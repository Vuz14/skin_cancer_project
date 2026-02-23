import os
from typing import Dict, Tuple, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


def identity(x):
    return x


class DermoscopyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1',
                 train: bool = True, selected_features: Optional[list] = None,
                 external_encoders=None, external_stats=None):
        # Làm sạch tên cột và tạo bản sao
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()

        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- 1. XỬ LÝ NHÃN ---
        if 'image_path' not in self.df.columns and 'isic_id' in self.df.columns:
            self.df['image_path'] = self.df['isic_id'].astype(str) + '.jpg'

        if 'label' not in self.df.columns and 'diagnosis_1' in self.df.columns:
            self.df['diagnosis_1'] = self.df['diagnosis_1'].astype(str).str.strip().str.lower()
            self.df['label'] = self.df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

        # --- 2. CẤU HÌNH METADATA ---
        self.all_categorical = ['anatom_site_general', 'anatom_site_special', 'diagnosis_confirm_type', 'sex']
        self.all_numeric = ['age_approx']

        if self.metadata_mode == 'diag1':
            self.categorical_cols = []
            self.numeric_cols = []
        else:
            self.categorical_cols = [c for c in self.all_categorical if c in self.df.columns]
            self.numeric_cols = [c for c in self.all_numeric if c in self.df.columns]

        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.num_mean_std: Dict[str, Tuple[float, float]] = {}

        if self.metadata_mode in ('full', 'full_weighted', 'late_fusion') and (
                self.categorical_cols or self.numeric_cols):
            # --- LOGIC MỚI: Ưu tiên dùng Encoder truyền vào (cho Test Cross) ---
            if external_encoders and external_stats:
                self.encoders = external_encoders
                self.num_mean_std = external_stats
                # Cập nhật cardinality từ encoder có sẵn
                for c in self.categorical_cols:
                    if c in self.encoders:
                        self.cat_cardinalities[c] = len(self.encoders[c].classes_)
            else:
                # --- LOGIC CŨ: Tự Fit (cho lúc Train) ---
                for c in self.categorical_cols:
                    vals = self.df[c].fillna('NA').astype(str).values
                    le = LabelEncoder()
                    le.fit(vals)
                    self.encoders[c] = le
                    self.cat_cardinalities[c] = len(le.classes_)

                for nc in self.numeric_cols:
                    arr = pd.to_numeric(self.df[nc], errors='coerce')
                    mean = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
                    std = float(np.nanstd(arr)) + 1e-6 if not np.all(np.isnan(arr)) else 1.0
                    self.num_mean_std[nc] = (mean, std)

                # --- AUGMENTATION (Albumentations API MỚI) ---
                if self.train:
                    self.transform = A.Compose([
                        A.Resize(img_size, img_size),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),

                        # 1. THAY THẾ ShiftScaleRotate BẰNG Affine
                        A.Affine(
                            scale=(0.9, 1.1),
                            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                            rotate=(-45, 45),
                            p=0.5
                        ),

                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

                        # 2. THAY THẾ THAM SỐ CỦA CoarseDropout
                        A.CoarseDropout(
                            num_holes_range=(1, 8),
                            hole_height_range=(0.05, 0.1),  # Tính theo phần trăm kích thước ảnh (5% - 10%)
                            hole_width_range=(0.05, 0.1),
                            p=0.3
                        ),

                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.Resize(img_size, img_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            full_path = os.path.join(self.img_root, os.path.basename(path))
        with Image.open(full_path) as img:
            return img.convert("RGB")

    def _encode_metadata(self, row: pd.Series):
        if self.metadata_mode == 'diag1':
            return torch.zeros(len(self.numeric_cols)), torch.zeros(len(self.categorical_cols), dtype=torch.long)

        nums = []
        for nc in self.numeric_cols:
            val = row.get(nc, np.nan)
            mean, std = self.num_mean_std[nc]
            nums.append((float(val) - mean) / std if not pd.isna(val) else 0.0)

        cats = []
        for cc in self.categorical_cols:
            raw = str(row.get(cc, 'NA'))
            le = self.encoders[cc]
            try:
                idx = int(le.transform([raw])[0])
            except:
                idx = 0
            cats.append(idx)

        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        augmented = self.transform(image=np.array(self._load_image(row['image_path'])))
        img = augmented['image']
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        if self.metadata_mode == 'late_fusion':
            meta_vec = torch.cat([meta_num, meta_cat.float()], dim=0)
            return img, (meta_vec, torch.zeros(0)), label

        return img, (meta_num, meta_cat), label
