import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DermoscopyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1', 
                 train: bool = True, selected_features: Optional[list] = None):
    
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- 1. XỬ LÝ NHÃN ---
        if 'image_path' not in self.df.columns and 'isic_id' in self.df.columns:
            self.df['image_path'] = self.df['isic_id'].astype(str) + '.jpg'
        
        # Logic gán nhãn chuẩn: Kiểm tra cột 'diagnosis_1' hoặc 'diagnosis'
        diag_col = 'diagnosis_1' if 'diagnosis_1' in self.df.columns else 'diagnosis'
        if 'label' not in self.df.columns and diag_col in self.df.columns:
            self.df[diag_col] = self.df[diag_col].astype(str).str.strip().str.lower()
            # Danh sách mở rộng để tránh lỗi IndexError (không tìm thấy nhãn 1)
            malignant_list = ['mel', 'bcc', 'scc', 'melanoma', 'basal cell', 'squamous cell', 'carcinoma']
            self.df['label'] = self.df[diag_col].apply(
                lambda x: 1 if any(m in x for m in malignant_list) else 0
            )

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

        if self.metadata_mode in ('full', 'full_weighted', 'late_fusion') and (self.categorical_cols or self.numeric_cols):
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

        # --- 3. AUGMENTATION (Sửa lỗi tham số Affine & CoarseDropout) ---
        
        if self.train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-30, 30),
                    p=0.5
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 8), 
                    hole_height_range=(0.02, 0.1), 
                    hole_width_range=(0.02, 0.1), 
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # ĐẢM BẢO KHỐI NÀY LUÔN CHẠY KHI train=False
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
        if not self.numeric_cols and not self.categorical_cols:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)
        
        nums = [((float(row.get(nc, np.nan)) - self.num_mean_std[nc][0]) / self.num_mean_std[nc][1]) 
                if not pd.isna(row.get(nc, np.nan)) else 0.0 for nc in self.numeric_cols]
            
        cats = []
        for cc in self.categorical_cols:
            raw = str(row.get(cc, 'NA'))
            try: idx = int(self.encoders[cc].transform([raw])[0])
            except: idx = 0
            cats.append(idx)
            
        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_pil = self._load_image(row['image_path'])
        augmented = self.transform(image=np.array(img_pil))
        
        return augmented['image'], self._encode_metadata(row), torch.tensor(row['label'], dtype=torch.float32)