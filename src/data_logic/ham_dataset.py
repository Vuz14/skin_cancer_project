import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.data_logic.common_transforms import build_dermoscopy_transform, uses_metadata


def identity(x):
    return x


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1',
                 train: bool = True, selected_features: Optional[list] = None,
                 external_encoders=None, external_stats=None):

        # Làm sạch tên cột và tạo bản sao để tránh rò rỉ dữ liệu
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()

        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- TỰ ĐỘNG XỬ LÝ CỘT CHO HAM10000 ---
        if 'image_path' not in self.df.columns and 'image_id' in self.df.columns:
            self.df['image_path'] = self.df['image_id'].astype(str) + '.jpg'

        if 'label' not in self.df.columns and 'dx' in self.df.columns:
            # Các lớp ác tính trong HAM10000: mel, bcc, akiec
            self.df['label'] = self.df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

        # --- CẤU HÌNH METADATA ---
        self.all_categorical = ['localization', 'sex']
        self.all_numeric = ['age']

        if not uses_metadata(self.metadata_mode):
            self.categorical_cols = []
            self.numeric_cols = []
        elif selected_features is not None:
            self.categorical_cols = [c for c in self.all_categorical if c in selected_features]
            self.numeric_cols = [c for c in self.all_numeric if c in selected_features]
        else:
            self.categorical_cols = [c for c in self.all_categorical if c in self.df.columns]
            self.numeric_cols = [c for c in self.all_numeric if c in self.df.columns]

        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.num_mean_std: Dict[str, Tuple[float, float]] = {}

        # ==========================================================
        # 1. KHỞI TẠO ENCODERS & STATS
        # ==========================================================
        if uses_metadata(self.metadata_mode):
            # Ưu tiên dùng Encoder/Stats truyền từ ngoài (Khi Test)
            if external_encoders is not None and external_stats is not None:
                self.encoders = external_encoders
                self.num_mean_std = external_stats
                for c in self.categorical_cols:
                    if c in self.encoders:
                        self.cat_cardinalities[c] = len(self.encoders[c].classes_)
            # Nếu không có thì tự Fit (Khi Train)
            else:
                for c in self.categorical_cols:
                    vals = self.df[c].fillna('unknown').astype(str).values
                    le = LabelEncoder()
                    le.fit(vals)
                    self.encoders[c] = le
                    self.cat_cardinalities[c] = len(le.classes_)

                for nc in self.numeric_cols:
                    arr = pd.to_numeric(self.df[nc], errors='coerce')
                    mean = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
                    std = float(np.nanstd(arr)) + 1e-6 if not np.all(np.isnan(arr)) else 1.0
                    self.num_mean_std[nc] = (mean, std)

        self.transform = build_dermoscopy_transform(img_size, train)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            # Fallback nếu path chỉ là tên file
            full_path = os.path.join(self.img_root, os.path.basename(path))
            
        with Image.open(full_path) as img:
            return img.convert("RGB")

    def _encode_metadata(self, row: pd.Series):
        if not uses_metadata(self.metadata_mode):
            return torch.zeros(0, dtype=torch.float32), torch.zeros(0, dtype=torch.long)
        
        nums = []
        for nc in self.numeric_cols:
            val = row.get(nc, np.nan)
            mean, std = self.num_mean_std[nc]
            nums.append((float(val) - mean) / std if not pd.isna(val) else 0.0)
            
        cats = []
        for cc in self.categorical_cols:
            raw = str(row.get(cc, 'unknown'))
            if cc in self.encoders:
                le = self.encoders[cc]
                try:
                    idx = int(le.transform([raw])[0])
                except ValueError:  # Bắt đúng lỗi nếu không tìm thấy nhãn
                    idx = 0
            else:
                idx = 0
            cats.append(idx)
            
        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- SỬA LỖI TRUYỀN ẢNH VÀO ALBUMENTATIONS ---
        img_pil = self._load_image(row['image_path'])
        img_np = np.array(img_pil)  # Ép kiểu sang Numpy
        augmented = self.transform(image=img_np)  # Truyền theo keyword argument
        img = augmented['image']  # Lấy tensor ảnh ra từ dictionary
        # ----------------------------------------------

        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        return img, (meta_num, meta_cat), label
