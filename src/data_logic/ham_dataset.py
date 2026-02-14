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

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1', 
                 train: bool = True, selected_features: Optional[list] = None):
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- XỬ LÝ ĐƯỜNG DẪN & NHÃN ---
        if 'image_path' not in self.df.columns and 'image_id' in self.df.columns:
            self.df['image_path'] = self.df['image_id'].astype(str) + '.jpg'
        
        if 'label' not in self.df.columns and 'dx' in self.df.columns:
            # HAM10000: mel, bcc, akiec là ác tính (1)
            self.df['label'] = self.df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

        # --- CẤU HÌNH METADATA ---
        self.all_categorical = ['localization', 'sex']
        self.all_numeric = ['age']

        if selected_features is not None:
            self.categorical_cols = [c for c in self.all_categorical if c in selected_features]
            self.numeric_cols = [c for c in self.all_numeric if c in selected_features]
        else:
            self.categorical_cols = [c for c in self.all_categorical if c in self.df.columns]
            self.numeric_cols = [c for c in self.all_numeric if c in self.df.columns]

        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.num_mean_std: Dict[str, Tuple[float, float]] = {}

        if self.metadata_mode in ('full', 'full_weighted', 'late_fusion'):
            for c in self.categorical_cols:
                # Fill NA bằng 'unknown' cho categorical
                vals = self.df[c].fillna('unknown').astype(str).values
                le = LabelEncoder()
                le.fit(vals)
                self.encoders[c] = le
                self.cat_cardinalities[c] = len(le.classes_)
            
            for nc in self.numeric_cols:
                # Fill NA bằng mean cho numeric
                arr = pd.to_numeric(self.df[nc], errors='coerce')
                mean = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
                std = float(np.nanstd(arr)) + 1e-6 if not np.all(np.isnan(arr)) else 1.0
                self.num_mean_std[nc] = (mean, std)

        # --- AUGMENTATION (Albumentations - Đồng bộ với BCN) ---
        if self.train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10, p=0.3),
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
            # Fallback nếu path chỉ là tên file
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
            raw = str(row.get(cc, 'unknown'))
            le = self.encoders[cc]
            try: idx = int(le.transform([raw])[0])
            except: idx = 0 # Fallback cho unknown classes
            cats.append(idx)
            
        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Albumentations cần input là numpy array
        img_np = np.array(self._load_image(row['image_path']))
        augmented = self.transform(image=img_np)
        img = augmented['image']
        
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        if self.metadata_mode == 'late_fusion':
             return img, (meta_num, meta_cat), label

        return img, (meta_num, meta_cat), label