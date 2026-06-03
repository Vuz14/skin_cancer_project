import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.data_logic.common_transforms import build_dermoscopy_transform, uses_metadata

class DermoscopyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1',
                  train: bool = True, selected_features: Optional[list] = None,
                  external_encoders=None, external_stats=None, augmentation_profile: str = "light"):
        
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
        # Restrict inputs to metadata available before diagnosis, as stated in
        # the experimental protocol. Confirmation/biopsy columns are targets'
        # downstream consequences and must never be model inputs.
        self.all_categorical = ['anatom_site_general', 'sex']
        self.all_numeric = ['age_approx']

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

        # Logic xử lý Encoder/Stats (Chỉ chạy nếu không phải mode diag1)
        if uses_metadata(self.metadata_mode):
            if external_encoders is not None and external_stats is not None:
                self.encoders = external_encoders
                self.num_mean_std = external_stats
                for c in self.categorical_cols:
                    if c in self.encoders:
                        self.cat_cardinalities[c] = len(self.encoders[c].classes_)
            else:
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

        self.transform = build_dermoscopy_transform(img_size, train, augmentation_profile)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            full_path = os.path.join(self.img_root, os.path.basename(path))
        with Image.open(full_path) as img:
            return img.convert("RGB")

    def _encode_metadata(self, row: pd.Series):
        if not uses_metadata(self.metadata_mode):
            return torch.zeros(0, dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        nums = []
        for nc in self.numeric_cols:
            val = row.get(nc, np.nan)
            mean, std = self.num_mean_std.get(nc, (0.0, 1.0))
            nums.append((float(val) - mean) / std if not pd.isna(val) else 0.0)

        cats = []
        for cc in self.categorical_cols:
            raw = str(row.get(cc, 'NA'))
            le = self.encoders.get(cc)
            if le:
                try:
                    idx = int(le.transform([raw])[0])
                except:
                    idx = 0
            else:
                idx = 0
            cats.append(idx)

        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_pil = self._load_image(row['image_path'])
        augmented = self.transform(image=np.array(img_pil))
        img = augmented['image']
        
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        return img, (meta_num, meta_cat), label
