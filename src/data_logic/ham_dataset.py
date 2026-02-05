import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

# Hàm identity hỗ trợ cho transform
def identity(x):
    return x

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1',
                 train: bool = True, selected_features: Optional[list] = None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # Danh sách tất cả các biến metadata gốc của HAM10000
        self.all_categorical = ['localization', 'sex']
        self.all_numeric = ['age']

        # Logic lọc biến: Nếu có selected_features từ quá trình SHAP, chỉ giữ lại các biến đó
        if selected_features is not None:
            self.categorical_cols = [c for c in self.all_categorical if c in selected_features]
            self.numeric_cols = [c for c in self.all_numeric if c in selected_features]
        else:
            self.categorical_cols = self.all_categorical
            self.numeric_cols = self.all_numeric

        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.num_mean_std: Dict[str, Tuple[float, float]] = {}

        if self.metadata_mode in ('full', 'full_weighted', 'late_fusion'):
            # Xử lý Categorical và lưu Cardinality
            for c in self.categorical_cols:
                vals = self.df[c].fillna('unknown').astype(str).values
                le = LabelEncoder()
                le.fit(vals)
                self.encoders[c] = le
                self.cat_cardinalities[c] = len(le.classes_)

            # Xử lý Numeric
            for nc in self.numeric_cols:
                arr = pd.to_numeric(self.df[nc], errors='coerce')
                mean = float(np.nanmean(arr))
                std = float(np.nanstd(arr)) + 1e-6
                self.num_mean_std[nc] = (mean, std)

        # Transform
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)) if train else transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(identity),
            transforms.RandomVerticalFlip() if train else transforms.Lambda(identity),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02) if train else transforms.Lambda(identity),
            transforms.RandomRotation(15) if train else transforms.Lambda(identity),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            full_path += '.jpg'
        with Image.open(full_path) as img:
            return img.convert("RGB")

    def _encode_metadata(self, row: pd.Series):
        if self.metadata_mode == 'diag1':
            return torch.zeros(len(self.numeric_cols)), torch.zeros(len(self.categorical_cols), dtype=torch.long)

        nums = []
        for nc in self.numeric_cols:
            val = row.get(nc, np.nan)
            mean, std = self.num_mean_std[nc]
            if pd.isna(val): val = mean
            nums.append((float(val) - mean) / std)

        cats = []
        for cc in self.categorical_cols:
            raw = str(row.get(cc, 'unknown'))
            le = self.encoders[cc]
            try:
                idx = int(le.transform([raw])[0])
            except:
                idx = 0
            cats.append(idx)

        return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = str(row['image_path'])

        img = self.transform(self._load_image(img_path))
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        if self.metadata_mode == 'late_fusion':
            meta_vec = torch.cat([meta_num, meta_cat.float()], dim=0)
            return img, (meta_vec, torch.zeros(0)), label

        return img, (meta_num, meta_cat), label