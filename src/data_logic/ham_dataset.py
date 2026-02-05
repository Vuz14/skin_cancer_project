import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

def identity(x):
    return x

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1',
                 train: bool = True, selected_features: Optional[list] = None):
        # Làm sạch tên cột và tạo bản sao để tránh rò rỉ dữ liệu
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- TỰ ĐỘNG XỬ LÝ CỘT CHO HAM10000 ---
        # 1. Tạo image_path từ image_id nếu chưa có
        if 'image_path' not in self.df.columns and 'image_id' in self.df.columns:
            self.df['image_path'] = self.df['image_id'].astype(str) + '.jpg'
        
        # 2. Tạo nhãn nhị phân từ cột dx (chẩn đoán)
        if 'label' not in self.df.columns and 'dx' in self.df.columns:
            # Các lớp ác tính trong HAM10000: mel (melanoma), bcc (basal cell), akiec (actinic keratoses)
            self.df['label'] = self.df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

        # Danh sách tất cả các biến metadata gốc của HAM10000
        self.all_categorical = ['localization', 'sex']
        self.all_numeric = ['age']

        # Logic lọc biến từ SHAP
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

        if self.train:
            self.transform = transforms.Compose([
                # 1. Cắt và Resize ngẫu nhiên
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                
                # 2. Xoay và lật đa hướng (rất quan trọng cho ảnh da liễu)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45),
                
                # 3. Biến dạng hình học nhẹ
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
                
                # 4. Thay đổi màu sắc và độ tương phản (Color Jitter mạnh hơn)
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                
                # 5. Làm mờ để tăng độ bền bỉ cho mô hình
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
                # 6. Random Erasing (Xóa vùng ngẫu nhiên để tránh học vẹt các đốm nhiễu/lông)
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            full_path = os.path.join(self.img_root, os.path.basename(path))
            if not full_path.endswith('.jpg'):
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
            nums.append((float(val) - mean) / std if not pd.isna(val) else 0.0)

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
        row = self.df.iloc[idx]
        img = self.transform(self._load_image(row['image_path']))
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        meta_num, meta_cat = self._encode_metadata(row)

        if self.metadata_mode == 'late_fusion':
            meta_vec = torch.cat([meta_num, meta_cat.float()], dim=0)
            return img, (meta_vec, torch.zeros(0)), label

        return img, (meta_num, meta_cat), label