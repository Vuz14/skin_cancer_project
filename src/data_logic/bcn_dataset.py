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

class DermoscopyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, img_size: int, metadata_mode: str = 'diag1', 
                 train: bool = True, selected_features: Optional[list] = None):
        # Làm sạch tên cột và tạo bản sao
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        
        self.img_root = img_root
        self.img_size = img_size
        self.train = train
        self.metadata_mode = metadata_mode

        # --- TỰ ĐỘNG XỬ LÝ CỘT ---
        if 'image_path' not in self.df.columns and 'isic_id' in self.df.columns:
            self.df['image_path'] = self.df['isic_id'].astype(str) + '.jpg'
        
        if 'label' not in self.df.columns and 'diagnosis_1' in self.df.columns:
            self.df['diagnosis_1'] = self.df['diagnosis_1'].astype(str).str.strip().str.lower()
            self.df['label'] = self.df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

        # Cấu hình Metadata
        self.all_categorical = ['anatom_site_general', 'anatom_site_special', 'diagnosis_confirm_type', 'sex']
        self.all_numeric = ['age_approx']

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

        if self.train:
            self.transform = transforms.Compose([
                # 1. Resize và Crop ngẫu nhiên mạnh hơn (từ 0.6 thay vì 0.7)
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                
                # 2. Lật xoay đa hướng
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45), 
                
                # 3. Biến dạng hình học: Dịch chuyển và co giãn
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                
                # 4. Thay đổi màu sắc mạnh tay hơn để mô hình không bị phụ thuộc vào sắc độ ảnh
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                
                # 5. Làm mờ nhẹ (Gaussian Blur) giúp mô hình bền bỉ với ảnh chất lượng thấp
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
                # 6. ✅ Random Erasing: Xóa ngẫu nhiên các vùng nhỏ (giả lập lông, thước đo, vật cản)
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
            ])
        else:
            # Đối với Validation/Test: Chỉ Resize và Chuẩn hóa
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
            try: idx = int(le.transform([raw])[0])
            except: idx = 0
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