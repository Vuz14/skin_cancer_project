import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_train.csv',
    'VAL_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_val.csv',
    'TEST_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_test.csv',
    'IMG_ROOT': '/mnt/d/skin_cancer_project/dataset/Bcn20000-preprocessed',
    'MODEL_OUT': '/mnt/d/skin_cancer_project/checkpoint_ResNet50_bcn20000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu', 
    'SEED': 42, 
    'IMG_SIZE': 384,
    'BATCH_SIZE': 16,
    'MODEL_NAME': 'resnet50',
    
    'EPOCHS': 20,           
    'BASE_LR': 1e-5,
    'WARMUP_EPOCHS': 3,     
    'WEIGHT_DECAY': 1e-2,
    # --------------------------------------------------

    'METADATA_MODE': 'full_weighted', 
    'METADATA_FEATURE_BOOST': 5.0,
    'META_CLASS_WEIGHT_BOOST': 2.0, 
    'PRETRAINED': True, 
    'FINE_TUNE_MODE': 'full_unfreeze',
    'ACCUM_STEPS': 1,
    'SHAP_THRESHOLD': 0.005, 
    'NSAMPLES_SHAP': 50       
}

def preprocess_bcn(df):
    """Làm sạch dữ liệu và tạo nhãn chuẩn"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'
    
    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
    # Loại bỏ các hàng không có chẩn đoán xác định
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    return df

def auto_feature_selection(train_df, config, device):
    """Giai đoạn thăm dò: Xác định các biến metadata quan trọng qua SHAP probe"""
    print("\n --- GIAI ĐOẠN: TỰ ĐỘNG LỌC BIẾN METADATA (SHAP) ---")
    temp_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    # Khởi tạo mô hình probe để đánh giá độ quan trọng ban đầu
    temp_model = get_model(config, temp_ds.cat_cardinalities, len(temp_ds.numeric_cols)).to(device)
    temp_model.eval()

    all_meta_features = temp_ds.numeric_cols + temp_ds.categorical_cols
    # Giả lập hoặc tính toán độ quan trọng thực tế qua SHAP
    importance_map = {feat: np.random.uniform(0.001, 0.02) for feat in all_meta_features}

    selected_features = [f for f, imp in importance_map.items() if imp > config['SHAP_THRESHOLD']]
    print(f"Biến metadata quan trọng được giữ lại: {selected_features}")
    return selected_features

def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    # Log thiết bị chạy (CUDA/CPU)
    print("="*50)
    print(f" Thiết bị đang sử dụng: {device}")
    if device.type == 'cuda':
        print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*50)

    # 1. Tải và chuẩn bị dữ liệu
    print(" Đang tải và làm sạch dữ liệu...")
    train_df = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    val_df = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    test_df = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    # 2. SHAP Selection
    important_features = advanced_feature_selection_rfe(train_df, config, device)

    # 3. Khởi tạo Datasets & Loaders
    train_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                                 config['METADATA_MODE'], train=True, selected_features=important_features)
    val_ds = DermoscopyDataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                               config['METADATA_MODE'], train=False, selected_features=important_features)
    test_ds = DermoscopyDataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                                config['METADATA_MODE'], train=False, selected_features=important_features)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 4. Khởi tạo Model
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'])

    # Thiết lập Loss với cân bằng trọng số
    y_train = train_df['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['BASE_LR'], 
        weight_decay=config['WEIGHT_DECAY']
    )
 
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

  
    print("\n🚀 --- BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC (BCN20000) ---")
    train_loop(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        config, 
        criterion, 
        optimizer, 
        scheduler, 
        device, 
        log_suffix="bcn_final_enhanced"
    )


def advanced_feature_selection_rfe(train_df, config, device):
    print("\n🔍 --- GIAI ĐOẠN: CHỌN LỌC ĐẶC TRƯNG NÂNG CAO (RFECV) - BCN20000 ---")

    # 1. Chuẩn bị dữ liệu metadata
    # Lưu ý: Dùng DermoscopyDataset (class của BCN)
    temp_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    all_cols = temp_ds.numeric_cols + temp_ds.categorical_cols

    X = train_df[all_cols].copy()
    y = train_df['label'].values

    # Xử lý dữ liệu thiếu
    if temp_ds.numeric_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X[temp_ds.numeric_cols] = num_imputer.fit_transform(X[temp_ds.numeric_cols])

    for col in temp_ds.categorical_cols:
        X[col] = X[col].fillna('unknown').astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # 2. Chạy RFECV
    print("🤖 Đang chạy RFE... (Tìm tập hợp biến tối ưu cho BCN)")
    rf = RandomForestClassifier(n_estimators=100, random_state=config['SEED'], n_jobs=-1)
    cv = StratifiedKFold(n_splits=5)

    # step=1: Loại từng biến một.
    selector = RFECV(estimator=rf, step=1, cv=cv, scoring='accuracy', min_features_to_select=3, n_jobs=-1)
    selector = selector.fit(X, y)

    # 3. Kết quả
    selected_mask = selector.support_
    selected_features = np.array(all_cols)[selected_mask].tolist()

    print(f"📊 Số lượng biến tối ưu: {selector.n_features_}/{len(all_cols)}")
    print(f"✅ QUYẾT ĐỊNH GIỮ LẠI: {selected_features}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.xlabel("Số lượng đặc trưng được chọn")
    plt.ylabel("Độ chính xác (CV Score)")
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
    plt.title("RFE Performance - BCN20000")
    plt.grid(True)
    plt.savefig(os.path.join(config['MODEL_OUT'], "rfe_performance_bcn.png"))

    return selected_features

if __name__ == '__main__':
    main(CONFIG)