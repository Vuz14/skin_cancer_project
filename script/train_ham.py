import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc của project
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.losses import FocalLoss
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/ham10000_train.csv',
    'VAL_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/ham10000_val.csv',
    'TEST_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/ham10000_test.csv',
    'IMG_ROOT': '/mnt/d/skin_cancer_project/dataset/Ham10000-preprocessed',
    'MODEL_OUT':  '/mnt/d/skin_cancer_project/checkpoint_ResNet50_ham10000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu', 
    'SEED': 42, 
    'IMG_SIZE': 384,
    'BATCH_SIZE': 16,
    'MODEL_NAME': 'resnet50',

    # --- CẬP NHẬT CHIẾN LƯỢC HỌC (STRATEGY) ---

    'EPOCHS': 20,           # Tăng lên 20 để hội tụ sâu hơn
    'BASE_LR': 1e-5,        # Giảm mạnh (từ 5e-4 xuống 8e-5) để Loss mượt hơn
    'WARMUP_EPOCHS': 3,     # Tăng Warmup lên 3 epoch đầu
    'WEIGHT_DECAY': 1e-2,   # Tăng Weight Decay để chống Overfit mạnh hơn
    # ------------------------------------------

    'METADATA_MODE': 'full_weighted',
    'METADATA_FEATURE_BOOST': 5.0,
    'PRETRAINED': True, 
    'FINE_TUNE_MODE': 'full_unfreeze',
    'UNFREEZE_SUBSTRINGS': [],
    'USE_SAMPLER': True,
    'ACCUM_STEPS': 1,
    'SHAP_THRESHOLD': 0.005, 
    'NSAMPLES_SHAP': 50       
}

def auto_feature_selection_ham(train_df, config, device):
    """Giai đoạn thăm dò: Xác định các biến metadata quan trọng cho HAM10000"""
    print("\n🔍 --- GIAI ĐOẠN: TỰ ĐỘNG LỌC BIẾN METADATA (SHAP) - HAM10000 ---")
    
    temp_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    temp_model = get_model(config, temp_ds.cat_cardinalities, len(temp_ds.numeric_cols)).to(device)
    temp_model.eval()

    all_meta_features = temp_ds.numeric_cols + temp_ds.categorical_cols
    # Placeholder: Trong thực tế sẽ chạy shap.KernelExplainer
    importance_map = {feat: np.random.uniform(0.001, 0.02) for feat in all_meta_features}

    selected_features = [f for f, imp in importance_map.items() if imp > config['SHAP_THRESHOLD']]
    
    print(f" Biến metadata giữ lại: {selected_features}")
    return selected_features

def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)
    
    # Log kiểm tra GPU
    print("="*50)
    print(f" Thiết bị đang sử dụng: {device}")
    if device.type == 'cuda':
        print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*50)

    # 1. Tải và chuẩn bị dữ liệu
    print(" Đang tải dữ liệu HAM10000...")
    train_df = pd.read_csv(config['TRAIN_CSV'])
    val_df = pd.read_csv(config['VAL_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])
    
    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.strip()
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        if 'dx' in df.columns: 
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
    
    # 2. SHAP Selection
    important_features = advanced_feature_selection_rfe(train_df, config, device)

    # 3. Khởi tạo Datasets (Với bộ Strong Augmentation đã cập nhật trong ham_dataset.py)
    train_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                               config['METADATA_MODE'], train=True, selected_features=important_features)
    val_ds = HAM10000Dataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                             config['METADATA_MODE'], train=False, selected_features=important_features)
    test_ds = HAM10000Dataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                              config['METADATA_MODE'], train=False, selected_features=important_features)

    # 4. Sampler & Loaders
    train_sampler = None
    if config['USE_SAMPLER']:
        targets = train_df['label'].values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = torch.from_numpy(weights[targets]).double()
        train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 5. Khởi tạo Model
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config['UNFREEZE_SUBSTRINGS'])

    # Optimizer sử dụng BASE_LR và WEIGHT_DECAY từ CONFIG
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['BASE_LR'], 
        weight_decay=config['WEIGHT_DECAY']
    )
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # 6. Huấn luyện
    print("\n🚀 --- BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC (HAM10000) ---")
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
        log_suffix="ham10k_final_enhanced"
    )

def advanced_feature_selection_rfe(train_df, config, device):
    print("\n🔍 --- GIAI ĐOẠN: CHỌN LỌC ĐẶC TRƯNG NÂNG CAO (RFECV) ---")

    # 1. Chuẩn bị dữ liệu metadata
    # Lưu ý: train=False để dataset không trộn ảnh clean/roi lung tung lúc này
    temp_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    all_cols = temp_ds.numeric_cols + temp_ds.categorical_cols

    X = train_df[all_cols].copy()
    y = train_df['label'].values

    # Xử lý dữ liệu thiếu (Imputation)
    if temp_ds.numeric_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X[temp_ds.numeric_cols] = num_imputer.fit_transform(X[temp_ds.numeric_cols])

    for col in temp_ds.categorical_cols:
        X[col] = X[col].fillna('unknown').astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # 2. Chạy RFECV (Recursive Feature Elimination with Cross-Validation)
    print("🤖 Đang chạy RFE... (Quá trình này tìm tập hợp biến tối ưu nhất)")
    rf = RandomForestClassifier(n_estimators=100, random_state=config['SEED'], n_jobs=-1)
    cv = StratifiedKFold(n_splits=5) # 5-Fold để đảm bảo khách quan

    # step=1: Loại từng biến một. min_features_to_select=3: Giữ ít nhất 3 biến
    selector = RFECV(estimator=rf, step=1, cv=cv, scoring='accuracy', min_features_to_select=3, n_jobs=-1)
    selector = selector.fit(X, y)

    # 3. Lấy kết quả
    selected_mask = selector.support_
    selected_features = np.array(all_cols)[selected_mask].tolist()

    print(f"📊 Số lượng biến tối ưu: {selector.n_features_}/{len(all_cols)}")
    print(f"✅ QUYẾT ĐỊNH GIỮ LẠI: {selected_features}")

    # Vẽ biểu đồ hiệu năng (Quan trọng cho bài báo)
    plt.figure(figsize=(10, 6))
    plt.xlabel("Số lượng đặc trưng được chọn")
    plt.ylabel("Độ chính xác (Cross Validation Score)")
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
    plt.title("Hiệu năng mô hình theo số lượng Metadata")
    plt.grid(True)
    plt.savefig(os.path.join(config['MODEL_OUT'], "rfe_performance.png"))
    print("📈 Đã lưu biểu đồ RFE vào folder checkpoint.")

    return selected_features

if __name__ == '__main__':
    main(CONFIG)