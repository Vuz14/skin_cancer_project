import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Thêm đường dẫn gốc của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# ------------------- KIỂM TRA GPU -------------------
def check_gpu_status():
    print("\n🔍 --- KIỂM TRA TRẠNG THÁI GPU ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Đã tìm thấy GPU: {gpu_name}")
        return 'cuda'
    else:
        print("❌ KHÔNG TÌM THẤY GPU! Code sẽ chạy chậm trên CPU.")
        return 'cpu'

# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_train.csv',
    'VAL_CSV':   r'D:\skin_cancer_project\dataset\metadata\bcn20000_val.csv',
    'TEST_CSV':  r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'IMG_ROOT':  r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,
 
    # --- MODEL & NAMING ---
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effb4',                 # Tên ngắn dùng để lưu file (effb4, res50)
    
    'IMG_SIZE': 300,  # BCN20000 thường dùng ảnh lớn hơn HAM10000
    'BATCH_SIZE': 16,

    'EPOCHS': 15,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA STRATEGY ---
    # Chọn 'full_weighted' (ghép sớm) hoặc 'late_fusion' (ghép muộn)
    'METADATA_MODE': 'full_weighted', 
    'METADATA_FEATURE_BOOST': 2.0,
    'META_CLASS_WEIGHT_BOOST': 1.0,
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',

    'UNFREEZE_KEYWORDS': ['conv_head', 'blocks.6', 'blocks.7'],

    'ACCUM_STEPS': 1,

    # --- CẤU HÌNH PHÂN TÍCH ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'palms/soles',

    # --- CẤU HÌNH GRAD-CAM ---
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,  
}

def preprocess_bcn(df):
    """Làm sạch dữ liệu cơ bản và tạo nhãn chuẩn"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Xử lý đường dẫn ảnh
    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    # Xử lý nhãn
    if 'diagnosis_1' in df.columns:
        df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
        # Loại bỏ dữ liệu rác
        df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
        # Tạo nhãn 0/1 (Ác tính/Lành tính)
        df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    
    return df

# ==============================================================================
# PHÂN TÍCH ĐỘ QUAN TRỌNG (Feature Importance)
# ==============================================================================
def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, config):
    print(f"\n🤖 [Analysis] Đang chạy Random Forest để đánh giá Metadata BCN...")
    valid_cat = [c for c in categorical_cols if c in train_df.columns]
    valid_num = [c for c in numeric_cols if c in train_df.columns]
    
    if not valid_cat and not valid_num: return

    meta_df = pd.DataFrame()
    # One-Hot Encoding cho Categorical
    if valid_cat:
        temp_cat = train_df[valid_cat].fillna('unknown')
        meta_df = pd.concat([meta_df, pd.get_dummies(temp_cat, columns=valid_cat, prefix_sep='=')], axis=1)
    
    # Impute cho Numeric
    if valid_num:
        temp_num = train_df[valid_num].copy()
        imputer = SimpleImputer(strategy='mean')
        temp_num_filled = pd.DataFrame(imputer.fit_transform(temp_num), columns=valid_num)
        meta_df = pd.concat([meta_df, temp_num_filled], axis=1)

    y = train_df['label'].values
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(meta_df, y)
    
    # Lấy Feature Importance
    importances = rf.feature_importances_
    feature_imp_list = sorted(zip(meta_df.columns, importances), key=lambda x: x[1], reverse=True)

    # In ra màn hình
    print("\n📊 BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG (Top 10):")
    anchor_score = next((s for n, s in feature_imp_list if config['ANCHOR_VALUE_NAME'] in n), 0)
    
    for i, (name, score) in enumerate(feature_imp_list[:10]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i + 1}. {name}: {score:.5f} [{status}]")

    # Lưu kết quả vào CSV chuẩn tên
    csv_name = f"bcn20k_{config['SHORT_NAME']}_meta_imp.csv"
    out_path = os.path.join(config['MODEL_OUT'], csv_name)
    pd.DataFrame(feature_imp_list, columns=['Feature', 'Importance']).to_csv(out_path, index=False)
    print(f"💾 Đã lưu bảng phân tích Metadata vào: {csv_name}")


# ==============================================================================
# MAIN
# ==============================================================================
def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print("📂 Loading Data BCN20000...")
    raw_train = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    # Metadata columns đặc thù của BCN
    numeric_cols = ['age_approx']
    categorical_cols = ['anatom_site_general', 'anatom_site_special', 'diagnosis_confirm_type', 'sex']

    # Chạy phân tích Metadata nếu được bật
    if config.get('ANALYZE_METADATA', False):
        analyze_feature_importance_only(raw_train, categorical_cols, numeric_cols, config)

    print("🚀 Khởi tạo Dataset & Loader...")
    train_ds = DermoscopyDataset(raw_train, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = DermoscopyDataset(raw_val, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = DermoscopyDataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # Khởi tạo Model
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

    # Loss Function (Weighted BCE)
    y_train = raw_train['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # --- TẠO LOG SUFFIX CHUẨN ---
    # Ví dụ: bcn20k_effb4
    log_suffix = f"bcn20k_{config['SHORT_NAME']}"

    # BẮT ĐẦU TRAINING
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix=log_suffix
    )

if __name__ == '__main__':
    main(CONFIG)