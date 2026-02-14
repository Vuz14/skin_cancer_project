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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# ------------------- CHECK GPU -------------------
def check_gpu_status():
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    return 'cpu'

# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_train.csv',
    'VAL_CSV':   r'D:\skin_cancer_project\dataset\metadata\ham10000_val.csv',
    'TEST_CSV':  r'D:\skin_cancer_project\dataset\metadata\ham10000_test.csv',
    'IMG_ROOT':  r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_ham10000',

    'DEVICE': 'cuda',
    'SEED': 42,
    
    # --- MODEL & NAMING ---
    'MODEL_NAME': 'tf_efficientnet_b4_ns', # Hoặc 'resnet50'
    'SHORT_NAME': 'effb4',                 # Tên ngắn để lưu file csv (effb4, res50)
    
    'IMG_SIZE': 224, # HAM10000 thường nhỏ hơn, 224 là chuẩn
    'BATCH_SIZE': 16,
    'EPOCHS': 20,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA ---
    'METADATA_MODE': 'full_weighted',
    'METADATA_FEATURE_BOOST': 2.0,
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_KEYWORDS': ['conv_head', 'blocks.6', 'blocks.5'], # Tinh chỉnh cho EffNet

    # --- ANALYSIS ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'lower extremity', # Một giá trị mẫu trong localization để so sánh

    # --- GRAD-CAM ---
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,  
}

def preprocess_ham(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'image_path' not in df.columns:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
    
    if 'dx' in df.columns and 'label' not in df.columns:
        df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
    return df

# --- PHÂN TÍCH QUAN TRỌNG (Random Forest) ---
def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, anchor_value):
    print(f"\n🤖 [Analysis] Đang chạy Random Forest để đánh giá Metadata...")
    valid_cat = [c for c in categorical_cols if c in train_df.columns]
    valid_num = [c for c in numeric_cols if c in train_df.columns]
    
    if not valid_cat and not valid_num: return

    meta_df = pd.DataFrame()
    if valid_cat:
        temp_cat = train_df[valid_cat].fillna('unknown')
        meta_df = pd.concat([meta_df, pd.get_dummies(temp_cat, columns=valid_cat)], axis=1)
    if valid_num:
        imputer = SimpleImputer(strategy='mean')
        temp_num = pd.DataFrame(imputer.fit_transform(train_df[valid_num]), columns=valid_num)
        meta_df = pd.concat([meta_df, temp_num], axis=1)

    y = train_df['label'].values
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(meta_df, y)
    
    imps = sorted(zip(meta_df.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    
    # Lưu CSV ngắn gọn
    csv_name = f"ham10k_{CONFIG['SHORT_NAME']}_meta_imp.csv"
    pd.DataFrame(imps, columns=['Feature', 'Importance']).to_csv(
        os.path.join(CONFIG['MODEL_OUT'], csv_name), index=False
    )
    print(f"📊 Đã lưu bảng xếp hạng Metadata vào: {csv_name}")

def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print("📂 Loading Data...")
    raw_train = preprocess_ham(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_ham(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_ham(pd.read_csv(config['TEST_CSV']))

    # Metadata columns của HAM10000
    cat_cols = ['localization', 'sex']
    num_cols = ['age']

    if config['ANALYZE_METADATA']:
        analyze_feature_importance_only(raw_train, cat_cols, num_cols, config['ANCHOR_VALUE_NAME'])

    print("🚀 Khởi tạo Dataset & Loader...")
    train_ds = HAM10000Dataset(raw_train, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = HAM10000Dataset(raw_val, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = HAM10000Dataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

    y_train = raw_train['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight = torch.tensor(weights[1] * 1.2, device=device) # Boost nhẹ cho lớp bệnh
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # Tên suffix chuẩn: ham10k_effb4
    log_suffix = f"ham10k_{config['SHORT_NAME']}"
    
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix=log_suffix
    )

if __name__ == '__main__':
    main(CONFIG)