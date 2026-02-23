import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Thêm đường dẫn gốc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.losses import FocalLossBCE
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
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effb4', 
    
    'IMG_SIZE': 224, 
    'BATCH_SIZE': 16,
    'EPOCHS': 15,
    'BASE_LR': 8e-5,        
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA (MODE 3 FiLM) ---
    'METADATA_MODE': 'late_fusion', 
    # Đã bỏ METADATA_FEATURE_BOOST theo yêu cầu vì dùng FiLM
    
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_SUBSTRINGS': ['layers', 'blocks', 'norm', 'conv_head', 'features', 'stem'],
    
    # --- IMBALANCE HANDLING (Xử lý mất cân bằng) ---
    'USE_SAMPLER': True,   
    'ACCUM_STEPS': 1,

    # --- ANALYSIS ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'lower extremity', 

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
        # 1: Ác tính (mel, bcc, akiec), 0: Lành tính
        df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
    return df

# --- PHÂN TÍCH QUAN TRỌNG (Random Forest) ---
def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, config):
    print(f"\n🤖 [Analysis] Đang chạy Random Forest để đánh giá Metadata HAM10000...")
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
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(meta_df, y)
    
    imps = sorted(zip(meta_df.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    
    anchor_score = next((s for n, s in imps if config['ANCHOR_VALUE_NAME'] in n), 0)
    print("\n📊 TOP METADATA FEATURES:")
    for i, (name, score) in enumerate(imps[:8]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i+1}. {name}: {score:.5f} [{status}]")

    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    csv_name = f"ham10k_{config['SHORT_NAME']}_meta_imp.csv"
    out_path = os.path.join(run_dir, csv_name)
    
    pd.DataFrame(imps, columns=['Feature', 'Importance']).to_csv(out_path, index=False)
    print(f"💾 Đã lưu bảng xếp hạng Metadata vào: {out_path}")

def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    
    # --- TẠO THƯ MỤC CON (RUN_DIR) ---
    run_name = f"{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    run_dir = os.path.join(config['MODEL_OUT'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    config['RUN_DIR'] = run_dir 
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print(f"📂 Thư mục gốc: {config['MODEL_OUT']}")
    print(f"📂 Thư mục chạy: {config['RUN_DIR']}")

    print("📂 Loading Data HAM10000...")
    raw_train = preprocess_ham(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_ham(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_ham(pd.read_csv(config['TEST_CSV']))

    cat_cols = ['localization', 'sex']
    num_cols = ['age']

    # Chạy phân tích Metadata (Random Forest)
    if config['ANALYZE_METADATA']:
        analyze_feature_importance_only(raw_train, cat_cols, num_cols, config)

    print("🚀 Khởi tạo Dataset & Loader...")
    train_ds = HAM10000Dataset(raw_train, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = HAM10000Dataset(raw_val, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = HAM10000Dataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    # --- XỬ LÝ MẤT CÂN BẰNG: WeightedRandomSampler ---
    train_sampler = None
    if config['USE_SAMPLER']:
        targets = raw_train['label'].values
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight).double()
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # Khởi tạo Model
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config['UNFREEZE_SUBSTRINGS'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    
    # --- XỬ LÝ MẤT CÂN BẰNG: Focal Loss ---
    # Alpha=0.8 tập trung vào lớp thiểu số (ác tính), Gamma=2.0 giảm trọng số các mẫu dễ
    criterion = FocalLossBCE(alpha=0.8, gamma=2.0)
    
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    log_suffix = f"ham10k_{config['SHORT_NAME']}"
    
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix=log_suffix
    )

if __name__ == '__main__':
    main(CONFIG)