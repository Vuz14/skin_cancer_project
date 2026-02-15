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
from src.utils.losses import FocalLoss
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode


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
    'SHORT_NAME': 'effb4',                 # Tên ngắn để lưu file
    
    'IMG_SIZE': 224, 
    'BATCH_SIZE': 16, 
    'BACKBONE': 'convnext',

    # --- CẬP NHẬT CHIẾN LƯỢC HỌC (STRATEGY) ---
    'EPOCHS': 20,           # Tăng lên 20 để hội tụ sâu hơn
    'BASE_LR': 8e-5,        # Giảm mạnh (từ 5e-4 xuống 8e-5) để Loss mượt hơn
    'WARMUP_EPOCHS': 3,     # Tăng Warmup lên 3 epoch đầu
    'WEIGHT_DECAY': 1e-3,   # Tăng Weight Decay để chống Overfit mạnh hơn
    # ------------------------------------------
    'BATCH_SIZE': 16,
    'EPOCHS': 20,
    'BASE_LR': 8e-5,        # Learning rate thấp hơn cho HAM
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA ---
    # Chọn 'full_weighted' hoặc 'late_fusion'
    'METADATA_MODE': 'full_weighted', 
    'METADATA_FEATURE_BOOST': 2.0,
    
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_SUBSTRINGS': ['layers', 'blocks', 'norm', 'conv_head', 'features', 'stem'],
    
    'USE_SAMPLER': True,   # HAM10000 rất mất cân bằng, nên dùng Sampler
    'ACCUM_STEPS': 1,

    # --- ANALYSIS ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'lower extremity', # Giá trị mẫu để so sánh độ quan trọng

    # --- GRAD-CAM ---
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,  
}
if CONFIG["BACKBONE"] == "convnext":
    from src.utils.trainer_convnext import train_loop
else:
    from src.utils.trainer import train_loop

def preprocess_ham(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'image_path' not in df.columns:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
    
    if 'dx' in df.columns and 'label' not in df.columns:
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
    
    # In ra màn hình Top 5
    anchor_score = next((s for n, s in imps if config['ANCHOR_VALUE_NAME'] in n), 0)
    print("\n📊 TOP METADATA FEATURES:")
    for i, (name, score) in enumerate(imps[:8]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i+1}. {name}: {score:.5f} [{status}]")

    # Lưu CSV vào thư mục con (RUN_DIR) - SỬA Ở ĐÂY
    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    csv_name = f"ham10k_{config['SHORT_NAME']}_meta_imp.csv"
    out_path = os.path.join(run_dir, csv_name)
    
    pd.DataFrame(imps, columns=['Feature', 'Importance']).to_csv(out_path, index=False)
    print(f"💾 Đã lưu bảng xếp hạng Metadata vào: {out_path}")

def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    
    # --- TẠO THƯ MỤC CON (RUN_DIR) - SỬA Ở ĐÂY ---
    # Tên thư mục: {METADATA_MODE}_{SHORT_NAME} (vd: full_weighted_effb4)
    run_name = f"{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    run_dir = os.path.join(config['MODEL_OUT'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Cập nhật Config:
    # RUN_DIR dùng để lưu file chi tiết, MODEL_OUT giữ nguyên để lưu file tổng
    config['RUN_DIR'] = run_dir 
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print(f"📂 Thư mục gốc (Summary): {config['MODEL_OUT']}")
    print(f"📂 Thư mục chạy (Run Dir): {config['RUN_DIR']}")

    print("📂 Loading Data HAM10000...")
    raw_train = preprocess_ham(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_ham(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_ham(pd.read_csv(config['TEST_CSV']))

    # Metadata columns của HAM10000
    cat_cols = ['localization', 'sex']
    num_cols = ['age']

    if config['ANALYZE_METADATA']:
        analyze_feature_importance_only(raw_train, cat_cols, num_cols, config)

    print("🚀 Khởi tạo Dataset & Loader...")
    # Lưu ý: HAM10000Dataset đã được cập nhật dùng Albumentations ở bước trước
    train_ds = HAM10000Dataset(raw_train, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = HAM10000Dataset(raw_val, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = HAM10000Dataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    # Sampler cho tập Train (Do HAM10000 lệch nhãn)
    train_sampler = None
    if config['USE_SAMPLER']:
        targets = raw_train['label'].values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = torch.from_numpy(weights[targets]).double()
        train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler, num_workers=4)
    # Val/Test không cần sampler, chỉ cần shuffle=False
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config['UNFREEZE_SUBSTRINGS'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    
    # Focal Loss hoạt động tốt với dữ liệu mất cân bằng như HAM
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # Suffix chuẩn: ham10k_effb4
    log_suffix = f"ham10k_{config['SHORT_NAME']}"
    
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix=log_suffix
    )

if __name__ == '__main__':
    main(CONFIG)