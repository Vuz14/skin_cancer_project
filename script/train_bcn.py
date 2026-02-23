import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Thêm đường dẫn gốc của dự án để import các module nội bộ
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.losses import FocalLossBCE
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
 
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effb4',
    
    'IMG_SIZE': 300, 
    'BATCH_SIZE': 16,

    'EPOCHS': 15,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    'METADATA_MODE': 'late_fusion',
    'METADATA_FEATURE_BOOST': 2.0,
    'META_CLASS_WEIGHT_BOOST': 1.2, 
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',

    'UNFREEZE_KEYWORDS': ['conv_head', 'blocks.6', 'blocks.7'],
    'ACCUM_STEPS': 1,
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'palms/soles',
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,  
    'LOSS_TYPE': 'focal'
}

def preprocess_bcn(df):
    """Tiền xử lý dữ liệu dựa trên thực tế file CSV"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    diag_col = 'diagnosis_1' if 'diagnosis_1' in df.columns else 'diagnosis'
    
    if diag_col in df.columns:
        # 1. Hiển thị mẫu dữ liệu gốc
        raw_values = df[diag_col].unique().tolist()
        print(f"🔍 Dữ liệu gốc trong {diag_col}: {raw_values}")
        
        # 2. Chuẩn hóa văn bản
        df[diag_col] = df[diag_col].astype(str).str.strip().str.lower()
        
        # 3. Loại bỏ Indeterminate (Mẫu không xác định)
        df = df[df[diag_col] != 'indeterminate'].reset_index(drop=True)
        
        # 4. Gán nhãn: Malignant=1, Còn lại (Benign)=0
        # malignant_list bao gồm 'malignant' để khớp với dữ liệu 'Malignant' trong CSV
        malignant_list = ['malignant', 'mel', 'bcc', 'scc', 'carcinoma']
        
        df['label'] = df[diag_col].apply(
            lambda x: 1 if any(m in x for m in malignant_list) else 0
        )
    
    # Kiểm tra số lượng nhãn
    print(f"📊 Các lớp thực tế: {df['label'].unique().tolist()}")
    print(f"📊 Phân phối nhãn:\n{df['label'].value_counts()}")
    
    return df

# ==============================================================================
# PHÂN TÍCH ĐỘ QUAN TRỌNG
# ==============================================================================
def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, config):
    print(f"\n🤖 [Analysis] Đang chạy Random Forest phân tích Metadata...")
    valid_cat = [c for c in categorical_cols if c in train_df.columns]
    valid_num = [c for c in numeric_cols if c in train_df.columns]
    
    if not valid_cat and not valid_num: return

    y = train_df['label'].values
    if len(np.unique(y)) < 2: 
        print("⚠️ Không thể phân tích Metadata vì chỉ có 1 lớp."); return

    meta_df = pd.DataFrame()
    if valid_cat:
        temp_cat = train_df[valid_cat].fillna('unknown')
        meta_df = pd.concat([meta_df, pd.get_dummies(temp_cat, columns=valid_cat, prefix_sep='=')], axis=1)
    
    if valid_num:
        temp_num = train_df[valid_num].copy()
        imputer = SimpleImputer(strategy='mean')
        temp_num_filled = pd.DataFrame(imputer.fit_transform(temp_num), columns=valid_num)
        meta_df = pd.concat([meta_df, temp_num_filled], axis=1)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(meta_df, y)
    
    feature_imp_list = sorted(zip(meta_df.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)

    print("\n📊 TOP 10 METADATA IMPORTANCE:")
    anchor_score = next((s for n, s in feature_imp_list if config['ANCHOR_VALUE_NAME'] in n), 0)
    for i, (name, score) in enumerate(feature_imp_list[:10]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i + 1}. {name}: {score:.5f} [{status}]")

    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    pd.DataFrame(feature_imp_list, columns=['Feature', 'Importance']).to_csv(os.path.join(run_dir, "meta_importance.csv"), index=False)

# ==============================================================================
# MAIN
# ==============================================================================
def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    
    run_name = f"{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    run_dir = os.path.join(config['MODEL_OUT'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    config['RUN_DIR'] = run_dir 

    print("📂 Loading Data BCN20000...")
    raw_train = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    numeric_cols = ['age_approx']
    categorical_cols = ['anatom_site_general', 'anatom_site_special', 'diagnosis_confirm_type', 'sex']

    if config.get('ANALYZE_METADATA', False):
        analyze_feature_importance_only(raw_train, categorical_cols, numeric_cols, config)

    print("🚀 Khởi tạo Dataset & Loader...")
    train_ds = DermoscopyDataset(raw_train, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = DermoscopyDataset(raw_val, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = DermoscopyDataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # Khởi tạo model - Ngắt metadata nếu là diag1
    use_meta_flag = (config['METADATA_MODE'] != 'diag1')
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols), use_metadata=use_meta_flag).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

    # Xử lý Class Weights an toàn
    y_train = raw_train['label'].values
    classes = np.unique(y_train)
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    else:
        pos_weight_val = 1.0

    if config['LOSS_TYPE'] == 'focal':
        criterion = FocalLossBCE(alpha=0.75, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device, log_suffix=f"bcn20k_{config['SHORT_NAME']}")

if __name__ == '__main__':
    main(CONFIG)