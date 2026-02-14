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
from src.utils.trainer import train_loop  # Lưu ý: Cần cập nhật hàm này để gọi Grad-CAM


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
    'VAL_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_val.csv',
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,

    # --- MODEL: RESNET50 ---
    'MODEL_NAME': 'resnet50',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,

    'EPOCHS': 20,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA ---
    'METADATA_MODE': 'full_weighted',
    'METADATA_FEATURE_BOOST': 5.0,
    'META_CLASS_WEIGHT_BOOST': 1.0,
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',

    'UNFREEZE_KEYWORDS': ['conv_head', 'bn2', 'blocks.6', 'blocks.5'],

    'ACCUM_STEPS': 1,

    # --- CẤU HÌNH PHÂN TÍCH (KHÔNG CAN THIỆP DỮ LIỆU) ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'palms/soles',

    # --- CẤU HÌNH GRAD-CAM (MỚI) ---
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,  # Chạy visualization mỗi 5 epoch
    'GRAD_CAM_TARGET_LAYER': 'layer4',  # Lớp Conv cuối của ResNet50
}


def preprocess_bcn(df):
    """Làm sạch dữ liệu cơ bản và tạo nhãn chuẩn"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    return df


# ==============================================================================
# PHÂN TÍCH ĐỘ QUAN TRỌNG
# ==============================================================================
def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, anchor_value='palms/soles'):
    print(f"\n🤖 Đang chạy phân tích Metadata (Tham khảo)...")
    valid_cat = [c for c in categorical_cols if c in train_df.columns]
    valid_num = [c for c in numeric_cols if c in train_df.columns]
    if not valid_cat and not valid_num: return {}

    meta_df = pd.DataFrame()
    if valid_cat:
        temp_cat = train_df[valid_cat].fillna('unknown')
        meta_cat_ohe = pd.get_dummies(temp_cat, columns=valid_cat, prefix_sep='===')
        meta_df = pd.concat([meta_df, meta_cat_ohe], axis=1)
    if valid_num:
        temp_num = train_df[valid_num].copy()
        imputer = SimpleImputer(strategy='mean')
        temp_num_filled = pd.DataFrame(imputer.fit_transform(temp_num), columns=valid_num, index=temp_num.index)
        meta_df = pd.concat([meta_df, temp_num_filled], axis=1)

    y = train_df['label'].values
    feature_names = meta_df.columns.tolist()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(meta_df, y)
    importances = rf.feature_importances_

    print("\n📊 BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG (Gợi ý từ RF):")
    feature_imp_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    anchor_score = next((s for n, s in feature_imp_list if anchor_value in n), 0)

    print(f"   ⚖️  Mốc chuẩn '{anchor_value}': {anchor_score:.5f}")
    for i, (name, score) in enumerate(feature_imp_list[:5]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i + 1}. {name.split('===')[-1] if '===' in name else name}: {score:.5f} [{status}]")

    full_structure = {col: [str(v) for v in train_df[col].fillna('unknown').unique()] for col in valid_cat}
    return full_structure


# ==============================================================================
# MAIN
# ==============================================================================
def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print("📂 Đang tải dữ liệu...")
    raw_train = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    numeric_cols = ['age_approx']
    categorical_cols = ['anatom_site_general', 'anatom_site_special', 'diagnosis_confirm_type', 'sex']

    if config.get('ANALYZE_METADATA', False):
        full_structure = analyze_feature_importance_only(raw_train, categorical_cols, numeric_cols,
                                                         config['ANCHOR_VALUE_NAME'])
        json_path = os.path.join(config['MODEL_OUT'], 'feature_structure_full.json')
        with open(json_path, 'w') as f:
            json.dump(full_structure, f, indent=4)
        print(f"💾 Đã lưu cấu trúc dữ liệu vào: {json_path}")

    print("🚀 TRAINING VỚI FULL FEATURES (X1-X18)...")
    train_df, val_df, test_df = raw_train, raw_val, raw_test

    train_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
    val_ds = DermoscopyDataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_ds = DermoscopyDataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

    y_train = train_df['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # BẮT ĐẦU TRAINING
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix="resnet_full_gradcam"
    )


if __name__ == '__main__':
    main(CONFIG)