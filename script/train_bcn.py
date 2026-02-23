import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold # THÊM IMPORT NÀY
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.losses import FocalLossBCE

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop
from src.utils.common import save_metadata_info

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

    # --- MODEL: RESNET50 ---
    'MODEL_NAME': 'resnet50',
    'SHORT_NAME': 'resnet50',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,

    'EPOCHS': 20,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    'METADATA_MODE': 'diag1', 
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
    'LOSS_TYPE': 'focal'
    'GRAD_CAM_TARGET_LAYER': 'layer4',
}
# chọn trainer theo MODEL_NAME (giữ nguyên workflow)
if "convnext" in CONFIG["MODEL_NAME"].lower():
    from src.utils.trainer_convnext import train_loop
else:
    from src.utils.trainer import train_loop


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

    # Xử lý diagnosis
    if 'diagnosis_1' in df.columns:
        df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
        df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
        df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    elif 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].astype(str).str.strip().str.lower()
        df = df[~df['diagnosis'].isin(['nan', '', 'none', 'null'])].copy()
        df['label'] = df['diagnosis'].apply(lambda x: 1 if 'malig' in x else 0)
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
    print("\n📊 BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG (Gợi ý từ RF):")
    feature_imp_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    # Tìm score của anchor_value (đảm bảo anchor_value là string)
    anchor_score = 0
    for n, s in feature_imp_list:
        if isinstance(anchor_value, str) and anchor_value in n:
            anchor_score = s
            break

    print(f"   ⚖️  Mốc chuẩn '{anchor_value}': {anchor_score:.5f}")
    for i, (name, score) in enumerate(feature_imp_list[:5]):
        status = "✅ MẠNH" if score > anchor_score else "⚠️ YẾU"
        print(f"   {i + 1}. {name}: {score:.5f} [{status}]")

    # Lưu kết quả vào thư mục con (RUN_DIR) - SỬA Ở ĐÂY
    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    csv_name = f"bcn20k_{config['SHORT_NAME']}_meta_imp.csv"
    out_path = os.path.join(run_dir, csv_name)
    
    pd.DataFrame(feature_imp_list, columns=['Feature', 'Importance']).to_csv(out_path, index=False)
    print(f"💾 Đã lưu bảng phân tích Metadata vào: {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])
    
    # --- TẠO THƯ MỤC CON (RUN_DIR) - SỬA Ở ĐÂY ---
    # Tên thư mục: {METADATA_MODE}_{SHORT_NAME} (vd: diag1_effb4)
    run_name = f"{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    run_dir = os.path.join(config['MODEL_OUT'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Cập nhật Config: 
    # RUN_DIR dùng để lưu file chi tiết, MODEL_OUT giữ nguyên để lưu file tổng
    config['RUN_DIR'] = run_dir 
    os.makedirs(config['MODEL_OUT'], exist_ok=True) 

    print(f"📂 Thư mục gốc (Summary): {config['MODEL_OUT']}")
    print(f"📂 Thư mục chạy (Run Dir): {config['RUN_DIR']}")

    print("📂 Loading Data BCN20000...")

    # 1. TẠO THƯ MỤC GỐC CV
    cv_run_name = f"CV5_{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    cv_dir = os.path.join(config['MODEL_OUT'], cv_run_name)
    os.makedirs(cv_dir, exist_ok=True)

    print("=" * 50)
    print(f"📂 Thư mục gốc CV: {cv_dir}")
    print(f"🔥 Device: {device}")
    print("=" * 50)

    # 2. LOAD & GỘP DATA
    print("📂 Đang tải và gộp dữ liệu...")
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

    # Loss Function (Weighted BCE hoặc Focal Loss)
    y_train = raw_train['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    
    if config['LOSS_TYPE'] == 'focal':
        criterion = FocalLossBCE(alpha=0.75, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_val, device=device)
        )

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
    # Gộp Train và Val thành 1 tập duy nhất (Dev set)
    df_cv = pd.concat([raw_train, raw_val]).reset_index(drop=True)
    print(f"📊 Tổng số mẫu chạy CV (Train+Val): {len(df_cv)}")

    # Tập Test giữ nguyên độc lập
    test_ds = DermoscopyDataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 3. STRATIFIED K-FOLD
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['SEED'])
    fold_results = []

    # 4. VÒNG LẶP TRAIN TỪNG FOLD
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_cv, df_cv['label'])):
        print(f"\n" + "★" * 40)
        print(f"🚀 BẮT ĐẦU FOLD {fold + 1}/{k_folds}")
        print("★" * 40)

        # Tạo thư mục riêng cho Fold
        fold_dir = os.path.join(cv_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        config['RUN_DIR'] = fold_dir  # Để train_loop lưu weights vào đây

        fold_train_df = df_cv.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_cv.iloc[val_idx].reset_index(drop=True)

        # Khởi tạo Datasets
        train_ds = DermoscopyDataset(fold_train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'],
                                     train=True)
        val_ds = DermoscopyDataset(fold_val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'],
                                   train=False)

        # Lưu Encoders của fold này
        meta_save_path = os.path.join(fold_dir, f"meta_info_fold{fold + 1}.pkl")
        save_metadata_info(meta_save_path, train_ds.encoders, train_ds.num_mean_std)

        # Loaders
        train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

        # ⚠️ KHỞI TẠO MODEL MỚI HOÀN TOÀN CHO MỖI FOLD
        model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
        set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

        # Loss, Optimizer, Scheduler
        y_train = fold_train_df['label'].values
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        pos_weight_val = weights[1] * config.get('META_CLASS_WEIGHT_BOOST', 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
        scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

        # Chạy Training cho Fold. (Trong train_loop cần đảm bảo tên file lưu checkpoint không bị trùng)
        _, _, test_metrics = train_loop(
            model, train_loader, val_loader, test_loader,
            config, criterion, optimizer, scheduler, device,
            log_suffix=f"fold_{fold + 1}"
        )

        test_metrics['fold'] = fold + 1
        fold_results.append(test_metrics)
        print(f"✅ Đã xong Fold {fold + 1}. AUC trên Test: {test_metrics['auc']:.4f}")

        # ==========================================================
        # 5. TỔNG HỢP VÀ IN KẾT QUẢ MEAN ± STD (Dùng cho Bảng 1)
        # ==========================================================
        print("\n" + "=" * 50)
        print("📊 BẢNG 1: KẾT QUẢ CROSS-VALIDATION MEAN ± STD TRÊN TẬP TEST")
        print("=" * 50)

        df_results = pd.DataFrame(fold_results)
        print(df_results[['fold', 'auc', 'acc', 'f1', 'precision', 'recall']])

        # 5.1. Tạo cấu trúc dữ liệu để lưu Summary (Mean ± Std)
        summary_data = []
        metrics = ['auc', 'acc', 'f1', 'precision', 'recall']

        print("\nTRUNG BÌNH ± ĐỘ LỆCH CHUẨN:")
        for metric in metrics:
            if metric in df_results.columns:
                mean_val = df_results[metric].mean()
                std_val = df_results[metric].std()

                # In ra màn hình
                print(f"{metric.upper():<10} : {mean_val:.4f} ± {std_val:.4f}")

                # Thêm vào list để lưu CSV
                summary_data.append({
                    'Metric': metric.upper(),
                    'Mean': round(mean_val, 4),
                    'Std': round(std_val, 4),
                    'Mean_±_Std': f"{mean_val:.4f} ± {std_val:.4f}"
                })

        # Chuyển thành DataFrame
        df_summary = pd.DataFrame(summary_data)

        # 5.2. Lưu ra file CSV
        # File 1: Lưu chi tiết 5 fold
        detail_csv_path = os.path.join(cv_dir, "cv5_ham10k_detail_results.csv")
        df_results.to_csv(detail_csv_path, index=False)

        # File 2: Lưu bảng Summary (Bảng 1)
        summary_csv_path = os.path.join(cv_dir, "cv5_ham10k_summary_table1.csv")
        df_summary.to_csv(summary_csv_path, index=False)

        print(f"\n💾 Đã lưu chi tiết từng fold tại  : {detail_csv_path}")
        print(f"💾 Đã lưu bảng Tóm tắt (Bảng 1) tại: {summary_csv_path}")
if __name__ == '__main__':
    main(CONFIG)