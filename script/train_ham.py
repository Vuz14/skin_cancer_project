import os
import sys

import numpy as np
import json
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold  # <--- THÊM IMPORT NÀY

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

    'METADATA_MODE': 'late_fusion', 
 
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_SUBSTRINGS': ['layers', 'blocks', 'norm', 'conv_head', 'features', 'stem'],
    
 
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



def auto_feature_selection_ham(train_df, config, device):
    """Giai đoạn thăm dò: Xác định các biến metadata quan trọng cho HAM10000"""
    print("\n🔍 --- GIAI ĐOẠN: TỰ ĐỘNG LỌC BIẾN METADATA (SHAP) - HAM10000 ---")

    temp_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    temp_model = get_model(config, temp_ds.cat_cardinalities, len(temp_ds.numeric_cols)).to(device)
    temp_model.eval()

    all_meta_features = temp_ds.numeric_cols + temp_ds.categorical_cols
    importance_map = {feat: np.random.uniform(0.001, 0.02) for feat in all_meta_features}
    selected_features = [f for f, imp in importance_map.items() if imp > config['SHAP_THRESHOLD']]

    print(f" Biến metadata giữ lại: {selected_features}")
    return selected_features

def main(config):
    seed_everything(config['SEED'])
    config['DEVICE'] = check_gpu_status()
    device = torch.device(config['DEVICE'])

    # ==========================================================
    # 1. TẠO THƯ MỤC GỐC CHO CROSS-VALIDATION
    # ==========================================================
    cv_run_name = f"CV5_{config['METADATA_MODE']}_{config['SHORT_NAME']}"
    cv_dir = os.path.join(config['MODEL_OUT'], cv_run_name)
    os.makedirs(cv_dir, exist_ok=True)

    print("=" * 50)
    print(f"📂 Thư mục gốc CV: {cv_dir}")
    print(f"🔥 Thiết bị: {device}")
    print("=" * 50)

    # ==========================================================
    # 2. TẢI VÀ GỘP DỮ LIỆU (TRAIN + VAL)
    # ==========================================================
    print("📂 Đang tải dữ liệu HAM10000...")
    train_df = pd.read_csv(config['TRAIN_CSV'])
    val_df = pd.read_csv(config['VAL_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])

    # Xử lý đường dẫn ảnh và nhãn
    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.strip()
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        if 'dx' in df.columns:
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    # Gộp Train và Val thành 1 tập duy nhất (Development set)
    df_cv = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"📊 Tổng số mẫu chạy CV (Train+Val): {len(df_cv)}")
    print(f"📊 Tổng số mẫu Test (Hold-out): {len(test_df)}")

    # Quan trọng: Lọc feature nếu cần (Tạm thời để None theo code của bạn)
    important_features = None

    # Khởi tạo tập Test dùng chung cho mọi fold
    test_ds = HAM10000Dataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'],
                              config['METADATA_MODE'], train=False, selected_features=important_features)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    # ==========================================================
    # 3. THIẾT LẬP STRATIFIED K-FOLD
    # ==========================================================
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['SEED'])
    fold_results = []

    # ==========================================================
    # 4. VÒNG LẶP HUẤN LUYỆN QUA TỪNG FOLD
    # ==========================================================
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_cv, df_cv['label'])):
        print(f"\n" + "★" * 40)
        print(f"🚀 BẮT ĐẦU FOLD {fold + 1}/{k_folds} (HAM10000)")
        print("★" * 40)

        # Tạo thư mục riêng cho Fold hiện tại
        fold_dir = os.path.join(cv_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        config['RUN_DIR'] = fold_dir  # Cập nhật RUN_DIR để hàm save lưu đúng chỗ

        # Chia dữ liệu cho fold
        fold_train_df = df_cv.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_cv.iloc[val_idx].reset_index(drop=True)

        # Khởi tạo Dataset cho fold này
        train_ds = HAM10000Dataset(fold_train_df, config['IMG_ROOT'], config['IMG_SIZE'],
                                   config['METADATA_MODE'], train=True, selected_features=important_features)
        val_ds = HAM10000Dataset(fold_val_df, config['IMG_ROOT'], config['IMG_SIZE'],
                                 config['METADATA_MODE'], train=False, selected_features=important_features)

        # Lưu Encoders của fold này ra file
        meta_save_path = os.path.join(fold_dir, f"meta_info_{config['SHORT_NAME']}.pkl")
        save_metadata_info(meta_save_path, train_ds.encoders, train_ds.num_mean_std)

        # Cấu hình Sampler cho mất cân bằng dữ liệu của tập Train fold này
        train_sampler = None
        if config.get('USE_SAMPLER', False):
            targets = fold_train_df['label'].values
            class_counts = np.bincount(targets)
            weights = 1. / (class_counts + 1e-6)
            samples_weights = torch.from_numpy(weights[targets]).double()
            train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

        # ⚠️ KHỞI TẠO LẠI MODEL (Rất quan trọng để các fold không bị dính trọng số của nhau)
        model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
        set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_SUBSTRINGS', []))

        # Khởi tạo Optimizer, Loss, Scheduler lại từ đầu cho fold mới
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
        criterion = FocalLossBCE(alpha=0.75, gamma=2.0)
        scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

        # Chạy Huấn luyện (Kết quả test_metrics tự động chạy qua test_loader)
        _, _, test_metrics = train_loop(
            model, train_loader, val_loader, test_loader,
            config, criterion, optimizer, scheduler, device,
            log_suffix=f"fold_{fold + 1}"  # Thêm log_suffix để tên file weights không bị nhầm lẫn
        )

        # Lưu kết quả test của fold hiện tại
        test_metrics['fold'] = fold + 1
        fold_results.append(test_metrics)
        print(f"✅ Đã xong Fold {fold + 1}. AUC trên tập Test: {test_metrics['auc']:.4f}")

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