import os
import sys
import gc
import traceback

import numpy as np
import json
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
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
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✅ Đã tìm thấy GPU: {gpu_name}")
        return 'cuda'
    else:
        print("\n❌ KHÔNG TÌM THẤY GPU! Code sẽ chạy chậm trên CPU.")
        return 'cpu'


# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_train.csv',
    'VAL_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_val.csv',
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_ham10000',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,

    # --- MODEL & NAMING ---
    'MODEL_NAME': 'resnet50',
    'SHORT_NAME': 'resnet50_ham',

    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,  # Khuyên dùng 32 để tránh tràn RAM
    'EPOCHS': 20,
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
    df.columns = df.columns.str.strip().str.lower()

    if 'image_path' not in df.columns and 'image_id' in df.columns:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'

    # Ép kiểu age về số và điền giá trị thiếu
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].mean())

    if 'dx' in df.columns and 'label' not in df.columns:
        # 1: Ác tính (mel, bcc, akiec), 0: Lành tính
        df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    # --- ĐẢM BẢO CÓ CỘT LESION_ID ĐỂ CHIA GROUP ---
    if 'lesion_id' not in df.columns:
        df['lesion_id'] = df['image_id'] if 'image_id' in df.columns else df.index.astype(str)
    df['lesion_id'] = df['lesion_id'].fillna(df['image_path'])

    # --- 🚀 CHỐT CHẶN: XÓA CỘT ĐÁP ÁN ĐỂ CHỐNG TARGET LEAKAGE ---
    df = df.drop(columns=['dx', 'dx_type'], errors='ignore')

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
        print(f"   {i + 1}. {name}: {score:.5f} [{status}]")

    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    os.makedirs(run_dir, exist_ok=True)
    csv_name = f"ham10k_{config['SHORT_NAME']}_meta_imp.csv"
    out_path = os.path.join(run_dir, csv_name)

    pd.DataFrame(imps, columns=['Feature', 'Importance']).to_csv(out_path, index=False)
    print(f"💾 Đã lưu bảng xếp hạng Metadata vào: {out_path}")


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
    raw_train = preprocess_ham(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_ham(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_ham(pd.read_csv(config['TEST_CSV']))

    # Gộp Train và Val thành 1 tập duy nhất (Development set)
    df_cv = pd.concat([raw_train, raw_val]).reset_index(drop=True)
    print(f"📊 Tổng số mẫu chạy CV (Train+Val): {len(df_cv)}")
    print(f"📊 Tổng số mẫu Test (Hold-out): {len(raw_test)}")

    # ==========================================================
    # 🛡️ KIỂM TRA BẢO MẬT 1: RÒ RỈ TOÀN CỤC (CV vs TEST)
    # ==========================================================
    group_col = 'lesion_id' if 'lesion_id' in df_cv.columns else 'image_id'

    if group_col in df_cv.columns and group_col in raw_test.columns:
        cv_ids = set(df_cv[group_col].dropna().unique())
        test_ids = set(raw_test[group_col].dropna().unique())
        leakage = cv_ids.intersection(test_ids)

        if len(leakage) > 0:
            print(
                f"\n❌ [LỖI NGHIÊM TRỌNG] Phát hiện {len(leakage)} '{group_col}' bị trùng lặp giữa tập CV và tập Test!")
            print(f"Danh sách ID bị trùng (sample): {list(leakage)[:5]}")
            raise ValueError(
                f"DATA LEAKAGE DETECTED (CV vs TEST). Vui lòng kiểm tra lại quá trình chia file CSV gốc. Đã dừng huấn luyện!")
        else:
            print(f"✅ CHỐT CHẶN 1: Tuyệt đối an toàn. Không có rò rỉ bệnh nhân từ tập CV sang tập Test.")

    if config.get('ANALYZE_METADATA'):
        categorical_cols = ['sex', 'localization']
        numeric_cols = ['age']
        analyze_feature_importance_only(raw_train, categorical_cols, numeric_cols, config)

    # ==========================================================
    # 3. THIẾT LẬP STRATIFIED GROUP K-FOLD
    # ==========================================================
    k_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=config['SEED'])
    fold_results = []

    # ==========================================================
    # 4. VÒNG LẶP HUẤN LUYỆN QUA TỪNG FOLD
    # ==========================================================
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df_cv, y=df_cv['label'], groups=df_cv[group_col])):
        print(f"\n" + "★" * 40)
        print(f"🚀 BẮT ĐẦU FOLD {fold + 1}/{k_folds}")
        print("★" * 40)

        fold_dir = os.path.join(cv_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        config['RUN_DIR'] = fold_dir

        fold_train_df = df_cv.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_cv.iloc[val_idx].reset_index(drop=True)

        # ==========================================================
        # 🛡️ KIỂM TRA BẢO MẬT 2: RÒ RỈ CỤC BỘ (TRAIN vs VAL)
        # ==========================================================
        train_ids = set(fold_train_df[group_col].dropna().unique())
        val_ids = set(fold_val_df[group_col].dropna().unique())
        fold_leakage = train_ids.intersection(val_ids)

        if len(fold_leakage) > 0:
            raise ValueError(
                f"❌ [LỖI NGHIÊM TRỌNG] Data Leakage tại Fold {fold + 1}! GroupKFold hoạt động không đúng. Dừng chương trình!")
        else:
            print(f"   ✅ CHỐT CHẶN 2: Fold {fold + 1} an toàn tuyệt đối (0 ID trùng lặp).")

        # Khởi tạo Dataset của HAM10000
        train_ds = HAM10000Dataset(fold_train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'],
                                   train=True)
        val_ds = HAM10000Dataset(fold_val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'],
                                 train=False)
        test_ds = HAM10000Dataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'],
                                  train=False)

        # 🚀 GÁN BỘ ENCODER & STATS CỦA TRAIN SANG VAL VÀ TEST
        val_ds.encoders = train_ds.encoders
        val_ds.num_mean_std = train_ds.num_mean_std

        test_ds.encoders = train_ds.encoders
        test_ds.num_mean_std = train_ds.num_mean_std

        # Lưu Encoders của fold này ra file
        meta_save_path = os.path.join(fold_dir, f"meta_info_fold{fold + 1}.pkl")
        save_metadata_info(meta_save_path, train_ds.encoders, train_ds.num_mean_std)

        # Cấu hình Sampler cho mất cân bằng dữ liệu
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
        test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

        # Khởi tạo Model MỚI CHO FOLD
        model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
        set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_SUBSTRINGS', []))

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
        criterion = FocalLossBCE(alpha=0.75, gamma=2.0)
        scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

        # Chạy Huấn luyện
        _, _, test_metrics = train_loop(
            model, train_loader, val_loader, test_loader,
            config, criterion, optimizer, scheduler, device,
            log_suffix=f"fold_{fold + 1}"
        )

        test_metrics['fold'] = fold + 1
        fold_results.append(test_metrics)
        print(f"✅ Đã xong Fold {fold + 1}. AUC trên tập Test: {test_metrics['auc']:.4f}")

    # ==========================================================
    # 5. TỔNG HỢP VÀ IN KẾT QUẢ MEAN ± STD
    # ==========================================================
    print("\n" + "=" * 50)
    print("📊 BẢNG 1: KẾT QUẢ CROSS-VALIDATION MEAN ± STD TRÊN TẬP TEST")
    print("=" * 50)

    df_results = pd.DataFrame(fold_results)
    print(df_results[['fold', 'auc', 'acc', 'f1', 'precision', 'recall']])

    summary_data = []
    metrics = ['auc', 'acc', 'f1', 'precision', 'recall']

    print("\nTRUNG BÌNH ± ĐỘ LỆCH CHUẨN:")
    for metric in metrics:
        if metric in df_results.columns:
            mean_val = df_results[metric].mean()
            std_val = df_results[metric].std()

            print(f"{metric.upper():<10} : {mean_val:.4f} ± {std_val:.4f}")

            summary_data.append({
                'Metric': metric.upper(),
                'Mean': round(mean_val, 4),
                'Std': round(std_val, 4),
                'Mean_±_Std': f"{mean_val:.4f} ± {std_val:.4f}"
            })

    df_summary = pd.DataFrame(summary_data)

    detail_csv_path = os.path.join(cv_dir, f"cv5_{config['SHORT_NAME']}_detail_results.csv")
    df_results.to_csv(detail_csv_path, index=False)

    summary_csv_path = os.path.join(cv_dir, f"cv5_{config['SHORT_NAME']}_summary_table1.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print(f"\n💾 Đã lưu chi tiết từng fold tại  : {detail_csv_path}")
    print(f"💾 Đã lưu bảng Tóm tắt (Bảng 1) tại: {summary_csv_path}")


if __name__ == '__main__':
    modes_to_run = ['diag1', 'full', 'full_weighted', 'late_fusion']

    print("\n" + "★" * 60)
    print("🌙 CHẾ ĐỘ CHẠY QUA ĐÊM (OVERNIGHT TRAINING) HAM10000 ĐÃ KÍCH HOẠT")
    print("★" * 60 + "\n")

    for mode in modes_to_run:
        try:
            print("\n" + "=" * 60)
            print(f"🚀 [TIẾN TRÌNH] ĐANG CHẠY MODE: {mode.upper()}")
            print("=" * 60 + "\n")

            CONFIG['METADATA_MODE'] = mode
            main(CONFIG)

            print(f"\n✅ [THÀNH CÔNG] ĐÃ XONG MODE: {mode.upper()}")

        except Exception as e:
            print(f"\n❌ [LỖI NGHIÊM TRỌNG] Mode {mode.upper()} gặp sự cố!")
            print(f"Chi tiết lỗi: {e}")
            traceback.print_exc()
            print("⏭️ ĐANG CHUYỂN SANG MODE TIẾP THEO...\n")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🧹 Đã dọn dẹp bộ nhớ GPU.\n")

    print("\n" + "🎉" * 20)
    print("ĐÃ KẾT THÚC TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN HAM10000!")
    print("🎉" * 20)