import json
import os
import sys
import gc
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc của dự án để import các module nội bộ
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
    'VAL_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_val.csv',
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,

    # --- MODEL---
    'MODEL_NAME': 'resnet50',
    'SHORT_NAME': 'resnet50_bcn',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32, # 32 là mức an toàn cho ảnh 224x224 trên hầu hết GPU

    'EPOCHS': 20,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    # --- METADATA ---
    'METADATA_MODE': 'late_fusion',
    'METADATA_FEATURE_BOOST': 2.0,
    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',

    # Cấu trúc unfreeze chuẩn cho ResNet
    'UNFREEZE_KEYWORDS': ['layers', 'blocks', 'norm', 'conv_head', 'features', 'stem'],

    'ACCUM_STEPS': 1,

    # --- CẤU HÌNH PHÂN TÍCH ---
    'ANALYZE_METADATA': True,
    'ANCHOR_VALUE_NAME': 'palms/soles',

    # --- CẤU HÌNH GRAD-CAM ---
    'ENABLE_GRAD_CAM': True,
    'GRAD_CAM_FREQ': 5,
}


def preprocess_bcn(df):
    """Tiền xử lý dữ liệu dựa trên thực tế file CSV của BCN20000"""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    # Xử lý diagnosis để tạo nhãn
    if 'diagnosis_1' in df.columns:
        df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
        df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
        df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    elif 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].astype(str).str.strip().str.lower()
        df = df[~df['diagnosis'].isin(['nan', '', 'none', 'null'])].copy()
        df['label'] = df['diagnosis'].apply(lambda x: 1 if 'malig' in x else 0)

    # --- PHỤC HỒI CHỐT CHẶN: ĐẢM BẢO CÓ LESION_ID & KHÔNG BỊ NAN ---
    if 'lesion_id' not in df.columns:
        if 'patient_id' in df.columns:
            df['lesion_id'] = df['patient_id']
        else:
            df['lesion_id'] = df['isic_id'] if 'isic_id' in df.columns else df.index.astype(str)
    df['lesion_id'] = df['lesion_id'].fillna(df['image_path'])

    # --- PHỤC HỒI CHỐT CHẶN: XÓA CỘT ĐÁP ÁN ĐỂ CHỐNG TARGET LEAKAGE ---
    df = df.drop(columns=['diagnosis', 'diagnosis_1', 'benign_malignant'], errors='ignore')

    return df


def analyze_feature_importance_only(train_df, categorical_cols, numeric_cols, config):
    print(f"\n🤖 [Analysis] Đang chạy Random Forest phân tích Metadata BCN20000...")
    valid_cat = [c for c in categorical_cols if c in train_df.columns]
    valid_num = [c for c in numeric_cols if c in train_df.columns]

    if not valid_cat and not valid_num:
        print("⚠️ Không tìm thấy cột Metadata hợp lệ để phân tích.")
        return

    y = train_df['label'].values
    if len(np.unique(y)) < 2:
        print("⚠️ Không thể phân tích Metadata vì chỉ có 1 lớp.")
        return

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

    feature_names = meta_df.columns
    importances = rf.feature_importances_
    anchor_value = config.get('ANCHOR_VALUE_NAME', 'palms/soles')

    feature_imp_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n📊 BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG (Gợi ý từ RF):")

    anchor_score = 0
    for n, s in feature_imp_list:
        if isinstance(anchor_value, str) and anchor_value in n:
            anchor_score = s
            break

    print(f"   ⚖️  Mốc chuẩn '{anchor_value}': {anchor_score:.5f}")
    for i, (name, score) in enumerate(feature_imp_list[:10]):
        status = "✅ MẠNH" if score >= anchor_score else "⚠️ YẾU"
        print(f"   {i + 1}. {name:<25}: {score:.5f} [{status}]")

    run_dir = config.get('RUN_DIR', config['MODEL_OUT'])
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(feature_imp_list, columns=['Feature', 'Importance']).to_csv(
        os.path.join(run_dir, f"{config['SHORT_NAME']}_meta_importance.csv"), index=False)


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
    print("📂 Đang tải dữ liệu BCN20000...")
    raw_train = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    raw_val = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    raw_test = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    # Gộp Train và Val thành 1 tập duy nhất (Development set)
    df_cv = pd.concat([raw_train, raw_val]).reset_index(drop=True)
    print(f"📊 Tổng số mẫu chạy CV (Train+Val): {len(df_cv)}")
    print(f"📊 Tổng số mẫu Test (Hold-out): {len(raw_test)}")

    # ==========================================================
    # 🛡️ KIỂM TRA BẢO MẬT 1: RÒ RỈ TOÀN CỤC (CV vs TEST)
    # ==========================================================
    if 'lesion_id' in df_cv.columns:
        group_col = 'lesion_id'
    elif 'patient_id' in df_cv.columns:
        group_col = 'patient_id'
    else:
        group_col = 'isic_id'

    if group_col in df_cv.columns and group_col in raw_test.columns:
        cv_ids = set(df_cv[group_col].dropna().unique())
        test_ids = set(raw_test[group_col].dropna().unique())
        leakage = cv_ids.intersection(test_ids)

        if len(leakage) > 0:
            print(f"\n❌ [LỖI NGHIÊM TRỌNG] Phát hiện {len(leakage)} '{group_col}' bị trùng lặp giữa tập CV và tập Test!")
            print(f"Danh sách ID bị trùng (sample): {list(leakage)[:5]}")
            raise ValueError(f"DATA LEAKAGE DETECTED (CV vs TEST). Vui lòng kiểm tra lại quá trình chia file CSV gốc. Đã dừng huấn luyện!")
        else:
            print(f"✅ CHỐT CHẶN 1: Tuyệt đối an toàn. Không có rò rỉ bệnh nhân từ tập CV sang tập Test.")

    if config.get('ANALYZE_METADATA'):
        categorical_cols = ['sex', 'anatom_site_general']
        numeric_cols = ['age_approx']
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
            raise ValueError(f"❌ [LỖI NGHIÊM TRỌNG] Data Leakage tại Fold {fold + 1}! GroupKFold hoạt động không đúng. Dừng chương trình!")
        else:
            print(f"   ✅ CHỐT CHẶN 2: Fold {fold + 1} an toàn tuyệt đối (0 ID trùng lặp).")

        # Khởi tạo Dataset của BCN20000 (DermoscopyDataset)
        train_ds = DermoscopyDataset(fold_train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=True)
        val_ds = DermoscopyDataset(fold_val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
        test_ds = DermoscopyDataset(raw_test, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)

        # 🚀 GÁN BỘ ENCODER & STATS CỦA TRAIN SANG VAL VÀ TEST
        val_ds.encoders = train_ds.encoders
        val_ds.num_mean_std = train_ds.num_mean_std

        test_ds.encoders = train_ds.encoders
        test_ds.num_mean_std = train_ds.num_mean_std

        # Lưu Encoders của fold này ra file
        meta_save_path = os.path.join(fold_dir, f"meta_info_fold{fold + 1}.pkl")
        save_metadata_info(meta_save_path, train_ds.encoders, train_ds.num_mean_std)

        # DataLoader (Sử dụng shuffle=True, bỏ Sampler vì đã dùng FocalLossBCE)
        train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

        # Khởi tạo Model MỚI CHO FOLD
        model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
        set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS', []))

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
    print("🌙 CHẾ ĐỘ CHẠY QUA ĐÊM (OVERNIGHT TRAINING) BCN20000 ĐÃ KÍCH HOẠT")
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
    print("ĐÃ KẾT THÚC TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN BCN20000!")
    print("🎉" * 20)