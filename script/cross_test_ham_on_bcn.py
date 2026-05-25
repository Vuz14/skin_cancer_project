import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.trainer import evaluate
from src.utils.common import load_metadata_info, seed_everything

# --- CẤU HÌNH ---
CONFIG = {
    # 1. DỮ LIỆU ĐÍCH (BCN20000)
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-paper-preprocessed',
    'SEED': 42,
    # 2. MODEL NGUỒN (HAM10000)
    # Trỏ đến checkpoint .pt của HAM
    'CHECKPOINT_PATH': r'D:\skin_cancer_project\checkpoint_ham10000\CV5_strategy3_effnet_b4_ham10000\fold_3\best_strategy3_fold_3.pt',

    # File .pkl chứa encoder của HAM
    'META_INFO_PATH': r'D:\skin_cancer_project\checkpoint_ham10000\CV5_strategy3_effnet_b4_ham10000\fold_3\meta_info_fold3.pkl',

    # 3. THÔNG SỐ KHÁC (Khớp với cấu hình lúc train HAM)
    'DEVICE': 'cuda',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'strategy3',
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'BATCH_SIZE': 32,
    'PRETRAINED': True,
    'METADATA_FEATURE_BOOST': 2.0
}

def map_bcn_to_ham(bcn_df):
    print("🔄 Đang mapping dữ liệu BCN sang format HAM...")
    df = bcn_df.copy()

    # Tạo cột image_path nếu chưa có
    if 'image_path' not in df.columns:
        if 'isic_id' in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        elif 'image_id' in df.columns:
            df['image_path'] = df['image_id'].astype(str) + '.jpg'

    # Mapping features
    df = df.rename(columns={'age_approx': 'age', 'anatom_site_general': 'localization'})
    if 'sex' in df.columns: df['sex'] = df['sex'].astype(str).str.lower()

    loc_mapping = {
        'anterior torso': 'chest', 'head/neck': 'neck', 'lateral torso': 'trunk',
        'lower extremity': 'lower extremity', 'oral/genital': 'genital',
        'palms/soles': 'acral', 'posterior torso': 'back',
        'upper extremity': 'upper extremity', 'nan': 'unknown'
    }
    df['localization'] = df['localization'].map(loc_mapping).fillna('unknown')

    print("✅ Mapping features hoàn tất.")
    return df


def main():
    seed_everything(42)
    device = torch.device(CONFIG['DEVICE'])

    print("📂 Loading Metadata Encoders của HAM...")
    if CONFIG['METADATA_MODE'] != 'strategy1':
        if not os.path.exists(CONFIG['META_INFO_PATH']):
            print(f"❌ Lỗi: Không tìm thấy file {CONFIG['META_INFO_PATH']}")
            return
        encoders, num_stats = load_metadata_info(CONFIG['META_INFO_PATH'])
    else:
        encoders, num_stats = None, None

    print("📂 Loading BCN Data...")
    test_df = pd.read_csv(CONFIG['TEST_CSV'])

    if 'label' not in test_df.columns:
        diag_col = 'diagnosis_1' if 'diagnosis_1' in test_df.columns else 'diagnosis'
        malignant_list = ['malignant', 'mel', 'bcc', 'scc', 'melanoma', 'basal cell', 'squamous cell', 'carcinoma']
        test_df['label'] = test_df[diag_col].astype(str).str.lower().apply(
            lambda x: 1 if any(m in x for m in malignant_list) else 0
        )

    # --- DEBUG: KIỂM TRA PHÂN PHỐI NHÃN ---
    label_counts = test_df['label'].value_counts()
    print("\n📊 THỐNG KÊ NHÃN TRONG TẬP TEST:")
    print(label_counts)
    if len(label_counts) < 2:
        print("⚠️ CẢNH BÁO: Tập test chỉ có 1 loại nhãn! AUC sẽ luôn bằng 0.")
    # --------------------------------------

    mapped_test_df = map_bcn_to_ham(test_df)

    test_ds = HAM10000Dataset(
        df=mapped_test_df,
        img_root=CONFIG['IMG_ROOT'],
        img_size=CONFIG['IMG_SIZE'],
        metadata_mode=CONFIG['METADATA_MODE'],
        train=False,
        external_encoders=encoders,
        external_stats=num_stats
    )

    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

    print("🤖 Loading Model...")
    cat_cardinalities = test_ds.cat_cardinalities if encoders else {}
    num_numeric = len(test_ds.numeric_cols) if encoders else 0

    model = get_model(CONFIG, cat_cardinalities, num_numeric).to(device)

    if not os.path.exists(CONFIG['CHECKPOINT_PATH']):
        print(f"❌ Không tìm thấy file checkpoint tại: {CONFIG['CHECKPOINT_PATH']}")
        return

    checkpoint = torch.load(CONFIG['CHECKPOINT_PATH'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("🚀 Đang chạy đánh giá chéo...")
    results = evaluate(model, test_loader, device=device)

    print("\n" + "=" * 30)
    print(f"KẾT QUẢ TEST HAM MODEL TRÊN TẬP BCN ({CONFIG['METADATA_MODE']})")
    print("=" * 30)
    print(f"AUC       : {results['auc']:.4f}")
    print(f"Accuracy  : {results['acc']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Specificity: {results.get('spec', 0):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
