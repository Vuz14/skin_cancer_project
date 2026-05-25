import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc để import các module từ src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Dùng Dataset của BCN (DermoscopyDataset) vì Model "hiểu" format của BCN
from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.trainer import evaluate
from src.utils.common import load_metadata_info, seed_everything

# ==============================================================================
# 1. CẤU HÌNH (CONFIG)
# ==============================================================================
CONFIG = {
    # 1. DỮ LIỆU ĐÍCH (HAM10000)
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-paper-preprocessed',
    'SEED': 42,
    # 2. MODEL NGUỒN (BCN20000)
    # Lưu ý: Thay đổi đường dẫn này trỏ đến file .pt tốt nhất của BCN (ví dụ trong thư mục fold_1 nếu dùng CV5)
    'CHECKPOINT_PATH': r'D:\skin_cancer_project\checkpoint_bcn20000\CV5_strategy3_effnet_b4_bcn20000\fold_4\best_strategy3_fold_4.pt',

    # File .pkl chứa Metadata Encoder và Stats được sinh ra lúc train model BCN
    'META_INFO_PATH': r'D:\skin_cancer_project\checkpoint_bcn20000\CV5_strategy3_effnet_b4_bcn20000\fold_4\meta_info_fold4.pkl',

    # 3. THÔNG SỐ KHÁC (Phải khớp chính xác với lúc bạn train model BCN)
    'DEVICE': 'cuda', # hoặc 'cpu'
    'IMG_SIZE': 224,
    'METADATA_MODE': 'strategy3',
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'BATCH_SIZE': 32,
    'PRETRAINED': True,
    'METADATA_FEATURE_BOOST': 1.0
}

# ==============================================================================
# 2. HÀM MAPPING: HAM -> BCN
# ==============================================================================
def map_ham_to_bcn(ham_df):
    """
    Biến đổi dữ liệu HAM (nguồn) -> Format của BCN (đích)
    BCN Columns cần: age_approx, sex, anatom_site_general
    """
    print("🔄 Đang mapping dữ liệu HAM sang format BCN...")
    df = ham_df.copy()

    # --- TẠO CỘT IMAGE_PATH NẾU CHƯA CÓ ---
    # HAM dataset thường có image_id, cần tạo image_path để loader đọc được
    if 'image_path' not in df.columns:
        if 'image_id' in df.columns:
            df['image_path'] = df['image_id'].astype(str) + '.jpg'
    # --------------------------------------

    # 1. Đổi tên cột cho khớp với BCN
    # HAM: age -> BCN: age_approx
    # HAM: localization -> BCN: anatom_site_general
    df = df.rename(columns={
        'age': 'age_approx',
        'localization': 'anatom_site_general'
    })

    # 2. Xử lý Giới tính (Sex) - Chuẩn hóa về chữ thường
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.lower()

    # 3. Xử lý Vị trí (Mapping quan trọng nhất)
    # BCN dùng các vị trí tổng quát hơn HAM
    loc_mapping = {
        # Vùng thân
        'abdomen': 'anterior torso',
        'chest': 'anterior torso',
        'back': 'posterior torso',
        'trunk': 'anterior torso',
        'lateral torso': 'anterior torso',  # Map tạm

        # Vùng đầu cổ
        'face': 'head/neck',
        'neck': 'head/neck',
        'scalp': 'head/neck',
        'ear': 'head/neck',

        # Chi dưới
        'foot': 'lower extremity',
        'lower extremity': 'lower extremity',

        # Chi trên
        'hand': 'upper extremity',
        'upper extremity': 'upper extremity',

        # Vùng đặc biệt
        'genital': 'oral/genital',
        'acral': 'palms/soles',

        # Không xác định
        'unknown': 'NA',
        'nan': 'NA'
    }

    if 'anatom_site_general' in df.columns:
        df['anatom_site_general'] = df['anatom_site_general'].map(loc_mapping).fillna('NA')
    else:
        print("⚠️ Cảnh báo: Không tìm thấy cột 'localization' (đã rename) trong HAM.")

    print("✅ Mapping hoàn tất.")
    return df


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    seed_everything(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])
    print(f"🔥 Thiết bị: {device}")

    # 1. Load Metadata Info từ tập TRAIN (BCN)
    print(f"📂 Loading Metadata Encoders của BCN từ: {CONFIG['META_INFO_PATH']}")
    if CONFIG['METADATA_MODE'] != 'strategy1':
        if not os.path.exists(CONFIG['META_INFO_PATH']):
            raise FileNotFoundError(f"❌ Không tìm thấy file metadata info: {CONFIG['META_INFO_PATH']}")
        encoders, num_stats = load_metadata_info(CONFIG['META_INFO_PATH'])
    else:
        encoders, num_stats = None, None

    # 2. Load và Map dữ liệu TEST (HAM)
    print(f"📂 Loading HAM Data từ: {CONFIG['TEST_CSV']}")
    test_df = pd.read_csv(CONFIG['TEST_CSV'])
    test_df.columns = test_df.columns.str.strip()  # Xóa khoảng trắng thừa ở tên cột

    # Xử lý nhãn cho HAM (Chuẩn hóa về 0/1)
    if 'dx' in test_df.columns:
        test_df['label'] = test_df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    # Áp dụng Mapping để biến DataFrame HAM thành dạng BCN
    mapped_test_df = map_ham_to_bcn(test_df)

    # 3. Tạo Dataset
    # QUAN TRỌNG: Dùng DermoscopyDataset (Class của BCN) nhưng chứa data HAM đã map
    print("🚀 Khởi tạo Dataset...")
    test_ds = DermoscopyDataset(
        df=mapped_test_df,
        img_root=CONFIG['IMG_ROOT'],
        img_size=CONFIG['IMG_SIZE'],
        metadata_mode=CONFIG['METADATA_MODE'],
        train=False,
        external_encoders=encoders,  # Truyền encoder của BCN vào
        external_stats=num_stats
    )

    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 4. Load Model
    print("🤖 Loading Model BCN...")
    cat_cardinalities = test_ds.cat_cardinalities if encoders else {}
    num_numeric = len(test_ds.numeric_cols) if encoders else 0

    # Khởi tạo model với cấu trúc y hệt lúc train BCN
    model = get_model(CONFIG, cat_cardinalities, num_numeric).to(device)

    # Load trọng số (Weights)
    if not os.path.exists(CONFIG['CHECKPOINT_PATH']):
        raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint: {CONFIG['CHECKPOINT_PATH']}")

    checkpoint = torch.load(CONFIG['CHECKPOINT_PATH'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # 5. Evaluate
    print(f"🚀 Đang chạy đánh giá Model BCN trên tập HAM10000...")
    print(f"   - Metadata Mode: {CONFIG['METADATA_MODE']}")
    print(f"   - Image Size: {CONFIG['IMG_SIZE']}")

    results = evaluate(model, test_loader, device=device)

    print("\n" + "=" * 40)
    print(f"KẾT QUẢ: TRAIN BCN20000 -> TEST HAM10000")
    print("=" * 40)
    print(f"AUC       : {results['auc']:.4f}")
    print(f"Accuracy  : {results['acc']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Specificity: {results.get('spec', 0):.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
