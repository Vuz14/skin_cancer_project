import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import load_metadata_info, seed_everything

# --- CẤU HÌNH ENSEMBLE ---
CONFIG = {
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',

    # Chỉ định thư mục gốc của 5 fold vừa train xong
    'CV_RUN_DIR': r'checkpoint_bcn20000/CV5_full_weighted_resnet50',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'PRETRAINED': True,
    'METADATA_FEATURE_BOOST': 5.0,
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',
    'MODEL_NAME': 'resnet50',
    'SHORT_NAME': 'resnet50',
    'BATCH_SIZE': 32,
    'SEED': 42
}


def main():
    seed_everything(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])

    # 1. LOAD TEST DATA (Lấy Encoder từ fold 1 là đủ vì chung tập mapping)
    print("📂 Đang tải dữ liệu Test...")
    test_df = pd.read_csv(CONFIG['TEST_CSV'])
    test_df.columns = test_df.columns.str.strip()
    if 'image_path' not in test_df.columns:
        test_df['image_path'] = test_df['isic_id'].astype(str) + '.jpg'

    test_df['diagnosis_1'] = test_df['diagnosis_1'].astype(str).str.strip().str.lower()
    test_df['label'] = test_df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    true_labels = test_df['label'].values

    meta_info_path = os.path.join(CONFIG['CV_RUN_DIR'], 'fold_1', f"meta_info_{CONFIG['SHORT_NAME']}.pkl")
    encoders, num_stats = load_metadata_info(meta_info_path) if CONFIG['METADATA_MODE'] != 'diag1' else (None, None)

    test_ds = DermoscopyDataset(test_df, CONFIG['IMG_ROOT'], CONFIG['IMG_SIZE'], CONFIG['METADATA_MODE'], train=False,
                                external_encoders=encoders, external_stats=num_stats)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 2. LOAD 5 MODELS
    models = []
    print("\n🤖 Đang tải 5 mô hình từ các Fold...")
    cat_cardinalities = test_ds.cat_cardinalities if encoders else {}
    num_numeric = len(test_ds.numeric_cols) if encoders else 0

    for fold in range(1, 6):
        model = get_model(CONFIG, cat_cardinalities, num_numeric).to(device)
        model.eval()

        # Cập nhật lại đường dẫn đọc file model khớp với file trainer.py đã sửa
        ckpt_path = os.path.join(CONFIG['CV_RUN_DIR'], f'fold_{fold}', f"best_{CONFIG['METADATA_MODE']}_fold_{fold}.pt")

        if not os.path.exists(ckpt_path):
            print(f"❌ KHÔNG TÌM THẤY file model: {ckpt_path}")
            continue  # Bỏ qua fold này nếu không thấy file

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        models.append(model)
        print(f"   ✅ Đã load Model Fold {fold}")

    # 3. CHẠY INFERENCE & ENSEMBLE (AVERAGE VOTING)
    print("\n🚀 Đang chạy dự đoán Ensemble trên tập Test...")
    ensemble_preds = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            meta = {k: v.to(device) for k, v in batch['metadata'].items()} if 'metadata' in batch else None

            # Lưu xác suất của batch này cho 5 model (Kích thước: [5, batch_size])
            batch_probs = []
            for model in models:
                outputs = model(imgs, meta) if meta else model(imgs)
                probs = torch.sigmoid(outputs).squeeze(-1)  # Xác suất từ 0 đến 1
                batch_probs.append(probs.cpu().numpy())

            # Tính trung bình xác suất của 5 model: (p1+p2+p3+p4+p5)/5
            avg_probs = np.mean(batch_probs, axis=0)
            ensemble_preds.extend(avg_probs)

    ensemble_preds = np.array(ensemble_preds)
    ensemble_preds_binary = (ensemble_preds >= 0.5).astype(int)

    # 4. TÍNH METRICS CUỐI CÙNG CHO BẢNG 2
    auc = roc_auc_score(true_labels, ensemble_preds)
    acc = accuracy_score(true_labels, ensemble_preds_binary)
    f1 = f1_score(true_labels, ensemble_preds_binary, zero_division=0)
    rec = recall_score(true_labels, ensemble_preds_binary, zero_division=0)
    prec = precision_score(true_labels, ensemble_preds_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, ensemble_preds_binary).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "🏆" * 20)
    print("BẢNG 2: KẾT QUẢ ĐỈNH (ENSEMBLE 5-FOLD)")
    print("🏆" * 20)
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()
