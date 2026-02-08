import sys
import os
import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- THÊM ĐƯỜNG DẪN SRC ---
sys.path.append(os.path.join(os.path.dirname(__file__), '/mnt/d/skin_cancer_project'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model

# ------------------- CONFIG (Đồng bộ với train_ham mới) -------------------
TEST_CONFIG = {
    'TRAIN_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_train.csv',
    'VAL_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_val.csv',
    'TEST_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_test.csv',
    'IMG_ROOT': '/mnt/d/skin_cancer_project/dataset/Bcn20k',
    'MODEL_OUT': '/mnt/d/skin_cancer_project/checkpoint_bcn20000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',
    'SEED': 42,
    'NSAMPLES_SHAP': 50,
    'MODEL_NAME': 'resnet50',
    'SELECTED_FEATURES': None # Danh sách biến nếu bạn đã lọc lúc train
}

# ------------------- LOAD MODEL -------------------
def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])
    
    # Load data
    train_df = pd.read_csv(config['TRAIN_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])

    # Tiền xử lý đồng bộ với train_ham
    for df in [train_df, test_df]:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        if 'dx' in df.columns:
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    # Khởi tạo dataset train để tái hiện lại các Encoders
    train_ds = HAM10000Dataset(
        train_df, config['IMG_ROOT'], config['IMG_SIZE'],
        metadata_mode=config['METADATA_MODE'], train=False,
        selected_features=config['SELECTED_FEATURES']
    )

    # --- Khởi tạo model dựa trên cardinality của DS đã lọc ---
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    # Load trọng số từ checkpoint
    ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f" Không tìm thấy checkpoint tại: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f" Đã load model từ: {ckpt_path}")
    print(f" Numeric cols: {train_ds.numeric_cols}")
    print(f" Categorical cols: {train_ds.categorical_cols}")
    
    return model, device, train_ds, test_df

# ------------------- ENCODE METADATA -------------------
def encode_metadata_with_train_encoders(row, train_ds):
    nums = []
    for nc in train_ds.numeric_cols:
        val = row.get(nc, np.nan)
        if pd.isna(val):
            val = train_ds.num_mean_std[nc][0]
        mean, std = train_ds.num_mean_std[nc]
        nums.append((float(val) - mean) / std)
    nums = torch.tensor(nums, dtype=torch.float32)

    cats = []
    for cc in train_ds.categorical_cols:
        raw = str(row.get(cc, 'unknown'))
        le = train_ds.encoders[cc]
        try:
            idx = int(le.transform([raw])[0])
        except:
            idx = 0
        cats.append(idx)
    cats = torch.tensor(cats, dtype=torch.long)
    return nums, cats

# ------------------- SHAP ANALYSIS -------------------
def test_metadata_shap_beeswarm(model, train_ds, test_df, device,
                                save_dir="explain_results_ham",
                                top_n_display=10, sample_n=100, bg_samples=20):
    os.makedirs(save_dir, exist_ok=True)

    # --- feature names ---
    feature_names = []
    for nc in train_ds.numeric_cols:
        feature_names.append(nc)
    for cc in train_ds.categorical_cols:
        le = train_ds.encoders[cc]
        for cls in le.classes_:
            feature_names.append(f"{cc}_{cls}")

    # Lấy mẫu ngẫu nhiên từ tập TEST để giải thích
    subset_df = test_df.sample(n=min(sample_n, len(test_df)), random_state=42)
    meta_list = []

    def to_onehot(meta_array):
        num_cols = len(train_ds.numeric_cols)
        onehot_list = []
        offset = num_cols
        for cc in train_ds.categorical_cols:
            card = train_ds.cat_cardinalities[cc]
            cat_idx = meta_array[:, offset].astype(int)
            onehot = np.zeros((meta_array.shape[0], card))
            onehot[np.arange(meta_array.shape[0]), cat_idx] = 1
            onehot_list.append(onehot)
            offset += 1
        return np.hstack([meta_array[:, :num_cols]] + onehot_list)

    # Encode metadata
    for _, row_i in subset_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).cpu().numpy()
        meta_list.append(to_onehot(meta_arr))
    meta_stack = np.vstack(meta_list)
    meta_df = pd.DataFrame(meta_stack, columns=feature_names)

    # Model wrapper
    def model_wrapper(meta_array):
        with torch.no_grad():
            K = meta_array.shape[0]
            dummy_img = torch.zeros((K, 3, 224, 224)).to(device)
            num_cols = len(train_ds.numeric_cols)
            meta_num = torch.tensor(meta_array[:, :num_cols], dtype=torch.float32).to(device)
            offset = num_cols
            meta_cat_list = []
            for cc in train_ds.categorical_cols:
                card = train_ds.cat_cardinalities[cc]
                onehot = meta_array[:, offset:offset + card]
                idx = np.argmax(onehot, axis=1)
                meta_cat_list.append(torch.tensor(idx, dtype=torch.long))
                offset += card
            
            if len(meta_cat_list) > 0:
                meta_cat = torch.stack(meta_cat_list, dim=1).to(device)
            else:
                meta_cat = torch.empty((K, 0), dtype=torch.long).to(device)
            
            logits = model(dummy_img, meta_num, meta_cat)
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # Background từ tập TRAIN để tính SHAP
    bg_df = train_ds.df.sample(n=min(bg_samples, len(train_ds.df)), random_state=123)
    bg_list = []
    for _, row_i in bg_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).cpu().numpy()
        bg_list.append(to_onehot(meta_arr))
    bg_stack = np.vstack(bg_list)

    # SHAP
    explainer = shap.KernelExplainer(model_wrapper, bg_stack)
    print(f"⌛ Computing SHAP values cho {sample_n} mẫu tập TEST...")
    shap_vals = explainer.shap_values(meta_stack, nsamples=TEST_CONFIG['NSAMPLES_SHAP'])

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    # Plot & Save
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, meta_df, show=False, max_display=top_n_display)
    plt.title(f"SHAP Analysis (HAM10000 Test Set) - Mode: {TEST_CONFIG['METADATA_MODE']}")
    plt.savefig(os.path.join(save_dir, "shap_summary_beeswarm_test.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Lưu CSV
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
    importance_df.to_csv(os.path.join(save_dir, "feature_importance_test.csv"), index=False)
    
    print(f"🎉 Kết quả giải thích tập TEST đã lưu tại: {save_dir}")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    print("🚀 Bắt đầu giải thích mô hình HAM10000 trên tập TEST...")
    model, device, train_ds, test_df = load_model_and_encoders(TEST_CONFIG)
    test_metadata_shap_beeswarm(model, train_ds, test_df, device)