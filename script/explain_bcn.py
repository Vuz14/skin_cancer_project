import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split
import shap

# --- THÊM ĐƯỜNG DẪN SRC ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'D:\skin_cancer_project\src'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model

# ------------------- CONFIG (Giữ nguyên) -------------------
TEST_CONFIG = {
    'CSV_PATH': r'D:\skin_cancer_project\dataset\metadata\metadata.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\ISIC-preprocessed-test11_refined',
    'MODEL_OUT': r'D:\skin_cancer_project\ouput\bcn20000_checkpint',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',  # full / late_fusion
    'SEED': 42,
    'NSAMPLES_SHAP': 20
}

# ------------------- W&B INIT -------------------
def init_wandb_existing(config, run_name="SHAP Analysis"):
    run = wandb.init(
        project="nckh_skin_cancer_2025",
        name=None,  # None để W&B tự tạo run mới
        config=config,
        reinit=True,
        dir="D:/wandb_temp"
    )
    return run

# ------------------- LOAD MODEL -------------------
def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])
    df = pd.read_csv(config['CSV_PATH'])

    # Logic ISIC
    if 'image_path' not in df.columns:
        if 'isic_id' in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        else:
            raise ValueError("CSV phải có image_path hoặc isic_id")

    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])]
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=config['SEED'], stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=1 / 3, random_state=config['SEED'], stratify=temp_df['label'])

    train_ds = DermoscopyDataset(
        train_df, config['IMG_ROOT'], config['IMG_SIZE'],
        metadata_mode=config['METADATA_MODE'], train=False
    )

    # --- Chọn model đúng mode qua Factory ---
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"✅ Đã load model từ: {ckpt_path}")
    return model, device, train_ds, train_df, test_df

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
        raw = str(row.get(cc, 'NA')) # ISIC dùng 'NA'
        le = train_ds.encoders[cc]
        try:
            idx = int(le.transform([raw])[0])
        except:
            idx = 0
        cats.append(idx)
    cats = torch.tensor(cats, dtype=torch.long)
    return nums, cats

# =========================================================
# SHAP + W&B LOGGING (Logic gốc)
# =========================================================
def test_metadata_shap_beeswarm(model, train_ds, test_df, device,
                                save_dir="test_shap_full",
                                top_n_display=5, nsamples=None, sample_n=100, bg_samples=20):
    os.makedirs(save_dir, exist_ok=True)

    # --- feature names ---
    feature_names = []
    for nc in train_ds.numeric_cols:
        feature_names.append(nc)
    for cc in train_ds.categorical_cols:
        for cls in train_ds.encoders[cc].classes_:
            feature_names.append(f"{cc}_{cls}")

    subset_df = test_df.sample(n=min(sample_n, len(test_df)), random_state=42)
    meta_list = []

    def to_onehot(meta_array):
        num_cols = len(train_ds.numeric_cols)
        offset = num_cols
        outs = [meta_array[:, :num_cols]]
        for cc in train_ds.categorical_cols:
            card = train_ds.cat_cardinalities[cc]
            idx = meta_array[:, offset].astype(int)
            onehot = np.zeros((meta_array.shape[0], card))
            onehot[np.arange(meta_array.shape[0]), idx] = 1
            outs.append(onehot)
            offset += 1
        return np.hstack(outs)

    for _, row_i in subset_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).numpy()
        meta_list.append(to_onehot(meta_arr))
    meta_stack = np.vstack(meta_list)
    meta_df = pd.DataFrame(meta_stack, columns=feature_names)

    # --- model wrapper ---
    def model_wrapper(meta_array, config=TEST_CONFIG):
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

            if config['METADATA_MODE'] == 'late_fusion':
                meta_vec = torch.cat([meta_num] + [c.float().unsqueeze(1).to(device) for c in meta_cat_list], dim=1)
                logits = model(dummy_img, meta_vec)
            else:
                if len(meta_cat_list) > 0:
                    meta_cat = torch.stack(meta_cat_list, dim=1).to(device)
                else:
                    meta_cat = torch.empty((K, 0), dtype=torch.long).to(device)
                logits = model(dummy_img, meta_num, meta_cat)
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # --- background ---
    bg_df = train_ds.df.sample(n=min(bg_samples, len(train_ds.df)), random_state=123)
    bg_list = []
    for _, row_i in bg_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).numpy()
        bg_list.append(to_onehot(meta_arr))
    bg_stack = np.vstack(bg_list)

    nsamples = nsamples if nsamples is not None else TEST_CONFIG['NSAMPLES_SHAP']
    explainer = shap.KernelExplainer(model_wrapper, bg_stack)
    shap_vals = explainer.shap_values(meta_stack, nsamples=nsamples)
    # Xử lý output list nếu có
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # --- Compute importance ---
    mean_abs = np.abs(shap_vals).mean(axis=0)
    tmp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)

    # --- short names ---
    short_names = [f"x{i+1}" for i in range(len(feature_names))]
    meta_df_short = meta_df.copy()
    meta_df_short.columns = short_names

    # --- Plot Top N ---
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_vals, meta_df_short, feature_names=short_names, show=False, max_display=top_n_display)
    top_path = os.path.join(save_dir, "shap_top.png")
    plt.savefig(top_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot ALL ---
    plt.figure(figsize=(12, max(6, len(feature_names) * 0.1)))
    shap.summary_plot(shap_vals, meta_df_short, feature_names=short_names, show=False, max_display=len(feature_names))
    all_path = os.path.join(save_dir, "shap_all.png")
    plt.savefig(all_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- SAVE CSV ---
    tmp_df.to_csv(os.path.join(save_dir, "shap_feature_importance.csv"), index=False)
    pd.DataFrame({"short": short_names, "full": feature_names}).to_csv(
        os.path.join(save_dir, "feature_mapping.csv"), index=False
    )

    # --- W&B LOG ---
    wandb.log({
        "SHAP/summary_top": wandb.Image(top_path),
        "SHAP/summary_all": wandb.Image(all_path),
        "SHAP/feature_importance": wandb.Table(dataframe=tmp_df.reset_index(drop=True)),
        "SHAP/feature_mapping": wandb.Table(dataframe=pd.DataFrame({"short": short_names, "full": feature_names}))
    })
    print("🎉 Logged SHAP to W&B successfully!")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    run = init_wandb_existing(TEST_CONFIG, run_name="SHAP Metadata Mode")
    model, device, train_ds, train_df, test_df = load_model_and_encoders(TEST_CONFIG)
    test_metadata_shap_beeswarm(model, train_ds, test_df, device)
    wandb.finish()