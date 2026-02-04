import sys
import os
import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- THÊM ĐƯỜNG DẪN SRC ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'D:\skin_cancer_project\src'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model

# ------------------- CONFIG (Giữ nguyên) -------------------
TEST_CONFIG = {
    'CSV_PATH': r'D:\skin_cancer_project\dataset\metadata\ham10000_processed.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\ouput\ham10000_checkpint',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',
    'SEED': 42,
    'NSAMPLES_SHAP': 10
}

# =========================================================
# ✅ LOAD MODEL + AUTO METADATA
# =========================================================
def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])
    df = pd.read_csv(config['CSV_PATH'])

    # ⚙️ Giữ lại các cột cần thiết (Logic cũ của bạn)
    keep_cols = ["image_id", "age", "sex", "localization", "dx"]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    # Fill missing values
    if 'age' in df.columns:
        df['age_scaled'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age_scaled'] = np.nan

    for cat_col in ["sex", "localization", "dx"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("unknown")

    # ✅ Gán nhãn nhị phân tự động
    if "dx" in df.columns:
        df["label"] = df["dx"].astype(str).apply(lambda x: 1 if "mel" in x or "malig" in x else 0)
    else:
        # Fallback nếu không có cột dx nhưng có cột label sẵn
        if "label" not in df.columns:
             raise ValueError("⚠️ Cột 'dx' hoặc 'label' bắt buộc phải có.")

    # Tách tập train/val/test (Giữ seed cố định để khớp với lúc train)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=config['SEED'], stratify=df['label'])
    # Lưu ý: Train_10k dùng test_size=0.2, sau đó split tiếp val/test từ temp
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config['SEED'], stratify=temp_df['label'])

    # 🧩 Dataset (Sử dụng class từ src)
    train_ds = HAM10000Dataset(
        train_df,
        config['IMG_ROOT'],
        config['IMG_SIZE'],
        metadata_mode=config['METADATA_MODE'],
        train=False
    )

    # ✅ Model (Sử dụng Factory từ src)
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    # 🔍 Load checkpoint
    ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device) # weights_only=True nếu pytorch mới
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"✅ Đã load model từ: {ckpt_path}")
    print(f"📊 Numeric cols: {train_ds.numeric_cols}")
    print(f"📊 Categorical cols: {train_ds.categorical_cols}")
    
    return model, device, train_ds, train_df, test_df


# =========================================================
# 🧩 Encode metadata
# =========================================================
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
        raw = str(row.get(cc, 'unknown')) # HAM10000 dùng 'unknown'
        le = train_ds.encoders[cc]
        try:
            idx = int(le.transform([raw])[0])
        except:
            idx = 0
        cats.append(idx)
    cats = torch.tensor(cats, dtype=torch.long)
    return nums, cats

# =========================================================
# ✅ test_metadata_shap_beeswarm (Logic gốc)
# =========================================================
def test_metadata_shap_beeswarm(
        model, train_ds, test_df, device,
        save_dir="test_shap_full_weighted",
        top_n_display=5, nsamples=None, bg_samples=10, sample_n=1000):

    os.makedirs(save_dir, exist_ok=True)

    # --- feature names ---
    feature_names = []
    for nc in train_ds.numeric_cols:
        feature_names.append(nc)
    for cc in train_ds.categorical_cols:
        le = train_ds.encoders[cc]
        for cls in le.classes_:
            feature_names.append(f"{cc}_{cls}")

    # --- sample subset ---
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

    # --- encode metadata ---
    for _, row_i in subset_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).cpu().numpy()
        meta_onehot = to_onehot(meta_arr)
        meta_list.append(meta_onehot)
    meta_stack = np.vstack(meta_list)
    
    meta_df = pd.DataFrame(meta_stack, columns=feature_names)

    # --- model wrapper ---
    def model_wrapper(meta_array):
        with torch.no_grad():
            K = meta_array.shape[0]
            dummy_img = torch.zeros((K, 3, 224, 224))
            num_cols = len(train_ds.numeric_cols)
            meta_num = torch.tensor(meta_array[:, :num_cols], dtype=torch.float32)
            offset = num_cols
            meta_cat_list = []
            for cc in train_ds.categorical_cols:
                card = train_ds.cat_cardinalities[cc]
                onehot = meta_array[:, offset:offset + card]
                idx = np.argmax(onehot, axis=1)
                meta_cat_list.append(torch.tensor(idx, dtype=torch.long))
                offset += card
            if len(meta_cat_list) > 0:
                meta_cat = torch.stack(meta_cat_list, dim=1)
            else:
                meta_cat = torch.empty((K, 0), dtype=torch.long)
            
            # Gọi model từ src
            logits = model(dummy_img.to(device), meta_num.to(device), meta_cat.to(device))
            out = torch.sigmoid(logits).cpu().numpy()
            return out.reshape(-1)

    # --- background for SHAP ---
    bg_df = train_ds.df.sample(n=min(bg_samples, len(train_ds.df)), random_state=123)
    bg_list = []
    for _, row_i in bg_df.iterrows():
        n_i, c_i = encode_metadata_with_train_encoders(row_i, train_ds)
        meta_arr = torch.cat([n_i, c_i.float()]).unsqueeze(0).cpu().numpy()
        meta_onehot = to_onehot(meta_arr)
        bg_list.append(meta_onehot)
    bg_stack = np.vstack(bg_list)

    # --- SHAP ---
    nsamples = nsamples if nsamples is not None else TEST_CONFIG.get('NSAMPLES_SHAP', 10)
    explainer = shap.KernelExplainer(model_wrapper, bg_stack)
    print(f"🔍 Computing SHAP values for {len(subset_df)} samples (nsamples={nsamples})...")
    shap_vals = explainer.shap_values(meta_stack, nsamples=nsamples)

    # --- xử lý output ---
    if isinstance(shap_vals, list):
        chosen = 1 if len(shap_vals) >= 2 else 0
        shap_values_for_plot = np.array(shap_vals[chosen])
    else:
        shap_values_for_plot = np.array(shap_vals)

    # --- auto xlim range ---
    q_low, q_high = np.percentile(shap_values_for_plot, [5, 95])
    xlim_low = q_low * 1.2
    xlim_high = q_high * 1.2

    # --- compute importance ---
    mean_abs = np.abs(shap_values_for_plot).mean(axis=0).flatten()
    tmp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
    print("🏆 Top features by mean |SHAP|:")
    print(tmp_df.head(15))

    # --- short name mapping ---
    short_names = [f"x{i+1}" for i in range(len(feature_names))]
    meta_df_short = meta_df.copy()
    meta_df_short.columns = short_names

    # --- plot Top N ---
    TOP_N_DISPLAY = min(top_n_display, len(feature_names))
    fig_height = max(6, TOP_N_DISPLAY * 0.6)
    plt.figure(figsize=(12, fig_height))
    shap.summary_plot(
        shap_values_for_plot,
        meta_df_short,
        feature_names=short_names,
        show=False,
        plot_type="dot",
        max_display=TOP_N_DISPLAY
    )
    plt.title(f"🌈 SHAP Beeswarm Plot (Top {TOP_N_DISPLAY} Metadata Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.xlim(xlim_low, xlim_high)
    summary_path = os.path.join(save_dir, f"shap_summary_beeswarm_top_{TOP_N_DISPLAY}_features.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved TOP {TOP_N_DISPLAY} beeswarm → {summary_path}")

    # --- plot all features ---
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.25)))
    shap.summary_plot(
        shap_values_for_plot,
        meta_df_short,
        feature_names=short_names,
        show=False,
        plot_type="dot",
        max_display=len(feature_names)
    )
    plt.title("🌈 SHAP Beeswarm Plot (All Metadata Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.xlim(xlim_low, xlim_high)
    all_path = os.path.join(save_dir, "shap_summary_beeswarm_all_features.png")
    plt.savefig(all_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ALL features beeswarm → {all_path}")

    # --- save importance & mapping ---
    all_features_df = tmp_df.reset_index(drop=True)
    all_features_df.to_csv(os.path.join(save_dir, "all_features_importance.csv"), index=False)
    pd.DataFrame({"short_name": short_names, "full_name": feature_names}).to_csv(
        os.path.join(save_dir, "feature_name_mapping.csv"), index=False)
    print(f"✅ Saved feature mapping + importance CSVs to → {save_dir}")

    print("🎯 SHAP analysis completed successfully!")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    print("🚀 Bắt đầu SHAP Beeswarm (HAM10000)...")
    model, device, train_ds, train_df, test_df = load_model_and_encoders(TEST_CONFIG)
    test_metadata_shap_beeswarm(model, train_ds, test_df, device)