import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json

# --- THÊM ĐƯỜNG DẪN SRC ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model

# ------------------- CONFIG -------------------
TEST_CONFIG = {
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_train.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',

    'MODEL_NAME': 'resnet50',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',
    'PRETRAINED': True,
    'SEED': 42,
    'NSAMPLES_SHAP': 50,
}


# ------------------- MAPPING X1-X18 ĐẦY ĐỦ -------------------
def get_full_x_mapping():
    """Mapping tên biến gốc sang ký hiệu X1-X18 chuẩn bài báo"""
    return {
        'age_approx': 'X1 (Age)',
        # General Sites
        'anatom_site_general_unknown': 'X2 (Site_NA)',
        'anatom_site_general_anterior torso': 'X3 (Site_Ant_Torso)',
        'anatom_site_general_head/neck': 'X4 (Site_Head/Neck)',
        'anatom_site_general_lower extremity': 'X5 (Site_Lower_Ext)',
        'anatom_site_general_oral/genital': 'X6 (Site_Oral/Genital)',
        'anatom_site_general_palms/soles': 'X7 (Site_Palms/Soles)',
        'anatom_site_general_upper extremity': 'X8 (Site_Upper_Ext)',

        # Special Sites
        'anatom_site_special_unknown': 'X9 (Spec_NA)',
        'anatom_site_special_acral palms or soles': 'X10 (Spec_Palms/Soles)',
        'anatom_site_special_oral or genital': 'X11 (Spec_Oral/Genital)',

        # Diagnosis
        'diagnosis_confirm_type_unknown': 'X12 (Confirm_NA)',
        'diagnosis_confirm_type_confocal microscopy with consensus dermoscopy': 'X13 (Confirm_Confocal)',
        'diagnosis_confirm_type_histopathology': 'X14/X15 (Confirm_Histo)',

        # Sex
        'sex_unknown': 'X16 (Sex_NA)',
        'sex_female': 'X17 (Sex_Female)',
        'sex_male': 'X18 (Sex_Male)',

        # Các biến thể tên khác có thể gặp
        'anatom_site_general_nan': 'X2 (Site_NA)',
        'anatom_site_special_nan': 'X9 (Spec_NA)',
        'diagnosis_confirm_type_nan': 'X12 (Confirm_NA)',
        'sex_nan': 'X16 (Sex_NA)'
    }


# ------------------- LOAD MODEL & DATA -------------------
def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])

    train_df = pd.read_csv(config['TRAIN_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])

    # Tiền xử lý cơ bản (Đồng bộ với train)
    for df in [train_df, test_df]:
        if 'image_path' not in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
        df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

    # Load Dataset Gốc (Không lọc, không gộp)
    # Model đã được train với Full Features nên ta dùng Dataset thường
    train_ds = DermoscopyDataset(
        train_df, config['IMG_ROOT'], config['IMG_SIZE'],
        metadata_mode=config['METADATA_MODE'], train=False
    )

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}. Hãy chạy train_bcn.py trước!")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"✅ Đã load model Full Features (X1-X18).")
    return model, device, train_ds, test_df


# ------------------- SHAP & PLOT -------------------
def encode_metadata(row, train_ds):
    nums = []
    for nc in train_ds.numeric_cols:
        val = row.get(nc, np.nan)
        mean, std = train_ds.num_mean_std[nc]
        nums.append((float(val) - mean) / std if not pd.isna(val) else 0.0)

    cats = []
    for cc in train_ds.categorical_cols:
        # Lấy giá trị gốc, nếu lạ thì về 'unknown' (không phải 'other' vì ta không gộp nữa)
        raw = str(row.get(cc, 'unknown'))
        le = train_ds.encoders[cc]
        try:
            idx = int(le.transform([raw])[0])
        except:
            # Nếu gặp giá trị cực lạ chưa từng thấy ở train set
            # Thử tìm index của 'unknown' hoặc 'nan'
            fallback_vals = ['unknown', 'nan', 'other']
            idx = 0
            for fb in fallback_vals:
                if fb in le.classes_:
                    idx = int(le.transform([fb])[0])
                    break
        cats.append(idx)
    return torch.tensor(nums, dtype=torch.float32), torch.tensor(cats, dtype=torch.long)


def test_metadata_shap_beeswarm(model, train_ds, test_df, device, save_dir="explain_results_bcn", top_n_display=25):
    os.makedirs(save_dir, exist_ok=True)

    # Tạo danh sách tên cột One-Hot
    feature_names = []
    for nc in train_ds.numeric_cols: feature_names.append(nc)
    for cc in train_ds.categorical_cols:
        for cls in train_ds.encoders[cc].classes_: feature_names.append(f"{cc}_{cls}")

    subset_df = test_df.sample(n=min(30, len(test_df)), random_state=42)

    def to_onehot(meta_array):
        num_cols = len(train_ds.numeric_cols)
        offset, outs = num_cols, [meta_array[:, :num_cols]]
        for cc in train_ds.categorical_cols:
            card = train_ds.cat_cardinalities[cc]
            idx = meta_array[:, offset].astype(int)
            oh = np.zeros((meta_array.shape[0], card));
            oh[np.arange(meta_array.shape[0]), idx] = 1
            outs.append(oh);
            offset += 1
        return np.hstack(outs)

    meta_list = []
    for _, row in subset_df.iterrows():
        n, c = encode_metadata(row, train_ds)
        meta_list.append(to_onehot(torch.cat([n, c.float()]).unsqueeze(0).numpy()))

    meta_stack = np.vstack(meta_list)
    meta_df = pd.DataFrame(meta_stack, columns=feature_names)

    # --- ĐỔI TÊN CỘT THÀNH X1-X18 ---
    mapping = get_full_x_mapping()
    new_cols = []
    for c in meta_df.columns:
        best_match = c
        # Logic match: Tìm tên biến gốc trong tên cột One-Hot
        # Ví dụ: anatom_site_general_palms/soles -> X7
        for k, v in mapping.items():
            # Match phần đuôi giá trị để chính xác hơn
            # c = "anatom_site_general_palms/soles"
            # k = "anatom_site_general_palms/soles"
            if k == c:
                best_match = v
                break
            # Fallback: substring match
            if k in c:
                best_match = v
        new_cols.append(best_match)
    meta_df.columns = new_cols

    def model_wrapper(m_arr):
        with torch.no_grad():
            K = m_arr.shape[0];
            num_c = len(train_ds.numeric_cols)
            m_num = torch.tensor(m_arr[:, :num_c], dtype=torch.float32).to(device)
            m_cat_l, off = [], num_c
            for cc in train_ds.categorical_cols:
                card = train_ds.cat_cardinalities[cc]
                m_cat_l.append(torch.tensor(np.argmax(m_arr[:, off:off + card], axis=1), dtype=torch.long))
                off += card
            m_cat = torch.stack(m_cat_l, dim=1).to(device) if m_cat_l else torch.empty((K, 0)).to(device)
            return torch.sigmoid(model(torch.zeros((K, 3, 224, 224)).to(device), m_num, m_cat)).cpu().numpy().reshape(
                -1)

    bg_df = train_ds.df.sample(n=min(15, len(train_ds.df)), random_state=123)
    bg_list = [to_onehot(torch.cat(list(encode_metadata(r, train_ds))).unsqueeze(0).numpy()) for _, r in
               bg_df.iterrows()]

    explainer = shap.KernelExplainer(model_wrapper, np.vstack(bg_list))
    print("⌛ Đang tính toán SHAP values (Full X1-X18)...")
    shap_vals = explainer.shap_values(meta_stack, nsamples=TEST_CONFIG['NSAMPLES_SHAP'])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    plt.figure(figsize=(12, 12))  # Cao hơn để hiện hết 18 dòng
    # max_display=30 để đảm bảo hiện hết tất cả biến dù điểm thấp
    shap.summary_plot(shap_vals, meta_df, show=False, max_display=30)
    plt.yticks(fontsize=10)
    plt.title("SHAP Feature Importance (Full X1-X18)")

    save_path = os.path.join(save_dir, "shap_full_X_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 Đã lưu biểu đồ Full SHAP tại: {save_path}")


if __name__ == "__main__":
    model, device, train_ds, test_df = load_model_and_encoders(TEST_CONFIG)
    test_metadata_shap_beeswarm(model, train_ds, test_df, device)