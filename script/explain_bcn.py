import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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

    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effb4', # Tên ngắn cho file output
    'IMG_SIZE': 224, # Lưu ý: Khi giải thích SHAP thường dùng size nhỏ hơn train (300) để nhanh, hoặc giữ 300 nếu GPU mạnh
    'METADATA_MODE': 'full_weighted',
    'PRETRAINED': True,
    'NSAMPLES_SHAP': 50,
}

# ------------------- MAPPING X1-X18 (Giữ nguyên logic cũ nhưng gọn hơn) -------------------
def get_full_x_mapping():
    return {
        'age_approx': 'X1 (Age)',
        'anatom_site_general_unknown': 'X2 (Site_NA)',
        'anatom_site_general_anterior torso': 'X3 (Site_Ant_Torso)',
        'anatom_site_general_head/neck': 'X4 (Site_Head/Neck)',
        'anatom_site_general_lower extremity': 'X5 (Site_Lower_Ext)',
        'anatom_site_general_oral/genital': 'X6 (Site_Oral/Genital)',
        'anatom_site_general_palms/soles': 'X7 (Site_Palms/Soles)',
        'anatom_site_general_upper extremity': 'X8 (Site_Upper_Ext)',
        'anatom_site_special_unknown': 'X9 (Spec_NA)',
        'anatom_site_special_acral palms or soles': 'X10 (Spec_Palms/Soles)',
        'anatom_site_special_oral or genital': 'X11 (Spec_Oral/Genital)',
        'diagnosis_confirm_type_unknown': 'X12 (Confirm_NA)',
        'diagnosis_confirm_type_confocal microscopy with consensus dermoscopy': 'X13 (Confirm_Confocal)',
        'diagnosis_confirm_type_histopathology': 'X14/X15 (Confirm_Histo)',
        'sex_unknown': 'X16 (Sex_NA)',
        'sex_female': 'X17 (Sex_Female)',
        'sex_male': 'X18 (Sex_Male)',
    }

def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])
    train_df = pd.read_csv(config['TRAIN_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])

    # Tiền xử lý
    for df in [train_df, test_df]:
        if 'image_path' not in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
        df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

    train_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
    model.eval()

    return model, device, train_ds, test_df

def run_shap_analysis(model, train_ds, test_df, device):
    print("⏳ Đang tính toán SHAP values (Full X1-X18)...")
    
    # 1. Prepare Data
    subset_df = test_df.sample(n=min(30, len(test_df)), random_state=42)
    bg_df = train_ds.df.sample(n=min(15, len(train_ds.df)), random_state=123)

    # Helper: Encode & One-hot flatten
    def to_onehot_flat(df_in):
        flat_list = []
        # Tên cột tạm thời (one-hot)
        feat_names = train_ds.numeric_cols + [f"{c}_{cls}" for c in train_ds.categorical_cols for cls in train_ds.encoders[c].classes_]

        for _, row in df_in.iterrows():
            nums = []
            for nc in train_ds.numeric_cols:
                mean, std = train_ds.num_mean_std[nc]
                val = row.get(nc, mean)
                nums.append((float(val) - mean)/std)
            
            cats_oh = []
            for cc in train_ds.categorical_cols:
                le = train_ds.encoders[cc]
                raw = str(row.get(cc, 'unknown'))
                try: idx = int(le.transform([raw])[0])
                except: idx = 0
                oh = np.zeros(len(le.classes_))
                oh[idx] = 1
                cats_oh.extend(oh)
            flat_list.append(np.concatenate([nums, cats_oh]))
            
        return np.array(flat_list), feat_names

    test_data, col_names = to_onehot_flat(subset_df)
    bg_data, _ = to_onehot_flat(bg_df)

    # 2. Rename Columns to X1-X18
    mapping = get_full_x_mapping()
    new_cols = []
    for c in col_names:
        name = mapping.get(c, c)
        if name == c: # Fallback substring match
            for k, v in mapping.items():
                if k in c: name = v; break
        new_cols.append(name)

    # 3. Model Wrapper
    def model_wrapper(m_arr):
        with torch.no_grad():
            K = m_arr.shape[0]
            dummy_img = torch.zeros((K, 3, config['IMG_SIZE'], config['IMG_SIZE'])).to(device)
            num_c = len(train_ds.numeric_cols)
            
            m_num = torch.tensor(m_arr[:, :num_c], dtype=torch.float32).to(device)
            m_cat_l = []
            off = num_c
            for cc in train_ds.categorical_cols:
                card = train_ds.cat_cardinalities[cc]
                chunk = m_arr[:, off:off+card]
                m_cat_l.append(torch.tensor(np.argmax(chunk, axis=1), dtype=torch.long))
                off += card
            
            m_cat = torch.stack(m_cat_l, dim=1).to(device)
            return torch.sigmoid(model(dummy_img, m_num, m_cat)).cpu().numpy().reshape(-1)

    # 4. SHAP
    explainer = shap.KernelExplainer(model_wrapper, bg_data)
    shap_vals = explainer.shap_values(test_data, nsamples=TEST_CONFIG['NSAMPLES_SHAP'])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    # 5. Plot & Save
    plt.figure(figsize=(12, 12))
    shap.summary_plot(shap_vals, pd.DataFrame(test_data, columns=new_cols), show=False, max_display=30)
    plt.yticks(fontsize=10)
    
    # Save tên chuẩn
    save_name = f"bcn20k_{TEST_CONFIG['SHORT_NAME']}_shap.png"
    save_path = os.path.join(TEST_CONFIG['MODEL_OUT'], save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 Đã lưu biểu đồ Full SHAP tại: {save_path}")

if __name__ == "__main__":
    model, device, train_ds, test_df = load_model_and_encoders(TEST_CONFIG)
    run_shap_analysis(model, train_ds, test_df, device)