import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model

# ------------------- CONFIG -------------------
TEST_CONFIG = {
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_test.csv',
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_train.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_ham10000',
    'DEVICE': 'cuda',
    
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effb4', # Dùng để lưu file
    'IMG_SIZE': 224,
    'METADATA_MODE': 'full_weighted',
    'PRETRAINED': True,
    'NSAMPLES_SHAP': 50,
}

# ------------------- MAPPING CHUẨN (X1-Xn) CHO HAM10000 -------------------
def get_ham_x_mapping():
    return {
        'age': 'X1 (Age)',
        
        # Sex
        'sex_male': 'X2 (Male)',
        'sex_female': 'X3 (Female)',
        'sex_unknown': 'X4 (Sex_NA)',
        
        # Localization (Vị trí)
        'localization_lower extremity': 'X5 (Loc_Lower_Ext)',
        'localization_trunk': 'X6 (Loc_Trunk)',
        'localization_back': 'X7 (Loc_Back)',
        'localization_abdomen': 'X8 (Loc_Abdomen)',
        'localization_upper extremity': 'X9 (Loc_Upper_Ext)',
        'localization_face': 'X10 (Loc_Face)',
        'localization_chest': 'X11 (Loc_Chest)',
        'localization_foot': 'X12 (Loc_Foot)',
        'localization_neck': 'X13 (Loc_Neck)',
        'localization_scalp': 'X14 (Loc_Scalp)',
        'localization_hand': 'X15 (Loc_Hand)',
        'localization_ear': 'X16 (Loc_Ear)',
        'localization_genital': 'X17 (Loc_Genital)',
        'localization_acral': 'X18 (Loc_Acral)',
        'localization_unknown': 'X19 (Loc_NA)',
    }

def load_model_and_data(config):
    device = torch.device(config['DEVICE'])
    train_df = pd.read_csv(config['TRAIN_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])
    
    for df in [train_df, test_df]:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        if 'dx' in df.columns:
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    train_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    
    ckpt = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
    model.eval()
    
    return model, device, train_ds, test_df

def run_shap_analysis(model, train_ds, test_df, device):
    print("⏳ Đang tính toán SHAP Values...")
    
    # 1. Prepare Data
    subset_df = test_df.sample(n=min(30, len(test_df)), random_state=42)
    bg_df = train_ds.df.sample(n=min(15, len(train_ds.df)), random_state=123)
    
    # Helper to encode & one-hot
    def encode_and_flat(df_in):
        flat_list = []
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

    test_data, col_names = encode_and_flat(subset_df)
    bg_data, _ = encode_and_flat(bg_df)
    
    # 2. Rename Columns (X1, X2...)
    mapping = get_ham_x_mapping()
    new_cols = []
    for c in col_names:
        name = mapping.get(c, c)
        # Fallback partial match
        if name == c:
            for k, v in mapping.items():
                if k in c: name = v; break
        new_cols.append(name)
    
    # 3. Model Wrapper
    def model_wrapper(data_arr):
        with torch.no_grad():
            B = data_arr.shape[0]
            dummy_img = torch.zeros((B, 3, 224, 224)).to(device)
            
            num_c = len(train_ds.numeric_cols)
            m_num = torch.tensor(data_arr[:, :num_c], dtype=torch.float32).to(device)
            
            m_cat_list = []
            curr = num_c
            for cc in train_ds.categorical_cols:
                card = train_ds.cat_cardinalities[cc]
                chunk = data_arr[:, curr:curr+card]
                idx = np.argmax(chunk, axis=1)
                m_cat_list.append(torch.tensor(idx, dtype=torch.long))
                curr += card
                
            m_cat = torch.stack(m_cat_list, dim=1).to(device)
            logits = model(dummy_img, m_num, m_cat)
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # 4. Compute SHAP
    explainer = shap.KernelExplainer(model_wrapper, bg_data)
    shap_vals = explainer.shap_values(test_data, nsamples=TEST_CONFIG['NSAMPLES_SHAP'])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    # 5. Plot & Save
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_vals, pd.DataFrame(test_data, columns=new_cols), show=False, max_display=20)
    
    save_name = f"ham10k_{TEST_CONFIG['SHORT_NAME']}_shap.png"
    save_path = os.path.join(TEST_CONFIG['MODEL_OUT'], save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 Đã lưu biểu đồ SHAP tại: {save_path}")

if __name__ == "__main__":
    model, device, train_ds, test_df = load_model_and_data(TEST_CONFIG)
    run_shap_analysis(model, train_ds, test_df, device)