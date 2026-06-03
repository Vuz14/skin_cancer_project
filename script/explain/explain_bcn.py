import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.experiment_runner import preprocess_bcn

# ------------------- CONFIG -------------------
TEST_CONFIG = {
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_test.csv',
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_train.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',

    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'SHORT_NAME': 'effnet_b4',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'strategy3',
    'PRETRAINED': True,
    'NSAMPLES_SHAP': 50,

    # --- ÄÆ¯á»œNG DáºªN Tá»šI FILE QUAN TRá»ŒNG ÄÃƒ Cáº¬P NHáº¬T ---
    'FEATURE_IMP_CSV': r'D:\skin_cancer_project\checkpoint_bcn20000\bcn_meta_importance.csv',
    'TOP_K_FEATURES': 3 # Giá»¯ láº¡i 3 biáº¿n
}

def get_selected_features(config):
    try:
        imp_df = pd.read_csv(config['FEATURE_IMP_CSV'])
        imp_df['Base_Feature'] = imp_df['Feature'].apply(lambda x: x.split('=')[0] if '=' in x else x)
        selected_base_features = imp_df['Base_Feature'].drop_duplicates().head(config['TOP_K_FEATURES']).tolist()
        print(f"âœ… Sáº½ chá»‰ hiá»ƒn thá»‹ {len(selected_base_features)} biáº¿n trÃªn biá»ƒu Ä‘á»“: {selected_base_features}")
        return selected_base_features
    except Exception as e:
        print(f"âš ï¸ Lá»—i Ä‘á»c file CSV: {e}. Máº·c Ä‘á»‹nh láº¥y 3 biáº¿n: age_approx, anatom_site_general, sex")
        return ['age_approx', 'anatom_site_general', 'sex']

def load_model_and_encoders(config):
    device = torch.device(config['DEVICE'])
    train_df = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    test_df = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    selected_features = get_selected_features(config)

    # ðŸš€ GIá»® NGUYÃŠN Bá»˜ BIáº¾N Äá»‚ MÃ” HÃŒNH KHÃ”NG Bá»Š Lá»–I INDEX TENSOR
    train_ds = DermoscopyDataset(
        train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False
    )

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)

    # Chá»‰ Ä‘á»‹nh Ä‘áº¿n Fold 4 (nÆ¡i cÃ³ file model cá»§a báº¡n)
    ckpt_dir = os.path.join(config['MODEL_OUT'], f"CV5_{config['METADATA_MODE']}_{config['SHORT_NAME']}_bcn20000", "fold_2")
    ckpt_path = os.path.join(ckpt_dir, f"best_{config['METADATA_MODE']}_fold_2.pt")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
    else:
        print(f"âŒ KhÃ´ng tÃ¬m tháº¥y: {ckpt_path}")
        return None, None, None, None, None

    model.eval()
    return model, device, train_ds, test_df, selected_features

def run_shap_analysis(model, train_ds, test_df, device, selected_features):
    if model is None: return
    print("â³ Äang tÃ­nh toÃ¡n SHAP values...")

    # Láº¥y máº«u dá»¯ liá»‡u
    subset_df = test_df.sample(n=min(30, len(test_df)), random_state=42)
    bg_df = train_ds.df.sample(n=min(15, len(train_ds.df)), random_state=123)

    def to_onehot_flat(df_in):
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

    test_data, col_names = to_onehot_flat(subset_df)
    bg_data, _ = to_onehot_flat(bg_df)

    def model_wrapper(m_arr):
        with torch.no_grad():
            K = m_arr.shape[0]
            BATCH_SIZE = 8
            all_preds = []
            for i in range(0, K, BATCH_SIZE):
                chunk = m_arr[i : i + BATCH_SIZE]
                bs = chunk.shape[0]
                dummy_img = torch.zeros((bs, 3, TEST_CONFIG['IMG_SIZE'], TEST_CONFIG['IMG_SIZE'])).to(device)
                num_c = len(train_ds.numeric_cols)
                m_num = torch.tensor(chunk[:, :num_c], dtype=torch.float32).to(device)
                m_cat_l = []
                off = num_c
                for cc in train_ds.categorical_cols:
                    card = train_ds.cat_cardinalities[cc]
                    c_chunk = chunk[:, off:off+card]
                    m_cat_l.append(torch.tensor(np.argmax(c_chunk, axis=1), dtype=torch.long))
                    off += card
                m_cat = torch.stack(m_cat_l, dim=1).to(device) if m_cat_l else torch.zeros((bs, 0), dtype=torch.long).to(device)
                preds = torch.sigmoid(model(dummy_img, m_num, m_cat)).cpu().numpy().reshape(-1)
                all_preds.extend(preds)
            return np.array(all_preds)

    explainer = shap.KernelExplainer(model_wrapper, bg_data)
    shap_vals = explainer.shap_values(test_data, nsamples=TEST_CONFIG['NSAMPLES_SHAP'])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    # --- Lá»ŒC BIáº¾N ---
    keep_indices, filtered_col_names = [], []
    for i, col in enumerate(col_names):
        if any(col == sf or col.startswith(sf + "_") for sf in selected_features):
            keep_indices.append(i)
            filtered_col_names.append(col)

    filtered_shap_vals = shap_vals[:, keep_indices]
    filtered_test_data = test_data[:, keep_indices]
    final_cols = [f"X{i+1}" for i in range(len(filtered_col_names))]

    # --- Cáº¤U HÃŒNH Váº¼ BIá»‚U Äá»’ ---
    FONT_SIZE = 26  # TÄƒng font chá»¯ lÃªn má»©c ráº¥t to
    plt.rcParams.update({'font.size': FONT_SIZE})

    # TÄƒng figsize rá»™ng ra Ä‘á»ƒ cÃ¡c dáº¥u cháº¥m to khÃ´ng bá»‹ Ä‘Ã¨ nhau
    fig = plt.figure(figsize=(18, 11))

    # Váº½ SHAP
    # Tham sá»‘ quan trá»ng: 's' Ä‘iá»u chá»‰nh kÃ­ch thÆ°á»›c cháº¥m (máº·c Ä‘á»‹nh thÆ°á»ng lÃ  15-20)
    shap.summary_plot(
        filtered_shap_vals,
        pd.DataFrame(filtered_test_data, columns=final_cols),
        show=False,
        max_display=len(final_cols),
        plot_size=None,
        alpha=0.8, # TÄƒng Ä‘á»™ Ä‘áº­m cá»§a cháº¥m
    )

    # Can thiá»‡p vÃ o cÃ¡c thÃ nh pháº§n Ä‘Ã£ váº½ Ä‘á»ƒ tÄƒng size dáº¥u cháº¥m
    ax = plt.gca()
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.collections.PathCollection):
            child.set_sizes([100]) # Ã‰p kÃ­ch thÆ°á»›c táº¥t cáº£ dáº¥u cháº¥m lÃªn 100 (to gáº¥p 5 láº§n máº·c Ä‘á»‹nh)

    # 1. XÃ³a Ä‘á»‹nh dáº¡ng 1e-5 (Scientific notation)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='x')

    # 2. TÄƒng kÃ­ch thÆ°á»›c chá»¯ cho Ticks & Labels
    ax.tick_params(axis='y', labelsize=FONT_SIZE + 6) # X1, X2... to vÆ°á»£t trá»™i
    ax.tick_params(axis='x', labelsize=FONT_SIZE)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE + 2, fontweight='bold')

    # 3. Chá»‰nh Colorbar (Thanh mÃ u bÃªn pháº£i)
    cfm = plt.gcf()
    if len(cfm.axes) > 1:
        cbar_ax = cfm.axes[-1]
        cbar_ax.set_ylabel('Feature value', fontsize=FONT_SIZE + 2, fontweight='bold')
        cbar_ax.tick_params(labelsize=FONT_SIZE)

    # LÆ°u áº£nh vá»›i cháº¥t lÆ°á»£ng cao
    save_path = os.path.join(TEST_CONFIG['MODEL_OUT'], f"bcn20k_{TEST_CONFIG['SHORT_NAME']}_big_dots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"ðŸŽ‰ Biá»ƒu Ä‘á»“ SIÃŠU TO KHá»”NG Lá»’ Ä‘Ã£ lÆ°u táº¡i: {save_path}")

if __name__ == "__main__":
    model, device, train_ds, test_df, selected_features = load_model_and_encoders(TEST_CONFIG)
    run_shap_analysis(model, train_ds, test_df, device, selected_features)
