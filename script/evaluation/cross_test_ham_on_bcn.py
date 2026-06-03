import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.trainer import evaluate
from src.utils.common import load_metadata_info, seed_everything

# --- Cáº¤U HÃŒNH ---
CONFIG = {
    # 1. Dá»® LIá»†U ÄÃCH (BCN20000)
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed',
    'SEED': 42,
    # 2. MODEL NGUá»’N (HAM10000)
    # Trá» Ä‘áº¿n checkpoint .pt cá»§a HAM
    'CHECKPOINT_PATH': r'D:\skin_cancer_project\checkpoint_ham10000\CV5_strategy3_effnet_b4_ham10000\fold_3\best_strategy3_fold_3.pt',

    # File .pkl chá»©a encoder cá»§a HAM
    'META_INFO_PATH': r'D:\skin_cancer_project\checkpoint_ham10000\CV5_strategy3_effnet_b4_ham10000\fold_3\meta_info_fold3.pkl',

    # 3. THÃ”NG Sá» KHÃC (Khá»›p vá»›i cáº¥u hÃ¬nh lÃºc train HAM)
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'strategy3',
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'BATCH_SIZE': 32,
    'PRETRAINED': True,
    'METADATA_FEATURE_BOOST': 2.0
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-test a HAM-trained model on BCN20000.")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--img-root", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--meta-info-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metadata-mode", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--metadata-feature-boost", type=float, default=None)
    return parser.parse_args()


def apply_overrides(args):
    for attr, key in {
        "test_csv": "TEST_CSV",
        "img_root": "IMG_ROOT",
        "checkpoint_path": "CHECKPOINT_PATH",
        "meta_info_path": "META_INFO_PATH",
        "device": "DEVICE",
        "metadata_mode": "METADATA_MODE",
        "model_name": "MODEL_NAME",
        "batch_size": "BATCH_SIZE",
        "metadata_feature_boost": "METADATA_FEATURE_BOOST",
    }.items():
        value = getattr(args, attr)
        if value is not None:
            CONFIG[key] = value

def map_bcn_to_ham(bcn_df):
    print("ðŸ”„ Äang mapping dá»¯ liá»‡u BCN sang format HAM...")
    df = bcn_df.copy()

    # Táº¡o cá»™t image_path náº¿u chÆ°a cÃ³
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

    print("âœ… Mapping features hoÃ n táº¥t.")
    return df


def main():
    apply_overrides(parse_args())
    seed_everything(42)
    device = torch.device(CONFIG['DEVICE'])

    print("ðŸ“‚ Loading Metadata Encoders cá»§a HAM...")
    if CONFIG['METADATA_MODE'] != 'strategy1':
        if not os.path.exists(CONFIG['META_INFO_PATH']):
            print(f"âŒ Lá»—i: KhÃ´ng tÃ¬m tháº¥y file {CONFIG['META_INFO_PATH']}")
            return
        encoders, num_stats = load_metadata_info(CONFIG['META_INFO_PATH'])
    else:
        encoders, num_stats = None, None

    print("ðŸ“‚ Loading BCN Data...")
    test_df = pd.read_csv(CONFIG['TEST_CSV'])

    if 'label' not in test_df.columns:
        diag_col = 'diagnosis_1' if 'diagnosis_1' in test_df.columns else 'diagnosis'
        malignant_list = ['malignant', 'mel', 'bcc', 'scc', 'melanoma', 'basal cell', 'squamous cell', 'carcinoma']
        test_df['label'] = test_df[diag_col].astype(str).str.lower().apply(
            lambda x: 1 if any(m in x for m in malignant_list) else 0
        )

    # --- DEBUG: KIá»‚M TRA PHÃ‚N PHá»I NHÃƒN ---
    label_counts = test_df['label'].value_counts()
    print("\nðŸ“Š THá»NG KÃŠ NHÃƒN TRONG Táº¬P TEST:")
    print(label_counts)
    if len(label_counts) < 2:
        print("âš ï¸ Cáº¢NH BÃO: Táº­p test chá»‰ cÃ³ 1 loáº¡i nhÃ£n! AUC sáº½ luÃ´n báº±ng 0.")
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

    print("ðŸ¤– Loading Model...")
    cat_cardinalities = test_ds.cat_cardinalities if encoders else {}
    num_numeric = len(test_ds.numeric_cols) if encoders else 0

    model = get_model(CONFIG, cat_cardinalities, num_numeric).to(device)

    if not os.path.exists(CONFIG['CHECKPOINT_PATH']):
        print(f"âŒ KhÃ´ng tÃ¬m tháº¥y file checkpoint táº¡i: {CONFIG['CHECKPOINT_PATH']}")
        return

    checkpoint = torch.load(CONFIG['CHECKPOINT_PATH'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print("ðŸš€ Äang cháº¡y Ä‘Ã¡nh giÃ¡ chÃ©o...")
    results = evaluate(model, test_loader, device=device)

    print("\n" + "=" * 30)
    print(f"Káº¾T QUáº¢ TEST HAM MODEL TRÃŠN Táº¬P BCN ({CONFIG['METADATA_MODE']})")
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
