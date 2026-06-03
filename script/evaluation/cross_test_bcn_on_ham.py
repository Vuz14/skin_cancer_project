import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# ThÃªm Ä‘Æ°á»ng dáº«n gá»‘c Ä‘á»ƒ import cÃ¡c module tá»« src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# DÃ¹ng Dataset cá»§a BCN (DermoscopyDataset) vÃ¬ Model "hiá»ƒu" format cá»§a BCN
from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.trainer import evaluate
from src.utils.common import load_metadata_info, seed_everything

# ==============================================================================
# 1. Cáº¤U HÃŒNH (CONFIG)
# ==============================================================================
CONFIG = {
    # 1. Dá»® LIá»†U ÄÃCH (HAM10000)
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_test.csv',
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-color-safe-preprocessed',
    'SEED': 42,
    # 2. MODEL NGUá»’N (BCN20000)
    # LÆ°u Ã½: Thay Ä‘á»•i Ä‘Æ°á»ng dáº«n nÃ y trá» Ä‘áº¿n file .pt tá»‘t nháº¥t cá»§a BCN (vÃ­ dá»¥ trong thÆ° má»¥c fold_1 náº¿u dÃ¹ng CV5)
    'CHECKPOINT_PATH': r'D:\skin_cancer_project\checkpoint_bcn20000\CV5_strategy3_effnet_b4_bcn20000\fold_4\best_strategy3_fold_4.pt',

    # File .pkl chá»©a Metadata Encoder vÃ  Stats Ä‘Æ°á»£c sinh ra lÃºc train model BCN
    'META_INFO_PATH': r'D:\skin_cancer_project\checkpoint_bcn20000\CV5_strategy3_effnet_b4_bcn20000\fold_4\meta_info_fold4.pkl',

    # 3. THÃ”NG Sá» KHÃC (Pháº£i khá»›p chÃ­nh xÃ¡c vá»›i lÃºc báº¡n train model BCN)
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'IMG_SIZE': 224,
    'METADATA_MODE': 'strategy3',
    'MODEL_NAME': 'tf_efficientnet_b4_ns',
    'BATCH_SIZE': 32,
    'PRETRAINED': True,
    'METADATA_FEATURE_BOOST': 1.0
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-test a BCN-trained model on HAM10000.")
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

# ==============================================================================
# 2. HÃ€M MAPPING: HAM -> BCN
# ==============================================================================
def map_ham_to_bcn(ham_df):
    """
    Biáº¿n Ä‘á»•i dá»¯ liá»‡u HAM (nguá»“n) -> Format cá»§a BCN (Ä‘Ã­ch)
    BCN Columns cáº§n: age_approx, sex, anatom_site_general
    """
    print("ðŸ”„ Äang mapping dá»¯ liá»‡u HAM sang format BCN...")
    df = ham_df.copy()

    # --- Táº O Cá»˜T IMAGE_PATH Náº¾U CHÆ¯A CÃ“ ---
    # HAM dataset thÆ°á»ng cÃ³ image_id, cáº§n táº¡o image_path Ä‘á»ƒ loader Ä‘á»c Ä‘Æ°á»£c
    if 'image_path' not in df.columns:
        if 'image_id' in df.columns:
            df['image_path'] = df['image_id'].astype(str) + '.jpg'
    # --------------------------------------

    # 1. Äá»•i tÃªn cá»™t cho khá»›p vá»›i BCN
    # HAM: age -> BCN: age_approx
    # HAM: localization -> BCN: anatom_site_general
    df = df.rename(columns={
        'age': 'age_approx',
        'localization': 'anatom_site_general'
    })

    # 2. Xá»­ lÃ½ Giá»›i tÃ­nh (Sex) - Chuáº©n hÃ³a vá» chá»¯ thÆ°á»ng
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.lower()

    # 3. Xá»­ lÃ½ Vá»‹ trÃ­ (Mapping quan trá»ng nháº¥t)
    # BCN dÃ¹ng cÃ¡c vá»‹ trÃ­ tá»•ng quÃ¡t hÆ¡n HAM
    loc_mapping = {
        # VÃ¹ng thÃ¢n
        'abdomen': 'anterior torso',
        'chest': 'anterior torso',
        'back': 'posterior torso',
        'trunk': 'anterior torso',
        'lateral torso': 'anterior torso',  # Map táº¡m

        # VÃ¹ng Ä‘áº§u cá»•
        'face': 'head/neck',
        'neck': 'head/neck',
        'scalp': 'head/neck',
        'ear': 'head/neck',

        # Chi dÆ°á»›i
        'foot': 'lower extremity',
        'lower extremity': 'lower extremity',

        # Chi trÃªn
        'hand': 'upper extremity',
        'upper extremity': 'upper extremity',

        # VÃ¹ng Ä‘áº·c biá»‡t
        'genital': 'oral/genital',
        'acral': 'palms/soles',

        # KhÃ´ng xÃ¡c Ä‘á»‹nh
        'unknown': 'NA',
        'nan': 'NA'
    }

    if 'anatom_site_general' in df.columns:
        df['anatom_site_general'] = df['anatom_site_general'].map(loc_mapping).fillna('NA')
    else:
        print("âš ï¸ Cáº£nh bÃ¡o: KhÃ´ng tÃ¬m tháº¥y cá»™t 'localization' (Ä‘Ã£ rename) trong HAM.")

    print("âœ… Mapping hoÃ n táº¥t.")
    return df


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    apply_overrides(parse_args())
    seed_everything(CONFIG['SEED'])
    device = torch.device(CONFIG['DEVICE'])
    print(f"ðŸ”¥ Thiáº¿t bá»‹: {device}")

    # 1. Load Metadata Info tá»« táº­p TRAIN (BCN)
    print(f"ðŸ“‚ Loading Metadata Encoders cá»§a BCN tá»«: {CONFIG['META_INFO_PATH']}")
    if CONFIG['METADATA_MODE'] != 'strategy1':
        if not os.path.exists(CONFIG['META_INFO_PATH']):
            raise FileNotFoundError(f"âŒ KhÃ´ng tÃ¬m tháº¥y file metadata info: {CONFIG['META_INFO_PATH']}")
        encoders, num_stats = load_metadata_info(CONFIG['META_INFO_PATH'])
    else:
        encoders, num_stats = None, None

    # 2. Load vÃ  Map dá»¯ liá»‡u TEST (HAM)
    print(f"ðŸ“‚ Loading HAM Data tá»«: {CONFIG['TEST_CSV']}")
    test_df = pd.read_csv(CONFIG['TEST_CSV'])
    test_df.columns = test_df.columns.str.strip()  # XÃ³a khoáº£ng tráº¯ng thá»«a á»Ÿ tÃªn cá»™t

    # Xá»­ lÃ½ nhÃ£n cho HAM (Chuáº©n hÃ³a vá» 0/1)
    if 'dx' in test_df.columns:
        test_df['label'] = test_df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

    # Ãp dá»¥ng Mapping Ä‘á»ƒ biáº¿n DataFrame HAM thÃ nh dáº¡ng BCN
    mapped_test_df = map_ham_to_bcn(test_df)

    # 3. Táº¡o Dataset
    # QUAN TRá»ŒNG: DÃ¹ng DermoscopyDataset (Class cá»§a BCN) nhÆ°ng chá»©a data HAM Ä‘Ã£ map
    print("ðŸš€ Khá»Ÿi táº¡o Dataset...")
    test_ds = DermoscopyDataset(
        df=mapped_test_df,
        img_root=CONFIG['IMG_ROOT'],
        img_size=CONFIG['IMG_SIZE'],
        metadata_mode=CONFIG['METADATA_MODE'],
        train=False,
        external_encoders=encoders,  # Truyá»n encoder cá»§a BCN vÃ o
        external_stats=num_stats
    )

    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 4. Load Model
    print("ðŸ¤– Loading Model BCN...")
    cat_cardinalities = test_ds.cat_cardinalities if encoders else {}
    num_numeric = len(test_ds.numeric_cols) if encoders else 0

    # Khá»Ÿi táº¡o model vá»›i cáº¥u trÃºc y há»‡t lÃºc train BCN
    model = get_model(CONFIG, cat_cardinalities, num_numeric).to(device)

    # Load trá»ng sá»‘ (Weights)
    if not os.path.exists(CONFIG['CHECKPOINT_PATH']):
        raise FileNotFoundError(f"âŒ KhÃ´ng tÃ¬m tháº¥y file checkpoint: {CONFIG['CHECKPOINT_PATH']}")

    checkpoint = torch.load(CONFIG['CHECKPOINT_PATH'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # 5. Evaluate
    print(f"ðŸš€ Äang cháº¡y Ä‘Ã¡nh giÃ¡ Model BCN trÃªn táº­p HAM10000...")
    print(f"   - Metadata Mode: {CONFIG['METADATA_MODE']}")
    print(f"   - Image Size: {CONFIG['IMG_SIZE']}")

    results = evaluate(model, test_loader, device=device)

    print("\n" + "=" * 40)
    print(f"Káº¾T QUáº¢: TRAIN BCN20000 -> TEST HAM10000")
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
