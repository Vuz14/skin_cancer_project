import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.losses import FocalLossBCE
from src.data_logic.bcn_dataset import DermoscopyDataset
from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode

# trainer giữ nguyên
from src.utils.trainer_convnext import train_loop


# ================= CONFIG =================
CONFIG = {
    # BCN train/val (80/20 bạn đã split)
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\bcn20000_train80.csv',
    'VAL_CSV':   r'D:\skin_cancer_project\dataset\bcn20000_val20.csv',

    # HAM full test
    'TEST_CSV':  r'D:\skin_cancer_project\dataset\metadata\HAM10000_metadata.csv',

    'BCN_IMG_ROOT': r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'HAM_IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',

    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_cross',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,

    'MODEL_NAME': 'convnext',
    'SHORT_NAME': 'conv',

    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EPOCHS': 15,
    'BASE_LR': 1e-4,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-3,

    'METADATA_MODE': 'diag1',
    'METADATA_FEATURE_BOOST': 2.0,
    'META_CLASS_WEIGHT_BOOST': 1.0,

    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_KEYWORDS': ['conv_head', 'blocks.6', 'blocks.7'],

    'LOSS_TYPE': 'focal'
}


# ================= PREPROCESS =================
def preprocess_bcn(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.lower()
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

    return df


def preprocess_ham(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'image_path' not in df.columns and 'image_id' in df.columns:
        df['image_path'] = df['image_id'].astype(str) + '.jpg'

    df['dx'] = df['dx'].astype(str).str.lower()
    df = df[~df['dx'].isin(['nan', '', 'none', 'null'])].copy()

    # malignant mapping HAM
    malignant = ['mel', 'bcc', 'akiec']
    df['label'] = df['dx'].apply(lambda x: 1 if x in malignant else 0)

    return df


# ================= MAIN =================
def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])

    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print("📂 Loading BCN train/val...")
    raw_train = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    raw_val   = preprocess_bcn(pd.read_csv(config['VAL_CSV']))

    print("📂 Loading HAM test...")
    raw_test  = preprocess_ham(pd.read_csv(config['TEST_CSV']))

    print("🚀 Dataset init...")

    train_ds = DermoscopyDataset(
        raw_train, config['BCN_IMG_ROOT'],
        config['IMG_SIZE'], config['METADATA_MODE'], train=True
    )

    val_ds = DermoscopyDataset(
        raw_val, config['BCN_IMG_ROOT'],
        config['IMG_SIZE'], config['METADATA_MODE'], train=False
    )

    test_ds = HAM10000Dataset(
        raw_test, config['HAM_IMG_ROOT'],
        config['IMG_SIZE'], config['METADATA_MODE'], train=False
    )

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    print("🧠 Model init...")
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config.get('UNFREEZE_KEYWORDS'))

    # class weight từ BCN train
    y_train = raw_train['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']

    if config['LOSS_TYPE'] == 'focal':
        criterion = FocalLossBCE(alpha=0.75, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_val, device=device)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    log_suffix = f"cross_{config['SHORT_NAME']}"

    print("🚀 TRAIN CROSS DATASET BCN → HAM")
    train_loop(
        model, train_loader, val_loader, test_loader,
        config, criterion, optimizer, scheduler, device,
        log_suffix=log_suffix
    )


if __name__ == "__main__":
    main(CONFIG)