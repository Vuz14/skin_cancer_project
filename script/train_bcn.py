import sys, os, torch, torch.nn as nn, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(os.path.join(os.path.dirname(__file__), 'D:\skin_cancer_project'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# , full, full_weight, late_fusion

CONFIG = {
    'CSV_PATH': 'D:\skin_cancer_project\dataset\metadata\metadata.csv',
    'IMG_ROOT': 'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': 'D:\\skin_cancer_project\checkpoint_bcn20000',
    'DEVICE': 'cuda', 
    'SEED': 42, 
    'IMG_SIZE': 224, 
    'BATCH_SIZE': 16, 
    'EPOCHS': 10,
    'METADATA_MODE': 'full_weighted', 
    'METADATA_FEATURE_BOOST': 5.0,
    'META_CLASS_WEIGHT_BOOST': 3.0, 
    'PRETRAINED': True, 
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'USE_WANDB': True, 
    'WANDB_PROJECT': "nckh_skin_cancer_2025", 
    'WANDB_API_KEY': "0792e63c409597338d7cdd72f51375cff888373f",
    
    # --- THÊM DÒNG NÀY ĐỂ FIX LỖI ---
    'WANDB_LOG_FREQ': 100,
    # -------------------------------
    'GRADCAM_SAVE_EVERY': 5,
    'LOG_GRADCAM_EVERY_EPOCH': True,
    'ACCUM_STEPS': 1
}
def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    df = pd.read_csv(config['CSV_PATH'])
    
    # --- FIX LỖI KEYERROR: Tạo cột image_path từ isic_id ---
    if 'image_path' not in df.columns:
        if 'isic_id' in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        else:
            raise ValueError("CSV must contain image_path or isic_id column")
    # -------------------------------------------------------

    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])]
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=config['SEED'])
    val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df['label'], random_state=config['SEED'])

    train_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], True)
    val_ds = DermoscopyDataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], False)
    test_ds = DermoscopyDataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], False)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'])

    y_train = train_df['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight_val = weights[1] * config['META_CLASS_WEIGHT_BOOST']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, 2, config['EPOCHS'])

    train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device, log_suffix="20k")

if __name__ == '__main__':
    main(CONFIG)