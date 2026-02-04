import sys, os, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), 'D:\skin_cancer_project'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.losses import FocalLoss
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop # Import train_loop dùng chung
# , full, full_weight, late_fusion
CONFIG = {
    'CSV_PATH': 'D:\skin_cancer_project\dataset\metadata\ham10000_processed.csv',
    'IMG_ROOT': 'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': 'D:\\skin_cancer_project\checkpoint_ham10000',
    'DEVICE': 'cuda', 
    'SEED': 42, 
    'IMG_SIZE': 224, 
    'BATCH_SIZE': 16, 
    'EPOCHS': 10,
    'BASE_LR': 5e-4, 
    'BACKBONE_LR_MULT': 0.1, 
    'WEIGHT_DECAY': 5e-4,
    'METADATA_MODE': 'full_weighted', 
    'METADATA_FEATURE_BOOST': 5.0,
    'PRETRAINED': True, 
    'FINE_TUNE_MODE': 'partial_unfreeze',
    'UNFREEZE_SUBSTRINGS': ['layers', 'blocks', 'norm', 'conv_head', 'features', 'stem'],
    'WARMUP_EPOCHS': 2, 
    'T_TOTAL': 45, 
    'SCHEDULER': 'warmup_cosine',
    'USE_WANDB': True, 
    'WANDB_PROJECT': "nckh_skin_cancer_2025", 
    'WANDB_API_KEY': "0792e63c409597338d7cdd72f51375cff888373f",
    'USE_SAMPLER': True,
    
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
    
    # 1. Load Data
    df = pd.read_csv(config['CSV_PATH'])
    
    # --- FIX LỖI KEYERROR: Tạo cột image_path từ image_id ---
    if 'image_id' in df.columns:
        df['image_path'] = df['image_id'].astype(str)
        # Nếu chưa có đuôi .jpg thì thêm vào
        if not df['image_path'].iloc[0].endswith('.jpg'):
            df['image_path'] += '.jpg'
    elif 'image_path' in df.columns:
        pass
    else:
        raise ValueError("CSV phải chứa cột 'image_id' hoặc 'image_path'")
    # ---------------------------------------------------------

    if 'dx' in df.columns: 
        df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
    
    # 2. Split
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=config['SEED'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=config['SEED'])

    # 3. Dataset
    train_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], True)
    val_ds = HAM10000Dataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], False)
    test_ds = HAM10000Dataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], False)

    # 4. Sampler
    train_sampler = None
    if config['USE_SAMPLER']:
        targets = train_df['label'].values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = torch.from_numpy(weights[targets]).double()
        train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler, num_workers=0) # Windows nên để num_workers=0 nếu lỗi
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 5. Model
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config['UNFREEZE_SUBSTRINGS'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['T_TOTAL'])

    # 6. Train
    train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device, log_suffix="10k")

if __name__ == '__main__':
    main(CONFIG)