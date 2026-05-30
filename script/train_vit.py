import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models.vit_fusion_head import ViT16_DualEmbeddingFusion
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# ------------------- CONFIG -------------------
CONFIG = {
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\bcn20000_train.csv',
    'VAL_CSV':   r'D:\skin_cancer_project\dataset\metadata\bcn20000_val.csv',
    'TEST_CSV':  r'D:\skin_cancer_project\dataset\metadata\bcn20000_test.csv',
    'IMG_ROOT':  r'D:\skin_cancer_project\dataset\Bcn20000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_bcn20000_vit',

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SEED': 42,

    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4,

    'EPOCHS': 20,
    'BASE_LR': 5e-5,
    'WARMUP_EPOCHS': 3,
    'WEIGHT_DECAY': 1e-4,

    # IMPORTANT: Late fusion đúng chuẩn trainer
    'METADATA_MODE': 'late_fusion',

    'PRETRAINED': True,
    'FINE_TUNE_MODE': 'partial_unfreeze',

    # repo logging
    'ACCUM_STEPS': 1,
    'LOG_GRADCAM_EVERY_EPOCH': False
}

def preprocess_bcn(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu và tạo nhãn chuẩn"""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'image_path' not in df.columns and 'isic_id' in df.columns:
        df['image_path'] = df['isic_id'].astype(str) + '.jpg'

    df['diagnosis_1'] = df['diagnosis_1'].astype(str).str.strip().str.lower()
    df = df[~df['diagnosis_1'].isin(['nan', '', 'none', 'null'])].copy()

    # Binary label: malignant vs others
    df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in x else 0)
    return df

def class_balanced_pos_weight(y, beta=0.9999) -> float:
    """CBW (effective number) -> pos_weight cho BCEWithLogitsLoss"""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    w_pos = (1 - beta) / (1 - beta ** n_pos)
    w_neg = (1 - beta) / (1 - beta ** n_neg)
    return float(w_pos / w_neg)

def infer_meta_dim_from_loader(loader) -> int:
    """
    Repo-strict: META_DIM phải lấy từ output dataset (late_fusion)
    Dataset trả: img, (meta_vec, zeros(0)), label
    """
    batch = next(iter(loader))
    if len(batch) != 3:
        raise ValueError(f"Unexpected batch length: {len(batch)}. Expected 3 items.")

    # Dataset của repo đang return: img, meta, label
    imgs, meta, labels = batch

    if not isinstance(meta, (tuple, list)) or len(meta) < 1:
        raise TypeError(f"meta must be tuple/list like (meta_vec, _). Got: {type(meta)}")

    meta_vec = meta[0]
    if not torch.is_tensor(meta_vec) or meta_vec.dim() != 2:
        raise ValueError(f"meta_vec must be 2D tensor (B, D). Got: {type(meta_vec)} shape={getattr(meta_vec,'shape',None)}")

    return int(meta_vec.shape[1])

def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    print("="*60)
    print(f"Thiết bị đang sử dụng: {device}")
    if device.type == 'cuda':
        print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*60)

    # 1) Load + preprocess
    print("Đang tải và làm sạch dữ liệu...")
    train_df = preprocess_bcn(pd.read_csv(config['TRAIN_CSV']))
    val_df   = preprocess_bcn(pd.read_csv(config['VAL_CSV']))
    test_df  = preprocess_bcn(pd.read_csv(config['TEST_CSV']))

    # 2) Datasets & Loaders (KHÔNG sửa repo)
    train_ds = DermoscopyDataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'],
                                 metadata_mode=config['METADATA_MODE'], train=True, selected_features=None)
    val_ds   = DermoscopyDataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'],
                                 metadata_mode=config['METADATA_MODE'], train=False, selected_features=None)
    test_ds  = DermoscopyDataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'],
                                 metadata_mode=config['METADATA_MODE'], train=False, selected_features=None)

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True,
                              num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False,
                              num_workers=config['NUM_WORKERS'], pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False,
                              num_workers=config['NUM_WORKERS'], pin_memory=True)

    # 3) Infer META_DIM từ dataset output (repo-strict)
    meta_dim = infer_meta_dim_from_loader(train_loader)
    print("✅ META_DIM inferred from dataset:", meta_dim)

    # 4) Init ViT Late Fusion model (signature đúng trainer)
    model = ViT16_DualEmbeddingFusion(
        pretrained=config['PRETRAINED'],
        meta_dim=meta_dim,
        num_classes=1,
        embed_dim=256
    ).to(device)
    print("✅ USING MODEL:", model.__class__.__name__)

    # finetune policy (repo utility)
    set_finetune_mode(model, config['FINE_TUNE_MODE'])

    # 5) CBW pos_weight cho BCE
    pos_w = class_balanced_pos_weight(train_df['label'].values, beta=0.9999)
    print("🔹 Class-Balanced pos_weight =", pos_w)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['BASE_LR'],
        weight_decay=config['WEIGHT_DECAY']
    )
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    print("\n🚀 --- BẮT ĐẦU HUẤN LUYỆN (ViT16 late_fusion + CBW) ---")
    train_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        criterion,
        optimizer,
        scheduler,
        device,
        log_suffix="vit16_latefusion_cbw"
    )

if __name__ == '__main__':
    main(CONFIG)
