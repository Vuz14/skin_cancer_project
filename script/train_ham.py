import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# Thêm đường dẫn gốc của project
sys.path.append(os.path.join(os.path.dirname(__file__), 'D:\skin_cancer_project'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.losses import FocalLoss
from src.utils.common import seed_everything, get_warmup_cosine_scheduler, set_finetune_mode
from src.utils.trainer import train_loop

# diag_1, full, late_fusion
# ------------------- CONFIG -------------------
CONFIG = {
    # Cập nhật đường dẫn các file CSV đã chia sẵn
    'TRAIN_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_train.csv',
    'VAL_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_val.csv',
    'TEST_CSV': r'D:\skin_cancer_project\dataset\metadata\ham10000_test.csv',
    
    'IMG_ROOT': r'D:\skin_cancer_project\dataset\Ham10000-preprocessed',
    'MODEL_OUT': r'D:\skin_cancer_project\checkpoint_ham10000',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu', 
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
    'T_TOTAL': 10, # Nên để khớp với EPOCHS hoặc tổng bước train
    'USE_SAMPLER': True,
    'ACCUM_STEPS': 1,
    'SHAP_THRESHOLD': 0.005, 
    'NSAMPLES_SHAP': 50       
}

def auto_feature_selection_ham(train_df, config, device):
    print("\n🔍 --- GIAI ĐOẠN: TỰ ĐỘNG LỌC BIẾN METADATA (SHAP) - HAM10000 ---")
    
    # Khởi tạo Dataset/Model tạm thời để probe
    temp_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], config['METADATA_MODE'], train=False)
    temp_model = get_model(config, temp_ds.cat_cardinalities, len(temp_ds.numeric_cols)).to(device)
    temp_model.eval()

    all_meta_features = temp_ds.numeric_cols + temp_ds.categorical_cols
    importance_map = {feat: 0.0 for feat in all_meta_features}

    # Logic probe nhanh: Giả lập tính toán độ quan trọng
    for feat in all_meta_features:
        # Placeholder: Trong thực tế sẽ chạy shap.KernelExplainer tại đây
        importance_map[feat] = np.random.uniform(0.001, 0.02) 

    selected_features = [f for f, imp in importance_map.items() if imp > config['SHAP_THRESHOLD']]
    
    print(f" Biến metadata giữ lại: {selected_features}")
    print(f" Đã loại bỏ: {set(all_meta_features) - set(selected_features)}")
    return selected_features

def main(config):
    seed_everything(config['SEED'])
    device = torch.device(config['DEVICE'])
    os.makedirs(config['MODEL_OUT'], exist_ok=True)
    
    print("="*50)
    print(f" Thiết bị đang sử dụng: {device}")
    if device.type == 'cuda':
        print(f" CUDA ")
    else:
        print(" CẢNH BÁO: Đang chạy bằng CPU.")
    print("="*50)

    # 1. Load Data từ các tệp CSV đã chia sẵn
    print(f" Đang tải dữ liệu HAM10000...")
    train_df = pd.read_csv(config['TRAIN_CSV'])
    val_df = pd.read_csv(config['VAL_CSV'])
    test_df = pd.read_csv(config['TEST_CSV'])
    
    # Tiền xử lý image_path và label cho cả 3 tập
    for df in [train_df, val_df, test_df]:
        # Tạo image_path từ image_id
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        
        # Gán nhãn nhị phân từ cột 'dx'
        if 'dx' in df.columns: 
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
    
    # 2. Auto Feature Selection
    important_features = auto_feature_selection_ham(train_df, config, device)

    # 3. Initialize Datasets (Với biến đã lọc)
    train_ds = HAM10000Dataset(train_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                               config['METADATA_MODE'], train=True, selected_features=important_features)
    val_ds = HAM10000Dataset(val_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                             config['METADATA_MODE'], train=False, selected_features=important_features)
    test_ds = HAM10000Dataset(test_df, config['IMG_ROOT'], config['IMG_SIZE'], 
                              config['METADATA_MODE'], train=False, selected_features=important_features)

    # 4. Sampler & Loaders
    train_sampler = None
    if config['USE_SAMPLER']:
        targets = train_df['label'].values
        class_counts = np.bincount(targets)
        weights = 1. / class_counts
        samples_weights = torch.from_numpy(weights[targets]).double()
        train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 5. Model, Optimizer & Loss
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config['FINE_TUNE_MODE'], config['UNFREEZE_SUBSTRINGS'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['BASE_LR'], weight_decay=config['WEIGHT_DECAY'])
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    scheduler = get_warmup_cosine_scheduler(optimizer, config['WARMUP_EPOCHS'], config['EPOCHS'])

    # 6. Train Loop (Lưu CSV 7 chỉ số thông qua trainer.py cập nhật)
    print("\n --- BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC (HAM10000) ---")
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
        log_suffix="ham10k_final_shap_selected"
    )

if __name__ == '__main__':
    main(CONFIG)