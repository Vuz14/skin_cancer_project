import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve

# Kiểm tra thư viện Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0

def generate_gradcam(model, img_tensor, save_dir, idx):
    """
    Grad-CAM chuẩn cho EfficientNet + dermoscopy
    """
    model.eval()
    device = next(model.parameters()).device

    # -------- Wrapper: image-only --------
    class _ImageOnlyWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            bs = x.size(0)
            dev = x.device
            num_numeric = getattr(self.base_model, 'num_numeric', 0)

            if hasattr(self.base_model, 'cat_names'):
                cat_dim = len(self.base_model.cat_names)
            elif hasattr(self.base_model, 'cat_cardinalities'):
                cat_dim = len(self.base_model.cat_cardinalities)
            else:
                cat_dim = 0

            meta_num = torch.zeros((bs, num_numeric), device=dev)
            meta_cat = torch.zeros((bs, cat_dim), dtype=torch.long, device=dev)
            return self.base_model(x, meta_num, meta_cat)

    target_layer = None
    layer_name = "unknown"

    try:
        # Trường hợp 1: EfficientNet (blocks)
        if hasattr(model.backbone, 'model'):
            eff = model.backbone.model
        if hasattr(eff, 'conv_head'):
            target_layer = eff.conv_head
            layer_name = "efficientnet_conv_head"
        elif hasattr(eff, 'blocks'):
            target_layer = eff.blocks[-1]
            layer_name = "efficientnet_blocks[-1]"
        # Trường hợp 2: ResNet (layer4)
        elif hasattr(model.backbone, 'backbone') and hasattr(model.backbone.backbone, 'layer4'):
            target_layer = model.backbone.backbone.layer4[-1]
            layer_name = "resnet_layer4"
    except Exception as e:
        return

    if target_layer is None: return

    cam = GradCAM(model=_ImageOnlyWrapper(model), target_layers=[target_layer])

    if img_tensor.ndim == 3: img_input = img_tensor.unsqueeze(0)
    else: img_input = img_tensor

    img_input = img_input.to(device).float()
    img_input.requires_grad_(True)

    try:
        targets = [ClassifierOutputTarget(0)]  
        grayscale_cam = cam(input_tensor=img_input, targets=targets)[0]
    except Exception: return

    img_np = img_input[0].detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = np.clip(img_np * std + mean, 0, 1)

    heatmap = np.uint8(255 * grayscale_cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    cam_image = np.uint8(255 * np.clip(0.6 * img_rgb + 0.4 * heatmap, 0, 1))
    
    cv2.putText(cam_image, f"Grad-CAM: {layer_name}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"sample_{idx}.png"), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

def plot_metrics(history_data, test_metrics, out_dir, log_suffix):
    """
    Vẽ biểu đồ Metric và lưu file không có chữ 'combined'
    Tên file: {metric}_{log_suffix}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(history_data)
    epochs = df['epoch']
    metrics = ['loss', 'auc', 'acc', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df[f'train_{metric}'], label=f'Train {metric.upper()}', marker='o')
        plt.plot(epochs, df[f'val_{metric}'], label=f'Val {metric.upper()}', marker='s')
        
        test_val = test_metrics.get(metric)
        if test_val is not None:
            plt.axhline(y=test_val, color='r', linestyle='--', label=f'Test {metric.upper()} ({test_val:.4f})')
        
        plt.title(f'{metric.upper()} History - {log_suffix}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        
        # Lưu file tên ngắn gọn
        save_path = os.path.join(out_dir, f"{metric}_{log_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def evaluate(model: nn.Module, loader, device='cpu', is_late_fusion=None, threshold=0.5, find_best_threshold=False):
    model.eval()
    loss_sum = 0.0
    all_probs, all_targets = [], []
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    if is_late_fusion is None:
        is_late_fusion = hasattr(model, "metadata_mode") and model.metadata_mode == "late_fusion"

    with torch.no_grad():
        for batch in loader:
            imgs, meta, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float32).view(-1, 1)

            if is_late_fusion:
                meta_vec, _ = meta
                logits = model(imgs, meta_vec.to(device).float(), None) # Late fusion dùng meta_vec
            else:
                m_num, m_cat = meta
                logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())

            loss = bce(logits, labels)
            loss_sum += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().reshape(-1).tolist())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)

    auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0.0

    if find_best_threshold and len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        youden_index = tpr - fpr
        threshold = thresholds[np.argmax(youden_index)]
        print(f"🔥 Best threshold found on Val: {threshold:.4f}")

    y_pred = (y_probs >= threshold).astype(int)

    return {
        'loss': loss_sum / len(loader.dataset),
        'auc': auc,
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'spec': calculate_specificity(y_true, y_pred),
        'threshold': threshold
    }

def train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device, log_suffix=""):
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    is_late_fusion = (config["METADATA_MODE"] == "late_fusion")
    
    best_val_auc = 0.0
    history_data = []
    
    patience = config.get('PATIENCE', 5)
    counter = 0
    
    # File log chi tiết từng epoch
    history_csv = os.path.join(config['MODEL_OUT'], f"history_epoch_{log_suffix}.csv")

    best_threshold = 0.5

    for epoch in range(1, config['EPOCHS'] + 1):
        model.train()
        train_loss_sum = 0.0
        
        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if is_late_fusion:
                    meta_vec, _ = meta
                    logits = model(imgs, meta_vec.to(device).float(), None)
                else:
                    m_num, m_cat = meta
                    logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())
                
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * imgs.size(0)

        # Đánh giá Metrics
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion, find_best_threshold=True)
        best_threshold = val_res['threshold'] # Update best threshold theo Val

        # Log kết quả Epoch
        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)
        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        print(f"Epoch {epoch} | Val AUC: {val_res['auc']:.4f} | Val Loss: {val_res['loss']:.4f}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_res['auc'])
            else:
                scheduler.step()
        
        # Checkpoint & Early Stopping
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            counter = 0
            torch.save({'state_dict': model.state_dict()}, 
                       os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt"))
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs.")
                break

        # Grad-CAM
        if HAS_GRADCAM and config.get('ENABLE_GRAD_CAM', False) and epoch % config.get('GRAD_CAM_FREQ', 5) == 0:
            save_dir = os.path.join(config['MODEL_OUT'], f"gradcam_ep{epoch}")
            val_batch = next(iter(val_loader))
            val_imgs = val_batch[0]
            for idx in range(min(4, len(val_imgs))):
                generate_gradcam(model, val_imgs[idx:idx+1], save_dir, idx)

    # --- ĐÁNH GIÁ TEST ---
    best_ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device)['state_dict'])

    print("\n🚀 Đánh giá trên tập Test...")
    test_metrics = evaluate(model, test_loader, device, is_late_fusion, threshold=best_threshold, find_best_threshold=False)

    # --- LƯU KẾT QUẢ TỔNG HỢP (APPEND MODE) ---
    summary_file = os.path.join(config['MODEL_OUT'], "experiment_summary_results.csv")
    new_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': config.get('SHORT_NAME', 'unknown'),
        'Metadata_Mode': config['METADATA_MODE'],
        'Epochs_Run': epoch,
        'Best_Val_AUC': best_val_auc,
        'Test_AUC': test_metrics['auc'],
        'Test_Acc': test_metrics['acc'],
        'Test_F1': test_metrics['f1'],
        'Test_Spec': test_metrics['spec'],
        'Threshold': best_threshold
    }
    
    df_new = pd.DataFrame([new_row])
    if os.path.exists(summary_file):
        df_new.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(summary_file, mode='w', header=True, index=False)
    
    print(f"✅ Đã lưu kết quả tóm tắt vào: {summary_file}")
    
    # Vẽ biểu đồ (đã bỏ chữ combined)
    plot_metrics(history_data, test_metrics, config['MODEL_OUT'], log_suffix)
    return model.state_dict(), history_data, test_metrics