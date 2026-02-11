import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

# Kiểm tra thư viện Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0

def generate_gradcam(model, img_tensor, save_dir, idx):
    """
    Tạo và lưu ảnh Grad-CAM cho phần Backbone hình ảnh.
    Sửa lỗi: Phải bật requires_grad và không chạy trong no_grad.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Truy cập vào layer cuối của backbone EfficientNet
    try:
        target_layer = model.backbone.backbone.conv_head 
    except AttributeError:
        target_layer = model.backbone.conv_head
    
    # Khởi tạo GradCAM
    cam = GradCAM(model=model.backbone, target_layers=[target_layer])
    
    # Quan trọng: Bật requires_grad cho input để tính toán heatmap
    img_input = img_tensor.to(device).float()
    img_input.requires_grad = True 
    
    # Tính toán CAM (Hàm này thực hiện backward nội bộ nên không được bọc trong no_grad)
    grayscale_cam = cam(input_tensor=img_input, targets=None)[0]
    
    # Chuẩn hóa ảnh gốc để vẽ đè CAM
    img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    save_path = os.path.join(save_dir, f"sample_{idx}.png")
    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

def plot_metrics_combined(history_data, test_metrics, out_dir, mode_name):
    """
    Vẽ biểu đồ gộp cả 3 tập Train, Val, Test vào 1 đồ thị.
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
        
        plt.title(f'{metric.upper()} Comparison - {mode_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"combined_{metric}_{mode_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def evaluate(model: nn.Module, loader, device='cpu', is_late_fusion=None):
    model.eval()
    loss_sum = 0.0
    all_probs, all_preds, all_targets = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    if is_late_fusion is None:
        is_late_fusion = hasattr(model, "metadata_mode") and model.metadata_mode == "late_fusion"

    with torch.no_grad():
        for batch in loader:
            imgs, meta, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            if labels.ndim == 1: labels = labels.unsqueeze(1)

            if is_late_fusion:
                meta_vec, _ = meta
                logits = model(imgs, meta_vec.to(device).float())
            else:
                m_num, m_cat = meta
                logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())

            loss = bce(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            if np.isnan(probs).any() or np.isinf(probs).any():
                print("\n⚠️ [NaN PROBS DETECTED IN EVAL]")
                print("Logits min/max:", logits.min().item(), logits.max().item())
                print("Valid probs:", np.isfinite(probs).sum(), "/", len(probs))
            all_probs.extend(probs.tolist())
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_targets.extend(labels.cpu().numpy().reshape(-1).tolist())

    y_true = np.array(all_targets)
    return {
        'loss': loss_sum / len(loader.dataset),
        'auc': roc_auc_score(y_true, all_probs) if len(set(y_true)) > 1 else 0.0,
        'acc': accuracy_score(y_true, all_preds),
        'f1': f1_score(y_true, all_preds, zero_division=0),
        'precision': precision_score(y_true, all_preds, zero_division=0),
        'recall': recall_score(y_true, all_preds, zero_division=0),
        'spec': calculate_specificity(y_true, all_preds)
    }

def train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device, log_suffix=""):
    # Cập nhật API GradScaler mới nhất của PyTorch
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    is_late_fusion = (config["METADATA_MODE"] == "late_fusion")
    best_val_auc = 0.0
    history_data = []
    history_csv = os.path.join(config['MODEL_OUT'], f"metrics_history_{config['METADATA_MODE']}_{log_suffix}.csv")

    for epoch in range(1, config['EPOCHS'] + 1):
        model.train()
        running_loss = 0.0
        
        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=False):
                if is_late_fusion:
                    meta_vec, _ = meta
                    logits = model(imgs, meta_vec.to(device).float())
                else:
                    m_num, m_cat = meta
                    logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())
                    # ===== DEBUG LOG =====
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("\n❌ [NaN LOGITS DETECTED]")
                        print("Epoch:", epoch)
                        print("Logits min/max:", logits.min().item(), logits.max().item())
                        print("Images min/max:", imgs.min().item(), imgs.max().item())
                        print("Labels unique:", torch.unique(labels))
                        
                        if not is_late_fusion:
                            print("Meta_num stats:",
                                torch.isnan(m_num).sum().item(),
                                torch.isinf(m_num).sum().item(),
                                m_num.min().item(),
                                m_num.max().item())
                            print("Meta_cat unique:", torch.unique(m_cat))
                        
                        continue  # bỏ batch lỗi
                    # =====================

                # Label smoothing nhẹ để ổn định train
                loss = criterion(logits.view(-1, 1), labels.view(-1, 1) * 0.9 + 0.05)

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        # Đánh giá Metrics cuối mỗi epoch
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion)
        
        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)
        
        # Lưu lịch sử ra CSV
        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        tqdm.write(
            f"Epoch {epoch} | Train AUC: {train_res['auc']:.4f} | Val AUC: {val_res['auc']:.4f}"
        )
        if scheduler: scheduler.step()
        
        # Lưu Checkpoint tốt nhất
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            torch.save({'state_dict': model.state_dict()}, 
                       os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt"))

        # --- LOGIC GRAD-CAM (Lưu mỗi 5 epoch) ---
        if config.get('LOG_GRADCAM_EVERY_EPOCH', True) and HAS_GRADCAM and epoch % config.get('GRADCAM_SAVE_EVERY', 5) == 0:
            cam_folder_name = f"{config['METADATA_MODE']}_gradcam_{epoch}"
            save_dir = os.path.join(config['MODEL_OUT'], cam_folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Lấy ảnh mẫu từ validation trong khối no_grad
            val_imgs = None
            with torch.no_grad():
                val_iter = iter(val_loader)
                try:
                    batch = next(val_iter)
                    val_imgs = batch[0]
                except StopIteration: pass
            
            # Thoát no_grad để chạy Grad-CAM
            if val_imgs is not None:
                model.eval()
                for idx in range(min(4, len(val_imgs))):
                    try:
                        generate_gradcam(model, val_imgs[idx:idx+1], save_dir, idx)
                    except Exception as e:
                        print(f"⚠️ Grad-CAM failed: {e}")
                model.train()

    # --- Kết thúc Training: Đánh giá tập Test & Vẽ biểu đồ ---
    print("\n🚀 Huấn luyện hoàn tất. Đang đánh giá tập Test...")
    test_metrics = evaluate(model, test_loader, device, is_late_fusion)
    
    # Lưu kết quả Test
    test_csv_path = os.path.join(config['MODEL_OUT'], f"test_metrics_{config['METADATA_MODE']}_{log_suffix}.csv")
    pd.DataFrame([test_metrics]).to_csv(test_csv_path, index=False)
    
    # Vẽ biểu đồ gộp 3 tập
    plot_metrics_combined(history_data, test_metrics, config['MODEL_OUT'], config['METADATA_MODE'])
    
    print(f"✅ Đã lưu biểu đồ và kết quả Test vào {config['MODEL_OUT']}")
    return model.state_dict(), history_data, test_metrics