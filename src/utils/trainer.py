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
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0

# Cập nhật trong src/utils/trainer.py

def generate_gradcam(model, img_tensor, save_dir, idx):
    """
    Generate and save Grad-CAM visualization.
    FIXED VERSION:
    - Không bắt layer quá sâu
    - Ưu tiên spatial feature (blocks[-2].conv_dw)
    """
    model.eval()
    device = next(model.parameters()).device

    # -------- 1. Wrapper: image-only forward (fake metadata) --------
    class _ImageOnlyWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            bs = x.size(0)
            dev = x.device

            num_numeric = getattr(self.base_model, 'num_numeric', 0)
            cat_dim = len(getattr(self.base_model, 'cat_names', []))
            if cat_dim == 0 and hasattr(self.base_model, 'cat_cardinalities'):
                cat_dim = len(self.base_model.cat_cardinalities)

            meta_num = torch.zeros((bs, num_numeric), device=dev)
            meta_cat = torch.zeros((bs, cat_dim), dtype=torch.long, device=dev)

            return self.base_model(x, meta_num, meta_cat)

    # -------- 2. CHỌN TARGET LAYER (KHÔNG QUÁ SÂU) --------
    layer_name = "unknown"
    try:
        # 🥇 TỐT NHẤT: block áp chót – conv_dw giữ spatial
        target_layer = model.backbone.model.blocks[-2][-1].conv_dw
        layer_name = "blocks[-2][-1].conv_dw"
    except Exception:
        try:
            # 🥈 Fallback: output block áp chót
            target_layer = model.backbone.model.blocks[-2]
            layer_name = "blocks[-2]"
        except Exception:
            try:
                # 🥉 Fallback cuối: conv_head
                target_layer = model.backbone.model.conv_head
                layer_name = "conv_head"
            except Exception:
                print("⚠️ Không tìm thấy layer phù hợp để Grad-CAM")
                return

    # -------- 3. Init Grad-CAM --------
    cam_model = _ImageOnlyWrapper(model)
    cam = GradCAM(
        model=cam_model,
        target_layers=[target_layer],
        use_cuda=(device.type == "cuda")
    )

    if img_tensor.ndim == 3:
        img_input = img_tensor.unsqueeze(0)
    else:
        img_input = img_tensor

    img_input = img_input.to(device).float()
    img_input.requires_grad_(True)

    # -------- 4. Run Grad-CAM --------
    try:
        grayscale_cam = cam(input_tensor=img_input, targets=None)[0]
    except Exception as e:
        print(f"⚠️ Grad-CAM error: {e}")
        return

    # -------- 5. De-normalize image --------
    img_np = img_input[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_rgb = np.clip((img_np * std) + mean, 0.0, 1.0)

    # -------- 6. Overlay heatmap --------
    heatmap = np.uint8(255 * np.clip(grayscale_cam, 0.0, 1.0))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    cam_image = 0.6 * img_rgb + 0.4 * heatmap
    cam_image = np.uint8(255 * np.clip(cam_image, 0.0, 1.0))

    # -------- 7. Annotation --------
    cv2.putText(
        cam_image,
        f"Grad-CAM layer: {layer_name}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # -------- 8. Save --------
    os.makedirs(save_dir, exist_ok=True)
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
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
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
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if is_late_fusion:
                    meta_vec, _ = meta
                    logits = model(imgs, meta_vec.to(device).float())
                else:
                    m_num, m_cat = meta
                    logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())
                # Label smoothing nhẹ để ổn định train
                loss = criterion(logits.view(-1, 1), labels.view(-1, 1) * 0.9 + 0.05)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Cần unscale trước khi clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) 
            scaler.step(optimizer)
            scaler.update()

        # Đánh giá Metrics cuối mỗi epoch
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion)
        
        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)
        
        # Lưu lịch sử ra CSV
        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        print(f"Epoch {epoch} | Train AUC: {train_res['auc']:.4f} | Val AUC: {val_res['auc']:.4f}")

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


    # --- Kết thúc Training: load checkpoint tốt nhất rồi đánh giá tập Test ---
    best_ckpt_path = os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt['state_dict'])

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