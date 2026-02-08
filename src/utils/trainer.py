import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import gc  # Thư viện để dọn rác bộ nhớ

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
    Tạo và lưu ảnh Grad-CAM.
    Phiên bản FIX: Hỗ trợ cả ResNet (layer4) và EfficientNet (conv_head).
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Lấy timm model gốc từ bên trong wrapper
    # model.backbone là class wrapper của mình
    # model.backbone.backbone là model timm thực tế
    try:
        backbone_timm = model.backbone.backbone
    except AttributeError:
        backbone_timm = model.backbone

    # 2. Tự động chọn Target Layer dựa trên tên model
    target_layer = None
    if hasattr(backbone_timm, 'conv_head'):  # EfficientNet
        target_layer = backbone_timm.conv_head
    elif hasattr(backbone_timm, 'layer4'):  # ResNet
        target_layer = backbone_timm.layer4[-1]

    # Nếu không tìm thấy layer phù hợp thì bỏ qua (không vẽ)
    if target_layer is None:
        return

    try:
        # Khởi tạo GradCAM
        cam = GradCAM(model=model.backbone, target_layers=[target_layer])

        # Bật requires_grad cho input để tính toán gradient
        img_input = img_tensor.to(device).float()
        img_input.requires_grad = True

        # Tính toán CAM
        grayscale_cam = cam(input_tensor=img_input, targets=None)[0]

        # 3. Denormalize ảnh gốc để hiển thị đẹp hơn
        img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Vẽ heatmap lên ảnh gốc
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # Lưu ảnh
        save_path = os.path.join(save_dir, f"sample_{idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"⚠️ Grad-CAM error: {e}")


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
        plt.legend();
        plt.grid(True)
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
    # Handle trường hợp chỉ có 1 lớp trong batch (tránh lỗi AUC)
    try:
        auc_score = roc_auc_score(y_true, all_probs)
    except ValueError:
        auc_score = 0.0

    return {
        'loss': loss_sum / len(loader.dataset),
        'auc': auc_score,
        'acc': accuracy_score(y_true, all_preds),
        'f1': f1_score(y_true, all_preds, zero_division=0),
        'precision': precision_score(y_true, all_preds, zero_division=0),
        'recall': recall_score(y_true, all_preds, zero_division=0),
        'spec': calculate_specificity(y_true, all_preds)
    }


def train_loop(model, train_loader, val_loader, test_loader, config, criterion, optimizer, scheduler, device,
               log_suffix=""):
    # Sử dụng GradScaler cho Mixed Precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    is_late_fusion = (config["METADATA_MODE"] == "late_fusion")
    best_val_auc = 0.0
    history_data = []
    history_csv = os.path.join(config['MODEL_OUT'], f"metrics_history_{config['METADATA_MODE']}_{log_suffix}.csv")

    for epoch in range(1, config['EPOCHS'] + 1):
        model.train()
        running_loss = 0.0

        # --- VÒNG LẶP TRAIN (BATCH) ---
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

                # Label smoothing nhẹ
                loss = criterion(logits.view(-1, 1), labels.view(-1, 1) * 0.9 + 0.05)

            scaler.scale(loss).backward()
            scaler.step(optimizer);
            scaler.update()

        # --- DỌN DẸP BỘ NHỚ (Ngang hàng với vòng for, chạy sau khi hết epoch) ---
        print("🧹 Đang dọn dẹp bộ nhớ...")
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ Đã dọn dẹp xong Epoch {epoch}")

        # --- ĐÁNH GIÁ (VALIDATION) ---
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion)

        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)

        # Lưu log
        pd.DataFrame(history_data).to_csv(history_csv, index=False)
        print(f"Epoch {epoch} | Train AUC: {train_res['auc']:.4f} | Val AUC: {val_res['auc']:.4f}")

        if scheduler: scheduler.step()

        # Lưu Checkpoint tốt nhất
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt"))

        # --- VẼ GRAD-CAM (Mỗi 5 epoch) ---
        if config.get('LOG_GRADCAM_EVERY_EPOCH', True) and HAS_GRADCAM and epoch % config.get('GRADCAM_SAVE_EVERY',
                                                                                              5) == 0:
            cam_folder_name = f"{config['METADATA_MODE']}_gradcam_{epoch}"
            save_dir = os.path.join(config['MODEL_OUT'], cam_folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # Lấy ảnh mẫu từ validation
            val_imgs = None
            with torch.no_grad():
                val_iter = iter(val_loader)
                try:
                    batch = next(val_iter)
                    val_imgs = batch[0]
                except StopIteration:
                    pass

            # Vẽ Grad-CAM
            if val_imgs is not None:
                model.eval()  # Chuyển sang eval mode
                for idx in range(min(4, len(val_imgs))):
                    # Gọi hàm generate_gradcam đã sửa lỗi
                    generate_gradcam(model, val_imgs[idx:idx + 1], save_dir, idx)
                model.train()  # Chuyển lại train mode

    # --- KẾT THÚC TRAIN: TEST & VẼ BIỂU ĐỒ ---
    print("\n🚀 Huấn luyện hoàn tất. Đang đánh giá tập Test...")
    test_metrics = evaluate(model, test_loader, device, is_late_fusion)

    test_csv_path = os.path.join(config['MODEL_OUT'], f"test_metrics_{config['METADATA_MODE']}_{log_suffix}.csv")
    pd.DataFrame([test_metrics]).to_csv(test_csv_path, index=False)

    plot_metrics_combined(history_data, test_metrics, config['MODEL_OUT'], config['METADATA_MODE'])

    print(f"✅ Đã lưu biểu đồ và kết quả Test vào {config['MODEL_OUT']}")
    return model.state_dict(), history_data, test_metrics