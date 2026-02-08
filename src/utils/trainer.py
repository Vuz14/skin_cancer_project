import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import gc

# Kiểm tra thư viện Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# --- CLASS WRAPPER CHO GRAD-CAM ---
class GradCAMModelWrapper(nn.Module):
    def __init__(self, model, meta_num, meta_cat):
        super().__init__()
        self.model = model
        self.meta_num = meta_num
        self.meta_cat = meta_cat

    def forward(self, x):
        return self.model(x, self.meta_num, self.meta_cat)


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0


def generate_gradcam(model, img_tensor, meta_num, meta_cat, save_dir, idx, device):
    """
    Tạo và lưu ảnh Grad-CAM sử dụng Wrapper để xử lý metadata.
    """
    if not HAS_GRADCAM: return

    # 1. Bọc model bằng Wrapper (để fix lỗi input 3 tham số)
    wrapped_model = GradCAMModelWrapper(model, meta_num, meta_cat)
    wrapped_model.eval()

    # 2. Xác định Target Layer chính xác
    # Path: ResNetCBAM -> ResNet50Backbone -> torchvision.models.resnet50 -> layer4
    try:
        # Trường hợp ResNet50
        target_layers = [model.backbone.model.layer4[-1]]
    except AttributeError:
        try:
            # Trường hợp EfficientNet (nếu có dùng)
            target_layers = [model.backbone.conv_head]
        except:
            return  # Không tìm thấy layer phù hợp

    try:
        # Khởi tạo GradCAM với Wrapper
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        # Tính toán CAM
        # targets=None tự động chọn class có score cao nhất
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]

        # 3. Denormalize ảnh gốc để hiển thị
        img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Vẽ heatmap lên ảnh gốc
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # Lưu ảnh
        save_path = os.path.join(save_dir, f"epoch_sample_{idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"⚠️ Grad-CAM error tại mẫu {idx}: {e}")


def plot_metrics_combined(history_data, test_metrics, out_dir, mode_name):
    """
    Vẽ biểu đồ so sánh Train/Val/Test
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(history_data)
    epochs = df['epoch']
    metrics = ['loss', 'auc', 'acc', 'f1']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        # Vẽ Train & Val
        plt.plot(epochs, df[f'train_{metric}'], label=f'Train {metric.upper()}', marker='o')
        plt.plot(epochs, df[f'val_{metric}'], label=f'Val {metric.upper()}', marker='s')

        # Vẽ đường ngang cho Test
        test_val = test_metrics.get(metric)
        if test_val is not None:
            plt.axhline(y=test_val, color='r', linestyle='--', label=f'Test {metric.upper()} ({test_val:.4f})')

        plt.title(f'{metric.upper()} History - {mode_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"combined_{metric}_{mode_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def evaluate(model, loader, device='cpu', is_late_fusion=None):
    model.eval()
    loss_sum = 0.0
    all_probs, all_preds, all_targets = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    with torch.no_grad():
        for batch in loader:
            imgs, meta, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            if labels.ndim == 1: labels = labels.unsqueeze(1)

            m_num, m_cat = meta
            # Forward pass
            logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())

            loss = bce(logits, labels)
            loss_sum += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.extend(probs.tolist())
            # Ngưỡng 0.5 (có thể chỉnh lại nếu cần)
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_targets.extend(labels.cpu().numpy().reshape(-1).tolist())

    y_true = np.array(all_targets)

    # Handle trường hợp chỉ có 1 class trong batch (tránh lỗi AUC)
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_val_auc = 0.0
    history_data = []
    history_csv = os.path.join(config['MODEL_OUT'], f"metrics_history_{config['METADATA_MODE']}_{log_suffix}.csv")

    for epoch in range(1, config['EPOCHS'] + 1):
        model.train()

        # --- VÒNG LẶP TRAIN ---
        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device).float()
            optimizer.zero_grad(set_to_none=True)

            m_num, m_cat = meta

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs, m_num.to(device).float(), m_cat.to(device).long())
                # Label smoothing nhẹ để tránh overfit
                loss = criterion(logits.view(-1, 1), labels.view(-1, 1) * 0.9 + 0.05)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- DỌN DẸP BỘ NHỚ ---
        print("🧹 Đang dọn dẹp bộ nhớ...")
        torch.cuda.empty_cache()
        gc.collect()

        # --- ĐÁNH GIÁ ---
        train_res = evaluate(model, train_loader, device)
        val_res = evaluate(model, val_loader, device)

        # Lưu metrics
        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)
        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        print(f"Epoch {epoch} | Train AUC: {train_res['auc']:.4f} | Val AUC: {val_res['auc']:.4f}")

        if scheduler: scheduler.step()

        # Lưu Checkpoint
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(config['MODEL_OUT'], f"best_{config['METADATA_MODE']}.pt"))

        # --- VẼ GRAD-CAM ---
        # Chỉ vẽ mỗi 5 epoch để tiết kiệm thời gian
        if config.get('LOG_GRADCAM_EVERY_EPOCH', True) and HAS_GRADCAM and epoch % config.get('GRADCAM_SAVE_EVERY',
                                                                                              5) == 0:
            cam_folder_name = f"{config['METADATA_MODE']}_gradcam_epoch{epoch}"
            save_dir = os.path.join(config['MODEL_OUT'], cam_folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # Lấy 1 batch từ Validation để vẽ mẫu
            try:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)  # (imgs, meta, labels)

                v_imgs, v_meta, v_labels = val_batch
                v_m_num, v_m_cat = v_meta

                # Vẽ cho 4 ảnh đầu tiên trong batch
                for idx in range(min(4, len(v_imgs))):
                    # Cần truyền từng mẫu đơn lẻ vào hàm vẽ
                    img_tensor = v_imgs[idx:idx + 1].to(device)
                    # Fake batch dimension cho metadata (1, D)
                    meta_num_sample = v_m_num[idx:idx + 1].to(device).float()
                    meta_cat_sample = v_m_cat[idx:idx + 1].to(device).long()

                    generate_gradcam(model, img_tensor, meta_num_sample, meta_cat_sample, save_dir, idx, device)

            except Exception as e:
                print(f"⚠️ Không thể vẽ GradCAM epoch {epoch}: {e}")

    # --- KẾT THÚC: TEST & VẼ BIỂU ĐỒ ---
    print("\n🚀 Huấn luyện hoàn tất. Đang đánh giá tập Test...")
    test_metrics = evaluate(model, test_loader, device)

    test_csv_path = os.path.join(config['MODEL_OUT'], f"test_metrics_{config['METADATA_MODE']}_{log_suffix}.csv")
    pd.DataFrame([test_metrics]).to_csv(test_csv_path, index=False)

    plot_metrics_combined(history_data, test_metrics, config['MODEL_OUT'], config['METADATA_MODE'])
    print(f"✅ Đã lưu kết quả vào {config['MODEL_OUT']}")

    return model.state_dict(), history_data, test_metrics