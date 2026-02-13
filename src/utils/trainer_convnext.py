import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

# =========================================================
# GradCAM availability check
# =========================================================
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# =========================================================
# Utility
# =========================================================
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0


# =========================================================
# 🔥 GRADCAM (CHỈ CHẠY BACKBONE – KHÔNG ĐỔI WORKFLOW)
# =========================================================
def generate_gradcam(model, img_tensor, save_dir, idx):
    """
    GradCAM chỉ chạy trên image backbone.
    Không ảnh hưởng metadata branch.
    """

    device = next(model.parameters()).device
    model.eval()

    if not hasattr(model, "backbone"):
        raise ValueError("❌ Model không có thuộc tính 'backbone'.")

    backbone_model = model.backbone

    # Tìm Conv2d cuối cùng trong backbone
    target_layer = None
    for m in reversed(list(backbone_model.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break

    if target_layer is None:
        raise ValueError("❌ Không tìm thấy Conv layer cho GradCAM.")

    cam = GradCAM(
        model=backbone_model,
        target_layers=[target_layer]
    )

    img_input = img_tensor.to(device).float()
    img_input.requires_grad_(True)

    grayscale_cam = cam(input_tensor=img_input)[0]

    img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    save_path = os.path.join(save_dir, f"sample_{idx}.png")
    cv2.imwrite(save_path, cv2.COLOR_RGB2BGR if False else cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))


# =========================================================
# Evaluation
# =========================================================
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
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

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


# =========================================================
# TRAIN LOOP (GIỮ NGUYÊN WORKFLOW)
# =========================================================
def train_loop(model, train_loader, val_loader, test_loader,
               config, criterion, optimizer, scheduler,
               device, log_suffix=""):

    scaler = torch.amp.GradScaler('cuda', enabled=False)
    is_late_fusion = (config["METADATA_MODE"] == "late_fusion")

    best_val_auc = 0.0
    history_data = []

    os.makedirs(config['MODEL_OUT'], exist_ok=True)
    history_csv = os.path.join(
        config['MODEL_OUT'],
        f"metrics_history_{config['METADATA_MODE']}_{log_suffix}.csv"
    )

    for epoch in range(1, config['EPOCHS'] + 1):

        model.train()

        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):

            imgs = imgs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=False):

                if is_late_fusion:
                    meta_vec, _ = meta
                    logits = model(imgs, meta_vec.to(device).float())
                else:
                    m_num, m_cat = meta
                    logits = model(imgs,
                                   m_num.to(device).float(),
                                   m_cat.to(device).long())

                loss = criterion(
                    logits.view(-1, 1),
                    labels.view(-1, 1) * 0.9 + 0.05
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ===== Evaluate =====
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion)

        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)

        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        tqdm.write(
            f"Epoch {epoch} | Train AUC: {train_res['auc']:.4f} | "
            f"Val AUC: {val_res['auc']:.4f}"
        )

        if scheduler:
            scheduler.step()

        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            torch.save(
                {'state_dict': model.state_dict()},
                os.path.join(config['MODEL_OUT'],
                             f"best_{config['METADATA_MODE']}.pt")
            )

        # ===== GradCAM =====
        if (config.get('LOG_GRADCAM_EVERY_EPOCH', True)
                and HAS_GRADCAM
                and epoch % config.get('GRADCAM_SAVE_EVERY', 5) == 0):

            save_dir = os.path.join(
                config['MODEL_OUT'],
                f"{config['METADATA_MODE']}_gradcam_{epoch}"
            )
            os.makedirs(save_dir, exist_ok=True)

            try:
                val_imgs, _, _ = next(iter(val_loader))

                model.eval()
                for idx in range(min(4, len(val_imgs))):
                    generate_gradcam(
                        model,
                        val_imgs[idx:idx+1],
                        save_dir,
                        idx
                    )
                model.train()

            except Exception as e:
                print(f"⚠️ Grad-CAM failed: {e}")

    # ===== TEST =====
    print("\n🚀 Training finished. Evaluating test set...")

    test_metrics = evaluate(model, test_loader, device, is_late_fusion)

    test_csv_path = os.path.join(
        config['MODEL_OUT'],
        f"test_metrics_{config['METADATA_MODE']}_{log_suffix}.csv"
    )
    pd.DataFrame([test_metrics]).to_csv(test_csv_path, index=False)

    print("Test AUC:", test_metrics['auc'])

    return model.state_dict(), history_data, test_metrics
