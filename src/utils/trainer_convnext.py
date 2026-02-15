import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix
)

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
    tn, fp, fn, tp = (
        confusion_matrix(y_true, y_pred).ravel()
        if len(set(y_true)) > 1 else (0, 0, 0, 0)
    )
    return tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0


# =========================================================
# GradCAM (BACKBONE ONLY)
# =========================================================
def generate_gradcam(model, img_tensor, save_dir, idx):

    device = next(model.parameters()).device
    model.eval()

    if not hasattr(model, "backbone"):
        raise ValueError("Model has no backbone attribute")

    backbone = model.backbone

    # find last Conv2d
    target_layer = None
    for m in reversed(list(backbone.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break

    if target_layer is None:
        raise ValueError("No Conv2d layer found for GradCAM")

    cam = GradCAM(
        model=backbone,
        target_layers=[target_layer]
    )

    img_input = img_tensor.to(device).float()
    img_input.requires_grad_(True)

    grayscale_cam = cam(input_tensor=img_input)[0]

    img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_dir, f"sample_{idx}.png"),
        cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    )


# =========================================================
# Evaluation
# =========================================================
def evaluate(model: nn.Module, loader, device='cpu', is_late_fusion=None):
    model.eval()
    loss_sum = 0.0
    all_probs, all_preds, all_targets = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    if is_late_fusion is None:
        is_late_fusion = (
            hasattr(model, "metadata_mode")
            and model.metadata_mode == "late_fusion"
        )

    with torch.no_grad():
        for batch in loader:
            imgs, meta, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            if is_late_fusion:
                meta_vec, _ = meta
                logits = model(imgs, meta_vec.to(device).float())
            else:
                m_num, m_cat = meta
                logits = model(
                    imgs,
                    m_num.to(device).float(),
                    m_cat.to(device).long()
                )

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
# TRAIN LOOP (ORIGINAL WORKFLOW + AMP OFF)
# =========================================================
def train_loop(
    model, train_loader, val_loader, test_loader,
    config, criterion, optimizer, scheduler,
    device, log_suffix=""
):

    # 🔴 AMP OFF for ConvNeXt stability
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    is_late_fusion = (config["METADATA_MODE"] == "late_fusion")

    best_val_auc = 0.0
    history_data = []

    patience = config.get('PATIENCE', 5)
    counter = 0

    os.makedirs(config['MODEL_OUT'], exist_ok=True)

    history_csv = os.path.join(
        config['MODEL_OUT'],
        f"metrics_history_{config['METADATA_MODE']}_{log_suffix}.csv"
    )

    for epoch in range(1, config['EPOCHS'] + 1):

        model.train()
        train_loss_sum = 0.0

        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):

            imgs = imgs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            # 🔴 AMP OFF
            with torch.amp.autocast('cuda', enabled=False):

                if is_late_fusion:
                    meta_vec, _ = meta
                    logits = model(imgs, meta_vec.to(device).float())
                else:
                    m_num, m_cat = meta
                    logits = model(
                        imgs,
                        m_num.to(device).float(),
                        m_cat.to(device).long()
                    )

                smooth = config.get('LABEL_SMOOTHING', 0.0)
                labels_smooth = labels * (1 - smooth) + 0.5 * smooth
                loss = criterion(logits, labels_smooth)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * imgs.size(0)

        # ===== Evaluate =====
        train_res = evaluate(model, train_loader, device, is_late_fusion)
        val_res = evaluate(model, val_loader, device, is_late_fusion)

        epoch_row = {'epoch': epoch}
        epoch_row.update({f'train_{k}': v for k, v in train_res.items()})
        epoch_row.update({f'val_{k}': v for k, v in val_res.items()})
        history_data.append(epoch_row)

        pd.DataFrame(history_data).to_csv(history_csv, index=False)

        print(
            f"Epoch {epoch} | Val AUC: {val_res['auc']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}"
        )

        # ===== Scheduler (original logic) =====
        if scheduler:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_res['auc'])
            else:
                scheduler.step()

        # ===== Early Stopping + Save =====
        if val_res['auc'] > best_val_auc:
            best_val_auc = val_res['auc']
            counter = 0
            torch.save(
                {'state_dict': model.state_dict()},
                os.path.join(
                    config['MODEL_OUT'],
                    f"best_{config['METADATA_MODE']}.pt"
                )
            )
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"Early stopping triggered "
                    f"after {patience} epochs"
                )
                break

        # ===== GradCAM =====
        if (
            HAS_GRADCAM
            and epoch % config.get('GRADCAM_SAVE_EVERY', 5) == 0
        ):
            save_dir = os.path.join(
                config['MODEL_OUT'],
                f"gradcam_ep{epoch}"
            )

            try:
                val_batch = next(iter(val_loader))
                val_imgs = val_batch[0]

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
                print(f"GradCAM failed: {e}")

    # ===== LOAD BEST BEFORE TEST =====
    best_ckpt_path = os.path.join(
        config['MODEL_OUT'],
        f"best_{config['METADATA_MODE']}.pt"
    )

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(
            torch.load(best_ckpt_path, map_location=device)['state_dict']
        )

    print("\nTraining finished. Evaluating test set...")

    test_metrics = evaluate(model, test_loader, device, is_late_fusion)

    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(
            config['MODEL_OUT'],
            f"test_metrics_{log_suffix}.csv"
        ),
        index=False
    )

    return model.state_dict(), history_data, test_metrics
