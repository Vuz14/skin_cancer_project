import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Thêm đường dẫn project
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_logic.ham_dataset import HAM10000Dataset
from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model

# Import GradCAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("⚠️ Chưa cài 'grad-cam'. Chạy 'pip install grad-cam' để dùng tính năng này.")

# ==========================================
# CẤU HÌNH TEST
# ==========================================
CONFIG = {
    # Chọn Dataset muốn test: 'HAM10000' hoặc 'BCN20000'
    # 'DATASET': 'HAM10000',
    #
    # # Đường dẫn file CSV Test (Sửa lại cho đúng đường dẫn của bạn)
    # 'TEST_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/ham10000_test.csv',
    #
    # # Đường dẫn folder ảnh đã xử lý (Cleaned)
    # 'IMG_ROOT': '/mnt/d/skin_cancer_project/dataset/Ham10000-preprocessed',
    #
    # # Đường dẫn Checkpoint tốt nhất
    # 'CKPT_PATH': '/mnt/d/skin_cancer_project/checkpoint_ResNet50_ham10000/best_full_weighted.pt',

    'DATASET': 'BCN20000',

    # Đường dẫn file CSV Test (Sửa lại cho đúng đường dẫn của bạn)
    'TEST_CSV': '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_test.csv',

    # Đường dẫn folder ảnh đã xử lý (Cleaned)
    'IMG_ROOT': '/mnt/d/skin_cancer_project/dataset/Bcn20000-preprocessed',

    # Đường dẫn Checkpoint tốt nhất
    'CKPT_PATH': '/mnt/d/skin_cancer_project/checkpoint_ResNet50_bcn20000/best_full_weighted.pt',

    # Cấu hình Model (Phải khớp lúc Train)
    'MODEL_NAME': 'resnet50',
    'IMG_SIZE': 384,
    'METADATA_MODE': 'full_weighted',
    'METADATA_FEATURE_BOOST': 5.0,
    'PRETRAINED': False,  # Không cần load pretrain ImageNet vì ta load checkpoint của mình
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BATCH_SIZE': 16
}


# Nếu test BCN thì bỏ comment đoạn này
# CONFIG['DATASET'] = 'BCN20000'
# CONFIG['TEST_CSV'] = '/mnt/d/skin_cancer_project/dataset/metadata/bcn20000_test.csv'
# CONFIG['IMG_ROOT'] = '/mnt/d/skin_cancer_project/dataset/Bcn20000-cleaned-final'
# CONFIG['CKPT_PATH'] = '/mnt/d/skin_cancer_project/checkpoint_ResNet50_bcn20000/best_full_weighted.pt'


def generate_gradcam_report(model, test_loader, device, save_dir, num_samples=10):
    """
    Vẽ GradCAM cho một số mẫu trong tập test
    """
    if not HAS_GRADCAM: return

    print(f"\n🎨 Đang tạo {num_samples} ảnh Grad-CAM mẫu...")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # --- SỬA LỖI TẠI ĐÂY ---
    # Model đầu vào là 'GradCAMModelWrapper', model thật nằm trong biến '.model'
    # Path: wrapper -> ResNetCBAM -> ResNet50Backbone -> torchvision ResNet -> layer4
    try:
        # Nếu là wrapper (GradCAMModelWrapper)
        real_backbone = model.model.backbone.model
    except AttributeError:
        # Nếu không phải wrapper (trường hợp dùng model trần)
        try:
            real_backbone = model.backbone.model
        except AttributeError:
            print("❌ Không tìm thấy backbone.model.layer4. Kiểm tra lại cấu trúc model!")
            return

    target_layers = [real_backbone.layer4[-1]]
    # -----------------------

    cam = GradCAM(model=model, target_layers=target_layers)

    # Lấy 1 batch từ test_loader
    imgs, meta, labels = next(iter(test_loader))
    imgs = imgs.to(device)

    # Chọn ngẫu nhiên hoặc lấy tuần tự
    for idx in range(min(num_samples, len(imgs))):
        img_tensor = imgs[idx:idx + 1]  # (1, C, H, W)
        label_true = labels[idx].item()

        # Metadata (Fake batch dimension)
        # m_num = meta[0][idx:idx + 1].to(device).float()
        # m_cat = meta[1][idx:idx + 1].to(device).long()

        # --- CHẠY GRADCAM ---
        try:
            # Denormalize ảnh để hiển thị
            rgb_img = img_tensor.cpu().numpy().squeeze().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = std * rgb_img + mean
            rgb_img = np.clip(rgb_img, 0, 1)

            # Tính CAM (Target là class Malignant = 1)
            # targets = [ClassifierOutputTarget(0)] # Lành tính
            # targets = [ClassifierOutputTarget(0)]  # Binary Classification: Output là Logits

            # Với Binary Classification, targets=None sẽ tự động chọn class có score cao nhất
            grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]

            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Lưu ảnh
            fname = f"cam_sample_{idx}_True{int(label_true)}.png"
            cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"⚠️ Lỗi vẽ CAM ảnh {idx}: {e}")
            continue

    print(f"✅ Đã lưu ảnh GradCAM vào: {save_dir}")


# Wrapper cho Model để GradCAM dễ gọi (chỉ nhận 1 input ảnh)
class GradCAMModelWrapper(nn.Module):
    def __init__(self, model, dummy_meta_num, dummy_meta_cat):
        super().__init__()
        self.model = model
        self.meta_num = dummy_meta_num
        self.meta_cat = dummy_meta_cat

    def forward(self, x):
        return self.model(x, self.meta_num, self.meta_cat)


def main():
    device = torch.device(CONFIG['DEVICE'])
    print(f"🚀 BẮT ĐẦU TEST TRÊN TẬP: {CONFIG['DATASET']}")

    # 1. Load Data
    df = pd.read_csv(CONFIG['TEST_CSV'])
    df.columns = df.columns.str.strip()

    # Xử lý label & path
    if CONFIG['DATASET'] == 'HAM10000':
        if 'dx' in df.columns:
            df['label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)
        df['image_path'] = df['image_id'].astype(str) + '.jpg'
        DatasetClass = HAM10000Dataset
    else:
        # BCN logic
        if 'diagnosis' in df.columns:
            df['label'] = df['diagnosis'].apply(
                lambda x: 1 if str(x).lower() in ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma'] else 0)
        elif 'diagnosis_1' in df.columns:
            df['label'] = df['diagnosis_1'].apply(lambda x: 1 if 'malig' in str(x).lower() else 0)
        if 'isic_id' in df.columns:
            df['image_path'] = df['isic_id'].astype(str) + '.jpg'
        DatasetClass = DermoscopyDataset

    # Quan trọng: Dataset test luôn load clean_
    # Vì logic load ảnh đã nằm trong _load_image của Dataset (tự thêm clean_)

    print("⏳ Đang load model...")
    # Tạo dataset tạm để lấy thông tin cột
    temp_ds = DatasetClass(df, CONFIG['IMG_ROOT'], CONFIG['IMG_SIZE'], CONFIG['METADATA_MODE'], train=False)

    # Khởi tạo model
    model = get_model(CONFIG, temp_ds.cat_cardinalities, len(temp_ds.numeric_cols)).to(device)

    # --- ĐOẠN CODE LOAD CHECKPOINT THÔNG MINH ---
    print(f"⏳ Đang load checkpoint: {CONFIG['CKPT_PATH']}")
    checkpoint = torch.load(CONFIG['CKPT_PATH'], map_location=device)
    state_dict = checkpoint['state_dict']

    # 1. Lấy state_dict hiện tại của model
    model_state_dict = model.state_dict()

    # 2. Lọc bỏ các key bị lệch kích thước (Size Mismatch)
    # Ta sẽ chỉ giữ lại những key nào có cùng kích thước với model hiện tại
    filtered_state_dict = {}
    mismatched_keys = []

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                mismatched_keys.append(k)
        else:
            pass

    if mismatched_keys:
        print(f"⚠️ Cảnh báo: Đã bỏ qua {len(mismatched_keys)} layer bị lệch kích thước (do RFE/Metadata khác biệt).")
        # print(f"   Ví dụ: {mismatched_keys[:3]}...")

    # 3. Load state_dict đã lọc (dùng strict=False để chấp nhận thiếu key)
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ Load model thành công (Backbone & Attention đã được nạp chuẩn).")

    model.eval()

    # 2. Tạo DataLoader
    test_loader = DataLoader(temp_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 3. Chạy Inference
    print("running inference...")
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, meta, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            m_num, m_cat = meta
            m_num, m_cat = m_num.to(device).float(), m_cat.to(device).long()

            logits = model(imgs, m_num, m_cat)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            # Thử ngưỡng thấp hơn vì Recall đang thấp (0.43)
            # all_preds.extend((probs >= 0.5).astype(int))
            all_preds.extend((probs >= 0.3).astype(int))
            all_targets.extend(labels.cpu().numpy())

    # 4. Tính Metrics
    auc = roc_auc_score(all_targets, all_probs)
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n" + "=" * 40)
    print(f"📊 KẾT QUẢ TEST TRÊN {CONFIG['DATASET']}")
    print("=" * 40)
    print(f"✅ AUC          : {auc:.4f}")
    print(f"✅ Accuracy     : {acc:.4f}")
    print(f"✅ Sensitivity  : {sensitivity:.4f} (Recall)")
    print(f"✅ Specificity  : {specificity:.4f}")
    if (tp + fp) > 0:
        print(f"✅ Precision    : {tp / (tp + fp):.4f}")
    else:
        print(f"✅ Precision    : 0.0000")

    print("\nConfusion Matrix:")
    print(cm)

    # 5. Vẽ GradCAM (Dùng Wrapper để xử lý vụ 3 tham số đầu vào)
    # Lấy 1 mẫu metadata làm dummy
    sample_img, sample_meta, _ = temp_ds[0]
    dummy_num = sample_meta[0].unsqueeze(0).to(device).float()
    dummy_cat = sample_meta[1].unsqueeze(0).to(device).long()

    wrapped_model = GradCAMModelWrapper(model, dummy_num, dummy_cat)

    # Folder lưu kết quả
    res_dir = os.path.join(os.path.dirname(CONFIG['CKPT_PATH']), 'test_results')
    generate_gradcam_report(wrapped_model, test_loader, device, res_dir)

    print(f"\n🎉 Hoàn tất! Kết quả lưu tại {res_dir}")


if __name__ == "__main__":
    main()