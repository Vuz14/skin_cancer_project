import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Setup path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import get_model

# CẤU HÌNH LOAD MODEL (Phải khớp lúc train)
CONFIG = {
    'MODEL_NAME': 'resnet50',        # BẮT BUỘC: Để gọi đúng backbone
    'IMG_SIZE': 384,                 # BẮT BUỘC: Để resize ảnh đầu vào cho khớp
    'METADATA_MODE': 'full_weighted',# BẮT BUỘC: Để model khởi tạo đúng Fusion Head
    'METADATA_FEATURE_BOOST': 5.0,   # BẮT BUỘC: Khớp tham số boost lúc train
    'PRETRAINED': False,             # Nên để False cho nhanh (vì ta sẽ load weight của mình đè lên)
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}
CKPT_PATH = '/mnt/d/skin_cancer_project/checkpoint_ResNet50_ham10000/best_full_weighted.pt'
IMG_PATH = '/mnt/d/skin_cancer_project/dataset/Bcn10000-preprocessed/clean_ISIC_0033158.jpg'  # Thay ảnh test vào đây


def run_lime():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    # Fake thông số metadata để khởi tạo khung model
    dummy_cats = {'age': 15, 'sex': 3, 'loc': 8}  # Ví dụ
    dummy_num = 1
    model = get_model(CONFIG, dummy_cats, dummy_num).to(device)

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. Hàm dự đoán cho LIME (Input: Numpy array ảnh)
    def batch_predict(images):
        # LIME đưa vào list ảnh numpy (H,W,C) -> Cần chuyển về Tensor (B,C,H,W)
        batch_tensors = []
        for img in images:
            # Chuẩn hóa giống lúc train
            img_pil = Image.fromarray(img.astype('uint8'))
            t = transforms.Compose([
                transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            batch_tensors.append(t(img_pil))

        batch = torch.stack(batch_tensors).to(device)

        # Fake metadata (batch size tương ứng)
        B = batch.size(0)
        meta_num = torch.zeros((B, dummy_num)).to(device)
        meta_cat = torch.zeros((B, len(dummy_cats)), dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(batch, meta_num, meta_cat)
            probs = torch.sigmoid(logits)

        # LIME cần output shape (N_samples, 2) cho bài toán Binary
        # probs hiện tại là (N, 1) -> ta tạo (N, 2) với [1-p, p]
        probs = probs.cpu().numpy()
        return np.hstack([1 - probs, probs])

    # 3. Chạy LIME
    print("🍋 Đang khởi chạy LIME (Sẽ mất vài phút)...")
    explainer = lime_image.LimeImageExplainer()

    # Đọc ảnh gốc để giải thích
    original_img = np.array(Image.open(IMG_PATH).convert('RGB').resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])))

    explanation = explainer.explain_instance(
        original_img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000  # Tăng lên nếu muốn chính xác hơn
    )

    # 4. Hiển thị và Lưu
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    img_boundry = mark_boundaries(temp / 255.0, mask)  # Normalize về 0-1 để hiển thị

    plt.figure(figsize=(8, 8))
    plt.imshow(img_boundry)
    plt.title(f"LIME Explanation: Why Malignant?")
    plt.axis('off')
    save_file = 'lime_result.png'
    plt.savefig(save_file, bbox_inches='tight')
    print(f"✅ Đã lưu kết quả LIME vào {save_file}")


if __name__ == '__main__':
    run_lime()