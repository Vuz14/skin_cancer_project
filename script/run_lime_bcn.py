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

# CẤU HÌNH LOAD MODEL BCN
CONFIG = {
    'MODEL_NAME': 'resnet50',
    'IMG_SIZE': 384,
    'METADATA_MODE': 'full_weighted',
    'METADATA_FEATURE_BOOST': 5.0,  # Lưu ý: Train BCN bạn dùng 5.0
    # 'META_CLASS_WEIGHT_BOOST': 2.0, # Cái này dùng tính Loss lúc train, inference ko cần
    'PRETRAINED': False,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ĐƯỜNG DẪN CỦA BCN
CKPT_PATH = '/mnt/d/skin_cancer_project/checkpoint_ResNet50_bcn20000/best_full_weighted.pt'

# Chọn một ảnh Test của BCN để giải thích
# Bạn hãy thay tên file này bằng một file thực tế trong folder BCN cleaned của bạn
IMG_PATH = '/mnt/d/skin_cancer_project/dataset/Bcn20000-preprocessed/clean_ISIC_0000000.jpg'


def run_lime_bcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    # Fake metadata cho BCN (BCN có anatomical_site, age, sex...)
    dummy_cats = {'age': 60, 'anatom_site_general': 2, 'sex': 1}  # Ví dụ dummy
    dummy_num = 0  # BCN thường ít numeric features hơn HAM, tùy dataset của bạn

    # Lưu ý: Cần truyền đúng số lượng features khớp với lúc train
    # Nếu RFE lọc bớt rồi thì ở đây phải khớp.
    # Mẹo: Để đơn giản lúc chạy LIME, bạn có thể load lại config đã lưu lúc train (nếu có)
    # Hoặc cứ để dummy dư ra, model.load_state_dict sẽ báo lỗi nếu lệch dimension.

    model = get_model(CONFIG, dummy_cats, dummy_num).to(device)

    print(f"⏳ Đang load checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'],
                          strict=False)  # strict=False để bỏ qua lỗi lệch head metadata nếu có
    model.eval()

    # 2. Hàm dự đoán (Giống bên HAM)
    def batch_predict(images):
        batch_tensors = []
        for img in images:
            img_pil = Image.fromarray(img.astype('uint8'))
            t = transforms.Compose([
                transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            batch_tensors.append(t(img_pil))

        batch = torch.stack(batch_tensors).to(device)
        B = batch.size(0)

        # Fake metadata
        meta_num = torch.zeros((B, dummy_num)).to(device)
        meta_cat = torch.zeros((B, len(dummy_cats)), dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(batch, meta_num, meta_cat)
            probs = torch.sigmoid(logits)

        probs = probs.cpu().numpy()
        return np.hstack([1 - probs, probs])

    # 3. Chạy LIME
    print("🍋 Đang chạy LIME cho BCN20000...")
    explainer = lime_image.LimeImageExplainer()

    if not os.path.exists(IMG_PATH):
        print(f"❌ Không tìm thấy ảnh: {IMG_PATH}")
        return

    original_img = np.array(Image.open(IMG_PATH).convert('RGB').resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])))

    explanation = explainer.explain_instance(
        original_img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # 4. Lưu ảnh
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    save_file = 'lime_result_bcn.png'
    plt.figure(figsize=(8, 8))
    plt.imshow(img_boundry)
    plt.axis('off')
    plt.savefig(save_file, bbox_inches='tight')
    print(f"✅ Đã lưu LIME BCN vào {save_file}")


if __name__ == '__main__':
    run_lime_bcn()