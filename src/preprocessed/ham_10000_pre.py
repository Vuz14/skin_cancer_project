import cv2
import numpy as np
import os
import pandas as pd
import sys

from skimage import morphology
from tqdm import tqdm

sys.path.append(r"C:\Users\ADMIN\PycharmProjects\NCKH_Test\U-2-Net-master")

import torch

# === Config ===
# (Giữ nguyên Config)
CSV_PATH = r"D:\skin_cancer_project\dataset\metadata\HAM10000_metadata.csv"
IMG_ROOT = r"D:\skin_cancer_project\dataset\Ham10k"
OUT_DIR = r"D:\skin_cancer_project\dataset\Ham10000-preprocessed"  # Đổi tên thư mục OUT để tránh ghi đè
IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)


# === Utility functions ===

# 1. Loại bỏ lông tóc
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)


# 2. Hiệu chỉnh ánh sáng không đều
def correct_illumination(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=30)
    illumination = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB).astype(np.float32)
    img_float = img.astype(np.float32)
    corrected = cv2.divide(img_float, illumination + 1e-6, scale=255)
    return np.clip(corrected, 0, 255).astype(np.uint8)


# 3. Cân bằng màu (Gray World)
def color_correction_grayworld(img):
    img = img.astype(np.float32)
    mean_r, mean_g, mean_b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    mean_gray = (mean_r + mean_g + mean_b) / 3
    img[:, :, 0] *= mean_gray / (mean_r + 1e-6)
    img[:, :, 1] *= mean_gray / (mean_g + 1e-6)
    img[:, :, 2] *= mean_gray / (mean_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


# 4. Tăng cường tương phản (Đã Tinh chỉnh)
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Tinh chỉnh: Giảm clipLimit từ 2.0 xuống 1.5 để giảm tăng tương phản quá mức, hạn chế nhiễu
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# 5. Thêm hàm Lọc nhiễu mới
def smooth_image(img):
    # Sử dụng Bilateral Filter: d=9, sigmaColor=75, sigmaSpace=75
    # Tinh chỉnh: Giảm sigmaColor và sigmaSpace để làm mịn ít hơn, giữ lại chi tiết
    # Thử nghiệm với (50, 50) hoặc (60, 60)
    return cv2.bilateralFilter(img, d=9, sigmaColor=60, sigmaSpace=60)  # Ví dụ


# === Main preprocessing ===
df = pd.read_csv(CSV_PATH)
if 'image_path' not in df.columns:
    df['image_path'] = df['image_id'].astype(str) + ".jpg"

print(f"🔍 Total images: {len(df)}")

error_count = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="🧠 Preprocessing images", ncols=100):
    img_path = os.path.join(IMG_ROOT, row['image_path'])
    out_path = os.path.join(OUT_DIR, row['image_path'])

    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Loại bỏ lông tóc
        img = remove_hair(img)

        # 2. Hiệu chỉnh ánh sáng không đều
        img = correct_illumination(img)

        # 3. Cân bằng màu
        img = color_correction_grayworld(img)

        # 4. Tăng cường tương phản (Đã tinh chỉnh CLAHE)
        img = enhance_contrast(img)

        # 5. LỌC NHIỄU (BƯỚC MỚI) - làm dịu các cạnh quá sắc nét và nhiễu sau CLAHE
        img = smooth_image(img)

        # 7. Thay đổi kích thước
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        error_count += 1
        print(f"❌ Error on {row['image_path']}: {e}")

# === Summary ===
print("\n✅ Hoàn tất tiền xử lý ảnh (Test mode).")
if error_count > 0:
    print(f"⚠️ Có {error_count} ảnh bị lỗi và đã bị bỏ qua.")
else:
    print("✅ Không có ảnh nào bị lỗi.")
print(f"📁 Ảnh đã xử lý được lưu tại: {OUT_DIR}")
