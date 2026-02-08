import os
import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
CONFIG = {
    "HAM10000": {
        "ENABLE": True,
        "SRC_DIR": "/mnt/d/skin_cancer_project/dataset/Ham10k",  # Đường dẫn folder ảnh GỐC
        "DST_DIR": "/mnt/d/skin_cancer_project/dataset/Ham10000-preprocessed",  # Folder ĐÍCH
        "IMG_SIZE_SAVE": 450  # Lưu dư ra để khi train crop 384 không bị vỡ
    },
    "BCN20000": {
        "ENABLE": True,
        "SRC_DIR": "/mnt/d/skin_cancer_project/dataset/Bcn20k",
        "DST_DIR": "/mnt/d/skin_cancer_project/dataset/Bcn20000-preprocessed",
        "IMG_SIZE_SAVE": 450
    }
}


# ==============================================================================
# 2. CORE LOGIC: XỬ LÝ ẢNH
# ==============================================================================
def remove_hair(image):
    """DullRazor: Xóa lông"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result


def shades_of_gray(img, power=6):
    """Color Constancy: Cân bằng màu"""
    img_dtype = img.dtype
    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    return np.clip(img, 0, 255).astype(img_dtype)


def _center_crop_square(img):
    """Cắt hình vuông ở giữa (Safe Crop)"""
    h, w = img.shape[:2]
    min_side = min(h, w)
    start_x = (w - min_side) // 2
    start_y = (h - min_side) // 2
    return img[start_y:start_y + min_side, start_x:start_x + min_side]


def crop_lesion_roi_smart(image, expansion_ratio=0.3):
    """
    Smart ROI: Tìm vết bệnh, cắt vuông và mở rộng vùng đệm.
    Nếu không tìm thấy hoặc quá nhỏ -> Trả về Center Crop (Fallback).
    """
    try:
        h_img, w_img = image.shape[:2]

        # 1. Tìm vết bệnh bằng Saturation channel
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        blurred = cv2.GaussianBlur(s_channel, (35, 35), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fallback 1: Không thấy contour
        if not contours:
            return _center_crop_square(image)

        c = max(contours, key=cv2.contourArea)
        # Fallback 2: Vết bệnh quá nhỏ (< 0.5% ảnh) -> Nhiễu
        if cv2.contourArea(c) < (h_img * w_img * 0.005):
            return _center_crop_square(image)

        x, y, w, h = cv2.boundingRect(c)

        # 2. Tính toán vùng cắt hình vuông
        center_x = x + w // 2
        center_y = y + h // 2

        max_side = max(w, h)
        new_side = int(max_side * (1 + expansion_ratio))  # Mở rộng thêm 30%
        new_side = max(new_side, 224)  # Không được nhỏ hơn 224

        half_side = new_side // 2

        x1 = max(0, center_x - half_side)
        y1 = max(0, center_y - half_side)
        x2 = min(w_img, center_x + half_side)
        y2 = min(h_img, center_y + half_side)

        roi = image[y1:y2, x1:x2]

        # Fallback 3: Kiểm tra lại kích thước sau cắt
        if roi.shape[0] < 50 or roi.shape[1] < 50:
            return _center_crop_square(image)

        return roi
    except:
        return _center_crop_square(image)


# ==============================================================================
# 3. WORKER: XỬ LÝ VÀ LƯU 2 PHIÊN BẢN
# ==============================================================================
def process_single_image(args):
    src_path, dst_dir, target_size = args
    fname = os.path.basename(src_path)

    # Định nghĩa tên file đầu ra
    path_clean = os.path.join(dst_dir, "clean_" + fname)  # Bản an toàn
    path_roi = os.path.join(dst_dir, "roi_" + fname)  # Bản tập trung

    # Nếu cả 2 đã tồn tại thì bỏ qua (Resume)
    if os.path.exists(path_clean) and os.path.exists(path_roi):
        return

    try:
        # Đọc ảnh
        img = cv2.imread(src_path)
        if img is None: return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- BƯỚC 1: XỬ LÝ CHUNG (Nặng nhất) ---
        img = remove_hair(img)
        img = shades_of_gray(img)

        # --- BƯỚC 2: TẠO BẢN CLEAN (Center Crop) ---
        # Dùng cho cả Train (bối cảnh) và Test (chuẩn)
        if not os.path.exists(path_clean):
            img_clean = _center_crop_square(img)
            img_clean_resized = cv2.resize(img_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path_clean, cv2.cvtColor(img_clean_resized, cv2.COLOR_RGB2BGR))

        # --- BƯỚC 3: TẠO BẢN ROI (Smart Crop) ---
        # Chỉ dùng cho Train để model học chi tiết
        if not os.path.exists(path_roi):
            img_roi = crop_lesion_roi_smart(img, expansion_ratio=0.3)
            img_roi_resized = cv2.resize(img_roi, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path_roi, cv2.cvtColor(img_roi_resized, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"\n❌ Lỗi file {fname}: {e}")


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    print("🚀 BẮT ĐẦU PIPELINE: HAIR REMOVAL + SoG + DUAL CROP (Clean & ROI)")
    print(f"🔥 CPU Cores: {os.cpu_count()}")

    for dataset_name, cfg in CONFIG.items():
        if not cfg["ENABLE"]: continue

        print(f"\nDataset: {dataset_name}")
        src_dir = cfg["SRC_DIR"]
        dst_dir = cfg["DST_DIR"]

        if not os.path.exists(src_dir):
            print(f"⚠️ Không tìm thấy nguồn: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        # Quét file
        extensions = ['*.jpg', '*.jpeg', '*.png']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(src_dir, ext)))

        print(f"📂 Nguồn: {src_dir}")
        print(f"📂 Đích : {dst_dir}")
        print(f"🖼️ Số lượng gốc: {len(files)} ảnh")
        print(f"💾 Số lượng sẽ tạo: {len(files) * 2} ảnh (Clean + ROI)")

        tasks = []
        for f in files:
            tasks.append((f, dst_dir, cfg["IMG_SIZE_SAVE"]))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"))

        print(f"✅ Hoàn tất {dataset_name}!")

    print("\n🎉 XONG! Dữ liệu đã sẵn sàng.")
    print("👉 Khi Train: Load cả 'clean_' và 'roi_'")
    print("👉 Khi Test : Chỉ load 'clean_'")


if __name__ == "__main__":
    main()