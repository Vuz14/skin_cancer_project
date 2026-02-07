import os
import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. CẤU HÌNH (Sửa đường dẫn tại đây)
# ==============================================================================
CONFIG = {
    # Cấu hình cho HAM10000
    "HAM10000": {
        "ENABLE": True,  # Đặt False nếu không muốn chạy dataset này
        "SRC_DIR": "/mnt/d/skin_cancer_project/dataset/Ham10k",
        "DST_DIR": "/mnt/d/skin_cancer_project/dataset/Ham10000-preprocessed",
        "IMG_SIZE_SAVE": 450  # Lưu lớn hơn 384 một chút để khi train crop là vừa đẹp
    },

    # Cấu hình cho BCN20000
    "BCN20000": {
        "ENABLE": True,
        "SRC_DIR": "/mnt/d/skin_cancer_project/dataset/Bcn20k",
        "DST_DIR": "/mnt/d/skin_cancer_project/dataset/Bcn20000-preprocessed",
        "IMG_SIZE_SAVE": 450
    }
}


# ==============================================================================
# 2. CÁC THUẬT TOÁN XỬ LÝ ẢNH (Core Logic)
# ==============================================================================
def remove_hair(image):
    """
    Thuật toán DullRazor: Loại bỏ lông và vật cản mảnh
    """
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. BlackHat Transform để tìm chi tiết tối (lông) trên nền sáng
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Tạo Mask (Ngưỡng 10 là kinh nghiệm thực tế tốt cho ảnh da)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # 4. Inpainting để lấp đầy vùng lông
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result


def shades_of_gray(img, power=6):
    """
    Color Constancy: Chuẩn hóa ánh sáng về tone trung tính
    """
    img_dtype = img.dtype
    img = img.astype('float32')

    # Tính vector ánh sáng
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))

    # Chuẩn hóa
    img = np.multiply(img, rgb_vec)
    return np.clip(img, 0, 255).astype(img_dtype)


# ==============================================================================
# 3. XỬ LÝ ĐA LUỒNG (Multiprocessing Worker)
# ==============================================================================
def process_single_image(args):
    """Worker xử lý 1 ảnh: Đọc -> Xóa lông -> SoG -> Resize -> Lưu"""
    src_path, dst_path, target_size = args

    # Nếu ảnh đích đã có thì bỏ qua (Resume capability)
    if os.path.exists(dst_path):
        return

    try:
        # Đọc ảnh (OpenCV đọc BGR)
        img = cv2.imread(src_path)
        if img is None: return

        # Chuyển sang RGB để xử lý
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- BƯỚC 1: XÓA LÔNG ---
        img = remove_hair(img)

        # --- BƯỚC 2: CHUẨN HÓA MÀU ---
        img = shades_of_gray(img)

        # --- BƯỚC 3: RESIZE ---
        # Resize về kích thước lưu trữ (ví dụ 450x450)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        # Chuyển lại BGR để lưu bằng OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, img)

    except Exception as e:
        print(f"\n❌ Lỗi file {os.path.basename(src_path)}: {e}")


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
def main():
    print("🚀 BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU DA LIỄU")
    print(f"🔥 CPU Cores Available: {os.cpu_count()}")

    for dataset_name, cfg in CONFIG.items():
        if not cfg["ENABLE"]:
            continue

        print(f"\ndataset: {dataset_name}")
        print("-" * 40)

        src_dir = cfg["SRC_DIR"]
        dst_dir = cfg["DST_DIR"]

        if not os.path.exists(src_dir):
            print(f"⚠️ Không tìm thấy thư mục nguồn: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        # Quét tất cả file ảnh
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(src_dir, ext)))

        print(f"📂 Nguồn: {src_dir}")
        print(f"📂 Đích : {dst_dir}")
        print(f"🖼️ Số lượng: {len(files)} ảnh")
        print("⏳ Đang xử lý (Xóa lông + SoG + Resize)...")

        # Chuẩn bị tham số cho worker
        tasks = []
        for f in files:
            fname = os.path.basename(f)
            dst_path = os.path.join(dst_dir, fname)
            tasks.append((f, dst_path, cfg["IMG_SIZE_SAVE"]))

        # Chạy Multiprocessing
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"))

        print(f"✅ Hoàn tất {dataset_name}!")

    print("\n🎉 TẤT CẢ HOÀN TẤT! Giờ bạn có thể dùng thư mục mới để train.")


if __name__ == "__main__":
    main()