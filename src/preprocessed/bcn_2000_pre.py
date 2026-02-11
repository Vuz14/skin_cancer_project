import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CSV_PATH = r"D:\skin_cancer_project\dataset\metadata\BCN20000_metadata.xlsx"
IMG_ROOT = r"D:\skin_cancer_project\dataset\BCN_20000"
OUT_DIR  = r"D:\skin_cancer_project\dataset\Bcn20000-preprocessed-1"
IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Hair Removal (SAFE – DullRazor-lite)
# =========================
def remove_hair(img):
    """
    Light hair removal for dermoscopy.
    Does NOT destroy pigment structure.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    img_clean = cv2.inpaint(
        img,
        mask,
        inpaintRadius=1,
        flags=cv2.INPAINT_TELEA
    )
    return img_clean


# =========================
# MAIN
# =========================
def main():
    print("📄 Loading metadata...")
    df = pd.read_excel(CSV_PATH)   # ✅ FIX: đọc đúng Excel

    if 'image_path' not in df.columns:
        if 'isic_id' not in df.columns:
            raise ValueError("Metadata phải có cột 'isic_id' hoặc 'image_path'")
        df['image_path'] = df['isic_id'].astype(str) + ".jpg"

    print(f"🔍 Total images: {len(df)}")

    error_count = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="🧠 Preprocessing",
        ncols=100
    ):
        img_path = os.path.join(IMG_ROOT, row['image_path'])
        out_path = os.path.join(OUT_DIR, row['image_path'])

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Cannot read image")

            # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1️⃣ Hair removal (light)
            img = remove_hair(img)

            # 2️⃣ Resize (INTER_AREA tốt cho downscale)
            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            # Ensure output subfolder exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save (RGB → BGR)
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            error_count += 1
            print(f"❌ Error on {row['image_path']}: {e}")

    print("\n✅ Preprocessing completed.")
    if error_count > 0:
        print(f"⚠️ {error_count} images failed and were skipped.")
    else:
        print("✅ No errors.")
    print(f"📁 Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
