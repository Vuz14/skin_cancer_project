import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DATASETS = {
    "ham10000": {
        "source": Path(r"D:\skin_cancer_project\dataset\HAM_10000"),
        "destination": Path(r"D:\skin_cancer_project\dataset\Ham10000-paper-preprocessed"),
    },
    "bcn20000": {
        "source": Path(r"D:\skin_cancer_project\dataset\BCN_20000"),
        "destination": Path(r"D:\skin_cancer_project\dataset\Bcn20000-paper-preprocessed"),
    },
}


def remove_hair(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def gray_world(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    channel_means = image_float.mean(axis=(0, 1))
    target = channel_means.mean()
    scaled = image_float * (target / np.maximum(channel_means, 1e-6))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=60, sigmaSpace=60)


def process_image(task: tuple[Path, Path, int]) -> tuple[str, bool]:
    source, destination, size = task
    image = cv2.imread(str(source))
    if image is None:
        return source.name, False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = remove_hair(image)
    image = gray_world(image)
    image = enhance_contrast(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return source.name, bool(success)


def preprocess_dataset(dataset_name: str, image_size: int, workers: int) -> None:
    config = DATASETS[dataset_name]
    source = config["source"]
    destination = config["destination"]
    if not source.exists():
        raise FileNotFoundError(f"Image source directory does not exist: {source}")
    files = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tasks = [(path, destination / path.name, image_size) for path in files]
    destination.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for name, success in tqdm(executor.map(process_image, tasks), total=len(tasks), desc=dataset_name):
            if not success:
                failures.append(name)
    print(f"[Preprocess] {dataset_name}: saved {len(files) - len(failures)}/{len(files)} images to {destination}")
    if failures:
        raise RuntimeError(f"Failed to preprocess {len(failures)} images; examples: {failures[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical preprocessing pipeline used by all reported experiments.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for dataset_name in args.datasets:
        preprocess_dataset(dataset_name, args.image_size, args.workers)


if __name__ == "__main__":
    main()
