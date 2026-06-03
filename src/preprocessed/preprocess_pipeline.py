import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DATASETS = {
    "ham10000": {
        "source": Path(r"D:\skin_cancer_project\dataset\HAM_10000"),
        "destinations": {
            "color_safe": Path(r"D:\skin_cancer_project\dataset\Ham10000-color-safe-preprocessed"),
            "legacy": Path(r"D:\skin_cancer_project\dataset\Ham10000-paper-preprocessed"),
            "raw_resize": Path(r"D:\skin_cancer_project\dataset\Ham10000-raw-resize"),
        },
    },
    "bcn20000": {
        "source": Path(r"D:\skin_cancer_project\dataset\BCN_20000"),
        "destinations": {
            "color_safe": Path(r"D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed"),
            "legacy": Path(r"D:\skin_cancer_project\dataset\Bcn20000-paper-preprocessed"),
            "raw_resize": Path(r"D:\skin_cancer_project\dataset\Bcn20000-raw-resize"),
        },
    },
}

PROFILES = ("color_safe", "legacy", "raw_resize")


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


def enhance_contrast_legacy(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=60, sigmaSpace=60)


def enhance_luminance_preserve_chroma(
    image: np.ndarray,
    clip_limit: float = 1.1,
    tile_grid_size: tuple[int, int] = (8, 8),
    blend: float = 0.25,
) -> np.ndarray:
    """Apply a weak LAB-L enhancement while leaving chroma channels untouched.

    The legacy pipeline replaced the full L channel after CLAHE and also ran
    Gray-World before it. That is difficult to defend for ABCDE color analysis.
    This version only blends a small amount of CLAHE-enhanced luminance back
    into the original L channel and keeps LAB a/b channels unchanged.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lightness = clahe.apply(lightness)
    blended_lightness = cv2.addWeighted(lightness, 1.0 - blend, enhanced_lightness, blend, 0)
    return cv2.cvtColor(cv2.merge((blended_lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def preprocess_image(image: np.ndarray, profile: str) -> np.ndarray:
    if profile == "raw_resize":
        return image
    if profile == "legacy":
        image = remove_hair(image)
        image = gray_world(image)
        return enhance_contrast_legacy(image)
    if profile == "color_safe":
        image = remove_hair(image)
        return enhance_luminance_preserve_chroma(image)
    raise ValueError(f"Unknown preprocessing profile: {profile}")


def process_image(task: tuple[Path, Path, int, str]) -> tuple[str, bool]:
    source, destination, size, profile = task
    image = cv2.imread(str(source))
    if image is None:
        return source.name, False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image, profile)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return source.name, bool(success)


def preprocess_dataset(
    dataset_name: str,
    image_size: int,
    workers: int,
    profile: str = "color_safe",
    destination_override: str | None = None,
) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Expected one of: {', '.join(PROFILES)}")
    config = DATASETS[dataset_name]
    source = config["source"]
    destination = Path(destination_override) if destination_override else config["destinations"][profile]
    if not source.exists():
        raise FileNotFoundError(f"Image source directory does not exist: {source}")
    files = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tasks = [(path, destination / path.name, image_size, profile) for path in files]
    destination.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        desc = f"{dataset_name}:{profile}"
        for name, success in tqdm(executor.map(process_image, tasks), total=len(tasks), desc=desc):
            if not success:
                failures.append(name)
    print(
        f"[Preprocess] {dataset_name} ({profile}): "
        f"saved {len(files) - len(failures)}/{len(files)} images to {destination}"
    )
    if failures:
        raise RuntimeError(f"Failed to preprocess {len(failures)} images; examples: {failures[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dermoscopy images for model input.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="color_safe",
        help=(
            "color_safe preserves chroma better and writes to *-color-safe-preprocessed; "
            "legacy reproduces the previous Gray-World + full L-CLAHE pipeline; "
            "raw_resize only resizes the RGB image."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--destination",
        default=None,
        help="Optional output folder. Only allowed when preprocessing one dataset.",
    )
    args = parser.parse_args()
    if args.destination and len(args.datasets) != 1:
        raise ValueError("--destination can only be used with exactly one dataset.")
    for dataset_name in args.datasets:
        preprocess_dataset(
            dataset_name,
            args.image_size,
            args.workers,
            profile=args.profile,
            destination_override=args.destination,
        )


if __name__ == "__main__":
    main()
