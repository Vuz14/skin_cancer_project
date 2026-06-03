"""Entrypoint for BCN20000 preprocessing profiles."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessed.preprocess_pipeline import PROFILES, preprocess_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BCN20000 images.")
    parser.add_argument("--profile", choices=PROFILES, default="color_safe")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--destination", default=None)
    args = parser.parse_args()
    preprocess_dataset(
        "bcn20000",
        image_size=args.image_size,
        workers=args.workers,
        profile=args.profile,
        destination_override=args.destination,
    )
