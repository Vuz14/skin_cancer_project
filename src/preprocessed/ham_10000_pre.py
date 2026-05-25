"""Backward-compatible entrypoint for the canonical HAM10000 preprocessing."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessed.preprocess_pipeline import preprocess_dataset


if __name__ == "__main__":
    preprocess_dataset("ham10000", image_size=224, workers=4)
