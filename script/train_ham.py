import argparse
import os
import sys
from pathlib import Path


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.experiment_runner import BACKBONES, STRATEGIES, run_suite


CONFIG = {
    "TRAIN_CSV": r"D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_train.csv",
    "VAL_CSV": r"D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_val.csv",
    "TEST_CSV": r"D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_test.csv",
    "IMG_ROOT": r"D:\skin_cancer_project\dataset\Ham10000-paper-preprocessed",
    "MODEL_OUT": r"D:\skin_cancer_project\checkpoint_ham10000",
    "SEED": 42,
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 4,
    "EPOCHS": 20,
    "BASE_LR": 8e-5,
    "WARMUP_EPOCHS": 3,
    "WEIGHT_DECAY": 1e-3,
    "PRETRAINED": True,
    "FINE_TUNE_MODE": "full_unfreeze",
    "METADATA_FEATURE_BOOST": 1.0,
    "FOCAL_ALPHA": 0.75,
    "FOCAL_GAMMA": 2.0,
    "PATIENCE": 5,
    "ENABLE_GRAD_CAM": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware HAM10000 experiments.")
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES))
    parser.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    parser.add_argument("--metadata-dir", default=None, help="Directory containing group-safe ham10000_{train,val,test}.csv.")
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.metadata_dir:
        root = Path(args.metadata_dir)
        config.update({
            "TRAIN_CSV": str(root / "ham10000_train.csv"),
            "VAL_CSV": str(root / "ham10000_val.csv"),
            "TEST_CSV": str(root / "ham10000_test.csv"),
        })
    run_suite(config, "ham10000", args.strategies, args.backbones)


if __name__ == "__main__":
    main()
