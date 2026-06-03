import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.data_logic.common_transforms import AUGMENTATION_PROFILES
from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.common import get_warmup_cosine_scheduler, save_metadata_info, seed_everything, set_finetune_mode
from src.utils.experiment_runner import (
    BACKBONE_TRAINING_OVERRIDES,
    BACKBONES,
    STRATEGIES,
    preprocess_bcn,
    preprocess_ham,
    select_group_column,
)
from src.utils.losses import FocalLossBCE
from src.utils.trainer import train_loop


DEFAULTS = {
    "ham10000": {
        "img_root": r"D:\skin_cancer_project\dataset\Ham10000-color-safe-preprocessed",
        "model_out": r"D:\skin_cancer_project\checkpoint_ham10000_single",
        "base_lr": 8e-5,
    },
    "bcn20000": {
        "img_root": r"D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed",
        "model_out": r"D:\skin_cancer_project\checkpoint_bcn20000_single",
        "base_lr": 1e-4,
    },
}


def assert_disjoint(first: pd.DataFrame, second: pd.DataFrame, group_col: str, names: tuple[str, str]) -> None:
    overlap = set(first[group_col].dropna()).intersection(set(second[group_col].dropna()))
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} shared {group_col} values in {names[0]} and {names[1]}.")


def build_config(args: argparse.Namespace) -> dict:
    defaults = DEFAULTS[args.dataset]
    backbone_overrides = BACKBONE_TRAINING_OVERRIDES.get(args.backbone, {})
    metadata_dir = Path(args.metadata_dir)
    config = {
        "TRAIN_CSV": str(metadata_dir / f"{args.dataset}_train.csv"),
        "VAL_CSV": str(metadata_dir / f"{args.dataset}_val.csv"),
        "TEST_CSV": str(metadata_dir / f"{args.dataset}_test.csv"),
        "IMG_ROOT": args.img_root or defaults["img_root"],
        "MODEL_OUT": args.model_out or defaults["model_out"],
        "SEED": args.seed,
        "IMG_SIZE": args.image_size,
        "BATCH_SIZE": args.batch_size if args.batch_size is not None else backbone_overrides.get("BATCH_SIZE", 32),
        "NUM_WORKERS": args.num_workers,
        "EPOCHS": args.epochs,
        "BASE_LR": (
            args.base_lr
            if args.base_lr is not None
            else backbone_overrides.get("BASE_LR", defaults["base_lr"])
        ),
        "WARMUP_EPOCHS": args.warmup_epochs,
        "WEIGHT_DECAY": (
            args.weight_decay
            if args.weight_decay is not None
            else backbone_overrides.get("WEIGHT_DECAY", 1e-3)
        ),
        "PRETRAINED": not args.no_pretrained,
        "FINE_TUNE_MODE": args.fine_tune_mode or backbone_overrides.get("FINE_TUNE_MODE", "full_unfreeze"),
        "UNFREEZE_KEYWORDS": backbone_overrides.get("UNFREEZE_KEYWORDS", []),
        "METADATA_FEATURE_BOOST": args.metadata_feature_boost,
        "FOCAL_ALPHA": args.focal_alpha,
        "FOCAL_GAMMA": args.focal_gamma,
        "PATIENCE": args.patience,
        "ENABLE_GRAD_CAM": False,
        "AUGMENTATION_PROFILE": args.augmentation_profile,
        "METADATA_MODE": args.strategy,
        "MODEL_NAME": BACKBONES[args.backbone],
        "SHORT_NAME": args.backbone,
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one train/val/test run without 5-fold CV.")
    parser.add_argument("--dataset", choices=["ham10000", "bcn20000"], required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="strategy3")
    parser.add_argument("--backbone", choices=list(BACKBONES), default="effnet_b4")
    parser.add_argument("--metadata-dir", default="dataset/metadata/group_safe")
    parser.add_argument("--img-root", default=None)
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--augmentation-profile", choices=sorted(AUGMENTATION_PROFILES), default="light")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--fine-tune-mode", default=None)
    parser.add_argument("--metadata-feature-boost", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    seed_everything(config["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["DEVICE"] = str(device)

    processor = preprocess_ham if args.dataset == "ham10000" else preprocess_bcn
    dataset_class = HAM10000Dataset if args.dataset == "ham10000" else DermoscopyDataset
    train_df = processor(pd.read_csv(config["TRAIN_CSV"]))
    val_df = processor(pd.read_csv(config["VAL_CSV"]))
    test_df = processor(pd.read_csv(config["TEST_CSV"]))

    group_col, protocol_level = select_group_column(pd.concat([train_df, val_df], ignore_index=True))
    assert_disjoint(train_df, val_df, group_col, ("train", "validation"))
    assert_disjoint(pd.concat([train_df, val_df], ignore_index=True), test_df, group_col, ("development", "test"))

    run_dir = (
        Path(config["MODEL_OUT"])
        / f"single_{args.strategy}_{args.backbone}_{args.dataset}_{config['AUGMENTATION_PROFILE']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config["RUN_DIR"] = str(run_dir)
    with (run_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "strategy": args.strategy,
                "strategy_name": STRATEGIES[args.strategy],
                "backbone": config["MODEL_NAME"],
                "img_root": config["IMG_ROOT"],
                "augmentation_profile": config["AUGMENTATION_PROFILE"],
                "protocol_level": protocol_level,
                "group_column": group_col,
                "training_mode": "single train/val/test split",
            },
            handle,
            indent=2,
        )

    train_ds = dataset_class(
        train_df,
        config["IMG_ROOT"],
        config["IMG_SIZE"],
        args.strategy,
        train=True,
        augmentation_profile=config["AUGMENTATION_PROFILE"],
    )
    val_ds = dataset_class(
        val_df,
        config["IMG_ROOT"],
        config["IMG_SIZE"],
        args.strategy,
        train=False,
        external_encoders=train_ds.encoders,
        external_stats=train_ds.num_mean_std,
        augmentation_profile=config["AUGMENTATION_PROFILE"],
    )
    test_ds = dataset_class(
        test_df,
        config["IMG_ROOT"],
        config["IMG_SIZE"],
        args.strategy,
        train=False,
        external_encoders=train_ds.encoders,
        external_stats=train_ds.num_mean_std,
        augmentation_profile=config["AUGMENTATION_PROFILE"],
    )
    save_metadata_info(str(run_dir / "meta_info_single.pkl"), train_ds.encoders, train_ds.num_mean_std)

    loaders = [
        DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"]),
        DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]),
        DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]),
    ]

    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    set_finetune_mode(model, config["FINE_TUNE_MODE"], config.get("UNFREEZE_KEYWORDS", []))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["BASE_LR"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = get_warmup_cosine_scheduler(optimizer, config["WARMUP_EPOCHS"], config["EPOCHS"])
    criterion = FocalLossBCE(alpha=config["FOCAL_ALPHA"], gamma=config["FOCAL_GAMMA"])
    _, _, val_metrics, test_metrics = train_loop(
        model, *loaders, config, criterion, optimizer, scheduler, device, log_suffix="single"
    )
    print(f"[Single] val AUC={val_metrics['auc']:.4f} test AUC={test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
