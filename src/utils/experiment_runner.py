from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.common import (
    get_warmup_cosine_scheduler,
    save_metadata_info,
    seed_everything,
    set_finetune_mode,
)
from src.utils.losses import FocalLossBCE
from src.utils.trainer import train_loop


STRATEGIES = {
    "strategy1": "image_only",
    "strategy2": "concatenation",
    "strategy3": "film",
    "strategy4": "gating",
}

BACKBONES = {
    "resnet50": "resnet50",
    "effnet_b4": "tf_efficientnet_b4_ns",
    "convnext": "convnext_base",
    "vit": "vit_base_patch16_224",
}

POST_DIAGNOSTIC_BCN_COLUMNS = [
    "concomitant_biopsy",
    "diagnosis",
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_confirm_type",
    "benign_malignant",
    "melanocytic",
]


def select_group_column(df: pd.DataFrame) -> tuple[str, str]:
    """Use lesion identifiers to prevent the same lesion crossing splits."""
    if "lesion_id" in df.columns and df["lesion_id"].notna().any():
        return "lesion_id", "lesion-level"
    raise ValueError("The dataset must provide lesion_id for lesion-level group-safe evaluation.")


def preprocess_ham(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "image_path" not in df.columns:
        df["image_path"] = df["image_id"].astype(str) + ".jpg"
    if "label" not in df.columns:
        if "dx" not in df.columns:
            raise ValueError("HAM input must contain either label or dx.")
        malignant = {"mel", "bcc", "akiec"}
        df["label"] = df["dx"].astype(str).str.lower().isin(malignant).astype(int)
    df["lesion_id"] = df["lesion_id"].fillna(df["image_path"])
    return df.drop(columns=["dx", "dx_type"], errors="ignore")


def preprocess_bcn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "image_path" not in df.columns:
        df["image_path"] = df["isic_id"].astype(str) + ".jpg"

    if "label" not in df.columns:
        if "diagnosis_1" not in df.columns:
            raise ValueError("BCN input must contain either label or diagnosis_1.")
        diagnosis = df["diagnosis_1"].astype(str).str.strip().str.lower()
        valid = diagnosis.isin(["benign", "malignant"])
        excluded = int((~valid).sum())
        if excluded:
            print(f"[Protocol] Excluding {excluded} BCN records without a definitive benign/malignant label.")
        df = df.loc[valid].copy()
        diagnosis = diagnosis.loc[valid]
        df["label"] = (diagnosis == "malignant").astype(int)

    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["image_path"]
    df["lesion_id"] = df["lesion_id"].fillna(df["image_path"])

    # Drop post-diagnostic attributes after creating the binary target.
    return df.drop(columns=POST_DIAGNOSTIC_BCN_COLUMNS, errors="ignore")


def _assert_disjoint(first: pd.DataFrame, second: pd.DataFrame, group_col: str, names: tuple[str, str]) -> None:
    overlap = set(first[group_col].dropna()).intersection(set(second[group_col].dropna()))
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} shared {group_col} values in {names[0]} and {names[1]}.")


def _summarize(metrics: list[dict[str, float]], output_path: Path) -> None:
    detail = pd.DataFrame(metrics)
    detail.to_csv(output_path.with_name(output_path.stem + "_detail.csv"), index=False)
    rows = []
    for metric in ("auc", "acc", "f1", "precision", "recall", "spec"):
        rows.append({
            "metric": metric,
            "mean": detail[metric].mean(),
            "std": detail[metric].std(),
            "mean_std": f"{detail[metric].mean():.4f} +/- {detail[metric].std():.4f}",
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_configuration(base_config: dict, dataset_name: str, strategy: str, backbone_key: str) -> None:
    config = copy.deepcopy(base_config)
    config["METADATA_MODE"] = strategy
    config["MODEL_NAME"] = BACKBONES[backbone_key]
    config["SHORT_NAME"] = backbone_key
    seed_everything(config["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["DEVICE"] = str(device)

    processor = preprocess_ham if dataset_name == "ham10000" else preprocess_bcn
    dataset_class = HAM10000Dataset if dataset_name == "ham10000" else DermoscopyDataset
    train = processor(pd.read_csv(config["TRAIN_CSV"]))
    val = processor(pd.read_csv(config["VAL_CSV"]))
    test = processor(pd.read_csv(config["TEST_CSV"]))
    dev = pd.concat([train, val], ignore_index=True)

    group_col, protocol_level = select_group_column(dev)
    if group_col not in test.columns:
        raise ValueError(f"Group column {group_col!r} is not present in the hold-out test set.")
    _assert_disjoint(dev, test, group_col, ("development", "hold-out test"))
    print(f"[Protocol] {protocol_level} grouping with column {group_col!r}; hold-out overlap = 0.")

    run_root = Path(config["MODEL_OUT"]) / f"CV5_{strategy}_{backbone_key}_{dataset_name}"
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": dataset_name,
            "strategy": strategy,
            "strategy_name": STRATEGIES[strategy],
            "backbone": config["MODEL_NAME"],
            "group_column": group_col,
            "protocol_level": protocol_level,
            "metadata": "pre-diagnostic only",
            "bcn_indeterminate_policy": "excluded" if dataset_name == "bcn20000" else None,
        }, handle, indent=2)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["SEED"])
    validation_results = []
    holdout_results = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(dev, dev["label"], dev[group_col]), start=1):
        fold_train = dev.iloc[train_idx].reset_index(drop=True)
        fold_val = dev.iloc[val_idx].reset_index(drop=True)
        _assert_disjoint(fold_train, fold_val, group_col, (f"fold {fold} train", f"fold {fold} validation"))

        fold_dir = run_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        config["RUN_DIR"] = str(fold_dir)

        train_ds = dataset_class(fold_train, config["IMG_ROOT"], config["IMG_SIZE"], strategy, train=True)
        val_ds = dataset_class(
            fold_val, config["IMG_ROOT"], config["IMG_SIZE"], strategy, train=False,
            external_encoders=train_ds.encoders, external_stats=train_ds.num_mean_std,
        )
        test_ds = dataset_class(
            test, config["IMG_ROOT"], config["IMG_SIZE"], strategy, train=False,
            external_encoders=train_ds.encoders, external_stats=train_ds.num_mean_std,
        )
        save_metadata_info(str(fold_dir / f"meta_info_fold{fold}.pkl"), train_ds.encoders, train_ds.num_mean_std)

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
            model, *loaders, config, criterion, optimizer, scheduler, device, log_suffix=f"fold_{fold}"
        )
        val_metrics["fold"] = fold
        test_metrics["fold"] = fold
        validation_results.append(val_metrics)
        holdout_results.append(test_metrics)
        print(f"[Result] {dataset_name} {backbone_key} {strategy} fold {fold}: val AUC={val_metrics['auc']:.4f}")

    _summarize(validation_results, run_root / "cv_validation_summary.csv")
    _summarize(holdout_results, run_root / "holdout_test_summary.csv")


def run_suite(base_config: dict, dataset_name: str, strategies: list[str], backbones: list[str]) -> None:
    for backbone_key in backbones:
        if backbone_key not in BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone_key}")
        for strategy in strategies:
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}")
            print(f"\n[Run] dataset={dataset_name} backbone={backbone_key} strategy={strategy}")
            run_configuration(base_config, dataset_name, strategy, backbone_key)
