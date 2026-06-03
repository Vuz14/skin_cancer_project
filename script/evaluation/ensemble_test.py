import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.models import get_model
from src.utils.common import load_metadata_info, seed_everything
from src.utils.experiment_runner import BACKBONES, preprocess_bcn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a five-fold BCN ensemble on the hold-out test set.")
    parser.add_argument("--strategy", default="strategy3", choices=["strategy1", "strategy2", "strategy3", "strategy4"])
    parser.add_argument("--backbone", default="effnet_b4", choices=list(BACKBONES))
    parser.add_argument("--img-root", default=r"D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed")
    parser.add_argument("--model-out", default=r"D:\skin_cancer_project\checkpoint_bcn20000")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = {
        "TEST_CSV": r"D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_test.csv",
        "IMG_ROOT": args.img_root,
        "MODEL_OUT": Path(args.model_out),
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "PRETRAINED": True,
        "METADATA_FEATURE_BOOST": 1.0,
        "IMG_SIZE": 224,
        "METADATA_MODE": args.strategy,
        "MODEL_NAME": BACKBONES[args.backbone],
        "BATCH_SIZE": args.batch_size,
        "SEED": 42,
    }
    seed_everything(config["SEED"])
    device = torch.device(config["DEVICE"])
    run_dir = config["MODEL_OUT"] / f"CV5_{args.strategy}_{args.backbone}_bcn20000"
    test_df = preprocess_bcn(pd.read_csv(config["TEST_CSV"]))
    labels = test_df["label"].to_numpy()
    fold_probabilities = []

    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        encoders, stats = load_metadata_info(str(fold_dir / f"meta_info_fold{fold}.pkl"))
        test_ds = DermoscopyDataset(
            test_df, config["IMG_ROOT"], config["IMG_SIZE"], args.strategy, train=False,
            external_encoders=encoders, external_stats=stats,
        )
        loader = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4)
        model = get_model(config, test_ds.cat_cardinalities, len(test_ds.numeric_cols)).to(device)
        checkpoint = torch.load(fold_dir / f"best_{args.strategy}_fold_{fold}.pt", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        probabilities = []
        with torch.no_grad():
            for images, (meta_num, meta_cat), _ in loader:
                logits = model(images.to(device), meta_num.to(device).float(), meta_cat.to(device).long())
                probabilities.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        fold_probabilities.append(probabilities)

    probabilities = np.mean(np.asarray(fold_probabilities), axis=0)
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(f"AUC: {roc_auc_score(labels, probabilities):.4f}")
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(f"F1: {f1_score(labels, predictions, zero_division=0):.4f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(labels, predictions, zero_division=0):.4f}")
    print(f"Specificity: {tn / (tn + fp + 1e-12):.4f}")


if __name__ == "__main__":
    main()
