import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_logic.bcn_dataset import DermoscopyDataset
from src.data_logic.ham_dataset import HAM10000Dataset
from src.models import get_model
from src.utils.experiment_runner import BACKBONES, STRATEGIES, preprocess_bcn, preprocess_ham


DATASET_DEFAULTS = {
    "ham10000": {
        "train_csv": r"D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_train.csv",
        "test_csv": r"D:\skin_cancer_project\dataset\metadata\group_safe\ham10000_test.csv",
        "img_root": r"D:\skin_cancer_project\dataset\Ham10000-color-safe-preprocessed",
        "model_out": r"D:\skin_cancer_project\checkpoint_ham10000",
    },
    "bcn20000": {
        "train_csv": r"D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_train.csv",
        "test_csv": r"D:\skin_cancer_project\dataset\metadata\group_safe\bcn20000_test.csv",
        "img_root": r"D:\skin_cancer_project\dataset\Bcn20000-color-safe-preprocessed",
        "model_out": r"D:\skin_cancer_project\checkpoint_bcn20000",
    },
}


def build_feature_names(dataset) -> list[str]:
    names = list(dataset.numeric_cols)
    for column in dataset.categorical_cols:
        encoder = dataset.encoders[column]
        names.extend(f"{column}={value}" for value in encoder.classes_)
    return names


def encode_flat(dataset, frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        nums = []
        for column in dataset.numeric_cols:
            mean, std = dataset.num_mean_std[column]
            value = row.get(column, mean)
            nums.append((float(value) - mean) / std if not pd.isna(value) else 0.0)

        one_hot = []
        for column in dataset.categorical_cols:
            encoder = dataset.encoders[column]
            raw = str(row.get(column, "unknown"))
            try:
                index = int(encoder.transform([raw])[0])
            except ValueError:
                index = 0
            vector = np.zeros(len(encoder.classes_), dtype=np.float32)
            vector[index] = 1.0
            one_hot.extend(vector)
        rows.append(np.asarray(nums + one_hot, dtype=np.float32))
    return np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def flat_to_metadata(dataset, data: np.ndarray, device: torch.device):
    num_count = len(dataset.numeric_cols)
    if num_count:
        meta_num = torch.tensor(data[:, :num_count], dtype=torch.float32, device=device)
    else:
        meta_num = torch.zeros((data.shape[0], 0), dtype=torch.float32, device=device)

    cat_indices = []
    cursor = num_count
    for column in dataset.categorical_cols:
        card = dataset.cat_cardinalities[column]
        chunk = data[:, cursor: cursor + card]
        cat_indices.append(torch.tensor(np.argmax(chunk, axis=1), dtype=torch.long))
        cursor += card
    if cat_indices:
        meta_cat = torch.stack(cat_indices, dim=1).to(device)
    else:
        meta_cat = torch.zeros((data.shape[0], 0), dtype=torch.long, device=device)
    return meta_num, meta_cat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metadata SHAP summary for a trained checkpoint.")
    parser.add_argument("--dataset", choices=["ham10000", "bcn20000"], required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="strategy3")
    parser.add_argument("--backbone", choices=list(BACKBONES), default="effnet_b4")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--img-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--background-size", type=int, default=15)
    parser.add_argument("--nsamples", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metadata-feature-boost", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    train_csv = args.train_csv or defaults["train_csv"]
    test_csv = args.test_csv or defaults["test_csv"]
    img_root = args.img_root or defaults["img_root"]
    output = Path(args.output) if args.output else Path(defaults["model_out"]) / (
        f"shap_{args.dataset}_{args.strategy}_{args.backbone}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    processor = preprocess_ham if args.dataset == "ham10000" else preprocess_bcn
    dataset_class = HAM10000Dataset if args.dataset == "ham10000" else DermoscopyDataset
    train_df = processor(pd.read_csv(train_csv))
    test_df = processor(pd.read_csv(test_csv))
    train_ds = dataset_class(train_df, img_root, args.image_size, args.strategy, train=False)

    if not train_ds.numeric_cols and not train_ds.categorical_cols:
        raise ValueError("SHAP metadata explanation requires a metadata strategy, not image-only strategy1.")

    config = {
        "METADATA_MODE": args.strategy,
        "MODEL_NAME": BACKBONES[args.backbone],
        "PRETRAINED": not args.no_pretrained,
        "METADATA_FEATURE_BOOST": args.metadata_feature_boost,
    }
    device = torch.device(args.device)
    model = get_model(config, train_ds.cat_cardinalities, len(train_ds.numeric_cols)).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    subset = test_df.sample(n=min(args.sample_size, len(test_df)), random_state=42)
    background = train_ds.df.sample(n=min(args.background_size, len(train_ds.df)), random_state=123)
    test_data = encode_flat(train_ds, subset)
    background_data = encode_flat(train_ds, background)
    feature_names = build_feature_names(train_ds)

    def predict_from_metadata(data: np.ndarray):
        with torch.no_grad():
            batch = data.shape[0]
            dummy_img = torch.zeros((batch, 3, args.image_size, args.image_size), device=device)
            meta_num, meta_cat = flat_to_metadata(train_ds, data, device)
            logits = model(dummy_img, meta_num, meta_cat)
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    explainer = shap.KernelExplainer(predict_from_metadata, background_data)
    shap_values = explainer.shap_values(test_data, nsamples=args.nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(test_data, columns=feature_names),
        show=False,
        max_display=20,
    )
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary: {output}")


if __name__ == "__main__":
    main()
