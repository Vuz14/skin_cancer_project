import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.experiment_runner import preprocess_bcn, preprocess_ham, select_group_column


def split_dataframe(df: pd.DataFrame, group_column: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    development_idx, test_idx = next(outer.split(df, df["label"], df[group_column]))
    development = df.iloc[development_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(development, development["label"], development[group_column]))
    train = development.iloc[train_idx].reset_index(drop=True)
    val = development.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate group-safe binary train/validation/test CSV files.")
    parser.add_argument("--dataset", choices=["ham10000", "bcn20000"], required=True)
    parser.add_argument("--source", required=True, help="CSV or XLSX metadata file containing lesion_id.")
    parser.add_argument("--output-dir", default="dataset/metadata/group_safe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    raw = pd.read_excel(source) if source.suffix.lower() == ".xlsx" else pd.read_csv(source)
    processor = preprocess_ham if args.dataset == "ham10000" else preprocess_bcn
    clean = processor(raw)
    group_column, protocol_level = select_group_column(clean)
    train, val, test = split_dataframe(clean, group_column, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(output_dir / f"{args.dataset}_{name}.csv", index=False)

    metadata = {
        "dataset": args.dataset,
        "source": str(source),
        "group_column": group_column,
        "protocol_level": protocol_level,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "groups": {
            "train": int(train[group_column].nunique()),
            "val": int(val[group_column].nunique()),
            "test": int(test[group_column].nunique()),
        },
        "bcn_indeterminate_policy": "excluded" if args.dataset == "bcn20000" else None,
        "seed": args.seed,
    }
    with (output_dir / f"{args.dataset}_split_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
