import argparse
from pathlib import Path

import pandas as pd


STRATEGY_LABELS = {
    "strategy1": "Strategy 1 (Image-only)",
    "strategy2": "Strategy 2 (Concatenation)",
    "strategy3": "Strategy 3 (FiLM)",
    "strategy4": "Strategy 4 (Gating)",
}

BACKBONE_LABELS = {
    "resnet50": "ResNet50",
    "effnet_b4": "EffNet-B4",
    "convnext": "ConvNeXt",
    "vit": "ViT",
}


def parse_run_name(path: Path) -> tuple[str, str, str]:
    parts = path.name.split("_")
    if len(parts) < 4 or parts[0] != "CV5":
        raise ValueError(f"Unexpected run directory: {path.name}")
    strategy = parts[1]
    dataset = parts[-1]
    backbone = "_".join(parts[2:-1])
    return dataset, backbone, strategy


def collect(checkpoint_roots: list[Path]) -> pd.DataFrame:
    records = []
    for root in checkpoint_roots:
        for run_dir in sorted(root.glob("CV5_strategy*")):
            detail_path = run_dir / "cv_validation_summary_detail.csv"
            if not detail_path.exists():
                continue
            dataset, backbone, strategy = parse_run_name(run_dir)
            detail = pd.read_csv(detail_path)
            record = {
                "Dataset": dataset.upper(),
                "Architecture": BACKBONE_LABELS.get(backbone, backbone),
                "Strategy": STRATEGY_LABELS.get(strategy, strategy),
                "strategy_key": strategy,
            }
            for column, label in (("auc", "AUC"), ("acc", "Accuracy"), ("f1", "F1-Score"), ("spec", "Specificity")):
                record[label] = f"{detail[column].mean() * 100:.2f} +/- {detail[column].std() * 100:.2f}%"
                record[label + "_mean"] = detail[column].mean()
            records.append(record)
    if not records:
        raise RuntimeError("No new CV validation files found. Run train_ham.py/train_bcn.py first.")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synchronized paper tables from fold validation outputs.")
    parser.add_argument(
        "--checkpoint-roots", nargs="+",
        default=["checkpoint_ham10000", "checkpoint_bcn20000"],
    )
    parser.add_argument("--output-dir", default="results/paper_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect([Path(path) for path in args.checkpoint_roots])
    result_columns = ["Dataset", "Architecture", "Strategy", "AUC", "Accuracy", "F1-Score", "Specificity"]
    table1 = results[result_columns].sort_values(["Dataset", "Architecture", "Strategy"])
    table2 = results[results["strategy_key"].isin(["strategy1", "strategy3"])][result_columns].sort_values(
        ["Dataset", "Architecture", "Strategy"]
    )
    table1.to_csv(output_dir / "table1_all_fusion_strategies.csv", index=False)
    table2.to_csv(output_dir / "table2_image_only_vs_film_cv.csv", index=False)
    print(table1.to_string(index=False))
    print(f"\nSaved: {output_dir / 'table1_all_fusion_strategies.csv'}")
    print(f"Saved: {output_dir / 'table2_image_only_vs_film_cv.csv'}")


if __name__ == "__main__":
    main()
