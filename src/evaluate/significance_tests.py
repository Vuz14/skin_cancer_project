"""
Run paired significance tests on 5-fold CV metrics.

This script is intentionally standalone and does not require retraining.
It reads saved CV result CSV files, extracts fold-wise AUC values, and runs
paired t-tests between strategies evaluated on the same folds.

Examples:
    python src/evaluate/significance_tests.py
    python src/evaluate/significance_tests.py --metric auc --alternative two-sided
    python src/evaluate/significance_tests.py --checkpoint-roots checkpoint_bcn20000 checkpoint_ham10000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_CHECKPOINT_ROOTS = ("checkpoint_bcn20000", "checkpoint_ham10000")


@dataclass(frozen=True)
class StrategyMetrics:
    dataset: str
    backbone: str
    strategy: str
    path: Path
    values: pd.Series


def parse_run_name(cv_dir: Path) -> tuple[str, str]:
    name = cv_dir.name
    if name.startswith("CV5_strategy"):
        parts = name.split("_")
        return parts[1], "_".join(parts[2:-1])
    raise ValueError(f"Unsupported legacy run directory for synchronized analysis: {cv_dir}")


def find_detail_csv(cv_dir: Path) -> Path | None:
    detail_files = sorted(cv_dir.glob("*detail_results.csv")) + sorted(cv_dir.glob("cv_validation_summary_detail.csv"))
    if detail_files:
        return detail_files[0]
    fold_files = sorted(cv_dir.glob("fold_*/test_metrics_fold_*.csv"))
    if fold_files:
        return fold_files[0]
    return None


def read_fold_metric(cv_dir: Path, metric: str) -> pd.Series:
    detail_csv = find_detail_csv(cv_dir)
    if detail_csv is None:
        raise FileNotFoundError(f"No detail or fold metrics CSV found in {cv_dir}")

    if "detail_results" in detail_csv.name:
        df = pd.read_csv(detail_csv)
    else:
        frames = []
        for fold_file in sorted(cv_dir.glob("fold_*/test_metrics_fold_*.csv")):
            fold_df = pd.read_csv(fold_file)
            fold_number = int(fold_file.parent.name.split("_")[-1])
            fold_df["fold"] = fold_number
            frames.append(fold_df)
        df = pd.concat(frames, ignore_index=True)

    metric_columns = {column.lower(): column for column in df.columns}
    metric_key = metric.lower()
    if metric_key not in metric_columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Metric '{metric}' not found in {detail_csv}. Available columns: {available}")

    if "fold" not in df.columns:
        df["fold"] = np.arange(1, len(df) + 1)

    fold_metric = (
        df[["fold", metric_columns[metric_key]]]
        .dropna()
        .groupby("fold", as_index=True)
        .mean()
        .iloc[:, 0]
        .sort_index()
    )
    fold_metric.name = metric_key
    return fold_metric


def load_strategy_metrics(checkpoint_root: Path, metric: str) -> list[StrategyMetrics]:
    if not checkpoint_root.exists():
        return []

    dataset = checkpoint_root.name.replace("checkpoint_", "")
    strategies: list[StrategyMetrics] = []
    for cv_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("CV5_strategy")):
        strategy, backbone = parse_run_name(cv_dir)
        values = read_fold_metric(cv_dir, metric)
        strategies.append(
            StrategyMetrics(
                dataset=dataset,
                backbone=backbone,
                strategy=strategy,
                path=cv_dir,
                values=values,
            )
        )
    return strategies


def paired_ttest(
    first: StrategyMetrics,
    second: StrategyMetrics,
    alternative: str,
    confidence: float,
) -> dict[str, object]:
    paired = pd.concat([first.values, second.values], axis=1, join="inner")
    paired.columns = [first.strategy, second.strategy]
    if len(paired) < 2:
        raise ValueError(f"Need at least 2 paired folds for {first.strategy} vs {second.strategy}")

    diff = paired[second.strategy].to_numpy(dtype=float) - paired[first.strategy].to_numpy(dtype=float)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = sd_diff / np.sqrt(n)

    test = stats.ttest_rel(
        paired[second.strategy],
        paired[first.strategy],
        alternative=alternative,
    )
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff
    cohen_dz = mean_diff / sd_diff if sd_diff > 0 else np.nan

    return {
        "dataset": first.dataset,
        "backbone": first.backbone,
        "metric": first.values.name,
        "baseline_strategy": first.strategy,
        "comparison_strategy": second.strategy,
        "n_paired_folds": n,
        "baseline_values": ";".join(f"{value:.6f}" for value in paired[first.strategy]),
        "comparison_values": ";".join(f"{value:.6f}" for value in paired[second.strategy]),
        "baseline_mean": float(paired[first.strategy].mean()),
        "comparison_mean": float(paired[second.strategy].mean()),
        "mean_diff": mean_diff,
        "mean_diff_percent_points": mean_diff * 100.0,
        f"ci{int(confidence * 100)}_low": ci_low,
        f"ci{int(confidence * 100)}_high": ci_high,
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "cohen_dz": float(cohen_dz),
        "alternative": alternative,
        "baseline_path": str(first.path),
        "comparison_path": str(second.path),
    }


def run_tests(
    checkpoint_roots: list[Path],
    metric: str,
    alternative: str,
    confidence: float,
) -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []
    for checkpoint_root in checkpoint_roots:
        strategies = load_strategy_metrics(checkpoint_root, metric)
        by_dataset: dict[tuple[str, str], list[StrategyMetrics]] = {}
        for strategy in strategies:
            by_dataset.setdefault((strategy.dataset, strategy.backbone), []).append(strategy)

        for (dataset, backbone), dataset_strategies in by_dataset.items():
            if len(dataset_strategies) < 2:
                print(f"[WARN] Skipping {dataset}/{backbone}: need at least 2 strategies, found {len(dataset_strategies)}")
                continue
            for first, second in combinations(dataset_strategies, 2):
                all_rows.append(paired_ttest(first, second, alternative, confidence))

    if not all_rows:
        raise RuntimeError("No paired tests were produced. Check checkpoint paths and metric names.")
    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired t-tests for fold-wise CV metrics.")
    parser.add_argument(
        "--checkpoint-roots",
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_ROOTS),
        help="Checkpoint root folders containing CV5_* strategy directories.",
    )
    parser.add_argument("--metric", default="auc", help="Metric column to test, e.g. auc, acc, f1.")
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help=(
            "Paired t-test alternative. With default pair ordering, 'greater' tests whether "
            "comparison_strategy > baseline_strategy."
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for mean difference CI.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output CSV report.",
    )
    args = parser.parse_args()

    checkpoint_roots = [Path(path) for path in args.checkpoint_roots]
    results = run_tests(
        checkpoint_roots=checkpoint_roots,
        metric=args.metric,
        alternative=args.alternative,
        confidence=args.confidence,
    )

    output_path = Path(args.output) if args.output else Path(f"results/significance_tests_{args.metric.lower()}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    display_cols = [
        "dataset",
        "backbone",
        "baseline_strategy",
        "comparison_strategy",
        "baseline_mean",
        "comparison_mean",
        "mean_diff_percent_points",
        "t_statistic",
        "p_value",
        "cohen_dz",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved detailed report to: {output_path}")


if __name__ == "__main__":
    main()
