import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# AUTO PATH (không sợ sai thư mục)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_CSV = os.path.join(BASE_DIR, "metrics_history_full_weighted_ham10k_final_enhanced.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metrics_full_weighted_ham10k_final_enhanced.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA (auto fix khoảng trắng)
# =========================
history_df = pd.read_csv(HISTORY_CSV, skipinitialspace=True)
test_df = pd.read_csv(TEST_CSV, skipinitialspace=True)

# Clean column names
history_df.columns = history_df.columns.str.strip().str.lower()
test_df.columns = test_df.columns.str.strip().str.lower()

# =========================
# HANDLE EPOCH COLUMN
# =========================
if "epoch" in history_df.columns:
    epochs = history_df["epoch"]
else:
    epochs = history_df.index + 1

plt.style.use("seaborn-v0_8-whitegrid")

# =========================
# PLOT FUNCTION
# =========================
def plot_metric(metric, ylabel):
    plt.figure(figsize=(12, 7))

    # Train line
    if f"train_{metric}" in history_df.columns:
        plt.plot(
            epochs,
            history_df[f"train_{metric}"],
            marker="o",
            markersize=7,
            linewidth=2.5,
            label=f"Train {metric.upper()}"
        )

    # Validation line
    if f"val_{metric}" in history_df.columns:
        plt.plot(
            epochs,
            history_df[f"val_{metric}"],
            marker="s",
            markersize=7,
            linewidth=2.5,
            label=f"Val {metric.upper()}"
        )

    # Test horizontal line
    if metric in test_df.columns:
        test_value = test_df[metric].values[0]
        plt.axhline(
            y=test_value,
            linestyle="--",
            linewidth=2.5,
            label=f"Test {metric.upper()} ({test_value:.4f})"
        )

    plt.title(f"{metric.upper()} Comparison", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_comparison.png"), dpi=400)
    plt.close()


# =========================
# GENERATE PLOTS
# =========================
plot_metric("loss", "Loss")
plot_metric("auc", "AUC")
plot_metric("acc", "Accuracy")
plot_metric("f1", "F1 Score")

print("✅ Done! All plots saved in folder:", OUTPUT_DIR)
