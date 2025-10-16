import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_results")
os.makedirs(OUT_DIR, exist_ok=True)

print("\n📊 Starting analysis for all PINN runs...\n")

summary_records = []

# ----------------------------------------------------------------------
# Loop through all results.npz files
# ----------------------------------------------------------------------
for root, _, files in os.walk(RUNS_DIR):
    for file in files:
        if not file.endswith("results.npz"):
            continue

        fpath = os.path.join(root, file)
        label = root.replace(RUNS_DIR + os.sep, "").replace("\\", " / ")

        try:
            data = np.load(fpath)
            X = data["X"]
            y_true = data["y_true"]
            y_pred = data["y_pred"]

            # Compute error metrics
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            summary_records.append({
                "Dataset": label,
                "MAE": mae,
                "RMSE": rmse,
                "Samples": len(X)
            })

            # Plot true vs predicted (for one output, e.g. temperature)
            plt.figure(figsize=(5, 4))
            plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, s=10)
            plt.xlabel("True Panel Temperature")
            plt.ylabel("Predicted Panel Temperature")
            plt.title(f"{label}\nTrue vs Predicted")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(root, "comparison_plot.png"))
            plt.close()

        except Exception as e:
            print(f"⚠️  Skipping {label}: {e}")

# ----------------------------------------------------------------------
# Save summary CSV
# ----------------------------------------------------------------------
df_summary = pd.DataFrame(summary_records)
csv_path = os.path.join(OUT_DIR, "pinn_summary_metrics.csv")
df_summary.to_csv(csv_path, index=False)
print(f"\n✅ Saved summary metrics → {csv_path}")

# ----------------------------------------------------------------------
# Overall comparison plot
# ----------------------------------------------------------------------
if not df_summary.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(df_summary["Dataset"], df_summary["RMSE"], color="teal")
    plt.xlabel("RMSE (Lower = Better)")
    plt.title("PINN Performance Across All Datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rmse_comparison.png"))
    plt.close()
    print("✅ Saved RMSE comparison bar chart → rmse_comparison.png")

print("\n🎉 Analysis complete! Check 'analysis_results/' for outputs.\n")
