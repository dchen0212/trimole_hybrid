from __future__ import annotations

from pathlib import Path
import pandas as pd

root = Path("/mnt/afs/250010150/zhensheng/trimole/results/model_log/gnn_v2_22tasks")
rows = []

for f in root.rglob("results_all.csv"):
    try:
        # .../<task>/seed_<seed>/run_xxx/results_all.csv
        task = f.parent.parent.parent.name
        seed = int(f.parent.parent.name.replace("seed_", ""))
        df = pd.read_csv(f)
        row = df.iloc[0].to_dict()
        row["task"] = task
        row["seed"] = seed
        rows.append(row)
    except Exception as e:
        print("skip:", f, e)

if not rows:
    raise SystemExit("No valid results parsed.")

raw = pd.DataFrame(rows)

metric_cols = [
    "primary_metric",
    "best_valid_primary",
    "test_auc",
    "test_auprc",
    "test_acc",
    "test_mae",
    "test_rmse",
    "test_spearman",
]
for c in metric_cols:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

agg = (
    raw.groupby(["task", "task_type", "primary_metric_name"], as_index=False)
       .agg(
           n_runs=("seed", "count"),
           primary_metric_mean=("primary_metric", "mean"),
           primary_metric_std=("primary_metric", "std"),
           primary_metric_best=("primary_metric", "max"),
           best_valid_primary_mean=("best_valid_primary", "mean"),
           test_auc_mean=("test_auc", "mean"),
           test_auc_std=("test_auc", "std"),
           test_auprc_mean=("test_auprc", "mean"),
           test_auprc_std=("test_auprc", "std"),
           test_acc_mean=("test_acc", "mean"),
           test_acc_std=("test_acc", "std"),
           test_mae_mean=("test_mae", "mean"),
           test_mae_std=("test_mae", "std"),
           test_rmse_mean=("test_rmse", "mean"),
           test_rmse_std=("test_rmse", "std"),
           test_spearman_mean=("test_spearman", "mean"),
           test_spearman_std=("test_spearman", "std"),
       )
       .sort_values("task")
)

raw_out = root / "gnn_v2_22tasks_raw.csv"
agg_out = root / "gnn_v2_22tasks_agg.csv"

raw.to_csv(raw_out, index=False)
agg.to_csv(agg_out, index=False)

print(agg.to_string(index=False))
print("\nSaved:", agg_out)
print("Saved:", raw_out)
