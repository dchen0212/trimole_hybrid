from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")

base_csv = ROOT / "results/model_log/benchmark_clean_rerun/run_20260415_2016/results_official_metrics.csv"
search_csv = ROOT / "results/model_log/txg_weight_search_benchmark_clean/txg_weight_search_benchmark_clean_summary.csv"
out_csv = ROOT / "results/model_log/benchmark_clean_rerun/benchmark_clean_txg_selective_patch.csv"

# 只 patch 这 5 个
SELECTIVE = {
    "clearance_microsome_az",
    "ppbr_az",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "caco2_wang",
}

base = pd.read_csv(base_csv)
search = pd.read_csv(search_csv)

search = search[search["task"].isin(SELECTIVE)].copy()

metric_col_map = {
    "AUROC": "test_auc",
    "AUPRC": "test_auprc",
    "MAE": "test_mae",
    "Spearman": "test_spearman",
}

# 从 benchmark_clean rerun 的 results_all.csv 取原始全列
results_all = ROOT / "results/model_log/benchmark_clean_rerun/run_20260415_2016/results_all.csv"
full = pd.read_csv(results_all)

df = full.copy()
df["txg_patch_applied"] = "no"
df["txg_metric"] = None
df["txg_best_t"] = None
df["txg_best_x"] = None
df["txg_best_g"] = None
df["txg_best_valid_score"] = None
df["txg_test_score"] = None

for _, r in search.iterrows():
    task = r["task"]
    metric = r["metric"]
    test_score = r["test_score"]

    m = df["task"] == task
    if m.sum() != 1:
        print(f"skip {task}: expected 1 row, got {m.sum()}")
        continue

    # official metric 列
    if metric == "AUROC":
        df.loc[m, "primary_metric_name"] = "AUROC"
        df.loc[m, "primary_metric"] = test_score
        df.loc[m, "test_auc"] = test_score
    elif metric == "AUPRC":
        df.loc[m, "primary_metric_name"] = "AUPRC"
        df.loc[m, "primary_metric"] = test_score
        df.loc[m, "test_auprc"] = test_score
    elif metric == "MAE":
        df.loc[m, "primary_metric_name"] = "MAE"
        df.loc[m, "primary_metric"] = test_score
        df.loc[m, "test_mae"] = test_score
    elif metric == "Spearman":
        df.loc[m, "primary_metric_name"] = "Spearman"
        df.loc[m, "primary_metric"] = test_score
        df.loc[m, "test_spearman"] = test_score
    else:
        raise ValueError(metric)

    df.loc[m, "txg_patch_applied"] = "yes"
    df.loc[m, "txg_metric"] = metric
    df.loc[m, "txg_best_t"] = r["best_t"]
    df.loc[m, "txg_best_x"] = r["best_x"]
    df.loc[m, "txg_best_g"] = r["best_g"]
    df.loc[m, "txg_best_valid_score"] = r["best_valid_score"]
    df.loc[m, "txg_test_score"] = r["test_score"]

df.to_csv(out_csv, index=False)

print("=== PATCHED ROWS ===")
print(df[df["txg_patch_applied"] == "yes"][[
    "task","primary_metric_name","primary_metric",
    "txg_metric","txg_best_t","txg_best_x","txg_best_g",
    "txg_best_valid_score","txg_test_score"
]].to_string(index=False))

print("\nSaved:", out_csv)
