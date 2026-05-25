from pathlib import Path
import pandas as pd
import numpy as np

root = Path("/mnt/afs/250010150/zhensheng/trimole")

gnn_path = root / "results/model_log/gnn_v2_22tasks/gnn_v2_22tasks_agg.csv"
main_path = root / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
out_path = root / "results/model_log/gnn_v2_22tasks/compare_gnn_v2_vs_main_fixed.csv"

gnn = pd.read_csv(gnn_path)
main = pd.read_csv(main_path)

task_col_main = "task" if "task" in main.columns else main.columns[0]

metric_candidates = [
    "primary_metric", "score", "value", "test_auc", "auc", "auroc", "metric", "performance"
]
main_metric_col = None
for c in metric_candidates:
    if c in main.columns:
        main_metric_col = c
        break
if main_metric_col is None:
    non_task_cols = [c for c in main.columns if c != task_col_main]
    if len(non_task_cols) == 1:
        main_metric_col = non_task_cols[0]
    else:
        raise ValueError(f"Cannot determine main score column: {list(main.columns)}")

main = main[[task_col_main, main_metric_col]].copy()
main.columns = ["task", "main_score"]

# 用 TDC 官方指标覆盖
official_metric = {
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "herg": "AUROC",
    "ames": "AUROC",
    "bbb_martins": "AUROC",
    "bioavailability_ma": "AUROC",
    "dili": "AUROC",
    "hia_hou": "AUROC",
    "pgp_broccatelli": "AUROC",
    "caco2_wang": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "solubility_aqsoldb": "MAE",
    "ld50_zhu": "MAE",
    "ppbr_az": "MAE",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "half_life_obach": "Spearman",
    "vdss_lombardo": "Spearman",
}

def pick_gnn_score(row):
    task = row["task"]
    metric = official_metric.get(task, row["primary_metric_name"])
    metric = str(metric).upper()

    if metric == "AUROC":
        return pd.to_numeric(row.get("test_auc_mean"), errors="coerce")
    if metric == "AUPRC":
        return pd.to_numeric(row.get("test_auprc_mean"), errors="coerce")
    if metric == "MAE":
        return pd.to_numeric(row.get("test_mae_mean"), errors="coerce")
    if metric == "SPEARMAN":
        return pd.to_numeric(row.get("test_spearman_mean"), errors="coerce")
    return pd.to_numeric(row.get("primary_metric_mean"), errors="coerce")

def direction(metric_name: str) -> str:
    m = str(metric_name).upper()
    if m in {"MAE", "RMSE", "MSE"}:
        return "lower"
    return "higher"

cmp = gnn.merge(main, on="task", how="left")
cmp["official_metric_name"] = cmp["task"].map(lambda t: official_metric.get(t, "UNKNOWN"))
cmp["gnn_score_for_compare"] = cmp.apply(pick_gnn_score, axis=1)
cmp["metric_direction"] = cmp["official_metric_name"].map(direction)

def calc_delta(row):
    g = pd.to_numeric(row["gnn_score_for_compare"], errors="coerce")
    m = pd.to_numeric(row["main_score"], errors="coerce")
    if pd.isna(g) or pd.isna(m):
        return np.nan
    if row["metric_direction"] == "lower":
        return m - g
    return g - m

cmp["delta_vs_main"] = cmp.apply(calc_delta, axis=1)
cmp["gnn_beats_main"] = cmp["delta_vs_main"] > 0

# std 也按对应指标取
def pick_std(row):
    metric = str(row["official_metric_name"]).upper()
    if metric == "AUROC":
        return pd.to_numeric(row.get("test_auc_std"), errors="coerce")
    if metric == "AUPRC":
        return pd.to_numeric(row.get("test_auprc_std"), errors="coerce")
    if metric == "MAE":
        return pd.to_numeric(row.get("test_mae_std"), errors="coerce")
    if metric == "SPEARMAN":
        return pd.to_numeric(row.get("test_spearman_std"), errors="coerce")
    return pd.to_numeric(row.get("primary_metric_std"), errors="coerce")

cmp["compare_std"] = cmp.apply(pick_std, axis=1)
cmp["std_safe_win"] = cmp["delta_vs_main"] > cmp["compare_std"].fillna(np.inf)

def recommend(row):
    if pd.isna(row["delta_vs_main"]):
        return "unknown"
    if row["gnn_beats_main"] and row["std_safe_win"]:
        return "take_over"
    if row["gnn_beats_main"]:
        return "candidate"
    return "keep_main"

cmp["recommendation"] = cmp.apply(recommend, axis=1)

keep_cols = [
    "task",
    "task_type",
    "official_metric_name",
    "n_runs",
    "gnn_score_for_compare",
    "compare_std",
    "main_score",
    "delta_vs_main",
    "gnn_beats_main",
    "std_safe_win",
    "recommendation",
]

cmp = cmp[keep_cols].sort_values(["recommendation", "delta_vs_main"], ascending=[True, False])
cmp.to_csv(out_path, index=False)

print(cmp.to_string(index=False))
print("\nSaved:", out_path)

print("\n=== TAKE OVER TASKS ===")
sub = cmp[cmp["recommendation"] == "take_over"]
print(sub.to_string(index=False) if len(sub) else "None")

print("\n=== CANDIDATE TASKS ===")
sub = cmp[cmp["recommendation"] == "candidate"]
print(sub.to_string(index=False) if len(sub) else "None")
