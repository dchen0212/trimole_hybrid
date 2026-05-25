from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")

base_csv = ROOT / "results/model_log/benchmark_clean_rerun/benchmark_clean_txg_selective_patch.csv"
xg_csv = ROOT / "results/model_log/xg_weight_search_missing_trimole/xg_weight_search_missing_trimole_summary.csv"
refine_csv = ROOT / "results/model_log/txg_refine_v2_5tasks/txg_refine_v2_5tasks_summary.csv"
out_csv = ROOT / "results/model_log/benchmark_clean_rerun/benchmark_clean_v2_patch.csv"

df = pd.read_csv(base_csv)
xg = pd.read_csv(xg_csv)
refine = pd.read_csv(refine_csv)

# 需要从 X:G 搜索接管的任务
xg_take = {
    "pgp_broccatelli",
    "solubility_aqsoldb",
}

# 需要从 TXG refine 接管的任务
txg_take = {
    "ppbr_az",
    "cyp2c9_veith",
    "caco2_wang",
}

# cyp3a4_veith 提升太小，这里默认不接管；想接管就加进去
# txg_take.add("cyp3a4_veith")

def patch_row(task: str, metric: str, score: float, extra: dict):
    m = df["task"] == task
    if m.sum() != 1:
        print(f"skip {task}: expected 1 row, got {m.sum()}")
        return

    df.loc[m, "primary_metric_name"] = metric
    df.loc[m, "primary_metric"] = score

    if metric == "AUROC":
        df.loc[m, "test_auc"] = score
    elif metric == "AUPRC":
        df.loc[m, "test_auprc"] = score
    elif metric == "MAE":
        df.loc[m, "test_mae"] = score
    elif metric == "Spearman":
        df.loc[m, "test_spearman"] = score

    for k, v in extra.items():
        df.loc[m, k] = v

# 先补列
for col in [
    "v2_patch_applied",
    "v2_patch_source",
    "v2_metric",
    "v2_best_t", "v2_best_x", "v2_best_g",
    "v2_best_valid_score", "v2_test_score",
]:
    if col not in df.columns:
        df[col] = None
df["v2_patch_applied"] = df["v2_patch_applied"].fillna("no")

# 接管 X:G 任务
for _, r in xg.iterrows():
    task = r["task"]
    if task not in xg_take:
        continue
    metric = r["metric"]
    score = r["test_score"]
    patch_row(task, metric, score, {
        "v2_patch_applied": "yes",
        "v2_patch_source": "xg_missing_trimole",
        "v2_metric": metric,
        "v2_best_t": None,
        "v2_best_x": r["best_x"],
        "v2_best_g": r["best_g"],
        "v2_best_valid_score": r["best_valid_score"],
        "v2_test_score": r["test_score"],
    })

# 接管 TXG refine 任务
for _, r in refine.iterrows():
    task = r["task"]
    if task not in txg_take:
        continue
    metric = r["metric"]
    score = r["test_score"]
    patch_row(task, metric, score, {
        "v2_patch_applied": "yes",
        "v2_patch_source": "txg_refine_v2",
        "v2_metric": metric,
        "v2_best_t": r["best_t"],
        "v2_best_x": r["best_x"],
        "v2_best_g": r["best_g"],
        "v2_best_valid_score": r["best_valid_score"],
        "v2_test_score": r["test_score"],
    })

df.to_csv(out_csv, index=False)

print("=== V2 PATCHED ROWS ===")
print(df[df["v2_patch_applied"].astype(str) == "yes"][[
    "task","primary_metric_name","primary_metric",
    "v2_patch_source","v2_metric","v2_best_t","v2_best_x","v2_best_g","v2_test_score"
]].to_string(index=False))

print("\nSaved:", out_csv)
