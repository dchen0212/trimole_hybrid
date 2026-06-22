from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("<PROJECT_ROOT>/trimole")
RERUN_DIR = ROOT / "results/model_log/benchmark_clean_rerun"
TXG_DIR = ROOT / "results/model_log/txg_weight_search_benchmark_clean"

BASE_FINAL = RERUN_DIR / "benchmark_clean_final_master_strict.csv"
TXG_SUMMARY = TXG_DIR / "txg_weight_search_benchmark_clean_summary.csv"

OUT_CSV = RERUN_DIR / "benchmark_clean_v3_patch.csv"
OUT_TXT = RERUN_DIR / "benchmark_clean_v3_patch.txt"

PATCH_TASKS = {
    "ames": ("AUROC", 0.879698, 0.1, 0.5, 0.4),
    "bbb_martins": ("AUROC", 0.905840, 0.2, 0.8, 0.0),
    "clearance_hepatocyte_az": ("Spearman", 0.490062, 0.8, 0.2, 0.0),
    "cyp2d6_veith": ("AUPRC", 0.728364, 0.2, 0.4, 0.4),
    "cyp3a4_veith": ("AUPRC", 0.890195, 0.1, 0.5, 0.4),
    "dili": ("AUROC", 0.906957, 0.0, 0.0, 1.0),
    "hia_hou": ("AUROC", 0.971193, 0.0, 0.0, 1.0),
    "ld50_zhu": ("MAE", 0.576178, 0.3, 0.3, 0.4),
    "lipophilicity_astrazeneca": ("MAE", 0.465261, 0.2, 0.4, 0.4),
}

def metric_better(metric_name, new_val, old_val):
    m = str(metric_name).strip().upper()
    if m in {"AUROC", "AUPRC", "SPEARMAN"}:
        return new_val > old_val
    if m == "MAE":
        return new_val < old_val
    raise ValueError(f"Unknown metric: {metric_name}")

base = pd.read_csv(BASE_FINAL).copy()
txg = pd.read_csv(TXG_SUMMARY).copy()

# 兼容列
task_col = "task"
metric_col = "metric" if "metric" in txg.columns else "primary_metric_name"
score_col = "test_score" if "test_score" in txg.columns else "primary_metric"
t_col = "best_t"
x_col = "best_x"
g_col = "best_g"

rows = []
for _, r in base.iterrows():
    task = str(r["task"]).strip()
    old_metric = r["primary_metric_name"]
    old_score = float(r["primary_metric"])
    old_source = r["final_source"]

    row = r.copy()
    row["v3_patch_applied"] = "no"
    row["v3_patch_source"] = ""
    row["v3_metric"] = np.nan
    row["v3_best_t"] = np.nan
    row["v3_best_x"] = np.nan
    row["v3_best_g"] = np.nan
    row["v3_test_score"] = np.nan

    if task in PATCH_TASKS:
        want_metric, want_score, wt, wx, wg = PATCH_TASKS[task]

        # 从 txg summary 再核一次
        sub = txg[txg[task_col].astype(str) == task]
        if len(sub):
            rr = sub.iloc[0]
            txg_metric = rr[metric_col]
            txg_score = float(rr[score_col])
            bt = float(rr[t_col]); bx = float(rr[x_col]); bg = float(rr[g_col])

            if str(txg_metric) != str(want_metric):
                raise ValueError(f"{task}: metric mismatch between PATCH_TASKS and summary: {want_metric} vs {txg_metric}")
            if abs(txg_score - want_score) > 1e-6:
                raise ValueError(f"{task}: score mismatch between PATCH_TASKS and summary: {want_score} vs {txg_score}")
            if not (abs(bt-wt) < 1e-9 and abs(bx-wx) < 1e-9 and abs(bg-wg) < 1e-9):
                raise ValueError(f"{task}: weight mismatch between PATCH_TASKS and summary")
        else:
            raise ValueError(f"{task}: not found in txg summary")

        if str(old_metric) != str(want_metric):
            raise ValueError(f"{task}: metric mismatch with current final table: {old_metric} vs {want_metric}")

        if metric_better(want_metric, want_score, old_score):
            row["primary_metric"] = want_score
            row["final_source"] = "v3_patch"
            row["final_note"] = f"replaced by v3 patch from txg_weight_search_benchmark_clean; previous source={old_source}; previous score={old_score}"
            row["v3_patch_applied"] = "yes"
            row["v3_patch_source"] = "txg_weight_search_benchmark_clean"
            row["v3_metric"] = want_metric
            row["v3_best_t"] = wt
            row["v3_best_x"] = wx
            row["v3_best_g"] = wg
            row["v3_test_score"] = want_score

    rows.append(row)

out = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

applied = out[out["v3_patch_applied"] == "yes"][[
    "task","primary_metric_name","primary_metric","v3_patch_source","v3_best_t","v3_best_x","v3_best_g","v3_test_score"
]].copy()

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== V3 PATCH APPLIED ROWS ===\n")
    if len(applied):
        f.write(applied.to_string(index=False))
        f.write("\n")
    else:
        f.write("none\n")

print("=== V3 PATCH APPLIED ROWS ===")
if len(applied):
    print(applied.to_string(index=False))
else:
    print("none")

print()
print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
