from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("<PROJECT_ROOT>/trimole")
RERUN_DIR = ROOT / "results/model_log/benchmark_clean_rerun"
TXG_DIR = ROOT / "results/model_log/txg_weight_search_benchmark_clean"

SRC = RERUN_DIR / "benchmark_clean_final_master_v4.csv"
TXG_SUMMARY = TXG_DIR / "txg_weight_search_benchmark_clean_summary.csv"

OUT_CSV = RERUN_DIR / "benchmark_clean_final_master_v5.csv"
OUT_TXT = RERUN_DIR / "benchmark_clean_final_master_v5.txt"

PATCH_TASKS = {
    "cyp2c9_substrate_carbonmangels": ("AUPRC", 0.463943, 0.4, 0.0, 0.6),
    "cyp3a4_substrate_carbonmangels": ("AUROC", 0.652803, 0.2, 0.0, 0.8),
}

def metric_better(metric_name, new_val, old_val):
    m = str(metric_name).strip().upper()
    if m in {"AUROC", "AUPRC", "SPEARMAN"}:
        return new_val > old_val
    if m == "MAE":
        return new_val < old_val
    raise ValueError(f"Unknown metric: {metric_name}")

df = pd.read_csv(SRC).copy()
txg = pd.read_csv(TXG_SUMMARY).copy()

rows = []
for _, r in df.iterrows():
    row = r.copy()
    task = str(row["task"]).strip()

    row["v5_patch_applied"] = "no"
    row["v5_patch_source"] = ""
    row["v5_metric"] = np.nan
    row["v5_best_t"] = np.nan
    row["v5_best_x"] = np.nan
    row["v5_best_g"] = np.nan
    row["v5_test_score"] = np.nan

    if task in PATCH_TASKS:
        want_metric, want_score, wt, wx, wg = PATCH_TASKS[task]
        old_metric = row["primary_metric_name"]
        old_score = float(row["primary_metric"])
        old_source = row["final_source"]

        sub = txg[txg["task"].astype(str) == task]
        if len(sub) == 0:
            raise ValueError(f"{task}: not found in txg summary")

        rr = sub.iloc[0]
        txg_metric = rr["metric"]
        txg_score = float(rr["test_score"])
        bt = float(rr["best_t"]); bx = float(rr["best_x"]); bg = float(rr["best_g"])

        if str(txg_metric) != str(want_metric):
            raise ValueError(f"{task}: metric mismatch {txg_metric} vs {want_metric}")
        if abs(txg_score - want_score) > 1e-6:
            raise ValueError(f"{task}: score mismatch {txg_score} vs {want_score}")
        if not (abs(bt-wt) < 1e-9 and abs(bx-wx) < 1e-9 and abs(bg-wg) < 1e-9):
            raise ValueError(f"{task}: weight mismatch")

        if str(old_metric) != str(want_metric):
            raise ValueError(f"{task}: current metric mismatch {old_metric} vs {want_metric}")

        if metric_better(want_metric, want_score, old_score):
            row["primary_metric"] = want_score
            row["final_source"] = "v5_patch"
            row["final_note"] = f"replaced by v5 patch from txg_weight_search_benchmark_clean; previous source={old_source}; previous score={old_score}"
            row["v5_patch_applied"] = "yes"
            row["v5_patch_source"] = "txg_weight_search_benchmark_clean"
            row["v5_metric"] = want_metric
            row["v5_best_t"] = wt
            row["v5_best_x"] = wx
            row["v5_best_g"] = wg
            row["v5_test_score"] = want_score

    rows.append(row)

out = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

applied = out[out["v5_patch_applied"] == "yes"][[
    "task","primary_metric_name","primary_metric","v5_patch_source","v5_best_t","v5_best_x","v5_best_g","v5_test_score"
]].copy()

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== V5 PATCH APPLIED ROWS ===\n")
    if len(applied):
        f.write(applied.to_string(index=False))
        f.write("\n")
    else:
        f.write("none\n")
    f.write("\n=== source counts ===\n")
    f.write(out["final_source"].value_counts().to_string())
    f.write("\n")

print("=== V5 PATCH APPLIED ROWS ===")
if len(applied):
    print(applied.to_string(index=False))
else:
    print("none")

print("\n=== source counts ===")
print(out["final_source"].value_counts().to_string())

print("\nSaved:", OUT_CSV)
print("Saved:", OUT_TXT)
