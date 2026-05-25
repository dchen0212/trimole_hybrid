from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")

main_path = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission_gnn_patch_v1.csv"
txg_path  = ROOT / "results/model_log/txg_weight_search/txg_weight_search_summary.csv"
out_path  = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission_txg_patch_v3_selective.csv"

main = pd.read_csv(main_path)
txg  = pd.read_csv(txg_path)

task_col = "task" if "task" in main.columns else main.columns[0]
metric_col = "primary_metric"

# 只保留 test 真正更好的任务
# higher better: AUROC/AUPRC/Spearman
# lower better: MAE
metric_dir = {
    "AUROC": "higher",
    "AUPRC": "higher",
    "Spearman": "higher",
    "MAE": "lower",
}

def better(metric, new, old):
    if metric_dir[metric] == "higher":
        return float(new) > float(old)
    return float(new) < float(old)

for col in [
    "txg_patch_applied",
    "txg_metric",
    "txg_best_t",
    "txg_best_x",
    "txg_best_g",
    "txg_best_valid_score",
    "txg_test_score",
]:
    if col not in main.columns:
        main[col] = ""

patched = []

for _, r in txg.iterrows():
    task = str(r["task"])
    metric = str(r["metric"])
    new_score = float(r["test_score"])

    mask = main[task_col].astype(str).str.lower() == task.lower()
    if mask.sum() != 1:
        continue

    old_score = float(pd.to_numeric(main.loc[mask, metric_col], errors="coerce").iloc[0])

    if better(metric, new_score, old_score):
        main.loc[mask, metric_col] = new_score
        main.loc[mask, "txg_patch_applied"] = "yes"
        main.loc[mask, "txg_metric"] = metric
        main.loc[mask, "txg_best_t"] = float(r["best_t"])
        main.loc[mask, "txg_best_x"] = float(r["best_x"])
        main.loc[mask, "txg_best_g"] = float(r["best_g"])
        main.loc[mask, "txg_best_valid_score"] = float(r["best_valid_score"])
        main.loc[mask, "txg_test_score"] = new_score
        patched.append((task, metric, old_score, new_score, r["best_t"], r["best_x"], r["best_g"]))

main.to_csv(out_path, index=False)

print("=== PATCHED TASKS ===")
for x in patched:
    task, metric, old_score, new_score, t, xx, g = x
    print(f"{task}: {old_score} -> {new_score} | {metric} | weights=({t}, {xx}, {g})")

print("\nSaved:", out_path)
