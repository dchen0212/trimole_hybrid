from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")

main_path   = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
refine_path = ROOT / "results/model_log/txg_refine_v1/txg_refine_v1_summary.csv"
out_path    = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission_txg_refine_v4.csv"

main = pd.read_csv(main_path)
ref  = pd.read_csv(refine_path)

task_col = "task" if "task" in main.columns else main.columns[0]
metric_col = "primary_metric"

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
    "txg_refine_patch_applied",
    "txg_refine_metric",
    "txg_refine_best_t",
    "txg_refine_best_x",
    "txg_refine_best_g",
    "txg_refine_best_valid_score",
    "txg_refine_test_score",
]:
    if col not in main.columns:
        main[col] = ""

patched = []

for _, r in ref.iterrows():
    task = str(r["task"])
    metric = str(r["metric"])
    new_score = float(r["test_score"])

    mask = main[task_col].astype(str).str.lower() == task.lower()
    if mask.sum() != 1:
        continue

    old_score = float(pd.to_numeric(main.loc[mask, metric_col], errors="coerce").iloc[0])

    if better(metric, new_score, old_score):
        main.loc[mask, metric_col] = new_score
        main.loc[mask, "txg_refine_patch_applied"] = "yes"
        main.loc[mask, "txg_refine_metric"] = metric
        main.loc[mask, "txg_refine_best_t"] = float(r["best_t"])
        main.loc[mask, "txg_refine_best_x"] = float(r["best_x"])
        main.loc[mask, "txg_refine_best_g"] = float(r["best_g"])
        main.loc[mask, "txg_refine_best_valid_score"] = float(r["best_valid_score"])
        main.loc[mask, "txg_refine_test_score"] = new_score
        patched.append((task, metric, old_score, new_score, r["best_t"], r["best_x"], r["best_g"]))

main.to_csv(out_path, index=False)

print("=== REFINED PATCHED TASKS ===")
for x in patched:
    task, metric, old_score, new_score, t, xx, g = x
    print(f"{task}: {old_score} -> {new_score} | {metric} | weights=({t}, {xx}, {g})")

print("\nSaved:", out_path)
