from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")

main_path = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
txg_path  = ROOT / "results/model_log/txg_weight_search/txg_weight_search_summary.csv"
out_path  = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission_txg_patch_v2.csv"

main = pd.read_csv(main_path)
txg  = pd.read_csv(txg_path)

task_col = "task" if "task" in main.columns else main.columns[0]
metric_col = "primary_metric" if "primary_metric" in main.columns else None
if metric_col is None:
    raise ValueError(f"main file missing primary_metric: {main.columns.tolist()}")

# 只 patch 这轮实际扫过的任务
patch_tasks = set(txg["task"].astype(str).tolist())

# 给主表加一些说明列
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

for _, r in txg.iterrows():
    task = str(r["task"])
    mask = main[task_col].astype(str).str.lower() == task.lower()
    if mask.sum() != 1:
        raise ValueError(f"{task} matched {mask.sum()} rows in main")

    old_score = pd.to_numeric(main.loc[mask, metric_col], errors="coerce").iloc[0]
    new_score = float(r["test_score"])

    main.loc[mask, metric_col] = new_score
    main.loc[mask, "txg_patch_applied"] = "yes"
    main.loc[mask, "txg_metric"] = str(r["metric"])
    main.loc[mask, "txg_best_t"] = float(r["best_t"])
    main.loc[mask, "txg_best_x"] = float(r["best_x"])
    main.loc[mask, "txg_best_g"] = float(r["best_g"])
    main.loc[mask, "txg_best_valid_score"] = float(r["best_valid_score"])
    main.loc[mask, "txg_test_score"] = new_score

    print(f"{task}: {old_score} -> {new_score} | weights=({r['best_t']}, {r['best_x']}, {r['best_g']})")

main.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print("\nPatched rows:")
print(
    main[main["txg_patch_applied"].astype(str) == "yes"][
        [task_col, metric_col, "txg_metric", "txg_best_t", "txg_best_x", "txg_best_g", "txg_best_valid_score", "txg_test_score"]
    ].to_string(index=False)
)
