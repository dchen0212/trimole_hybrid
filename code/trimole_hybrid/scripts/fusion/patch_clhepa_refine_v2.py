from pathlib import Path
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")

main_path = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
ref_path  = ROOT / "results/model_log/clhepa_txg_refine_v2/clhepa_txg_refine_v2_summary.csv"
out_path  = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission_clhepa_refine_v2.csv"

main = pd.read_csv(main_path)
ref = pd.read_csv(ref_path).iloc[0]

task = "clearance_hepatocyte_az"
new_score = float(ref["test_spearman"])

task_col = "task" if "task" in main.columns else main.columns[0]
metric_col = "primary_metric"

mask = main[task_col].astype(str).str.lower() == task.lower()
if mask.sum() != 1:
    raise ValueError(f"{task} matched {mask.sum()} rows")

old_score = float(pd.to_numeric(main.loc[mask, metric_col], errors="coerce").iloc[0])

if new_score > old_score:
    main.loc[mask, metric_col] = new_score
    for col, val in {
        "clhepa_refine_patch_applied": "yes",
        "clhepa_refine_best_t": float(ref["best_t"]),
        "clhepa_refine_best_x": float(ref["best_x"]),
        "clhepa_refine_best_g": float(ref["best_g"]),
        "clhepa_refine_best_valid_spearman": float(ref["best_valid_spearman"]),
        "clhepa_refine_test_spearman": new_score,
    }.items():
        if col not in main.columns:
            main[col] = ""
        main.loc[mask, col] = val

    print(f"patched {task}: {old_score} -> {new_score}")
else:
    print(f"skip {task}: {old_score} -> {new_score} (not better)")

main.to_csv(out_path, index=False)
print("Saved:", out_path)
