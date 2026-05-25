#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASES = [
    Path("results/model_log/final_best_of_all_22_v6_selected_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_v5_selected_with_config.csv"),
]
PATCH = Path("results/model_log/gated_3d_downweight_all_cls/best_by_task.csv")

base_file = None
for p in BASES:
    if p.exists():
        base_file = p
        break
if base_file is None:
    raise FileNotFoundError("No v6/v5 final file found")
if not PATCH.exists():
    raise FileNotFoundError(f"Missing: {PATCH}")

base = pd.read_csv(base_file)
patch = pd.read_csv(PATCH)

OUT1 = Path("results/model_log/final_best_of_all_22_routeB_cls.csv")
OUT2 = Path("results/model_log/final_best_of_all_22_routeB_cls_with_config.csv")
OUT3 = Path("results/model_log/final_best_routeB_cls_patch_only.csv")
OUT4 = Path("results/model_log/final_best_routeB_cls_summary.json")

merged = base.copy()
patch_rows = []

for _, row in patch.iterrows():
    task = row["task"]
    idxs = merged.index[merged["task"] == task].tolist()
    if not idxs:
        continue
    idx = idxs[0]

    old_auc = merged.at[idx, "test_auc"] if "test_auc" in merged.columns else np.nan
    new_auc = row.get("test_auc", np.nan)
    better = pd.notna(new_auc) and (pd.isna(old_auc) or float(new_auc) > float(old_auc))

    patch_rows.append({
        "task": task,
        "old_auc": old_auc,
        "new_auc": new_auc,
        "delta_auc": float(new_auc - old_auc) if pd.notna(old_auc) and pd.notna(new_auc) else np.nan,
        "is_selected": bool(better),
        "cfg_name": row["cfg_name"],
        "results_file": row["results_file"],
    })

    if better:
        for c in ["primary_metric", "best_valid_primary", "test_auc", "test_auprc", "test_acc", "loss_type"]:
            if c in row.index and pd.notna(row.get(c, np.nan)):
                merged.at[idx, c] = row[c]
        if "primary_metric_name" in merged.columns:
            merged.at[idx, "primary_metric_name"] = "AUROC"
        merged.at[idx, "fusion_type"] = "gated_3d_downweight"
        merged.at[idx, "dropout_head"] = 0.2
        merged.at[idx, "hidden_dim"] = 128
        merged.at[idx, "rerun_results_file"] = row["results_file"]
        merged.at[idx, "rerun_run_name"] = row["run_name"]

patch_df = pd.DataFrame(patch_rows).sort_values(["is_selected", "task"], ascending=[False, True]).reset_index(drop=True)
selected = patch_df[patch_df["is_selected"]].copy()

merged = merged.sort_values(["task_type", "task"]).reset_index(drop=True)
merged.to_csv(OUT1, index=False)
merged.to_csv(OUT2, index=False)
selected.to_csv(OUT3, index=False)

summary = {
    "base_file": str(base_file),
    "patch_file": str(PATCH),
    "n_selected_improvements": int(len(selected)),
    "selected_tasks": selected["task"].tolist(),
    "output_file": str(OUT1),
}
OUT4.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT1)
print(" -", OUT2)
print(" -", OUT3)
print(" -", OUT4)
print()
print("Selected improvements:")
if len(selected):
    print(selected[["task", "old_auc", "new_auc", "delta_auc", "cfg_name"]].to_string(index=False))
else:
    print("None")
