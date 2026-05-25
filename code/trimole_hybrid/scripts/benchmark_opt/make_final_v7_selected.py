#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_CANDIDATES = [
    Path("results/model_log/final_best_of_all_22_v6_selected_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_v5_selected_with_config.csv"),
]
PATCH = Path("results/model_log/bench_opt_targeted_v7/best_targeted_v7_by_task.csv")

base_file = None
for p in BASE_CANDIDATES:
    if p.exists():
        base_file = p
        break
if base_file is None:
    raise FileNotFoundError("No base final file found (v6_selected_with_config or v5_selected_with_config).")
if not PATCH.exists():
    raise FileNotFoundError(f"Missing: {PATCH}")

OUT1 = Path("results/model_log/final_best_of_all_22_v7_selected.csv")
OUT2 = Path("results/model_log/final_best_of_all_22_v7_selected_with_config.csv")
OUT3 = Path("results/model_log/final_best_v7_selected_summary.json")
OUT4 = Path("results/model_log/final_best_v7_selected_patch_only.csv")

base = pd.read_csv(base_file)
patch = pd.read_csv(PATCH)

def parse_cfg(cfg_name: str):
    out = {"fusion_type": None, "loss_type_cfg": None, "hidden_dim": None, "dropout_head": None}
    parts = str(cfg_name).split("_")
    if "mlp" in parts:
        out["fusion_type"] = "mlp"
    elif "gated" in parts:
        out["fusion_type"] = "gated"
    if "focal" in parts:
        out["loss_type_cfg"] = "focal"
    elif "auto" in parts:
        out["loss_type_cfg"] = "auto"
    for p in parts:
        if p.startswith("h") and p[1:].isdigit():
            out["hidden_dim"] = int(p[1:])
        if p.startswith("d") and p[1:].isdigit():
            num = p[1:]
            if len(num) == 3:
                out["dropout_head"] = float(num[0] + "." + num[1:])
            elif len(num) == 2:
                out["dropout_head"] = float(num[0] + "." + num[1])
    return out

merged = base.copy()
patch_rows = []

for _, row in patch.iterrows():
    task = row["task"]
    idxs = merged.index[merged["task"] == task].tolist()
    if not idxs:
        continue
    idx = idxs[0]
    task_type = row["task_type"]

    if task_type == "classification":
        old_val = merged.at[idx, "test_auc"] if "test_auc" in merged.columns else np.nan
        new_val = row.get("test_auc", np.nan)
        better = pd.notna(new_val) and (pd.isna(old_val) or float(new_val) > float(old_val))
        metric_name = "test_auc"
    else:
        old_val = merged.at[idx, "test_mae"] if "test_mae" in merged.columns else np.nan
        new_val = row.get("test_mae", np.nan)
        better = pd.notna(new_val) and (pd.isna(old_val) or float(new_val) < float(old_val))
        metric_name = "test_mae"

    cfg = parse_cfg(row["cfg_name"])
    patch_rows.append({
        "task": task,
        "task_type": task_type,
        "metric_name": metric_name,
        "old_value": old_val,
        "new_value": new_val,
        "delta_new_minus_old": (float(new_val) - float(old_val)) if pd.notna(old_val) and pd.notna(new_val) else np.nan,
        "is_selected": bool(better),
        "cfg_name": row["cfg_name"],
        "fusion_type": cfg["fusion_type"],
        "loss_type_cfg": cfg["loss_type_cfg"],
        "hidden_dim": cfg["hidden_dim"],
        "dropout_head": cfg["dropout_head"],
        "results_file": row["results_file"],
    })

    if better:
        for c in ["test_auc", "test_auprc", "test_acc", "test_mae", "test_rmse", "test_spearman", "loss_type"]:
            if c in row.index and pd.notna(row.get(c, np.nan)):
                merged.at[idx, c] = row[c]
        if "primary_metric" in merged.columns:
            merged.at[idx, "primary_metric"] = new_val
        merged.at[idx, "fusion_type"] = cfg["fusion_type"]
        if cfg["hidden_dim"] is not None:
            merged.at[idx, "hidden_dim"] = cfg["hidden_dim"]
        if cfg["dropout_head"] is not None:
            merged.at[idx, "dropout_head"] = cfg["dropout_head"]
        merged.at[idx, "rerun_results_file"] = row["results_file"]
        merged.at[idx, "rerun_run_name"] = row["run_name"]

patch_df = pd.DataFrame(patch_rows).sort_values(["is_selected", "task"], ascending=[False, True]).reset_index(drop=True)
selected_patch_df = patch_df[patch_df["is_selected"]].copy()

merged = merged.sort_values(["task_type", "task"]).reset_index(drop=True)
merged.to_csv(OUT1, index=False)
merged.to_csv(OUT2, index=False)
selected_patch_df.to_csv(OUT4, index=False)

summary = {
    "base_file": str(base_file),
    "patch_file": str(PATCH),
    "output_file": str(OUT1),
    "n_selected_improvements": int(len(selected_patch_df)),
    "selected_tasks": selected_patch_df["task"].tolist(),
}
OUT3.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT1)
print(" -", OUT2)
print(" -", OUT3)
print(" -", OUT4)
print()
print("Selected improvements:")
if len(selected_patch_df):
    print(selected_patch_df[["task", "metric_name", "old_value", "new_value", "delta_new_minus_old", "cfg_name"]].to_string(index=False))
else:
    print("None")
