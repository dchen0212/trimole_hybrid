#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
import numpy as np

V4 = Path("results/model_log/final_best_of_all_22_v4_rerun_with_config.csv")
V5T = Path("results/model_log/bench_opt_targeted_v5/best_targeted_v5_by_task.csv")

OUT1 = Path("results/model_log/final_best_of_all_22_v5_selected.csv")
OUT2 = Path("results/model_log/final_best_of_all_22_v5_selected_with_config.csv")
OUT3 = Path("results/model_log/final_best_v5_selected_summary.json")
OUT4 = Path("results/model_log/final_best_v5_selected_patch_only.csv")

if not V4.exists():
    raise FileNotFoundError(f"Missing: {V4}")
if not V5T.exists():
    raise FileNotFoundError(f"Missing: {V5T}")

v4 = pd.read_csv(V4)
v5t = pd.read_csv(V5T)

def infer_metric_col(df_row):
    task_type = df_row.get("task_type", None)
    if task_type == "classification":
        return "test_auc"
    return "test_mae"

def is_better(task_type, old_val, new_val):
    if pd.isna(new_val):
        return False
    if pd.isna(old_val):
        return True
    if task_type == "classification":
        return float(new_val) > float(old_val)
    return float(new_val) < float(old_val)

# 取 v4 的简洁配置列
v4_cols_keep = [
    c for c in [
        "task", "task_type", "primary_metric_name", "primary_metric", "best_valid_primary",
        "loss_type", "test_auc", "test_auprc", "test_acc", "test_mae", "test_rmse", "test_spearman",
        "best_epoch", "device", "seed", "rerun_results_file", "rerun_task_dir", "rerun_run_name",
        "fusion_type", "hidden_dim", "dropout_head"
    ] if c in v4.columns
]
base = v4[v4_cols_keep].copy()

# 给 targeted v5 解析配置
def parse_cfg(cfg_name: str):
    out = {
        "fusion_type": None,
        "loss_type_cfg": None,
        "hidden_dim": None,
        "dropout_head": None,
        "seed_cfg": None,
    }
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
        if p.startswith("seed") and p[4:].isdigit():
            out["seed_cfg"] = int(p[4:])
    return out

patch_rows = []
merged = base.copy()

for _, row in v5t.iterrows():
    task = row["task"]
    if task not in set(merged["task"]):
        continue

    task_type = row["task_type"]
    old = merged.loc[merged["task"] == task].iloc[0]

    if task_type == "classification":
        old_metric = old.get("test_auc", np.nan)
        new_metric = row.get("test_auc", np.nan)
        metric_name = "test_auc"
    else:
        old_metric = old.get("test_mae", np.nan)
        new_metric = row.get("test_mae", np.nan)
        metric_name = "test_mae"

    better = is_better(task_type, old_metric, new_metric)
    delta = None
    if pd.notna(old_metric) and pd.notna(new_metric):
        delta = float(new_metric) - float(old_metric)

    parsed = parse_cfg(row["cfg_name"])
    patch_rows.append({
        "task": task,
        "task_type": task_type,
        "metric_name": metric_name,
        "old_value": old_metric,
        "new_value": new_metric,
        "delta_new_minus_old": delta,
        "is_selected": bool(better),
        "cfg_name": row["cfg_name"],
        "results_file": row["results_file"],
        "fusion_type": parsed["fusion_type"],
        "loss_type_cfg": parsed["loss_type_cfg"],
        "hidden_dim": parsed["hidden_dim"],
        "dropout_head": parsed["dropout_head"],
        "seed_cfg": parsed["seed_cfg"],
        "test_auc": row.get("test_auc", np.nan),
        "test_auprc": row.get("test_auprc", np.nan),
        "test_mae": row.get("test_mae", np.nan),
        "test_spearman": row.get("test_spearman", np.nan),
        "model_selection_score": row.get("model_selection_score", np.nan),
    })

    if better:
        idx = merged.index[merged["task"] == task][0]
        # 用 targeted v5 覆盖关键结果列
        for c in ["test_auc", "test_auprc", "test_acc", "test_mae", "test_rmse", "test_spearman", "loss_type"]:
            if c in row.index and pd.notna(row.get(c, np.nan)):
                merged.at[idx, c] = row[c]
        merged.at[idx, "fusion_type"] = parsed["fusion_type"]
        if parsed["hidden_dim"] is not None:
            merged.at[idx, "hidden_dim"] = parsed["hidden_dim"]
        if parsed["dropout_head"] is not None:
            merged.at[idx, "dropout_head"] = parsed["dropout_head"]
        if parsed["seed_cfg"] is not None:
            merged.at[idx, "seed"] = parsed["seed_cfg"]
        merged.at[idx, "rerun_results_file"] = row["results_file"]
        merged.at[idx, "rerun_run_name"] = row["run_name"]
        merged.at[idx, "loss_type"] = row.get("loss_type", old.get("loss_type", np.nan))

patch_df = pd.DataFrame(patch_rows).sort_values(["is_selected", "task"], ascending=[False, True]).reset_index(drop=True)
selected_patch_df = patch_df[patch_df["is_selected"]].copy()

merged = merged.sort_values(["task_type", "task"]).reset_index(drop=True)

merged.to_csv(OUT1, index=False)
merged.to_csv(OUT2, index=False)
selected_patch_df.to_csv(OUT4, index=False)

summary = {
    "base_v4_file": str(V4),
    "targeted_v5_file": str(V5T),
    "output_v5_selected": str(OUT1),
    "output_patch_only": str(OUT4),
    "n_tasks_total": int(len(merged)),
    "n_targeted_tasks_checked": int(len(patch_df)),
    "n_selected_improvements": int(len(selected_patch_df)),
    "selected_tasks": selected_patch_df["task"].tolist(),
    "rejected_tasks": patch_df.loc[~patch_df["is_selected"], "task"].tolist(),
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
print()
print("Rejected targeted updates:")
rej = patch_df.loc[~patch_df["is_selected"], ["task", "metric_name", "old_value", "new_value", "delta_new_minus_old", "cfg_name"]]
if len(rej):
    print(rej.to_string(index=False))
else:
    print("None")
