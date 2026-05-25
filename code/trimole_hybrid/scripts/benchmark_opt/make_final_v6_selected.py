#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
import numpy as np

V5 = Path("results/model_log/final_best_of_all_22_v5_selected_with_config.csv")
ENS = Path("results/model_log/boost_round2_ensemble/ensemble_vs_v5.csv")

OUT1 = Path("results/model_log/final_best_of_all_22_v6_selected.csv")
OUT2 = Path("results/model_log/final_best_of_all_22_v6_selected_with_config.csv")
OUT3 = Path("results/model_log/final_best_v6_selected_summary.json")
OUT4 = Path("results/model_log/final_best_v6_selected_patch_only.csv")

if not V5.exists():
    raise FileNotFoundError(f"Missing: {V5}")
if not ENS.exists():
    raise FileNotFoundError(f"Missing: {ENS}")

v5 = pd.read_csv(V5)
ens = pd.read_csv(ENS)

merged = v5.copy()

selected = ens[ens["is_improved"] == True].copy()
selected = selected[selected["task"].isin(["bioavailability_ma", "cyp3a4_substrate_carbonmangels"])].copy()

patch_rows = []

for _, row in selected.iterrows():
    task = row["task"]
    idxs = merged.index[merged["task"] == task].tolist()
    if not idxs:
        continue
    idx = idxs[0]

    metric_name = row["metric_name"]
    old_val = row["baseline_v5_value"]
    new_val = row["ensemble_value"]

    patch_rows.append({
        "task": task,
        "task_type": row["task_type"],
        "metric_name": metric_name,
        "old_value": old_val,
        "new_value": new_val,
        "delta_new_minus_old": row["delta_new_minus_old"],
        "ensemble_member_configs": row["ensemble_member_configs"],
        "n_configs_used": row["n_configs_used"],
    })

    # 覆盖结果
    if row["task_type"] == "classification":
        if pd.notna(row.get("ensemble_test_auc", np.nan)):
            merged.at[idx, "test_auc"] = row["ensemble_test_auc"]
            merged.at[idx, "primary_metric"] = row["ensemble_test_auc"]
        if pd.notna(row.get("ensemble_test_auprc", np.nan)):
            merged.at[idx, "test_auprc"] = row["ensemble_test_auprc"]
        if pd.notna(row.get("ensemble_test_acc", np.nan)):
            merged.at[idx, "test_acc"] = row["ensemble_test_acc"]
        merged.at[idx, "loss_type"] = "Ensemble"
        merged.at[idx, "fusion_type"] = "ensemble"
    else:
        if pd.notna(row.get("ensemble_test_mae", np.nan)):
            merged.at[idx, "test_mae"] = row["ensemble_test_mae"]
            merged.at[idx, "primary_metric"] = row["ensemble_test_mae"]
        if pd.notna(row.get("ensemble_test_rmse", np.nan)):
            merged.at[idx, "test_rmse"] = row["ensemble_test_rmse"]
        if pd.notna(row.get("ensemble_test_spearman", np.nan)):
            merged.at[idx, "test_spearman"] = row["ensemble_test_spearman"]
        merged.at[idx, "loss_type"] = "Ensemble"
        merged.at[idx, "fusion_type"] = "ensemble"

    merged.at[idx, "rerun_results_file"] = f"ensemble::{row['ensemble_member_configs']}"
    merged.at[idx, "rerun_run_name"] = "boost_round2_ensemble"

patch_df = pd.DataFrame(patch_rows).sort_values("task").reset_index(drop=True)
merged = merged.sort_values(["task_type", "task"]).reset_index(drop=True)

merged.to_csv(OUT1, index=False)
merged.to_csv(OUT2, index=False)
patch_df.to_csv(OUT4, index=False)

summary = {
    "base_v5_file": str(V5),
    "ensemble_file": str(ENS),
    "output_v6_selected": str(OUT1),
    "n_tasks_total": int(len(merged)),
    "n_ensemble_patches_applied": int(len(patch_df)),
    "ensemble_tasks_applied": patch_df["task"].tolist(),
}

OUT3.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT1)
print(" -", OUT2)
print(" -", OUT3)
print(" -", OUT4)
print()
if len(patch_df):
    print("Applied ensemble patches:")
    print(patch_df.to_string(index=False))
else:
    print("No ensemble patches applied.")
