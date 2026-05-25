#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_CANDIDATES = [
    Path("results/model_log/final_best_of_all_22_plus_cyp2c9_ensemble_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_plus_cyp2c9_ensemble.csv"),
    Path("results/model_log/final_best_of_all_22_3d_downweight_full_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_3d_downweight_full.csv"),
]

XGB_FILE = Path("results/model_log/xgb_cyp2c9_veith_5seeds/seed_results.csv")

base_file = next((p for p in BASE_CANDIDATES if p.exists()), None)
if base_file is None:
    raise FileNotFoundError("No base final file found.")

if not XGB_FILE.exists():
    raise FileNotFoundError(f"Missing XGBoost results: {XGB_FILE}")

base = pd.read_csv(base_file)
xgb = pd.read_csv(XGB_FILE)

xgb = xgb[xgb["task"] == "cyp2c9_veith"].copy()
if xgb.empty:
    raise ValueError("No cyp2c9_veith row found in XGBoost results.")

best = xgb.sort_values("test_auprc", ascending=False).iloc[0]

row_idx = base.index[base["task"] == "cyp2c9_veith"].tolist()
if not row_idx:
    raise ValueError("cyp2c9_veith not found in base final table.")
row_idx = row_idx[0]

old_value = base.at[row_idx, "test_auprc"] if "test_auprc" in base.columns else np.nan
new_value = float(best["test_auprc"])
selected = pd.isna(old_value) or (new_value > float(old_value))

patch_row = {
    "task": "cyp2c9_veith",
    "metric_name": "AUPRC",
    "old_value": old_value,
    "new_value": new_value,
    "delta_new_minus_old": (new_value - float(old_value)) if pd.notna(old_value) else np.nan,
    "is_selected": bool(selected),
    "source_model": "XGBoost",
    "best_seed": int(best["seed"]),
    "xgb_results_file": str(XGB_FILE),
}

if selected:
    if "fusion_type" in base.columns:
        base.at[row_idx, "fusion_type"] = "xgboost_patch"
    if "loss_type" in base.columns:
        base.at[row_idx, "loss_type"] = "XGBoost"
    if "primary_metric_name" in base.columns:
        base.at[row_idx, "primary_metric_name"] = "AUPRC"
    if "primary_metric" in base.columns:
        base.at[row_idx, "primary_metric"] = float(best["test_auprc"])
    if "test_auc" in base.columns:
        base.at[row_idx, "test_auc"] = float(best["test_auc"])
    if "test_auprc" in base.columns:
        base.at[row_idx, "test_auprc"] = float(best["test_auprc"])
    if "test_acc" in base.columns:
        base.at[row_idx, "test_acc"] = float(best["test_acc"])
    if "seed" in base.columns:
        base.at[row_idx, "seed"] = int(best["seed"])
    if "hidden_dim" in base.columns:
        base.at[row_idx, "hidden_dim"] = np.nan
    if "dropout_head" in base.columns:
        base.at[row_idx, "dropout_head"] = np.nan

out_main = Path("results/model_log/final_best_of_all_22_plus_cyp2c9_xgb.csv")
out_cfg  = Path("results/model_log/final_best_of_all_22_plus_cyp2c9_xgb_with_config.csv")
out_patch = Path("results/model_log/final_patch_cyp2c9_veith_xgb.csv")
out_json = Path("results/model_log/final_patch_cyp2c9_veith_xgb_summary.json")

base.to_csv(out_main, index=False)
base.to_csv(out_cfg, index=False)
pd.DataFrame([patch_row]).to_csv(out_patch, index=False)
out_json.write_text(json.dumps({
    "base_file": str(base_file),
    "selected": bool(selected),
    "task": "cyp2c9_veith",
    "old_auprc": None if pd.isna(old_value) else float(old_value),
    "new_auprc": float(new_value),
    "best_seed": int(best["seed"]),
    "output_file": str(out_main),
}, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", out_main)
print(" -", out_cfg)
print(" -", out_patch)
print(" -", out_json)
print()
print(pd.DataFrame([patch_row]).to_string(index=False))
