#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_CANDIDATES = [
    Path("results/model_log/final_best_of_all_22_3d_downweight_full_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_3d_downweight_full.csv"),
    Path("results/model_log/final_best_of_all_22_routeB_cls_with_config.csv"),
    Path("results/model_log/final_best_of_all_22_routeB_cls.csv"),
]

ENSEMBLE_CANDIDATES = [
    Path("results/model_log/ensemble_trimole_maplight_5tasks_equal/results_all.csv"),
    Path("results/model_log/ensemble_trimole_maplight_5tasks_73/results_all.csv"),
    Path("results/model_log/ensemble_trimole_maplight_5tasks_82/results_all.csv"),
]

base_file = next((p for p in BASE_CANDIDATES if p.exists()), None)
if base_file is None:
    raise FileNotFoundError("No base final file found.")

ens_frames = []
for p in ENSEMBLE_CANDIDATES:
    if p.exists():
        df = pd.read_csv(p)
        df["ensemble_source"] = str(p)
        ens_frames.append(df)

if not ens_frames:
    raise FileNotFoundError("No ensemble result files found.")

ens = pd.concat(ens_frames, ignore_index=True)
ens = ens[ens["task"] == "cyp2c9_veith"].copy()
if ens.empty:
    raise ValueError("No cyp2c9_veith row found in ensemble results.")

# 官方指标是 AUPRC，所以按 primary_metric 选最优
best = ens.sort_values("primary_metric", ascending=False).iloc[0]

base = pd.read_csv(base_file)
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
    "weight_trimole": float(best["weight_trimole"]),
    "weight_maplight": float(best["weight_maplight"]),
    "ensemble_source": best["ensemble_source"],
}

if selected:
    base.at[row_idx, "fusion_type"] = "trimole_maplight_ensemble"
    if "loss_type" in base.columns:
        base.at[row_idx, "loss_type"] = "Ensemble"
    if "primary_metric_name" in base.columns:
        base.at[row_idx, "primary_metric_name"] = "AUPRC"
    if "primary_metric" in base.columns:
        base.at[row_idx, "primary_metric"] = float(best["primary_metric"])
    if "test_auc" in base.columns:
        base.at[row_idx, "test_auc"] = float(best["test_auc"])
    if "test_auprc" in base.columns:
        base.at[row_idx, "test_auprc"] = float(best["test_auprc"])
    if "test_acc" in base.columns:
        base.at[row_idx, "test_acc"] = float(best["test_acc"])
    if "hidden_dim" in base.columns:
        base.at[row_idx, "hidden_dim"] = np.nan
    if "dropout_head" in base.columns:
        base.at[row_idx, "dropout_head"] = np.nan

out_main = Path("results/model_log/final_best_of_all_22_plus_cyp2c9_ensemble.csv")
out_cfg  = Path("results/model_log/final_best_of_all_22_plus_cyp2c9_ensemble_with_config.csv")
out_patch = Path("results/model_log/final_patch_cyp2c9_veith_ensemble.csv")
out_json = Path("results/model_log/final_patch_cyp2c9_veith_ensemble_summary.json")

base.to_csv(out_main, index=False)
base.to_csv(out_cfg, index=False)
pd.DataFrame([patch_row]).to_csv(out_patch, index=False)
out_json.write_text(json.dumps({
    "base_file": str(base_file),
    "selected": bool(selected),
    "task": "cyp2c9_veith",
    "best_weight_trimole": float(best["weight_trimole"]),
    "best_weight_maplight": float(best["weight_maplight"]),
    "old_auprc": None if pd.isna(old_value) else float(old_value),
    "new_auprc": float(new_value),
    "output_file": str(out_main),
}, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", out_main)
print(" -", out_cfg)
print(" -", out_patch)
print(" -", out_json)
print()
print(pd.DataFrame([patch_row]).to_string(index=False))
