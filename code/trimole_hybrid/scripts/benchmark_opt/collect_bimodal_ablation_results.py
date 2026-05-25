#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("results/model_log/ablation_bimodal")
OUT_ALL = ROOT / "ablation_all_results.csv"
OUT_PIVOT = ROOT / "ablation_metric_pivot.csv"
OUT_DELTA = ROOT / "ablation_delta_vs_full.csv"
OUT_SUMMARY = ROOT / "ablation_summary.json"

result_files = sorted(ROOT.glob("*/*/run_*/results_all.csv"))
if not result_files:
    raise FileNotFoundError(f"No results_all.csv found under {ROOT}")

frames = []
for f in result_files:
    setting = f.parents[2].name
    task = f.parents[1].name
    run_name = f.parents[0].name
    df = pd.read_csv(f)
    if "task" not in df.columns:
        continue
    df["ablation_setting"] = setting
    df["ablation_task_dir"] = task
    df["ablation_run_name"] = run_name
    df["results_file"] = str(f)
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

def choose_metric_name(row):
    if pd.notna(row.get("test_auc", np.nan)):
        return "AUROC"
    if pd.notna(row.get("test_mae", np.nan)):
        return "MAE"
    return "UNKNOWN"

def choose_metric_value(row):
    if pd.notna(row.get("test_auc", np.nan)):
        return float(row["test_auc"])
    if pd.notna(row.get("test_mae", np.nan)):
        return float(row["test_mae"])
    return np.nan

all_df["metric_name"] = all_df.apply(choose_metric_name, axis=1)
all_df["metric_value"] = all_df.apply(choose_metric_value, axis=1)

all_df = all_df.sort_values(["task", "ablation_setting"]).reset_index(drop=True)
all_df.to_csv(OUT_ALL, index=False)

pivot = all_df.pivot_table(
    index=["task", "task_type", "metric_name"],
    columns="ablation_setting",
    values="metric_value",
    aggfunc="first"
).reset_index()

for col in ["full", "drop_smiles", "drop_graph", "drop_3d"]:
    if col not in pivot.columns:
        pivot[col] = np.nan

pivot.to_csv(OUT_PIVOT, index=False)

delta_rows = []
for _, row in pivot.iterrows():
    task = row["task"]
    task_type = row["task_type"]
    metric_name = row["metric_name"]
    full = row["full"]

    for setting in ["drop_smiles", "drop_graph", "drop_3d"]:
        val = row[setting]
        if pd.isna(full) or pd.isna(val):
            delta = np.nan
        else:
            # classification: lower is worse => val-full
            # regression MAE: higher is worse => val-full
            delta = float(val - full)
        delta_rows.append({
            "task": task,
            "task_type": task_type,
            "metric_name": metric_name,
            "setting": setting,
            "full_value": full,
            "setting_value": val,
            "delta_setting_minus_full": delta,
        })

delta_df = pd.DataFrame(delta_rows).sort_values(["task", "setting"]).reset_index(drop=True)
delta_df.to_csv(OUT_DELTA, index=False)

summary = {
    "n_result_files": int(len(result_files)),
    "n_rows": int(len(all_df)),
    "settings": sorted(all_df["ablation_setting"].unique().tolist()),
    "tasks": sorted(all_df["task"].unique().tolist()),
    "outputs": {
        "all_results": str(OUT_ALL),
        "pivot": str(OUT_PIVOT),
        "delta": str(OUT_DELTA),
    },
}
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT_ALL)
print(" -", OUT_PIVOT)
print(" -", OUT_DELTA)
print(" -", OUT_SUMMARY)
print()
print(pivot.to_string(index=False))
