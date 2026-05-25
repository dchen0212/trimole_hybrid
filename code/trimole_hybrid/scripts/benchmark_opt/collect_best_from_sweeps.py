#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/model_log/bench_opt_minimal")
META = ROOT / "task_metadata.csv"
OUT_CSV = ROOT / "best_by_task_from_sweeps.csv"
ALL_CSV = ROOT / "all_sweep_results_flat.csv"

if not META.exists():
    raise FileNotFoundError(f"Missing metadata file: {META}. Run discover_benchmark_tasks.py first.")

meta = pd.read_csv(META)
task_type_map = dict(zip(meta["task"], meta["task_type"]))

result_files = sorted(ROOT.glob("*/run_*/results_all.csv"))
if not result_files:
    raise FileNotFoundError(f"No results_all.csv found under {ROOT}")

frames = []
for f in result_files:
    df = pd.read_csv(f)
    if "task" not in df.columns:
        continue
    cfg_name = f.parents[1].name
    run_name = f.parents[0].name
    df["cfg_name"] = cfg_name
    df["run_name"] = run_name
    df["results_file"] = str(f)
    df["task_type"] = df["task"].map(task_type_map)
    frames.append(df)

if not frames:
    raise RuntimeError("Found result files, but none had a 'task' column.")

all_df = pd.concat(frames, ignore_index=True)

def score_row(row):
    t = row.get("task_type")
    cols = row.index.tolist()

    # classification: maximize AUROC first, then AUPRC, then ACC
    if t == "classification":
        if "test_auc" in cols and pd.notna(row["test_auc"]):
            return float(row["test_auc"])
        if "test_auroc" in cols and pd.notna(row["test_auroc"]):
            return float(row["test_auroc"])
        if "test_auprc" in cols and pd.notna(row["test_auprc"]):
            return float(row["test_auprc"])
        if "test_acc" in cols and pd.notna(row["test_acc"]):
            return float(row["test_acc"])
        return -np.inf

    # regression:
    # prefer minimizing MAE; if absent, maximize Spearman; if absent, minimize RMSE/MSE
    if "test_mae" in cols and pd.notna(row["test_mae"]):
        return -float(row["test_mae"])
    if "test_spearman" in cols and pd.notna(row["test_spearman"]):
        return float(row["test_spearman"])
    if "test_rmse" in cols and pd.notna(row["test_rmse"]):
        return -float(row["test_rmse"])
    if "test_mse" in cols and pd.notna(row["test_mse"]):
        return -float(row["test_mse"])
    return -np.inf

all_df["model_selection_score"] = all_df.apply(score_row, axis=1)
all_df = all_df.sort_values(["task", "model_selection_score"], ascending=[True, False]).reset_index(drop=True)
all_df.to_csv(ALL_CSV, index=False)

best_df = all_df.groupby("task", as_index=False).first()
best_df = best_df.sort_values(["task_type", "task"]).reset_index(drop=True)
best_df.to_csv(OUT_CSV, index=False)

print("Saved:")
print(" -", ALL_CSV)
print(" -", OUT_CSV)
print()
print(best_df[["task", "task_type", "cfg_name", "run_name", "model_selection_score"]].to_string(index=False))
