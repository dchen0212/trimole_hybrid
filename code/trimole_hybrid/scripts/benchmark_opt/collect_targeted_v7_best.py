#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/model_log/bench_opt_targeted_v7")
OUT_ALL = ROOT / "all_targeted_v7_results.csv"
OUT_BEST = ROOT / "best_targeted_v7_by_task.csv"

result_files = sorted(ROOT.glob("*/run_*/results_all.csv"))
if not result_files:
    raise FileNotFoundError(f"No results_all.csv found under {ROOT}")

frames = []
for f in result_files:
    df = pd.read_csv(f)
    if "task" not in df.columns:
        continue
    df["cfg_name"] = f.parents[1].name
    df["run_name"] = f.parents[0].name
    df["results_file"] = str(f)
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

def infer_task_type(row):
    if pd.notna(row.get("test_auc", np.nan)) or pd.notna(row.get("test_auprc", np.nan)):
        return "classification"
    return "regression"

all_df["task_type"] = all_df.apply(infer_task_type, axis=1)

def score_row(row):
    if row["task_type"] == "classification":
        return float(row["test_auc"]) if pd.notna(row.get("test_auc", np.nan)) else -np.inf
    return -float(row["test_mae"]) if pd.notna(row.get("test_mae", np.nan)) else -np.inf

all_df["model_selection_score"] = all_df.apply(score_row, axis=1)
all_df = all_df.sort_values(["task", "model_selection_score"], ascending=[True, False]).reset_index(drop=True)
all_df.to_csv(OUT_ALL, index=False)

best_df = all_df.groupby("task", as_index=False).first()
best_df = best_df.sort_values(["task_type", "task"]).reset_index(drop=True)
best_df.to_csv(OUT_BEST, index=False)

print("Saved:")
print(" -", OUT_ALL)
print(" -", OUT_BEST)
print()
show_cols = [c for c in ["task", "task_type", "cfg_name", "test_auc", "test_auprc", "test_mae", "test_spearman", "model_selection_score"] if c in best_df.columns]
print(best_df[show_cols].to_string(index=False))
