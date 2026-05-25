#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/model_log/gated_3d_downweight_all_22")
OUT_ALL = ROOT / "all_results.csv"
OUT_BEST = ROOT / "best_by_task.csv"

files = sorted(ROOT.glob("*/run_*/results_all.csv"))
if not files:
    raise FileNotFoundError(f"No results_all.csv found under {ROOT}")

frames = []
for f in files:
    df = pd.read_csv(f)
    if "task" not in df.columns:
        continue
    df["cfg_name"] = f.parents[1].name
    df["run_name"] = f.parents[0].name
    df["results_file"] = str(f)
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

def score_row(row):
    if pd.notna(row.get("test_auc", np.nan)):
        return float(row["test_auc"])
    if pd.notna(row.get("test_mae", np.nan)):
        return -float(row["test_mae"])
    return -np.inf

all_df["model_selection_score"] = all_df.apply(score_row, axis=1)
all_df = all_df.sort_values(["task", "model_selection_score"], ascending=[True, False]).reset_index(drop=True)
all_df.to_csv(OUT_ALL, index=False)

best_df = all_df.groupby("task", as_index=False).first()
best_df = best_df.sort_values(["task"]).reset_index(drop=True)
best_df.to_csv(OUT_BEST, index=False)

print("Saved:")
print(" -", OUT_ALL)
print(" -", OUT_BEST)
print()
cols = [c for c in ["task", "task_type", "cfg_name", "test_auc", "test_auprc", "test_acc", "test_mae", "test_spearman"] if c in best_df.columns]
print(best_df[cols].to_string(index=False))
