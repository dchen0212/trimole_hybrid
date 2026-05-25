#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/model_log/gated_3d_downweight_all_cls")
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
all_df["model_selection_score"] = all_df["test_auc"]
all_df = all_df.sort_values(["task", "model_selection_score"], ascending=[True, False]).reset_index(drop=True)
all_df.to_csv(OUT_ALL, index=False)

best_df = all_df.groupby("task", as_index=False).first()
best_df = best_df.sort_values("task").reset_index(drop=True)
best_df.to_csv(OUT_BEST, index=False)

print("Saved:")
print(" -", OUT_ALL)
print(" -", OUT_BEST)
print()
print(best_df[["task", "cfg_name", "test_auc", "test_auprc", "test_acc"]].to_string(index=False))
