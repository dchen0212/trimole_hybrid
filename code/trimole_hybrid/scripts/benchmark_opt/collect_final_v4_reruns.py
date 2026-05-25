#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

RUN_ROOT = Path("results/model_log/final_best_v4_runs")
CFG_FILE = Path("results/model_log/final_best_v4_task_config.csv")
OUT1 = Path("results/model_log/final_best_of_all_22_v4_rerun.csv")
OUT2 = Path("results/model_log/final_best_of_all_22_v4_rerun_with_config.csv")
OUT3 = Path("results/model_log/final_best_v4_rerun_summary.json")

result_files = sorted(RUN_ROOT.glob("*/run_*/results_all.csv"))
if not result_files:
    raise FileNotFoundError(f"No rerun results_all.csv found under: {RUN_ROOT}")

frames = []
for f in result_files:
    df = pd.read_csv(f)
    if "task" not in df.columns:
        continue
    df["rerun_results_file"] = str(f)
    df["rerun_task_dir"] = str(f.parent.parent)
    df["rerun_run_name"] = f.parent.name
    frames.append(df)

if not frames:
    raise RuntimeError("Found files, but none had a 'task' column.")

rerun_df = pd.concat(frames, ignore_index=True)

# 一个任务只取一行；如果不小心有多个 run，优先取最后一个
rerun_df = rerun_df.sort_values(["task", "rerun_run_name"]).groupby("task", as_index=False).last()

rerun_df = rerun_df.sort_values("task").reset_index(drop=True)
rerun_df.to_csv(OUT1, index=False)

if CFG_FILE.exists():
    cfg_df = pd.read_csv(CFG_FILE)
    merged = rerun_df.merge(cfg_df, on="task", how="left", suffixes=("", "_cfg"))
else:
    cfg_df = pd.DataFrame()
    merged = rerun_df.copy()

merged = merged.sort_values("task").reset_index(drop=True)
merged.to_csv(OUT2, index=False)

summary = {
    "run_root": str(RUN_ROOT),
    "n_result_files_found": len(result_files),
    "n_tasks_collected": int(len(rerun_df)),
    "output_csv": str(OUT1),
    "output_with_config_csv": str(OUT2),
    "config_file_used": str(CFG_FILE) if CFG_FILE.exists() else None,
    "tasks": merged["task"].tolist(),
}

OUT3.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT1)
print(" -", OUT2)
print(" -", OUT3)
print()
show_cols = [c for c in ["task", "task_type", "test_auc", "test_auroc", "test_auprc", "test_acc", "test_mae", "test_spearman", "fusion_type", "loss_type", "dropout_head"] if c in merged.columns]
print(merged[show_cols].to_string(index=False))
