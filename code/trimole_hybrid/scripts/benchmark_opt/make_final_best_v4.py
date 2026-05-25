#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

ROOT = Path("results/model_log/bench_opt_minimal")
BEST_SWEEP = ROOT / "best_by_task_from_sweeps.csv"
TASK_META = ROOT / "task_metadata.csv"

PREV_FINALS = [
    Path("results/model_log/final_best_of_all_22_v3.csv"),
    Path("results/model_log/final_best_of_all_22_v2.csv"),
    Path("results/model_log/final_best_of_all_22.csv"),
]

OUT_FINAL = Path("results/model_log/final_best_of_all_22_v4.csv")
OUT_CONFIG = Path("results/model_log/final_best_v4_task_config.csv")
OUT_SUMMARY = Path("results/model_log/final_best_v4_summary.json")

if not BEST_SWEEP.exists():
    raise FileNotFoundError(f"Missing file: {BEST_SWEEP}")

best_df = pd.read_csv(BEST_SWEEP)
meta_df = pd.read_csv(TASK_META) if TASK_META.exists() else pd.DataFrame()

def parse_cfg(cfg_name: str) -> dict:
    cfg = {
        "fusion_type": None,
        "loss_type": None,
        "dropout_head": None,
        "seed": None,
    }
    parts = cfg_name.split("_")

    if "mlp" in parts:
        cfg["fusion_type"] = "mlp"
    elif "gated" in parts:
        cfg["fusion_type"] = "gated"

    if "focal" in parts:
        cfg["loss_type"] = "focal"
    elif "auto" in parts:
        cfg["loss_type"] = "auto"

    for p in parts:
        if p.startswith("d") and p[1:].isdigit():
            # d03 -> 0.3 ; d02 -> 0.2
            num = p[1:]
            if len(num) == 2:
                cfg["dropout_head"] = float(num[0] + "." + num[1])
        if p.startswith("seed") and p[4:].isdigit():
            cfg["seed"] = int(p[4:])

    return cfg

cfg_rows = []
for _, row in best_df.iterrows():
    cfg = parse_cfg(str(row["cfg_name"]))
    cfg_rows.append({
        "task": row["task"],
        "task_type": row.get("task_type"),
        "cfg_name": row["cfg_name"],
        "run_name": row["run_name"],
        "results_file": row["results_file"],
        "fusion_type": cfg["fusion_type"],
        "loss_type": cfg["loss_type"],
        "hidden_dim": 128,
        "dropout_head": cfg["dropout_head"],
        "seed": cfg["seed"],
        "model_selection_score": row["model_selection_score"],
    })

cfg_df = pd.DataFrame(cfg_rows).sort_values(["task_type", "task"]).reset_index(drop=True)
cfg_df.to_csv(OUT_CONFIG, index=False)

# 直接用当前 best sweep 作为 v4 正式表基础
final_df = best_df.copy()

# 如果有历史正式表，就补一些缺失列，尽量保持列风格一致
prev_df = None
for p in PREV_FINALS:
    if p.exists():
        prev_df = pd.read_csv(p)
        break

if prev_df is not None and "task" in prev_df.columns:
    prev_cols = [c for c in prev_df.columns if c not in final_df.columns]
    if prev_cols:
        merge_cols = ["task"] + prev_cols
        final_df = final_df.merge(prev_df[merge_cols], on="task", how="left")

# 再把配置列并进去
final_df = final_df.merge(
    cfg_df[["task", "fusion_type", "loss_type", "hidden_dim", "dropout_head", "seed"]],
    on="task",
    how="left"
)

# 排序
if "task_type" in final_df.columns:
    final_df = final_df.sort_values(["task_type", "task"]).reset_index(drop=True)
else:
    final_df = final_df.sort_values(["task"]).reset_index(drop=True)

final_df.to_csv(OUT_FINAL, index=False)

summary = {
    "source_best_sweep": str(BEST_SWEEP),
    "output_final_v4": str(OUT_FINAL),
    "output_config_table": str(OUT_CONFIG),
    "n_tasks": int(len(final_df)),
    "n_classification": int((final_df["task_type"] == "classification").sum()) if "task_type" in final_df.columns else None,
    "n_regression": int((final_df["task_type"] == "regression").sum()) if "task_type" in final_df.columns else None,
    "fusion_counts": cfg_df["fusion_type"].value_counts(dropna=False).to_dict(),
    "loss_counts": cfg_df["loss_type"].value_counts(dropna=False).to_dict(),
}

OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("Saved:")
print(" -", OUT_FINAL)
print(" -", OUT_CONFIG)
print(" -", OUT_SUMMARY)
print()
print(cfg_df.to_string(index=False))
