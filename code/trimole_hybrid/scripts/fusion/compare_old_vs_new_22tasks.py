#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import math

ROOT = Path("<PROJECT_ROOT>/trimole/results/model_log")

OLD_P = ROOT / "final_validation_selected_submission" / "final_validation_selected_submission.csv"
NEW_P = ROOT / "final_tx_router_22tasks" / "final_tx_router_22tasks.csv"

OUT_DIR = ROOT / "final_tx_router_22tasks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_P = OUT_DIR / "compare_old_vs_new_22tasks.csv"
OUT_READABLE_P = OUT_DIR / "compare_old_vs_new_22tasks_readable.csv"

higher_better = {"AUROC", "AUPRC", "Spearman"}
lower_better = {"MAE", "RMSE"}

old = pd.read_csv(OLD_P)
new = pd.read_csv(NEW_P)

rows = []
for _, nr in new.iterrows():
    task = nr["task"]
    hit = old[old["task"] == task]
    if hit.empty:
        continue
    orow = hit.iloc[0]

    metric = str(nr["primary_metric_name"])
    old_val = float(orow["primary_metric"])
    new_val = float(nr["primary_metric"])

    if metric in higher_better:
        delta = new_val - old_val
        better = new_val > old_val
    elif metric in lower_better:
        delta = old_val - new_val
        better = new_val < old_val
    else:
        delta = math.nan
        better = False

    rows.append({
        "task": task,
        "metric": metric,
        "old_primary": old_val,
        "new_primary": new_val,
        "delta_improvement": delta,
        "better": better,
        "old_loss_type": orow["loss_type"] if "loss_type" in old.columns else "",
        "new_loss_type": nr["loss_type"] if "loss_type" in new.columns else "",
        "old_fusion_type": orow["fusion_type"] if "fusion_type" in old.columns else "",
        "new_fusion_type": nr["fusion_type"] if "fusion_type" in new.columns else "",
        "router_strategy": nr["router_strategy"] if "router_strategy" in new.columns else "",
        "router_valid_primary": nr["router_valid_primary"] if "router_valid_primary" in new.columns else math.nan,
        "router_test_primary": nr["router_test_primary"] if "router_test_primary" in new.columns else math.nan,
    })

df = pd.DataFrame(rows).sort_values(
    ["better", "delta_improvement", "task"],
    ascending=[False, False, True]
)

df.to_csv(OUT_P, index=False)

show = df[[
    "task",
    "metric",
    "old_primary",
    "new_primary",
    "delta_improvement",
    "better",
    "router_strategy",
    "router_valid_primary",
    "router_test_primary",
    "old_fusion_type",
    "new_fusion_type",
]].copy()

show.to_csv(OUT_READABLE_P, index=False)

print("=== OLD vs NEW 22 TASKS ===")
print(show.to_string(index=False))

print("\nSummary:")
print(" improved:", int(df["better"].sum()))
print(" unchanged_or_worse:", int((~df["better"]).sum()))

print("\nSaved:")
print(" -", OUT_P)
print(" -", OUT_READABLE_P)
