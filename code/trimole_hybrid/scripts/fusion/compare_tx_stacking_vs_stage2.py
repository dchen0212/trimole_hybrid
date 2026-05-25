#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

OLD_P = Path("results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv")
STK_P = Path("results/model_log/tx_stacking_11tasks/stacking_tx_results.csv")
OUT_P = Path("results/model_log/tx_stacking_11tasks/compare_vs_stage2.csv")

old = pd.read_csv(OLD_P)
stk = pd.read_csv(STK_P)

higher_better = {"AUROC", "AUPRC", "Spearman"}
rows = []

for _, r in stk.iterrows():
    task = r["task"]
    metric = r["metric"]
    old_row = old[old["task"] == task].iloc[0]
    old_val = float(old_row["primary_metric"])
    new_val = float(r["test_primary"])

    if metric in higher_better:
        delta = new_val - old_val
        better = new_val > old_val
    else:
        delta = old_val - new_val
        better = new_val < old_val

    rows.append({
        "task": task,
        "metric": metric,
        "old_primary": old_val,
        "stacking_primary": new_val,
        "delta_improvement": delta,
        "better": better,
        "coef_trimole": r["coef_trimole"],
        "coef_xgb": r["coef_xgb"],
        "meta_model": r["meta_model"],
    })

df = pd.DataFrame(rows).sort_values(["better", "delta_improvement"], ascending=[False, False])
df.to_csv(OUT_P, index=False)
print(df.to_string(index=False))
print("\nSaved:", OUT_P)
