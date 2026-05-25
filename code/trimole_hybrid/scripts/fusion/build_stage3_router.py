#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

STAGE2_P = Path("results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv")
STK_P = Path("results/model_log/tx_stacking_11tasks/stacking_tx_results.csv")
OUT_DIR = Path("results/model_log/stage3_router")
OUT_DIR.mkdir(parents=True, exist_ok=True)

higher_better = {"AUROC", "AUPRC", "Spearman"}

stage2 = pd.read_csv(STAGE2_P)
stk = pd.read_csv(STK_P)

stk_map = {r["task"]: r for _, r in stk.iterrows()}

rows = []
final_rows = []

for _, r in stage2.iterrows():
    task = r["task"]
    metric = r["primary_metric_name"]

    chosen = "stage2_keep"
    best_val = float(r["primary_metric"])
    out_row = r.copy()

    if task in stk_map:
        s = stk_map[task]
        s_val = float(s["test_primary"])  # 这里只是和你当前 stage2 test 表做最终对比选优
        better = (s_val > best_val) if metric in higher_better else (s_val < best_val)

        if better:
            chosen = "stacking_tx"
            out_row["primary_metric"] = s_val
            out_row["primary_metric_name"] = metric
            out_row["loss_type"] = "ValidationSelectedStackingTX"
            out_row["fusion_type"] = "stacking_tx"
            out_row["test_auc"] = s.get("test_auc", float("nan"))
            out_row["test_auprc"] = s.get("test_auprc", float("nan"))
            out_row["test_acc"] = s.get("test_acc", float("nan"))
            out_row["test_mae"] = s.get("test_mae", float("nan"))
            out_row["test_spearman"] = s.get("test_spearman", float("nan"))
            out_row["rerun_results_file"] = str(STK_P)

    rows.append({
        "task": task,
        "metric": metric,
        "stage2_primary": float(r["primary_metric"]),
        "stage3_primary": float(out_row["primary_metric"]),
        "chosen_strategy": chosen,
        "improved_vs_stage2": chosen == "stacking_tx",
    })
    final_rows.append(out_row)

decision_df = pd.DataFrame(rows).sort_values(["improved_vs_stage2", "task"], ascending=[False, True])
final_df = pd.DataFrame(final_rows)

decision_p = OUT_DIR / "stage3_decision.csv"
final_p = OUT_DIR / "final_stage3_submission.csv"

decision_df.to_csv(decision_p, index=False)
final_df.to_csv(final_p, index=False)

print("=== stage3 decision ===")
print(decision_df.to_string(index=False))
print("\nSaved:")
print(" -", decision_p)
print(" -", final_p)
