#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import math

ROOT = Path("<PROJECT_ROOT>/trimole/results/model_log")

# 你“之前那个版本”的 22 任务总表
BASE_P = ROOT / "final_validation_selected_submission" / "final_validation_selected_submission.csv"

# 这次 TX router 的 6 任务结果
TX_P = ROOT / "tx_router_v1" / "tx_router_results.csv"

# xgb 原始结果，给 strategy=xgb 时补完整字段用
XGB_P = ROOT / "xgb_baseline_22tasks_with_valid" / "run_xgb_baseline_22tasks_with_valid" / "results_all.csv"

OUT_DIR = ROOT / "final_tx_router_22tasks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_P = OUT_DIR / "final_tx_router_22tasks.csv"

base = pd.read_csv(BASE_P)
tx = pd.read_csv(TX_P)
xgb = pd.read_csv(XGB_P)

base = base.copy()
tx_map = {r["task"]: r for _, r in tx.iterrows()}
xgb_map = {r["task"]: r for _, r in xgb.iterrows()}

# 尽量兼容你之前表的字段
for col in [
    "router_strategy",
    "router_valid_primary",
    "router_test_primary",
    "router_source",
]:
    if col not in base.columns:
        base[col] = ""

for idx, row in base.iterrows():
    task = row["task"]
    if task not in tx_map:
        continue

    r = tx_map[task]
    strat = str(r["strategy"])
    base.at[idx, "router_strategy"] = strat
    base.at[idx, "router_valid_primary"] = float(r["valid_primary"])
    base.at[idx, "router_test_primary"] = float(r["test_primary"])
    base.at[idx, "router_source"] = "tx_router_v1"

    # 1) xgb 直接替换成 xgb baseline 行
    if strat == "xgb":
        if task not in xgb_map:
            continue
        xr = xgb_map[task]

        if "primary_metric_name" in base.columns and "primary_metric_name" in xr.index:
            base.at[idx, "primary_metric_name"] = xr["primary_metric_name"]
        if "primary_metric" in base.columns and "primary_metric" in xr.index:
            base.at[idx, "primary_metric"] = float(xr["primary_metric"])
        if "loss_type" in base.columns:
            base.at[idx, "loss_type"] = "ValidationSelectedPureXGB_TXRouter"
        if "fusion_type" in base.columns:
            base.at[idx, "fusion_type"] = "xgb_tx_router"

        for c in ["test_auc", "test_auprc", "test_acc", "test_mae", "test_rmse", "test_spearman"]:
            if c in base.columns:
                base.at[idx, c] = float(xr[c]) if c in xr.index and pd.notna(xr[c]) else math.nan

    # 2) trimole 直接保留 router 结果
    elif strat == "trimole":
        if "primary_metric" in base.columns:
            base.at[idx, "primary_metric"] = float(r["test_primary"])
        if "loss_type" in base.columns:
            base.at[idx, "loss_type"] = "ValidationSelectedTrimole_TXRouter"
        if "fusion_type" in base.columns:
            base.at[idx, "fusion_type"] = "trimole_tx_router"
        for c in ["test_auc", "test_auprc", "test_acc"]:
            if c in base.columns and c in r.index:
                base.at[idx, c] = float(r[c])

    # 3) late fusion tx
    elif strat.startswith("late_fusion_tx_"):
        if "primary_metric" in base.columns:
            base.at[idx, "primary_metric"] = float(r["test_primary"])
        if "loss_type" in base.columns:
            base.at[idx, "loss_type"] = "ValidationSelectedLateFusionTXRouter"
        if "fusion_type" in base.columns:
            base.at[idx, "fusion_type"] = strat
        for c in ["test_auc", "test_auprc", "test_acc"]:
            if c in base.columns and c in r.index:
                base.at[idx, c] = float(r[c])

    # 4) stacking tx
    elif strat == "stacking_tx":
        if "primary_metric" in base.columns:
            base.at[idx, "primary_metric"] = float(r["test_primary"])
        if "loss_type" in base.columns:
            base.at[idx, "loss_type"] = "ValidationSelectedStackingTXRouter"
        if "fusion_type" in base.columns:
            base.at[idx, "fusion_type"] = "stacking_tx_router"
        for c in ["test_auc", "test_auprc", "test_acc"]:
            if c in base.columns and c in r.index:
                base.at[idx, c] = float(r[c])

# 排序
base = base.sort_values("task").reset_index(drop=True)
base.to_csv(OUT_P, index=False)

# 另外导出一个精简可读版
show_cols = [c for c in [
    "task",
    "primary_metric_name",
    "primary_metric",
    "loss_type",
    "fusion_type",
    "router_strategy",
    "router_valid_primary",
    "router_test_primary",
    "test_auc",
    "test_auprc",
    "test_acc",
] if c in base.columns]

READABLE_P = OUT_DIR / "final_tx_router_22tasks_readable.csv"
base[show_cols].to_csv(READABLE_P, index=False)

print("=== FINAL 22 TASKS (TX router patched) ===")
print(base[show_cols].to_string(index=False))
print("\nSaved:")
print(" -", OUT_P)
print(" -", READABLE_P)
