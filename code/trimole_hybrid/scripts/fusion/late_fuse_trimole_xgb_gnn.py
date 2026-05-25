#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error
from scipy.stats import spearmanr

TASK_METRIC = {
    "ames": "AUROC",
    "bbb_martins": "AUROC",
    "bioavailability_ma": "AUROC",
    "caco2_wang": "MAE",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "cyp3a4_veith": "AUPRC",
    "dili": "AUROC",
    "half_life_obach": "Spearman",
    "herg": "AUROC",
    "hia_hou": "AUROC",
    "ld50_zhu": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "pgp_broccatelli": "AUROC",
    "ppbr_az": "MAE",
    "solubility_aqsoldb": "MAE",
    "vdss_lombardo": "Spearman",
}

WEIGHTS = [
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.6, 0.2, 0.2),
    (0.5, 0.4, 0.1),
    (0.5, 0.3, 0.2),
]

def eval_pred(y_true, y_pred, metric_name):
    if metric_name == "AUROC":
        y_cls = (y_pred >= 0.5).astype(int)
        return {
            "test_auc": float(roc_auc_score(y_true, y_pred)),
            "test_auprc": float(average_precision_score(y_true, y_pred)),
            "test_acc": float(accuracy_score(y_true, y_cls)),
            "primary_metric": float(roc_auc_score(y_true, y_pred)),
        }
    if metric_name == "AUPRC":
        y_cls = (y_pred >= 0.5).astype(int)
        return {
            "test_auc": float(roc_auc_score(y_true, y_pred)),
            "test_auprc": float(average_precision_score(y_true, y_pred)),
            "test_acc": float(accuracy_score(y_true, y_cls)),
            "primary_metric": float(average_precision_score(y_true, y_pred)),
        }
    if metric_name == "MAE":
        mae = float(mean_absolute_error(y_true, y_pred))
        return {
            "test_mae": mae,
            "primary_metric": mae,
        }
    if metric_name == "Spearman":
        sp = float(spearmanr(y_true, y_pred).statistic)
        return {
            "test_spearman": sp,
            "primary_metric": sp,
        }
    raise ValueError(f"Unsupported metric_name: {metric_name}")

def load_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true_col = cols.get("y_true")
    y_pred_col = cols.get("y_pred") or cols.get("y_prob") or cols.get("y_pred_mean") or cols.get("pred") or cols.get("prob")
    if y_true_col is None or y_pred_col is None:
        raise ValueError(f"{path} missing y_true and prediction column. columns={list(df.columns)}")
    return df[y_true_col].to_numpy(), df[y_pred_col].to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trimole-dir", required=True, type=str)
    ap.add_argument("--xgb-dir", required=True, type=str)
    ap.add_argument("--gnn-dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    trimole_dir = Path(args.trimole_dir)
    xgb_dir = Path(args.xgb_dir)
    gnn_dir = Path(args.gnn_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_rows = []

    for task, metric_name in TASK_METRIC.items():
        trimole_file = trimole_dir / f"{task}_test_predictions.csv"
        xgb_file = xgb_dir / f"{task}_test_predictions.csv"
        gnn_file = gnn_dir / f"{task}_test_predictions.csv"

        y_true_t, y_pred_t = load_pred(trimole_file)
        y_true_x, y_pred_x = load_pred(xgb_file)
        y_true_g, y_pred_g = load_pred(gnn_file)

        same_tx = (len(y_true_t) == len(y_true_x)) and np.allclose(y_true_t, y_true_x, rtol=1e-6, atol=1e-8, equal_nan=True)
        same_tg = (len(y_true_t) == len(y_true_g)) and np.allclose(y_true_t, y_true_g, rtol=1e-6, atol=1e-8, equal_nan=True)
        if not (same_tx and same_tg):
            raise ValueError(
                f"y_true mismatch for task {task}: "
                f"trimole={len(y_true_t)} xgb={len(y_true_x)} gnn={len(y_true_g)}"
            )

        best_primary = -1.0
        best_info = None

        for wt, wx, wg in WEIGHTS:
            y_prob = wt * y_pred_t + wx * y_pred_x + wg * y_pred_g
            metrics = eval_pred(y_true_t, y_prob, metric_name)
            primary = metrics["primary_metric"]

            row = {
                "task": task,
                "primary_metric_name": metric_name,
                "weight_trimole": wt,
                "weight_xgb": wx,
                "weight_gnn": wg,
                **metrics,
            }
            rows.append(row)

            better = (primary > best_primary) if metric_name in ("AUROC", "AUPRC", "Spearman") else (best_primary < 0 or primary < best_primary)
            if better:
                best_primary = float(primary)
                best_info = row.copy()

        assert best_info is not None
        best_rows.append(best_info)

        pd.DataFrame({
            "task": task,
            "y_true": y_true_t,
            "y_pred": (
                best_info["weight_trimole"] * y_pred_t
                + best_info["weight_xgb"] * y_pred_x
                + best_info["weight_gnn"] * y_pred_g
            )
        }).to_csv(out_dir / f"{task}_test_predictions.csv", index=False)

    all_df = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)

    all_df.to_csv(out_dir / "all_weight_results.csv", index=False)
    best_df.to_csv(out_dir / "best_by_task.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(best_rows, indent=2, ensure_ascii=False)
    )

    print("Saved:")
    print(" -", out_dir / "all_weight_results.csv")
    print(" -", out_dir / "best_by_task.csv")
    print(" -", out_dir / "summary.json")
    print()
    print(best_df.to_string(index=False))

if __name__ == "__main__":
    main()
