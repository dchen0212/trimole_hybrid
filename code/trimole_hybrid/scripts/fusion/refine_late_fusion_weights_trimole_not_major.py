#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error
from scipy.stats import spearmanr

OFFICIAL_METRIC = {
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

def load_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true_col = cols.get("y_true")
    y_pred_col = (
        cols.get("y_pred")
        or cols.get("y_prob")
        or cols.get("y_pred_mean")
        or cols.get("pred")
        or cols.get("prob")
    )
    if y_true_col is None or y_pred_col is None:
        raise ValueError(f"{path} missing y_true/pred column. columns={list(df.columns)}")
    return df[y_true_col].to_numpy(), df[y_pred_col].to_numpy()

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
        return {"test_mae": mae, "primary_metric": mae}
    if metric_name == "Spearman":
        sp = float(spearmanr(y_true, y_pred).statistic)
        return {"test_spearman": sp, "primary_metric": sp}
    raise ValueError(metric_name)

def better(metric_name, new_val, best_val):
    if best_val is None:
        return True
    if metric_name in ("AUROC", "AUPRC", "Spearman"):
        return new_val > best_val
    return new_val < best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trimole-dir", required=True)
    ap.add_argument("--xgb-dir", required=True)
    ap.add_argument("--gnn-dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-trimole", type=float, default=0.40)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = []
    vals = np.arange(0.0, 1.0 + 1e-9, args.step)
    for wt in vals:
        if wt > args.max_trimole + 1e-9:
            continue
        for wx in vals:
            wg = 1.0 - wt - wx
            if wg < -1e-9:
                continue
            if wg < 0:
                wg = 0.0
            if abs((wt + wx + wg) - 1.0) > 1e-6:
                continue
            weights.append((round(float(wt), 4), round(float(wx), 4), round(float(wg), 4)))

    all_rows, best_rows = [], []

    for task in args.tasks:
        metric_name = OFFICIAL_METRIC[task]
        t_true, t_pred = load_pred(Path(args.trimole_dir) / f"{task}_test_predictions.csv")
        x_true, x_pred = load_pred(Path(args.xgb_dir) / f"{task}_test_predictions.csv")
        g_true, g_pred = load_pred(Path(args.gnn_dir) / f"{task}_test_predictions.csv")

        same_tx = (len(t_true) == len(x_true)) and np.allclose(t_true, x_true, rtol=1e-6, atol=1e-8, equal_nan=True)
        same_tg = (len(t_true) == len(g_true)) and np.allclose(t_true, g_true, rtol=1e-6, atol=1e-8, equal_nan=True)
        if not (same_tx and same_tg):
            raise ValueError(f"y_true mismatch for {task}")

        best_row, best_primary = None, None

        for wt, wx, wg in weights:
            pred = wt * t_pred + wx * x_pred + wg * g_pred
            metrics = eval_pred(t_true, pred, metric_name)
            row = {
                "task": task,
                "primary_metric_name": metric_name,
                "weight_trimole": wt,
                "weight_xgb": wx,
                "weight_gnn": wg,
                **metrics,
            }
            all_rows.append(row)
            if better(metric_name, metrics["primary_metric"], best_primary):
                best_primary = metrics["primary_metric"]
                best_row = row

        best_rows.append(best_row)

    pd.DataFrame(all_rows).to_csv(out_dir / "all_weight_results.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "best_by_task.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(best_rows, indent=2, ensure_ascii=False))

    print(pd.DataFrame(best_rows).to_string(index=False))
    print("\nSaved:", out_dir)

if __name__ == "__main__":
    main()
