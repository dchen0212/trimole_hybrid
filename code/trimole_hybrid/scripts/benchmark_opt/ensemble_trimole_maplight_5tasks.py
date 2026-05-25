#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

OFFICIAL_METRIC = {
    "bioavailability_ma": "AUROC",
    "bbb_martins": "AUROC",
    "cyp2c9_veith": "AUPRC",
    "pgp_broccatelli": "AUROC",
    "herg": "AUROC",
}

TASKS = list(OFFICIAL_METRIC.keys())

def evaluate(task, y_true, y_pred):
    out = {
        "task": task,
        "test_auc": float(roc_auc_score(y_true, y_pred)),
        "test_auprc": float(average_precision_score(y_true, y_pred)),
        "test_acc": float(accuracy_score(y_true, (y_pred >= 0.5).astype(int))),
        "primary_metric_name": OFFICIAL_METRIC[task],
    }
    out["primary_metric"] = out["test_auc"] if OFFICIAL_METRIC[task] == "AUROC" else out["test_auprc"]
    return out

def load_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true_col = cols.get("y_true")
    y_pred_col = cols.get("y_pred")
    if y_pred_col is None:
        y_pred_col = cols.get("y_prob")
    if y_true_col is None or y_pred_col is None:
        raise ValueError(
            f"{path} missing prediction columns. "
            f"Need y_true + (y_pred or y_prob). columns={list(df.columns)}"
        )
    return df[y_true_col].to_numpy(), df[y_pred_col].to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trimole-dir", required=True, type=str)
    ap.add_argument("--maplight-dir", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--weight-trimole", type=float, default=0.5)
    ap.add_argument("--weight-maplight", type=float, default=0.5)
    args = ap.parse_args()

    trimole_dir = Path(args.trimole_dir)
    maplight_dir = Path(args.maplight_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wt = args.weight_trimole
    wm = args.weight_maplight
    s = wt + wm
    wt, wm = wt / s, wm / s

    rows = []
    for task in TASKS:
        trimole_file = trimole_dir / f"{task}_test_predictions.csv"
        maplight_file = maplight_dir / f"{task}_test_predictions.csv"

        if not trimole_file.exists():
            raise FileNotFoundError(f"Missing Trimole prediction file: {trimole_file}")
        if not maplight_file.exists():
            raise FileNotFoundError(f"Missing MapLight prediction file: {maplight_file}")

        y_true_t, y_pred_t = load_pred(trimole_file)
        y_true_m, y_pred_m = load_pred(maplight_file)

        if len(y_true_t) != len(y_true_m):
            raise ValueError(f"{task}: prediction length mismatch: trimole={len(y_true_t)} maplight={len(y_true_m)}")
        if not np.allclose(y_true_t.astype(float), y_true_m.astype(float)):
            raise ValueError(f"{task}: y_true mismatch between Trimole and MapLight predictions")

        y_ens = wt * y_pred_t + wm * y_pred_m

        pred_df = pd.DataFrame({
            "task": task,
            "y_true": y_true_t,
            "y_pred_trimole": y_pred_t,
            "y_pred_maplight": y_pred_m,
            "y_pred_ensemble": y_ens,
        })
        pred_df.to_csv(out_dir / f"{task}_ensemble_predictions.csv", index=False)

        row = evaluate(task, y_true_t, y_ens)
        row["weight_trimole"] = wt
        row["weight_maplight"] = wm
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    res.to_csv(out_dir / "results_all.csv", index=False)
    print(res.to_string(index=False))
    print(f"\nSaved: {out_dir / 'results_all.csv'}")

if __name__ == "__main__":
    main()
