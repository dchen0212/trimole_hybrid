#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

MAPLIGHT_ROOT = Path("<PROJECT_ROOT>/trimole/external/MapLight-TDC")
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

BENCHMARK_CONFIG = {
    "ames": ("classification", "AUROC"),
    "bioavailability_ma": ("classification", "AUROC"),
    "bbb_martins": ("classification", "AUROC"),
    "cyp2c9_veith": ("classification", "AUPRC"),
    "pgp_broccatelli": ("classification", "AUROC"),
    "herg": ("classification", "AUROC"),
    "hia_hou": ("classification", "AUROC"),
    "dili": ("classification", "AUROC"),
}

def detect_cols(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    smiles_col = cols_lower.get("smiles") or cols_lower.get("drug")
    y_col = cols_lower.get("label") or cols_lower.get("y") or cols_lower.get("target")
    if smiles_col is None or y_col is None:
        raise KeyError(f"Cannot detect smiles/label columns: {list(df.columns)}")
    return smiles_col, y_col

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    smiles_col, y_col = detect_cols(df)
    X = np.asarray(get_fingerprints(df[smiles_col].astype(str)), dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    y = df[y_col].to_numpy()
    return X, y

def eval_cls(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_true, y_prob)),
        "test_auprc": float(average_precision_score(y_true, y_prob)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
    }

def fit_one(task_dir: Path, task: str, seed: int):
    task_type, official_metric = BENCHMARK_CONFIG[task]
    X_train, y_train = load_xy(task_dir / "train.csv")
    X_valid, y_valid = load_xy(task_dir / "valid.csv")
    X_test, y_test = load_xy(task_dir / "test.csv")

    clf = XGBClassifier(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_jobs=8,
        eval_metric="auc",
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    p_valid = clf.predict_proba(X_valid)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    valid_auc = roc_auc_score(y_valid, p_valid)
    valid_auprc = average_precision_score(y_valid, p_valid)
    best_valid_primary = valid_auc if official_metric == "AUROC" else valid_auprc

    row = {
        "task": task,
        "task_type": task_type,
        "primary_metric_name": official_metric,
        "best_valid_primary": float(best_valid_primary),
        "loss_type": "XGBoost",
        "seed": seed,
    }
    row.update(eval_cls(y_test, p_test))
    row["primary_metric"] = row["test_auc"] if official_metric == "AUROC" else row["test_auprc"]
    return row, y_test, p_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / "run_xgb_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = {}

    for task in args.tasks:
        try:
            row, y_true, y_pred = fit_one(data_root / task, task, args.seed)
            pred_df = pd.DataFrame({"task": task, "y_true": y_true, "y_pred": y_pred})
            pred_df.to_csv(run_dir / f"{task}_test_predictions.csv", index=False)
            rows.append(row)
            print(f"[{task}] {row['primary_metric_name']}={row['primary_metric']:.6f}")
        except Exception as e:
            errors[task] = str(e)
            print(f"[{task}] FAILED: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "results_all.csv", index=False)
        print(f"\nDone. Summary: {run_dir / 'results_all.csv'}")

    if errors:
        (run_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Failures: {len(errors)} (see {run_dir / 'errors.json'})")

if __name__ == "__main__":
    main()
