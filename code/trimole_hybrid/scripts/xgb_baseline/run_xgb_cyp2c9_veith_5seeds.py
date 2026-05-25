#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

MAPLIGHT_ROOT = Path("/mnt/afs/250010150/zhensheng/trimole/external/MapLight-TDC")
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

TASK = "cyp2c9_veith"
SEEDS = [1, 2, 3, 4, 5]

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

def main():
    data_root = Path("./data/data_benchmark") / TASK
    out_root = Path("results/model_log/xgb_cyp2c9_veith_5seeds")
    out_root.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_xy(data_root / "train.csv")
    X_valid, y_valid = load_xy(data_root / "valid.csv")
    X_test, y_test = load_xy(data_root / "test.csv")

    rows = []
    pred_cols = {"y_true": y_test}

    for seed in SEEDS:
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

        row = {
            "task": TASK,
            "seed": seed,
            "primary_metric_name": "AUPRC",
            "best_valid_primary": float(average_precision_score(y_valid, p_valid)),
        }
        row.update(eval_cls(y_test, p_test))
        row["primary_metric"] = row["test_auprc"]
        rows.append(row)

        pred_cols[f"y_pred_seed{seed}"] = p_test

        pd.DataFrame({
            "task": TASK,
            "y_true": y_test,
            "y_pred": p_test,
        }).to_csv(out_root / f"{TASK}_seed{seed}_test_predictions.csv", index=False)

        print(f"[seed={seed}] AUPRC={row['test_auprc']:.6f} AUROC={row['test_auc']:.6f}")

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.to_csv(out_root / "seed_results.csv", index=False)

    pred_df = pd.DataFrame(pred_cols)
    seed_pred_cols = [c for c in pred_df.columns if c.startswith("y_pred_seed")]
    pred_df["y_pred_mean"] = pred_df[seed_pred_cols].mean(axis=1)

    mean_metrics = eval_cls(pred_df["y_true"].to_numpy(), pred_df["y_pred_mean"].to_numpy())
    summary = {
        "task": TASK,
        "metric": "AUPRC",
        "mean_of_seed_test_auprc": float(df["test_auprc"].mean()),
        "std_of_seed_test_auprc": float(df["test_auprc"].std(ddof=1)),
        "best_single_seed": int(df.loc[df["test_auprc"].idxmax(), "seed"]),
        "best_single_seed_auprc": float(df["test_auprc"].max()),
        "mean_prediction_ensemble_auprc": float(mean_metrics["test_auprc"]),
        "mean_prediction_ensemble_auroc": float(mean_metrics["test_auc"]),
        "mean_prediction_ensemble_acc": float(mean_metrics["test_acc"]),
    }

    pred_df.to_csv(out_root / f"{TASK}_5seed_mean_test_predictions.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Seed Results ===")
    print(df.to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_root}")

if __name__ == "__main__":
    main()
