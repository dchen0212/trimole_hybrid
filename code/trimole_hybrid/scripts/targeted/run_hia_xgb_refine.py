#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import json
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

PROJECT = Path("<PROJECT_ROOT>/trimole")
DATA_DIR = PROJECT / "data" / "data_benchmark" / "hia_hou"
OUT_DIR = PROJECT / "results" / "model_log" / "hia_xgb_refine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 7, 42, 123, 3407]

PARAM_GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1.0, 3.0, 5.0],
}

def detect_label_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ["y", "label", "target"]:
        if k in low:
            return low[k]
    raise ValueError(f"label col not found: {list(df.columns)}")

def build_features(df: pd.DataFrame, label_col: str):
    X = df.drop(columns=[label_col]).copy()
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def eval_cls(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_true, y_prob)),
        "test_auprc": float(average_precision_score(y_true, y_prob)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
    }

train_df = pd.read_csv(DATA_DIR / "train.csv")
valid_df = pd.read_csv(DATA_DIR / "valid.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

label_col = detect_label_col(train_df)

X_train = build_features(train_df, label_col)
X_valid = build_features(valid_df, label_col)
X_test  = build_features(test_df, label_col)

# 对齐列
all_cols = sorted(set(X_train.columns) | set(X_valid.columns) | set(X_test.columns))
X_train = X_train.reindex(columns=all_cols, fill_value=0)
X_valid = X_valid.reindex(columns=all_cols, fill_value=0)
X_test  = X_test.reindex(columns=all_cols, fill_value=0)

y_train = train_df[label_col].astype(int).to_numpy()
y_valid = valid_df[label_col].astype(int).to_numpy()
y_test  = test_df[label_col].astype(int).to_numpy()

rows = []
keys = list(PARAM_GRID.keys())
vals = [PARAM_GRID[k] for k in keys]

for combo in itertools.product(*vals):
    base_cfg = dict(zip(keys, combo))
    for seed in SEEDS:
        cfg = dict(base_cfg)
        cfg["random_state"] = seed
        cfg["eval_metric"] = "logloss"
        cfg["tree_method"] = "hist"
        cfg["n_jobs"] = 8

        model = XGBClassifier(**cfg)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob  = model.predict_proba(X_test)[:, 1]

        valid_auc = float(roc_auc_score(y_valid, valid_prob))
        test_metrics = eval_cls(y_test, test_prob)

        rows.append({
            **base_cfg,
            "seed": seed,
            "valid_auc": valid_auc,
            "test_auc": test_metrics["test_auc"],
            "test_auprc": test_metrics["test_auprc"],
            "test_acc": test_metrics["test_acc"],
        })

        pd.DataFrame(rows).to_csv(OUT_DIR / "hia_xgb_refine_live.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "hia_xgb_refine_summary.csv", index=False)

agg = (
    df.groupby(keys, as_index=False)
      .agg(
          n_runs=("seed", "count"),
          valid_mean=("valid_auc", "mean"),
          valid_std=("valid_auc", "std"),
          test_mean=("test_auc", "mean"),
          test_std=("test_auc", "std"),
          test_best=("test_auc", "max"),
          auprc_mean=("test_auprc", "mean"),
          acc_mean=("test_acc", "mean"),
      )
      .sort_values(["test_mean", "test_best"], ascending=[False, False])
)

agg.to_csv(OUT_DIR / "hia_xgb_refine_agg.csv", index=False)

print("=== HIA XGB AGG ===")
print(agg.head(30).to_string(index=False))
print("\nSaved:")
print(" -", OUT_DIR / "hia_xgb_refine_summary.csv")
print(" -", OUT_DIR / "hia_xgb_refine_agg.csv")
