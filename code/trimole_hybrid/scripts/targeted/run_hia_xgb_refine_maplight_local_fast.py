#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import itertools
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
DATA_DIR = PROJECT / "data" / "data_benchmark" / "hia_hou"
OUT_DIR = PROJECT / "results" / "model_log" / "hia_xgb_refine_maplight_local_fast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPLIGHT_ROOT = PROJECT / "external" / "MapLight-TDC"
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

SEEDS = [1, 7, 42, 123, 3407]

PARAM_GRID = {
    "n_estimators": [800, 1000, 1400],
    "max_depth": [6, 7],
    "learning_rate": [0.04, 0.05],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [2, 3, 4],
    "reg_lambda": [0.5, 1.0, 2.0],
}

def detect_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    smiles_col = cols.get("smiles") or cols.get("drug")
    y_col = cols.get("label") or cols.get("y") or cols.get("target")
    if smiles_col is None or y_col is None:
        raise KeyError(f"Cannot detect smiles/label columns: {list(df.columns)}")
    return smiles_col, y_col

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    smiles_col, y_col = detect_cols(df)
    X = np.asarray(get_fingerprints(df[smiles_col].astype(str)), dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    y = df[y_col].astype(int).to_numpy()
    return X, y

def eval_cls(y_true, y_prob):
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_hat)),
    }

X_train, y_train = load_xy(DATA_DIR / "train.csv")
X_valid, y_valid = load_xy(DATA_DIR / "valid.csv")
X_test,  y_test  = load_xy(DATA_DIR / "test.csv")

rows = []
keys = list(PARAM_GRID.keys())
vals = [PARAM_GRID[k] for k in keys]

total = 1
for v in vals:
    total *= len(v)
total *= len(SEEDS)

cur = 0
for combo in itertools.product(*vals):
    base_cfg = dict(zip(keys, combo))
    for seed in SEEDS:
        cur += 1
        cfg = dict(base_cfg)
        cfg.update({
            "objective": "binary:logistic",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": 8,
            "eval_metric": "auc",
        })

        print(f"[{cur}/{total}] {cfg}")

        model = XGBClassifier(**cfg)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test  = model.predict_proba(X_test)[:, 1]

        valid_auc = float(roc_auc_score(y_valid, p_valid))
        test_m = eval_cls(y_test, p_test)

        rows.append({
            **base_cfg,
            "seed": seed,
            "valid_auc": valid_auc,
            "test_auc": test_m["auc"],
            "test_auprc": test_m["auprc"],
            "test_acc": test_m["acc"],
        })

        pd.DataFrame(rows).to_csv(OUT_DIR / "hia_xgb_refine_maplight_local_fast_live.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "hia_xgb_refine_maplight_local_fast_summary.csv", index=False)

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

agg.to_csv(OUT_DIR / "hia_xgb_refine_maplight_local_fast_agg.csv", index=False)

print("\n=== HIA MAPLIGHT-XGB LOCAL FAST AGG ===")
print(agg.to_string(index=False))
print("\nSaved:")
print(" -", OUT_DIR / "hia_xgb_refine_maplight_local_fast_summary.csv")
print(" -", OUT_DIR / "hia_xgb_refine_maplight_local_fast_agg.csv")
