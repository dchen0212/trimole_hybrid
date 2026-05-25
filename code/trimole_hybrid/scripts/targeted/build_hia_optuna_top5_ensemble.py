#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
DATA_DIR = PROJECT / "data" / "data_benchmark" / "hia_hou"
TRIALS_CSV = PROJECT / "results" / "model_log" / "hia_xgb_optuna" / "optuna_all_trials.csv"
OUT_DIR = PROJECT / "results" / "model_log" / "hia_xgb_optuna_top5_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPLIGHT_ROOT = PROJECT / "external" / "MapLight-TDC"
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

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

trials = pd.read_csv(TRIALS_CSV)
top = trials.sort_values(["valid_auc","test_auc"], ascending=[False,False]).head(5).copy()

valid_preds = []
test_preds = []
chosen = []

for i, r in top.iterrows():
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": int(r["random_state"]) if "random_state" in r else 42,
        "n_jobs": 8,
        "n_estimators": int(r["n_estimators"]),
        "max_depth": int(r["max_depth"]),
        "learning_rate": float(r["learning_rate"]),
        "subsample": float(r["subsample"]),
        "colsample_bytree": float(r["colsample_bytree"]),
        "colsample_bynode": float(r["colsample_bynode"]),
        "min_child_weight": int(r["min_child_weight"]),
        "reg_lambda": float(r["reg_lambda"]),
        "reg_alpha": float(r["reg_alpha"]),
        "gamma": float(r["gamma"]),
        "max_bin": int(r["max_bin"]),
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    p_valid = model.predict_proba(X_valid)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    valid_preds.append(p_valid)
    test_preds.append(p_test)
    chosen.append(params)

p_valid_mean = np.mean(valid_preds, axis=0)
p_test_mean = np.mean(test_preds, axis=0)

valid_m = eval_cls(y_valid, p_valid_mean)
test_m = eval_cls(y_test, p_test_mean)

pd.DataFrame({"task":"hia_hou","y_true":y_valid,"y_pred":p_valid_mean}).to_csv(
    OUT_DIR / "hia_hou_valid_predictions.csv", index=False
)
pd.DataFrame({"task":"hia_hou","y_true":y_test,"y_pred":p_test_mean}).to_csv(
    OUT_DIR / "hia_hou_test_predictions.csv", index=False
)

pd.DataFrame([{
    "task": "hia_hou",
    "task_type": "classification",
    "primary_metric_name": "AUROC",
    "best_valid_primary": valid_m["auc"],
    "primary_metric": test_m["auc"],
    "loss_type": "OptunaTop5Ensemble",
    "test_auc": test_m["auc"],
    "test_auprc": test_m["auprc"],
    "test_acc": test_m["acc"],
}]).to_csv(OUT_DIR / "results_all.csv", index=False)

(OUT_DIR / "chosen_top5_params.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False))

print("=== HIA OPTUNA TOP5 ENSEMBLE ===")
print(f"valid_auc = {valid_m['auc']:.6f}")
print(f"test_auc  = {test_m['auc']:.6f}")
print(f"test_auprc= {test_m['auprc']:.6f}")
print(f"test_acc  = {test_m['acc']:.6f}")
print("\nSaved:")
print(" -", OUT_DIR / "results_all.csv")
print(" -", OUT_DIR / "chosen_top5_params.json")
