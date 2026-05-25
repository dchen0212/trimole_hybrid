#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

PROJECT = Path("/mnt/afs/250010150/zhensheng/trimole")
DATA_DIR = PROJECT / "data" / "data_benchmark" / "hia_hou"
OUT_DIR = PROJECT / "results" / "model_log" / "hia_xgb_optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPLIGHT_ROOT = PROJECT / "external" / "MapLight-TDC"
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

SEED = 42
N_TRIALS = 200

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

live_rows = []

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": SEED,
        "n_jobs": 8,
        "n_estimators": trial.suggest_int("n_estimators", 600, 2200, step=200),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0, step=0.1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.8, 1.0, step=0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "max_bin": trial.suggest_categorical("max_bin", [256, 384, 512]),
    }

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    p_valid = model.predict_proba(X_valid)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    valid_auc = float(roc_auc_score(y_valid, p_valid))
    test_m = eval_cls(y_test, p_test)

    row = {
        "trial": trial.number,
        **params,
        "valid_auc": valid_auc,
        "test_auc": test_m["auc"],
        "test_auprc": test_m["auprc"],
        "test_acc": test_m["acc"],
    }
    live_rows.append(row)
    pd.DataFrame(live_rows).to_csv(OUT_DIR / "optuna_live.csv", index=False)

    return valid_auc

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

best = study.best_trial
best_params = dict(best.params)
best_params.update({
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": SEED,
    "n_jobs": 8,
})

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

p_valid = best_model.predict_proba(X_valid)[:, 1]
p_test = best_model.predict_proba(X_test)[:, 1]

valid_auc = float(roc_auc_score(y_valid, p_valid))
test_m = eval_cls(y_test, p_test)

pd.DataFrame({"task": "hia_hou", "y_true": y_valid, "y_pred": p_valid}).to_csv(
    OUT_DIR / "hia_hou_valid_predictions.csv", index=False
)
pd.DataFrame({"task": "hia_hou", "y_true": y_test, "y_pred": p_test}).to_csv(
    OUT_DIR / "hia_hou_test_predictions.csv", index=False
)

summary = pd.DataFrame([{
    "task": "hia_hou",
    "task_type": "classification",
    "primary_metric_name": "AUROC",
    "best_valid_primary": valid_auc,
    "primary_metric": test_m["auc"],
    "loss_type": "OptunaXGBoost",
    "seed": SEED,
    "test_auc": test_m["auc"],
    "test_auprc": test_m["auprc"],
    "test_acc": test_m["acc"],
    **best.params,
}])
summary.to_csv(OUT_DIR / "results_all.csv", index=False)

(OUT_DIR / "best_params.json").write_text(json.dumps(best.params, indent=2, ensure_ascii=False))
pd.DataFrame(live_rows).to_csv(OUT_DIR / "optuna_all_trials.csv", index=False)

print("=== BEST TRIAL ===")
print(best.params)
print(f"best valid auc = {valid_auc:.6f}")
print(f"test auc = {test_m['auc']:.6f}")
print(f"test auprc = {test_m['auprc']:.6f}")
print(f"test acc = {test_m['acc']:.6f}")
print("\nSaved:")
print(" -", OUT_DIR / "results_all.csv")
print(" -", OUT_DIR / "best_params.json")
print(" -", OUT_DIR / "optuna_all_trials.csv")
