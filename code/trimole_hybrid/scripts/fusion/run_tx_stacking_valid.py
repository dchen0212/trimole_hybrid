#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error
from scipy.stats import spearmanr

OFFICIAL_METRIC = {
    "ames": "AUROC",
    "caco2_wang": "MAE",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "ld50_zhu": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "ppbr_az": "MAE",
}

DATA_ROOT = Path("data/data_benchmark")
TRIMOLE_VALID_DIR = Path("results/model_log/fusion_inputs_valid_trimole_22tasks")
XGB_VALID_DIR = Path("results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid")
TRIMOLE_TEST_DIRS = [
    Path("results/model_log/validation_dump_rerun_keytasks/run_20260413_2116"),
    Path("results/model_log/validation_dump_rerun_regfix/run_20260413_2136"),
    Path("results/model_log/tdc_benchmark_22check/run_20260411_2101"),
]
XGB_TEST_DIR = Path("results/model_log/xgb_baseline_22tasks/run_xgb_baseline_22tasks")
OUT_DIR = Path("results/model_log/tx_stacking_11tasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_label_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ("label", "y", "target"):
        if k in low:
            return low[k]
    raise ValueError(f"Cannot detect label column: {list(df.columns)}")

def detect_pred_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ("y_pred", "y_prob", "y_pred_mean", "pred", "prob"):
        if k in low:
            return low[k]
    raise ValueError(f"Cannot detect pred column: {list(df.columns)}")

def load_true(task: str, split: str) -> np.ndarray:
    df = pd.read_csv(DATA_ROOT / task / f"{split}.csv")
    y_col = detect_label_col(df)
    return df[y_col].to_numpy()

def load_pred(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    p_col = detect_pred_col(df)
    return df[p_col].to_numpy()

def find_trimole_test(task: str) -> Path:
    for base in TRIMOLE_TEST_DIRS:
        p = base / task / "test_predictions.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"No trimole test predictions found for {task}")

def eval_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if metric == "AUROC":
        y_hat = (y_pred >= 0.5).astype(int)
        return {
            "primary_metric": float(roc_auc_score(y_true, y_pred)),
            "test_auc": float(roc_auc_score(y_true, y_pred)),
            "test_auprc": float(average_precision_score(y_true, y_pred)),
            "test_acc": float(accuracy_score(y_true, y_hat)),
            "test_mae": np.nan,
            "test_spearman": np.nan,
        }
    if metric == "AUPRC":
        y_hat = (y_pred >= 0.5).astype(int)
        return {
            "primary_metric": float(average_precision_score(y_true, y_pred)),
            "test_auc": float(roc_auc_score(y_true, y_pred)),
            "test_auprc": float(average_precision_score(y_true, y_pred)),
            "test_acc": float(accuracy_score(y_true, y_hat)),
            "test_mae": np.nan,
            "test_spearman": np.nan,
        }
    if metric == "MAE":
        return {
            "primary_metric": float(mean_absolute_error(y_true, y_pred)),
            "test_auc": np.nan,
            "test_auprc": np.nan,
            "test_acc": np.nan,
            "test_mae": float(mean_absolute_error(y_true, y_pred)),
            "test_spearman": float(spearmanr(y_true, y_pred).statistic),
        }
    if metric == "Spearman":
        return {
            "primary_metric": float(spearmanr(y_true, y_pred).statistic),
            "test_auc": np.nan,
            "test_auprc": np.nan,
            "test_acc": np.nan,
            "test_mae": float(mean_absolute_error(y_true, y_pred)),
            "test_spearman": float(spearmanr(y_true, y_pred).statistic),
        }
    raise ValueError(metric)

rows = []

for task, metric in OFFICIAL_METRIC.items():
    y_valid = load_true(task, "valid")
    y_test = load_true(task, "test")

    p_t_valid = load_pred(TRIMOLE_VALID_DIR / f"{task}_valid_predictions.csv")
    p_x_valid = load_pred(XGB_VALID_DIR / f"{task}_valid_predictions.csv")

    p_t_test = load_pred(find_trimole_test(task))
    p_x_test = load_pred(XGB_TEST_DIR / f"{task}_test_predictions.csv")

    if not (len(y_valid) == len(p_t_valid) == len(p_x_valid)):
        raise ValueError(f"valid length mismatch: {task}, y={len(y_valid)}, t={len(p_t_valid)}, x={len(p_x_valid)}")
    if not (len(y_test) == len(p_t_test) == len(p_x_test)):
        raise ValueError(f"test length mismatch: {task}, y={len(y_test)}, t={len(p_t_test)}, x={len(p_x_test)}")

    X_valid = np.column_stack([p_t_valid, p_x_valid])
    X_test = np.column_stack([p_t_test, p_x_test])

    if metric in ("AUROC", "AUPRC"):
        meta = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        meta.fit(X_valid, y_valid.astype(int))
        p_valid_stack = meta.predict_proba(X_valid)[:, 1]
        p_test_stack = meta.predict_proba(X_test)[:, 1]
        coef = meta.coef_[0].tolist()
        intercept = float(meta.intercept_[0])
    else:
        meta = Ridge(alpha=1.0, random_state=42)
        meta.fit(X_valid, y_valid.astype(float))
        p_valid_stack = meta.predict(X_valid)
        p_test_stack = meta.predict(X_test)
        coef = meta.coef_.tolist()
        intercept = float(meta.intercept_)

    valid_res = eval_metric(metric, y_valid, p_valid_stack)
    test_res = eval_metric(metric, y_test, p_test_stack)

    rows.append({
        "task": task,
        "metric": metric,
        "meta_model": meta.__class__.__name__,
        "coef_trimole": float(coef[0]),
        "coef_xgb": float(coef[1]),
        "intercept": intercept,
        "valid_primary": valid_res["primary_metric"],
        "test_primary": test_res["primary_metric"],
        **test_res,
    })

df = pd.DataFrame(rows).sort_values("task")
out = OUT_DIR / "stacking_tx_results.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("\nSaved:", out)
