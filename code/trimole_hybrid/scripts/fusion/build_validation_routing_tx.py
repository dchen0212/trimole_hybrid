#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
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

def score(y_true, y_pred, metric):
    if metric == "AUROC":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "AUPRC":
        return float(average_precision_score(y_true, y_pred))
    if metric == "MAE":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "Spearman":
        return float(spearmanr(y_true, y_pred).statistic)
    raise ValueError(metric)

def better(metric, a, b):
    if b is None:
        return True
    if metric in ("AUROC", "AUPRC", "Spearman"):
        return a > b
    return a < b

def detect_label_col(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in ("label", "y", "target"):
        if k in cols:
            return cols[k]
    raise ValueError(f"Cannot detect label column: {list(df.columns)}")

def load_official_valid_y(task: str, data_root: Path) -> np.ndarray:
    df = pd.read_csv(data_root / task / "valid.csv")
    y_col = detect_label_col(df)
    return df[y_col].to_numpy()

def load_pred_only(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    pred_col = cols.get("y_pred") or cols.get("y_prob") or cols.get("y_pred_mean") or cols.get("pred") or cols.get("prob")
    if pred_col is None:
        raise ValueError(f"{path} missing prediction col: {list(df.columns)}")
    return df[pred_col].to_numpy()

tasks = list(OFFICIAL_METRIC.keys())
data_root = Path("data/data_benchmark")
trimole_dir = Path("results/model_log/fusion_inputs_valid_trimole_22tasks")
xgb_dir = Path("results/model_log/fusion_inputs_valid_xgb_22tasks")
out_dir = Path("results/model_log/validation_routing_tx")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
grid = np.arange(0.0, 1.0 + 1e-9, 0.05)

for task in tasks:
    metric = OFFICIAL_METRIC[task]
    t_file = trimole_dir / f"{task}_valid_predictions.csv"
    x_file = xgb_dir / f"{task}_valid_predictions.csv"
    if not (t_file.exists() and x_file.exists()):
        continue

    y_true = load_official_valid_y(task, data_root)
    p_t = load_pred_only(t_file)
    p_x = load_pred_only(x_file)

    if len(y_true) != len(p_t):
        raise ValueError(f"length mismatch trimole: {task} y_true={len(y_true)} pred={len(p_t)}")
    if len(y_true) != len(p_x):
        raise ValueError(f"length mismatch xgb: {task} y_true={len(y_true)} pred={len(p_x)}")

    candidates = []

    s_t = score(y_true, p_t, metric)
    candidates.append({
        "task": task, "metric": metric, "strategy": "keep_original",
        "weight_trimole": 1.0, "weight_xgb": 0.0, "weight_gnn": 0.0,
        "valid_score": s_t
    })

    s_x = score(y_true, p_x, metric)
    candidates.append({
        "task": task, "metric": metric, "strategy": "pure_xgb",
        "weight_trimole": 0.0, "weight_xgb": 1.0, "weight_gnn": 0.0,
        "valid_score": s_x
    })

    best_lf = None
    best_lf_score = None
    for wt in grid:
        wx = 1.0 - wt
        pred = wt * p_t + wx * p_x
        sc = score(y_true, pred, metric)
        if better(metric, sc, best_lf_score):
            best_lf_score = sc
            best_lf = {
                "task": task, "metric": metric, "strategy": "late_fusion_tx",
                "weight_trimole": round(float(wt), 2),
                "weight_xgb": round(float(wx), 2),
                "weight_gnn": 0.0,
                "valid_score": sc
            }
    candidates.append(best_lf)

    best = None
    best_score = None
    for c in candidates:
        if better(metric, c["valid_score"], best_score):
            best = c
            best_score = c["valid_score"]

    rows.append(best)

df = pd.DataFrame(rows)
if df.empty:
    print("No paired trimole/xgb validation predictions found.")
    df.to_csv(out_dir / "validation_routing_tx.csv", index=False)
    raise SystemExit(0)

df = df.sort_values("task")
df.to_csv(out_dir / "validation_routing_tx.csv", index=False)
print(df.to_string(index=False))
print("\nSaved:", out_dir / "validation_routing_tx.csv")
