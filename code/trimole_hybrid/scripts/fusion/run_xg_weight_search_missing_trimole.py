from __future__ import annotations

from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

ROOT = Path("<PROJECT_ROOT>/trimole")
OUT_DIR = ROOT / "results/model_log/xg_weight_search_missing_trimole"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "pgp_broccatelli",
    "solubility_aqsoldb",
    "vdss_lombardo",
]

OFFICIAL_METRIC = {
    "bioavailability_ma": "AUROC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "half_life_obach": "Spearman",
    "pgp_broccatelli": "AUROC",
    "solubility_aqsoldb": "MAE",
    "vdss_lombardo": "Spearman",
}

def weights(step=0.05):
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    out = []
    for x in vals:
        g = round(1 - x, 10)
        if 0 <= g <= 1:
            out.append((x, g))
    return out

WEIGHTS = weights(0.05)

def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def score(metric, y_true, y_pred):
    m = metric.upper()
    if m == "AUROC":
        return float(roc_auc_score(y_true, y_pred))
    if m == "AUPRC":
        return float(average_precision_score(y_true, y_pred))
    if m == "MAE":
        return float(mean_absolute_error(y_true, y_pred))
    if m == "SPEARMAN":
        return float(spearman_corr(y_true, y_pred))
    raise ValueError(metric)

def better(metric, a, b):
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    if metric.upper() in {"MAE", "RMSE", "MSE"}:
        return a < b
    return a > b

def read_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true = df[cols["y_true"]].to_numpy()
    if "y_prob" in cols:
        y_pred = df[cols["y_prob"]].to_numpy()
    elif "y_pred" in cols:
        y_pred = df[cols["y_pred"]].to_numpy()
    else:
        raise ValueError(f"unsupported prediction format: {path}")
    sample_idx = df[cols["sample_idx"]].to_numpy() if "sample_idx" in cols else None
    if sample_idx is not None:
        order = np.argsort(sample_idx)
        y_true = y_true[order]
        y_pred = y_pred[order]
    return y_true, y_pred

def newest(cands):
    cands = [p for p in cands if p.exists()]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def find_xgb_valid(task: str):
    candidates = [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_valid_xgb_22tasks" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "valid_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"[X-valid] {task}")

def find_xgb_test(task: str):
    cands = [p for p in ROOT.rglob(f"{task}_test_predictions.csv") if ("xgb" in str(p).lower() or "baseline" in str(p).lower() or "maplight" in str(p).lower())]
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[X-test] {task}")
    return p

def find_gnn_valid(task: str):
    p = newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("valid_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-valid] {task}")
    return p

def find_gnn_test(task: str):
    p = newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("test_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-test] {task}")
    return p

summary_rows = []
detail_rows = []
failed_rows = []

for task in TASKS:
    metric = OFFICIAL_METRIC[task]
    try:
        vx, vg = find_xgb_valid(task), find_gnn_valid(task)
        tx, tg = find_xgb_test(task), find_gnn_test(task)

        yv_x, pv_x = read_pred(vx)
        yv_g, pv_g = read_pred(vg)
        yt_x, pt_x = read_pred(tx)
        yt_g, pt_g = read_pred(tg)

        if metric in {"AUROC", "AUPRC"}:
            if not np.allclose(yv_x, yv_g):
                raise ValueError(f"{task} valid y_true mismatch")
            if not np.allclose(yt_x, yt_g):
                raise ValueError(f"{task} test y_true mismatch")

        yv_ref = yv_x
        yt_ref = yt_x

        best_w = None
        best_valid = float("inf") if metric == "MAE" else -float("inf")

        for x, g in WEIGHTS:
            valid_pred = x * pv_x + g * pv_g
            s = score(metric, yv_ref, valid_pred)
            detail_rows.append({"task": task, "metric": metric, "x": x, "g": g, "valid_score": s})
            if better(metric, s, best_valid):
                best_valid = s
                best_w = (x, g)

        x, g = best_w
        test_pred = x * pt_x + g * pt_g
        test_score = score(metric, yt_ref, test_pred)

        summary_rows.append({
            "task": task,
            "metric": metric,
            "best_x": x,
            "best_g": g,
            "best_valid_score": best_valid,
            "test_score": test_score,
            "valid_xgb": score(metric, yv_ref, pv_x),
            "valid_gnn": score(metric, yv_ref, pv_g),
            "xgb_valid_file": str(vx),
            "gnn_valid_file": str(vg),
            "xgb_test_file": str(tx),
            "gnn_test_file": str(tg),
        })
        print(f"[done] {task} metric={metric} best_w=({x},{g}) valid={best_valid:.6f} test={test_score:.6f}", flush=True)
    except Exception as e:
        failed_rows.append({"task": task, "error": str(e)})
        print(f"[failed] {task}: {e}", flush=True)

summary = pd.DataFrame(summary_rows).sort_values("task") if summary_rows else pd.DataFrame()
detail = pd.DataFrame(detail_rows)
failed = pd.DataFrame(failed_rows)

if len(summary):
    summary.to_csv(OUT_DIR / "xg_weight_search_missing_trimole_summary.csv", index=False)
if len(detail):
    detail.to_csv(OUT_DIR / "xg_weight_search_missing_trimole_detail.csv", index=False)
if len(failed):
    failed.to_csv(OUT_DIR / "xg_weight_search_missing_trimole_failed.csv", index=False)

if len(summary):
    print("\n=== SUMMARY ===")
    print(summary[["task","metric","best_x","best_g","best_valid_score","test_score","valid_xgb","valid_gnn"]].to_string(index=False))
if len(failed):
    print("\n=== FAILED ===")
    print(failed.to_string(index=False))

print("\nSaved:", OUT_DIR / "xg_weight_search_missing_trimole_summary.csv")
