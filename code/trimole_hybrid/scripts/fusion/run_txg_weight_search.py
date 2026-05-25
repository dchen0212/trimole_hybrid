from __future__ import annotations

from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_DIR = ROOT / "results/model_log/txg_weight_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "ames",
    "bbb_martins",
    "caco2_wang",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "ld50_zhu",
    "lipophilicity_astrazeneca",
    "ppbr_az",
]

OFFICIAL_METRIC = {
    "ames": "AUROC",
    "bbb_martins": "AUROC",
    "caco2_wang": "MAE",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "dili": "AUROC",
    "herg": "AUROC",
    "hia_hou": "AUROC",
    "ld50_zhu": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "ppbr_az": "MAE",
}

def all_weight_triplets(step=0.1):
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    out = []
    for t, x, g in itertools.product(vals, vals, vals):
        if abs((t + x + g) - 1.0) < 1e-9:
            out.append((t, x, g))
    return out

WEIGHTS = all_weight_triplets(0.1)

def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def score_metric(metric, y_true, y_pred):
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

    y_true = df[cols["y_true"]].to_numpy() if "y_true" in cols else None

    if "y_prob" in cols:
        y_pred = df[cols["y_prob"]].to_numpy()
    elif "y_pred" in cols:
        y_pred = df[cols["y_pred"]].to_numpy()
    else:
        raise ValueError(f"unsupported prediction format: {path} cols={df.columns.tolist()}")

    sample_idx = df[cols["sample_idx"]].to_numpy() if "sample_idx" in cols else None
    return y_true, y_pred, sample_idx

def newest(cands):
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def is_regression_metric(metric: str) -> bool:
    return metric.upper() in {"MAE", "RMSE", "MSE", "SPEARMAN"}

def align_by_idx_or_order(y_true, y_pred, sample_idx):
    if sample_idx is None:
        return y_true, y_pred
    order = np.argsort(sample_idx)
    y_true2 = y_true[order] if y_true is not None else None
    y_pred2 = y_pred[order]
    return y_true2, y_pred2

def find_trimole_valid(task: str):
    roots = [
        ROOT / "results/model_log/fusion_inputs_valid_trimole_22tasks",
        ROOT / "results/model_log/fusion_inputs_trimole_calib_ready_fixed",
        ROOT / "results/model_log/fusion_inputs_trimole_calib_ready",
    ]
    for r in roots:
        cands = list(r.glob(f"{task}_valid_predictions.csv"))
        p = newest(cands)
        if p is not None:
            return p
    raise FileNotFoundError(f"[T] {task} valid not found")

def find_trimole_test(task: str):
    root = ROOT / "results/model_log/final_best_v4_runs" / task
    cands = list(root.rglob("test_predictions.csv"))
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[T] {task} test not found")
    return p

def find_xgb_valid(task: str):
    roots = [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid",
        ROOT / "results/model_log/fusion_inputs_valid_xgb_22tasks",
    ]
    for r in roots:
        cands = list(r.glob(f"{task}_valid_predictions.csv"))
        p = newest(cands)
        if p is not None:
            return p
    raise FileNotFoundError(f"[X] {task} valid not found")

def find_xgb_test(task: str):
    root = ROOT / "results/model_log"
    cands = list(root.rglob(f"{task}_test_predictions.csv"))
    cands = [p for p in cands if "xgb" in str(p).lower() or "baseline" in str(p).lower()]
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[X] {task} test not found")
    return p

def find_gnn_valid(task: str):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    cands = list(root.rglob("valid_predictions.csv"))
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[G] {task} valid not found")
    return p

def find_gnn_test(task: str):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    cands = list(root.rglob("test_predictions.csv"))
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[G] {task} test not found")
    return p

summary_rows = []
detail_rows = []

for task in TASKS:
    metric = OFFICIAL_METRIC[task]

    vt, tt = find_trimole_valid(task), find_trimole_test(task)
    vx, tx = find_xgb_valid(task), find_xgb_test(task)
    vg, tg = find_gnn_valid(task), find_gnn_test(task)

    yv_t, pv_t, iv_t = read_pred(vt)
    yv_x, pv_x, iv_x = read_pred(vx)
    yv_g, pv_g, iv_g = read_pred(vg)

    yt_t, pt_t, it_t = read_pred(tt)
    yt_x, pt_x, it_x = read_pred(tx)
    yt_g, pt_g, it_g = read_pred(tg)

    yv_t, pv_t = align_by_idx_or_order(yv_t, pv_t, iv_t)
    yv_x, pv_x = align_by_idx_or_order(yv_x, pv_x, iv_x)
    yv_g, pv_g = align_by_idx_or_order(yv_g, pv_g, iv_g)

    yt_t, pt_t = align_by_idx_or_order(yt_t, pt_t, it_t)
    yt_x, pt_x = align_by_idx_or_order(yt_x, pt_x, it_x)
    yt_g, pt_g = align_by_idx_or_order(yt_g, pt_g, it_g)

    if is_regression_metric(metric):
        yv_ref = yv_x if yv_x is not None else yv_g
        yt_ref = yt_x if yt_x is not None else yt_g
    else:
        if not (np.allclose(yv_t, yv_x) and np.allclose(yv_t, yv_g)):
            raise ValueError(f"{task} valid y_true mismatch across models")
        if not (np.allclose(yt_t, yt_x) and np.allclose(yt_t, yt_g)):
            raise ValueError(f"{task} test y_true mismatch across models")
        yv_ref = yv_t
        yt_ref = yt_t

    best_w = None
    best_valid = float("inf") if metric.upper() in {"MAE", "RMSE", "MSE"} else -float("inf")

    for t, x, g in WEIGHTS:
        valid_pred = t * pv_t + x * pv_x + g * pv_g
        s = score_metric(metric, yv_ref, valid_pred)
        detail_rows.append({"task": task, "metric": metric, "t": t, "x": x, "g": g, "valid_score": s})
        if better(metric, s, best_valid):
            best_valid = s
            best_w = (t, x, g)

    t, x, g = best_w
    test_pred = t * pt_t + x * pt_x + g * pt_g
    test_score = score_metric(metric, yt_ref, test_pred)

    summary_rows.append({
        "task": task,
        "metric": metric,
        "best_t": t,
        "best_x": x,
        "best_g": g,
        "best_valid_score": best_valid,
        "test_score": test_score,
        "valid_trimole": score_metric(metric, yv_ref, pv_t),
        "valid_xgb": score_metric(metric, yv_ref, pv_x),
        "valid_gnn": score_metric(metric, yv_ref, pv_g),
    })

summary = pd.DataFrame(summary_rows).sort_values("task")
detail = pd.DataFrame(detail_rows)

summary.to_csv(OUT_DIR / "txg_weight_search_summary.csv", index=False)
detail.to_csv(OUT_DIR / "txg_weight_search_detail.csv", index=False)

print(summary.to_string(index=False))
print("\nSaved:", OUT_DIR / "txg_weight_search_summary.csv")
print("Saved:", OUT_DIR / "txg_weight_search_detail.csv")
