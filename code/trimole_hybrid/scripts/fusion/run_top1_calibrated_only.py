from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import itertools

ROOT = Path("<PROJECT_ROOT>/trimole")
OUT = ROOT / "results/model_log/top1_calibrated_only"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "bbb_martins": {"metric": "AUROC"},
    "clearance_hepatocyte_az": {"metric": "Spearman"},
}

def newest(cands):
    cands = [Path(x) for x in cands if x and Path(x).exists()]
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0] if cands else None

def first_existing(cands):
    for c in cands:
        p = Path(c)
        if p.exists():
            return p
    return None

def read_pred_file(p):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next((cols[c] for c in ["y_true", "label", "labels", "target", "y"] if c in cols), None)
    pred_col = next((cols[c] for c in ["y_prob", "pred", "prediction", "predictions", "y_pred", "score", "prob", "probability"] if c in cols), None)
    if y_col is None or pred_col is None:
        raise ValueError(f"Cannot parse {p}, cols={list(df.columns)}")
    return df[y_col].to_numpy(), df[pred_col].to_numpy()

def metric_score(metric, y, pred):
    if metric == "AUROC":
        return roc_auc_score(y, pred)
    if metric == "Spearman":
        return spearmanr(y, pred).statistic
    raise ValueError(metric)

def better(metric, a, b):
    return a > b

def find_trimole_valid(task):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    cands = []
    for rp in root.glob("run_*"):
        if (rp / task).exists():
            cands.extend((rp / task).rglob("valid_predictions.csv"))
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(task)
    return p

def find_trimole_test(task):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    cands = []
    for rp in root.glob("run_*"):
        if (rp / task).exists():
            cands.extend((rp / task).rglob("test_predictions.csv"))
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(task)
    return p

def find_xgb_valid(task):
    return first_existing([
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "valid_predictions.csv",
    ])

def find_xgb_test(task):
    return first_existing([
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_test_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_test_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_test_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "test_predictions.csv",
    ])

def find_gnn_valid(task):
    return newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("valid_predictions.csv")))

def find_gnn_test(task):
    return newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("test_predictions.csv")))

def calibrate_binary(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(p.reshape(-1, 1), y)
    return lambda x: lr.predict_proba(np.clip(x, eps, 1-eps).reshape(-1, 1))[:, 1]

def calibrate_regression(y, p):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return lambda x: iso.predict(x)

def make_grid(step=0.02):
    vals = np.arange(0, 1.0001, step)
    out = []
    for t, x in itertools.product(vals, vals):
        g = round(1.0 - t - x, 10)
        if g < -1e-9 or g > 1 + 1e-9:
            continue
        out.append((round(float(t),2), round(float(x),2), round(float(g),2)))
    return sorted(set(out))

rows = []
for task, info in TASKS.items():
    metric = info["metric"]
    tv, tt = find_trimole_valid(task), find_trimole_test(task)
    xv, xt = find_xgb_valid(task), find_xgb_test(task)
    gv, gt = find_gnn_valid(task), find_gnn_test(task)

    y_tv, p_tv = read_pred_file(tv)
    y_xv, p_xv = read_pred_file(xv)
    y_gv, p_gv = read_pred_file(gv)

    y_tt, p_tt = read_pred_file(tt)
    y_xt, p_xt = read_pred_file(xt)
    y_gt, p_gt = read_pred_file(gt)

    if metric == "AUROC":
        f_t = calibrate_binary(y_tv, p_tv)
        f_x = calibrate_binary(y_xv, p_xv)
        f_g = calibrate_binary(y_gv, p_gv)
    else:
        f_t = calibrate_regression(y_tv, p_tv)
        f_x = calibrate_regression(y_xv, p_xv)
        f_g = calibrate_regression(y_gv, p_gv)

    cv_t, cv_x, cv_g = f_t(p_tv), f_x(p_xv), f_g(p_gv)
    ct_t, ct_x, ct_g = f_t(p_tt), f_x(p_xt), f_g(p_gt)

    best = None
    for wt, wx, wg in make_grid():
        pv = wt*cv_t + wx*cv_x + wg*cv_g
        vv = metric_score(metric, y_tv, pv)
        if best is None or better(metric, vv, best["best_valid_score"]):
            pt = wt*ct_t + wx*ct_x + wg*ct_g
            ts = metric_score(metric, y_tt, pt)
            best = {
                "task": task, "metric": metric,
                "best_t": wt, "best_x": wx, "best_g": wg,
                "best_valid_score": vv, "test_score": ts
            }

    rows.append(best)
    print(f"[done] {task} metric={metric} best=({best['best_t']},{best['best_x']},{best['best_g']}) valid={best['best_valid_score']:.6f} test={best['test_score']:.6f}")

pd.DataFrame(rows).to_csv(OUT / "top1_calibrated_summary.csv", index=False)
