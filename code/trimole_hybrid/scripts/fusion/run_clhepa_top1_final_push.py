from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT = ROOT / "results/model_log/clhepa_top1_final_push"
OUT.mkdir(parents=True, exist_ok=True)

TASK = "clearance_hepatocyte_az"

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
    pred_col = next((cols[c] for c in ["y_pred", "pred", "prediction", "predictions", "score", "probability", "prob", "y_prob"] if c in cols), None)
    if y_col is None or pred_col is None:
        raise ValueError(f"Cannot parse {p}, cols={list(df.columns)}")
    return df[y_col].to_numpy(), df[pred_col].to_numpy()

def metric(y, p):
    return spearmanr(y, p).statistic

def find_trimole_valid(task):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    cands = []
    for rp in root.glob("run_*"):
        if (rp / task).exists():
            cands.extend((rp / task).rglob("valid_predictions.csv"))
    return newest(cands)

def find_trimole_test(task):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    cands = []
    for rp in root.glob("run_*"):
        if (rp / task).exists():
            cands.extend((rp / task).rglob("test_predictions.csv"))
    return newest(cands)

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

def make_grid():
    vals_t = np.arange(0.50, 0.86 + 1e-9, 0.01)
    vals_x = np.arange(0.00, 0.26 + 1e-9, 0.01)
    out = []
    for t, x in itertools.product(vals_t, vals_x):
        g = round(1.0 - t - x, 10)
        if g < -1e-9 or g > 0.50:
            continue
        out.append((round(float(t),2), round(float(x),2), round(float(g),2)))
    return sorted(set(out))

tv, tt = find_trimole_valid(TASK), find_trimole_test(TASK)
xv, xt = find_xgb_valid(TASK), find_xgb_test(TASK)
gv, gt = find_gnn_valid(TASK), find_gnn_test(TASK)

y_tv, p_tv = read_pred_file(tv)
y_xv, p_xv = read_pred_file(xv)
y_gv, p_gv = read_pred_file(gv)

y_tt, p_tt = read_pred_file(tt)
y_xt, p_xt = read_pred_file(xt)
y_gt, p_gt = read_pred_file(gt)

f_t = IsotonicRegression(out_of_bounds="clip").fit(p_tv, y_tv)
f_x = IsotonicRegression(out_of_bounds="clip").fit(p_xv, y_xv)
f_g = IsotonicRegression(out_of_bounds="clip").fit(p_gv, y_gv)

cv_t, cv_x, cv_g = f_t.predict(p_tv), f_x.predict(p_xv), f_g.predict(p_gv)
ct_t, ct_x, ct_g = f_t.predict(p_tt), f_x.predict(p_xt), f_g.predict(p_gt)

best = None
for wt, wx, wg in make_grid():
    pv = wt*cv_t + wx*cv_x + wg*cv_g
    vv = metric(y_tv, pv)
    if best is None or vv > best["best_valid_score"]:
        pt = wt*ct_t + wx*ct_x + wg*ct_g
        ts = metric(y_tt, pt)
        best = {
            "task": TASK,
            "metric": "Spearman",
            "best_t": wt,
            "best_x": wx,
            "best_g": wg,
            "best_valid_score": vv,
            "test_score": ts,
        }

df = pd.DataFrame([best])
df.to_csv(OUT / "clhepa_top1_final_push.csv", index=False)
print(df.to_string(index=False))
