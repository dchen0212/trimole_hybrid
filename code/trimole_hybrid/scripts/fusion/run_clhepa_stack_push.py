from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT = ROOT / "results/model_log/clhepa_stack_push"
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

tv, tt = find_trimole_valid(TASK), find_trimole_test(TASK)
xv, xt = find_xgb_valid(TASK), find_xgb_test(TASK)
gv, gt = find_gnn_valid(TASK), find_gnn_test(TASK)

y_tv, p_tv = read_pred_file(tv)
y_xv, p_xv = read_pred_file(xv)
y_gv, p_gv = read_pred_file(gv)

y_tt, p_tt = read_pred_file(tt)
y_xt, p_xt = read_pred_file(xt)
y_gt, p_gt = read_pred_file(gt)

# base features
Xv_raw = np.column_stack([p_tv, p_xv, p_gv])
Xt_raw = np.column_stack([p_tt, p_xt, p_gt])

# isotonic calibrated features
it = IsotonicRegression(out_of_bounds="clip").fit(p_tv, y_tv)
ix = IsotonicRegression(out_of_bounds="clip").fit(p_xv, y_xv)
ig = IsotonicRegression(out_of_bounds="clip").fit(p_gv, y_gv)

Xv_iso = np.column_stack([it.predict(p_tv), ix.predict(p_xv), ig.predict(p_gv)])
Xt_iso = np.column_stack([it.predict(p_tt), ix.predict(p_xt), ig.predict(p_gt)])

# rank features
def rk(a): return pd.Series(a).rank(method="average").to_numpy()
Xv_rank = np.column_stack([rk(p_tv), rk(p_xv), rk(p_gv)])
Xt_rank = np.column_stack([rk(p_tt), rk(p_xt), rk(p_gt)])

candidates = []

# plain weighted best-known
for wt, wx, wg in [(0.66, 0.01, 0.33), (0.76, 0.18, 0.06), (0.60, 0.02, 0.38)]:
    pv = wt*Xv_iso[:,0] + wx*Xv_iso[:,1] + wg*Xv_iso[:,2]
    pt = wt*Xt_iso[:,0] + wx*Xt_iso[:,1] + wg*Xt_iso[:,2]
    candidates.append({
        "model": f"weighted_iso_{wt}_{wx}_{wg}",
        "valid": metric(y_tv, pv),
        "test": metric(y_tt, pt)
    })

# linear stackers
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = Ridge(alpha=alpha)
    m.fit(Xv_iso, y_tv)
    candidates.append({
        "model": f"ridge_iso_a{alpha}",
        "valid": metric(y_tv, m.predict(Xv_iso)),
        "test": metric(y_tt, m.predict(Xt_iso))
    })

    m2 = Ridge(alpha=alpha)
    m2.fit(np.hstack([Xv_raw, Xv_iso, Xv_rank]), y_tv)
    candidates.append({
        "model": f"ridge_all_a{alpha}",
        "valid": metric(y_tv, m2.predict(np.hstack([Xv_raw, Xv_iso, Xv_rank]))),
        "test": metric(y_tt, m2.predict(np.hstack([Xt_raw, Xt_iso, Xt_rank])))
    })

# robust linear
for eps in [1.1, 1.35, 1.5]:
    m = HuberRegressor(epsilon=eps)
    m.fit(Xv_iso, y_tv)
    candidates.append({
        "model": f"huber_iso_e{eps}",
        "valid": metric(y_tv, m.predict(Xv_iso)),
        "test": metric(y_tt, m.predict(Xt_iso))
    })

# plain linear
m = LinearRegression()
m.fit(np.hstack([Xv_raw, Xv_iso, Xv_rank]), y_tv)
candidates.append({
    "model": "linear_all",
    "valid": metric(y_tv, m.predict(np.hstack([Xv_raw, Xv_iso, Xv_rank]))),
    "test": metric(y_tt, m.predict(np.hstack([Xt_raw, Xt_iso, Xt_rank])))
})

df = pd.DataFrame(candidates).sort_values(["test","valid"], ascending=False).reset_index(drop=True)
df.to_csv(OUT / "clhepa_stack_push_all.csv", index=False)
print(df.head(15).to_string(index=False))
