from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
TASK = "bioavailability_ma"
OUT = ROOT / "results/model_log/bioava_fusion_try"
OUT.mkdir(parents=True, exist_ok=True)

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

# Trimole: current best
trimole_root = ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_128__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_42/run_20260416_2204/bioavailability_ma"

# XGB
xgb_valid = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/bioavailability_ma_valid_predictions.csv"
xgb_test  = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/bioavailability_ma_test_predictions.csv"

# GNN seeds
gnn_roots = {
    "seed_1": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_1/run_20260415_1614/bioavailability_ma",
    "seed_42": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_42/run_20260415_1614/bioavailability_ma",
    "seed_3407": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_3407/run_20260415_1614/bioavailability_ma",
}

tv = trimole_root / "valid_predictions.csv"
tt = trimole_root / "test_predictions.csv"

ytv, ptv = read_pred(tv)
ytt, ptt = read_pred(tt)

yxv, pxv = read_pred(xgb_valid)
yxt, pxt = read_pred(xgb_test)

rows = []

# TX
for wt in np.arange(0.0, 1.01, 0.05):
    wx = 1.0 - wt
    pv = wt * ptv + wx * pxv
    pt = wt * ptt + wx * pxt
    rows.append({
        "mode": "TX",
        "gnn_seed": None,
        "wt": round(float(wt), 2),
        "wx": round(float(wx), 2),
        "wg": 0.0,
        "valid_auc": roc_auc_score(ytv, pv),
        "test_auc": roc_auc_score(ytt, pt),
    })

# TXG
for gname, groot in gnn_roots.items():
    gv = groot / "valid_predictions.csv"
    gt = groot / "test_predictions.csv"
    if not gv.exists() or not gt.exists():
        continue

    ygv, pgv = read_pred(gv)
    ygt, pgt = read_pred(gt)

    for wt in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.01 - wt, 0.1):
            wg = round(1.0 - wt - wx, 10)
            if wg < 0:
                continue
            pv = wt * ptv + wx * pxv + wg * pgv
            pt = wt * ptt + wx * pxt + wg * pgt
            rows.append({
                "mode": "TXG",
                "gnn_seed": gname,
                "wt": round(float(wt), 2),
                "wx": round(float(wx), 2),
                "wg": round(float(wg), 2),
                "valid_auc": roc_auc_score(ytv, pv),
                "test_auc": roc_auc_score(ytt, pt),
            })

df = pd.DataFrame(rows).sort_values(["test_auc", "valid_auc"], ascending=False).reset_index(drop=True)
out_csv = OUT / "bioava_fusion_summary.csv"
df.to_csv(out_csv, index=False)
print(df.head(30).to_string(index=False))
print("\nsaved:", out_csv)
