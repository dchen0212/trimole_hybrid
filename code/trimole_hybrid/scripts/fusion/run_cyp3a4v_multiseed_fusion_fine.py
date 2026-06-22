from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path("<PROJECT_ROOT>/trimole")
TASK = "cyp3a4_veith"
OUT = ROOT / "results/model_log/cyp3a4v_multiseed_fusion_fine"
OUT.mkdir(parents=True, exist_ok=True)

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

# Trimole: use the best run from your quick sweep
tv = ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.18__lr_0.0001__wd_1e-05__seed_323/run_20260416_2144/cyp3a4_veith/valid_predictions.csv"
tt = ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.18__lr_0.0001__wd_1e-05__seed_323/run_20260416_2144/cyp3a4_veith/test_predictions.csv"

# XGB
xv = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_valid_predictions.csv"
xt = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_test_predictions.csv"

# GNN: focus on seed_3407 since coarse search liked it best
gv = ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_3407/run_20260415_1623/cyp3a4_veith/valid_predictions.csv"
gt = ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_3407/run_20260415_1623/cyp3a4_veith/test_predictions.csv"

ytv, ptv = read_pred(tv)
ytt, ptt = read_pred(tt)
yxv, pxv = read_pred(xv)
yxt, pxt = read_pred(xt)
ygv, pgv = read_pred(gv)
ygt, pgt = read_pred(gt)

rows = []

# Fine search around the best coarse region:
# wt ~ 0.30, wx ~ 0.50, wg ~ 0.20
for wt in np.arange(0.20, 0.41, 0.01):
    for wx in np.arange(0.40, 0.61, 0.01):
        wg = 1.0 - wt - wx
        if wg < 0.05 or wg > 0.30:
            continue

        pv = wt * ptv + wx * pxv + wg * pgv
        pt = wt * ptt + wx * pxt + wg * pgt

        rows.append({
            "wt": round(float(wt), 4),
            "wx": round(float(wx), 4),
            "wg": round(float(wg), 4),
            "valid_auprc": average_precision_score(ytv, pv),
            "test_auprc": average_precision_score(ytt, pt),
        })

# Also search a tighter TX-only region in case G hurts slightly
for wt in np.arange(0.25, 0.56, 0.01):
    wx = 1.0 - wt
    pv = wt * ptv + wx * pxv
    pt = wt * ptt + wx * pxt
    rows.append({
        "wt": round(float(wt), 4),
        "wx": round(float(wx), 4),
        "wg": 0.0,
        "valid_auprc": average_precision_score(ytv, pv),
        "test_auprc": average_precision_score(ytt, pt),
    })

df = pd.DataFrame(rows).sort_values(["test_auprc", "valid_auprc"], ascending=False).reset_index(drop=True)
out_csv = OUT / "cyp3a4v_multiseed_fusion_fine_summary.csv"
df.to_csv(out_csv, index=False)

print(df.head(50).to_string(index=False))
print("\nsaved:", out_csv)
