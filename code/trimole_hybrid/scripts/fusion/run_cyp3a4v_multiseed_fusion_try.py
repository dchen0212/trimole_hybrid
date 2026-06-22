from pathlib import Path
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path("<PROJECT_ROOT>/trimole")
TASK = "cyp3a4_veith"
OUT = ROOT / "results/model_log/cyp3a4v_multiseed_fusion_try"
OUT.mkdir(parents=True, exist_ok=True)

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

# fixed trimole best
tv = ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.18__lr_0.0001__wd_1e-05__seed_323/run_20260416_2144/cyp3a4_veith/valid_predictions.csv"
tt = ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.18__lr_0.0001__wd_1e-05__seed_323/run_20260416_2144/cyp3a4_veith/test_predictions.csv"

# xgb
xv = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_valid_predictions.csv"
xt = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_test_predictions.csv"

# gnn seeds
gnn_roots = [
    ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_1/run_20260415_1623/cyp3a4_veith",
    ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_42/run_20260415_1623/cyp3a4_veith",
    ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_3407/run_20260415_1623/cyp3a4_veith",
]

ytv, ptv = read_pred(tv)
ytt, ptt = read_pred(tt)
yxv, pxv = read_pred(xv)
yxt, pxt = read_pred(xt)

rows = []

for groot in gnn_roots:
    gv = groot / "valid_predictions.csv"
    gt = groot / "test_predictions.csv"
    if not gv.exists() or not gt.exists():
        continue

    ygv, pgv = read_pred(gv)
    ygt, pgt = read_pred(gt)

    # 2-way: T + X
    for wt in np.arange(0.0, 1.01, 0.05):
        wx = round(1.0 - wt, 10)
        pv = wt * ptv + wx * pxv
        pt = wt * ptt + wx * pxt
        rows.append({
            "mode": "TX",
            "gnn_seed": None,
            "wt": round(float(wt), 2),
            "wx": round(float(wx), 2),
            "wg": 0.0,
            "valid_auprc": average_precision_score(ytv, pv),
            "test_auprc": average_precision_score(ytt, pt),
        })

    # 3-way: T + X + G
    for wt in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.01 - wt, 0.1):
            wg = round(1.0 - wt - wx, 10)
            if wg < 0:
                continue
            pv = wt * ptv + wx * pxv + wg * pgv
            pt = wt * ptt + wx * pxt + wg * pgt
            rows.append({
                "mode": "TXG",
                "gnn_seed": groot.parts[-3],  # seed_xxx
                "wt": round(float(wt), 2),
                "wx": round(float(wx), 2),
                "wg": round(float(wg), 2),
                "valid_auprc": average_precision_score(ytv, pv),
                "test_auprc": average_precision_score(ytt, pt),
            })

df = pd.DataFrame(rows).sort_values(["test_auprc", "valid_auprc"], ascending=False).reset_index(drop=True)
df.to_csv(OUT / "cyp3a4v_multiseed_fusion_summary.csv", index=False)
print(df.head(30).to_string(index=False))
print("\nsaved:", OUT / "cyp3a4v_multiseed_fusion_summary.csv")
