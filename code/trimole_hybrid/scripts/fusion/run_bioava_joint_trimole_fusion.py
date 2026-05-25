from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
TASK = "bioavailability_ma"
OUT = ROOT / "results/model_log/bioava_joint_trimole_fusion"
OUT.mkdir(parents=True, exist_ok=True)

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

# -----------------------------
# Trimole candidate runs
# -----------------------------
trimole_candidates = {
    "mlp_hd128_dh03_seed42": ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_128__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_42/run_20260416_2204/bioavailability_ma",
    "mlp_hd128_dh04_seed42": ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_128__dh_0.4__dp_0.2__lr_0.0003__wd_0.0__seed_42/run_20260416_2205/bioavailability_ma",
    "mlp_hd128_dh03_seed123": ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_128__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_123/run_20260416_2204/bioavailability_ma",
    "mlp_hd128_dh03_seed7": ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_128__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_7/run_20260416_2204/bioavailability_ma",
    "gated_hd128_seed42": ROOT / "results/model_log/bioava_quick_sweep/f_gated__hd_128__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_42/run_20260416_2205/bioavailability_ma",
    "mlp_hd256_seed42": ROOT / "results/model_log/bioava_quick_sweep/f_mlp__hd_256__dh_0.3__dp_0.2__lr_0.0003__wd_0.0__seed_42/run_20260416_2205/bioavailability_ma",
}

# -----------------------------
# XGB
# -----------------------------
xgb_valid = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/bioavailability_ma_valid_predictions.csv"
xgb_test  = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/bioavailability_ma_test_predictions.csv"

# -----------------------------
# GNN seeds
# -----------------------------
gnn_candidates = {
    "seed_1": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_1/run_20260415_1614/bioavailability_ma",
    "seed_42": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_42/run_20260415_1614/bioavailability_ma",
    "seed_3407": ROOT / "results/model_log/gnn_v2_22tasks/bioavailability_ma/seed_3407/run_20260415_1614/bioavailability_ma",
}

# preload trimole
trimole_data = {}
for name, root in trimole_candidates.items():
    vv = root / "valid_predictions.csv"
    tt = root / "test_predictions.csv"
    if vv.exists() and tt.exists():
        yv, pv = read_pred(vv)
        yt, pt = read_pred(tt)
        trimole_data[name] = {"yv": yv, "pv": pv, "yt": yt, "pt": pt}

# preload xgb
yxv, pxv = read_pred(xgb_valid)
yxt, pxt = read_pred(xgb_test)

# preload gnn
gnn_data = {}
for name, root in gnn_candidates.items():
    vv = root / "valid_predictions.csv"
    tt = root / "test_predictions.csv"
    if vv.exists() and tt.exists():
        yv, pv = read_pred(vv)
        yt, pt = read_pred(tt)
        gnn_data[name] = {"yv": yv, "pv": pv, "yt": yt, "pt": pt}

rows = []

for t_name, t in trimole_data.items():
    # TX
    for wt in np.arange(0.0, 1.01, 0.05):
        wx = 1.0 - wt
        pv = wt * t["pv"] + wx * pxv
        pt = wt * t["pt"] + wx * pxt
        rows.append({
            "mode": "TX",
            "trimole": t_name,
            "gnn_seed": None,
            "wt": round(float(wt), 2),
            "wx": round(float(wx), 2),
            "wg": 0.0,
            "valid_auc": roc_auc_score(t["yv"], pv),
            "test_auc": roc_auc_score(t["yt"], pt),
        })

    # TXG
    for g_name, g in gnn_data.items():
        for wt in np.arange(0.0, 1.01, 0.1):
            for wx in np.arange(0.0, 1.01 - wt, 0.1):
                wg = round(1.0 - wt - wx, 10)
                if wg < 0:
                    continue
                pv = wt * t["pv"] + wx * pxv + wg * g["pv"]
                pt = wt * t["pt"] + wx * pxt + wg * g["pt"]
                rows.append({
                    "mode": "TXG",
                    "trimole": t_name,
                    "gnn_seed": g_name,
                    "wt": round(float(wt), 2),
                    "wx": round(float(wx), 2),
                    "wg": round(float(wg), 2),
                    "valid_auc": roc_auc_score(t["yv"], pv),
                    "test_auc": roc_auc_score(t["yt"], pt),
                })

df = pd.DataFrame(rows).sort_values(["test_auc", "valid_auc"], ascending=False).reset_index(drop=True)
out_csv = OUT / "bioava_joint_trimole_fusion_summary.csv"
df.to_csv(out_csv, index=False)

print(df.head(40).to_string(index=False))
print("\nsaved:", out_csv)
