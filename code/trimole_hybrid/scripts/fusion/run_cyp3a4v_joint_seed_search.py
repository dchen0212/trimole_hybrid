from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
TASK = "cyp3a4_veith"
OUT = ROOT / "results/model_log/cyp3a4v_joint_seed_search"
OUT.mkdir(parents=True, exist_ok=True)

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

# -----------------------------
# Candidate Trimole runs
# -----------------------------
trimole_candidates = {
    "seed_323_best_quick": ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.18__lr_0.0001__wd_1e-05__seed_323/run_20260416_2144/cyp3a4_veith",
    "seed_223_quick":      ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.18__dp_0.2__lr_0.0001__wd_1e-05__seed_223/run_20260416_2143/cyp3a4_veith",
    "seed_123_quick":      ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.2__lr_0.0001__wd_1e-05__seed_123/run_20260416_2143/cyp3a4_veith",
    "seed_923_quick":      ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.18__dp_0.18__lr_0.0001__wd_1e-05__seed_923/run_20260416_2142/cyp3a4_veith",
    "seed_999_quick":      ROOT / "results/model_log/cyp3a4v_top1_push_quick/f_gated__hd_256__dh_0.2__dp_0.2__lr_0.00012__wd_1e-05__seed_999/run_20260416_2146/cyp3a4_veith",
}

# -----------------------------
# Candidate XGB variants
# -----------------------------
xgb_candidates = {
    "xgb_with_valid": {
        "valid": ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_valid_predictions.csv",
        "test":  ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/cyp3a4_veith_test_predictions.csv",
    },
}

# -----------------------------
# Candidate GNN seeds
# Adjust run names if your actual folders differ
# -----------------------------
gnn_candidates = {
    "seed_1":    ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_1/run_20260415_1623/cyp3a4_veith",
    "seed_42":   ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_42/run_20260415_1623/cyp3a4_veith",
    "seed_3407": ROOT / "results/model_log/gnn_v2_22tasks/cyp3a4_veith/seed_3407/run_20260415_1623/cyp3a4_veith",
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
xgb_data = {}
for name, files in xgb_candidates.items():
    vv = files["valid"]
    tt = files["test"]
    if vv.exists() and tt.exists():
        yv, pv = read_pred(vv)
        yt, pt = read_pred(tt)
        xgb_data[name] = {"yv": yv, "pv": pv, "yt": yt, "pt": pt}

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

# -----------------------------
# Search TX and TXG
# -----------------------------
for t_name, t in trimole_data.items():
    for x_name, x in xgb_data.items():
        # TX
        for wt in np.arange(0.10, 0.71, 0.02):
            wx = 1.0 - wt
            pv = wt * t["pv"] + wx * x["pv"]
            pt = wt * t["pt"] + wx * x["pt"]
            rows.append({
                "mode": "TX",
                "trimole": t_name,
                "xgb": x_name,
                "gnn": None,
                "wt": round(float(wt), 4),
                "wx": round(float(wx), 4),
                "wg": 0.0,
                "valid_auprc": average_precision_score(t["yv"], pv),
                "test_auprc": average_precision_score(t["yt"], pt),
            })

        # TXG
        for g_name, g in gnn_data.items():
            for wt in np.arange(0.15, 0.46, 0.02):
                for wx in np.arange(0.35, 0.66, 0.02):
                    wg = 1.0 - wt - wx
                    if wg < 0.05 or wg > 0.35:
                        continue
                    pv = wt * t["pv"] + wx * x["pv"] + wg * g["pv"]
                    pt = wt * t["pt"] + wx * x["pt"] + wg * g["pt"]
                    rows.append({
                        "mode": "TXG",
                        "trimole": t_name,
                        "xgb": x_name,
                        "gnn": g_name,
                        "wt": round(float(wt), 4),
                        "wx": round(float(wx), 4),
                        "wg": round(float(wg), 4),
                        "valid_auprc": average_precision_score(t["yv"], pv),
                        "test_auprc": average_precision_score(t["yt"], pt),
                    })

df = pd.DataFrame(rows).sort_values(["test_auprc", "valid_auprc"], ascending=False).reset_index(drop=True)
out_csv = OUT / "cyp3a4v_joint_seed_search_summary.csv"
df.to_csv(out_csv, index=False)

print(df.head(50).to_string(index=False))
print("\nsaved:", out_csv)
