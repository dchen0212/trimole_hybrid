from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_DIR = ROOT / "results/model_log/clhepa_txg_refine_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "clearance_hepatocyte_az"
CENTER = (0.772727, 0.181818, 0.045455)
STEP = 0.02

def local_triplets(center, step=0.02):
    vals_t = [round(center[0] + d, 10) for d in (-2*step, -step, 0.0, step, 2*step)]
    vals_x = [round(center[1] + d, 10) for d in (-2*step, -step, 0.0, step, 2*step)]
    vals_g = [round(center[2] + d, 10) for d in (-2*step, -step, 0.0, step, 2*step)]
    out = []
    for t, x, g in itertools.product(vals_t, vals_x, vals_g):
        if min(t, x, g) < 0:
            continue
        s = t + x + g
        if s <= 0:
            continue
        t, x, g = round(t / s, 10), round(x / s, 10), round(g / s, 10)
        if (t, x, g) not in out:
            out.append((t, x, g))
    return out

def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def read_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true = df[cols["y_true"]].to_numpy()
    y_pred = df[cols["y_pred"]].to_numpy() if "y_pred" in cols else df[cols["y_prob"]].to_numpy()
    sample_idx = df[cols["sample_idx"]].to_numpy() if "sample_idx" in cols else None
    return y_true, y_pred, sample_idx

def align(y_true, y_pred, sample_idx):
    if sample_idx is None:
        return y_true, y_pred
    order = np.argsort(sample_idx)
    return y_true[order], y_pred[order]

def newest(cands):
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

vt = ROOT / "results/model_log/fusion_inputs_valid_trimole_22tasks/clearance_hepatocyte_az_valid_predictions.csv"
vx = ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid/clearance_hepatocyte_az_valid_predictions.csv"
vg = newest(list((ROOT / "results/model_log/gnn_v2_22tasks/clearance_hepatocyte_az").rglob("valid_predictions.csv")))

tt = newest(list((ROOT / "results/model_log/final_best_v4_runs/clearance_hepatocyte_az").rglob("test_predictions.csv")))
tx = newest([p for p in (ROOT / "results/model_log").rglob("clearance_hepatocyte_az_test_predictions.csv") if "xgb" in str(p).lower() or "baseline" in str(p).lower()])
tg = newest(list((ROOT / "results/model_log/gnn_v2_22tasks/clearance_hepatocyte_az").rglob("test_predictions.csv")))

yv_t, pv_t, iv_t = read_pred(vt)
yv_x, pv_x, iv_x = read_pred(vx)
yv_g, pv_g, iv_g = read_pred(vg)

yt_t, pt_t, it_t = read_pred(tt)
yt_x, pt_x, it_x = read_pred(tx)
yt_g, pt_g, it_g = read_pred(tg)

yv_t, pv_t = align(yv_t, pv_t, iv_t)
yv_x, pv_x = align(yv_x, pv_x, iv_x)
yv_g, pv_g = align(yv_g, pv_g, iv_g)

yt_t, pt_t = align(yt_t, pt_t, it_t)
yt_x, pt_x = align(yt_x, pt_x, it_x)
yt_g, pt_g = align(yt_g, pt_g, it_g)

yv_ref = yv_x
yt_ref = yt_x

rows = []
best_valid = -float("inf")
best_w = None

grid = local_triplets(CENTER, STEP)
for i, (t, x, g) in enumerate(grid, 1):
    valid_pred = t * pv_t + x * pv_x + g * pv_g
    s = float(spearman_corr(yv_ref, valid_pred))
    rows.append({"t": t, "x": x, "g": g, "valid_spearman": s})
    if s > best_valid:
        best_valid = s
        best_w = (t, x, g)
    if i % 20 == 0 or i == len(grid):
        print(f"[{i}/{len(grid)}] best_valid_spearman={best_valid:.6f} best_w={best_w}", flush=True)

t, x, g = best_w
test_pred = t * pt_t + x * pt_x + g * pt_g
test_spearman = float(spearman_corr(yt_ref, test_pred))

detail = pd.DataFrame(rows).sort_values("valid_spearman", ascending=False)
summary = pd.DataFrame([{
    "task": TASK,
    "best_t": t,
    "best_x": x,
    "best_g": g,
    "best_valid_spearman": best_valid,
    "test_spearman": test_spearman,
    "base_center_t": CENTER[0],
    "base_center_x": CENTER[1],
    "base_center_g": CENTER[2],
}])

detail.to_csv(OUT_DIR / "clhepa_txg_refine_v2_detail.csv", index=False)
summary.to_csv(OUT_DIR / "clhepa_txg_refine_v2_summary.csv", index=False)

print("\n=== SUMMARY ===")
print(summary.to_string(index=False))
print("\nSaved:", OUT_DIR / "clhepa_txg_refine_v2_summary.csv")
print("Saved:", OUT_DIR / "clhepa_txg_refine_v2_detail.csv")
