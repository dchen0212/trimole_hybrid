from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

ROOT = Path("<PROJECT_ROOT>/trimole")
OUT_DIR = ROOT / "results/model_log/txg_refine_v2_5tasks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "caco2_wang": ("MAE", (0.60, 0.20, 0.20)),
    "clearance_microsome_az": ("Spearman", (0.40, 0.60, 0.00)),
    "cyp2c9_veith": ("AUPRC", (0.20, 0.10, 0.70)),
    "cyp3a4_veith": ("AUPRC", (0.10, 0.50, 0.40)),
    "ppbr_az": ("MAE", (0.40, 0.00, 0.60)),
}

def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def score(metric, y_true, y_pred):
    if metric == "AUROC":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "AUPRC":
        return float(average_precision_score(y_true, y_pred))
    if metric == "MAE":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "Spearman":
        return float(spearman_corr(y_true, y_pred))
    raise ValueError(metric)

def better(metric, a, b):
    return a < b if metric == "MAE" else a > b

def read_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true = df[cols["y_true"]].to_numpy()
    y_pred = df[cols["y_prob"]].to_numpy() if "y_prob" in cols else df[cols["y_pred"]].to_numpy()
    if "sample_idx" in cols:
        idx = df[cols["sample_idx"]].to_numpy()
        order = np.argsort(idx)
        y_true = y_true[order]
        y_pred = y_pred[order]
    return y_true, y_pred

def newest(cands):
    cands = [p for p in cands if p.exists()]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def find_trimole_valid(task):
    for p in [
        ROOT / "results/model_log/fusion_inputs_valid_trimole_22tasks" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_trimole_calib_ready_fixed" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_trimole_calib_ready" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/trimole" / task / "calibrated_valid_predictions.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(task)

def find_xgb_valid(task):
    for p in [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_valid_xgb_22tasks" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_valid_predictions.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(task)

def find_gnn_valid(task):
    p = newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("valid_predictions.csv")))
    if p is None:
        raise FileNotFoundError(task)
    return p

def find_trimole_test(task):
    run_dir = newest([p for p in (ROOT / "results/model_log/benchmark_clean_rerun").glob("run_*") if p.is_dir()])
    cands = [run_dir / task / "test_predictions.csv", *list(run_dir.rglob(f"{task}/test_predictions.csv"))]
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(task)
    return p

def find_xgb_test(task):
    p = newest([p for p in ROOT.rglob(f"{task}_test_predictions.csv") if ("xgb" in str(p).lower() or "baseline" in str(p).lower() or "maplight" in str(p).lower())])
    if p is None:
        raise FileNotFoundError(task)
    return p

def find_gnn_test(task):
    p = newest(list((ROOT / "results/model_log/gnn_v2_22tasks" / task).rglob("test_predictions.csv")))
    if p is None:
        raise FileNotFoundError(task)
    return p

def local_grid(center, radius=0.08, step=0.02):
    ct, cx, cg = center
    vals_t = np.round(np.arange(max(0, ct-radius), min(1, ct+radius)+1e-9, step), 6)
    vals_x = np.round(np.arange(max(0, cx-radius), min(1, cx+radius)+1e-9, step), 6)
    out = []
    for t in vals_t:
        for x in vals_x:
            g = round(1 - t - x, 6)
            if g < 0 or g > 1:
                continue
            if abs(g - cg) <= radius + 1e-9:
                out.append((float(t), float(x), float(g)))
    return out

summary = []
detail = []

for task, (metric, center) in TARGETS.items():
    yv_t, pv_t = read_pred(find_trimole_valid(task))
    yv_x, pv_x = read_pred(find_xgb_valid(task))
    yv_g, pv_g = read_pred(find_gnn_valid(task))

    yt_t, pt_t = read_pred(find_trimole_test(task))
    yt_x, pt_x = read_pred(find_xgb_test(task))
    yt_g, pt_g = read_pred(find_gnn_test(task))

    yv_ref, yt_ref = yv_x, yt_x

    best_score = float("inf") if metric == "MAE" else -1e18
    best_w = center

    grid = local_grid(center)
    for i, (t, x, g) in enumerate(grid, 1):
        valid_pred = t*pv_t + x*pv_x + g*pv_g
        s = score(metric, yv_ref, valid_pred)
        detail.append({"task": task, "metric": metric, "t": t, "x": x, "g": g, "valid_score": s})
        if better(metric, s, best_score):
            best_score = s
            best_w = (t, x, g)

    t, x, g = best_w
    test_pred = t*pt_t + x*pt_x + g*pt_g
    test_score = score(metric, yt_ref, test_pred)

    summary.append({
        "task": task, "metric": metric,
        "best_t": t, "best_x": x, "best_g": g,
        "best_valid_score": best_score,
        "test_score": test_score,
        "center_t": center[0], "center_x": center[1], "center_g": center[2],
    })
    print(f"[done] {task} best_w={best_w} valid={best_score:.6f} test={test_score:.6f}", flush=True)

summary_df = pd.DataFrame(summary).sort_values("task")
detail_df = pd.DataFrame(detail)
summary_df.to_csv(OUT_DIR / "txg_refine_v2_5tasks_summary.csv", index=False)
detail_df.to_csv(OUT_DIR / "txg_refine_v2_5tasks_detail.csv", index=False)

print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))
