from __future__ import annotations

from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
OUT_DIR = ROOT / "results/model_log/txg_weight_search_benchmark_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "ames","bbb_martins","bioavailability_ma","caco2_wang",
    "clearance_hepatocyte_az","clearance_microsome_az",
    "cyp2c9_substrate_carbonmangels","cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels","cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels","cyp3a4_veith",
    "dili","half_life_obach","herg","hia_hou","ld50_zhu",
    "lipophilicity_astrazeneca","pgp_broccatelli","ppbr_az",
    "solubility_aqsoldb","vdss_lombardo",
]

OFFICIAL_METRIC = {
    "ames": "AUROC",
    "bbb_martins": "AUROC",
    "bioavailability_ma": "AUROC",
    "caco2_wang": "MAE",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "cyp3a4_veith": "AUPRC",
    "dili": "AUROC",
    "half_life_obach": "Spearman",
    "herg": "AUROC",
    "hia_hou": "AUROC",
    "ld50_zhu": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "pgp_broccatelli": "AUROC",
    "ppbr_az": "MAE",
    "solubility_aqsoldb": "MAE",
    "vdss_lombardo": "Spearman",
}

def all_weight_triplets(step=0.1):
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    out = []
    for t, x, g in itertools.product(vals, vals, vals):
        if abs((t + x + g) - 1.0) < 1e-9:
            out.append((t, x, g))
    return out

WEIGHTS = all_weight_triplets(0.1)

def spearman_corr(y_true, y_pred):
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def score_metric(metric, y_true, y_pred):
    m = metric.upper()
    if m == "AUROC":
        return float(roc_auc_score(y_true, y_pred))
    if m == "AUPRC":
        return float(average_precision_score(y_true, y_pred))
    if m == "MAE":
        return float(mean_absolute_error(y_true, y_pred))
    if m == "SPEARMAN":
        return float(spearman_corr(y_true, y_pred))
    raise ValueError(metric)

def better(metric, a, b):
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    if metric.upper() in {"MAE", "RMSE", "MSE"}:
        return a < b
    return a > b

def read_pred(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true = df[cols["y_true"]].to_numpy()
    if "y_prob" in cols:
        y_pred = df[cols["y_prob"]].to_numpy()
    elif "y_pred" in cols:
        y_pred = df[cols["y_pred"]].to_numpy()
    else:
        raise ValueError(f"unsupported prediction format: {path} cols={df.columns.tolist()}")
    sample_idx = df[cols["sample_idx"]].to_numpy() if "sample_idx" in cols else None
    return y_true, y_pred, sample_idx

def align(y_true, y_pred, sample_idx):
    if sample_idx is None:
        return y_true, y_pred
    order = np.argsort(sample_idx)
    return y_true[order], y_pred[order]

def newest(cands):
    cands = [p for p in cands if p.exists()]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def find_trimole_valid(task: str):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    valid_candidates = []
    for rp in root.glob("run_*"):
        if not rp.is_dir():
            continue
        task_dir = rp / task
        if not task_dir.exists():
            continue
        valid_candidates.extend(task_dir.rglob("valid_predictions.csv"))
    p = newest(valid_candidates)
    if p is None:
        raise FileNotFoundError(f"[T-valid] {task} not found in known trimole valid locations")
    return p

def find_xgb_valid(task: str):
    candidates = [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_valid_xgb_22tasks" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "valid_predictions.csv",
    ]
    p = first_existing(candidates)
    if p is None:
        raise FileNotFoundError(f"[X-valid] {task} not found in known xgb valid locations")
    return p

def find_gnn_valid(task: str):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    p = newest(list(root.rglob("valid_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-valid] {task} not found under {root}")
    return p

def find_trimole_test(task: str):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    # choose the newest run_* that actually contains this task
    run_candidates = []
    for rp in root.glob("run_*"):
        if not rp.is_dir():
            continue
        task_dir = rp / task
        if task_dir.exists():
            run_candidates.append(rp)
    if not run_candidates:
        raise FileNotFoundError(f"[T-run] {task} not found under any run_* in {root}")
    run_dir = newest(run_candidates)
    if run_dir is None:
        raise FileNotFoundError("[T-test] benchmark_clean_rerun run_* not found")
    candidates = [
        run_dir / task / "test_predictions.csv",
        *list(run_dir.rglob(f"{task}/test_predictions.csv")),
    ]
    p = newest(candidates)
    if p is None:
        raise FileNotFoundError(f"[T-test] {task} not found under {run_dir}")
    return p

def find_xgb_test(task: str):
    root = ROOT / "results/model_log"
    cands = [p for p in root.rglob(f"{task}_test_predictions.csv") if ("xgb" in str(p).lower() or "baseline" in str(p).lower() or "maplight" in str(p).lower())]
    p = newest(cands)
    if p is None:
        raise FileNotFoundError(f"[X-test] {task} not found")
    return p

def find_gnn_test(task: str):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    p = newest(list(root.rglob("test_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-test] {task} not found under {root}")
    return p

summary_rows = []
detail_rows = []
failed_rows = []

for task in TASKS:
    metric = OFFICIAL_METRIC[task]
    try:
        vt, vx, vg = find_trimole_valid(task), find_xgb_valid(task), find_gnn_valid(task)
        tt, tx, tg = find_trimole_test(task), find_xgb_test(task), find_gnn_test(task)

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

        if metric in {"AUROC", "AUPRC"}:
            if not (np.allclose(yv_t, yv_x) and np.allclose(yv_t, yv_g)):
                raise ValueError(f"{task} valid y_true mismatch across models")
            if not (np.allclose(yt_t, yt_x) and np.allclose(yt_t, yt_g)):
                raise ValueError(f"{task} test y_true mismatch across models")
            yv_ref, yt_ref = yv_t, yt_t
        else:
            # 回归任务允许以 X 作为参照
            yv_ref, yt_ref = yv_x, yt_x

        best_w = None
        best_valid = float("inf") if metric in {"MAE"} else -float("inf")

        for t, x, g in WEIGHTS:
            valid_pred = t * pv_t + x * pv_x + g * pv_g
            s = score_metric(metric, yv_ref, valid_pred)
            detail_rows.append({"task": task, "metric": metric, "t": t, "x": x, "g": g, "valid_score": s})
            if better(metric, s, best_valid):
                best_valid = s
                best_w = (t, x, g)

        t, x, g = best_w
        test_pred = t * pt_t + x * pt_x + g * pt_g
        best_test = score_metric(metric, yt_ref, test_pred)

        summary_rows.append({
            "task": task,
            "metric": metric,
            "best_t": t,
            "best_x": x,
            "best_g": g,
            "best_valid_score": best_valid,
            "test_score": best_test,
            "valid_trimole": score_metric(metric, yv_ref, pv_t),
            "valid_xgb": score_metric(metric, yv_ref, pv_x),
            "valid_gnn": score_metric(metric, yv_ref, pv_g),
            "trimole_valid_file": str(vt),
            "xgb_valid_file": str(vx),
            "gnn_valid_file": str(vg),
            "trimole_test_file": str(tt),
            "xgb_test_file": str(tx),
            "gnn_test_file": str(tg),
        })
        print(f"[done] {task} metric={metric} best_w={best_w} valid={best_valid:.6f} test={best_test:.6f}", flush=True)

    except Exception as e:
        failed_rows.append({"task": task, "error": str(e)})
        print(f"[failed] {task}: {e}", flush=True)

summary = pd.DataFrame(summary_rows).sort_values("task") if summary_rows else pd.DataFrame()
detail = pd.DataFrame(detail_rows)
failed = pd.DataFrame(failed_rows)

if len(summary):
    summary.to_csv(OUT_DIR / "txg_weight_search_benchmark_clean_summary.csv", index=False)
if len(detail):
    detail.to_csv(OUT_DIR / "txg_weight_search_benchmark_clean_detail.csv", index=False)
if len(failed):
    failed.to_csv(OUT_DIR / "txg_weight_search_benchmark_clean_failed.csv", index=False)

if len(summary):
    print("\n=== SUMMARY ===")
    print(summary[["task","metric","best_t","best_x","best_g","best_valid_score","test_score","valid_trimole","valid_xgb","valid_gnn"]].to_string(index=False))
if len(failed):
    print("\n=== FAILED ===")
    print(failed.to_string(index=False))

print("\nSaved:", OUT_DIR / "txg_weight_search_benchmark_clean_summary.csv")
print("Saved:", OUT_DIR / "txg_weight_search_benchmark_clean_detail.csv")
print("Saved:", OUT_DIR / "txg_weight_search_benchmark_clean_failed.csv")
