from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")

TASKS = {
    "bbb_martins": {"metric": "AUROC", "center": (0.2, 0.8, 0.0)},
    "clearance_hepatocyte_az": {"metric": "Spearman", "center": (0.8, 0.2, 0.0)},
    "cyp2d6_veith": {"metric": "AUPRC", "center": (0.2, 0.5, 0.3)},
    "ld50_zhu": {"metric": "MAE", "center": (0.3, 0.3, 0.4)},
    "lipophilicity_astrazeneca": {"metric": "MAE", "center": (0.2, 0.4, 0.4)},
    "cyp2c9_substrate_carbonmangels": {"metric": "AUPRC", "center": (0.4, 0.0, 0.6)},
}

OUT_DIR = ROOT / "results/model_log/txg_weight_search_top1_sprint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def newest(cands):
    cands = [Path(x) for x in cands if x and Path(x).exists()]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def first_existing(cands):
    for c in cands:
        c = Path(c)
        if c.exists():
            return c
    return None

def read_pred_file(p):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}

    y_col = None
    pred_col = None
    for cand in ["y_true", "label", "labels", "target", "y"]:
        if cand in cols:
            y_col = cols[cand]
            break
    for cand in ["pred", "prediction", "predictions", "y_pred", "y_prob", "score", "prob", "probability"]:
        if cand in cols:
            pred_col = cols[cand]
            break

    if y_col is None or pred_col is None:
        raise ValueError(f"Cannot parse prediction file: {p}, cols={list(df.columns)}")

    return df[y_col].to_numpy(), df[pred_col].to_numpy()

def metric_score(metric, y, pred):
    m = metric.upper()
    if m == "AUROC":
        return roc_auc_score(y, pred)
    if m == "AUPRC":
        return average_precision_score(y, pred)
    if m == "MAE":
        return mean_absolute_error(y, pred)
    if m == "SPEARMAN":
        return spearmanr(y, pred).statistic
    raise ValueError(metric)

def better(metric, a, b):
    m = metric.upper()
    if m in {"AUROC", "AUPRC", "SPEARMAN"}:
        return a > b
    if m == "MAE":
        return a < b
    raise ValueError(metric)

def find_trimole_valid(task):
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
        raise FileNotFoundError(f"[T-valid] {task} not found")
    return p

def find_trimole_test(task):
    root = ROOT / "results/model_log/benchmark_clean_rerun"
    run_candidates = []
    for rp in root.glob("run_*"):
        if rp.is_dir() and (rp / task).exists():
            run_candidates.append(rp)
    if not run_candidates:
        raise FileNotFoundError(f"[T-run] {task} not found")
    run_dir = newest(run_candidates)
    candidates = [run_dir / task / "test_predictions.csv", *list(run_dir.rglob(f"{task}/test_predictions.csv"))]
    p = newest(candidates)
    if p is None:
        raise FileNotFoundError(f"[T-test] {task} not found under {run_dir}")
    return p

def find_xgb_valid(task):
    candidates = [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_valid_xgb_22tasks" / f"{task}_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_valid_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "valid_predictions.csv",
    ]
    p = first_existing(candidates)
    if p is None:
        raise FileNotFoundError(f"[X-valid] {task} not found")
    return p

def find_xgb_test(task):
    candidates = [
        ROOT / "results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid" / f"{task}_test_predictions.csv",
        ROOT / "results/model_log/fusion_inputs_test_xgb_22tasks" / f"{task}_test_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_tx_v3/xgb" / task / "calibrated_test_predictions.csv",
        ROOT / "results/model_log/calibrated_experts_v1/xgb" / task / "calibrated_test_predictions.csv",
        ROOT / "results/model_log/maplight_catboost" / task / "test_predictions.csv",
    ]
    p = first_existing(candidates)
    if p is None:
        raise FileNotFoundError(f"[X-test] {task} not found")
    return p

def find_gnn_valid(task):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    p = newest(list(root.rglob("valid_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-valid] {task} not found")
    return p

def find_gnn_test(task):
    root = ROOT / "results/model_log/gnn_v2_22tasks" / task
    p = newest(list(root.rglob("test_predictions.csv")))
    if p is None:
        raise FileNotFoundError(f"[G-test] {task} not found")
    return p

def make_grid(center, radius=0.12, step=0.02):
    ct, cx, cg = center
    vals = np.arange(0.0, 1.0001, step)
    out = []
    for t, x in itertools.product(vals, vals):
        g = round(1.0 - t - x, 10)
        if g < -1e-9 or g > 1 + 1e-9:
            continue
        g = max(0.0, min(1.0, g))
        if abs(t - ct) <= radius and abs(x - cx) <= radius and abs(g - cg) <= radius:
            out.append((round(float(t), 2), round(float(x), 2), round(float(g), 2)))
    out = sorted(set(out))
    return out

rows, failed = [], []

for task, info in TASKS.items():
    metric = info["metric"]
    center = info["center"]
    try:
        tv = find_trimole_valid(task); tt = find_trimole_test(task)
        xv = find_xgb_valid(task);    xt = find_xgb_test(task)
        gv = find_gnn_valid(task);    gt = find_gnn_test(task)

        y_tv, p_tv = read_pred_file(tv)
        y_xv, p_xv = read_pred_file(xv)
        y_gv, p_gv = read_pred_file(gv)

        y_tt, p_tt = read_pred_file(tt)
        y_xt, p_xt = read_pred_file(xt)
        y_gt, p_gt = read_pred_file(gt)

        best = None
        for wt, wx, wg in make_grid(center):
            pv = wt * p_tv + wx * p_xv + wg * p_gv
            vv = metric_score(metric, y_tv, pv)
            if best is None or better(metric, vv, best["best_valid_score"]):
                pt = wt * p_tt + wx * p_xt + wg * p_gt
                ts = metric_score(metric, y_tt, pt)
                best = {
                    "task": task, "metric": metric,
                    "best_t": wt, "best_x": wx, "best_g": wg,
                    "best_valid_score": vv, "test_score": ts,
                    "valid_trimole": metric_score(metric, y_tv, p_tv),
                    "valid_xgb": metric_score(metric, y_xv, p_xv),
                    "valid_gnn": metric_score(metric, y_gv, p_gv),
                    "trimole_valid_file": str(tv), "xgb_valid_file": str(xv), "gnn_valid_file": str(gv),
                    "trimole_test_file": str(tt), "xgb_test_file": str(xt), "gnn_test_file": str(gt),
                }

        rows.append(best)
        print(f"[done] {task} metric={metric} best_w=({best['best_t']}, {best['best_x']}, {best['best_g']}) valid={best['best_valid_score']:.6f} test={best['test_score']:.6f}")
    except Exception as e:
        failed.append({"task": task, "error": str(e)})
        print(f"[failed] {task}: {e}")

summary = pd.DataFrame(rows)
failed_df = pd.DataFrame(failed)

summary.to_csv(OUT_DIR / "top1_sprint_summary.csv", index=False)
failed_df.to_csv(OUT_DIR / "top1_sprint_failed.csv", index=False)

print("\n=== SUMMARY ===")
if len(summary):
    print(summary[["task","metric","best_t","best_x","best_g","best_valid_score","test_score"]].to_string(index=False))
print("\n=== FAILED ===")
if len(failed_df):
    print(failed_df.to_string(index=False))
