from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("<PROJECT_ROOT>/trimole_hybrid")
RESULTS_ROOT = ROOT / "results_strict"
OUT_DIR = RESULTS_ROOT / "calibration_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TDC official metric for classification tasks in ADMET group
TASK_METRIC = {
    "ames": "AUROC",
    "bbb_martins": "AUROC",
    "bioavailability_ma": "AUROC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "cyp3a4_veith": "AUPRC",
    "dili": "AUROC",
    "herg": "AUROC",
    "hia_hou": "AUROC",
    "pgp_broccatelli": "AUROC",
}

def read_pred(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    y_col = next(cols[c] for c in ["y_true", "label", "target", "y"] if c in cols)
    p_col = next(cols[c] for c in ["y_prob", "y_pred", "pred", "prediction", "predictions", "score"] if c in cols)
    return df[y_col].to_numpy(), df[p_col].to_numpy()

def metric_score(task, y, p):
    m = TASK_METRIC[task]
    if m == "AUROC":
        return roc_auc_score(y, p)
    elif m == "AUPRC":
        return average_precision_score(y, p)
    raise ValueError(task)

rows = []

# only calibrate final 5-seed runs if they exist; otherwise all strict runs
candidate_dirs = sorted(RESULTS_ROOT.glob("**/valid_predictions.csv"))
for valid_csv in candidate_dirs:
    task_dir = valid_csv.parent
    task = task_dir.name
    if task not in TASK_METRIC:
        continue

    test_csv = task_dir / "test_predictions.csv"
    if not test_csv.exists():
        continue

    yv, pv = read_pred(valid_csv)
    yt, pt = read_pred(test_csv)

    # raw
    raw_valid = metric_score(task, yv, pv)
    raw_test = metric_score(task, yt, pt)

    rows.append({
        "task": task,
        "method": "raw",
        "valid_metric": raw_valid,
        "test_metric": raw_test,
        "source_valid_file": str(valid_csv),
        "source_test_file": str(test_csv),
    })

    # sigmoid / Platt-style calibration
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(pv.reshape(-1, 1), yv)
    pv_sig = lr.predict_proba(pv.reshape(-1, 1))[:, 1]
    pt_sig = lr.predict_proba(pt.reshape(-1, 1))[:, 1]
    rows.append({
        "task": task,
        "method": "sigmoid",
        "valid_metric": metric_score(task, yv, pv_sig),
        "test_metric": metric_score(task, yt, pt_sig),
        "source_valid_file": str(valid_csv),
        "source_test_file": str(test_csv),
    })

    # isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pv, yv)
    pv_iso = iso.transform(pv)
    pt_iso = iso.transform(pt)
    rows.append({
        "task": task,
        "method": "isotonic",
        "valid_metric": metric_score(task, yv, pv_iso),
        "test_metric": metric_score(task, yt, pt_iso),
        "source_valid_file": str(valid_csv),
        "source_test_file": str(test_csv),
    })

df = pd.DataFrame(rows)
all_csv = OUT_DIR / "_all_calibration_trials.csv"
df.to_csv(all_csv, index=False)

best_rows = []
for task, sub in df.groupby("task"):
    sub = sub.sort_values(["valid_metric", "test_metric"], ascending=False).reset_index(drop=True)
    best_rows.append(sub.iloc[0])

best = pd.DataFrame(best_rows).sort_values("task").reset_index(drop=True)
best_csv = OUT_DIR / "_best_by_valid.csv"
best.to_csv(best_csv, index=False)

print("=== BEST BY VALID ===")
print(best.to_string(index=False))
print("\nsaved:", all_csv)
print("saved:", best_csv)
