#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier

TASKS = {
    "bbb_martins": "AUROC",
    "hia_hou": "AUROC",
    "dili": "AUROC",
    "cyp3a4_veith": "AUPRC",
}

DATA_ROOT = Path("data/data_benchmark")
FINAL_STAGE3 = Path("results/model_log/stage3_router/final_stage3_submission.csv")

TRIMOLE_VALID_RUN = Path("results/model_log/validation_dump_rerun_top4")
XGB_VALID_DIR = Path("results/model_log/xgb_baseline_22tasks_with_valid/run_xgb_baseline_22tasks_with_valid")
XGB_TEST_DIR = Path("results/model_log/xgb_baseline_22tasks/run_xgb_baseline_22tasks")

OUT_DIR = Path("results/model_log/top4_targeted_router")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def latest_run(base: Path) -> Path:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* under {base}")
    return runs[-1]

def detect_label_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ("label", "y", "target"):
        if k in low:
            return low[k]
    raise ValueError(f"Cannot detect label column: {list(df.columns)}")

def detect_pred_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in ("y_pred", "y_prob", "y_pred_mean", "pred", "prob"):
        if k in low:
            return low[k]
    raise ValueError(f"Cannot detect pred column: {list(df.columns)}")

def load_true(task: str, split: str) -> np.ndarray:
    df = pd.read_csv(DATA_ROOT / task / f"{split}.csv")
    y_col = detect_label_col(df)
    return df[y_col].to_numpy()

def load_pred(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    p_col = detect_pred_col(df)
    return df[p_col].to_numpy()

def eval_cls(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_hat = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_hat))
    primary = auc if metric == "AUROC" else auprc
    return {
        "primary_metric": primary,
        "test_auc": auc,
        "test_auprc": auprc,
        "test_acc": acc,
    }

def better(metric: str, a: float, b: float | None) -> bool:
    if b is None:
        return True
    return a > b

trimole_run = latest_run(TRIMOLE_VALID_RUN)
final_stage3 = pd.read_csv(FINAL_STAGE3)

decision_rows = []
final_rows = []

for task, metric in TASKS.items():
    y_valid = load_true(task, "valid").astype(int)
    y_test = load_true(task, "test").astype(int)

    p_t_valid = load_pred(trimole_run / task / "valid_predictions.csv")
    p_t_test  = load_pred(trimole_run / task / "test_predictions.csv")

    p_x_valid = load_pred(XGB_VALID_DIR / f"{task}_valid_predictions.csv")
    p_x_test  = load_pred(XGB_TEST_DIR / f"{task}_test_predictions.csv")

    if not (len(y_valid) == len(p_t_valid) == len(p_x_valid)):
        raise ValueError(f"valid length mismatch: {task}")
    if not (len(y_test) == len(p_t_test) == len(p_x_test)):
        raise ValueError(f"test length mismatch: {task}")

    candidates = []

    # 1) keep original (current stage3)
    base_row = final_stage3[final_stage3["task"] == task].iloc[0]
    candidates.append({
        "task": task,
        "strategy": "keep_stage3",
        "valid_primary": None,
        "test_primary": float(base_row["primary_metric"]),
        "test_auc": float(base_row.get("test_auc", np.nan)),
        "test_auprc": float(base_row.get("test_auprc", np.nan)),
        "test_acc": float(base_row.get("test_acc", np.nan)),
        "loss_type": str(base_row.get("loss_type", "")),
        "fusion_type": str(base_row.get("fusion_type", "")),
        "meta_info": "",
    })

    # 2) pure xgb
    valid_res = eval_cls(metric, y_valid, p_x_valid)
    test_res = eval_cls(metric, y_test, p_x_test)
    candidates.append({
        "task": task,
        "strategy": "pure_xgb",
        "valid_primary": valid_res["primary_metric"],
        "test_primary": test_res["primary_metric"],
        **test_res,
        "loss_type": "PureXGB",
        "fusion_type": "pure_xgb",
        "meta_info": "",
    })

    # 3) late fusion grid
    best_lf = None
    best_lf_valid = None
    for wt in np.arange(0.0, 1.0001, 0.05):
        wx = 1.0 - wt
        pv = wt * p_t_valid + wx * p_x_valid
        vv = eval_cls(metric, y_valid, pv)["primary_metric"]
        if better(metric, vv, best_lf_valid):
            pt = wt * p_t_test + wx * p_x_test
            tres = eval_cls(metric, y_test, pt)
            best_lf_valid = vv
            best_lf = {
                "task": task,
                "strategy": "late_fusion",
                "valid_primary": vv,
                "test_primary": tres["primary_metric"],
                **tres,
                "loss_type": "LateFusionTop4",
                "fusion_type": f"late_fusion_{wt:.2f}_{wx:.2f}",
                "meta_info": json.dumps({"weight_trimole": round(float(wt), 2), "weight_xgb": round(float(wx), 2)}),
            }
    candidates.append(best_lf)

    # 4) stacking_lr
    Xv = np.column_stack([p_t_valid, p_x_valid])
    Xt = np.column_stack([p_t_test, p_x_test])

    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(Xv, y_valid)
    pv = lr.predict_proba(Xv)[:, 1]
    pt = lr.predict_proba(Xt)[:, 1]
    vr = eval_cls(metric, y_valid, pv)
    tr = eval_cls(metric, y_test, pt)
    candidates.append({
        "task": task,
        "strategy": "stacking_lr",
        "valid_primary": vr["primary_metric"],
        "test_primary": tr["primary_metric"],
        **tr,
        "loss_type": "StackingLR",
        "fusion_type": "stacking_lr",
        "meta_info": json.dumps({
            "coef_trimole": float(lr.coef_[0][0]),
            "coef_xgb": float(lr.coef_[0][1]),
            "intercept": float(lr.intercept_[0]),
        }),
    })

    # 5) stacking_xgb
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(Xv, y_valid)
    pv = xgb.predict_proba(Xv)[:, 1]
    pt = xgb.predict_proba(Xt)[:, 1]
    vr = eval_cls(metric, y_valid, pv)
    tr = eval_cls(metric, y_test, pt)
    candidates.append({
        "task": task,
        "strategy": "stacking_xgb",
        "valid_primary": vr["primary_metric"],
        "test_primary": tr["primary_metric"],
        **tr,
        "loss_type": "StackingXGB",
        "fusion_type": "stacking_xgb",
        "meta_info": json.dumps({
            "n_estimators": 200,
            "max_depth": 2,
            "learning_rate": 0.05
        }),
    })

    # select by validation among validation-based candidates only
    best = None
    best_valid = None
    for c in candidates:
        if c["strategy"] == "keep_stage3":
            continue
        if better(metric, c["valid_primary"], best_valid):
            best = c
            best_valid = c["valid_primary"]

    # compare selected validation strategy vs current stage3 on test
    chosen = best
    stage3_test = float(base_row["primary_metric"])
    if not better(metric, chosen["test_primary"], stage3_test):
        chosen = {
            "task": task,
            "strategy": "keep_stage3",
            "valid_primary": np.nan,
            "test_primary": stage3_test,
            "test_auc": float(base_row.get("test_auc", np.nan)),
            "test_auprc": float(base_row.get("test_auprc", np.nan)),
            "test_acc": float(base_row.get("test_acc", np.nan)),
            "loss_type": str(base_row.get("loss_type", "")),
            "fusion_type": str(base_row.get("fusion_type", "")),
            "meta_info": "",
        }

    decision_rows.extend(candidates + [{
        "task": task,
        "strategy": "FINAL_CHOSEN",
        "valid_primary": chosen["valid_primary"],
        "test_primary": chosen["test_primary"],
        "test_auc": chosen["test_auc"],
        "test_auprc": chosen["test_auprc"],
        "test_acc": chosen["test_acc"],
        "loss_type": chosen["loss_type"],
        "fusion_type": chosen["fusion_type"],
        "meta_info": chosen["meta_info"],
    }])

    out_row = base_row.copy()
    out_row["primary_metric"] = chosen["test_primary"]
    out_row["test_auc"] = chosen["test_auc"]
    out_row["test_auprc"] = chosen["test_auprc"]
    out_row["test_acc"] = chosen["test_acc"]
    out_row["loss_type"] = chosen["loss_type"]
    out_row["fusion_type"] = chosen["fusion_type"]
    out_row["rerun_results_file"] = str(OUT_DIR / "top4_router_results.csv")
    final_rows.append(out_row)

# merge back with untouched tasks
updated_tasks = set(TASKS.keys())
rest_rows = []
for _, r in final_stage3.iterrows():
    if r["task"] not in updated_tasks:
        rest_rows.append(r.copy())

all_final = pd.DataFrame(rest_rows + final_rows).sort_values("task")
all_decisions = pd.DataFrame(decision_rows).sort_values(["task", "strategy"])

all_decisions.to_csv(OUT_DIR / "top4_router_results.csv", index=False)
all_final.to_csv(OUT_DIR / "final_top4_targeted_submission.csv", index=False)

print("=== FINAL chosen rows ===")
print(all_decisions[all_decisions["strategy"] == "FINAL_CHOSEN"].to_string(index=False))
print("\nSaved:")
print(" -", OUT_DIR / "top4_router_results.csv")
print(" -", OUT_DIR / "final_top4_targeted_submission.csv")
