#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

TASKS = [
    "bbb_martins",
    "hia_hou",
    "herg",
    "dili",
    "cyp3a4_veith",
    "cyp2d6_veith",
]

PRIMARY = {
    "bbb_martins": "AUROC",
    "hia_hou": "AUROC",
    "herg": "AUROC",
    "dili": "AUROC",
    "cyp3a4_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
}

TRIMOLE_DIR = Path("results/model_log/calibrated_experts_tx_v3/trimole")
XGB_DIR = Path("results/model_log/calibrated_experts_tx_v3/xgb")
OUT_DIR = Path("results/model_log/tx_router_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_calibrated(base: Path, task: str, split: str):
    p = base / task / f"calibrated_{split}_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df["y_true"].to_numpy().astype(int), df["y_pred_calibrated"].to_numpy().astype(float)

def score_cls(y_true, y_prob, metric):
    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    primary = auc if metric == "AUROC" else auprc
    return {
        "primary": primary,
        "auc": auc,
        "auprc": auprc,
        "acc": acc,
    }

rows = []

for task in TASKS:
    metric = PRIMARY[task]

    t_val = load_calibrated(TRIMOLE_DIR, task, "valid")
    t_test = load_calibrated(TRIMOLE_DIR, task, "test")
    x_val = load_calibrated(XGB_DIR, task, "valid")
    x_test = load_calibrated(XGB_DIR, task, "test")

    candidates = []

    # trimole only
    if t_val is not None and t_test is not None:
        yv, ptv = t_val
        yt, ptt = t_test
        s_val = score_cls(yv, ptv, metric)
        s_test = score_cls(yt, ptt, metric)
        candidates.append({
            "task": task,
            "strategy": "trimole",
            "valid_primary": s_val["primary"],
            "test_primary": s_test["primary"],
            "test_auc": s_test["auc"],
            "test_auprc": s_test["auprc"],
            "test_acc": s_test["acc"],
        })

    # xgb only
    if x_val is not None and x_test is not None:
        yv, pxv = x_val
        yt, pxt = x_test
        s_val = score_cls(yv, pxv, metric)
        s_test = score_cls(yt, pxt, metric)
        candidates.append({
            "task": task,
            "strategy": "xgb",
            "valid_primary": s_val["primary"],
            "test_primary": s_test["primary"],
            "test_auc": s_test["auc"],
            "test_auprc": s_test["auprc"],
            "test_acc": s_test["acc"],
        })

    # late fusion TX
    if t_val is not None and x_val is not None and t_test is not None and x_test is not None:
        yv_t, ptv = t_val
        yv_x, pxv = x_val
        yt_t, ptt = t_test
        yt_x, pxt = x_test

        if np.array_equal(yv_t, yv_x) and np.array_equal(yt_t, yt_x):
            best = None
            for wt in np.arange(0.0, 1.01, 0.05):
                wx = 1.0 - wt
                pv = wt * ptv + wx * pxv
                st = score_cls(yv_t, pv, metric)
                cand = {
                    "wt": float(round(wt, 2)),
                    "wx": float(round(wx, 2)),
                    "valid_primary": st["primary"],
                }
                if best is None or cand["valid_primary"] > best["valid_primary"]:
                    best = cand

            pv = best["wt"] * ptv + best["wx"] * pxv
            pt = best["wt"] * ptt + best["wx"] * pxt
            s_val = score_cls(yv_t, pv, metric)
            s_test = score_cls(yt_t, pt, metric)
            candidates.append({
                "task": task,
                "strategy": f"late_fusion_tx_{best['wt']:.2f}_{best['wx']:.2f}",
                "valid_primary": s_val["primary"],
                "test_primary": s_test["primary"],
                "test_auc": s_test["auc"],
                "test_auprc": s_test["auprc"],
                "test_acc": s_test["acc"],
            })

            # stacking TX
            Xv = np.column_stack([ptv, pxv])
            Xt = np.column_stack([ptt, pxt])
            meta = LogisticRegression(solver="lbfgs", max_iter=1000)
            meta.fit(Xv, yv_t)
            pv_stack = meta.predict_proba(Xv)[:, 1]
            pt_stack = meta.predict_proba(Xt)[:, 1]
            s_val = score_cls(yv_t, pv_stack, metric)
            s_test = score_cls(yt_t, pt_stack, metric)
            candidates.append({
                "task": task,
                "strategy": "stacking_tx",
                "valid_primary": s_val["primary"],
                "test_primary": s_test["primary"],
                "test_auc": s_test["auc"],
                "test_auprc": s_test["auprc"],
                "test_acc": s_test["acc"],
            })

    if not candidates:
        continue

    cand_df = pd.DataFrame(candidates).sort_values(
        ["valid_primary", "test_primary"], ascending=[False, False]
    )
    cand_df.to_csv(OUT_DIR / f"{task}_candidates.csv", index=False)

    best = cand_df.iloc[0].to_dict()
    rows.append(best)

final_df = pd.DataFrame(rows).sort_values("task")
final_df.to_csv(OUT_DIR / "tx_router_results.csv", index=False)

print("=== TX ROUTER RESULTS ===")
print(final_df.to_string(index=False))
print("\nSaved:", OUT_DIR / "tx_router_results.csv")
