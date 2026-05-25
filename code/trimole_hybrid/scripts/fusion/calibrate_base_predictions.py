#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


CLASSIFICATION_TASKS = {
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
}

DEFAULT_MODELS = ["trimole", "xgb", "gnn"]


def _find_pred_file(base_dir: Path, task: str, split: str) -> Optional[Path]:
    candidates = [
        base_dir / f"{task}_{split}_predictions.csv",
        base_dir / task / f"{split}_predictions.csv",
        base_dir / f"{task}_{split}.csv",
        base_dir / task / f"{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    y_true_col = cols.get("y_true") or cols.get("label") or cols.get("y") or cols.get("target")
    y_pred_col = cols.get("y_pred") or cols.get("y_prob") or cols.get("prob") or cols.get("pred")
    if y_true_col is None or y_pred_col is None:
        raise ValueError(f"Cannot detect y_true/y_pred columns from {list(df.columns)}")
    return y_true_col, y_pred_col


def _load_pred_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y_true_col, y_pred_col = _detect_cols(df)
    y_true = df[y_true_col].to_numpy().astype(int)
    y_pred = df[y_pred_col].to_numpy().astype(float)
    return y_true, y_pred


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _fit_sigmoid_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    x = np.log(_clip_prob(y_prob) / (1.0 - _clip_prob(y_prob))).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x, y_true)
    return clf


def _apply_sigmoid_calibrator(model: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    x = np.log(_clip_prob(y_prob) / (1.0 - _clip_prob(y_prob))).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def _fit_isotonic_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(_clip_prob(y_prob), y_true)
    return iso


def _apply_isotonic_calibrator(model: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(_clip_prob(y_prob)), dtype=float)


def _eval_cls(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_prob = _clip_prob(y_prob)
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_hat)),
    }


def calibrate_one(
    task: str,
    model_name: str,
    valid_file: Path,
    test_file: Path,
    method: str,
    out_dir: Path,
) -> Dict[str, object]:
    y_valid, p_valid = _load_pred_csv(valid_file)
    y_test, p_test = _load_pred_csv(test_file)

    if not np.array_equal(y_valid, y_valid.astype(int)):
        raise ValueError(f"{task}/{model_name}: valid labels are not integer class labels")
    if not np.array_equal(y_test, y_test.astype(int)):
        raise ValueError(f"{task}/{model_name}: test labels are not integer class labels")

    raw_valid = _eval_cls(y_valid, p_valid)
    raw_test = _eval_cls(y_test, p_test)

    if method == "auto":
        # 小样本优先 sigmoid，大一点再给 isotonic
        method = "sigmoid" if len(y_valid) < 1000 else "isotonic"

    if method == "sigmoid":
        cal = _fit_sigmoid_calibrator(y_valid, p_valid)
        p_valid_cal = _apply_sigmoid_calibrator(cal, p_valid)
        p_test_cal = _apply_sigmoid_calibrator(cal, p_test)
        meta = {
            "type": "sigmoid",
            "coef": cal.coef_.ravel().tolist(),
            "intercept": cal.intercept_.ravel().tolist(),
        }
    elif method == "isotonic":
        cal = _fit_isotonic_calibrator(y_valid, p_valid)
        p_valid_cal = _apply_isotonic_calibrator(cal, p_valid)
        p_test_cal = _apply_isotonic_calibrator(cal, p_test)
        meta = {
            "type": "isotonic",
            "x_thresholds": getattr(cal, "X_thresholds_", []).tolist(),
            "y_thresholds": getattr(cal, "y_thresholds_", []).tolist(),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    cal_valid = _eval_cls(y_valid, p_valid_cal)
    cal_test = _eval_cls(y_test, p_test_cal)

    task_out = out_dir / model_name / task
    task_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "task": task,
        "y_true": y_valid,
        "y_pred_raw": _clip_prob(p_valid),
        "y_pred_calibrated": _clip_prob(p_valid_cal),
    }).to_csv(task_out / "calibrated_valid_predictions.csv", index=False)

    pd.DataFrame({
        "task": task,
        "y_true": y_test,
        "y_pred_raw": _clip_prob(p_test),
        "y_pred_calibrated": _clip_prob(p_test_cal),
    }).to_csv(task_out / "calibrated_test_predictions.csv", index=False)

    (task_out / "calibrator_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    return {
        "task": task,
        "model": model_name,
        "method": method,
        "n_valid": int(len(y_valid)),
        "raw_valid_auc": raw_valid["auc"],
        "raw_valid_auprc": raw_valid["auprc"],
        "raw_valid_acc": raw_valid["acc"],
        "cal_valid_auc": cal_valid["auc"],
        "cal_valid_auprc": cal_valid["auprc"],
        "cal_valid_acc": cal_valid["acc"],
        "raw_test_auc": raw_test["auc"],
        "raw_test_auprc": raw_test["auprc"],
        "raw_test_acc": raw_test["acc"],
        "cal_test_auc": cal_test["auc"],
        "cal_test_auprc": cal_test["auprc"],
        "cal_test_acc": cal_test["acc"],
        "valid_file": str(valid_file),
        "test_file": str(test_file),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root dir")
    ap.add_argument("--tasks", nargs="*", default=None, help="Classification tasks only")
    ap.add_argument("--method", default="auto", choices=["auto", "sigmoid", "isotonic"])
    ap.add_argument("--trimole-dir", default=None)
    ap.add_argument("--xgb-dir", default=None)
    ap.add_argument("--gnn-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "trimole": args.trimole_dir,
        "xgb": args.xgb_dir,
        "gnn": args.gnn_dir,
    }
    model_dirs = {k: Path(v) for k, v in model_dirs.items() if v}

    tasks = args.tasks if args.tasks else sorted(CLASSIFICATION_TASKS)
    tasks = [t for t in tasks if t in CLASSIFICATION_TASKS]

    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, str]] = []

    for model_name, base_dir in model_dirs.items():
        for task in tasks:
            valid_file = _find_pred_file(base_dir, task, "valid")
            test_file = _find_pred_file(base_dir, task, "test")

            if valid_file is None or test_file is None:
                missing.append({
                    "task": task,
                    "model": model_name,
                    "valid_found": str(valid_file) if valid_file else "",
                    "test_found": str(test_file) if test_file else "",
                    "base_dir": str(base_dir),
                })
                continue

            try:
                row = calibrate_one(
                    task=task,
                    model_name=model_name,
                    valid_file=valid_file,
                    test_file=test_file,
                    method=args.method,
                    out_dir=out_dir,
                )
                rows.append(row)
                print(
                    f"[{model_name}][{task}] "
                    f"valid_auc {row['raw_valid_auc']:.6f} -> {row['cal_valid_auc']:.6f} | "
                    f"test_auc {row['raw_test_auc']:.6f} -> {row['cal_test_auc']:.6f}"
                )
            except Exception as e:
                missing.append({
                    "task": task,
                    "model": model_name,
                    "valid_found": str(valid_file),
                    "test_found": str(test_file),
                    "base_dir": str(base_dir),
                    "error": str(e),
                })
                print(f"[{model_name}][{task}] FAILED: {e}")

    if rows:
        pd.DataFrame(rows).sort_values(["model", "task"]).to_csv(
            out_dir / "calibration_summary.csv", index=False
        )
        print("\nSaved:", out_dir / "calibration_summary.csv")

    if missing:
        pd.DataFrame(missing).to_csv(out_dir / "calibration_missing_or_failed.csv", index=False)
        print("Saved:", out_dir / "calibration_missing_or_failed.csv")


if __name__ == "__main__":
    main()
