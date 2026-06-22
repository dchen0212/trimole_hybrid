#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

MAPLIGHT_ROOT = Path("<PROJECT_ROOT>/trimole/external/MapLight-TDC")
sys.path.insert(0, str(MAPLIGHT_ROOT))
from maplight import get_fingerprints  # noqa

TASK_CONFIG = {
    "ames": ("classification", "AUROC"),
    "bbb_martins": ("classification", "AUROC"),
    "bioavailability_ma": ("classification", "AUROC"),
    "caco2_wang": ("regression", "MAE"),
    "clearance_hepatocyte_az": ("regression", "Spearman"),
    "clearance_microsome_az": ("regression", "Spearman"),
    "cyp2c9_substrate_carbonmangels": ("classification", "AUPRC"),
    "cyp2c9_veith": ("classification", "AUPRC"),
    "cyp2d6_substrate_carbonmangels": ("classification", "AUPRC"),
    "cyp2d6_veith": ("classification", "AUPRC"),
    "cyp3a4_substrate_carbonmangels": ("classification", "AUROC"),
    "cyp3a4_veith": ("classification", "AUPRC"),
    "dili": ("classification", "AUROC"),
    "half_life_obach": ("regression", "Spearman"),
    "herg": ("classification", "AUROC"),
    "hia_hou": ("classification", "AUROC"),
    "ld50_zhu": ("regression", "MAE"),
    "lipophilicity_astrazeneca": ("regression", "MAE"),
    "pgp_broccatelli": ("classification", "AUROC"),
    "ppbr_az": ("regression", "MAE"),
    "solubility_aqsoldb": ("regression", "MAE"),
    "vdss_lombardo": ("regression", "Spearman"),
}
ALL_TASKS = list(TASK_CONFIG.keys())

def detect_cols(df):
    cols = {c.lower(): c for c in df.columns}
    smiles_col = cols.get("smiles") or cols.get("drug")
    y_col = cols.get("label") or cols.get("y") or cols.get("target")
    if smiles_col is None or y_col is None:
        raise KeyError(f"Cannot detect smiles/label columns: {list(df.columns)}")
    return smiles_col, y_col

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    smiles_col, y_col = detect_cols(df)
    X = np.asarray(get_fingerprints(df[smiles_col].astype(str)), dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    y = df[y_col].to_numpy()
    return X, y

def eval_cls(y_true, y_prob, metric):
    y_hat = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_hat))
    primary = auc if metric == "AUROC" else auprc
    return {"test_auc": auc, "test_auprc": auprc, "test_acc": acc, "primary_metric": primary}

def eval_reg(y_true, y_pred, metric):
    mae = float(mean_absolute_error(y_true, y_pred))
    sp = float(spearmanr(y_true, y_pred).statistic)
    primary = mae if metric == "MAE" else sp
    return {"test_mae": mae, "test_spearman": sp, "primary_metric": primary}

def fit_one(task_dir: Path, task: str, seed: int, run_dir: Path):
    task_type, metric = TASK_CONFIG[task]
    X_train, y_train = load_xy(task_dir / "train.csv")
    X_valid, y_valid = load_xy(task_dir / "valid.csv")
    X_test, y_test = load_xy(task_dir / "test.csv")

    if task_type == "classification":
        model = XGBClassifier(
            n_estimators=1200, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=1, objective="binary:logistic",
            tree_method="hist", random_state=seed, n_jobs=8, eval_metric="auc"
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        pd.DataFrame({"task": task, "y_true": y_valid, "y_pred": p_valid}).to_csv(
            run_dir / f"{task}_valid_predictions.csv", index=False
        )
        pd.DataFrame({"task": task, "y_true": y_test, "y_pred": p_test}).to_csv(
            run_dir / f"{task}_test_predictions.csv", index=False
        )

        valid_primary = roc_auc_score(y_valid, p_valid) if metric == "AUROC" else average_precision_score(y_valid, p_valid)
        row = {
            "task": task, "task_type": task_type, "primary_metric_name": metric,
            "best_valid_primary": float(valid_primary), "loss_type": "XGBoost", "seed": seed,
        }
        row.update(eval_cls(y_test, p_test, metric))
        return row

    model = XGBRegressor(
        n_estimators=1200, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=1, objective="reg:squarederror",
        tree_method="hist", random_state=seed, n_jobs=8, eval_metric="mae"
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    p_valid = model.predict(X_valid)
    p_test = model.predict(X_test)

    pd.DataFrame({"task": task, "y_true": y_valid, "y_pred": p_valid}).to_csv(
        run_dir / f"{task}_valid_predictions.csv", index=False
    )
    pd.DataFrame({"task": task, "y_true": y_test, "y_pred": p_test}).to_csv(
        run_dir / f"{task}_test_predictions.csv", index=False
    )

    valid_primary = mean_absolute_error(y_valid, p_valid) if metric == "MAE" else spearmanr(y_valid, p_valid).statistic
    row = {
        "task": task, "task_type": task_type, "primary_metric_name": metric,
        "best_valid_primary": float(valid_primary), "loss_type": "XGBoost", "seed": seed,
    }
    row.update(eval_reg(y_test, p_test, metric))
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = args.tasks if args.tasks else ALL_TASKS
    out_root = Path(args.out)
    run_dir = out_root / "run_xgb_baseline_22tasks_with_valid"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows, errors = [], {}
    for task in tasks:
        try:
            row = fit_one(Path(args.data_root) / task, task, args.seed, run_dir)
            rows.append(row)
            print(f"[{task}] {row['primary_metric_name']}={row['primary_metric']:.6f}")
        except Exception as e:
            errors[task] = str(e)
            print(f"[{task}] FAILED: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "results_all.csv", index=False)
        print(f"\nDone. Summary: {run_dir / 'results_all.csv'}")
    if errors:
        (run_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Failures: {len(errors)} (see {run_dir / 'errors.json'})")

if __name__ == "__main__":
    main()
