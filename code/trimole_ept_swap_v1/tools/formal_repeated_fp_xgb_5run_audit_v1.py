from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
OUT_ROOT = REPO / "results_strict" / "formal_repeated_fp_xgb_5run_audit_v1"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested


TASKS: dict[str, dict[str, object]] = {
    "solubility_aqsoldb": {"metric": "MAE", "direction": "min", "top1_ref": 0.741},
    "clearance_hepatocyte_az": {"metric": "Spearman", "direction": "max", "top1_ref": 0.536},
    "clearance_microsome_az": {"metric": "Spearman", "direction": "max", "top1_ref": 0.630},
    "cyp3a4_substrate_carbonmangels": {"metric": "AUROC", "direction": "max", "top1_ref": 0.667},
    "cyp2d6_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.736},
    "cyp2c9_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.474},
    "hia_hou": {"metric": "AUROC", "direction": "max", "top1_ref": 0.993},
    "bbb_martins": {"metric": "AUROC", "direction": "max", "top1_ref": 0.924},
}

DEFAULT_GROUPS = [
    [11, 22, 33, 44, 55],
    [111, 122, 133, 144, 155],
    [211, 222, 233, 244, 255],
    [311, 322, 333, 344, 355],
    [411, 422, 433, 444, 455],
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=["solubility_aqsoldb", "clearance_hepatocyte_az", "clearance_microsome_az"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--xgb-estimators", type=int, default=700)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_data(data_root: Path, task: str):
    task_dir = data_root / task
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    smiles_col = nested.get_smiles_col(train_df)
    label_col = base.find_label_col(train_df)
    X_tr = base.get_fingerprints(train_df[smiles_col])
    X_va = base.get_fingerprints(valid_df[smiles_col])
    X_te = base.get_fingerprints(test_df[smiles_col])
    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    X_tv = np.concatenate([X_tr, X_va], axis=0).astype(np.float32)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()
    return X_tv, y_tv, X_te.astype(np.float32), y_te, smiles_tv


def score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    if metric == "MAE":
        return float(mean_absolute_error(y, pred))
    if metric == "Spearman":
        val = spearmanr(y, pred).correlation
        return float(val) if val is not None and not math.isnan(float(val)) else float("-inf")
    if metric == "AUROC":
        return float(roc_auc_score(y, pred))
    if metric == "AUPRC":
        return float(average_precision_score(y, pred))
    raise ValueError(metric)


def fit_xgb(X: np.ndarray, y: np.ndarray, metric: str, seed: int, args: argparse.Namespace):
    if metric in {"AUROC", "AUPRC"}:
        pos = max(float(np.sum(np.asarray(y) == 1)), 1.0)
        neg = max(float(len(y) - pos), 1.0)
        model = base.XGBClassifier(
            n_estimators=args.xgb_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=args.n_jobs,
            scale_pos_weight=max(1.0, neg / pos),
        )
    else:
        model = base.XGBRegressor(
            n_estimators=args.xgb_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    model.fit(X, y, verbose=False)
    return model


def predict(model, X: np.ndarray, metric: str) -> np.ndarray:
    if metric in {"AUROC", "AUPRC"}:
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def write_pred(path: Path, y: np.ndarray, pred: np.ndarray, metric: str) -> None:
    col = "y_prob" if metric in {"AUROC", "AUPRC"} else "y_pred"
    pd.DataFrame({"sample_idx": np.arange(len(y), dtype=int), "y_true": y, col: pred}).to_csv(path, index=False)


def run_inner_seed(
    task: str,
    seed: int,
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    smiles_tv: list[str],
    metric: str,
    args: argparse.Namespace,
    out_root: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    seed_dir = out_root / task / "inner_seed_models" / f"seed_{seed}"
    result_path = seed_dir / "result.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text())
        train_col = "y_prob" if metric in {"AUROC", "AUPRC"} else "y_pred"
        test_col = train_col
        train_pred = pd.read_csv(seed_dir / "trainval_predictions.csv").sort_values("sample_idx")[train_col].to_numpy(dtype=np.float32)
        test_pred = pd.read_csv(seed_dir / "test_predictions.csv").sort_values("sample_idx")[test_col].to_numpy(dtype=np.float32)
        return result, train_pred, test_pred

    seed_dir.mkdir(parents=True, exist_ok=True)
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, seed)
    oof = np.zeros(len(y_tv), dtype=np.float32)
    test_parts: list[np.ndarray] = []
    fold_scores: list[float] = []
    for fold_idx, valid_idx in enumerate(folds):
        train_mask = np.ones(len(y_tv), dtype=bool)
        train_mask[valid_idx] = False
        train_idx = np.where(train_mask)[0]
        model = fit_xgb(X_tv[train_idx], y_tv[train_idx], metric, seed * 100 + fold_idx, args)
        oof[valid_idx] = predict(model, X_tv[valid_idx], metric)
        test_parts.append(predict(model, X_te, metric))
        fold_scores.append(score(metric, y_tv[valid_idx], oof[valid_idx]))
    test_pred = np.mean(test_parts, axis=0).astype(np.float32)
    result = {
        "task": task,
        "seed": seed,
        "metric": metric,
        "cv_fold_scores": ",".join(f"{x:.12f}" for x in fold_scores),
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores, ddof=0)),
        "oof_score": score(metric, y_tv, oof),
        "test_score": score(metric, y_te, test_pred),
        "trainval_pred_file": str(seed_dir / "trainval_predictions.csv"),
        "test_pred_file": str(seed_dir / "test_predictions.csv"),
    }
    write_pred(seed_dir / "trainval_predictions.csv", y_tv, oof, metric)
    write_pred(seed_dir / "test_predictions.csv", y_te, test_pred, metric)
    result_path.write_text(json.dumps(result, indent=2))
    return result, oof, test_pred


def run_task(task: str, args: argparse.Namespace) -> dict[str, object]:
    meta = TASKS[task]
    metric = str(meta["metric"])
    direction = str(meta["direction"])
    top1_ref = float(meta["top1_ref"])
    out_root = Path(args.out_root)
    task_dir = out_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())

    X_tv, y_tv, X_te, y_te, smiles_tv = load_data(Path(args.data_root), task)
    run_rows: list[dict[str, object]] = []
    formal_test_preds: list[np.ndarray] = []
    for run_idx, seeds in enumerate(DEFAULT_GROUPS, 1):
        print(f"[{task}] formal_run={run_idx} seeds={','.join(map(str, seeds))}", flush=True)
        seed_results: list[dict[str, object]] = []
        train_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        for seed in seeds:
            result, train_pred, test_pred = run_inner_seed(task, seed, X_tv, y_tv, X_te, y_te, smiles_tv, metric, args, out_root)
            seed_results.append(result)
            train_preds.append(train_pred)
            test_preds.append(test_pred)
        train_ensemble = np.mean(train_preds, axis=0).astype(np.float32)
        test_ensemble = np.mean(test_preds, axis=0).astype(np.float32)
        formal_test_preds.append(test_ensemble)
        run_dir = task_dir / f"formal_seed_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_pred(run_dir / "trainval_predictions.csv", y_tv, train_ensemble, metric)
        write_pred(run_dir / "test_predictions.csv", y_te, test_ensemble, metric)
        test_score = score(metric, y_te, test_ensemble)
        run_rows.append(
            {
                "formal_seed": run_idx,
                "inner_seeds": ",".join(str(seed) for seed in seeds),
                "inner_seed_scores": ",".join(f"{float(row['test_score']):.12f}" for row in seed_results),
                "inner_seed_mean": float(np.mean([float(row["test_score"]) for row in seed_results])),
                "inner_seed_std": float(np.std([float(row["test_score"]) for row in seed_results], ddof=0)),
                "test_score": test_score,
                "beats_top1": test_score <= top1_ref if direction == "min" else test_score >= top1_ref,
            }
        )
    scores = np.array([float(row["test_score"]) for row in run_rows], dtype=np.float64)
    cross_run_pred = np.mean(formal_test_preds, axis=0).astype(np.float32)
    summary = {
        "task": task,
        "metric": metric,
        "direction": direction,
        "top1_ref": top1_ref,
        "formal_run_scores": ",".join(f"{x:.12f}" for x in scores),
        "test_mean": float(np.mean(scores)),
        "test_std": float(np.std(scores, ddof=0)),
        "test_ensemble_score": score(metric, y_te, cross_run_pred),
        "beats_top1_mean": bool(np.mean(scores) <= top1_ref if direction == "min" else np.mean(scores) >= top1_ref),
        "beats_top1_ensemble": bool(score(metric, y_te, cross_run_pred) <= top1_ref if direction == "min" else score(metric, y_te, cross_run_pred) >= top1_ref),
        "backend": f"xgboost_fp_{args.xgb_estimators}",
        "folds": args.folds,
        "internal_repeats_per_formal_seed": len(DEFAULT_GROUPS[0]),
        "selection_note": "Pre-registered fingerprint XGBoost repeated scaffold fold-bagging; no test-side recipe selection.",
    }
    fields = list(run_rows[0].keys())
    with (task_dir / "formal_seed_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(run_rows)
    write_pred(task_dir / "cross_run_ensemble_test_predictions.csv", y_te, cross_run_pred, metric)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for task in args.tasks:
        if task not in TASKS:
            raise KeyError(f"unknown task {task}; available={sorted(TASKS)}")
        rows.append(run_task(task, args))
    fields = sorted({k for row in rows for k in row})
    with (out_root / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
