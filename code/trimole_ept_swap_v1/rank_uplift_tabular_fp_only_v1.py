from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import descriptor_sidecar_official_v1 as base
import official_sidecar_nested_refit_v1 as nested


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_bagged_blend_pgp_frozen_microsome_seedbag_audited_v10.csv"
)
OUT_ROOT = REPO / "results_strict" / "rank_uplift_tabular_fp_only_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(MASTER))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat-seeds", nargs="*", type=int, default=[])
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--xgb-estimators", type=int, default=300)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def direction_better(new_value: float, old_value: float, direction: str) -> bool:
    return new_value > old_value if direction == "max" else new_value < old_value


def adjusted_score(mean: float, std: float, direction: str, lam: float) -> float:
    return mean - lam * std if direction == "max" else mean + lam * std


def fit_model(X, y, task_type: str, metric: str, seed: int, n_estimators: int):
    if not base.HAS_XGB:
        return base.fit_model(X, y, X, y, task_type, metric, seed=seed)
    if task_type == "classification":
        pos = float(np.sum(np.asarray(y) == 1))
        neg = float(len(y) - pos)
        scale_pos_weight = max(1.0, neg / max(pos, 1.0))
        model = base.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        model = base.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=1,
        )
    model.fit(X, y, verbose=False)
    return model, f"xgboost_{n_estimators}"


def score_foldwise(metric: str, y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> list[float]:
    return [float(base.score_metric(metric, y[idx], pred[idx])) for idx in folds]


def load_master(path: Path, tasks: list[str]) -> list[dict[str, str]]:
    rows = [row for row in csv.DictReader(path.open()) if str(row.get("selected", "False")).lower() == "true"]
    if tasks:
        wanted = set(tasks)
        rows = [row for row in rows if row["task"] in wanted]
    return rows


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    task = row["task"]
    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    task_dir = Path(args.data_root) / task
    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    s_col = nested.get_smiles_col(train_df)
    y_col = base.find_label_col(train_df)

    X_tr = base.get_fingerprints(train_df[s_col])
    X_va = base.get_fingerprints(valid_df[s_col])
    X_te = base.get_fingerprints(test_df[s_col])
    y_tr = train_df[y_col].to_numpy()
    y_va = valid_df[y_col].to_numpy()
    y_te = test_df[y_col].to_numpy()
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[s_col], valid_df[s_col]], ignore_index=True).astype(str).tolist()
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, args.seed)
    task_type = base.infer_task_type(y_tv)

    repeat_seeds = args.repeat_seeds or [args.seed]
    oof_sum = np.zeros(len(y_tv), dtype=np.float64)
    oof_count = np.zeros(len(y_tv), dtype=np.float64)
    test_preds: list[np.ndarray] = []
    all_fold_scores: list[float] = []
    backend = ""
    for repeat_seed in repeat_seeds:
        folds = nested.build_scaffold_folds(smiles_tv, args.folds, repeat_seed)
        for fold_idx, valid_idx in enumerate(folds):
            mask = np.ones(len(y_tv), dtype=bool)
            mask[valid_idx] = False
            train_idx = np.where(mask)[0]
            model, backend = fit_model(
                X_tv[train_idx],
                y_tv[train_idx],
                task_type,
                row["tdc_metric"],
                repeat_seed * 100 + fold_idx,
                args.xgb_estimators,
            )
            fold_pred = base.predict_model(model, X_tv[valid_idx], task_type).astype(np.float32)
            oof_sum[valid_idx] += fold_pred
            oof_count[valid_idx] += 1.0
            all_fold_scores.append(float(base.score_metric(row["tdc_metric"], y_tv[valid_idx], fold_pred)))
            test_preds.append(base.predict_model(model, X_te, task_type).astype(np.float32))
    oof = (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32)
    test_pred = np.mean(test_preds, axis=0).astype(np.float32)

    fold_scores = all_fold_scores
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores, ddof=0))
    cv_adj = adjusted_score(cv_mean, cv_std, row["metric_direction"], args.lambda_std)
    cv_oof = float(base.score_metric(row["tdc_metric"], y_tv, oof))
    test_score = float(base.score_metric(row["tdc_metric"], y_te, test_pred))

    base.write_predictions(out_dir / "trainval_predictions.csv", y_tv, oof, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, test_pred, task_type)

    incumbent_test = float(row.get("test_tdc_score_mean") or row.get("test_tdc_score") or row.get("test_score"))
    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": row["tdc_metric"],
        "metric_direction": row["metric_direction"],
        "selected_variant": "tabular_fp_only",
        "backend": backend,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_adjusted_score": cv_adj,
        "cv_oof_score": cv_oof,
        "test_tdc_score": test_score,
        "incumbent_test_tdc_score": incumbent_test,
        "improved_test": direction_better(test_score, incumbent_test, row["metric_direction"]),
        "tdc_top1_ref": float(row["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(test_score - float(row["tdc_top1_ref"])),
        "endpoint": "tabular_fp_only_repeated_fold_bagging" if len(repeat_seeds) > 1 else "tabular_fp_only_fold_bagging",
        "repeat_seeds": ",".join(str(x) for x in repeat_seeds),
        "trainval_pred_file": str(out_dir / "trainval_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    rows = load_master(Path(args.master), args.tasks)
    for idx, row in enumerate(rows, 1):
        print(f"[{idx}/{len(rows)}] {row['task']}", flush=True)
        try:
            results.append(run_one(row, args))
        except Exception as exc:
            results.append({"task": row["task"], "status": "error", "error": str(exc)})
    fields = sorted({k for row in results for k in row})
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (out_root / "meta.json").write_text(json.dumps({"tasks": [row["task"] for row in rows]}, indent=2))
    print(out_root / "summary.csv")


if __name__ == "__main__":
    main()
