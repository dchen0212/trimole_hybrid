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
    / "ept_family_routing_master_v1_metric_cv_sidecar_layerwise_selected_v4.csv"
)
OUT_ROOT = REPO / "results_strict" / "official_sidecar_bagged_blend_v1"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_BASE_ROOTS = [
    REPO / "results_strict" / "official_layerwise_selected_5seed_v1",
    REPO / "results_strict" / "official_selected_5seed_materialize_v1",
    REPO / "results_strict" / "ept_family_official_v1_5seed_runs",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default=str(MASTER))
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--base-5seed-roots", nargs="*", default=[str(x) for x in DEFAULT_BASE_ROOTS])
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=["pgp_broccatelli"])
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--xgb-estimators", type=int, default=800)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def find_seed_pred_files_any(
    roots: list[Path], task: str, candidate: str, seeds: list[int]
) -> tuple[list[Path], list[Path], list[Path], str]:
    last_error: Exception | None = None
    for root in roots:
        try:
            tr, va, te = nested.find_seed_pred_files(root, task, candidate, seeds)
            return tr, va, te, str(root)
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(f"missing 5-seed predictions for {task}; last error: {last_error}")


def fit_fold_model(
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    task_type: str,
    metric: str,
    seed: int,
    n_estimators: int,
):
    if n_estimators == 800:
        return base.fit_model(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            task_type,
            metric,
            seed=seed,
        )
    if not base.HAS_XGB:
        return base.fit_model(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            task_type,
            metric,
            seed=seed,
        )
    if task_type == "classification":
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
            early_stopping_rounds=40,
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
            early_stopping_rounds=40,
        )
    model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
    return model, f"xgboost_{n_estimators}"


def rank_values(x: np.ndarray) -> np.ndarray:
    import scipy.stats as st

    return st.rankdata(np.asarray(x, dtype=float), method="average").astype(np.float32)


def blend_prediction(a: np.ndarray, b: np.ndarray, weight_a: float, mode: str) -> np.ndarray:
    if mode == "rank":
        a = rank_values(a)
        b = rank_values(b)
    return (weight_a * a + (1.0 - weight_a) * b).astype(np.float32)


def score_foldwise(metric: str, y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> list[float]:
    scores: list[float] = []
    for idx in folds:
        scores.append(float(base.score_metric(metric, y[idx], pred[idx])))
    return scores


def better_adjusted(new_adj: float, old_adj: float, new_mean: float, old_mean: float, direction: str) -> bool:
    if direction == "max":
        return (new_adj > old_adj) or (new_adj == old_adj and new_mean > old_mean)
    return (new_adj < old_adj) or (new_adj == old_adj and new_mean < old_mean)


def build_variants(
    emb_tr: np.ndarray,
    emb_va: np.ndarray,
    emb_te: np.ndarray,
    fp_tr: np.ndarray,
    fp_va: np.ndarray,
    fp_te: np.ndarray,
    pred_tr: np.ndarray,
    pred_va: np.ndarray,
    pred_te: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variants = {
        "tabular_fp": (fp_tr, fp_va, fp_te),
        "tabular_fp_base_pred": (
            np.concatenate([fp_tr, pred_tr.reshape(-1, 1)], axis=1),
            np.concatenate([fp_va, pred_va.reshape(-1, 1)], axis=1),
            np.concatenate([fp_te, pred_te.reshape(-1, 1)], axis=1),
        ),
    }
    variants.update(nested.build_variant_mats(emb_tr, emb_va, emb_te, fp_tr, fp_va, fp_te, pred_tr, pred_va, pred_te))
    return {k: tuple(base.sanitize_features(x.astype(np.float32)) for x in v) for k, v in variants.items()}


def run_one(task: str, row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
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

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, row["candidate"], n_tr, n_va, n_te)
    fp_tr = base.get_fingerprints(train_df[s_col])
    fp_va = base.get_fingerprints(valid_df[s_col])
    fp_te = base.get_fingerprints(test_df[s_col])

    seed_list = [int(x) for x in args.seeds]
    pred_roots = [Path(x) for x in args.base_5seed_roots]
    train_pred_files, valid_pred_files, test_pred_files, pred_root = find_seed_pred_files_any(
        pred_roots, task, row["candidate"], seed_list
    )
    pred_tr = nested.average_prediction_files(train_pred_files)
    pred_va = nested.average_prediction_files(valid_pred_files)
    pred_te = nested.average_prediction_files(test_pred_files)

    variants = build_variants(emb_tr, emb_va, emb_te, fp_tr, fp_va, fp_te, pred_tr, pred_va, pred_te)
    if args.variants:
        wanted = set(args.variants)
        variants = {name: mats for name, mats in variants.items() if name in wanted}
        missing = sorted(wanted.difference(variants))
        if missing:
            raise KeyError(f"requested variants not available for {task}: {missing}")
    y_tr = train_df[y_col].to_numpy()
    y_va = valid_df[y_col].to_numpy()
    y_te = test_df[y_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    base_tv = np.concatenate([pred_tr, pred_va], axis=0).astype(np.float32)
    base_te = pred_te.astype(np.float32)
    smiles_tv = pd.concat([train_df[s_col], valid_df[s_col]], axis=0, ignore_index=True).astype(str).tolist()
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, args.seed)

    task_type = base.infer_task_type(y_tv)
    direction = row["metric_direction"]
    metric = row["tdc_metric"]
    weight_values = np.round(np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step), 6)
    blend_modes = ["raw", "rank"] if str(metric).upper() == "SPEARMAN" else ["raw"]

    cv_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_adj = -math.inf if direction == "max" else math.inf
    best_mean = -math.inf if direction == "max" else math.inf

    for name, (X_tr, X_va, X_te) in variants.items():
        print(f"[variant] {task}::{name}", flush=True)
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        oof = np.zeros(len(y_tv), dtype=np.float32)
        test_preds: list[np.ndarray] = []
        backend = ""
        for fold_idx, valid_idx in enumerate(folds):
            train_mask = np.ones(len(y_tv), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.where(train_mask)[0]
            model, backend = fit_fold_model(
                X_tv,
                y_tv,
                train_idx,
                valid_idx,
                task_type,
                metric,
                seed=args.seed + fold_idx,
                n_estimators=args.xgb_estimators,
            )
            oof[valid_idx] = base.predict_model(model, X_tv[valid_idx], task_type)
            test_preds.append(base.predict_model(model, X_te, task_type).astype(np.float32))
        bag_test = np.stack(test_preds, axis=0).mean(axis=0).astype(np.float32)

        for mode in blend_modes:
            for weight in weight_values:
                # weight applies to the bagged tabular/sidecar branch; 0.0 is incumbent-only.
                oof_blend = blend_prediction(oof, base_tv, float(weight), mode)
                test_blend = blend_prediction(bag_test, base_te, float(weight), mode)
                fold_scores = score_foldwise(metric, y_tv, oof_blend, folds)
                mean = float(np.mean(fold_scores))
                std = float(np.std(fold_scores, ddof=0))
                adj = nested.adjusted_score(mean, std, direction, args.lambda_std)
                oof_score = float(base.score_metric(metric, y_tv, oof_blend))
                test_score = float(base.score_metric(metric, y_te, test_blend))
                row_out = {
                    "task": task,
                    "variant": name,
                    "blend_mode": mode,
                    "weight_sidecar": float(weight),
                    "tdc_metric": metric,
                    "metric_direction": direction,
                    "cv_mean": mean,
                    "cv_std": std,
                    "cv_adjusted_score": adj,
                    "cv_oof_score": oof_score,
                    "test_tdc_score": test_score,
                    "backend": backend,
                }
                cv_rows.append(row_out)
                if best is None or better_adjusted(adj, best_adj, mean, best_mean, direction):
                    best = dict(row_out)
                    best["test_predictions"] = test_blend
                    best["trainval_predictions"] = oof_blend
                    best_adj = adj
                    best_mean = mean

    if best is None:
        raise RuntimeError(f"no bagged-blend candidate evaluated for {task}")

    trainval_pred = np.asarray(best.pop("trainval_predictions"), dtype=np.float32)
    test_pred = np.asarray(best.pop("test_predictions"), dtype=np.float32)
    base.write_predictions(out_dir / "trainval_predictions.csv", y_tv, trainval_pred, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, test_pred, task_type)
    with (out_dir / "cv_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in cv_rows for k in r}))
        w.writeheader()
        w.writerows(cv_rows)

    incumbent_test = float(row.get("test_tdc_score_mean") or row.get("test_tdc_score"))
    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": metric,
        "metric_direction": direction,
        "selected_variant": best["variant"],
        "blend_mode": best["blend_mode"],
        "weight_sidecar": best["weight_sidecar"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "cv_adjusted_score": best["cv_adjusted_score"],
        "cv_oof_score": best["cv_oof_score"],
        "test_tdc_score": best["test_tdc_score"],
        "incumbent_valid_tdc_score": float(row.get("valid_tdc_score_mean") or row.get("valid_tdc_score")),
        "incumbent_test_tdc_score": incumbent_test,
        "improved_test": base.direction_better(float(best["test_tdc_score"]), incumbent_test, direction),
        "tdc_top1_ref": float(row["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(float(best["test_tdc_score"]) - float(row["tdc_top1_ref"])),
        "backend": best["backend"],
        "base_pred_source_root": pred_root,
        "endpoint": "fold_bagging_plus_weighted_blend",
        "trainval_pred_file": str(out_dir / "trainval_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "cv_rows_file": str(out_dir / "cv_rows.csv"),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    master = nested.read_master(Path(args.master))

    results: list[dict[str, object]] = []
    for task in args.tasks:
        if task not in master:
            results.append({"task": task, "status": "missing_from_master"})
            continue
        print(f"[task] {task}", flush=True)
        try:
            results.append(run_one(task, master[task], args))
        except Exception as exc:
            results.append({"task": task, "status": "error", "error": str(exc)})

    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        w.writeheader()
        w.writerows(results)
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "master": str(args.master),
                "tasks": args.tasks,
                "folds": args.folds,
                "seed": args.seed,
                "base_5seed_roots": args.base_5seed_roots,
                "endpoint": "fold_bagging_plus_weighted_blend",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
