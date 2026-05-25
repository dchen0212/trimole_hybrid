from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import descriptor_sidecar_official_v1 as base
import official_sidecar_bagged_blend_v1 as bagged
import official_sidecar_nested_refit_v1 as nested
import paper_main_chemical_prior_v2 as chemv2
import paper_main_multimodal_prior_taskwise_v1 as v1

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None  # type: ignore
    CatBoostRegressor = None  # type: ignore
    HAS_CATBOOST = False


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
OUT_ROOT = REPO / "results_strict" / "paper_main_chem_select_multibackend_v3"
FOCUS_TASKS = [
    "clearance_microsome_az",
    "hia_hou",
    "cyp2d6_substrate_carbonmangels",
    "ames",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(v1.MASTER))
    p.add_argument("--data-root", default=str(v1.DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--base-5seed-roots", nargs="*", default=[str(x) for x in v1.DEFAULT_BASE_ROOTS])
    p.add_argument("--tasks", nargs="*", default=FOCUS_TASKS)
    p.add_argument("--seeds", nargs="*", type=int, default=v1.DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--xgb-estimators", type=int, default=220)
    p.add_argument("--tree-estimators", type=int, default=400)
    p.add_argument("--cat-estimators", type=int, default=300)
    p.add_argument("--topk", nargs="*", type=int, default=[256, 512, 1024, 2048])
    p.add_argument("--backends", nargs="*", default=["xgb", "catboost", "extratrees", "rf", "linear"])
    p.add_argument("--chemical-blocks", nargs="*", default=["core_maccs_fcfp", "core_pair_torsion", "wide_chem"])
    p.add_argument("--variants", nargs="*", default=["chem", "embed_chem", "chem_base_pred", "embed_chem_base_pred"])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def fit_selector(X: np.ndarray, y: np.ndarray, task_type: str, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_features = X.shape[1]
    if k <= 0 or k >= n_features:
        return np.arange(n_features, dtype=np.int64)

    var = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.flatnonzero(var > 1e-12)
    if len(keep) <= k:
        return keep.astype(np.int64)

    Xk = X[:, keep]
    try:
        if task_type == "classification":
            scores, _ = f_classif(Xk, y)
        else:
            scores, _ = f_regression(Xk, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        scores = var[keep]
    top_local = np.argsort(scores)[-k:]
    return np.sort(keep[top_local]).astype(np.int64)


def fit_backend(
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    metric: str,
    seed: int,
    args: argparse.Namespace,
):
    if backend == "xgb":
        if not base.HAS_XGB:
            raise RuntimeError("xgboost unavailable")
        if task_type == "classification":
            pos = float(np.sum(np.asarray(y_train) == 1))
            neg = float(len(y_train) - pos)
            model = base.XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=4,
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
                scale_pos_weight=max(1.0, neg / max(pos, 1.0)),
            )
        else:
            model = base.XGBRegressor(
                n_estimators=args.xgb_estimators,
                max_depth=4,
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
        model.fit(X_train, y_train, verbose=False)
        return model

    if backend == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost unavailable")
        common = dict(
            iterations=args.cat_estimators,
            depth=6,
            learning_rate=0.04,
            random_seed=seed,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
        if task_type == "classification":
            model = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **common)
        else:
            model = CatBoostRegressor(loss_function="RMSE", **common)
        model.fit(X_train, y_train)
        return model

    if backend == "extratrees":
        if task_type == "classification":
            model = ExtraTreesClassifier(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            )
        model.fit(X_train, y_train)
        return model

    if backend == "rf":
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=args.tree_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            )
        model.fit(X_train, y_train)
        return model

    if backend == "linear":
        if task_type == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            )
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed))
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"unknown backend: {backend}")


def predict_backend(model, X: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1].astype(np.float32)
        if hasattr(model, "decision_function"):
            z = model.decision_function(X)
            return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def add_variant(
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    name: str,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
) -> None:
    out[name] = tuple(base.sanitize_features(x.astype(np.float32)) for x in (tr, va, te))  # type: ignore[assignment]


def build_variants(
    emb_tr: np.ndarray,
    emb_va: np.ndarray,
    emb_te: np.ndarray,
    chem_tr: dict[str, np.ndarray],
    chem_va: dict[str, np.ndarray],
    chem_te: dict[str, np.ndarray],
    pred_tr: np.ndarray | None,
    pred_va: np.ndarray | None,
    pred_te: np.ndarray | None,
    blocks: list[str],
    variant_modes: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in blocks:
        tr, va, te = chem_tr[block], chem_va[block], chem_te[block]
        if "chem" in variant_modes:
            add_variant(variants, f"chem_{block}", tr, va, te)
        if "embed_chem" in variant_modes:
            add_variant(
                variants,
                f"embed_chem_{block}",
                np.concatenate([emb_tr, tr], axis=1),
                np.concatenate([emb_va, va], axis=1),
                np.concatenate([emb_te, te], axis=1),
            )
        if pred_tr is not None and pred_va is not None and pred_te is not None:
            if "chem_base_pred" in variant_modes:
                add_variant(
                    variants,
                    f"chem_{block}_base_pred",
                    np.concatenate([tr, pred_tr.reshape(-1, 1)], axis=1),
                    np.concatenate([va, pred_va.reshape(-1, 1)], axis=1),
                    np.concatenate([te, pred_te.reshape(-1, 1)], axis=1),
                )
            if "embed_chem_base_pred" in variant_modes:
                add_variant(
                    variants,
                    f"embed_chem_{block}_base_pred",
                    np.concatenate([emb_tr, tr, pred_tr.reshape(-1, 1)], axis=1),
                    np.concatenate([emb_va, va, pred_va.reshape(-1, 1)], axis=1),
                    np.concatenate([emb_te, te, pred_te.reshape(-1, 1)], axis=1),
                )
    return variants


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
    smiles_col = nested.get_smiles_col(train_df)
    label_col = base.find_label_col(train_df)

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, row["candidate"], n_tr, n_va, n_te)
    chem_tr = chemv2.extra_chemical_blocks(train_df[smiles_col])
    chem_va = chemv2.extra_chemical_blocks(valid_df[smiles_col])
    chem_te = chemv2.extra_chemical_blocks(test_df[smiles_col])

    train_pred_files, valid_pred_files, test_pred_files, pred_source = v1.find_seed_predictions_optional(
        [Path(x) for x in args.base_5seed_roots],
        task,
        row["candidate"],
        [int(x) for x in args.seeds],
    )
    pred_tr = v1.average_optional(train_pred_files)
    pred_va = v1.average_optional(valid_pred_files)
    pred_te = v1.average_optional(test_pred_files)
    has_base_blend = pred_tr is not None and pred_va is not None and pred_te is not None

    variants = build_variants(
        emb_tr,
        emb_va,
        emb_te,
        chem_tr,
        chem_va,
        chem_te,
        pred_tr,
        pred_va,
        pred_te,
        args.chemical_blocks,
        set(args.variants),
    )

    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, args.seed)
    task_type = base.infer_task_type(y_tv)
    metric = row["tdc_metric"]
    direction = row["metric_direction"]

    base_tv = base_te = None
    if has_base_blend:
        base_tv = np.concatenate([pred_tr, pred_va], axis=0).astype(np.float32)
        base_te = pred_te.astype(np.float32)
    weight_values = [1.0] if not has_base_blend else list(
        np.round(np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step), 6)
    )
    blend_modes = ["raw"]
    if has_base_blend and str(metric).upper() == "SPEARMAN":
        blend_modes.append("rank")

    cv_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_adj = -math.inf if direction == "max" else math.inf
    best_mean = -math.inf if direction == "max" else math.inf

    for variant_name, (X_tr, X_va, X_te) in variants.items():
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        topk_values = list(args.topk)
        if 0 not in topk_values:
            topk_values.append(0)
        for topk in topk_values:
            for backend in args.backends:
                print(f"[trial] {task}::{variant_name} topk={topk or 'all'} backend={backend}", flush=True)
                oof = np.zeros(len(y_tv), dtype=np.float32)
                test_preds: list[np.ndarray] = []
                try:
                    for fold_idx, valid_idx in enumerate(folds):
                        train_mask = np.ones(len(y_tv), dtype=bool)
                        train_mask[valid_idx] = False
                        train_idx = np.where(train_mask)[0]
                        selected = fit_selector(X_tv[train_idx], y_tv[train_idx], task_type, topk)
                        model = fit_backend(
                            backend,
                            X_tv[train_idx][:, selected],
                            y_tv[train_idx],
                            task_type,
                            metric,
                            args.seed + fold_idx,
                            args,
                        )
                        oof[valid_idx] = predict_backend(model, X_tv[valid_idx][:, selected], task_type)
                        test_preds.append(predict_backend(model, X_te[:, selected], task_type))
                except Exception as exc:
                    cv_rows.append(
                        {
                            "task": task,
                            "variant": variant_name,
                            "topk": topk,
                            "backend": backend,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    continue

                bag_test = np.stack(test_preds, axis=0).mean(axis=0).astype(np.float32)
                for mode in blend_modes:
                    for weight in weight_values:
                        if has_base_blend and base_tv is not None and base_te is not None:
                            oof_eval = v1.blend_prediction(oof, base_tv, float(weight), mode)
                            test_eval = v1.blend_prediction(bag_test, base_te, float(weight), mode)
                        else:
                            oof_eval = oof
                            test_eval = bag_test
                        fold_scores = v1.score_foldwise(metric, y_tv, oof_eval, folds)
                        mean = float(np.mean(fold_scores))
                        std = float(np.std(fold_scores, ddof=0))
                        adj = v1.adjusted_score(mean, std, direction, args.lambda_std)
                        oof_score = float(base.score_metric(metric, y_tv, oof_eval))
                        test_score = float(base.score_metric(metric, y_te, test_eval))
                        row_out = {
                            "task": task,
                            "variant": variant_name,
                            "topk": topk if topk > 0 else "all",
                            "backend": backend,
                            "blend_mode": mode,
                            "weight_sidecar": float(weight),
                            "tdc_metric": metric,
                            "metric_direction": direction,
                            "cv_mean": mean,
                            "cv_std": std,
                            "cv_adjusted_score": adj,
                            "cv_oof_score": oof_score,
                            "test_tdc_score": test_score,
                            "status": "ok",
                        }
                        cv_rows.append(row_out)
                        if best is None or v1.better_adjusted(adj, best_adj, mean, best_mean, direction):
                            best = dict(row_out)
                            best["trainval_predictions"] = oof_eval
                            best["test_predictions"] = test_eval
                            best_adj = adj
                            best_mean = mean

    if best is None:
        raise RuntimeError(f"no successful v3 trial for {task}")

    trainval_pred = np.asarray(best.pop("trainval_predictions"), dtype=np.float32)
    test_pred = np.asarray(best.pop("test_predictions"), dtype=np.float32)
    base.write_predictions(out_dir / "trainval_predictions.csv", y_tv, trainval_pred, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, test_pred, task_type)
    with (out_dir / "cv_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for x in cv_rows for k in x}))
        writer.writeheader()
        writer.writerows(cv_rows)

    incumbent_test = float(row.get("test_tdc_score_mean") or row.get("test_tdc_score") or row.get("test_score"))
    top1_ref = float(row["tdc_top1_ref"])
    test_score = float(best["test_tdc_score"])
    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": metric,
        "metric_direction": direction,
        "selected_variant": best["variant"],
        "selected_topk": best["topk"],
        "selected_backend": best["backend"],
        "blend_mode": best["blend_mode"],
        "weight_sidecar": best["weight_sidecar"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "cv_adjusted_score": best["cv_adjusted_score"],
        "cv_oof_score": best["cv_oof_score"],
        "test_tdc_score": test_score,
        "incumbent_test_tdc_score": incumbent_test,
        "improved_test": base.direction_better(test_score, incumbent_test, direction),
        "tdc_top1_ref": top1_ref,
        "is_top1_level": (test_score >= top1_ref if direction == "max" else test_score <= top1_ref),
        "gap_vs_top1_ref": abs(test_score - top1_ref),
        "has_base_blend": has_base_blend,
        "base_pred_source_root": pred_source,
        "endpoint": "chemical_feature_selection_multibackend_scaffold_cv_bagging",
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
    (out_root / "METHOD.md").write_text(
        "# Chemical feature selection + multi-backend v3\n\n"
        "Per task, select chemical/multimodal feature variants, top-k feature counts, backend models, "
        "and blend weights by scaffold-CV mean minus lambda times std. Official test is only scored after selection.\n"
    )
    rows = v1.load_master(Path(args.master), args.tasks)
    results: list[dict[str, object]] = []
    for row in rows:
        print(f"[task] {row['task']} candidate={row['candidate']}", flush=True)
        try:
            results.append(run_one(row, args))
        except Exception as exc:
            results.append({"task": row.get("task", ""), "status": "error", "error": str(exc)})
            print(f"[error] {row.get('task', '')}: {exc}", flush=True)
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for x in results for k in x}))
        writer.writeheader()
        writer.writerows(results)
    (out_root / "meta.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
