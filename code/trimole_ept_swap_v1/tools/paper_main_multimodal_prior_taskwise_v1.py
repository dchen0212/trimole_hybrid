from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import descriptor_sidecar_official_v1 as base
import official_sidecar_bagged_blend_v1 as bagged
import official_sidecar_nested_refit_v1 as nested


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_layerwise_selected_v4.csv"
)
OUT_ROOT = REPO / "results_strict" / "paper_main_multimodal_prior_taskwise_v1"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_BASE_ROOTS = [
    REPO / "results_strict" / "official_layerwise_selected_5seed_v1",
    REPO / "results_strict" / "official_selected_5seed_materialize_v1",
    REPO / "results_strict" / "ept_family_official_v1_5seed_runs",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=str(MASTER))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--base-5seed-roots", nargs="*", default=[str(x) for x in DEFAULT_BASE_ROOTS])
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--xgb-estimators", type=int, default=500)
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_master(path: Path, tasks: list[str]) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    rows = [r for r in rows if not r.get("selected") or str(r.get("selected")).lower() == "true"]
    if tasks:
        wanted = set(tasks)
        rows = [r for r in rows if r["task"] in wanted]
    return rows


def load_pred_column_optional(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return nested.load_pred_column(path)


def find_seed_predictions_optional(
    roots: list[Path], task: str, candidate: str, seeds: list[int]
) -> tuple[list[Path] | None, list[Path] | None, list[Path] | None, str]:
    last_error = ""
    for root in roots:
        train_files: list[Path] = []
        valid_files: list[Path] = []
        test_files: list[Path] = []
        found_any = False
        for seed in seeds:
            exact_root = root / f"{task}__{candidate}__seed_{seed}"
            seed_roots = [exact_root] if exact_root.exists() else sorted(root.glob(f"{task}__{candidate}*__seed_{seed}"))
            if not seed_roots:
                last_error = f"missing seed root for {task} seed {seed} under {root}"
                break
            seed_root = seed_roots[-1]
            valid_candidates = sorted(seed_root.glob(f"run_*/{task}/valid_predictions.csv"))
            test_candidates = sorted(seed_root.glob(f"run_*/{task}/test_predictions.csv"))
            train_candidates = sorted(seed_root.glob(f"run_*/{task}/train_predictions.csv"))
            if not (valid_candidates and test_candidates):
                last_error = f"missing valid/test predictions for {task} seed {seed} under {seed_root}"
                break
            valid_files.append(valid_candidates[-1])
            test_files.append(test_candidates[-1])
            if train_candidates:
                train_files.append(train_candidates[-1])
            found_any = True
        else:
            if found_any:
                return (train_files if len(train_files) == len(seeds) else None), valid_files, test_files, str(root)
    return None, None, None, last_error


def average_optional(paths: list[Path] | None) -> np.ndarray | None:
    if not paths:
        return None
    return nested.average_prediction_files(paths)


def adjusted_score(mean: float, std: float, direction: str, lam: float) -> float:
    return mean - lam * std if direction == "max" else mean + lam * std


def better_adjusted(new_adj: float, old_adj: float, new_mean: float, old_mean: float, direction: str) -> bool:
    if direction == "max":
        return (new_adj > old_adj) or (new_adj == old_adj and new_mean > old_mean)
    return (new_adj < old_adj) or (new_adj == old_adj and new_mean < old_mean)


def blend_prediction(side: np.ndarray, base_pred: np.ndarray, weight_side: float, mode: str) -> np.ndarray:
    if mode == "rank":
        side = bagged.rank_values(side)
        base_pred = bagged.rank_values(base_pred)
    return (weight_side * side + (1.0 - weight_side) * base_pred).astype(np.float32)


def score_foldwise(metric: str, y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> list[float]:
    return [float(base.score_metric(metric, y[idx], pred[idx])) for idx in folds]


def build_variants(
    emb_tr: np.ndarray,
    emb_va: np.ndarray,
    emb_te: np.ndarray,
    fp_tr: np.ndarray,
    fp_va: np.ndarray,
    fp_te: np.ndarray,
    pred_tr: np.ndarray | None,
    pred_va: np.ndarray | None,
    pred_te: np.ndarray | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "chemical_prior_fp": (fp_tr, fp_va, fp_te),
        "multimodal_embedding_fp": (
            np.concatenate([emb_tr, fp_tr], axis=1),
            np.concatenate([emb_va, fp_va], axis=1),
            np.concatenate([emb_te, fp_te], axis=1),
        ),
    }
    if pred_tr is not None and pred_va is not None and pred_te is not None:
        variants["chemical_prior_fp_base_pred"] = (
            np.concatenate([fp_tr, pred_tr.reshape(-1, 1)], axis=1),
            np.concatenate([fp_va, pred_va.reshape(-1, 1)], axis=1),
            np.concatenate([fp_te, pred_te.reshape(-1, 1)], axis=1),
        )
        variants["multimodal_embedding_fp_base_pred"] = (
            np.concatenate([emb_tr, fp_tr, pred_tr.reshape(-1, 1)], axis=1),
            np.concatenate([emb_va, fp_va, pred_va.reshape(-1, 1)], axis=1),
            np.concatenate([emb_te, fp_te, pred_te.reshape(-1, 1)], axis=1),
        )
    return {name: tuple(base.sanitize_features(x.astype(np.float32)) for x in mats) for name, mats in variants.items()}


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
    fp_tr = base.get_fingerprints(train_df[smiles_col])
    fp_va = base.get_fingerprints(valid_df[smiles_col])
    fp_te = base.get_fingerprints(test_df[smiles_col])

    train_pred_files, valid_pred_files, test_pred_files, pred_source = find_seed_predictions_optional(
        [Path(x) for x in args.base_5seed_roots], task, row["candidate"], [int(x) for x in args.seeds]
    )
    pred_tr = average_optional(train_pred_files)
    pred_va = average_optional(valid_pred_files)
    pred_te = average_optional(test_pred_files)
    has_base_blend = pred_tr is not None and pred_va is not None and pred_te is not None

    variants = build_variants(emb_tr, emb_va, emb_te, fp_tr, fp_va, fp_te, pred_tr, pred_va, pred_te)
    if args.variants:
        wanted = set(args.variants)
        variants = {name: mats for name, mats in variants.items() if name in wanted}
        missing = sorted(wanted.difference(variants))
        if missing:
            raise KeyError(f"requested variants unavailable for {task}: {missing}")

    y_tr = train_df[label_col].to_numpy()
    y_va = valid_df[label_col].to_numpy()
    y_te = test_df[label_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[smiles_col], valid_df[smiles_col]], ignore_index=True).astype(str).tolist()
    folds = nested.build_scaffold_folds(smiles_tv, args.folds, args.seed)
    task_type = base.infer_task_type(y_tv)
    metric = row["tdc_metric"]
    direction = row["metric_direction"]

    base_tv = None
    base_te = None
    if has_base_blend:
        base_tv = np.concatenate([pred_tr, pred_va], axis=0).astype(np.float32)
        base_te = pred_te.astype(np.float32)

    weight_values = [1.0] if not has_base_blend else list(np.round(np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step), 6))
    blend_modes = ["raw"]
    if has_base_blend and str(metric).upper() == "SPEARMAN":
        blend_modes.append("rank")

    cv_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_adj = -math.inf if direction == "max" else math.inf
    best_mean = -math.inf if direction == "max" else math.inf

    for variant_name, (X_tr, X_va, X_te) in variants.items():
        print(f"[variant] {task}::{variant_name}", flush=True)
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        oof = np.zeros(len(y_tv), dtype=np.float32)
        test_preds: list[np.ndarray] = []
        backend = ""
        for fold_idx, valid_idx in enumerate(folds):
            train_mask = np.ones(len(y_tv), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.where(train_mask)[0]
            model, backend = bagged.fit_fold_model(
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
                if has_base_blend and base_tv is not None and base_te is not None:
                    oof_eval = blend_prediction(oof, base_tv, float(weight), mode)
                    test_eval = blend_prediction(bag_test, base_te, float(weight), mode)
                else:
                    oof_eval = oof
                    test_eval = bag_test
                fold_scores = score_foldwise(metric, y_tv, oof_eval, folds)
                mean = float(np.mean(fold_scores))
                std = float(np.std(fold_scores, ddof=0))
                adj = adjusted_score(mean, std, direction, args.lambda_std)
                oof_score = float(base.score_metric(metric, y_tv, oof_eval))
                test_score = float(base.score_metric(metric, y_te, test_eval))
                cv_row = {
                    "task": task,
                    "variant": variant_name,
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
                cv_rows.append(cv_row)
                if best is None or better_adjusted(adj, best_adj, mean, best_mean, direction):
                    best = dict(cv_row)
                    best["trainval_predictions"] = oof_eval
                    best["test_predictions"] = test_eval
                    best_adj = adj
                    best_mean = mean

    if best is None:
        raise RuntimeError(f"no variants evaluated for {task}")

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
        "backend": best["backend"],
        "has_base_blend": has_base_blend,
        "base_pred_source_root": pred_source,
        "endpoint": "multimodal_embedding_plus_chemical_prior_taskwise_cv_bagging",
        "trainval_pred_file": str(out_dir / "trainval_predictions.csv"),
        "test_pred_file": str(out_dir / "test_predictions.csv"),
        "cv_rows_file": str(out_dir / "cv_rows.csv"),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def write_method_note(out_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Paper-main ADMET pipeline v1

Core framing: multimodal molecular representation + chemical prior sidecar + task-adaptive robust selection.

Pipeline:
1. Use the frozen EPT-family routing master as the deep representation backbone for each task.
2. Build task-specific multimodal embeddings from the selected ChemBERTa/KPGT/EPT route.
3. Add chemistry priors using RDKit descriptors plus ECFP/Avalon/ErG fingerprints.
4. Train sidecar branches with scaffold-CV fold bagging.
5. Select variants and simple blend weights using CV mean - lambda * CV std, never using official test for selection.
6. Report official test scores only after CV selection.

Run config:
- master: {args.master}
- folds: {args.folds}
- seed: {args.seed}
- lambda_std: {args.lambda_std}
- xgb_estimators: {args.xgb_estimators}
- weight_step: {args.weight_step}
"""
    (out_root / "METHOD.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_method_note(out_root, args)

    rows = load_master(Path(args.master), args.tasks)
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
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "master": args.master,
                "data_root": args.data_root,
                "out_root": args.out_root,
                "tasks": [r["task"] for r in rows],
                "folds": args.folds,
                "seed": args.seed,
                "lambda_std": args.lambda_std,
                "xgb_estimators": args.xgb_estimators,
                "weight_step": args.weight_step,
                "base_5seed_roots": args.base_5seed_roots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
