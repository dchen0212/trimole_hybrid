from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import descriptor_sidecar_official_v1 as base


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
DATA_ROOT = REPO / "data" / "data_benchmark_official_v1"
MASTER = (
    REPO
    / "results_strict"
    / "ept_family_routing_master_v1"
    / "ept_family_routing_master_v1_metric_cv_sidecar_layerwise_selected_v4.csv"
)
BASE_5SEED_ROOT = REPO / "results_strict" / "ept_family_official_v1_5seed_runs"
OUT_ROOT = REPO / "results_strict" / "official_sidecar_nested_refit_v1"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default=str(MASTER))
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--base-5seed-root", type=str, default=str(BASE_5SEED_ROOT))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument("--tasks", nargs="*", default=["pgp_broccatelli"])
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_master(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["task"]: row for row in rows if str(row.get("selected", "False")).lower() == "true"}


def get_smiles_col(df: pd.DataFrame) -> str:
    for col in ("smiles", "Drug", "SMILES", "drug"):
        if col in df.columns:
            return col
    raise KeyError("smiles column not found")


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid::{smiles}"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def build_scaffold_folds(smiles: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        scaffolds.setdefault(scaffold_from_smiles(smi), []).append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    fold_bins = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for g in groups:
        tgt = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_bins[tgt].extend(g)
        fold_sizes[tgt] += len(g)
    return [np.array(sorted(x), dtype=int) for x in fold_bins]


def load_pred_column(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "sample_idx" in df.columns:
        df = df.sort_values("sample_idx").reset_index(drop=True)
    for col in ("y_prob", "y_pred", "prediction", "pred"):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float32)
    raise KeyError(f"prediction column not found in {path}")


def find_seed_pred_files(base_root: Path, task: str, candidate: str, seeds: list[int]) -> tuple[list[Path], list[Path], list[Path]]:
    train_files: list[Path] = []
    valid_files: list[Path] = []
    test_files: list[Path] = []
    for seed in seeds:
        exact_root = base_root / f"{task}__{candidate}__seed_{seed}"
        if exact_root.exists():
            seed_roots = [exact_root]
        else:
            seed_roots = sorted(base_root.glob(f"{task}__{candidate}*__seed_{seed}"))
        if not seed_roots:
            raise FileNotFoundError(
                f"missing base prediction directory for {task} seed {seed} under {base_root}"
            )
        seed_root = seed_roots[-1]
        valid_candidates = sorted(seed_root.glob(f"run_*/{task}/valid_predictions.csv"))
        test_candidates = sorted(seed_root.glob(f"run_*/{task}/test_predictions.csv"))
        train_candidates = sorted(seed_root.glob(f"run_*/{task}/train_predictions.csv"))
        if not (train_candidates and valid_candidates and test_candidates):
            raise FileNotFoundError(f"missing base predictions for {task} seed {seed} under {seed_root}")
        train_files.append(train_candidates[-1])
        valid_files.append(valid_candidates[-1])
        test_files.append(test_candidates[-1])
    return train_files, valid_files, test_files


def average_prediction_files(paths: list[Path]) -> np.ndarray:
    arrays = [load_pred_column(path) for path in paths]
    return np.stack(arrays, axis=0).mean(axis=0).astype(np.float32)


def fit_full_model(X: np.ndarray, y: np.ndarray, task_type: str, seed: int):
    if base.HAS_XGB:
        if task_type == "classification":
            model = base.XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=8,
            )
            model.fit(X, y, verbose=False)
            return model, "xgboost"
        model = base.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=8,
        )
        model.fit(X, y, verbose=False)
        return model, "xgboost"

    if base.HAS_SK:
        if task_type == "classification":
            model = base.RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=8,
                random_state=seed,
            )
            model.fit(X, y)
            return model, "random_forest"
        model = base.RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=8,
            random_state=seed,
        )
        model.fit(X, y)
        return model, "random_forest"

    raise RuntimeError("Need xgboost or sklearn in runtime for nested sidecar refit")


def adjusted_score(mean: float, std: float, direction: str, lam: float) -> float:
    return mean - lam * std if direction == "max" else mean + lam * std


def build_variant_mats(
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
        "embedding_fp": (
            np.concatenate([emb_tr, fp_tr], axis=1),
            np.concatenate([emb_va, fp_va], axis=1),
            np.concatenate([emb_te, fp_te], axis=1),
        ),
        "embedding_fp_base_pred": (
            np.concatenate([emb_tr, fp_tr, pred_tr.reshape(-1, 1)], axis=1),
            np.concatenate([emb_va, fp_va, pred_va.reshape(-1, 1)], axis=1),
            np.concatenate([emb_te, fp_te, pred_te.reshape(-1, 1)], axis=1),
        ),
    }
    return {name: tuple(base.sanitize_features(x.astype(np.float32)) for x in mats) for name, mats in variants.items()}


def run_one(task: str, row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    task_dir = Path(args.data_root) / task
    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    train_df = pd.read_csv(task_dir / "train.csv")
    valid_df = pd.read_csv(task_dir / "valid.csv")
    test_df = pd.read_csv(task_dir / "test.csv")
    s_col = get_smiles_col(train_df)
    y_col = base.find_label_col(train_df)

    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    emb_tr, emb_va, emb_te = base.build_embedding_features(task_dir, row["candidate"], n_tr, n_va, n_te)
    fp_tr = base.get_fingerprints(train_df[s_col])
    fp_va = base.get_fingerprints(valid_df[s_col])
    fp_te = base.get_fingerprints(test_df[s_col])

    seed_list = [int(x) for x in args.seeds]
    train_pred_files, valid_pred_files, test_pred_files = find_seed_pred_files(
        Path(args.base_5seed_root), task, row["candidate"], seed_list
    )
    pred_tr = average_prediction_files(train_pred_files)
    pred_va = average_prediction_files(valid_pred_files)
    pred_te = average_prediction_files(test_pred_files)

    variants = build_variant_mats(
        emb_tr, emb_va, emb_te, fp_tr, fp_va, fp_te, pred_tr, pred_va, pred_te
    )
    y_tr = train_df[y_col].to_numpy()
    y_va = valid_df[y_col].to_numpy()
    y_te = test_df[y_col].to_numpy()
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    smiles_tv = pd.concat([train_df[s_col], valid_df[s_col]], axis=0, ignore_index=True).astype(str).tolist()
    folds = build_scaffold_folds(smiles_tv, args.folds, args.seed)
    task_type = base.infer_task_type(y_tv)
    direction = row["metric_direction"]

    cv_rows: list[dict[str, object]] = []
    best_name = ""
    best_adj = -math.inf if direction == "max" else math.inf
    best_mean = -math.inf if direction == "max" else math.inf
    best_std = math.nan

    for name, (X_tr, X_va, X_te) in variants.items():
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        fold_scores: list[float] = []
        for fold_idx, valid_idx in enumerate(folds):
            train_mask = np.ones(len(y_tv), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.where(train_mask)[0]
            model, backend = base.fit_model(
                X_tv[train_idx],
                y_tv[train_idx],
                X_tv[valid_idx],
                y_tv[valid_idx],
                task_type,
                row["tdc_metric"],
                seed=args.seed + fold_idx,
            )
            pred = base.predict_model(model, X_tv[valid_idx], task_type)
            score = float(base.score_metric(row["tdc_metric"], y_tv[valid_idx], pred))
            fold_scores.append(score)
            cv_rows.append(
                {
                    "task": task,
                    "variant": name,
                    "fold_idx": fold_idx,
                    "tdc_metric": row["tdc_metric"],
                    "metric_direction": direction,
                    "fold_score": score,
                    "backend": backend,
                }
            )

        mean = float(np.mean(fold_scores))
        std = float(np.std(fold_scores, ddof=0))
        adj = adjusted_score(mean, std, direction, args.lambda_std)
        cv_rows.append(
            {
                "task": task,
                "variant": name,
                "fold_idx": "summary",
                "tdc_metric": row["tdc_metric"],
                "metric_direction": direction,
                "cv_mean": mean,
                "cv_std": std,
                "cv_adjusted_score": adj,
            }
        )
        if direction == "max":
            is_better = (adj > best_adj) or (adj == best_adj and mean > best_mean)
        else:
            is_better = (adj < best_adj) or (adj == best_adj and mean < best_mean)
        if is_better:
            best_name = name
            best_adj = adj
            best_mean = mean
            best_std = std

    X_best_tv = np.concatenate([variants[best_name][0], variants[best_name][1]], axis=0)
    X_best_te = variants[best_name][2]
    model, backend = fit_full_model(X_best_tv, y_tv, task_type, seed=args.seed)
    pred_tv = base.predict_model(model, X_best_tv, task_type)
    pred_te = base.predict_model(model, X_best_te, task_type)
    test_score = float(base.score_metric(row["tdc_metric"], y_te, pred_te))

    base.write_predictions(out_dir / "trainval_predictions.csv", y_tv, pred_tv, task_type)
    base.write_predictions(out_dir / "test_predictions.csv", y_te, pred_te, task_type)

    with (out_dir / "cv_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in cv_rows for k in r}))
        w.writeheader()
        w.writerows(cv_rows)

    result = {
        "task": task,
        "candidate": row["candidate"],
        "head": row.get("head", ""),
        "tdc_metric": row["tdc_metric"],
        "metric_direction": direction,
        "selected_variant": best_name,
        "cv_mean": best_mean,
        "cv_std": best_std,
        "cv_adjusted_score": best_adj,
        "test_tdc_score": test_score,
        "incumbent_valid_tdc_score": float(row.get("valid_tdc_score_mean") or row.get("valid_tdc_score")),
        "incumbent_test_tdc_score": float(row.get("test_tdc_score_mean") or row.get("test_tdc_score")),
        "improved_test": base.direction_better(test_score, float(row.get("test_tdc_score_mean") or row.get("test_tdc_score")), direction),
        "tdc_top1_ref": float(row["tdc_top1_ref"]),
        "gap_vs_top1_ref": abs(test_score - float(row["tdc_top1_ref"])),
        "backend": backend,
        "base_pred_source": "avg_seed_1_to_5_official_5seed_predictions",
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
    master = read_master(Path(args.master))

    results: list[dict[str, object]] = []
    for task in args.tasks:
        if task not in master:
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
                "base_5seed_root": str(args.base_5seed_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
