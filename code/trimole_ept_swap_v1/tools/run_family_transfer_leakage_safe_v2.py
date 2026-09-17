"""Leakage-safe family-transfer training for the Bioinformatics revision.

Candidate selection uses target validation labels only. Official test labels are
scored only after one feature/configuration candidate has been frozen per target.
Cross-task training molecules that overlap a target holdout are removed using
connectivity-level InChIKey identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from family_transfer_safety import (
    canonical_connectivity_keys,
    forbidden_keys,
    keep_mask,
    select_validation_candidate,
)


DEFAULT_REPO = Path(__file__).resolve().parents[1]
FAMILIES: dict[str, dict[str, Any]] = {
    "cyp_substrate": {
        "kind": "classification",
        "tasks": [
            "cyp2c9_substrate_carbonmangels",
            "cyp2d6_substrate_carbonmangels",
            "cyp3a4_substrate_carbonmangels",
        ],
        "metrics": {
            "cyp2c9_substrate_carbonmangels": "auprc",
            "cyp2d6_substrate_carbonmangels": "auprc",
            "cyp3a4_substrate_carbonmangels": "auroc",
        },
    },
    "clearance": {
        "kind": "regression",
        "tasks": ["clearance_hepatocyte_az", "clearance_microsome_az"],
        "metrics": {
            "clearance_hepatocyte_az": "spearman",
            "clearance_microsome_az": "spearman",
        },
    },
}
SEEDS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument(
        "--target",
        action="append",
        help="Target endpoint to evaluate; repeat for multiple targets. Defaults to every family task.",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--selection-stat", choices=("valid_mean", "valid_adjusted"), default="valid_mean")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def label_col(frame: pd.DataFrame) -> str:
    ignored = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    for column in frame.columns:
        if column.lower() not in ignored:
            return column
    raise KeyError("label column not found")


def smiles_col(frame: pd.DataFrame) -> str:
    lookup = {column.lower(): column for column in frame.columns}
    for candidate in ("smiles", "drug"):
        if candidate in lookup:
            return lookup[candidate]
    raise KeyError("SMILES column not found")


def read_feature_frame(path: Path, include_labels: bool) -> pd.DataFrame:
    """Read only label-free test inputs until candidate selection is frozen."""
    if include_labels:
        return pd.read_csv(path)
    header = pd.read_csv(path, nrows=0)
    return pd.read_csv(path, usecols=[smiles_col(header)])


def split_concat(array: np.ndarray, n_train: int, n_valid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return array[:n_train], array[n_train : n_train + n_valid], array[n_train + n_valid :]


def add_task_onehot(array: np.ndarray, task_index: int, n_tasks: int) -> np.ndarray:
    onehot = np.zeros((len(array), n_tasks), dtype=np.float32)
    onehot[:, task_index] = 1.0
    return np.concatenate([array.astype(np.float32), onehot], axis=1)


def hstack(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_dense(
    data_root: Path,
    task: str,
    frames: dict[str, pd.DataFrame],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    task_dir = data_root / task
    sizes = tuple(len(frames[split]) for split in ("train", "valid", "test"))
    dense: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("chemberta", "kpgt", "unimol"):
        path = task_dir / "embeddings" / f"{name}.npy"
        if path.exists():
            array = np.load(path).astype(np.float32)
            if len(array) != sum(sizes):
                raise ValueError(f"embedding row mismatch: {path}")
            dense[name] = split_concat(array, sizes[0], sizes[1])
    ept_paths = tuple(task_dir / "embeddings_ept" / f"{split}_ept.npy" for split in ("train", "valid", "test"))
    if all(path.exists() for path in ept_paths):
        arrays = tuple(np.load(path).astype(np.float32) for path in ept_paths)
        if tuple(len(array) for array in arrays) != sizes:
            raise ValueError(f"EPT embedding row mismatch for {task}")
        dense["ept"] = arrays  # type: ignore[assignment]
    return dense


def load_inputs(
    repo: Path,
    data_root: Path,
    tasks: list[str],
    kind: str,
) -> tuple[
    dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, list[str]]],
    dict[str, tuple[float, float]],
]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import descriptor_sidecar_official_v1 as sidecar  # noqa: PLC0415

    features: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    identities: dict[str, dict[str, list[str]]] = {}
    scales: dict[str, tuple[float, float]] = {}
    for task_index, task in enumerate(tasks):
        frames = {
            split: read_feature_frame(
                data_root / task / f"{split}.csv", include_labels=split != "test"
            )
            for split in ("train", "valid", "test")
        }
        label = label_col(frames["train"])
        smiles = {
            split: frames[split][smiles_col(frames[split])].tolist()
            for split in ("train", "valid", "test")
        }
        identities[task] = {
            split: canonical_connectivity_keys(values) for split, values in smiles.items()
        }
        all_smiles = pd.Series(smiles["train"] + smiles["valid"] + smiles["test"])
        # Morgan/descriptor generation is fixed and label-free; concatenating rows
        # here preserves the precomputed split order without fitting on holdouts.
        fp = split_concat(
            sidecar.get_fingerprints(all_smiles),
            len(frames["train"]),
            len(frames["valid"]),
        )
        dense = load_dense(data_root, task, frames)
        task_features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "fp": tuple(add_task_onehot(part, task_index, len(tasks)) for part in fp),  # type: ignore[assignment]
        }
        if "kpgt" in dense:
            task_features["fp_kpgt"] = tuple(
                add_task_onehot(hstack([fp[index], dense["kpgt"][index]]), task_index, len(tasks))
                for index in range(3)
            )  # type: ignore[assignment]
        deep_names = [name for name in ("chemberta", "kpgt", "ept") if name in dense]
        if deep_names:
            task_features["fp_chemberta_kpgt_ept"] = tuple(
                add_task_onehot(
                    hstack([fp[index], *[dense[name][index] for name in deep_names]]),
                    task_index,
                    len(tasks),
                )
                for index in range(3)
            )  # type: ignore[assignment]
        features[task] = task_features

        task_labels = {
            split: frames[split][label].to_numpy(dtype=np.float64)
            for split in ("train", "valid")
        }
        if kind == "regression":
            mean = float(np.mean(task_labels["train"]))
            std = float(np.std(task_labels["train"]) + 1e-8)
            scales[task] = (mean, std)
            labels[task] = {split: (values - mean) / std for split, values in task_labels.items()}
        else:
            scales[task] = (0.0, 1.0)
            labels[task] = task_labels
    return features, labels, identities, scales


def load_test_labels(
    data_root: Path,
    target: str,
    kind: str,
    scale: tuple[float, float],
) -> np.ndarray:
    """Read official test labels only after candidate selection is frozen."""
    frame = pd.read_csv(data_root / target / "test.csv")
    values = frame[label_col(frame)].to_numpy(dtype=np.float64)
    if kind == "regression":
        mean, std = scale
        return (values - mean) / std
    return values


def common_feature_sets(features: dict[str, dict[str, object]]) -> list[str]:
    names = set.intersection(*(set(task_features) for task_features in features.values()))
    return [name for name in ("fp", "fp_kpgt", "fp_chemberta_kpgt_ept") if name in names]


def pooled_stage(
    features: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    labels: dict[str, dict[str, np.ndarray]],
    identities: dict[str, dict[str, list[str]]],
    tasks: list[str],
    target: str,
    feature_set: str,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    split_index = {"train": 0, "valid": 1}
    included_splits = ("train",) if stage == "selection" else ("train", "valid")
    blocked = forbidden_keys(identities[target]["valid"], identities[target]["test"], stage)
    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for source in tasks:
        for split in included_splits:
            mask = np.asarray(keep_mask(identities[source][split], blocked), dtype=bool)
            pooled_x.append(features[source][feature_set][split_index[split]][mask])
            pooled_y.append(labels[source][split][mask])
            audit.append(
                {
                    "target_task": target,
                    "stage": stage,
                    "source_task": source,
                    "source_split": split,
                    "rows_before": int(len(mask)),
                    "rows_removed_for_holdout_overlap": int(np.sum(~mask)),
                    "rows_after": int(np.sum(mask)),
                }
            )
    return np.concatenate(pooled_x), np.concatenate(pooled_y), audit


def classification_grid(y: np.ndarray) -> list[dict[str, float | int]]:
    positive = max(float(np.sum(y == 1)), 1.0)
    negative = max(float(np.sum(y == 0)), 1.0)
    ratio = negative / positive
    return [
        {"max_depth": 2, "learning_rate": 0.025, "min_child_weight": 1, "scale_pos_weight": ratio},
        {"max_depth": 2, "learning_rate": 0.040, "min_child_weight": 2, "scale_pos_weight": ratio},
        {"max_depth": 3, "learning_rate": 0.025, "min_child_weight": 1, "scale_pos_weight": ratio},
        {"max_depth": 3, "learning_rate": 0.040, "min_child_weight": 3, "scale_pos_weight": ratio * 1.25},
    ]


def regression_grid() -> list[dict[str, float | int]]:
    return [
        {"max_depth": 2, "learning_rate": 0.020, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.75},
        {"max_depth": 2, "learning_rate": 0.035, "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.70},
        {"max_depth": 3, "learning_rate": 0.020, "min_child_weight": 1, "subsample": 0.80, "colsample_bytree": 0.80},
        {"max_depth": 3, "learning_rate": 0.035, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.75},
    ]


def make_model(
    kind: str,
    metric: str,
    seed: int,
    config: dict[str, float | int],
    n_jobs: int,
    n_estimators: int = 1400,
    early_stopping: bool = True,
) -> XGBClassifier | XGBRegressor:
    common: dict[str, object] = {
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "random_state": seed,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "n_jobs": n_jobs,
        **config,
    }
    if early_stopping:
        common["early_stopping_rounds"] = 80
    if kind == "classification":
        common.setdefault("subsample", 0.85)
        common.setdefault("colsample_bytree", 0.75)
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc" if metric == "auroc" else "aucpr",
            **common,
        )
    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)


def metric_score(metric: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "auroc":
        return float(roc_auc_score(y_true, prediction))
    if metric == "auprc":
        return float(average_precision_score(y_true, prediction))
    if metric == "spearman":
        true_rank = pd.Series(y_true).rank(method="average").to_numpy()
        pred_rank = pd.Series(prediction).rank(method="average").to_numpy()
        return float(np.corrcoef(true_rank, pred_rank)[0, 1])
    raise ValueError(metric)


def predict(model: XGBClassifier | XGBRegressor, kind: str, array: np.ndarray) -> np.ndarray:
    if kind == "classification":
        return model.predict_proba(array)[:, 1]  # type: ignore[union-attr]
    return model.predict(array)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def git_status(repo: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "status-unavailable"


def main() -> None:
    args = parse_args()
    spec = FAMILIES[args.family]
    tasks: list[str] = spec["tasks"]
    targets = args.target or tasks
    invalid_targets = sorted(set(targets) - set(tasks))
    if invalid_targets:
        raise ValueError(
            f"targets do not belong to {args.family}: {', '.join(invalid_targets)}"
        )
    if len(targets) != len(set(targets)):
        raise ValueError("target endpoints must not be repeated")
    kind: str = spec["kind"]
    metrics: dict[str, str] = spec["metrics"]
    data_root = args.data_root or args.repo / "data" / "data_benchmark_official_v1"
    out_root = args.out_root or args.repo / "results_strict" / f"{args.family}_pooled_family_xgb_leakage_safe_v2"
    initial_git_status = git_status(args.repo)
    if initial_git_status and not args.allow_dirty:
        raise RuntimeError(
            "formal runs require a clean Git checkout; commit the protocol or pass --allow-dirty for debugging only"
        )
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to mix outputs in non-empty directory: {out_root}; pass --force to overwrite")
    out_root.mkdir(parents=True, exist_ok=True)

    features, labels, identities, scales = load_inputs(args.repo, data_root, tasks, kind)
    feature_sets = common_feature_sets(features)
    if not feature_sets:
        raise RuntimeError("no feature set is available for every task in the family")

    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    removal_audit: list[dict[str, object]] = []
    best_iterations: dict[tuple[str, str, int, int], int] = {}

    for target in targets:
        metric = metrics[target]
        valid_index = 1
        valid_x_by_feature = {name: features[target][name][valid_index] for name in feature_sets}
        valid_y_model = labels[target]["valid"]
        mean, std = scales[target]
        valid_y_score = valid_y_model * std + mean
        for feature_set in feature_sets:
            pool_x, pool_y, audit = pooled_stage(
                features, labels, identities, tasks, target, feature_set, "selection"
            )
            if feature_set == feature_sets[0]:
                removal_audit.extend(audit)
            grid = classification_grid(pool_y) if kind == "classification" else regression_grid()
            for config_index, config in enumerate(grid):
                scores: list[float] = []
                for seed in SEEDS:
                    model = make_model(kind, metric, seed, config, args.n_jobs, early_stopping=True)
                    model.fit(pool_x, pool_y, eval_set=[(valid_x_by_feature[feature_set], valid_y_model)], verbose=False)
                    prediction = predict(model, kind, valid_x_by_feature[feature_set]) * std + mean
                    score = metric_score(metric, valid_y_score, prediction)
                    best_iteration = int(model.best_iteration) + 1
                    best_iterations[(target, feature_set, config_index, seed)] = best_iteration
                    scores.append(score)
                    seed_rows.append(
                        {
                            "target_task": target,
                            "metric": metric,
                            "feature_set": feature_set,
                            "config_index": config_index,
                            "seed": seed,
                            "valid_score": score,
                            "selected_n_estimators": best_iteration,
                            **config,
                        }
                    )
                values = np.asarray(scores, dtype=float)
                summary_rows.append(
                    {
                        "target_task": target,
                        "metric": metric,
                        "feature_set": feature_set,
                        "config_index": config_index,
                        "valid_mean": float(np.mean(values)),
                        "valid_std": float(np.std(values, ddof=1)),
                        "valid_adjusted": float(np.mean(values) - np.std(values, ddof=1)),
                        **config,
                    }
                )

        candidates = [row for row in summary_rows if row["target_task"] == target]
        selected = select_validation_candidate(candidates, args.selection_stat)
        selected_rows.append({"selection_stat": args.selection_stat, **selected})

    write_csv(out_root / "candidate_validation_seed_scores.csv", seed_rows)
    write_csv(out_root / "candidate_validation_summary.csv", summary_rows)
    write_csv(out_root / "selected_candidates.csv", selected_rows)
    write_csv(out_root / "holdout_overlap_filter_audit.csv", removal_audit)

    # Test labels are deliberately unlocked only after the selected-candidate
    # manifest has been materialized.
    final_seed_rows: list[dict[str, object]] = []
    final_summary_rows: list[dict[str, object]] = []
    for selected in selected_rows:
        target = str(selected["target_task"])
        metric = str(selected["metric"])
        feature_set = str(selected["feature_set"])
        config_index = int(selected["config_index"])
        config = {
            key: selected[key]
            for key in ("max_depth", "learning_rate", "min_child_weight", "scale_pos_weight", "subsample", "colsample_bytree")
            if key in selected and selected[key] not in (None, "")
        }
        pool_x, pool_y, audit = pooled_stage(
            features, labels, identities, tasks, target, feature_set, "final"
        )
        removal_audit.extend(audit)
        test_x = features[target][feature_set][2]
        test_y_model = load_test_labels(data_root, target, kind, scales[target])
        mean, std = scales[target]
        test_y = test_y_model * std + mean
        predictions: list[np.ndarray] = []
        for seed in SEEDS:
            n_estimators = best_iterations[(target, feature_set, config_index, seed)]
            model = make_model(
                kind,
                metric,
                seed,
                config,
                args.n_jobs,
                n_estimators=n_estimators,
                early_stopping=False,
            )
            model.fit(pool_x, pool_y, verbose=False)
            prediction = predict(model, kind, test_x) * std + mean
            predictions.append(prediction)
            score = metric_score(metric, test_y, prediction)
            final_seed_rows.append(
                {
                    "target_task": target,
                    "metric": metric,
                    "feature_set": feature_set,
                    "config_index": config_index,
                    "seed": seed,
                    "n_estimators": n_estimators,
                    "test_score": score,
                }
            )
            pd.DataFrame(
                {"sample_idx": np.arange(len(test_y)), "y_true": test_y, "prediction": prediction}
            ).to_csv(out_root / f"{target}__selected__seed_{seed}__test_predictions.csv", index=False)
        scores = np.asarray(
            [float(row["test_score"]) for row in final_seed_rows if row["target_task"] == target]
        )
        ensemble = np.mean(predictions, axis=0)
        final_summary_rows.append(
            {
                "target_task": target,
                "metric": metric,
                "feature_set": feature_set,
                "config_index": config_index,
                "test_mean": float(np.mean(scores)),
                "test_std": float(np.std(scores, ddof=1)),
                "test_ensemble": metric_score(metric, test_y, ensemble),
            }
        )

    write_csv(out_root / "final_selected_test_seed_scores.csv", final_seed_rows)
    write_csv(out_root / "final_selected_test_summary.csv", final_summary_rows)
    write_csv(out_root / "holdout_overlap_filter_audit.csv", removal_audit)

    input_files = [data_root / task / f"{split}.csv" for task in tasks for split in ("train", "valid", "test")]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "targets": targets,
        "selection_rule": args.selection_stat,
        "test_policy": "one frozen candidate per target; no test-based candidate ranking",
        "molecule_identity": "RDKit connectivity-level InChIKey first block",
        "git_commit": git_commit(args.repo),
        "git_status_at_start": initial_git_status or "clean",
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "rdkit")},
        "inputs": {str(path): file_sha256(path) for path in input_files},
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"completed leakage-safe {args.family} run: {out_root}")


if __name__ == "__main__":
    main()
