from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


MODALITIES = ("chemberta", "kpgt", "ept")
CLASSIFICATION_METRICS = {"AUROC", "AUPRC"}
ALLOWED_LABEL_SPLITS = {
    "select": {"train", "valid"},
    "final": {"train", "valid"},
    "score": {"test"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "final", "score"), required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--task-metadata", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["task"]: row for row in csv.DictReader(handle)}


def label_column(frame: pd.DataFrame) -> str:
    return "Y" if "Y" in frame.columns else "label"


def load_labels(task_root: Path, split: str) -> np.ndarray:
    frame = pd.read_csv(task_root / f"{split}.csv")
    return frame[label_column(frame)].to_numpy(dtype=np.float64)


def load_labels_for_phase(task_root: Path, split: str, phase: str) -> np.ndarray:
    if split not in ALLOWED_LABEL_SPLITS[phase]:
        raise ValueError(f"{phase} phase may not read {split} labels")
    return load_labels(task_root, split)


def split_counts(task_root: Path) -> dict[str, int]:
    return {
        split: len(pd.read_csv(task_root / f"{split}.csv", usecols=["smiles"]))
        for split in ("train", "valid", "test")
    }


def load_modality(task_root: Path, modality: str, split: str) -> np.ndarray:
    counts = split_counts(task_root)
    if modality == "ept":
        return np.asarray(
            np.load(task_root / "embeddings_ept" / f"{split}_ept.npy", mmap_mode="r"),
            dtype=np.float32,
        )
    full = np.load(task_root / "embeddings" / f"{modality}.npy", mmap_mode="r")
    starts = {
        "train": 0,
        "valid": counts["train"],
        "test": counts["train"] + counts["valid"],
    }
    start = starts[split]
    return np.asarray(full[start : start + counts[split]], dtype=np.float32)


def load_features(task_root: Path, split: str) -> np.ndarray:
    arrays = [load_modality(task_root, name, split) for name in MODALITIES]
    if len({len(array) for array in arrays}) != 1:
        raise ValueError(f"feature row mismatch for {task_root.name}/{split}")
    features = np.concatenate(arrays, axis=1)
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite features for {task_root.name}/{split}")
    return features


def predict_values(automl: AutoML, features: np.ndarray, classification: bool) -> np.ndarray:
    if classification:
        values = np.asarray(automl.predict_proba(features))[:, 1]
    else:
        values = np.asarray(automl.predict(features))
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("AutoML produced non-finite predictions")
    return values


def average_precision_metric(
    X_test, y_test, estimator, labels, X_train, y_train, *args, **kwargs
):
    prediction = np.asarray(estimator.predict_proba(X_test))[:, 1]
    value = float(average_precision_score(y_test, prediction))
    return 1.0 - value, {"average_precision": value}


def spearman_metric(
    X_test, y_test, estimator, labels, X_train, y_train, *args, **kwargs
):
    prediction = np.asarray(estimator.predict(X_test))
    value = float(spearmanr(y_test, prediction).correlation)
    if not np.isfinite(value):
        value = -1.0
    return 1.0 - value, {"spearman": value}


def flaml_metric(metric: str):
    if metric == "AUROC":
        return "roc_auc"
    if metric == "AUPRC":
        return average_precision_metric
    if metric == "MAE":
        return "mae"
    if metric == "Spearman":
        return spearman_metric
    raise ValueError(metric)


def official_score(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    if metric == "Spearman":
        return float(spearmanr(labels, predictions).correlation)
    raise ValueError(metric)


def write_prediction(path: Path, prediction: np.ndarray) -> None:
    if not np.isfinite(prediction).all():
        raise ValueError(f"refusing to write non-finite prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_idx": np.arange(len(prediction)), "prediction": prediction}
    ).to_csv(path, index=False)


def read_prediction(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if not np.array_equal(frame.sample_idx.to_numpy(), np.arange(len(frame))):
        raise ValueError(f"sample index mismatch: {path}")
    prediction = frame.prediction.to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"non-finite prediction: {path}")
    return prediction


def automl_settings(args: argparse.Namespace, metric: str, seed: int, log_path: Path) -> dict:
    classification = metric in CLASSIFICATION_METRICS
    return {
        "task": "classification" if classification else "regression",
        "metric": flaml_metric(metric),
        "estimator_list": ["xgboost", "rf", "extra_tree"] + (["lrl1"] if classification else []),
        "time_budget": args.time_budget,
        "max_iter": args.max_iter,
        "n_jobs": args.n_jobs,
        "seed": seed,
        "eval_method": "holdout",
        "retrain_full": False,
        "log_file_name": str(log_path),
        "log_type": "all",
        "verbose": 0,
    }


def select_phase(args: argparse.Namespace, metadata: dict[str, dict[str, str]], tasks: list[str]) -> None:
    root = args.out_root / "selection"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    rows: list[dict[str, object]] = []
    for task in tasks:
        print("[select]", task, flush=True)
        task_root = args.data_root / task
        train_x, valid_x = load_features(task_root, "train"), load_features(task_root, "valid")
        train_y = load_labels_for_phase(task_root, "train", "select")
        valid_y = load_labels_for_phase(task_root, "valid", "select")
        metric = metadata[task]["metric"]
        classification = metric in CLASSIFICATION_METRICS
        for seed in args.seeds:
            log_path = root / "logs" / task / f"seed_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            automl = AutoML()
            automl.fit(
                X_train=train_x,
                y_train=train_y.astype(int) if classification else train_y,
                X_val=valid_x,
                y_val=valid_y.astype(int) if classification else valid_y,
                **automl_settings(args, metric, seed, log_path),
            )
            prediction = predict_values(automl, valid_x, classification)
            prediction_path = root / "predictions" / task / f"seed_{seed}.csv"
            write_prediction(prediction_path, prediction)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "seed": seed,
                    "valid_score": official_score(metric, valid_y, prediction),
                    "best_estimator": automl.best_estimator,
                    "best_config_json": json.dumps(automl.best_config, sort_keys=True),
                    "best_iteration": automl.best_iteration,
                    "log_file": str(log_path),
                    "prediction_file": str(prediction_path),
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "selected_models.csv", index=False)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_at_start": args.git_commit_at_start,
                "policy": "train fit and official-validation selection; no test labels read",
                "modalities": list(MODALITIES),
                "time_budget_seconds_per_task_seed": args.time_budget,
                "max_iterations_per_task_seed": args.max_iter,
                "seeds": args.seeds,
            },
            indent=2,
        )
        + "\n"
    )


def final_phase(args: argparse.Namespace, metadata: dict[str, dict[str, str]], tasks: list[str]) -> None:
    root = args.out_root / "final"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    selected = pd.read_csv(args.out_root / "selection" / "selected_models.csv")
    expected = {(task, seed) for task in tasks for seed in args.seeds}
    observed = set(zip(selected.task, selected.seed))
    if observed != expected:
        raise ValueError("selection manifest does not cover the requested task/seed grid")
    rows: list[dict[str, object]] = []
    for row in selected.to_dict("records"):
        task, seed = str(row["task"]), int(row["seed"])
        print("[final]", task, seed, flush=True)
        task_root = args.data_root / task
        train_x, valid_x = load_features(task_root, "train"), load_features(task_root, "valid")
        dev_x = np.concatenate([train_x, valid_x], axis=0)
        dev_y = np.concatenate(
            [
                load_labels_for_phase(task_root, "train", "final"),
                load_labels_for_phase(task_root, "valid", "final"),
            ]
        )
        test_x = load_features(task_root, "test")
        metric = str(row["metric"])
        classification = metric in CLASSIFICATION_METRICS
        automl = AutoML()
        automl.retrain_from_log(
            str(row["log_file"]),
            X_train=dev_x,
            y_train=dev_y.astype(int) if classification else dev_y,
            task="classification" if classification else "regression",
            train_best=True,
            train_full=True,
            record_id=int(row["best_iteration"]),
            n_jobs=args.n_jobs,
        )
        prediction = predict_values(automl, test_x, classification)
        prediction_path = root / "predictions" / task / f"seed_{seed}.csv"
        write_prediction(prediction_path, prediction)
        rows.append({**row, "prediction_file": str(prediction_path)})
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "prediction_manifest.csv", index=False)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_at_start": args.git_commit_at_start,
                "policy": "frozen FLAML logs retrained on train+validation; test labels not read",
            },
            indent=2,
        )
        + "\n"
    )


def score_phase(args: argparse.Namespace) -> None:
    root = args.out_root / "score"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    manifest = pd.read_csv(args.out_root / "final" / "prediction_manifest.csv")
    rows = []
    for row in manifest.to_dict("records"):
        labels = load_labels_for_phase(
            args.data_root / str(row["task"]), "test", "score"
        )
        prediction = read_prediction(Path(str(row["prediction_file"])))
        rows.append({**row, "test_score": official_score(str(row["metric"]), labels, prediction)})
    scores = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(root / "test_seed_scores.csv", index=False)
    scores.groupby(["task", "metric"], as_index=False).agg(
        test_mean=("test_score", "mean"), test_std=("test_score", "std")
    ).to_csv(root / "test_summary.csv", index=False)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_at_start": args.git_commit_at_start,
                "policy": "official test labels first read in score phase",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.git_commit_at_start = git_commit()
    metadata = read_metadata(args.task_metadata)
    tasks = args.tasks or sorted(metadata)
    unknown = sorted(set(tasks) - set(metadata))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "select":
        select_phase(args, metadata, tasks)
    elif args.phase == "final":
        final_phase(args, metadata, tasks)
    else:
        score_phase(args)


if __name__ == "__main__":
    main()
