from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODALITIES = ("chemberta", "kpgt", "ept")
CLASSIFICATION_METRICS = {"AUROC", "AUPRC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "final", "score"), required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--task-metadata", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
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


def split_counts(task_root: Path) -> dict[str, int]:
    return {
        split: len(pd.read_csv(task_root / f"{split}.csv", usecols=["smiles"]))
        for split in ("train", "valid", "test")
    }


def load_features(task_root: Path, modality: str, split: str) -> np.ndarray:
    counts = split_counts(task_root)
    if modality == "ept":
        return np.load(task_root / "embeddings_ept" / f"{split}_ept.npy")
    full = np.load(task_root / "embeddings" / f"{modality}.npy", mmap_mode="r")
    starts = {"train": 0, "valid": counts["train"], "test": counts["train"] + counts["valid"]}
    start = starts[split]
    return np.asarray(full[start : start + counts[split]], dtype=np.float32)


def load_development_features(task_root: Path, modality: str) -> np.ndarray:
    return np.concatenate(
        [load_features(task_root, modality, "train"), load_features(task_root, modality, "valid")],
        axis=0,
    )


def make_base_model(classification: bool, seed: int, max_iter: int):
    if classification:
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=max_iter,
            tol=1e-4,
            class_weight="balanced",
            average=True,
            random_state=seed,
        )
    else:
        # A deterministic LSQR solve is stable for the high-dimensional frozen
        # embedding matrices. SGDRegressor produced unbounded predictions on
        # several ADMET regression tasks and is unsuitable as a fair control.
        estimator = Ridge(alpha=1.0, solver="lsqr")
    return make_pipeline(StandardScaler(), estimator)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    classification: bool,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    model = make_base_model(classification, seed, max_iter)
    if classification:
        model.fit(train_x, train_y.astype(int))
        return model.predict_proba(predict_x)[:, 1].astype(np.float64)
    mean = float(np.mean(train_y))
    scale = float(np.std(train_y)) or 1.0
    model.fit(train_x, (train_y - mean) / scale)
    return (model.predict(predict_x) * scale + mean).astype(np.float64)


def score(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "Spearman":
        value = spearmanr(labels, predictions).correlation
        return float(value) if value is not None else float("nan")
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    raise ValueError(metric)


def utility(metric: str, value: float) -> float:
    return -value if metric == "MAE" else value


def write_prediction(path: Path, predictions: np.ndarray) -> None:
    if not np.isfinite(predictions).all():
        raise ValueError(f"non-finite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_idx": np.arange(len(predictions)), "prediction": predictions}
    ).to_csv(path, index=False)


def read_prediction(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if not np.array_equal(frame["sample_idx"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"sample index mismatch: {path}")
    return frame["prediction"].to_numpy(dtype=np.float64)


def select_phase(args: argparse.Namespace, metadata: dict[str, dict[str, str]], tasks: list[str]) -> None:
    phase_root = args.out_root / "selection"
    if phase_root.exists():
        raise FileExistsError(f"refusing to overwrite {phase_root}")
    rows: list[dict[str, object]] = []
    for task in tasks:
        print("[select]", task, flush=True)
        task_root = args.data_root / task
        train_y = load_labels(task_root, "train")
        valid_y = load_labels(task_root, "valid")
        metric = metadata[task]["metric"]
        classification = metric in CLASSIFICATION_METRICS
        for modality in MODALITIES:
            train_x = load_features(task_root, modality, "train")
            valid_x = load_features(task_root, modality, "valid")
            for seed in args.seeds:
                prediction = fit_predict(
                    train_x, train_y, valid_x, classification, seed, args.max_iter
                )
                path = phase_root / "predictions" / task / f"{modality}_seed_{seed}.csv"
                write_prediction(path, prediction)
                value = score(metric, valid_y, read_prediction(path))
                rows.append(
                    {"task": task, "metric": metric, "modality": modality, "seed": seed, "valid_score": value, "prediction_file": str(path)}
                )
    scores = pd.DataFrame(rows)
    phase_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(phase_root / "candidate_valid_scores.csv", index=False)
    means = scores.groupby(["task", "metric", "modality"], as_index=False).valid_score.mean()
    task_choices: dict[str, str] = {}
    rank_rows: list[dict[str, object]] = []
    for task, group in means.groupby("task"):
        metric = str(group.metric.iloc[0])
        ordered = group.assign(utility=group.valid_score.map(lambda value: utility(metric, value))).sort_values("utility", ascending=False)
        task_choices[str(task)] = str(ordered.modality.iloc[0])
        for rank, row in enumerate(ordered.to_dict("records"), start=1):
            rank_rows.append({"task": task, "modality": row["modality"], "rank": rank, "valid_score": row["valid_score"]})
    ranks = pd.DataFrame(rank_rows)
    global_choice = str(
        ranks.groupby("modality")["rank"].mean().sort_values().index[0]
    )
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": args.git_commit_at_start,
        "policy": "train-only base fitting; five-seed validation means; no test labels read",
        "modalities": list(MODALITIES),
        "seeds": args.seeds,
        "tasks": tasks,
        "per_task_single": task_choices,
        "global_single": global_choice,
    }
    (phase_root / "selected_controls.json").write_text(json.dumps(selection, indent=2) + "\n")
    ranks.to_csv(phase_root / "candidate_valid_ranks.csv", index=False)


def oof_and_test(
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    classification: bool,
    seed: int,
    folds: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if classification
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    oof = np.empty(len(dev_y), dtype=np.float64)
    for fold, (train_index, heldout_index) in enumerate(splitter.split(dev_x, dev_y if classification else None)):
        oof[heldout_index] = fit_predict(
            dev_x[train_index], dev_y[train_index], dev_x[heldout_index], classification, seed + fold, max_iter
        )
    test_prediction = fit_predict(dev_x, dev_y, test_x, classification, seed, max_iter)
    return oof, test_prediction


def final_phase(args: argparse.Namespace, metadata: dict[str, dict[str, str]], tasks: list[str]) -> None:
    phase_root = args.out_root / "final"
    if phase_root.exists():
        raise FileExistsError(f"refusing to overwrite {phase_root}")
    selection = json.loads((args.out_root / "selection" / "selected_controls.json").read_text())
    if sorted(selection["tasks"]) != sorted(tasks):
        raise ValueError("selection task set does not match final task set")
    manifest_rows: list[dict[str, object]] = []
    for task in tasks:
        print("[final]", task, flush=True)
        task_root = args.data_root / task
        dev_y = np.concatenate([load_labels(task_root, "train"), load_labels(task_root, "valid")])
        metric = metadata[task]["metric"]
        classification = metric in CLASSIFICATION_METRICS
        for seed in args.seeds:
            base_test: dict[str, np.ndarray] = {}
            oof_columns: list[np.ndarray] = []
            for modality in MODALITIES:
                dev_x = load_development_features(task_root, modality)
                test_x = load_features(task_root, modality, "test")
                oof, test_prediction = oof_and_test(
                    dev_x, dev_y, test_x, classification, seed, args.folds, args.max_iter
                )
                oof_columns.append(oof)
                base_test[modality] = test_prediction
            stacked_oof = np.column_stack(oof_columns)
            stacked_test = np.column_stack([base_test[name] for name in MODALITIES])
            if classification:
                meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
                meta_model.fit(stacked_oof, dev_y.astype(int))
                stacking_prediction = meta_model.predict_proba(stacked_test)[:, 1]
            else:
                meta_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr"))
                meta_model.fit(stacked_oof, dev_y)
                stacking_prediction = meta_model.predict(stacked_test)
            predictions = {
                "global_single": base_test[selection["global_single"]],
                "per_task_single": base_test[selection["per_task_single"][task]],
                "uniform_average": np.mean(list(base_test.values()), axis=0),
                "oof_stacking": stacking_prediction,
            }
            for control, prediction in predictions.items():
                path = phase_root / "predictions" / task / f"{control}_seed_{seed}.csv"
                write_prediction(path, prediction)
                manifest_rows.append({"task": task, "metric": metric, "control": control, "seed": seed, "prediction_file": str(path)})
    phase_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(phase_root / "prediction_manifest.csv", index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": args.git_commit_at_start,
        "policy": "selection frozen before final train+valid refit; test labels not read",
        "selection_file": str(args.out_root / "selection" / "selected_controls.json"),
        "folds": args.folds,
        "seeds": args.seeds,
    }
    (phase_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def score_phase(args: argparse.Namespace, metadata: dict[str, dict[str, str]], tasks: list[str]) -> None:
    phase_root = args.out_root / "score"
    if phase_root.exists():
        raise FileExistsError(f"refusing to overwrite {phase_root}")
    manifest = pd.read_csv(args.out_root / "final" / "prediction_manifest.csv")
    rows: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        task = str(row["task"])
        labels = load_labels(args.data_root / task, "test")
        predictions = read_prediction(Path(row["prediction_file"]))
        rows.append({**row, "test_score": score(str(row["metric"]), labels, predictions)})
    scores = pd.DataFrame(rows)
    phase_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(phase_root / "test_seed_scores.csv", index=False)
    summary = scores.groupby(["task", "metric", "control"], as_index=False).agg(
        test_mean=("test_score", "mean"),
        test_std=("test_score", "std"),
    )
    summary.to_csv(phase_root / "test_summary.csv", index=False)
    (phase_root / "provenance.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit_at_start": args.git_commit_at_start, "policy": "test labels first read in score phase"}, indent=2) + "\n"
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
        score_phase(args, metadata, tasks)


if __name__ == "__main__":
    main()
