from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = (101, 202, 303, 404, 505)
CLASSIFICATION_METRICS = {"AUROC", "AUPRC"}
ALLOWED_SPLITS = {"select": {"valid"}, "score": {"valid", "test"}}
STABILITY_RATIO_THRESHOLD = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "score"), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-split-root", type=Path)
    parser.add_argument("--evaluation-data-root", type=Path)
    parser.add_argument(
        "--candidate-family-policy",
        choices=("all_complete", "stable_regularized"),
        default="all_complete",
        help=(
            "stable_regularized excludes unregularized deep linear heads after "
            "validation-only numerical-stability screening"
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def candidate_file(
    root: Path, family: str, task: str, seed: int, split: str
) -> Path:
    return root / family / task / f"seed_{seed}" / f"{split}_predictions.csv"


def discover_complete_pool(
    root: Path, tasks: list[str], seeds: list[int]
) -> list[str]:
    """Return families with both files present for every requested task and seed.

    Test-file existence is provenance metadata. The selection phase never opens a
    test file or reads its labels or predictions.
    """

    families: list[str] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        complete = all(
            candidate_file(root, directory.name, task, seed, split).is_file()
            for task in tasks
            for seed in seeds
            for split in ("valid", "test")
        )
        if complete:
            families.append(directory.name)
    if not families:
        raise RuntimeError("no candidate family is complete for every task and seed")
    return families


def apply_candidate_family_policy(
    families: list[str], policy: str, stability_audit: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """Apply a predeclared, test-independent numerical-stability policy."""

    audit = stability_audit.copy()
    if set(audit["candidate_family"]) != set(families):
        raise ValueError("stability audit does not match the complete candidate pool")
    if policy == "all_complete":
        audit["eligible"] = True
    else:
        audit["eligible"] = audit["max_regression_mae_to_median_baseline_ratio"].le(
            STABILITY_RATIO_THRESHOLD
        )
    audit["policy"] = policy
    audit["stability_ratio_threshold"] = STABILITY_RATIO_THRESHOLD
    audit["reason"] = np.where(
        audit["eligible"],
        "eligible complete family",
        "excluded because validation-only regression MAE exceeded 100x the median-prediction baseline",
    )
    selected = audit.loc[audit["eligible"], "candidate_family"].astype(str).tolist()
    if not selected:
        raise RuntimeError(f"candidate-family policy removed every family: {policy}")
    return selected, audit


def build_validation_stability_audit(
    root: Path, families: list[str], tasks: list[str], seeds: list[int]
) -> pd.DataFrame:
    """Quantify numerical stability using validation labels and predictions only."""

    rows: list[dict[str, object]] = []
    for family in families:
        max_abs_prediction = 0.0
        ratios: list[float] = []
        regression_files = 0
        files_checked = 0
        for task in tasks:
            metric = metric_from_task_file(root, family, task, seeds[0])
            for seed in seeds:
                path = candidate_file(root, family, task, seed, "valid")
                labels, predictions = load_prediction_for_phase(path, "valid", "select")
                files_checked += 1
                max_abs_prediction = max(
                    max_abs_prediction, float(np.max(np.abs(predictions)))
                )
                if metric == "MAE":
                    regression_files += 1
                    baseline = float(np.mean(np.abs(labels - np.median(labels))))
                    if baseline <= 1e-12:
                        raise ValueError(f"degenerate validation baseline: {path}")
                    ratios.append(mean_absolute_error(labels, predictions) / baseline)
        rows.append(
            {
                "candidate_family": family,
                "complete_for_all_tasks_seeds": True,
                "validation_files_checked": files_checked,
                "regression_validation_files_checked": regression_files,
                "max_abs_validation_prediction": max_abs_prediction,
                "max_regression_mae_to_median_baseline_ratio": max(ratios),
            }
        )
    return pd.DataFrame(rows).sort_values("candidate_family").reset_index(drop=True)


def load_prediction_for_phase(
    path: Path, split: str, phase: str
) -> tuple[np.ndarray, np.ndarray]:
    if split not in ALLOWED_SPLITS[phase]:
        raise ValueError(f"{phase} phase may not read {split} predictions")
    frame = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing {sorted(required)} in {path}")
    labels = frame["y_true"].to_numpy(dtype=np.float64)
    predictions = frame["y_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError(f"non-finite values in {path}")
    return labels, predictions


def metric_from_task_file(root: Path, family: str, task: str, seed: int) -> str:
    metadata = root / family / task / f"seed_{seed}" / "metrics.json"
    value = str(json.loads(metadata.read_text())["metric"])
    if value not in {"AUROC", "AUPRC", "MAE", "Spearman"}:
        raise ValueError(f"unsupported metric {value}: {metadata}")
    return value


def score(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    if metric == "Spearman":
        value = spearmanr(labels, predictions).correlation
        return float(value) if value is not None else float("nan")
    raise ValueError(metric)


def utility(metric: str, value: float) -> float:
    return -value if metric == "MAE" else value


def verify_labels(reference: np.ndarray | None, observed: np.ndarray, path: Path) -> np.ndarray:
    if reference is None:
        return observed
    if not np.array_equal(reference, observed):
        raise ValueError(f"label order mismatch: {path}")
    return reference


def file_set_digest(root: Path, families: list[str], tasks: list[str], seeds: list[int]) -> str:
    digest = hashlib.sha256()
    for family in families:
        for task in tasks:
            for seed in seeds:
                for split in ("valid", "test"):
                    path = candidate_file(root, family, task, seed, split)
                    stat = path.stat()
                    relative = path.relative_to(root)
                    digest.update(f"{relative}\t{stat.st_size}\n".encode())
    return digest.hexdigest()


def select_phase(args: argparse.Namespace, tasks: list[str], seeds: list[int]) -> None:
    selection_root = args.out_root / "selection"
    if selection_root.exists():
        raise FileExistsError(f"refusing to overwrite {selection_root}")
    complete_families = discover_complete_pool(args.candidate_root, tasks, seeds)
    stability_audit = build_validation_stability_audit(
        args.candidate_root, complete_families, tasks, seeds
    )
    families, eligibility = apply_candidate_family_policy(
        complete_families, args.candidate_family_policy, stability_audit
    )
    rows: list[dict[str, object]] = []
    task_metrics: dict[str, str] = {}
    for task in tasks:
        reference_labels: np.ndarray | None = None
        for family in families:
            metric = metric_from_task_file(args.candidate_root, family, task, seeds[0])
            task_metrics.setdefault(task, metric)
            if task_metrics[task] != metric:
                raise ValueError(f"metric mismatch for {task}")
            for seed in seeds:
                path = candidate_file(args.candidate_root, family, task, seed, "valid")
                labels, predictions = load_prediction_for_phase(path, "valid", "select")
                reference_labels = verify_labels(reference_labels, labels, path)
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "candidate_family": family,
                        "seed": seed,
                        "valid_score": score(metric, labels, predictions),
                        "prediction_file": str(path),
                    }
                )

    scores = pd.DataFrame(rows)
    means = (
        scores.groupby(["task", "metric", "candidate_family"], as_index=False)
        .valid_score.mean()
    )
    rank_rows: list[dict[str, object]] = []
    rankings: dict[str, list[str]] = {}
    for task, group in means.groupby("task", sort=True):
        metric = str(group.metric.iloc[0])
        ordered = group.assign(
            utility=group.valid_score.map(lambda value: utility(metric, value))
        ).sort_values(["utility", "candidate_family"], ascending=[False, True])
        rankings[str(task)] = ordered.candidate_family.astype(str).tolist()
        for rank, row in enumerate(ordered.to_dict("records"), start=1):
            rank_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "candidate_family": row["candidate_family"],
                    "valid_score": row["valid_score"],
                    "rank": rank,
                    "normalized_rank_utility": 1.0
                    - (rank - 1) / (len(families) - 1),
                }
            )
    ranks = pd.DataFrame(rank_rows)
    global_summary = (
        ranks.groupby("candidate_family", as_index=False)
        .agg(
            mean_valid_rank=("rank", "mean"),
            mean_normalized_rank_utility=("normalized_rank_utility", "mean"),
        )
        .sort_values(
            ["mean_normalized_rank_utility", "candidate_family"],
            ascending=[False, True],
        )
    )
    global_choice = str(global_summary.candidate_family.iloc[0])
    selection_root.mkdir(parents=True)
    scores.to_csv(selection_root / "candidate_valid_scores.csv", index=False)
    ranks.to_csv(selection_root / "candidate_valid_ranks.csv", index=False)
    global_summary.to_csv(selection_root / "global_candidate_ranking.csv", index=False)
    eligibility.to_csv(selection_root / "candidate_family_eligibility.csv", index=False)
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": git_commit(),
        "policy": "fixed complete candidate pool with test-independent family policy; validation-only selection; test file contents inaccessible until score phase",
        "candidate_family_policy": args.candidate_family_policy,
        "complete_candidate_pool": complete_families,
        "complete_candidate_count": len(complete_families),
        "candidate_root": str(args.candidate_root),
        "candidate_pool": families,
        "candidate_count": len(families),
        "tasks": tasks,
        "task_count": len(tasks),
        "seeds": seeds,
        "candidate_run_count": len(families) * len(tasks) * len(seeds),
        "file_set_size_digest": file_set_digest(
            args.candidate_root, families, tasks, seeds
        ),
        "global_single": global_choice,
        "per_task_ranking": rankings,
        "task_metrics": task_metrics,
    }
    (selection_root / "frozen_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )


def oriented_average(
    metric: str,
    valid_predictions: list[np.ndarray],
    test_predictions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(valid_predictions) != len(test_predictions) or not valid_predictions:
        raise ValueError("paired non-empty prediction lists are required")
    if metric != "Spearman":
        return np.mean(valid_predictions, axis=0), np.mean(test_predictions, axis=0)
    valid_scaled: list[np.ndarray] = []
    test_scaled: list[np.ndarray] = []
    for valid, test in zip(valid_predictions, test_predictions, strict=True):
        center = float(np.mean(valid))
        scale = float(np.std(valid)) or 1.0
        valid_scaled.append((valid - center) / scale)
        test_scaled.append((test - center) / scale)
    return np.mean(valid_scaled, axis=0), np.mean(test_scaled, axis=0)


def fixed_stacking_prediction(
    metric: str,
    valid_matrix: np.ndarray,
    valid_labels: np.ndarray,
    test_matrix: np.ndarray,
    seed: int,
) -> np.ndarray:
    if metric in CLASSIFICATION_METRICS:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
        )
        model.fit(valid_matrix, valid_labels.astype(int))
        return model.predict_proba(test_matrix)[:, 1]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr"))
    model.fit(valid_matrix, valid_labels)
    return model.predict(test_matrix)


def write_prediction(
    path: Path, labels: np.ndarray, predictions: np.ndarray
) -> None:
    if not np.isfinite(predictions).all():
        raise ValueError(f"non-finite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_idx": np.arange(len(predictions)),
            "y_true": labels,
            "prediction": predictions,
        }
    ).to_csv(path, index=False)


def _data_columns(frame: pd.DataFrame, path: Path) -> tuple[str, str]:
    molecule_column = next(
        (name for name in ("Drug", "smiles", "SMILES") if name in frame.columns),
        None,
    )
    label_column = next(
        (name for name in ("Y", "label", "y_true") if name in frame.columns),
        None,
    )
    if molecule_column is None or label_column is None:
        raise ValueError(f"missing molecule or label column: {path}")
    return molecule_column, label_column


def _row_keys(frame: pd.DataFrame, molecule_column: str, label_column: str) -> pd.Series:
    molecules = frame[molecule_column].astype(str)
    labels = frame[label_column].astype(np.float64).map(lambda value: f"{value:.12g}")
    base = molecules + "\t" + labels
    occurrence = base.groupby(base, sort=False).cumcount().astype(str)
    return base + "\t" + occurrence


def build_test_alignment(
    candidate_split_path: Path, evaluation_split_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Map legacy candidate rows onto the current evaluation split by row identity."""

    candidate = pd.read_csv(candidate_split_path)
    evaluation = pd.read_csv(evaluation_split_path)
    candidate_molecule, candidate_label = _data_columns(candidate, candidate_split_path)
    evaluation_molecule, evaluation_label = _data_columns(
        evaluation, evaluation_split_path
    )
    candidate_keys = _row_keys(candidate, candidate_molecule, candidate_label)
    evaluation_keys = _row_keys(evaluation, evaluation_molecule, evaluation_label)
    if candidate_keys.duplicated().any() or evaluation_keys.duplicated().any():
        raise ValueError("occurrence-qualified row keys must be unique")
    candidate_lookup = pd.Series(
        np.arange(len(candidate), dtype=np.int64), index=candidate_keys
    )
    missing = evaluation_keys[~evaluation_keys.isin(candidate_lookup.index)]
    if not missing.empty:
        raise ValueError(
            f"evaluation rows absent from candidate split: {evaluation_split_path}: "
            f"{missing.iloc[:3].tolist()}"
        )
    indices = candidate_lookup.loc[evaluation_keys].to_numpy(dtype=np.int64)
    candidate_labels = candidate[candidate_label].to_numpy(dtype=np.float64)
    evaluation_labels = evaluation[evaluation_label].to_numpy(dtype=np.float64)
    if not np.allclose(
        candidate_labels[indices], evaluation_labels, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"aligned label mismatch: {evaluation_split_path}")
    kept = set(indices.tolist())
    excluded = candidate.loc[
        [index not in kept for index in range(len(candidate))],
        [candidate_molecule, candidate_label],
    ]
    audit = {
        "candidate_split": str(candidate_split_path),
        "evaluation_split": str(evaluation_split_path),
        "candidate_rows": len(candidate),
        "evaluation_rows": len(evaluation),
        "excluded_rows": len(candidate) - len(evaluation),
        "excluded_molecules": ";".join(excluded[candidate_molecule].astype(str)),
        "excluded_labels": ";".join(excluded[candidate_label].astype(str)),
        "alignment_key": "molecule+label+within-key occurrence",
    }
    return indices, evaluation_labels, audit


def score_phase(args: argparse.Namespace, tasks: list[str], seeds: list[int]) -> None:
    score_root = args.out_root / "score"
    if score_root.exists():
        raise FileExistsError(f"refusing to overwrite {score_root}")
    selection_path = args.out_root / "selection" / "frozen_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["tasks"] != tasks or selection["seeds"] != seeds:
        raise ValueError("requested tasks or seeds differ from frozen selection")
    families = list(selection["candidate_pool"])
    current_digest = file_set_digest(args.candidate_root, families, tasks, seeds)
    if current_digest != selection["file_set_size_digest"]:
        raise ValueError("candidate file set changed after selection freeze")
    if (args.candidate_split_root is None) != (args.evaluation_data_root is None):
        raise ValueError(
            "candidate-split-root and evaluation-data-root must be provided together"
        )

    rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    for task in tasks:
        metric = str(selection["task_metrics"][task])
        ranking = list(selection["per_task_ranking"][task])
        selections = {
            "global_single": [selection["global_single"]],
            "per_task_single": ranking[:1],
            "validation_top2_average": ranking[:2],
            "validation_top3_average": ranking[:3],
            "uniform_candidate_average": families,
        }
        test_indices: np.ndarray | None = None
        evaluation_labels: np.ndarray | None = None
        if args.candidate_split_root is not None:
            test_indices, evaluation_labels, alignment_audit = build_test_alignment(
                args.candidate_split_root / task / "test.csv",
                args.evaluation_data_root / task / "test.csv",
            )
            alignment_rows.append({"task": task, **alignment_audit})
        for seed in seeds:
            valid_labels: np.ndarray | None = None
            test_labels: np.ndarray | None = None
            valid_by_family: dict[str, np.ndarray] = {}
            test_by_family: dict[str, np.ndarray] = {}
            for family in families:
                valid_path = candidate_file(
                    args.candidate_root, family, task, seed, "valid"
                )
                test_path = candidate_file(
                    args.candidate_root, family, task, seed, "test"
                )
                vy, vp = load_prediction_for_phase(valid_path, "valid", "score")
                ty, tp = load_prediction_for_phase(test_path, "test", "score")
                valid_labels = verify_labels(valid_labels, vy, valid_path)
                test_labels = verify_labels(test_labels, ty, test_path)
                valid_by_family[family] = vp
                test_by_family[family] = tp
            assert valid_labels is not None and test_labels is not None
            if test_indices is not None:
                if len(test_labels) <= int(test_indices.max(initial=-1)):
                    raise ValueError(f"candidate prediction rows missing for {task}")
                legacy_labels = pd.read_csv(
                    args.candidate_split_root / task / "test.csv"
                )
                _, legacy_label_column = _data_columns(
                    legacy_labels, args.candidate_split_root / task / "test.csv"
                )
                expected_legacy_labels = legacy_labels[legacy_label_column].to_numpy(
                    dtype=np.float64
                )
                if not np.allclose(
                    test_labels, expected_legacy_labels, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(f"candidate prediction label mismatch for {task}")
                test_labels = evaluation_labels
                test_by_family = {
                    family: prediction[test_indices]
                    for family, prediction in test_by_family.items()
                }

            predictions: dict[str, np.ndarray] = {}
            for method, selected in selections.items():
                _, predictions[method] = oriented_average(
                    metric,
                    [valid_by_family[name] for name in selected],
                    [test_by_family[name] for name in selected],
                )
            valid_matrix = np.column_stack([valid_by_family[name] for name in families])
            test_matrix = np.column_stack([test_by_family[name] for name in families])
            predictions["validation_stacking"] = fixed_stacking_prediction(
                metric, valid_matrix, valid_labels, test_matrix, seed
            )

            for method, prediction in predictions.items():
                path = score_root / "predictions" / task / f"{method}_seed_{seed}.csv"
                write_prediction(path, test_labels, prediction)
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "method": method,
                        "seed": seed,
                        "candidate_count": len(families),
                        "selected_candidates": ";".join(
                            selections.get(method, families)
                        ),
                        "test_score": score(metric, test_labels, prediction),
                        "prediction_file": str(path),
                    }
                )

    runs = pd.DataFrame(rows)
    score_root.mkdir(parents=True, exist_ok=True)
    if alignment_rows:
        pd.DataFrame(alignment_rows).to_csv(
            score_root / "test_split_alignment_audit.csv", index=False
        )
    runs.to_csv(score_root / "expanded_pool_control_runs.csv", index=False)
    runs.rename(columns={"method": "model"})[
        ["task", "metric", "model", "seed", "prediction_file"]
    ].to_csv(score_root / "prediction_manifest.csv", index=False)
    summary = (
        runs.groupby(["task", "metric", "method", "candidate_count"], as_index=False)
        .agg(
            score_mean=("test_score", "mean"),
            score_std=("test_score", "std"),
            n_seeds=("seed", "nunique"),
            selected_candidates=("selected_candidates", "first"),
        )
    )
    summary.to_csv(score_root / "expanded_pool_control_summary.csv", index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": git_commit(),
        "selection_file": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "policy": "selection frozen before any test prediction content was read; all methods share the same complete candidate pool",
        "methods": sorted(runs.method.unique().tolist()),
        "candidate_count": len(families),
        "candidate_pool": families,
        "tasks": len(tasks),
        "seeds": seeds,
        "run_rows": len(runs),
        "candidate_split_root": (
            str(args.candidate_split_root) if args.candidate_split_root else None
        ),
        "evaluation_data_root": (
            str(args.evaluation_data_root) if args.evaluation_data_root else None
        ),
        "alignment_policy": (
            "molecule+label+within-key occurrence; candidate labels verified before filtering"
            if alignment_rows
            else "candidate prediction row order"
        ),
    }
    (score_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique seeds are required")
    tasks = list(args.tasks)
    if not tasks:
        tasks = sorted(
            path.name
            for path in next(
                directory
                for directory in sorted(args.candidate_root.iterdir())
                if directory.is_dir()
            ).iterdir()
            if path.is_dir()
        )
    if args.phase == "select":
        select_phase(args, tasks, seeds)
    else:
        score_phase(args, tasks, seeds)


if __name__ == "__main__":
    main()
