"""Nested validation audit for task-wise candidate selection.

The audit never opens official test predictions or labels.  It partitions the
official validation predictions into outer folds, chooses a recipe on the
remaining validation rows, and scores that frozen recipe on the held-out
validation rows.  This measures selection stability without relabelling the
official test set as a development set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import run_expanded_candidate_pool_controls_v1 as common


RULES = (
    "per_task_single",
    "validation_top2_average",
    "validation_top3_average",
    "uniform_candidate_average",
    "validation_stacking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def _folds(metric: str, labels: np.ndarray, n_splits: int, seed: int):
    if metric in common.CLASSIFICATION_METRICS:
        _, counts = np.unique(labels.astype(int), return_counts=True)
        splits = min(n_splits, int(counts.min()))
        if splits < 2:
            raise ValueError("nested validation needs at least two samples per class")
        return StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels.astype(int)
        )
    splits = min(n_splits, len(labels))
    if splits < 2:
        raise ValueError("nested validation needs at least two validation rows")
    return KFold(n_splits=splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)))


def _recipe_predictions(
    metric: str,
    families: list[str],
    ranking: list[str],
    valid_by_family: dict[str, np.ndarray],
    fit_matrix: np.ndarray,
    fit_labels: np.ndarray,
    evaluation_matrix: np.ndarray,
    seed: int,
    stacking_oof: bool,
) -> dict[str, np.ndarray]:
    selections = {
        "per_task_single": ranking[:1],
        "validation_top2_average": ranking[:2],
        "validation_top3_average": ranking[:3],
        "uniform_candidate_average": families,
    }
    output: dict[str, np.ndarray] = {}
    for rule, selected in selections.items():
        _, output[rule] = common.oriented_average(
            metric,
            [valid_by_family[name] for name in selected],
            [valid_by_family[name] for name in selected],
        )
    if stacking_oof:
        output["validation_stacking"] = _oof_stacking(
            metric, fit_matrix, fit_labels, seed
        )
    else:
        output["validation_stacking"] = common.fixed_stacking_prediction(
            metric, fit_matrix, fit_labels, evaluation_matrix, seed
        )
    return output


def _oof_stacking(
    metric: str, matrix: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    """Produce inner-fold OOF predictions for the stacking candidate."""

    output = np.empty(len(labels), dtype=np.float64)
    for train_index, holdout_index in _folds(metric, labels, min(5, len(labels)), seed + 17):
        model_prediction = common.fixed_stacking_prediction(
            metric,
            matrix[train_index],
            labels[train_index],
            matrix[holdout_index],
            seed,
        )
        output[holdout_index] = model_prediction
    if not np.isfinite(output).all():
        raise ValueError("non-finite nested OOF stacking prediction")
    return output


def _choose(metric: str, scores: dict[str, float]) -> str:
    return max(scores, key=lambda rule: common.utility(metric, scores[rule]))


def _ranking(
    metric: str,
    families: list[str],
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[str]:
    scores = {
        family: common.score(metric, labels, predictions[family]) for family in families
    }
    return sorted(
        families,
        key=lambda family: (-common.utility(metric, scores[family]), family),
    )


def run(args: argparse.Namespace) -> None:
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    base = json.loads(args.base_selection.read_text())
    families = list(base["candidate_pool"])
    tasks = list(base["tasks"])
    seeds = [int(seed) for seed in base["seeds"]]
    if common.file_set_digest(args.candidate_root, families, tasks, seeds) != base["file_set_size_digest"]:
        raise ValueError("candidate pool differs from frozen validation selection")

    rows: list[dict[str, object]] = []
    for task in tasks:
        metric = str(base["task_metrics"][task])
        for seed in seeds:
            labels: np.ndarray | None = None
            predictions: dict[str, np.ndarray] = {}
            for family in families:
                path = common.candidate_file(args.candidate_root, family, task, seed, "valid")
                observed, prediction = common.load_prediction_for_phase(path, "valid", "select")
                labels = common.verify_labels(labels, observed, path)
                predictions[family] = prediction
            assert labels is not None
            matrix = np.column_stack([predictions[name] for name in families])
            for fold, (inner, outer) in enumerate(_folds(metric, labels, args.n_splits, seed), start=1):
                inner_labels = labels[inner]
                outer_labels = labels[outer]
                inner_by_family = {name: value[inner] for name, value in predictions.items()}
                outer_by_family = {name: value[outer] for name, value in predictions.items()}
                inner_ranking = _ranking(metric, families, inner_labels, inner_by_family)
                inner_matrix = matrix[inner]
                outer_matrix = matrix[outer]
                inner_predictions = _recipe_predictions(
                    metric,
                    families,
                    inner_ranking,
                    inner_by_family,
                    inner_matrix,
                    inner_labels,
                    inner_matrix,
                    seed,
                    True,
                )
                outer_predictions = _recipe_predictions(
                    metric,
                    families,
                    inner_ranking,
                    outer_by_family,
                    inner_matrix,
                    inner_labels,
                    outer_matrix,
                    seed,
                    False,
                )
                inner_scores = {
                    rule: common.score(metric, inner_labels, prediction)
                    for rule, prediction in inner_predictions.items()
                }
                selected = _choose(metric, inner_scores)
                outer_scores = {
                    rule: common.score(metric, outer_labels, prediction)
                    for rule, prediction in outer_predictions.items()
                }
                rule_values = np.asarray(list(outer_scores.values()), dtype=np.float64)
                score_min = float(rule_values.min())
                score_max = float(rule_values.max())
                score_range = score_max - score_min
                if score_range <= 1e-12:
                    normalized = {rule: 0.0 for rule in outer_scores}
                elif metric == "MAE":
                    normalized = {
                        rule: (score_max - value) / score_range
                        for rule, value in outer_scores.items()
                    }
                else:
                    normalized = {
                        rule: (value - score_min) / score_range
                        for rule, value in outer_scores.items()
                    }
                normalized_effect = normalized[selected] - normalized["per_task_single"]
                raw_effect = outer_scores[selected] - outer_scores["per_task_single"]
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "seed": seed,
                        "outer_fold": fold,
                        "selected_rule": selected,
                        "selected_inner_score": inner_scores[selected],
                        "selected_outer_score": outer_scores[selected],
                        "per_task_single_outer_score": outer_scores["per_task_single"],
                        "selected_minus_single_outer": raw_effect,
                        "outer_rule_score_range": score_range,
                        "normalized_selected_score": normalized[selected],
                        "normalized_single_score": normalized["per_task_single"],
                        "normalized_selected_minus_single": normalized_effect,
                        "selected_wins_single": int(normalized_effect > 1e-12),
                        "selected_ties_single": int(abs(normalized_effect) <= 1e-12),
                        "validation_top2_outer_score": outer_scores["validation_top2_average"],
                        "validation_top3_outer_score": outer_scores["validation_top3_average"],
                        "uniform_outer_score": outer_scores["uniform_candidate_average"],
                        "stacking_outer_score": outer_scores["validation_stacking"],
                    }
                )

    frame = pd.DataFrame(rows)
    args.out_root.mkdir(parents=True)
    frame.to_csv(args.out_root / "nested_validation_outer_scores.csv", index=False)
    summary = (
        frame.groupby(["task", "metric"], as_index=False)
        .agg(
            outer_folds=("outer_fold", "count"),
            selected_minus_single_outer_mean=("selected_minus_single_outer", "mean"),
            selected_minus_single_outer_sd=("selected_minus_single_outer", "std"),
            selected_outer_score_mean=("selected_outer_score", "mean"),
            per_task_single_outer_score_mean=("per_task_single_outer_score", "mean"),
            normalized_selected_minus_single_mean=("normalized_selected_minus_single", "mean"),
            normalized_selected_minus_single_sd=("normalized_selected_minus_single", "std"),
            selected_wins=("selected_wins_single", "sum"),
            selected_ties=("selected_ties_single", "sum"),
        )
    )
    summary.to_csv(args.out_root / "nested_validation_summary.csv", index=False)
    selection_frequency = (
        frame.groupby(["task", "selected_rule"], as_index=False)
        .size()
        .rename(columns={"size": "outer_fold_count"})
    )
    selection_frequency.to_csv(args.out_root / "nested_validation_selection_frequency.csv", index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "validation-only nested audit; official test predictions and labels were not opened",
        "candidate_root": str(args.candidate_root),
        "base_selection": str(args.base_selection),
        "base_selection_sha256": hashlib.sha256(args.base_selection.read_bytes()).hexdigest(),
        "candidate_pool": families,
        "tasks": tasks,
        "seeds": seeds,
        "rules": list(RULES),
        "n_splits_requested": args.n_splits,
        "outer_rows": len(frame),
    }
    (args.out_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    run(parse_args())
