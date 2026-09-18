"""Strict common-pool comparison for the Bioinformatics revision.

Every method receives the same nine frozen candidate-prediction families. The
selection phase reads validation predictions only and freezes both a task-wise
recipe and a six-configuration AutoML-style meta-learner before the score phase
opens test predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_expanded_candidate_pool_controls_v1 as common


RULES = (
    "global_single",
    "per_task_single",
    "validation_top2_average",
    "validation_top3_average",
    "uniform_candidate_average",
    "validation_stacking",
)
AUTOML_CONFIGS = tuple(range(6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "score"), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-split-root", type=Path)
    parser.add_argument("--evaluation-data-root", type=Path)
    return parser.parse_args()


def _make_meta_model(metric: str, config_index: int, seed: int):
    if metric in common.CLASSIFICATION_METRICS:
        models = (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=seed)),
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
            make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000, random_state=seed)),
            RandomForestClassifier(n_estimators=300, max_depth=2, min_samples_leaf=5, random_state=seed, n_jobs=1),
            ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=seed, n_jobs=1),
            HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, learning_rate=0.05, random_state=seed),
        )
    else:
        models = (
            make_pipeline(StandardScaler(), Ridge(alpha=0.1, solver="lsqr")),
            make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr")),
            make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr")),
            RandomForestRegressor(n_estimators=300, max_depth=2, min_samples_leaf=5, random_state=seed, n_jobs=1),
            ExtraTreesRegressor(n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=seed, n_jobs=1),
            HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, learning_rate=0.05, random_state=seed),
        )
    return models[config_index]


def _predict(model, metric: str, matrix: np.ndarray) -> np.ndarray:
    if metric in common.CLASSIFICATION_METRICS:
        return model.predict_proba(matrix)[:, 1]
    return model.predict(matrix)


def _folds(metric: str, labels: np.ndarray, seed: int):
    if metric in common.CLASSIFICATION_METRICS:
        _, counts = np.unique(labels.astype(int), return_counts=True)
        n_splits = min(5, int(counts.min()))
        if n_splits < 2:
            raise ValueError("classification validation set needs at least two samples per class")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels.astype(int)
        )
    n_splits = min(5, len(labels))
    if n_splits < 2:
        raise ValueError("validation set needs at least two rows")
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)))


def oof_meta_prediction(
    metric: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    config_index: int,
    seed: int,
) -> np.ndarray:
    prediction = np.empty(len(labels), dtype=np.float64)
    for train_index, holdout_index in _folds(metric, labels, seed):
        model = _make_meta_model(metric, config_index, seed)
        model.fit(matrix[train_index], labels[train_index])
        prediction[holdout_index] = _predict(model, metric, matrix[holdout_index])
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite OOF meta-prediction")
    return prediction


def fit_meta_prediction(
    metric: str,
    valid_matrix: np.ndarray,
    valid_labels: np.ndarray,
    test_matrix: np.ndarray,
    config_index: int,
    seed: int,
) -> np.ndarray:
    model = _make_meta_model(metric, config_index, seed)
    model.fit(valid_matrix, valid_labels)
    prediction = _predict(model, metric, test_matrix)
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite test meta-prediction")
    return prediction


def _load_valid_pool(
    candidate_root: Path,
    families: list[str],
    task: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels: np.ndarray | None = None
    predictions: dict[str, np.ndarray] = {}
    for family in families:
        path = common.candidate_file(candidate_root, family, task, seed, "valid")
        observed_labels, prediction = common.load_prediction_for_phase(path, "valid", "select")
        labels = common.verify_labels(labels, observed_labels, path)
        predictions[family] = prediction
    assert labels is not None
    return labels, predictions


def _simple_predictions(
    metric: str,
    families: list[str],
    ranking: list[str],
    global_single: str,
    valid_by_family: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    selections = {
        "global_single": [global_single],
        "per_task_single": ranking[:1],
        "validation_top2_average": ranking[:2],
        "validation_top3_average": ranking[:3],
        "uniform_candidate_average": families,
    }
    output: dict[str, np.ndarray] = {}
    for method, selected in selections.items():
        output[method], _ = common.oriented_average(
            metric,
            [valid_by_family[name] for name in selected],
            [valid_by_family[name] for name in selected],
        )
    return output


def _choose(metric: str, frame: pd.DataFrame, item_column: str) -> str | int:
    ordered = frame.assign(
        selection_utility=frame.valid_score_mean.map(lambda value: common.utility(metric, value))
    ).sort_values(["selection_utility", item_column], ascending=[False, True])
    return ordered.iloc[0][item_column]


def select_phase(args: argparse.Namespace) -> None:
    output = args.out_root / "selection"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = json.loads(args.base_selection.read_text())
    families = list(base["candidate_pool"])
    tasks = list(base["tasks"])
    seeds = [int(seed) for seed in base["seeds"]]
    if common.file_set_digest(args.candidate_root, families, tasks, seeds) != base["file_set_size_digest"]:
        raise ValueError("candidate pool differs from the frozen base selection")

    recipe_rows: list[dict[str, object]] = []
    automl_rows: list[dict[str, object]] = []
    for task in tasks:
        metric = str(base["task_metrics"][task])
        ranking = list(base["per_task_ranking"][task])
        for seed in seeds:
            labels, valid_by_family = _load_valid_pool(args.candidate_root, families, task, seed)
            simple = _simple_predictions(
                metric, families, ranking, str(base["global_single"]), valid_by_family
            )
            matrix = np.column_stack([valid_by_family[name] for name in families])
            simple["validation_stacking"] = oof_meta_prediction(metric, matrix, labels, 1, seed)
            for rule in RULES:
                recipe_rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "seed": seed,
                        "rule": rule,
                        "valid_score": common.score(metric, labels, simple[rule]),
                    }
                )
            for config_index in AUTOML_CONFIGS:
                prediction = oof_meta_prediction(metric, matrix, labels, config_index, seed)
                automl_rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "seed": seed,
                        "config_index": config_index,
                        "valid_score": common.score(metric, labels, prediction),
                    }
                )

    recipes = pd.DataFrame(recipe_rows)
    automl = pd.DataFrame(automl_rows)
    recipe_summary = recipes.groupby(["task", "metric", "rule"], as_index=False).agg(
        valid_score_mean=("valid_score", "mean"), valid_score_sd=("valid_score", "std")
    )
    automl_summary = automl.groupby(["task", "metric", "config_index"], as_index=False).agg(
        valid_score_mean=("valid_score", "mean"), valid_score_sd=("valid_score", "std")
    )
    selected_rules: dict[str, str] = {}
    selected_configs: dict[str, int] = {}
    for task in tasks:
        metric = str(base["task_metrics"][task])
        selected_rules[task] = str(_choose(metric, recipe_summary[recipe_summary.task.eq(task)], "rule"))
        selected_configs[task] = int(_choose(metric, automl_summary[automl_summary.task.eq(task)], "config_index"))

    output.mkdir(parents=True)
    recipes.to_csv(output / "recipe_validation_seed_scores.csv", index=False)
    recipe_summary.to_csv(output / "recipe_validation_summary.csv", index=False)
    automl.to_csv(output / "automl_validation_seed_scores.csv", index=False)
    automl_summary.to_csv(output / "automl_validation_summary.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "same nine-family pool; validation-only selection; six recipe evaluations and six AutoML meta-configurations per task; test predictions inaccessible",
        "base_selection": str(args.base_selection),
        "base_selection_sha256": hashlib.sha256(args.base_selection.read_bytes()).hexdigest(),
        "candidate_pool": families,
        "candidate_count": len(families),
        "tasks": tasks,
        "seeds": seeds,
        "rule_budget": list(RULES),
        "automl_configuration_budget": len(AUTOML_CONFIGS),
        "selected_rule_by_task": selected_rules,
        "selected_automl_config_by_task": selected_configs,
    }
    (output / "frozen_common_pool_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")


def score_phase(args: argparse.Namespace) -> None:
    output = args.out_root / "score"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base = json.loads(args.base_selection.read_text())
    frozen_path = args.out_root / "selection" / "frozen_common_pool_selection.json"
    frozen = json.loads(frozen_path.read_text())
    families = list(frozen["candidate_pool"])
    tasks = list(frozen["tasks"])
    seeds = [int(seed) for seed in frozen["seeds"]]
    rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    for task in tasks:
        metric = str(base["task_metrics"][task])
        ranking = list(base["per_task_ranking"][task])
        selections = {
            "global_single": [base["global_single"]],
            "per_task_single": ranking[:1],
            "validation_top2_average": ranking[:2],
            "validation_top3_average": ranking[:3],
            "uniform_candidate_average": families,
        }
        test_indices: np.ndarray | None = None
        evaluation_labels: np.ndarray | None = None
        if args.candidate_split_root and args.evaluation_data_root:
            test_indices, evaluation_labels, audit = common.build_test_alignment(
                args.candidate_split_root / task / "test.csv",
                args.evaluation_data_root / task / "test.csv",
            )
            alignment_rows.append({"task": task, **audit})
        for seed in seeds:
            valid_labels: np.ndarray | None = None
            test_labels: np.ndarray | None = None
            valid_by_family: dict[str, np.ndarray] = {}
            test_by_family: dict[str, np.ndarray] = {}
            for family in families:
                valid_path = common.candidate_file(args.candidate_root, family, task, seed, "valid")
                test_path = common.candidate_file(args.candidate_root, family, task, seed, "test")
                vy, vp = common.load_prediction_for_phase(valid_path, "valid", "score")
                ty, tp = common.load_prediction_for_phase(test_path, "test", "score")
                valid_labels = common.verify_labels(valid_labels, vy, valid_path)
                test_labels = common.verify_labels(test_labels, ty, test_path)
                valid_by_family[family] = vp
                test_by_family[family] = tp
            assert valid_labels is not None and test_labels is not None
            if test_indices is not None:
                test_labels = evaluation_labels
                test_by_family = {name: value[test_indices] for name, value in test_by_family.items()}
            predictions: dict[str, np.ndarray] = {}
            for method, selected in selections.items():
                _, predictions[method] = common.oriented_average(
                    metric,
                    [valid_by_family[name] for name in selected],
                    [test_by_family[name] for name in selected],
                )
            valid_matrix = np.column_stack([valid_by_family[name] for name in families])
            test_matrix = np.column_stack([test_by_family[name] for name in families])
            predictions["validation_stacking"] = common.fixed_stacking_prediction(
                metric, valid_matrix, valid_labels, test_matrix, seed
            )
            selected_rule = str(frozen["selected_rule_by_task"][task])
            predictions["common_pool_taskwise_selector"] = predictions[selected_rule]
            config_index = int(frozen["selected_automl_config_by_task"][task])
            predictions["common_pool_automl"] = fit_meta_prediction(
                metric, valid_matrix, valid_labels, test_matrix, config_index, seed
            )
            for method, prediction in predictions.items():
                path = output / "predictions" / task / f"{method}_seed_{seed}.csv"
                common.write_prediction(path, test_labels, prediction)
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "model": method,
                        "seed": seed,
                        "candidate_count": len(families),
                        "selection_budget": 6 if method in {"common_pool_taskwise_selector", "common_pool_automl"} else 0,
                        "selected_recipe": selected_rule if method == "common_pool_taskwise_selector" else "",
                        "selected_automl_config": config_index if method == "common_pool_automl" else "",
                        "test_score": common.score(metric, test_labels, prediction),
                        "prediction_file": str(path),
                    }
                )
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "common_pool_fair_runs.csv", index=False)
    frame[["task", "metric", "model", "seed", "prediction_file"]].to_csv(
        output / "prediction_manifest.csv", index=False
    )
    frame.groupby(["task", "metric", "model"], as_index=False).agg(
        score_mean=("test_score", "mean"), score_sd=("test_score", "std"), n_seeds=("seed", "nunique")
    ).to_csv(output / "common_pool_fair_summary.csv", index=False)
    if alignment_rows:
        pd.DataFrame(alignment_rows).to_csv(output / "test_split_alignment_audit.csv", index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": str(frozen_path),
        "selection_manifest_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "candidate_pool": families,
        "candidate_count": len(families),
        "methods": sorted(frame.model.unique().tolist()),
        "policy": "all eight methods use the same frozen nine-family prediction pool; task-wise selector and AutoML each have six validation-only meta-selection evaluations",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if (args.candidate_split_root is None) != (args.evaluation_data_root is None):
        raise ValueError("candidate-split-root and evaluation-data-root must be provided together")
    if args.phase == "select":
        select_phase(args)
    else:
        score_phase(args)


if __name__ == "__main__":
    main()
