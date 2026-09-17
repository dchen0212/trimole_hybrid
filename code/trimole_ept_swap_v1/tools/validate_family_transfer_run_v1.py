#!/usr/bin/env python3
"""Independently validate a completed leakage-safe family-transfer run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def label_col(frame: pd.DataFrame) -> str:
    ignored = {"smiles", "drug", "drug_id", "mol", "id", "sample_idx"}
    return next(column for column in frame.columns if column.lower() not in ignored)


def score(metric: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "auroc":
        return float(roc_auc_score(y_true, prediction))
    if metric == "auprc":
        return float(average_precision_score(y_true, prediction))
    if metric == "spearman":
        return float(
            np.corrcoef(
                pd.Series(y_true).rank(method="average"),
                pd.Series(prediction).rank(method="average"),
            )[0, 1]
        )
    raise ValueError(metric)


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--official-test", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--metric", choices=("auroc", "auprc", "spearman"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    candidates = pd.read_csv(args.run_root / "candidate_validation_summary.csv")
    selected = pd.read_csv(args.run_root / "selected_candidates.csv")
    final_seed = pd.read_csv(args.run_root / "final_selected_test_seed_scores.csv")
    final_summary = pd.read_csv(args.run_root / "final_selected_test_summary.csv").iloc[0]
    provenance = json.loads((args.run_root / "provenance.json").read_text())

    if any("test" in column.lower() for column in candidates.columns):
        raise ValueError("candidate table contains a test-derived column")
    if len(selected) != 1 or selected.iloc[0]["target_task"] != args.target:
        raise ValueError("selected candidate manifest is not target-specific")
    selection_stat = str(selected.iloc[0]["selection_stat"])
    best = candidates.sort_values(
        [selection_stat, "feature_set", "config_index"],
        ascending=[False, True, True],
    ).iloc[0]
    for column in ("feature_set", "config_index"):
        if str(selected.iloc[0][column]) != str(best[column]):
            raise ValueError(f"selected candidate is not validation-best by {column}")

    official = pd.read_csv(args.official_test)
    official_y = official[label_col(official)].to_numpy(dtype=float)
    prediction_files = sorted(args.run_root.glob(f"{args.target}__selected__seed_*__test_predictions.csv"))
    if len(prediction_files) != 5 or len(final_seed) != 5:
        raise ValueError("formal result must contain exactly five seeds")

    scores: list[float] = []
    predictions: list[np.ndarray] = []
    for path in prediction_files:
        frame = pd.read_csv(path)
        if not np.array_equal(frame["sample_idx"].to_numpy(), np.arange(len(official_y))):
            raise ValueError(f"sample index mismatch: {path}")
        y_true = frame["y_true"].to_numpy(dtype=float)
        if not np.allclose(y_true, official_y, rtol=0.0, atol=1e-12):
            raise ValueError(f"official label mismatch: {path}")
        prediction = frame["prediction"].to_numpy(dtype=float)
        scores.append(score(args.metric, y_true, prediction))
        predictions.append(prediction)

    reported_scores = final_seed.sort_values("seed")["test_score"].to_numpy(dtype=float)
    if not np.allclose(scores, reported_scores, rtol=0.0, atol=1e-12):
        raise ValueError("recomputed seed scores do not match final seed table")
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    ensemble = score(args.metric, official_y, np.mean(np.stack(predictions), axis=0))
    checks = {
        "seed_mean_matches": close(mean, float(final_summary["test_mean"])),
        "seed_std_matches": close(std, float(final_summary["test_std"])),
        "ensemble_matches": close(ensemble, float(final_summary["test_ensemble"])),
        "candidate_selection_validation_only": True,
        "official_test_alignment": True,
        "five_seed_predictions_present": True,
    }
    if not all(checks.values()):
        raise ValueError(f"validation checks failed: {checks}")

    report = {
        "target": args.target,
        "metric": args.metric,
        "selection_stat": selection_stat,
        "selected_feature_set": str(selected.iloc[0]["feature_set"]),
        "selected_config_index": int(selected.iloc[0]["config_index"]),
        "valid_mean": float(selected.iloc[0]["valid_mean"]),
        "test_seed_scores": scores,
        "test_mean": mean,
        "test_std_ddof1": std,
        "test_ensemble": ensemble,
        "max_abs_test_label_difference": float(
            max(
                np.max(
                    np.abs(
                        pd.read_csv(path)["y_true"].to_numpy(dtype=float)
                        - official_y
                    )
                )
                for path in prediction_files
            )
        ),
        "git_commit": provenance["git_commit"],
        "checks": checks,
        "provenance_warning": (
            "xgboost distribution is xgboost-cpu 3.0.4; original provenance lookup used the module name"
            if provenance["packages"].get("xgboost") == "not-installed"
            else ""
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
