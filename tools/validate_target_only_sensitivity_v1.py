"""Independently verify five-seed target-only and target-wide task scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def task_score(metric: str, labels: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "auroc":
        return float(roc_auc_score(labels, prediction))
    if metric == "spearman":
        return float(spearmanr(labels, prediction).correlation)
    raise ValueError(f"unsupported metric: {metric}")


def verify_run(root: Path, data_root: Path, policy: str) -> dict[str, object]:
    summary = pd.read_csv(root / "final_selected_test_summary.csv")
    if len(summary) != 1:
        raise ValueError(f"expected one target summary in {root}")
    row = summary.iloc[0]
    task = str(row.target_task)
    official = pd.read_csv(data_root / task / "test.csv")
    label_column = "Y" if "Y" in official else "label"
    labels = official[label_column].to_numpy(dtype=float)
    predictions: list[np.ndarray] = []
    scores: list[float] = []
    for seed in (1, 2, 3, 4, 5):
        path = root / f"{task}__selected__seed_{seed}__test_predictions.csv"
        frame = pd.read_csv(path)
        if len(frame) != len(labels) or not np.array_equal(frame.sample_idx, np.arange(len(labels))):
            raise ValueError(f"test row mismatch in {path}")
        if not np.allclose(frame.y_true, labels, rtol=0, atol=1e-10):
            raise ValueError(f"test label mismatch in {path}")
        prediction = frame.prediction.to_numpy(dtype=float)
        predictions.append(prediction)
        scores.append(task_score(str(row.metric), labels, prediction))
    observed_mean = float(np.mean(scores))
    observed_std = float(np.std(scores, ddof=1))
    observed_ensemble = task_score(str(row.metric), labels, np.mean(predictions, axis=0))
    error = max(
        abs(observed_mean - float(row.test_mean)),
        abs(observed_std - float(row.test_std)),
        abs(observed_ensemble - float(row.test_ensemble)),
    )
    if error > 1e-10:
        raise ValueError(f"score mismatch in {root}: {error}")
    provenance = json.loads((root / "provenance.json").read_text())
    if provenance.get("source_policy", "target_universe") != policy:
        raise ValueError(f"unexpected run provenance: {root}")
    if policy == "target_only" and provenance.get("git_status_at_start") != "clean":
        raise ValueError(f"target-only run was not started from a clean checkout: {root}")
    audit = pd.read_csv(root / "holdout_overlap_filter_audit.csv")
    source = audit[audit.source_task.ne(task)]
    if policy == "target_only" and int(source.rows_after.sum()) != 0:
        raise ValueError(f"target-only run retained source rows: {root}")
    return {
        "task": task,
        "metric": row.metric,
        "policy": policy,
        "uses_target_test_structure_for_source_filter": policy == "target_universe",
        "cross_task_source_rows_after_selection": int(source[source.stage.eq("selection")].rows_after.sum()),
        "cross_task_source_rows_after_refit": int(source[source.stage.eq("final")].rows_after.sum()),
        "selected_feature_set": row.feature_set,
        "selected_config_index": int(row.config_index),
        "test_mean": observed_mean,
        "test_std": observed_std,
        "test_ensemble": observed_ensemble,
        "n_seeds": 5,
        "n_test_rows": len(labels),
        "max_score_recalculation_error": error,
        "git_commit": provenance.get("git_commit", "unavailable"),
        "checkout_clean_at_start": provenance.get("git_status_at_start") == "clean",
        "provenance_warning": (
            "none" if provenance.get("git_status_at_start") == "clean"
            else "historical target-wide run started from a dirty checkout"
        ),
        "result_directory": root.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-only-clearance", type=Path, required=True)
    parser.add_argument("--target-only-cyp3a4", type=Path, required=True)
    parser.add_argument("--target-universe-clearance", type=Path, required=True)
    parser.add_argument("--target-universe-cyp3a4", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    rows = [
        verify_run(args.target_only_clearance, args.data_root, "target_only"),
        verify_run(args.target_universe_clearance, args.data_root, "target_universe"),
        verify_run(args.target_only_cyp3a4, args.data_root, "target_only"),
        verify_run(args.target_universe_cyp3a4, args.data_root, "target_universe"),
    ]
    args.out_root.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.out_root / "target_only_vs_target_universe.csv", index=False)
    (args.out_root / "audit_summary.json").write_text(
        json.dumps({"rows": len(rows), "all_scores_recomputed": True, "max_error": max(row["max_score_recalculation_error"] for row in rows)}, indent=2) + "\n"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
