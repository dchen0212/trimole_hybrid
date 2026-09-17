from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", default="")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_labels(data_root: Path, task: str, split: str) -> np.ndarray:
    frame = pd.read_csv(data_root / task / f"{split}.csv")
    column = "Y" if "Y" in frame.columns else "label"
    return frame[column].to_numpy(dtype=np.float64)


def score(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "Spearman":
        value = spearmanr(labels, predictions).correlation
        if value is None:
            raise ValueError("undefined Spearman correlation")
        return float(value)
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    raise ValueError(f"unsupported metric: {metric}")


def assert_close(actual: float, expected: float, context: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    per_seed = pd.read_csv(run_root / "per_seed_scores.csv")
    summary = pd.read_csv(run_root / "summary.csv").set_index("task")
    provenance = json.loads((run_root / "provenance.json").read_text())

    if args.expected_commit and provenance.get("git_commit") != args.expected_commit:
        raise AssertionError(
            f"commit mismatch: {provenance.get('git_commit')} != {args.expected_commit}"
        )

    task_results: dict[str, object] = {}
    prediction_hashes: dict[str, str] = {}
    for task, rows in per_seed.groupby("task", sort=True):
        if len(rows) != 5 or sorted(rows["seed_group"].astype(int)) != [1, 2, 3, 4, 5]:
            raise AssertionError(f"{task}: expected exactly five seed groups")
        metric = str(rows["metric"].iloc[0])
        valid_scores: list[float] = []
        test_scores: list[float] = []
        for row in rows.sort_values("seed_group").to_dict("records"):
            for split, scores in (("valid", valid_scores), ("test", test_scores)):
                path = Path(row[f"{split}_prediction_file"])
                frame = pd.read_csv(path)
                labels = official_labels(data_root, task, split)
                expected_index = np.arange(len(labels))
                if not np.array_equal(frame["sample_idx"].to_numpy(), expected_index):
                    raise AssertionError(f"{task} {split}: sample_idx mismatch in {path}")
                if not np.allclose(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    labels,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise AssertionError(f"{task} {split}: official labels mismatch in {path}")
                value = score(
                    metric,
                    labels,
                    frame["prediction"].to_numpy(dtype=np.float64),
                )
                assert_close(value, float(row[f"{split}_score"]), f"{task} {split}")
                scores.append(value)
                prediction_hashes[str(path)] = sha256(path)

        task_summary = summary.loc[task]
        checks = {
            "valid_mean": float(np.mean(valid_scores)),
            "valid_std": float(np.std(valid_scores, ddof=1)),
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores, ddof=1)),
        }
        for key, value in checks.items():
            assert_close(value, float(task_summary[f"formal_{key}"]), f"{task} {key}")
        task_results[task] = {
            "metric": metric,
            "n_valid": len(official_labels(data_root, task, "valid")),
            "n_test": len(official_labels(data_root, task, "test")),
            "valid_scores": valid_scores,
            "test_scores": test_scores,
            **checks,
        }

    expected_tasks = sorted(str(task) for task in provenance["tasks"])
    if sorted(task_results) != expected_tasks:
        raise AssertionError(
            f"task mismatch: validated={sorted(task_results)}, provenance={expected_tasks}"
        )

    report = {
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "generator_git_commit": provenance.get("git_commit"),
        "task_count": len(task_results),
        "prediction_file_count": len(prediction_hashes),
        "tasks": task_results,
        "prediction_sha256": prediction_hashes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
