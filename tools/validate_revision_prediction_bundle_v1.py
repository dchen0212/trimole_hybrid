"""Recompute published method scores from the label-free prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


def metric_score(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    if metric == "Spearman":
        return float(spearmanr(labels, predictions).correlation)
    raise ValueError(f"unsupported metric: {metric}")


def validate(bundle: Path, score_table: Path, data_root: Path) -> pd.DataFrame:
    scores = pd.read_csv(score_table)
    labels_by_task: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    with ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        if archive.testzip() is not None:
            raise ValueError("prediction bundle failed ZIP integrity check")
        for row in scores.itertuples(index=False):
            task = str(row.task)
            if task not in labels_by_task:
                frame = pd.read_csv(data_root / task / "test.csv")
                label_column = "Y" if "Y" in frame else "label"
                labels_by_task[task] = frame[label_column].to_numpy(dtype=float)
            member = f"method/{row.model}/{task}/seed_{int(row.seed)}/test.csv"
            with archive.open(member) as handle:
                predictions = pd.read_csv(handle)
            labels = labels_by_task[task]
            if len(predictions) != len(labels) or not np.array_equal(
                predictions.sample_idx.to_numpy(), np.arange(len(labels))
            ):
                raise ValueError(f"test row alignment failed: {member}")
            actual = metric_score(str(row.metric), labels, predictions.prediction.to_numpy(dtype=float))
            rows.append(
                {
                    "task": task,
                    "metric": row.metric,
                    "model": row.model,
                    "seed": int(row.seed),
                    "published_score": float(row.test_score),
                    "recomputed_score": actual,
                    "absolute_error": abs(actual - float(row.test_score)),
                    "n_test_rows": len(labels),
                    "bundle_member": member,
                }
            )
        if len(rows) != len(manifest["tasks"]) * len(manifest["methods"]) * len(manifest["seeds"]):
            raise ValueError("published score grid does not match bundle manifest")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    frame = validate(args.bundle, args.score_table, args.data_root)
    args.out_root.mkdir(parents=True)
    frame.to_csv(args.out_root / "independent_score_recalculation.csv", index=False)
    summary = {
        "scores_recomputed": len(frame),
        "tasks": int(frame.task.nunique()),
        "methods": int(frame.model.nunique()),
        "seeds": int(frame.seed.nunique()),
        "max_absolute_error": float(frame.absolute_error.max()),
        "all_match_at_1e-10": bool((frame.absolute_error <= 1e-10).all()),
    }
    (args.out_root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["all_match_at_1e-10"]:
        raise RuntimeError(f"score recalculation mismatch: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
