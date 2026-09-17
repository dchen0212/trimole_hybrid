from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


HIGHER_IS_BETTER = {"AUROC", "AUPRC", "Spearman"}
SUPPORTED_METRICS = HIGHER_IS_BETTER | {"MAE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reference-model", default="trimole_hybrid")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def label_column(frame: pd.DataFrame) -> str:
    return "Y" if "Y" in frame.columns else "label"


def load_test_labels(data_root: Path, task: str) -> np.ndarray:
    frame = pd.read_csv(data_root / task / "test.csv")
    return frame[label_column(frame)].to_numpy(dtype=np.float64)


def load_prediction(path: str | Path, expected_rows: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if "prediction" not in frame:
        raise ValueError(f"missing prediction column: {path}")
    if "sample_idx" in frame and not np.array_equal(
        frame.sample_idx.to_numpy(), np.arange(len(frame))
    ):
        raise ValueError(f"sample index mismatch: {path}")
    values = frame.prediction.to_numpy(dtype=np.float64)
    if len(values) != expected_rows:
        raise ValueError(f"row mismatch: {path}: {len(values)} != {expected_rows}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite prediction: {path}")
    return values


def metric_value(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(labels, predictions))
    if metric == "AUPRC":
        return float(average_precision_score(labels, predictions))
    if metric == "MAE":
        return float(mean_absolute_error(labels, predictions))
    if metric == "Spearman":
        return float(spearmanr(labels, predictions).correlation)
    raise ValueError(f"unsupported metric: {metric}")


def improvement(metric: str, reference: float, comparator: float) -> float:
    if metric in HIGHER_IS_BETTER:
        return reference - comparator
    return comparator - reference


def paired_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    comparator: np.ndarray,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    observed_reference = metric_value(metric, labels, reference)
    observed_comparator = metric_value(metric, labels, comparator)
    observed_delta = improvement(metric, observed_reference, observed_comparator)
    deltas = np.empty(replicates, dtype=np.float64)
    n = len(labels)
    valid = 0
    for _ in range(replicates):
        index = rng.integers(0, n, size=n)
        try:
            reference_score = metric_value(metric, labels[index], reference[index])
            comparator_score = metric_value(metric, labels[index], comparator[index])
        except ValueError:
            continue
        delta = improvement(metric, reference_score, comparator_score)
        if np.isfinite(delta):
            deltas[valid] = delta
            valid += 1
    if valid < max(100, int(0.8 * replicates)):
        raise RuntimeError(f"too few valid bootstrap replicates: {valid}/{replicates}")
    deltas = deltas[:valid]
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2.0
        * min(
            (np.count_nonzero(deltas <= 0.0) + 1.0) / (valid + 1.0),
            (np.count_nonzero(deltas >= 0.0) + 1.0) / (valid + 1.0),
        ),
    )
    return {
        "n_samples": n,
        "reference_score": observed_reference,
        "comparator_score": observed_comparator,
        "improvement": observed_delta,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_value_two_sided": float(p_two_sided),
        "bootstrap_replicates_valid": valid,
    }


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def aggregate_predictions(
    rows: pd.DataFrame, labels: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    predictions = [load_prediction(path, len(labels)) for path in rows.prediction_file]
    seeds = sorted(rows.seed.astype(int).tolist())
    return np.mean(np.stack(predictions, axis=0), axis=0), seeds


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    manifest = pd.read_csv(args.prediction_manifest)
    required = {"task", "metric", "model", "seed", "prediction_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if not set(manifest.metric).issubset(SUPPORTED_METRICS):
        raise ValueError("manifest contains unsupported metrics")

    output_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for task, task_rows in manifest.groupby("task", sort=True):
        metrics = task_rows.metric.unique().tolist()
        if len(metrics) != 1:
            raise ValueError(f"multiple metrics for {task}: {metrics}")
        metric = str(metrics[0])
        labels = load_test_labels(args.data_root, str(task))
        reference_rows = task_rows[task_rows.model == args.reference_model]
        if reference_rows.empty:
            raise ValueError(f"missing reference model for {task}")
        reference_prediction, reference_seeds = aggregate_predictions(reference_rows, labels)
        for comparator_model, comparator_rows in task_rows[
            task_rows.model != args.reference_model
        ].groupby("model", sort=True):
            comparator_prediction, comparator_seeds = aggregate_predictions(
                comparator_rows, labels
            )
            result = paired_bootstrap(
                labels,
                reference_prediction,
                comparator_prediction,
                metric,
                args.bootstrap_replicates,
                rng,
            )
            output_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "reference_model": args.reference_model,
                    "comparator_model": comparator_model,
                    "reference_seeds": ";".join(map(str, reference_seeds)),
                    "comparator_seeds": ";".join(map(str, comparator_seeds)),
                    **result,
                }
            )

    results = pd.DataFrame(output_rows)
    if results.empty:
        raise ValueError("no model comparisons were generated")
    results["p_value_bh_fdr"] = bh_adjust(
        results.p_value_two_sided.to_numpy(dtype=np.float64)
    )
    results["significant_fdr_0_05"] = results.p_value_bh_fdr < 0.05
    args.out_root.mkdir(parents=True, exist_ok=False)
    results.to_csv(args.out_root / "paired_bootstrap_results.csv", index=False)
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_manifest": str(args.prediction_manifest),
                "reference_model": args.reference_model,
                "bootstrap_replicates_requested": args.bootstrap_replicates,
                "random_seed": args.seed,
                "delta_definition": "positive means reference model is better",
                "p_value": "two-sided paired bootstrap with plus-one correction",
                "multiplicity": "Benjamini-Hochberg across emitted comparisons",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
