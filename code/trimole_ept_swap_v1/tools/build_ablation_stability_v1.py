from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


from run_paired_bootstrap_v1 import load_prediction, load_test_labels, metric_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-table", type=Path, required=True)
    parser.add_argument("--score-corrections", type=Path)
    parser.add_argument("--selection-scores", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def normalized_rank_loss(
    frame: pd.DataFrame,
    task_col: str = "task",
    variant_col: str = "variant",
    metric_col: str = "metric",
    score_col: str = "score_mean",
    full_variant: str = "full_v36_final",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, group in frame.groupby(task_col, sort=True):
        group = group[group[score_col].notna()].copy()
        if group.empty:
            raise ValueError(f"no finite ablation scores for {task}")
        metrics = group[metric_col].dropna().unique().tolist()
        if len(metrics) != 1:
            raise ValueError(f"expected one metric for {task}, found {metrics}")
        metric = str(metrics[0])
        ascending = metric == "MAE"
        ranked = group.copy()
        ranked["task_rank"] = ranked[score_col].rank(
            method="average", ascending=ascending
        )
        full_rows = ranked[ranked[variant_col] == full_variant]
        if len(full_rows) != 1:
            raise ValueError(f"expected one {full_variant} row for {task}")
        full_rank = float(full_rows.task_rank.iloc[0])
        denominator = max(len(ranked) - 1, 1)
        for record in ranked.to_dict("records"):
            rows.append(
                {
                    **record,
                    "full_variant": full_variant,
                    "full_rank": full_rank,
                    "n_ranked_variants": len(ranked),
                    "normalized_rank_loss": (
                        float(record["task_rank"]) - full_rank
                    )
                    / denominator,
                }
            )
    return pd.DataFrame(rows)


def validation_labels(data_root: Path, task: str) -> np.ndarray:
    frame = pd.read_csv(data_root / task / "valid.csv")
    column = "Y" if "Y" in frame else "label"
    return frame[column].to_numpy(dtype=np.float64)


def selection_stability(
    selection_scores: pd.DataFrame,
    data_root: Path,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, task_rows in selection_scores.groupby("task", sort=True):
        metric_values = task_rows.metric.unique().tolist()
        if len(metric_values) != 1:
            raise ValueError(f"multiple metrics for {task}")
        metric = str(metric_values[0])
        labels = validation_labels(data_root, str(task))
        predictions = {}
        for modality, modality_rows in task_rows.groupby("modality", sort=True):
            values = [
                load_prediction(path, len(labels))
                for path in modality_rows.prediction_file
            ]
            predictions[str(modality)] = np.mean(np.stack(values), axis=0)
        modalities = sorted(predictions)
        counts = dict.fromkeys(modalities, 0)
        valid = 0
        for _ in range(replicates):
            index = rng.integers(0, len(labels), size=len(labels))
            scored = []
            for modality in modalities:
                try:
                    value = metric_value(
                        metric, labels[index], predictions[modality][index]
                    )
                except ValueError:
                    value = np.nan
                if np.isfinite(value):
                    scored.append(((-value if metric == "MAE" else value), modality))
            if len(scored) != len(modalities):
                continue
            selected = max(scored)[1]
            counts[selected] += 1
            valid += 1
        if valid < max(100, int(0.8 * replicates)):
            raise RuntimeError(f"too few valid bootstrap replicates for {task}")
        frequencies = np.array([counts[name] / valid for name in modalities])
        positive = frequencies[frequencies > 0]
        entropy = float(-np.sum(positive * np.log(positive)) / np.log(len(modalities)))
        switch_rate = float(1.0 - np.max(frequencies))
        for modality, frequency in zip(modalities, frequencies):
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "candidate": modality,
                    "selection_count": counts[modality],
                    "selection_frequency": frequency,
                    "bootstrap_replicates_valid": valid,
                    "selection_entropy_normalized": entropy,
                    "switch_rate": switch_rate,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ablation = pd.read_csv(args.ablation_table)
    score_col = "score_mean" if "score_mean" in ablation else "ablation_score_mean"
    variant_col = "variant" if "variant" in ablation else "ablation"
    if args.score_corrections:
        corrections = pd.read_csv(args.score_corrections)
        required = {"task", variant_col, score_col}
        missing = required - set(corrections.columns)
        if missing:
            raise ValueError(f"score corrections missing columns: {sorted(missing)}")
        for correction in corrections.to_dict("records"):
            mask = (ablation.task == correction["task"]) & (
                ablation[variant_col] == correction[variant_col]
            )
            if mask.sum() != 1:
                raise ValueError(
                    f"correction must match one row: {correction['task']} / "
                    f"{correction[variant_col]}"
                )
            for column, value in correction.items():
                if column in ablation.columns and column not in {"task", variant_col}:
                    ablation.loc[mask, column] = value
    ranked = normalized_rank_loss(
        ablation, variant_col=variant_col, score_col=score_col
    )
    stability = selection_stability(
        pd.read_csv(args.selection_scores),
        args.data_root,
        args.bootstrap_replicates,
        np.random.default_rng(args.seed),
    )
    args.out_root.mkdir(parents=True, exist_ok=False)
    ranked.to_csv(args.out_root / "ablation_normalized_rank_loss.csv", index=False)
    stability.to_csv(args.out_root / "validation_selection_stability.csv", index=False)
    combined = pd.concat(
        [
            ranked.assign(record_type="ablation_rank_loss"),
            stability.assign(record_type="validation_selection_stability"),
        ],
        ignore_index=True,
        sort=False,
    )
    first = ["record_type", "task", "metric"]
    combined[first + [column for column in combined.columns if column not in first]].to_csv(
        args.out_root / "Table_S23_ablation_selection_stability.csv", index=False
    )
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "ablation_table": str(args.ablation_table),
                "score_corrections": str(args.score_corrections) if args.score_corrections else None,
                "selection_scores": str(args.selection_scores),
                "bootstrap_replicates_requested": args.bootstrap_replicates,
                "seed": args.seed,
                "normalized_rank_loss": "(variant rank - full-model rank)/(K-1) within task",
                "switch_rate": "1 - maximum bootstrap selection frequency",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
