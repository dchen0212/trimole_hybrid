#!/usr/bin/env python3
"""Recover selected fixed-endpoint predictions from the migrated archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


SEEDS = (101, 202, 303, 404, 505)
SOURCES = {
    "bioavailability_ma": ("top1_method_transfer_chem_lite_v1", "AUROC"),
    "cyp2c9_substrate_carbonmangels": (
        "paper_main_chemical_prior_xl_v4_all22_32core",
        "AUPRC",
    ),
    "cyp2c9_veith": ("top1_method_transfer_chem_lite_v1", "AUPRC"),
    "herg": ("paper_main_chemical_prior_xl_v4_all22_32core", "AUROC"),
    "ld50_zhu": ("paper_main_chemical_prior_xl_v4_all22_32core", "MAE"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_score(metric: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "AUROC":
        return float(roc_auc_score(y_true, prediction))
    if metric == "AUPRC":
        return float(average_precision_score(y_true, prediction))
    if metric == "MAE":
        return float(mean_absolute_error(y_true, prediction))
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    task_summaries: list[dict[str, object]] = []
    for task, (source_group, metric) in SOURCES.items():
        source_dir = args.archive_root / source_group / task
        source_summary = json.loads((source_dir / "summary.json").read_text())
        reported_scores = [
            float(value) for value in source_summary["test_scores"].split(";")
        ]
        if len(reported_scores) != len(SEEDS):
            raise ValueError(f"unexpected reported seed count for {task}")

        task_out = args.out_root / task
        task_out.mkdir(parents=True)
        shutil.copy2(source_dir / "summary.json", task_out / "source_summary.json")
        predictions: list[np.ndarray] = []
        y_reference: np.ndarray | None = None
        recomputed_scores: list[float] = []
        for index, seed in enumerate(SEEDS):
            source = source_dir / f"test_predictions_formal_seed_{seed}.csv"
            destination = task_out / source.name
            frame = pd.read_csv(source)
            pred_col = "y_prob" if "y_prob" in frame else "y_pred"
            y_true = frame["y_true"].to_numpy(dtype=float)
            prediction = frame[pred_col].to_numpy(dtype=float)
            if y_reference is None:
                y_reference = y_true
            elif not np.array_equal(y_reference, y_true):
                raise ValueError(f"seed label mismatch for {task}, seed {seed}")
            score = metric_score(metric, y_true, prediction)
            if abs(score - reported_scores[index]) > 1e-10:
                raise ValueError(
                    f"metric mismatch for {task}, seed {seed}: {score} vs {reported_scores[index]}"
                )
            shutil.copy2(source, destination)
            predictions.append(prediction)
            recomputed_scores.append(score)
            manifest.append(
                {
                    "task": task,
                    "metric": metric,
                    "seed": seed,
                    "rows": len(frame),
                    "recomputed_score": score,
                    "reported_score": reported_scores[index],
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256(source),
                    "destination_path": str(destination.resolve()),
                    "destination_sha256": sha256(destination),
                }
            )

        assert y_reference is not None
        seed_mean = np.mean(np.stack(predictions), axis=0)
        pred_col = "y_prob" if metric in {"AUROC", "AUPRC"} else "y_pred"
        ensemble_path = task_out / "test_predictions_seed_mean.csv"
        pd.DataFrame(
            {
                "sample_idx": np.arange(len(y_reference)),
                "y_true": y_reference,
                pred_col: seed_mean,
            }
        ).to_csv(ensemble_path, index=False)
        task_summaries.append(
            {
                "task": task,
                "metric": metric,
                "reported_seed_score_mean": source_summary["test_mean"],
                "reported_seed_score_std": source_summary["test_std"],
                "recomputed_seed_score_mean": float(np.mean(recomputed_scores)),
                "recomputed_seed_score_std_ddof1": float(
                    np.std(recomputed_scores, ddof=1)
                ),
                "seed_mean_prediction_score": metric_score(
                    metric, y_reference, seed_mean
                ),
                "seed_mean_prediction_file": str(ensemble_path.resolve()),
                "seed_mean_prediction_sha256": sha256(ensemble_path),
            }
        )

    with (args.out_root / "recovery_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (args.out_root / "recovered_task_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_summaries[0]))
        writer.writeheader()
        writer.writerows(task_summaries)
    print(f"recovered and verified {len(manifest)} seed prediction files")


if __name__ == "__main__":
    main()
