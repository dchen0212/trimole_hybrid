from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_paired_bootstrap_v1 import load_test_labels, metric_value


SAFE_TASKS = {
    "clearance_hepatocyte_az",
    "cyp3a4_substrate_carbonmangels",
}

SAFE_RESULT_DIRECTORIES = {
    "clearance_hepatocyte_az": "revision_20260917_clearance_hepatocyte_leakage_safe_v2",
    "cyp3a4_substrate_carbonmangels": "revision_20260917_cyp3a4_substrate_leakage_safe_v2",
}

SOLUBILITY_TASK = "solubility_aqsoldb"
SINGLE_RUN_TASKS = {"bbb_martins"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-metadata", type=Path, required=True)
    parser.add_argument("--legacy-long", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--v29-seed-scores", type=Path, required=True)
    parser.add_argument("--safe-results-root", type=Path, required=True)
    parser.add_argument("--solubility-results-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_many(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def prediction_column(frame: pd.DataFrame) -> str:
    for column in ("prediction", "y_pred", "y_prob", "pred", "score"):
        if column in frame.columns:
            return column
    excluded = {"sample_idx", "y_true", "Y", "label"}
    candidates = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if len(candidates) != 1:
        raise ValueError(f"cannot identify prediction column: {frame.columns.tolist()}")
    return candidates[0]


def normalize_frame(frame: pd.DataFrame, labels: np.ndarray, source: str) -> pd.DataFrame:
    if "sample_idx" in frame:
        frame = frame.sort_values("sample_idx")
        if not np.array_equal(frame.sample_idx.to_numpy(), np.arange(len(frame))):
            raise ValueError(f"sample index mismatch: {source}")
    if len(frame) != len(labels):
        raise ValueError(f"row mismatch: {source}: {len(frame)} != {len(labels)}")
    for label_name in ("y_true", "Y", "label"):
        if label_name in frame and not np.allclose(
            frame[label_name].to_numpy(dtype=float), labels, equal_nan=True
        ):
            raise ValueError(f"label alignment mismatch: {source}")
    prediction = frame[prediction_column(frame)].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"non-finite prediction: {source}")
    return pd.DataFrame(
        {"sample_idx": np.arange(len(prediction)), "prediction": prediction}
    )


def seed_from_text(value: object, fallback: int) -> int:
    matches = re.findall(r"\d+", str(value))
    return int(matches[-1]) if matches else fallback


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.task_metadata)
    metric_by_task = dict(zip(metadata.task, metadata.metric))
    expected_tasks = set(metric_by_task)
    sources: dict[str, list[dict[str, object]]] = {}

    legacy = pd.read_csv(args.legacy_long)
    for (task, run_label), rows in legacy.groupby(["task", "run_label"], sort=True):
        if task in SAFE_TASKS or task == SOLUBILITY_TASK:
            continue
        sources.setdefault(str(task), []).append(
            {
                "seed": seed_from_text(run_label, len(sources.get(str(task), [])) + 1),
                "source_kind": "legacy_materialized_final",
                "source_path": str(args.legacy_long),
                "frame": rows[["sample_idx", "y_true", "y_pred"]].copy(),
            }
        )
    legacy_tasks = set(sources)

    metric_path = (
        args.solubility_results_root
        / "official_metric_loss_push_all22_v1"
        / "solubility_aqsoldb__metric_auto__seed_42"
        / "test_predictions.csv"
    )
    xl_path = (
        args.solubility_results_root
        / "paper_main_chemical_prior_xl_v4_remaining4_32core"
        / SOLUBILITY_TASK
        / "test_predictions.csv"
    )
    metric_frame = pd.read_csv(metric_path)
    xl_frame = pd.read_csv(xl_path)
    for seed in range(1, 6):
        fp_path = (
            args.solubility_results_root
            / "formal_repeated_fp_xgb_5run_audit_v1"
            / SOLUBILITY_TASK
            / f"formal_seed_{seed}"
            / "test_predictions.csv"
        )
        fp_frame = pd.read_csv(fp_path)
        prediction = (
            0.3 * fp_frame[prediction_column(fp_frame)].to_numpy(dtype=float)
            + 0.3 * metric_frame[prediction_column(metric_frame)].to_numpy(dtype=float)
            + 0.4 * xl_frame[prediction_column(xl_frame)].to_numpy(dtype=float)
        )
        frame = pd.DataFrame(
            {
                "sample_idx": np.arange(len(fp_frame)),
                "y_true": fp_frame["y_true"].to_numpy(dtype=float),
                "y_pred": prediction,
            }
        )
        sources.setdefault(SOLUBILITY_TASK, []).append(
            {
                "seed": seed,
                "source_kind": "formula_reconstruction_current_split",
                "source_path": ";".join(map(str, (fp_path, metric_path, xl_path))),
                "source_paths": [fp_path, metric_path, xl_path],
                "frame": frame,
            }
        )

    recovery = pd.read_csv(args.recovery_manifest)
    for row in recovery.to_dict("records"):
        task = str(row["task"])
        if task in legacy_tasks or task in SAFE_TASKS or task == SOLUBILITY_TASK:
            continue
        path = Path(str(row["destination_path"]))
        sources.setdefault(task, []).append(
            {
                "seed": int(row["seed"]),
                "source_kind": "recovered_fixed_endpoint",
                "source_path": str(path),
                "frame": pd.read_csv(path),
            }
        )
    recovery_tasks = set(recovery.task.astype(str))

    v29 = pd.read_csv(args.v29_seed_scores)
    for row in v29.to_dict("records"):
        task = str(row["task"])
        if (
            task in legacy_tasks
            or task in recovery_tasks
            or task in SAFE_TASKS
            or task == SOLUBILITY_TASK
        ):
            continue
        path = Path(str(row["test_prediction_file"]))
        sources.setdefault(task, []).append(
            {
                "seed": int(row["seed_group"]),
                "source_kind": "v29_formula_reconstruction",
                "source_path": str(path),
                "frame": pd.read_csv(path),
            }
        )

    for task in SAFE_TASKS:
        task_root = args.safe_results_root / SAFE_RESULT_DIRECTORIES[task]
        matches = sorted(
            task_root.glob(f"{task}__selected__seed_*__test_predictions.csv")
        )
        if len(matches) != 5:
            raise ValueError(f"expected five leakage-safe predictions for {task}")
        sources[task] = [
            {
                "seed": seed_from_text(path.stem, index),
                "source_kind": "leakage_safe_family_transfer",
                "source_path": str(path),
                "frame": pd.read_csv(path),
            }
            for index, path in enumerate(matches, start=1)
        ]

    observed_tasks = set(sources)
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"task coverage mismatch; missing={sorted(expected_tasks-observed_tasks)}, "
            f"extra={sorted(observed_tasks-expected_tasks)}"
        )
    invalid_run_counts = {
        task: len(items)
        for task, items in sources.items()
        if len(items) != (1 if task in SINGLE_RUN_TASKS else 5)
    }
    if invalid_run_counts:
        raise ValueError(f"unexpected per-task run counts: {invalid_run_counts}")

    prediction_root = args.out_root / "predictions"
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    manifest_rows: list[dict[str, object]] = []
    for task in sorted(sources):
        labels = load_test_labels(args.data_root, task)
        metric = metric_by_task[task]
        seen_seeds: set[int] = set()
        for index, item in enumerate(sources[task], start=1):
            seed = int(item["seed"])
            if seed in seen_seeds:
                seed = index
            seen_seeds.add(seed)
            normalized = normalize_frame(item["frame"], labels, str(item["source_path"]))
            output = prediction_root / task / f"seed_{seed}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(output, index=False)
            manifest_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "model": "trimole_hybrid",
                    "seed": seed,
                    "prediction_file": str(output),
                    "n_samples": len(normalized),
                    "recomputed_score": metric_value(
                        metric, labels, normalized.prediction.to_numpy()
                    ),
                    "source_kind": item["source_kind"],
                    "source_path": item["source_path"],
                    "source_sha256": sha256_many(
                        [Path(path) for path in item.get("source_paths", [item["source_path"]])]
                    ),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_root / "prediction_manifest.csv", index=False)
    manifest.groupby(["task", "metric", "model", "source_kind"], as_index=False).agg(
        n_runs=("seed", "nunique"),
        n_samples=("n_samples", "first"),
        score_mean=("recomputed_score", "mean"),
        score_std=("recomputed_score", "std"),
    ).to_csv(args.out_root / "score_summary.csv", index=False)
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "policy": "legacy sources normalized; leakage-safe family-transfer predictions override submitted files",
                "tasks": len(observed_tasks),
                "manifest_rows": len(manifest),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
