from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from run_paired_bootstrap_v1 import (
    HIGHER_IS_BETTER,
    aggregate_predictions,
    bh_adjust,
    load_test_labels,
    metric_value,
    paired_bootstrap,
)


DESCRIPTORS = {
    "molecular_weight": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reference-model", default="trimole_hybrid")
    parser.add_argument("--comparator-model", default="per_task_single")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--minimum-group-size", type=int, default=30)
    parser.add_argument("--minimum-class-count", type=int, default=5)
    return parser.parse_args()


def smiles_column(frame: pd.DataFrame) -> str:
    for column in ("smiles", "Drug", "drug"):
        if column in frame:
            return column
    raise ValueError("test split has no SMILES column")


def calculate_descriptors(smiles: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES at test row {index}")
        rows.append({name: float(function(molecule)) for name, function in DESCRIPTORS.items()})
    result = pd.DataFrame(rows)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("non-finite molecular descriptor")
    return result


def group_validity(
    metric: str,
    labels: np.ndarray,
    minimum_group_size: int,
    minimum_class_count: int,
) -> tuple[bool, str, int | None, float | None]:
    n = len(labels)
    if n < minimum_group_size:
        return False, "group_below_minimum_size", None, None
    if metric not in {"AUROC", "AUPRC"}:
        return True, "valid", None, None
    values, counts = np.unique(labels.astype(int), return_counts=True)
    positive_n = int(np.count_nonzero(labels == 1))
    prevalence = positive_n / n
    if len(values) != 2:
        return False, "classification_group_has_one_class", positive_n, prevalence
    if int(counts.min()) < minimum_class_count:
        return False, "classification_group_below_minimum_class_count", positive_n, prevalence
    return True, "valid", positive_n, prevalence


def task_subgroups(
    task: str,
    task_rows: pd.DataFrame,
    data_root: Path,
    reference_model: str,
    comparator_model: str,
    bootstrap_replicates: int,
    minimum_group_size: int,
    minimum_class_count: int,
    seed: int,
) -> list[dict[str, object]]:
    metric_values = task_rows.metric.unique().tolist()
    if len(metric_values) != 1:
        raise ValueError(f"multiple metrics for {task}: {metric_values}")
    metric = str(metric_values[0])
    labels = load_test_labels(data_root, task)
    test_frame = pd.read_csv(data_root / task / "test.csv")
    descriptors = calculate_descriptors(test_frame[smiles_column(test_frame)])
    reference_rows = task_rows[task_rows.model == reference_model]
    comparator_rows = task_rows[task_rows.model == comparator_model]
    if reference_rows.empty or comparator_rows.empty:
        raise ValueError(f"missing matched predictions for {task}")
    reference, reference_seeds = aggregate_predictions(reference_rows, labels)
    comparator, comparator_seeds = aggregate_predictions(comparator_rows, labels)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for descriptor in DESCRIPTORS:
        values = descriptors[descriptor].to_numpy(dtype=float)
        threshold = float(np.median(values))
        groups = {
            "low_or_equal_median": values <= threshold,
            "above_median": values > threshold,
        }
        for group_name, mask in groups.items():
            group_labels = labels[mask]
            valid, reason, positive_n, prevalence = group_validity(
                metric,
                group_labels,
                minimum_group_size,
                minimum_class_count,
            )
            row: dict[str, object] = {
                "task": task,
                "metric": metric,
                "descriptor": descriptor,
                "group": group_name,
                "threshold_task_median": threshold,
                "n_samples": int(mask.sum()),
                "positive_n": positive_n,
                "prevalence": prevalence,
                "metric_valid": valid,
                "validity_reason": reason,
                "reference_model": reference_model,
                "comparator_model": comparator_model,
                "reference_seeds": ";".join(map(str, reference_seeds)),
                "comparator_seeds": ";".join(map(str, comparator_seeds)),
                "bootstrap_replicates_requested": bootstrap_replicates,
            }
            if valid:
                result = paired_bootstrap(
                    group_labels,
                    reference[mask],
                    comparator[mask],
                    metric,
                    bootstrap_replicates,
                    rng,
                )
                row.update(result)
            rows.append(row)
    return rows


def task_subgroups_job(
    payload: tuple[str, pd.DataFrame, Path, str, str, int, int, int, int]
) -> list[dict[str, object]]:
    return task_subgroups(*payload)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.prediction_manifest)
    required = {"task", "metric", "model", "seed", "prediction_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be at least 1")

    grouped = [(str(task), rows.copy()) for task, rows in manifest.groupby("task", sort=True)]
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(grouped))
    payloads = [
        (
            task,
            task_rows,
            args.data_root,
            args.reference_model,
            args.comparator_model,
            args.bootstrap_replicates,
            args.minimum_group_size,
            args.minimum_class_count,
            int(child_seed.generate_state(1, dtype=np.uint32)[0]),
        )
        for (task, task_rows), child_seed in zip(grouped, child_seeds)
    ]
    if args.n_jobs == 1:
        nested_rows = map(task_subgroups_job, payloads)
        rows = [row for task_rows in nested_rows for row in task_rows]
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            nested_rows = executor.map(task_subgroups_job, payloads)
            rows = [row for task_rows in nested_rows for row in task_rows]

    results = pd.DataFrame(rows)
    valid_mask = results.metric_valid.astype(bool)
    results["p_value_bh_fdr"] = np.nan
    results["significant_fdr_0_05"] = False
    if valid_mask.any():
        adjusted = bh_adjust(
            results.loc[valid_mask, "p_value_two_sided"].to_numpy(dtype=float)
        )
        results.loc[valid_mask, "p_value_bh_fdr"] = adjusted
        results.loc[valid_mask, "significant_fdr_0_05"] = adjusted < 0.05

    args.out_root.mkdir(parents=True)
    results.to_csv(args.out_root / "Table_S22_subgroup_uncertainty.csv", index=False)
    (args.out_root / "provenance.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_manifest": str(args.prediction_manifest),
                "reference_model": args.reference_model,
                "comparator_model": args.comparator_model,
                "descriptors": list(DESCRIPTORS),
                "split_rule": "within-task official-test median; low group includes ties",
                "minimum_group_size": args.minimum_group_size,
                "minimum_class_count": args.minimum_class_count,
                "bootstrap_replicates": args.bootstrap_replicates,
                "parallel_jobs": args.n_jobs,
                "random_seed": args.seed,
                "randomization": "fixed child seed per sorted task; invariant to n_jobs",
                "delta_direction": "positive means reference model is better",
                "metric_direction": {
                    "higher_is_better": sorted(HIGHER_IS_BETTER),
                    "lower_is_better": ["MAE"],
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
