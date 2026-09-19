#!/usr/bin/env python3
"""Promote the predeclared test-blind target-only control to the main two-task table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from propagate_family_transfer_v4 import (
    load_result,
    update_ablation_summary,
    update_dataset_tables,
    update_group_summaries,
    update_provenance_tables,
    update_s23,
    update_single_task_rows,
    update_task_variant_table,
)

RUNS = {
    "clearance_hepatocyte_az": "revision_20260919_clearance_target_only_v1",
    "cyp3a4_substrate_carbonmangels": "revision_20260919_cyp3a4_target_only_v1",
}


def flag_incomparable_historical_analyses(table_dir: Path) -> None:
    excluded = set(RUNS)
    for filename in (
        "Table_S4_formal_ablation_long.csv",
        "Table_S4c_formal_ablation_delta_vs_full.csv",
        "Table_S4e_naive_mlp_late_fusion_control.csv",
        "Table_S23_ablation_selection_stability.csv",
    ):
        path = table_dir / filename
        frame = pd.read_csv(path)
        frame["primary_ablation_comparable"] = ~frame["task"].isin(excluded)
        frame.to_csv(path, index=False)

    source = pd.read_csv(table_dir / "Table_S4_formal_ablation_long.csv")
    source = source[source.primary_ablation_comparable].copy()
    full = source[source.variant.eq("full_v36_final")].set_index("task")
    rows = []
    for variant, group in source.groupby("variant", sort=False):
        available = group[group.score_mean.notna()].copy()
        full_positive = full.loc[available.task, "direction_normalized_margin"].to_numpy() >= 0
        current_positive = available.direction_normalized_margin.to_numpy() >= 0
        rows.append({
            "variant": variant,
            "available_tasks": len(available),
            "missing_tasks": int(group.score_mean.isna().sum()),
            "mean_margin": available.direction_normalized_margin.mean(),
            "median_margin": available.direction_normalized_margin.median(),
            "mean_delta_vs_full": available.delta_margin_vs_full.mean(),
            "median_delta_vs_full": available.delta_margin_vs_full.median(),
            "mean_abs_delta_vs_full": available.delta_margin_vs_full.abs().mean(),
            "diagnostic_margin_ge_0_count": int(current_positive.sum()),
            "full_top1_retention_count": int((current_positive & full_positive).sum()),
            "n_tasks_with_5run_mean_std": int(((available.n_runs >= 5) & available.score_std.notna()).sum()),
            "n_tasks_single_seed_only": int((available.n_runs == 1).sum()),
        })
    pd.DataFrame(rows).to_csv(table_dir / "Table_S4b_formal_ablation_summary.csv", index=False)

    for filename in ("Table_S22_uncertainty_and_subgroups.csv", "Table_S22b_subgroup_uncertainty.csv"):
        path = table_dir / filename
        frame = pd.read_csv(path)
        frame["primary_subgroup_comparable"] = ~frame.task.isin(excluded)
        frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--clearance", required=True, type=Path)
    parser.add_argument("--cyp3a4", required=True, type=Path)
    args = parser.parse_args()
    results = {}
    for task, path in (("clearance_hepatocyte_az", args.clearance), ("cyp3a4_substrate_carbonmangels", args.cyp3a4)):
        result = load_result(path, task, RUNS[task])
        result.update(
            source_kind="test_blind_target_only_five_seed",
            split_status="test_blind_target_only_official_split",
            source_evidence="Predeclared target-only control; no cross-task source rows and no target test identities in training-pool construction: " + str(result["source"]),
            selection_protocol="Target-only training pool fixed without target test molecule identities; train-only fitting, validation-only configuration selection, five-seed train+validation refit, and test-only scoring.",
        )
        results[task] = result

    update_dataset_tables(args.table_dir, results)
    for filename in (
        "Table_S2_endpoint_recipe_and_variant_ledger.csv",
        "Table_S4_formal_ablation_long.csv",
        "Table_S4c_formal_ablation_delta_vs_full.csv",
        "Table_S4e_naive_mlp_late_fusion_control.csv",
    ):
        update_task_variant_table(args.table_dir / filename, results)
    update_single_task_rows(args.table_dir / "Table_S2c_final_endpoint_summary.csv", results)
    update_provenance_tables(args.table_dir, results)
    update_group_summaries(args.table_dir)
    update_ablation_summary(args.table_dir)
    update_s23(args.table_dir)
    flag_incomparable_historical_analyses(args.table_dir)
    for task, result in results.items():
        print(task, result["score_mean"], result["score_std"])


if __name__ == "__main__":
    main()
