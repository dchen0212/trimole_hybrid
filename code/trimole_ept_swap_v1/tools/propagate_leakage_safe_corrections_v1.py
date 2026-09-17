#!/usr/bin/env python3
"""Propagate frozen leakage-safe endpoint results through legacy tables.

The submitted supplementary tables duplicated the two family-transfer scores
in several derived views. This script updates those views from one frozen
correction record and recomputes every dependent margin and ablation delta.
It intentionally fails if an expected table or endpoint row is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORRECTIONS = {
    "clearance_hepatocyte_az": {
        "dataset": "TDC.CL-Hepa",
        "mean": 0.2737101158416089,
        "std": 0.020235030266734705,
        "top1": 0.536,
        "candidate": "leakage_safe_fp_chemberta_kpgt_ept_xgb",
        "config": "Leakage-safe family-transfer XGB; FP+ChemBERTa+KPGT+EPT",
        "source": "results_strict/revision_20260917_clearance_hepatocyte_leakage_safe_v2/final_selected_test_summary.csv",
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "TDC.CYP3A4-S",
        "mean": 0.655628390596745,
        "std": 0.007590024136700193,
        "top1": 0.667,
        "candidate": "leakage_safe_fp_xgb",
        "config": "Leakage-safe family-transfer XGB; fingerprint features",
        "source": "results_strict/revision_20260917_cyp3a4_substrate_leakage_safe_v2/final_selected_test_summary.csv",
    },
}

EXPECTED_TABLES = {
    "Table_S1_TDC_ADMET22_benchmark.csv",
    "Table_S2_endpoint_recipe_and_variant_ledger.csv",
    "Table_S2c_final_endpoint_summary.csv",
    "Table_S3_frozen_reference_snapshot.csv",
    "Table_S3b_multibaseline_long.csv",
    "Table_S4_formal_ablation_long.csv",
    "Table_S4c_formal_ablation_delta_vs_full.csv",
    "Table_S4e_naive_mlp_late_fusion_control.csv",
    "Table_S5c_22task_plot_data_long.csv",
    "Table_S11_split_and_source_provenance_audit.csv",
    "Table_S13_reproducibility_manifest.csv",
    "Table_S14_frozen_reference_snapshot_audit.csv",
}

DERIVED_TABLES = {
    "Table_S2b_endpoint_configs_wide.csv",
    "Table_S2d_endpoint_overview_for_pdf.csv",
    "Table_S4b_formal_ablation_summary.csv",
    "Table_S4d_formal_ablation_heatmap_ready.csv",
    "Table_S5_ADMET_category_summary.csv",
    "Table_S5b_metric_type_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, required=True)
    return parser.parse_args()


def endpoint_mask(frame: pd.DataFrame, task: str, dataset: str) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if "task" in frame:
        mask |= frame["task"].astype(str).isin([task, dataset])
    if "Dataset" in frame:
        mask |= frame["Dataset"].astype(str).eq(dataset)
    if "dataset" in frame:
        mask |= frame["dataset"].astype(str).eq(dataset)
    return mask


def full_mask(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    result = mask.copy()
    if "variant" in frame:
        result &= frame["variant"].astype(str).eq("full_v36_final")
    if "is_trimole" in frame:
        result &= frame["is_trimole"].astype(bool)
    return result


def assign_if_present(
    frame: pd.DataFrame, mask: pd.Series, column: str, value: object
) -> None:
    if column in frame:
        frame.loc[mask, column] = value


def row_margin(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    score = pd.to_numeric(frame.loc[mask, "score_mean"])
    reference = pd.to_numeric(frame.loc[mask, "top1_ref"])
    if "direction" in frame:
        direction = frame.loc[mask, "direction"].astype(str)
        sign = np.where(
            direction.isin(["max", "higher_better"]), 1.0, -1.0
        )
        return pd.Series(sign * (score - reference), index=score.index)
    return score - reference


def update_table(path: Path) -> int:
    frame = pd.read_csv(path)
    changed = 0
    for task, correction in CORRECTIONS.items():
        mask = endpoint_mask(frame, task, str(correction["dataset"]))
        if not mask.any():
            raise ValueError(f"{path.name}: missing endpoint {task}")
        selected = full_mask(frame, mask)
        if not selected.any():
            raise ValueError(f"{path.name}: missing full result for {task}")

        mean = float(correction["mean"])
        std = float(correction["std"])
        top1 = float(correction["top1"])
        margin = mean - top1
        score_text = f"{mean:.6f} ± {std:.6f}"
        compact_score = f"{mean:.6f}±{std:.6f}"

        assign_if_present(frame, selected, "score_mean", mean)
        assign_if_present(frame, selected, "score_std", std)
        assign_if_present(frame, selected, "n_runs", 5)
        assign_if_present(frame, selected, "Trimole-Hybrid", compact_score)
        assign_if_present(frame, selected, "score_text", score_text)
        assign_if_present(frame, selected, "Margin", margin)
        assign_if_present(frame, selected, "margin", margin)
        assign_if_present(frame, selected, "margin_percent_points", margin * 100.0)
        assign_if_present(frame, selected, "direction_normalized_margin", margin)
        assign_if_present(frame, selected, "diagnostic_margin_ge_0", False)
        assign_if_present(frame, selected, "Rank", "Descriptive")
        assign_if_present(frame, selected, "rank_num", np.nan)
        assign_if_present(frame, selected, "top1_flag", 0)
        assign_if_present(frame, selected, "top3_flag", 0)
        assign_if_present(frame, selected, "top5_flag", 0)
        assign_if_present(frame, selected, "top10_flag", 0)
        assign_if_present(frame, selected, "top1_margin_flag", 0)
        assign_if_present(frame, selected, "selected_candidate", correction["candidate"])
        assign_if_present(frame, selected, "Endpoint config", correction["config"])
        assign_if_present(frame, selected, "endpoint_config", correction["config"])
        assign_if_present(frame, selected, "selected_endpoint_config", correction["config"])
        assign_if_present(frame, selected, "selected_endpoint_config_full", correction["config"])
        assign_if_present(frame, selected, "test_score_source_file", correction["source"])
        # In S14, `source` describes the frozen public comparator rather than
        # the Trimole prediction provenance and must remain `TDC leaderboard`.
        if path.name != "Table_S14_frozen_reference_snapshot_audit.csv":
            assign_if_present(frame, selected, "source", correction["source"])
        assign_if_present(frame, selected, "source_provenance", "leakage_safe_family_transfer")
        assign_if_present(frame, selected, "split_status", "leakage_audited_official_split")
        assign_if_present(
            frame,
            selected,
            "selection_protocol",
            "Validation-only selection after stage-specific cross-task identity filtering; official test labels used only for final scoring.",
        )
        assign_if_present(
            frame,
            selected,
            "evidence",
            f"Frozen leakage-safe five-seed rerun: {correction['source']}",
        )
        assign_if_present(
            frame,
            selected,
            "source_evidence",
            f"Frozen leakage-safe five-seed rerun: {correction['source']}",
        )
        assign_if_present(
            frame,
            selected,
            "paper_use_recommendation",
            "Use corrected leakage-safe result; submitted family-transfer score is invalidated.",
        )
        assign_if_present(frame, selected, "uses_chemistry_sidecar", True)
        assign_if_present(
            frame,
            selected,
            "uses_ept_or_3d",
            task == "clearance_hepatocyte_az",
        )
        assign_if_present(frame, selected, "uses_prediction_level_ensemble", False)
        assign_if_present(frame, selected, "uses_seedbag", True)

        if "direction_normalized_margin" in frame and "score_mean" in frame:
            frame.loc[mask, "direction_normalized_margin"] = row_margin(frame, mask)
        if "full_margin" in frame:
            frame.loc[mask, "full_margin"] = margin
        if "delta_margin_vs_full" in frame and "direction_normalized_margin" in frame:
            frame.loc[mask, "delta_margin_vs_full"] = (
                pd.to_numeric(frame.loc[mask, "direction_normalized_margin"]) - margin
            )
        changed += int(selected.sum())

    frame.to_csv(path, index=False)
    return changed


def rebuild_derived_tables(table_dir: Path) -> None:
    ledger = pd.read_csv(table_dir / "Table_S2_endpoint_recipe_and_variant_ledger.csv")
    variant_order = [
        "full_v36_final",
        "no_task_adaptive_selection",
        "no_chemistry_sidecar",
        "no_ept_or_no_3d",
        "no_prediction_level_ensemble",
        "no_seedbag_single_seed",
    ]
    configs = ledger.pivot(
        index=["task", "metric"],
        columns="variant",
        values="selected_endpoint_config",
    ).reset_index()
    configs = configs[["task", "metric", *variant_order]]
    configs.to_csv(table_dir / "Table_S2b_endpoint_configs_wide.csv", index=False)

    overview_path = table_dir / "Table_S2d_endpoint_overview_for_pdf.csv"
    overview = pd.read_csv(overview_path)
    for task, correction in CORRECTIONS.items():
        mask = overview["Dataset"].eq(correction["dataset"])
        if mask.sum() != 1:
            raise ValueError(f"{overview_path.name}: expected one row for {task}")
        overview.loc[mask, "Frozen endpoint family"] = "Leakage-safe family-transfer XGBoost"
        overview.loc[mask, "Aggregation"] = "five-seed mean"
        overview.loc[mask, "Chem"] = True
        overview.loc[mask, "Seq"] = task == "clearance_hepatocyte_az"
        overview.loc[mask, "Graph"] = task == "clearance_hepatocyte_az"
        overview.loc[mask, "EPT_3D"] = task == "clearance_hepatocyte_az"
        overview.loc[mask, "Blend"] = False
        overview.loc[mask, "Seed_or_fold_bagging"] = True
        if task == "clearance_hepatocyte_az":
            streams = "chemistry priors, sequence embeddings, graph embeddings, EPT/3D"
            flags = "Chem, Seq, Graph, EPT/3D, Seedbag"
        else:
            streams = "chemistry priors"
            flags = "Chem, Seedbag"
        overview.loc[mask, "Main evidence streams"] = streams
        overview.loc[mask, "Component flags"] = flags
    overview.to_csv(overview_path, index=False)

    ablation = pd.read_csv(table_dir / "Table_S4_formal_ablation_long.csv")
    full_positive = set(
        ablation.loc[
            ablation.variant.eq("full_v36_final")
            & (ablation.direction_normalized_margin >= 0),
            "task",
        ]
    )
    summary_rows = []
    for variant in variant_order:
        group = ablation[ablation.variant.eq(variant)].copy()
        available = group.score_mean.notna()
        retained = group.task.isin(full_positive) & (
            group.direction_normalized_margin >= 0
        )
        summary_rows.append(
            {
                "variant": variant,
                "available_tasks": int(available.sum()),
                "missing_tasks": int((~available).sum()),
                "mean_margin": group.direction_normalized_margin.mean(),
                "median_margin": group.direction_normalized_margin.median(),
                "mean_delta_vs_full": group.delta_margin_vs_full.mean(),
                "median_delta_vs_full": group.delta_margin_vs_full.median(),
                "mean_abs_delta_vs_full": group.delta_margin_vs_full.abs().mean(),
                "diagnostic_margin_ge_0_count": int(
                    (group.direction_normalized_margin >= 0).sum()
                ),
                "full_top1_retention_count": int(retained.sum()),
                "n_tasks_with_5run_mean_std": int(
                    ((group.n_runs >= 5) & group.score_std.notna()).sum()
                ),
                "n_tasks_single_seed_only": int((group.n_runs == 1).sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        table_dir / "Table_S4b_formal_ablation_summary.csv", index=False
    )
    ablation[
        [
            "task",
            "variant",
            "delta_margin_vs_full",
            "direction_normalized_margin",
            "selected_candidate",
        ]
    ].to_csv(table_dir / "Table_S4d_formal_ablation_heatmap_ready.csv", index=False)

    plot_data = pd.read_csv(table_dir / "Table_S5c_22task_plot_data_long.csv")
    for group_column, order_column, output_name in (
        ("Category", "category_order", "Table_S5_ADMET_category_summary.csv"),
        ("Metric", "metric_order", "Table_S5b_metric_type_summary.csv"),
    ):
        rows = []
        for group_name, group in plot_data.groupby(group_column, sort=False):
            rows.append(
                {
                    group_column: group_name,
                    "n_tasks": len(group),
                    "total_size": int(group.Size.sum()),
                    "mean_margin": group.margin.mean(),
                    "median_margin": group.margin.median(),
                    "mean_margin_percent_points": group.margin_percent_points.mean(),
                    "median_margin_percent_points": group.margin_percent_points.median(),
                    "top1_count": int(group.top1_flag.sum()),
                    "top3_count": int(group.top3_flag.sum()),
                    "top5_count": int(group.top5_flag.sum()),
                    "top10_count": int(group.top10_flag.sum()),
                    "mean_rank": group.rank_num.mean(),
                    "median_rank": group.rank_num.median(),
                    "top1_rate": group.top1_flag.mean(),
                    "top3_rate": group.top3_flag.mean(),
                    "top5_rate": group.top5_flag.mean(),
                    "top10_rate": group.top10_flag.mean(),
                    order_column: (
                        int(group[order_column].iloc[0])
                        if order_column in group
                        else {"AUROC": 1, "AUPRC": 2, "MAE": 3, "Spearman": 4}[
                            str(group_name)
                        ]
                    ),
                }
            )
        pd.DataFrame(rows).sort_values(order_column).to_csv(
            table_dir / output_name, index=False
        )


def main() -> None:
    args = parse_args()
    observed = {path.name for path in args.table_dir.glob("*.csv")}
    missing = (EXPECTED_TABLES | DERIVED_TABLES) - observed
    if missing:
        raise FileNotFoundError(f"missing expected tables: {sorted(missing)}")

    changed: dict[str, int] = {}
    for name in sorted(EXPECTED_TABLES):
        path = args.table_dir / name
        changed[name] = update_table(path)
    rebuild_derived_tables(args.table_dir)

    stale = []
    for path in args.table_dir.glob("*.csv"):
        text = path.read_text()
        if "0.552040" in text or "0.725249" in text:
            stale.append(path.name)
    if stale:
        raise RuntimeError(f"stale invalid scores remain in: {sorted(stale)}")
    print(changed)


if __name__ == "__main__":
    main()
