#!/usr/bin/env python3
"""Propagate target-wide leakage-safe family-transfer reruns into paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = {
    "clearance_hepatocyte_az": {
        "dataset": "TDC.CL-Hepa",
        "result_dir": "revision_20260918_clearance_hepatocyte_presplit_similarity_v4",
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "TDC.CYP3A4-S",
        "result_dir": "revision_20260918_cyp3a4_substrate_presplit_similarity_v4",
    },
}

FEATURE_LABELS = {
    "fp": ("targetwide_safe_fp_xgb", "Target-wide leakage-safe family-transfer XGBoost; Morgan fingerprint"),
    "fp_kpgt": (
        "targetwide_safe_fp_kpgt_xgb",
        "Target-wide leakage-safe family-transfer XGBoost; Morgan fingerprint + KPGT",
    ),
    "fp_chemberta_kpgt_ept": (
        "targetwide_safe_fp_chemberta_kpgt_ept_xgb",
        "Target-wide leakage-safe family-transfer XGBoost; Morgan fingerprint + ChemBERTa + KPGT + EPT",
    ),
}

PROTOCOL = (
    "Validation-only selection after one label-free target-universe exact-identity and "
    "Morgan-Tanimoto >=0.90 source filter fixed before split-specific fitting; official "
    "test labels used only for final scoring."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--clearance", required=True, type=Path)
    parser.add_argument("--cyp3a4", required=True, type=Path)
    return parser.parse_args()


def load_result(root: Path, task: str, result_dir: str) -> dict[str, object]:
    summary = pd.read_csv(root / "final_selected_test_summary.csv").iloc[0]
    selected = pd.read_csv(root / "selected_candidates.csv").iloc[0]
    feature_set = str(selected["feature_set"])
    candidate, endpoint = FEATURE_LABELS[feature_set]
    return {
        "task": task,
        "dataset": TASKS[task]["dataset"],
        "result_dir": result_dir,
        "score_mean": float(summary["test_mean"]),
        "score_std": float(summary["test_std"]),
        "score_ensemble": float(summary["test_ensemble"]),
        "n_runs": 5,
        "selected_candidate": candidate,
        "selected_endpoint_config": endpoint,
        "source": f"results_strict/{result_dir}/final_selected_test_summary.csv",
        "selection_protocol": PROTOCOL,
        "uses_chemistry_sidecar": True,
        "uses_ept_or_3d": feature_set == "fp_chemberta_kpgt_ept",
        "uses_prediction_level_ensemble": False,
        "uses_seedbag": True,
    }


def direction_margin(score: pd.Series, reference: pd.Series, direction: pd.Series) -> pd.Series:
    sign = direction.astype(str).str.lower().map(
        {"max": 1.0, "higher_better": 1.0, "min": -1.0, "lower_better": -1.0}
    )
    if sign.isna().any():
        raise ValueError(f"unknown metric direction: {direction[sign.isna()].unique()}")
    return sign * (score.astype(float) - reference.astype(float))


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def update_task_variant_table(path: Path, results: dict[str, dict[str, object]]) -> None:
    frame = pd.read_csv(path)
    for task, result in results.items():
        task_mask = frame["task"].eq(task)
        full_mask = task_mask & frame["variant"].eq("full_v36_final")
        if full_mask.sum() != 1:
            raise ValueError(f"{path.name}: expected one full row for {task}")
        for column in ("n_runs", "score_mean", "score_std", "selected_candidate"):
            if column in frame:
                frame.loc[full_mask, column] = result[column]
        if "selected_endpoint_config" in frame:
            frame.loc[full_mask, "selected_endpoint_config"] = result["selected_endpoint_config"]
        if "test_score_source_file" in frame:
            frame.loc[full_mask, "test_score_source_file"] = result["source"]
        if "selection_protocol" in frame:
            frame.loc[full_mask, "selection_protocol"] = result["selection_protocol"]
        for column in (
            "uses_chemistry_sidecar",
            "uses_ept_or_3d",
            "uses_prediction_level_ensemble",
            "uses_seedbag",
        ):
            if column in frame:
                frame.loc[full_mask, column] = result[column]
        if "direction_normalized_margin" in frame:
            frame.loc[task_mask, "direction_normalized_margin"] = direction_margin(
                frame.loc[task_mask, "score_mean"],
                frame.loc[task_mask, "top1_ref"],
                frame.loc[task_mask, "direction"],
            )
        full_margin = float(frame.loc[full_mask, "direction_normalized_margin"].iloc[0])
        if "full_margin" in frame:
            frame.loc[task_mask, "full_margin"] = full_margin
        if "delta_margin_vs_full" in frame:
            frame.loc[task_mask, "delta_margin_vs_full"] = (
                frame.loc[task_mask, "direction_normalized_margin"] - full_margin
            )
        if "diagnostic_margin_ge_0" in frame:
            frame.loc[task_mask, "diagnostic_margin_ge_0"] = (
                frame.loc[task_mask, "direction_normalized_margin"] >= 0
            )
    write(frame, path)


def update_single_task_rows(path: Path, results: dict[str, dict[str, object]]) -> None:
    frame = pd.read_csv(path)
    for task, result in results.items():
        mask = frame["task"].eq(task)
        if mask.sum() != 1:
            raise ValueError(f"{path.name}: expected one row for {task}")
        for column in ("n_runs", "score_mean", "score_std", "selected_candidate"):
            if column in frame:
                frame.loc[mask, column] = result[column]
        mappings = {
            "selected_endpoint_config": "selected_endpoint_config",
            "selected_endpoint_config_full": "selected_endpoint_config",
            "test_score_source_file": "source",
            "selection_protocol": "selection_protocol",
            "uses_chemistry_sidecar": "uses_chemistry_sidecar",
            "uses_ept_or_3d": "uses_ept_or_3d",
            "uses_prediction_level_ensemble": "uses_prediction_level_ensemble",
            "uses_seedbag": "uses_seedbag",
        }
        for column, key in mappings.items():
            if column in frame:
                frame.loc[mask, column] = result[key]
        if "direction_normalized_margin" in frame:
            frame.loc[mask, "direction_normalized_margin"] = direction_margin(
                frame.loc[mask, "score_mean"],
                frame.loc[mask, "top1_ref"],
                pd.Series(["max"] * int(mask.sum()), index=frame.index[mask]),
            )
        if "source_kind" in frame:
            frame.loc[mask, "source_kind"] = result.get("source_kind", "targetwide_similarity_safe_family_transfer")
        if "source_provenance" in frame:
            frame.loc[mask, "source_provenance"] = result.get("source_kind", "targetwide_similarity_safe_family_transfer")
        if "source_evidence" in frame:
            frame.loc[mask, "source_evidence"] = str(result.get("source_evidence", "Frozen target-wide and similarity-filtered five-seed rerun: " + str(result["source"])))
    write(frame, path)


def update_dataset_tables(table_dir: Path, results: dict[str, dict[str, object]]) -> None:
    for filename in (
        "Table_S1_TDC_ADMET22_benchmark.csv",
        "Table_S3_frozen_reference_snapshot.csv",
        "Table_S14_frozen_reference_snapshot_audit.csv",
    ):
        path = table_dir / filename
        frame = pd.read_csv(path)
        for result in results.values():
            mask = frame["Dataset"].eq(result["dataset"])
            if mask.sum() != 1:
                raise ValueError(f"{filename}: expected one row for {result['dataset']}")
            score = float(result["score_mean"])
            std = float(result["score_std"])
            reference = float(frame.loc[mask, "Top-1 ref."].iloc[0])
            margin = score - reference
            frame.loc[mask, "Trimole-Hybrid"] = f"{score:.6f}±{std:.6f}"
            frame.loc[mask, "Margin"] = margin
            if "Endpoint config" in frame:
                frame.loc[mask, "Endpoint config"] = result["selected_endpoint_config"]
            if "score_mean" in frame:
                frame.loc[mask, "score_mean"] = score
            if "score_std" in frame:
                frame.loc[mask, "score_std"] = std
        write(frame, path)

    path = table_dir / "Table_S3b_multibaseline_long.csv"
    frame = pd.read_csv(path)
    for result in results.values():
        mask = frame["task"].eq(result["dataset"]) & frame["is_trimole"].astype(bool)
        if mask.sum() != 1:
            raise ValueError(f"{path.name}: expected one Trimole row for {result['dataset']}")
        frame.loc[mask, ["score_mean", "score_std"]] = [result["score_mean"], result["score_std"]]
        frame.loc[mask, "source"] = result["source"]
        frame.loc[mask, "endpoint_config"] = result["selected_endpoint_config"]
    write(frame, path)

    path = table_dir / "Table_S5c_22task_plot_data_long.csv"
    frame = pd.read_csv(path)
    for result in results.values():
        mask = frame["Dataset"].eq(result["dataset"])
        if mask.sum() != 1:
            raise ValueError(f"{path.name}: expected one row for {result['dataset']}")
        score = float(result["score_mean"])
        std = float(result["score_std"])
        reference = float(frame.loc[mask, "top1_ref"].iloc[0])
        frame.loc[mask, "score_mean"] = score
        frame.loc[mask, "score_std"] = std
        frame.loc[mask, "margin"] = score - reference
        frame.loc[mask, "margin_percent_points"] = 100 * (score - reference)
        frame.loc[mask, "Endpoint config"] = result["selected_endpoint_config"]
    write(frame, path)


def update_provenance_tables(table_dir: Path, results: dict[str, dict[str, object]]) -> None:
    path = table_dir / "Table_S11_split_and_source_provenance_audit.csv"
    frame = pd.read_csv(path)
    for task, result in results.items():
        mask = frame["task"].eq(task)
        frame.loc[mask, "source"] = result["source"]
        frame.loc[mask, "score_text"] = f"{result['score_mean']:.6f} ± {result['score_std']:.6f}"
        frame.loc[mask, "split_status"] = result.get("split_status", "targetwide_similarity_audited_official_split")
        frame.loc[mask, "evidence"] = str(result.get("source_evidence", "Frozen target-wide and similarity-filtered five-seed rerun: " + str(result["source"])))
    write(frame, path)

    update_single_task_rows(table_dir / "Table_S13_reproducibility_manifest.csv", results)
    update_single_task_rows(table_dir / "trimole_score_summary_frozen.csv", results)


def update_group_summaries(table_dir: Path) -> None:
    plot = pd.read_csv(table_dir / "Table_S5c_22task_plot_data_long.csv")
    for group_col, order_col, filename in (
        ("Category", "category_order", "Table_S5_ADMET_category_summary.csv"),
        ("Metric", "metric_order", "Table_S5b_metric_type_summary.csv"),
    ):
        existing = pd.read_csv(table_dir / filename).set_index(group_col)
        rows = []
        for name, group in plot.groupby(group_col, sort=False):
            valid_rank = pd.to_numeric(group["rank_num"], errors="coerce")
            rows.append(
                {
                    group_col: name,
                    "n_tasks": len(group),
                    "total_size": pd.to_numeric(group["Size"].astype(str).str.replace(",", "", regex=False)).sum(),
                    "mean_margin": group["margin"].mean(),
                    "median_margin": group["margin"].median(),
                    "mean_margin_percent_points": group["margin_percent_points"].mean(),
                    "median_margin_percent_points": group["margin_percent_points"].median(),
                    "top1_count": int(group["top1_flag"].sum()),
                    "top3_count": int(group["top3_flag"].sum()),
                    "top5_count": int(group["top5_flag"].sum()),
                    "top10_count": int(group["top10_flag"].sum()),
                    "mean_rank": valid_rank.mean(),
                    "median_rank": valid_rank.median(),
                    "top1_rate": group["top1_flag"].mean(),
                    "top3_rate": group["top3_flag"].mean(),
                    "top5_rate": group["top5_flag"].mean(),
                    "top10_rate": group["top10_flag"].mean(),
                    order_col: existing.loc[name, order_col],
                }
            )
        write(pd.DataFrame(rows).sort_values(order_col), table_dir / filename)


def update_ablation_summary(table_dir: Path) -> None:
    frame = pd.read_csv(table_dir / "Table_S4_formal_ablation_long.csv")
    rows = []
    full = frame[frame["variant"].eq("full_v36_final")].set_index("task")
    for variant, group in frame.groupby("variant", sort=False):
        available = group[group["score_mean"].notna()].copy()
        full_positive = full.loc[available["task"], "direction_normalized_margin"].to_numpy() >= 0
        current_positive = available["direction_normalized_margin"].to_numpy() >= 0
        rows.append(
            {
                "variant": variant,
                "available_tasks": len(available),
                "missing_tasks": int(group["score_mean"].isna().sum()),
                "mean_margin": available["direction_normalized_margin"].mean(),
                "median_margin": available["direction_normalized_margin"].median(),
                "mean_delta_vs_full": available["delta_margin_vs_full"].mean(),
                "median_delta_vs_full": available["delta_margin_vs_full"].median(),
                "mean_abs_delta_vs_full": available["delta_margin_vs_full"].abs().mean(),
                "diagnostic_margin_ge_0_count": int(current_positive.sum()),
                "full_top1_retention_count": int((current_positive & full_positive).sum()),
                "n_tasks_with_5run_mean_std": int(((available["n_runs"] >= 5) & available["score_std"].notna()).sum()),
                "n_tasks_single_seed_only": int((available["n_runs"] == 1).sum()),
            }
        )
    write(pd.DataFrame(rows), table_dir / "Table_S4b_formal_ablation_summary.csv")


def update_s23(table_dir: Path) -> None:
    source = pd.read_csv(table_dir / "Table_S4_formal_ablation_long.csv")
    rows: list[dict[str, object]] = []
    for task, group in source.groupby("task", sort=True):
        group = group[group["score_mean"].notna()].copy()
        ascending = str(group["direction"].iloc[0]).lower() == "min"
        group["task_rank"] = group["score_mean"].rank(method="average", ascending=ascending)
        full_rank = float(group.loc[group["variant"].eq("full_v36_final"), "task_rank"].iloc[0])
        denominator = max(len(group) - 1, 1)
        for row in group.to_dict("records"):
            rows.append(
                {
                    "record_type": "ablation_rank_loss",
                    **row,
                    "full_variant": "full_v36_final",
                    "full_rank": full_rank,
                    "n_ranked_variants": len(group),
                    "normalized_rank_loss": (float(row["task_rank"]) - full_rank) / denominator,
                }
            )
    old = pd.read_csv(table_dir / "Table_S23_ablation_selection_stability.csv")
    stability = old[old["record_type"].eq("validation_selection_stability")].copy()
    combined = pd.concat([pd.DataFrame(rows), stability], ignore_index=True, sort=False)
    first = ["record_type", "task", "metric"]
    write(combined[first + [column for column in combined.columns if column not in first]], table_dir / "Table_S23_ablation_selection_stability.csv")


def mark_superseded_sensitivities(table_dir: Path) -> None:
    path = table_dir / "Table_S20_cross_task_leakage_audit.csv"
    frame = pd.read_csv(path)
    mask = frame["record_type"].eq("filter_and_corrected_score")
    frame.loc[mask, "record_type"] = "superseded_stage_specific_sensitivity"
    frame.loc[mask, "audit_interpretation"] = frame.loc[mask, "audit_interpretation"].map(
        lambda value: "SUPERSEDED exact-only, target-split-specific sensitivity; retained for audit trail and not used as final evidence. " + str(value)
    )
    write(frame, path)

    for filename in (
        "Table_S21_controlled_baselines_automl.csv",
        "Table_S21b_controlled_baselines_summary.csv",
    ):
        path = table_dir / filename
        frame = pd.read_csv(path)
        automl = frame["model"].eq("flaml_automl")
        frame.loc[automl, "source_family"] = "historical_three_view_automl_sensitivity"
        trimole = frame["model"].eq("trimole_hybrid")
        frame.loc[trimole, "source_family"] = "superseded_historical_taskwise_sensitivity"
        frame.loc[trimole, "prediction_evidence_note"] = (
            "archived five-seed prediction arrays retained as sensitivity evidence; "
            "superseded for strict fairness by S24g-S24j"
        )
        write(frame, path)


def main() -> None:
    args = parse_args()
    results = {
        "clearance_hepatocyte_az": load_result(
            args.clearance,
            "clearance_hepatocyte_az",
            str(TASKS["clearance_hepatocyte_az"]["result_dir"]),
        ),
        "cyp3a4_substrate_carbonmangels": load_result(
            args.cyp3a4,
            "cyp3a4_substrate_carbonmangels",
            str(TASKS["cyp3a4_substrate_carbonmangels"]["result_dir"]),
        ),
    }
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
    mark_superseded_sensitivities(args.table_dir)
    print(pd.DataFrame(results.values()).to_string(index=False))


if __name__ == "__main__":
    main()
