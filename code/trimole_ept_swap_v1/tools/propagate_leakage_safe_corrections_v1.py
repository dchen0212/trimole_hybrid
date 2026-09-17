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


def main() -> None:
    args = parse_args()
    observed = {path.name for path in args.table_dir.glob("*.csv")}
    missing = EXPECTED_TABLES - observed
    if missing:
        raise FileNotFoundError(f"missing expected tables: {sorted(missing)}")

    changed: dict[str, int] = {}
    for name in sorted(EXPECTED_TABLES):
        path = args.table_dir / name
        changed[name] = update_table(path)

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
