#!/usr/bin/env python3
"""Build a conservative task-level evidence gap matrix for the revision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LEAKAGE_TARGETS = {
    "clearance_hepatocyte_az",
    "cyp3a4_substrate_carbonmangels",
}


def read_by_task(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["task"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--recovered", type=Path)
    parser.add_argument("--v29-validation", type=Path)
    parser.add_argument("--leakage-validations", nargs="*", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    coverage = read_by_task(args.coverage)
    manifest = read_by_task(args.manifest)
    recovered = read_by_task(args.recovered) if args.recovered else {}
    v29_report = (
        json.loads(args.v29_validation.read_text()) if args.v29_validation else {}
    )
    if v29_report and v29_report.get("status") != "ok":
        raise ValueError("v29 materialization validation did not pass")
    v29_tasks = v29_report.get("tasks", {})
    leakage_reports = {
        report["target"]: report
        for path in args.leakage_validations
        for report in [json.loads(path.read_text())]
    }
    for task, report in leakage_reports.items():
        if not all(report.get("checks", {}).values()):
            raise ValueError(f"leakage-safe validation did not pass for {task}")
    if set(coverage) != set(manifest):
        raise ValueError("coverage and reproducibility manifest task sets differ")

    rows: list[dict[str, str]] = []
    for task in sorted(coverage):
        c = coverage[task]
        m = manifest[task]
        has_predictions = bool(c["primary_prediction_file"].strip())
        recovered_predictions = task in recovered
        reconstructed_v29 = task in v29_tasks
        leakage_safe = task in leakage_reports
        if task in LEAKAGE_TARGETS and not leakage_safe:
            priority = "P0_family_transfer_leakage_rerun_missing"
        elif leakage_safe:
            priority = "P1_replace_submitted_score_and_add_controls"
        elif not (has_predictions or recovered_predictions or reconstructed_v29):
            priority = "P1_reconstruct_final_sample_predictions"
        elif recovered_predictions or reconstructed_v29:
            priority = "P2_add_paired_statistics_and_controls"
        elif "matched_final_score_close" not in c["coverage_status"]:
            priority = "P1_reconcile_prediction_score_vs_reported_mean"
        else:
            priority = "P2_add_paired_statistics_and_controls"
        if leakage_safe:
            prediction_status = "leakage_safe_rerun_independently_verified"
            reconciliation = "five_seed_metrics_and_official_alignment_verified"
            evidence_source = "leakage_safe_rerun_validation"
            revision_mean = leakage_reports[task]["test_mean"]
            revision_std = leakage_reports[task]["test_std_ddof1"]
        elif reconstructed_v29:
            prediction_status = "reconstructed_verified_v29_formula"
            reconciliation = "all_five_materialized_seed_metrics_independently_verified"
            evidence_source = "v29_formula_predictions_v3_validation"
            revision_mean = v29_tasks[task]["test_mean"]
            revision_std = v29_tasks[task]["test_std"]
        elif recovered_predictions:
            prediction_status = "recovered_verified_archive"
            reconciliation = "all_five_seed_metrics_exactly_match_reported_summary"
            evidence_source = "fixed_endpoint_archive_recovery"
            revision_mean = c["final_score_mean"]
            revision_std = c["final_score_sd"]
        else:
            prediction_status = "available" if has_predictions else "missing"
            reconciliation = c["coverage_status"]
            evidence_source = "submitted_prediction_inventory" if has_predictions else ""
            revision_mean = c["final_score_mean"] if has_predictions else ""
            revision_std = c["final_score_sd"] if has_predictions else ""

        rows.append(
            {
                "task": task,
                "metric": c["metric"],
                "submitted_score_mean": c["final_score_mean"],
                "submitted_score_sd": c["final_score_sd"],
                "revision_score_mean": revision_mean,
                "revision_score_sd": revision_std,
                "n_runs_reported": m["n_runs"],
                "selected_candidate": m["selected_candidate"],
                "split_status": m["split_status"],
                "sample_level_final_prediction": prediction_status,
                "prediction_rows": (
                    c["official_test_rows"]
                    if recovered_predictions or reconstructed_v29 or leakage_safe
                    else c["n_prediction_rows"]
                ),
                "official_test_rows": c["official_test_rows"],
                "prediction_score_reconciliation": reconciliation,
                "evidence_source": evidence_source,
                "selected_final_model_artifact": "not_yet_verified",
                "paired_ci_or_test": "missing",
                "uniform_average_control": "missing",
                "standard_oof_stacking_control": "missing",
                "same_budget_automl_control": "missing",
                "revision_priority": priority,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    available = sum(row["sample_level_final_prediction"] != "missing" for row in rows)
    print(f"wrote {len(rows)} tasks; sample-level final predictions available for {available}")


if __name__ == "__main__":
    main()
