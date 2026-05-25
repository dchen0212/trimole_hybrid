# Supplementary Tables for Trimole-Hybrid ADMET

This directory contains audit-oriented supplementary tables generated from the local Trimole-Hybrid manuscript artifacts.
The tables are intended to support reproducibility, selection-protocol auditing, case-study provenance, and leakage checks.

## Tables
- S1: 22-task benchmark table used for the main benchmark comparison. File: `Table_S1_TDC_ADMET22_benchmark.csv`
- S10: Not-promoted candidate audit documenting cases that were not elevated to exact main claims. File: `Table_S10_not_promoted_case_audit.csv`
- S11: Main-result official-split and source-provenance audit. File: `Table_S11_split_and_source_provenance_audit.csv`
- S12: External case alignment feasibility and selected external search snapshot. File: `Table_S12_external_case_alignment_feasibility.csv`
- S13: One-row-per-task reproducibility manifest linking endpoint recipes, seeds, split provenance, and source files. File: `Table_S13_reproducibility_manifest.csv`
- S14: Frozen TDC reference snapshot with public reference model/source metadata where available. File: `Table_S14_frozen_reference_snapshot_audit.csv`
- S15: Validation-only endpoint-selection evidence summary. File: `Table_S15_validation_only_selection_evidence.csv`
- S16: Metric direction, margin formula, and endpoint scale guide. File: `Table_S16_metric_direction_and_unit_guide.csv`
- S17: Case-study decision-boundary table for promoted and non-promoted molecule-level examples. File: `Table_S17_case_study_decision_boundary.csv`
- S18: Software and execution-environment manifest read from the server environments. File: `Table_S18_software_environment_manifest.csv`
- S2: Selected endpoint recipes and full ablation candidate ledger. S2d provides the compact endpoint overview used in the PDF supplement. File: `Table_S2_endpoint_recipe_and_variant_ledger.csv`
- S3: Frozen leaderboard/reference and multibaseline comparison data. File: `Table_S3_frozen_reference_snapshot.csv`
- S4: Formal ablation summary, long-form ablation scores, deltas, and heatmap-ready data. File: `Table_S4_formal_ablation_long.csv`
- S5: Figure 2 stratified analysis data by ADMET category and metric type. File: `Table_S5_ADMET_category_summary.csv`
- S6: Extended molecule-level official-test case examples retained as supplementary audit material. File: `Table_S6_extended_official_test_case_examples.csv`
- S7: Main Figure 4 two-case sensitivity audit with exact/proxy status. File: `Table_S7_Figure4_two_case_sensitivity_audit.csv`
- S8: External molecule leakage audit, including P-gp split-overlap summary. File: `Table_S8_external_candidate_leakage_audit.csv`
- S9: P-gp matched-replacement sweep supporting the Deutivacaftor case. File: `Table_S9_Pgp_matched_replacement_sweep.csv`

The workbook `Supplementary_Tables_S1_S18.xlsx` contains all tables as separate sheets.
Large source CSV files are also exported individually for easier inspection.
