# Supplementary data

[Download the complete supplementary data](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.28/Supplementary_Data.zip). This download contains all 203 supporting files, including run scores, model settings, row identifiers, protocols and the integrity manifest.

The paper uses descriptive references to these records. After extracting the download, use this index to find the source files. Original filenames and experiment identifiers are retained within the download.

| Paper item | Contents | Location inside the data download |
|---|---|---|
| Supplementary Table S1 | Dataset sources, endpoint definitions and coverage | `historical_tables/Table_S19_dataset_provenance.csv` |
| Supplementary Table S2 | Task-specific model components | `historical_tables/Table_S2d_endpoint_overview_for_pdf.csv`<br>`historical_tables/Table_S2_endpoint_recipe_and_variant_ledger.csv`<br>`historical_tables/Table_S2c_final_endpoint_summary.csv`<br>`current_tables/Table_S2_task_configurations.csv` |
| Supplementary Table S3 | Trimole-Hybrid, MiniMol and Chemprop across 22 tasks (mean +/- SD) | `complete_results/Table_S31_external22_long.csv` |
| Supplementary Table S4 | Shared-pool comparison with two FLAML search budgets (mean +/- population SD) | `complete_results/Table_S34_recovered_pool_summary.csv`<br>`complete_results/Table_S34a_recovered_pool_runs.csv`<br>`complete_results/Table_S34b_pool_and_rows.json`<br>`complete_results/Table_S34b_reporting_metadata.json`<br>`complete_results/Table_S34c_search_budget_runs.csv`<br>`revision_budget_checks/Table_S4_complete_eight_arms.csv`<br>`revision_budget_checks/SEARCH_BUDGET_RECORDS_180s.csv` |
| Supplementary Table S4a | FLAML: 32 trials, 60 s, 120 s and 180 s (fixed nine-predictor pool) | `revision_budget_checks/Table_S4a_four_settings_by_task.csv`<br>`revision_budget_checks/Table_S4a_four_settings_summary.csv`<br>`revision_budget_checks/EARLIER_POOL_SEARCH_BUDGET_RECORDS_180s.csv` |
| Supplementary Table S5 | Chemprop configurations selected using seed 101 | `complete_results/Table_S33_Chemprop_configurations.csv`<br>`complete_results/Table_S32b_Chemprop_runs.csv`<br>`complete_results/Table_S33a_half_life_validation.csv`<br>`complete_results/Table_S33b_Chemprop_original_updated.csv` |
| Supplementary Table S6 | Paired differences against MiniMol and Chemprop (95% intervals) | `complete_results/Whole_task_WHOLE_TASK_CI.csv`<br>`complete_results/Whole_task_PROTOCOL.md` |
| Supplementary Table S7 | Task-wise selection versus FLAML (120 s): paired differences on identical inputs | `complete_results/Whole_task_WHOLE_TASK_CI.csv`<br>`complete_results/Whole_task_PROTOCOL.md` |
| Supplementary Table S8 | Ablation changes and MLP late-fusion scores across 22 tasks | `complete_results/Table_S37c_relative_to_reselected_reference.csv`<br>`complete_results/Table_S37b_late_fusion.csv` |
| Supplementary Table S9 | Validation ranks for the global model-family choice | `complete_results/Table_S37d_global_family_validation_ranks.csv` |
| Supplementary Table S10 | Mean ranks of 27 methods across 20 development tasks | `complete_results/S42_FINAL_SUMMARY.json`<br>`complete_results/S42_TASK_SCORES.csv`<br>`complete_results/S42_CONTRASTS.csv`<br>`protocols/development_study_initial_design.md`<br>`reports/development_study_results.md` |
| Supplementary Table S11 | Seven-method comparison on six official test tasks (five-run mean +/- population SD) | `complete_results/S43_RESULTS.json`<br>`complete_results/S43_all_runs.csv`<br>`complete_results/S43_FREEZE.json` |
| Supplementary Table S12 | Source tasks for the original related-task models | `historical_tables/Table_S20_cross_task_leakage_audit.csv`<br>`historical_tables/Table_S20b_cross_task_similarity_audit.csv`<br>`historical_tables/Table_S20e_target_only_sensitivity.csv` |
| Supplementary Table S13 | Results for the two target-only models | `complete_results/Table_S36_target_only_source.json`<br>`protocols/target_only_training.md` |
| Supplementary Table S14 | Molecular-subgroup scores and paired differences across 22 tasks | `complete_results/Table_S38_paired_subgroups.csv`<br>`complete_results/Table_S38a_subgroup_scores.csv`<br>`complete_results/Table_S38b_subgroup_counts.csv`<br>`complete_results/Subgroup_PROTOCOL.md` |
| Supplementary Figure S1 | FLAML (120 s) and the Trimole-Hybrid reference across 22 tasks | `complete_results/Table_S31_external22_long.csv`<br>`complete_results/Table_S34_recovered_pool_summary.csv`<br>`complete_results/Table_S34a_recovered_pool_runs.csv`<br>`complete_results/Table_S34b_pool_and_rows.json`<br>`complete_results/Table_S34b_reporting_metadata.json`<br>`complete_results/Table_S34c_search_budget_runs.csv` |
| Supplementary Figure S2 | Trimole-Hybrid, MiniMol and Chemprop across 22 tasks, grouped by official metric | `complete_results/Table_S31_external22_long.csv` |
| Supplementary Figure S3 | Related-task and target-only model results against the archived public top-1 reference (CFA) | `complete_results/Table_S36_target_only_source.json`<br>`protocols/target_only_training.md` |
| Manuscript Figure 4 | Two molecule-level sensitivity examples and AMES replay audit | `historical_tables/Table_S7_Figure4_two_case_sensitivity_audit.csv` |

## Additional methods records

| Record | Location inside the data download |
|---|---|
| Target-only model training settings | `protocols/target_only_training.md` |
| Dataset class prevalence and measurement limitations | `historical_tables/Table_S19_dataset_provenance.csv` |
| Original ablation component matrix | `historical_tables/Table_S2c_final_endpoint_summary.csv` |
| Matched 1,995-row solubility rescoring | `revision_solubility_1995/` |
| Shared-pool FLAML 120/180-second runs | `complete_results/Table_S34c_search_budget_runs.csv` and `revision_budget_checks/` |

## Verify the download

Run `python3 verify_manifest.py .` in the extracted directory. The manifest checks byte counts and SHA256 hashes and reports missing or extra files. The complete file-level table/figure mapping is included in the download.

## Model records and archived release

[Final model configurations and checkpoints](../models/) are indexed separately. The immutable [2026.10.24 research archive](https://doi.org/10.5281/zenodo.23602309) remains available; this GitHub release supplies the current supplementary-data package.
