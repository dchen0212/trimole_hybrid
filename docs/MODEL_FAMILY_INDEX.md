# Model Family Index for Trimole-Hybrid Code Pull

This index maps the manuscript-level model families to concrete files in this targeted server code pull. It is intended to avoid the misleading impression that the package only contains KPGT source code. KPGT is included as the graph-encoder dependency, but the pull also contains the Trimole multimodal package, 3D/EPT wrappers, chemistry-prior sidecars, fusion modules, endpoint-selection scripts, and strict audit outputs.

## Core multimodal package

- `trimole_hybrid/trimole/`
- `trimole_ept_swap_v1/trimole/`

Both workspaces include the same core package layout:

- `trimole/embeddings/`
- `trimole/models/`
- `trimole/training/`
- `trimole/configs/`
- `trimole/analysis/`

## Sequence / SMILES branch

The sequence branch is represented by ChemBERTa-style molecular language model embeddings.

- `trimole_hybrid/trimole/embeddings/chemberta.py`
- `trimole_ept_swap_v1/trimole/embeddings/chemberta.py`
- `trimole_hybrid/scripts/analyze_ablation_results.py`
- `trimole_ept_swap_v1/trimole/configs/task_configs.py`
- `trimole_ept_swap_v1/trimole/configs/task_configs_fixed.py`

## Graph branch

The graph branch is represented by KPGT embeddings and the KPGT source dependency.

- `trimole_hybrid/trimole/embeddings/kpgt.py`
- `trimole_ept_swap_v1/trimole/embeddings/kpgt.py`
- `KPGT/scripts/extract_features.py`
- `KPGT/scripts/finetune.py`
- `KPGT/scripts/train_kpgt.py`
- `KPGT/src/model/light.py`
- `KPGT/src/data/featurizer.py`

KPGT appears as a large top-level folder because it is an external graph-model dependency. It is not the whole method.

## 3D / EPT / UniMol branch

The 3D-sensitive branch is represented by UniMol/EPT-style embeddings and EPT routing or endpoint scripts.

- `trimole_hybrid/trimole/embeddings/unimol.py`
- `trimole_ept_swap_v1/trimole/embeddings/unimol.py`
- `trimole_ept_swap_v1/build_maplight_style_ept_hybrid_summary_v1.py`
- `trimole_ept_swap_v1/results_strict/ept_family_routing_master_v1/`
- `trimole_ept_swap_v1/results_strict/clearance_microsome_kpgt_ept_seedbag_endpoint_v1/`
- `trimole_hybrid/results_strict/ablation_v36_formal/formal_candidate_family_metrics.csv`
- `trimole_hybrid/results_strict/ablation_v36_formal/formal_ablation_delta_heatmap_ready.csv`

## Multimodal fusion model

The main neural fusion model projects and combines SMILES, graph, and 3D embeddings. The code supports MLP, gated, residual dynamic, and 3D-downweighted gated fusion variants.

- `trimole_hybrid/trimole/models/model.py`
- `trimole_hybrid/trimole/models/fusion_blocks.py`
- `trimole_ept_swap_v1/trimole/models/model.py`
- `trimole_ept_swap_v1/trimole/models/fusion_blocks.py`
- `trimole_hybrid/trimole/training/trainer.py`
- `trimole_hybrid/trimole/training/trainer_moddrop.py`
- `trimole_hybrid/trimole/training/trainer_transfer.py`
- `trimole_ept_swap_v1/trimole/training/trainer.py`
- `trimole_ept_swap_v1/trimole/training/trainer_moddrop.py`
- `trimole_ept_swap_v1/trimole/training/trainer_transfer.py`

Useful code-level evidence:

- `MultiModalFusionMLP` in `trimole_hybrid/trimole/models/model.py` takes `emb_smiles`, `emb_3d`, and `emb_graph`.
- `ResidualDynamicGatedFusion3DDownweight` is used for the `gated_3d_downweight` fusion option.

## Chemistry-prior sidecars

The chemistry-prior branch uses RDKit/Morgan fingerprints, descriptors, and tabular chemical features with classical machine-learning heads.

- `trimole_ept_swap_v1/descriptor_sidecar_official_v1.py`
- `trimole_ept_swap_v1/descriptor_sidecar_official_v2.py`
- `trimole_ept_swap_v1/paper_main_chemical_prior_v2.py`
- `trimole_ept_swap_v1/paper_main_chemical_prior_xl_v4.py`
- `trimole_ept_swap_v1/tools/run_fixed_chemical_endpoint_5run_v1.py`
- `trimole_ept_swap_v1/tools/run_pure_chem_multibackend_endpoint_v1.py`
- `trimole_ept_swap_v1/results_strict/fixed_chemical_endpoint_5run_v1/`
- `trimole_ept_swap_v1/results_strict/pure_chem_multibackend_endpoint_v1/`

These scripts contain the RDKit fingerprint and descriptor feature construction used in the chemistry sidecar candidate pool.

## Classical endpoint heads

The sidecar and prediction-zoo scripts include classical model heads such as XGBoost, ExtraTrees, RandomForest, LogisticRegression, Ridge, and related scikit-learn/XGBoost models.

- `trimole_ept_swap_v1/descriptor_sidecar_official_v1.py`
- `trimole_ept_swap_v1/descriptor_sidecar_official_v2.py`
- `trimole_ept_swap_v1/tools/run_cyp2c9_substrate_seedmatched_xgb_grid_v2.py`
- `trimole_ept_swap_v1/tools/run_cyp3a4_substrate_clean_backend_expansion_v1.py`
- `trimole_ept_swap_v1/tools/run_clearance_pooled_family_xgb_v1.py`
- `trimole_ept_swap_v1/tools/run_cyp_substrate_pooled_family_xgb_v1.py`
- `trimole_hybrid/scripts/fusion/run_tx_stacking_valid.py`
- `trimole_hybrid/scripts/fusion/run_top4_targeted_router.py`

## Prediction-level ensembles, blends, and seed bagging

The final framework is not just a single neural backbone. It includes prediction-level blends, seed bags, rank/logit/z-score/raw blends, endpoint selectors, and formal 5-run summaries.

- `trimole_ept_swap_v1/prediction_zoo_ensemble_v2.py`
- `trimole_ept_swap_v1/offline_prediction_zoo_blend_v1.py`
- `trimole_ept_swap_v1/cv_selected_prediction_ensemble_builder_fast_v2.py`
- `trimole_ept_swap_v1/tools/run_final_formal_seed5_prediction_zoo_v1.py`
- `trimole_ept_swap_v1/tools/run_strict_5run_seedwise_prediction_zoo_v1.py`
- `trimole_ept_swap_v1/tools/run_trainval_scaffold_foldbag_endpoint_v1.py`
- `trimole_ept_swap_v1/results_strict/final_formal_seed5_prediction_zoo_v1/`
- `trimole_ept_swap_v1/results_strict/ppbr_kpgt_seedbag_endpoint_v1/`
- `trimole_ept_swap_v1/results_strict/pgp_seedmatched_blend_formal_v2/`
- `trimole_ept_swap_v1/results_strict/solubility_clean_selector_v2/`

## Validation-only endpoint selection and formal audits

These files document endpoint selection, strict TDC split usage, formal ablations, and audit checks.

- `trimole_hybrid/notes/STRICT_RULES.md`
- `trimole_hybrid/scripts/audit/audit_benchmark_provenance.py`
- `trimole_hybrid/scripts/audit/audit_benchmark_provenance_strict.py`
- `trimole_hybrid/scripts/audit/audit_final_submission_provenance.py`
- `trimole_hybrid/scripts/audit/audit_final_submission_provenance_v2.py`
- `trimole_hybrid/results_strict/tdc_aligned_22task_table_strict_v1/`
- `trimole_hybrid/results_strict/final_5seed_v1/README_strict_v1.txt`
- `trimole_hybrid/results_strict/ablation_v36_formal/README.md`
- `trimole_hybrid/results_strict/ablation_v36_formal/audit_ablation_v36_formal_numeric.py`
- `trimole_hybrid/results_strict/ablation_v36_formal/formal_ablation_summary.csv`
- `trimole_hybrid/results_strict/ablation_v36_formal/formal_valid_selection_evidence.csv`
- `trimole_hybrid/results_strict/ablation_v36_formal/formal_global_endpoint_valid_selection_evidence.csv`

## Case-study and external-inference scripts

These files support the manuscript case-study and external molecule diagnostics.

- `trimole_hybrid/results_strict/case_study_v36_exact_full_main/`
- `trimole_hybrid/results_strict/alyftrek_minimol_trimole_case_v1/`
- `trimole_hybrid/results_strict/pgp_external_matched_replacement_sweep_v1/`
- `trimole_hybrid/results_strict/ames_official_case_matched_replacement_sweep_v1/`
- `trimole_hybrid/results_strict/ames_exact_replay_salvage_v1/`
- `trimole_hybrid/reconstruct_pgp_exact_external_v36.py`
- `trimole_hybrid/reconstruct_cyp3a4s_exact_external_v36.py`
- `trimole_hybrid/external_case_perturbation_v1.py`
- `trimole_hybrid/pgp_external_matched_replacement_sweep_v1.py`
- `trimole_hybrid/cyp3a4s_external_matched_replacement_sweep_v1.py`
- `trimole_hybrid/bbb_external_matched_replacement_sweep_v1.py`
- `trimole_hybrid/make_deutivacaftor_pgp_perturbation_figure.py`

## Practical reviewer note

This targeted pull excludes large trained weights, cached arrays, and official datasets. Therefore it is not a one-command full rerun package. Its purpose is to provide a compact, inspectable source-and-audit bundle that demonstrates the model families and manuscript-relevant selection/ablation/case-study provenance.
