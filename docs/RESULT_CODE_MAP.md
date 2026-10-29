# Paper results and implementation entry points

Current tables and figures refer to the [20-page submission supplement](../supplementary/supplementary.pdf). The [data index](../supplementary_data/) binds the source records. Code paths beginning `revision/` are inside the [2026.10.24 analysis archive](https://github.com/dchen0212/trimole_hybrid/releases/download/v2026.10.24/Trimole_Hybrid_revision_20261024.zip).

| Paper item | Implementation or source | Inputs |
|---|---|---|
| Model architecture and component fitting | [Feature-head model](../code/trimole/models/model.py), [trainer](../code/trimole/training/trainer.py), [final task JSONs](../configs/tasks/) | Prepared task features and the recorded component configuration |
| S1–S2: datasets and task components | [Current table files](../supplementary_tables/); final task configuration package | Saved split and component records |
| S3/S5: MiniMol and Chemprop | `revision/code/minimol_kpgt_benchmark22_20261014_v01/`, `revision/code/chemprop22_20261015_v01/` | Official task splits, upstream encoders and documented fit settings |
| S4: shared-pool controls, FLAML 120 s | `revision/code/original_pool_matched_controls22_20261015_v01/` | Saved candidate prediction columns and matched rows |
| S4: FLAML 180 s | `revision/s4_pool_180s/fit.py`, `common.py`, `controller.py` | Same candidate inputs, split and seeds; increased requested time budget |
| S4a: four budgets, nine-predictor pool | `revision/revision_budget_checks/` and the source bindings recorded there | The separate fixed nine-predictor pool |
| S6–S7: paired intervals | `revision/statistics/analyze.py` | Saved matched row-level predictions |
| Figure 3/S8–S9: ablations | `revision/code/figure3_abcd_labels_right_20261021_v12/`; the archived model-family index and result map | Saved ablation and validation-rank tables |
| S10–S11: 20-task OOF and six-task studies | Protocols and results under `complete_results/S42_*` and `S43_*` in the current data download | The study-specific candidate and row definitions |
| S12–S13: target-only models | `revision/code/target_only_expanded_20261013_v02/`, `target_only_kernel_refine_20261014_v01/`, `hepatocyte_final/` | Target-task data, encoders and fitted artifacts |
| S14: subgroups | `revision/code/subgroup_four_recovery_20261020_v01/`, `subgroups_publication_style_20261020_v11/` | Saved matched predictions and molecular subgroup records |
| Figure 4: molecule-level examples | `revision/case_studies/`; manuscript Figure 4 source records in the current data index | Archived predictions and case-specific reconstruction records |

The code paths identify the implementations and packaging entry points. The release scope and fitted-artifact availability are stated in [Code availability](../CODE_AVAILABILITY.md). Earlier [model-family](https://github.com/dchen0212/trimole_hybrid/blob/6c263364f79b8c8f21fbf482812636fb801ec233/archive/MODEL_FAMILY_INDEX.md) and [result-to-code](https://github.com/dchen0212/trimole_hybrid/blob/6c263364f79b8c8f21fbf482812636fb801ec233/archive/REVISION_RESULT_CODE_MAP.csv) records retain the original source-path conventions.

The complete original source tree is retained in the [pinned pre-cleanup snapshot](https://github.com/dchen0212/trimole_hybrid/tree/6c263364f79b8c8f21fbf482812636fb801ec233). The current fixed model commands are [train.py](../train.py) and [predict.py](../predict.py). Their [source bindings](MODEL_SOURCE_MAP.json) identify the retained original implementations.

The complete original source tree is retained in the [pinned pre-cleanup snapshot](https://github.com/dchen0212/trimole_hybrid/tree/6c263364f79b8c8f21fbf482812636fb801ec233). The current fixed model commands are [train.py](../train.py) and [predict.py](../predict.py). Their [source bindings](MODEL_SOURCE_MAP.json) identify the retained original implementations.

The complete original source tree is retained in the [pinned pre-cleanup snapshot](https://github.com/dchen0212/trimole_hybrid/tree/6c263364f79b8c8f21fbf482812636fb801ec233). The current fixed model commands are [train.py](../train.py) and [predict.py](../predict.py). Their [source bindings](MODEL_SOURCE_MAP.json) identify the retained original implementations.
