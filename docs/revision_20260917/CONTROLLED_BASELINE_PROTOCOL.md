# Strict Common-Pool Baseline Protocol

## Purpose

The primary experiment isolates the effect of task-wise selection and prediction combination under one fixed candidate pool. It does not reuse the submitted task-specific recipes as if they were a controlled comparison. The earlier three-view controls and FLAML run are retained only as historical sensitivity analyses because they did not share the broader historical Trimole-Hybrid candidate space.

## Fixed candidate pool

Every primary method consumes the same nine validation-stable candidate families, complete for all 22 tasks and five seeds. These nine families were frozen from 15 initially complete families after a validation-only numerical-stability audit excluded six unregularized deep linear heads globally. The exclusion was completed before test predictions were opened.

The historical sensitivity experiment used only three frozen representation caches:

1. ChemBERTa for the SMILES/string view;
2. KPGT for the two-dimensional molecular-graph view;
3. EPT for the geometry-related view.

Those three-view results remain in Tables S21-S22 for transparency but are not the strict fairness result.

## Data boundaries

The runner has three separate phases.

- `select`: reads validation predictions only, evaluates the declared recipe or meta-model configurations and writes the frozen selection manifest. It cannot open test predictions or labels.
- `score`: verifies the manifest hash, opens aligned test predictions only after selection is frozen and computes the reported metrics.

## Controls

- `global_single`: one family selected by mean validation rank across all 22 tasks.
- `per_task_single`: the highest-ranked validation family for each task.
- `validation_top2_average` and `validation_top3_average`: unweighted averages of the two or three highest-ranked validation families.
- `uniform_candidate_average`: an unweighted average of all nine families.
- `validation_stacking`: a validation-fitted linear combination using the same nine inputs.
- `common_pool_taskwise_selector`: selects among the six predeclared rules above.
- `common_pool_automl`: selects among six predeclared meta-model configurations using the same nine inputs.

Five fixed seeds are used: 101, 202, 303, 404 and 505. Every method receives the same base prediction pool. The two selectable systems each evaluate six validation configurations; deterministic controls require no additional search.

## Scope and limitation

The strict outputs contain 880 method-task-seed scores, 176 task-method summaries, 154 hierarchical seed-and-sample bootstrap comparisons and complete validation-selection records (Tables S24g-S24j). These controls establish candidate-pool parity and a transparent, but not equal-compute, selection comparison; Table S24k reports the unequal meta-model fit counts. A separate validation-only nested audit is provided in Table S25. The mixed results do not support universal superiority of the task-wise selector.
