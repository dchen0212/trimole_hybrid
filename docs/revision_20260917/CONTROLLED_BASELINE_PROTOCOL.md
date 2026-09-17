# Controlled Baseline Protocol

## Purpose

This experiment isolates the effect of task-wise selection and prediction combination under one fixed candidate pool. It does not reuse the submitted task-specific recipes as if they were a controlled comparison.

## Fixed candidate pool

Every task uses the same three frozen representation caches:

1. ChemBERTa for the SMILES/string view;
2. KPGT for the two-dimensional molecular-graph view;
3. EPT for the geometry-related view.

Each representation is paired with the same standardized linear stochastic-gradient learner. Classification uses logistic loss and regression uses squared loss with target standardization. Model settings, seeds and stopping criteria are fixed across tasks.

## Data boundaries

The runner has three separate phases.

- `select`: fits base models on official train only, scores official validation only, and writes the frozen global and per-task modality choices. It does not read test labels.
- `final`: refits on train plus validation, creates official-test predictions and creates five-fold out-of-fold development predictions for stacking. It does not read test labels.
- `score`: reads official test labels only after the selection file and all prediction files exist.

## Controls

- `global_single`: one modality selected by mean validation rank across all 22 tasks and then used for every task.
- `per_task_single`: the best validation modality for each task, without custom ensembles.
- `validation_top2_average`: an unweighted average of the two highest-ranked modalities for that task, ranked by five-seed validation mean.
- `validation_top3_average`: an explicit average of all three validation-ranked modalities; with this three-candidate pool it is numerically equivalent to `uniform_average` and is retained to make the top-3 control auditable.
- `uniform_average`: an unweighted average of the three modality predictions.
- `oof_stacking`: logistic regression for classification or ridge regression for regression, trained only on five-fold out-of-fold development predictions.

Five fixed seeds are used: 101, 202, 303, 404 and 505. The same base predictions feed all four controls.

## Scope and limitation

This is a deliberately reduced, symmetric candidate pool intended to isolate selection and combination behavior. It is not identical to the much larger submitted engineering search space. A separate equal-budget AutoML control is still required and must not be described as complete until its implementation, frozen search budget and outputs are independently validated.
