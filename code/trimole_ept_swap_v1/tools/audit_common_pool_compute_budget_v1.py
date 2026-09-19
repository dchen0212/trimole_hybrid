"""Account for meta-model fits in the frozen common-pool comparison.

The historical base-family training wall time is unavailable, so this audit
does not claim an end-to-end matched compute budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_common_pool_fair_comparison_v1 as fair


def audit_budget(candidate_root: Path, frozen: dict[str, object]) -> pd.DataFrame:
    families = list(frozen["candidate_pool"])
    tasks = list(frozen["tasks"])
    seeds = [int(seed) for seed in frozen["seeds"]]
    selected_rules = dict(frozen["selected_rule_by_task"])
    rows: list[dict[str, object]] = []
    for task in tasks:
        metric = fair.common.metric_from_task_file(candidate_root, families[0], task, seeds[0])
        for seed in seeds:
            labels, _ = fair.common.load_prediction_for_phase(
                fair.common.candidate_file(candidate_root, families[0], task, seed, "valid"),
                "valid", "select",
            )
            n_folds = len(list(fair._folds(metric, labels, seed)))
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "seed": seed,
                    "candidate_families_shared": len(families),
                    "validation_options_each": len(fair.RULES),
                    "meta_oof_folds": n_folds,
                    "selector_meta_cv_fits": n_folds,
                    "automl_meta_cv_fits": len(fair.AUTOML_CONFIGS) * n_folds,
                    "selector_meta_final_fits": int(selected_rules[task] == "validation_stacking"),
                    "automl_meta_final_fits": 1,
                    "base_family_training_walltime_available": False,
                    "end_to_end_compute_matched": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_root}")
    frozen = json.loads(args.selection_manifest.read_text())
    frame = audit_budget(args.candidate_root, frozen)
    args.out_root.mkdir(parents=True)
    frame.to_csv(args.out_root / "common_pool_compute_budget_by_seed.csv", index=False)
    summary = {
        "candidate_families_shared": int(frame.candidate_families_shared.iloc[0]),
        "validation_options_each": int(frame.validation_options_each.iloc[0]),
        "task_seed_rows": int(len(frame)),
        "selector_meta_cv_fits": int(frame.selector_meta_cv_fits.sum()),
        "automl_meta_cv_fits": int(frame.automl_meta_cv_fits.sum()),
        "selector_meta_final_fits": int(frame.selector_meta_final_fits.sum()),
        "automl_meta_final_fits": int(frame.automl_meta_final_fits.sum()),
        "base_family_training_walltime_available": False,
        "end_to_end_compute_matched": False,
        "interpretation": "Candidate count and validation-option count match; meta-model fit counts and end-to-end compute do not.",
    }
    (args.out_root / "budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
