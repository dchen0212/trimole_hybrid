from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv_selected_prediction_ensemble_builder_fast_v2 as base


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
RESULTS = REPO / "results_strict"

ALL22_TASKS = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
]


EXTRA_TASKS = {
    "pgp_broccatelli": {"metric": "AUROC", "direction": "max", "top1_ref": 0.938},
    "bioavailability_ma": {"metric": "AUROC", "direction": "max", "top1_ref": 0.942},
    "cyp2c9_substrate_carbonmangels": {"metric": "AUPRC", "direction": "max", "top1_ref": 0.474},
    "ld50_zhu": {"metric": "MAE", "direction": "min", "top1_ref": 0.552},
}


EXTRA_SUMMARIES = [
    "paper_main_chemical_prior_xl_v4_all22_32core/summary.csv",
    "paper_main_chemical_prior_xl_v4_remaining4_32core/summary.csv",
    "rank_uplift_tabular_fp_repeated_v1_focus/summary.csv",
    "cv_selected_prediction_ensemble_builder_fast_v4_rank_batch/summary.csv",
    "cv_selected_prediction_ensemble_builder_fast_v4_rank_batch_mae_raw/summary.csv",
]


def write_xl_summary(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/result.json")):
        result = json.load(open(path))
        rows.append(
            {
                "task": result.get("task", path.parent.name),
                "candidate": f"xl_v4_{result.get('candidate', '')}",
                "head": result.get("head", ""),
                "tdc_metric": result.get("tdc_metric", ""),
                "metric_direction": result.get("metric_direction", ""),
                "selected_variant": result.get("selected_variant", ""),
                "selected_topk": result.get("selected_topk", ""),
                "selected_backend": result.get("selected_backend", ""),
                "weight_sidecar": result.get("weight_sidecar", ""),
                "cv_mean": result.get("cv_mean", ""),
                "cv_std": result.get("cv_std", ""),
                "test_tdc_score": result.get("test_tdc_score", ""),
                "incumbent_test_tdc_score": result.get("incumbent_test_tdc_score", ""),
                "improved_test": result.get("improved_test", ""),
                "tdc_top1_ref": result.get("tdc_top1_ref", ""),
                "is_top1_level": result.get("is_top1_level", ""),
                "trainval_pred_file": result.get("trainval_pred_file", ""),
                "test_pred_file": result.get("test_pred_file", ""),
                "endpoint": result.get("endpoint", ""),
            }
        )
    if not rows:
        return
    fields = list(rows[0].keys())
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for directory in [
        RESULTS / "paper_main_chemical_prior_xl_v4_all22_32core",
        RESULTS / "paper_main_chemical_prior_xl_v4_remaining4_32core",
    ]:
        write_xl_summary(directory)

    base.TASKS.update(EXTRA_TASKS)
    for summary in EXTRA_SUMMARIES:
        if summary not in base.PRED_SUMMARIES:
            base.PRED_SUMMARIES.append(summary)

    sys.argv = [
        sys.argv[0],
        "--out-root",
        str(RESULTS / "xl_v4_prediction_blend_all22_v1"),
        "--tasks",
        *ALL22_TASKS,
        "--weight-step",
        "0.1",
        "--max-streams",
        "8",
    ]
    base.main()


if __name__ == "__main__":
    main()
