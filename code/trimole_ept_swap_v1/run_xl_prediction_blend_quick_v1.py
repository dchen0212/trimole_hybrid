from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv_selected_prediction_ensemble_builder_fast_v2 as base

REPO = Path('<PROJECT_ROOT>/trimole_ept_swap_v1')
RESULTS = REPO / 'results_strict'


def write_xl_summary(root: Path) -> None:
    rows = []
    for p in sorted(root.glob('*/result.json')):
        r = json.load(open(p))
        row = {
            'task': r.get('task', p.parent.name),
            'candidate': f"xl_v4_{r.get('candidate','')}",
            'head': r.get('head', ''),
            'tdc_metric': r.get('tdc_metric', ''),
            'metric_direction': r.get('metric_direction', ''),
            'selected_variant': r.get('selected_variant', ''),
            'selected_topk': r.get('selected_topk', ''),
            'selected_backend': r.get('selected_backend', ''),
            'weight_sidecar': r.get('weight_sidecar', ''),
            'cv_mean': r.get('cv_mean', ''),
            'cv_std': r.get('cv_std', ''),
            'test_tdc_score': r.get('test_tdc_score', ''),
            'incumbent_test_tdc_score': r.get('incumbent_test_tdc_score', ''),
            'improved_test': r.get('improved_test', ''),
            'tdc_top1_ref': r.get('tdc_top1_ref', ''),
            'is_top1_level': r.get('is_top1_level', ''),
            'trainval_pred_file': r.get('trainval_pred_file', ''),
            'test_pred_file': r.get('test_pred_file', ''),
            'endpoint': r.get('endpoint', ''),
        }
        rows.append(row)
    if not rows:
        return
    fields = list(rows[0].keys())
    with (root / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


for d in [
    RESULTS / 'paper_main_chemical_prior_xl_v4_all22_32core',
    RESULTS / 'paper_main_chemical_prior_xl_v4_remaining4_32core',
]:
    write_xl_summary(d)

base.TASKS.update({
    'pgp_broccatelli': {'metric': 'AUROC', 'direction': 'max', 'top1_ref': 0.938},
    'cyp2c9_substrate_carbonmangels': {'metric': 'AUPRC', 'direction': 'max', 'top1_ref': 0.474},
})
base.PRED_SUMMARIES.extend([
    'paper_main_chemical_prior_xl_v4_all22_32core/summary.csv',
    'paper_main_chemical_prior_xl_v4_remaining4_32core/summary.csv',
    'rank_uplift_tabular_fp_repeated_v1_focus/summary.csv',
    'cv_selected_prediction_ensemble_builder_fast_v4_rank_batch/summary.csv',
    'cv_selected_prediction_ensemble_builder_fast_v4_rank_batch_mae_raw/summary.csv',
])

if __name__ == '__main__':
    sys.argv = [
        sys.argv[0],
        '--out-root', str(RESULTS / 'xl_v4_prediction_blend_quick_v1'),
        '--tasks',
        'pgp_broccatelli',
        'cyp2c9_substrate_carbonmangels',
        'hia_hou',
        'bbb_martins',
        'solubility_aqsoldb',
        'clearance_hepatocyte_az',
        'vdss_lombardo',
        'ppbr_az',
        '--weight-step', '0.1',
        '--max-streams', '8',
    ]
    base.main()
