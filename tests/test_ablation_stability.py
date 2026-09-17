from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "build_ablation_stability_v1.py"
SPEC = importlib.util.spec_from_file_location("ablation_stability", MODULE_PATH)
assert SPEC and SPEC.loader
STABILITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STABILITY
SPEC.loader.exec_module(STABILITY)


class AblationStabilityTest(unittest.TestCase):
    def test_rank_loss_respects_metric_direction(self) -> None:
        frame = pd.DataFrame(
            [
                {"task": "auc", "variant": "full_v36_final", "metric": "AUROC", "score_mean": 0.9},
                {"task": "auc", "variant": "weak", "metric": "AUROC", "score_mean": 0.7},
                {"task": "mae", "variant": "full_v36_final", "metric": "MAE", "score_mean": 0.2},
                {"task": "mae", "variant": "weak", "metric": "MAE", "score_mean": 0.5},
            ]
        )
        result = STABILITY.normalized_rank_loss(frame)
        weak = result[result.variant == "weak"]
        self.assertTrue((weak.normalized_rank_loss == 1.0).all())

    def test_missing_scores_are_excluded_from_rank_denominator(self) -> None:
        frame = pd.DataFrame(
            [
                {"task": "auc", "variant": "full_v36_final", "metric": "AUROC", "score_mean": 0.9},
                {"task": "auc", "variant": "weak", "metric": "AUROC", "score_mean": 0.7},
                {"task": "auc", "variant": "missing", "metric": "AUROC", "score_mean": None},
            ]
        )
        result = STABILITY.normalized_rank_loss(frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.n_ranked_variants.unique().tolist(), [2])


if __name__ == "__main__":
    unittest.main()
