from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "run_common_pool_fair_comparison_v1.py"
SPEC = importlib.util.spec_from_file_location("common_pool_fair", MODULE_PATH)
assert SPEC and SPEC.loader
FAIR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FAIR
SPEC.loader.exec_module(FAIR)


class CommonPoolFairComparisonTest(unittest.TestCase):
    def test_six_predeclared_budgets(self) -> None:
        self.assertEqual(len(FAIR.RULES), 6)
        self.assertEqual(len(FAIR.AUTOML_CONFIGS), 6)

    def test_oof_meta_prediction_is_finite(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=float)
        matrix = np.column_stack(
            [np.linspace(0.1, 0.9, len(labels)), np.linspace(0.9, 0.1, len(labels))]
        )
        prediction = FAIR.oof_meta_prediction("AUROC", matrix, labels, 1, 101)
        self.assertEqual(prediction.shape, labels.shape)
        self.assertTrue(np.isfinite(prediction).all())

    def test_meta_prediction_supports_regression(self) -> None:
        labels = np.linspace(-1.0, 1.0, 12)
        matrix = np.column_stack([labels + 0.1, labels - 0.1])
        prediction = FAIR.fit_meta_prediction(
            "MAE", matrix, labels, matrix[:3], 1, 101
        )
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
