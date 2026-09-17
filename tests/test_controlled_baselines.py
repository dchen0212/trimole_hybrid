from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "code"
    / "trimole_ept_swap_v1"
    / "tools"
    / "run_controlled_selection_baselines_v1.py"
)
SPEC = importlib.util.spec_from_file_location("controlled_baselines", MODULE_PATH)
assert SPEC and SPEC.loader
BASELINES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASELINES
SPEC.loader.exec_module(BASELINES)


class ControlledBaselineTest(unittest.TestCase):
    def test_test_labels_are_blocked_before_score_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import pandas as pd

            pd.DataFrame({"smiles": ["CC"], "Y": [1]}).to_csv(
                root / "test.csv", index=False
            )
            for phase in ("select", "final"):
                with self.assertRaisesRegex(ValueError, "may not read test labels"):
                    BASELINES.load_labels_for_phase(root, "test", phase)
            labels = BASELINES.load_labels_for_phase(root, "test", "score")
            np.testing.assert_array_equal(labels, np.array([1.0]))

    def test_regression_fit_is_finite_on_high_scale_features(self) -> None:
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(80, 32)) * 1e4
        train_y = rng.normal(loc=2.0, scale=0.5, size=80)
        test_x = rng.normal(size=(20, 32)) * 1e4

        prediction = BASELINES.fit_predict(
            train_x,
            train_y,
            test_x,
            classification=False,
            seed=101,
            max_iter=2000,
        )

        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(float(np.max(np.abs(prediction))), 100.0)

    def test_prediction_writer_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.csv"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                BASELINES.write_prediction(path, np.array([0.1, np.inf]))

    def test_top3_and_uniform_use_identical_canonical_average(self) -> None:
        base = {
            "chemberta": np.array([0.1, 0.7, 0.3]),
            "kpgt": np.array([0.4, 0.2, 0.8]),
            "ept": np.array([0.9, 0.5, 0.6]),
        }
        top3 = BASELINES.mean_selected_modalities(base, ["ept", "kpgt", "chemberta"])
        uniform = BASELINES.mean_selected_modalities(base, BASELINES.MODALITIES)
        np.testing.assert_array_equal(top3, uniform)


if __name__ == "__main__":
    unittest.main()
