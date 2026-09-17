from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1]
    / "code"
    / "trimole_ept_swap_v1"
    / "tools"
    / "run_flaml_matched_baseline_v1.py"
)
SPEC = importlib.util.spec_from_file_location("flaml_matched_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
FLAML_BASELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FLAML_BASELINE
SPEC.loader.exec_module(FLAML_BASELINE)


class FlamlMatchedBaselineTest(unittest.TestCase):
    def test_test_labels_are_blocked_before_score_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"smiles": ["CC"], "Y": [1]}).to_csv(
                root / "test.csv", index=False
            )
            for phase in ("select", "final"):
                with self.assertRaisesRegex(ValueError, "may not read test labels"):
                    FLAML_BASELINE.load_labels_for_phase(root, "test", phase)

            labels = FLAML_BASELINE.load_labels_for_phase(root, "test", "score")
            np.testing.assert_array_equal(labels, np.array([1.0]))

    def test_selection_and_final_label_policy(self) -> None:
        self.assertEqual(
            FLAML_BASELINE.ALLOWED_LABEL_SPLITS,
            {
                "select": {"train", "valid"},
                "final": {"train", "valid"},
                "score": {"test"},
            },
        )

    def test_prediction_writer_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.csv"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                FLAML_BASELINE.write_prediction(path, np.array([0.1, np.nan]))


if __name__ == "__main__":
    unittest.main()
