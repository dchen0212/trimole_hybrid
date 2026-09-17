from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "materialize_trimole_prediction_manifest_v1.py"
SPEC = importlib.util.spec_from_file_location("materialize_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MATERIALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATERIALIZE
SPEC.loader.exec_module(MATERIALIZE)


class MaterializeTrimoleManifestTest(unittest.TestCase):
    def test_safe_result_directories_are_explicit_for_both_corrected_tasks(self) -> None:
        self.assertEqual(
            set(MATERIALIZE.SAFE_RESULT_DIRECTORIES), MATERIALIZE.SAFE_TASKS
        )

    def test_normalize_supports_y_prob_and_checks_labels(self) -> None:
        frame = pd.DataFrame(
            {"sample_idx": [0, 1], "y_true": [0, 1], "y_prob": [0.2, 0.8]}
        )
        normalized = MATERIALIZE.normalize_frame(frame, np.array([0.0, 1.0]), "test")
        np.testing.assert_allclose(normalized.prediction, [0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "label alignment mismatch"):
            MATERIALIZE.normalize_frame(frame, np.array([1.0, 0.0]), "test")


if __name__ == "__main__":
    unittest.main()
