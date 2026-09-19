from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "family_transfer_target_only", TOOLS / "run_family_transfer_leakage_safe_v2.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class TargetOnlyPoolTest(unittest.TestCase):
    def test_target_test_identity_cannot_change_training_pool(self) -> None:
        features = {
            "target": {"fp": (np.array([[1.0], [2.0]]), np.array([[3.0]]), np.array([[4.0]]))},
            "source": {"fp": (np.array([[5.0], [6.0]]), np.array([[7.0]]), np.array([[8.0]]))},
        }
        labels = {
            "target": {"train": np.array([0.0, 1.0]), "valid": np.array([1.0])},
            "source": {"train": np.array([1.0, 0.0]), "valid": np.array([0.0])},
        }
        identities = {
            "target": {"train": ["a", "b"], "valid": ["c"], "test": ["d"]},
            "source": {"train": ["d", "e"], "valid": ["f"], "test": ["g"]},
        }

        for stage, expected_rows in (("selection", 2), ("final", 3)):
            first = RUNNER.pooled_stage(
                features, labels, identities, ["target", "source"],
                "target", "fp", stage, source_policy="target_only",
            )
            identities["target"]["test"] = ["e", "f", "different"]
            second = RUNNER.pooled_stage(
                features, labels, identities, ["target", "source"],
                "target", "fp", stage, source_policy="target_only",
            )
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(len(first[0]), expected_rows)
            self.assertTrue(all(row["rows_after"] == 0 for row in first[2] if row["source_task"] == "source"))


if __name__ == "__main__":
    unittest.main()
