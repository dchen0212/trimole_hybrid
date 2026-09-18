from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "code"
    / "trimole_ept_swap_v1"
    / "tools"
    / "run_paired_bootstrap_v1.py"
)
SPEC = importlib.util.spec_from_file_location("paired_bootstrap", MODULE_PATH)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)


class PairedBootstrapTest(unittest.TestCase):
    def test_mae_improvement_has_consistent_direction(self) -> None:
        labels = np.arange(20, dtype=float)
        reference = labels + 0.1
        comparator = labels + 1.0
        result = BOOTSTRAP.paired_bootstrap(
            labels,
            reference,
            comparator,
            "MAE",
            500,
            np.random.default_rng(1),
        )
        self.assertGreater(result["improvement"], 0.0)
        self.assertGreater(result["ci95_lower"], 0.0)

    def test_bh_adjustment_matches_known_values(self) -> None:
        adjusted = BOOTSTRAP.bh_adjust(np.array([0.01, 0.04, 0.03, 0.002]))
        np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])

    def test_hierarchical_bootstrap_resamples_seed_and_sample(self) -> None:
        labels = np.arange(30, dtype=float)
        reference = np.stack([labels + 0.1, labels + 0.2, labels + 0.3])
        comparator = np.stack([labels + 1.0, labels + 1.1, labels + 1.2])
        result = BOOTSTRAP.hierarchical_seed_sample_bootstrap(
            labels,
            reference,
            comparator,
            "MAE",
            500,
            np.random.default_rng(9),
            True,
        )
        self.assertGreater(result["ci95_lower"], 0.0)
        self.assertEqual(result["resampling_mode"], "hierarchical_seed_and_sample")
        self.assertEqual(result["seed_resampling"], "paired")
        self.assertTrue(result["reference_seed_uncertainty_available"])

    def test_single_seed_is_explicitly_flagged(self) -> None:
        labels = np.arange(20, dtype=float)
        reference = np.stack([labels + 0.1])
        comparator = np.stack([labels + 1.0, labels + 1.2])
        result = BOOTSTRAP.hierarchical_seed_sample_bootstrap(
            labels,
            reference,
            comparator,
            "MAE",
            200,
            np.random.default_rng(4),
            False,
        )
        self.assertFalse(result["reference_seed_uncertainty_available"])
        self.assertEqual(result["seed_resampling"], "independent")


if __name__ == "__main__":
    unittest.main()
