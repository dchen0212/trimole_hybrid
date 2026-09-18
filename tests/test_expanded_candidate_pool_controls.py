from __future__ import annotations

import importlib.util
import json
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
    / "run_expanded_candidate_pool_controls_v1.py"
)
SPEC = importlib.util.spec_from_file_location("expanded_pool", MODULE_PATH)
assert SPEC and SPEC.loader
EXPANDED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPANDED
SPEC.loader.exec_module(EXPANDED)


class ExpandedCandidatePoolControlsTest(unittest.TestCase):
    def test_selection_phase_cannot_read_test_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_predictions.csv"
            pd.DataFrame({"y_true": [0, 1], "y_pred": [0.2, 0.8]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "may not read test"):
                EXPANDED.load_prediction_for_phase(path, "test", "select")

    def test_complete_pool_requires_all_tasks_seeds_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in ("complete", "incomplete"):
                for task in ("task_a", "task_b"):
                    for seed in (101, 202):
                        run = root / family / task / f"seed_{seed}"
                        run.mkdir(parents=True)
                        for split in ("valid", "test"):
                            if family == "incomplete" and task == "task_b" and seed == 202 and split == "test":
                                continue
                            pd.DataFrame(
                                {"y_true": [0, 1], "y_pred": [0.2, 0.8]}
                            ).to_csv(run / f"{split}_predictions.csv", index=False)
            self.assertEqual(
                EXPANDED.discover_complete_pool(
                    root, ["task_a", "task_b"], [101, 202]
                ),
                ["complete"],
            )

    def test_spearman_average_uses_validation_scaling(self) -> None:
        valid = [np.array([0.0, 1.0]), np.array([0.0, 100.0])]
        test = [np.array([2.0, 3.0]), np.array([200.0, 300.0])]
        _, averaged = EXPANDED.oriented_average("Spearman", valid, test)
        np.testing.assert_allclose(averaged, np.array([3.0, 5.0]))

    def test_stable_policy_uses_validation_ratio_only(self) -> None:
        families = [
            "chem_fp_xgb",
            "deep_kpgt_linear",
            "deep_kpgt_ridge_linear",
        ]
        stability = pd.DataFrame(
            {
                "candidate_family": families,
                "complete_for_all_tasks_seeds": True,
                "validation_files_checked": 10,
                "regression_validation_files_checked": 5,
                "max_abs_validation_prediction": [2.0, 1e9, 3.0],
                "max_regression_mae_to_median_baseline_ratio": [0.8, 1e8, 1.2],
            }
        )
        selected, audit = EXPANDED.apply_candidate_family_policy(
            families, "stable_regularized", stability
        )
        self.assertEqual(
            selected,
            [
                "chem_fp_xgb",
                "deep_kpgt_ridge_linear",
            ],
        )
        self.assertEqual(int(audit["eligible"].sum()), 2)
        self.assertTrue(
            audit.loc[
                audit["candidate_family"].eq("deep_kpgt_linear"), "reason"
            ].iloc[0].startswith("excluded")
        )

    def test_fixed_stacking_returns_finite_prediction(self) -> None:
        valid_x = np.array(
            [[0.1, 0.4], [0.2, 0.3], [0.8, 0.6], [0.9, 0.7]], dtype=float
        )
        labels = np.array([0, 0, 1, 1], dtype=float)
        test_x = np.array([[0.3, 0.4], [0.7, 0.6]], dtype=float)
        prediction = EXPANDED.fixed_stacking_prediction(
            "AUROC", valid_x, labels, test_x, 101
        )
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_metric_direction_orients_mae_as_utility(self) -> None:
        self.assertEqual(EXPANDED.utility("AUROC", 0.8), 0.8)
        self.assertEqual(EXPANDED.utility("MAE", 0.8), -0.8)

    def test_alignment_filters_candidate_only_rows_by_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.csv"
            evaluation_path = root / "evaluation.csv"
            pd.DataFrame(
                {
                    "Drug": ["A", "excluded-1", "B", "excluded-2", "A"],
                    "Y": [1.0, 7.0, 2.0, 8.0, 1.0],
                }
            ).to_csv(candidate_path, index=False)
            pd.DataFrame(
                {"smiles": ["A", "B", "A"], "label": [1.0, 2.0, 1.0]}
            ).to_csv(evaluation_path, index=False)
            indices, labels, audit = EXPANDED.build_test_alignment(
                candidate_path, evaluation_path
            )
            np.testing.assert_array_equal(indices, np.array([0, 2, 4]))
            np.testing.assert_array_equal(labels, np.array([1.0, 2.0, 1.0]))
            self.assertEqual(audit["excluded_rows"], 2)
            self.assertEqual(audit["excluded_molecules"], "excluded-1;excluded-2")

    def test_alignment_fails_when_evaluation_row_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.csv"
            evaluation_path = root / "evaluation.csv"
            pd.DataFrame({"Drug": ["A"], "Y": [1.0]}).to_csv(
                candidate_path, index=False
            )
            pd.DataFrame({"Drug": ["B"], "Y": [1.0]}).to_csv(
                evaluation_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "absent from candidate split"):
                EXPANDED.build_test_alignment(candidate_path, evaluation_path)


if __name__ == "__main__":
    unittest.main()
