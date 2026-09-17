from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_family_transfer_leakage_safe_v2 as runner
import run_strict_5run_seedwise_prediction_zoo_v1 as prediction_zoo
from family_transfer_safety import forbidden_keys, select_validation_candidate


class FamilyTransferRevisionTest(unittest.TestCase):
    def test_selection_blocks_validation_and_test_identities(self) -> None:
        self.assertEqual(
            forbidden_keys(["valid-a", "shared"], ["test-a", "shared"], "selection"),
            {"valid-a", "test-a", "shared"},
        )

    def test_final_blocks_test_identities_only(self) -> None:
        self.assertEqual(
            forbidden_keys(["valid-a"], ["test-a"], "final"),
            {"test-a"},
        )

    def test_candidate_selector_rejects_test_derived_statistics(self) -> None:
        candidates = [
            {"feature_set": "fp", "config_index": 0, "valid_mean": 0.7},
            {"feature_set": "fp_kpgt", "config_index": 1, "valid_mean": 0.8},
        ]
        with self.assertRaises(ValueError):
            select_validation_candidate(candidates, "test_mean")

    def test_preselection_test_frame_contains_smiles_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            pd.DataFrame(
                {
                    "Drug_ID": ["a"],
                    "Drug": ["CCO"],
                    "Y": [1],
                    "smiles": ["CCO"],
                    "label": [1],
                }
            ).to_csv(path, index=False)
            frame = runner.read_feature_frame(path, include_labels=False)
        self.assertEqual(list(frame.columns), ["smiles"])

    def test_migrated_result_path_rebases_only_results_suffix(self) -> None:
        original_results = prediction_zoo.RESULTS
        try:
            prediction_zoo.RESULTS = Path("/new/project/results_strict")
            resolved = prediction_zoo.resolve_results_path(
                "/old/afs/repository/results_strict/example/task/test_predictions.csv"
            )
        finally:
            prediction_zoo.RESULTS = original_results
        self.assertEqual(
            resolved,
            Path("/new/project/results_strict/example/task/test_predictions.csv"),
        )


if __name__ == "__main__":
    unittest.main()
