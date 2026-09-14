from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "code"
    / "trimole_ept_swap_v1"
    / "tools"
    / "family_transfer_safety.py"
)
SPEC = importlib.util.spec_from_file_location("family_transfer_safety", MODULE_PATH)
assert SPEC and SPEC.loader
SAFETY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAFETY
SPEC.loader.exec_module(SAFETY)


class FamilyTransferSafetyTest(unittest.TestCase):
    def test_selection_blocks_validation_and_test(self) -> None:
        blocked = SAFETY.forbidden_keys(["valid-a", "shared"], ["test-a", "shared"], "selection")
        self.assertEqual(blocked, {"valid-a", "test-a", "shared"})
        self.assertEqual(
            SAFETY.keep_mask(["train-a", "shared", "test-a"], blocked),
            [True, False, False],
        )


    def test_final_blocks_test_only(self) -> None:
        blocked = SAFETY.forbidden_keys(["valid-a"], ["test-a"], "final")
        self.assertEqual(blocked, {"test-a"})
        self.assertEqual(SAFETY.keep_mask(["valid-a", "test-a"], blocked), [True, False])


    def test_overlap_stats_counts_rows_and_unique_molecules(self) -> None:
        stats = SAFETY.overlap_stats(["a", "a", "b", "c"], ["a", "d", "d"])
        self.assertEqual(
            stats,
            {
                "source_rows": 4,
                "source_unique_molecules": 3,
                "holdout_rows": 3,
                "holdout_unique_molecules": 2,
                "overlap_source_rows": 2,
                "overlap_unique_molecules": 1,
            },
        )


    def test_anonymized_key_is_deterministic(self) -> None:
        self.assertEqual(
            SAFETY.anonymized_key("ABCDEFGHIJKLMN"),
            SAFETY.anonymized_key("ABCDEFGHIJKLMN"),
        )
        self.assertNotEqual(
            SAFETY.anonymized_key("ABCDEFGHIJKLMN"),
            SAFETY.anonymized_key("ABCDEFGHIJKLMO"),
        )

    def test_candidate_selection_is_validation_only(self) -> None:
        candidates = [
            {
                "feature_set": "fp",
                "config_index": 0,
                "valid_mean": 0.7,
                "valid_adjusted": 0.6,
                "test_mean": 0.9,
            },
            {
                "feature_set": "fp_kpgt",
                "config_index": 1,
                "valid_mean": 0.8,
                "valid_adjusted": 0.75,
                "test_mean": 0.1,
            },
        ]
        selected = SAFETY.select_validation_candidate(candidates, "valid_mean")
        self.assertEqual(selected["feature_set"], "fp_kpgt")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            SAFETY.select_validation_candidate(candidates, "test_mean")


if __name__ == "__main__":
    unittest.main()
