from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "nested_validation", TOOLS / "run_nested_validation_selection_v1.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NestedValidationSelectionTest(unittest.TestCase):
    def test_classification_folds_are_stratified(self) -> None:
        labels = np.array([0, 1] * 10)
        folds = list(MODULE._folds("AUROC", labels, 5, 101))
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(len(np.unique(labels[test])) == 2 for _, test in folds))

    def test_recipe_choice_uses_metric_direction(self) -> None:
        self.assertEqual(MODULE._choose("AUROC", {"a": 0.8, "b": 0.7}), "a")
        self.assertEqual(MODULE._choose("MAE", {"a": 0.8, "b": 0.7}), "b")


if __name__ == "__main__":
    unittest.main()
