from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "build_subgroup_uncertainty_v1.py"
SPEC = importlib.util.spec_from_file_location("subgroup_uncertainty", MODULE_PATH)
assert SPEC and SPEC.loader
SUBGROUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBGROUP
SPEC.loader.exec_module(SUBGROUP)


class SubgroupUncertaintyTest(unittest.TestCase):
    def test_classification_requires_both_classes_and_minimum_count(self) -> None:
        valid = SUBGROUP.group_validity("AUROC", np.array([0] * 25 + [1] * 5), 30, 5)
        self.assertEqual(valid, (True, "valid", 5, 1 / 6))
        one_class = SUBGROUP.group_validity("AUPRC", np.zeros(30), 30, 5)
        self.assertFalse(one_class[0])
        self.assertEqual(one_class[1], "classification_group_has_one_class")

    def test_regression_only_requires_group_size(self) -> None:
        valid = SUBGROUP.group_validity("MAE", np.arange(30), 30, 5)
        self.assertEqual(valid, (True, "valid", None, None))
        small = SUBGROUP.group_validity("Spearman", np.arange(29), 30, 5)
        self.assertFalse(small[0])


if __name__ == "__main__":
    unittest.main()
