from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "common_pool_budget", TOOLS / "audit_common_pool_compute_budget_v1.py"
)
assert SPEC and SPEC.loader
BUDGET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUDGET
SPEC.loader.exec_module(BUDGET)


class ComputeBudgetTest(unittest.TestCase):
    def test_equal_option_count_does_not_imply_equal_fit_count(self) -> None:
        frozen = {
            "candidate_pool": [f"family_{index}" for index in range(9)],
            "tasks": ["one"],
            "seeds": [101],
            "selected_rule_by_task": {"one": "per_task_single"},
        }
        labels = np.array([0, 1] * 10, dtype=float)
        with patch.object(BUDGET.fair.common, "metric_from_task_file", return_value="AUROC"), patch.object(
            BUDGET.fair.common, "load_prediction_for_phase", return_value=(labels, labels)
        ):
            frame = BUDGET.audit_budget(Path("/unused"), frozen)
        row = frame.iloc[0]
        self.assertEqual(row["validation_options_each"], 6)
        self.assertEqual(row["selector_meta_cv_fits"], 5)
        self.assertEqual(row["automl_meta_cv_fits"], 30)
        self.assertFalse(bool(row["end_to_end_compute_matched"]))


if __name__ == "__main__":
    unittest.main()
