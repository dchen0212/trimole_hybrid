from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).parents[1] / "code" / "trimole_ept_swap_v1" / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "build_controlled_comparison_table_v1.py"
SPEC = importlib.util.spec_from_file_location("controlled_table", MODULE_PATH)
assert SPEC and SPEC.loader
CONTROLLED_TABLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROLLED_TABLE
SPEC.loader.exec_module(CONTROLLED_TABLE)


class ControlledComparisonTableTest(unittest.TestCase):
    def test_normalize_assigns_model_names(self) -> None:
        common = {"task": ["x"], "metric": ["AUROC"], "seed": [1], "prediction_file": ["x.csv"]}
        trimole = pd.DataFrame({**common, "model": ["trimole_hybrid"]})
        controlled = pd.DataFrame({**common, "control": ["global_single"]})
        flaml = pd.DataFrame(common)
        result = CONTROLLED_TABLE.normalize_manifests(trimole, controlled, flaml)
        self.assertEqual(
            result.model.tolist(), ["trimole_hybrid", "global_single", "flaml_automl"]
        )


if __name__ == "__main__":
    unittest.main()
