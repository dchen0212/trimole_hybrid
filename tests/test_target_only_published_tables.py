from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "supplementary_tables"


def rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class TargetOnlyPublishedTablesTest(unittest.TestCase):
    def test_primary_scores_are_test_blind_and_match_s20e(self) -> None:
        benchmark = rows("Table_S1_TDC_ADMET22_benchmark.csv")
        sensitivity = rows("Table_S20e_target_only_sensitivity.csv")
        expected = {
            "TDC.CL-Hepa": ("clearance_hepatocyte_az", 0.18123691452116247),
            "TDC.CYP3A4-S": ("cyp3a4_substrate_carbonmangels", 0.6536618444846293),
        }
        self.assertEqual(len(benchmark), 22)
        for dataset, (task, score) in expected.items():
            benchmark_row = next(row for row in benchmark if row["Dataset"] == dataset)
            primary_row = next(row for row in sensitivity if row["task"] == task and row["policy"] == "target_only")
            self.assertAlmostEqual(float(benchmark_row["Trimole-Hybrid"].split("±")[0]), score, places=6)
            self.assertAlmostEqual(float(primary_row["test_mean"]), score, places=12)
            self.assertEqual(primary_row["uses_target_test_structure_for_source_filter"], "False")
            self.assertEqual(primary_row["cross_task_source_rows_after_selection"], "0")
            self.assertEqual(primary_row["cross_task_source_rows_after_refit"], "0")

    def test_historical_analyses_are_not_paired_with_target_only_results(self) -> None:
        ablation = rows("Table_S4_formal_ablation_long.csv")
        subgroup = rows("Table_S22b_subgroup_uncertainty.csv")
        summary = rows("Table_S4b_formal_ablation_summary.csv")
        self.assertEqual({int(row["available_tasks"]) for row in summary}, {20})
        self.assertEqual(len(subgroup), 132)
        self.assertEqual(sum(row["primary_subgroup_comparable"] == "True" for row in subgroup), 120)
        for task in ("clearance_hepatocyte_az", "cyp3a4_substrate_carbonmangels"):
            self.assertTrue(all(row["primary_ablation_comparable"] == "False" for row in ablation if row["task"] == task))
            self.assertTrue(all(row["primary_subgroup_comparable"] == "False" for row in subgroup if row["task"] == task))


if __name__ == "__main__":
    unittest.main()
