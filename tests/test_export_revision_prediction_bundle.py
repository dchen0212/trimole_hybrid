from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


SCRIPT = Path(__file__).parents[1] / "tools" / "export_revision_prediction_bundle_v1.py"
SPEC = importlib.util.spec_from_file_location("prediction_bundle", SCRIPT)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class PredictionBundleTest(unittest.TestCase):
    def test_exports_predictions_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate" / "family" / "task" / "seed_101"
            candidate.mkdir(parents=True)
            (candidate / "valid_predictions.csv").write_text("y_true,y_pred\n1,0.8\n0,0.2\n")
            (candidate / "test_predictions.csv").write_text("y_true,y_pred\n0,0.1\n1,0.9\n")
            method = root / "score" / "predictions" / "task"
            method.mkdir(parents=True)
            (method / "selector_seed_101.csv").write_text("sample_idx,y_true,prediction\n0,0,0.1\n1,1,0.9\n")
            selection = root / "selection.json"
            selection.write_text(json.dumps({"candidate_pool": ["family"], "tasks": ["task"], "seeds": [101]}))
            destination = root / "bundle.zip"
            result = BUNDLE.build_bundle(root / "candidate", root / "score", selection, destination)
            self.assertEqual(result["files"], 3)
            with ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                for member in archive.namelist():
                    if member.endswith(".csv"):
                        payload = archive.read(member).decode()
                        self.assertEqual(payload.splitlines()[0], "sample_idx,prediction")
                        self.assertNotIn("y_true", payload)


if __name__ == "__main__":
    unittest.main()
