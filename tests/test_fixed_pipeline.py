"""Regression fixtures for fixed feature schemas and final prediction assembly."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from trimole_hybrid.registry import load_task
from trimole_hybrid.composition import task_prediction
from trimole_hybrid.chemical.features import build_features, task_profile

FIXTURE=json.loads((Path(__file__).parent/"fixed_pipeline_golden.json").read_text())

class FixedPipelineTests(unittest.TestCase):
    def test_original_feature_vectors(self):
        import hashlib
        for profile, record in FIXTURE["features"].items():
            with self.subTest(profile=profile):
                x,names=build_features(FIXTURE["smiles"],profile)
                self.assertEqual(list(x.shape),record["shape"])
                self.assertEqual(hashlib.sha256(x.tobytes()).hexdigest(),record["sha256"])
                self.assertEqual(len(names),len(set(names)))

    def test_fixed_combinations(self):
        for case in FIXTURE["combinations"]:
            with self.subTest(task=case["task"],run=case["run"]):
                inputs={k:np.asarray(v) for k,v in case["inputs"].items()}
                actual=task_prediction(load_task(case["task"]),inputs,case["run"])
                np.testing.assert_array_equal(actual,case["expected"])

    def test_individual_seed_inputs(self):
        case=FIXTURE["combinations"][0]
        p={"seed_"+str(seed):np.asarray(values) for seed,values in zip(case["inputs"]["seed_ids"],case["inputs"]["seedbag_heads"])}
        np.testing.assert_array_equal(task_prediction(load_task(case["task"]),p,case["run"]),case["expected"])

    def test_guard_mismatched_seed_ids(self):
        case=FIXTURE["combinations"][0]
        inputs={k:np.asarray(v) for k,v in case["inputs"].items()}
        inputs["seed_ids"]=np.arange(10,15)
        with self.assertRaisesRegex(ValueError,"seed_ids"):
            task_prediction(load_task(case["task"]),inputs)

    def test_guard_misaligned_components(self):
        with self.assertRaisesRegex(ValueError,"row"):
            task_prediction(load_task("ppbr_az"),{"seedbag_mean":[1,2],"fp_only":[1]})

    def test_guard_invalid_smiles(self):
        with self.assertRaisesRegex(ValueError,"row"):
            build_features(["CCO","invalid_smiles"])

    def test_guard_invalid_run(self):
        with self.assertRaises(ValueError):
            task_prediction(load_task("clearance_microsome_az"),{"seedbag_mean":[1,2]},0)

    def test_completed_prediction_export(self):
        for task in ("clearance_hepatocyte_az","clearance_microsome_az","ppbr_az","cyp2d6_substrate_carbonmangels"):
            np.testing.assert_array_equal(task_prediction(load_task(task),{"prediction":[.1,.2]}),[.1,.2])

    def test_chemical_only_profile_does_not_add_embedding(self):
        for task in ("bioavailability_ma","caco2_wang","cyp2c9_veith","ld50_zhu"):
            self.assertIsNone(task_profile(load_task(task))["candidate"])

if __name__=="__main__":
    unittest.main()
