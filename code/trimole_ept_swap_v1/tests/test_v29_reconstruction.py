import unittest

import numpy as np

import run_v29_exact_endpoint_5run_v1 as reconstruction


class V29ReconstructionTest(unittest.TestCase):
    def test_raw_labels_follow_official_order(self):
        official = np.array([1.0, 3.0, 2.0])
        status, residual = reconstruction.label_alignment(official, official)
        self.assertEqual(status, "raw")
        self.assertEqual(residual, 0.0)

    def test_positive_affine_standardization_preserves_order(self):
        official = np.array([4.3, 730.0, 22.0, 52.0])
        standardized = (official - official.mean()) / official.std()
        status, residual = reconstruction.label_alignment(standardized, official)
        self.assertEqual(status, "positive_affine")
        self.assertLess(residual, 1e-12)

    def test_permuted_labels_are_rejected(self):
        official = np.array([1.0, 4.0, 2.0, 8.0])
        permuted = official[[1, 0, 3, 2]]
        status, _ = reconstruction.label_alignment(permuted, official)
        self.assertEqual(status, "mismatch")


if __name__ == "__main__":
    unittest.main()
