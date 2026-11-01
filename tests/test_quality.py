import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from quality import drift, passes_fullbody_gate


class FullBodyQualityTests(unittest.TestCase):
    def test_ankle_error_cannot_be_hidden_by_68_accurate_face_points(self):
        ref = np.zeros((1, 133, 3), np.float32)
        score = np.ones((1, 133), np.float32)
        candidate = ref.copy()
        candidate[0, 15, 0] = 20
        parts = drift(candidate, score, ref, score)
        self.assertGreater(parts['ankles']['xyP95_px'], 3)
        self.assertFalse(passes_fullbody_gate([{'parts': parts}]))

    def test_all_candidates_must_preserve_all_133_points(self):
        with self.assertRaises(ValueError):
            drift(np.zeros((1, 68, 3)), np.ones((1, 68)), np.zeros((1, 133, 3)), np.ones((1, 133)))

    def test_missing_feet_are_not_treated_as_verified(self):
        ref = np.zeros((1, 133, 3), np.float32)
        score = np.ones((1, 133), np.float32)
        score[:, 17:23] = 0
        self.assertFalse(passes_fullbody_gate([{'parts': drift(ref, score, ref, score)}]))

    def test_unchanged_full_body_passes(self):
        ref = np.zeros((1, 133, 3), np.float32)
        score = np.ones((1, 133), np.float32)
        self.assertTrue(passes_fullbody_gate([{'parts': drift(ref, score, ref, score)}]))
