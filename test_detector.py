"""Checks for gated scans, aggregation and schema failures with one model."""
import math
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from detector import analyze_pixels, prepare_pixels, ANALYSIS_METHOD


class FakeSession:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def run(self, names, inputs):
        self.calls += 1
        assert inputs["pixel_values"].shape == (1, 3, 384, 384)
        score = next(self.scores)
        return [np.array([[math.log(score / (1 - score))]], dtype=np.float32)]


class DetectorTests(unittest.TestCase):
    def test_low_and_high_initial_scores_are_not_rescanned(self):
        for score in (.005, .1, .49, .99):
            session = FakeSession([score])
            result = analyze_pixels(Image.new("RGB", (800, 600)), session)
            self.assertEqual(session.calls, 1)
            self.assertAlmostEqual(result["ai_score"], score, places=6)
            self.assertEqual(len(result["crop_scores"]), 1)

    def test_borderline_rescan_retains_max_and_provenance(self):
        session = FakeSession([.5, .94, .983])
        result = analyze_pixels(Image.new("RGB", (800, 600)), session)
        self.assertEqual(session.calls, 3)
        self.assertAlmostEqual(result["ai_score"], .983, places=6)
        self.assertAlmostEqual(result["baseline_score"], .5, places=6)
        self.assertEqual([r["short_edge"] for r in result["crop_scores"]], [440, 576, 768])
        self.assertEqual(result["analysis_method"], ANALYSIS_METHOD)
        self.assertIn("AI-like", result["signal"])

    def test_rescan_can_remain_inconclusive_and_invalid_scale_is_rejected(self):
        result = analyze_pixels(Image.new("RGB", (800, 600)), FakeSession([.6, .2, .3]))
        self.assertAlmostEqual(result["ai_score"], .6, places=6)
        self.assertEqual(result["signal"], "Inconclusive")
        with self.assertRaises(ValueError):
            prepare_pixels(Image.new("RGB", (64, 64)), 1024)

    def test_rescanning_is_the_only_default_pipeline(self):
        session = FakeSession([.6, .7, .8])
        result = analyze_pixels(Image.new("RGB", (800, 600)), session)
        self.assertEqual(session.calls, 3)
        self.assertEqual(result["analysis_method"], ANALYSIS_METHOD)
        self.assertEqual(len(result["crop_scores"]), 3)


if __name__ == "__main__":
    unittest.main()
