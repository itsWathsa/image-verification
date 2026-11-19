"""Research candidate checks; synthetic examples are not accuracy evidence."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from columbia_enhancement import adjusted, source_family, split_families, adjustment_metrics
from enhancement_model import ARCHITECTURE, EnhancementMLP, enhancement_features, enhancement_score, load_enhancement_model


class EnhancementTests(unittest.TestCase):
    def test_original_subimages_stay_in_one_split(self):
        self.assertEqual(source_family("canong3_02_sub_01.tif"), source_family("canong3_02_sub_09.tif"))
        groups = [f"camera_scene_{i}" for i in range(20)]
        first = split_families(groups)
        self.assertEqual(first, split_families(list(reversed(groups))))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        self.assertEqual(len(first), 20)

    def test_features_ignore_metadata_and_controls_are_identity(self):
        pixels = np.random.default_rng(5).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        features = enhancement_features(image)
        tagged = image.copy()
        tagged.info["exif"] = b"arbitrary metadata"
        np.testing.assert_array_equal(features, enhancement_features(tagged))
        self.assertTrue(np.isfinite(features).all())
        for operation in ("brightness", "contrast", "gamma"):
            np.testing.assert_array_equal(np.asarray(adjusted(image, operation, 1.)), pixels)
        with self.assertRaises(ValueError):
            adjusted(image, "unknown", .5)

    def test_weights_only_loading_and_adjustment_logit_mapping(self):
        image = Image.new("RGB", (64, 64), "gray")
        count = len(enhancement_features(image))
        model = EnhancementMLP(count)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.layers[-1].bias.data.fill_(2.)
        checkpoint = {"architecture": ARCHITECTURE, "features": count, "state_dict": model.state_dict(),
                      "positive_class": "synthetic_tonal_adjustment",
                      "mean": torch.zeros(count), "std": torch.ones(count)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            torch.save(checkpoint, path)
            self.assertAlmostEqual(enhancement_score(image, load_enhancement_model(path)), torch.sigmoid(torch.tensor(2.)).item())
            checkpoint["architecture"] = "unknown"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_enhancement_model(path)

    def test_report_metrics_distinguish_adjustments_from_ai_generation(self):
        report = adjustment_metrics([{"label": "camera", "ai_score": .2},
                                     {"label": "generated", "ai_score": .8}])
        self.assertEqual(report["accuracy"], 1.)
        self.assertEqual(report["adjusted_count"], 1)
        self.assertEqual(report["control_count"], 1)
        self.assertEqual(report["confusion_matrix"]["true_adjustments"], 1)
