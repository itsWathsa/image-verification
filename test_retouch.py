"""Guard the research split, class mapping and threshold accounting."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from fivek_retouch import assign_splits, operating_point
from retouch_model import ARCHITECTURE, RetouchClassifier, load_retouch_model, retouch_score


class RetouchResearchTests(unittest.TestCase):
    def test_official_test_ids_remain_unseen_and_variants_share_id(self):
        train = [f"FiveK/train/input/a{i:04d}-photo.jpg" for i in range(1, 4501)]
        test = [f"FiveK/test/input/a{i:04d}-photo.jpg" for i in range(4501, 5001)]
        names = train + test + ["FiveK/train/input/"]
        splits = assign_splits(names)
        self.assertEqual(splits, assign_splits(list(reversed(names))))
        self.assertEqual(sum(value == "train" for value in splits.values()), 4000)
        self.assertEqual(sum(value == "validation" for value in splits.values()), 500)
        self.assertTrue(all(splits[name] == "test" for name in test))
        with self.assertRaises(ValueError):
            assign_splits(names[:-2])
        with self.assertRaisesRegex(ValueError, "Repeated photo IDs"):
            assign_splits(train + test[:-1] + ["FiveK/test/input/a0001-photo.jpg"])

    def test_flagged_threshold_ties_count_as_positive(self):
        result = operating_point(np.array([0., 0., 1., 1.]), np.array([.1, .6, .4, .9]), .6)
        self.assertEqual(result["false_adjustments"], 1)
        self.assertEqual(result["missed_adjustments"], 1)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["control_false_positive_rate"], .5)

    def test_checkpoint_rejects_wrong_task_and_prediction_ignores_exif(self):
        model = RetouchClassifier()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.head[-1].bias.data.fill_(-2.)
        checkpoint = {"architecture": ARCHITECTURE, "positive_class": "photographic_adjustment", "features": 272,
                      "state_dict": model.state_dict(), "mean": torch.zeros(272), "std": torch.ones(272)}
        image = Image.new("RGB", (160, 100), "gray")
        tagged = image.copy()
        tagged.info["exif"] = b"untrusted camera information"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            loaded = load_retouch_model(path)
            self.assertAlmostEqual(retouch_score(image, loaded), torch.sigmoid(torch.tensor(-2.)).item())
            self.assertEqual(retouch_score(image, loaded), retouch_score(tagged, loaded))
            checkpoint["positive_class"] = "AI_generated"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_retouch_model(path)
