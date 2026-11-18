"""Class mapping, preprocessing and dataset isolation checks for CIFAKE."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from cifake_experiment import split_indices
from cifake_model import ARCHITECTURE, CifakeCNN, cifake_score, load_cifake_model, prepare_cifake_pixels


class CifakeTests(unittest.TestCase):
    def test_rgb_normalization_and_checkpoint_class_mapping(self):
        image = Image.new("RGB", (80, 40), (255, 128, 0))
        pixels = prepare_cifake_pixels(image)
        self.assertEqual(tuple(pixels.shape), (1, 3, 32, 32))
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), 1)
        self.assertAlmostEqual(float(pixels[0, 1, 0, 0]), 128 / 255, places=6)
        self.assertEqual(float(pixels[0, 2, 0, 0]), 0)
        model = CifakeCNN()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.classifier[-1].bias.data.fill_(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            checkpoint = {"architecture": ARCHITECTURE, "generated_class": 1, "state_dict": model.state_dict()}
            torch.save(checkpoint, path)
            self.assertAlmostEqual(cifake_score(image, load_cifake_model(path)), torch.sigmoid(torch.tensor(2.)).item())
            checkpoint["generated_class"] = 0
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_cifake_model(path)

    def test_duplicates_conflicts_and_official_test_isolation(self):
        hashes = [f"image-{i}" for i in range(40)] + ["image-2", "image-3", "conflict", "conflict"]
        labels = np.array([i % 2 for i in range(40)] + [0, 1, 0, 1])
        official = ["train"] * 40 + ["test", "test", "train", "test"]
        first, conflicts = split_indices(hashes, labels, official)
        second, _ = split_indices(hashes, labels, official)
        self.assertEqual(conflicts, 2)
        self.assertEqual(set(first["test"]), {40, 41})
        self.assertNotIn(2, first["train"])
        self.assertNotIn(3, first["validation"])
        groups = [{hashes[i] for i in indices} for indices in first.values()]
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertFalse(groups[a] & groups[b])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
