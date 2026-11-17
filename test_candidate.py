import unittest

from PIL import Image

from candidate_detector import prepare_pixels, sigmoid


class CandidateTests(unittest.TestCase):
    def test_preprocessing_shape_and_normalization(self):
        import numpy as np

        for size in ((600, 400), (400, 600), (100, 100)):
            image = Image.new("RGB", size, "white")
            image.info["exif"] = b"untrusted metadata"
            result = prepare_pixels(image)
            self.assertEqual(result.shape, (1, 3, 384, 384))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(result.flags.c_contiguous)
            self.assertAlmostEqual(float(result[0, 0, 0, 0]), (1 - .485) / .229, places=5)

    def test_stable_sigmoid(self):
        self.assertEqual(sigmoid(0), .5)
        self.assertEqual(sigmoid(1000), 1)
        self.assertEqual(sigmoid(-1000), 0)
        with self.assertRaises(ValueError):
            sigmoid(float("nan"))
