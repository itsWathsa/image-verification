"""Registration and reference-difference checks, independent of AI detectors."""

import io
import unittest

import numpy as np
from PIL import Image, ImageDraw

from reference_compare import _checked_transform, compare_reference


def textured_image(seed=17):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(30, 225, (480, 640, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(70):
        x, y = rng.integers([0, 0], [600, 440])
        shade = tuple(map(int, rng.integers(0, 255, 3)))
        draw.rectangle((int(x), int(y), int(x + 20), int(y + 20)), outline=shade, width=3)
    return image


class ReferenceComparisonTests(unittest.TestCase):
    def test_identical_pixels_are_unmarked_not_authenticity_proof(self):
        image = textured_image()
        result = compare_reference(image, image.copy())
        self.assertEqual(result["changed_fraction"], 0.)
        self.assertEqual(result["overlap_fraction"], 1.)
        self.assertIn("authenticity inconclusive", result["assessment"])
        self.assertFalse(np.asarray(result["mask"]).any())

    def test_resize_and_jpeg_do_not_become_large_content_change(self):
        image = textured_image()
        scaled = image.resize((480, 360), Image.Resampling.LANCZOS)
        data = io.BytesIO()
        scaled.save(data, format="JPEG", quality=90)
        with Image.open(io.BytesIO(data.getvalue())) as compressed:
            result = compare_reference(image, compressed)
        self.assertLess(result["changed_fraction"], .03)
        self.assertGreater(result["overlap_fraction"], .9)
        self.assertGreaterEqual(result["alignment"]["inliers"], 12)

    def test_local_splice_is_marked_and_boundaries_are_excluded(self):
        image = textured_image()
        candidate = image.copy()
        ImageDraw.Draw(candidate).rectangle((230, 180, 370, 300), fill=(255, 0, 0))
        result = compare_reference(image, candidate)
        mask = np.asarray(result["mask"]) > 0
        self.assertGreater(mask[185:295, 235:365].mean(), .9)
        self.assertLess(result["changed_fraction"], .15)
        overlap = np.asarray(result["overlap_mask"]) > 0
        self.assertFalse(mask[~overlap].any())
        self.assertTrue(np.all(np.asarray(result["overlay"])[~overlap] == 128))

    def test_unrelated_and_featureless_images_are_inconclusive(self):
        for image in (textured_image(seed=55), Image.new("RGB", (640, 480), "white")):
            with self.assertRaisesRegex(ValueError, "Inconclusive"):
                compare_reference(textured_image(), image)

    def test_degenerate_mirrored_and_extreme_projections_rejected(self):
        for transform in (np.zeros((3, 3)), np.diag([-1., 1., 1.]),
                          np.diag([100., 100., 1.]), np.full((3, 3), np.nan)):
            with self.assertRaisesRegex(ValueError, "Inconclusive"):
                _checked_transform(transform, (480, 640, 3), (480, 640, 3))
