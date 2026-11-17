import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from edit_detector import analyze_edits, render_maps


class EditTests(unittest.TestCase):
    def test_tensor_preprocessing_and_output_mapping(self):
        import torch

        class Model:
            def __call__(self, pixels):
                self.pixels = pixels
                _, _, height, width = pixels.shape
                return (torch.zeros(1, 2, height, width), torch.zeros(1, 1, height, width), torch.zeros(1), None)

        model = Model()
        image = Image.new("RGB", (800, 400), "white")
        result = analyze_edits(image, model)
        self.assertEqual(result["analyzed_size"], (384, 192))
        self.assertEqual(image.size, (800, 400))
        self.assertAlmostEqual(float(model.pixels.max()), 255 / 256)
        np.testing.assert_array_equal(result["map"], .5)
        np.testing.assert_array_equal(result["reliability"], .5)
        self.assertEqual(result["score"], .5)
        self.assertEqual(result["assessment"], "inconclusive")
        overlay, reliability = render_maps(result)
        self.assertEqual(overlay.size, (384, 192))
        self.assertEqual(reliability.size, (384, 192))
        with self.assertRaises(ValueError):
            analyze_edits(Image.new("RGB", (1, 100)), model)

    def test_edit_button_and_graceful_failure(self):
        data = io.BytesIO()
        Image.new("RGB", (64, 64)).save(data, format="PNG")
        result = {"image": Image.new("RGB", (64, 64)), "map": np.zeros((64, 64)),
                  "reliability": np.ones((64, 64)), "score": .7, "analyzed_size": (64, 64)}
        for failure in (None, RuntimeError("model unavailable")):
            with patch("streamlit.file_uploader", return_value=data), patch(
                "edit_detector.load_edit_detector", return_value=object()
            ), patch("edit_detector.analyze_edits", return_value=result, side_effect=failure):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.button(key="analyze_edits").click().run(timeout=30)
                self.assertFalse(app.exception)
                if failure:
                    self.assertIn("Edit analysis unavailable", app.error[0].value)
                else:
                    self.assertEqual(app.metric[0].value, "0.700")
                    self.assertIn("Edit assessment: inconclusive", app.warning[0].value)
