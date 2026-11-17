"""Regression checks for image handling and experimental model integration."""

import io
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError
from streamlit.testing.v1 import AppTest

from app import get_detector, read_image
from detector import analyze_pixels, interpret_score


class ImageTests(unittest.TestCase):
    def test_hosted_model_download_and_windows_offline(self):
        for platform, download in (("linux", True), ("win32", False)):
            get_detector.clear()
            with patch("app.sys.platform", platform), patch("detector.load_detector", return_value=object()) as loader:
                get_detector()
                get_detector()
                loader.assert_called_once_with(download=download)
        get_detector.clear()

    def test_cloud_pixel_limit_before_decode(self):
        data = io.BytesIO()
        Image.new("RGB", (40, 30)).save(data, format="PNG")
        with patch("app.MAX_IMAGE_PIXELS", 1000), self.assertRaisesRegex(ValueError, "20 megapixels"):
            read_image(data.getvalue())

    def test_file_details_and_absent_metadata(self):
        for fmt in ("JPEG", "PNG"):
            data = io.BytesIO()
            Image.new("RGB", (40, 30)).save(data, format=fmt)
            _, actual, size, make, model = read_image(data.getvalue())
            self.assertEqual((actual, size), (fmt, (40, 30)))
            self.assertEqual((make, model), ("Not available", "Not available"))

    def test_invalid_file(self):
        with self.assertRaises(UnidentifiedImageError):
            read_image(b"not an image")

    def test_bands_and_invalid_scores(self):
        self.assertIn("AI-like", interpret_score(0.95))
        self.assertIn("Camera-photo-like", interpret_score(0.05))
        self.assertEqual(interpret_score(0.5), "Inconclusive")
        for value in (float("nan"), -1, 2):
            with self.assertRaises(ValueError):
                interpret_score(value)

    def test_metadata_does_not_enter_processor(self):
        class StopBeforeInference(Exception):
            pass

        received = []

        def processor(*, images, return_tensors):
            received.append((images.tobytes(), dict(images.info)))
            raise StopBeforeInference

        clean = Image.new("RGB", (40, 30), "blue")
        tagged = clean.copy()
        tagged.info["exif"] = b"arbitrary camera metadata"
        for image in (clean, tagged):
            with self.assertRaises(StopBeforeInference):
                analyze_pixels(image, (processor, None, None, {0: "artificial", 1: "human"}))
        self.assertEqual(received[0], received[1])
        self.assertEqual(received[0][1], {})

    def test_streamlit_initial_render(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertFalse(app.exception)

    def test_streamlit_analysis_and_failure(self):
        data = io.BytesIO()
        Image.new("RGB", (40, 30)).save(data, format="PNG")
        failures = (None, RuntimeError("offline"), ImportError("cannot import name 'NP_SUPPORTED_MODULES'"), ModuleNotFoundError("missing package", name="torch"), ImportError("DLL load failed: An Application Control policy has blocked this file."))
        for failure in failures:
            with patch("streamlit.file_uploader", return_value=data), patch(
                "detector.load_detector", return_value=object()
            ), patch(
                "detector.analyze_pixels",
                return_value={"ai_score": 0.5, "signal": "Inconclusive"},
                side_effect=failure,
            ):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.button[0].click().run(timeout=30)
                self.assertFalse(app.exception)
                if failure:
                    self.assertTrue(app.error)
                    if "Application Control" in str(failure):
                        self.assertIn("Application Control", app.error[0].value)
                    elif isinstance(failure, ImportError) and not isinstance(failure, ModuleNotFoundError):
                        self.assertIn("version compatibility", app.error[0].value)
                    if isinstance(failure, ModuleNotFoundError):
                        self.assertIn("module is unavailable: torch", app.error[0].value)
                else:
                    self.assertEqual(app.metric[0].value, "0.500")


if __name__ == "__main__":
    unittest.main()
