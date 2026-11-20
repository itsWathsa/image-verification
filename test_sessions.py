"""Cross-check isolation and invalidation across Streamlit reruns."""

import io
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest


def upload(color):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(stream, format="PNG")
    return stream


class SessionTests(unittest.TestCase):
    def test_colour_only_reference_difference_is_reported_separately(self):
        comparison = {"aligned_reference": Image.new("RGB", (64, 64)), "candidate": Image.new("RGB", (64, 64)),
                      "overlay": Image.new("RGB", (64, 64)), "difference": Image.new("L", (64, 64)),
                      "changed_fraction": 0., "overlap_fraction": 1., "analysis_size": (64, 64), "alignment": {},
                      "tonal_comparison": {"brightness_difference_levels": 0., "contrast_spread_ratio": 1.,
                                           "notable_tonal_difference": False, "notable_colour_difference": True,
                                           "mean_colour_difference_levels": 5.2}}

        def uploader(*args, **kwargs):
            return upload("white") if kwargs.get("key") == "reference_upload" else upload("blue")

        with patch("streamlit.file_uploader", side_effect=uploader), patch(
            "reference_compare.compare_reference", return_value=comparison
        ) as compare:
            app = AppTest.from_file("app.py").run(timeout=30)
            self.assertEqual([tab.label for tab in app.tabs], ["AI detection", "Compare original", "File details"])
            app.checkbox(key="has_reference").check().run(timeout=30)
            app.button(key="compare_reference").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(compare.call_count, 1)
            self.assertTrue(any("Differences detected compared" in item.value for item in app.info))
            self.assertTrue(any(item.value == "5.2 levels" for item in app.metric))
            self.assertNotIn("edits", app.session_state)
            from app import combined_report
            states = {"image_digest": "test", "reference": {"ok": True, "result": comparison}}
            with patch("app.st.session_state", states):
                report = combined_report("PNG", (64, 64))
            self.assertTrue(report["checks"]["reference"]["result"]["tonal_comparison"]["notable_colour_difference"])

    def test_removed_edit_section_cannot_run_or_reappear(self):
        with patch("streamlit.file_uploader", return_value=upload("blue")), patch.dict("sys.modules", {"edit_detector": None, "torch": None}), patch("detector.load_detector", return_value=object()), patch(
            "detector.analyze_pixels", return_value={"ai_score": .98, "signal": "test"}
        ) as generation:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.session_state["edits"] = {"ok": True, "result": {"score": .99}}
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertNotIn("edits", app.session_state)
            self.assertNotIn("analyze_edits", [item.key for item in app.button])
            self.assertEqual([tab.label for tab in app.tabs], ["AI detection", "Compare original", "File details"])
            app.button(key="analyze_generation").click().run(timeout=30)
            app.run(timeout=30)
            self.assertEqual(generation.call_count, 1)
            self.assertEqual(app.metric[0].value, "0.980000")

    def test_report_contains_qualified_answers_without_removed_feature(self):
        from app import combined_report
        states = {"image_digest": "test", "generation": {"ok": True, "result": {"ai_score": .98}}, "edits": {"ok": True, "result": {"score": .99}}}
        with patch("app.st.session_state", states):
            report = combined_report("JPEG", (64, 64))
        self.assertIn("Likely AI", report["checks"]["generation"]["result"]["conclusion"])
        self.assertEqual(set(report["checks"]), {"generation", "reference"})

    def test_independent_results_survive_rerun_and_new_image_clears_them(self):
        current = [upload("blue")]

        with patch("streamlit.file_uploader", side_effect=lambda *args, **kw: current[0]), patch(
            "detector.load_detector", return_value=object()
        ), patch("detector.analyze_pixels", return_value={"ai_score": .0054, "signal": "Low AI-like signal"}) as generation:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.button(key="analyze_generation").click().run(timeout=30)
            app.run(timeout=30)  # same rerun triggered by a download or panel action
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 1)
            self.assertEqual(app.metric[0].value, "0.005400")
            self.assertEqual(generation.call_count, 1)
            current[0] = upload("red")
            app.run(timeout=30)
            self.assertFalse(app.metric)
            self.assertNotIn("generation", app.session_state)
            self.assertNotIn("edits", app.session_state)
            current[0] = None
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["image_digest"])

    def test_changed_reference_clears_only_comparison(self):
        current_reference = [upload("white")]

        def uploader(*args, **kwargs):
            return current_reference[0] if kwargs.get("key") == "reference_upload" else upload("blue")

        comparison = {"aligned_reference": Image.new("RGB", (64, 64)), "candidate": Image.new("RGB", (64, 64)),
                      "overlay": Image.new("RGB", (64, 64)), "difference": Image.new("L", (64, 64)),
                      "changed_fraction": .1, "overlap_fraction": .9, "analysis_size": (64, 64), "alignment": {}}
        with patch("streamlit.file_uploader", side_effect=uploader), patch(
            "detector.load_detector", return_value=object()
        ), patch("detector.analyze_pixels", return_value={"ai_score": .5, "signal": "Inconclusive"}), patch(
            "reference_compare.compare_reference", return_value=comparison
        ) as compare:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.button(key="analyze_generation").click().run(timeout=30)
            app.checkbox(key="has_reference").check().run(timeout=30)
            app.button(key="compare_reference").click().run(timeout=30)
            self.assertEqual(compare.call_count, 1)
            self.assertIn("reference", app.session_state)
            current_reference[0] = upload("green")
            app.run(timeout=30)
            self.assertNotIn("reference", app.session_state)
            self.assertIn("generation", app.session_state)
            self.assertEqual(app.metric[0].value, "0.500000")
            self.assertFalse(app.exception)
