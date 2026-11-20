import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from video_analysis import analyze_video


def clip():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "test.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
        if not writer.isOpened():
            raise RuntimeError("MJPG test encoder unavailable")
        try:
            for i in range(20):
                writer.write(np.full((64, 64, 3), i * 10, np.uint8))
        finally:
            writer.release()
        return path.read_bytes()


class VideoTests(unittest.TestCase):
    def test_real_decode_sampling_scores_and_cleanup(self):
        scores = [.01] * 11 + [.99]
        progress = []
        with patch("video_analysis.analyze_pixels", side_effect=[{"ai_score": s} for s in scores]) as analyze:
            result = analyze_video(clip(), object(), progress.append)
        report = result["report"]
        self.assertEqual(analyze.call_count, 12)
        self.assertEqual(report["sample_count"], 12)
        self.assertAlmostEqual(report["mean_frame_score"], sum(scores)/12)
        self.assertEqual(report["high_signal_frames"], 1)
        self.assertEqual(report["frames"][0]["frame_index"], 0)
        self.assertEqual(report["frames"][-1]["frame_index"], 19)
        self.assertEqual(report["frames"][-1]["timestamp_seconds"], 1.9)
        self.assertIn("does not establish", report["conclusion"])
        self.assertEqual(progress[-1], 1)

    def test_low_and_inconclusive_never_claim_authenticity(self):
        for score, phrase in ((.01, "does not confirm"), (.5, "Unable to determine")):
            with patch("video_analysis.analyze_pixels", return_value={"ai_score": score}):
                self.assertIn(phrase, analyze_video(clip(), object())["report"]["conclusion"])

    def test_bad_upload_and_partial_decode_fail_closed(self):
        for data in (b"", b"not a video"):
            with self.assertRaises(ValueError):
                analyze_video(data, object())
        with patch("video_analysis.MAX_VIDEO_BYTES", 1), self.assertRaises(ValueError):
            analyze_video(b"12", object())
        with patch("video_analysis.cv2.VideoCapture") as factory:
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda key: {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 20, cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 64}[key]
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(ValueError, "could not be decoded"):
                analyze_video(b"video", object())
            capture.release.assert_called_once()
        with patch("video_analysis.MAX_DURATION_SECONDS", 1), self.assertRaisesRegex(ValueError, "10 minutes"):
            analyze_video(clip(), object())

    def test_video_ui_persists_and_new_upload_invalidates(self):
        uploaded = io.BytesIO(clip())
        uploaded.name = "test.avi"
        with patch("streamlit.file_uploader", return_value=uploaded), patch("detector.load_detector", return_value=object()), patch("video_analysis.analyze_pixels", return_value={"ai_score": .02}) as analyze:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.radio(key="media_type").set_value("Video").run(timeout=30)
            app.button(key="analyze_video").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "0.020000")
            app.run(timeout=30)
            self.assertEqual(analyze.call_count, 12)
            self.assertEqual(app.metric[0].value, "0.020000")
            uploaded.seek(0)
            uploaded.truncate()
            uploaded.write(b"changed")
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.metric)
            self.assertNotIn("video", app.session_state)
