"""Bounded video-frame screening using the existing image model, not a video classifier."""
import hashlib
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from detector import ANALYSIS_METHOD, MODEL_ID, MODEL_REVISION, analyze_pixels

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_DURATION_SECONDS = 600
MAX_VIDEO_PIXELS = 4096 * 2160
SAMPLE_COUNT = 12
VIDEO_METHOD = "uniform-frame-screen-v1:" + ANALYSIS_METHOD


def analyze_video(data, session, progress=None):
    if not data or len(data) > MAX_VIDEO_BYTES:
        raise ValueError("Choose a nonempty video no larger than 100 MB.")
    # Closed before OpenCV opens it, including on Windows. Removed on every exit.
    with tempfile.TemporaryDirectory(prefix="video-review-") as folder:
        path = Path(folder) / "upload.video"
        path.write_bytes(data)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Cannot decode this video. Try an MP4 encoded with H.264.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
                raise ValueError("This video's duration or dimensions could not be read.")
            duration = count / fps
            if duration > MAX_DURATION_SECONDS:
                raise ValueError(
                    f"The video decoder reports {duration:.1f} seconds "
                    f"({int(count):,} frames at {fps:.3f} fps), exceeding the 10-minute limit. "
                    "If your player shows a shorter clip, its timing metadata may be incompatible. "
                    "Re-export it as an H.264 MP4 at a fixed frame rate and try again."
                )
            if width * height > MAX_VIDEO_PIXELS:
                raise ValueError(
                    f"The video resolution is {int(width)} x {int(height)} "
                    f"({width * height / 1_000_000:.2f} megapixels per frame), exceeding the "
                    "8.85-megapixel limit. Resize to 4096 x 2160 or smaller "
                    "(2160 x 4096 for portrait video) and try again."
                )
            indices = np.unique(np.linspace(0, int(count) - 1, min(SAMPLE_COUNT, int(count)), dtype=int))
            records, failed, preview = [], [], None
            started = time.monotonic()
            for step, index in enumerate(indices):
                if time.monotonic() - started > 120:
                    raise ValueError("Video analysis took too long. Try a shorter or smaller video.")
                if index != 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    failed.append(int(index))
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed.append(int(index))
                    continue
                if frame.shape[0] * frame.shape[1] > MAX_VIDEO_PIXELS:
                    raise ValueError(f"Decoded frame resolution is {frame.shape[1]} x {frame.shape[0]}, exceeding the 8.85-megapixel limit. Resize to 4K or smaller and try again.")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = analyze_pixels(image, session)
                score = float(result["ai_score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid frame score.")
                records.append({"frame_index": int(index), "timestamp_seconds": float(index / fps), "ai_score": score})
                if preview is None or score > preview[0]:
                    image.thumbnail((640, 640))
                    preview = (score, image.copy(), float(index / fps))
                if progress:
                    progress((step + 1) / len(indices))
            if not records or len(records) != len(indices):
                raise ValueError("Some sampled frames could not be decoded. No complete assessment is available; try re-exporting the video as MP4.")
            scores = [item["ai_score"] for item in records]
            high = sum(score >= .9 for score in scores)
            if high:
                conclusion = "AI-like visual signals found in sampled frames. This does not establish that the video is AI-generated."
            elif all(score <= .1 for score in scores):
                conclusion = "Low AI-like signal in sampled frames. This does not confirm the video is authentic."
            else:
                conclusion = "Unable to determine AI involvement from the sampled frames."
            report = {"video_sha256": hashlib.sha256(data).hexdigest(), "analysis_method": VIDEO_METHOD,
                      "model": MODEL_ID, "revision": MODEL_REVISION, "duration_seconds": duration,
                      "dimensions": [int(width), int(height)], "sample_count": len(records),
                      "mean_frame_score": sum(scores) / len(scores), "max_frame_score": max(scores),
                      "high_signal_frames": high, "frames": records, "conclusion": conclusion,
                      "notice": "Experimental image-model screening of sampled frames; not validated for video. Scores are not probabilities. Audio, motion and unsampled frames are not assessed."}
            return {"report": report, "preview": preview[1], "preview_timestamp": preview[2]}
        finally:
            capture.release()
