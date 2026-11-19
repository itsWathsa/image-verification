"""Experimental pixel-only Detectra-OpenFake ONNX detector.

Attribution and preprocessing: https://huggingface.co/s-gehring/detectra-openfake
Weights: CC-BY-NC-4.0. No target-data accuracy or calibration is established.
"""

import argparse
import math
from pathlib import Path

MODEL_ID = "s-gehring/detectra-openfake"
MODEL_REVISION = "c84565e"
CACHE_DIR = Path(__file__).resolve().parent / ".models"
ANALYSIS_METHOD = "single-center-v1"
DETAIL_METHOD = "conditional-center-rescan-v1"
RESCAN_EDGES = (576, 768)
RESCAN_MIN = .5
RESCAN_MAX = .95


def sigmoid(value):
    if not math.isfinite(value):
        raise ValueError("Non-finite model output")
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def prepare_pixels(image, short_edge=440):
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    if short_edge not in (440, *RESCAN_EDGES):
        raise ValueError("Unsupported detector rescan size.")
    # Same 384x384 input and normalization at each model-card center scale.
    if width <= height:
        size = (short_edge, int(short_edge * height / width))
    else:
        size = (int(short_edge * width / height), short_edge)
    if max(size) > 16000:
        raise ValueError("Image aspect ratio is too extreme for safe model preprocessing.")
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 384) / 2), round((size[1] - 384) / 2)
    crop = resized.crop((left, top, left + 384, top + 384))
    pixels = np.asarray(crop, dtype=np.float32) / np.float32(255)
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None])


def load_detector(download=False):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, "model.onnx", revision=MODEL_REVISION, cache_dir=str(CACHE_DIR), local_files_only=not download)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def analyze_pixels(image, session, detailed=False):
    import numpy as np

    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "pixel_values":
        raise ValueError("Unexpected detector input schema")
    def score_crop(edge):
        output = np.asarray(session.run(["logits"], {"pixel_values": prepare_pixels(image, edge)})[0])
        if output.size != 1:
            raise ValueError("Unexpected detector output schema")
        return sigmoid(float(output.reshape(-1)[0]))

    baseline = score_crop(440)
    crops = [{"short_edge": 440, "ai_score": baseline}]
    # Fixed, author-inspired gate, applied identically to every file. Raw
    # scores are retained; the author's Platt calibration is NOT applied.
    if detailed and RESCAN_MIN <= baseline < RESCAN_MAX:
        crops.extend({"short_edge": edge, "ai_score": score_crop(edge)} for edge in RESCAN_EDGES)
    score = max(crop["ai_score"] for crop in crops)
    return {"ai_score": score, "signal": interpret_score(score), "baseline_score": baseline,
            "crop_scores": crops, "analysis_method": DETAIL_METHOD if detailed else ANALYSIS_METHOD,
            "model": MODEL_ID, "revision": MODEL_REVISION}


def interpret_score(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score.")
    if score >= .9:
        return "AI-like signal (experimental)"
    if score <= .1:
        return "Low AI-like signal (experimental; camera origin unproven)"
    return "Inconclusive"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        load_detector(download=True)
        print(f"Model ready: {MODEL_ID} @ {MODEL_REVISION}")
    else:
        parser.print_help()
