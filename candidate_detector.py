"""Experimental ONNX candidate; compare before replacing the SDXL baseline.

Attribution and preprocessing: https://huggingface.co/s-gehring/detectra-openfake
Weights: CC-BY-NC-4.0. No target-data accuracy or calibration is established.
"""

import math

CANDIDATE_ID = "s-gehring/detectra-openfake"
CANDIDATE_REVISION = "c84565e"


def sigmoid(value):
    if not math.isfinite(value):
        raise ValueError("Non-finite model output")
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def prepare_pixels(image):
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    # Model card: shorter edge 440, bilinear; center crop 384; ImageNet norm.
    if width <= height:
        size = (440, int(440 * height / width))
    else:
        size = (int(440 * width / height), 440)
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 384) / 2), round((size[1] - 384) / 2)
    crop = resized.crop((left, top, left + 384, top + 384))
    pixels = np.asarray(crop, dtype=np.float32) / np.float32(255)
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None])


def load_candidate():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CANDIDATE_ID, "model.onnx", revision=CANDIDATE_REVISION)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def candidate_score(image, session):
    import numpy as np

    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "pixel_values":
        raise ValueError("Unexpected candidate input schema")
    output = np.asarray(session.run(["logits"], {"pixel_values": prepare_pixels(image)})[0])
    if output.size != 1:
        raise ValueError("Unexpected candidate output schema")
    logit = float(output.reshape(-1)[0])
    return {
        "ai_score": sigmoid(logit),
        "author_scaled_score": sigmoid(.552 * logit + .1933),
    }
