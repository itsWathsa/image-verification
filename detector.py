"""Experimental pixel-only SDXL baseline; no EXIF fields enter the model."""

import argparse
import math
from pathlib import Path

from PIL import Image

MODEL_ID = "Organika/sdxl-detector"
MODEL_REVISION = "b37fede"
CACHE_DIR = Path(__file__).resolve().parent / ".models"


def load_detector(download=False):
    # Check this explicitly: torch._dynamo can swallow its DLL ImportError and
    # replace it with an unrelated-looking NP_SUPPORTED_MODULES import failure.
    import numpy.fft  # noqa: F401
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    options = dict(
        revision=MODEL_REVISION,
        cache_dir=str(CACHE_DIR),
        local_files_only=not download,
        trust_remote_code=False,
    )
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **options)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, use_safetensors=True, **options
    )
    labels = {int(key): value.lower() for key, value in model.config.id2label.items()}
    if set(labels.values()) != {"artificial", "human"}:
        raise ValueError("Unexpected model labels; inference stopped.")
    model.to("cpu").eval()
    return processor, model, torch, labels


def interpret_score(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score.")
    # Exploratory bands only, not thresholds validated on the user's data.
    if score >= 0.9:
        return "AI-like signal (experimental)"
    if score <= 0.1:
        return "Camera-photo-like signal (experimental)"
    return "Inconclusive"


def analyze_pixels(image, detector):
    processor, model, torch, labels = detector
    # Make a new RGB image so EXIF and other metadata cannot enter inference.
    rgb = image.convert("RGB")
    pixels = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    inputs = processor(images=pixels, return_tensors="pt")
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]
    ai_index = next(index for index, name in labels.items() if name == "artificial")
    score = float(probabilities[ai_index].item())
    return {"ai_score": score, "signal": interpret_score(score)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download pinned model files into .models")
    args = parser.parse_args()
    if args.download:
        load_detector(download=True)
        print(f"Model ready: {MODEL_ID} @ {MODEL_REVISION}")
    else:
        parser.print_help()
