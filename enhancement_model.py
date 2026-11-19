"""Research-only photographic-adjustment classifier; not an AI-use detector."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

ARCHITECTURE = "tonal-frequency-mlp-v2"
MODEL_PATH = Path(__file__).resolve().parent / ".models/columbia/enhancement.pt"


def enhancement_features(image):
    """Sample native RGB levels without interpolation or metadata features."""
    import cv2
    rgb = np.asarray(image.convert("RGB"))
    stride = max(1, int(np.ceil(np.sqrt(image.width * image.height / 262144))))
    sampled = np.ascontiguousarray(rgb[::stride, ::stride])
    gray = cv2.cvtColor(sampled, cv2.COLOR_RGB2GRAY)
    channels = [sampled[:, :, i].ravel() for i in range(3)] + [gray.ravel()]
    histograms = [np.bincount(channel, minlength=256).astype(np.float32) / len(channel) for channel in channels]
    frequencies = [np.abs(np.fft.rfft(hist))[1:65].astype(np.float32) for hist in histograms]
    summaries = np.asarray([[float(channel.mean()) / 255, float(channel.std()) / 255,
                            float((channel <= 1).mean()), float((channel >= 254).mean())]
                           for channel in channels], dtype=np.float32).ravel()
    features = np.concatenate([*frequencies, summaries])
    if not np.isfinite(features).all():
        raise ValueError("Invalid enhancement features.")
    return features


class EnhancementMLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 16), nn.ReLU(), nn.Dropout(.5),
                                    nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, features):
        return self.layers(features).squeeze(1)


def load_enhancement_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("positive_class") != "synthetic_tonal_adjustment":
        raise ValueError("Unsupported enhancement checkpoint.")
    model = EnhancementMLP(checkpoint["features"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def enhancement_score(image, loaded):
    model, checkpoint = loaded
    features = torch.from_numpy(enhancement_features(image))
    normalized = ((features - checkpoint["mean"]) / checkpoint["std"]).clamp(-5, 5)
    with torch.inference_mode():
        return float(model(normalized[None]).sigmoid().item())
