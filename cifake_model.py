"""Small pixel classifier trained for CIFAKE's 32x32 benchmark domain.

This separate candidate does not calibrate or replace Detectra-OpenFake.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

ARCHITECTURE = "cifake-cnn-v1"
MODEL_PATH = Path(__file__).resolve().parent / ".models" / "cifake" / "candidate.pt"


class CifakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 64),
                                        nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 1))

    def forward(self, pixels):
        return self.classifier(self.features(pixels)).squeeze(1)


def prepare_cifake_pixels(image):
    rgb = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    pixels = np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return torch.from_numpy(np.ascontiguousarray(pixels))[None]


def load_cifake_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("generated_class") != 1:
        raise ValueError("Unsupported CIFAKE checkpoint schema or class mapping.")
    model = CifakeCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def cifake_score(image, model):
    with torch.inference_mode():
        score = float(model(prepare_cifake_pixels(image)).sigmoid().item())
    if not np.isfinite(score):
        raise ValueError("Non-finite CIFAKE model output.")
    return score
