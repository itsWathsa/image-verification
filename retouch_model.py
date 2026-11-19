"""Research classifier for photographic retouching, separate from AI and TruFor.

The positive class is a dataset retouch/tonal adjustment, not image fraud.
No candidate is used in the website unless separately accepted and documented.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from enhancement_model import enhancement_features

ARCHITECTURE = "fivek-paired-cnn-frequency-v1"
IMAGE_SIZE = 64
MODEL_PATH = Path(__file__).resolve().parent / ".models/fivek/retouch.pt"


def image_tensor(image):
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255


class RetouchClassifier(nn.Module):
    def __init__(self, features=272):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(24 * 4 * 4, 32), nn.ReLU(), nn.Dropout(.25))
        self.frequency_branch = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Dropout(.35),
                                             nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 24), nn.ReLU(), nn.Dropout(.25), nn.Linear(24, 1))

    def forward(self, images, features):
        return self.head(torch.cat((self.image_branch(images), self.frequency_branch(features)), dim=1)).squeeze(1)


def load_retouch_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("positive_class") != "photographic_adjustment":
        raise ValueError("Unsupported retouching checkpoint.")
    model = RetouchClassifier(checkpoint["features"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def retouch_score(image, loaded):
    model, checkpoint = loaded
    features = torch.from_numpy(enhancement_features(image))
    normalized = ((features - checkpoint["mean"]) / checkpoint["std"]).clamp(-5, 5)
    with torch.inference_mode():
        return float(model(image_tensor(image)[None], normalized[None]).sigmoid().item())
