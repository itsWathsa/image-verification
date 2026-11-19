"""TruFor CPU localization adapter, independent of generation classification."""

import hashlib
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request
import zipfile

from PIL import Image

WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
WEIGHTS_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
CACHE = Path(__file__).resolve().parent / ".models" / "trufor"
SOURCE_REVISION = "ae54475df6f41a491d7615100feb19263dec13f7"
MAX_EDGE = 384
_lock = threading.Lock()


def get_weights(download=False):
    path = CACHE / "trufor.pth.tar"
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError("TruFor weights are not downloaded.")
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "weights.zip.part"
    try:
        digest = hashlib.md5()
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as response, archive.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 400 * 1024 * 1024:
                    raise ValueError("TruFor download exceeded expected size.")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != WEIGHTS_MD5:
            raise ValueError("Official TruFor archive checksum mismatch.")
        with zipfile.ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist() if Path(name).name == "trufor.pth.tar"]
            if len(matches) != 1:
                raise ValueError("Unexpected TruFor archive layout.")
            temporary = CACHE / "model.part"
            with bundle.open(matches[0]) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(path)
    finally:
        archive.unlink(missing_ok=True)
    return path


class Options(dict):
    __getattr__ = dict.__getitem__


def load_edit_detector(download=False):
    import numpy as np
    import torch
    from trufor_vendor.models.cmx.builder_np_conf import myEncoderDecoder

    torch.set_num_threads(2)
    extra = Options(BACKBONE="mit_b2", DECODER="MLPDecoder", DECODER_EMBED_DIM=512,
                    PREPRC="imagenet", BN_EPS=.001, BN_MOMENTUM=.1,
                    DETECTION="confpool", CONF=True)
    config = SimpleNamespace(MODEL=SimpleNamespace(EXTRA=extra, MODS=("RGB", "NP++"), PRETRAINED=None),
                             DATASET=SimpleNamespace(NUM_CLASSES=2))
    # Official checkpoint includes NumPy scalar training metrics. Allow only
    # those primitive scalar/dtype constructors; retain weights-only loading.
    safe = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
            type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(get_weights(download), map_location="cpu", weights_only=True)
    model = myEncoderDecoder(cfg=config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def analyze_edits(image, model, max_edge=None):
    import numpy as np
    import torch

    original_size = image.size
    edge = MAX_EDGE if max_edge is None else max_edge
    if not isinstance(edge, int) or not 32 <= edge <= 768:
        raise ValueError("TruFor analysis edge must be an integer from 32 to 768.")
    analyzed = image.convert("RGB")
    analyzed.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    if min(analyzed.size) < 32:
        raise ValueError("Image is too small or narrow for TruFor (minimum 32 pixels per side).")
    # Official dataset preprocessing divides RGB bytes by 256, not 255.
    pixels = np.asarray(analyzed, dtype=np.float32).transpose(2, 0, 1).copy() / 256.
    with _lock, torch.inference_mode():
        pred, confidence, detection, _ = model(torch.from_numpy(pixels)[None])
        anomaly = pred.softmax(dim=1)[0, 1].cpu().numpy()
        reliability = confidence.sigmoid()[0, 0].cpu().numpy()
        score = float(detection.sigmoid().item())
    for array in (anomaly, reliability):
        if array.shape != (analyzed.height, analyzed.width) or not np.isfinite(array).all():
            raise ValueError("Invalid TruFor output map.")
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid TruFor score.")
    return {"image": analyzed, "map": anomaly, "reliability": reliability, "score": score,
            "original_size": original_size, "analyzed_size": analyzed.size, "assessment": "inconclusive"}


def render_maps(result):
    import numpy as np

    anomaly = result["map"]
    heat = np.zeros((*anomaly.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = np.round(anomaly * 255).astype(np.uint8)
    heat[:, :, 2] = np.round((1 - anomaly) * 255).astype(np.uint8)
    overlay = Image.blend(result["image"], Image.fromarray(heat), .45)
    confidence = Image.fromarray(np.round(result["reliability"] * 255).astype(np.uint8))
    return overlay, confidence


if __name__ == "__main__":
    load_edit_detector(download=True)
    print("TruFor CPU model ready.")
