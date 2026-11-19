"""Reproducible COVERAGE processed-release download and grouped evaluation data.

The original author source is https://github.com/wenbihan/coverage. This mirror
contains the ForgeryVCR 832-square edition, not native camera TIFFs. This
downloader uses viewer assets, which can be JPEG-reencoded, then saves PNGs.
It is an exploration cache, NOT a verified native forensic benchmark loader.
Research/noncommercial use only; cite Wen et al., ICIP 2016 and the mirror.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".models/coverage"
DATASET = "ThaneJoss/COVERAGE"
SEED = 20261119


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(12 * 1024**2 + 1)
            if len(data) > 12 * 1024**2:
                raise ValueError("Unexpected large COVERAGE asset")
            return data
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503} or attempt == 2:
                raise
            delay = min(60, max(5, int(error.headers.get("Retry-After", "10"))))
            time.sleep(delay)


def split_pairs(ids):
    unique = sorted(set(str(value) for value in ids), key=int)
    if len(unique) != 100:
        raise ValueError("Expected 100 COVERAGE original-photo IDs")
    random.Random(SEED).shuffle(unique)
    return {value: "test" if i < 20 else "validation" if i < 40 else "development" for i, value in enumerate(unique)}


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in (0, 100):
        path = CACHE / f"rows_{offset}.json"
        if not path.exists():
            query = {"dataset": DATASET, "config": "coverage", "split": "full", "offset": offset, "length": 100}
            path.write_bytes(fetch("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)))
        response = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(item["row"] for item in response["rows"])
    if len(rows) != 200:
        raise ValueError("Unexpected mirror inventory")
    splits = split_pairs(row["filename_source_id"] for row in rows)
    destination = CACHE / "images"
    destination.mkdir(exist_ok=True)
    def save(row):
        photo_id, label = str(row["filename_source_id"]), int(row["label"])
        if not photo_id.isdigit() or label not in {0, 1}:
            raise ValueError("Invalid photo ID or label")
        stem = photo_id + ("_fake" if label else "_real")
        assets = {}
        for kind in ("image", "mask"):
            if row[kind] is None:
                continue
            path = destination / (stem + ("_mask" if kind == "mask" else "") + ".png")
            if not path.exists():
                data = fetch(row[kind]["src"])
                with Image.open(io.BytesIO(data)) as decoded:
                    if decoded.size != (832, 832):
                        raise ValueError("Mirror preprocessing changed; inspect before using")
                    decoded.convert("L" if kind == "mask" else "RGB").save(path)
            assets[kind] = path.relative_to(CACHE).as_posix()
            assets[kind + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"photo_id": photo_id, "label": label, "split": splits[photo_id], **assets}
    records = []
    with ThreadPoolExecutor(4) as pool:
        for index, record in enumerate(pool.map(save, rows), 1):
            records.append(record)
            if index % 20 == 0:
                print(f"Saved {index}/200 COVERAGE images and masks", flush=True)
    if len({(row["photo_id"], row["label"]) for row in records}) != 200:
        raise ValueError("Missing original/forged pair")
    # Exclude any identical decoded image families that cross split boundaries.
    hashes = {}
    for row in records:
        with Image.open(CACHE / row["image"]) as image:
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(row)
    if any(len({row["split"] for row in group}) > 1 for group in hashes.values()):
        raise ValueError("Decoded image duplication across partitions; regroup before evaluation")
    report = {"source": "https://github.com/wenbihan/coverage", "mirror": "https://huggingface.co/datasets/" + DATASET,
              "edition": "ForgeryVCR 832x832 dataset-viewer assets, potentially JPEG-reencoded, saved as PNG; NOT verified native TIFF pixels",
              "seed": SEED, "partitions": {split: len({row['photo_id'] for row in records if row['split'] == split})
                                              for split in ("development", "validation", "test")},
              "records": records, "terms": "Noncommercial research; Wen et al., ICIP 2016; ThaneJoss processed packaging"}
    (CACHE / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Complete; original/forged photo groups kept together.", flush=True)


if __name__ == "__main__":
    download()
