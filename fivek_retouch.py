"""Train/evaluate a research retouch classifier on FiveK pairs and tonal variants.

Archive is the sRGB FiveK release linked by the Retinexformer authors, kept in
.models/fivek/source.zip with its provenance and original research licenses.
Dataset inputs are rendered photos, not certified untouched camera JPEGs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn

from columbia_enhancement import adjusted, adjustment_metrics, exported, write_json
from enhancement_model import enhancement_features
from retouch_model import ARCHITECTURE, IMAGE_SIZE, MODEL_PATH, RetouchClassifier, load_retouch_model, retouch_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".models/fivek"
SEED = 20261119
ARCHIVE_SHA256 = "1c2c79f92c103c337517c8fff32ecc434d08c0287ecf49325447344945d05251"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR&export=download&confirm=t"


def download():
    """Explicit research-data download; never called by the Streamlit app."""
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / "source.zip"
    if not destination.exists():
        partial = CACHE / "source.zip.part"
        size = 0
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=90) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 700 * 1024**2:
                    raise ValueError("Unexpected archive size; download stopped.")
                output.write(chunk)
        if size != 658823577:
            raise ValueError("Incomplete or changed dataset archive.")
        partial.replace(destination)
    with destination.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Source archive checksum mismatch; inspect the release before use.")
    for filename in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt", "filesAdobe.txt", "filesAdobeMIT.txt"):
        with urllib.request.urlopen("https://data.csail.mit.edu/graphics/fivek/legal/" + filename, timeout=60) as response:
            (CACHE / filename).write_bytes(response.read())
    write_json(CACHE / "provenance.json", {"source_release": "https://github.com/caiyuanhao1998/Retinexformer",
               "download": DOWNLOAD_URL, "original_dataset": "https://data.csail.mit.edu/graphics/fivek/",
               "bytes": destination.stat().st_size, "sha256": digest})
    print("Verified FiveK archive and retained original research licenses.", flush=True)


def assign_splits(names, seed=SEED):
    """Use the release's official 500 test IDs; hold out 500 training IDs."""
    train = sorted(name for name in names if name.startswith("FiveK/train/input/") and name.endswith(".jpg"))
    test = sorted(name for name in names if name.startswith("FiveK/test/input/") and name.endswith(".jpg"))
    if len(train) != 4500 or len(test) != 500:
        raise ValueError("Expected the 4500/500 paired FiveK release.")
    ids = [PurePosixPath(name).name for name in train + test]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated photo IDs in the source split.")
    random.Random(seed).shuffle(train)
    validation = set(train[:500])
    return {name: "test" if name in test else "validation" if name in validation else "train" for name in train + test}


def synthetic_adjustment(image, rng):
    """Explicit extra labels for global adjustments; no AI generation involved."""
    kind = rng.choice(("brightness", "contrast", "gamma", "saturation", "combined"))
    strength = rng.choice((.65, .8, 1.2, 1.4))
    if kind == "saturation":
        return ImageEnhance.Color(image).enhance(strength), kind, strength
    if kind == "combined":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.25)).enhance(1.2), kind, [1.25, 1.2]
    return adjusted(image, kind, strength), kind, strength


def prepare():
    if (CACHE / "prepared.npz").exists():
        raise FileExistsError("Prepared experiment already exists; preserve the split.")
    thumbnails, features, labels, records = [], [], [], []
    rng = random.Random(SEED)
    unchanged, mismatched = [], []
    duplicates = []
    with zipfile.ZipFile(CACHE / "source.zip") as archive:
        split_map = assign_splits(archive.namelist())
        # Inspect decoded source pixels before training. Exact duplicate families
        # crossing partitions are excluded completely, rather than leaked.
        sources = sorted(split_map)
        digest_names = {}
        for name in sources:
            pair_hashes = []
            pair_sizes = []
            for path in (name, name.replace("/input/", "/target/")):
                with Image.open(io.BytesIO(archive.read(path))) as decoded:
                    image = decoded.convert("RGB")
                    pair_sizes.append(image.size)
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                digest_names.setdefault(digest, []).append(name)
                pair_hashes.append(digest)
            if pair_hashes[0] == pair_hashes[1]:
                unchanged.append(name)
            if pair_sizes[0] != pair_sizes[1]:
                mismatched.append(name)
        excluded = set(unchanged + mismatched)
        for digest, names in digest_names.items():
            families = set(names)
            if len(families) > 1:
                duplicates.append({"sha256": digest, "photo_ids": sorted(families)})
                if len({split_map[name] for name in families}) > 1:
                    excluded.update(families)
        for index, name in enumerate(sources, 1):
            if name in excluded:
                continue
            with Image.open(io.BytesIO(archive.read(name))) as decoded:
                original = decoded.convert("RGB")
            with Image.open(io.BytesIO(archive.read(name.replace("/input/", "/target/")))) as decoded:
                target = decoded.convert("RGB")
            if original.size != target.size:
                raise ValueError("Pair dimensions differ: " + name)
            modified, operation, strength = synthetic_adjustment(original, rng)
            # Re-export ALL examples using the same pair-specific quality.
            # Paired controls are repetitions, never independent photo samples.
            quality = rng.randint(80, 98)
            for label, version, kind in ((0, original, "input"), (1, target, "expert_retouch"),
                                         (0, original, "input_control"), (1, modified, "synthetic_" + operation)):
                image = exported(version, quality)
                thumbnails.append(np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)).copy())
                features.append(enhancement_features(image))
                labels.append(label)
                records.append({"photo_id": PurePosixPath(name).name, "split": split_map[name], "label": label,
                                "kind": kind, "quality": quality, "strength": strength if kind.startswith("synthetic") else None})
            if index % 250 == 0:
                print(f"Prepared {index}/5000 photo pairs", flush=True)
    np.savez(CACHE / "prepared.npz", images=np.stack(thumbnails), features=np.stack(features), labels=np.asarray(labels, dtype=np.float32))
    write_json(CACHE / "manifest.json", records)
    write_json(CACHE / "preparation.json", {
        "seed": SEED, "image_size": IMAGE_SIZE, "source": json.loads((CACHE / "provenance.json").read_text()),
        "duplicates": duplicates, "unchanged_pairs": unchanged, "mismatched_size_pairs": mismatched, "excluded_photo_ids": sorted(excluded),
        "partitions": {split: {"photo_ids": len({r["photo_id"] for r in records if r["split"] == split}),
                               "derived_examples": sum(r["split"] == split for r in records)}
                       for split in ("train", "validation", "test")},
        "caution": "Rendered FiveK inputs versus expert targets plus explicit synthetic tonal variants; not a universal edit-history ground truth. Near-duplicate scenes are not certified absent."})
    print(json.dumps(json.loads((CACHE / "preparation.json").read_text())["partitions"]), flush=True)


def prepared():
    arrays = np.load(CACHE / "prepared.npz")
    records = json.loads((CACHE / "manifest.json").read_text())
    return torch.from_numpy(arrays["images"]).permute(0, 3, 1, 2).float() / 255, torch.from_numpy(arrays["features"]), torch.from_numpy(arrays["labels"]), records


def scores_for(model, images, features, indices, batch=256):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(images[i], features[i]).sigmoid() for i in indices.split(batch)]).numpy()


def measured(labels, scores):
    return adjustment_metrics([{"label": "generated" if label else "camera", "ai_score": float(score)}
                               for label, score in zip(labels, scores)])


def operating_point(labels, scores, threshold):
    positive = labels > .5
    predicted = scores >= threshold
    tp = int((positive & predicted).sum()); fp = int((~positive & predicted).sum())
    tn = int((~positive & ~predicted).sum()); fn = int((positive & ~predicted).sum())
    return {"threshold": threshold, "true_adjustments": tp, "false_adjustments": fp,
            "true_controls": tn, "missed_adjustments": fn,
            "adjustment_recall": tp / (tp + fn), "control_false_positive_rate": fp / (fp + tn),
            "precision": tp / (tp + fp) if tp + fp else None}


def train(epochs=100):
    if MODEL_PATH.exists():
        raise FileExistsError("Checkpoint exists; do not silently overwrite a completed experiment.")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    images, raw, labels, records = prepared()
    train_idx = torch.tensor([i for i, row in enumerate(records) if row["split"] == "train"])
    validation_idx = torch.tensor([i for i, row in enumerate(records) if row["split"] == "validation"])
    mean, std = raw[train_idx].mean(0), raw[train_idx].std(0).clamp_min(.0001)
    features = ((raw - mean) / std).clamp(-5, 5)
    model = RetouchClassifier(raw.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.02)
    best_loss, best_epoch, history = float("inf"), 0, []
    start = time.perf_counter()
    # Indices contain adjacent control/expert and control/synthetic pairs.
    pair_ids = train_idx.reshape(-1, 2)
    for epoch in range(1, epochs + 1):
        model.train()
        order = pair_ids[torch.randperm(len(pair_ids), generator=rng)].reshape(-1)
        total = 0.
        for index in order.split(128):
            inputs = images[index]
            if torch.rand((), generator=rng) < .5:
                inputs = inputs.flip(-1)
            logits = model(inputs, features[index])
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels[index])
            paired = logits.reshape(-1, 2)
            ranking = nn.functional.relu(.5 - (paired[:, 1] - paired[:, 0])).mean()
            loss = bce + .15 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.item()) * len(index)
        scores = scores_for(model, images, features, validation_idx)
        y = labels[validation_idx].numpy()
        validation_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(scores).clamp(1e-7, 1-1e-7), torch.from_numpy(y)).item())
        metrics = measured(y, scores)
        history.append({"epoch": epoch, "train_loss": total / len(train_idx), "validation_loss": validation_loss,
                        "validation_accuracy": metrics["accuracy"], "validation_auc": metrics["roc_auc"]})
        if validation_loss < best_loss - .0001:
            best_loss, best_epoch = validation_loss, epoch
            torch.save({"architecture": ARCHITECTURE, "positive_class": "photographic_adjustment", "features": raw.shape[1],
                        "state_dict": model.state_dict(), "mean": mean, "std": std, "seed": SEED,
                        "best_epoch": best_epoch, "image_size": IMAGE_SIZE}, MODEL_PATH)
        print(f"Epoch {epoch}: val accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}, loss={validation_loss:.4f}, best={best_epoch}, elapsed={time.perf_counter()-start:.0f}s", flush=True)
        write_json(CACHE / "history.json", history)
        if epoch - best_epoch >= 15:
            break
    # Freeze an operating point on validation controls BEFORE test evaluation.
    model, checkpoint = load_retouch_model()
    scores = scores_for(model, images, features, validation_idx)
    y = labels[validation_idx].numpy()
    threshold = float(np.quantile(scores[y == 0], .95, method="higher"))
    write_json(CACHE / "selection.json", {"best_epoch": best_epoch, "validation_loss": best_loss,
               "validation_metrics": measured(y, scores), "operating_point": operating_point(y, scores, threshold),
               "selection": "Minimum validation BCE, early stopping patience 15; high-adjustment threshold validation-control 95th percentile.",
               "training_seconds": time.perf_counter()-start, "parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()})


def evaluate_candidate():
    if (CACHE / "evaluation.json").exists():
        raise FileExistsError("Test already evaluated; preserve the original result.")
    torch.set_num_threads(4)
    model, checkpoint = load_retouch_model()
    images, raw, labels, records = prepared()
    features = ((raw - checkpoint["mean"]) / checkpoint["std"]).clamp(-5, 5)
    selection = json.loads((CACHE / "selection.json").read_text())
    threshold = selection["operating_point"]["threshold"]
    report = {"architecture": ARCHITECTURE, "selection": selection, "test": {}}
    for title, kinds in (("expert_retouch", {"input", "expert_retouch"}),
                         ("synthetic_adjustments", {row["kind"] for row in records if row["kind"].startswith("synthetic")} | {"input_control"}),
                         ("combined", {row["kind"] for row in records})):
        index = torch.tensor([i for i, row in enumerate(records) if row["split"] == "test" and row["kind"] in kinds])
        scores = scores_for(model, images, features, index)
        y = labels[index].numpy()
        report["test"][title] = {"metrics": measured(y, scores), "operating_point": operating_point(y, scores, threshold)}
    # Never train on the user's images. These are small, previously seen diagnostics.
    diagnostics = []
    for name, label in (("raw1.jpg", 0), ("raw2.jpg", 0), ("raw3.jpg", 0), ("raw 1 edited.jpg", 1)):
        path = ROOT / "data" / name
        if path.exists():
            with Image.open(path) as image:
                score = retouch_score(image.convert("RGB"), (model, checkpoint))
            diagnostics.append({"filename": name, "adjusted": label, "score": score, "flagged": score >= threshold})
    report["private_diagnostics"] = diagnostics
    # A source-transfer check, not additional threshold-fitting data.
    external = sorted((ROOT / ".models/columbia/originals").glob("*.png"))
    if external:
        control_scores, adjustment_scores = [], []
        for path in external:
            with Image.open(path) as image:
                original = image.convert("RGB")
            control_scores.append(retouch_score(original, (model, checkpoint)))
            modified = ImageEnhance.Contrast(ImageEnhance.Brightness(original).enhance(1.25)).enhance(1.2)
            adjustment_scores.append(retouch_score(modified, (model, checkpoint)))
        all_scores = np.asarray(control_scores + adjustment_scores)
        y = np.asarray([0.] * len(control_scores) + [1.] * len(adjustment_scores))
        report["external_columbia_combined_tonal_check"] = {"source_photos": len(external), "metrics": measured(y, all_scores),
                                                           "operating_point": operating_point(y, all_scores, threshold)}
    report["production_enabled"] = False
    write_json(CACHE / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "train", "evaluate"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.command == "download": download()
    elif args.command == "prepare": prepare()
    elif args.command == "train": train(args.epochs)
    else: evaluate_candidate()
