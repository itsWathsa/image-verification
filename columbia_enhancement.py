"""Grouped Columbia-original tonal-adjustment experiment, separate from TruFor.

Requires the author's source.zip in .models/columbia. Synthetic adjustment
labels are generated here; Columbia's spliced images are NOT tonal-edit labels.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import time
import zipfile

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn

from enhancement_model import ARCHITECTURE, EnhancementMLP, enhancement_features, enhancement_score, load_enhancement_model
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".models/columbia"
SEED = 20261119
OPERATIONS = [(kind, strength) for kind in ("brightness", "contrast") for strength in (.65, .8, 1.2, 1.4)]
OPERATIONS += [("gamma", strength) for strength in (.7, .85, 1.2, 1.5)]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def adjustment_metrics(records):
    """Reuse binary metrics while keeping this task distinct from AI detection."""
    measured = evaluate(records)
    matrix = measured["confusion_matrix"]
    return {"count": measured["count"], "adjusted_count": measured["generated_count"],
            "control_count": measured["camera_count"], "forced_cutoff": .5,
            "accuracy": measured["accuracy"], "adjustment_precision": measured["ai_precision"],
            "adjustment_recall": measured["ai_recall"], "adjustment_f1": measured["ai_f1"],
            "control_false_positive_rate": measured["camera_false_positive_rate"], "roc_auc": measured["roc_auc"],
            "confusion_matrix": {"true_adjustments": matrix["true_ai"], "false_adjustments": matrix["false_ai"],
                                 "true_controls": matrix["true_camera"], "missed_adjustments": matrix["missed_ai"]}}


def source_family(name):
    base = PurePosixPath(name).stem.lower()
    return re.sub(r"_sub_?\d+$", "", base)


def split_families(groups, seed=SEED):
    unique = sorted(set(groups))
    if len(unique) < 10:
        raise ValueError("Too few original-photo families for three-way evaluation.")
    random.Random(seed).shuffle(unique)
    holdout = max(2, round(len(unique) * .2))
    mapping = {group: "test" if i < holdout else "validation" if i < 2 * holdout else "train"
               for i, group in enumerate(unique)}
    return mapping


def adjusted(image, kind, strength):
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(strength)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(strength)
    if kind == "gamma":
        table = [round(255 * (i / 255) ** strength) for i in range(256)]
        return image.point(table * 3)
    raise ValueError("Unsupported tonal adjustment")


def exported(image, quality):
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=quality)
    with Image.open(io.BytesIO(data.getvalue())) as decoded:
        return decoded.convert("RGB")


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "features.npz").exists():
        raise FileExistsError("Prepared experiment exists; preserve the frozen split.")
    originals = CACHE / "originals"
    originals.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile(CACHE / "source.zip") as bundle:
        (CACHE / "README.txt").write_bytes(bundle.read("README.txt"))
        with bundle.open("4cam_auth.tar.bz2") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|bz2") as archive:
                for member in archive:
                    name = PurePosixPath(member.name).name
                    if not member.isfile() or "edgemask" in name.lower() or "mask" in name.lower():
                        continue
                    if PurePosixPath(name).suffix.lower() not in {".tif", ".tiff", ".bmp", ".jpg", ".png"}:
                        continue
                    if member.size > 40 * 1024**2:
                        raise ValueError("Unexpected large source image")
                    with Image.open(io.BytesIO(archive.extractfile(member).read())) as source:
                        if source.width * source.height > 20_000_000:
                            raise ValueError("Unexpected source dimensions")
                        image = source.convert("RGB")
                        path = originals / (PurePosixPath(name).stem + ".png")
                        image.save(path)
                    rows.append({"source": member.name, "file": path.name, "family": source_family(name)})
    if len(rows) != 183:
        write_json(CACHE / "source_inventory.json", rows)
        raise ValueError(f"Expected 183 originals, found {len(rows)}; inspect inventory before continuing.")
    partitions = split_families([row["family"] for row in rows])
    rng = random.Random(SEED)
    features, labels, records = [], [], []
    for index, row in enumerate(rows, 1):
        with Image.open(originals / row["file"]) as source:
            image = source.convert("RGB")
        for operation, strength in OPERATIONS:
            quality = rng.randint(65, 98)
            modified = adjusted(image, operation, strength)
            # A paired baseline has identical export/compression settings.
            # Repeated originals are controls, not independent new photographs.
            for label, version in ((0, image), (1, modified)):
                features.append(enhancement_features(exported(version, quality)))
                labels.append(label)
                records.append({**row, "split": partitions[row["family"]], "label": label,
                                "operation": operation if label else "none", "strength": strength if label else 1.,
                                "paired_operation": operation, "jpeg_quality": quality})
        if index % 20 == 0:
            print(f"Prepared {index}/183 source images", flush=True)
    np.savez(CACHE / "features.npz", x=np.stack(features), y=np.asarray(labels, dtype=np.float32))
    write_json(CACHE / "manifest.json", records)
    write_json(CACHE / "preparation.json", {
        "source_original_count": len(rows), "family_count": len(partitions), "seed": SEED,
        "source_archive_sha256": json.loads((CACHE / "download.json").read_text())["sha256"],
        "split_families": partitions,
        "partition_counts": {split: {"families": sum(value == split for value in partitions.values()),
                                     "source_images": sum(partitions[row["family"]] == split for row in rows),
                                     "derived_examples": sum(row["split"] == split for row in records)}
                             for split in ("train", "validation", "test")},
        "labels": "0 = no additional adjustment; 1 = synthetic brightness/contrast/gamma adjustment",
        "limitations": "Only 183 originals/four cameras; related subimages grouped by filename prefix before _sub. Cross-camera scene overlap and undeclared near duplicates not exhaustively audited. Paired controls are repeated originals, not independent photographs."
    })
    print(json.dumps(json.loads((CACHE / "preparation.json").read_text())["partition_counts"], indent=2), flush=True)


def train():
    if (CACHE / "enhancement.pt").exists():
        raise FileExistsError("Training already performed; preserve the experiment.")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = np.load(CACHE / "features.npz", allow_pickle=False)
    rows = json.loads((CACHE / "manifest.json").read_text())
    masks = {name: np.array([row["split"] == name for row in rows]) for name in ("train", "validation", "test")}
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    mean = x[masks["train"]].mean(0)
    std = x[masks["train"]].std(0).clamp_min(.0001)
    x = ((x - mean) / std).clamp(-5, 5)
    model = EnhancementMLP(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.1)
    best, stale, history = float("inf"), 0, []
    train_indices = torch.from_numpy(np.where(masks["train"])[0])
    for epoch in range(1, 201):
        model.train()
        order = train_indices[torch.randperm(len(train_indices))]
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x[masks["validation"]])
            loss = float(nn.functional.binary_cross_entropy_with_logits(logits, y[masks["validation"]]))
            accuracy = float(((logits.sigmoid() >= .5) == y[masks["validation"]].bool()).float().mean())
        history.append({"epoch": epoch, "validation_bce": loss, "validation_accuracy": accuracy})
        if loss < best:
            best, stale = loss, 0
            torch.save({"architecture": ARCHITECTURE, "features": x.shape[1], "state_dict": model.state_dict(),
                        "positive_class": "synthetic_tonal_adjustment",
                        "mean": mean, "std": std, "best_epoch": epoch, "seed": SEED,
                        "note": "Research candidate; not integrated into app. Validation-only checkpoint selection."}, CACHE / "enhancement.pt")
        else:
            stale += 1
        if epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
        if stale >= 25:
            break
    write_json(CACHE / "training_history.json", {"optimizer": "AdamW lr .0005 weight_decay .1", "patience": 25, "history": history})
    print("Training complete; checkpoint selected without test scores.", flush=True)


def evaluate_candidate():
    if (CACHE / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; do not tune repeatedly on the test set.")
    torch.set_num_threads(2)
    model, checkpoint = load_enhancement_model()
    data = np.load(CACHE / "features.npz", allow_pickle=False)
    rows = json.loads((CACHE / "manifest.json").read_text())
    with torch.inference_mode():
        normalized = ((torch.from_numpy(data["x"]) - checkpoint["mean"]) / checkpoint["std"]).clamp(-5, 5)
        scores = model(normalized).sigmoid().numpy()
    output = {"architecture": ARCHITECTURE, "best_epoch": checkpoint["best_epoch"], "splits": {}}
    for split in ("validation", "test"):
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        records = [{"label": "generated" if rows[i]["label"] else "camera", "ai_score": float(scores[i])} for i in indices]
        output["splits"][split] = adjustment_metrics(records)
    validation_negative = [float(scores[i]) for i, row in enumerate(rows) if row["split"] == "validation" and row["label"] == 0]
    threshold = max(.5, float(np.quantile(validation_negative, .95)))
    output["validation_selected_positive_threshold"] = threshold
    output["threshold_target"] = "Empirical validation negative 95th percentile; not a guaranteed 5% population false-positive rate."
    test_indices = [i for i, row in enumerate(rows) if row["split"] == "test"]
    output["test_at_selected_threshold"] = {
        "true_adjustments": int(sum(rows[i]["label"] == 1 and scores[i] >= threshold for i in test_indices)),
        "missed_adjustments": int(sum(rows[i]["label"] == 1 and scores[i] < threshold for i in test_indices)),
        "false_adjustments": int(sum(rows[i]["label"] == 0 and scores[i] >= threshold for i in test_indices)),
        "true_controls": int(sum(rows[i]["label"] == 0 and scores[i] < threshold for i in test_indices)),
    }
    loaded = (model, checkpoint)
    target = []
    for filename, label in (("raw1.jpg", 0), ("raw2.jpg", 0), ("raw3.jpg", 0), ("raw 1 edited.jpg", 1)):
        with Image.open(ROOT / "data" / filename) as image:
            score = enhancement_score(ImageOps.exif_transpose(image).convert("RGB"), loaded)
        target.append({"label": label, "score": score, "indication": score >= threshold})
    output["known_target_diagnostics"] = target
    output["limitations"] = "These are synthetic adjustments from a small splicing dataset's authentic images. No general-use improvement is established. The known phone examples were not used in training/selection. No AI-generation classifier or TruFor weights were trained in this experiment."
    write_json(CACHE / "evaluation.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate_candidate}[args.command]()
