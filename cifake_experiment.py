"""Download, audit, train and evaluate a separate CIFAKE candidate.

All dataset/cache/checkpoint artifacts stay under .models/cifake, ignored by Git.
The official test split is excluded from training and model selection.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from cifake_model import ARCHITECTURE, CifakeCNN, cifake_score, load_cifake_model
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".models" / "cifake"
REVISION = "e112a942abaecd02b6b1f6f646c807d56be8fb62"
DATASET_URL = f"https://codeload.github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/zip/{REVISION}"
SEED = 20261118


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset():
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "dataset.zip"
    if not archive.exists():
        partial = CACHE / "dataset.zip.part"
        print("Downloading pinned author dataset archive...", flush=True)
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response, partial.open("wb") as target:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 2 * 1024**3:
                    raise ValueError("Dataset download exceeds the 2 GiB safety limit.")
                target.write(chunk)
        partial.replace(archive)
    print(f"Dataset archive: {archive.stat().st_size:,} bytes", flush=True)
    return archive


def split_indices(hashes, labels, official_splits, seed=SEED):
    """Keep official test separate; remove exact overlap and conflicting labels.

    Each identical decoded RGB image has one representative. Near duplicates
    cannot be reliably identified from these exact hashes and remain a limitation.
    """
    groups = {}
    for index, digest in enumerate(hashes):
        groups.setdefault(str(digest), []).append(index)
    training_pool, testing, conflicts = [], [], 0
    for indices in groups.values():
        if len({int(labels[i]) for i in indices}) != 1:
            conflicts += len(indices)
            continue
        test_indices = [i for i in indices if official_splits[i] == "test"]
        if test_indices:
            testing.append(test_indices[0])
        else:
            training_pool.append(indices[0])
    rng = random.Random(seed)
    training, validation = [], []
    for label in (0, 1):
        indices = sorted(i for i in training_pool if labels[i] == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * .1))
        validation.extend(indices[:count])
        training.extend(indices[count:])
    return {"train": np.asarray(sorted(training), dtype=np.int64),
            "validation": np.asarray(sorted(validation), dtype=np.int64),
            "test": np.asarray(sorted(testing), dtype=np.int64)}, conflicts


def prepare_dataset():
    archive = download_dataset()
    archive_hash = file_sha256(archive)
    prepared = CACHE / "prepared.npz"
    audit_path = CACHE / "data_audit.json"
    if prepared.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["archive_sha256"] == archive_hash and audit["revision"] == REVISION:
            print("Using audited prepared dataset.", flush=True)
            return
    images, labels, splits, names, hashes = [], [], [], [], []
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(bundle.infolist(), key=lambda member: member.filename)
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 5 or parts[1] != "DATASET" or parts[2] not in {"train", "test"}:
                continue
            if parts[3] not in {"REAL", "FAKE"} or PurePosixPath(parts[4]).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if member.file_size > 1024 * 1024:
                raise ValueError(f"Unexpected oversized dataset image: {member.filename}")
            # Never extract archive paths onto the filesystem.
            with Image.open(io.BytesIO(bundle.read(member))) as image:
                if image.size != (32, 32):
                    raise ValueError(f"Expected 32x32 image: {member.filename}")
                pixels = np.array(image.convert("RGB"), dtype=np.uint8)
            images.append(pixels)
            labels.append(int(parts[3] == "FAKE"))
            splits.append(parts[2])
            names.append("/".join(parts[2:]))
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            if len(images) % 20000 == 0:
                print(f"Decoded {len(images):,} images", flush=True)
    counts = {f"{split}/{label}": sum(s == split and y == label for s, y in zip(splits, labels))
              for split in ("train", "test") for label in (0, 1)}
    expected = {"train/0": 50000, "train/1": 50000, "test/0": 10000, "test/1": 10000}
    if counts != expected:
        raise ValueError(f"Unexpected official dataset counts: {counts}")
    labels = np.asarray(labels, dtype=np.uint8)
    partitions, conflicts = split_indices(hashes, labels, splits)
    arrays = dict(images=np.stack(images), labels=labels, names=np.asarray(names), hashes=np.asarray(hashes), **partitions)
    np.savez(prepared, **arrays)
    audit = {"dataset": "CIFAKE", "revision": REVISION, "archive_url": DATASET_URL,
             "archive_sha256": archive_hash, "seed": SEED, "class_mapping": {"REAL": 0, "FAKE": 1},
             "official_counts": counts, "conflicting_records_removed": conflicts,
             "exact_duplicate_records_removed": len(images) - sum(len(i) for i in partitions.values()) - conflicts,
             "partitions": {name: {"count": len(indices), "generated": int(labels[indices].sum()),
                                   "real": int(len(indices) - labels[indices].sum()),
                                   "manifest_sha256": hashlib.sha256("\n".join(names[i] for i in indices).encode()).hexdigest()}
                            for name, indices in partitions.items()},
             "note": "Validation sampled only from official training pool. Exact RGB duplicates grouped; conflicts removed; official test takes precedence over train overlap. Near duplicates and semantic-family leakage have not been exhaustively audited."}
    save_json(audit_path, audit)
    print(json.dumps(audit["partitions"], indent=2), flush=True)


def load_arrays():
    if not (CACHE / "prepared.npz").exists():
        raise FileNotFoundError("Run prepare before training/evaluation.")
    return dict(np.load(CACHE / "prepared.npz", allow_pickle=False))


def batches(data, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        current = indices[start:start + batch_size]
        x = torch.from_numpy(data["images"][current].transpose(0, 3, 1, 2).copy()).float().div_(255)
        y = torch.from_numpy(data["labels"][current].copy()).float()
        yield current, x, y


def predict(model, data, indices, batch_size):
    scores = []
    loss = 0.
    model.eval()
    with torch.inference_mode():
        for _, x, y in batches(data, indices, batch_size):
            logits = model(x)
            loss += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            scores.extend(logits.sigmoid().tolist())
    return np.asarray(scores), loss / len(indices)


def score_rows(data, indices, scores):
    return [{"filename": str(data["names"][i]), "label": "generated" if data["labels"][i] else "camera",
             "ai_score": float(score)} for i, score in zip(indices, scores)]


def save_scores(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["filename", "label", "ai_score"])
        writer.writeheader()
        writer.writerows(rows)


def load_baseline():
    from detector import load_detector
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return load_detector(download=False)
    except (FileNotFoundError, LocalEntryNotFoundError):
        return load_detector(download=True)


def train(args):
    checkpoint_path = CACHE / "candidate.pt"
    if checkpoint_path.exists():
        raise FileExistsError("Candidate already exists. Preserve the experiment; use a separate workspace for a new training run.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_arrays()
    model = CifakeCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, stale = float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.
        order = data["train"][np.random.permutation(len(data["train"]))]
        for _, x, y in batches(data, order, args.batch_size):
            # No image augmentations: preserve forensic traces for this first baseline.
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        scores, validation_loss = predict(model, data, data["validation"], args.batch_size)
        accuracy = float(((scores >= .5) == data["labels"][data["validation"]]).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(order),
                        "validation_loss": validation_loss, "validation_accuracy": accuracy,
                        "seconds": round(time.perf_counter() - start, 3)})
        print(json.dumps(history[-1]), flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save({"architecture": ARCHITECTURE, "generated_class": 1, "state_dict": model.state_dict(),
                        "dataset_revision": REVISION, "seed": SEED, "best_epoch": epoch,
                        "preprocessing": "RGB; bilinear resize 32x32; CHW float32 /255",
                        "selection": "lowest validation BCE; official test excluded"}, checkpoint_path)
        else:
            stale += 1
        save_json(CACHE / "training_history.json", {"architecture": ARCHITECTURE, "seed": SEED,
                   "epochs_requested": args.epochs, "batch_size": args.batch_size, "threads": args.threads,
                   "optimizer": "AdamW lr=0.001 weight_decay=0.0001; cosine LR schedule",
                   "selection": "lowest validation BCE; patience 4; no test-based tuning", "history": history})
        scheduler.step()
        if stale >= 4:
            break
    print(f"Candidate trained: {checkpoint_path}", flush=True)


def evaluate_candidate(args):
    output = CACHE / "test_metrics.json"
    if output.exists():
        raise FileExistsError("Official test results already exist; preserve them instead of repeatedly tuning on this set.")
    torch.set_num_threads(args.threads)
    data = load_arrays()
    model = load_cifake_model()
    scores, loss = predict(model, data, data["test"], args.batch_size)
    rows = score_rows(data, data["test"], scores)
    report = evaluate(rows)
    report.update({"dataset": "CIFAKE official test after exact duplicate/conflict audit", "bce": loss,
                   "architecture": ARCHITECTURE, "checkpoint_sha256": file_sha256(CACHE / "candidate.pt"),
                   "warning": "CIFAKE-domain results only; not accuracy on Gemini, modern generators or phone photos. No edit detection is trained."})
    save_scores(CACHE / "test_scores.csv", rows)
    save_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


def compare_baseline(args):
    """Fixed random test subset for a paired, compute-limited baseline comparison."""
    from detector import MODEL_ID, MODEL_REVISION, analyze_pixels
    torch.set_num_threads(args.threads)
    data = load_arrays()
    rng = np.random.default_rng(SEED)
    subset = np.concatenate([rng.choice(data["test"][data["labels"][data["test"]] == label],
                                       args.per_class, replace=False) for label in (0, 1)])
    subset.sort()
    baseline = load_baseline()
    candidate = load_cifake_model()
    baseline_rows, candidate_rows = [], []
    for count, i in enumerate(subset, 1):
        image = Image.fromarray(data["images"][i])
        baseline_rows.extend(score_rows(data, [i], [analyze_pixels(image, baseline)["ai_score"]]))
        candidate_rows.extend(score_rows(data, [i], [cifake_score(image, candidate)]))
        if count % 100 == 0:
            print(f"Compared {count}/{len(subset)} held-out images", flush=True)
    report = {"subset_seed": SEED, "per_class": args.per_class,
              "baseline_model": MODEL_ID, "baseline_revision": MODEL_REVISION,
              "baseline": evaluate(baseline_rows), "candidate": evaluate(candidate_rows),
              "note": "Same fixed CIFAKE test subset; reports in-domain behavior only. Both models see identical decoded pixels."}
    save_scores(CACHE / "baseline_subset_scores.csv", baseline_rows)
    save_scores(CACHE / "candidate_subset_scores.csv", candidate_rows)
    save_json(CACHE / "baseline_comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


def target_diagnostic(args):
    from detector import analyze_pixels
    torch.set_num_threads(args.threads)
    candidate = load_cifake_model()
    baseline = load_baseline()
    cases = [("raw1.jpg", "camera"), ("raw2.jpg", "camera"), ("raw3.jpg", "camera"),
             ("dog.jpg", "generated"), ("school.jpg", "generated"),
             ("CineMate AI Security Audit-2.png", "generated")]
    rows = []
    for filename, label in cases:
        path = ROOT / "data" / filename
        if not path.exists():
            continue
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        rows.append({"filename": filename, "label": label,
                     "baseline_score": analyze_pixels(image, baseline)["ai_score"],
                     "candidate_score": cifake_score(image, candidate)})
    report = {"cases": rows,
              "baseline_metrics": evaluate([{"label": row["label"], "ai_score": row["baseline_score"]} for row in rows]) if rows else None,
              "candidate_metrics": evaluate([{"label": row["label"], "ai_score": row["candidate_score"]} for row in rows]) if rows else None,
              "note": "Previously inspected user-labeled development diagnostics, not an independent target test set. Not used in training or checkpoint selection."}
    save_json(CACHE / "target_diagnostic.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "compare", "target"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=250)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.per_class) <= 0:
        parser.error("Numeric settings must be positive.")
    {"prepare": prepare_dataset, "train": lambda: train(args), "evaluate": lambda: evaluate_candidate(args),
     "compare": lambda: compare_baseline(args), "target": lambda: target_diagnostic(args)}[args.command]()


if __name__ == "__main__":
    main()
