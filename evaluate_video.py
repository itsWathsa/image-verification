"""Evaluate video screening against labelled, prompt-disjoint video manifests.

No training or changes to the application model are performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def validate_manifest(rows):
    if not rows:
        raise ValueError("The manifest is empty.")
    groups, paths = {}, set()
    for row in rows:
        if row.get("label") not in ("real", "generated") or row.get("split") not in ("development", "test"):
            raise ValueError("Use real/generated labels and development/test splits.")
        for key in ("path", "family", "source", "provenance"):
            if not row.get(key):
                raise ValueError(f"Missing {key}.")
        group = str(row["family"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("An action/prompt family crosses evaluation splits.")
        groups[group] = row["split"]
        path = str(Path(row["path"]).resolve())
        if path in paths:
            raise ValueError("Duplicate video path in manifest.")
        paths.add(path)
    return rows


def metrics(rows, rule):
    tp = fp = tn = fn = 0
    by_source = {}
    for row in rows:
        positive = rule(row)
        actual = row["label"] == "generated"
        tp += int(actual and positive)
        fn += int(actual and not positive)
        fp += int(not actual and positive)
        tn += int(not actual and not positive)
        source = by_source.setdefault(row["source"], {"total": 0, "flagged": 0})
        source["total"] += 1
        source["flagged"] += int(positive)
    return {"generated_detected": tp, "generated_missed": fn, "real_false_alarms": fp, "real_not_flagged": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None, "by_source": by_source}


def ready_rows(rows, wait_seconds):
    import time
    pending = list(rows)
    deadline = time.monotonic() + wait_seconds
    while pending:
        ready = [row for row in pending if Path(row["path"]).exists() and
                 ("size" not in row or Path(row["path"]).stat().st_size == row["size"])]
        if not ready:
            if not wait_seconds or time.monotonic() >= deadline:
                yield from pending
                return
            time.sleep(1)
            continue
        for row in ready:
            pending.remove(row)
            yield row


def run(manifest, output, wait_seconds=0):
    from detector import load_detector
    from video_analysis import analyze_video, VIDEO_METHOD
    rows = validate_manifest(json.loads(Path(manifest).read_text(encoding="utf-8")))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session = load_detector(download=False)
    scored, errors, hashes = [], [], {}
    for index, row in enumerate(ready_rows(rows, wait_seconds)):
        try:
            if "size" in row and (not Path(row["path"]).exists() or Path(row["path"]).stat().st_size != row["size"]):
                raise ValueError("Dataset file is missing or incomplete; not scored.")
            data = Path(row["path"]).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if row.get("sha256") and digest != row["sha256"]:
                raise ValueError("Dataset checksum mismatch.")
            if digest in hashes:
                raise ValueError("Duplicate video content in evaluation manifest.")
            hashes[digest] = row["split"]
            cached = output / (digest + ".json")
            if cached.exists():
                report = json.loads(cached.read_text())
                if report.get("analysis_method") != VIDEO_METHOD:
                    raise ValueError("Stale evaluation cache; choose a fresh output folder.")
            else:
                report = analyze_video(data, session)["report"]
                cached.write_text(json.dumps(report, indent=2))
            scored.append({**row, "report": report})
            print(f"{index + 1}/{len(rows)} {row['source']} {row['split']} max={report['max_frame_score']:.4f}", flush=True)
        except Exception as error:
            errors.append({**row, "error": str(error)})
            print(f"{index + 1}/{len(rows)} unavailable: {error}", flush=True)
        partial = {"scored": scored, "errors": errors}
        (output / "scores.json").write_text(json.dumps(partial, indent=2))
    rules = {"current_any_frame_090": lambda row: row["report"]["max_frame_score"] >= .9,
             "candidate_consensus_050": lambda row: sum(frame["ai_score"] >= .5 for frame in row["report"]["frames"]) >= max(2, (row["report"]["sample_count"] + 3) // 4)}
    summary = {"method": VIDEO_METHOD, "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
               "counts": dict(Counter(row["split"] for row in rows)), "decode_or_analysis_errors": errors,
               "results": {split: {name: metrics([row for row in scored if row["split"] == split], rule)
                           for name, rule in rules.items()} for split in ("development", "test")},
               "notice": "Exploratory bounded subset, not a canonical benchmark. Errors are reported separately, not counted as correct. Candidate is offline only; no detector weights were trained."}
    development = summary["results"]["development"]
    baseline = development["current_any_frame_090"]
    candidate = development["candidate_consensus_050"]
    eligible = not errors and baseline["recall"] is not None and candidate["false_positive_rate"] is not None
    summary["candidate_development_gate"] = {
        "required_recall_gain": .10, "maximum_false_positive_rate": .05,
        "must_not_increase_false_alarms": True,
        "passed": bool(eligible and candidate["recall"] >= baseline["recall"] + .10
                       and candidate["false_positive_rate"] <= .05
                       and candidate["real_false_alarms"] <= baseline["real_false_alarms"]),
        "enabled_in_app": False,
        "reason": "Offline experiment only; an initial small benchmark does not establish deployment reliability."
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0, help="Wait for pending benchmark file downloads")
    args = parser.parse_args()
    run(args.manifest, args.output, args.wait_seconds)
