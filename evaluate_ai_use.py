"""Evaluate AI involvement separately from ordinary editing.

Ground-truth labels must come from known creation history. This tool evaluates
the existing Detectra model; it does not fit weights or thresholds.
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate import evaluate

LABELS = {"camera_original": False, "camera_conventional_edit": False,
          "fully_ai_generated": True, "camera_ai_edited": True}


def evaluate_ai_use(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No labeled AI-use examples supplied.")
    if any(row["label"] not in LABELS for row in rows):
        raise ValueError(f"Labels must be one of: {', '.join(LABELS)}")

    def metrics(records, field):
        result = evaluate([{"label": "generated" if LABELS[row["label"]] else "camera",
                            "ai_score": row[field]} for row in records])
        result["answer_counts"] = {
            "ai_use_indicated": sum(float(row[field]) >= .9 for row in records),
            "low_ai_signal": sum(float(row[field]) <= .1 for row in records),
            "inconclusive": sum(.1 < float(row[field]) < .9 for row in records),
            "false_ai_indications": sum(not LABELS[row["label"]] and float(row[field]) >= .9 for row in records),
            "ai_cases_with_low_signal": sum(LABELS[row["label"]] and float(row[field]) <= .1 for row in records),
        }
        return result

    return {"task": "AI involvement: fully generated OR camera image edited using AI",
            "positive_labels": [label for label, positive in LABELS.items() if positive],
            "negative_labels": [label for label, positive in LABELS.items() if not positive],
            "initial_center_view": metrics(rows, "baseline_score"),
            "conditional_rescan": metrics(rows, "ai_score"),
            "by_creation_type": {label: {"count": len(selected),
                "initial_center_view": metrics(selected, "baseline_score"),
                "conditional_rescan": metrics(selected, "ai_score")}
                for label in LABELS if (selected := [row for row in rows if row["label"] == label])},
            "note": "The evaluator's generated fields mean AI involvement, including AI edits. Camera fields mean the no-AI labels. Scores are not calibrated probabilities; creation history is supplied, not inferred. Known development cases do not estimate independent target accuracy."}


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for row in rows:
        if row.get("label") not in LABELS or row.get("split") not in {"train", "validation", "test", "development"}:
            raise ValueError("Each manifest row needs a supported label and split.")
        if not row.get("path") or not row.get("group_id", "").strip():
            raise ValueError("Each row needs an image path and original-family group_id.")
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Original-image family crosses splits: {row['group_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV: path,label,group_id,split. Image paths are relative to the CSV.")
    parser.add_argument("--split", choices=["development", "validation", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not records:
        parser.error("The requested split contains no images.")
    from app import read_image
    from detector import analyze_pixels, load_detector, MODEL_ID, MODEL_REVISION, ANALYSIS_METHOD
    model = load_detector(download=False)
    results = []
    for index, row in enumerate(records, 1):
        image, *_ = read_image((args.manifest.parent / row["path"]).read_bytes(), full_resolution=True)
        result = analyze_pixels(image, model)
        results.append({"path": row["path"], "label": row["label"], "group_id": row["group_id"], **result})
        print(f"Evaluated {index}/{len(records)}", flush=True)
    report = {"model": MODEL_ID, "revision": MODEL_REVISION, "method": ANALYSIS_METHOD,
              "split": args.split, "metrics": evaluate_ai_use(results), "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
