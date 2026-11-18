"""Evaluate labeled scores without loading a model or tuning on test examples."""

import argparse
import csv
import json
import math
from pathlib import Path


def evaluate(rows):
    records = list(rows)
    if not records:
        raise ValueError("No scored images supplied.")
    parsed = []
    for row in records:
        label = row["label"]
        score = float(row["ai_score"])
        if label not in {"camera", "generated"}:
            raise ValueError("Labels must be camera or generated.")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Scores must be finite and between 0 and 1.")
        parsed.append((label == "generated", score))
    positives = [score for label, score in parsed if label]
    negatives = [score for label, score in parsed if not label]
    tp = sum(label and score >= .5 for label, score in parsed)
    fp = sum(not label and score >= .5 for label, score in parsed)
    tn = sum(not label and score < .5 for label, score in parsed)
    fn = sum(label and score < .5 for label, score in parsed)
    decided = [(label, score) for label, score in parsed if score <= .1 or score >= .9]
    correct = sum(label == (score >= .9) for label, score in decided)
    ratio = lambda a, b: a / b if b else None
    # Rank/group ties in O(n log n), rather than comparing every positive to
    # every negative (100 million pairs on CIFAKE's official test split).
    ordered = sorted(parsed, key=lambda item: item[1])
    concordant, negatives_before, position = 0., 0, 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        group = ordered[position:end]
        group_positive = sum(label for label, _ in group)
        group_negative = len(group) - group_positive
        concordant += group_positive * (negatives_before + .5 * group_negative)
        negatives_before += group_negative
        position = end
    auc = ratio(concordant, len(positives) * len(negatives))
    return {
        "count": len(parsed), "generated_count": len(positives), "camera_count": len(negatives),
        "forced_cutoff": .5,
        "confusion_matrix": {"true_ai": tp, "false_ai": fp, "true_camera": tn, "missed_ai": fn},
        "accuracy": ratio(tp + tn, len(parsed)), "ai_precision": ratio(tp, tp + fp),
        "ai_recall": ratio(tp, tp + fn), "ai_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "camera_false_positive_rate": ratio(fp, len(negatives)), "roc_auc": auc,
        "exploratory_bands": {"camera_max": .1, "ai_min": .9},
        "coverage": ratio(len(decided), len(parsed)),
        "inconclusive_count": len(parsed) - len(decided),
        "accuracy_among_decided": ratio(correct, len(decided)),
        "note": "Evaluation only; this function fits neither thresholds nor weights. Results describe the supplied images and may not generalize.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", help="CSV with filename,label,ai_score columns")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    try:
        with open(args.scores, newline="", encoding="utf-8-sig") as source:
            report = evaluate(csv.DictReader(source))
        output = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
