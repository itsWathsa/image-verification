import tempfile
from pathlib import Path
import unittest

from evaluate_ai_use import evaluate_ai_use, read_manifest


class AIUseTests(unittest.TestCase):
    def test_ai_edits_are_positive_and_conventional_edits_are_negative(self):
        report = evaluate_ai_use([
            {"label": "camera_original", "baseline_score": .01, "ai_score": .01},
            {"label": "camera_conventional_edit", "baseline_score": .02, "ai_score": .02},
            {"label": "fully_ai_generated", "baseline_score": .99, "ai_score": .99},
            {"label": "camera_ai_edited", "baseline_score": .6, "ai_score": .98},
        ])
        self.assertEqual(report["conditional_rescan"]["accuracy"], 1)
        self.assertEqual(report["initial_center_view"]["inconclusive_count"], 1)
        self.assertEqual(report["conditional_rescan"]["inconclusive_count"], 0)
        self.assertEqual(report["conditional_rescan"]["generated_count"], 2)
        self.assertEqual(report["initial_center_view"]["answer_counts"]["inconclusive"], 1)
        self.assertEqual(report["conditional_rescan"]["answer_counts"]["false_ai_indications"], 0)
        self.assertEqual(report["by_creation_type"]["camera_conventional_edit"]["count"], 1)

    def test_invalid_labels_and_family_leakage_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_ai_use([{"label": "unknown", "baseline_score": .2, "ai_score": .2}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("path,label,group_id,split\na.jpg,camera_original,original-1,train\nb.jpg,camera_ai_edited,original-1,test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                read_manifest(path)

    def test_false_indications_are_separate_from_forced_cutoff_metrics(self):
        result = evaluate_ai_use([
            {"label": "camera_conventional_edit", "baseline_score": .8, "ai_score": .99},
            {"label": "camera_ai_edited", "baseline_score": .05, "ai_score": .05},
        ])
        self.assertEqual(result["initial_center_view"]["answer_counts"]["false_ai_indications"], 0)
        self.assertEqual(result["conditional_rescan"]["answer_counts"]["false_ai_indications"], 1)
        self.assertEqual(result["conditional_rescan"]["answer_counts"]["ai_cases_with_low_signal"], 1)
