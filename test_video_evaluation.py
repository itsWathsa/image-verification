import unittest
from evaluate_video import validate_manifest, metrics


class VideoEvaluationTests(unittest.TestCase):
    def test_family_leakage_and_duplicate_paths_are_rejected(self):
        row={"path":"a.mp4","label":"real","split":"development","family":"walking","source":"camera","provenance":"verified"}
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_manifest([row,{**row,"path":"b.mp4","split":"test"}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest([row,row])
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_manifest([{**row,"label":"looks fake"}])
        self.assertEqual(validate_manifest([row]),[row])

    def test_false_alarms_and_misses_are_reported_separately(self):
        rows=[{"label":label,"source":label,"score":score} for label,score in (("real",.99),("real",.01),("generated",.99),("generated",.01),("generated",.5))]
        result=metrics(rows,lambda r:r['score']>=.9)
        self.assertEqual(result['generated_detected'],1)
        self.assertEqual(result['generated_missed'],2)
        self.assertEqual(result['real_false_alarms'],1)
        self.assertEqual(result['false_positive_rate'],.5)
        self.assertAlmostEqual(result['recall'],1/3)

    def test_subset_is_reproducible_and_keeps_action_families_separate(self):
        from prepare_video_benchmark import select_rows
        from pathlib import Path
        sources = ["Pexels", "A", "B", "C", "D", "E", "F"]
        files = [{"path": f"{source}/{family}/a.mp4", "size": 100} for source in sources for family in range(40)]
        first = select_rows(files, Path("dataset"))
        self.assertEqual(first, select_rows(list(reversed(files)), Path("dataset")))
        self.assertEqual(len(first), 96)
        validate_manifest(first)
        self.assertEqual(sum(row["label"] == "real" for row in first), 32)
        self.assertEqual(len({row["family"] for row in first}), 32)
