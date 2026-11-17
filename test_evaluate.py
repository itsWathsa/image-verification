import unittest

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_confusion_and_abstention(self):
        rows = [{"label": label, "ai_score": score} for label, score in
                [("generated", .99), ("generated", .14), ("generated", .29),
                 ("camera", .95), ("camera", 0), ("camera", .92)]]
        report = evaluate(rows)
        self.assertEqual(report["confusion_matrix"], {"true_ai": 1, "false_ai": 2, "true_camera": 1, "missed_ai": 2})
        self.assertAlmostEqual(report["accuracy"], 1 / 3)
        self.assertAlmostEqual(report["roc_auc"], 5 / 9)
        self.assertEqual(report["inconclusive_count"], 2)
        self.assertEqual(report["accuracy_among_decided"], .5)

    def test_ties_single_class_and_invalid_inputs(self):
        report = evaluate([{"label": "camera", "ai_score": .5}, {"label": "generated", "ai_score": .5}])
        self.assertEqual(report["roc_auc"], .5)
        self.assertIsNone(report["accuracy_among_decided"])
        self.assertIsNone(evaluate([{"label": "camera", "ai_score": 0}])["roc_auc"])
        for rows in ([], [{"label": "unknown", "ai_score": .5}], [{"label": "camera", "ai_score": "nan"}]):
            with self.assertRaises(ValueError):
                evaluate(rows)
