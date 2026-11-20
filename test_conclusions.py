import unittest
from conclusions import generation_conclusion

class ConclusionTests(unittest.TestCase):
    def test_answers_preserve_uncertainty(self):
        self.assertIn("Likely AI-generated", generation_conclusion(.95))
        self.assertIn("may still be present", generation_conclusion(.05))
        self.assertIn("Unable to determine", generation_conclusion(.5007))
        for fn in (generation_conclusion,):
            for value in (-1, 2, float('nan')):
                with self.assertRaises(ValueError): fn(value)

