import unittest
from conclusions import generation_conclusion, edit_conclusion

class ConclusionTests(unittest.TestCase):
    def test_answers_preserve_uncertainty(self):
        self.assertIn("Likely AI-generated", generation_conclusion(.95))
        self.assertIn("may still be present", generation_conclusion(.05))
        self.assertIn("Unable to determine", generation_conclusion(.5007))
        self.assertIn("Possible editing", edit_conclusion(.95))
        self.assertIn("does not confirm", edit_conclusion(.09))
        self.assertIn("Unable to determine", edit_conclusion(.44))
        for fn in (generation_conclusion, edit_conclusion):
            for value in (-1, 2, float('nan')):
                with self.assertRaises(ValueError): fn(value)
