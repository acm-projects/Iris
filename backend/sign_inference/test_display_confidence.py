import unittest
from display_confidence import scale_top_display


class DisplayScalingTests(unittest.TestCase):
    def test_selected_top_only_and_original_unchanged(self):
        original = [{'gloss': 'Call', 'confidence': .4}, {'gloss': 'help', 'confidence': .2}]
        result = scale_top_display(original, [' call ', 'help'])
        self.assertEqual(result[0]['display_confidence'], .8)
        self.assertEqual(result[0]['display_scalar'], 2)
        self.assertEqual(result[0]['confidence'], .4)
        self.assertEqual(result[1]['display_confidence'], .2)
        self.assertNotIn('display_confidence', original[0])

    def test_unselected_high_boundary_and_zero_unchanged(self):
        for raw, words in [(.4, []), (.8, ['call']), (.9, ['call']), (0, ['call'])]:
            result = scale_top_display([{'gloss': 'call', 'confidence': raw}], words)
            self.assertEqual(result[0]['display_confidence'], raw)
            self.assertEqual(result[0]['display_scalar'], 1)
        self.assertEqual(scale_top_display([], ['call']), [])

    def test_invalid_scores(self):
        for score in [float('nan'), float('inf'), -.1, 1.1]:
            with self.assertRaises(ValueError):
                scale_top_display([{'gloss': 'call', 'confidence': score}], ['call'])
