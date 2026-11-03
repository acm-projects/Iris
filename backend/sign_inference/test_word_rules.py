import unittest
from word_rules import select_word_predictions
from display_confidence import scale_top_display


class WordRuleTests(unittest.TestCase):
    def test_help_uses_mirrored_score_then_scales_if_top(self):
        original = [{'gloss': 'call', 'confidence': .5}, {'gloss': 'help', 'confidence': .1}]
        mirrored = [{'gloss': 'help', 'confidence': .678}, {'gloss': 'call', 'confidence': .01}]
        ranked = select_word_predictions(original, mirrored, {'help': {'invert': True}}, 2)
        result = scale_top_display(ranked, ['help'])
        self.assertEqual(result[0]['gloss'], 'help')
        self.assertEqual(result[0]['confidence'], .678)
        self.assertEqual(result[0]['original_confidence'], .1)
        self.assertEqual(result[0]['display_confidence'], .8)
        self.assertEqual(result[1]['confidence'], .5)
        self.assertAlmostEqual(result[0]['display_scalar'], .8/.678)

    def test_inversion_does_not_force_help_to_win(self):
        original = [{'gloss': 'call', 'confidence': .9}, {'gloss': 'help', 'confidence': .1}]
        mirrored = [{'gloss': 'help', 'confidence': .6}, {'gloss': 'call', 'confidence': .2}]
        ranked = select_word_predictions(original, mirrored, {'help': {'invert': True}}, 2)
        self.assertEqual(ranked[0]['gloss'], 'call')
        self.assertEqual(scale_top_display(ranked, ['help'])[1]['display_confidence'], .6)

    def test_disabled_word_removed_before_ranking(self):
        original = [{'gloss': 'help', 'confidence': .9}, {'gloss': 'call', 'confidence': .1}]
        ranked = select_word_predictions(original, [], {'help': {'disabled': True}}, 2)
        self.assertEqual([p['gloss'] for p in ranked], ['call'])
