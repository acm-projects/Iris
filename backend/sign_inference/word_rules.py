"""Select each word's configured orientation before ranking raw model scores.

word_rules.json lists, per word:
  invert    use the score from a mirrored copy of the keypoints (helps words
            the model learned mostly from signers using the other hand)
  disabled  drop the word from the results entirely
"""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_word_rules():
    """word_rules.json as a dict, read from disk once (continuous mode predicts many times a second)."""
    return json.loads(Path(__file__).with_name('word_rules.json').read_text(encoding='utf-8'))


def select_word_predictions(original, mirrored, rules, top_k):
    """Pick each word's score from the original or mirrored pass, drop disabled words, rank.

    original / mirrored: every word's prediction ({"gloss", "confidence"}) from
    the normal and the mirrored model pass. Returns the top_k, each tagged with
    which orientation was used and its original-pass score.
    """
    mirror_by_word = {p['gloss']: p for p in mirrored}
    output = []
    for prediction in original:
        word = prediction['gloss']
        rule = rules.get(word, {})
        if rule.get('disabled', False):
            continue
        inverted = rule.get('invert', False)
        selected = mirror_by_word[word] if inverted else prediction
        output.append(dict(selected, input_orientation='mirrored_keypoints' if inverted else 'original',
                           original_confidence=prediction['confidence']))
    return sorted(output, key=lambda p: p['confidence'], reverse=True)[:top_k]
