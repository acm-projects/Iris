"""Select each word's configured orientation before ranking raw model scores."""
import json
from pathlib import Path


def load_word_rules():
    return json.loads(Path(__file__).with_name('word_rules.json').read_text(encoding='utf-8'))


def select_word_predictions(original, mirrored, rules, top_k):
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
