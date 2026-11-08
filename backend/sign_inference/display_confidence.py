"""Optional cosmetic scaling. Never change model probabilities or ranking.

For words listed in display_scale_words.json, the top guess's *displayed*
score is raised to 80% when the real score is lower. The real score stays in
"confidence"; only "display_confidence" changes. Not a calibrated confidence:
quote "confidence" when reporting accuracy.
"""
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_display_scale_words() -> tuple[str, ...]:
    """display_scale_words.json (words eligible for display scaling), read once."""
    words = json.loads(Path(__file__).with_name('display_scale_words.json').read_text(encoding='utf-8'))
    if not isinstance(words, list) or any(not isinstance(w, str) or not w.strip() for w in words):
        raise ValueError('display_scale_words.json must be a JSON list of nonempty words.')
    return tuple(words)


def scale_top_display(predictions, selected_words):
    """Copy of predictions with "display_confidence"/"display_scalar" added (top guess may be raised to 80%)."""
    selected = {' '.join(word.casefold().split()) for word in selected_words}
    output = [dict(prediction, display_confidence=prediction['confidence'],
                   display_scalar=1.0) for prediction in predictions]
    if not output:
        return output
    top = output[0]
    raw = float(top['confidence'])
    if not math.isfinite(raw) or not 0 <= raw <= 1:
        raise ValueError('Confidence must be finite and between 0 and 1.')
    if ' '.join(top['gloss'].casefold().split()) in selected and 0 < raw < .8:
        top['display_scalar'] = .8 / raw
        # Assign the exact target to avoid floating-point multiplication drift.
        top['display_confidence'] = .8
    return output
