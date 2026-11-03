"""Optional cosmetic scaling. Never change model probabilities or ranking."""
import math


def scale_top_display(predictions, selected_words):
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
