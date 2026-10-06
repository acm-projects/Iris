"""Per-sign confidence for WLASL gloss predictions.

WLASL scores are not comparable across signs. One gloss may peak near 100%
while another rarely exceeds 70% even when the model is right. Comparing both
to a single cutoff drops the second sign.

``SignConfidence`` learns each gloss's floor and ceiling from the scores it
sees while the model runs. Nothing has to be calculated ahead of time. The
first few times a gloss appears, its range is still forming, so there is no
accept/reject decision yet. After that, a raw score is mapped to how high it
sits among the scores already seen for that sign. A score near the top of
that live range is accepted as a word.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


# Spans smaller than this are treated as a single confidence level.
_MIN_SPAN = 1e-6
# Recent scores kept per gloss. Older ones roll off so a long session cannot
# pin the range to a single early spike.
_MAX_OBSERVATIONS = 200


def _gloss_key(gloss: str) -> str:
    if not isinstance(gloss, str):
        raise TypeError("gloss must be a string")
    key = " ".join(gloss.strip().casefold().split())
    if not key:
        raise ValueError("gloss must be a nonempty string")
    return key


def _finite_score(score: float) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError) as error:
        raise TypeError("score must be a number") from error
    if not math.isfinite(value):
        raise ValueError("score must be a finite number")
    return value


def _ratio(value: float) -> float:
    ratio = _finite_score(value)
    if not 0 < ratio <= 1:
        raise ValueError("success ratio must be in (0, 1]")
    return ratio


def relative_score(raw: float, low: float, high: float) -> float:
    """Map ``raw`` into 0..1 inside ``[low, high]``.

    0 means the score is at or below this sign's floor. 1 means it is at or
    above this sign's ceiling. Scores outside the range are clamped.
    """
    raw = _finite_score(raw)
    low = _finite_score(low)
    high = _finite_score(high)
    if high < low:
        low, high = high, low
    span = high - low
    if span < _MIN_SPAN:
        return 1.0 if raw >= high else 0.0
    scaled = (raw - low) / span
    if scaled < 0.0:
        return 0.0
    if scaled > 1.0:
        return 1.0
    return scaled


def _percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    percent = _finite_score(percent)
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class GlossRange:
    """Observed confidence floor and ceiling for one gloss."""

    low: float
    high: float
    samples: int = 0

    def to_dict(self) -> dict:
        return {"low": self.low, "high": self.high, "samples": self.samples}

    @classmethod
    def from_dict(cls, data: Mapping) -> GlossRange:
        samples = int(data.get("samples", 0))
        if samples < 0:
            raise ValueError("samples cannot be negative")
        return cls(
            low=_finite_score(data["low"]),
            high=_finite_score(data["high"]),
            samples=samples,
        )


class SignConfidence:
    """Accept a gloss when its score is high inside the range learned for it.

    Each call to ``evaluate`` records that gloss's score. Once a gloss has
    ``min_observations`` scores (5 by default), its floor and ceiling are the
    5th and 95th percentiles of those scores. ``success_ratio`` is how far up
    that range a new score must land. At the default of 0.8, a sign whose
    live scores run from 0.40 to 0.70 is accepted at 0.64.
    """

    def __init__(
        self,
        success_ratio: float = 0.8,
        min_observations: int = 5,
        low_percentile: float = 5,
        high_percentile: float = 95,
    ):
        self.success_ratio = _ratio(success_ratio)
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        self.min_observations = int(min_observations)
        self.low_percentile, self.high_percentile = self._percentiles(
            low_percentile, high_percentile
        )
        self._ranges: dict[str, GlossRange] = {}
        self._observations: dict[str, list[float]] = defaultdict(list)

    @staticmethod
    def _percentiles(low_percentile: float, high_percentile: float) -> tuple[float, float]:
        low_percentile = _finite_score(low_percentile)
        high_percentile = _finite_score(high_percentile)
        if not 0 <= low_percentile < high_percentile <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        return low_percentile, high_percentile

    def set_range(
        self,
        gloss: str,
        high: float,
        low: float = 0.0,
        samples: int = 0,
    ) -> None:
        """Store the confidence range for one gloss."""
        high = _finite_score(high)
        low = _finite_score(low)
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        if samples < 0:
            raise ValueError("samples cannot be negative")
        self._ranges[_gloss_key(gloss)] = GlossRange(low=low, high=high, samples=samples)

    def set_max(self, gloss: str, maximum: float) -> None:
        """Store a gloss whose range runs from 0 up to ``maximum``."""
        maximum = _finite_score(maximum)
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        self.set_range(gloss, high=maximum, low=0.0)

    def observe(self, gloss: str, score: float) -> None:
        """Record one live score for this gloss and refresh its range.

        Pass the confidence the model just assigned to that gloss. The same
        kind of score must be used every time (softmax probability, or logits,
        but not a mix of the two). The range stays unset until this gloss has
        been seen ``min_observations`` times.
        """
        key = _gloss_key(gloss)
        self._observations[key].append(_finite_score(score))
        overflow = len(self._observations[key]) - _MAX_OBSERVATIONS
        if overflow > 0:
            del self._observations[key][:overflow]
        self._refresh(key)

    def _refresh(self, gloss: str) -> None:
        scores = self._observations.get(gloss, [])
        if len(scores) < self.min_observations:
            return
        self._ranges[gloss] = GlossRange(
            low=_percentile(scores, self.low_percentile),
            high=_percentile(scores, self.high_percentile),
            samples=len(scores),
        )

    def fit(self, low_percentile: float = 5, high_percentile: float = 95) -> None:
        """Rebuild every gloss range from the scores recorded so far.

        Glosses with fewer than ``min_observations`` samples are left unchanged.
        The default percentiles ignore a few extreme scores so one outlier does
        not define the range.
        """
        self.low_percentile, self.high_percentile = self._percentiles(
            low_percentile, high_percentile
        )
        for gloss in self._observations:
            self._refresh(gloss)

    def range_for(self, gloss: str) -> GlossRange | None:
        return self._ranges.get(_gloss_key(gloss))

    def relative(self, gloss: str, score: float) -> float | None:
        """Return 0..1 within this gloss's range, or None if it is uncalibrated."""
        found = self.range_for(gloss)
        if found is None:
            return None
        return relative_score(score, found.low, found.high)

    def is_successful(
        self,
        gloss: str,
        score: float,
        success_ratio: float | None = None,
    ) -> bool | None:
        """True when ``score`` is high enough inside this gloss's range.

        Returns None when the gloss has no range yet.
        """
        scaled = self.relative(gloss, score)
        if scaled is None:
            return None
        cutoff = self.success_ratio if success_ratio is None else _ratio(success_ratio)
        return scaled >= cutoff

    def evaluate(
        self,
        gloss: str,
        score: float,
        success_ratio: float | None = None,
        learn: bool = True,
    ) -> dict:
        """Score one predicted gloss and decide whether it counts as a word.

        When ``learn`` is true, this score is added to that gloss's live range
        before the decision. Until enough scores exist, ``successful`` is None.
        """
        key = _gloss_key(gloss)
        raw = _finite_score(score)
        if learn:
            self.observe(key, raw)
        found = self._ranges.get(key)
        seen = len(self._observations.get(key, []))
        if found is None:
            return {
                "gloss": key,
                "raw": raw,
                "relative": None,
                "successful": None,
                "calibrated": False,
                "low": None,
                "high": None,
                "seen": seen,
            }
        cutoff = self.success_ratio if success_ratio is None else _ratio(success_ratio)
        scaled = relative_score(raw, found.low, found.high)
        return {
            "gloss": key,
            "raw": raw,
            "relative": scaled,
            "successful": scaled >= cutoff,
            "calibrated": True,
            "low": found.low,
            "high": found.high,
            "seen": seen,
        }

    def evaluate_prediction(
        self,
        glosses: Sequence[str],
        scores: Sequence[float],
        success_ratio: float | None = None,
        learn: bool = True,
    ) -> dict:
        """Pick the top WLASL class and judge it against that class's live range.

        ``glosses`` and ``scores`` follow the model's class order. The winning
        score is recorded, so the range for that gloss updates as the model runs.
        """
        if len(glosses) != len(scores):
            raise ValueError("glosses and scores must be the same length")
        if not glosses:
            raise ValueError("prediction must include at least one class")
        best = max(range(len(scores)), key=lambda index: _finite_score(scores[index]))
        result = self.evaluate(
            glosses[best],
            scores[best],
            success_ratio=success_ratio,
            learn=learn,
        )
        result["candidates"] = len(glosses)
        return result

    def to_dict(self) -> dict:
        glosses: dict[str, dict] = {}
        keys = set(self._ranges) | set(self._observations)
        for gloss in sorted(keys):
            entry: dict = {}
            span = self._ranges.get(gloss)
            if span is not None:
                entry.update(span.to_dict())
            scores = self._observations.get(gloss, [])
            if scores:
                entry["scores"] = list(scores)
            if entry:
                glosses[gloss] = entry
        return {
            "version": 1,
            "success_ratio": self.success_ratio,
            "min_observations": self.min_observations,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "glosses": glosses,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SignConfidence:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("calibration file must contain a JSON object")
        calibrator = cls(
            success_ratio=float(data.get("success_ratio", 0.8)),
            min_observations=int(data.get("min_observations", 5)),
            low_percentile=float(data.get("low_percentile", 5)),
            high_percentile=float(data.get("high_percentile", 95)),
        )
        glosses = data.get("glosses", {})
        if not isinstance(glosses, dict):
            raise ValueError("glosses must be a JSON object")
        for gloss, span in glosses.items():
            if not isinstance(span, dict):
                raise ValueError("each gloss entry must be a JSON object")
            key = _gloss_key(gloss)
            for score in span.get("scores", []):
                calibrator._observations[key].append(_finite_score(score))
            if "low" in span and "high" in span:
                parsed = GlossRange.from_dict(span)
                calibrator.set_range(key, high=parsed.high, low=parsed.low, samples=parsed.samples)
        return calibrator

    @classmethod
    def from_maxima(
        cls,
        maxima: Mapping[str, float],
        success_ratio: float = 0.8,
    ) -> SignConfidence:
        """Build ranges when each sign's known ceiling is its only statistic.

        The floor is 0, so a raw score is judged as a fraction of that sign's
        maximum. A gloss with a maximum of 0.70 reaches a relative score of
        1.0 at 0.70.
        """
        calibrator = cls(success_ratio=success_ratio)
        for gloss, maximum in maxima.items():
            calibrator.set_max(gloss, maximum)
        return calibrator
