"""Ground-truth confidence calibration for live sign predictions.

The runtime must never learn acceptance thresholds from its own unlabelled
predictions. This module fits thresholds only from labelled evaluation rows
and writes a deterministic, versioned artifact that the live recognizer can
load without retaining session observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_TARGET_PRECISION = 0.80
DEFAULT_MIN_TRUE_POSITIVES = 3


def _gloss(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = " ".join(value.strip().casefold().replace("_", " ").split())
    if not cleaned:
        raise ValueError(f"{field} must not be blank")
    return cleaned


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be blank")
    return cleaned


def _score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("score must be numeric") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")
    return score


@dataclass(frozen=True)
class LabelledPrediction:
    """One top-one prediction from a clip with a known expected gloss."""

    expected: str
    predicted: str
    score: float
    clip: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _gloss(self.expected, "expected"))
        object.__setattr__(self, "predicted", _gloss(self.predicted, "predicted"))
        object.__setattr__(self, "score", _score(self.score))
        object.__setattr__(self, "clip", str(self.clip))

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "LabelledPrediction":
        return cls(
            expected=_gloss(row.get("expected"), "expected"),
            predicted=_gloss(row.get("predicted"), "predicted"),
            score=_score(row.get("score")),
            clip=str(row.get("clip", "")),
        )

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted

    def canonical(self) -> dict[str, object]:
        return {
            "clip": self.clip.replace("\\", "/"),
            "expected": self.expected,
            "predicted": self.predicted,
            "score": self.score,
        }


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accepted: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "accepted": self.accepted,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "precision": self.precision,
            "recall": self.recall,
        }


def choose_threshold(
    rows: Sequence[LabelledPrediction],
    *,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_true_positives: int = DEFAULT_MIN_TRUE_POSITIVES,
) -> ThresholdMetrics | None:
    """Choose the highest-recall score cutoff that meets a precision floor."""
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    if min_true_positives < 1:
        raise ValueError("min_true_positives must be positive")
    total_correct = sum(row.correct for row in rows)
    if total_correct < min_true_positives:
        return None

    choices: list[ThresholdMetrics] = []
    for threshold in sorted({row.score for row in rows}):
        accepted = [row for row in rows if row.score >= threshold]
        true_positives = sum(row.correct for row in accepted)
        false_positives = len(accepted) - true_positives
        precision = true_positives / len(accepted)
        recall = true_positives / total_correct
        if true_positives >= min_true_positives and precision >= target_precision:
            choices.append(
                ThresholdMetrics(
                    threshold=threshold,
                    accepted=len(accepted),
                    true_positives=true_positives,
                    false_positives=false_positives,
                    precision=precision,
                    recall=recall,
                )
            )
    if not choices:
        return None
    return max(
        choices,
        key=lambda item: (item.recall, item.precision, item.true_positives, item.threshold),
    )


@dataclass(frozen=True)
class ConfidenceCalibration:
    """Versioned thresholds fitted from a labelled camera evaluation."""

    model: str
    preprocessing: str
    dataset: str
    source_sha256: str
    target_precision: float
    global_metrics: ThresholdMetrics
    per_gloss: Mapping[str, ThresholdMetrics]
    observation_count: int

    def threshold_for(self, gloss: str) -> tuple[float, str]:
        cleaned = _gloss(gloss, "gloss")
        metrics = self.per_gloss.get(cleaned)
        if metrics is not None:
            return metrics.threshold, "per-gloss"
        return self.global_metrics.threshold, "global"

    def evaluate(self, gloss: str, score: float) -> dict[str, object]:
        threshold, source = self.threshold_for(gloss)
        raw = _score(score)
        return {
            "accepted": raw >= threshold,
            "threshold": threshold,
            "thresholdSource": source,
            "calibrationDataset": self.dataset,
            "calibrationSha256": self.source_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "model": self.model,
            "preprocessing": self.preprocessing,
            "dataset": self.dataset,
            "sourceSha256": self.source_sha256,
            "targetPrecision": self.target_precision,
            "observationCount": self.observation_count,
            "global": self.global_metrics.to_dict(),
            "perGloss": {
                gloss: metrics.to_dict()
                for gloss, metrics in sorted(self.per_gloss.items())
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError(f"calibration must use schema version {SCHEMA_VERSION}")

        def metrics(value: object) -> ThresholdMetrics:
            if not isinstance(value, dict):
                raise ValueError("calibration metrics must be an object")
            return ThresholdMetrics(
                threshold=_score(value.get("threshold")),
                accepted=int(value.get("accepted", 0)),
                true_positives=int(value.get("truePositives", 0)),
                false_positives=int(value.get("falsePositives", 0)),
                precision=_score(value.get("precision")),
                recall=_score(value.get("recall")),
            )

        per_gloss_data = data.get("perGloss", {})
        if not isinstance(per_gloss_data, dict):
            raise ValueError("perGloss must be an object")
        return cls(
            model=_text(data.get("model"), "model"),
            preprocessing=_text(data.get("preprocessing"), "preprocessing"),
            dataset=_text(data.get("dataset"), "dataset"),
            source_sha256=str(data.get("sourceSha256", "")),
            target_precision=_score(data.get("targetPrecision")),
            global_metrics=metrics(data.get("global")),
            per_gloss={
                _gloss(gloss, "perGloss key"): metrics(value)
                for gloss, value in per_gloss_data.items()
            },
            observation_count=int(data.get("observationCount", 0)),
        )


def fit_calibration(
    observations: Iterable[LabelledPrediction],
    *,
    model: str,
    preprocessing: str,
    dataset: str,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_true_positives: int = DEFAULT_MIN_TRUE_POSITIVES,
) -> ConfidenceCalibration:
    """Fit global and sufficiently supported per-gloss thresholds."""
    rows = sorted(
        list(observations),
        key=lambda row: (row.clip.replace("\\", "/"), row.expected, row.predicted, row.score),
    )
    if not rows:
        raise ValueError("at least one labelled observation is required")
    global_metrics = choose_threshold(
        rows,
        target_precision=target_precision,
        min_true_positives=min_true_positives,
    )
    if global_metrics is None:
        raise ValueError(
            "labelled observations cannot meet the requested precision and sample minimum"
        )
    predictions: dict[str, list[LabelledPrediction]] = {}
    for row in rows:
        predictions.setdefault(row.predicted, []).append(row)
    per_gloss = {}
    for gloss, gloss_rows in predictions.items():
        selected = choose_threshold(
            gloss_rows,
            target_precision=target_precision,
            min_true_positives=min_true_positives,
        )
        if selected is not None:
            per_gloss[gloss] = selected
    canonical = "\n".join(
        json.dumps(row.canonical(), sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return ConfidenceCalibration(
        model=_text(model, "model"),
        preprocessing=_text(preprocessing, "preprocessing"),
        dataset=_text(dataset, "dataset"),
        source_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        target_precision=target_precision,
        global_metrics=global_metrics,
        per_gloss=per_gloss,
        observation_count=len(rows),
    )


def read_jsonl(path: str | Path) -> list[LabelledPrediction]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("row must be an object")
            rows.append(LabelledPrediction.from_mapping(value))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid observation on line {line_number}: {error}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a versioned confidence artifact from labelled JSONL predictions."
    )
    parser.add_argument("observations", help="JSONL produced by camera_evaluation.py")
    parser.add_argument("--output", required=True, help="Calibration JSON destination")
    parser.add_argument("--dataset", required=True, help="Stable dataset/version name")
    parser.add_argument("--model", default="Pose-TGCN WLASL-100")
    parser.add_argument("--preprocessing", default="wlasl_affine_256_v1")
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--min-true-positives", type=int, default=DEFAULT_MIN_TRUE_POSITIVES)
    args = parser.parse_args()
    artifact = fit_calibration(
        read_jsonl(args.observations),
        model=args.model,
        preprocessing=args.preprocessing,
        dataset=args.dataset,
        target_precision=args.target_precision,
        min_true_positives=args.min_true_positives,
    )
    artifact.save(args.output)
    print(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
