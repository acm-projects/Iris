"""Evaluate labelled camera clips through the exact live recognition path.

The manifest is JSON with either a top-level list or ``{"clips": [...]}``.
Each clip entry has ``path`` and ``label`` fields. Video paths are resolved
relative to the manifest. The command writes top-one observations as JSONL
and fits a deterministic confidence artifact from those labelled outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from confidence_calibration import LabelledPrediction, fit_calibration
from live_recognizer import LiveSignRecognizer, MotionSegmenter
from sign_inference import SignInference


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("clips")
    if not isinstance(data, list) or not data:
        raise ValueError("manifest must contain a nonempty clips list")
    clips = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"clip {index} must be an object")
        label = str(item.get("label", "")).strip()
        relative = str(item.get("path", "")).strip()
        if not label or not relative:
            raise ValueError(f"clip {index} requires path and label")
        video = (manifest_path.parent / relative).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"clip {index} does not exist: {video}")
        clips.append({"path": str(video), "label": label})
    return clips


def evaluate_clip(
    session: SignInference,
    path: str | Path,
    *,
    first_timestamp_ms: int,
) -> tuple[list[dict], int]:
    """Return predictions for every live segment found in one labelled clip."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    sample_period_ms = LiveSignRecognizer.SAMPLE_INTERVAL_MS
    next_sample_ms = 0.0
    frame_index = 0
    timestamp_ms = int(first_timestamp_ms)
    segmenter = MotionSegmenter()
    segments = []
    try:
        while True:
            available, frame_bgr = capture.read()
            if not available:
                break
            source_time_ms = frame_index * 1000.0 / source_fps
            frame_index += 1
            if source_time_ms + 0.001 < next_sample_ms:
                continue
            next_sample_ms += sample_period_ms
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            session.process_frame(frame_rgb, timestamp_ms)
            points = session.last_model_points
            if points is not None:
                update = segmenter.observe(
                    timestamp_ms,
                    points,
                    (256, 256, 3),
                )
                if update.completed:
                    segments.append(update.completed)
            timestamp_ms += sample_period_ms
        final = segmenter.flush()
        if final.completed:
            segments.append(final.completed)
    finally:
        capture.release()
    return [session.predict_keypoints(segment) for segment in segments], timestamp_ms


def evaluate_manifest(
    manifest: str | Path,
    *,
    device: str = "cpu",
) -> tuple[list[LabelledPrediction], dict[str, object]]:
    clips = read_manifest(manifest)
    observations: list[LabelledPrediction] = []
    failures = []
    preprocessing = ""
    for clip in clips:
        try:
            # MediaPipe VIDEO trackers retain temporal state, so each isolated
            # labelled clip gets a fresh session instead of inheriting the
            # previous clip's final pose.
            with SignInference(device=device) as session:
                preprocessing = session.preprocessing
                results, _ = evaluate_clip(
                    session,
                    clip["path"],
                    first_timestamp_ms=0,
                )
            if not results:
                failures.append({"clip": clip["path"], "reason": "no live segment"})
                continue
            for segment_index, result in enumerate(results):
                top = result["predictions"][0]
                observations.append(
                    LabelledPrediction(
                        expected=clip["label"],
                        predicted=top["gloss"],
                        score=float(top["score"]),
                        clip=f"{Path(clip['path']).name}#{segment_index + 1}",
                    )
                )
        except Exception as error:  # keep the complete evaluation report
            failures.append(
                {
                    "clip": clip["path"],
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    correct = sum(row.correct for row in observations)
    summary = {
        "clips": len(clips),
        "segments": len(observations),
        "correct": correct,
        "top1Accuracy": correct / len(observations) if observations else 0.0,
        "failedClips": failures,
        "preprocessing": preprocessing,
    }
    return observations, summary


def write_jsonl(path: str | Path, rows: list[LabelledPrediction]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row.canonical(), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate labelled camera clips and fit confidence thresholds."
    )
    parser.add_argument("manifest")
    parser.add_argument("--observations", required=True, help="JSONL evaluation output")
    parser.add_argument("--calibration", required=True, help="Calibration JSON output")
    parser.add_argument("--dataset", required=True, help="Stable dataset/version name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--min-true-positives", type=int, default=3)
    args = parser.parse_args()

    observations, summary = evaluate_manifest(args.manifest, device=args.device)
    write_jsonl(args.observations, observations)
    if not observations:
        raise SystemExit("No live sign segments were recognized; calibration was not created.")
    calibration = fit_calibration(
        observations,
        model="Pose-TGCN WLASL-100",
        preprocessing=str(summary["preprocessing"]),
        dataset=args.dataset,
        target_precision=args.target_precision,
        min_true_positives=args.min_true_positives,
    )
    calibration.save(args.calibration)
    print(
        json.dumps(
            {
                "summary": summary,
                "calibration": calibration.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
