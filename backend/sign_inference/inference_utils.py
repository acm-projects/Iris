import json
from pathlib import Path
import numpy as np
from landmark_normalization import NORMALIZATION_VERSION, preprocess_keypoints

MODEL_DIR = Path(__file__).resolve().parent / 'mediapipe_models'
CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints' / 'asl100'
SPLIT_FILE = Path(__file__).resolve().parent / 'asl100.json'
MODEL_NUM_FRAMES = 50
MODEL_NUM_LANDMARKS = 55
LEFT_WRIST, RIGHT_WRIST = 7, 4  # signer's wrists in the 55-joint body block


def remap_landmarks(pose_result, hand_result, hand_assignment="label"):
    """MediaPipe pose + hands -> (55, 2) points in OpenPose order, NaN = missing.

    hand_assignment chooses which slot (signer's left 13..33, right 34..54)
    each detected hand goes to:
      "label": MediaPipe's handedness label (original behavior).
      "wrist": the nearer pose wrist. MediaPipe's labels assume a mirrored
               (selfie) image, so on unmirrored frames they can come out
               swapped; the pose wrists are anatomical, so matching each hand
               to the closest wrist puts it in the right slot either way.
               Falls back to labels when no body is detected.
    """
    if hand_assignment not in ("label", "wrist"):
        raise ValueError(f"hand_assignment must be 'label' or 'wrist', got {hand_assignment!r}")
    # NaN marks missing coordinates during this debugging stage.
    points = np.full((55, 2), np.nan, dtype=np.float32)

    if pose_result.pose_landmarks:
        pose = pose_result.pose_landmarks[0]

        def xy(index):
            return np.array(
                [pose[index].x, pose[index].y],
                dtype=np.float32,
            )

        # Retained OpenPose BODY_25 order.
        points[:13] = np.array([
            xy(0),                  # Nose
            (xy(11) + xy(12)) / 2,  # Estimated neck
            xy(12),                 # Right shoulder
            xy(14),                 # Right elbow
            xy(16),                 # Right wrist
            xy(11),                 # Left shoulder
            xy(13),                 # Left elbow
            xy(15),                 # Left wrist
            (xy(23) + xy(24)) / 2,  # Hip midpoint
            xy(5),                  # Right eye
            xy(2),                  # Left eye
            xy(8),                  # Right ear
            xy(7),                  # Left ear
        ])

    if hand_assignment == "wrist" and pose_result.pose_landmarks:
        return _assign_hands_by_wrist(points, hand_result)

    # Use handedness labels, not detection-list order.
    best_score = {"Left": -1.0, "Right": -1.0}
    detected_sides = []

    for landmarks, categories in zip(
        hand_result.hand_landmarks,
        hand_result.handedness,
    ):
        if not categories:
            continue

        category = categories[0]
        side = category.category_name
        detected_sides.append(side)

        if side not in best_score or len(landmarks) != 21:
            continue

        # If two detections receive the same label, keep the stronger one.
        if category.score <= best_score[side]:
            continue

        best_score[side] = category.score
        start = 13 if side == "Left" else 34

        points[start:start + 21] = np.array(
            [[landmark.x, landmark.y] for landmark in landmarks],
            dtype=np.float32,
        )

    return points, detected_sides


def _assign_hands_by_wrist(points, hand_result):
    """Place up to two detected hands by distance from each hand's wrist to the pose wrists."""
    hands = []
    for landmarks, categories in zip(hand_result.hand_landmarks, hand_result.handedness):
        if len(landmarks) != 21:
            continue
        xy = np.array([[landmark.x, landmark.y] for landmark in landmarks], dtype=np.float32)
        score = categories[0].score if categories else 0.0
        hands.append((score, xy))
    # At most two hands; keep the most confident ones.
    hands = sorted(hands, key=lambda h: -h[0])[:2]

    def dist(xy, wrist):
        target = points[wrist]
        return float(np.linalg.norm(xy[0] - target)) if np.isfinite(target).all() else np.inf

    if len(hands) == 2:
        (_, a), (_, b) = hands
        # Pick the pairing with the smaller total wrist distance.
        straight = dist(a, LEFT_WRIST) + dist(b, RIGHT_WRIST)
        crossed = dist(a, RIGHT_WRIST) + dist(b, LEFT_WRIST)
        pairs = [(a, "Left"), (b, "Right")] if straight <= crossed else [(a, "Right"), (b, "Left")]
    elif len(hands) == 1:
        (_, a), = hands
        pairs = [(a, "Left" if dist(a, LEFT_WRIST) <= dist(a, RIGHT_WRIST) else "Right")]
    else:
        pairs = []
    sides = []
    for xy, side in pairs:
        start = 13 if side == "Left" else 34
        points[start:start + 21] = xy
        sides.append(side)
    return points, sides


def normalize_landmarks(points, preprocessing=NORMALIZATION_VERSION):
    """Normalize equal-unit x/y coordinates using shared training pivots."""
    return preprocess_keypoints(points, preprocessing)


def assemble_model_input(normalized_frames):
    """Pack 50 normalized frames as (1, 55, 100), preserving frame order."""
    frames = np.asarray(normalized_frames, dtype=np.float32)
    expected_shape = (MODEL_NUM_FRAMES, MODEL_NUM_LANDMARKS, 2)
    if frames.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} normalized frames; got {frames.shape}."
        )
    if not np.isfinite(frames).all():
        raise ValueError("Model input cannot contain NaN or infinite coordinates.")

    return frames.transpose(1, 0, 2).reshape(
        1,
        MODEL_NUM_LANDMARKS,
        MODEL_NUM_FRAMES * 2,
    )


def prepare_recording(records, num_frames=MODEL_NUM_FRAMES):
    """Select/pad a recording to a fixed count without dropping missing frames.

    Long recordings use a consecutive middle window, with an odd extra frame
    taken from the end of that window. Short recordings repeat the last frame
    and timestamp. No temporal interpolation or compression is performed.
    """
    if not records:
        raise ValueError("No frames were recorded; press R before S.")
    if num_frames <= 0:
        raise ValueError("The prepared frame count must be positive.")

    validated_records = []
    for index, record in enumerate(records):
        if len(record) != 2:
            raise ValueError(
                f"Recorded item {index} must contain a timestamp and landmarks."
            )
        timestamp, points = record
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Recorded item {index} has an invalid timestamp."
            ) from error
        if not np.isfinite(timestamp):
            raise ValueError(f"Recorded item {index} has a non-finite timestamp.")

        points = np.asarray(points, dtype=np.float32)
        if points.shape != (MODEL_NUM_LANDMARKS, 2):
            raise ValueError(
                f"Recorded item {index} has landmark shape {points.shape}; "
                f"expected ({MODEL_NUM_LANDMARKS}, 2)."
            )
        validated_records.append((timestamp, points))

    if len(validated_records) > num_frames:
        start = (len(validated_records) - num_frames) // 2
        validated_records = validated_records[start:start + num_frames]
    elif len(validated_records) < num_frames:
        validated_records.extend(
            [validated_records[-1]] * (num_frames - len(validated_records))
        )

    timestamps = [timestamp for timestamp, _ in validated_records]
    frames = np.stack([points for _, points in validated_records])
    return frames, timestamps


def count_missing_points(frames):
    """Count missing joint observations by body, left hand, and right hand."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[1:] != (MODEL_NUM_LANDMARKS, 2):
        raise ValueError(
            "Expected frames shaped (T, 55, 2) to count missing landmarks."
        )
    point_missing = ~np.isfinite(frames).all(axis=2)
    return (
        int(point_missing[:, :13].sum()),
        int(point_missing[:, 13:34].sum()),
        int(point_missing[:, 34:55].sum()),
    )


def class_labels_from_entries(entries, expected_class_count=100):
    """Match Sign_Dataset's alphabetically sorted gloss-label encoding."""
    if not isinstance(entries, list):
        raise ValueError("ASL split JSON must contain a list of gloss entries.")
    labels = []
    for index, entry in enumerate(entries):
        gloss = entry.get("gloss") if isinstance(entry, dict) else None
        if not isinstance(gloss, str) or not gloss:
            raise ValueError(f"ASL split entry {index} has no valid gloss label.")
        labels.append(gloss)

    sorted_labels = sorted(labels)
    if len(sorted_labels) != expected_class_count:
        raise ValueError(
            f"Expected {expected_class_count} gloss labels; "
            f"found {len(sorted_labels)}."
        )
    if len(set(sorted_labels)) != expected_class_count:
        raise ValueError(
            f"Expected {expected_class_count} unique gloss labels; "
            f"found {len(set(sorted_labels))}."
        )
    return sorted_labels


def load_class_labels(split_file=SPLIT_FILE, expected_class_count=100):
    with open(split_file, "r", encoding="utf-8") as split:
        entries = json.load(split)
    return class_labels_from_entries(entries, expected_class_count)



