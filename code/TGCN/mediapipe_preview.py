import json
from pathlib import Path
import time
import numpy as np

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmark_normalization import NORMALIZATION_VERSION, preprocess_keypoints
from checkpoint_compatibility import ASL100_AUDIT, require_compatible
from configs import Config
from tgcn_model import GCN_muti_att

MODEL_DIR = Path(__file__).resolve().parent / "mediapipe_models"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "asl100"
SPLIT_FILE = (
    Path(__file__).resolve().parents[2]
    / "WLASL"
    / "data"
    / "splits"
    / "asl100.json"
)
MODEL_NUM_FRAMES = 50
MODEL_NUM_LANDMARKS = 55
MODEL_MISSING_COORDINATE = -1.0


def draw_points(frame, landmarks, color):
    """Convert normalized coordinates to pixels for display."""
    height, width = frame.shape[:2]

    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 3, color, -1)

def draw_connections(frame, landmarks, connections, color):
    height, width = frame.shape[:2]

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]

        start_xy = (int(start.x * width), int(start.y * height))
        end_xy = (int(end.x * width), int(end.y * height))

        if all(
            0 <= x < width and 0 <= y < height
            for x, y in (start_xy, end_xy)
        ):
            cv2.line(frame, start_xy, end_xy, color, 2)

from inference_utils import (
    remap_landmarks, normalize_landmarks, assemble_model_input, prepare_recording,
    count_missing_points, class_labels_from_entries, load_class_labels,
)


def load_asl100_model():
    """Load and validate the matching ASL100 configuration and checkpoint."""
    require_compatible(ASL100_AUDIT)
    config_path = CHECKPOINT_DIR / "config.ini"
    checkpoint_path = CHECKPOINT_DIR / "pytorch_model.bin"
    for required_path in (config_path, checkpoint_path, SPLIT_FILE):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required ASL100 file not found: {required_path}")

    config = Config(str(config_path))
    if config.num_samples != MODEL_NUM_FRAMES:
        raise ValueError(
            f"ASL100 config expects {config.num_samples} frames, "
            f"but recording preparation uses {MODEL_NUM_FRAMES}."
        )

    labels = load_class_labels(SPLIT_FILE, expected_class_count=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN_muti_att(
        input_feature=config.num_samples * 2,
        hidden_feature=config.hidden_size,
        num_class=len(labels),
        p_dropout=config.drop_p,
        num_stage=config.num_stages,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} does not contain a state dictionary."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    print("Checkpoint missing keys:", list(incompatible.missing_keys))
    print("Checkpoint unexpected keys:", list(incompatible.unexpected_keys))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "ASL100 checkpoint keys do not match the configured model; "
            "refusing to run inference."
        )

    model.preprocessing_audit = ASL100_AUDIT
    model.eval()
    print(f"Loaded ASL100 model on {device}; labels: {len(labels)}")
    return model, labels, device, config


def predict_top_words(model, labels, device, model_input):
    """Return the top three labels and softmax model scores."""
    model_input = np.asarray(model_input)
    expected_shape = (1, MODEL_NUM_LANDMARKS, MODEL_NUM_FRAMES * 2)
    if model_input.shape != expected_shape:
        raise ValueError(
            f"Expected model input {expected_shape}; got {model_input.shape}."
        )
    if model_input.dtype != np.float32:
        raise ValueError("Model input must have dtype float32.")
    if not np.isfinite(model_input).all():
        raise ValueError("Model input cannot contain NaN or infinite values.")
    if len(labels) != 100:
        raise ValueError(f"Expected 100 class labels; got {len(labels)}.")

    require_compatible(getattr(model, "preprocessing_audit", None))
    input_tensor = torch.from_numpy(model_input).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        scores = torch.softmax(logits, dim=1)
        top_scores, top_indices = torch.topk(scores[0], k=3)
    return [
        (labels[index], float(score))
        for index, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist())
    ]


def format_landmark_debug(points, normalized_points, detected_sides):
    """Summarize remapping, missing-point counts, and normalized coordinates."""
    missing_body = int((~np.isfinite(points[:13]).all(axis=1)).sum())
    missing_left = int((~np.isfinite(points[13:34]).all(axis=1)).sum())
    missing_right = int((~np.isfinite(points[34:55]).all(axis=1)).sum())

    samples = (
        ("nose[0]", 0),
        ("neck[1]", 1),
        ("L0[13]", 13),
        ("L20[33]", 33),
        ("R0[34]", 34),
        ("R20[54]", 54),
    )
    coordinates = " ".join(
        f"{name}=({normalized_points[index, 0]:.3f},"
        f"{normalized_points[index, 1]:.3f})"
        for name, index in samples
    )
    return (
        f"Detected labels: {detected_sides} | "
        f"Missing body/L/R: {missing_body}/{missing_left}/{missing_right} | "
        f"{coordinates}"
    )

def main():
    try:
        model, labels, device, config = load_asl100_model()
    except (RuntimeError, ValueError, OSError) as error:
        model, labels, device, config = None, None, None, None
        print(f"Predictions disabled: {error}")
    recording = []
    recording_active = False
    prediction_lines = []
    message = "Press R to record a sign."

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_DIR / "pose_landmarker.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_DIR / "hand_landmarker.task")
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the webcam.")

    start_time = time.monotonic()
    previous_timestamp = -1
    last_report = -1000

    try:
        with (
            vision.PoseLandmarker.create_from_options(pose_options) as pose,
            vision.HandLandmarker.create_from_options(hand_options) as hands,
        ):
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a webcam frame.")
                    break

                # OpenCV supplies BGR; MediaPipe needs RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # Both detectors receive the same frame and timestamp.
                timestamp = int((time.monotonic() - start_time) * 1000)
                timestamp = max(timestamp, previous_timestamp + 1)
                previous_timestamp = timestamp

                pose_result = pose.detect_for_video(image, timestamp)
                hand_result = hands.detect_for_video(image, timestamp)

                points, detected_sides = remap_landmarks(pose_result, hand_result)
                # Convert to pixels so x/y distances respect camera aspect ratio.
                height, width = frame.shape[:2]
                points *= np.array([width, height], dtype=np.float32)

                normalized_points = normalize_landmarks(points)
                if recording_active:
                    recording.append((timestamp, points.copy()))

                # Print once per second so the terminal stays readable.
                if timestamp - last_report >= 1000:
                    print(
                        f"Remapped shape/dtype: {points.shape}/{points.dtype} | "
                        f"Normalized shape/dtype: "
                        f"{normalized_points.shape}/{normalized_points.dtype}"
                    )
                    print(format_landmark_debug(
                        points,
                        normalized_points,
                        detected_sides,
                    ))
                    last_report = timestamp

                for landmarks in pose_result.pose_landmarks:
                    draw_connections(
                        frame,
                        landmarks,
                        vision.PoseLandmarksConnections.POSE_LANDMARKS,
                        (0, 255, 0),
                    )
                    draw_points(frame, landmarks, (0, 255, 0))

                for landmarks in hand_result.hand_landmarks:
                    draw_connections(
                        frame,
                        landmarks,
                        [
                            c for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
                            if (c.start, c.end) not in {(0, 5), (5, 9), (9, 13), (13, 17)}
                        ],
                        (0, 165, 255),
                    )
                    draw_points(frame, landmarks, (0, 165, 255))

                # Mirror only the preview, after landmark detection.
                preview = cv2.flip(frame, 1)

                status = (
                    f"{'RECORDING' if recording_active else 'IDLE'} "
                    f"frames: {len(recording)} | "
                    f"Body: {len(pose_result.pose_landmarks)} | "
                    f"Hands: {len(hand_result.hand_landmarks)} | "
                    f"R record S stop Q quit"
                )
                cv2.putText(
                    preview, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2,
                )
                cv2.putText(
                    preview, message, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2,
                )
                for line_index, prediction in enumerate(prediction_lines):
                    cv2.putText(
                        preview, prediction, (10, 84 + line_index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2,
                    )

                cv2.imshow("MediaPipe preview", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    if recording_active:
                        message = "Already recording; press S to stop."
                    else:
                        recording.clear()
                        prediction_lines = []
                        recording_active = True
                        message = "Recording started."
                elif key == ord("s"):
                    if not recording_active:
                        message = "Not recording; press R to start."
                        continue

                    recording_active = False
                    captured_count = len(recording)
                    try:
                        frames, frame_timestamps = prepare_recording(
                            recording,
                            MODEL_NUM_FRAMES,
                        )
                        missing_body, missing_left, missing_right = (
                            count_missing_points(frames)
                        )
                        normalized_frames = np.stack([
                            normalize_landmarks(frame)
                            for frame in frames
                        ])
                        model_input = assemble_model_input(normalized_frames)
                        input_is_finite = bool(np.isfinite(model_input).all())
                        input_diagnostic = (
                            f"Recorded model input: shape={model_input.shape}, "
                            f"dtype={model_input.dtype}, "
                            f"all_finite={input_is_finite}"
                        )
                        print(input_diagnostic)
                        if model is None:
                            prediction_lines = []
                            message = f"Recording stopped: {captured_count} frames. Predictions disabled."
                            print(message)
                            continue
                        top_words = predict_top_words(
                            model,
                            labels,
                            device,
                            model_input,
                        )
                        prediction_lines = [
                            f"{rank}. {word}: model score {score:.3f}"
                            for rank, (word, score) in enumerate(top_words, start=1)
                        ]
                        message = (
                            f"Input (1,55,100) finite={input_is_finite} | "
                            f"Frames captured/prepared: {captured_count}/"
                            f"{len(frame_timestamps)} | "
                            f"Missing B/L/R observations: "
                            f"{missing_body}/650, {missing_left}/1050, "
                            f"{missing_right}/1050 | "
                            f"timestamp span: {frame_timestamps[0]:g}-"
                            f"{frame_timestamps[-1]:g} ms"
                        )
                        print(message)
                        print("Top predictions (model scores, not verified confidence):")
                        for prediction in prediction_lines:
                            print(prediction)
                    except ValueError as error:
                        prediction_lines = []
                        message = f"Cannot infer: {error}"
                        print(message)

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
