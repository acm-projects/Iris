import collections
import numpy as np

NUM_SAMPLES = 50            # frames the model expects (check num_samples in the TGCN config)
USE_SHOULDER_NORM = True    # True = shoulder-based, False = the model's original frame-based scaling


def hand_xy(hand):
    if hand is None:
        return None
    points = []
    for lm in hand.landmark:
        points.append([lm.x, lm.y])
    return np.array(points)


def body_xy(pose):
    if pose is None:
        return None
    pts = np.array([[lm.x, lm.y] for lm in pose.landmark])   # (33, 2)

    neck = (pts[11] + pts[12]) / 2

    hips_visible = pose.landmark[23].visibility > 0.5 and pose.landmark[24].visibility > 0.5
    if hips_visible:
        mid_hip = (pts[23] + pts[24]) / 2
    else:
        mid_hip = np.array([0.0, 0.0])

    rows = [
        pts[0],     # 0  nose
        neck,       # 1  neck
        pts[12],    # 2  right shoulder
        pts[14],    # 3  right elbow
        pts[16],    # 4  right wrist
        pts[11],    # 5  left shoulder
        pts[13],    # 6  left elbow
        pts[15],    # 7  left wrist
        mid_hip,    # 8  mid-hip
        pts[5],     # 9  right eye
        pts[2],     # 10 left eye
        pts[8],     # 11 right ear
        pts[7],     # 12 left ear
    ]
    return np.array(rows)


def normalize_frame(raw):
    # the model's original scaling: 0..1 becomes -1..1 across the whole frame
    return 2 * (raw - 0.5)


def normalize_shoulder(raw):
    # shift so the neck is (0, 0), then scale so the shoulder width is 1
    neck = raw[1]
    shoulder_width = np.linalg.norm(raw[2] - raw[5])   # right shoulder to left shoulder
    if shoulder_width < 1e-6:
        return None                                    # can't scale, caller repeats the last frame

    missing = np.all(raw == 0, axis=1)                 # points that were never detected
    out = (raw - neck) / shoulder_width
    out[missing] = 0                                   # keep missing points as 0, not garbage
    return out.astype(np.float32)


def frame_to_keypoints(results):
    """MediaPipe results -> normalized (55, 2) array, or None if unusable."""
    body = body_xy(results.pose_landmarks)
    if body is None:
        return None

    left = hand_xy(results.left_hand_landmarks)
    right = hand_xy(results.right_hand_landmarks)

    raw = np.zeros((55, 2), dtype=np.float32)   # undetected stays 0
    raw[0:13] = body
    if left is not None:
        raw[13:34] = left
    if right is not None:
        raw[34:55] = right

    if USE_SHOULDER_NORM:
        return normalize_shoulder(raw)
    return normalize_frame(raw)


def new_buffer():
    return collections.deque(maxlen=NUM_SAMPLES)


def add_frame(buffer, kp):
    if kp is None and len(buffer) > 0:
        kp = buffer[-1]          # nothing usable: repeat the previous frame
    if kp is not None:
        buffer.append(kp)


def buffer_to_model_input(buffer):
    if len(buffer) < NUM_SAMPLES:
        return None
    clip = np.stack(buffer)                          # (50, 55, 2)
    x = clip.transpose(1, 0, 2).reshape(55, -1)      # (55, 100): x and y interleaved per frame
    return x[np.newaxis, ...]                        # (1, 55, 100)