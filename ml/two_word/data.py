"""Load WLASL OpenPose keypoints for a handful of glosses.

Each video becomes a (T, 55, 2) array: 13 upper-body joints + 21 left-hand
+ 21 right-hand joints, (x, y) per joint, for T frames. Coordinates are
centered on the neck and scaled by shoulder width so the model learns the
motion of the sign, not where the signer stood in the frame.

The train / val / test labels come straight from the WLASL split file.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Where the unzipped WLASL data lives: Cue/data (two folders up from this
# file; it's gitignored). Can be overridden with the WLASL_DATA environment
# variable or the --data-root flag in train.py.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_DATA_ROOT = os.environ.get("WLASL_DATA", os.path.join(_REPO_ROOT, "data"))

# OpenPose's BODY_25 model outputs 25 body joints. We keep only the 13 that
# matter for signing (head, shoulders, arms, hips); legs and feet are never
# visible in WLASL clips. The numbers are BODY_25 joint indices:
#   0 nose, 1 neck, 2 right shoulder, 3 right elbow, 4 right wrist,
#   5 left shoulder, 6 left elbow, 7 left wrist, 8 mid hip,
#   15 right eye, 16 left eye, 17 right ear, 18 left ear
BODY_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
NUM_HAND = 21  # OpenPose detects 21 points per hand (wrist + 4 per finger)
NUM_JOINTS = len(BODY_KEEP) + 2 * NUM_HAND  # 13 + 21 + 21 = 55

# Positions of these joints *within BODY_KEEP* (not BODY_25), used to
# normalize each clip.
NECK, R_SHOULDER, L_SHOULDER = 1, 2, 5

# When we mirror a clip for data augmentation, the right arm becomes the left
# arm and vice versa, so their joint indices have to be swapped too.
# These pairs are positions within BODY_KEEP:
#   (2, 5) shoulders, (3, 6) elbows, (4, 7) wrists, (9, 10) eyes, (11, 12) ears
_BODY_PAIRS = [(2, 5), (3, 6), (4, 7), (9, 10), (11, 12)]

# FLIP_PERM[i] = which joint should end up at position i after mirroring.
# Start with "every joint stays put", then swap the left/right pairs.
FLIP_PERM = list(range(NUM_JOINTS))
for a, b in _BODY_PAIRS:
    FLIP_PERM[a], FLIP_PERM[b] = b, a

# The two hands are stored back to back after the body joints:
# left hand at positions 13..33, right hand at 34..54. Swap them wholesale.
_L0, _R0 = len(BODY_KEEP), len(BODY_KEEP) + NUM_HAND
for k in range(NUM_HAND):
    FLIP_PERM[_L0 + k], FLIP_PERM[_R0 + k] = _R0 + k, _L0 + k


def load_instances(split_file: str, glosses: list[str]) -> list[dict]:
    """Return [{video_id, label, split, frame_start, frame_end, signer_id}] for the given glosses."""
    # The split file is a list of {"gloss": "book", "instances": [...]}.
    # Turn it into a dict so we can look words up by name.
    with open(split_file) as f:
        content = {entry["gloss"]: entry["instances"] for entry in json.load(f)}

    # Fail early with a clear message if a word was misspelled.
    missing = [g for g in glosses if g not in content]
    if missing:
        raise ValueError(f"Glosses not in {split_file}: {missing}")

    rows = []
    # The model predicts numbers, not words, so each gloss gets a label equal
    # to its position in the list (book -> 0, drink -> 1).
    for label, gloss in enumerate(glosses):
        for inst in content[gloss]:
            rows.append({
                "video_id": inst["video_id"],        # folder name under pose_per_individual_videos/
                "label": label,                      # class number the model should predict
                "split": inst["split"],              # "train", "val" or "test", decided by WLASL
                "frame_start": inst["frame_start"],  # first frame of the sign (1-based)
                "frame_end": inst["frame_end"],      # last frame of the sign, or -1 for "to the end"
                "signer_id": inst["signer_id"],      # which person is signing (handy for analysis)
            })
    return rows


def _read_frame(path: str) -> np.ndarray | None:
    """One OpenPose JSON -> (55, 3) array of x, y, confidence, or None if no person."""
    try:
        with open(path) as f:
            people = json.load(f)["people"]
    except FileNotFoundError:
        # A few frames are missing from the WLASL download; treat them like
        # frames where nobody was detected.
        return None
    if not people:
        # OpenPose didn't find anyone in this frame.
        return None

    # OpenPose stores keypoints as one flat list: [x0, y0, c0, x1, y1, c1, ...]
    # where c is a 0..1 confidence (0 means "not detected"). reshape(-1, 3)
    # turns that into one row per joint: [[x0, y0, c0], [x1, y1, c1], ...].
    p = people[0]  # WLASL videos have a single signer, so take the first person
    body = np.asarray(p["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)[BODY_KEEP]
    left = np.asarray(p["hand_left_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    right = np.asarray(p["hand_right_keypoints_2d"], dtype=np.float32).reshape(-1, 3)

    # Stack into one (55, 3) array: body joints, then left hand, then right hand.
    return np.concatenate([body, left, right], axis=0)


def _normalize(frames: np.ndarray) -> np.ndarray:
    """(T, 55, 3) raw pixels -> (T, 55, 2) neck-centered, shoulder-scaled coordinates."""
    xy, conf = frames[..., :2], frames[..., 2]  # split off the confidence column
    visible = conf > 0                          # True where OpenPose actually found the joint

    # Pick a center point for the clip: the neck. We take the median neck
    # position over all frames (rather than per frame) so a jittery neck
    # detection doesn't shake the whole skeleton around.
    neck_ok = visible[:, NECK]
    center = np.median(xy[neck_ok, NECK], axis=0) if neck_ok.any() else xy[visible].mean(axis=0)

    # Pick a scale for the clip: the distance between the shoulders. Dividing
    # by it means a signer close to the camera and one far away end up the
    # same size. Fall back to 100 pixels if the shoulders were never seen.
    sh_ok = visible[:, R_SHOULDER] & visible[:, L_SHOULDER]
    if sh_ok.any():
        scale = np.median(np.linalg.norm(xy[sh_ok, R_SHOULDER] - xy[sh_ok, L_SHOULDER], axis=-1))
    else:
        scale = 100.0
    scale = max(float(scale), 1e-3)  # guard against dividing by zero

    # After this, the neck sits at (0, 0) and the shoulders are about 1 unit apart.
    out = (xy - center) / scale
    out[~visible] = 0.0  # undetected joints sit at the origin
    return out.astype(np.float32)


def load_video(pose_root: str, inst: dict) -> np.ndarray:
    """Read every frame of one clip -> (T, 55, 2). Frames with no detected person reuse the previous one."""
    video_dir = os.path.join(pose_root, inst["video_id"])
    start = inst["frame_start"]
    end = inst["frame_end"]
    if end < 0:  # -1 means "until the last frame"
        # Count the keypoint files to find out how long the video is.
        end = len([n for n in os.listdir(video_dir) if n.endswith("_keypoints.json")])

    frames = []
    for i in range(start, end + 1):
        # Files are named image_00001_keypoints.json, image_00002_keypoints.json, ...
        fr = _read_frame(os.path.join(video_dir, f"image_{i:05d}_keypoints.json"))
        if fr is None:
            if frames:
                # No person in this frame: repeat the last good frame so the
                # clip has no holes.
                fr = frames[-1]
            else:
                # No good frame yet to copy, so just skip it.
                continue
        frames.append(fr)

    if not frames:
        raise RuntimeError(f"No usable pose frames for video {inst['video_id']}")
    return _normalize(np.stack(frames))  # np.stack: list of (55, 3) -> (T, 55, 3)


def sample_frames(num_frames: int, length: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """Pick `length` frame indices spanning the whole clip.

    Eval: evenly spaced. Train: each index jitters within its segment, so
    every epoch sees a slightly different version of the clip.
    """
    # Clips vary from ~18 to ~124 frames but the model needs a fixed length.
    # Split the clip into `length` equal segments and take one frame from each.
    # e.g. a 64-frame clip with length=32 -> segments [0,2), [2,4), [4,6), ...
    edges = np.linspace(0, num_frames, length + 1)
    if train:
        # Random point inside each segment: a cheap form of augmentation that
        # also simulates the sign being performed slightly faster or slower.
        idx = edges[:-1] + rng.random(length) * (edges[1:] - edges[:-1])
    else:
        # Middle of each segment, so evaluation is the same every time.
        idx = (edges[:-1] + edges[1:]) / 2
    # Short clips (fewer frames than `length`) simply repeat some frames.
    return np.clip(idx.astype(int), 0, num_frames - 1)


def augment(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random mirror, rotation, scale, shift and jitter on a (T, 55, 2) clip.

    With only ~55 training videos the model would memorize them exactly.
    Small random changes each epoch make it learn the general shape of the
    motion instead. Each change is one that a real recording could plausibly
    have (different camera angle, distance, position, a left-handed signer).
    """
    clip = clip.copy()  # don't modify the cached original
    visible = np.any(clip != 0, axis=-1)  # remember which joints were undetected (sitting at 0, 0)

    # Mirror half the time (simulates a left-handed signer): swap left/right
    # joints, then flip the x axis.
    if rng.random() < 0.5:
        clip = clip[:, FLIP_PERM]
        visible = visible[:, FLIP_PERM]
        clip[..., 0] *= -1

    # Rotate by up to ±10° around the neck (the origin): slightly tilted camera.
    theta = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    clip = clip @ rot.T

    # Zoom in or out by up to 10%: signer slightly closer or farther away.
    clip *= rng.uniform(0.9, 1.1)

    # Shift the whole skeleton a little: signer not perfectly centered.
    clip += rng.uniform(-0.1, 0.1, size=2).astype(np.float32)

    # Tiny per-joint noise: OpenPose detections are never perfectly precise.
    clip += rng.normal(0, 0.01, size=clip.shape).astype(np.float32)

    # The steps above moved undetected joints away from (0, 0); put them back
    # so "missing" still looks like "missing" to the model.
    clip[~visible] = 0.0
    return clip


class PoseClipDataset(Dataset):
    """Clips for one split, loaded into memory once (they're tiny).

    PyTorch's DataLoader calls __len__ to know how many examples there are and
    __getitem__(i) to fetch example i, then batches them together.
    """

    def __init__(self, instances: list[dict], pose_root: str, num_frames: int = 32,
                 train: bool = False, seed: int = 0):
        self.instances = instances
        # Read every video's JSON files up front. Parsing JSON is the slow
        # part, and this way it happens once instead of every epoch.
        self.clips = [load_video(pose_root, inst) for inst in instances]
        self.num_frames = num_frames  # how many frames each example is cut down to
        self.train = train            # True -> random frame sampling + augmentation
        self.rng = np.random.default_rng(seed)  # seeded so runs are reproducible

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        # Cut the full clip down to exactly num_frames frames.
        clip = clip[sample_frames(len(clip), self.num_frames, self.train, self.rng)]
        if self.train:
            clip = augment(clip, self.rng)
        # The model takes one flat vector per frame, so (T, 55, 2) -> (T, 110).
        x = torch.from_numpy(clip.reshape(self.num_frames, -1))
        # Return the input, the correct answer, and the video id (useful for
        # reporting which videos the model got wrong).
        return x, self.instances[i]["label"], self.instances[i]["video_id"]
