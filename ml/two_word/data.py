"""Load WLASL OpenPose keypoints for a handful of glosses and turn them into model features.

Each frame has 55 joints: 13 upper-body + 21 left-hand + 21 right-hand, the
same layout the live MediaPipe pipeline in backend/sign_inference produces.

Two feature modes:

  legacy           (x, y) of every joint, centered on the neck and scaled by
                   shoulder width over the whole clip. 110 numbers per frame.
                   This is what the first experiments used.

  pos,vel,bone,mask  (default, 371 numbers per frame) built on
                   backend/sign_inference's shared `normalize_keypoints`, so
                   training matches live inference:
                   pos   body joints relative to the neck / shoulder width,
                         and each hand relative to its own wrist / size
                         (= hand shape, independent of where the hand is)
                   vel   change in pos from the previous frame (movement)
                   bone  vectors along each finger and arm bone (joint angles)
                   mask  1 if the joint was detected in this frame, else 0
                   Any subset can be picked, e.g. --features pos,mask.

The train / val / test labels come straight from the WLASL split file.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

# Where the unzipped WLASL data lives: Cue/data (two folders up from this
# file; it's gitignored). Can be overridden with the WLASL_DATA environment
# variable or the --data-root flag in train.py.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_DATA_ROOT = os.environ.get("WLASL_DATA", os.path.join(_REPO_ROOT, "data"))

# Reuse the live-inference normalization so a model trained here sees exactly
# the same kind of numbers the camera pipeline will feed it.
sys.path.insert(0, os.path.join(_REPO_ROOT, "backend", "sign_inference"))
from landmark_normalization import normalize_keypoints  # noqa: E402

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

# The two hands are stored back to back after the body joints:
# left hand at positions 13..33, right hand at 34..54.
_L0, _R0 = len(BODY_KEEP), len(BODY_KEEP) + NUM_HAND

# When we mirror a clip for data augmentation, the right arm becomes the left
# arm and vice versa, so their joint indices have to be swapped too.
# These pairs are positions within BODY_KEEP:
#   (2, 5) shoulders, (3, 6) elbows, (4, 7) wrists, (9, 10) eyes, (11, 12) ears
_BODY_PAIRS = [(2, 5), (3, 6), (4, 7), (9, 10), (11, 12)]

# FLIP_PERM[i] = which joint should end up at position i after mirroring.
# Start with "every joint stays put", then swap the left/right pairs and the
# two hands.
FLIP_PERM = list(range(NUM_JOINTS))
for a, b in _BODY_PAIRS:
    FLIP_PERM[a], FLIP_PERM[b] = b, a
for k in range(NUM_HAND):
    FLIP_PERM[_L0 + k], FLIP_PERM[_R0 + k] = _R0 + k, _L0 + k

# Bones = pairs of connected joints. Each bone feature is the vector from the
# first joint to the second, which encodes the angle of that bone.
#   Body (positions within BODY_KEEP): neck-nose, neck-hip, and both arms
#   (neck -> shoulder -> elbow -> wrist).
_BODY_BONES = [(1, 0), (1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
#   Hand (same 21-point layout in OpenPose and MediaPipe): the wrist (0)
#   connects to the base of each finger, then each finger is a chain of 4.
_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),         # thumb
               (0, 5), (5, 6), (6, 7), (7, 8),         # index
               (0, 9), (9, 10), (10, 11), (11, 12),    # middle
               (0, 13), (13, 14), (14, 15), (15, 16),  # ring
               (0, 17), (17, 18), (18, 19), (19, 20)]  # pinky
BONES = np.array(_BODY_BONES
                 + [(_L0 + a, _L0 + b) for a, b in _HAND_BONES]
                 + [(_R0 + a, _R0 + b) for a, b in _HAND_BONES])  # (48, 2)

POS_CLIP = 4.0  # max |value| of a normalized joint coordinate (see build_features)

FEATURE_COMPONENTS = ("pos", "vel", "bone", "mask")
DEFAULT_FEATURES = ",".join(FEATURE_COMPONENTS)


def parse_features(spec: str) -> list[str]:
    """'pos,vel,mask' -> ['pos', 'vel', 'mask']; 'legacy' -> ['legacy']."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if parts == ["legacy"]:
        return parts
    bad = [p for p in parts if p not in FEATURE_COMPONENTS]
    if bad or not parts:
        raise ValueError(f"Unknown features {bad or spec!r}; use 'legacy' or any of {FEATURE_COMPONENTS}")
    return parts


def feature_dim(components: list[str]) -> int:
    """Numbers per frame for a feature spec (the model's input size)."""
    sizes = {"legacy": NUM_JOINTS * 2, "pos": NUM_JOINTS * 2, "vel": NUM_JOINTS * 2,
             "bone": len(BONES) * 2, "mask": NUM_JOINTS}
    return sum(sizes[c] for c in components)


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


def load_raw_video(pose_root: str, inst: dict) -> np.ndarray:
    """Read every frame of one clip -> (T, 55, 3) raw pixels + confidence.

    Frames with no detected person reuse the previous one.
    """
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
    return np.stack(frames)  # list of (55, 3) -> (T, 55, 3)


# ---- Legacy features -----------------------------------------------------------

def legacy_normalize(frames: np.ndarray) -> np.ndarray:
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


def legacy_augment(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random mirror, rotation, scale, shift and jitter on a normalized (T, 55, 2) clip."""
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


# ---- New features --------------------------------------------------------------

def augment_raw(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random mirror, rotation and jitter on raw (T, 55, 3) pixel keypoints.

    Done before normalization so the hand-shape features see the change too.
    (Shift and zoom aren't needed here: normalization removes them anyway.)
    """
    clip = clip.copy()  # don't modify the cached original
    xy, visible = clip[..., :2], clip[..., 2] > 0

    # Mirror half the time (simulates a left-handed signer): swap left/right
    # joints, then flip the x axis.
    if rng.random() < 0.5:
        clip = clip[:, FLIP_PERM]
        xy, visible = clip[..., :2], clip[..., 2] > 0
        xy[..., 0] *= -1

    # Rotate by up to ±10° around the neck: slightly tilted camera. Hands
    # rotate with it, so their local shape features change too.
    neck_ok = visible[:, NECK]
    center = np.median(xy[neck_ok, NECK], axis=0) if neck_ok.any() else np.zeros(2, np.float32)
    theta = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    xy[:] = (xy - center) @ rot.T + center

    # Per-joint noise of ~1% of shoulder width: detections are never exact.
    sh_ok = visible[:, R_SHOULDER] & visible[:, L_SHOULDER]
    sw = np.median(np.linalg.norm(xy[sh_ok, R_SHOULDER] - xy[sh_ok, L_SHOULDER], axis=-1)) if sh_ok.any() else 100.0
    xy += rng.normal(0, 0.01 * sw, size=xy.shape).astype(np.float32)

    return clip


def build_features(raw: np.ndarray, components: list[str]) -> np.ndarray:
    """(T, 55, 3) raw pixels + confidence -> (T, feature_dim) model input."""
    # normalize_keypoints expects NaN for undetected joints.
    pts = raw[..., :2].astype(np.float32).copy()
    pts[raw[..., 2] <= 0] = np.nan

    # Per-frame shared normalization: body relative to neck / shoulder width,
    # each hand relative to its wrist / wrist-to-middle-knuckle distance.
    # It marks joints it couldn't normalize with (-1, -1).
    pos = np.stack([normalize_keypoints(p) for p in pts])  # (T, 55, 2)
    detected = ~np.all(pos == -1, axis=-1)                  # (T, 55)
    pos[~detected] = 0.0  # the mask feature says "missing", so zero is fine here

    # A real hand stays within about ±2.5 of its wrist in these units, but a
    # bad OpenPose hand detection (wrist and middle knuckle almost on top of
    # each other) can blow values up to ±30. Cap them so a few broken frames
    # can't dominate training. Live inference must apply the same cap.
    np.clip(pos, -POS_CLIP, POS_CLIP, out=pos)

    parts = []
    if "pos" in components:
        parts.append(pos)
    if "vel" in components:
        # Movement since the previous frame; 0 for the first frame and for
        # joints missing in either frame.
        vel = np.zeros_like(pos)
        both = detected[1:] & detected[:-1]
        vel[1:] = np.where(both[..., None], pos[1:] - pos[:-1], 0.0)
        parts.append(vel)
    if "bone" in components:
        # Vector along each bone; 0 if either end wasn't detected.
        a, b = BONES[:, 0], BONES[:, 1]
        ok = detected[:, a] & detected[:, b]
        parts.append(np.where(ok[..., None], pos[:, b] - pos[:, a], 0.0))
    if "mask" in components:
        parts.append(detected.astype(np.float32))

    T = len(raw)
    return np.concatenate([p.reshape(T, -1) for p in parts], axis=1).astype(np.float32)


# ---- Shared ----------------------------------------------------------------------

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


class PoseClipDataset(Dataset):
    """Clips for one split, loaded into memory once (they're tiny).

    PyTorch's DataLoader calls __len__ to know how many examples there are and
    __getitem__(i) to fetch example i, then batches them together.
    """

    def __init__(self, instances: list[dict], pose_root: str, num_frames: int = 32,
                 train: bool = False, seed: int = 0, features: str = DEFAULT_FEATURES):
        self.instances = instances
        self.components = parse_features(features)
        self.legacy = self.components == ["legacy"]
        self.feature_dim = feature_dim(self.components)  # model input size per frame

        # Read every video's JSON files up front. Parsing JSON is the slow
        # part, and this way it happens once instead of every epoch.
        raw = [load_raw_video(pose_root, inst) for inst in instances]
        # Legacy normalizes the whole clip once; the new features are built
        # per example, after augmentation.
        self.clips = [legacy_normalize(r) for r in raw] if self.legacy else raw

        self.num_frames = num_frames  # how many frames each example is cut down to
        self.train = train            # True -> random frame sampling + augmentation
        self.rng = np.random.default_rng(seed)  # seeded so runs are reproducible

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        # Cut the full clip down to exactly num_frames frames.
        clip = clip[sample_frames(len(clip), self.num_frames, self.train, self.rng)]

        if self.legacy:
            if self.train:
                clip = legacy_augment(clip, self.rng)
            feats = clip.reshape(self.num_frames, -1)  # (T, 55, 2) -> (T, 110)
        else:
            if self.train:
                clip = augment_raw(clip, self.rng)
            feats = build_features(clip, self.components)  # (T, feature_dim)

        # Return the input, the correct answer, and the video id (useful for
        # reporting which videos the model got wrong).
        return torch.from_numpy(feats), self.instances[i]["label"], self.instances[i]["video_id"]
