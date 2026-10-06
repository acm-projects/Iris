"""Shared pivot normalization for training and live 55-joint inputs."""
import numpy as np

NORMALIZATION_VERSION = "pivot_v1"
MIN_SCALE_PIXELS = 1.0


def normalize_keypoints(points):
    """Use neck/shoulder width for body; wrist/middle MCP distance for hands.

    Input coordinates must be pixels with equal units for x/y. Invalid
    points or groups with missing/degenerate reference joints become (-1,-1).
    Hand coordinates describe local shape; body wrists retain hand placement.
    Values are not clipped, and inputs are not modified.
    References below one pixel are unresolved at image resolution and masked;
    exactly one pixel is accepted. The absolute cutoff breaks scale invariance
    when rescaling crosses its boundary. No epsilon or clipping is used.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (55, 2):
        raise ValueError(f"Expected (55, 2) landmarks; got {points.shape}.")
    output = np.full((55, 2), -1, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    for start, end, pivot, scale_a, scale_b in (
        (0, 13, 1, 2, 5),
        (13, 34, 13, 13, 22),
        (34, 55, 34, 34, 43),
    ):
        if not valid[[pivot, scale_a, scale_b]].all():
            continue
        scale = np.linalg.norm(points[scale_a].astype(np.float64) - points[scale_b])
        if not np.isfinite(scale) or scale < MIN_SCALE_PIXELS:
            continue
        indices = np.arange(start, end)[valid[start:end]]
        values = (points[indices].astype(np.float64) - points[pivot]) / scale
        finite = (np.isfinite(values) & (np.abs(values) <= np.finfo(np.float32).max)).all(axis=1)
        output[indices[finite]] = values[finite]
    return output


LEGACY_AFFINE_VERSION = "legacy_affine_256_candidate"


def legacy_affine_keypoints(points):
    """Reproduce historical raw-JSON xy arithmetic, not a verified HF contract.

    Preserve global body/hand placement: x and y are each 2*(pixel/256 - .5).
    Historical confidence is ignored, and raw zero coordinates become -1.
    Live nonfinite observations map individually to zero pixels before scaling.
    No pivot distance, one-pixel threshold, clipping, or epsilon applies here.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (55, 2):
        raise ValueError(f"Expected (55, 2) landmarks; got {points.shape}.")
    if np.isinf(points).any():
        raise ValueError("Infinite pixel coordinates are invalid.")
    pixels = np.where(np.isnan(points), np.float32(0), points)
    return np.float32(2) * (pixels / np.float32(256) - np.float32(.5))


def preprocess_keypoints(points, preprocessing=NORMALIZATION_VERSION):
    """Select an explicit coordinate contract; neither proves weight provenance."""
    if preprocessing == NORMALIZATION_VERSION:
        return normalize_keypoints(points)
    if preprocessing == LEGACY_AFFINE_VERSION:
        return legacy_affine_keypoints(points)
    raise ValueError(f"Unknown preprocessing: {preprocessing}")
