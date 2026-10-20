"""Measure how visible each word's hands are in WLASL, to pick signs a pose model can see.

For every word in a split file, over its *training* videos, this reports:
  hands          one / two / mixed: is the second hand raised and moving, or resting?
  visibility     share of hand joints detected with confidence > 0.3 while a hand is signing
  overlap        share of two-hand frames where the wrists are under 0.6 shoulder-widths apart
                 (hands touching or crossing, where joints get hidden)
  hand_conf      mean OpenPose confidence of the signing hand(s)

and writes a CSV, most visible first.

  python ml/hand_visibility.py
  python ml/hand_visibility.py --split-file data/splits/asl2000.json --output ml/two_word/vocab/hand_visibility_asl2000.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("two_word_data", REPO / "ml" / "two_word" / "data.py")
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

NECK, R_SH, L_SH, R_WRIST, L_WRIST = 1, 2, 5, 4, 7
HANDS = {"left": (slice(13, 34), L_WRIST), "right": (slice(34, 55), R_WRIST)}
CONF = 0.3


def clip_stats(raw: np.ndarray) -> dict | None:
    """(T, 55, 3) OpenPose pixels + confidence -> per-clip hand statistics."""
    xy, conf = raw[..., :2], raw[..., 2]
    sh_ok = (conf[:, R_SH] > 0) & (conf[:, L_SH] > 0) & (conf[:, NECK] > 0)
    if sh_ok.sum() < 3:
        return None
    sw = np.median(np.linalg.norm(xy[sh_ok, R_SH] - xy[sh_ok, L_SH], axis=-1))
    neck_y = np.median(xy[sh_ok, NECK, 1])
    if sw < 1:
        return None

    active, vis, hconf = {}, [], []
    for name, (joints, wrist) in HANDS.items():
        w_ok = conf[:, wrist] > 0
        # Signing space: wrist above roughly mid-chest (image y grows downward).
        raised = w_ok & (xy[:, wrist, 1] < neck_y + 1.0 * sw)
        motion = float(np.std(xy[w_ok, wrist], axis=0).mean() / sw) if w_ok.sum() > 2 else 0.0
        is_active = raised.mean() > 0.3 or motion > 0.15
        active[name] = (is_active, raised)
        if is_active and raised.any():
            c = conf[raised][:, joints]
            vis.append(float((c > CONF).mean()))
            hconf.append(float(c.mean()))

    if not vis:  # no hand in signing space at all
        return {"two": False, "vis": 0.0, "conf": 0.0, "overlap": 0.0}
    two = active["left"][0] and active["right"][0]
    overlap = 0.0
    if two:
        both = active["left"][1] & active["right"][1]
        if both.any():
            d = np.linalg.norm(xy[both, L_WRIST] - xy[both, R_WRIST], axis=-1) / sw
            overlap = float((d < 0.6).mean())
    return {"two": two, "vis": float(np.mean(vis)), "conf": float(np.mean(hconf)), "overlap": overlap}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", default=str(REPO / "data" / "splits" / "asl2000.json"))
    ap.add_argument("--pose-root", default=str(REPO / "data" / "pose_per_individual_videos"))
    ap.add_argument("--output", default=str(REPO / "ml" / "two_word" / "vocab" / "hand_visibility_asl2000.csv"))
    args = ap.parse_args()

    entries = json.load(open(args.split_file))
    rows = []
    for k, e in enumerate(entries, 1):
        stats = []
        for inst in e["instances"]:
            if inst["split"] != "train":
                continue
            try:
                s = clip_stats(data.load_raw_video(args.pose_root, inst))
            except (RuntimeError, FileNotFoundError):
                s = None
            if s:
                stats.append(s)
        if not stats:
            continue
        two = np.mean([s["two"] for s in stats])
        rows.append({
            "word": e["gloss"],
            "train_clips": len(stats),
            "hands": "two" if two >= 0.7 else "one" if two <= 0.3 else "mixed",
            "two_hand_share": round(float(two), 3),
            "visibility": round(float(np.mean([s["vis"] for s in stats])), 3),
            "overlap": round(float(np.mean([s["overlap"] for s in stats])), 3),
            "hand_conf": round(float(np.mean([s["conf"] for s in stats])), 3),
        })
        if k % 250 == 0:
            print(f"  {k}/{len(entries)} words")

    rows.sort(key=lambda r: (-r["visibility"], r["overlap"]))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} words to {args.output}")


if __name__ == "__main__":
    main()
