"""Evaluate a trained checkpoint on one WLASL split (test by default).

  python ml/two_word/evaluate.py
  python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/book_drink/best.pt --split val
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data import NUM_JOINTS, PoseClipDataset, load_instances
from model import PoseGRU

# Folder this script lives in, used to find runs/ by default.
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "runs", "book_drink", "best.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    # Load the checkpoint saved by train.py. It holds the weights plus the
    # settings used for training, so we can rebuild everything the same way.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, glosses = ckpt["config"], ckpt["glosses"]

    # Load the clips for the requested split. No augmentation here
    # (PoseClipDataset defaults to train=False).
    split_file = cfg["split_file"] or os.path.join(cfg["data_root"], "splits", "asl100.json")
    pose_root = os.path.join(cfg["data_root"], "pose_per_individual_videos")
    rows = [r for r in load_instances(split_file, glosses) if r["split"] == args.split]
    loader = DataLoader(PoseClipDataset(rows, pose_root, cfg["num_frames"]), batch_size=64)

    # Rebuild the model with the same shape as during training, then load the
    # trained weights into it. eval() turns dropout off.
    model = PoseGRU(NUM_JOINTS * 2, len(glosses), cfg["hidden_size"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # confusion[true][predicted] counts how often each true word was predicted
    # as each word. The diagonal is the correct predictions.
    confusion = [[0] * len(glosses) for _ in glosses]
    mistakes = []
    with torch.no_grad():  # no gradients needed, we're not training
        for x, y, vids in loader:
            probs = model(x).softmax(1)  # softmax turns raw scores into probabilities that sum to 1
            for p, t, v in zip(probs, y.tolist(), vids):
                pred = int(p.argmax())   # the word with the highest probability
                confusion[t][pred] += 1
                if pred != t:
                    # Record (video id, true word, predicted word, model's confidence).
                    mistakes.append((v, glosses[t], glosses[pred], float(p[pred])))

    # Overall accuracy = correct predictions (the diagonal) / all predictions.
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(len(glosses)))
    print(f"Checkpoint: epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.2%}")
    print(f"{args.split} accuracy: {correct}/{total} = {correct / total:.2%}\n")

    # Print the confusion matrix as an aligned table.
    width = max(len(g) for g in glosses) + 2
    print("rows = true, columns = predicted")
    print(" " * width + "".join(g.rjust(width) for g in glosses))
    for g, row in zip(glosses, confusion):
        print(g.ljust(width) + "".join(str(n).rjust(width) for n in row))

    # List the videos it got wrong, so you can look at them (or their pose
    # files) to see why.
    if mistakes:
        print("\nMisclassified:")
        for v, t, p, conf in mistakes:
            print(f"  video {v}: true {t}, predicted {p} ({conf:.0%})")


if __name__ == "__main__":
    main()
