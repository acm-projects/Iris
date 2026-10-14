"""Evaluate trained model(s) and rank words by how well they're recognized.

  # one checkpoint, test split
  python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/book_drink/best.pt

  # a multi-seed run folder: scores each seed and their combined (averaged) prediction
  python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl100_15_rich

  # rank words on val + test together (more clips per word = a steadier ranking)
  python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl100_15_rich --split val test

Prints overall top-1 / top-5 accuracy and a per-word table, best first, and
saves that table as per_word_<split>.csv next to the checkpoint.
Add --confusion for the full confusion matrix and --mistakes to list every
misclassified video.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import Counter

import torch
from torch.utils.data import DataLoader

from data import PoseClipDataset, load_instances
from model import PoseGRU

# Folder this script lives in, used to find runs/ by default.
HERE = os.path.dirname(os.path.abspath(__file__))


def find_checkpoints(path: str) -> list[str]:
    """A .pt file -> [it]; a run folder -> its best.pt, or every seed*/best.pt inside it."""
    if os.path.isfile(path):
        return [path]
    if os.path.isfile(os.path.join(path, "best.pt")):
        return [os.path.join(path, "best.pt")]
    found = sorted(glob.glob(os.path.join(path, "seed*", "best.pt")))
    if not found:
        raise FileNotFoundError(f"No best.pt or seed*/best.pt under {path}")
    return found


def predict(ckpt: dict, feature_dim: int, loader: DataLoader) -> torch.Tensor:
    """Run one checkpoint over the loader -> (num_clips, num_words) probabilities."""
    cfg = ckpt["config"]
    # Rebuild the model with the same shape as during training, then load the
    # trained weights into it. eval() turns dropout off.
    model = PoseGRU(feature_dim, len(ckpt["glosses"]), cfg["hidden_size"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():  # no gradients needed, we're not training
        # softmax turns raw scores into probabilities that sum to 1 per clip.
        return torch.cat([model(x).softmax(1) for x, _, _ in loader])


def topk_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Fraction of clips whose true word is among the k highest-probability guesses."""
    k = min(k, probs.shape[1])
    hits = (probs.topk(k, dim=1).indices == labels[:, None]).any(dim=1)
    return hits.float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "runs", "book_drink", "best.pt"),
                    help="a best.pt file, or a run folder (multi-seed runs are combined)")
    ap.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test"],
                    help="one or more splits to score together, e.g. --split val test")
    ap.add_argument("--confusion", action="store_true", help="also print the confusion matrix")
    ap.add_argument("--mistakes", action="store_true", help="also list every misclassified video")
    ap.add_argument("--show", type=int, default=40, help="max words to print in the table (the CSV has all)")
    args = ap.parse_args()

    # Load every checkpoint saved by train.py. Each holds the weights plus the
    # settings used for training, so we can rebuild everything the same way.
    paths = find_checkpoints(args.checkpoint)
    ckpts = [torch.load(p, map_location="cpu") for p in paths]
    cfg, glosses = ckpts[0]["config"], ckpts[0]["glosses"]
    features = cfg.get("features", "legacy")  # checkpoints from before --features used legacy
    for p, c in zip(paths, ckpts):
        same = (c["glosses"] == glosses and c["config"].get("features", "legacy") == features
                and c["config"]["num_frames"] == cfg["num_frames"])
        if not same:
            raise ValueError(f"{p} was trained on different words or features than {paths[0]}")

    # Load the clips for the requested split(s). No augmentation here
    # (PoseClipDataset defaults to train=False).
    split_file = cfg["split_file"] or os.path.join(cfg["data_root"], "splits", "asl100.json")
    pose_root = os.path.join(cfg["data_root"], "pose_per_individual_videos")
    rows = [r for r in load_instances(split_file, glosses) if r["split"] in args.split]
    dataset = PoseClipDataset(rows, pose_root, cfg["num_frames"], features=features)
    loader = DataLoader(dataset, batch_size=64)
    labels = torch.tensor([r["label"] for r in rows])
    split_name = "+".join(args.split)
    show_top5 = len(glosses) > 5  # top-5 is meaningless with 5 words or fewer

    # ---- Overall accuracy ----------------------------------------------------
    all_probs = [predict(c, dataset.feature_dim, loader) for c in ckpts]
    print(f"{split_name}: {len(rows)} clips, {len(glosses)} words, {len(ckpts)} model(s)\n")
    for p, c, probs in zip(paths, ckpts, all_probs):
        line = f"  {os.path.relpath(p)} (epoch {c['epoch']}): top-1 {topk_accuracy(probs, labels, 1):.2%}"
        if show_top5:
            line += f"  top-5 {topk_accuracy(probs, labels, 5):.2%}"
        print(line)

    # With several seeds, average their probabilities: a simple ensemble that
    # is usually a little more accurate than any single model.
    probs = torch.stack(all_probs).mean(0)
    if len(ckpts) > 1:
        top1s = [topk_accuracy(p, labels, 1) for p in all_probs]
        print(f"\n  mean of single models: top-1 {statistics.mean(top1s):.2%} ± {statistics.stdev(top1s):.2%}")
        label = "combined (averaged) prediction"
    else:
        label = "accuracy"
    correct = int((probs.argmax(1) == labels).sum())
    line = f"  {label}: top-1 {correct}/{len(rows)} = {correct / len(rows):.2%}"
    if show_top5:
        line += f"  top-5 {topk_accuracy(probs, labels, 5):.2%}"
    print(line)

    # ---- Per-word table --------------------------------------------------------
    preds = probs.argmax(1)
    table = []
    for i, gloss in enumerate(glosses):
        idx = (labels == i).nonzero().flatten()
        if len(idx) == 0:
            continue  # this word has no clips in the chosen split(s)
        hits = int((preds[idx] == i).sum())
        # Average probability the model gave the *right* word: how sure it is,
        # even on clips it got wrong.
        confidence = probs[idx, i].mean().item()
        wrong = Counter(glosses[int(preds[j])] for j in idx if preds[j] != i)
        table.append({
            "word": gloss,
            "correct": hits,
            "clips": len(idx),
            "accuracy": hits / len(idx),
            "avg_confidence": confidence,
            "most_confused_with": ", ".join(f"{w} ({n})" for w, n in wrong.most_common(2)),
        })
    # Best recognized first: highest accuracy, then highest confidence.
    table.sort(key=lambda r: (-r["accuracy"], -r["avg_confidence"]))

    print(f"\nWords ranked by how well they're recognized ({split_name}):")
    width = max(len(r["word"]) for r in table) + 2
    print(f"  {'#':>3}  {'word'.ljust(width)}{'correct':>9}{'accuracy':>10}{'confidence':>12}  confused with")
    for n, r in enumerate(table[:args.show], start=1):
        print(f"  {n:>3}  {r['word'].ljust(width)}{r['correct']:>4}/{r['clips']:<4}{r['accuracy']:>10.0%}"
              f"{r['avg_confidence']:>12.0%}  {r['most_confused_with']}")
    if len(table) > args.show:
        print(f"  ... {len(table) - args.show} more words in the CSV")

    out_csv = os.path.join(os.path.dirname(paths[0]) if len(paths) == 1 else args.checkpoint,
                           f"per_word_{split_name}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSaved {os.path.relpath(out_csv)}")

    # ---- Optional extras ---------------------------------------------------------
    if args.confusion:
        # confusion[true][predicted]; the diagonal is the correct predictions.
        confusion = [[0] * len(glosses) for _ in glosses]
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1
        print("\nConfusion matrix (rows = true, columns = predicted)")
        print(" " * width + "".join(g.rjust(width) for g in glosses))
        for g, row in zip(glosses, confusion):
            print(g.ljust(width) + "".join(str(n).rjust(width) for n in row))

    if args.mistakes:
        print("\nMisclassified:")
        for j, r in enumerate(rows):
            if preds[j] != labels[j]:
                print(f"  video {r['video_id']}: true {glosses[labels[j]]}, "
                      f"predicted {glosses[preds[j]]} ({probs[j, preds[j]]:.0%})")


if __name__ == "__main__":
    main()
