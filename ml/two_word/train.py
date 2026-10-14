"""Train the two-word sign classifier on the WLASL asl100 train split.

The checkpoint with the best validation accuracy is kept (ties go to the
lower validation loss). The test split is not touched here; run evaluate.py
for that once you're done tuning.

  python ml/two_word/train.py
  python ml/two_word/train.py --glosses book drink --epochs 200 --run-name gru_h64
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DEFAULT_DATA_ROOT, NUM_JOINTS, PoseClipDataset, load_instances
from model import PoseGRU

# Folder this script lives in, so runs/ always ends up next to it no matter
# which directory you launch the script from.
HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device(name: str) -> torch.device:
    """Turn the --device flag into a torch device. "auto" picks the fastest available."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():         # NVIDIA GPU
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given. Returns (loss, accuracy)."""
    # train(True) turns dropout on; train(False) (evaluation mode) turns it off
    # so predictions are deterministic.
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0

    # Only track gradients when training; skipping them during evaluation is
    # faster and uses less memory.
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, _ in loader:  # x: batch of clips, y: correct labels, _: video ids (unused here)
            x, y = x.to(device), y.to(device)
            logits = model(x)         # model's scores for each word
            loss = loss_fn(logits, y)  # how wrong those scores are

            if optimizer is not None:
                optimizer.zero_grad()  # clear gradients left over from the previous batch
                loss.backward()        # work out how each weight contributed to the loss
                optimizer.step()       # nudge the weights to reduce the loss

            # Accumulate totals. loss.item() is the batch *average*, so multiply
            # by batch size to get a correct overall average at the end.
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()  # argmax = the word with the highest score
            count += len(y)
    return total_loss / count, correct / count


def main():
    # ---- Command-line options ------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glosses", nargs="+", default=["book", "drink"])
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="folder containing splits/ and pose_per_individual_videos/")
    ap.add_argument("--split-file", default=None, help="defaults to <data-root>/splits/asl100.json")
    ap.add_argument("--num-frames", type=int, default=32)       # frames sampled per clip
    ap.add_argument("--hidden-size", type=int, default=64)      # model width (see model.py)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=150)          # full passes over the training set
    ap.add_argument("--batch-size", type=int, default=16)       # clips per weight update
    ap.add_argument("--lr", type=float, default=1e-3)           # learning rate: how big each update step is
    ap.add_argument("--weight-decay", type=float, default=1e-4)  # small penalty on large weights, reduces overfitting
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, mps, cuda or auto (the model is tiny; cpu is fine)")
    ap.add_argument("--run-name", default=None, help="folder under runs/ for the checkpoint")
    args = ap.parse_args()

    # Seed every source of randomness so the same command gives the same result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Data ----------------------------------------------------------------
    split_file = args.split_file or os.path.join(args.data_root, "splits", "asl100.json")
    pose_root = os.path.join(args.data_root, "pose_per_individual_videos")
    instances = load_instances(split_file, args.glosses)

    # Use WLASL's own split labels. Test videos are deliberately left out of
    # this script entirely so they can't influence any training decisions.
    train_rows = [r for r in instances if r["split"] == "train"]
    val_rows = [r for r in instances if r["split"] == "val"]

    # Print how many clips of each word are in each split, e.g.
    # "train: 55 clips {'book': 30, 'drink': 25}".
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        counts = {g: sum(r["label"] == i for r in rows) for i, g in enumerate(args.glosses)}
        print(f"{name}: {len(rows)} clips {counts}")

    # Training data gets random sampling + augmentation; validation doesn't,
    # so validation scores are comparable from one epoch to the next.
    train_ds = PoseClipDataset(train_rows, pose_root, args.num_frames, train=True, seed=args.seed)
    val_ds = PoseClipDataset(val_rows, pose_root, args.num_frames, train=False)
    # shuffle=True gives a different batch order each epoch, which helps training.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- Model, optimizer, loss ----------------------------------------------
    device = pick_device(args.device)
    # Input size is 55 joints x 2 coordinates = 110 numbers per frame.
    model = PoseGRU(NUM_JOINTS * 2, len(args.glosses), args.hidden_size, dropout=args.dropout).to(device)
    # Adam is a standard optimizer that adapts the step size for each weight.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cross-entropy is the standard loss for "pick one of N classes".
    loss_fn = nn.CrossEntropyLoss()
    print(f"device: {device}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Checkpoints and history go in runs/<run-name>/, e.g. runs/book_drink/.
    run_name = args.run_name or "_".join(args.glosses)
    out_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Training loop -------------------------------------------------------
    best = (-1.0, float("inf"))  # (val acc, val loss) of the saved checkpoint
    best_epoch = -1
    history = []  # per-epoch numbers, saved to history.json for plotting later
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)  # learn
        va_loss, va_acc = run_epoch(model, val_loader, device, loss_fn)               # check
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})

        # Keep the epoch that does best on validation. With only 12 val clips,
        # accuracy often ties, so lower val loss (more confident correct
        # answers) breaks the tie.
        improved = va_acc > best[0] or (va_acc == best[0] and va_loss < best[1])
        if improved:
            best, best_epoch = (va_acc, va_loss), epoch
            # Save the weights plus everything evaluate.py needs to rebuild
            # the model and data pipeline exactly.
            torch.save({
                "model_state": model.state_dict(),  # the learned weights
                "glosses": args.glosses,            # label number -> word
                "config": vars(args),               # all the command-line settings
                "epoch": epoch,
                "val_acc": va_acc,
                "val_loss": va_loss,
            }, os.path.join(out_dir, "best.pt"))

        # Print the first epoch, every 10th, and any new best (marked with *).
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.2%}  "
                  f"val loss {va_loss:.3f} acc {va_acc:.2%}{'  *' if improved else ''}")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)

    print(f"\nBest val acc {best[0]:.2%} (loss {best[1]:.3f}) at epoch {best_epoch}")
    print(f"Saved {os.path.join(out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
