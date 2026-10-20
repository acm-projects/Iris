"""Train the sign classifier on a WLASL train split.

The checkpoint with the best validation accuracy is kept (ties go to the
lower validation loss). The test split is not touched here; run evaluate.py
for that once you're done tuning.

  # one model
  python ml/two_word/train.py --glosses book drink

  # several seeds (saved as runs/<name>/seed0, seed1, ...; evaluate.py averages them)
  python ml/two_word/train.py --glosses book drink --seeds 0 1 2

  # keep training as long as validation improves (stop after 15 epochs without a new best)
  python ml/two_word/train.py --glosses book drink --epochs 1000 --patience 15

  # pretrain on every asl2000 word, then fine-tune on a few words
  python ml/two_word/train.py --split-file data/splits/asl2000.json --glosses all \\
      --hidden-size 128 --epochs 40 --batch-size 64 --run-name pretrain_asl2000
  python ml/two_word/train.py --glosses book drink --init-from ml/two_word/runs/pretrain_asl2000/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DEFAULT_DATA_ROOT, DEFAULT_FEATURES, PoseClipDataset, all_glosses, load_instances
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


def describe_split(name: str, rows: list[dict], glosses: list[str]) -> None:
    """Print how many clips each word has, e.g. "train: 55 clips {'book': 30, 'drink': 25}"."""
    counts = [sum(r["label"] == i for r in rows) for i in range(len(glosses))]
    if len(glosses) <= 20:
        print(f"{name}: {len(rows)} clips {dict(zip(glosses, counts))}")
    else:
        # Too many words to list; summarize instead.
        print(f"{name}: {len(rows)} clips, {len(glosses)} words, per word "
              f"min {min(counts)} / median {statistics.median(counts):g} / max {max(counts)}")


def load_pretrained(model: PoseGRU, path: str) -> None:
    """Copy every pretrained weight except the final word classifier into `model`.

    The pretrained model scored e.g. 2000 words, ours scores a few, so the last
    layer has a different shape and starts fresh. Everything before it (how
    to read hand shapes and motion) is reused.
    """
    state = torch.load(path, map_location="cpu")["model_state"]
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(keep, strict=False)
    skipped = sorted(set(own) - set(keep))
    print(f"loaded {len(keep)} pretrained tensors from {path}; new: {', '.join(skipped)}")


def train_one(args, seed: int, train_ds: PoseClipDataset, val_ds: PoseClipDataset, out_dir: str) -> dict:
    """Train one model with one seed; save its best checkpoint to out_dir/best.pt."""
    # Seed every source of randomness so the same command gives the same result.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Each seed gets its own stream of random frame sampling / augmentation.
    train_ds.rng = np.random.default_rng(seed)
    # shuffle=True gives a different batch order each epoch, which helps training.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- Model, optimizer, loss ----------------------------------------------
    device = pick_device(args.device)
    # Input size depends on --features (110 numbers per frame for legacy, 371 for all four parts).
    model = PoseGRU(train_ds.feature_dim, len(args.glosses), args.hidden_size, dropout=args.dropout)
    if args.init_from:
        load_pretrained(model, args.init_from)
    model.to(device)
    # Adam is a standard optimizer that adapts the step size for each weight.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cross-entropy is the standard loss for "pick one of N classes".
    loss_fn = nn.CrossEntropyLoss()
    print(f"seed {seed}: features {args.features} ({train_ds.feature_dim} per frame), "
          f"device {device}, parameters {sum(p.numel() for p in model.parameters()):,}")

    # ---- Training loop -------------------------------------------------------
    best = (-1.0, float("inf"))  # (val acc, val loss) of the saved checkpoint
    best_epoch = -1
    history = []  # per-epoch numbers, saved to history.json for plotting later
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)  # learn
        va_loss, va_acc = run_epoch(model, val_loader, device, loss_fn)               # check
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})

        # Keep the epoch that does best on validation. With few val clips,
        # accuracy often ties, so lower val loss (more confident correct
        # answers) breaks the tie.
        improved = va_acc > best[0] or (va_acc == best[0] and va_loss < best[1])
        if improved:
            best, best_epoch = (va_acc, va_loss), epoch
            # Save the weights plus everything evaluate.py needs to rebuild
            # the model and data pipeline exactly.
            torch.save({
                "model_state": model.state_dict(),          # the learned weights
                "glosses": args.glosses,                    # label number -> word
                "config": {**vars(args), "seed": seed},     # all the command-line settings
                "epoch": epoch,
                "val_acc": va_acc,
                "val_loss": va_loss,
            }, os.path.join(out_dir, "best.pt"))

        # Print the first epoch, every 10th, and any new best (marked with *).
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.2%}  "
                  f"val loss {va_loss:.3f} acc {va_acc:.2%}{'  *' if improved else ''}")

        # Save the history every epoch, so stopping a long run early (Ctrl+C)
        # still leaves a record of it next to the best checkpoint.
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        # Early stopping: quit once validation hasn't improved for
        # --patience epochs in a row. --epochs is then just an upper limit.
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"stopping at epoch {epoch}: no improvement for {args.patience} epochs")
            break

    print(f"seed {seed}: best val acc {best[0]:.2%} (loss {best[1]:.3f}) at epoch {best_epoch}")
    print(f"Saved {os.path.join(out_dir, 'best.pt')}\n")
    return {"seed": seed, "val_acc": best[0], "val_loss": best[1], "epoch": best_epoch}


def main():
    # ---- Command-line options ------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glosses", nargs="+", default=["book", "drink"],
                    help="words to learn, or 'all' for every word in the split file")
    ap.add_argument("--glosses-file", default=None,
                    help="text file with one word per line (overrides --glosses), e.g. ml/two_word/vocab/pretrain1000.txt")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="folder containing splits/ and pose_per_individual_videos/")
    ap.add_argument("--split-file", default=None, help="defaults to <data-root>/splits/asl100.json")
    ap.add_argument("--num-frames", type=int, default=32)       # frames sampled per clip
    ap.add_argument("--features", default=DEFAULT_FEATURES,
                    help="'legacy' (original neck-centered xy) or any of pos,vel,bone,mask (see data.py)")
    ap.add_argument("--hidden-size", type=int, default=64)      # model width (see model.py)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=150)          # full passes over the training set (upper limit)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs without a new best on validation (0 = never stop early)")
    ap.add_argument("--batch-size", type=int, default=16)       # clips per weight update
    ap.add_argument("--lr", type=float, default=1e-3)           # learning rate: how big each update step is
    ap.add_argument("--weight-decay", type=float, default=1e-4)  # small penalty on large weights, reduces overfitting
    ap.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0],
                    help="one or more random seeds; each trains a separate model")
    ap.add_argument("--init-from", default=None,
                    help="pretrained checkpoint to start from (its features / hidden size / frames are reused)")
    ap.add_argument("--device", default="cpu", help="cpu, mps, cuda or auto (the model is tiny; cpu is fine)")
    ap.add_argument("--run-name", default=None, help="folder under runs/ for the checkpoint(s)")
    args = ap.parse_args()

    split_file = args.split_file or os.path.join(args.data_root, "splits", "asl100.json")
    args.split_file = split_file
    if args.glosses_file:
        with open(args.glosses_file) as f:
            args.glosses = [line.strip() for line in f if line.strip()]
    if args.glosses == ["all"]:
        args.glosses = all_glosses(split_file)

    # A fine-tuned model must be built exactly like the pretrained one, so
    # take those settings from its checkpoint.
    if args.init_from:
        pre = torch.load(args.init_from, map_location="cpu")["config"]
        for key in ("features", "hidden_size", "num_frames"):
            if getattr(args, key) != pre[key]:
                print(f"using pretrained {key}={pre[key]} (instead of {getattr(args, key)})")
                setattr(args, key, pre[key])

    # ---- Data ----------------------------------------------------------------
    pose_root = os.path.join(args.data_root, "pose_per_individual_videos")
    instances = load_instances(split_file, args.glosses)

    # Use WLASL's own split labels. Test videos are deliberately left out of
    # this script entirely so they can't influence any training decisions.
    train_rows = [r for r in instances if r["split"] == "train"]
    val_rows = [r for r in instances if r["split"] == "val"]
    describe_split("train", train_rows, args.glosses)
    describe_split("val", val_rows, args.glosses)

    # Checkpoints and history go in runs/<run-name>/, e.g. runs/book_drink/.
    # With several seeds, each gets its own subfolder: runs/<run-name>/seed0/ ...
    default_name = "_".join(args.glosses) if len(args.glosses) <= 5 else f"{len(args.glosses)}_words"
    run_dir = os.path.join(HERE, "runs", args.run_name or default_name)

    # Read the videos once and share them across seeds. Training data gets
    # random sampling + augmentation; validation doesn't, so validation scores
    # are comparable from one epoch to the next.
    train_ds = PoseClipDataset(train_rows, pose_root, args.num_frames, train=True, features=args.features)
    val_ds = PoseClipDataset(val_rows, pose_root, args.num_frames, train=False, features=args.features)

    results = []
    for seed in args.seeds:
        out_dir = run_dir if len(args.seeds) == 1 else os.path.join(run_dir, f"seed{seed}")
        results.append(train_one(args, seed, train_ds, val_ds, out_dir))

    if len(results) > 1:
        accs = [r["val_acc"] for r in results]
        print("Best val acc per seed: " + ", ".join(f"seed {r['seed']} {r['val_acc']:.2%}" for r in results))
        print(f"Mean val acc {statistics.mean(accs):.2%} ± {statistics.stdev(accs):.2%} over {len(accs)} seeds")
        print(f"Run `python ml/two_word/evaluate.py --checkpoint {os.path.relpath(run_dir)}` "
              "to test all seeds and their combined prediction.")


if __name__ == "__main__":
    main()
