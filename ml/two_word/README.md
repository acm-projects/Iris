# Two-word sign classifier (book vs. drink)

A small model trained from scratch on WLASL pose keypoints, using the
train / val / test labels from `asl100.json`.

| split | book | drink | total |
|-------|------|-------|-------|
| train | 30   | 25    | 55    |
| val   | 6    | 6     | 12    |
| test  | 4    | 4     | 8     |

## Data

Expects the WLASL data in `Cue/data/` (gitignored, so each teammate copies it
in locally). Override with `--data-root` or the `WLASL_DATA` environment variable.

```
Cue/data/
  splits/asl100.json
  pose_per_individual_videos/<video_id>/image_00001_keypoints.json ...
```

## Run

```bash
pip install -r ml/two_word/requirements.txt
python ml/two_word/train.py          # trains, keeps best-on-val checkpoint in runs/book_drink/
python ml/two_word/evaluate.py       # scores that checkpoint on the test split
```

Useful flags: `train.py --glosses book drink computer` (any asl100 words),
`--epochs`, `--hidden-size`, `--num-frames`, `--seed`, `--run-name`;
`evaluate.py --split val --checkpoint runs/<name>/best.pt`.

## How it works

- **`data.py`**: reads each OpenPose frame and keeps 13 upper-body + 21 + 21 hand
  joints (55 × (x, y)). Each clip is centered on the neck and scaled by
  shoulder width. Training samples 32 frames with random jitter and applies
  mirror / rotate / scale / shift / noise augmentation; evaluation samples
  32 evenly spaced frames with no augmentation.
- **`model.py`**: linear embedding → bidirectional GRU (hidden 64) → mean over
  time → linear classifier. ~57k parameters.
- **`train.py`**: Adam, cross-entropy, 150 epochs; saves the epoch with the
  best val accuracy (ties → lower val loss). Test is never used for selection.

## Results

| experiment | split file | train / val / test | val acc | test acc | chance |
|---|---|---|---|---|---|
| book vs drink (seeds 0–4) | asl100 | 55 / 12 / 8 | 91.7% every seed | 87.5–100% | 50% |
| 15 words (seed 0) | asl100 | 276 / 65 / 50 | 75.4% | 66.0% | 6.7% |
| sun vs support (seeds 0–4) | asl2000 | 14 / 4 / 2 | 75–100% | 100% every seed | 50% |

15 words = the 15 asl100 glosses with the most videos: book drink computer
before chair go clothes who candy cousin deaf fine help no thin.

```bash
python ml/two_word/train.py --run-name asl100_15 --glosses book drink computer before chair go clothes who candy cousin deaf fine help no thin
python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl100_15/best.pt

python ml/two_word/train.py --split-file data/splits/asl2000.json --glosses sun support --run-name asl2000_sun_support
python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl2000_sun_support/best.pt
```

Test sets are tiny (8, 50 and 2 clips), so treat these as rough. Note that
asl2000 contains asl100's videos with the same split labels, so book vs drink
from asl2000 is identical to the asl100 run; sun / support only exist in asl2000.
