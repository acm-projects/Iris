# Pose-based sign classifier

Recognizes isolated ASL signs from body and hand keypoints. The model is our
own (a small GRU), trained on the [WLASL](https://github.com/dxli94/WLASL)
dataset's OpenPose keypoints: first **pretrained on all 2,000 WLASL words**,
then **fine-tuned** on the word list we care about.

The folder is still called `two_word` because it started as a book-vs-drink
experiment; it now handles any number of words.

## Results

All numbers are on WLASL's **test** split: videos never used for training or
for picking the best epoch.

| Words | Split | Train videos / word | Test top-1 | Test top-5 |
|---|---|---|---|---|
| 15 (asl100's 15 most frequent) | asl100 | ~18 | **80.0%** (40/50) | 96.0% |
| 50 (asl2000's 50 most frequent) | asl2000 | ~16 | **75.5%** (avg of 3 seeds) | ~92% |
| **100 (all of asl100)** | asl100 | ~14 | **72.1%** (186/258) | **91.5%** |
| 50 (words only in asl2000) | asl2000 | ~7 | **67.3%** (avg of 3 seeds, noisy) | ~94% |
| 300 (all of asl300) | asl300 | ~11 | **53.3%** (356/668) | 81.1% |

### Compared with the WLASL paper (asl100, same 258 test videos)

| Model | Input | Top-1 | Top-5 |
|---|---|---|---|
| Pose-TGCN (paper; reproduced in `ml/TGCN`) | pose keypoints | 55.4% | 78.7% |
| I3D (paper) | raw video | ~65.9%* | – |
| **This model** | pose keypoints | **72.1%** | **91.5%** |

\*From memory of the paper; check before quoting. Our model is pretrained on
the asl2000 *training* videos, which the paper's asl100 Pose-TGCN didn't use,
so part of the gap is extra training data. No test video is used anywhere.

### What pretraining adds (test top-1, same features)

| Words | From scratch | Pretrained + fine-tuned | Gain |
|---|---|---|---|
| 15 | 74.0% | 80.0% | +6.0 |
| 50 (most frequent) | 66.4% | 75.5% | +9.1 |
| 50 (asl2000-only, ~7 videos / word) | 52.8% | 67.3% | +14.5 |
| 300 | 44.5% | 53.3% | +8.8 |

The less data a word has, the more pretraining helps. (The from-scratch 15-
and 50-word runs used hidden size 64; the 300-word one used 128, the same size
as the pretrained model.)

Notes:
- Small test sets (50–143 videos) move about ±5–7 points between seeds;
  asl100 (258) and asl300 (668) are the most reliable numbers.
- The 15- and 100-word results use the 100-epoch pretraining
  (`pretrain_asl2000_e100`); the 50- and 300-word results use the earlier
  40-epoch one and may improve slightly when rerun.
- These are OpenPose keypoints from WLASL videos. The live app uses MediaPipe
  on a webcam, so **live accuracy still has to be measured separately**.

## Setup

**Python environment** (separate from `backend/.venv`, which has no PyTorch):

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/two_word/requirements.txt
source ml/.venv/bin/activate
```

**Data.** Unzip the WLASL `splits.zip` and `pose_per_individual_videos.zip`
into `Cue/data/` (gitignored, about 2 GB / 1.3M files):

```
Cue/data/
  splits/asl100.json  asl300.json  asl1000.json  asl2000.json
  pose_per_individual_videos/<video_id>/image_00001_keypoints.json ...
```

Override the location with `--data-root` or the `WLASL_DATA` environment variable.

## Reproducing the results

Run from the `Cue` folder with `ml/.venv` active.

**1. Pretrain on all 2,000 words** — or skip this step: the result is
committed as `ml/two_word/pretrained/pretrain_asl2000_e100.pt` (3 MB, epoch 95,
26.4% val) with its per-epoch history next to it. To redo it (about 2 hours):

```bash
caffeinate -i python ml/two_word/train.py --split-file data/splits/asl2000.json --glosses all \
    --hidden-size 128 --batch-size 64 --epochs 100 --run-name pretrain_asl2000_e100
```

Best validation accuracy: 26.4% at epoch 95 (on 2,000 words; chance is 0.05%).
The point isn't that number: it's that the early layers learn general
handshape and motion patterns from 14,296 videos.

**2. Fine-tune** on a word list. `--init-from` copies the pretrained weights;
`--patience` stops once validation stops improving.

```bash
# all 100 asl100 words -> 72.1% test
python ml/two_word/train.py --glosses all --init-from ml/two_word/pretrained/pretrain_asl2000_e100.pt \
    --lr 3e-4 --epochs 200 --patience 20 --run-name asl100_all_ft_e100

# the 15 most frequent asl100 words -> 80.0% test
python ml/two_word/train.py --init-from ml/two_word/pretrained/pretrain_asl2000_e100.pt \
    --lr 3e-4 --epochs 200 --patience 20 --run-name asl100_15_ft_e100 \
    --glosses book drink computer before chair go clothes who candy cousin deaf fine help no thin

# all 300 asl300 words
python ml/two_word/train.py --split-file data/splits/asl300.json --glosses all \
    --init-from ml/two_word/pretrained/pretrain_asl2000_e100.pt \
    --lr 3e-4 --epochs 200 --patience 20 --run-name asl300_ft_e100
```

**3. Evaluate:**

```bash
python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl100_all_ft_e100
```

Prints top-1 / top-5 accuracy and a per-word table (best recognized first,
with accuracy, average confidence in the right word, and what it's confused
with), and saves the table as `per_word_<split>.csv` in the run folder.

A from-scratch baseline is the same command without `--init-from` (add
`--hidden-size 128` to match the pretrained model's size).

## How it works

```
OpenPose JSON per frame ─► 55 joints ─► 371 features per frame ─► 32 frames ─► GRU ─► score per word
```

**Input** (`data.py`). Each frame gives 55 (x, y) keypoints: 13 upper-body
joints, 21 left-hand and 21 right-hand points. This is the same layout the live
MediaPipe pipeline in `backend/sign_inference` produces. 32 frames are sampled
across the sign (randomly jittered in training, evenly spaced in evaluation).

**Features** (`--features`, default `pos,vel,bone,mask`, 371 per frame):

| Part | Numbers | What it is |
|---|---|---|
| `pos` | 110 | body joints relative to the neck / shoulder width; each hand relative to its own wrist / size (hand shape, independent of position). Uses `backend/sign_inference/landmark_normalization.py`, so training matches live inference. |
| `vel` | 110 | change in position since the previous frame (movement) |
| `bone` | 96 | vector along each of 48 arm and finger bones (joint angles) |
| `mask` | 55 | 1 if the joint was detected in that frame, else 0 |

Positions are capped at ±4 (`POS_CLIP` in `data.py`): bad OpenPose hand
detections can otherwise produce values up to ±30. **Live inference must apply
the same cap.** `--features legacy` gives the original 110-number
neck-centered coordinates.

**Augmentation** (training only, on raw keypoints before normalization):
random left/right mirror, ±10° rotation, small per-joint noise, and jittered
frame sampling.

**Model** (`model.py`). A linear layer turns each frame's 371 numbers into
128 features, a bidirectional GRU reads the 32 frames in order, the outputs
are averaged over time, and a final linear layer gives one score per word.
About 250k parameters for 15 words, 760k for 2,000.

**Training** (`train.py`). Adam optimizer, cross-entropy loss, batches of 16
(64 for pretraining). After each epoch it scores the validation split and
keeps the best epoch as `best.pt` (ties go to the lower validation loss). The
test split is never loaded during training.

**Fine-tuning** (`--init-from`). The new model copies the pretrained
embedding and GRU weights and gets a fresh final layer sized for the new word
list (the pretrained one scored 2,000 words). Training then runs as usual,
with a smaller learning rate (`--lr 3e-4`) so the pretrained knowledge is
adjusted rather than overwritten. Settings that must match the pretrained
model (`--features`, `--hidden-size`, `--num-frames`) are taken from its
checkpoint automatically.

## Options

**`train.py`**

| Flag | Default | |
|---|---|---|
| `--glosses` | `book drink` | words to learn, or `all` for every word in the split file |
| `--split-file` | `data/splits/asl100.json` | which WLASL split to use |
| `--init-from` | – | pretrained checkpoint to fine-tune from |
| `--epochs` | 150 | maximum number of passes over the training data |
| `--patience` | 0 (off) | stop after this many epochs without a new best |
| `--lr` | 1e-3 | learning rate (use `3e-4` for fine-tuning) |
| `--seeds` | 0 | one or more seeds; several seeds save to `runs/<name>/seed0`, `seed1`, … |
| `--hidden-size` | 64 | model width (128 for pretraining) |
| `--batch-size` | 16 | clips per update |
| `--features` | `pos,vel,bone,mask` | see above |
| `--num-frames` | 32 | frames sampled per clip |
| `--run-name` | from the words | output folder under `runs/` |

**`evaluate.py`**

| Flag | |
|---|---|
| `--checkpoint` | a `best.pt`, or a run folder (multi-seed runs are scored individually and averaged) |
| `--split` | `test` (default), `val`, `train`, or several, e.g. `--split val test` |
| `--show` | how many words to print in the table (the CSV has all) |
| `--confusion` | also print the confusion matrix |
| `--mistakes` | also list every misclassified video |

## Picking demo words

Rank words on validation, then check the chosen ones on test or, better,
live on camera. Choosing words by their test score and then quoting that test
score is optimistic.

```bash
python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/asl100_all_ft_e100 --split val --show 100
```

Words recognized perfectly on val + test by the 50-word model: family, chair,
can, bowling, year, many, help, orange, black, tall, book, dog, study, fish,
white, blue, man.

## Files

| File | |
|---|---|
| `data.py` | reads WLASL splits and OpenPose JSON, builds features, augmentation, `PoseClipDataset` |
| `model.py` | `PoseGRU` |
| `train.py` | training, multi-seed runs, pretraining / fine-tuning, early stopping |
| `evaluate.py` | top-1 / top-5, per-word ranking, CSV export |
| `pretrained/` | the committed asl2000 pretrained model, ready for `--init-from` |
| `runs/` | checkpoints and training history (gitignored) |

Related: `ml/split_stats.py` writes a spreadsheet of train / val / test video
counts per word for every WLASL subset, and `ml/TGCN/` holds the paper's
Pose-TGCN code adapted to run on macOS.

## Next steps

- Connect this model to the camera app in `backend/sign_inference` (371
  features, 32 frames, ±4 cap) and measure live accuracy on recordings of the
  team.
- Check MediaPipe's left / right hand labels on unmirrored frames in
  `backend/sign_inference/inference_utils.py`.
- Fine-tune on our own MediaPipe recordings to close the gap between WLASL and
  live webcam input.
- Possible accuracy gains: averaging several views at test time, label
  smoothing and a learning-rate schedule, a bigger GRU, or a transformer model.
