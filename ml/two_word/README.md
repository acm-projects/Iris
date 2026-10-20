# Pose-based sign classifier: meeting100

Recognizes isolated ASL signs from body and hand keypoints, for words that
come up in online meetings. The model is our own (a small GRU), trained on the
[WLASL](https://github.com/dxli94/WLASL) dataset's OpenPose keypoints in two
steps:

1. **Pretrained** on all 2,000 WLASL words (`pretrain_asl2000_e100.pt`).
2. **Fine-tuned** on 100 meeting words (`meeting100`).

Both model files are shared on the team Drive, not in GitHub (see Quick start).

The same model then runs live on a webcam through MediaPipe
(`backend/sign_inference`).

> **Just want to try it on your camera?** Jump to [Quick start](#quick-start-run-meeting100-on-your-camera): about 10 minutes, no training needed.

The folder is still called `two_word` for historical reasons; it handles any
number of words.

## Results

All numbers are on WLASL's **test** split: videos never used for training or
for picking the best epoch.

| Model | Words | Val | Test top-1 | Test top-5 | Random guess |
|---|---|---|---|---|---|
| `pretrain_asl2000_e100` (pretraining) | 2,000 | 26.4% | – | – | 0.05% |
| **`meeting100`** (fine-tuned) | 100 | **69.7%** | **69.0%** (140/203) | **93.6%** | 1% |

Top-5 means the right word is among the model's five best guesses.

**By category** (meeting100, test):

| Category | Words | Correct | Accuracy |
|---|---|---|---|
| Meeting work | 15 | 25/28 | 89% |
| Greetings & manners | 9 | 12/15 | 80% |
| Understanding & talking | 13 | 19/26 | 73% |
| Wants, needs & actions | 12 | 17/25 | 68% |
| Time | 14 | 21/31 | 68% |
| Tech | 8 | 12/18 | 67% |
| Describing | 8 | 12/18 | 67% |
| Responses | 9 | 10/18 | 56% |
| Questions | 7 | 9/16 | 56% |
| People | 5 | 3/8 | 38% |

**Most reliable words** (correct on test, and confident): camera, team,
problem, can, explain, have, time, tell, late, how, meet, meeting, share,
wrong, plan, email.

**Weakest words** (wrong on test): thank you, good, maybe, you, what, hard,
need, before, stop, me, ready, call, listen, right, why, disagree.

Most words have only 1–3 test clips, so per-word results are rough. The full
table is in `runs/meeting100/per_word_test.csv`.

### Known look-alikes
- **thank you / good** are nearly the same sign in ASL; the model confuses them.
- **now / today, hear / listen, like / same, email / send** share a few
  identical WLASL videos.
- **before** is weak in every model (WLASL's before / past / ago clips overlap).

### Reference point: the WLASL paper
On asl100 (100 words, the same 258 test videos the paper uses), the same
pretraining + fine-tuning gets **72.1%** top-1 / 91.5% top-5, against **55.4%**
for the paper's Pose-TGCN (reproduced in `ml/TGCN`). Our model is pretrained
on the asl2000 *training* videos, which the paper's asl100 model didn't use,
so part of the gap is extra training data. No test video is used anywhere.

### Live accuracy
These are OpenPose keypoints from WLASL videos. Live, the app uses MediaPipe on
a webcam, with different signers and lighting, so **live accuracy has to be
measured separately**: sign each word a few times on camera and count.

## The 100 words

Listed in `vocab/meeting100.txt`, all from WLASL asl2000 (9–30 videos each,
median 15):

- **Responses:** yes, no, ok, maybe, not, right, wrong, agree, disagree
- **Questions:** who, what, where, when, why, how, question
- **People:** you, your, we, me, my
- **Greetings & manners:** hello, goodbye, please, thank you, sorry, excuse, welcome, nice, meet
- **Understanding & talking:** understand, know, think, remember, forget, explain, slow, hear, listen, see, talk, tell, ready
- **Wants, needs & actions:** want, need, like, help, have, can, will, try, wait, stop, start, finish
- **Meeting work:** meeting, work, plan, idea, problem, discuss, share, show, ask, answer, decide, vote, project, team, important
- **Time:** now, later, soon, today, tomorrow, yesterday, time, minute, hour, week, next, again, late, before
- **Tech:** computer, internet, phone, email, camera, chat, call, send
- **Describing:** good, bad, more, all, same, different, easy, hard

## Quick start: run meeting100 on your camera

You don't need to train anything. You need the code (this repo), one Python
environment, and two model files from the team's shared Drive.

### 1. Get the model files from the Drive

The models are **not in GitHub**. Download them from the team Drive and put
them at exactly these paths inside your `Cue` folder (create the folders if
they don't exist):

| File on the Drive | Put it at |
|---|---|
| `meeting100/best.pt` | `Cue/ml/two_word/runs/meeting100/best.pt` |
| `pretrain_asl2000_e100.pt` (only needed to train) | `Cue/ml/two_word/pretrained/pretrain_asl2000_e100.pt` |

```bash
mkdir -p ~/Cue/ml/two_word/runs/meeting100 ~/Cue/ml/two_word/pretrained
```

Then move the downloaded files into those folders (Finder works fine).

### 2. Create the camera environment (once)

The camera needs MediaPipe, which runs on Python 3.11 or 3.12 (not 3.13+).
From the `Cue` folder:

```bash
cd ~/Cue
python3.11 -m venv .venv-wlasl
.venv-wlasl/bin/pip install -r backend/sign_inference/requirements-inference.txt
.venv-wlasl/bin/python backend/sign_inference/setup_assets.py
```

The last command downloads MediaPipe's hand and pose models plus the old TGCN checkpoint (about 17 MB).
`.venv-wlasl` is gitignored. If `python3.11` isn't found, install it with
`brew install python@3.11` (or use `python3.12`).

### 3. Check the model is found

```bash
.venv-wlasl/bin/python backend/sign_inference/camera_inference.py --list-models
```

`meeting100` should be in the list. If it isn't, the file from step 1 is in
the wrong place.

### 4. Run it

```bash
.venv-wlasl/bin/python backend/sign_inference/camera_inference.py --model meeting100 --landmarks
```

The terminal should print
`Model meeting100: 100 words, 32 frames, trained val acc 69.7%`, and a camera
window opens with your skeleton drawn in green.

### 5. Sign

1. **Click the camera window** so it receives your key presses.
2. Press **R** to start recording.
3. Do **one** sign.
4. Press **S** to stop. The top five guesses appear with their confidence.
5. Repeat for the next sign. Press **Q** to quit.

The status line under the guesses shows:

- **frames**: how many frames were recorded
- **hands L / R**: how often each hand was detected (low numbers mean MediaPipe
  is losing your hands, so the model can't see the sign)
- **margin**: how far the best guess is ahead of the second (bigger = surer)

The full result is also printed as JSON in the terminal.

**Tips for good results**

- Press R right before the sign and S right after it. Standing still before or
  after dilutes the recording.
- Keep your shoulders and both hands in view, with decent light.
- Copy the WLASL version of each sign (look it up on signingsavvy.com or
  handspeak.com). Signs vary between signers and regions.
- Start with the most reliable words: camera, team, problem, can, explain,
  have, time, tell, late, how, meeting, share, wrong, plan, email.

## Camera options

Add any of these to the command in step 4:

| Option | What it does | Example |
|---|---|---|
| `--model NAME` | which model to use: a name from `--list-models`, or a path to a `.pt` file | `--model meeting100` |
| `--list-models` | list the models it can find, then exit | |
| `--landmarks` | draw your skeleton on the video | |
| `--landmark-style lines` / `dots` | draw the skeleton as lines (default) or dots | `--landmark-style dots` |
| `--top-k N` | how many guesses to show (default 5) | `--top-k 3` |
| `--mirror-input` | flip each frame left-right before MediaPipe and the model see it (the preview is always mirrored like a selfie either way) | `--mirror-input` |
| `--hand-assignment wrist` / `label` | how left and right hands are told apart: by the nearest wrist on the body (default, recommended) or by MediaPipe's own label | `--hand-assignment label` |
| `--camera N` | use a different camera (0 is the built-in one) | `--camera 1` |

Examples:

```bash
# show only the top 3 guesses
.venv-wlasl/bin/python backend/sign_inference/camera_inference.py --model meeting100 --landmarks --top-k 3

# compare with mirrored input
.venv-wlasl/bin/python backend/sign_inference/camera_inference.py --model meeting100 --landmarks --mirror-input

# external webcam, dots instead of lines
.venv-wlasl/bin/python backend/sign_inference/camera_inference.py --model meeting100 --landmarks --camera 1 --landmark-style dots
```

Without `--model`, the script runs the original WLASL-100 TGCN demo instead
(and asks for `--experimental-legacy`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` or `'mediapipe'` | You're using the wrong Python. Run the commands exactly as written, starting with `.venv-wlasl/bin/python`. (`backend/.venv` and `ml/.venv` don't have MediaPipe.) |
| `can't open file ... camera_inference.py` | You're not in the `Cue` folder. Run `cd ~/Cue` first. |
| `No model named 'meeting100'` | The model file isn't at `ml/two_word/runs/meeting100/best.pt` (step 1). |
| `Missing MediaPipe models` | Run `.venv-wlasl/bin/python backend/sign_inference/setup_assets.py`. |
| `Could not open camera index 0` | Allow camera access: System Settings → Privacy & Security → Camera, turn it on for Terminal / VS Code, then reopen that app. Or try `--camera 1`. |
| `No body detected` | Step back so your shoulders are in the frame. |
| `Recording too short` | Hold the sign a little longer between R and S. |
| Keys do nothing | Click the camera window first; it needs focus. |

## Training it yourself (optional)

Only needed if you want to retrain or change the word list.

**Training environment** (separate from the camera one):

```bash
cd ~/Cue
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/two_word/requirements.txt
```

**Data.** Unzip the WLASL `splits.zip` and `pose_per_individual_videos.zip`
into `Cue/data/` (gitignored, about 2 GB / 1.3M files):

```
Cue/data/
  splits/asl100.json  asl300.json  asl1000.json  asl2000.json
  pose_per_individual_videos/<video_id>/image_00001_keypoints.json ...
```

**1. Pretraining** (skip it: use `pretrain_asl2000_e100.pt` from the Drive).
To redo it (about 2 hours; best epoch 95, 26.4% val):

```bash
caffeinate -i ml/.venv/bin/python ml/two_word/train.py --split-file data/splits/asl2000.json --glosses all \
    --hidden-size 128 --batch-size 64 --epochs 100 --run-name pretrain_asl2000_e100
```

That saves to `runs/pretrain_asl2000_e100/best.pt`; copy it to
`pretrained/pretrain_asl2000_e100.pt` to use the commands below unchanged.

**2. Fine-tune on the 100 meeting words** (a few minutes; stops by itself once
validation stops improving). This creates `runs/meeting100/best.pt`:

```bash
ml/.venv/bin/python ml/two_word/train.py --split-file data/splits/asl2000.json \
    --glosses-file ml/two_word/vocab/meeting100.txt \
    --init-from ml/two_word/pretrained/pretrain_asl2000_e100.pt \
    --lr 3e-4 --epochs 200 --patience 20 --run-name meeting100
```

Expected: best val 69.7% at epoch 22.

**3. Evaluate:**

```bash
ml/.venv/bin/python ml/two_word/evaluate.py --checkpoint ml/two_word/runs/meeting100
```

Expected: top-1 140/203 = 69.0%, top-5 93.6%. It prints a per-word table
(best recognized first, with accuracy, average confidence in the right word,
and what it's confused with) and saves it as `runs/meeting100/per_word_test.csv`.
Add `--split val test` for a steadier per-word ranking.

To train on a different word list, put one word per line in a text file and
pass it with `--glosses-file`; the camera then runs it with `--model <run-name>`.

## How it works

```
keypoints per frame ─► 55 joints ─► 371 features per frame ─► 32 frames ─► GRU ─► score per word
```

**Input** (`data.py`). Each frame gives 55 (x, y) keypoints: 13 upper-body
joints, 21 left-hand and 21 right-hand points. Training reads them from
WLASL's OpenPose files; the camera gets the same layout from MediaPipe. 32
frames are sampled across the sign (randomly jittered in training, evenly
spaced in evaluation and on camera).

**Features** (default `pos,vel,bone,mask`, 371 per frame):

| Part | Numbers | What it is |
|---|---|---|
| `pos` | 110 | body joints relative to the neck / shoulder width; each hand relative to its own wrist / size (hand shape, independent of position). Uses `backend/sign_inference/landmark_normalization.py`, so training matches live inference. |
| `vel` | 110 | change in position since the previous frame (movement) |
| `bone` | 96 | vector along each of 48 arm and finger bones (joint angles) |
| `mask` | 55 | 1 if the joint was detected in that frame, else 0 |

Positions are capped at ±4 (`POS_CLIP` in `data.py`): bad hand detections can
otherwise produce values up to ±30.

**Augmentation** (training only): random left / right mirror, ±10° rotation,
small per-joint noise, and jittered frame sampling.

**Model** (`model.py`). A linear layer turns each frame's 371 numbers into 128
features, a bidirectional GRU reads the 32 frames in order, the outputs are
averaged over time, and a final linear layer gives one score per word. About
270k parameters for 100 words.

**Training** (`train.py`). Adam optimizer, cross-entropy loss. After each epoch
it scores the validation split and keeps the best epoch as `best.pt`. The test
split is never loaded during training.

**Pretraining then fine-tuning.** Each word has only about 7–18 training
videos, too few to learn signing from scratch without memorizing. Pretraining
on all 2,000 words (14,296 videos) teaches the first layers general handshape
and motion patterns. Fine-tuning (`--init-from`) copies those layers into a new
model with a fresh output layer for the 100 meeting words, then trains on just
those words with a smaller learning rate (`--lr 3e-4`) so the pretrained
knowledge is adjusted rather than overwritten. Settings that must match the
pretrained model (`--features`, `--hidden-size`, `--num-frames`) are copied
from its checkpoint automatically.

## Training options

**`train.py`**

| Flag | Default | |
|---|---|---|
| `--glosses` | `book drink` | words to learn, or `all` for every word in the split file |
| `--glosses-file` | – | text file with one word per line (overrides `--glosses`) |
| `--split-file` | `data/splits/asl100.json` | which WLASL split to use (`asl2000.json` for meeting100) |
| `--init-from` | – | pretrained checkpoint to fine-tune from |
| `--epochs` | 150 | maximum number of passes over the training data |
| `--patience` | 0 (off) | stop after this many epochs without a new best |
| `--lr` | 1e-3 | learning rate (use `3e-4` for fine-tuning) |
| `--seeds` | 0 | one or more seeds; several seeds save to `runs/<name>/seed0`, `seed1`, … |
| `--hidden-size` | 64 | model width (128 for pretraining) |
| `--batch-size` | 16 | clips per update |
| `--run-name` | from the words | output folder under `runs/` |

**`evaluate.py`**

| Flag | |
|---|---|
| `--checkpoint` | a `best.pt`, or a run folder (multi-seed runs are scored individually and averaged) |
| `--split` | `test` (default), `val`, `train`, or several, e.g. `--split val test` |
| `--show` | how many words to print in the table (the CSV has all) |
| `--confusion` | also print the confusion matrix |
| `--mistakes` | also list every misclassified video |

## Files

| File | |
|---|---|
| `data.py` | reads WLASL splits and OpenPose JSON, builds features, augmentation, `PoseClipDataset` |
| `model.py` | `PoseGRU` |
| `train.py` | training, pretraining / fine-tuning, multi-seed runs, early stopping |
| `evaluate.py` | top-1 / top-5, per-word ranking, CSV export |
| `pretrained/pretrain_asl2000_e100.pt` | the 2,000-word pretrained model (from the Drive) |
| `runs/meeting100/best.pt` | the meeting100 model (from the Drive) |
| `vocab/meeting100.txt` | the 100 meeting words |
| `vocab/hand_visibility_asl2000.csv` | per-word hand visibility, one- vs two-handed, overlap (from `ml/hand_visibility.py`) |
| `runs/` | your own training runs: checkpoints, history, per-word CSVs (gitignored) |

Related: `backend/sign_inference/` (camera), `ml/hand_visibility.py`,
`ml/split_stats.py` (train / val / test counts per word), and `ml/TGCN/` (the
paper's Pose-TGCN, adapted to run on macOS).

## Next steps

- Measure live accuracy on camera: each word a few times per teammate.
- Drop or merge the look-alike words (keep thank you, drop good; keep one of
  now / today, hear / listen, like / same).
- Fine-tune on our own MediaPipe recordings to close the gap between WLASL and
  webcam input.
- Show the top three guesses in the app instead of one: top-5 accuracy is 93.6%.
