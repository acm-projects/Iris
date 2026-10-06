# MediaPipe preprocessing audit

The supplied ASL100 checkpoint is **unknown**, and normal predictions are blocked.
An explicit experimental camera path is now available; see the section below.
An explicit historical affine candidate is now available for regression and
recovery, but is not selected automatically for this checkpoint. Preview, R recording, S stopping/preparation, and Q quitting remain available;
raw recordings remain in memory. No compatible inference path is claimed.

## Experimental reusable inference (2026-10-05)

`sign_inference.SignInference(experimental_legacy=True)` loads the ASL100
configuration, strict checkpoint state dictionary, and candidate sorted split
labels once per instance. Each session owns its MediaPipe VIDEO pose/hand
trackers and recording buffer. Normal construction retains the compatibility
gate and fails before weight loading. The experimental option does not change
`ASL100_AUDIT`, mark the model compatible, or enable the old preview gate.

Methods: `start_sequence()`, `process_frame(rgb_uint8, timestamp_ms)`,
`finish_and_predict()`, `reset()`, `close()`; context-manager use is supported.
Timestamps are integer milliseconds, strictly increasing across the entire
session, including resets. Reset clears recording state but retains trackers
and their timestamp history; a new instance creates fresh tracking state.
Use one instance per session and call its methods serially. `process_frame`
returns remapped pixel coordinates for optional external drawing.

Only `legacy_affine_256_candidate` is used for experimental pretrained
inference. MediaPipe RGB observations are remapped to 55 joints and converted
using actual image width and height. Missing body detections approximate
OpenPose empty-people frames: omit initial missing frames, repeat the preceding
pose thereafter. Missing individual coordinates map to zero pixels before
affine transformation; no pivot normalization or one-pixel threshold applies.
Hand-only detections without a body are treated as missing-person frames.
This detector correspondence remains an experimental approximation.

The audited live helper selects a consecutive central 50-frame window from
usable records and repeats the final frame for short recordings. Initial
missing frames are omitted before this selection; padding always produces 50
frames, rather than reproducing the historical short-tensor bug. This live
single-window policy differs from upstream four-copy evaluation and is
reported explicitly as `central_consecutive_50_repeat_last` in results.
Shared packing produces finite float32 `(1,55,100)` tensors with interleaved
xy per joint. Model execution uses `eval()` and `torch.inference_mode()`;
finite `(1,100)` output is required to match the 100 unique sorted labels.

JSON-compatible results contain top-three `{gloss, score}` predictions,
captured/usable/prepared frame counts, preprocessing mode, experimental status,
preprocessing verification status (`unknown`), label-map verification status
(`unverified`), temporal policy, and score type. Scores are softmax model
scores; checkpoint-linked label ordering remains unverified.

`inference_utils.py` holds the existing remapping/preparation/packing/label
helpers. The old preview reexports these functions for compatibility. UI,
keyboard handling, BGR/RGB conversion, display mirroring, optional landmark
drawing, and camera transport are in the thin `camera_inference.py` runner.

Run in Windows PowerShell:

```powershell
Set-Location 'C:\Users\hi\OneDrive\Documents\ACM\Cue'
python .\code\TGCN\camera_inference.py --experimental-legacy --landmarks
```

Omit `--landmarks` to hide overlays. Use `--camera 1` for another camera.
R starts recording, S finishes and displays/prints top-three predictions,
Q quits. Empty/all-missing recordings and invalid inputs produce clear errors.
Resources are released on exit. No web app or continuous segmentation was added.

Validation: `python -m unittest test_sign_inference test_legacy_preprocessing
test_mediapipe_preview -v` passes 38 tests with zero skips. Integration tests
use the actual checkpoint with deterministic detector fixtures, checking
pixel conversion, independent affine/packing outputs and top-three scores,
single-load behavior, experimental gating, missing frames, invalid outputs,
label validation, result serialization, reset, resource closure, and separate
session tracking/buffers. Existing real WLASL regression remains passing.
An additional smoke check loaded the real MediaPipe task assets, processed a
synthetic 240x320 RGB frame, and closed both trackers. **Webcam hardware was
not tested.** Compatibility remains unverified; this enables experimental
predictions only.

## Evidence inspected

- `checkpoints/asl100/pytorch_model.bin` is an OrderedDict with 330 tensor
  entries, no preprocessing, training revision, cache provenance, or labels.
- `checkpoints/asl100/config.ini` specifies 50 frames, hidden size 64,
  20 stages and dropout 0.3; it does not identify preprocessing or the split.
- Historical `sign_dataset.py` at commit `2330a47` and the preserved
  `temp_WLASL/code/TGCN/sign_dataset.py` use cached feature coordinates or
  affine `2*(pixel/256 - 0.5)` coordinates, without body/hand pivots.
  Zero coordinates become (-1,-1); confidence is ignored. The upstream cache
  generator has now been located and reproduces this affine xy transform;
  its use and coordinate provenance are not tied to the supplied checkpoint.
- Historical and current loaders retain BODY_25 indices
  0,1,2,3,4,5,6,7,8,15,16,17,18, then 21 left and 21 right hand joints.
  `torch.cat(poses, dim=1)` packs x0,y0,x1,y1,... per joint.
- Training uses random consecutive starts; evaluation uses multiple copies
  of consecutive windows, repeating final frames for short clips. Historical
  missing frames repeat the preceding pose or omit initial missing frames.
  Live preparation keeps missing frames, takes a central consecutive window,
  and repeats the final frame to 50. These are not identical policies.
- `LabelEncoder` sorts glosses alphabetically and training logs label order.
  No checkpoint-linked training log or original split/label map is available.
  The current split's sorted labels alone cannot establish checkpoint labels.
- `train_tgcn.py` and `train_utils.py` save bare state dictionaries; model
  dimensions and successful weight loading cannot prove input compatibility.

Missing evidence: an authenticated connection from these weights to the
training revision, preprocessing and feature-cache generation, image coordinate
units/resizing, temporal/missing-frame policies, and original class-index map.
Historical affine code is evidence of a candidate pipeline, not verification
of this checkpoint. Do not enable inference merely by changing an audit flag.
`checkpoint_compatibility.py` records the reviewed unknown decision and gates
both loading and prediction. Synthetic compatible test fixtures are not claims
about the shipped checkpoint.

## Selected preprocessing for preview and future training

`pivot_v1` remains available through shared `normalize_keypoints`:
body 0:13 subtracts neck 1 and divides by shoulder distance 2-to-5;
left hand 13:34 subtracts wrist 13 and divides by distance 13-to-22;
right hand 34:55 subtracts wrist 34 and divides by distance 34-to-43.
Hands encode local shape; body wrists retain placement. Training reads raw
OpenPose pixel JSON and masks zero-confidence/(0,0) joints; live converts
MediaPipe coordinates to pixels using width and height before normalization.
Nonfinite points and invalid reference groups use (-1,-1).

The existing 1e-6 cutoff was replaced by `MIN_SCALE_PIXELS = 1.0`: references
below one pixel are unresolved at image resolution and amplify subpixel noise.
Exactly one pixel is accepted. This conservative cutoff applies to raw pixel
inputs in both paths; normalized [0,1] coordinates must not be passed directly.
An absolute threshold breaks uniform scale invariance when scaling crosses its
boundary, including changes of image resolution. There is no denominator
clipping or epsilon. Float64 intermediate arithmetic and float32 range checks
keep output finite without changing input or valid independent groups.

Retrain with this pipeline and retain checkpoint-linked provenance and labels
before auditing it as compatible. The one-pixel rule is part of that training
contract; this update does not establish compatibility with older pivot runs.


## Download-source investigation (follow-up audit)

Verified source: local ASL100 SHA-256
`0a66a799f12dd5d8a48a02b441e7499b2ed89ada3e130e059c7e9afc63312e19`
is exactly the LFS SHA-256 published for
`sharonn18/tgcn-wlasl/checkpoints/asl100/pytorch_model.bin` at revision
`dacb4568719caa03c44764034f599a9f8a0f63f4`.
The fetched API response is retained in `checkpoint_source.json`; the revision's
model card is retained in `checkpoint_model_card.md`.
Source URLs:
https://huggingface.co/api/models/sharonn18/tgcn-wlasl/revision/dacb4568719caa03c44764034f599a9f8a0f63f4?blobs=true
https://huggingface.co/sharonn18/tgcn-wlasl/blob/dacb4568719caa03c44764034f599a9f8a0f63f4/README.md

The local Hugging Face cache contains this revision's ASL2000 files, not ASL100.
Commit `2330a47` explicitly points to that ASL2000 snapshot. Thus the source
of ASL100 bytes is established by the remote content hash, not a surviving
ASL100 download log. The snapshot file listing contains model/config files
but **no dataset loader, training script, split, label map, or cache generator**.
Its model card describes normalized coordinates without a formula and claims
MediaPipe 55-point ordering without defining it; this conflicts with the
historical OpenPose mapping. These claims do not verify pivot_v1 or affine.

The local archived ASL100 file has SHA-256
`a61d7dda5f875ce5ebd9d407c56874f77d1cd2aeb4bc7cd0d98a6e1ca4669a0c`.
Its INI uses MAX_STEPS, dropout .5 and batch size 5, unlike the HF config.
It is not byte-identical and no conversion record establishes it as the
ancestor of these weights. Weight/shape similarity would not settle that.

### Explicit historical candidate, not verified checkpoint default

`legacy_affine_keypoints` implements the historical raw-JSON branch in one
shared function; `read_pose_file`, `Sign_Dataset(preprocessing=...)`, and
`normalize_landmarks` accept `legacy_affine_256_candidate` explicitly.
Current defaults remain pivot_v1 for preview/future training; predictions stay
blocked. The affine candidate ignores confidence exactly as historical code
and reads raw JSON, never unproven cached features. Live nonfinite missing
coordinates become zero pixels before affine conversion; infinities are
rejected. No scaling distance or one-pixel threshold is used. No resizing to
256 is invented: the historical formula divides the existing pixel coordinates
by 256. The input extraction/resizing used to generate training pixels still
needs verification. MediaPipe's estimated neck/hip and detector differences
also cannot be made identical to OpenPose merely by changing normalization.

| Property | Historical raw-JSON candidate | pivot_v1 |
|---|---|---|
| Joint order | 13 retained BODY_25, left 21, right 21 | Same retained order |
| Input units | Pixel values divided by fixed 256 | Pixel distances |
| Center/scale | Global affine pixel/128 - 1 | Neck/shoulders; separate wrist/MCP hands |
| Body/hand relationship | Global hand placement and size retained | Local hand shape; placement through body wrists |
| Missing | Confidence ignored; each zero coordinate maps to -1 | Confidence/NaN joint masking; invalid references mask group |
| Tiny reference | No reference distance or threshold | Below 1 pixel masks group |
| Frames | Random training start; evaluation multiple windows; last-frame padding | Live central 50, last-frame padding, missing frames retained |
| Packing | Per joint x0,y0,x1,y1,... | Same |
| Labels | Historical LabelEncoder alphabetical split glosses | Current split sorted; original map unverified |

### Regression evidence and limits

`test_legacy_preprocessing.py` checks manually calculated affine outputs,
including zero pixels, fractions, values outside the image, coincident pivots,
and global hand translation. It compares the shared function and live packing
with a separate transcription of historical raw-JSON arithmetic on real WLASL
video 00295 frames, pads the final frame to 50, loads the real HF ASL100 weights
strictly, and compares logits and decoded labels using identical current split
labels. This proves candidate arithmetic/packing reproduction, **not that the
HF checkpoint was trained with that candidate or those labels**, and does not
validate historical feature caches or multiple-window evaluation.

### Exact evidence needed to restore pretrained predictions

From the publisher of the exact HF ASL100 hash/revision, obtain:
1. The training repository URL and commit, especially `sign_dataset.py`,
   pose extraction/resizing script, and any feature-cache generator.
2. The source checkpoint filename/hash and conversion script or record if
   `pytorch_model.bin` was converted from an upstream checkpoint.
3. The original `asl100.json` training split or ordered 100-class label map
   (or training log containing class indices and glosses).
4. The training/evaluation temporal settings and handling of missing frames,
   and confirmation of whether training loaded cached features or raw JSON.
5. If cached: a representative original raw pose JSON plus its generated
   feature tensor, with coordinate/image-size provenance.

These are concrete missing artifacts at the identified publisher revision,
not a request to retrain by default. Once established, select the verified
shared preprocessing and temporal policy and restore the prediction gate.

## Original-source follow-up, 2026-10-05

**Decision: checkpoint compatibility remains unknown. There is no selected
pretrained inference default, and predictions remain blocked.** `pivot_v1`
continues to be the preview/future-training default only. The explicit shared
`legacy_affine_256_candidate` is a reproduced upstream contract, not an
authenticated contract for the HF weights. No normalization safety rules were
changed in this follow-up.

### Newly pursued evidence

- [Publisher commit history](https://huggingface.co/sharonn18/tgcn-wlasl/commits/main)
  has six commits, all uploads on 2025-12-01, with empty extended messages.
  Retained API evidence: `checkpoint_upload_history.json`.
- The original checkpoint upload revision
  `3ad42062f8c191272ca7f7f4c09199e815250856` already has ASL100 LFS SHA-256
  `0a66a799f12dd5d8a48a02b441e7499b2ed89ada3e130e059c7e9afc63312e19`.
  Its file list has only model/config code and checkpoint files. The next
  revision `28fc4588680b460227cd8b4789d90560aa239986` adds configs, not training
  code. Their complete API listings are retained in
  `checkpoint_source_<revision>.json`. The later model card does not establish
  how the already-uploaded weights were trained or converted.
- [Publisher discussion #1](https://huggingface.co/sharonn18/tgcn-wlasl/discussions/1)
  asks for the joint order, coordinate system, and temporal packing. The
  inspected page contains no publisher answer or preprocessing artifact.
  [Publisher profile](https://huggingface.co/sharonn18) links no training repo;
  exact-name searches produced no publisher-linked training source. The GitHub
  API lookup for the same username returned 404; this does not establish that
  the publisher has no GitHub account under another name. No message was sent.
- Located original [gen_features.py at upstream revision
  997fc58280e20fd69f1f695bcc5d8edd33d7c64a](https://github.com/dxli94/WLASL/blob/997fc58280e20fd69f1f695bcc5d8edd33d7c64a/code/TGCN/gen_features.py).
  Lines 44-72 concatenate BODY_25/left/right hands, exclude the 12 body joints,
  apply `2*(pixel/256-.5)`, and save xy as the first two cache columns. The
  local immutable-source copy is `provenance/upstream_gen_features.py` (UTF-8
  BOM/line endings may differ). Existing caches are skipped, so finding this
  generator does not authenticate every cache encountered by a training run.
- The upstream [training script at that revision](https://github.com/dxli94/WLASL/blob/997fc58280e20fd69f1f695bcc5d8edd33d7c64a/code/TGCN/train_tgcn.py)
  constructs hidden width `num_samples*2` (100 for 50 frames), whereas the HF
  ASL100 input graph weight is `(100,64)`. This revision cannot directly train
  this model as configured without modifications. The local archived official
  ASL100 dictionary also has 344 entries versus HF's 330. These differences
  leave any ancestor/conversion claim unestablished.

### What is verified about the upstream pipeline

Sources refer to upstream revision `997fc58280e20fd69f1f695bcc5d8edd33d7c64a`,
also retained in this repository's history, not the modified current loader:

| Property | Original source and established behavior | Checkpoint linkage |
|---|---|---|
| Normalization/cache | `gen_features.py:44-72`, `sign_dataset.py:43-92`: fixed affine xy; cache reads first two columns | Unknown |
| Units/resizing | Both scripts consume JSON numbers directly, divide by 256; neither resizes/extracts poses | Extraction dimensions/settings missing |
| 55 joints | `sign_dataset.py:31,54-61`: BODY_25 0..8,15..18, then left 0..20, right 0..20 | Unknown; HF card's MediaPipe wording is insufficient |
| Missing joints | Confidence ignored; each zero coordinate maps to -1 independently | Unknown |
| Missing frames | `sign_dataset.py:175-199`: empty people repeats prior pose; initial empty frames omitted; all-empty clips fail; initial omissions can leave short tensors because padding uses requested-frame count | Unknown; live retention differs |
| Sampling/padding | `sign_dataset.py:204-265`: random consecutive training window; four consecutive evaluation windows; short clips repeat last frame per copy | Unknown |
| Packing/evaluation | `sign_dataset.py:197-199`: concatenate (55,2) frames along columns; `train_utils.py:79-94`: slice four copies and average logits | Unknown; live single center window differs |
| Class order | `sign_dataset.py:134-150`: LabelEncoder over sorted split glosses; `train_tgcn.py:40-41` logs indexed labels | Exact training split/log missing |

The one-pixel threshold **does not match this upstream affine preprocessing**:
there is no reference distance or threshold at all. It remains exclusive to
`pivot_v1`. Camera width/height conversion to pixels is preserved; no speculative
256-by-256 resize has been introduced. Even verified affine arithmetic would
not make MediaPipe estimated neck/hip and detector outputs identical to
original OpenPose observations.

### Added regression coverage and result

The regression suite now executes the preserved original cache-generator
functions, redirecting only its hardcoded filesystem roots and excluding its
top-level multiprocessing/I/O. Generated xy is exactly equal to both the shared
affine function and an independent historical arithmetic oracle, including
nonzero coordinates with zero confidence. Separate manually enumerated tests
cover all three four-copy evaluation-window branches and short-clip padding.
The original generator emits a ResourceWarning for its unclosed JSON read;
its source is deliberately preserved.

Validation: `python -m unittest test_legacy_preprocessing test_mediapipe_preview -v`
passes **33 tests, zero skips**, including real WLASL 00295 identical-frame
tensor and exact checkpoint-logit comparisons. These verify reproduction of the
upstream candidate; they do not supply the absent publisher training link,
original extraction dimensions, or class map. The preprocessing prerequisite
therefore remains unresolved pending the concrete artifacts listed above.
