# ASL100 checkpoint contract

Status: **compatible for Iris's WLASL-100 MVP**  
Reviewed: 2026-10-18

## Pinned artifacts

- Model repository: `sharonn18/tgcn-wlasl`
- Revision: `dacb4568719caa03c44764034f599a9f8a0f63f4`
- Checkpoint: `checkpoints/asl100/pytorch_model.bin`
- SHA-256: `0a66a799f12dd5d8a48a02b441e7499b2ed89ada3e130e059c7e9afc63312e19`
- Architecture/config: strict state-dict load into the upstream WLASL
  `GCN_muti_att` architecture, 50 frames, 55 joints, 100 classes.

## Input contract

Iris reproduces the public WLASL TGCN loader in
`dxli94/WLASL/code/TGCN/sign_dataset.py`:

1. OpenPose BODY_25 points 0–8 and 15–18 (13 body points), followed by 21
   left-hand and 21 right-hand points.
2. Pixel coordinates from a 256×256 pose frame.
3. Each x/y value is transformed with `2 * (pixel / 256 - 0.5)`.
4. Fifty consecutive frames are packed as `(1, 55, 100)`.
5. Class indices are alphabetical WLASL glosses, matching the upstream
   `LabelEncoder` fitted over the selected `asl100.json` split.

MediaPipe normalized landmarks are remapped into that joint order and projected
into a virtual 256×256 frame before applying the affine transform. Preview
overlay points are scaled separately to the actual camera resolution. This is
a compatibility adapter; the checkpoint was not trained directly on MediaPipe
detections, so real-camera accuracy must still be measured and reported
separately.

## Regression lock

`test_fixtures/static_keypoints.json` contains a deterministic 50-frame
precomputed keypoint sequence and the expected top-three output from the pinned
checkpoint. `test_sign_inference.py` verifies the checkpoint hash, transform,
label order, and output scores. A checkpoint, split, joint-map, or
preprocessing change must update this document and fixture in the same review.
