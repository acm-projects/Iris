# MediaPipe preprocessing status

- **Model joints (55 total):** Positions 0-12 are pose points in this order:
  nose (Pose 0), shoulder midpoint (11/12), right shoulder/elbow/wrist
  (12/14/16), left shoulder/elbow/wrist (11/13/15), hip midpoint (23/24),
  right eye (5), left eye (2), right ear (8), left ear (7). Positions 13-33
  contain left-hand landmarks 0-20, and positions 34-54 contain right-hand
  landmarks 0-20, assigned using MediaPipe handedness labels.
- **Provisional normalization:** each full-frame MediaPipe normalized
  coordinate is mapped independently with `value = 2 * value - 1`.
  This does not apply shoulder centering, cropping, or bounding-box scaling.
- **Missing points:** remapping uses NaN internally. Before model input, if
  either coordinate of a point is missing, both normalized coordinates become
  `(-1, -1)`. This matches the original loader's numeric result for OpenPose
  `(0, 0)` coordinates.
- **Unresolved training compatibility:** the original training scale and
  coordinate origin are not established. The original loader divides pixel
  coordinates by 256, while the WLASL paper describes resizing based on the
  person bounding-box diagonal. The available sources do not show how that
  scaling relates to the keypoint frame's crop or origin. Treat live predictions
  as experimental until that preprocessing is verified against reference
  training inputs.
