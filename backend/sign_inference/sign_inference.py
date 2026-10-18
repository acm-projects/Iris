"""Session-scoped camera inference; no UI or transport dependencies.

Frames are uint8 RGB images and timestamps are strictly increasing milliseconds.
Each instance owns its model, MediaPipe VIDEO trackers, and recording buffer.
"""
import numpy as np
from pathlib import Path
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from checkpoint_compatibility import ASL100_AUDIT, require_compatible
from configs import Config
from tgcn_model import GCN_muti_att
from landmark_normalization import WLASL_AFFINE_VERSION, preprocess_keypoints
from inference_utils import (
    CHECKPOINT_DIR, MODEL_DIR, SPLIT_FILE, load_class_labels,
    remap_landmarks, prepare_recording, assemble_model_input,
)


class SignInference:
    """One instance per session; methods must be called serially.

    Normal operation requires the repository's reviewed compatibility audit.
    ``experimental_legacy`` remains accepted for older scripts but now selects
    the same audited WLASL affine contract.
    reset clears recordings but preserves tracker timestamps; close releases them.
    """
    def __init__(self, *, experimental_legacy=False, checkpoint_dir=CHECKPOINT_DIR,
                 split_file=SPLIT_FILE, model_dir=MODEL_DIR, device=None):
        self.experimental = bool(experimental_legacy)
        require_compatible(ASL100_AUDIT)
        if ASL100_AUDIT.preprocessing != WLASL_AFFINE_VERSION:
            raise RuntimeError('The ASL100 audit selected an unsupported preprocessing contract.')
        self.preprocessing = ASL100_AUDIT.preprocessing
        checkpoint_dir, model_dir = Path(checkpoint_dir), Path(model_dir)
        required_files = {
            'WLASL-100 class split': Path(split_file),
            'TGCN configuration': checkpoint_dir / 'config.ini',
            'TGCN checkpoint': checkpoint_dir / 'pytorch_model.bin',
            'MediaPipe pose task': model_dir / 'pose_landmarker.task',
            'MediaPipe hand task': model_dir / 'hand_landmarker.task',
        }
        missing = [
            f'{name}: {path}'
            for name, path in required_files.items()
            if not path.is_file()
        ]
        if missing:
            details = '\n  - '.join(missing)
            raise FileNotFoundError(
                f'Missing inference files:\n  - {details}\n'
                'Run backend/sign_inference/setup_assets.py, then retry.'
            )
        self.labels = load_class_labels(split_file, 100)
        self.config = Config(str(checkpoint_dir / 'config.ini'))
        if self.config.num_samples != 50:
            raise ValueError('ASL100 inference requires a 50-frame configuration.')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = GCN_muti_att(input_feature=100, hidden_feature=self.config.hidden_size,
                                 num_class=100, p_dropout=self.config.drop_p,
                                 num_stage=self.config.num_stages).to(self.device)
        checkpoint = torch.load(checkpoint_dir / 'pytorch_model.bin',
                                map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
        self.model.eval()
        self._pose = self._hands = None
        self._closed = False
        self._timestamp = -1
        self.reset()
        try:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / 'pose_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / 'hand_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO, num_hands=2))
        except Exception:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('Inference session is closed.')

    @property
    def captured_frame_count(self):
        return self._captured

    @property
    def recording(self):
        return self._recording

    @property
    def last_model_points(self):
        """Newest keypoints in the checkpoint's 256×256 coordinate space."""
        return None if self._last_model_points is None else self._last_model_points.copy()

    def reset(self):
        self._ensure_open()
        self._records = []
        self._captured = 0
        self._recording = False
        self._previous_pose = None
        self._last_model_points = None

    def start_sequence(self):
        self._ensure_open()
        if self._recording:
            raise RuntimeError('Already recording; finish or reset first.')
        self.reset()
        self._recording = True

    def process_frame(self, rgb_frame, timestamp_ms):
        """Track a frame; buffer only while recording. Return pixel overlay data.

        A missing body detection stands in for original OpenPose empty people:
        initial missing frames are omitted, later ones repeat the preceding pose.
        Missing joints within detected frames map to zero pixels during affine.
        """
        self._ensure_open()
        frame = np.asarray(rgb_frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0:
            raise ValueError('Expected a nonempty uint8 RGB image shaped (H,W,3).')
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, np.integer)):
            raise ValueError('Timestamp must be integer milliseconds.')
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms < 0 or timestamp_ms <= self._timestamp:
            raise ValueError('Timestamps must be nonnegative and strictly increasing across the session.')
        # Consume the timestamp even if one detector fails: tracker state may advance.
        self._timestamp = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        pose = self._pose.detect_for_video(image, timestamp_ms)
        hands = self._hands.detect_for_video(image, timestamp_ms)
        normalized_points, _ = remap_landmarks(pose, hands)
        # The upstream checkpoint was trained from OpenPose coordinates on
        # 256×256 frames. Keep model coordinates independent of camera
        # resolution while returning frame-pixel points for preview overlays.
        points = normalized_points * np.float32(256)
        display_points = normalized_points * np.array(
            [frame.shape[1], frame.shape[0]], dtype=np.float32
        )
        self._last_model_points = points.copy()
        if self._recording:
            self._captured += 1
            if pose.pose_landmarks:
                self._previous_pose = points.copy()
            if self._previous_pose is not None:
                self._records.append((timestamp_ms, self._previous_pose.copy()))
        return display_points.copy()

    def predict_keypoints(self, records):
        """Predict a sequence of precomputed 55-joint pixel keypoints.

        This is also the deterministic seam used by the static regression test:
        MediaPipe extraction and model-contract validation can be tested
        independently.
        """
        self._ensure_open()
        records = list(records)
        frames, _ = prepare_recording(records, 50)
        normalized = np.stack([preprocess_keypoints(p, self.preprocessing) for p in frames])
        packed = assemble_model_input(normalized)
        if packed.shape != (1,55,100) or packed.dtype != np.float32 or not np.isfinite(packed).all():
            raise ValueError('Invalid model input: expected finite float32 (1,55,100).')
        require_compatible(ASL100_AUDIT)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(packed).to(self.device))
            if logits.shape != (1,len(self.labels)) or not torch.isfinite(logits).all():
                raise ValueError('Model output must be finite (1,100), matching the label map.')
            scores, indices = torch.softmax(logits, dim=1)[0].topk(3)
        return {
            'predictions': [{'gloss':self.labels[i], 'score':float(s)}
                            for i,s in zip(indices.cpu().tolist(), scores.cpu().tolist())],
            'captured_frame_count':len(records),
            'usable_frame_count':len(records),
            'prepared_frame_count':len(frames),
            'preprocessing_mode':self.preprocessing,
            'experimental':self.experimental,
            'preprocessing_verification_status':ASL100_AUDIT.status,
            'label_map_verification_status':'verified',
            'temporal_policy':'central_consecutive_50_repeat_last',
            'score_type':'softmax_model_score',
        }

    def finish_and_predict(self):
        self._ensure_open()
        if not self._recording:
            raise RuntimeError('No active sequence; start recording first.')
        self._recording = False
        if not self._records:
            raise ValueError('No detected body poses in recording; record a visible signer.')
        return self.predict_keypoints(self._records)

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self._records = []
        self._recording = False
        self._previous_pose = None
        self._last_model_points = None
        try:
            if self._pose is not None:
                self._pose.close()
        finally:
            if self._hands is not None:
                self._hands.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()
