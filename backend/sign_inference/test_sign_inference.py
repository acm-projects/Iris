import json
import hashlib
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sign_inference import SignInference
from inference_utils import CHECKPOINT_DIR
from checkpoint_compatibility import PreprocessingAudit


FIXTURE = Path(__file__).resolve().parent / 'test_fixtures' / 'static_keypoints.json'


def make_pose_result():
    landmarks = [
        SimpleNamespace(x=index / 100, y=-index / 100)
        for index in range(33)
    ]
    return SimpleNamespace(pose_landmarks=[landmarks])


class Detector:
    def __init__(self, result):
        self.result = result
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp):
        self.timestamps.append(timestamp)
        return self.result

    def close(self):
        self.closed = True


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pose = Detector(make_pose_result())
        self.hands = Detector(SimpleNamespace(hand_landmarks=[],handedness=[]))
        self.frame = np.zeros((240,320,3),dtype=np.uint8)

    def session(self):
        with patch('sign_inference.vision.PoseLandmarker.create_from_options',return_value=self.pose), \
             patch('sign_inference.vision.HandLandmarker.create_from_options',return_value=self.hands):
            session = SignInference(experimental_legacy=True,device='cpu')
        self.addCleanup(session.close)
        return session

    def test_incompatible_gate_before_load(self):
        blocked = PreprocessingAudit('unknown', 'test contract is not reviewed')
        with patch('sign_inference.ASL100_AUDIT', blocked), \
             patch('sign_inference.torch.load') as load:
            with self.assertRaisesRegex(RuntimeError,'predictions blocked'):
                SignInference()
            load.assert_not_called()

    def test_missing_assets_report_setup_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'setup_assets.py'):
                SignInference(
                    experimental_legacy=True,
                    checkpoint_dir=root / 'checkpoint',
                    split_file=root / 'asl100.json',
                    model_dir=root / 'mediapipe',
                )

    def test_actual_model_pixel_conversion_packing_result_and_reset(self):
        if not (CHECKPOINT_DIR/'pytorch_model.bin').is_file():
            self.skipTest('Checkpoint unavailable')
        torch.set_num_threads(1)
        with patch('sign_inference.torch.load',wraps=torch.load) as load:
            session = self.session()
            session.start_sequence()
            points = session.process_frame(self.frame,1)
            np.testing.assert_allclose(points[0],[0,0])
            np.testing.assert_allclose(points[2],[38.4,-28.8],rtol=1e-6)
            inputs = []
            def observe(model, args):
                self.assertFalse(model.training)
                self.assertTrue(torch.is_inference_mode_enabled())
                inputs.append(args[0].clone())
            hook = session.model.register_forward_pre_hook(observe)
            self.addCleanup(hook.remove)
            result = session.finish_and_predict()
            self.assertEqual(inputs[0].shape,(1,55,100))
            self.assertEqual(inputs[0].dtype,torch.float32)
            self.assertTrue(torch.isfinite(inputs[0]).all())
            np.testing.assert_array_equal(inputs[0][0,0].numpy(),-np.ones(100))
            model_points = points / np.array([320, 240], dtype=np.float32) * np.float32(256)
            expected_xy = np.nan_to_num(model_points,nan=0)/np.float32(128)-np.float32(1)
            expected_input = np.repeat(expected_xy[:,None,:],50,axis=1).reshape(1,55,100)
            np.testing.assert_allclose(inputs[0].numpy(), expected_input, rtol=1e-6, atol=1e-7)
            with torch.inference_mode():
                expected_scores, expected_indices = session.model(torch.from_numpy(expected_input)).softmax(1)[0].topk(3)
            self.assertEqual([p['gloss'] for p in result['predictions']],
                             [session.labels[i] for i in expected_indices.tolist()])
            np.testing.assert_allclose(
                [p['score'] for p in result['predictions']],
                expected_scores.tolist(),
                rtol=1e-5,
                atol=1e-7,
            )
            self.assertEqual(result['captured_frame_count'],1)
            self.assertEqual(result['prepared_frame_count'],50)
            self.assertEqual(len(result['predictions']),3)
            self.assertEqual(result['preprocessing_mode'],'wlasl_affine_256_v1')
            self.assertTrue(result['experimental'])
            self.assertEqual(result['preprocessing_verification_status'],'compatible')
            self.assertEqual(result['label_map_verification_status'],'verified')
            json.dumps(result,allow_nan=False)
            session.reset()
            self.assertEqual(session.captured_frame_count,0)
            self.assertFalse(session.recording)
            session.start_sequence()
            with self.assertRaisesRegex(ValueError,'strictly increasing'):
                session.process_frame(self.frame,1)
            session.process_frame(self.frame,2)
            session.finish_and_predict()
            self.assertEqual(load.call_count,1)

    def test_missing_body_policy_and_empty_recording(self):
        session = self.session()
        session.start_sequence()
        self.pose.result = SimpleNamespace(pose_landmarks=[])
        session.process_frame(self.frame,1)
        self.assertEqual(len(session._records),0)
        with self.assertRaisesRegex(ValueError,'No detected body'):
            session.finish_and_predict()
        session.start_sequence()
        session.process_frame(self.frame,2)
        self.pose.result = make_pose_result()
        session.process_frame(self.frame,3)
        self.pose.result = SimpleNamespace(pose_landmarks=[])
        session.process_frame(self.frame,4)
        self.assertEqual(session.captured_frame_count,3)
        self.assertEqual(len(session._records),2)
        np.testing.assert_array_equal(session._records[0][1],session._records[1][1])
        session.close()
        session.close()
        self.assertTrue(self.pose.closed and self.hands.closed)
        with self.assertRaisesRegex(RuntimeError,'closed'):
            session.start_sequence()

    def test_output_count_finite_validation_and_label_map(self):
        session = self.session()
        self.assertEqual(len(set(session.labels)),100)
        self.assertEqual(session.labels,sorted(session.labels))
        for output in (torch.zeros((1,99)),torch.full((1,100),float('nan'))):
            session.start_sequence()
            session.process_frame(self.frame,session._timestamp+1)
            with patch.object(session.model,'forward',return_value=output):
                with self.assertRaisesRegex(ValueError,'Model output'):
                    session.finish_and_predict()
        with patch('sign_inference.load_class_labels',side_effect=ValueError('Expected 100')):
            with self.assertRaisesRegex(ValueError,'100'):
                SignInference(experimental_legacy=True)

    def test_sessions_keep_separate_buffers_and_tracking(self):
        first = self.session()
        pose2 = Detector(make_pose_result())
        hands2 = Detector(self.hands.result)
        with patch('sign_inference.vision.PoseLandmarker.create_from_options',return_value=pose2), \
             patch('sign_inference.vision.HandLandmarker.create_from_options',return_value=hands2):
            second = SignInference(experimental_legacy=True,device='cpu')
        self.addCleanup(second.close)
        first.start_sequence()
        second.start_sequence()
        first.process_frame(self.frame,10)
        second.process_frame(self.frame,1)
        first.reset()
        self.assertEqual(second.captured_frame_count,1)
        self.assertEqual(pose2.timestamps,[1])

    def test_static_keypoint_checkpoint_regression(self):
        checkpoint = CHECKPOINT_DIR / 'pytorch_model.bin'
        if not checkpoint.is_file():
            self.skipTest('Checkpoint unavailable')
        fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(digest, fixture['expected']['checkpoint_sha256'])
        base = np.asarray(fixture['base_points'], dtype=np.float32)
        motion = fixture['motion']
        records = []
        for index in range(motion['frames']):
            phase = math.sin(2 * math.pi * index / motion['period_denominator'])
            points = base.copy()
            points[13:, 0] += phase * motion['all_hands_x_amplitude']
            points[13:34, 1] += phase * motion['left_hand_y_amplitude']
            points[34:, 1] += phase * motion['right_hand_y_amplitude']
            records.append((index, points))
        result = self.session().predict_keypoints(records)
        self.assertEqual(result['preprocessing_mode'], fixture['expected']['preprocessing'])
        self.assertEqual(
            [prediction['gloss'] for prediction in result['predictions']],
            fixture['expected']['top_glosses'],
        )
        np.testing.assert_allclose(
            [prediction['score'] for prediction in result['predictions']],
            fixture['expected']['top_scores'],
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == '__main__':
    unittest.main()
