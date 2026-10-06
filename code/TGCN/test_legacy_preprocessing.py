"""Independent historical affine regression; not checkpoint training proof."""
import json
import ast
import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from landmark_normalization import LEGACY_AFFINE_VERSION, legacy_affine_keypoints
from mediapipe_preview import assemble_model_input, normalize_landmarks
from sign_dataset import read_pose_file, k_copies_fixed_length_sequential_sampling

ROOT = Path(__file__).resolve().parents[2]
RETAINED = [0,1,2,3,4,5,6,7,8,15,16,17,18]


def historical_oracle(path):
    person = json.loads(path.read_text())["people"][0]
    values = person["pose_keypoints_2d"] + person["hand_left_keypoints_2d"] + person["hand_right_keypoints_2d"]
    excluded = {9,10,11,12,13,14,19,20,21,22,23,24}
    x = torch.FloatTensor([v for i,v in enumerate(values) if i%3 == 0 and i//3 not in excluded])
    y = torch.FloatTensor([v for i,v in enumerate(values) if i%3 == 1 and i//3 not in excluded])
    return torch.stack([2*((x/256)-.5), 2*((y/256)-.5)], dim=1)


class HistoricalAffineTests(unittest.TestCase):
    def test_original_cache_generator_xy_matches_shared_affine(self):
        # Execute the preserved upstream functions, without its top-level Pool
        # and dataset I/O. Only redirect its two machine-specific root paths.
        source = Path(__file__).parent/'provenance/upstream_gen_features.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'features'
            raw = Path(directory)/'poses'
            cache.mkdir()
            (raw/'fixture').mkdir(parents=True)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value == '/home/dxli/workspace/nslt/code/Pose-GCN/posegcn/features':
                        node.value = str(cache)
                    elif node.value == '/home/dxli/workspace/nslt/data/pose/pose_per_individual_videos':
                        node.value = str(raw)
            person = {}
            for key, count in [('pose_keypoints_2d',25),('hand_left_keypoints_2d',21),('hand_right_keypoints_2d',21)]:
                # Deliberately distinct joints and zero-confidence nonzero points.
                person[key] = [value for i in range(count) for value in (i*8+.25,i*4+.5,0)]
            path = raw/'fixture/image_00001_keypoints.json'
            path.write_text(json.dumps({'people':[person]}))
            namespace = {'json':json,'os':os,'torch':torch,
                         'body_pose_exclude':set(range(25))-set(RETAINED)}
            exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
            namespace['gen']([{'instances':[{'video_id':'fixture','frame_start':1,'frame_end':1}]}])
            original = torch.load(cache/'fixture/image_00001_ft.pt',weights_only=True)[:,:2]
            torch.testing.assert_close(read_pose_file(path,LEGACY_AFFINE_VERSION),original,rtol=0,atol=0)
            torch.testing.assert_close(original,historical_oracle(path),rtol=0,atol=0)

    def test_historical_evaluation_windows_and_padding(self):
        sample = k_copies_fixed_length_sequential_sampling
        self.assertEqual(sample(10,12,5,4),[10,11,12,12,12]*4)
        self.assertEqual(sample(1,30,5,4),list(range(5,25)))
        self.assertEqual(sample(1,12,5,4),
                         [1,2,3,4,5,3,4,5,6,7,5,6,7,8,9,7,8,9,10,11])

    def test_independent_known_pixel_values(self):
        points = np.tile([128., 256.], (55,1)).astype(np.float32)
        points[:5] = [[0,0],[64,192],[256,512],[-128,128],[128.25,127.75]]
        expected = np.tile([0.,1.], (55,1)).astype(np.float32)
        expected[:5] = [[-1,-1],[-.5,.5],[1,3],[-2,0],[.001953125,-.001953125]]
        original = points.copy()
        np.testing.assert_array_equal(legacy_affine_keypoints(points), expected)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(legacy_affine_keypoints(points).dtype, np.float32)

    def test_no_pivot_threshold_or_independent_hand_centering(self):
        points = np.full((55,2),128.,dtype=np.float32)
        points[22] = [128.25,128]
        actual = legacy_affine_keypoints(points)
        np.testing.assert_array_equal(actual[22],[.001953125,0])
        points[13:34] += 64
        shifted = legacy_affine_keypoints(points)
        np.testing.assert_array_equal(shifted[13:34],actual[13:34]+.5)
        np.testing.assert_array_equal(shifted[:13],actual[:13])

    def test_live_missing_points_are_affine_zero_pixels(self):
        points = np.full((55,2),np.nan,dtype=np.float32)
        actual = normalize_landmarks(points,LEGACY_AFFINE_VERSION)
        np.testing.assert_array_equal(actual,-np.ones((55,2)))
        points[0,0] = np.inf
        with self.assertRaises(ValueError): legacy_affine_keypoints(points)

    def test_real_frames_inputs_and_checkpoint_logits_against_historical_oracle(self):
        folder = ROOT/'WLASL/data/pose_per_individual_videos/pose_per_individual_videos/00295'
        checkpoint = ROOT/'code/TGCN/checkpoints/asl100/pytorch_model.bin'
        paths = sorted(folder.glob('*keypoints.json'))
        if not paths or not checkpoint.is_file():
            self.skipTest('Real WLASL fixture or checkpoint absent')
        paths = (paths + [paths[-1]]*50)[:50]
        oracle = torch.cat([historical_oracle(p) for p in paths],dim=1).unsqueeze(0)
        updated = assemble_model_input(np.stack([read_pose_file(p,LEGACY_AFFINE_VERSION).numpy() for p in paths]))
        np.testing.assert_array_equal(updated,oracle.numpy())
        # Raw pixel input for live conversion's downstream path, same real frames.
        live = []
        for p in paths:
            person=json.loads(p.read_text())["people"][0]
            body=np.array(person['pose_keypoints_2d']).reshape(25,3)[RETAINED,:2]
            hands=[np.array(person[k]).reshape(21,3)[:,:2] for k in ('hand_left_keypoints_2d','hand_right_keypoints_2d')]
            live.append(normalize_landmarks(np.concatenate([body,*hands]),LEGACY_AFFINE_VERSION))
        np.testing.assert_array_equal(assemble_model_input(live),oracle.numpy())
        from configs import Config
        from tgcn_model import GCN_muti_att
        config=Config(str(checkpoint.parent/'config.ini'))
        model=GCN_muti_att(input_feature=100,hidden_feature=config.hidden_size,num_class=100,p_dropout=config.drop_p,num_stage=config.num_stages)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval()
        torch.set_num_threads(1)
        with torch.inference_mode():
            expected=model(oracle)
            actual=model(torch.from_numpy(updated))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        split=json.loads((ROOT/'WLASL/data/splits/asl100.json').read_text())
        from sklearn.preprocessing import LabelEncoder
        labels=LabelEncoder().fit([entry['gloss'] for entry in split]).classes_.tolist()
        from mediapipe_preview import class_labels_from_entries
        self.assertEqual(class_labels_from_entries(split),labels)
        self.assertEqual(labels[int(actual.argmax())],labels[int(expected.argmax())])
