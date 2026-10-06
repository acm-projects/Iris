import json
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils
from landmark_normalization import NORMALIZATION_VERSION, preprocess_keypoints

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff


def read_pose_file(filepath, preprocessing=NORMALIZATION_VERSION):
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    try:
        with open(filepath, encoding="utf-8") as pose_file:
            content = json.load(pose_file)["people"][0]
    except IndexError:
        return None

    # Read raw JSON every time: legacy feature caches use incompatible affine
    # coordinates and must not bypass the shared pivot normalization.
    body = np.asarray(content["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    left = np.asarray(content["hand_left_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    right = np.asarray(content["hand_right_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    retained = [index for index in range(25) if index not in body_pose_exclude]
    landmarks = np.concatenate((body[retained], left, right))
    points = landmarks[:, :2].copy()
    # OpenPose uses (0,0) and zero confidence for undetected joints.
    missing = (landmarks[:, 2] <= 0) | (points == 0).all(axis=1)
    if preprocessing == NORMALIZATION_VERSION:
        points[missing] = np.nan
    return torch.from_numpy(preprocess_keypoints(points, preprocessing))


class Sign_Dataset(Dataset):
    def __init__(self, index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=25, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None, preprocessing=NORMALIZATION_VERSION):
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.preprocessing = preprocessing
        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]

        self.test_index_file = test_index_file
        self._make_dataset(index_file_path, split)

        self.index_file_path = index_file_path
        self.pose_root = pose_root
        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples

        self.img_transforms = img_transforms
        self.video_transforms = video_transforms

        self.num_copies = num_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]
        # frames of dimensions (T, H, W, C)
        x = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples)

        if self.video_transforms:
            x = self.video_transforms(x)

        y = gloss_cat

        return x, y, video_id

    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        if self.test_index_file is not None:
            print('Trained on {}, tested on {}'.format(index_file_path, self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, instances = gloss_entry['gloss'], gloss_entry['instances']
            gloss_cat = utils.labels2cat(self.label_encoder, [gloss])[0]

            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)

    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        else:
            raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sample_strategy))

        for i in frames_to_sample:
            pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
            # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
            try:
                pose = read_pose_file(pose_path, self.preprocessing)
            except OSError:
                print("Skipping unreadable keypoint file:", pose_path)
                pose = None

            if pose is not None:
                if self.img_transforms:
                    pose = self.img_transforms(pose)

                poses.append(pose)
            else:
                try:
                    poses.append(poses[-1])
                except IndexError:
                    print(pose_path)

        pad = None

        # if len(frames_to_sample) < num_samples:
        if len(poses) < num_samples:
            num_padding = num_samples - len(frames_to_sample)
            last_pose = poses[-1]
            pad = last_pose.repeat(1, num_padding)

        poses_across_time = torch.cat(poses, dim=1)
        if pad is not None:
            poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample


if __name__ == '__main__':
    # root = '/home/dxli/workspace/nslt'
    #
    # split_file = os.path.join(root, 'data/splits-with-dialect-annotated/asl100.json')
    # pose_data_root = os.path.join(root, 'data/pose/pose_per_individual_videos')
    #
    # num_samples = 64
    #
    # train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
    #                              img_transforms=None, video_transforms=None,
    #                              num_samples=num_samples)
    #
    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #
    # cnt = 0
    # for batch_idx, data in enumerate(train_data_loader):
    #     print(batch_idx)
    #     x = data[0]
    #     y = data[1]
    #     print(x.size())
    #     print(y.size())

    print(k_copies_fixed_length_sequential_sampling(0, 2, 20, num_copies=3))
