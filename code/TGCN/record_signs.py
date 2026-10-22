"""Record sign clips from the webcam as OpenPose-format keypoint JSON.

R starts the first clip, S saves it and starts the next, D discards the current
clip, Q quits. Each clip becomes my_signs/pose/<clip_id>/image_NNNNN_keypoints.json
(BODY_25 + 21-point hands, pixel x/y/confidence, zeros for missing points), and
my_signs/index.json lists clips in the WLASL split-file format.
"""
import argparse
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from inference_utils import MODEL_DIR

OUTPUT_DIR = Path(__file__).resolve().parent / 'my_signs'
# BODY_25 index -> MediaPipe pose landmark(s); a pair is averaged, None has no equivalent.
BODY_25_FROM_MEDIAPIPE = [0, (11, 12), 12, 14, 16, 11, 13, 15, (23, 24), 24, 26, 28, 23, 25, 27,
                          5, 2, 8, 7, 31, None, 29, 32, None, 30]


def body_keypoints(pose, width, height):
    values = []
    for source in BODY_25_FROM_MEDIAPIPE:
        if source is None:
            values += [0, 0, 0]
            continue
        marks = [pose[i] for i in (source if isinstance(source, tuple) else (source,))]
        values += [float(np.mean([m.x for m in marks]) * width), float(np.mean([m.y for m in marks]) * height),
                   float(min(m.visibility for m in marks))]
    return values


def hand_keypoints(hand_result, width, height):
    """Match inference_utils.remap_landmarks: handedness label picks the slot, strongest wins."""
    hands = {'Left': (-1.0, [0] * 63), 'Right': (-1.0, [0] * 63)}
    for landmarks, categories in zip(hand_result.hand_landmarks, hand_result.handedness):
        if not categories or len(landmarks) != 21:
            continue
        side, score = categories[0].category_name, categories[0].score
        if side in hands and score > hands[side][0]:
            hands[side] = (score, [v for m in landmarks for v in (m.x * width, m.y * height, score)])
    return hands['Left'][1], hands['Right'][1]


def frame_json(pose_result, hand_result, width, height):
    # No body detection is written like an OpenPose empty-people frame.
    if not pose_result.pose_landmarks:
        return {'version': 1.3, 'people': []}
    left, right = hand_keypoints(hand_result, width, height)
    return {'version': 1.3, 'people': [{
        'person_id': [-1],
        'pose_keypoints_2d': body_keypoints(pose_result.pose_landmarks[0], width, height),
        'face_keypoints_2d': [], 'hand_left_keypoints_2d': left, 'hand_right_keypoints_2d': right,
        'pose_keypoints_3d': [], 'face_keypoints_3d': [],
        'hand_left_keypoints_3d': [], 'hand_right_keypoints_3d': []}]}


def next_clip_number(gloss, signer):
    pattern = re.compile(rf'^{re.escape(gloss)}_{re.escape(signer)}_(\d+)$')
    numbers = [int(m.group(1)) for p in (OUTPUT_DIR / 'pose').glob('*') if (m := pattern.match(p.name))]
    return max(numbers, default=0) + 1


def save_clip(gloss, signer, frames):
    clip_id = f'{gloss}_{signer}_{next_clip_number(gloss, signer):03d}'
    clip_dir = OUTPUT_DIR / 'pose' / clip_id
    clip_dir.mkdir(parents=True)
    for i, frame in enumerate(frames, 1):
        (clip_dir / f'image_{i:05d}_keypoints.json').write_text(json.dumps(frame))
    index_path = OUTPUT_DIR / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.is_file() else []
    entry = next((e for e in index if e['gloss'] == gloss), None)
    if entry is None:
        entry = {'gloss': gloss, 'instances': []}
        index.append(entry)
    entry['instances'].append({'video_id': clip_id, 'signer': signer, 'frame_start': 1,
                               'frame_end': len(frames), 'split': 'train'})
    index_path.write_text(json.dumps(index, indent=2))
    return clip_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gloss', required=True, help='Sign being recorded, e.g. book.')
    parser.add_argument('--signer', required=True, help='Your name, used in clip IDs.')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    gloss, signer = args.gloss.strip().lower(), args.signer.strip().lower()
    if not re.fullmatch(r'[a-z0-9-]+', gloss) or not re.fullmatch(r'[a-z0-9-]+', signer):
        parser.error('--gloss and --signer may only use letters, numbers and hyphens.')
    camera = pose_tracker = hand_tracker = None
    try:
        pose_tracker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_DIR / 'pose_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        hand_tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError('Could not open webcam.')
        print(f'Recording "{gloss}" as {signer}. R start | S save + next | D discard | Q quit')
        origin = time.monotonic()
        previous = -1
        frames = None
        saved = 0
        message = 'R start | S save + next | D discard | Q quit'
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError('Could not read webcam frame.')
            timestamp = max(previous+1, int((time.monotonic()-origin)*1000))
            previous = timestamp
            height, width = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            pose = pose_tracker.detect_for_video(image, timestamp)
            hands = hand_tracker.detect_for_video(image, timestamp)
            record = frame_json(pose, hands, width, height)
            if frames is not None:
                frames.append(record)
            for person in record['people']:
                for key in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d'):
                    for x, y, c in np.reshape(person[key], (-1, 3)):
                        if c > 0 and 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            preview = cv2.flip(frame, 1)
            status = f'RECORDING {len(frames)} frames' if frames is not None else 'IDLE'
            for i, line in enumerate([f'{gloss} ({signer}) - {saved} saved', status, message]):
                cv2.putText(preview, line, (10, 25+i*26), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.imshow('Record signs', preview)
            key = chr(cv2.waitKey(1) & 0xff).lower()
            if key == 'q':
                break
            if key == 'r' and frames is None:
                frames = []
                message = 'Recording: S saves + starts next, D discards'
            elif key == 's' and frames is not None:
                if any(f['people'] for f in frames):
                    clip_id = save_clip(gloss, signer, frames)
                    saved += 1
                    message = f'Saved {clip_id} ({len(frames)} frames); recording next'
                else:
                    message = 'Not saved: no body detected in that clip; recording next'
                print(message)
                frames = []
            elif key == 'd' and frames is not None:
                frames = None
                message = 'Discarded. R starts a new clip'
                print(message)
    except Exception as error:
        parser.exit(1, f'Recording failed: {error}\n')
    finally:
        for resource in (pose_tracker, hand_tracker):
            if resource is not None:
                resource.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
