"""Thin OpenCV runner for reusable inference sessions.

  # our pose models, picked by name (--list-models shows them)
  python camera_inference.py --model meeting100 --landmarks

  # the original WLASL-100 TGCN checkpoint (unverified preprocessing)
  python camera_inference.py --experimental-legacy --landmarks
"""
import argparse
import json
import time

try:
    import cv2
    import numpy as np
except ImportError as error:
    dependency = getattr(error, "name", None) or str(error)
    raise SystemExit(
        f"Missing or incompatible Python dependency '{dependency}'. Create "
        "the Python 3.12 environment and install "
        "backend/sign_inference/requirements-inference.txt."
    ) from error


# Connected joint pairs in the 55-point layout (13 body, then 21 left hand,
# then 21 right hand), drawn as lines by --landmark-style lines.
BODY_LINES = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),  # head, arms, torso
    (0, 9), (9, 11), (0, 10), (10, 12),                              # eyes and ears
]
HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]
SKELETON_LINES = (
    BODY_LINES
    + [(13 + a, 13 + b) for a, b in HAND_LINES]
    + [(34 + a, 34 + b) for a, b in HAND_LINES]
)


def draw_landmarks(frame, points, style):
    """Draw detected landmarks on a BGR frame as dots, or as a skeleton of lines."""
    height, width = frame.shape[:2]

    def visible(index):
        x, y = points[index]
        return np.isfinite([x, y]).all() and 0 <= x < width and 0 <= y < height

    if style == 'lines':
        for a, b in SKELETON_LINES:
            if visible(a) and visible(b):
                cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])),
                         (0, 255, 0), 2, cv2.LINE_AA)
    radius = 2 if style == 'lines' else 3
    for index in range(len(points)):
        if visible(index):
            cv2.circle(frame, tuple(map(int, points[index])), radius, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=None,
                        help='one of our pose models by name or path (e.g. meeting100); '
                             'omit to use the legacy TGCN checkpoint')
    parser.add_argument('--list-models', action='store_true', help='List available model names and exit.')
    parser.add_argument('--top-k', type=int, default=5, help='How many guesses to show (our models).')
    parser.add_argument('--hand-assignment', choices=['wrist', 'label'], default='wrist',
                        help="Our models: place each hand by nearest pose wrist (default) or MediaPipe's label.")
    parser.add_argument('--experimental-legacy', action='store_true',
                        help='Allow unverified affine preprocessing and candidate labels (TGCN only).')
    parser.add_argument('--landmarks', action='store_true', help='Draw optional landmarks.')
    parser.add_argument('--landmark-style', choices=['lines', 'dots'], default='lines',
                        help='How --landmarks draws them: a skeleton of lines (default) or dots.')
    parser.add_argument('--camera', type=int, default=0, help='OpenCV camera index (default: 0).')
    parser.add_argument('--mirror-input', action='store_true',
                        help='Flip each frame left-right before MediaPipe and the model see it '
                             '(by default only the on-screen preview is mirrored).')
    args = parser.parse_args()
    if args.list_models:
        from pose_models import list_models
        print('\n'.join(list_models()) or 'No models found.')
        return
    if args.model is None and not args.experimental_legacy:
        parser.error(
            'pass --model NAME to use one of our pose models, or --experimental-legacy '
            'to acknowledge and run the unverified TGCN checkpoint'
        )
    camera = None
    try:
        if args.model is not None:
            from pose_model_inference import PoseModelInference
            session_cm = PoseModelInference(args.model, hand_assignment=args.hand_assignment,
                                            top_k=args.top_k)
            m = session_cm.model
            print(f'Model {m.name}: {len(m.labels)} words, {m.num_frames} frames, '
                  f'trained val acc {m.trained_val_acc:.1%} ({m.path})')
            print('Words: ' + ', '.join(m.labels[:30]) + (' ...' if len(m.labels) > 30 else ''))
            title = f'{m.name} ({len(m.labels)} words)'
        else:
            from sign_inference import SignInference
            session_cm = SignInference(experimental_legacy=args.experimental_legacy)
            print('EXPERIMENTAL: preprocessing and checkpoint label ordering unverified.')
            title = 'Experimental WLASL100'
        with session_cm as session:
            print('R starts recording, S finishes and predicts, Q quits.')
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(
                    f'Could not open camera index {args.camera}. Close other camera '
                    'apps or select another device with --camera.'
                )
            origin = time.monotonic()
            previous = -1
            message = 'R record | S predict | Q quit'
            predictions = []
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError('Could not read webcam frame.')
                if args.mirror_input:
                    # MediaPipe, the model and the landmark overlay all get the mirrored frame.
                    frame = cv2.flip(frame, 1)
                timestamp = max(previous + 1, int((time.monotonic() - origin) * 1000))
                previous = timestamp
                try:
                    points = session.process_frame(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        timestamp,
                    )
                except Exception as error:
                    session.reset()
                    predictions = []
                    message = f'Frame error; recording reset: {error}'
                    print(message)
                    points = []
                if args.landmarks and len(points):
                    draw_landmarks(frame, points, args.landmark_style)
                # The preview always looks like a mirror; with --mirror-input the
                # frame is already flipped, so don't flip it back.
                preview = frame if args.mirror_input else cv2.flip(frame, 1)
                lines = [
                    (title if args.model else 'EXPERIMENTAL: preprocessing / labels unverified')
                    + (' | MIRRORED INPUT' if args.mirror_input else ''),
                    (
                        f'{"RECORDING" if session.recording else "IDLE"} '
                        f'{session.captured_frame_count} frames'
                    ),
                    message,
                    *predictions,
                ]
                for index, line in enumerate(lines):
                    cv2.putText(
                        preview,
                        line,
                        (10, 25 + index * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .55,
                        (0, 255, 255),
                        2,
                    )
                cv2.imshow(title, preview)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q')):
                    break
                try:
                    if key in (ord('r'), ord('R')):
                        session.start_sequence()
                        predictions = []
                        message = 'Recording: S finishes'
                    elif key in (ord('s'), ord('S')):
                        result = session.finish_and_predict()
                        print(json.dumps(result, allow_nan=False))
                        if args.model:
                            predictions = [
                                f'{index}. {p["gloss"]:<12} {p["confidence"]:6.1%}  '
                                + '|' * round(p['confidence'] * 30)
                                for index, p in enumerate(result['predictions'], 1)
                            ]
                            rate = result['detection_rate']
                            message = (
                                f'{result["usable_frame_count"]} frames | hands L {rate["left_hand"]:.0%} '
                                f'R {rate["right_hand"]:.0%} | margin {result["margin"]:.1%}'
                            )
                        else:
                            predictions = [
                                f'{index}. {prediction["gloss"]}: '
                                f'model score {prediction["score"]:.3f}'
                                for index, prediction in enumerate(result['predictions'], 1)
                            ]
                            message = (
                                f'Captured/prepared: {result["captured_frame_count"]}/'
                                f'{result["prepared_frame_count"]}'
                            )
                except Exception as error:
                    predictions = []
                    message = str(error)
                    print(f'Cannot infer: {error}')
    except Exception as error:
        parser.exit(1, f'Camera inference failed: {error}\n')
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
