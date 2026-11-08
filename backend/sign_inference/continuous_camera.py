"""Live webcam runner for continuous sign recognition (rolling landmark buffer).

  python continuous_camera.py --model meeting100 --landmarks

Just sign; stabilized words submit automatically after a pause with hands out of view.
C clears the phrase, Q quits. Recognition continues during translation.
"""
import argparse
import time

import cv2
from gloss_translation import PhraseTranslator, PauseSubmitter

from camera_inference import draw_landmarks
from continuous_inference import ContinuousInference, SentenceBuilder, WINDOW_FRAMES
from pose_model_inference import PoseModelInference


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='meeting100', help='pose model name or path (default: meeting100)')
    parser.add_argument('--top-k', type=int, default=3, help='How many live guesses to show.')
    parser.add_argument('--hand-assignment', choices=['wrist', 'label'], default='wrist')
    parser.add_argument('--predict-every', type=int, default=5, help='Classify the window every N frames.')
    parser.add_argument('--min-conf', type=float, default=0.30,
                        help='Minimum averaged confidence for accepting a gloss (default: 0.30).')
    parser.add_argument('--stable', type=int, default=2,
                        help='Consecutive winning windows required (default: 2).')
    parser.add_argument('--smooth', type=int, default=3,
                        help='Windows to average for acceptance (default: 3).')
    parser.add_argument('--landmarks', action='store_true', help='Draw landmarks.')
    parser.add_argument('--landmark-style', choices=['lines', 'dots'], default='lines')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--submit-pause', type=float, default=2.0,
                        help='Seconds without detected signing before automatic translation (default: 2).')
    parser.add_argument('--manual-submit', action='store_true',
                        help='Disable automatic translation; submit only with Enter.')
    parser.add_argument('--mirror-input', action='store_true',
                        help='Flip frames before MediaPipe and the model see them (default: preview only).')
    args = parser.parse_args()
    try:
        pause_submitter = PauseSubmitter(args.submit_pause)
        builder = SentenceBuilder(min_conf=args.min_conf, stable=args.stable, smooth=args.smooth)
    except ValueError as error:
        parser.error(str(error))

    camera = None
    translator = PhraseTranslator()
    submit_mode = 'Enter submits' if args.manual_submit else f'Auto-submit after {args.submit_pause:g}s rest'
    translation = submit_mode + ' | C clears | Q quits'
    try:
        with PoseModelInference(args.model, hand_assignment=args.hand_assignment, top_k=args.top_k) as session:
            live = ContinuousInference(session, predict_every=args.predict_every, builder=builder)
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(f'Could not open camera index {args.camera}.')
            print(f'{session.model.name}: {len(session.model.labels)} words, '
                  f'{WINDOW_FRAMES}-frame rolling window. {submit_mode}; C clears, Q quits.')
            print(f'Acceptance: confidence >= {builder.min_conf:.0%}, '
                  f'{builder.stable} consecutive wins, smoothing {builder.smooth} windows.')
            origin, previous = time.monotonic(), -1
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError('Could not read webcam frame.')
                if args.mirror_input:
                    frame = cv2.flip(frame, 1)
                previous = max(previous + 1, int((time.monotonic() - origin) * 1000))
                update = live.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), previous)
                completed = translator.poll()
                if completed:
                    translation = completed
                    print(completed)
                if update.committed:
                    print(f'+ {update.committed}  ->  {live.builder.text}')
                if not args.manual_submit:
                    submitted = pause_submitter.update(
                        signing=update.signing, committed=update.committed is not None,
                        now=time.monotonic(), words=live.builder.words, translator=translator,
                    )
                    if submitted:
                        translation = submitted
                        print('Submitting gloss phrase: ' + live.builder.text)
                if args.landmarks:
                    draw_landmarks(frame, update.points, args.landmark_style)
                preview = frame if args.mirror_input else cv2.flip(frame, 1)

                status = 'SIGNING' if update.signing else 'resting'
                lines = [f'{status} | buffer {update.buffered}/{WINDOW_FRAMES}', translation,
                         live.builder.status,
                         *(f'{g["gloss"]:<12} {g["confidence"]:5.0%}' for g in update.guesses)]
                for i, line in enumerate(lines):
                    cv2.putText(preview, line, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (0, 255, 255), 2)
                cv2.putText(preview, live.builder.text or '...', (10, preview.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
                cv2.imshow('Continuous signing', preview)

                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q')):
                    break
                if key in (ord('c'), ord('C')):
                    live.reset()
                elif key in (10, 13):
                    pause_submitter.mark_submitted()
                    translation = translator.submit(live.builder.words)
    except Exception as error:
        parser.exit(1, f'Continuous inference failed: {error}\n')
    finally:
        translator.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
