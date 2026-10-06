"""Thin OpenCV runner for reusable SignInference sessions."""
import argparse
import json
import time
import cv2
import numpy as np
from sign_inference import SignInference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimental-legacy', action='store_true',
                        help='Allow unverified affine preprocessing and candidate labels.')
    parser.add_argument('--landmarks', action='store_true', help='Draw optional landmark points.')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    camera = None
    try:
        with SignInference(experimental_legacy=args.experimental_legacy) as session:
            print('EXPERIMENTAL: preprocessing and checkpoint label ordering unverified.')
            print('R starts recording, S finishes and predicts, Q quits.')
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError('Could not open webcam.')
            origin = time.monotonic()
            previous = -1
            message = 'R record | S predict | Q quit'
            predictions = []
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError('Could not read webcam frame.')
                timestamp = max(previous+1, int((time.monotonic()-origin)*1000))
                previous = timestamp
                try:
                    points = session.process_frame(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),timestamp)
                except Exception as error:
                    session.reset()
                    predictions = []
                    message = f'Frame error; recording reset: {error}'
                    print(message)
                    points = []
                if args.landmarks:
                    for x,y in points:
                        if np.isfinite([x,y]).all() and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(frame,(int(x),int(y)),3,(0,255,0),-1)
                preview = cv2.flip(frame,1)
                lines = ['EXPERIMENTAL: preprocessing / labels unverified',
                         f'{"RECORDING" if session.recording else "IDLE"} {session.captured_frame_count} frames',
                         message, *predictions]
                for i,line in enumerate(lines):
                    cv2.putText(preview,line,(10,25+i*26),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
                cv2.imshow('Experimental WLASL100',preview)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                try:
                    if key == ord('r'):
                        session.start_sequence()
                        predictions = []
                        message = 'Recording: S finishes'
                    elif key == ord('s'):
                        result = session.finish_and_predict()
                        print(json.dumps(result,allow_nan=False))
                        predictions = [f'{i}. {p["gloss"]}: model score {p["score"]:.3f}'
                                       for i,p in enumerate(result['predictions'],1)]
                        message = f'Captured/prepared: {result["captured_frame_count"]}/{result["prepared_frame_count"]}'
                except Exception as error:
                    predictions = []
                    message = str(error)
                    print(f'Cannot infer: {error}')
    except Exception as error:
        parser.exit(1,f'Camera inference failed: {error}\n')
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
