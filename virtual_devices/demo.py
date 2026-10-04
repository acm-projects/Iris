"""
demo.py

Minimal end-to-end example:
  1. Reads frames from your real webcam, overlays a text label (a
     stand-in for "currently detected sign"), and streams the result
     out through the virtual camera.
  2. Plays a short audio clip out through the virtual microphone (a
     stand-in for TTS output of a recognized sign/phrase).

Run this script, then in Google Meet click the settings gear and, under
Video/Audio, pick the virtual devices it created (see SETUP.md for the
exact names on your OS). Meet will show your annotated webcam feed and
will "hear" the tone this script plays.
"""

import threading
import time

import cv2
import numpy as np

from virtual_camera import VirtualCamera
from virtual_microphone import VirtualMicrophone

WIDTH, HEIGHT, FPS = 1280, 720, 30

# Change this to match the virtual audio device you installed:
#   Windows -> "CABLE Input"
#   macOS   -> "BlackHole"   (matches to 'BlackHole 2ch')
MIC_DEVICE_HINT = "BlackHole"


def video_loop(label_text="Hello is this thing on?"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open the real webcam :( (index 0).")

    with VirtualCamera(width=WIDTH, height=HEIGHT, fps=FPS) as vcam:
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frame_bgr = cv2.resize(frame_bgr, (WIDTH, HEIGHT))

                # Stand-in for Iris's "signer-facing feedback" overlay -
                # replace with the real recognized-sign label later.
                cv2.putText(
                    frame_bgr, label_text, (30, HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )

                vcam.send_bgr_frame(frame_bgr)
        finally:
            cap.release()


def audio_demo():
    mic = VirtualMicrophone(device_name_hint=MIC_DEVICE_HINT, samplerate=48000)

    # Stand-in for TTS output, it's just A = 440hz
    # Swap this for the TTS API once we get one
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    time.sleep(2)  # let the video loop start up first
    mic.speak(tone)


if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_demo, daemon=True)
    audio_thread.start()

    video_loop()