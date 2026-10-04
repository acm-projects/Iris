"""
virtual_camera.py

Wraps pyvirtualcam so the rest of the pipeline only ever deals with plain
RGB numpy frames. pyvirtualcam doesn't create a camera device itself -
it feeds frames into a virtual camera driver that must already be
installed on the OS. See SETUP.md for the one-time driver install.

Basic usage:

    with VirtualCamera(width=1280, height=720, fps=30) as vcam:
        while True:
            frame = get_next_rgb_frame()   # HxWx3 uint8, RGB order
            vcam.send_frame(frame)
"""

import numpy as np
import pyvirtualcam


class VirtualCamera:
    def __init__(self, width=1280, height=720, fps=30, device=None, backend=None):
        """
        width, height, fps : stream resolution/framerate. Must match what
            you pass to send_frame().
        device : optional, force a specific virtual camera device name/path.
        Leave None to use the first
            one pyvirtualcam finds.
        backend : optional, force a specific backend ("obs", "unitycapture",
            "v4l2loopback", ...). Leave None to auto-detect.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self._device = device
        self._backend = backend
        self._cam = None

    def __enter__(self):
        kwargs = {"width": self.width, "height": self.height, "fps": self.fps}
        if self._device:
            kwargs["device"] = self._device
        if self._backend:
            kwargs["backend"] = self._backend

        self._cam = pyvirtualcam.Camera(**kwargs)
        self._cam.__enter__()
        print(f"[VirtualCamera] streaming to virtual device: {self._cam.device}")
        print("    -> select this as the Camera in Google Meet's settings")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cam is not None:
            self._cam.__exit__(exc_type, exc_val, exc_tb)

    def send_frame(self, frame_rgb: np.ndarray):
        """Send one HxWx3 uint8 RGB frame. Blocks briefly to pace output at self.fps."""
        h, w = frame_rgb.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"frame is {w}x{h}, camera expects {self.width}x{self.height}"
            )
        self._cam.send(frame_rgb)
        self._cam.sleep_until_next_frame()

    def send_bgr_frame(self, frame_bgr: np.ndarray):
        """Convenience for OpenCV users (cv2 reads frames as BGR)."""
        import cv2
        self.send_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
