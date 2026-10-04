"""
virtual_microphone.py

Plays audio (e.g. TTS output for a recognized sign once we get there) out through a
system-level virtual audio cable, so Google Meet can pick it up as if
it were a real microphone. Like the camera, the driver itself
(VB-CABLE / BlackHole) must be installed first
- see SETUP.md. sounddevice just needs to know which output device to
write to.

Basic usage:

    mic = VirtualMicrophone(device_name_hint="CABLE Input")  # Windows example
    mic.speak(audio_array, samplerate=48000)
"""

import numpy as np
import sounddevice as sd


def list_devices():
    """Print all audio devices sounddevice can see, with their index.
    Run this first if find_device_index() can't find your virtual cable -
    the exact name varies by OS/driver version.
    """
    print(sd.query_devices())


def find_device_index(name_substring: str) -> int:
    """Find the index of the first *output* device whose name contains
    name_substring (case-insensitive)."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if name_substring.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return idx
    raise ValueError(
        f"No output device matching '{name_substring}' found.\n"
        f"Available devices:\n{devices}\n"
        f"Run virtual_microphone.list_devices() to see exact names."
    )


class VirtualMicrophone:
    def __init__(self, device_name_hint: str, samplerate: int = 48000):
        """
        device_name_hint : substring of the virtual cable's output-side
            device name, e.g. "CABLE Input" (Windows/VB-CABLE) or
            "BlackHole" (macOS).
        samplerate : default sample rate for speak() when none is given.
        """
        self.device_index = find_device_index(device_name_hint)
        self.samplerate = samplerate
        print(f"[VirtualMicrophone] writing to device #{self.device_index}: "
              f"{sd.query_devices(self.device_index)['name']}")
        print("    -> select the matching input device as the Microphone in Google Meet")

    def speak(self, audio: np.ndarray, samplerate: int = None, blocking: bool = True):
        """
        audio : float32 numpy array in [-1, 1], shape (n_samples,) for mono
            or (n_samples, n_channels).
        samplerate : overrides the default for this call (match your TTS
            engine's output rate).
        blocking : wait for playback to finish before returning. Set False
            if you want to keep processing video while audio plays.
        """
        sd.play(audio.astype(np.float32), samplerate or self.samplerate,
                 device=self.device_index)
        if blocking:
            sd.wait()

    def stop(self):
        sd.stop()
