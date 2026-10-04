"""
test_blackhole.py

Standalone check that Python can talk to BlackHole 2ch - no webcam, no
Google Meet, no demo.py needed. Confirms the "virtual microphone" half
of the pipeline in isolation.

How to use:
  1. Open QuickTime Player -> File -> New Audio Recording.
  2. Click the small dropdown arrow next to the record button and pick
     "BlackHole 2ch" as the input.
  3. Click record, then run this script.
  4. Watch QuickTime's level meter jump while the 440Hz tone plays.
"""

import numpy as np

from virtual_microphone import VirtualMicrophone, list_devices

if __name__ == "__main__":
    print("All audio devices sounddevice can see:")
    list_devices()
    print()

    mic = VirtualMicrophone(device_name_hint="BlackHole", samplerate=48000)

    t = np.linspace(0, 2.0, 48000 * 2, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Playing a 2-second 440Hz tone through BlackHole 2ch now...")
    mic.speak(tone, blocking=True)
    print("Done - check QuickTime's level meter/waveform for the tone.")