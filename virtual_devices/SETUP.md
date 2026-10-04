# Setting up the virtual camera + microphone

`pyvirtualcam` and `sounddevice` cannot create virtual devices themselves -
they write into ones that already exist on the OS. This is OS-dependent, so
see below to install the drivers for your OS.

## Virtual camera

| OS | Install | Notes |
|---|---|---|
| Windows | Install [OBS Studio](https://obsproject.com/) | Ships a virtual camera driver pyvirtualcam can use directly. Just installing OBS is enough - you don't need to open it. |
| macOS | Install [OBS Studio](https://obsproject.com/) | Same as Windows; OBS's virtual cam works on macOS 10.13+ as of OBS version 26.1. |

## Virtual microphone

| OS | Install | Device name to use |
|---|---|---|
| Windows | Install [VB-CABLE](https://vb-audio.com/Cable/) | Python writes to **"CABLE Input"**; in Meet, pick **"CABLE Output"** as the microphone. |
| macOS | `brew install blackhole-2ch` | Python writes to **"BlackHole 2ch"**; in Meet, pick **"BlackHole 2ch"** as the microphone too (it's a single loopback pair, not two separate devices). |

If `find_device_index()` in `virtual_microphone.py` can't find your device, run:

```python
from virtual_microphone import list_devices
list_devices()
```

and copy the exact name (or a distinctive substring of it) into `device_name_hint`.

## Using them in Google Meet

1. Run:
```bash

python3 -m venv .venv

source venv/bin/activate

pip install -r requirements.txt
```
2. Run `demo.py` (or your real Cue pipeline) so the virtual devices are actively receiving frames/audio.
3. In Google Meet, click the **Settings gear** → **Video** tab → set Camera to the OBS.
4. Same menu → **Audio** tab → set Microphone to CABLE Output / BlackHole 2ch.
5. Meet now sees your processed video and generated audio exactly as if they came from a real webcam and mic.