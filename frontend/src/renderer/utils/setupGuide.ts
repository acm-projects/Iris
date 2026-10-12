// Step-by-step setup instructions for the Translation page, per operating
// system. Google Meet needs a virtual camera and a virtual microphone, and
// these come from different free tools on macOS, Windows and Linux.

/** The operating systems Iris has instructions for. */
export type OsKind = "mac" | "windows" | "linux";

/** This computer's operating system (from Electron's process.platform). */
export const osKind: OsKind =
  window.iris?.platform === "darwin"
    ? "mac"
    : window.iris?.platform === "win32"
      ? "windows"
      : "linux";

/** A terminal command shown with a Copy button. */
export type GuideCommand = { label?: string; command: string };

/**
 * How to fix one setup step. Kept deliberately short: a few brief steps, a
 * download button, the one command most people need, and alternatives folded
 * under "More options".
 */
export type Guide = {
  steps: string[];
  download?: { label: string; url: string };
  commands?: GuideCommand[];
  more?: GuideCommand[];
};

/** What Meet calls Iris's devices on this computer. */
export const MEET_DEVICE_NAMES: Record<
  OsKind,
  { camera: string; microphone: string }
> = {
  mac: { camera: "OBS Virtual Camera", microphone: "BlackHole 2ch" },
  windows: { camera: "OBS Virtual Camera", microphone: "CABLE Output" },
  linux: { camera: "Iris Virtual Camera", microphone: "Iris Microphone" },
};

const OBS_DOWNLOAD = {
  label: "Get OBS",
  url: "https://obsproject.com/download",
};

// Linux: v4l2loopback provides the virtual camera. The options give it a
// friendly name in Meet and make Chrome treat it as a camera.
const V4L2_OPTIONS =
  'devices=1 exclusive_caps=1 card_label="Iris Virtual Camera"';

/** Virtual camera: lets Meet show Iris's captioned video. */
export const VIRTUAL_CAMERA_GUIDE: Record<OsKind, Guide> = {
  mac: {
    download: OBS_DOWNLOAD,
    steps: [
      "Install OBS (free).",
      "Open it and click Start Virtual Camera.",
      "Allow it if macOS asks, then quit OBS.",
    ],
  },
  windows: {
    download: OBS_DOWNLOAD,
    steps: ["Install OBS (free).", "No need to open it."],
  },
  linux: {
    steps: ["Install v4l2loopback.", "Turn it on."],
    commands: [
      { command: "sudo apt install v4l2loopback-dkms" },
      { command: `sudo modprobe v4l2loopback ${V4L2_OPTIONS}` },
    ],
    more: [
      { label: "Fedora", command: "sudo dnf install v4l2loopback" },
      { label: "Arch", command: "sudo pacman -S v4l2loopback-dkms" },
      {
        label: "Keep it on after a restart",
        command: "echo v4l2loopback | sudo tee /etc/modules-load.d/iris.conf",
      },
      {
        label: "…with the same options",
        command: `echo 'options v4l2loopback ${V4L2_OPTIONS}' | sudo tee /etc/modprobe.d/iris.conf`,
      },
    ],
  },
};

/** Virtual microphone: lets Meet hear Iris's voice. */
export const VIRTUAL_MIC_GUIDE: Record<OsKind, Guide> = {
  mac: {
    download: {
      label: "Get BlackHole",
      url: "https://existential.audio/blackhole/download/",
    },
    steps: ["Install BlackHole 2ch (free)."],
    more: [{ label: "With Homebrew", command: "brew install blackhole-2ch" }],
  },
  windows: {
    download: { label: "Get VB-CABLE", url: "https://vb-audio.com/Cable/" },
    steps: [
      "Download and unzip it.",
      "Run VBCABLE_Setup_x64 as administrator.",
      "Restart Windows.",
    ],
  },
  linux: {
    // The "Create Iris Microphone" button does the work.
    steps: ["Nothing to download."],
    more: [
      {
        label: "If it fails, install the sound tools",
        command: "sudo apt install pulseaudio-utils",
      },
    ],
  },
};

/** Text-to-speech voice (only Linux needs an install). */
export const VOICE_GUIDE: Record<OsKind, Guide> = {
  mac: { steps: ["Restart Iris."] },
  windows: { steps: ["Restart Iris."] },
  linux: {
    steps: ["Install espeak-ng."],
    commands: [{ command: "sudo apt install espeak-ng" }],
    more: [
      { label: "Fedora", command: "sudo dnf install espeak-ng" },
      { label: "Arch", command: "sudo pacman -S espeak-ng" },
    ],
  },
};
