// Translation helpers: saved settings, device detection, the engine's system
// check result, and the frame pump that streams video to the virtual camera.

/** Choices on the Translation page, saved per computer. */
export type TranslationSettings = {
  cameraId: string;
  microphoneId: string;
  virtualMicId: string;
  signLanguage: "ASL";
  resolution: "720p" | "540p";
  mirrorPreview: boolean;
  showCaptions: boolean;
  speakSigns: boolean;
  captionSize: "small" | "medium" | "large";
};

const settingsKey = "iris-translation-settings";

export const DEFAULT_SETTINGS: TranslationSettings = {
  cameraId: "",
  microphoneId: "",
  virtualMicId: "",
  signLanguage: "ASL",
  resolution: "720p",
  mirrorPreview: true,
  showCaptions: true,
  speakSigns: true,
  captionSize: "medium",
};

/** Saved settings merged over the defaults (so new options get defaults). */
export function loadSettings(): TranslationSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey) ?? "{}");
    return { ...DEFAULT_SETTINGS, ...saved };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

/** Remembers settings on this computer. */
export function saveSettings(settings: TranslationSettings) {
  try {
    localStorage.setItem(settingsKey, JSON.stringify(settings));
  } catch {
    /* storage unavailable: settings last until Iris restarts */
  }
}

/** Pixel size and frame rate for each resolution option. */
export const RESOLUTIONS = {
  "720p": { width: 1280, height: 720, fps: 24 },
  "540p": { width: 960, height: 540, fps: 24 },
} as const;

// Device names created by the drivers Iris uses.
export const isVirtualCamera = (label: string) =>
  /obs virtual camera|obs-camera|iris virtual/i.test(label);
export const isVirtualMic = (label: string) =>
  /blackhole|cable input|cable output|vb-audio|loopback/i.test(label);

/** What engine.py's `system_check` returns. */
export type EngineCheck = {
  python: string;
  platform: string;
  modules: Record<string, true | string>;
  missingPackages: string[];
  virtualCamera: {
    ok: boolean;
    device?: string;
    error?: string;
    inUse?: boolean;
  };
  virtualMic: { ok: boolean; device?: string; error?: string };
  speech: { ok: boolean; engine: string };
  signModel: { ok: boolean; status: string };
};

/** Engine events forwarded by Electron. */
export type EngineEvent =
  | {
      event: "status";
      camera: "running" | "stopped" | "error";
      device?: string;
      error?: string;
      framesOut?: number;
      engine?: string;
    }
  | { event: "sign"; text: string };

/** Strips Electron's "Error invoking remote method" prefix from engine errors. */
export function engineError(error: unknown, fallback: string) {
  const message = error instanceof Error ? error.message : "";
  return (
    message.replace(/^Error invoking remote method '[^']+': (Error: )?/, "") ||
    fallback
  );
}

// Chromium's MediaStreamTrackProcessor (not yet in TypeScript's DOM types):
// turns a camera track into a stream of VideoFrames.
type TrackProcessor = { readable: ReadableStream<VideoFrame> };
type TrackProcessorConstructor = new (init: {
  track: MediaStreamTrack;
}) => TrackProcessor;

/**
 * Draws the camera plus the caption overlay onto a canvas and sends JPEG
 * frames to the engine at a fixed rate. The same canvas is what Meet sees.
 *
 * Frames are read straight from the camera track (not from the on-screen
 * <video>), so streaming keeps going while Iris is in the background or the
 * preview is hidden. The <video> element is only a fallback.
 */
export class FramePump {
  private timer = 0;
  private busy = false;
  private canvas = document.createElement("canvas");
  private reader: ReadableStreamDefaultReader<VideoFrame> | null = null;
  private latest: VideoFrame | null = null;

  constructor(
    private video: HTMLVideoElement,
    private getCaption: () => {
      text: string;
      size: TranslationSettings["captionSize"];
      show: boolean;
    },
  ) {}

  /** Starts sending frames; pass the camera track so hidden windows keep streaming. */
  start(width: number, height: number, fps: number, track?: MediaStreamTrack) {
    this.stop();
    this.canvas.width = width;
    this.canvas.height = height;
    const Processor = (
      window as unknown as {
        MediaStreamTrackProcessor?: TrackProcessorConstructor;
      }
    ).MediaStreamTrackProcessor;
    if (track && Processor) void this.readTrack(new Processor({ track }));
    this.timer = window.setInterval(() => void this.tick(), 1000 / fps);
  }

  stop() {
    window.clearInterval(this.timer);
    this.timer = 0;
    void this.reader?.cancel().catch(() => undefined);
    this.reader = null;
    this.latest?.close();
    this.latest = null;
  }

  /** Keeps only the newest camera frame (older ones are closed to free memory). */
  private async readTrack(processor: TrackProcessor) {
    const reader = processor.readable.getReader();
    this.reader = reader;
    try {
      while (this.reader === reader) {
        const { value, done } = await reader.read();
        if (done || !value) break;
        if (this.reader !== reader) {
          value.close();
          break;
        }
        this.latest?.close();
        this.latest = value;
      }
    } catch {
      /* the track ended or the pump stopped */
    }
  }

  private async tick() {
    // Skip a frame rather than queue up if encoding is slower than the frame rate.
    if (this.busy) return;
    const frame = this.latest;
    const source = frame
      ? {
          image: frame as CanvasImageSource,
          width: frame.displayWidth,
          height: frame.displayHeight,
        }
      : this.video.readyState >= 2
        ? {
            image: this.video,
            width: this.video.videoWidth,
            height: this.video.videoHeight,
          }
        : null;
    if (!source) return;
    this.busy = true;
    try {
      const context = this.canvas.getContext("2d");
      if (!context) return;
      drawFrame(
        context,
        source,
        this.canvas.width,
        this.canvas.height,
        this.getCaption(),
      );
      const blob = await new Promise<Blob | null>((resolve) =>
        this.canvas.toBlob(resolve, "image/jpeg", 0.78),
      );
      if (blob)
        window.iris.engine.sendFrame(new Uint8Array(await blob.arrayBuffer()));
    } finally {
      this.busy = false;
    }
  }
}

/** Fills the frame with the camera image (cropped to fit) and adds the caption bar. */
export function drawFrame(
  context: CanvasRenderingContext2D,
  source: { image: CanvasImageSource; width: number; height: number },
  width: number,
  height: number,
  caption: {
    text: string;
    size: TranslationSettings["captionSize"];
    show: boolean;
  },
) {
  const scale = Math.max(width / source.width, height / source.height);
  const drawWidth = source.width * scale;
  const drawHeight = source.height * scale;
  context.drawImage(
    source.image,
    (width - drawWidth) / 2,
    (height - drawHeight) / 2,
    drawWidth,
    drawHeight,
  );
  if (!caption.show || !caption.text) return;

  const fontSize =
    { small: 0.034, medium: 0.044, large: 0.056 }[caption.size] * height;
  context.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
  const text =
    caption.text.length > 90 ? `${caption.text.slice(0, 88)}…` : caption.text;
  const textWidth = context.measureText(text).width;
  const padX = fontSize * 0.8;
  const boxHeight = fontSize * 1.9;
  const boxWidth = Math.min(textWidth + padX * 2, width * 0.92);
  const x = (width - boxWidth) / 2;
  const y = height - boxHeight - height * 0.05;
  // Deep Navy caption bar with a gold accent, matching the Iris palette.
  context.fillStyle = "rgba(0, 34, 78, 0.82)";
  context.beginPath();
  context.roundRect(x, y, boxWidth, boxHeight, fontSize * 0.5);
  context.fill();
  context.fillStyle = "#f6bf41";
  context.fillRect(x + fontSize * 0.5, y + boxHeight - 4, fontSize * 1.4, 3);
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, width / 2, y + boxHeight / 2, boxWidth - padX * 2);
}
