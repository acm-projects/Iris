// Translation page: live camera preview with captions, device + translation
// settings, a system check for everything Google Meet needs, Type to speak,
// and Start/Stop translation (streams to the virtual camera and microphone).
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  FramePump,
  RESOLUTIONS,
  engineError,
  isVirtualCamera,
  isVirtualMic,
  loadSettings,
  saveSettings,
  type EngineCheck,
  type EngineEvent,
  type TranslationSettings,
} from "../../utils/translation";
import "../css/translation.css";

/** Whether video is currently being sent to the virtual camera. */
type LiveState = "off" | "starting" | "live";
/** Result of one system-check row (drives its icon and colour). */
type CheckState = "ok" | "warn" | "fail" | "pending" | "info";
/** Devices the OS reports, grouped for the settings dropdowns. */
type Devices = {
  cameras: MediaDeviceInfo[];
  mics: MediaDeviceInfo[];
  outputs: MediaDeviceInfo[];
  // True when the OS lists a virtual camera (Iris's output, not offered as an input).
  virtualCamera: boolean;
};

/** Tabs in the right-hand card. */
const TABS = [
  { id: "setup", label: "Setup" },
  { id: "settings", label: "Settings" },
  { id: "meet", label: "Meet guide" },
] as const;
type TabId = (typeof TABS)[number]["id"];

// Short phrases a signer can send with one click.
const QUICK_PHRASES = [
  "Hello everyone!",
  "Could you repeat that?",
  "One moment, please.",
  "Thank you!",
];
// Commands that install the engine's Python packages into backend/.venv.
// Windows uses the `py` launcher, a Scripts\ folder and `;` (PowerShell 5 has no &&).
const INSTALL_COMMAND =
  window.iris?.platform === "win32"
    ? "cd backend; py -m venv .venv; .venv\\Scripts\\pip install -r requirements.txt"
    : "cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt";
// Where users unblock the camera/microphone if they said no.
const PRIVACY_SETTINGS =
  window.iris?.platform === "win32"
    ? "Windows Settings → Privacy & security"
    : "System Settings → Privacy & Security";

/** Names of the small line icons used on this page. */
type IconName =
  | "camera"
  | "mic"
  | "volume"
  | "cpu"
  | "video"
  | "headphones"
  | "message"
  | "check"
  | "info";

/** Which icon goes with which system-check row. */
const CHECK_ICONS: Record<string, IconName> = {
  camera: "camera",
  mic: "mic",
  engine: "cpu",
  vcam: "video",
  vmic: "headphones",
  voice: "volume",
  model: "info",
};

/** Short status word shown under each setup tile. */
const stepLabel = (state: CheckState) =>
  state === "ok" ? "Ready" : state === "pending" ? "Waiting" : "Fix needed";

/**
 * Translation tab. Owns the real camera/microphone preview, talks to the
 * Python engine (via Electron) for the virtual camera, virtual microphone and
 * text-to-speech, and shows a system check of everything Meet needs.
 */
export default function TranslationPage() {
  const [settings, setSettings] = useState<TranslationSettings>(loadSettings);
  const [devices, setDevices] = useState<Devices>({
    cameras: [],
    mics: [],
    outputs: [],
    virtualCamera: false,
  });
  const [permission, setPermission] = useState({
    camera: "unknown",
    microphone: "unknown",
  });
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [level, setLevel] = useState(0);
  const [heardVoice, setHeardVoice] = useState(false);
  const [engineCheck, setEngineCheck] = useState<EngineCheck | null>(null);
  const [engineProblem, setEngineProblem] = useState("");
  const [checking, setChecking] = useState(false);
  const [live, setLive] = useState<LiveState>("off");
  const [liveDevice, setLiveDevice] = useState("");
  const [liveError, setLiveError] = useState("");
  const [caption, setCaption] = useState("");
  // Which tab of the right-hand card is open, and which setup tile is selected
  // (null = show the first problem, "" = none).
  const [tab, setTab] = useState<TabId>("setup");
  const [openRow, setOpenRow] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pumpRef = useRef<FramePump | null>(null);
  // Refs give callbacks (frame pump, engine events) the latest values.
  const settingsRef = useRef(settings);
  const captionRef = useRef(caption);
  const devicesRef = useRef(devices);
  const streamRef = useRef<MediaStream | null>(null);
  settingsRef.current = settings;
  captionRef.current = caption;
  devicesRef.current = devices;
  streamRef.current = stream;

  /** Updates one or more settings and remembers them. */
  const update = (patch: Partial<TranslationSettings>) =>
    setSettings((current) => {
      const next = { ...current, ...patch };
      saveSettings(next);
      return next;
    });

  // --- Devices and permission ---------------------------------------------

  /** Re-reads the list of cameras, microphones and audio outputs. */
  const refreshDevices = useCallback(async () => {
    const all = await navigator.mediaDevices.enumerateDevices();
    setDevices({
      // The virtual camera is Iris's own output, so never offer it as an input.
      cameras: all.filter(
        (d) => d.kind === "videoinput" && !isVirtualCamera(d.label),
      ),
      mics: all.filter(
        (d) => d.kind === "audioinput" && !isVirtualMic(d.label),
      ),
      outputs: all.filter((d) => d.kind === "audiooutput"),
      virtualCamera: all.some(
        (d) => d.kind === "videoinput" && isVirtualCamera(d.label),
      ),
    });
    return all;
  }, []);

  /** Reads the macOS camera/microphone permission state. */
  const refreshPermission = useCallback(async () => {
    const status = await window.iris.media?.status?.().catch(() => null);
    if (status) setPermission(status);
    return status;
  }, []);

  /** Shows the macOS permission prompts, then starts the preview. */
  const allowAccess = async () => {
    await window.iris.media?.request?.("camera").catch(() => false);
    await window.iris.media?.request?.("microphone").catch(() => false);
    await refreshPermission();
    await openPreview();
  };

  // --- Preview (camera + microphone level) --------------------------------

  /** Opens the chosen camera + microphone for the preview (and the outgoing video). */
  const openPreview = useCallback(async () => {
    setPreviewError("");
    try {
      const { cameraId, microphoneId } = settingsRef.current;
      const next = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: cameraId ? { exact: cameraId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: { deviceId: microphoneId ? { exact: microphoneId } : undefined },
      });
      setStream((previous) => {
        previous?.getTracks().forEach((track) => track.stop());
        return next;
      });
      const all = await refreshDevices();
      // First run: remember the devices the OS picked, and a virtual mic if there is one.
      const current = settingsRef.current;
      const patch: Partial<TranslationSettings> = {};
      if (!current.cameraId)
        patch.cameraId = next.getVideoTracks()[0]?.getSettings().deviceId ?? "";
      if (!current.microphoneId)
        patch.microphoneId =
          next.getAudioTracks()[0]?.getSettings().deviceId ?? "";
      if (!current.virtualMicId)
        patch.virtualMicId =
          all.find((d) => d.kind === "audiooutput" && isVirtualMic(d.label))
            ?.deviceId ?? "";
      if (Object.keys(patch).length) update(patch);
    } catch (error) {
      setStream(null);
      setPreviewError(
        error instanceof DOMException && error.name === "NotAllowedError"
          ? `Iris doesn't have permission to use your camera and microphone. Allow it in ${PRIVACY_SETTINGS}.`
          : error instanceof DOMException &&
              error.name === "OverconstrainedError"
            ? "That camera or microphone is no longer connected. Pick another one."
            : "Your camera or microphone couldn't be opened. Close other apps using it and try again.",
      );
    }
    void refreshPermission();
  }, [refreshDevices, refreshPermission]);

  // Open the preview on arrival, reopen when the chosen camera/mic changes,
  // and release the camera when leaving the page.
  useEffect(() => {
    // Skip if the open stream already uses the chosen devices (e.g. right
    // after the first launch saved the OS defaults).
    const current = streamRef.current;
    const usingCamera = current?.getVideoTracks()[0]?.getSettings().deviceId;
    const usingMic = current?.getAudioTracks()[0]?.getSettings().deviceId;
    if (
      current &&
      usingCamera === settings.cameraId &&
      usingMic === settings.microphoneId
    )
      return;
    void openPreview();
  }, [openPreview, settings.cameraId, settings.microphoneId]);
  // Release the camera/mic when the stream is replaced or the page closes.
  useEffect(
    () => () => stream?.getTracks().forEach((track) => track.stop()),
    [stream],
  );
  // Keep the device lists current when something is plugged in or removed.
  useEffect(() => {
    navigator.mediaDevices.addEventListener("devicechange", refreshDevices);
    return () =>
      navigator.mediaDevices.removeEventListener(
        "devicechange",
        refreshDevices,
      );
  }, [refreshDevices]);

  // Attach the stream to the <video> element.
  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = stream;
  }, [stream]);

  // Microphone level meter (0–1), also used by the system check.
  useEffect(() => {
    const track = stream?.getAudioTracks()[0];
    if (!stream || !track) return;
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(new MediaStream([track])).connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    let frame = 0;
    const read = () => {
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(
        samples.reduce((sum, v) => sum + v * v, 0) / samples.length,
      );
      const value = Math.min(1, rms * 6);
      setLevel(value);
      if (value > 0.18) setHeardVoice(true);
      frame = requestAnimationFrame(read);
    };
    read();
    return () => {
      cancelAnimationFrame(frame);
      void context.close();
    };
  }, [stream]);

  // --- Translation engine ---------------------------------------------------

  /** Asks the engine to check Python packages, virtual devices and speech. */
  const runEngineCheck = useCallback(async () => {
    setChecking(true);
    setEngineProblem("");
    try {
      setEngineCheck(
        await window.iris.engine.request<EngineCheck>("system_check"),
      );
    } catch (error) {
      setEngineCheck(null);
      setEngineProblem(
        typeof window.iris.engine?.request !== "function"
          ? "Restart Iris to finish updating, then run the check again."
          : engineError(error, "The translation engine could not start."),
      );
    } finally {
      setChecking(false);
      void refreshDevices();
    }
  }, [refreshDevices]);

  // Run the system check when the page opens.
  useEffect(() => {
    void runEngineCheck();
  }, [runEngineCheck]);

  /** Speaks text into the meeting through the virtual microphone. */
  const speak = useCallback(async (text: string) => {
    const label = devicesRef.current.outputs.find(
      (d) => d.deviceId === settingsRef.current.virtualMicId,
    )?.label;
    setCaption(text);
    await window.iris.engine.request("speak", { text, device: label });
  }, []);

  // Engine events: virtual camera status and recognised signs.
  useEffect(() => {
    if (typeof window.iris.engine?.onEvent !== "function") return;
    return window.iris.engine.onEvent((raw) => {
      const message = raw as EngineEvent;
      if (message.event === "sign") {
        setCaption(message.text);
        if (settingsRef.current.speakSigns)
          void speak(message.text).catch(() => undefined);
      } else if (message.event === "status") {
        if (message.camera === "running") setLiveDevice(message.device ?? "");
        if (message.camera === "error")
          setLiveError(message.error ?? "The virtual camera stopped.");
        if (message.camera === "stopped" || message.camera === "error") {
          pumpRef.current?.stop();
          setLive("off");
        }
      }
    });
  }, [speak]);

  /** Starts sending video (with captions) to the virtual camera. */
  const startLive = async () => {
    if (!videoRef.current || !stream) return;
    setLive("starting");
    setLiveError("");
    const size = RESOLUTIONS[settings.resolution];
    try {
      const result = await window.iris.engine.request<{ device: string }>(
        "start_virtual_camera",
        size,
      );
      setLiveDevice(result.device);
      pumpRef.current ??= new FramePump(videoRef.current, () => ({
        text: captionRef.current,
        size: settingsRef.current.captionSize,
        show: settingsRef.current.showCaptions,
      }));
      pumpRef.current.start(
        size.width,
        size.height,
        size.fps,
        stream.getVideoTracks()[0],
      );
      setLive("live");
    } catch (error) {
      setLive("off");
      setLiveError(engineError(error, "The virtual camera could not start."));
    }
  };

  /** Stops sending video to Meet (the virtual camera shows nothing afterwards). */
  const stopLive = async () => {
    pumpRef.current?.stop();
    setLive("off");
    await window.iris.engine
      .request("stop_virtual_camera")
      .catch(() => undefined);
  };

  // If the camera is switched while live, keep streaming from the new camera.
  useEffect(() => {
    if (live !== "live" || !stream || !pumpRef.current) return;
    const size = RESOLUTIONS[settingsRef.current.resolution];
    pumpRef.current.start(
      size.width,
      size.height,
      size.fps,
      stream.getVideoTracks()[0],
    );
    // Only re-run when the stream itself changes (not when going live/off).
  }, [stream]);

  // Leaving the page stops streaming to Meet.
  useEffect(
    () => () => {
      pumpRef.current?.stop();
      void window.iris.engine
        ?.request?.("stop_virtual_camera")
        .catch(() => undefined);
    },
    [],
  );

  // --- System check rows ----------------------------------------------------

  const videoTrack = stream?.getVideoTracks()[0];
  const audioTrack = stream?.getAudioTracks()[0];
  const hasVirtualCamera =
    Boolean(engineCheck?.virtualCamera.ok) ||
    live === "live" ||
    devices.virtualCamera;
  const hasVirtualMic =
    Boolean(engineCheck?.virtualMic.ok) ||
    devices.outputs.some((d) => isVirtualMic(d.label));
  const isMac = window.iris.platform === "darwin";

  const checks = useMemo(() => {
    const rows: {
      id: string;
      title: string;
      state: CheckState;
      detail: string;
      action?: { label: string; run: () => void };
      command?: string;
    }[] = [];
    rows.push(
      videoTrack?.readyState === "live"
        ? {
            id: "camera",
            title: "Camera",
            state: "ok",
            detail: `${videoTrack.label} · ${videoTrack.getSettings().width}×${videoTrack.getSettings().height}`,
          }
        : {
            id: "camera",
            title: "Camera",
            state: permission.camera === "denied" ? "fail" : "warn",
            detail:
              permission.camera === "denied"
                ? `Blocked. Allow Iris in ${PRIVACY_SETTINGS} → Camera.`
                : previewError || "Waiting for camera access.",
            action: { label: "Allow access", run: () => void allowAccess() },
          },
    );
    rows.push(
      audioTrack?.readyState === "live"
        ? {
            id: "mic",
            title: "Microphone",
            state: heardVoice ? "ok" : "pending",
            detail: heardVoice
              ? `${audioTrack.label} · hearing you`
              : "Say something to test your microphone.",
          }
        : {
            id: "mic",
            title: "Microphone",
            state: permission.microphone === "denied" ? "fail" : "warn",
            detail:
              permission.microphone === "denied"
                ? `Blocked. Allow Iris in ${PRIVACY_SETTINGS} → Microphone.`
                : "Waiting for microphone access.",
            action: { label: "Allow access", run: () => void allowAccess() },
          },
    );
    rows.push(
      engineCheck && engineCheck.missingPackages.length === 0
        ? {
            id: "engine",
            title: "Translation engine",
            state: "ok",
            detail: `Python ${engineCheck.python} · all packages installed`,
          }
        : {
            id: "engine",
            title: "Translation engine",
            state: checking ? "pending" : "fail",
            detail: checking
              ? "Checking…"
              : engineCheck
                ? `Missing Python packages: ${engineCheck.missingPackages.join(", ")}. Run this in the project folder, then check again:`
                : engineProblem || "Not started yet.",
            command: engineCheck?.missingPackages.length
              ? INSTALL_COMMAND
              : undefined,
          },
    );
    rows.push(
      hasVirtualCamera
        ? {
            id: "vcam",
            title: "Virtual camera",
            state: "ok",
            detail:
              engineCheck?.virtualCamera.device ||
              "OBS Virtual Camera is installed",
          }
        : {
            id: "vcam",
            title: "Virtual camera",
            state: "fail",
            // On macOS the OBS camera is a system extension that OBS installs the
            // first time "Start Virtual Camera" is clicked.
            detail: isMac
              ? "Install OBS Studio (free), open it once, click Start Virtual Camera, then allow the extension in System Settings → General → Login Items & Extensions. You can quit OBS afterwards."
              : "Install OBS Studio (free). Iris uses its virtual camera driver; you don't need to open OBS.",
            action: {
              label: "Get OBS",
              run: () =>
                void window.iris.auth.openExternal(
                  "https://obsproject.com/download",
                ),
            },
          },
    );
    rows.push(
      hasVirtualMic
        ? {
            id: "vmic",
            title: "Virtual microphone",
            state: "ok",
            detail:
              engineCheck?.virtualMic.device || "Loopback audio driver found",
            action:
              engineCheck?.missingPackages.length === 0
                ? { label: "Play test sound", run: () => void playTestSound() }
                : undefined,
          }
        : {
            id: "vmic",
            title: "Virtual microphone",
            state: "fail",
            detail: isMac
              ? "Install BlackHole 2ch (free) so Meet can hear Iris's voice."
              : "Install VB-CABLE (free) so Meet can hear Iris's voice, then restart Windows.",
            action: {
              label: isMac ? "Get BlackHole" : "Get VB-CABLE",
              run: () =>
                void window.iris.auth.openExternal(
                  isMac
                    ? "https://existential.audio/blackhole/"
                    : "https://vb-audio.com/Cable/",
                ),
            },
          },
    );
    rows.push(
      engineCheck?.speech.ok
        ? {
            id: "voice",
            title: "Voice",
            state: "ok",
            detail: `${engineCheck.speech.engine} text-to-speech`,
          }
        : {
            id: "voice",
            title: "Voice",
            state: checking ? "pending" : "warn",
            detail: "Text-to-speech will be checked with the engine.",
          },
    );
    rows.push({
      id: "model",
      title: "Sign recognition",
      state: "info",
      detail: "Not connected yet. Captions and Type to speak work today.",
    });
    return rows;
  }, [
    videoTrack,
    audioTrack,
    permission,
    previewError,
    heardVoice,
    engineCheck,
    engineProblem,
    checking,
    hasVirtualCamera,
    hasVirtualMic,
    isMac,
  ]);

  // Items that must be fixed before going live (failures and warnings).
  const blocking = checks.filter(
    (row) => row.state === "fail" || row.state === "warn",
  ).length;
  const ready = blocking === 0 && Boolean(stream);

  // What the Setup tab shows: one tile per step, a progress ring, and the
  // selected tile's details (by default the first problem).
  const stepRows = checks.filter((row) => row.state !== "info");
  const noteRow = checks.find((row) => row.state === "info");
  const okCount = stepRows.filter((row) => row.state === "ok").length;
  const ringLength = 2 * Math.PI * 22;
  const activeId = openRow || undefined;
  const activeRow = stepRows.find((row) => row.id === activeId);

  /** Plays a chime into the virtual microphone so the user can hear it in Meet. */
  const [testNote, setTestNote] = useState("");
  async function playTestSound() {
    setTestNote("Playing a test sound into the virtual microphone…");
    try {
      const label = devices.outputs.find(
        (d) => d.deviceId === settings.virtualMicId,
      )?.label;
      const result = await window.iris.engine.request<{ device: string }>(
        "test_tone",
        { device: label },
      );
      setTestNote(
        `Played into ${result.device}. In Meet, the microphone bar should have moved.`,
      );
    } catch (error) {
      setTestNote(engineError(error, "Could not play the test sound."));
    }
  }

  return (
    <section className="translation-page">
      {/* Banner: title, readiness and Start/Stop */}
      <header className="tr-header">
        <span aria-hidden="true" className="tr-leaf one" />
        <span aria-hidden="true" className="tr-leaf two" />
        <span aria-hidden="true" className="tr-leaf three" />
        <div className="tr-header-text">
          <p>Translation</p>
          <h1>Live translation</h1>
          <span className="tr-sub">
            Sign and type in Iris. Meet sees your video and hears Iris.
          </span>
        </div>
        <div className="tr-header-actions">
          <span
            className={`tr-pill ${live === "live" ? "live" : ready ? "ready" : "setup"}`}
          >
            {live === "live"
              ? "Live in Meet"
              : ready
                ? "Ready"
                : `${blocking || 1} to set up`}
          </span>
          {live === "live" ? (
            <button
              className="tr-stop"
              onClick={() => void stopLive()}
              type="button"
            >
              Stop translation
            </button>
          ) : (
            <button
              className="tr-start"
              disabled={!ready || live === "starting"}
              onClick={() => void startLive()}
              title={
                ready
                  ? "Send your captioned video to Google Meet"
                  : "Finish the system check first"
              }
              type="button"
            >
              {live === "starting" ? "Starting…" : "Start translation"}
            </button>
          )}
        </div>
      </header>
      {liveError && <p className="tr-error">{liveError}</p>}

      <div className="tr-grid">
        <div className="tr-column">
          {/* Live preview with the caption overlay Meet will see */}
          <section className="tr-card tr-preview-card">
            <div
              className={`tr-preview ${settings.mirrorPreview ? "mirrored" : ""}`}
            >
              <video autoPlay muted playsInline ref={videoRef} />
              {!stream && (
                <div className="tr-preview-empty">
                  <strong>Camera preview</strong>
                  <span>{previewError || "Starting your camera…"}</span>
                  {previewError && (
                    <button onClick={() => void allowAccess()} type="button">
                      Allow camera & microphone
                    </button>
                  )}
                </div>
              )}
              {live === "live" ? (
                <span className="tr-live-badge">
                  ● LIVE · {liveDevice || "Virtual camera"}
                </span>
              ) : (
                stream && <span className="tr-tag">Preview</span>
              )}
              <div className="tr-meter" aria-label="Microphone level">
                <span>Mic</span>
                <i>
                  <em style={{ width: `${Math.round(level * 100)}%` }} />
                </i>
              </div>
              {settings.showCaptions && caption && stream && (
                <div className={`tr-caption ${settings.captionSize}`}>
                  {caption}
                </div>
              )}
            </div>
          </section>

          <TypeToSpeak
            disabled={
              !engineCheck ||
              engineCheck.missingPackages.length > 0 ||
              !hasVirtualMic
            }
            onClearCaption={() => setCaption("")}
            onSpeak={speak}
          />
        </div>

        <div className="tr-column tr-column-side">
          {/* One card, three tabs: Setup, Settings, Meet guide */}
          <section className="tr-card tr-tabs-card">
            <div className="tr-tabs" role="tablist" aria-label="Translation">
              {TABS.map((item) => (
                <button
                  aria-selected={tab === item.id}
                  className={tab === item.id ? "active" : ""}
                  key={item.id}
                  onClick={() => setTab(item.id)}
                  role="tab"
                  type="button"
                >
                  {item.label}
                  {item.id === "setup" && blocking > 0 && <b>{blocking}</b>}
                </button>
              ))}
            </div>

            {/* Setup: the system check as a progress ring and a tile per step */}
            {tab === "setup" && (
              <>
                <div className="tr-progress">
                  <svg
                    aria-hidden="true"
                    className="tr-ring"
                    height="64"
                    viewBox="0 0 54 54"
                    width="64"
                  >
                    <circle
                      className="track"
                      cx="27"
                      cy="27"
                      fill="none"
                      r="22"
                      strokeWidth="6"
                    />
                    <circle
                      className="value"
                      cx="27"
                      cy="27"
                      fill="none"
                      r="22"
                      strokeDasharray={`${(okCount / Math.max(stepRows.length, 1)) * ringLength} ${ringLength}`}
                      strokeLinecap="round"
                      strokeWidth="6"
                      transform="rotate(-90 27 27)"
                    />
                    <text textAnchor="middle" x="27" y="31">
                      {okCount}/{stepRows.length}
                    </text>
                  </svg>
                  <div>
                    <h2>
                      {blocking === 0
                        ? "You're all set"
                        : okCount >= stepRows.length / 2
                          ? "Almost there"
                          : "Let's get you set up"}
                    </h2>
                    <p>
                      {blocking
                        ? `Fix ${blocking} step${blocking > 1 ? "s" : ""} to go live in Meet.`
                        : "Everything Meet needs is ready."}
                    </p>
                  </div>
                  <button
                    className="tr-ghost"
                    disabled={checking}
                    onClick={() => void runEngineCheck()}
                    type="button"
                  >
                    {checking ? "Checking…" : "Run again"}
                  </button>
                </div>

                <ul className="tr-steps">
                  {stepRows.map((row) => (
                    <li key={row.id}>
                      <button
                        aria-label={`${row.title}: ${stepLabel(row.state)}`}
                        aria-pressed={activeId === row.id}
                        className={`tr-step ${row.state} ${activeId === row.id ? "selected" : ""}`}
                        onClick={() => setOpenRow(activeId === row.id ? "" : row.id)}
                        title={`${row.title}: ${stepLabel(row.state)}`}
                        type="button"
                      >
                        {row.state === "ok" && (
                          <span className="tr-step-check">
                            <Icon name="check" size={12} />
                          </span>
                        )}
                        <span className="tr-tile">
                          <Icon name={CHECK_ICONS[row.id] ?? "info"} />
                        </span>
                        <strong>{row.title}</strong>
                      </button>
                    </li>
                  ))}
                </ul>

                {activeRow && (
                  <div className="tr-step-detail">
                    <strong>{activeRow.title}</strong>
                    <span>{activeRow.detail}</span>
                    {activeRow.command && (
                      <CopyCommand command={activeRow.command} />
                    )}
                    {activeRow.action && (
                      <button
                        className="tr-ghost"
                        onClick={activeRow.action.run}
                        type="button"
                      >
                        {activeRow.action.label}
                      </button>
                    )}
                  </div>
                )}
                {testNote && <p className="tr-note">{testNote}</p>}
                {noteRow && (
                  <p className="tr-footnote">
                    <Icon name="info" size={16} />
                    {noteRow.detail}
                  </p>
                )}
              </>
            )}

            {/* Settings: devices and translation options */}
            {tab === "settings" && (
              <>
                <header className="tr-card-head">
                  <div>
                    <h2>Settings</h2>
                    <p>Saved on this computer.</p>
                  </div>
                </header>
                <div className="tr-settings">
                  <label>
                    Camera
                    <select
                      onChange={(e) => update({ cameraId: e.target.value })}
                      value={settings.cameraId}
                    >
                      {devices.cameras.map((d) => (
                        <option key={d.deviceId} value={d.deviceId}>
                          {d.label || "Camera"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Microphone
                    <select
                      onChange={(e) => update({ microphoneId: e.target.value })}
                      value={settings.microphoneId}
                    >
                      {devices.mics.map((d) => (
                        <option key={d.deviceId} value={d.deviceId}>
                          {d.label || "Microphone"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Iris voice goes to
                    <select
                      onChange={(e) => update({ virtualMicId: e.target.value })}
                      value={settings.virtualMicId}
                    >
                      <option value="">Automatic (virtual microphone)</option>
                      {devices.outputs
                        .filter((d) => isVirtualMic(d.label))
                        .map((d) => (
                          <option key={d.deviceId} value={d.deviceId}>
                            {d.label}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label>
                    Sign language
                    <select
                      onChange={() => update({ signLanguage: "ASL" })}
                      value={settings.signLanguage}
                    >
                      <option value="ASL">American Sign Language (ASL)</option>
                      <option disabled>
                        British Sign Language (coming soon)
                      </option>
                      <option disabled>
                        French Sign Language (coming soon)
                      </option>
                    </select>
                  </label>
                  <label>
                    Video quality
                    <select
                      onChange={(e) =>
                        update({
                          resolution: e.target
                            .value as TranslationSettings["resolution"],
                        })
                      }
                      value={settings.resolution}
                    >
                      <option value="720p">HD (1280×720)</option>
                      <option value="540p">Balanced (960×540)</option>
                    </select>
                  </label>
                  <div className="tr-field">
                    Caption size
                    <div
                      className="tr-segment"
                      role="radiogroup"
                      aria-label="Caption size"
                    >
                      {(["small", "medium", "large"] as const).map((size) => (
                        <button
                          aria-checked={settings.captionSize === size}
                          className={
                            settings.captionSize === size ? "active" : ""
                          }
                          key={size}
                          onClick={() => update({ captionSize: size })}
                          role="radio"
                          type="button"
                        >
                          {size[0].toUpperCase() + size.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                  <Toggle
                    checked={settings.showCaptions}
                    description="Captions appear under your video in Meet."
                    label="Show captions on my video"
                    onChange={(value) => update({ showCaptions: value })}
                  />
                  <Toggle
                    checked={settings.speakSigns}
                    description="Iris speaks each recognised sign."
                    label="Speak recognised signs aloud"
                    onChange={(value) => update({ speakSigns: value })}
                  />
                  <Toggle
                    checked={settings.mirrorPreview}
                    description="Preview only. Meet gets the unmirrored video."
                    label="Mirror my preview"
                    onChange={(value) => update({ mirrorPreview: value })}
                  />
                </div>
              </>
            )}

            {/* Meet guide: how to pick the devices in Google Meet */}
            {tab === "meet" && (
              <>
                <header className="tr-card-head">
                  <div>
                    <h2>In Google Meet</h2>
                    <p>Three steps to use Iris in a call.</p>
                  </div>
                </header>
                <ol className="tr-meet-list">
                  <li>
                    Click <strong>Start translation</strong> here first.
                  </li>
                  <li>
                    In Meet, open <strong>Settings ⚙ → Video</strong> and
                    choose <strong>OBS Virtual Camera</strong>.
                  </li>
                  <li>
                    Open <strong>Audio</strong> and choose{" "}
                    <strong>{isMac ? "BlackHole 2ch" : "CABLE Output"}</strong>{" "}
                    as the microphone.
                  </li>
                </ol>
              </>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

/** Type to speak: send typed text (or a quick phrase) as Iris's voice. */
function TypeToSpeak({
  disabled,
  onClearCaption,
  onSpeak,
}: {
  disabled: boolean;
  onClearCaption: () => void;
  onSpeak: (text: string) => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Speaks the text, then clears the box; errors are shown under it.
  const say = async (value: string) => {
    if (!value.trim()) return;
    setBusy(true);
    setError("");
    try {
      await onSpeak(value.trim());
      setText("");
    } catch (problem) {
      setError(engineError(problem, "Iris couldn't speak that."));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="tr-card">
      <header className="tr-card-head">
        <div className="tr-title">
          <span className="tr-tile purple">
            <Icon name="message" />
          </span>
          <div>
            <h2>Type to speak</h2>
            <p>Iris says it in the meeting and shows a caption.</p>
          </div>
        </div>
        <button className="tr-ghost" onClick={onClearCaption} type="button">
          Clear caption
        </button>
      </header>
      <form
        className={`tr-speak ${disabled ? "locked" : ""}`}
        onSubmit={(event) => {
          event.preventDefault();
          void say(text);
        }}
      >
        <textarea
          disabled={disabled || busy}
          maxLength={500}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends; Shift+Enter adds a new line.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void say(text);
            }
          }}
          placeholder={
            disabled
              ? "Finish setup to use Type to speak."
              : "Type a message and press Enter…"
          }
          rows={3}
          value={text}
        />
        <div className="tr-speak-row">
          <div className="tr-phrases">
            {QUICK_PHRASES.map((phrase) => (
              <button
                disabled={disabled || busy}
                key={phrase}
                onClick={() => void say(phrase)}
                type="button"
              >
                {phrase}
              </button>
            ))}
          </div>
          <button
            className="tr-start"
            disabled={disabled || busy || !text.trim()}
            type="submit"
          >
            {busy ? "Speaking…" : "Speak"}
          </button>
        </div>
        {error && <p className="tr-error">{error}</p>}
      </form>
    </section>
  );
}

/** An on/off switch with a label and short description. */
function Toggle({
  checked,
  description,
  label,
  onChange,
}: {
  checked: boolean;
  description: string;
  label: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      aria-checked={checked}
      className="tr-toggle"
      onClick={() => onChange(!checked)}
      role="switch"
      type="button"
    >
      <span>
        <strong>{label}</strong>
        <small>{description}</small>
      </span>
      <i className={checked ? "on" : ""} />
    </button>
  );
}

/** A shell command with a Copy button. */
function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="tr-command">
      <code>{command}</code>
      <button
        onClick={() =>
          void navigator.clipboard.writeText(command).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          })
        }
        type="button"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Small icons
// ---------------------------------------------------------------------------

/** Line-icon drawings (24×24 grid), drawn in the current text colour. */
const ICON_PATHS: Record<IconName, ReactNode> = {
  camera: (
    <>
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </>
  ),
  mic: (
    <>
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <path d="M12 19v4M8 23h8" />
    </>
  ),
  volume: (
    <>
      <path d="M11 5 6 9H2v6h4l5 4V5z" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
    </>
  ),
  cpu: (
    <>
      <rect height="16" rx="2" width="16" x="4" y="4" />
      <rect height="6" width="6" x="9" y="9" />
      <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
    </>
  ),
  video: (
    <>
      <path d="m23 7-7 5 7 5V7z" />
      <rect height="14" rx="2" width="15" x="1" y="5" />
    </>
  ),
  headphones: (
    <>
      <path d="M3 18v-6a9 9 0 0 1 18 0v6" />
      <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z" />
    </>
  ),
  message: (
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
  ),
  check: <path d="m20 6-11 11-5-5" />,
  info: (
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </>
  ),
};

/** One of the small line icons above. */
function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.9"
      viewBox="0 0 24 24"
      width={size}
    >
      {ICON_PATHS[name]}
    </svg>
  );
}
