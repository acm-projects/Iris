// Electron main process for Iris: creates the window, handles sign-in deep
// links, stores secrets encrypted, and calls the Google Calendar API on behalf
// of the React UI (which runs sandboxed without access to Node.js).
import {
  app,
  BrowserWindow,
  ipcMain,
  nativeTheme,
  powerSaveBlocker,
  safeStorage,
  shell,
  systemPreferences,
} from "electron";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Electron's main process owns the native window and app lifecycle.
// ---------------------------------------------------------------------------
// App state shared by the window, deep-link handling and the OAuth callback server
// ---------------------------------------------------------------------------
const __dirname = path.dirname(fileURLToPath(import.meta.url));
let mainWindow: BrowserWindow | null = null;
// An iris://auth/callback URL waiting for React to pick it up.
let pendingAuthCallback: string | null = null;
// Tiny local web server that receives Supabase OAuth redirects (see below).
let authCallbackServer: Server | null = null;
const loopbackAuthPort = 54321;
const loopbackAuthPath = "/auth/callback";

// Set the process name before Electron creates any macOS UI, including the dock item.
app.setName("Iris");

// ---------------------------------------------------------------------------
// Dock / window icon
// ---------------------------------------------------------------------------

/** Finds an image from frontend/assets in dev, packaged and preview builds. */
function assetPath(filename: string) {
  return process.env.VITE_DEV_SERVER_URL
    ? path.join(process.cwd(), "assets", filename)
    : app.isPackaged
      ? path.join(process.resourcesPath, "brand", filename)
      : path.join(__dirname, "../dist", filename);
}

// ---------------------------------------------------------------------------
// Preferences chosen in Settings → Appearance (stored unencrypted; not secret)
// ---------------------------------------------------------------------------

/** Dock icon styles; "auto" follows the app's light/dark appearance. */
const appIconChoices = ["auto", "light", "dark", "mono"] as const;
type AppIconChoice = (typeof appIconChoices)[number];
/** Everything saved in iris-preferences.json. */
type Preferences = { appIcon: AppIconChoice };

/** Preferences file in the user's app-data folder. */
function preferencesPath() {
  return path.join(app.getPath("userData"), "iris-preferences.json");
}

/** Reads saved preferences, falling back to defaults for anything missing. */
function readPreferences(): Preferences {
  try {
    const saved = JSON.parse(
      readFileSync(preferencesPath(), "utf8"),
    ) as Partial<Preferences>;
    return {
      appIcon: appIconChoices.includes(saved.appIcon as AppIconChoice)
        ? (saved.appIcon as AppIconChoice)
        : "auto",
    };
  } catch {
    return { appIcon: "auto" };
  }
}

/** Saves preferences (overwrites the whole file). */
function writePreferences(preferences: Preferences) {
  mkdirSync(path.dirname(preferencesPath()), { recursive: true });
  writeFileSync(preferencesPath(), JSON.stringify(preferences), "utf8");
}

/** Picks the dock icon for the saved choice ("auto" matches light/dark mode). */
function currentAppIcon() {
  const choice = readPreferences().appIcon;
  const style =
    choice === "auto"
      ? nativeTheme.shouldUseDarkColors
        ? "dark"
        : "light"
      : choice;
  return assetPath(`iris-dock-${style}.png`);
}

/** Applies the current icon to the macOS dock and every open window. */
function updateAppIcon() {
  const icon = currentAppIcon();
  if (process.platform === "darwin" && app.dock) {
    app.dock.setIcon(icon);
  }
  for (const window of BrowserWindow.getAllWindows()) {
    window.setIcon(icon);
  }
}

// ---------------------------------------------------------------------------
// Encrypted key/value storage (Supabase session, Google tokens)
// Values are encrypted with the OS keychain via Electron safeStorage.
// ---------------------------------------------------------------------------

/** File in the user's app-data folder that holds the encrypted values. */
function secureStorePath() {
  return path.join(app.getPath("userData"), "iris-auth-session.json");
}

/** True when real OS encryption is available (Linux may only offer plain text). */
function canPersistSecurely() {
  return (
    safeStorage.isEncryptionAvailable() &&
    !(
      process.platform === "linux" &&
      safeStorage.getSelectedStorageBackend() === "basic_text"
    )
  );
}

/** Reads and decrypts one stored value, or null if missing/unreadable. */
function readSecureValue(key: string) {
  if (!canPersistSecurely() || !existsSync(secureStorePath())) return null;
  try {
    const stored = JSON.parse(
      readFileSync(secureStorePath(), "utf8"),
    ) as Record<string, string>;
    return stored[key]
      ? safeStorage.decryptString(Buffer.from(stored[key], "base64"))
      : null;
  } catch {
    return null;
  }
}

/** Encrypts and stores a value; passing null deletes the key. */
function writeSecureValue(key: string, value: string | null) {
  if (!canPersistSecurely()) return;
  const file = secureStorePath();
  let stored: Record<string, string> = {};
  try {
    stored = JSON.parse(readFileSync(file, "utf8")) as Record<string, string>;
  } catch {
    /* first session */
  }
  if (value === null) delete stored[key];
  else stored[key] = safeStorage.encryptString(value).toString("base64");
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(stored), {
    encoding: "utf8",
    mode: 0o600,
  });
}

// ---------------------------------------------------------------------------
// Sign-in callbacks (iris://auth/callback?code=...)
// ---------------------------------------------------------------------------

/** Brings Iris to the front and hands an auth callback URL to React. */
function deliverAuthCallback(url: string) {
  if (!url.startsWith("iris://auth/callback")) return;
  pendingAuthCallback = url;
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show();
    mainWindow.focus();
    mainWindow.webContents.send("iris:auth-callback", url);
  }
}

/** Finds an iris:// callback URL in command-line arguments (Windows/Linux, cold start). */
function callbackFromArguments(argumentsList: readonly string[]) {
  return (
    argumentsList.find((argument) =>
      argument.startsWith("iris://auth/callback"),
    ) ?? null
  );
}

/**
 * Starts http://127.0.0.1:54321/auth/callback. Google/Supabase redirect the
 * browser here after sign-in; the page shows an "Open Iris" button that
 * forwards the one-time code to the app through the iris:// link.
 */
function startLoopbackAuthServer() {
  if (authCallbackServer) return;
  authCallbackServer = createServer((request, response) => {
    const requestUrl = new URL(
      request.url ?? "/",
      `http://127.0.0.1:${loopbackAuthPort}`,
    );
    if (request.method !== "GET" || requestUrl.pathname !== loopbackAuthPath) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }

    // Supabase sends the short-lived OAuth code here. The completion page asks
    // the user before handing it to Iris via the operating-system app link.
    const irisCallbackUrl = `iris://auth/callback${requestUrl.search}`.replace(
      /&/g,
      "&amp;",
    );
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Type": "text/html; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
    });
    response.end(
      `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Continue to Iris</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f2fbfc;color:#00224e;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:28rem;padding:3rem;text-align:center;border:1px solid #d5e0ed;border-radius:1.5rem;background:#fff;box-shadow:0 18px 42px rgba(0,34,78,.1)}h1{margin:0 0 .6rem;font-size:2rem}p{margin:0;color:#4376ab;line-height:1.5}a{display:inline-block;margin-top:1.7rem;padding:.9rem 1.4rem;border-radius:.75rem;background:#00224e;color:#fff;font-weight:700;text-decoration:none}small{display:block;margin-top:1rem;color:#4376ab}</style><main><h1>Signed in successfully</h1><p>Would you like to return to Iris now?</p><a href="${irisCallbackUrl}" onclick="this.textContent='Opening Iris…'">Open Iris</a><small>After Iris opens, you can close this completed browser tab.</small></main></html>`,
    );
  });
  authCallbackServer.on("error", (error) => {
    console.error(
      `Iris could not start its local authentication callback server: ${error.message}`,
    );
    authCallbackServer = null;
  });
  authCallbackServer.listen(loopbackAuthPort, "127.0.0.1");
}

// macOS delivers iris:// links through this event.
app.on("open-url", (event, url) => {
  event.preventDefault();
  deliverAuthCallback(url);
});

// Only one Iris instance may run; a second launch (e.g. from an iris:// link on
// Windows/Linux) forwards its callback URL to the existing window and exits.
const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", (_event, commandLine) => {
    const callback = callbackFromArguments(commandLine);
    if (callback) deliverAuthCallback(callback);
  });
}

/** Creates the desktop window and loads the appropriate renderer bundle. */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    title: "Iris",
    icon: currentAppIcon(),
    webPreferences: {
      // Preload runs before React and exposes a carefully limited API.
      preload: path.join(__dirname, "preload.cjs"),
      // Keep the renderer separate from Electron's privileged Node.js APIs.
      contextIsolation: true,
      nodeIntegration: false,
      // Keep timers and video running while Iris is behind other windows, so
      // the video sent to Google Meet doesn't freeze when you switch to Meet.
      backgroundThrottling: false,
    },
  });
  const devServerUrl = process.env.VITE_DEV_SERVER_URL;

  mainWindow.webContents.once("did-finish-load", () => {
    if (pendingAuthCallback) deliverAuthCallback(pendingAuthCallback);
  });
  mainWindow.webContents.on(
    "console-message",
    (_event, level, message, line, sourceId) => {
      if (level >= 2)
        console.error(`[renderer:${level}] ${message} (${sourceId}:${line})`);
    },
  );

  if (devServerUrl) {
    void mainWindow.loadURL(devServerUrl);
    return;
  }

  // Production loads Vite's static build from the local filesystem.
  void mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
}

// ---------------------------------------------------------------------------
// App start-up
// ---------------------------------------------------------------------------
app.whenReady().then(() => {
  app.setAsDefaultProtocolClient("iris");
  startLoopbackAuthServer();
  // macOS normally uses open-url, but a launch-services handoff can also
  // provide the URL in argv. Supporting both avoids a silent OAuth return.
  const launchCallback = callbackFromArguments(process.argv);
  if (launchCallback) deliverAuthCallback(launchCallback);
  updateAppIcon();
  nativeTheme.on("updated", updateAppIcon);
  createWindow();

  // On macOS, re-create a window when the dock icon is selected.
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// ---------------------------------------------------------------------------
// IPC handlers: the only privileged actions React may request (see preload.cts)
// ---------------------------------------------------------------------------

/** Opens a sign-in page (https only) in the user's default browser. */
ipcMain.handle("iris:auth:open-external", async (_event, url: string) => {
  const parsed = new URL(url);
  if (parsed.protocol !== "https:")
    throw new Error("Only secure authentication URLs can be opened.");
  await shell.openExternal(parsed.toString());
});
/** Opens a Google Meet link in the default browser (Meet's pre-join screen). */
ipcMain.handle("iris:meetings:open-external", async (_event, url: string) => {
  const parsed = new URL(url);
  // Meeting URLs originate in user-owned data, so allow only Google Meet.
  if (parsed.protocol !== "https:" || parsed.hostname !== "meet.google.com") {
    throw new Error("Only secure Google Meet URLs may be opened.");
  }
  await shell.openExternal(parsed.toString());
});
// --- Google Calendar API -------------------------------------------------
// Errors prefixed with "GOOGLE_AUTH:" tell React the token is missing/expired
// so it can ask the user to reconnect Google.

/** Data React sends when creating a calendar event with a Meet link. */
type MeetEventRequest = {
  title: string;
  startsAt: string;
  endsAt: string;
  description?: string;
};
// Calendar events are created here rather than in React so the request is not
// subject to browser CORS rules (production loads from file://).
ipcMain.handle(
  "iris:google:create-meet-event",
  async (_event, accessToken: string, request: MeetEventRequest) => {
    if (typeof accessToken !== "string" || !accessToken)
      throw new Error("Connect your Google account to create Meet links.");
    const startsAt = new Date(request?.startsAt);
    const endsAt = new Date(request?.endsAt);
    const title = String(request?.title ?? "")
      .trim()
      .slice(0, 200);
    if (!title || isNaN(startsAt.getTime()) || !(endsAt > startsAt))
      throw new Error(
        "A meeting needs a title and a valid start and end time.",
      );
    const response = await fetch(
      "https://www.googleapis.com/calendar/v3/calendars/primary/events?conferenceDataVersion=1",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          summary: title,
          description: request.description?.slice(0, 2000),
          start: { dateTime: startsAt.toISOString() },
          end: { dateTime: endsAt.toISOString() },
          conferenceData: {
            createRequest: {
              requestId: crypto.randomUUID(),
              conferenceSolutionKey: { type: "hangoutsMeet" },
            },
          },
        }),
      },
    );
    const body = (await response.json().catch(() => ({}))) as {
      id?: string;
      htmlLink?: string;
      hangoutLink?: string;
      conferenceData?: {
        entryPoints?: { entryPointType?: string; uri?: string }[];
      };
      error?: { message?: string };
    };
    if (response.status === 401 || response.status === 403)
      throw new Error(
        "GOOGLE_AUTH: Google needs permission to create Meet links. Reconnect Google and allow calendar access.",
      );
    if (!response.ok)
      throw new Error(
        body.error?.message || `Google Calendar returned ${response.status}.`,
      );
    const meetUrl =
      body.conferenceData?.entryPoints?.find(
        (entry) => entry.entryPointType === "video",
      )?.uri || body.hangoutLink;
    if (!meetUrl || new URL(meetUrl).hostname !== "meet.google.com")
      throw new Error(
        "Google created the event but did not return a Meet link.",
      );
    return {
      eventId: body.id ?? null,
      htmlLink: body.htmlLink ?? null,
      meetUrl,
    };
  },
);
/** The subset of a Google Calendar API event that Iris reads. */
type GoogleApiEvent = {
  id?: string;
  status?: string;
  colorId?: string;
  summary?: string;
  location?: string;
  htmlLink?: string;
  hangoutLink?: string;
  start?: { date?: string; dateTime?: string };
  end?: { date?: string; dateTime?: string };
  conferenceData?: {
    entryPoints?: { entryPointType?: string; uri?: string }[];
  };
};
/** Returns the URL only if it is a real meet.google.com link. */
const safeMeetUrl = (value: string | undefined) => {
  try {
    return value && new URL(value).hostname === "meet.google.com"
      ? value
      : null;
  } catch {
    return null;
  }
};
// Reads the primary calendar for the visible date range of the Calendar page.
ipcMain.handle(
  "iris:google:list-events",
  async (_event, accessToken: string, timeMin: string, timeMax: string) => {
    if (typeof accessToken !== "string" || !accessToken)
      throw new Error("GOOGLE_AUTH: Connect Google to see your calendar.");
    const min = new Date(timeMin);
    const max = new Date(timeMax);
    if (
      isNaN(min.getTime()) ||
      isNaN(max.getTime()) ||
      max <= min ||
      max.getTime() - min.getTime() > 93 * 86_400_000
    )
      throw new Error("That calendar date range is not valid.");
    const events: GoogleApiEvent[] = [];
    let pageToken: string | undefined;
    // A month grid rarely needs more than one page; cap it to stay responsive.
    for (let page = 0; page < 4; page++) {
      const params = new URLSearchParams({
        singleEvents: "true",
        orderBy: "startTime",
        timeMin: min.toISOString(),
        timeMax: max.toISOString(),
        maxResults: "250",
      });
      if (pageToken) params.set("pageToken", pageToken);
      const response = await fetch(
        `https://www.googleapis.com/calendar/v3/calendars/primary/events?${params}`,
        { headers: { Authorization: `Bearer ${accessToken}` } },
      );
      const body = (await response.json().catch(() => ({}))) as {
        items?: GoogleApiEvent[];
        nextPageToken?: string;
        error?: { message?: string };
      };
      if (response.status === 401 || response.status === 403)
        throw new Error(
          "GOOGLE_AUTH: Google needs permission to read your calendar. Reconnect Google.",
        );
      if (!response.ok)
        throw new Error(
          body.error?.message || `Google Calendar returned ${response.status}.`,
        );
      events.push(...(body.items ?? []));
      pageToken = body.nextPageToken;
      if (!pageToken) break;
    }
    return events
      .filter((item) => item.id && item.status !== "cancelled")
      .map((item) => ({
        id: item.id as string,
        title: item.summary?.trim() || "(No title)",
        start: item.start?.dateTime ?? item.start?.date ?? "",
        end: item.end?.dateTime ?? item.end?.date ?? "",
        allDay: !item.start?.dateTime,
        meetUrl:
          safeMeetUrl(
            item.conferenceData?.entryPoints?.find(
              (entry) => entry.entryPointType === "video",
            )?.uri,
          ) ?? safeMeetUrl(item.hangoutLink),
        htmlLink: item.htmlLink ?? null,
        location: item.location ?? null,
        colorId: item.colorId ?? null,
      }))
      .filter((item) => item.start && item.end);
  },
);
// Changes an event's colour on the user's primary Google Calendar.
ipcMain.handle(
  "iris:google:set-event-color",
  async (_event, accessToken: string, eventId: string, colorId: string) => {
    if (typeof accessToken !== "string" || !accessToken)
      throw new Error("GOOGLE_AUTH: Connect Google to change event colours.");
    if (typeof eventId !== "string" || !/^[a-v0-9_]{5,1024}$/i.test(eventId))
      throw new Error("That Google Calendar event ID is not valid.");
    if (!/^(?:[1-9]|1[01])$/.test(String(colorId)))
      throw new Error("That calendar colour is not valid.");
    const response = await fetch(
      `https://www.googleapis.com/calendar/v3/calendars/primary/events/${eventId}`,
      {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ colorId: String(colorId) }),
      },
    );
    if (response.ok) return;
    if (response.status === 401)
      throw new Error("GOOGLE_AUTH: Reconnect Google to change event colours.");
    const body = (await response.json().catch(() => ({}))) as {
      error?: { message?: string };
    };
    // 403 here usually means the event belongs to someone else's calendar.
    throw new Error(
      response.status === 403
        ? "You can only recolour events you own in Google Calendar."
        : body.error?.message || `Google Calendar returned ${response.status}.`,
    );
  },
);
/** Opens an event in Google Calendar (only Google Calendar URLs are allowed). */
ipcMain.handle("iris:google:open-event", async (_event, url: string) => {
  const parsed = new URL(url);
  const isCalendar =
    parsed.protocol === "https:" &&
    (parsed.hostname === "calendar.google.com" ||
      (parsed.hostname === "www.google.com" &&
        parsed.pathname.startsWith("/calendar")));
  if (!isCalendar) throw new Error("Only Google Calendar links may be opened.");
  await shell.openExternal(parsed.toString());
});
ipcMain.handle(
  // Deletes an event from the user's primary Google Calendar.
  "iris:google:delete-event",
  async (_event, accessToken: string, eventId: string) => {
    if (typeof accessToken !== "string" || !accessToken)
      throw new Error("GOOGLE_AUTH: Connect Google to remove calendar events.");
    // Google event IDs are base32hex; reject anything else before building the URL.
    if (typeof eventId !== "string" || !/^[a-v0-9_]{5,1024}$/i.test(eventId))
      throw new Error("That Google Calendar event ID is not valid.");
    const response = await fetch(
      `https://www.googleapis.com/calendar/v3/calendars/primary/events/${eventId}?sendUpdates=all`,
      { method: "DELETE", headers: { Authorization: `Bearer ${accessToken}` } },
    );
    // 404/410 mean the event is already gone, which is the outcome we want.
    if (response.ok || response.status === 404 || response.status === 410)
      return;
    if (response.status === 401 || response.status === 403)
      throw new Error(
        "GOOGLE_AUTH: Google needs permission to remove calendar events. Reconnect Google.",
      );
    const body = (await response.json().catch(() => ({}))) as {
      error?: { message?: string };
    };
    throw new Error(
      body.error?.message || `Google Calendar returned ${response.status}.`,
    );
  },
);
// Keep the callback until React explicitly consumes it. On macOS, open-url can
// arrive before the renderer has registered its IPC listener.
ipcMain.handle("iris:auth:consume-callback", () => {
  const callback = pendingAuthCallback;
  pendingAuthCallback = null;
  return callback;
});
// --- Appearance ------------------------------------------------------------

/** Makes native UI (title bar, menus) follow the app's Light/Dark/System theme. */
ipcMain.handle("iris:app:set-theme", (_event, theme: string) => {
  if (theme !== "light" && theme !== "dark" && theme !== "system")
    throw new Error("Unknown theme.");
  nativeTheme.themeSource = theme;
  updateAppIcon();
});
/** Returns the saved dock icon choice. */
ipcMain.handle("iris:app:get-icon", () => readPreferences().appIcon);
/** Saves a dock icon choice and applies it immediately. */
ipcMain.handle("iris:app:set-icon", (_event, choice: string) => {
  if (!appIconChoices.includes(choice as AppIconChoice))
    throw new Error("Unknown app icon.");
  writePreferences({ ...readPreferences(), appIcon: choice as AppIconChoice });
  updateAppIcon();
});

// Encrypted storage access for React (used for the Supabase session and Google tokens).
ipcMain.handle("iris:auth:secure-get", (_event, key: string) =>
  readSecureValue(key),
);
ipcMain.handle(
  // Store (or, with null, delete) an encrypted value.
  "iris:auth:secure-set",
  (_event, key: string, value: string | null) => writeSecureValue(key, value),
);

// ---------------------------------------------------------------------------
// Camera / microphone permission (macOS asks the user once per app)
// ---------------------------------------------------------------------------

/** Current OS permission for the camera and microphone. */
ipcMain.handle("iris:media:status", () => {
  if (process.platform !== "darwin")
    return { camera: "granted", microphone: "granted" };
  return {
    camera: systemPreferences.getMediaAccessStatus("camera"),
    microphone: systemPreferences.getMediaAccessStatus("microphone"),
  };
});
/** Shows the macOS permission prompt; returns whether access was granted. */
ipcMain.handle("iris:media:request", async (_event, kind: string) => {
  if (kind !== "camera" && kind !== "microphone")
    throw new Error("Unknown device.");
  if (process.platform !== "darwin") return true;
  return systemPreferences.askForMediaAccess(kind);
});

// ---------------------------------------------------------------------------
// Translation engine (backend/engine/engine.py)
// A Python child process that drives the virtual camera and microphone.
// Messages are JSON lines over stdin/stdout (see engine.py for the protocol).
// ---------------------------------------------------------------------------

let engine: ChildProcessWithoutNullStreams | null = null;
let engineRequestId = 0;
const enginePending = new Map<
  string,
  {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }
>();

/** Folder containing engine.py: the repo's backend/ in development, bundled copy when packaged. */
function engineFolder() {
  return process.env.VITE_DEV_SERVER_URL || !app.isPackaged
    ? path.resolve(process.cwd(), "..", "backend", "engine")
    : path.join(process.resourcesPath, "backend", "engine");
}

/** Prefers the project's virtual environment (backend/.venv), then the system Python. */
function pythonCandidates() {
  const backend = path.resolve(engineFolder(), "..");
  return [
    path.join(backend, ".venv", "bin", "python3"),
    path.join(backend, ".venv", "Scripts", "python.exe"),
    process.platform === "win32" ? "python" : "python3",
  ].filter((candidate) => !path.isAbsolute(candidate) || existsSync(candidate));
}

/** Sends a message to every window (engine events such as status updates). */
function broadcastEngineEvent(message: unknown) {
  for (const window of BrowserWindow.getAllWindows())
    window.webContents.send("iris:engine:event", message);
}

/** Starts the engine if needed; resolves once it is accepting requests. */
function ensureEngine() {
  if (engine && engine.exitCode === null) return Promise.resolve();
  const script = path.join(engineFolder(), "engine.py");
  if (!existsSync(script))
    return Promise.reject(
      new Error(`The translation engine was not found at ${script}.`),
    );
  const python = pythonCandidates()[0];
  const child = spawn(python, ["-u", script], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  engine = child;

  // Replies and events arrive as JSON lines on stdout.
  let buffer = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buffer += chunk;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      try {
        const message = JSON.parse(line) as {
          id?: string;
          result?: unknown;
          error?: string;
          event?: string;
        };
        const pending = message.id ? enginePending.get(message.id) : undefined;
        if (pending) {
          clearTimeout(pending.timer);
          enginePending.delete(message.id as string);
          if (message.error) pending.reject(new Error(message.error));
          else pending.resolve(message.result);
        } else if (message.event) {
          broadcastEngineEvent(message);
        }
      } catch {
        console.error(`[engine] unreadable output: ${line}`);
      }
    }
  });
  child.stderr.on("data", (chunk) => console.error(String(chunk).trimEnd()));

  // If the engine exits, reject every request still waiting for a reply.
  const failAll = (reason: string) => {
    for (const [id, pending] of enginePending) {
      clearTimeout(pending.timer);
      pending.reject(new Error(reason));
      enginePending.delete(id);
    }
  };
  child.on("exit", (code) => {
    if (engine === child) engine = null;
    failAll(`The translation engine stopped (exit code ${code}).`);
    keepAwakeWhileLive(false);
    broadcastEngineEvent({
      event: "status",
      camera: "stopped",
      engine: "stopped",
    });
  });

  return new Promise<void>((resolve, reject) => {
    child.once("spawn", () => resolve());
    child.once("error", (error) => {
      if (engine === child) engine = null;
      reject(
        new Error(
          `Could not start Python (${python}). Install Python 3 and the packages in backend/requirements.txt. (${error.message})`,
        ),
      );
    });
  });
}

/** Sends one request to the engine and waits for its reply. */
async function engineRequest(
  method: string,
  params: unknown = {},
  timeoutMs = 20_000,
) {
  await ensureEngine();
  const id = String(++engineRequestId);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      enginePending.delete(id);
      reject(
        new Error(`The translation engine did not answer "${method}" in time.`),
      );
    }, timeoutMs);
    enginePending.set(id, { resolve, reject, timer });
    engine?.stdin.write(JSON.stringify({ id, method, params }) + "\n");
  });
}

const engineMethods = new Set([
  "health",
  "system_check",
  "start_virtual_camera",
  "stop_virtual_camera",
  "speak",
  "test_tone",
]);
/** Runs an engine method requested by the Translation page. */
ipcMain.handle(
  "iris:engine:request",
  async (_event, method: string, params: unknown) => {
    if (!engineMethods.has(method)) throw new Error("Unknown engine method.");
    // Speech can take a while for long text; everything else should be quick.
    const result = await engineRequest(
      method,
      params,
      method === "speak" ? 90_000 : 20_000,
    );
    // While streaming to the virtual camera, stop macOS from napping Iris.
    if (method === "start_virtual_camera") keepAwakeWhileLive(true);
    if (method === "stop_virtual_camera") keepAwakeWhileLive(false);
    return result;
  },
);

// Id of the active power-save blocker while translation is live (-1 when off).
let liveBlocker = -1;
/** Prevents App Nap / suspension while video is being sent to Meet. */
function keepAwakeWhileLive(live: boolean) {
  if (live && liveBlocker === -1) {
    liveBlocker = powerSaveBlocker.start("prevent-app-suspension");
  } else if (!live && liveBlocker !== -1) {
    powerSaveBlocker.stop(liveBlocker);
    liveBlocker = -1;
  }
}
/** Video frames (JPEG bytes) for the virtual camera; dropped if the engine is busy. */
ipcMain.on("iris:engine:frame", (_event, jpeg: Uint8Array) => {
  if (!engine || engine.exitCode !== null || engine.stdin.writableNeedDrain)
    return;
  engine.stdin.write(
    JSON.stringify({
      method: "frame",
      data: Buffer.from(jpeg).toString("base64"),
    }) + "\n",
  );
});

// macOS applications stay open after their last window closes.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// Release the callback server's port when Iris quits.
app.on("before-quit", () => {
  authCallbackServer?.close();
  // Closing stdin lets the engine stop the virtual camera and exit cleanly.
  engine?.stdin.end();
});
