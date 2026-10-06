import {
  app,
  BrowserWindow,
  ipcMain,
  nativeTheme,
  safeStorage,
  shell,
} from "electron";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Electron's main process owns the native window and app lifecycle.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
let mainWindow: BrowserWindow | null = null;
let pendingAuthCallback: string | null = null;
let authCallbackServer: Server | null = null;
const loopbackAuthPort = 54321;
const loopbackAuthPath = "/auth/callback";

// Set the process name before Electron creates any macOS UI, including the dock item.
app.setName("Iris");

function assetPath(filename: string) {
  return process.env.VITE_DEV_SERVER_URL
    ? path.join(process.cwd(), "public", filename)
    : app.isPackaged
      ? path.join(process.resourcesPath, "brand", filename)
      : path.join(__dirname, "../../dist", filename);
}

function currentAppIcon() {
  return assetPath(
    nativeTheme.shouldUseDarkColors
      ? "iris-dock-dark.png"
      : "iris-dock-light.png",
  );
}

function updateAppIcon() {
  const icon = currentAppIcon();
  if (process.platform === "darwin" && app.dock) {
    app.dock.setIcon(icon);
  }
  for (const window of BrowserWindow.getAllWindows()) {
    window.setIcon(icon);
  }
}

function secureStorePath() {
  return path.join(app.getPath("userData"), "iris-auth-session.json");
}

function canPersistSecurely() {
  return (
    safeStorage.isEncryptionAvailable() &&
    !(
      process.platform === "linux" &&
      safeStorage.getSelectedStorageBackend() === "basic_text"
    )
  );
}

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

function deliverAuthCallback(url: string) {
  if (!url.startsWith("iris://auth/callback")) return;
  pendingAuthCallback = url;
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show();
    mainWindow.focus();
    mainWindow.webContents.send("iris:auth-callback", url);
  }
}

function callbackFromArguments(argumentsList: readonly string[]) {
  return (
    argumentsList.find((argument) =>
      argument.startsWith("iris://auth/callback"),
    ) ?? null
  );
}

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

app.on("open-url", (event, url) => {
  event.preventDefault();
  deliverAuthCallback(url);
});

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
  void mainWindow.loadFile(path.join(__dirname, "../../dist/index.html"));
}

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

ipcMain.handle("iris:auth:open-external", async (_event, url: string) => {
  const parsed = new URL(url);
  if (parsed.protocol !== "https:")
    throw new Error("Only secure authentication URLs can be opened.");
  await shell.openExternal(parsed.toString());
});
ipcMain.handle("iris:meetings:open-external", async (_event, url: string) => {
  const parsed = new URL(url);
  // Meeting URLs originate in user-owned data, so allow only Google Meet.
  if (parsed.protocol !== "https:" || parsed.hostname !== "meet.google.com") {
    throw new Error("Only secure Google Meet URLs may be opened.");
  }
  await shell.openExternal(parsed.toString());
});
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
    const title = String(request?.title ?? "").trim().slice(0, 200);
    if (!title || isNaN(startsAt.getTime()) || !(endsAt > startsAt))
      throw new Error("A meeting needs a title and a valid start and end time.");
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
      throw new Error("Google created the event but did not return a Meet link.");
    return { eventId: body.id ?? null, htmlLink: body.htmlLink ?? null, meetUrl };
  },
);
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
ipcMain.handle("iris:auth:secure-get", (_event, key: string) =>
  readSecureValue(key),
);
ipcMain.handle(
  "iris:auth:secure-set",
  (_event, key: string, value: string | null) => writeSecureValue(key, value),
);
ipcMain.handle("iris:auth:can-persist", () => canPersistSecurely());

// macOS applications stay open after their last window closes.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => authCallbackServer?.close());
