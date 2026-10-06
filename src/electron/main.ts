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
