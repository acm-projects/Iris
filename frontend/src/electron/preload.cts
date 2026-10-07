// Preload bridge: runs before React and exposes a small, typed `window.iris`
// API. Every call is forwarded to an ipcMain handler in main.ts.
import { contextBridge, ipcRenderer } from "electron";

// Expose only deliberate, safe values to the renderer process.
// Add new desktop capabilities here instead of importing Electron in React.
contextBridge.exposeInMainWorld("iris", {
  // Operating system name (darwin, win32, linux).
  platform: process.platform,
  // Appearance: native theme (title bar) and the dock icon style.
  app: {
    setTheme: (theme: "light" | "dark" | "system") =>
      ipcRenderer.invoke("iris:app:set-theme", theme),
    getIcon: () => ipcRenderer.invoke("iris:app:get-icon"),
    setIcon: (choice: "auto" | "light" | "dark" | "mono") =>
      ipcRenderer.invoke("iris:app:set-icon", choice),
  },
  // Sign-in helpers: open the browser, receive iris:// callbacks, and read or
  // write encrypted values (the Supabase session and Google tokens).
  auth: {
    openExternal: (url: string) =>
      ipcRenderer.invoke("iris:auth:open-external", url),
    consumeCallback: () => ipcRenderer.invoke("iris:auth:consume-callback"),
    secureGet: (key: string) => ipcRenderer.invoke("iris:auth:secure-get", key),
    secureSet: (key: string, value: string | null) =>
      ipcRenderer.invoke("iris:auth:secure-set", key, value),
    // Subscribes to sign-in callbacks; returns a function that unsubscribes.
    onCallback: (listener: (url: string) => void) => {
      const handler = (_event: Electron.IpcRendererEvent, url: string) =>
        listener(url);
      ipcRenderer.on("iris:auth-callback", handler);
      return () => ipcRenderer.removeListener("iris:auth-callback", handler);
    },
  },
  // Camera/microphone permission and the Python translation engine.
  media: {
    status: () => ipcRenderer.invoke("iris:media:status"),
    request: (kind: "camera" | "microphone") =>
      ipcRenderer.invoke("iris:media:request", kind),
  },
  engine: {
    request: (method: string, params?: unknown) =>
      ipcRenderer.invoke("iris:engine:request", method, params),
    // Fire-and-forget video frame (JPEG bytes) for the virtual camera.
    sendFrame: (jpeg: Uint8Array) =>
      ipcRenderer.send("iris:engine:frame", jpeg),
    // Subscribes to engine events; returns a function that unsubscribes.
    onEvent: (listener: (message: unknown) => void) => {
      const handler = (_event: Electron.IpcRendererEvent, message: unknown) =>
        listener(message);
      ipcRenderer.on("iris:engine:event", handler);
      return () => ipcRenderer.removeListener("iris:engine:event", handler);
    },
  },
  // Opens Google Meet links in the default browser.
  meetings: {
    openExternal: (url: string) =>
      ipcRenderer.invoke("iris:meetings:open-external", url),
  },
  // Google Calendar API calls (performed by main.ts with the user's token).
  google: {
    createMeetEvent: (
      accessToken: string,
      request: {
        title: string;
        startsAt: string;
        endsAt: string;
        description?: string;
      },
    ) =>
      ipcRenderer.invoke("iris:google:create-meet-event", accessToken, request),
    listEvents: (accessToken: string, timeMin: string, timeMax: string) =>
      ipcRenderer.invoke(
        "iris:google:list-events",
        accessToken,
        timeMin,
        timeMax,
      ),
    setEventColor: (accessToken: string, eventId: string, colorId: string) =>
      ipcRenderer.invoke(
        "iris:google:set-event-color",
        accessToken,
        eventId,
        colorId,
      ),
    setEventTimes: (
      accessToken: string,
      eventId: string,
      startsAt: string,
      endsAt: string,
    ) =>
      ipcRenderer.invoke(
        "iris:google:set-event-times",
        accessToken,
        eventId,
        startsAt,
        endsAt,
      ),
    openEvent: (url: string) =>
      ipcRenderer.invoke("iris:google:open-event", url),
    deleteEvent: (accessToken: string, eventId: string) =>
      ipcRenderer.invoke("iris:google:delete-event", accessToken, eventId),
  },
});
