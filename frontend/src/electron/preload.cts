import { contextBridge, ipcRenderer } from "electron";

// Expose only deliberate, safe values to the renderer process.
// Add new desktop capabilities here instead of importing Electron in React.
contextBridge.exposeInMainWorld("iris", {
  platform: process.platform,
  auth: {
    openExternal: (url: string) =>
      ipcRenderer.invoke("iris:auth:open-external", url),
    consumeCallback: () => ipcRenderer.invoke("iris:auth:consume-callback"),
    secureGet: (key: string) => ipcRenderer.invoke("iris:auth:secure-get", key),
    secureSet: (key: string, value: string | null) =>
      ipcRenderer.invoke("iris:auth:secure-set", key, value),
    canPersistSecurely: () => ipcRenderer.invoke("iris:auth:can-persist"),
    onCallback: (listener: (url: string) => void) => {
      const handler = (_event: Electron.IpcRendererEvent, url: string) =>
        listener(url);
      ipcRenderer.on("iris:auth-callback", handler);
      return () => ipcRenderer.removeListener("iris:auth-callback", handler);
    },
  },
  meetings: {
    openExternal: (url: string) =>
      ipcRenderer.invoke("iris:meetings:open-external", url),
  },
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
    openEvent: (url: string) =>
      ipcRenderer.invoke("iris:google:open-event", url),
    deleteEvent: (accessToken: string, eventId: string) =>
      ipcRenderer.invoke("iris:google:delete-event", accessToken, eventId),
  },
});
