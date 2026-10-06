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
});
