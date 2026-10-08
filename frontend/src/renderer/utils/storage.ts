// Small wrappers around localStorage for preferences that must survive a
// restart. Chromium writes localStorage to disk lazily, so `saveLocal` asks
// Electron to flush it right away (otherwise a setting changed just before
// Iris is stopped can be lost).

/** Reads a saved value, or null if missing or storage is unavailable. */
export function readLocal(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** Saves (or, with null, removes) a value and flushes it to disk. */
export function saveLocal(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: the value lasts until Iris restarts */
  }
  // Older running copies of Iris may not have this bridge yet.
  void window.iris?.app?.flushStorage?.().catch(() => undefined);
}
