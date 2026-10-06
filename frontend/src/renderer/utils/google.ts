/** Google Calendar scope that lets Iris create events with Meet links. */
export const googleCalendarScope =
  "https://www.googleapis.com/auth/calendar.events";

const tokenKey = "iris-google-calendar-token";
// Google access tokens last an hour; treat them as stale a little early.
const tokenLifetimeMs = 55 * 60_000;

type StoredToken = { token: string; expiresAt: number };

/** Saves the Google access token returned alongside a Supabase OAuth session. */
export async function saveGoogleToken(token: string | null | undefined) {
  if (!token) return;
  const stored: StoredToken = { token, expiresAt: Date.now() + tokenLifetimeMs };
  await window.iris.auth.secureSet(tokenKey, JSON.stringify(stored));
}

/** Returns a still-valid Google access token, or null if Google must be reconnected. */
export async function loadGoogleToken() {
  try {
    const raw = await window.iris.auth.secureGet(tokenKey);
    if (!raw) return null;
    const stored = JSON.parse(raw) as StoredToken;
    return stored.expiresAt > Date.now() ? stored.token : null;
  } catch {
    return null;
  }
}

export async function clearGoogleToken() {
  await window.iris.auth.secureSet(tokenKey, null);
}

/**
 * Accepts a full Meet URL, a URL without https://, or a bare meeting code
 * (abc-defg-hij) and returns a safe https://meet.google.com URL, or null.
 */
export function normalizeMeetUrl(value: string) {
  const input = value.trim();
  if (!input) return null;
  if (/^[a-z]{3}-?[a-z]{4}-?[a-z]{3}$/i.test(input)) {
    const code = input.replace(/-/g, "").toLowerCase();
    return `https://meet.google.com/${code.slice(0, 3)}-${code.slice(3, 7)}-${code.slice(7)}`;
  }
  try {
    const url = new URL(/^https?:\/\//i.test(input) ? input : `https://${input}`);
    if (url.hostname !== "meet.google.com" || url.pathname === "/") return null;
    url.protocol = "https:";
    return url.toString();
  } catch {
    return null;
  }
}

/** Strips Electron's "Error invoking remote method" wrapper from IPC errors. */
export function ipcErrorMessage(error: unknown, fallback: string) {
  const message = error instanceof Error ? error.message : "";
  return (
    message
      .replace(/^Error invoking remote method '[^']+': (Error: )?/, "")
      .replace(/^GOOGLE_AUTH: /, "") || fallback
  );
}

export const isGoogleAuthError = (error: unknown) =>
  error instanceof Error && error.message.includes("GOOGLE_AUTH");

/** A Google Calendar event as returned by the Electron main process. */
export type GoogleEvent = Awaited<
  ReturnType<Window["iris"]["google"]["listEvents"]>
>[number];
