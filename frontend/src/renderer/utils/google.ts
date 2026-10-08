// Google helpers: storing/renewing the Google Calendar access token per Iris
// account, disconnecting, and parsing Google Meet links.
import { getSupabase } from "./supabase";
import { friendlyError } from "./errors";
import { readLocal, saveLocal } from "./storage";

/** Google Calendar scope that lets Iris create events with Meet links. */
export const googleCalendarScope =
  "https://www.googleapis.com/auth/calendar.events";

// Access tokens are kept per Iris account (encrypted by Electron safeStorage)
// so signing out and back in does not lose the Google connection.
const tokenKey = (userId: string) => `iris-google-calendar-token:${userId}`;
// Google access tokens last an hour; treat them as stale a little early.
const tokenLifetimeMs = 55 * 60_000;
// Edge Function that stores the Google refresh token server-side and trades it
// for fresh access tokens (it holds the OAuth client secret, the app never does).
// Its code is backend/supabase/functions/google-calendar-token; set
// VITE_GOOGLE_TOKEN_FUNCTION if it was deployed under a different name.
const tokenFunction =
  import.meta.env.VITE_GOOGLE_TOKEN_FUNCTION || "google-calendar-token";

type StoredToken = { token: string; expiresAt: number };

/** Saves a Google access token for this Iris account. */
export async function saveGoogleToken(
  userId: string,
  token: string | null | undefined,
  lifetimeSeconds?: number,
) {
  if (!token) return;
  const lifetime = lifetimeSeconds
    ? Math.min(lifetimeSeconds * 1000 - 5 * 60_000, tokenLifetimeMs)
    : tokenLifetimeMs;
  const stored: StoredToken = { token, expiresAt: Date.now() + lifetime };
  await window.iris.auth.secureSet(tokenKey(userId), JSON.stringify(stored));
}

/** Returns this account's stored access token if it has not expired yet. */
async function loadStoredToken(userId: string) {
  try {
    const raw = await window.iris.auth.secureGet(tokenKey(userId));
    if (!raw) return null;
    const stored = JSON.parse(raw) as StoredToken;
    return stored.expiresAt > Date.now() ? stored.token : null;
  } catch {
    return null;
  }
}

export async function clearGoogleToken(userId: string) {
  await window.iris.auth.secureSet(tokenKey(userId), null);
}

/** Sends the long-lived refresh token to the Edge Function for safekeeping. */
export async function saveGoogleRefreshToken(
  refreshToken: string | null | undefined,
) {
  if (!refreshToken) return;
  const { error } = await getSupabase().functions.invoke(tokenFunction, {
    body: { action: "save", refreshToken },
  });
  // Without the function, Iris still works; users just reconnect hourly.
  if (error)
    console.warn("Could not save the Google refresh token:", error.message);
}

/**
 * Returns a valid Google access token for this account: the stored one if it
 * is still fresh, otherwise a new one from the Edge Function. Null means the
 * user needs to connect (or reconnect) Google Calendar.
 */
export async function getGoogleAccessToken(userId: string) {
  // After "Disconnect", never renew access until the user connects again.
  if (isCalendarDisconnected(userId)) return null;
  const stored = await loadStoredToken(userId);
  if (stored) return stored;
  try {
    const { data, error } = await getSupabase().functions.invoke<{
      accessToken?: string;
      expiresIn?: number;
    }>(tokenFunction, { body: { action: "token" } });
    if (error || !data?.accessToken) return null;
    await saveGoogleToken(userId, data.accessToken, data.expiresIn);
    return data.accessToken;
  } catch {
    return null;
  }
}

// Set when the user disconnects Google Calendar in Settings; cleared only when
// they click "Connect Google Calendar" again. A Google *sign-in* alone does not
// reconnect the calendar.
const disconnectedKey = (userId: string) =>
  `iris-google-disconnected:${userId}`;

/** True if this account chose to disconnect Google Calendar on this computer. */
export const isCalendarDisconnected = (userId: string) =>
  readLocal(disconnectedKey(userId)) === "true";

/** Clears the "disconnected" choice after the user connects again. */
export const markCalendarConnected = (userId: string) =>
  saveLocal(disconnectedKey(userId), null);

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
    const url = new URL(
      /^https?:\/\//i.test(input) ? input : `https://${input}`,
    );
    if (url.hostname !== "meet.google.com" || url.pathname === "/") return null;
    url.protocol = "https:";
    return url.toString();
  } catch {
    return null;
  }
}

/** A user-friendly message for an error from Electron/Google (see utils/errors.ts). */
export function ipcErrorMessage(error: unknown, fallback: string) {
  return friendlyError(error, fallback);
}

export const isGoogleAuthError = (error: unknown) =>
  error instanceof Error && error.message.includes("GOOGLE_AUTH");

/** A Google Calendar event as returned by the Electron main process. */
export type GoogleEvent = Awaited<
  ReturnType<Window["iris"]["google"]["listEvents"]>
>[number];

/**
 * Disconnects Google Calendar for this account: revokes Iris's access at
 * Google (directly with the token kept on this computer, and through the Edge
 * Function for the server-side refresh token), then forgets the local token.
 * `revoked` is false only if neither step worked, in which case access can
 * still be removed from the Google Account page.
 */
export async function disconnectGoogleCalendar(userId: string) {
  // Remember the choice first, so nothing reconnects in the background even
  // if revoking at Google fails below.
  saveLocal(disconnectedKey(userId), "true");
  // Revoke with the stored token (even an expired one is worth trying).
  let revoked = false;
  try {
    const raw = await window.iris.auth.secureGet(tokenKey(userId));
    const stored = raw ? (JSON.parse(raw) as StoredToken) : null;
    if (stored?.token) revoked = await window.iris.google.revoke(stored.token);
  } catch {
    /* fall through to the Edge Function */
  }
  try {
    const { error } = await getSupabase().functions.invoke(tokenFunction, {
      body: { action: "disconnect" },
    });
    if (!error) revoked = true;
  } catch {
    /* the Edge Function may not be deployed */
  }
  await clearGoogleToken(userId);
  return { revoked };
}
