// Turns any error (Supabase, Google, Electron, the Python engine, the network)
// into a short sentence a user can act on. Every message shown on screen should
// go through `friendlyError`, so nobody sees stack traces or raw API errors.

/** Pulls a message out of anything that was thrown or returned as an error. */
function rawMessage(error: unknown) {
  if (typeof error === "string") return error;
  if (error && typeof error === "object" && "message" in error)
    return String((error as { message: unknown }).message ?? "");
  return "";
}

/** Known technical messages → plain language (first match wins). */
const TRANSLATIONS: [RegExp, string][] = [
  // Network
  [
    /failed to fetch|networkerror|network request failed|fetch failed|load failed|ENOTFOUND|ECONNREFUSED|ECONNRESET|ETIMEDOUT|ERR_INTERNET_DISCONNECTED/i,
    "Iris can't reach the internet right now. Check your connection and try again.",
  ],
  // Session / sign-in
  [
    /jwt expired|invalid jwt|refresh token|not authenticated|auth session missing/i,
    "Your session has expired. Sign out and sign back in, then try again.",
  ],
  // Database permissions and setup
  [
    /permission denied|row-level security|violates row level|42501/i,
    "You don't have permission to do that. If this keeps happening, sign out and back in.",
  ],
  [
    /schema cache|could not find the table|could not find the .* column|PGRST20[45]|does not exist/i,
    "Iris's database isn't fully set up yet. Ask your team to run the latest Supabase migrations.",
  ],
  [/duplicate key|already exists|23505/i, "That already exists."],
  // Uploads
  [
    /exceeded the maximum allowed size|payload too large|entity too large|413/i,
    "That file is too large. Choose an image under 5 MB.",
  ],
  [
    /mime type|invalid file type|not supported.*(image|file)/i,
    "That file type isn't supported. Use a PNG, JPEG or WebP image.",
  ],
  // Google
  [
    /insufficient authentication scopes|insufficientPermissions/i,
    "Iris needs permission to use your Google Calendar. Reconnect Google and allow calendar access.",
  ],
  [
    /rate limit|quota|too many requests|429/i,
    "Google is busy right now. Wait a minute and try again.",
  ],
  [
    /backend error|internal error|503|502|500/i,
    "The service had a problem. Please try again in a moment.",
  ],
  // Translation engine (Python)
  [
    /OBS Virtual Camera is not installed|'obs' backend|pyvirtualcam/i,
    "The OBS virtual camera isn't set up. Open OBS once, click Start Virtual Camera and approve it, then try again.",
  ],
  [
    /PortAudio|Invalid sample rate|Error opening OutputStream|sounddevice/i,
    "Iris couldn't play audio into the virtual microphone. Check that BlackHole (or VB-CABLE) is installed, then try again.",
  ],
  [
    /No module named|ModuleNotFoundError|ImportError/i,
    "The translation engine is missing some Python packages. Run the install command shown in the system check.",
  ],
  [
    /Could not start Python|spawn .*ENOENT/i,
    "Python couldn't be started. Install Python 3 and the packages in backend/requirements.txt.",
  ],
];

// Messages that look like code or internals rather than a sentence for people.
const TECHNICAL =
  /TypeError|ReferenceError|SyntaxError|undefined|null|NaN|Traceback|stack|\bat \w+ \(|[{}[\]<>]|0x[0-9a-f]+|\.(py|ts|js):\d+/i;

/**
 * A short, friendly message for any error. Readable messages we wrote
 * ourselves pass through; technical ones are translated or replaced by the
 * fallback (and logged to the console for debugging).
 */
export function friendlyError(error: unknown, fallback: string) {
  const message = rawMessage(error)
    // Electron wraps errors thrown in the main process.
    .replace(/^Error invoking remote method '[^']+': (Error: )?/, "")
    // Marker used by main.ts for "Google needs reconnecting" errors.
    .replace(/^GOOGLE_AUTH: /, "")
    .trim();
  if (!message) return fallback;
  for (const [pattern, friendly] of TRANSLATIONS)
    if (pattern.test(message)) {
      console.warn("[iris]", message);
      return friendly;
    }
  if (TECHNICAL.test(message) || message.length > 220) {
    console.error("[iris]", error);
    return fallback;
  }
  return message;
}
