// Supabase connection: database, auth, storage and Edge Functions all go
// through the single client created here.
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Project URL and publishable key come from frontend/.env.
const url = import.meta.env.VITE_SUPABASE_URL;
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

// Where email links (confirmation, password reset) send the user back to Iris.
export const authRedirectUrl =
  import.meta.env.VITE_SUPABASE_REDIRECT_URL || "iris://auth/callback";

// A loopback callback lets the browser show a clear completion page after
// OAuth while the Electron process securely receives the one-time code.
export const oauthRedirectUrl = "http://127.0.0.1:54321/auth/callback";

// One shared Supabase client for the whole app, created on first use.
let client: SupabaseClient | null = null;

/** True when the Supabase variables are present in .env. */
export function isSupabaseConfigured() {
  return Boolean(url && key);
}

/** Creates an Auth client whose session is persisted through Electron safeStorage. */
export function getSupabase() {
  if (!url || !key) {
    throw new Error(
      "Supabase is not configured. Add the Iris Supabase variables to .env.",
    );
  }

  if (!client) {
    client = createClient(url, key, {
      auth: {
        autoRefreshToken: true,
        detectSessionInUrl: false,
        flowType: "pkce",
        persistSession: true,
        storage: {
          getItem: (storageKey) => window.iris.auth.secureGet(storageKey),
          setItem: (storageKey, value) =>
            window.iris.auth.secureSet(storageKey, value),
          removeItem: (storageKey) =>
            window.iris.auth.secureSet(storageKey, null),
        },
      },
    });
  }

  return client;
}
