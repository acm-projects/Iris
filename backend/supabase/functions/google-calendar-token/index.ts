// Supabase Edge Function: keeps each user's Google refresh token server-side
// and exchanges it for short-lived access tokens, so the Iris desktop app stays
// connected to Google Calendar without shipping the OAuth client secret.
//
// Secrets (Supabase → Edge Functions → Secrets):
//   GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET  — the same Google OAuth client
//   configured under Authentication → Sign In / Providers → Google.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided automatically.
//
// Body: { action: "save", refreshToken } | { action: "token" } | { action: "disconnect" }
import { createClient } from "jsr:@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  const admin = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    { auth: { persistSession: false } },
  );

  // Identify the caller from their Supabase session token.
  const jwt = req.headers.get("Authorization")?.replace(/^Bearer\s+/i, "");
  if (!jwt) return json({ error: "Not signed in" }, 401);
  const { data: auth, error: authError } = await admin.auth.getUser(jwt);
  if (authError || !auth.user) return json({ error: "Not signed in" }, 401);
  const userId = auth.user.id;

  let body: { action?: string; refreshToken?: string };
  try {
    body = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  if (body.action === "save") {
    if (typeof body.refreshToken !== "string" || body.refreshToken.length < 10)
      return json({ error: "Missing refresh token" }, 400);
    const { error } = await admin.from("google_connections").upsert({
      user_id: userId,
      refresh_token: body.refreshToken,
      updated_at: new Date().toISOString(),
    });
    return error ? json({ error: error.message }, 500) : json({ saved: true });
  }

  if (body.action === "disconnect") {
    const { data: row } = await admin
      .from("google_connections")
      .select("refresh_token")
      .eq("user_id", userId)
      .maybeSingle();
    // Revoking the refresh token removes Iris's Calendar access at Google too.
    if (row)
      await fetch("https://oauth2.googleapis.com/revoke", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ token: row.refresh_token }),
      }).catch(() => undefined);
    await admin.from("google_connections").delete().eq("user_id", userId);
    return json({ disconnected: true });
  }

  if (body.action === "token") {
    const { data: row } = await admin
      .from("google_connections")
      .select("refresh_token")
      .eq("user_id", userId)
      .maybeSingle();
    if (!row) return json({ error: "Google Calendar is not connected" }, 404);

    const response = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: Deno.env.get("GOOGLE_CLIENT_ID")!,
        client_secret: Deno.env.get("GOOGLE_CLIENT_SECRET")!,
        grant_type: "refresh_token",
        refresh_token: row.refresh_token,
      }),
    });
    const token = await response.json();
    if (!response.ok) {
      // The user revoked access or the token expired: forget it so the app
      // asks them to reconnect instead of retrying forever.
      if (token.error === "invalid_grant")
        await admin.from("google_connections").delete().eq("user_id", userId);
      return json({ error: token.error_description || token.error }, 410);
    }
    return json({ accessToken: token.access_token, expiresIn: token.expires_in });
  }

  return json({ error: "Unknown action" }, 400);
});
