import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import {
  authRedirectUrl,
  getSupabase,
  isSupabaseConfigured,
  oauthRedirectUrl,
} from "./lib/supabase";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import CreateAccountPage from "./pages/CreateAccountPage";

export type Meeting = {
  id: string;
  title: string;
  starts_at: string;
  ends_at: string;
  meeting_url: string | null;
  accessibility_mode: string;
  color: string;
  description: string | null;
};
type Profile = { display_name: string | null; avatar_path: string | null };
const weekStartFor = (date: Date) => {
  const start = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  start.setDate(start.getDate() - ((start.getDay() + 6) % 7));
  return start;
};
const isMeetUrl = (value: string) => {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "meet.google.com";
  } catch {
    return false;
  }
};

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [weekStart, setWeekStart] = useState(() => weekStartFor(new Date()));
  const [showCreate, setShowCreate] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [joinUrl, setJoinUrl] = useState("");
  const [now, setNow] = useState(() => new Date());
  const fileInput = useRef<HTMLInputElement>(null);
  const configured = isSupabaseConfigured();
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!configured) return;
    const supabase = getSupabase();
    void supabase.auth
      .getSession()
      .then(({ data }) => setSession(data.session));
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => subscription.unsubscribe();
  }, [configured]);
  useEffect(() => {
    if (!session || !configured) return;
    const fallback =
      session.user.user_metadata.full_name ||
      session.user.user_metadata.name ||
      session.user.email?.split("@")[0] ||
      "there";
    void (async () => {
      const supabase = getSupabase();
      const { data } = await supabase
        .from("profiles")
        .select("display_name,avatar_path")
        .eq("id", session.user.id)
        .maybeSingle();
      const next = data || { display_name: fallback, avatar_path: null };
      if (!data)
        await supabase
          .from("profiles")
          .upsert({ id: session.user.id, display_name: fallback });
      setProfile(next);
      if (next.avatar_path) {
        const { data: signed } = await supabase.storage
          .from("avatars")
          .createSignedUrl(next.avatar_path, 3600);
        setAvatarUrl(signed?.signedUrl || null);
      }
      const { data: records, error } = await supabase
        .from("meetings")
        .select(
          "id,title,starts_at,ends_at,meeting_url,accessibility_mode,color,description",
        )
        .order("starts_at");
      if (error) setMessage(error.message);
      else setMeetings(records || []);
    })();
  }, [configured, session]);
  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!configured || !email || !password)
      return setMessage("Enter an email and password.");
    setBusy(true);
    const { error } = creating
      ? await getSupabase().auth.signUp({
          email,
          password,
          options: { emailRedirectTo: authRedirectUrl },
        })
      : await getSupabase().auth.signInWithPassword({ email, password });
    setBusy(false);
    setMessage(
      error
        ? error.message
        : creating
          ? "Check your email to confirm your account."
          : "Signed in.",
    );
  }
  async function oauth(provider: "google" | "azure") {
    if (!configured) return;
    const { data, error } = await getSupabase().auth.signInWithOAuth({
      provider,
      options: { redirectTo: oauthRedirectUrl, skipBrowserRedirect: true },
    });
    if (error || !data.url)
      return setMessage(error?.message || "Could not start sign-in.");
    await window.iris.auth.openExternal(data.url);
  }
  async function signOut() {
    await getSupabase().auth.signOut();
    setSession(null);
  }
  async function uploadAvatar(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !session) return;
    const path = `${session.user.id}/avatar.${file.name.split(".").pop() || "jpg"}`;
    const supabase = getSupabase();
    const { error } = await supabase.storage
      .from("avatars")
      .upload(path, file, { upsert: true, contentType: file.type });
    if (error) return setMessage(error.message);
    const { error: profileError } = await supabase
      .from("profiles")
      .upsert({ id: session.user.id, avatar_path: path });
    if (profileError) return setMessage(profileError.message);
    const { data } = await supabase.storage
      .from("avatars")
      .createSignedUrl(path, 3600);
    setAvatarUrl(data?.signedUrl || null);
    event.target.value = "";
  }
  async function openMeeting(url: string) {
    if (!isMeetUrl(url)) {
      setMessage("Enter a valid Google Meet link.");
      return false;
    }
    try {
      await window.iris.meetings.openExternal(url);
      return true;
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not open meeting.",
      );
      return false;
    }
  }
  async function saveMeeting(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) return;
    const form = new FormData(event.currentTarget);
    const title = String(form.get("title") || "").trim(),
      date = String(form.get("date") || ""),
      time = String(form.get("time") || ""),
      url = String(form.get("url") || "").trim();
    if (!title || !date || !time || (url && !isMeetUrl(url)))
      return setMessage(
        "Add a title, date and time; the Google Meet link is optional.",
      );
    const starts = new Date(`${date}T${time}`),
      ends = new Date(
        starts.getTime() + Number(form.get("duration") || 60) * 60000,
      );
    const { data, error } = await getSupabase()
      .from("meetings")
      .insert({
        owner_id: session.user.id,
        title,
        starts_at: starts.toISOString(),
        ends_at: ends.toISOString(),
        meeting_url: url || null,
        accessibility_mode: String(
          form.get("accessibility") || "Captions + Sign translation",
        ),
        color: "blue",
      })
      .select()
      .single();
    if (error || !data)
      return setMessage(error?.message || "Could not create meeting.");
    setMeetings((current) => [...current, data]);
    setShowCreate(false);
  }
  if (!session)
    return creating ? (
      <CreateAccountPage
        email={email}
        message={message}
        onEmail={setEmail}
        onPassword={setPassword}
        onSubmit={submitAuth}
        onSwitch={() => {
          setCreating(false);
          setMessage("");
        }}
        password={password}
        busy={busy}
      />
    ) : (
      <LoginPage
        email={email}
        message={message}
        onEmail={setEmail}
        onOAuth={oauth}
        onPassword={setPassword}
        onSubmit={submitAuth}
        onSwitch={() => {
          setCreating(true);
          setMessage("");
        }}
        password={password}
        busy={busy}
      />
    );
  const name =
    profile?.display_name || session.user.email?.split("@")[0] || "there";
  return (
    <HomePage
      avatarUrl={avatarUrl}
      fileInput={fileInput}
      joinUrl={joinUrl}
      meetings={meetings}
      name={name}
      now={now}
      onAvatarChange={uploadAvatar}
      onCreate={saveMeeting}
      onJoin={openMeeting}
      onJoinUrlChange={setJoinUrl}
      onShiftWeek={(days) =>
        setWeekStart((current) => {
          const next = new Date(current);
          next.setDate(next.getDate() + days);
          return next;
        })
      }
      onSignOut={() => void signOut()}
      setShowCreate={setShowCreate}
      showCreate={showCreate}
      setShowJoin={setShowJoin}
      showJoin={showJoin}
      weekStart={weekStart}
    />
  );
}
