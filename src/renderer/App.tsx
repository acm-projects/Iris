import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import {
  authRedirectUrl,
  getSupabase,
  isSupabaseConfigured,
  oauthRedirectUrl,
} from "./lib/supabase";

type Provider = "google" | "azure";
type Profile = { display_name: string | null; avatar_path: string | null };
type Meeting = {
  id: string;
  title: string;
  starts_at: string;
  ends_at: string;
  meeting_url: string | null;
  accessibility_mode: string;
  color: string;
  description: string | null;
};
const navItems = [
  "Home",
  "Meetings",
  "Calendar",
  "Translation",
  "Analytics",
  "Settings",
];
const meetings = [
  ["9:00 AM", "Product Design Sync", "Zoom · Captions + Sign translation"],
  ["11:00 AM", "Client Check-in", "Teams · Captions + Sign translation"],
  ["2:00 PM", "Team Retro", "Google Meet · Captions"],
];
const greetingForHour = (hour: number) =>
  hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase() || "I";
const atStartOfDay = (date: Date) =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate());
const weekStartFor = (date: Date) => {
  const start = atStartOfDay(date);
  start.setDate(start.getDate() - ((start.getDay() + 6) % 7));
  return start;
};
const dateKey = (date: Date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
const formatTime = (iso: string) =>
  new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(iso));
const isMeetUrl = (value: string) => {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "meet.google.com";
  } catch {
    return false;
  }
};

export default function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isCreatingAccount, setIsCreatingAccount] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [now, setNow] = useState(() => new Date());
  const [weekStart, setWeekStart] = useState(() => weekStartFor(new Date()));
  const [meetingsForWeek, setMeetingsForWeek] = useState<Meeting[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [joinUrl, setJoinUrl] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const handledCallbackUrls = useRef(new Set<string>());
  const configured = isSupabaseConfigured();
  const authBridgeAvailable = Boolean(window.iris?.auth);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!configured || !authBridgeAvailable) return;
    const completeCallback = async (url: string) => {
      if (handledCallbackUrls.current.has(url)) return;
      handledCallbackUrls.current.add(url);
      const callbackUrl = new URL(url);
      const code = callbackUrl.searchParams.get("code");
      const flowId = callbackUrl.searchParams.get("sb_flow_id");
      if (!code)
        return setMessage(
          "The sign-in callback did not include an authorization code.",
        );
      setIsBusy(true);
      const { data, error } = await getSupabase().auth.exchangeCodeForSession(
        code,
        flowId ? { flowId } : undefined,
      );
      setIsBusy(false);
      if (data.session) setSession(data.session);
      setMessage(error ? error.message : "You are signed in to Iris.");
    };
    const unsubscribe = window.iris.auth.onCallback(completeCallback);
    if (typeof window.iris.auth.consumeCallback === "function")
      void window.iris.auth.consumeCallback().then((url) => {
        if (url) void completeCallback(url);
      });
    return unsubscribe;
  }, [authBridgeAvailable, configured]);
  useEffect(() => {
    if (!configured || !authBridgeAvailable) return;
    let active = true;
    const supabase = getSupabase();
    void supabase.auth.getSession().then(({ data }) => {
      if (active) setSession(data.session);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      if (active) setSession(nextSession);
    });
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [authBridgeAvailable, configured]);
  useEffect(() => {
    if (!session || !configured) {
      setProfile(null);
      setAvatarUrl(null);
      return;
    }
    let active = true;
    const defaultName =
      session.user.user_metadata.full_name ||
      session.user.user_metadata.name ||
      session.user.email?.split("@")[0] ||
      "there";
    void (async () => {
      const supabase = getSupabase();
      const { data } = await supabase
        .from("profiles")
        .select("display_name, avatar_path")
        .eq("id", session.user.id)
        .maybeSingle();
      const next = data || { display_name: defaultName, avatar_path: null };
      if (!data)
        await supabase
          .from("profiles")
          .upsert({ id: session.user.id, display_name: defaultName });
      if (!active) return;
      setProfile(next);
      if (next.avatar_path) {
        const { data: signed } = await supabase.storage
          .from("avatars")
          .createSignedUrl(next.avatar_path, 3600);
        if (active) setAvatarUrl(signed?.signedUrl || null);
      }
    })();
    return () => {
      active = false;
    };
  }, [configured, session]);
  async function handleSignIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email || !password)
      return setMessage("Enter your email address and password to continue.");
    if (!authBridgeAvailable || !configured)
      return setMessage("Supabase authentication is not configured yet.");
    setIsBusy(true);
    const { error } = isCreatingAccount
      ? await getSupabase().auth.signUp({
          email,
          password,
          options: { emailRedirectTo: authRedirectUrl },
        })
      : await getSupabase().auth.signInWithPassword({ email, password });
    setIsBusy(false);
    setMessage(
      error
        ? error.message
        : isCreatingAccount
          ? "Check your email to confirm your Iris account."
          : "You are signed in to Iris.",
    );
  }
  async function continueWith(provider: Provider) {
    if (!authBridgeAvailable || !configured)
      return setMessage("Supabase authentication is not configured yet.");
    setIsBusy(true);
    const { data, error } = await getSupabase().auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: oauthRedirectUrl,
        skipBrowserRedirect: true,
        scopes: provider === "azure" ? "email" : "email profile",
      },
    });
    if (error || !data.url) {
      setIsBusy(false);
      return setMessage(error?.message || "Could not start sign-in.");
    }
    await window.iris.auth.openExternal(data.url);
    setIsBusy(false);
    setMessage(
      "Continue sign-in in your browser. Iris will reopen automatically.",
    );
  }
  async function resetPassword() {
    if (!email)
      return setMessage(
        "Enter your email address first, then choose Forgot password.",
      );
    setIsBusy(true);
    const { error } = await getSupabase().auth.resetPasswordForEmail(email, {
      redirectTo: authRedirectUrl,
    });
    setIsBusy(false);
    setMessage(
      error
        ? error.message
        : "Password reset instructions have been sent if that account exists.",
    );
  }
  async function signOut() {
    setIsBusy(true);
    await getSupabase().auth.signOut();
    setIsBusy(false);
    setSession(null);
  }
  async function uploadAvatar(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !session) return;
    if (!file.type.startsWith("image/") || file.size > 5 * 1024 * 1024) {
      setMessage("Choose an image smaller than 5 MB.");
      return;
    }
    setIsBusy(true);
    const ext =
      file.name
        .split(".")
        .pop()
        ?.replace(/[^a-z0-9]/gi, "") || "jpg";
    const path = `${session.user.id}/avatar.${ext}`;
    const supabase = getSupabase();
    const { error: uploadError } = await supabase.storage
      .from("avatars")
      .upload(path, file, { upsert: true, contentType: file.type });
    if (uploadError) {
      setIsBusy(false);
      setMessage(`Photo upload failed: ${uploadError.message}`);
      return;
    }
    const { error: profileError } = await supabase
      .from("profiles")
      .upsert({ id: session.user.id, avatar_path: path });
    if (profileError) {
      setIsBusy(false);
      setMessage(
        `Photo uploaded, but could not save it to your profile: ${profileError.message}`,
      );
      return;
    }
    const { data, error: signedUrlError } = await supabase.storage
      .from("avatars")
      .createSignedUrl(path, 3600);
    setAvatarUrl(data?.signedUrl || null);
    setProfile((current) => ({
      display_name: current?.display_name || null,
      avatar_path: path,
    }));
    setIsBusy(false);
    setMessage(
      signedUrlError
        ? `Photo saved, but could not display it: ${signedUrlError.message}`
        : "Profile photo updated.",
    );
    event.target.value = "";
  }
  useEffect(() => {
    if (!session || !configured) return;
    const start = new Date();
    start.setFullYear(start.getFullYear() - 1);
    void getSupabase()
      .from("meetings")
      .select(
        "id,title,starts_at,ends_at,meeting_url,accessibility_mode,color,description",
      )
      .gte("starts_at", start.toISOString())
      .order("starts_at")
      .then(({ data, error }) => {
        if (error) setMessage(`Could not load meetings: ${error.message}`);
        else setMeetingsForWeek(data || []);
      });
  }, [configured, session, weekStart]);
  async function openMeeting(url: string) {
    if (!isMeetUrl(url)) {
      setMessage("Enter a valid https://meet.google.com link.");
      return false;
    }
    try {
      await window.iris.meetings.openExternal(url);
      return true;
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not open the meeting.",
      );
      return false;
    }
  }
  async function saveMeeting(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) return;
    const form = new FormData(event.currentTarget);
    const title = String(form.get("title") || "").trim();
    const date = String(form.get("date") || "");
    const time = String(form.get("time") || "");
    const url = String(form.get("url") || "").trim();
    const id = String(form.get("id") || "");
    const duration = Number(form.get("duration") || 60);
    if (!title || !date || !time || (url && !isMeetUrl(url)))
      return setMessage(
        "Add a title, date and time. A meeting link is optional, but must be a valid Google Meet link.",
      );
    const startsAt = new Date(`${date}T${time}`);
    const endsAt = new Date(startsAt.getTime() + duration * 60_000);
    const meeting = {
      owner_id: session.user.id,
      title,
      starts_at: startsAt.toISOString(),
      ends_at: endsAt.toISOString(),
      meeting_url: url || null,
      accessibility_mode: String(
        form.get("accessibility") || "Captions + Sign translation",
      ),
      color: String(form.get("color") || "blue"),
      description: String(form.get("description") || "").trim() || null,
    };
    const request = id
      ? getSupabase()
          .from("meetings")
          .update(meeting)
          .eq("id", id)
          .select()
          .single()
      : getSupabase().from("meetings").insert(meeting).select().single();
    const { data, error } = await request;
    if (error || !data)
      return setMessage(
        `Could not save meeting: ${error?.message || "No record returned."}`,
      );
    setMeetingsForWeek((current) =>
      id
        ? current.map((item) => (item.id === id ? data : item))
        : [...current, data].sort((a, b) =>
            a.starts_at.localeCompare(b.starts_at),
          ),
    );
    setMessage(id ? "Meeting updated." : "Meeting created.");
  }
  if (session) {
    const name =
      profile?.display_name ||
      session.user.user_metadata.full_name ||
      session.user.user_metadata.name ||
      session.user.email?.split("@")[0] ||
      "there";
    return (
      <Dashboard
        avatarUrl={avatarUrl}
        fileInput={fileInput}
        joinUrl={joinUrl}
        meetings={meetingsForWeek}
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
        setShowCreate={setShowCreate}
        showCreate={showCreate}
        setShowJoin={setShowJoin}
        showJoin={showJoin}
        weekStart={weekStart}
      />
    );
  }
  return (
    <main className="login-shell">
      <section className="brand-panel">
        <a className="wordmark" href="#sign-in">
          <img src="/iris-mark.png" alt="" />
          <span>Iris</span>
        </a>
        <div className="brand-copy">
          <h1>More accessible meetings for a more connected world.</h1>
          <p>
            Real-time sign language translation, live captions, and inclusive
            conversations.
          </p>
        </div>
      </section>
      <section className="sign-in-panel" id="sign-in">
        <form className="sign-in-card" onSubmit={handleSignIn}>
          <header>
            <p className="kicker">WELCOME TO IRIS</p>
            <h2>
              {isCreatingAccount ? "Create your account" : "Welcome back"}
            </h2>
            <p className="subtitle">
              {isCreatingAccount
                ? "Start making meetings more inclusive."
                : "Sign in to continue making meetings more inclusive."}
            </p>
          </header>
          <div className="field-group">
            <label htmlFor="email">Email</label>
            <input
              autoComplete="email"
              id="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              type="email"
              value={email}
            />
          </div>
          <div className="field-group">
            <label htmlFor="password">Password</label>
            <input
              autoComplete={
                isCreatingAccount ? "new-password" : "current-password"
              }
              id="password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              type="password"
              value={password}
            />
            {!isCreatingAccount && (
              <button
                className="forgot-password"
                onClick={() => void resetPassword()}
                type="button"
              >
                Forgot password?
              </button>
            )}
          </div>
          <button className="primary-button" disabled={isBusy} type="submit">
            {isBusy
              ? "Please wait…"
              : isCreatingAccount
                ? "Create account"
                : "Sign in"}
          </button>
          <p className="divider">
            <span>or continue with</span>
          </p>
          <div className="provider-buttons">
            <button
              disabled={isBusy}
              onClick={() => void continueWith("google")}
              type="button"
            >
              Continue with Google
            </button>
            <button
              disabled={isBusy}
              onClick={() => void continueWith("azure")}
              type="button"
            >
              Continue with Microsoft
            </button>
          </div>
          <p className="form-message" aria-live="polite">
            {message}
          </p>
          <p className="create-account">
            {isCreatingAccount
              ? "Already have an account?"
              : "Don't have an account?"}
            <button
              className="text-button"
              onClick={() => {
                setIsCreatingAccount(!isCreatingAccount);
                setMessage("");
              }}
              type="button"
            >
              {isCreatingAccount ? "Sign in" : "Create account"}
            </button>
          </p>
        </form>
      </section>
    </main>
  );
}

function Workspace({
  avatarUrl,
  fileInput,
  joinUrl,
  meetings,
  name,
  now,
  onAvatarChange,
  onJoin,
  onJoinUrlChange,
  onSave,
  setShowJoin,
  showJoin,
}: {
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  joinUrl: string;
  meetings: Meeting[];
  name: string;
  now: Date;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onJoin: (url: string) => Promise<boolean>;
  onJoinUrlChange: (value: string) => void;
  onSave: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  setShowJoin: (show: boolean) => void;
  showJoin: boolean;
}) {
  const [page, setPage] = useState<"home" | "calendar">("home");
  const [month, setMonth] = useState(
    new Date(now.getFullYear(), now.getMonth(), 1),
  );
  const [editor, setEditor] = useState<Partial<Meeting> | null>(null);
  const week = Array.from({ length: 7 }, (_, index) => {
    const date = weekStartFor(now);
    date.setDate(date.getDate() + index);
    return date;
  });
  const monthDays = Array.from(
    {
      length: new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate(),
    },
    (_, index) => new Date(month.getFullYear(), month.getMonth(), index + 1),
  );
  const homeItems = meetings.filter((meeting) =>
    week.some((day) => dateKey(new Date(meeting.starts_at)) === dateKey(day)),
  );
  const draftFor = (date: Date): Partial<Meeting> => ({
    title: "",
    starts_at: `${dateKey(date)}T09:00`,
    ends_at: `${dateKey(date)}T10:00`,
    accessibility_mode: "Captions + Sign translation",
    color: "blue",
    meeting_url: null,
    description: null,
  });
  const formatInput = (iso?: string) =>
    iso ? new Date(iso).toISOString().slice(0, 16) : "";
  return (
    <main className="dashboard-shell">
      <aside className="sidebar">
        <a className="wordmark" href="#home">
          <img src="/iris-mark.png" alt="" />
          <span>Iris</span>
        </a>
        <nav>
          {navItems.map((item) => (
            <button
              className={
                page === item.toLowerCase() ||
                (page === "home" && item === "Meetings")
                  ? "active"
                  : ""
              }
              key={item}
              onClick={() => {
                if (item === "Calendar") setPage("calendar");
                if (item === "Home" || item === "Meetings") setPage("home");
              }}
              type="button"
            >
              <span>
                {item === "Calendar"
                  ? "▦"
                  : item === "Home"
                    ? "⌂"
                    : item === "Meetings"
                      ? "▣"
                      : item === "Translation"
                        ? "⌁"
                        : "⚙"}
              </span>
              {item}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            className="user-chip"
            onClick={() => fileInput.current?.click()}
            type="button"
          >
            {avatarUrl ? (
              <img src={avatarUrl} alt="" />
            ) : (
              <span className="avatar-fallback">{initials(name)}</span>
            )}
            <span>
              <strong>{name}</strong>
              <small>Change photo</small>
            </span>
          </button>
          <input
            accept="image/png,image/jpeg,image/webp"
            className="visually-hidden"
            onChange={onAvatarChange}
            ref={fileInput}
            type="file"
          />
        </div>
      </aside>
      {page === "home" ? (
        <section className="calendar-dashboard">
          <header className="calendar-header">
            <div>
              <h1>
                {greetingForHour(now.getHours())}, {name}
              </h1>
              <p>
                {new Intl.DateTimeFormat(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                }).format(now)}
              </p>
            </div>
            <button className="join-button" onClick={() => setShowJoin(true)}>
              ▣ Join meeting
            </button>
          </header>
          <section className="week-card">
            <header>
              <h2>Your week</h2>
              <span>Quick view</span>
            </header>
            <div className="week-days">
              {week.map((day) => (
                <article
                  className={dateKey(day) === dateKey(now) ? "today" : ""}
                  key={dateKey(day)}
                >
                  <b>
                    {new Intl.DateTimeFormat(undefined, {
                      weekday: "short",
                    }).format(day)}
                  </b>
                  <strong>{day.getDate()}</strong>
                  {homeItems
                    .filter(
                      (item) =>
                        dateKey(new Date(item.starts_at)) === dateKey(day),
                    )
                    .map((item) => (
                      <button
                        className={`week-event event-${item.color || "blue"}`}
                        key={item.id}
                        onClick={() =>
                          item.meeting_url && void onJoin(item.meeting_url)
                        }
                        type="button"
                      >
                        <small>{formatTime(item.starts_at)}</small>
                        {item.title}
                      </button>
                    ))}
                </article>
              ))}
            </div>
          </section>
          <section className="home-note">
            <h2>Manage your calendar</h2>
            <p>
              Create, edit, color-code, and add a meeting link from the Calendar
              tab.
            </p>
            <button onClick={() => setPage("calendar")}>Open calendar</button>
          </section>
        </section>
      ) : (
        <section className="calendar-page">
          <header className="calendar-page-header">
            <div>
              <h1>Calendar</h1>
              <p>
                {new Intl.DateTimeFormat(undefined, {
                  month: "long",
                  year: "numeric",
                }).format(month)}
              </p>
            </div>
            <div>
              <button
                onClick={() =>
                  setMonth(
                    new Date(month.getFullYear(), month.getMonth() - 1, 1),
                  )
                }
              >
                ‹
              </button>
              <button
                onClick={() =>
                  setMonth(
                    new Date(month.getFullYear(), month.getMonth() + 1, 1),
                  )
                }
              >
                ›
              </button>
              <button
                className="create-button"
                onClick={() => setEditor(draftFor(now))}
              >
                ＋ New meeting
              </button>
            </div>
          </header>
          <div className="calendar-workspace">
            <section className="month-grid">
              <div className="month-weekdays">
                {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map(
                  (day) => (
                    <b key={day}>{day}</b>
                  ),
                )}
              </div>
              <div className="month-cells">
                {Array.from(
                  { length: (monthDays[0].getDay() + 6) % 7 },
                  (_, index) => (
                    <div className="month-cell muted" key={`blank-${index}`} />
                  ),
                )}
                {monthDays.map((day) => (
                  <button
                    className="month-cell"
                    key={dateKey(day)}
                    onClick={() => setEditor(draftFor(day))}
                    type="button"
                  >
                    <time>{day.getDate()}</time>
                    {meetings
                      .filter(
                        (item) =>
                          dateKey(new Date(item.starts_at)) === dateKey(day),
                      )
                      .map((item) => (
                        <span
                          className={`calendar-event event-${item.color || "blue"}`}
                          key={item.id}
                          onClick={(event) => {
                            event.stopPropagation();
                            setEditor(item);
                          }}
                        >
                          {formatTime(item.starts_at)} {item.title}
                        </span>
                      ))}
                  </button>
                ))}
              </div>
            </section>
            {editor && (
              <aside className="meeting-editor">
                <header>
                  <h2>{editor.id ? "Edit meeting" : "New meeting"}</h2>
                  <button onClick={() => setEditor(null)} type="button">
                    ×
                  </button>
                </header>
                <form
                  onSubmit={async (event) => {
                    await onSave(event);
                    setEditor(null);
                  }}
                >
                  <input name="id" type="hidden" value={editor.id || ""} />
                  <label>
                    Title
                    <input
                      defaultValue={editor.title || ""}
                      name="title"
                      placeholder="Untitled meeting"
                      required
                    />
                  </label>
                  <label>
                    Start
                    <input
                      defaultValue={formatInput(editor.starts_at)}
                      name="dateTime"
                      type="hidden"
                    />
                    <input
                      defaultValue={formatInput(editor.starts_at)}
                      name="starts"
                      onChange={() => undefined}
                      type="datetime-local"
                    />
                  </label>
                  <div className="editor-times">
                    <label>
                      Date
                      <input
                        defaultValue={(editor.starts_at || "").slice(0, 10)}
                        name="date"
                        required
                        type="date"
                      />
                    </label>
                    <label>
                      Time
                      <input
                        defaultValue={(editor.starts_at || "").slice(11, 16)}
                        name="time"
                        required
                        type="time"
                      />
                    </label>
                  </div>
                  <label>
                    Duration
                    <select
                      defaultValue={
                        Math.max(
                          15,
                          (new Date(editor.ends_at || "").getTime() -
                            new Date(editor.starts_at || "").getTime()) /
                            60000,
                        ) || 60
                      }
                      name="duration"
                    >
                      <option value="30">30 minutes</option>
                      <option value="60">1 hour</option>
                      <option value="90">1.5 hours</option>
                      <option value="120">2 hours</option>
                    </select>
                  </label>
                  <label>
                    Google Meet link <small>Optional — add it later</small>
                    <input
                      defaultValue={editor.meeting_url || ""}
                      name="url"
                      placeholder="https://meet.google.com/..."
                      type="url"
                    />
                  </label>
                  <label>
                    Color
                    <div className="color-picks">
                      {["blue", "purple", "green", "yellow", "coral"].map(
                        (color) => (
                          <label className={`color-dot ${color}`} key={color}>
                            <input
                              defaultChecked={
                                (editor.color || "blue") === color
                              }
                              name="color"
                              type="radio"
                              value={color}
                            />
                            <span />
                          </label>
                        ),
                      )}
                    </div>
                  </label>
                  <label>
                    Accessibility
                    <select
                      defaultValue={editor.accessibility_mode}
                      name="accessibility"
                    >
                      <option>Captions + Sign translation</option>
                      <option>Captions only</option>
                    </select>
                  </label>
                  <label>
                    Description
                    <textarea
                      defaultValue={editor.description || ""}
                      name="description"
                      placeholder="Notes for this meeting"
                    />
                  </label>
                  <button className="modal-submit" type="submit">
                    {editor.id ? "Save changes" : "Create meeting"}
                  </button>
                </form>
              </aside>
            )}
          </div>
        </section>
      )}
      {showJoin && (
        <div className="modal-backdrop">
          <form
            className="meeting-modal join-modal"
            onSubmit={async (event) => {
              event.preventDefault();
              if (await onJoin(joinUrl)) setShowJoin(false);
            }}
          >
            <header>
              <h2>Join a meeting</h2>
              <button onClick={() => setShowJoin(false)} type="button">
                ×
              </button>
            </header>
            <p>
              Paste your Google Meet link. Iris will use your default browser.
            </p>
            <label>
              Google Meet link
              <input
                autoFocus
                onChange={(event) => onJoinUrlChange(event.target.value)}
                pattern="https://meet\.google\.com/.*"
                placeholder="https://meet.google.com/abc-defg-hij"
                required
                type="url"
                value={joinUrl}
              />
            </label>
            <button className="modal-submit" type="submit">
              Join meeting
            </button>
          </form>
        </div>
      )}
    </main>
  );
}

function Dashboard({
  avatarUrl,
  fileInput,
  joinUrl,
  meetings: items,
  name,
  now,
  onAvatarChange,
  onCreate,
  onJoin,
  onJoinUrlChange,
  onShiftWeek,
  setShowCreate,
  showCreate,
  setShowJoin,
  showJoin,
  weekStart,
}: {
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  joinUrl: string;
  meetings: Meeting[];
  name: string;
  now: Date;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onCreate: (event: FormEvent<HTMLFormElement>) => void;
  onJoin: (url: string) => Promise<boolean>;
  onJoinUrlChange: (value: string) => void;
  onShiftWeek: (days: number) => void;
  setShowCreate: (show: boolean) => void;
  showCreate: boolean;
  setShowJoin: (show: boolean) => void;
  showJoin: boolean;
  weekStart: Date;
}) {
  const days = Array.from({ length: 7 }, (_, offset) => {
    const date = new Date(weekStart);
    date.setDate(date.getDate() + offset);
    return date;
  });
  const next =
    items.find((meeting) => new Date(meeting.starts_at) >= now) || items[0];
  return (
    <main className="dashboard-shell">
      <aside className="sidebar">
        <a className="wordmark" href="#home">
          <img src="/iris-mark.png" alt="" />
          <span>Iris</span>
        </a>
        <nav>
          {navItems.map((item) => (
            <a
              className={item === "Home" ? "active" : ""}
              href={`#${item.toLowerCase()}`}
              key={item}
            >
              <span>
                {item === "Home"
                  ? "⌂"
                  : item === "Meetings"
                    ? "▣"
                    : item === "Calendar"
                      ? "▦"
                      : item === "Translation"
                        ? "⌁"
                        : "⚙"}
              </span>
              {item}
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            className="user-chip"
            onClick={() => fileInput.current?.click()}
            title="Change profile photo"
          >
            {avatarUrl ? (
              <img src={avatarUrl} alt="" />
            ) : (
              <span className="avatar-fallback">{initials(name)}</span>
            )}
            <span>
              <strong>{name}</strong>
              <small>Change photo</small>
            </span>
          </button>
          <input
            accept="image/png,image/jpeg,image/webp"
            className="visually-hidden"
            onChange={onAvatarChange}
            ref={fileInput}
            type="file"
          />
        </div>
      </aside>
      <section className="calendar-dashboard" id="home">
        <header className="calendar-header">
          <div>
            <h1>
              {greetingForHour(now.getHours())}, {name}
            </h1>
            <p>
              {new Intl.DateTimeFormat(undefined, {
                weekday: "long",
                month: "long",
                day: "numeric",
                year: "numeric",
              }).format(now)}
            </p>
          </div>
          <div className="header-actions">
            <button onClick={() => setShowJoin(true)}>▣ Join meeting</button>
            <button
              className="create-button"
              onClick={() => setShowCreate(true)}
            >
              ＋ Create meeting
            </button>
          </div>
        </header>
        <section className="week-card">
          <header>
            <h2>Your week</h2>
            <div>
              <button onClick={() => onShiftWeek(-7)}>‹</button>
              <span>
                {new Intl.DateTimeFormat(undefined, {
                  month: "short",
                  day: "numeric",
                }).format(days[0])}{" "}
                –{" "}
                {new Intl.DateTimeFormat(undefined, {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                }).format(days[6])}
              </span>
              <button onClick={() => onShiftWeek(7)}>›</button>
            </div>
          </header>
          <div className="week-days">
            {days.map((day) => {
              const dayMeetings = items.filter(
                (meeting) =>
                  dateKey(new Date(meeting.starts_at)) === dateKey(day),
              );
              return (
                <article
                  className={dateKey(day) === dateKey(now) ? "today" : ""}
                  key={dateKey(day)}
                >
                  <b>
                    {new Intl.DateTimeFormat(undefined, {
                      weekday: "short",
                    }).format(day)}
                  </b>
                  <strong>{day.getDate()}</strong>
                  {dayMeetings.map((meeting) => (
                    <button
                      className="week-event"
                      key={meeting.id}
                      onClick={() => onJoin(meeting.meeting_url || "")}
                    >
                      <small>{formatTime(meeting.starts_at)}</small>
                      {meeting.title}
                    </button>
                  ))}
                  <button
                    className="add-day"
                    onClick={() => setShowCreate(true)}
                  >
                    ＋
                  </button>
                </article>
              );
            })}
          </div>
        </section>
        <div className="dashboard-row">
          <section className="next-card">
            <h2>Next meeting</h2>
            {next ? (
              <div className="next-content">
                <div className="calendar-icon">▣</div>
                <div>
                  <h3>{next.title}</h3>
                  <p>
                    ◷ {formatTime(next.starts_at)} – {formatTime(next.ends_at)}
                  </p>
                  <p>▣ Google Meet</p>
                  <p>◉ {next.accessibility_mode}</p>
                </div>
                <button onClick={() => onJoin(next.meeting_url || "")}>
                  ▣ Join meeting →
                </button>
              </div>
            ) : (
              <div className="empty-state">
                <p>No upcoming meeting this week.</p>
                <button onClick={() => setShowCreate(true)}>
                  Create your first meeting
                </button>
              </div>
            )}
          </section>
          <section className="iris-status">
            <h2>
              Iris status <em>✓ All systems ready</em>
            </h2>
            {[
              ["▣", "Camera", "Detected and tracking"],
              ["CC", "Live captions", "Ready"],
              ["⌁", "Sign to speech", "Ready"],
            ].map(([icon, title, detail]) => (
              <p key={title}>
                <i>{icon}</i>
                <span>
                  <b>{title}</b>
                  <small>{detail}</small>
                </span>
                ›
              </p>
            ))}
          </section>
        </div>
        <section className="today-card">
          <header>
            <h2>
              Today{" "}
              <small>
                {new Intl.DateTimeFormat(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                }).format(now)}
              </small>
            </h2>
            <button onClick={() => onShiftWeek(0)}>▦ Open calendar</button>
          </header>
          {items.filter(
            (meeting) => dateKey(new Date(meeting.starts_at)) === dateKey(now),
          ).length ? (
            items
              .filter(
                (meeting) =>
                  dateKey(new Date(meeting.starts_at)) === dateKey(now),
              )
              .map((meeting) => (
                <article key={meeting.id}>
                  <time>
                    {formatTime(meeting.starts_at)}
                    <small>{formatTime(meeting.ends_at)}</small>
                  </time>
                  <b>{meeting.title}</b>
                  <span>▣ Google Meet</span>
                  <span>◉ {meeting.accessibility_mode}</span>
                  <button onClick={() => onJoin(meeting.meeting_url || "")}>
                    Join
                  </button>
                </article>
              ))
          ) : (
            <p className="empty-today">Nothing scheduled for today.</p>
          )}
        </section>
      </section>
      {showJoin && (
        <div className="modal-backdrop" role="presentation">
          <form
            className="meeting-modal join-modal"
            onSubmit={async (event) => {
              event.preventDefault();
              if (await onJoin(joinUrl)) setShowJoin(false);
            }}
          >
            <header>
              <h2>Join a meeting</h2>
              <button
                onClick={() => setShowJoin(false)}
                type="button"
                aria-label="Close join meeting dialog"
              >
                ×
              </button>
            </header>
            <p>
              Paste your Google Meet link below. Iris will open it in your
              preferred browser.
            </p>
            <label>
              Google Meet link
              <input
                autoFocus
                onChange={(event) => onJoinUrlChange(event.target.value)}
                pattern="https://meet\.google\.com/.*"
                placeholder="https://meet.google.com/abc-defg-hij"
                required
                type="url"
                value={joinUrl}
              />
            </label>
            <button className="modal-submit" type="submit">
              Join meeting
            </button>
          </form>
        </div>
      )}
      {showCreate && (
        <div className="modal-backdrop">
          <form className="meeting-modal" onSubmit={onCreate}>
            <header>
              <h2>Create meeting</h2>
              <button onClick={() => setShowCreate(false)} type="button">
                ×
              </button>
            </header>
            <label>
              Meeting name
              <input name="title" required />
            </label>
            <div className="modal-fields">
              <label>
                Date
                <input
                  defaultValue={dateKey(now)}
                  name="date"
                  required
                  type="date"
                />
              </label>
              <label>
                Time
                <input defaultValue="09:00" name="time" required type="time" />
              </label>
            </div>
            <label>
              Google Meet link
              <input
                name="url"
                placeholder="https://meet.google.com/abc-defg-hij"
                required
                type="url"
              />
            </label>
            <div className="modal-fields">
              <label>
                Duration
                <select defaultValue="60" name="duration">
                  <option value="30">30 minutes</option>
                  <option value="60">1 hour</option>
                  <option value="90">1.5 hours</option>
                </select>
              </label>
              <label>
                Accessibility
                <select name="accessibility">
                  <option>Captions + Sign translation</option>
                  <option>Captions only</option>
                </select>
              </label>
            </div>
            <button className="modal-submit" type="submit">
              Create meeting
            </button>
          </form>
        </div>
      )}
    </main>
  );
}
function StatusCard() {
  return (
    <section className="card status-card">
      <h2>Translation status</h2>
      {[
        ["Camera", "Detected and tracking"],
        ["Live captions", "Active"],
        ["Sign to speech", "Active"],
        ["Audio output", "Speaking to meeting"],
      ].map(([title, description]) => (
        <p className="status-item" key={title}>
          <i />
          <span>
            <strong>{title}</strong>
            <small>{description}</small>
          </span>
        </p>
      ))}
      <em>ALL SYSTEMS READY</em>
    </section>
  );
}
function StatsCard() {
  return (
    <section className="card stats-card">
      <h2>This week</h2>
      <div>
        {[
          ["Accessible meetings", "8", "All supported"],
          ["Live caption time", "12h", "This week"],
          ["Caption accuracy", "98%", "Reliable"],
          ["Participation", "+35%", "More engagement"],
        ].map(([label, value, note]) => (
          <article key={label}>
            <small>{label}</small>
            <strong>{value}</strong>
            <span>{note}</span>
          </article>
        ))}
      </div>
    </section>
  );
}
