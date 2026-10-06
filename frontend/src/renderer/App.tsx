import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import {
  authRedirectUrl,
  getSupabase,
  isSupabaseConfigured,
  oauthRedirectUrl,
} from "./utils/supabase";
import {
  clearGoogleToken,
  googleCalendarScope,
  ipcErrorMessage,
  isGoogleAuthError,
  loadGoogleToken,
  normalizeMeetUrl,
  saveGoogleToken,
} from "./utils/google";
import {
  friendlyAuthError,
  passwordProblem,
  type AuthNotice,
} from "./utils/auth";
import {
  DEFAULT_IRIS_COLOR,
  googleIdFor,
  type EventColorKey,
} from "./utils/colors";
import HomePage from "./pages/tsx/HomePage";
import LoginPage from "./pages/tsx/LoginPage";
import CreateAccountPage from "./pages/tsx/CreateAccountPage";
import ForgotPasswordPage from "./pages/tsx/ForgotPasswordPage";
import ResetPasswordPage from "./pages/tsx/ResetPasswordPage";

export type Meeting = {
  id: string;
  title: string;
  starts_at: string;
  ends_at: string;
  meeting_url: string | null;
  accessibility_mode: string;
  color: string;
  description: string | null;
  google_event_id: string | null;
};
type Profile = { display_name: string | null; avatar_path: string | null };
const displayNameFor = (user: Session["user"]) => {
  const metadata = user.user_metadata as Record<string, unknown>;
  const fullName =
    metadata.full_name ||
    metadata.name ||
    [metadata.given_name, metadata.family_name].filter(Boolean).join(" ");
  if (typeof fullName === "string" && fullName.trim()) return fullName.trim();
  const localPart = user.email?.split("@")[0] || "there";
  return localPart
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};
const weekStartFor = (date: Date) => {
  const start = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  start.setDate(start.getDate() - ((start.getDay() + 6) % 7));
  return start;
};
const meetingColumns =
  "id,title,starts_at,ends_at,meeting_url,accessibility_mode,color,description,google_event_id";
export type MeetingDraft = {
  title: string;
  startsAt: Date;
  endsAt: Date;
  meetingUrl: string;
  generateMeetLink: boolean;
  accessibility: string;
};

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authNotice, setAuthNotice] = useState<AuthNotice | null>(null);
  const [authView, setAuthView] = useState<"signin" | "signup" | "forgot">(
    "signin",
  );
  // True after a password reset link signs the user in: ask for a new password.
  const [recovering, setRecovering] = useState(false);
  const [busy, setBusy] = useState(false);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [weekStart, setWeekStart] = useState(() => weekStartFor(new Date()));
  const [showCreate, setShowCreate] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [joinUrl, setJoinUrl] = useState("");
  const [now, setNow] = useState(() => new Date());
  const [notice, setNotice] = useState("");
  const [googleConnected, setGoogleConnected] = useState(false);
  const [meetingBusy, setMeetingBusy] = useState(false);
  const oauthProvider = useRef<"google" | "azure" | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const handledCallbackUrls = useRef(new Set<string>());
  const configured = isSupabaseConfigured();
  useEffect(() => {
    // Tick exactly on each minute boundary (rather than 60s after launch) so the
    // clock never shows the previous minute, and resync after sleep or focus.
    let timer: number;
    const tick = () => {
      const current = new Date();
      setNow(current);
      timer = window.setTimeout(
        tick,
        60_000 - (current.getSeconds() * 1000 + current.getMilliseconds()) + 50,
      );
    };
    const resync = () => {
      clearTimeout(timer);
      tick();
    };
    tick();
    window.addEventListener("focus", resync);
    document.addEventListener("visibilitychange", resync);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", resync);
      document.removeEventListener("visibilitychange", resync);
    };
  }, []);
  useEffect(() => {
    if (!configured || !window.iris?.auth) return;
    const completeCallback = async (url: string) => {
      if (handledCallbackUrls.current.has(url)) return;
      handledCallbackUrls.current.add(url);
      const callback = new URL(url);
      const code = callback.searchParams.get("code");
      const flowId = callback.searchParams.get("sb_flow_id");
      const linkError =
        callback.searchParams.get("error_description") ||
        callback.searchParams.get("error");
      if (!code)
        return setAuthNotice({
          text: linkError
            ? `${linkError}. Request a new link and try again.`
            : "That link didn't include a sign-in code. Request a new one.",
          tone: "error",
        });
      setBusy(true);
      const { data, error } = await getSupabase().auth.exchangeCodeForSession(
        code,
        flowId ? { flowId } : undefined,
      );
      setBusy(false);
      if (data.session) {
        // Supabase hands back Google's access token only on this exchange.
        const provider =
          oauthProvider.current ?? data.session.user.app_metadata.provider;
        if (provider === "google" && data.session.provider_token) {
          await saveGoogleToken(data.session.provider_token);
          setGoogleConnected(true);
          setNotice("Google connected. Iris can now create Meet links.");
        }
        oauthProvider.current = null;
        setSession(data.session);
      }
      if (error)
        setAuthNotice({ text: friendlyAuthError(error), tone: "error" });
    };
    const unsubscribe = window.iris.auth.onCallback(completeCallback);
    void window.iris.auth.consumeCallback().then((url) => {
      if (url) void completeCallback(url);
    });
    return unsubscribe;
  }, [configured]);
  useEffect(() => {
    if (!configured) return;
    const supabase = getSupabase();
    void supabase.auth
      .getSession()
      .then(({ data }) => setSession(data.session));
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, next) => {
      if (event === "PASSWORD_RECOVERY") setRecovering(true);
      if (event === "SIGNED_OUT") setRecovering(false);
      setSession(next);
    });
    return () => subscription.unsubscribe();
  }, [configured]);
  useEffect(() => {
    if (!session || !configured) return;
    const fallback = displayNameFor(session.user);
    void loadGoogleToken().then((token) => setGoogleConnected(Boolean(token)));
    void (async () => {
      const supabase = getSupabase();
      const { data } = await supabase
        .from("profiles")
        .select("display_name,avatar_path")
        .eq("id", session.user.id)
        .maybeSingle();
      const existingName = data?.display_name?.trim();
      const emailUsername = session.user.email?.split("@")[0];
      const shouldUpgradeName = Boolean(
        existingName &&
        emailUsername &&
        existingName.toLowerCase() === emailUsername.toLowerCase() &&
        existingName !== fallback,
      );
      const next = {
        display_name:
          shouldUpgradeName || !existingName ? fallback : existingName,
        avatar_path: data?.avatar_path || null,
      };
      if (!data || shouldUpgradeName)
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
        .select(meetingColumns)
        .order("starts_at");
      if (error) setNotice(error.message);
      else setMeetings(records || []);
    })();
  }, [configured, session]);
  function showAuthView(view: "signin" | "signup" | "forgot") {
    setAuthView(view);
    setAuthNotice(null);
    setPassword("");
  }
  async function resendConfirmation() {
    setBusy(true);
    const { error } = await getSupabase().auth.resend({
      type: "signup",
      email,
      options: { emailRedirectTo: authRedirectUrl },
    });
    setBusy(false);
    setAuthNotice(
      error
        ? { text: friendlyAuthError(error), tone: "error" }
        : { text: `We sent a new confirmation link to ${email}.`, tone: "success" },
    );
  }
  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!configured) return;
    setBusy(true);
    setAuthNotice(null);
    const { error } = await getSupabase().auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    setBusy(false);
    if (!error) return setPassword("");
    setAuthNotice({
      text: friendlyAuthError(error),
      tone: "error",
      action:
        error.code === "email_not_confirmed"
          ? { label: "Resend confirmation email", run: () => void resendConfirmation() }
          : error.code === "invalid_credentials"
            ? { label: "Forgot password?", run: () => showAuthView("forgot") }
            : undefined,
    });
  }
  async function signUp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!configured) return;
    const confirm = String(new FormData(event.currentTarget).get("confirm") ?? "");
    const problem = passwordProblem(password, confirm);
    if (problem) return setAuthNotice({ text: problem, tone: "error" });
    setBusy(true);
    setAuthNotice(null);
    const { data, error } = await getSupabase().auth.signUp({
      email: email.trim(),
      password,
      options: { emailRedirectTo: authRedirectUrl },
    });
    setBusy(false);
    // Supabase does not reveal existing accounts as an error when email
    // confirmation is on; it returns a user with no identities instead.
    const alreadyExists =
      error?.code === "user_already_exists" ||
      error?.code === "email_exists" ||
      (!error && data.user && data.user.identities?.length === 0);
    if (alreadyExists)
      return setAuthNotice({
        text: "An account with this email already exists. Sign in instead, or reset your password if you've forgotten it.",
        tone: "info",
        action: { label: "Forgot your password?", run: () => showAuthView("forgot") },
      });
    if (error)
      return setAuthNotice({ text: friendlyAuthError(error), tone: "error" });
    setPassword("");
    // With email confirmation off, Supabase signs the user straight in.
    if (data.session) return;
    setAuthNotice({
      text: `Almost there! We sent a confirmation link to ${email.trim()}. Open it to activate your account.`,
      tone: "success",
      action: { label: "Resend email", run: () => void resendConfirmation() },
    });
  }
  async function sendPasswordReset(target: string) {
    const { error } = await getSupabase().auth.resetPasswordForEmail(
      target.trim(),
      { redirectTo: authRedirectUrl },
    );
    return error ? friendlyAuthError(error) : null;
  }
  async function requestReset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    const error = await sendPasswordReset(email);
    setBusy(false);
    // Supabase responds the same way whether or not the account exists, so
    // the message stays neutral to avoid revealing who has an account.
    setAuthNotice(
      error
        ? { text: error, tone: "error" }
        : {
            text: `If an account exists for ${email.trim()}, a reset link is on its way. Open it on this computer to choose a new password.`,
            tone: "success",
          },
    );
  }
  async function updatePassword(next: string) {
    const { error } = await getSupabase().auth.updateUser({ password: next });
    return error ? friendlyAuthError(error) : null;
  }
  /** Settings: verify the current password (when there is one), then update it. */
  async function changePassword(current: string | null, next: string) {
    if (!session?.user.email) return "Your account has no email address.";
    if (current !== null) {
      const { error } = await getSupabase().auth.signInWithPassword({
        email: session.user.email,
        password: current,
      });
      if (error)
        return error.code === "invalid_credentials"
          ? "Your current password is incorrect."
          : friendlyAuthError(error);
    }
    return updatePassword(next);
  }
  async function oauth(provider: "google" | "azure") {
    if (!configured) return;
    oauthProvider.current = provider;
    const { data, error } = await getSupabase().auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: oauthRedirectUrl,
        skipBrowserRedirect: true,
        // Google sign-in also asks for Calendar access so Iris can create Meet links.
        ...(provider === "google" && {
          scopes: googleCalendarScope,
          queryParams: { include_granted_scopes: "true", prompt: "consent" },
        }),
      },
    });
    if (error || !data.url) {
      const text = error?.message || "Could not start sign-in.";
      setAuthNotice({ text, tone: "error" });
      setNotice(text);
      return;
    }
    await window.iris.auth.openExternal(data.url);
  }
  /** Grants Calendar access; links Google if this account signed in another way. */
  async function connectGoogle() {
    if (!session) return;
    const hasGoogle = session.user.identities?.some(
      (identity) => identity.provider === "google",
    );
    if (hasGoogle) return oauth("google");
    oauthProvider.current = "google";
    const { data, error } = await getSupabase().auth.linkIdentity({
      provider: "google",
      options: {
        redirectTo: oauthRedirectUrl,
        skipBrowserRedirect: true,
        scopes: googleCalendarScope,
        queryParams: { include_granted_scopes: "true", prompt: "consent" },
      },
    });
    if (error || !data.url)
      return setNotice(
        error?.message ||
          "Could not connect Google. Enable manual identity linking in Supabase.",
      );
    await window.iris.auth.openExternal(data.url);
    setNotice("Finish connecting Google in your browser, then return to Iris.");
  }
  async function signOut() {
    await clearGoogleToken();
    setGoogleConnected(false);
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
    if (error) return setNotice(error.message);
    const { error: profileError } = await supabase
      .from("profiles")
      .upsert({ id: session.user.id, avatar_path: path });
    if (profileError) return setNotice(profileError.message);
    const { data } = await supabase.storage
      .from("avatars")
      .createSignedUrl(path, 3600);
    setAvatarUrl(data?.signedUrl || null);
    event.target.value = "";
  }
  async function openMeeting(value: string) {
    const url = normalizeMeetUrl(value);
    if (!url) {
      setNotice(
        "Enter a Google Meet link (meet.google.com/abc-defg-hij) or a meeting code.",
      );
      return false;
    }
    try {
      await window.iris.meetings.openExternal(url);
      setNotice("");
      return true;
    } catch (error) {
      setNotice(ipcErrorMessage(error, "Could not open the meeting."));
      return false;
    }
  }
  /** Asks Google Calendar for an event with a fresh Meet link. */
  async function createMeetLink(title: string, startsAt: Date, endsAt: Date) {
    const token = await loadGoogleToken();
    if (!token) {
      setGoogleConnected(false);
      throw new Error(
        "Connect Google to generate a Meet link, or paste an existing link instead.",
      );
    }
    try {
      const created = await window.iris.google.createMeetEvent(token, {
        title,
        startsAt: startsAt.toISOString(),
        endsAt: endsAt.toISOString(),
        description: "Created with Iris — captions and sign translation.",
      });
      return created;
    } catch (error) {
      if (isGoogleAuthError(error)) {
        await clearGoogleToken();
        setGoogleConnected(false);
      }
      throw new Error(ipcErrorMessage(error, "Could not create a Meet link."));
    }
  }
  async function insertMeeting(
    title: string,
    startsAt: Date,
    endsAt: Date,
    meetingUrl: string | null,
    accessibility: string,
    googleEventId: string | null = null,
  ) {
    if (!session) return null;
    const { data, error } = await getSupabase()
      .from("meetings")
      .insert({
        owner_id: session.user.id,
        title,
        starts_at: startsAt.toISOString(),
        ends_at: endsAt.toISOString(),
        meeting_url: meetingUrl,
        accessibility_mode: accessibility,
        color: DEFAULT_IRIS_COLOR,
        google_event_id: googleEventId,
      })
      .select(meetingColumns)
      .single();
    if (error || !data) {
      setNotice(error?.message || "Could not save the meeting.");
      return null;
    }
    setMeetings((current) =>
      [...current, data].sort((a, b) => a.starts_at.localeCompare(b.starts_at)),
    );
    return data;
  }
  async function saveMeeting(draft: MeetingDraft) {
    if (!session || meetingBusy) return false;
    const pasted = draft.meetingUrl.trim();
    const pastedUrl = pasted ? normalizeMeetUrl(pasted) : null;
    if (pasted && !pastedUrl) {
      setNotice("That doesn't look like a Google Meet link.");
      return false;
    }
    setMeetingBusy(true);
    setNotice("");
    try {
      const created =
        !pastedUrl && draft.generateMeetLink
          ? await createMeetLink(draft.title, draft.startsAt, draft.endsAt)
          : null;
      const meetingUrl = pastedUrl || created?.meetUrl || null;
      const saved = await insertMeeting(
        draft.title,
        draft.startsAt,
        draft.endsAt,
        meetingUrl,
        draft.accessibility,
        created?.eventId ?? null,
      );
      if (!saved) return false;
      setNotice(
        meetingUrl
          ? `“${saved.title}” is scheduled with a Google Meet link.`
          : `“${saved.title}” is scheduled. Add a Meet link any time.`,
      );
      return true;
    } catch (error) {
      setNotice(ipcErrorMessage(error, "Could not create the meeting."));
      return false;
    } finally {
      setMeetingBusy(false);
    }
  }
  /** Reads Google Calendar events for the Calendar page; null if not connected. */
  async function listGoogleEvents(start: Date, end: Date) {
    const token = await loadGoogleToken();
    if (!token) {
      setGoogleConnected(false);
      return null;
    }
    try {
      return await window.iris.google.listEvents(
        token,
        start.toISOString(),
        end.toISOString(),
      );
    } catch (error) {
      if (isGoogleAuthError(error)) {
        await clearGoogleToken();
        setGoogleConnected(false);
        return null;
      }
      throw new Error(ipcErrorMessage(error, "Could not load Google Calendar."));
    }
  }
  /**
   * Recolours a calendar event: the Iris meeting row (if any) and the Google
   * Calendar event (if any). Returns an error message, or null on success.
   */
  async function setEventColor(
    target: { meetingId?: string; googleEventId?: string },
    color: EventColorKey,
  ) {
    if (target.meetingId) {
      const { error } = await getSupabase()
        .from("meetings")
        .update({ color })
        .eq("id", target.meetingId);
      if (error) return error.message;
      setMeetings((current) =>
        current.map((m) => (m.id === target.meetingId ? { ...m, color } : m)),
      );
    }
    if (!target.googleEventId) return null;
    // An Iris window started before this feature existed has an older preload.
    if (typeof window.iris.google.setEventColor !== "function")
      return "Restart Iris to finish updating, then change the colour again.";
    const token = await loadGoogleToken();
    if (!token) {
      setGoogleConnected(false);
      return target.meetingId
        ? "Saved in Iris. Reconnect Google to update Google Calendar too."
        : "Reconnect Google to change this event's colour.";
    }
    try {
      await window.iris.google.setEventColor(
        token,
        target.googleEventId,
        googleIdFor(color),
      );
      return null;
    } catch (error) {
      if (isGoogleAuthError(error)) {
        await clearGoogleToken();
        setGoogleConnected(false);
      }
      return ipcErrorMessage(error, "Could not update Google Calendar.");
    }
  }
  async function openInGoogleCalendar(url: string) {
    try {
      await window.iris.google.openEvent(url);
    } catch (error) {
      setNotice(ipcErrorMessage(error, "Could not open Google Calendar."));
    }
  }
  /**
   * Removes a meeting from Iris and, when Iris created it, from Google Calendar.
   * Google is tried first so a failure there leaves both copies in place.
   */
  async function deleteMeeting(meeting: Meeting, keepGoogleEvent = false) {
    if (!session) return false;
    setNotice("");
    let removedFromGoogle = false;
    if (meeting.google_event_id && !keepGoogleEvent) {
      const token = await loadGoogleToken();
      if (!token) {
        setGoogleConnected(false);
        setNotice(
          "Connect Google to also remove this meeting from your Google Calendar.",
        );
        return false;
      }
      try {
        await window.iris.google.deleteEvent(token, meeting.google_event_id);
        removedFromGoogle = true;
      } catch (error) {
        if (isGoogleAuthError(error)) {
          await clearGoogleToken();
          setGoogleConnected(false);
        }
        setNotice(ipcErrorMessage(error, "Could not remove the Google event."));
        return false;
      }
    }
    const { error } = await getSupabase()
      .from("meetings")
      .delete()
      .eq("id", meeting.id);
    if (error) {
      setNotice(error.message);
      return false;
    }
    setMeetings((current) => current.filter((m) => m.id !== meeting.id));
    setNotice(
      removedFromGoogle
        ? `“${meeting.title}” was deleted from Iris and Google Calendar.`
        : `“${meeting.title}” was deleted.`,
    );
    return true;
  }
  /** Starts a meeting right now and opens Google Meet's pre-join screen. */
  async function startInstantMeeting() {
    if (meetingBusy) return;
    const token = await loadGoogleToken();
    if (!token) {
      // Without Calendar access, Google can still create a meeting in the browser.
      await openMeeting("https://meet.google.com/new");
      return;
    }
    setMeetingBusy(true);
    setNotice("");
    try {
      const startsAt = new Date();
      const endsAt = new Date(startsAt.getTime() + 60 * 60_000);
      const title = "Instant Iris meeting";
      const { meetUrl, eventId } = await createMeetLink(
        title,
        startsAt,
        endsAt,
      );
      await insertMeeting(
        title,
        startsAt,
        endsAt,
        meetUrl,
        "Captions + Sign translation",
        eventId,
      );
      await openMeeting(meetUrl);
    } catch (error) {
      setNotice(ipcErrorMessage(error, "Could not start the meeting."));
    } finally {
      setMeetingBusy(false);
    }
  }
  if (!session) {
    if (authView === "forgot")
      return (
        <ForgotPasswordPage
          busy={busy}
          email={email}
          notice={authNotice}
          onBack={() => showAuthView("signin")}
          onEmail={setEmail}
          onSubmit={requestReset}
        />
      );
    if (authView === "signup")
      return (
        <CreateAccountPage
          busy={busy}
          email={email}
          notice={authNotice}
          onEmail={setEmail}
          onPassword={setPassword}
          onSubmit={signUp}
          onSwitch={() => showAuthView("signin")}
          password={password}
        />
      );
    return (
      <LoginPage
        busy={busy}
        email={email}
        notice={authNotice}
        onEmail={setEmail}
        onForgot={() => showAuthView("forgot")}
        onGoogle={() => void oauth("google")}
        onPassword={setPassword}
        onSubmit={signIn}
        onSwitch={() => showAuthView("signup")}
        password={password}
      />
    );
  }
  if (recovering)
    return (
      <ResetPasswordPage
        email={session.user.email ?? "your account"}
        onSave={async (next) => {
          const error = await updatePassword(next);
          if (!error) {
            setRecovering(false);
            setNotice("Your password has been updated.");
          }
          return error;
        }}
        onSkip={() => setRecovering(false)}
      />
    );
  const name = profile?.display_name || displayNameFor(session.user);
  return (
    <HomePage
      avatarUrl={avatarUrl}
      fileInput={fileInput}
      joinUrl={joinUrl}
      meetings={meetings}
      name={name}
      now={now}
      googleConnected={googleConnected}
      meetingBusy={meetingBusy}
      notice={notice}
      account={{
        email: session.user.email ?? "",
        hasPassword: Boolean(
          session.user.identities?.some((i) => i.provider === "email"),
        ),
        providers: [
          ...new Set(
            (session.user.identities ?? []).map((i) => i.provider),
          ),
        ],
      }}
      onAvatarChange={uploadAvatar}
      onChangePassword={changePassword}
      onSendPasswordReset={() => sendPasswordReset(session.user.email ?? "")}
      onConnectGoogle={() => void connectGoogle()}
      onCreate={saveMeeting}
      onDelete={deleteMeeting}
      onDismissNotice={() => setNotice("")}
      onInstantMeeting={() => void startInstantMeeting()}
      onJoin={openMeeting}
      onLoadGoogleEvents={listGoogleEvents}
      onOpenInGoogle={(url) => void openInGoogleCalendar(url)}
      onSetEventColor={setEventColor}
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
