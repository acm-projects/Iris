// Signed-in dashboard: sidebar, Home page, and the meeting dialogs.
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import type { Meeting, MeetingDraft } from "../../App";
import type { GoogleEvent } from "../../utils/google";
import {
  DEFAULT_IRIS_COLOR,
  colorHex,
  isEventColor,
  type EventColorKey,
} from "../../utils/colors";
import CalendarPage from "./CalendarPage";
import TranslationPage from "./TranslationPage";
import SettingsPage from "./SettingsPage";
import Sidebar from "../../components/Sidebar";
import "../css/calendar.css";
import "../css/home.css";
import Corner from "../../components/Corner";
import { asset } from "../../utils/assets";
import { useAutoDismiss } from "../../utils/useAutoDismiss";
import { readLocal, saveLocal } from "../../utils/storage";
import {
  ASL_ONLY,
  ASL_WITH_CAPTIONS,
  modeLabel,
} from "../../utils/meetingModes";

/** Everything App passes down: user data, meetings, Google state, and actions. */
type Props = {
  account: {
    id: string;
    email: string;
    hasPassword: boolean;
    providers: string[];
  };
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  joinUrl: string;
  googleChecked: boolean;
  googleConnected: boolean;
  googleConnecting: boolean;
  meetingBusy: boolean;
  meetings: Meeting[];
  name: string;
  notice: string;
  now: Date;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onChangePassword: (
    current: string | null,
    next: string,
  ) => Promise<string | null>;
  onSendPasswordReset: () => Promise<string | null>;
  onConnectGoogle: () => void;
  onDisconnectGoogle: () => Promise<{ revoked: boolean }>;
  onCreate: (draft: MeetingDraft) => Promise<boolean>;
  onDelete: (meeting: Meeting, keepGoogleEvent?: boolean) => Promise<boolean>;
  onDismissNotice: () => void;
  onInstantMeeting: () => void;
  onJoin: (url: string) => Promise<boolean>;
  onJoinUrlChange: (url: string) => void;
  onLoadGoogleEvents: (start: Date, end: Date) => Promise<GoogleEvent[] | null>;
  onOpenInGoogle: (url: string) => void;
  onSetEventColor: (
    target: { meetingId?: string; googleEventId?: string },
    color: EventColorKey,
  ) => Promise<string | null>;
  onSetEventTime: (
    target: { meetingId?: string; googleEventId?: string },
    startsAt: Date,
    endsAt: Date,
  ) => Promise<string | null>;
  onSignOut: () => void;
  setShowCreate: (open: boolean) => void;
  showCreate: boolean;
  setShowJoin: (open: boolean) => void;
  showJoin: boolean;
};
/** "YYYY-MM-DD" in local time, for comparing calendar days. */
const dateKey = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
/** Formats an ISO timestamp as a short local time, e.g. "2:30 PM". */
const time = (v: string) =>
  new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(v));

/** The full signed-in dashboard lives here; App.tsx supplies data and actions. */
/**
 * Signed-in shell: the sidebar plus whichever page is selected. The Home page
 * (meeting launcher + today's schedule) is rendered here; dialogs, the Google
 * Calendar prompt and the toast sit on top of every page.
 */
export default function HomePage(p: Props) {
  // Selected sidebar page.
  const [page, setPage] = useState("Home");
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  // Pre-fills the Create dialog when a day or time slot is clicked on the calendar.
  const [createAt, setCreateAt] = useState<Date | null>(null);
  // Start time of the meeting being drafted inline on Today's Timeline.
  const [composeAt, setComposeAt] = useState<Date | null>(null);
  // The meeting last joined from Iris (its mode is applied on Translation).
  const [joinedId, setJoinedId] = useState<string | null>(null);
  /** Opens a meeting link; remembers which Iris meeting it was, if any. */
  const joinMeeting = (url: string) => {
    // Match by Meet code, so a pasted code or a link written differently
    // still finds the scheduled meeting (and its ASL / captions mode).
    const code = meetCode(url);
    const meeting = code
      ? p.meetings.find((m) => meetCode(m.meeting_url ?? "") === code)
      : undefined;
    if (meeting) setJoinedId(meeting.id);
    return p.onJoin(url);
  };
  /** Opens the full Create dialog (other days, existing links), pre-filled. */
  const openCreateDialog = (start: Date | null) => {
    setComposeAt(null);
    setCreateAt(start);
    p.setShowCreate(true);
  };
  // Ask users without a Google connection to connect once per launch,
  // unless they chose "Don't ask me again" (remembered per account).
  const promptKey = `iris-google-prompt-dismissed:${p.account.id}`;
  const [promptClosed, setPromptClosed] = useState(
    () => readLocal(promptKey) === "true",
  );
  const showGooglePrompt =
    p.googleChecked && !p.googleConnected && !promptClosed;
  // Meeting waiting for delete confirmation.
  const [pendingDelete, setPendingDelete] = useState<Meeting | null>(null);
  // The toast fades away after a few seconds (hover keeps it open).
  const toastTimer = useAutoDismiss(p.notice, p.onDismissNotice);
  // Meetings that haven't ended, soonest first.
  const upcoming = p.meetings
    .filter((meeting) => new Date(meeting.ends_at) >= p.now)
    .sort((a, b) => a.starts_at.localeCompare(b.starts_at));
  // The meeting Translation should follow: the one joined from Iris (until it
  // ends), else one happening now (from 10 minutes before its start).
  const activeMeeting =
    upcoming.find((m) => m.id === joinedId) ??
    upcoming.find(
      (m) => new Date(m.starts_at).getTime() - 10 * 60_000 <= p.now.getTime(),
    ) ??
    null;
  return (
    <main className="dashboard-shell">
      {/* Left navigation */}
      <Sidebar
        active={page}
        avatarUrl={p.avatarUrl}
        email={p.account.email}
        fileInput={p.fileInput}
        name={p.name}
        onAvatarChange={p.onAvatarChange}
        onNavigate={setPage}
      />
      {/* Selected page: Calendar, Settings, a placeholder, or Home */}
      {page === "Calendar" ? (
        <CalendarPage
          googleConnected={p.googleConnected}
          googleConnecting={p.googleConnecting}
          meetings={p.meetings}
          now={p.now}
          onConnectGoogle={p.onConnectGoogle}
          onCreateAt={(start) => {
            setCreateAt(start);
            p.setShowCreate(true);
          }}
          onJoin={joinMeeting}
          onLoadGoogleEvents={p.onLoadGoogleEvents}
          onOpenInGoogle={p.onOpenInGoogle}
          onSetEventColor={p.onSetEventColor}
          onSetEventTime={p.onSetEventTime}
        />
      ) : page === "Settings" ? (
        <SettingsPage
          avatarUrl={p.avatarUrl}
          email={p.account.email}
          hasPassword={p.account.hasPassword}
          name={p.name}
          onChangePhoto={() => p.fileInput.current?.click()}
          onSignOut={p.onSignOut}
          providers={p.account.providers}
          googleConnected={p.googleConnected}
          googleConnecting={p.googleConnecting}
          onConnectGoogle={p.onConnectGoogle}
          onDisconnectGoogle={async () => {
            const result = await p.onDisconnectGoogle();
            // They chose to disconnect, so don't nag with the connect popup.
            setPromptClosed(true);
            saveLocal(promptKey, "true");
            return result;
          }}
          onChangePassword={p.onChangePassword}
          onSendReset={p.onSendPasswordReset}
        />
      ) : page === "Translation" ? (
        <TranslationPage
          meeting={
            activeMeeting && {
              id: activeMeeting.id,
              title: activeMeeting.title,
              mode: activeMeeting.accessibility_mode,
            }
          }
        />
      ) : (
        <section className="calendar-dashboard">
          <HomeAurora />
          {/* Clock banner across the top of the Home page */}
          <header className="iris-hero">
            <div>
              <strong>
                {new Intl.DateTimeFormat(undefined, {
                  hour: "numeric",
                  minute: "2-digit",
                }).format(p.now)}
              </strong>
              <span>
                {new Intl.DateTimeFormat(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                }).format(p.now)}
              </span>
            </div>
          </header>
          <div className="iris-control-center">
            {/* Left: start, join, shortcuts, and camera / mic status */}
            <section className="home-launch" aria-label="Meeting actions">
              {/* Start Meeting: the main button starts an instant meeting;
                  the arrow opens more options */}
              <div className="start-action-wrap start-split">
                <button
                  className="start-split-main"
                  disabled={p.meetingBusy}
                  onClick={() => p.onInstantMeeting()}
                  type="button"
                >
                  <span aria-hidden="true">+</span>
                  {p.meetingBusy ? "Starting…" : "Start Meeting"}
                </button>
                <button
                  aria-expanded={startMenuOpen}
                  aria-haspopup="menu"
                  aria-label="More ways to start a meeting"
                  className="start-split-more"
                  disabled={p.meetingBusy}
                  onClick={() => setStartMenuOpen((open) => !open)}
                  type="button"
                >
                  <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
                    <path
                      d="m6 9 6 6 6-6"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2.4"
                    />
                  </svg>
                </button>
                {startMenuOpen && (
                  <div
                    className="start-menu"
                    onMouseLeave={() => setStartMenuOpen(false)}
                    role="menu"
                  >
                    <button
                      onClick={() => {
                        setStartMenuOpen(false);
                        p.onInstantMeeting();
                      }}
                      role="menuitem"
                      type="button"
                    >
                      <LineIcon type="camera" />
                      Start an instant meeting
                    </button>
                    <button
                      onClick={() => {
                        setStartMenuOpen(false);
                        // Drafted right on the timeline; late at night the
                        // next slot is tomorrow, so use the full dialog.
                        const start = nextSlot(p.now);
                        if (dateKey(start) === dateKey(p.now))
                          setComposeAt(start);
                        else openCreateDialog(start);
                      }}
                      role="menuitem"
                      type="button"
                    >
                      <LineIcon type="calendar" />
                      Create a meeting for later
                    </button>
                  </div>
                )}
              </div>
              {/* Join right here: paste a Meet link or code */}
              <JoinInline
                joinUrl={p.joinUrl}
                onJoin={joinMeeting}
                onJoinUrlChange={p.onJoinUrlChange}
              />
              <div className="home-cards">
                <button
                  className="home-card"
                  onClick={() => setPage("Calendar")}
                  type="button"
                >
                  <ActionIcon type="calendar" />
                  <span>Schedule</span>
                </button>
                <button
                  className="home-card"
                  onClick={() => setPage("Translation")}
                  type="button"
                >
                  <ActionIcon type="translation" />
                  <span>Translation</span>
                </button>
              </div>
              <DeviceStatus />
            </section>
            {/* Right: today's meetings on an hour-by-hour timeline */}
            <TodayTimeline
              composeAt={composeAt}
              googleConnected={p.googleConnected}
              meetingBusy={p.meetingBusy}
              meetings={p.meetings}
              now={p.now}
              onCompose={setComposeAt}
              onConnectGoogle={p.onConnectGoogle}
              onCreate={p.onCreate}
              onDelete={(m) => setPendingDelete(m)}
              onJoin={joinMeeting}
              onMoreOptions={openCreateDialog}
              upcoming={upcoming}
            />
          </div>
        </section>
      )}
      {/* Dialogs and overlays, shown on top of any page */}
      {p.showJoin && <JoinDialog {...p} />}
      {p.showCreate && (
        <CreateDialog
          {...p}
          initialStart={createAt}
          setShowCreate={(open) => {
            if (!open) setCreateAt(null);
            p.setShowCreate(open);
          }}
        />
      )}
      {pendingDelete && (
        <DeleteDialog
          googleConnected={p.googleConnected}
          meeting={pendingDelete}
          onClose={() => setPendingDelete(null)}
          onConnectGoogle={p.onConnectGoogle}
          onDelete={p.onDelete}
        />
      )}
      {showGooglePrompt && (
        <GoogleCalendarPrompt
          connecting={p.googleConnecting}
          hasGoogle={p.account.providers.includes("google")}
          onClose={(remember) => {
            setPromptClosed(true);
            if (remember) saveLocal(promptKey, "true");
          }}
          onConnect={() => {
            setPromptClosed(true);
            p.onConnectGoogle();
          }}
        />
      )}
      {/* Toast message (bottom-right), with Connect Google when relevant */}
      {p.notice && (
        <Corner>
          <div className="iris-toast" role="status" {...toastTimer}>
            <span>{p.notice}</span>
            {!p.googleConnected && /Google/.test(p.notice) && (
              <button onClick={p.onConnectGoogle} type="button">
                Connect Google
              </button>
            )}
            <button
              aria-label="Dismiss"
              className="toast-close"
              onClick={p.onDismissNotice}
              type="button"
            >
              ×
            </button>
          </div>
        </Corner>
      )}
    </main>
  );
}
/** Brand icon for a launcher tile, drawn as a CSS mask so CSS can colour it. */
function ActionIcon({
  type,
}: {
  type: "camera" | "join" | "calendar" | "translation";
}) {
  // Brand icons from the Iris design file, served from public/action-icons.
  // Drawn as a CSS mask so each tile can choose its icon colour.
  return (
    <i
      aria-hidden="true"
      className="action-icon"
      style={{
        ["--icon" as string]: `url(${asset(`action-icons/${type}.svg`)})`,
      }}
    />
  );
}
/** Small outline icon (camera or calendar) for the Start Meeting menu. */
function LineIcon({ type }: { type: "camera" | "calendar" }) {
  return (
    <svg
      aria-hidden="true"
      className="line-icon"
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d={
          type === "camera"
            ? "M3 7.5A1.5 1.5 0 0 1 4.5 6h9A1.5 1.5 0 0 1 15 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 3 16.5zM15 10.5l5-3v9l-5-3"
            : "M4 6.5A1.5 1.5 0 0 1 5.5 5h13A1.5 1.5 0 0 1 20 6.5v12a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5zM4 10h16M8.5 3v4M15.5 3v4"
        }
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
/** The Meet code in a link or typed code ("abcdefghij"), or "" if none. */
function meetCode(text: string) {
  const match = text
    .toLowerCase()
    .match(/(?:^|\/|\s)([a-z]{3})-?([a-z]{4})-?([a-z]{3})(?:$|[/?#\s])/);
  return match ? match[1] + match[2] + match[3] : "";
}
/** True for a Google Meet link or a bare meeting code like "abc-defg-hij". */
function isMeetLink(text: string) {
  const value = text.trim().toLowerCase();
  return (
    /^[a-z]{3}-?[a-z]{4}-?[a-z]{3}$/.test(value) ||
    /(^|\/\/|\.)meet\.google\.com\/[a-z]{3}-?[a-z]{4}-?[a-z]{3}(\b|$)/.test(
      value,
    )
  );
}
/** "Join with a code or link…": a text box with a Join button (opens Meet). */
function JoinInline({
  joinUrl,
  onJoin,
  onJoinUrlChange,
}: {
  joinUrl: string;
  onJoin: (url: string) => Promise<boolean>;
  onJoinUrlChange: (url: string) => void;
}) {
  const [joining, setJoining] = useState(false);
  // A Meet link or code (abc-defg-hij) turns the Join button purple.
  const ready = isMeetLink(joinUrl);
  return (
    <form
      className={`join-inline ${ready ? "ready" : ""}`}
      onSubmit={async (event) => {
        event.preventDefault();
        if (!joinUrl.trim()) return;
        setJoining(true);
        const opened = await onJoin(joinUrl);
        setJoining(false);
        if (opened) onJoinUrlChange("");
      }}
    >
      {/* Visually hidden; screen readers still announce it */}
      <label className="visually-hidden" htmlFor="home-join">
        Join with a code or link
      </label>
      <div>
        <input
          id="home-join"
          onChange={(event) => onJoinUrlChange(event.target.value)}
          placeholder="Join with a code or link…"
          value={joinUrl}
        />
        <button disabled={joining || !joinUrl.trim()} type="submit">
          {joining ? "Opening…" : "Join"}
        </button>
      </div>
    </form>
  );
}
/**
 * Microphone and camera status under the shortcuts. Uses the browser's
 * permission state and the connected devices; updates when either changes.
 */
/**
 * Moving background behind the Home page: soft brand-coloured glows that
 * slowly swirl around the screen and lean a little toward the mouse.
 */
function HomeAurora() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches)
      return;
    let frame = 0;
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        // -1..1 from the centre of the window
        el.style.setProperty("--mx", String((e.clientX / innerWidth) * 2 - 1));
        el.style.setProperty("--my", String((e.clientY / innerHeight) * 2 - 1));
      });
    };
    window.addEventListener("pointermove", onMove);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
    };
  }, []);
  return (
    <div className="home-aurora" aria-hidden="true" ref={ref}>
      <div className="home-aurora-swirl">
        <i />
        <i />
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}

function DeviceStatus() {
  const [status, setStatus] = useState<"ok" | "blocked" | "missing" | null>(
    null,
  );
  useEffect(() => {
    let cancelled = false;
    const permissions: PermissionStatus[] = [];
    const check = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const hasMic = devices.some((d) => d.kind === "audioinput");
        const hasCam = devices.some((d) => d.kind === "videoinput");
        const states = await Promise.all(
          (["microphone", "camera"] as PermissionName[]).map((name) =>
            navigator.permissions
              .query({ name })
              .then((result) => {
                if (!permissions.includes(result)) {
                  permissions.push(result);
                  result.onchange = () => void check();
                }
                return result.state;
              })
              .catch(() => "prompt" as PermissionState),
          ),
        );
        if (cancelled) return;
        if (!hasMic || !hasCam) setStatus("missing");
        else if (states.every((state) => state === "granted")) setStatus("ok");
        else setStatus("blocked");
      } catch {
        if (!cancelled) setStatus(null);
      }
    };
    void check();
    navigator.mediaDevices?.addEventListener?.("devicechange", check);
    return () => {
      cancelled = true;
      navigator.mediaDevices?.removeEventListener?.("devicechange", check);
      for (const permission of permissions) permission.onchange = null;
    };
  }, []);
  if (!status) return null;
  const text =
    status === "ok"
      ? "Connected: Microphone & Camera"
      : status === "missing"
        ? "Microphone or camera not found"
        : "Allow microphone & camera in Translation";
  return (
    <div className="device-status" role="status">
      <span aria-hidden="true" className="device-icons">
        <svg fill="none" viewBox="0 0 24 24">
          <path
            d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"
            stroke="currentColor"
            strokeLinecap="round"
            strokeWidth="1.8"
          />
        </svg>
        <svg fill="currentColor" viewBox="0 0 24 24">
          <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h9A1.5 1.5 0 0 1 15 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 3 16.5zM16 10.5l5-3v9l-5-3z" />
        </svg>
      </span>
      <span className={`device-pill ${status}`}>
        <i aria-hidden="true" />
        {text}
      </span>
    </div>
  );
}
/** Height of one hour on the timeline, in pixels. */
const HOUR_PX = 56;
/**
 * "Today's Timeline": the hours of today down the left, a line at the current
 * time, and today's meetings placed at their times (Join / Copy link /
 * Delete). Clicking a free time opens the inline composer there. Scrolls to
 * the current time when it opens.
 */
function TodayTimeline({
  composeAt,
  googleConnected,
  meetingBusy,
  meetings,
  now,
  onCompose,
  onConnectGoogle,
  onCreate,
  onDelete,
  onJoin,
  onMoreOptions,
  upcoming,
}: {
  composeAt: Date | null;
  googleConnected: boolean;
  meetingBusy: boolean;
  meetings: Meeting[];
  now: Date;
  onCompose: (start: Date | null) => void;
  onConnectGoogle: () => void;
  onCreate: (draft: MeetingDraft) => Promise<boolean>;
  onDelete: (meeting: Meeting) => void;
  onJoin: (url: string) => Promise<boolean>;
  onMoreOptions: (start: Date) => void;
  upcoming: Meeting[];
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const today = meetings
    .filter((m) => dateKey(new Date(m.starts_at)) === dateKey(now))
    .sort((a, b) => a.starts_at.localeCompare(b.starts_at));
  // Minutes since midnight -> pixels from the top of the day.
  const minutes = (d: Date) => d.getHours() * 60 + d.getMinutes();
  const y = (d: Date) => (minutes(d) / 60) * HOUR_PX;
  // Start scrolled to about an hour before now (or the first meeting).
  useEffect(() => {
    const first = today[0] ? new Date(today[0].starts_at) : now;
    const target = Math.min(y(now), y(first)) - HOUR_PX;
    scroller.current?.scrollTo({ top: Math.max(target, 0) });
    // Only when the page opens, not on every clock tick.
  }, []);
  // Bring the composer into view when it opens.
  useEffect(() => {
    if (composeAt)
      scroller.current?.scrollTo({
        top: Math.max(y(composeAt) - HOUR_PX, 0),
        behavior: "smooth",
      });
    // Only when a new draft opens.
  }, [composeAt?.getTime()]);
  /** A click on an empty part of the day: draft a meeting at that half hour
   *  (or at the next free half hour if the time has already passed). */
  const composeFromClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (
      event.target !== event.currentTarget &&
      !(event.target as HTMLElement).classList.contains("timeline-hour")
    )
      return;
    const box = event.currentTarget.getBoundingClientRect();
    const minute = Math.floor(((event.clientY - box.top) / HOUR_PX) * 2) * 30;
    const start = new Date(now);
    start.setHours(0, minute, 0, 0);
    const earliest = nextSlot(now);
    const chosen = start < now ? earliest : start;
    if (dateKey(chosen) === dateKey(now)) onCompose(chosen);
    else onMoreOptions(chosen);
  };
  const next = upcoming.find(
    (m) => dateKey(new Date(m.starts_at)) !== dateKey(now),
  );
  return (
    <section className="today-timeline" aria-label="Today's timeline">
      <h2>Today's Timeline</h2>
      <div className="timeline-scroll" ref={scroller}>
        {/* Clicking an empty time drafts a meeting there */}
        <div
          className="timeline-day"
          onClick={composeFromClick}
          style={{ height: 24 * HOUR_PX }}
        >
          {/* Hour lines; labels every two hours */}
          {Array.from({ length: 24 }, (_, hour) => (
            <div
              className={`timeline-hour ${hour % 2 === 0 ? "labeled" : ""}`}
              key={hour}
              style={{ top: hour * HOUR_PX }}
            >
              {hour % 2 === 0 && (
                <span>
                  {new Intl.DateTimeFormat(undefined, {
                    hour: "numeric",
                  }).format(new Date(2000, 0, 1, hour))}
                </span>
              )}
            </div>
          ))}
          {/* Current time */}
          <div className="timeline-now" style={{ top: y(now) }}>
            <span>Now</span>
          </div>
          {/* Today's meetings, placed at their start time */}
          {today.map((m) => (
            <TimelineEvent
              key={m.id}
              meeting={m}
              now={now}
              onDelete={() => onDelete(m)}
              onJoin={onJoin}
              top={y(new Date(m.starts_at))}
              height={Math.max(
                ((new Date(m.ends_at).getTime() -
                  new Date(m.starts_at).getTime()) /
                  3_600_000) *
                  HOUR_PX,
                48,
              )}
            />
          ))}
          {/* The meeting being drafted inline */}
          {composeAt && (
            <TimelineComposer
              booked={today}
              googleConnected={googleConnected}
              key={composeAt.getTime()}
              meetingBusy={meetingBusy}
              onClose={() => onCompose(null)}
              onConnectGoogle={onConnectGoogle}
              onCreate={onCreate}
              start={composeAt}
            />
          )}
        </div>
      </div>
      {/* Nothing today: a calm message (and the next meeting, if any) */}
      {!today.length && !composeAt && (
        <div className="timeline-empty">
          <svg aria-hidden="true" fill="none" viewBox="0 0 48 48">
            <path
              d="M8 13a3 3 0 0 1 3-3h22a3 3 0 0 1 3 3v8M8 19h28M15 6v8M29 6v8M8 13v22a3 3 0 0 0 3 3h13"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
            <circle
              cx="35"
              cy="35"
              r="8"
              stroke="currentColor"
              strokeWidth="2"
            />
            <path
              d="M35 31v4l3 2"
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="2"
            />
          </svg>
          <p>{next ? "No meetings today" : "No upcoming meetings"}</p>
          {next && (
            <small>
              Next: {next.title} ·{" "}
              {new Intl.DateTimeFormat(undefined, {
                weekday: "short",
                hour: "numeric",
                minute: "2-digit",
              }).format(new Date(next.starts_at))}
            </small>
          )}
        </div>
      )}
    </section>
  );
}
/** Lengths the length pill cycles through, in minutes. Any other length is
 *  set by dragging the block's bottom edge (like Google Calendar). */
const COMPOSE_DURATIONS = [15, 30, 60];
/** "30 min", "1 hr", "1 hr 15", "2 hr". */
const durationLabel = (minutes: number) => {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (!h) return `${m} min`;
  return m ? `${h} hr ${m}` : `${h} hr`;
};
/** Drag (move or resize) moves the draft in steps of this many minutes. */
const DRAG_STEP = 5;
/**
 * Reads a typed time like "2:07 pm", "14:07", "2pm" or "207p" as minutes since
 * midnight. Without am/pm, 1–11 keeps the period of `near` (the current value).
 */
function parseTime(text: string, near: number): number | null {
  const m = text
    .trim()
    .toLowerCase()
    .match(/^(\d{1,2})(?::?(\d{2}))?\s*([ap])?\.?m?\.?$/);
  if (!m) return null;
  let hour = Number(m[1]);
  const minute = Number(m[2] ?? 0);
  if (minute > 59 || hour > 23) return null;
  if (m[3]) {
    if (hour < 1 || hour > 12) return null;
    hour = (hour % 12) + (m[3] === "p" ? 12 : 0);
  } else if (hour >= 1 && hour <= 11 && near >= 12 * 60) hour += 12;
  else if (hour === 12 && near < 12 * 60) hour = 0;
  return hour * 60 + minute;
}
/** A time you can click and type into; shows e.g. "2:07 AM" otherwise. */
function TimeField({
  label,
  minute,
  onChange,
}: {
  label: string;
  minute: number;
  onChange: (minute: number) => void;
}) {
  const shown = (() => {
    const d = new Date();
    d.setHours(0, minute, 0, 0);
    return time(d.toISOString());
  })();
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    if (text !== null) {
      const parsed = parseTime(text, minute);
      if (parsed !== null) onChange(parsed);
    }
    setText(null);
  };
  return (
    <input
      aria-label={`${label}. Type a time, or use the up and down arrow keys.`}
      className="compose-time-input"
      onBlur={commit}
      onChange={(event) => setText(event.target.value)}
      onFocus={(event) => {
        setText(shown);
        event.target.select();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault(); // don't submit the form
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          event.stopPropagation();
          setText(null);
          event.currentTarget.blur();
        } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
          event.preventDefault();
          setText(null);
          onChange(minute + (event.key === "ArrowUp" ? -DRAG_STEP : DRAG_STEP));
        }
      }}
      size={Math.max((text ?? shown).length, 4)}
      value={text ?? shown}
    />
  );
}
/**
 * Inline meeting composer on Today's Timeline (Notion Calendar style): a
 * glowing block at the chosen time. Type a name; drag the block up or down
 * to move it (15-minute steps) and its bottom edge to change the length; the
 * pills set the length, accessibility and Meet link. Saves through the same onCreate as the full Create dialog;
 * "More options" opens that dialog for other days or existing links.
 */
function TimelineComposer({
  booked,
  googleConnected,
  meetingBusy,
  onClose,
  onConnectGoogle,
  onCreate,
  start: initialStart,
}: {
  /** Today's meetings; the draft can't be created on top of one. */
  booked: Meeting[];
  googleConnected: boolean;
  meetingBusy: boolean;
  onClose: () => void;
  onConnectGoogle: () => void;
  onCreate: (draft: MeetingDraft) => Promise<boolean>;
  start: Date;
}) {
  const [title, setTitle] = useState("");
  const [start, setStart] = useState(initialStart);
  const [duration, setDuration] = useState(60);
  // ASL + Captions (true) or ASL only (false); saved with the meeting.
  const [captions, setCaptions] = useState(true);
  const [newLink, setNewLink] = useState(true);
  // Shown when "Create" is pressed while the draft overlaps a meeting.
  const [clash, setClash] = useState<Meeting | null>(null);
  const [shaking, setShaking] = useState(false);
  const clashTimer = useRef<number>(undefined);
  useEffect(() => () => window.clearTimeout(clashTimer.current), []);
  // Drag state for resizing from the bottom edge.
  const drag = useRef<{ y: number; minutes: number } | null>(null);
  // Drag state for moving the whole block (changes the start time).
  const move = useRef<{ y: number; minute: number } | null>(null);
  const startMinute = start.getHours() * 60 + start.getMinutes();
  // Earliest allowed start today: now, rounded up to the next 15 minutes.
  const firstMinute = (() => {
    const today = new Date();
    if (dateKey(today) !== dateKey(start)) return 0;
    return today.getHours() * 60 + today.getMinutes() + 1;
  })();
  /** Moves the start to `minute` (snapped, kept within today and not past). */
  const moveTo = (minute: number, step = DRAG_STEP) => {
    const snapped = Math.round(minute / step) * step;
    const latest = 24 * 60 - Math.min(duration, 24 * 60 - firstMinute);
    const clamped = Math.min(Math.max(snapped, firstMinute), latest);
    const next = new Date(start);
    next.setHours(0, clamped, 0, 0);
    setStart(next);
  };
  /** Pointer handlers that drag the block up/down (not from inputs or buttons). */
  const moveHandlers = {
    onPointerDown: (event: React.PointerEvent<HTMLElement>) => {
      if ((event.target as HTMLElement).closest("input, button")) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      move.current = { y: event.clientY, minute: startMinute };
    },
    onPointerMove: (event: React.PointerEvent<HTMLElement>) => {
      if (!move.current) return;
      moveTo(
        move.current.minute + ((event.clientY - move.current.y) / HOUR_PX) * 60,
      );
    },
    onPointerUp: () => {
      move.current = null;
    },
  };
  // A meeting can't run past midnight on the timeline.
  const maxDuration = Math.min(240, 24 * 60 - startMinute);
  const top = (startMinute / 60) * HOUR_PX;
  const height = (Math.min(duration, maxDuration) / 60) * HOUR_PX;
  const needsGoogle = newLink && !googleConnected;
  const end = new Date(start.getTime() + duration * 60_000);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!title.trim() || meetingBusy || needsGoogle) return;
    // Two meetings can't share the same time: shake the draft and say why.
    const overlap = booked.find(
      (m) => new Date(m.starts_at) < end && new Date(m.ends_at) > start,
    );
    if (overlap) {
      setClash(overlap);
      setShaking(false);
      requestAnimationFrame(() => setShaking(true)); // restart the shake
      window.clearTimeout(clashTimer.current);
      clashTimer.current = window.setTimeout(() => {
        setClash(null);
        setShaking(false);
      }, 4500);
      return;
    }
    setClash(null);
    const created = await onCreate({
      title: title.trim(),
      startsAt: start,
      endsAt: end,
      meetingUrl: "",
      generateMeetLink: newLink,
      accessibility: captions ? ASL_WITH_CAPTIONS : ASL_ONLY,
      color: "gold", // stays gold (with a soft glow) on the timeline
    });
    if (created) onClose();
  };
  return (
    <form
      aria-label="New meeting"
      className={`timeline-compose ${shaking ? "is-shaking" : ""}`}
      onAnimationEnd={(event) => {
        if (event.animationName === "compose-shake") setShaking(false);
      }}
      // Clicks inside the draft shouldn't start another draft.
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key === "Escape") onClose();
      }}
      onSubmit={submit}
      style={{ top }}
    >
      {/* Overlap warning, bottom-right of the window */}
      {clash && (
        <Corner>
          <div className="iris-toast iris-toast-warn" role="alert">
            <b aria-hidden="true">!</b>
            <span>
              <strong>{clash.title}</strong> is already at{" "}
              {time(clash.starts_at)} – {time(clash.ends_at)}. Drag the gold box
              to a free time.
            </span>
            <button
              aria-label="Dismiss"
              className="toast-close"
              onClick={() => setClash(null)}
              type="button"
            >
              ×
            </button>
          </div>
        </Corner>
      )}
      {/* The exact time range, with a handle to drag the end time */}
      <div
        aria-hidden="true"
        className="compose-range"
        style={{ height }}
        {...moveHandlers}
      >
        <span
          className="compose-resize"
          onPointerDown={(event) => {
            event.preventDefault();
            event.stopPropagation(); // resizing, not moving
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = { y: event.clientY, minutes: duration };
          }}
          onPointerMove={(event) => {
            if (!drag.current) return;
            const moved = ((event.clientY - drag.current.y) / HOUR_PX) * 60;
            const snapped =
              Math.round((drag.current.minutes + moved) / DRAG_STEP) *
              DRAG_STEP;
            setDuration(Math.min(Math.max(snapped, DRAG_STEP), maxDuration));
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
        />
      </div>
      {/* Drag the card (anywhere but the fields and buttons) to move it.
          The card always sits outside the gold block: just below it, or just
          above it when the block is near the end of the day. */}
      <div
        className="compose-card"
        style={
          startMinute + duration > 21 * 60
            ? { marginTop: 0, transform: "translateY(calc(-100% - 12px))" }
            : { marginTop: height + 14 }
        }
        {...moveHandlers}
      >
        <div className="compose-title">
          <input
            aria-label="Meeting name"
            autoFocus
            onChange={(event) => setTitle(event.target.value)}
            placeholder="Meeting name"
            value={title}
          />
          <button
            className="compose-create"
            disabled={!title.trim() || meetingBusy || needsGoogle}
            type="submit"
          >
            {meetingBusy ? "Creating…" : "Create"}
          </button>
          <button
            aria-label="Cancel"
            className="compose-close"
            onClick={onClose}
            type="button"
          >
            ×
          </button>
        </div>
        <div className="compose-pills">
          {/* Start – end: type exact times, or drag the block (5-min steps) */}
          <div
            className="compose-pill compose-time"
            title="Type a time (e.g. 2:07 pm) or drag the block"
          >
            <span aria-hidden="true">↕</span>
            <TimeField
              label="Start time"
              minute={startMinute}
              onChange={(minute) => moveTo(minute, 1)}
            />
            –
            <TimeField
              label="End time"
              minute={startMinute + duration}
              onChange={(minute) => {
                const length = minute - startMinute;
                if (length >= 5) setDuration(Math.min(length, maxDuration));
              }}
            />
          </div>
          <button
            className="compose-pill gold"
            onClick={() => {
              // 15 → 30 → 60 min; a custom (dragged) length goes back to 15.
              const i = COMPOSE_DURATIONS.indexOf(duration);
              const nextDuration =
                COMPOSE_DURATIONS[(i + 1) % COMPOSE_DURATIONS.length];
              setDuration(Math.min(nextDuration, maxDuration));
            }}
            title="15 min, 30 min or 1 hr. For any other length, drag the bottom edge."
            type="button"
          >
            {durationLabel(duration)} <small>⌄</small>
          </button>
          <button
            aria-pressed={captions}
            className="compose-pill on"
            onClick={() => setCaptions((v) => !v)}
            title="What Iris does in this meeting: translate signs to speech, with or without captions on your video"
            type="button"
          >
            {modeLabel(captions ? ASL_WITH_CAPTIONS : ASL_ONLY)}
          </button>
          <button
            className={`compose-pill ${newLink ? "on" : ""}`}
            onClick={() => setNewLink((v) => !v)}
            title={
              newLink
                ? "Creates a Google Meet link and a Google Calendar event"
                : "Add a link later"
            }
            type="button"
          >
            {newLink ? "+ New Meet link" : "Link later"}
          </button>
        </div>
        {needsGoogle && (
          <div className="compose-google">
            <span>Connect Google to create a Meet link.</span>
            <button onClick={onConnectGoogle} type="button">
              Connect Google
            </button>
          </div>
        )}
      </div>
    </form>
  );
}
/** One meeting on the timeline: title, time, Join / Copy link / Delete. */
function TimelineEvent({
  meeting,
  now,
  onDelete,
  onJoin,
  top,
  height,
}: {
  meeting: Meeting;
  now: Date;
  onDelete: () => void;
  onJoin: (url: string) => Promise<boolean>;
  top: number;
  height: number;
}) {
  const [copied, setCopied] = useState(false);
  // "Live" from 10 minutes before the start until the meeting ends.
  const live =
    new Date(meeting.starts_at).getTime() - 10 * 60_000 <= now.getTime() &&
    new Date(meeting.ends_at) > now;
  return (
    <article
      className={`timeline-event ${live ? "live" : ""} ${height < 64 ? "short" : ""} ${meeting.color === "gold" ? "gold" : ""}`}
      // Tinted with the colour chosen for this meeting on the calendar.
      style={{
        top,
        height,
        ["--accent" as string]: colorHex(
          isEventColor(meeting.color) ? meeting.color : DEFAULT_IRIS_COLOR,
        ),
      }}
    >
      <div className="timeline-event-text">
        <strong>{meeting.title}</strong>
        <small>
          {time(meeting.starts_at)} – {time(meeting.ends_at)} ·{" "}
          {modeLabel(meeting.accessibility_mode)}
          {live && <em>Live</em>}
        </small>
      </div>
      <div className="timeline-event-actions">
        {meeting.meeting_url && (
          <>
            <button
              className="join"
              onClick={() => void onJoin(meeting.meeting_url || "")}
              type="button"
            >
              Join
            </button>
            <button
              aria-label="Copy meeting link"
              className="icon"
              onClick={() =>
                void navigator.clipboard
                  .writeText(meeting.meeting_url || "")
                  .then(() => {
                    setCopied(true);
                    window.setTimeout(() => setCopied(false), 1800);
                  })
              }
              title={copied ? "Copied" : "Copy link"}
              type="button"
            >
              {copied ? (
                "✓"
              ) : (
                <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
                  <path
                    d="M9 9h10v10H9zM5 15V5h10"
                    stroke="currentColor"
                    strokeLinejoin="round"
                    strokeWidth="2"
                  />
                </svg>
              )}
            </button>
          </>
        )}
        <button
          aria-label={`Delete ${meeting.title}`}
          className="icon delete"
          onClick={onDelete}
          title="Delete meeting"
          type="button"
        >
          <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
            <path
              d="M4 7h16M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7V4h6v3"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        </button>
      </div>
    </article>
  );
}
/** Shared modal chrome: closes on Escape or a click outside the dialog. */
function Modal({
  corner = false,
  children,
  className,
  label,
  onClose,
  onSubmit,
}: {
  children: React.ReactNode;
  className?: string;
  /** Show as a card in the bottom-right corner instead of a centred dialog. */
  corner?: boolean;
  label: string;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  // Close on Escape.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const header = (
    <header>
      <h2>{label}</h2>
      <button aria-label="Close" onClick={onClose} type="button">
        ×
      </button>
    </header>
  );
  if (corner)
    return (
      <Corner>
        <form
          aria-label={label}
          className={`meeting-modal corner-card ${className ?? ""}`}
          onSubmit={onSubmit}
          role="alertdialog"
        >
          {header}
          {children}
        </form>
      </Corner>
    );
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form
        aria-label={label}
        aria-modal="true"
        className={`meeting-modal ${className ?? ""}`}
        onSubmit={onSubmit}
        role="dialog"
      >
        {header}
        {children}
      </form>
    </div>
  );
}
/** Popup asking the user to connect (or reconnect) Google Calendar. */
function GoogleCalendarPrompt({
  connecting,
  hasGoogle,
  onClose,
  onConnect,
}: {
  connecting: boolean;
  hasGoogle: boolean;
  onClose: (remember: boolean) => void;
  onConnect: () => void;
}) {
  const [remember, setRemember] = useState(false);
  return (
    <Modal
      className="google-prompt"
      corner
      label={
        hasGoogle ? "Reconnect Google Calendar" : "Connect Google Calendar"
      }
      onClose={() => onClose(remember)}
      onSubmit={(event) => {
        event.preventDefault();
        onConnect();
      }}
    >
      <p className="modal-copy">
        {hasGoogle
          ? "Your Google Calendar connection has expired. Reconnect to keep your calendar in sync."
          : "Link your Google account so Iris can work with your schedule."}
      </p>
      <ul className="google-prompt-list">
        <li>See your Google Calendar events in the Calendar tab</li>
        <li>Create meetings with Google Meet links in one click</li>
        <li>Keep colours and changes in sync both ways</li>
      </ul>
      <label className="google-prompt-remember">
        <input
          checked={remember}
          onChange={(event) => setRemember(event.target.checked)}
          type="checkbox"
        />
        Don't ask me again
      </label>
      <div className="modal-actions">
        <button
          className="modal-secondary"
          onClick={() => onClose(remember)}
          type="button"
        >
          Not now
        </button>
        <button
          autoFocus
          className="modal-connect"
          disabled={connecting}
          type="submit"
        >
          <span aria-hidden="true" className="google-mark" />
          {connecting ? "Opening Google…" : "Connect Google Calendar"}
        </button>
      </div>
    </Modal>
  );
}
/** Join a meeting: paste a Meet link or code and open it in the browser. */
function JoinDialog(p: Props) {
  const [joining, setJoining] = useState(false);
  return (
    <Modal
      className="join-modal"
      label="Join a meeting"
      onClose={() => {
        p.onJoinUrlChange("");
        p.setShowJoin(false);
      }}
      onSubmit={async (e) => {
        e.preventDefault();
        setJoining(true);
        const opened = await p.onJoin(p.joinUrl);
        setJoining(false);
        if (opened) {
          p.onJoinUrlChange("");
          p.setShowJoin(false);
        }
      }}
    >
      <p className="modal-copy">
        Paste a Google Meet link or meeting code. Meet opens in your default
        browser, where you can check your camera and mic before joining.
      </p>
      <label>
        Meeting link or code
        <input
          autoFocus
          onChange={(e) => p.onJoinUrlChange(e.target.value)}
          placeholder="meet.google.com/abc-defg-hij or abc-defg-hij"
          required
          value={p.joinUrl}
        />
      </label>
      <button className="modal-submit" disabled={joining} type="submit">
        {joining ? "Opening Google Meet…" : "Join meeting"}
      </button>
    </Modal>
  );
}
/**
 * Create meeting: name, date/time, duration, accessibility, and either a new
 * Meet link (via Google Calendar) or an existing/pasted one.
 */
function CreateDialog(p: Props & { initialStart?: Date | null }) {
  const [generate, setGenerate] = useState(true);
  // Past slots fall back to the next half hour today so the form stays valid.
  const start =
    p.initialStart && p.initialStart > p.now ? p.initialStart : null;
  const startTime = start
    ? `${String(start.getHours()).padStart(2, "0")}:${String(start.getMinutes()).padStart(2, "0")}`
    : nextHalfHour(p.now);
  const close = () => {
    if (!p.meetingBusy) p.setShowCreate(false);
  };
  return (
    <Modal
      label="Create meeting"
      onClose={close}
      onSubmit={async (event) => {
        event.preventDefault();
        const form = new FormData(event.currentTarget);
        const startsAt = new Date(`${form.get("date")}T${form.get("time")}`);
        if (isNaN(startsAt.getTime())) return;
        const created = await p.onCreate({
          title: String(form.get("title") || "").trim(),
          startsAt,
          endsAt: new Date(
            startsAt.getTime() + Number(form.get("duration") || 60) * 60_000,
          ),
          meetingUrl: generate ? "" : String(form.get("url") || ""),
          generateMeetLink: generate,
          accessibility: String(form.get("accessibility") || ASL_WITH_CAPTIONS),
        });
        if (created) p.setShowCreate(false);
      }}
    >
      <label>
        Meeting name
        <input autoFocus name="title" placeholder="Team sync" required />
      </label>
      <div className="modal-fields">
        <label>
          Date
          <input
            defaultValue={dateKey(start ?? p.now)}
            min={dateKey(p.now)}
            name="date"
            required
            type="date"
          />
        </label>
        <label>
          Time
          <input defaultValue={startTime} name="time" required type="time" />
        </label>
      </div>
      <div className="modal-fields">
        <label>
          Duration
          <select defaultValue="60" name="duration">
            <option value="15">15 minutes</option>
            <option value="30">30 minutes</option>
            <option value="45">45 minutes</option>
            <option value="60">1 hour</option>
            <option value="90">1.5 hours</option>
            <option value="120">2 hours</option>
          </select>
        </label>
        <label>
          Accessibility
          <select name="accessibility">
            <option value={ASL_WITH_CAPTIONS}>ASL + Captions</option>
            <option value={ASL_ONLY}>ASL only</option>
          </select>
        </label>
      </div>
      {/* Generate a Meet link vs. use an existing one */}
      <fieldset className="link-choice">
        <legend>Google Meet link</legend>
        <label className={generate ? "selected" : ""}>
          <input
            checked={generate}
            name="link-mode"
            onChange={() => setGenerate(true)}
            type="radio"
          />
          <span>
            <strong>Generate a new Meet link</strong>
            <small>
              {p.googleConnected
                ? "Adds the event to your Google Calendar"
                : "Requires connecting your Google account"}
            </small>
          </span>
        </label>
        <label className={!generate ? "selected" : ""}>
          <input
            checked={!generate}
            name="link-mode"
            onChange={() => setGenerate(false)}
            type="radio"
          />
          <span>
            <strong>Use an existing link</strong>
            <small>Paste one now, or leave blank to add later</small>
          </span>
        </label>
      </fieldset>
      {generate && !p.googleConnected && (
        <div className="google-connect">
          <span>Connect Google so Iris can create Meet links for you.</span>
          <button onClick={p.onConnectGoogle} type="button">
            Connect Google
          </button>
        </div>
      )}
      {!generate && (
        <label>
          Meeting link <small>Optional</small>
          <input name="url" placeholder="https://meet.google.com/..." />
        </label>
      )}
      <button
        className="modal-submit"
        disabled={p.meetingBusy || (generate && !p.googleConnected)}
        type="submit"
      >
        {p.meetingBusy ? "Creating meeting…" : "Create meeting"}
      </button>
    </Modal>
  );
}
/** Confirms deleting a meeting from Iris (and from Google Calendar if linked). */
function DeleteDialog({
  googleConnected,
  meeting,
  onClose,
  onConnectGoogle,
  onDelete,
}: {
  googleConnected: boolean;
  meeting: Meeting;
  onClose: () => void;
  onConnectGoogle: () => void;
  onDelete: (meeting: Meeting, keepGoogleEvent?: boolean) => Promise<boolean>;
}) {
  const [deleting, setDeleting] = useState(false);
  const inGoogle = Boolean(meeting.google_event_id);
  const remove = async (keepGoogleEvent: boolean) => {
    setDeleting(true);
    const deleted = await onDelete(meeting, keepGoogleEvent);
    setDeleting(false);
    if (deleted) onClose();
  };
  // Escape cancels.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !deleting) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleting, onClose]);
  return (
    <Corner>
      <form
        aria-label="Delete meeting?"
        className="delete-toast"
        onSubmit={(event) => {
          event.preventDefault();
          void remove(false);
        }}
        role="alertdialog"
      >
        <p>
          <span>
            Delete <strong>{meeting.title}</strong>?
          </span>
          <small>
            {new Intl.DateTimeFormat(undefined, {
              weekday: "short",
              month: "short",
              day: "numeric",
            }).format(new Date(meeting.starts_at))}
            , {time(meeting.starts_at)}
            {inGoogle ? " · also from Google Calendar" : ""}
          </small>
        </p>
        {inGoogle && !googleConnected && (
          <button
            className="delete-toast-link"
            onClick={onConnectGoogle}
            type="button"
          >
            Connect Google to remove it there too
          </button>
        )}
        {/* Delete underneath the message */}
        <button
          autoFocus
          className="delete-toast-confirm"
          disabled={deleting || (inGoogle && !googleConnected)}
          type="submit"
        >
          {deleting ? "Deleting…" : "Delete"}
        </button>
        <div className="delete-toast-row">
          <button disabled={deleting} onClick={onClose} type="button">
            Cancel
          </button>
          {inGoogle && (
            <button
              disabled={deleting}
              onClick={() => void remove(true)}
              type="button"
            >
              Iris only
            </button>
          )}
        </div>
      </form>
    </Corner>
  );
}
/** The next :00 or :30 after now, as a Date (may be tomorrow). */
function nextSlot(now: Date) {
  const next = new Date(now);
  next.setMinutes(now.getMinutes() < 30 ? 30 : 60, 0, 0);
  return next;
}
/** Default start time for new meetings ("HH:MM"): the next :00 or :30. */
function nextHalfHour(now: Date) {
  const next = nextSlot(now);
  return `${String(next.getHours()).padStart(2, "0")}:${String(next.getMinutes()).padStart(2, "0")}`;
}
