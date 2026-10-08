// Signed-in dashboard: sidebar, Home page, and the meeting dialogs.
import { ChangeEvent, FormEvent, useEffect, useState } from "react";
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
import { asset } from "../../utils/assets";
import { useAutoDismiss } from "../../utils/useAutoDismiss";
import { readLocal, saveLocal } from "../../utils/storage";

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
  // Meetings that have not ended yet, and the ones happening today.
  // Meetings that haven't ended, soonest first.
  const upcoming = p.meetings
    .filter((meeting) => new Date(meeting.ends_at) >= p.now)
    .sort((a, b) => a.starts_at.localeCompare(b.starts_at));
  const selectedMeetings = upcoming.filter(
    (meeting) => dateKey(new Date(meeting.starts_at)) === dateKey(p.now),
  );
  return (
    <main className="dashboard-shell">
      {/* Left navigation */}
      <Sidebar
        active={page}
        avatarUrl={p.avatarUrl}
        fileInput={p.fileInput}
        name={p.name}
        onAvatarChange={p.onAvatarChange}
        onNavigate={setPage}
        onSignOut={p.onSignOut}
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
          onJoin={p.onJoin}
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
        <TranslationPage />
      ) : (
        <section className="calendar-dashboard">
          {/* Clock banner across the top of the Home page */}
          <header className="iris-hero">
            <span className="petal p1" />
            <span className="petal p2" />
            <span className="petal p3" />
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
            {/* Launcher: Start / Join / Schedule / Translation tiles */}
            <section className="iris-actions" aria-label="Meeting actions">
              <div className="start-action-wrap">
                <button
                  aria-expanded={startMenuOpen}
                  aria-haspopup="menu"
                  className="iris-action start"
                  disabled={p.meetingBusy}
                  onClick={() => setStartMenuOpen((open) => !open)}
                >
                  <b>
                    <ActionIcon type="camera" />
                  </b>
                  <span>
                    {p.meetingBusy ? "Starting…" : "Start meeting"}
                    <small aria-hidden="true">⌄</small>
                  </span>
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
                      <strong>Start an instant meeting</strong>
                      <small>Opens Google Meet now</small>
                    </button>
                    <button
                      onClick={() => {
                        setStartMenuOpen(false);
                        p.setShowCreate(true);
                      }}
                      role="menuitem"
                      type="button"
                    >
                      <strong>Create a meeting for later</strong>
                      <small>Schedule it with a Meet link</small>
                    </button>
                  </div>
                )}
              </div>
              <button
                className="iris-action join"
                onClick={() => p.setShowJoin(true)}
              >
                <b>
                  <ActionIcon type="join" />
                </b>
                <span>Join meeting</span>
              </button>
              <button
                className="iris-action schedule"
                onClick={() => setPage("Calendar")}
              >
                <b>
                  <ActionIcon type="calendar" />
                </b>
                <span>Schedule</span>
              </button>
              <button
                className="iris-action translation"
                onClick={() => setPage("Translation")}
              >
                <b>
                  <ActionIcon type="translation" />
                </b>
                <span>Translation</span>
              </button>
            </section>
            {/* Upcoming meetings (today's first); the list scrolls if long */}
            <section className="iris-schedule">
              {/* Title and list only when there is something to show;
                  otherwise the empty state fills the card, centred */}
              {upcoming.length > 0 && (
                <>
                  <h2 className="iris-schedule-title">
                    {selectedMeetings.length
                      ? "Today & upcoming"
                      : "Upcoming meetings"}
                    <small>{upcoming.length}</small>
                  </h2>
                  <div className="iris-schedule-list">
                    {upcoming.slice(0, 6).map((m) => (
                      <MeetingCard
                        key={m.id}
                        meeting={m}
                        now={p.now}
                        onDelete={() => setPendingDelete(m)}
                        onJoin={p.onJoin}
                        showDate={
                          dateKey(new Date(m.starts_at)) !== dateKey(p.now)
                        }
                      />
                    ))}
                  </div>
                </>
              )}
              {!upcoming.length && (
                <div className="schedule-empty">
                  <p>No upcoming meetings</p>
                  <button onClick={() => p.setShowCreate(true)} type="button">
                    ＋ Create a meeting
                  </button>
                </div>
              )}
            </section>
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
/** One meeting in today's schedule: time, code, Join / Copy link, delete. */
function MeetingCard({
  meeting,
  now,
  onDelete,
  onJoin,
  showDate,
}: {
  meeting: Meeting;
  now: Date;
  onDelete: () => void;
  onJoin: (url: string) => Promise<boolean>;
  showDate: boolean;
}) {
  const [copied, setCopied] = useState(false);
  // "Live now" from 10 minutes before the start until the meeting ends.
  const live =
    new Date(meeting.starts_at).getTime() - 10 * 60_000 <= now.getTime() &&
    new Date(meeting.ends_at) > now;
  return (
    <article
      className={`iris-meeting ${live ? "live" : ""}`}
      // The left edge always uses the colour chosen for this meeting on the calendar.
      style={{
        borderLeftColor: colorHex(
          isEventColor(meeting.color) ? meeting.color : DEFAULT_IRIS_COLOR,
        ),
      }}
    >
      <button
        aria-label={`Delete ${meeting.title}`}
        className="meeting-delete"
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
      <strong>
        {meeting.title}
        {live && <em>Live now</em>}
      </strong>
      <span>
        {showDate &&
          `${new Intl.DateTimeFormat(undefined, {
            weekday: "short",
            month: "short",
            day: "numeric",
          }).format(new Date(meeting.starts_at))} · `}
        {time(meeting.starts_at)} – {time(meeting.ends_at)}
      </span>
      <small>{meeting.accessibility_mode}</small>
      {meetingCode(meeting.meeting_url) && (
        <small className="meeting-code">
          Meeting code <code>{meetingCode(meeting.meeting_url)}</code>
        </small>
      )}
      {meeting.meeting_url ? (
        <div className="meeting-card-actions">
          <button
            onClick={() => void onJoin(meeting.meeting_url || "")}
            type="button"
          >
            Join with Iris
          </button>
          <button
            className="secondary"
            onClick={() =>
              void navigator.clipboard
                .writeText(meeting.meeting_url || "")
                .then(() => {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1800);
                })
            }
            type="button"
          >
            {copied ? "Copied ✓" : "Copy link"}
          </button>
        </div>
      ) : (
        <small className="no-link">No Google Meet link yet</small>
      )}
    </article>
  );
}
/** Shared modal chrome: closes on Escape or a click outside the dialog. */
function Modal({
  children,
  className,
  label,
  onClose,
  onSubmit,
}: {
  children: React.ReactNode;
  className?: string;
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
        <header>
          <h2>{label}</h2>
          <button aria-label="Close" onClick={onClose} type="button">
            ×
          </button>
        </header>
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
          accessibility: String(
            form.get("accessibility") || "Captions + Sign translation",
          ),
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
            <option>Captions + Sign translation</option>
            <option>Captions only</option>
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
  return (
    <Modal
      className="delete-modal"
      label="Delete meeting?"
      onClose={() => !deleting && onClose()}
      onSubmit={(event) => {
        event.preventDefault();
        void remove(false);
      }}
    >
      <p className="modal-copy">
        <strong>{meeting.title}</strong> on{" "}
        {new Intl.DateTimeFormat(undefined, {
          weekday: "short",
          month: "short",
          day: "numeric",
        }).format(new Date(meeting.starts_at))}{" "}
        at {time(meeting.starts_at)} will be removed
        {inGoogle ? " from Iris and your Google Calendar." : " from Iris."} This
        can't be undone.
      </p>
      {inGoogle && !googleConnected && (
        <div className="google-connect">
          <span>Connect Google to remove it from your calendar too.</span>
          <button onClick={onConnectGoogle} type="button">
            Connect Google
          </button>
        </div>
      )}
      <div className="modal-actions">
        {inGoogle && (
          <button
            className="modal-secondary"
            disabled={deleting}
            onClick={() => void remove(true)}
            type="button"
          >
            Remove from Iris only
          </button>
        )}
        <button
          autoFocus
          className="modal-danger"
          disabled={deleting || (inGoogle && !googleConnected)}
          type="submit"
        >
          {deleting ? "Deleting…" : "Delete meeting"}
        </button>
      </div>
    </Modal>
  );
}
/** The abc-defg-hij part of a Google Meet link, or null for other URLs. */
function meetingCode(url: string | null) {
  if (!url) return null;
  try {
    const code = new URL(url).pathname.slice(1);
    return /^[a-z]{3}-[a-z]{4}-[a-z]{3}$/i.test(code) ? code : null;
  } catch {
    return null;
  }
}
/** Default start time for new meetings: the next :00 or :30. */
function nextHalfHour(now: Date) {
  const next = new Date(now);
  next.setMinutes(now.getMinutes() < 30 ? 30 : 60, 0, 0);
  return `${String(next.getHours()).padStart(2, "0")}:${String(next.getMinutes()).padStart(2, "0")}`;
}
