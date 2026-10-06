import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import type { Meeting, MeetingDraft } from "../App";
import type { GoogleEvent } from "../lib/google";
import {
  DEFAULT_IRIS_COLOR,
  colorHex,
  isEventColor,
  type EventColorKey,
} from "../lib/colors";
import MeetingsPage from "./MeetingsPage";
import CalendarPage from "./CalendarPage";
import TranslationPage from "./TranslationPage";
import SettingsPage from "./SettingsPage";
import Sidebar from "../components/Sidebar";
import "./calendar.css";

type Props = {
  account: { email: string; hasPassword: boolean; providers: string[] };
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  joinUrl: string;
  googleConnected: boolean;
  meetingBusy: boolean;
  meetings: Meeting[];
  name: string;
  notice: string;
  now: Date;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onChangePassword: (current: string | null, next: string) => Promise<string | null>;
  onSendPasswordReset: () => Promise<string | null>;
  onConnectGoogle: () => void;
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
  onShiftWeek: (days: number) => void;
  onSignOut: () => void;
  setShowCreate: (open: boolean) => void;
  showCreate: boolean;
  setShowJoin: (open: boolean) => void;
  showJoin: boolean;
  weekStart: Date;
};
const dateKey = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const time = (v: string) =>
  new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(v));

/** The full signed-in dashboard lives here; App.tsx supplies data and actions. */
export default function HomePage(p: Props) {
  const [page, setPage] = useState("Home");
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  // Pre-fills the Create dialog when a day or time slot is clicked on the calendar.
  const [createAt, setCreateAt] = useState<Date | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Meeting | null>(null);
  const upcoming = p.meetings.filter(
    (meeting) => new Date(meeting.ends_at) >= p.now,
  );
  const selectedMeetings = upcoming.filter(
    (meeting) => dateKey(new Date(meeting.starts_at)) === dateKey(p.now),
  );
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(p.weekStart);
    d.setDate(d.getDate() + i);
    return d;
  });
  const Page = page === "Meetings" ? MeetingsPage : TranslationPage;
  const greeting =
    p.now.getHours() < 12
      ? "Good morning"
      : p.now.getHours() < 18
        ? "Good afternoon"
        : "Good evening";
  return (
    <main className="dashboard-shell">
      <Sidebar
        active={page}
        avatarUrl={p.avatarUrl}
        fileInput={p.fileInput}
        name={p.name}
        onAvatarChange={p.onAvatarChange}
        onNavigate={setPage}
        onSignOut={p.onSignOut}
      />
      {page === "Calendar" ? (
        <CalendarPage
          googleConnected={p.googleConnected}
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
        />
      ) : page === "Settings" ? (
        <SettingsPage
          {...p.account}
          onChangePassword={p.onChangePassword}
          onSendReset={p.onSendPasswordReset}
        />
      ) : page !== "Home" ? (
        <Page />
      ) : (
        <section className="calendar-dashboard">
          <div className="iris-control-center">
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
            <section className="iris-schedule">
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
              {(selectedMeetings.length ? selectedMeetings : upcoming.slice(0, 3)).map(
                (m) => (
                  <MeetingCard
                    key={m.id}
                    meeting={m}
                    now={p.now}
                    onDelete={() => setPendingDelete(m)}
                    onJoin={p.onJoin}
                    showDate={!selectedMeetings.length}
                  />
                ),
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
          <header className="calendar-header">
            <div>
              <h1>
                {greeting}, {p.name}
              </h1>
              <p>
                {new Intl.DateTimeFormat(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                }).format(p.now)}
              </p>
            </div>
            <div className="header-actions">
              <button onClick={() => setPage("Calendar")}>▦ Schedule</button>
              <button
                onClick={() =>
                  void navigator.mediaDevices
                    ?.getDisplayMedia({ video: true })
                    .then((stream) =>
                      stream.getTracks().forEach((track) => track.stop()),
                    )
                }
              >
                ↑ Share screen
              </button>
              <button onClick={() => p.setShowJoin(true)}>
                ▣ Join meeting
              </button>
              <button
                className="create-button"
                onClick={() => p.setShowCreate(true)}
              >
                ＋ Create meeting
              </button>
            </div>
          </header>
          <section className="week-card">
            <header>
              <h2>Your week</h2>
              <div>
                <button onClick={() => p.onShiftWeek(-7)}>‹</button>
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
                <button onClick={() => p.onShiftWeek(7)}>›</button>
              </div>
            </header>
            <div className="week-days">
              {days.map((day) => (
                <article
                  className={dateKey(day) === dateKey(p.now) ? "today" : ""}
                  key={dateKey(day)}
                >
                  <b>
                    {new Intl.DateTimeFormat(undefined, {
                      weekday: "short",
                    }).format(day)}
                  </b>
                  <strong>{day.getDate()}</strong>
                  {p.meetings
                    .filter(
                      (m) => dateKey(new Date(m.starts_at)) === dateKey(day),
                    )
                    .map((m) => (
                      <button
                        className={`week-event event-${m.color || "blue"}`}
                        key={m.id}
                        onClick={() =>
                          m.meeting_url && void p.onJoin(m.meeting_url)
                        }
                        type="button"
                      >
                        <small>{time(m.starts_at)}</small>
                        {m.title}
                      </button>
                    ))}
                </article>
              ))}
            </div>
          </section>
        </section>
      )}
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
      {p.notice && (
        <div className="iris-toast" role="status">
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
      style={{ ["--icon" as string]: `url(/action-icons/${type}.svg)` }}
    />
  );
}
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
  const live =
    new Date(meeting.starts_at).getTime() - 10 * 60_000 <= now.getTime() &&
    new Date(meeting.ends_at) > now;
  return (
    <article
      className={`iris-meeting ${live ? "live" : ""}`}
      style={
        live
          ? undefined
          : {
              borderLeftColor: colorHex(
                isEventColor(meeting.color) ? meeting.color : DEFAULT_IRIS_COLOR,
              ),
            }
      }
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
          <input
            defaultValue={startTime}
            name="time"
            required
            type="time"
          />
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
        {inGoogle ? " from Iris and your Google Calendar." : " from Iris."}{" "}
        This can't be undone.
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
function nextHalfHour(now: Date) {
  const next = new Date(now);
  next.setMinutes(now.getMinutes() < 30 ? 30 : 60, 0, 0);
  return `${String(next.getHours()).padStart(2, "0")}:${String(next.getMinutes()).padStart(2, "0")}`;
}
