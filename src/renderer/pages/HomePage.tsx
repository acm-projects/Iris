import { ChangeEvent, FormEvent, useState } from "react";
import MeetingsPage from "./MeetingsPage";
import CalendarPage from "./CalendarPage";
import TranslationPage from "./TranslationPage";
import AnalyticsPage from "./AnalyticsPage";
import SettingsPage from "./SettingsPage";
import Sidebar from "../components/Sidebar";
import "./calendar.css";

type Meeting = {
  id: string;
  title: string;
  starts_at: string;
  ends_at: string;
  meeting_url: string | null;
  accessibility_mode: string;
  color: string;
};
type Props = {
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  joinUrl: string;
  meetings: Meeting[];
  name: string;
  now: Date;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onCreate: (event: FormEvent<HTMLFormElement>) => void;
  onJoin: (url: string) => Promise<boolean>;
  onJoinUrlChange: (url: string) => void;
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
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(p.weekStart);
    d.setDate(d.getDate() + i);
    return d;
  });
  const Page =
    page === "Meetings"
      ? MeetingsPage
      : page === "Calendar"
        ? CalendarPage
        : page === "Translation"
          ? TranslationPage
          : page === "Analytics"
            ? AnalyticsPage
            : SettingsPage;
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
      {page !== "Home" ? (
        <Page />
      ) : (
        <section className="calendar-dashboard">
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
      {p.showJoin && <JoinDialog {...p} />}{" "}
      {p.showCreate && <CreateDialog {...p} />}
    </main>
  );
}
function Icon({ active, item }: { active: boolean; item: string }) {
  const icon =
    item === "Home"
      ? "home"
      : item === "Meetings"
        ? "meetings"
        : item === "Calendar"
          ? "calendar"
          : item === "Settings"
            ? "settings"
            : null;
  return icon ? (
    <img
      alt=""
      className="sidebar-icon"
      src={`/sidebar-icons/${icon}${active ? "-active" : ""}.svg`}
    />
  ) : (
    <span className="sidebar-fallback">
      {item === "Translation" ? "⌁" : "◌"}
    </span>
  );
}
function JoinDialog(p: Props) {
  return (
    <div className="modal-backdrop">
      <form
        className="meeting-modal join-modal"
        onSubmit={async (e) => {
          e.preventDefault();
          if (await p.onJoin(p.joinUrl)) p.setShowJoin(false);
        }}
      >
        <header>
          <h2>Join a meeting</h2>
          <button onClick={() => p.setShowJoin(false)} type="button">
            ×
          </button>
        </header>
        <p>Paste your Google Meet link. Iris will use your default browser.</p>
        <label>
          Google Meet link
          <input
            autoFocus
            onChange={(e) => p.onJoinUrlChange(e.target.value)}
            pattern="https://meet\.google\.com/.*"
            placeholder="https://meet.google.com/abc-defg-hij"
            required
            type="url"
            value={p.joinUrl}
          />
        </label>
        <button className="modal-submit" type="submit">
          Join meeting
        </button>
      </form>
    </div>
  );
}
function CreateDialog(p: Props) {
  return (
    <div className="modal-backdrop">
      <form className="meeting-modal" onSubmit={p.onCreate}>
        <header>
          <h2>Create meeting</h2>
          <button onClick={() => p.setShowCreate(false)} type="button">
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
              defaultValue={dateKey(p.now)}
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
          Google Meet link <small>Optional — add it later</small>
          <input
            name="url"
            placeholder="https://meet.google.com/..."
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
  );
}
