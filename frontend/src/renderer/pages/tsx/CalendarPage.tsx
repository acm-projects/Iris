// Calendar page: Month / Week / Day views of Iris meetings and Google Calendar
// events, with live sync, event details, and per-event colours.
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { Meeting } from "../../App";
import type { GoogleEvent } from "../../utils/google";
import {
  DEFAULT_GOOGLE_COLOR,
  DEFAULT_IRIS_COLOR,
  EVENT_COLORS,
  colorFromGoogle,
  colorHex,
  isEventColor,
  type EventColorKey,
} from "../../utils/colors";
import "../css/calendar-page.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Which calendar layout is showing. */
type View = "month" | "week" | "day";
/** One event as the calendar draws it, whether it came from Iris or Google. */
type CalendarItem = {
  id: string;
  title: string;
  start: Date;
  end: Date;
  allDay: boolean;
  source: "iris" | "google";
  meetUrl: string | null;
  htmlLink: string | null;
  location: string | null;
  note: string | null;
  color: EventColorKey;
  meetingId?: string;
  googleEventId?: string;
};
/** Inline style that hands an event's colour to the CSS as --ev. */
const evStyle = (item: CalendarItem) =>
  ({ "--ev": colorHex(item.color) }) as React.CSSProperties;
/** Data and actions supplied by App (via HomePage). */
type Props = {
  googleConnected: boolean;
  googleConnecting: boolean;
  meetings: Meeting[];
  now: Date;
  onConnectGoogle: () => void;
  onCreateAt: (start: Date) => void;
  onJoin: (url: string) => Promise<boolean>;
  onLoadGoogleEvents: (start: Date, end: Date) => Promise<GoogleEvent[] | null>;
  onOpenInGoogle: (url: string) => void;
  onSetEventColor: (
    target: { meetingId?: string; googleEventId?: string },
    color: EventColorKey,
  ) => Promise<string | null>;
};

// ---------------------------------------------------------------------------
// Layout constants and date helpers
// ---------------------------------------------------------------------------

// Pixel height of one hour in Week/Day view (must match the grid lines in CSS).
const HOUR_HEIGHT = 52;
// Events shown per day in Month view before "N more".
const MONTH_CHIP_LIMIT = 3;
// How often Google Calendar is re-fetched while the page is open.
const SYNC_INTERVAL_MS = 60_000;

/** Midnight at the start of `d`. */
const startOfDay = (d: Date) =>
  new Date(d.getFullYear(), d.getMonth(), d.getDate());
/** `d` moved by a number of whole days (DST-safe). */
const addDays = (d: Date, days: number) =>
  new Date(d.getFullYear(), d.getMonth(), d.getDate() + days);
// Weeks start on Monday, matching the dashboard.
const startOfWeek = (d: Date) =>
  addDays(startOfDay(d), -((d.getDay() + 6) % 7));
const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString();
/** Locale-aware date formatting shortcut. */
const format = (date: Date, options: Intl.DateTimeFormatOptions) =>
  new Intl.DateTimeFormat(undefined, options).format(date);
/** Short local time, e.g. "9:30 AM". */
const clock = (date: Date) =>
  format(date, { hour: "numeric", minute: "2-digit" });
/** Google all-day dates ("2026-10-06") are local calendar days, not UTC instants. */
const parseGoogleDate = (value: string, allDay: boolean) => {
  if (!allDay) return new Date(value);
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day);
};
/** The abc-defg-hij code from a Meet link, or null. */
const meetCode = (url: string | null) => {
  if (!url) return null;
  const code = url.split("/").pop()?.split("?")[0] ?? "";
  return /^[a-z]{3}-[a-z]{4}-[a-z]{3}$/i.test(code) ? code : null;
};

/** First and last day shown for the current view (Month shows whole weeks). */
function visibleRange(view: View, cursor: Date) {
  if (view === "day") {
    const start = startOfDay(cursor);
    return { start, end: addDays(start, 1), weeks: 0 };
  }
  if (view === "week") {
    const start = startOfWeek(cursor);
    return { start, end: addDays(start, 7), weeks: 1 };
  }
  const first = new Date(cursor.getFullYear(), cursor.getMonth(), 1);
  const start = startOfWeek(first);
  const daysInMonth = new Date(
    cursor.getFullYear(),
    cursor.getMonth() + 1,
    0,
  ).getDate();
  const weeks = Math.ceil((((first.getDay() + 6) % 7) + daysInMonth) / 7);
  return { start, end: addDays(start, weeks * 7), weeks };
}

/** Events that touch the given day, all-day first, then by start time. */
function itemsOnDay(items: CalendarItem[], day: Date) {
  const dayStart = day.getTime();
  const dayEnd = addDays(day, 1).getTime();
  return items
    .filter(
      (item) =>
        item.start.getTime() < dayEnd &&
        Math.max(item.end.getTime(), item.start.getTime() + 1) > dayStart,
    )
    .sort(
      (a, b) =>
        Number(b.allDay) - Number(a.allDay) ||
        a.start.getTime() - b.start.getTime(),
    );
}

/** A positioned event block in Week/Day view (pixels + overlap column). */
type Block = {
  item: CalendarItem;
  top: number;
  height: number;
  column: number;
  columns: number;
};
/** Places a day's timed events side by side wherever they overlap. */
function layoutDay(items: CalendarItem[], day: Date): Block[] {
  const dayStart = day.getTime();
  const dayEnd = addDays(day, 1).getTime();
  const spans = items
    .filter((item) => !item.allDay)
    .map((item) => {
      const start = Math.max(item.start.getTime(), dayStart);
      // Give very short events enough height to read.
      const end = Math.min(
        Math.max(item.end.getTime(), item.start.getTime() + 20 * 60_000),
        dayEnd,
      );
      return { item, start, end };
    })
    .filter((span) => span.end > dayStart && span.start < dayEnd)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const blocks: Block[] = [];
  let cluster: typeof spans = [];
  let clusterEnd = -Infinity;
  const flush = () => {
    const columnEnds: number[] = [];
    const placed = cluster.map((span) => {
      let column = columnEnds.findIndex((end) => end <= span.start);
      if (column === -1) column = columnEnds.push(span.end) - 1;
      else columnEnds[column] = span.end;
      return { span, column };
    });
    for (const { span, column } of placed)
      blocks.push({
        item: span.item,
        top: ((span.start - dayStart) / 3_600_000) * HOUR_HEIGHT,
        height: Math.max(
          ((span.end - span.start) / 3_600_000) * HOUR_HEIGHT - 2,
          18,
        ),
        column,
        columns: columnEnds.length,
      });
    cluster = [];
  };
  for (const span of spans) {
    if (cluster.length && span.start >= clusterEnd) flush();
    if (!cluster.length) clusterEnd = span.end;
    cluster.push(span);
    clusterEnd = Math.max(clusterEnd, span.end);
  }
  if (cluster.length) flush();
  return blocks;
}

/** Notion-style calendar showing Iris meetings and the user's Google Calendar. */
// ---------------------------------------------------------------------------
// Calendar page
// ---------------------------------------------------------------------------

/** Notion-style calendar showing Iris meetings and the user's Google Calendar. */
export default function CalendarPage(p: Props) {
  // Current view and the date it is centred on.
  const [view, setView] = useState<View>("month");
  const [cursor, setCursor] = useState(() => startOfDay(new Date()));
  // Google Calendar sync state.
  const [googleEvents, setGoogleEvents] = useState<GoogleEvent[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [syncedAt, setSyncedAt] = useState<Date | null>(null);
  const [syncError, setSyncError] = useState("");
  // Event whose details card is open, and where the user clicked.
  const [selected, setSelected] = useState<{
    item: CalendarItem;
    x: number;
    y: number;
  } | null>(null);
  // Colours picked this session, shown immediately while they save.
  const [colorOverrides, setColorOverrides] = useState<
    Record<string, EventColorKey>
  >({});
  const [colorError, setColorError] = useState("");
  const range = useMemo(() => visibleRange(view, cursor), [view, cursor]);
  // Refs so the sync effect always calls the latest loader and ignores stale responses.
  const loader = useRef(p.onLoadGoogleEvents);
  const requestId = useRef(0);
  const syncNow = useRef<() => void>(() => {});
  useEffect(() => {
    loader.current = p.onLoadGoogleEvents;
  });

  // Keep Google events fresh: on range change, after Iris meetings change,
  // every minute, and whenever the window regains focus.
  const rangeStart = range.start.getTime();
  const rangeEnd = range.end.getTime();
  useEffect(() => {
    if (!p.googleConnected) {
      setGoogleEvents([]);
      setSyncedAt(null);
      syncNow.current = () => {};
      return;
    }
    const sync = async () => {
      const id = ++requestId.current;
      setSyncing(true);
      try {
        const events = await loader.current(
          new Date(rangeStart),
          new Date(rangeEnd),
        );
        if (id !== requestId.current) return;
        if (events) setGoogleEvents(events);
        setSyncedAt(new Date());
        setSyncError("");
      } catch (error) {
        if (id === requestId.current)
          setSyncError(
            error instanceof Error ? error.message : "Could not sync.",
          );
      } finally {
        if (id === requestId.current) setSyncing(false);
      }
    };
    syncNow.current = () => void sync();
    void sync();
    const timer = window.setInterval(() => void sync(), SYNC_INTERVAL_MS);
    window.addEventListener("focus", sync);
    return () => {
      clearInterval(timer);
      window.removeEventListener("focus", sync);
    };
  }, [p.googleConnected, rangeStart, rangeEnd, p.meetings]);

  const refresh = () => syncNow.current();

  // Iris meetings that Iris also put on Google Calendar are shown once, using
  // Google's copy so edits made in Google Calendar appear here too.
  const items = useMemo<CalendarItem[]>(() => {
    const linked = new Map(
      p.meetings
        .filter((m) => m.google_event_id)
        .map((m) => [m.google_event_id as string, m]),
    );
    const fromGoogle = new Set<string>();
    const google = googleEvents.map((event): CalendarItem => {
      const meeting = linked.get(event.id);
      if (meeting) fromGoogle.add(meeting.id);
      const id = `google-${event.id}`;
      const saved =
        meeting && isEventColor(meeting.color) ? meeting.color : null;
      return {
        id,
        title: event.title,
        start: parseGoogleDate(event.start, event.allDay),
        end: parseGoogleDate(event.end, event.allDay),
        allDay: event.allDay,
        source: meeting ? "iris" : "google",
        meetUrl: event.meetUrl ?? meeting?.meeting_url ?? null,
        htmlLink: event.htmlLink,
        location: event.location,
        note: meeting?.accessibility_mode ?? null,
        color:
          colorOverrides[id] ??
          saved ??
          colorFromGoogle(event.colorId) ??
          (meeting ? DEFAULT_IRIS_COLOR : DEFAULT_GOOGLE_COLOR),
        meetingId: meeting?.id,
        googleEventId: event.id,
      };
    });
    const iris = p.meetings
      .filter((m) => !fromGoogle.has(m.id))
      .map((m): CalendarItem => ({
        id: `iris-${m.id}`,
        title: m.title,
        start: new Date(m.starts_at),
        end: new Date(m.ends_at),
        allDay: false,
        source: "iris",
        meetUrl: m.meeting_url,
        htmlLink: null,
        location: null,
        note: m.accessibility_mode,
        color:
          colorOverrides[`iris-${m.id}`] ??
          (isEventColor(m.color) ? m.color : DEFAULT_IRIS_COLOR),
        meetingId: m.id,
        googleEventId: m.google_event_id ?? undefined,
      }));
    return [...google, ...iris];
  }, [googleEvents, p.meetings, colorOverrides]);

  /** Recolours an event immediately, then saves; reverts if a Google-only save fails. */
  const changeColor = async (item: CalendarItem, color: EventColorKey) => {
    const previous = item.color;
    setColorError("");
    setColorOverrides((current) => ({ ...current, [item.id]: color }));
    setSelected((current) =>
      current ? { ...current, item: { ...current.item, color } } : current,
    );
    const error = await p.onSetEventColor(
      { meetingId: item.meetingId, googleEventId: item.googleEventId },
      color,
    );
    if (!error) return;
    // An Iris meeting keeps its new colour even if Google could not be updated.
    if (!item.meetingId) {
      setColorOverrides((current) => ({ ...current, [item.id]: previous }));
      setSelected((current) =>
        current
          ? { ...current, item: { ...current.item, color: previous } }
          : current,
      );
    }
    setColorError(error);
  };

  // Toolbar navigation: previous/next month, week or day.
  const go = (direction: -1 | 1) =>
    setCursor((current) =>
      view === "month"
        ? new Date(current.getFullYear(), current.getMonth() + direction, 1)
        : addDays(current, direction * (view === "week" ? 7 : 1)),
    );
  const openDay = (day: Date) => {
    setCursor(startOfDay(day));
    setView("day");
  };
  const select = (item: CalendarItem, event: MouseEvent) => {
    event.stopPropagation();
    setSelected({ item, x: event.clientX, y: event.clientY });
  };

  // Notion/Google-style shortcuts: T today, M/W/D views, arrows to move.
  // Keyboard shortcuts while the calendar is focused (not while typing).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        target?.closest("input, textarea, select, [role='dialog']")
      )
        return;
      const key = event.key.toLowerCase();
      if (key === "escape") setSelected(null);
      else if (key === "t") setCursor(startOfDay(new Date()));
      else if (key === "m") setView("month");
      else if (key === "w") setView("week");
      else if (key === "d") setView("day");
      else if (key === "arrowleft") go(-1);
      else if (key === "arrowright") go(1);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Heading text for the current view.
  const title =
    view === "month"
      ? format(cursor, { month: "long", year: "numeric" })
      : view === "week"
        ? weekTitle(range.start, addDays(range.end, -1))
        : format(cursor, {
            weekday: "long",
            month: "long",
            day: "numeric",
            year: "numeric",
          });

  return (
    <section className="calendar-page" onClick={() => setSelected(null)}>
      {/* Toolbar: title, sync status, Today, ‹ ›, view switch, + New */}
      <header className="cal-toolbar">
        <div className="cal-heading">
          <p>Calendar</p>
          <h1>{title}</h1>
        </div>
        <div className="cal-controls">
          <SyncStatus
            connected={p.googleConnected}
            connecting={p.googleConnecting}
            error={syncError}
            onConnect={p.onConnectGoogle}
            onRefresh={refresh}
            syncedAt={syncedAt}
            syncing={syncing}
          />
          <button
            className="cal-button"
            onClick={() => setCursor(startOfDay(new Date()))}
            title="Today (T)"
            type="button"
          >
            Today
          </button>
          <div className="cal-nav">
            <button
              aria-label="Previous"
              onClick={() => go(-1)}
              title="Previous (←)"
              type="button"
            >
              ‹
            </button>
            <button
              aria-label="Next"
              onClick={() => go(1)}
              title="Next (→)"
              type="button"
            >
              ›
            </button>
          </div>
          <div className="cal-views" role="tablist" aria-label="Calendar view">
            {(["month", "week", "day"] as const).map((option) => (
              <button
                aria-selected={view === option}
                className={view === option ? "active" : ""}
                key={option}
                onClick={() => setView(option)}
                role="tab"
                title={`${option[0].toUpperCase()}${option.slice(1)} (${option[0].toUpperCase()})`}
                type="button"
              >
                {option[0].toUpperCase() + option.slice(1)}
              </button>
            ))}
          </div>
          <button
            className="cal-new"
            onClick={() => {
              const start = new Date(p.now);
              start.setMinutes(start.getMinutes() < 30 ? 30 : 60, 0, 0);
              p.onCreateAt(
                view === "month" || sameDay(cursor, p.now)
                  ? start
                  : new Date(
                      cursor.getFullYear(),
                      cursor.getMonth(),
                      cursor.getDate(),
                      9,
                    ),
              );
            }}
            type="button"
          >
            ＋ New
          </button>
        </div>
      </header>

      {/* The grid for the current view */}
      {view === "month" ? (
        <MonthView
          cursor={cursor}
          items={items}
          now={p.now}
          onCreateAt={p.onCreateAt}
          onOpenDay={openDay}
          onSelect={select}
          range={range}
        />
      ) : (
        <TimeGrid
          days={Array.from({ length: view === "week" ? 7 : 1 }, (_, i) =>
            addDays(range.start, i),
          )}
          items={items}
          now={p.now}
          onCreateAt={p.onCreateAt}
          onOpenDay={openDay}
          onSelect={select}
        />
      )}

      {/* Details card for the clicked event */}
      {selected && (
        <EventPopover
          colorError={colorError}
          item={selected.item}
          onClose={() => {
            setSelected(null);
            setColorError("");
          }}
          onColor={(color) => void changeColor(selected.item, color)}
          onJoin={p.onJoin}
          onOpenInGoogle={p.onOpenInGoogle}
          x={selected.x}
          y={selected.y}
        />
      )}
    </section>
  );
}

/** "October 5 – 11, 2026" (or across months/years when needed). */
function weekTitle(start: Date, end: Date) {
  if (start.getMonth() === end.getMonth())
    return `${format(start, { month: "long" })} ${start.getDate()} – ${end.getDate()}, ${end.getFullYear()}`;
  return `${format(start, { month: "short", day: "numeric" })} – ${format(end, { month: "short", day: "numeric", year: "numeric" })}`;
}

/** Google Calendar status in the toolbar: Connect button, or synced time + refresh. */
function SyncStatus({
  connected,
  connecting,
  error,
  onConnect,
  onRefresh,
  syncedAt,
  syncing,
}: {
  connected: boolean;
  connecting: boolean;
  error: string;
  onConnect: () => void;
  onRefresh: () => void;
  syncedAt: Date | null;
  syncing: boolean;
}) {
  if (!connected)
    return (
      <button
        className="cal-connect"
        disabled={connecting}
        onClick={onConnect}
        type="button"
      >
        <span className="cal-google-dot" />
        {connecting ? "Finish in your browser…" : "Connect Google Calendar"}
      </button>
    );
  return (
    <div className={`cal-sync ${error ? "error" : ""}`} title={error}>
      <span className="cal-sync-dot" />
      <span>
        {error
          ? "Sync failed"
          : syncing && !syncedAt
            ? "Syncing…"
            : syncedAt
              ? `Synced ${clock(syncedAt)}`
              : "Google Calendar"}
      </span>
      <button
        aria-label="Refresh Google Calendar"
        className={syncing ? "spinning" : ""}
        onClick={onRefresh}
        title="Refresh"
        type="button"
      >
        ↻
      </button>
    </div>
  );
}

/** A compact event in Month view or the all-day row. */
function EventChip({
  item,
  onSelect,
}: {
  item: CalendarItem;
  onSelect: (item: CalendarItem, event: MouseEvent) => void;
}) {
  return (
    <button
      className={`cal-chip ${item.source} ${item.allDay ? "all-day" : ""}`}
      onClick={(event) => onSelect(item, event)}
      style={evStyle(item)}
      title={item.title}
      type="button"
    >
      {!item.allDay && <i />}
      <span>{item.title}</span>
      {!item.allDay && <time>{clock(item.start)}</time>}
    </button>
  );
}

/** Month grid: weekday header, then one cell per day with its events. */
function MonthView({
  cursor,
  items,
  now,
  onCreateAt,
  onOpenDay,
  onSelect,
  range,
}: {
  cursor: Date;
  items: CalendarItem[];
  now: Date;
  onCreateAt: (start: Date) => void;
  onOpenDay: (day: Date) => void;
  onSelect: (item: CalendarItem, event: MouseEvent) => void;
  range: { start: Date; weeks: number };
}) {
  const days = Array.from({ length: range.weeks * 7 }, (_, i) =>
    addDays(range.start, i),
  );
  return (
    <div
      className="cal-month"
      style={{
        gridTemplateRows: `auto repeat(${range.weeks}, minmax(0, 1fr))`,
      }}
    >
      {days.slice(0, 7).map((day) => (
        <div className="cal-weekday" key={`head-${day.getDay()}`}>
          {format(day, { weekday: "short" })}
        </div>
      ))}
      {days.map((day) => {
        const dayItems = itemsOnDay(items, day);
        const overflow = dayItems.length - MONTH_CHIP_LIMIT;
        const shown = dayItems.slice(
          0,
          overflow > 0 ? MONTH_CHIP_LIMIT - 1 : MONTH_CHIP_LIMIT,
        );
        const outside = day.getMonth() !== cursor.getMonth();
        const weekend = day.getDay() === 0 || day.getDay() === 6;
        return (
          <div
            className={`cal-day ${outside ? "outside" : ""} ${weekend ? "weekend" : ""} ${sameDay(day, now) ? "today" : ""}`}
            key={day.toISOString()}
            onDoubleClick={() =>
              onCreateAt(
                new Date(day.getFullYear(), day.getMonth(), day.getDate(), 9),
              )
            }
          >
            <header>
              <button
                className="cal-date"
                onClick={(event) => {
                  event.stopPropagation();
                  onOpenDay(day);
                }}
                title="Open day"
                type="button"
              >
                {day.getDate() === 1
                  ? format(day, { month: "short", day: "numeric" })
                  : day.getDate()}
              </button>
              <button
                aria-label={`New meeting on ${format(day, { month: "long", day: "numeric" })}`}
                className="cal-add"
                onClick={(event) => {
                  event.stopPropagation();
                  onCreateAt(
                    new Date(
                      day.getFullYear(),
                      day.getMonth(),
                      day.getDate(),
                      9,
                    ),
                  );
                }}
                title="New meeting"
                type="button"
              >
                ＋
              </button>
            </header>
            <div className="cal-chips">
              {shown.map((item) => (
                <EventChip
                  item={item}
                  key={`${item.id}-${day.getDate()}`}
                  onSelect={onSelect}
                />
              ))}
              {overflow > 0 && (
                <button
                  className="cal-more"
                  onClick={(event) => {
                    event.stopPropagation();
                    onOpenDay(day);
                  }}
                  type="button"
                >
                  {overflow + 1} more
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Week/Day view: day headers, all-day row, and a scrolling 24-hour grid. */
function TimeGrid({
  days,
  items,
  now,
  onCreateAt,
  onOpenDay,
  onSelect,
}: {
  days: Date[];
  items: CalendarItem[];
  now: Date;
  onCreateAt: (start: Date) => void;
  onOpenDay: (day: Date) => void;
  onSelect: (item: CalendarItem, event: MouseEvent) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const firstDay = days[0].getTime();
  // Open scrolled to the working day (or just above the current time).
  useEffect(() => {
    const hour = days.some((day) => sameDay(day, new Date()))
      ? Math.max(new Date().getHours() - 2, 0)
      : 7;
    scroller.current?.scrollTo({ top: hour * HOUR_HEIGHT });
    // Only re-scroll when the visible days change, not on every minute tick.
  }, [firstDay, days.length]);
  const columns = { "--days": days.length } as React.CSSProperties;
  const allDay = days.map((day) =>
    itemsOnDay(items, day).filter((item) => item.allDay),
  );
  const hasAllDay = allDay.some((list) => list.length);
  return (
    <div className={`cal-grid ${days.length === 1 ? "single" : ""}`}>
      <div className="cal-grid-head" style={columns}>
        <span />
        {days.map((day) => (
          <button
            className={sameDay(day, now) ? "today" : ""}
            key={day.toISOString()}
            onClick={() => onOpenDay(day)}
            type="button"
          >
            <small>{format(day, { weekday: "short" })}</small>
            <strong>{day.getDate()}</strong>
          </button>
        ))}
      </div>
      {hasAllDay && (
        <div className="cal-allday" style={columns}>
          <span>All-day</span>
          {allDay.map((list, index) => (
            <div key={days[index].toISOString()}>
              {list.map((item) => (
                <EventChip item={item} key={item.id} onSelect={onSelect} />
              ))}
            </div>
          ))}
        </div>
      )}
      <div className="cal-grid-scroll" ref={scroller}>
        <div
          className="cal-grid-body"
          style={{ ...columns, height: 24 * HOUR_HEIGHT }}
        >
          {/* Hour labels in the left gutter */}
          <div className="cal-hours">
            {Array.from({ length: 23 }, (_, i) => i + 1).map((hour) => (
              <span key={hour} style={{ top: hour * HOUR_HEIGHT }}>
                {format(new Date(2000, 0, 1, hour), { hour: "numeric" })}
              </span>
            ))}
          </div>
          {/* One column per day; clicking empty space creates a meeting at that time */}
          {days.map((day) => (
            <div
              className="cal-column"
              key={day.toISOString()}
              onClick={(event) => {
                if (event.target !== event.currentTarget) return;
                const rect = event.currentTarget.getBoundingClientRect();
                const minutes =
                  Math.floor(((event.clientY - rect.top) / HOUR_HEIGHT) * 2) *
                  30;
                onCreateAt(
                  new Date(
                    day.getFullYear(),
                    day.getMonth(),
                    day.getDate(),
                    0,
                    minutes,
                  ),
                );
              }}
              title="Click to create a meeting"
            >
              {layoutDay(itemsOnDay(items, day), day).map((block) => (
                <button
                  className={`cal-block ${block.item.source} ${block.height < 40 ? "compact" : ""}`}
                  key={block.item.id}
                  onClick={(event) => onSelect(block.item, event)}
                  style={{
                    ...evStyle(block.item),
                    top: block.top,
                    height: block.height,
                    left: `calc(${(block.column / block.columns) * 100}% + 2px)`,
                    width: `calc(${100 / block.columns}% - 4px)`,
                  }}
                  type="button"
                >
                  <strong>{block.item.title}</strong>
                  <small>
                    {clock(block.item.start)} – {clock(block.item.end)}
                  </small>
                </button>
              ))}
              {sameDay(day, now) && (
                <div
                  className="cal-now"
                  style={{
                    top:
                      ((now.getHours() * 60 + now.getMinutes()) / 60) *
                      HOUR_HEIGHT,
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Event details card: time, source, Meet code, colour picker, actions. */
function EventPopover({
  colorError,
  item,
  onClose,
  onColor,
  onJoin,
  onOpenInGoogle,
  x,
  y,
}: {
  colorError: string;
  item: CalendarItem;
  onClose: () => void;
  onColor: (color: EventColorKey) => void;
  onJoin: (url: string) => Promise<boolean>;
  onOpenInGoogle: (url: string) => void;
  x: number;
  y: number;
}) {
  const [copied, setCopied] = useState(false);
  const width = 320;
  // Keep the card on screen next to where the event was clicked.
  const left = Math.min(Math.max(x + 12, 12), window.innerWidth - width - 12);
  const top = Math.min(Math.max(y - 20, 12), window.innerHeight - 300);
  const code = meetCode(item.meetUrl);
  const multiDay = !sameDay(item.start, new Date(item.end.getTime() - 1));
  const when = item.allDay
    ? multiDay
      ? `${format(item.start, { month: "short", day: "numeric" })} – ${format(new Date(item.end.getTime() - 1), { month: "short", day: "numeric" })} · All day`
      : `${format(item.start, { weekday: "long", month: "long", day: "numeric" })} · All day`
    : `${format(item.start, { weekday: "short", month: "short", day: "numeric" })} · ${clock(item.start)} – ${clock(item.end)}`;
  return (
    <div
      aria-label={item.title}
      className={`cal-popover ${item.source}`}
      onClick={(event) => event.stopPropagation()}
      role="dialog"
      style={{ ...evStyle(item), left, top, width }}
    >
      <header>
        <i />
        <h2>{item.title}</h2>
        <button aria-label="Close" onClick={onClose} type="button">
          ×
        </button>
      </header>
      <p>{when}</p>
      <p className="cal-popover-meta">
        {item.source === "iris" ? "Iris meeting" : "Google Calendar"}
        {item.note ? ` · ${item.note}` : ""}
      </p>
      {item.location && !meetCode(item.location) && (
        <p className="cal-popover-meta">📍 {item.location}</p>
      )}
      {code && (
        <p className="cal-popover-meta">
          Meeting code <code>{code}</code>
        </p>
      )}
      {/* Colour swatches (brand palette) */}
      <div className="cal-colors" role="radiogroup" aria-label="Event colour">
        {EVENT_COLORS.map((color) => (
          <button
            aria-checked={item.color === color.key}
            aria-label={color.label}
            className={item.color === color.key ? "selected" : ""}
            key={color.key}
            onClick={() => onColor(color.key)}
            role="radio"
            style={{ "--swatch": color.hex } as React.CSSProperties}
            title={color.label}
            type="button"
          />
        ))}
      </div>
      {colorError && <p className="cal-popover-error">{colorError}</p>}
      {/* Join, Copy link, Open in Google */}
      <div className="cal-popover-actions">
        {item.meetUrl && (
          <button
            className="primary"
            onClick={() => void onJoin(item.meetUrl as string)}
            type="button"
          >
            Join with Iris
          </button>
        )}
        {item.meetUrl && (
          <button
            onClick={() =>
              void navigator.clipboard
                .writeText(item.meetUrl as string)
                .then(() => setCopied(true))
            }
            type="button"
          >
            {copied ? "Copied ✓" : "Copy link"}
          </button>
        )}
        {item.htmlLink && (
          <button
            onClick={() => onOpenInGoogle(item.htmlLink as string)}
            type="button"
          >
            Open in Google
          </button>
        )}
      </div>
    </div>
  );
}
