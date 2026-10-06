// Settings page: Appearance (theme + app icon), Account, Integrations
// (Google Calendar) and Security (password), with an in-page section menu.
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  PASSWORD_HINT,
  passwordProblem,
  type AuthNotice,
} from "../../utils/auth";
import {
  loadAppIcon,
  loadTheme,
  saveAppIcon,
  saveTheme,
  type AppIconChoice,
  type ThemeChoice,
} from "../../utils/theme";
import { AuthMessage, Field, PasswordInput } from "./LoginPage";
import "../css/auth.css";
import "../css/settings.css";
import { asset } from "../../utils/assets";

/** Account details and actions supplied by App (via HomePage). */
type Props = {
  avatarUrl: string | null;
  email: string;
  googleConnected: boolean;
  googleConnecting: boolean;
  hasPassword: boolean;
  name: string;
  providers: string[];
  onChangePhoto: () => void;
  onConnectGoogle: () => void;
  onDisconnectGoogle: () => Promise<{ revoked: boolean }>;
  onChangePassword: (
    current: string | null,
    next: string,
  ) => Promise<string | null>;
  onSendReset: () => Promise<string | null>;
};

/** Sections in the order they appear, with their menu icons. */
const SECTIONS = [
  { id: "appearance", label: "Appearance", icon: "palette" },
  { id: "account", label: "Account", icon: "user" },
  { id: "integrations", label: "Integrations", icon: "link" },
  { id: "security", label: "Security", icon: "lock" },
] as const;
/** Id of a settings section (also its element id for scrolling). */
type SectionId = (typeof SECTIONS)[number]["id"];

/** Human-readable name for a Supabase sign-in provider. */
const providerName = (provider: string) =>
  provider === "email"
    ? "Email & password"
    : provider === "google"
      ? "Google"
      : provider === "azure"
        ? "Microsoft"
        : provider;

/** Settings page: section menu on the left, cards on the right. */
export default function SettingsPage(p: Props) {
  const [active, setActive] = useState<SectionId>("appearance");
  const scroller = useRef<HTMLElement>(null);

  // Highlight the menu item for whichever section is currently in view.
  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (a, b) => a.boundingClientRect.top - b.boundingClientRect.top,
          )[0];
        if (visible) setActive(visible.target.id as SectionId);
      },
      { root, rootMargin: "0px 0px -60% 0px" },
    );
    root
      .querySelectorAll(".settings-section")
      .forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <section className="settings-page" ref={scroller}>
      <header className="settings-hero">
        <p>Settings</p>
        <h1>Make Iris yours</h1>
        <span>Appearance, your account, connected apps and security.</span>
      </header>

      <div className="settings-layout">
        {/* In-page menu: jumps to a section and tracks the one in view */}
        <nav className="settings-nav" aria-label="Settings sections">
          {SECTIONS.map((section) => (
            <button
              aria-current={active === section.id ? "true" : undefined}
              className={active === section.id ? "active" : ""}
              key={section.id}
              onClick={() => {
                setActive(section.id);
                document
                  .getElementById(section.id)
                  ?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
              type="button"
            >
              <SettingsIcon name={section.icon} />
              {section.label}
            </button>
          ))}
        </nav>

        <div className="settings-sections">
          <Section
            description="Choose how Iris looks on this computer."
            icon="palette"
            id="appearance"
            title="Appearance"
          >
            <ThemePicker />
            <AppIconPicker />
          </Section>

          <Section
            description="Your profile and the ways you can sign in."
            icon="user"
            id="account"
            title="Account"
          >
            <ProfileRow {...p} />
          </Section>

          <Section
            description="Apps connected to your Iris account."
            icon="link"
            id="integrations"
            title="Integrations"
          >
            <GoogleCalendarRow {...p} />
          </Section>

          <Section
            description="Keep your account secure."
            icon="lock"
            id="security"
            title="Security"
          >
            <PasswordRow {...p} />
          </Section>
        </div>
      </div>
    </section>
  );
}

/** A titled settings card with an icon badge. */
function Section({
  children,
  description,
  icon,
  id,
  title,
}: {
  children: ReactNode;
  description: string;
  icon: IconName;
  id: SectionId;
  title: string;
}) {
  return (
    <section className="settings-section" id={id}>
      <header>
        <span className="settings-badge">
          <SettingsIcon name={icon} />
        </span>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </header>
      <div className="settings-section-body">{children}</div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Appearance
// ---------------------------------------------------------------------------

const THEMES: { id: ThemeChoice; label: string; hint: string }[] = [
  { id: "light", label: "Light", hint: "Bright and airy" },
  { id: "dark", label: "Dark", hint: "Easy on the eyes" },
  { id: "system", label: "System", hint: "Match your Mac" },
];

/** Light / Dark / System cards, each with a miniature preview of the app. */
function ThemePicker() {
  const [theme, setTheme] = useState<ThemeChoice>(loadTheme);
  return (
    <div className="settings-group">
      <h3>Theme</h3>
      <div className="choice-grid" role="radiogroup" aria-label="Theme">
        {THEMES.map((option) => (
          <button
            aria-checked={theme === option.id}
            className={`choice-card ${theme === option.id ? "selected" : ""}`}
            key={option.id}
            onClick={() => {
              setTheme(option.id);
              saveTheme(option.id);
            }}
            role="radio"
            type="button"
          >
            {/* Miniature window. System stacks a dark copy over a light one,
                cut along a diagonal. */}
            {option.id === "system" ? (
              <span className="theme-preview system" aria-hidden="true">
                <MiniWindow theme="light" />
                <MiniWindow theme="dark" />
              </span>
            ) : (
              <span className="theme-preview" aria-hidden="true">
                <MiniWindow theme={option.id} />
              </span>
            )}
            <span className="choice-label">
              <strong>{option.label}</strong>
              <small>{option.hint}</small>
            </span>
            <CheckMark />
          </button>
        ))}
      </div>
    </div>
  );
}

/** A tiny drawing of the app (sidebar, title, two cards) in one theme. */
function MiniWindow({ theme }: { theme: "light" | "dark" }) {
  return (
    <span className={`mini-window ${theme}`}>
      <i className="tp-sidebar" />
      <i className="tp-title" />
      <i className="tp-card" />
      <i className="tp-card short" />
    </span>
  );
}

const APP_ICONS: { id: AppIconChoice; label: string; image: string | null }[] =
  [
    { id: "auto", label: "Auto", image: null },
    { id: "light", label: "Light", image: asset("iris-dock-light.png") },
    { id: "dark", label: "Dark", image: asset("iris-dock-dark.png") },
    { id: "mono", label: "Monochrome", image: asset("iris-dock-mono.png") },
  ];

/** Dock icon choices; Auto switches between Light and Dark with the theme. */
function AppIconPicker() {
  const [icon, setIcon] = useState<AppIconChoice>("auto");
  const [note, setNote] = useState("");
  // Show the icon Electron has saved.
  useEffect(() => {
    void loadAppIcon().then(setIcon);
  }, []);
  return (
    <div className="settings-group">
      <h3>App icon</h3>
      <p className="settings-help">Shown in your Dock while Iris is open.</p>
      <div className="icon-grid" role="radiogroup" aria-label="App icon">
        {APP_ICONS.map((option) => (
          <button
            aria-checked={icon === option.id}
            className={`icon-choice ${icon === option.id ? "selected" : ""}`}
            key={option.id}
            onClick={async () => {
              const previous = icon;
              setIcon(option.id);
              setNote("");
              if (!(await saveAppIcon(option.id))) {
                setIcon(previous);
                setNote(
                  "Restart Iris to finish updating, then pick your icon again.",
                );
              }
            }}
            role="radio"
            type="button"
          >
            {option.image ? (
              <img alt="" src={option.image} />
            ) : (
              // Auto: half light icon, half dark icon
              <span className="icon-auto" aria-hidden="true">
                <img alt="" src={asset("iris-dock-light.png")} />
                <img alt="" src={asset("iris-dock-dark.png")} />
              </span>
            )}
            <span>{option.label}</span>
            <CheckMark />
          </button>
        ))}
      </div>
      {note && <p className="settings-note">{note}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Account
// ---------------------------------------------------------------------------

/** Photo, name, email and sign-in methods. */
function ProfileRow(p: Props) {
  return (
    <div className="profile-row">
      <button
        className="profile-avatar"
        onClick={p.onChangePhoto}
        title="Change photo"
        type="button"
      >
        {p.avatarUrl ? (
          <img alt="" src={p.avatarUrl} />
        ) : (
          <span>{p.name.slice(0, 1).toUpperCase()}</span>
        )}
        <em>Edit</em>
      </button>
      <div className="profile-copy">
        <strong>{p.name}</strong>
        <span>{p.email}</span>
        <div className="settings-tags">
          {p.providers.map((provider) => (
            <span key={provider}>{providerName(provider)}</span>
          ))}
        </div>
      </div>
      <button
        className="settings-button"
        onClick={p.onChangePhoto}
        type="button"
      >
        Change photo
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Integrations
// ---------------------------------------------------------------------------

/** Google Calendar connection status with Connect / Disconnect. */
function GoogleCalendarRow(p: Props) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<AuthNotice | null>(null);
  // Runs after the user confirms; reports whether access was revoked at Google.
  const disconnect = async () => {
    setBusy(true);
    const { revoked } = await p.onDisconnectGoogle();
    setBusy(false);
    setConfirming(false);
    setNotice(
      revoked
        ? {
            text: "Google Calendar is disconnected and Iris's access was removed.",
            tone: "success",
          }
        : {
            text: "Disconnected on this computer. To fully remove Iris's access, open your Google Account → Security → Third-party connections.",
            tone: "info",
          },
    );
  };
  return (
    <div className="integration">
      <div className="integration-row">
        <span className="integration-logo" aria-hidden="true">
          <GoogleCalendarLogo />
        </span>
        <div className="integration-copy">
          <strong>
            Google Calendar
            <span
              className={`settings-status ${p.googleConnected ? "on" : ""}`}
            >
              {p.googleConnected ? "Connected" : "Not connected"}
            </span>
          </strong>
          <span>
            {p.googleConnected
              ? "Your Google events appear in the Calendar tab, and new meetings get Google Meet links."
              : "See your Google events in Iris and create meetings with Google Meet links."}
          </span>
        </div>
        {!confirming &&
          (p.googleConnected ? (
            <button
              className="settings-button danger"
              onClick={() => {
                setNotice(null);
                setConfirming(true);
              }}
              type="button"
            >
              Disconnect
            </button>
          ) : (
            <button
              className="settings-button primary"
              disabled={p.googleConnecting}
              onClick={() => {
                setNotice(null);
                p.onConnectGoogle();
              }}
              type="button"
            >
              {p.googleConnecting ? "Finish in browser…" : "Connect"}
            </button>
          ))}
      </div>
      {confirming && (
        <div className="settings-confirm">
          <p>
            Disconnect Google Calendar? Your Google events will disappear from
            Iris and new meetings won't get Meet links until you reconnect.
            Nothing is deleted from your Google Calendar.
          </p>
          <div>
            <button
              className="settings-button"
              disabled={busy}
              onClick={() => setConfirming(false)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="settings-button danger solid"
              disabled={busy}
              onClick={() => void disconnect()}
              type="button"
            >
              {busy ? "Disconnecting…" : "Disconnect"}
            </button>
          </div>
        </div>
      )}
      <AuthMessage notice={notice} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Security
// ---------------------------------------------------------------------------

/**
 * Password row that expands into the Change password form (or Set a password
 * for Google-only accounts). App verifies the current password before saving.
 */
function PasswordRow(p: Props) {
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<AuthNotice | null>(null);
  // Validates locally first, then asks App to verify and update.
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const problem = passwordProblem(next, confirm);
    if (problem) return setNotice({ text: problem, tone: "error" });
    if (p.hasPassword && !current)
      return setNotice({ text: "Enter your current password.", tone: "error" });
    setBusy(true);
    const error = await p.onChangePassword(
      p.hasPassword ? current : null,
      next,
    );
    setBusy(false);
    if (error) return setNotice({ text: error, tone: "error" });
    setCurrent("");
    setNext("");
    setConfirm("");
    setOpen(false);
    setNotice({
      text: p.hasPassword
        ? "Your password has been changed."
        : "Password set. You can now also sign in with your email and password.",
      tone: "success",
    });
  };
  // "Forgot it?" — emails a reset link to the account's address.
  const sendReset = async () => {
    setBusy(true);
    const error = await p.onSendReset();
    setBusy(false);
    setNotice(
      error
        ? { text: error, tone: "error" }
        : { text: `We sent a reset link to ${p.email}.`, tone: "success" },
    );
  };
  return (
    <div className="integration">
      <div className="integration-row">
        <span className="integration-logo key" aria-hidden="true">
          <SettingsIcon name="lock" />
        </span>
        <div className="integration-copy">
          <strong>Password</strong>
          <span>
            {p.hasPassword
              ? "Change the password you use to sign in with your email."
              : "You sign in with Google. Add a password to also sign in with your email."}
          </span>
        </div>
        {!open && (
          <button
            className="settings-button"
            onClick={() => {
              setNotice(null);
              setOpen(true);
            }}
            type="button"
          >
            {p.hasPassword ? "Change password" : "Set password"}
          </button>
        )}
      </div>
      {open && (
        <form className="password-form" onSubmit={submit}>
          {p.hasPassword && (
            <Field
              aside={
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => void sendReset()}
                  type="button"
                >
                  Forgot it?
                </button>
              }
              htmlFor="settings-current"
              label="Current password"
            >
              <PasswordInput
                autoComplete="current-password"
                id="settings-current"
                onChange={setCurrent}
                placeholder="Current password"
                value={current}
              />
            </Field>
          )}
          <Field
            hint={PASSWORD_HINT}
            htmlFor="settings-new"
            label="New password"
          >
            <PasswordInput
              autoComplete="new-password"
              id="settings-new"
              onChange={setNext}
              placeholder="New password"
              value={next}
            />
          </Field>
          <Field htmlFor="settings-confirm" label="Confirm new password">
            <PasswordInput
              autoComplete="new-password"
              id="settings-confirm"
              onChange={setConfirm}
              placeholder="Type it again"
              value={confirm}
            />
          </Field>
          <div className="password-actions">
            <button
              className="settings-button"
              disabled={busy}
              onClick={() => {
                setOpen(false);
                setNotice(null);
              }}
              type="button"
            >
              Cancel
            </button>
            <button
              className="settings-button primary"
              disabled={busy}
              type="submit"
            >
              {busy
                ? "Saving…"
                : p.hasPassword
                  ? "Update password"
                  : "Set password"}
            </button>
          </div>
        </form>
      )}
      <AuthMessage notice={notice} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small icons
// ---------------------------------------------------------------------------

/** Names of the small line icons used in Settings. */
type IconName = "palette" | "user" | "link" | "lock";

/** Line icons for the section menu and badges (inherit the text colour). */
function SettingsIcon({ name }: { name: IconName }) {
  const paths: Record<IconName, string> = {
    palette:
      "M12 3a9 9 0 1 0 0 18c1.1 0 1.8-.9 1.8-1.8 0-.5-.2-.9-.5-1.2-.3-.3-.5-.8-.5-1.2 0-1 .8-1.8 1.8-1.8H17a4 4 0 0 0 4-4C21 6.6 17 3 12 3ZM7.5 12.5h.01M9.5 8h.01M14.5 8h.01M16.5 11.5h.01",
    user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4 21a8 8 0 0 1 16 0",
    link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    lock: "M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3",
  };
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height="18"
      viewBox="0 0 24 24"
      width="18"
    >
      <path
        d={paths[name]}
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

/** Check badge shown on the selected choice card. */
function CheckMark() {
  return (
    <span className="choice-check" aria-hidden="true">
      <svg fill="none" height="12" viewBox="0 0 24 24" width="12">
        <path
          d="M5 12.5 10 17l9-10"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="3"
        />
      </svg>
    </span>
  );
}

/** Simplified Google Calendar mark in Google's colours. */
function GoogleCalendarLogo() {
  return (
    <svg aria-hidden="true" height="26" viewBox="0 0 48 48" width="26">
      <path
        d="M36 8H12a4 4 0 0 0-4 4v24a4 4 0 0 0 4 4h24a4 4 0 0 0 4-4V12a4 4 0 0 0-4-4Z"
        fill="#fff"
      />
      <path d="M8 16V12a4 4 0 0 1 4-4h24a4 4 0 0 1 4 4v4Z" fill="#4285F4" />
      <path d="M40 32v4a4 4 0 0 1-4 4h-4v-8Z" fill="#EA4335" />
      <path d="M32 40H12a4 4 0 0 1-4-4v-4h24Z" fill="#34A853" />
      <path d="M8 16h4v16H8Z" fill="#FBBC04" />
      <text
        fill="#4285F4"
        fontFamily="Inter, Arial"
        fontSize="14"
        fontWeight="700"
        textAnchor="middle"
        x="24"
        y="30"
      >
        31
      </text>
    </svg>
  );
}
