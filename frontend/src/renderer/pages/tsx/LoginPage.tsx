import { useState, type FormEvent } from "react";
import type { AuthNotice } from "../../utils/auth";
import "../css/auth.css";
type Props = {
  email: string;
  password: string;
  notice: AuthNotice | null;
  busy: boolean;
  onEmail: (v: string) => void;
  onPassword: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onSwitch: () => void;
  onForgot: () => void;
  onGoogle: () => void;
};
export default function LoginPage(p: Props) {
  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to continue making meetings more inclusive."
    >
      <form className="sign-in-card" onSubmit={p.onSubmit}>
        <Field htmlFor="signin-email" label="Email">
          <input
            autoComplete="email"
            id="signin-email"
            onChange={(e) => p.onEmail(e.target.value)}
            placeholder="you@example.com"
            required
            type="email"
            value={p.email}
          />
        </Field>
        <Field
          htmlFor="signin-password"
          label="Password"
          aside={
            <button className="text-button" onClick={p.onForgot} type="button">
              Forgot password?
            </button>
          }
        >
          <PasswordInput
            autoComplete="current-password"
            id="signin-password"
            onChange={p.onPassword}
            value={p.password}
          />
        </Field>
        <button className="primary-button" disabled={p.busy} type="submit">
          {p.busy ? "Signing in…" : "Sign in"}
        </button>
        <div className="auth-divider">
          <span>or</span>
        </div>
        {/* Microsoft sign-in is paused for now; add it back here when ready. */}
        <div className="provider-buttons">
          <button onClick={p.onGoogle} type="button">
            <GoogleLogo />
            Continue with Google
          </button>
        </div>
        <AuthMessage notice={p.notice} />
        <p className="create-account">
          Don't have an account?{" "}
          <button className="text-button" onClick={p.onSwitch} type="button">
            Create account
          </button>
        </p>
      </form>
    </AuthShell>
  );
}
export function AuthShell({
  children,
  title,
  subtitle,
}: {
  children: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <main className="login-shell">
      <section className="brand-panel">
        {/* Decorative layers: soft light orbs and a large Iris flower. */}
        <span aria-hidden="true" className="brand-orb orb-glow" />
        <span aria-hidden="true" className="brand-orb orb-teal" />
        <img
          alt=""
          aria-hidden="true"
          className="brand-flower"
          src="/brand/iris-flower-solid.svg"
        />
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
          <ul className="brand-features">
            <li>
              <span aria-hidden="true">✦</span> ASL to speech
            </li>
            <li>
              <span aria-hidden="true">✦</span> Live captions
            </li>
            <li>
              <span aria-hidden="true">✦</span> Works with Google Meet
            </li>
          </ul>
        </div>
        <p className="brand-footnote">Built for signers and hearing teammates alike.</p>
      </section>
      <section className="sign-in-panel" id="sign-in">
        <div className="sign-in-content">
          <p className="kicker">WELCOME TO IRIS</p>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
          {children}
        </div>
      </section>
    </main>
  );
}
export function Field({
  aside,
  children,
  hint,
  htmlFor,
  label,
}: {
  aside?: React.ReactNode;
  children: React.ReactNode;
  hint?: string;
  htmlFor: string;
  label: string;
}) {
  return (
    <div className="field-group">
      <span className="field-heading">
        <label className="field-label" htmlFor={htmlFor}>
          {label}
        </label>
        {aside}
      </span>
      {children}
      {hint && <small className="field-hint">{hint}</small>}
    </div>
  );
}
/** Status line under an auth form, with an optional follow-up button. */
export function AuthMessage({ notice }: { notice: AuthNotice | null }) {
  if (!notice) return null;
  return (
    <div className={`form-message ${notice.tone}`} role="status">
      <span>{notice.text}</span>
      {notice.action && (
        <button className="text-button" onClick={notice.action.run} type="button">
          {notice.action.label}
        </button>
      )}
    </div>
  );
}
/** Password field with a show/hide toggle, shared by sign-in and sign-up. */
export function PasswordInput({
  autoComplete,
  id,
  name,
  onChange,
  placeholder = "Enter your password",
  value,
}: {
  autoComplete: string;
  id: string;
  name?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  value: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <span className="password-field">
      <input
        autoComplete={autoComplete}
        id={id}
        name={name}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required
        type={visible ? "text" : "password"}
        value={value}
      />
      <button
        aria-label={visible ? "Hide password" : "Show password"}
        onClick={() => setVisible((v) => !v)}
        type="button"
      >
        {visible ? "Hide" : "Show"}
      </button>
    </span>
  );
}
function GoogleLogo() {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" width="20" height="20">
      <path
        fill="#FFC107"
        d="M43.6 20.1H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.6-.4-3.9z"
      />
      <path
        fill="#FF3D00"
        d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"
      />
      <path
        fill="#4CAF50"
        d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-7.9l-6.5 5C9.5 39.6 16.2 44 24 44z"
      />
      <path
        fill="#1976D2"
        d="M43.6 20.1H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.6-.4-3.9z"
      />
    </svg>
  );
}
