import type { FormEvent } from "react";
type Props = {
  email: string;
  password: string;
  message: string;
  busy: boolean;
  onEmail: (v: string) => void;
  onPassword: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onSwitch: () => void;
  onOAuth: (p: "google" | "azure") => void;
};
export default function LoginPage(p: Props) {
  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to continue making meetings more inclusive."
    >
      <form className="sign-in-card" onSubmit={p.onSubmit}>
        <Field label="Email">
          <input
            autoComplete="email"
            onChange={(e) => p.onEmail(e.target.value)}
            type="email"
            value={p.email}
          />
        </Field>
        <Field label="Password">
          <input
            autoComplete="current-password"
            onChange={(e) => p.onPassword(e.target.value)}
            type="password"
            value={p.password}
          />
        </Field>
        <button className="primary-button" disabled={p.busy} type="submit">
          Sign in
        </button>
        <div className="provider-buttons">
          <button onClick={() => p.onOAuth("google")} type="button">
            Continue with Google
          </button>
          <button onClick={() => p.onOAuth("azure")} type="button">
            Continue with Microsoft
          </button>
        </div>
        <p className="form-message">{p.message}</p>
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
        <div>
          <p className="kicker">WELCOME TO IRIS</p>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
          {children}
        </div>
      </section>
    </main>
  );
}
function Field({
  children,
  label,
}: {
  children: React.ReactNode;
  label: string;
}) {
  return (
    <label className="field-group">
      {label}
      {children}
    </label>
  );
}
