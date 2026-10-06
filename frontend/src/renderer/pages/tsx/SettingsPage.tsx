import { useState, type FormEvent } from "react";
import { PASSWORD_HINT, passwordProblem, type AuthNotice } from "../../utils/auth";
import { AuthMessage, Field, PasswordInput } from "./LoginPage";
import "../css/auth.css";
import "../css/settings.css";

type Props = {
  email: string;
  hasPassword: boolean;
  providers: string[];
  onChangePassword: (current: string | null, next: string) => Promise<string | null>;
  onSendReset: () => Promise<string | null>;
};

const providerName = (provider: string) =>
  provider === "email"
    ? "Email & password"
    : provider === "google"
      ? "Google"
      : provider === "azure"
        ? "Microsoft"
        : provider;

export default function SettingsPage(p: Props) {
  return (
    <section className="settings-page">
      <header>
        <p>Settings</p>
        <h1>Account</h1>
      </header>
      <div className="settings-card">
        <h2>Profile</h2>
        <dl>
          <div>
            <dt>Email</dt>
            <dd>{p.email}</dd>
          </div>
          <div>
            <dt>Sign-in methods</dt>
            <dd className="settings-tags">
              {p.providers.map((provider) => (
                <span key={provider}>{providerName(provider)}</span>
              ))}
            </dd>
          </div>
        </dl>
      </div>
      <PasswordCard {...p} />
    </section>
  );
}

function PasswordCard(p: Props) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<AuthNotice | null>(null);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const problem = passwordProblem(next, confirm);
    if (problem) return setNotice({ text: problem, tone: "error" });
    if (p.hasPassword && !current)
      return setNotice({ text: "Enter your current password.", tone: "error" });
    setBusy(true);
    const error = await p.onChangePassword(p.hasPassword ? current : null, next);
    setBusy(false);
    if (error) return setNotice({ text: error, tone: "error" });
    setCurrent("");
    setNext("");
    setConfirm("");
    setNotice({
      text: p.hasPassword
        ? "Your password has been changed."
        : "Password set. You can now also sign in with your email and password.",
      tone: "success",
    });
  };
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
    <form className="settings-card" onSubmit={submit}>
      <h2>{p.hasPassword ? "Change password" : "Set a password"}</h2>
      <p className="settings-copy">
        {p.hasPassword
          ? "Enter your current password, then choose a new one."
          : "You signed up with Google. Add a password to also sign in with your email."}
      </p>
      <div className="settings-fields">
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
      </div>
      <AuthMessage notice={notice} />
      <button className="primary-button" disabled={busy} type="submit">
        {busy ? "Saving…" : p.hasPassword ? "Update password" : "Set password"}
      </button>
    </form>
  );
}
