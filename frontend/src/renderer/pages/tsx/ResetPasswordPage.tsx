// Choose a new password: shown after a reset link signs the user in.
import { useState } from "react";
import {
  PASSWORD_HINT,
  passwordProblem,
  type AuthNotice,
} from "../../utils/auth";
import { AuthMessage, AuthShell, Field, PasswordInput } from "./LoginPage";

/** Shown after a password reset link opens Iris: choose a new password. */
export default function ResetPasswordPage(p: {
  email: string;
  onSave: (password: string) => Promise<string | null>;
  onSkip: () => void;
}) {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<AuthNotice | null>(null);
  return (
    <AuthShell
      title="Choose a new password"
      subtitle={`Set a new password for ${p.email}.`}
    >
      <form
        className="sign-in-card"
        onSubmit={async (event) => {
          event.preventDefault();
          const problem = passwordProblem(password, confirm);
          if (problem) return setNotice({ text: problem, tone: "error" });
          setBusy(true);
          const error = await p.onSave(password);
          setBusy(false);
          if (error) setNotice({ text: error, tone: "error" });
        }}
      >
        <Field
          hint={PASSWORD_HINT}
          htmlFor="reset-password"
          label="New password"
        >
          <PasswordInput
            autoComplete="new-password"
            id="reset-password"
            onChange={setPassword}
            placeholder="Create a new password"
            value={password}
          />
        </Field>
        <Field htmlFor="reset-confirm" label="Confirm new password">
          <PasswordInput
            autoComplete="new-password"
            id="reset-confirm"
            onChange={setConfirm}
            placeholder="Type it again"
            value={confirm}
          />
        </Field>
        <button className="primary-button" disabled={busy} type="submit">
          {busy ? "Saving…" : "Save new password"}
        </button>
        <AuthMessage notice={notice} />
        <p className="create-account">
          <button className="text-button" onClick={p.onSkip} type="button">
            Skip for now
          </button>
        </p>
      </form>
    </AuthShell>
  );
}
