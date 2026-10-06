import { useState, type FormEvent } from "react";
import { PASSWORD_HINT, type AuthNotice } from "../lib/auth";
import { AuthMessage, AuthShell, Field, PasswordInput } from "./LoginPage";
type Props = {
  email: string;
  password: string;
  notice: AuthNotice | null;
  busy: boolean;
  onEmail: (v: string) => void;
  onPassword: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onSwitch: () => void;
};
export default function CreateAccountPage(p: Props) {
  const [confirm, setConfirm] = useState("");
  return (
    <AuthShell
      title="Create your account"
      subtitle="Start making meetings more inclusive."
    >
      <form className="sign-in-card" onSubmit={p.onSubmit}>
        <Field htmlFor="signup-email" label="Email">
          <input
            autoComplete="email"
            id="signup-email"
            onChange={(e) => p.onEmail(e.target.value)}
            placeholder="you@example.com"
            required
            type="email"
            value={p.email}
          />
        </Field>
        <Field
          hint={PASSWORD_HINT}
          htmlFor="signup-password"
          label="Password"
        >
          <PasswordInput
            autoComplete="new-password"
            id="signup-password"
            onChange={p.onPassword}
            placeholder="Create a password"
            value={p.password}
          />
        </Field>
        <Field htmlFor="signup-confirm" label="Confirm password">
          <PasswordInput
            autoComplete="new-password"
            id="signup-confirm"
            name="confirm"
            onChange={setConfirm}
            placeholder="Type it again"
            value={confirm}
          />
        </Field>
        <button className="primary-button" disabled={p.busy} type="submit">
          {p.busy ? "Creating account…" : "Create account"}
        </button>
        <AuthMessage notice={p.notice} />
        <p className="create-account">
          Already have an account?{" "}
          <button className="text-button" onClick={p.onSwitch} type="button">
            Sign in
          </button>
        </p>
      </form>
    </AuthShell>
  );
}
