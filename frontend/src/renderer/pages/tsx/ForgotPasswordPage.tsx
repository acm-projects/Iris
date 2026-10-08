// Forgot password screen: requests a password reset email.
import type { FormEvent } from "react";
import type { AuthNotice } from "../../utils/auth";
import { AuthMessage, AuthShell, Field } from "./LoginPage";

/** Asks for an email and sends a password reset link to it. */
export default function ForgotPasswordPage(p: {
  busy: boolean;
  email: string;
  notice: AuthNotice | null;
  onBack: () => void;
  onEmail: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <AuthShell
      title="Reset your password"
      subtitle="Enter your account email and we'll send you a link to choose a new password."
    >
      <form className="sign-in-card" onSubmit={p.onSubmit}>
        <Field htmlFor="forgot-email" label="Email">
          <input
            autoComplete="email"
            autoFocus
            id="forgot-email"
            onChange={(e) => p.onEmail(e.target.value)}
            placeholder="you@example.com"
            required
            type="email"
            value={p.email}
          />
        </Field>
        <button className="primary-button" disabled={p.busy} type="submit">
          {p.busy ? "Sending…" : "Send reset link"}
        </button>
        <AuthMessage notice={p.notice} />
        <p className="create-account">
          Remembered it?{" "}
          <button className="text-button" onClick={p.onBack} type="button">
            Back to sign in
          </button>
        </p>
      </form>
    </AuthShell>
  );
}
