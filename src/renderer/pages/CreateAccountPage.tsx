import type { FormEvent } from "react";
import { AuthShell } from "./LoginPage";
type Props = {
  email: string;
  password: string;
  message: string;
  busy: boolean;
  onEmail: (v: string) => void;
  onPassword: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onSwitch: () => void;
};
export default function CreateAccountPage(p: Props) {
  return (
    <AuthShell
      title="Create your account"
      subtitle="Start making meetings more inclusive."
    >
      <form className="sign-in-card" onSubmit={p.onSubmit}>
        <label className="field-group">
          Email
          <input
            autoComplete="email"
            onChange={(e) => p.onEmail(e.target.value)}
            type="email"
            value={p.email}
          />
        </label>
        <label className="field-group">
          Password
          <input
            autoComplete="new-password"
            onChange={(e) => p.onPassword(e.target.value)}
            type="password"
            value={p.password}
          />
        </label>
        <button className="primary-button" disabled={p.busy} type="submit">
          Create account
        </button>
        <p className="form-message">{p.message}</p>
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
