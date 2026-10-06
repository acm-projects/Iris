// Sign-in helpers shared by the auth screens and Settings: password rules
// and plain-language versions of Supabase's error messages.
import type { AuthError } from "@supabase/supabase-js";

/** A message shown on the auth screens, optionally with a follow-up action. */
export type AuthNotice = {
  text: string;
  tone: "error" | "success" | "info";
  action?: { label: string; run: () => void };
};

export const MIN_PASSWORD_LENGTH = 8;
export const PASSWORD_HINT = `At least ${MIN_PASSWORD_LENGTH} characters, with uppercase and lowercase letters and a number.`;

/** Returns why a new password is not acceptable, or null if it is fine. */
export function passwordProblem(password: string, confirm?: string) {
  if (password.length < MIN_PASSWORD_LENGTH)
    return `Use at least ${MIN_PASSWORD_LENGTH} characters for your password.`;
  // Mirrors the Supabase project's password policy (lower, upper, digit).
  if (
    !/[a-z]/.test(password) ||
    !/[A-Z]/.test(password) ||
    !/\d/.test(password)
  )
    return "Use at least one lowercase letter, one uppercase letter, and one number.";
  if (confirm !== undefined && password !== confirm)
    return "The two passwords don't match.";
  return null;
}

/** Turns Supabase auth errors into plain-language messages. */
export function friendlyAuthError(error: AuthError) {
  // PKCE links only work in the app that requested them.
  if (error.name === "AuthPKCECodeVerifierMissingError")
    return "Open the link on the same computer where you requested it, or request a new one.";
  switch (error.code) {
    case "flow_state_not_found":
    case "flow_state_expired":
      return "That link has expired or was already used. Request a new one.";
    case "invalid_credentials":
      return "That email and password don't match. Check them, or reset your password.";
    case "email_not_confirmed":
      return "Confirm your email before signing in. Check your inbox for the link.";
    case "user_already_exists":
    case "email_exists":
      return "An account with this email already exists.";
    case "weak_password":
      return "That password is too weak. Use at least 8 characters with lowercase and uppercase letters and a number.";
    case "same_password":
      return "Your new password must be different from your current one.";
    case "over_email_send_rate_limit":
      return "Too many emails were sent recently. Wait a few minutes and try again.";
    case "over_request_rate_limit":
      return "Too many attempts. Wait a moment and try again.";
    case "reauthentication_needed":
      return "For security, sign out and back in, then change your password.";
    case "otp_expired":
      return "That link has expired. Request a new one.";
    default:
      return error.message || "Something went wrong. Please try again.";
  }
}
