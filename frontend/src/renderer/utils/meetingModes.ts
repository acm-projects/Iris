// Accessibility mode of a meeting: what Iris should do while it's happening.
// Stored as text in the meetings table (accessibility_mode) and applied to
// the Translation page's caption setting when the meeting is on.

/** ASL is translated to speech AND shown as captions on the video. */
export const ASL_WITH_CAPTIONS = "Captions + Sign translation";
/** ASL is translated to speech only; no captions on the video. */
export const ASL_ONLY = "Sign translation only";

/** Short label for a stored mode ("ASL + Captions" / "ASL only"). */
export function modeLabel(mode: string | null | undefined) {
  return mode === ASL_ONLY ? "ASL only" : "ASL + Captions";
}

/**
 * Should captions be shown for this mode? Only "ASL only" turns them off.
 * Older meetings saved as "Captions only" (or anything unknown) keep
 * captions on, so nobody loses captions they were expecting.
 */
export function showsCaptions(mode: string | null | undefined) {
  return mode !== ASL_ONLY;
}
