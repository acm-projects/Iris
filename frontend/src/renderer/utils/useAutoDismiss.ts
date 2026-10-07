// Hides a temporary message (toast, banner, note) a few seconds after it
// appears. The timer pauses while the mouse is over the message so buttons in
// it (e.g. "Connect Google") can still be clicked, and restarts on leave.
import { useEffect, useRef, useState } from "react";

/** How long temporary messages stay on screen. */
export const MESSAGE_DURATION_MS = 3000;

/**
 * Calls `dismiss` once `active` (the message) has been shown for `delay` ms.
 * Spread the returned handlers onto the message element.
 */
export function useAutoDismiss(
  active: unknown,
  dismiss: () => void,
  delay = MESSAGE_DURATION_MS,
) {
  const [hovered, setHovered] = useState(false);
  // Keep the latest callback without restarting the timer on every render.
  const dismissRef = useRef(dismiss);
  useEffect(() => {
    dismissRef.current = dismiss;
  });
  useEffect(() => {
    // A message that disappears while hovered must not leave the timer paused.
    if (!active) setHovered(false);
  }, [active]);
  useEffect(() => {
    if (!active || hovered) return;
    const timer = window.setTimeout(() => dismissRef.current(), delay);
    return () => window.clearTimeout(timer);
  }, [active, hovered, delay]);
  return {
    onMouseEnter: () => setHovered(true),
    onMouseLeave: () => setHovered(false),
  };
}
