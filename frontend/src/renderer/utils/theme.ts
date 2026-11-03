// Appearance helpers: Light / Dark / System theme. The dock icon follows the
// theme automatically (Electron picks it in main.ts).
// The theme is stored in localStorage (a per-computer preference) and applied
// as <html data-theme="light|dark">, which theme.css uses for dark colours.
import { saveLocal } from "./storage";

/** Theme options in Settings → Appearance. */
export type ThemeChoice = "light" | "dark" | "system";

const themeKey = "iris-theme";
// Media query that reports whether the OS (or Electron's theme) is dark.
const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)");

/** The saved theme choice, defaulting to Light. */
export function loadTheme(): ThemeChoice {
  try {
    const saved = localStorage.getItem(themeKey);
    if (saved === "light" || saved === "dark" || saved === "system")
      return saved;
  } catch {
    /* storage unavailable: use the default */
  }
  return "light";
}

/** Turns "system" into the light/dark appearance the OS is using right now. */
export function resolveTheme(choice: ThemeChoice): "light" | "dark" {
  if (choice !== "system") return choice;
  return darkQuery().matches ? "dark" : "light";
}

/** Applies a theme to the page (and tells Electron so the title bar matches). */
function applyTheme(choice: ThemeChoice) {
  const resolved = resolveTheme(choice);
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
  // Older running copies of Iris may not have this bridge yet.
  void window.iris?.app?.setTheme?.(choice).catch(() => undefined);
}

/** Saves and applies a new theme choice. */
export function saveTheme(choice: ThemeChoice) {
  saveLocal(themeKey, choice);
  applyTheme(choice);
}

/** Call once before rendering: applies the saved theme and follows OS changes. */
export function initTheme() {
  applyTheme(loadTheme());
  darkQuery().addEventListener("change", () => {
    if (loadTheme() === "system") applyTheme("system");
  });
}
