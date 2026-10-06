/**
 * Event colours users can pick on the calendar, limited to the Iris brand
 * palette. Each maps to the nearest Google Calendar colorId so a choice made
 * in Iris also shows up in Google Calendar.
 */
export const EVENT_COLORS = [
  { key: "purple", label: "Iris Purple", hex: "#5c4bab", googleId: "3" },
  { key: "navy", label: "Deep Navy", hex: "#00224e", googleId: "9" },
  { key: "blue", label: "Iris Blue", hex: "#4376ab", googleId: "7" },
  { key: "teal", label: "Turquoise", hex: "#0d9da4", googleId: "2" },
  { key: "gold", label: "Brand Gold", hex: "#f6bf41", googleId: "5" },
  { key: "yellow", label: "Soft Yellow", hex: "#fce688", googleId: "5" },
] as const;

export type EventColorKey = (typeof EVENT_COLORS)[number]["key"];

/** Default colours: Iris meetings are purple, other Google events blue. */
export const DEFAULT_IRIS_COLOR: EventColorKey = "purple";
export const DEFAULT_GOOGLE_COLOR: EventColorKey = "blue";

export const isEventColor = (value: unknown): value is EventColorKey =>
  EVENT_COLORS.some((color) => color.key === value);

export const colorHex = (key: EventColorKey) =>
  EVENT_COLORS.find((color) => color.key === key)?.hex ?? "#4376ab";

export const googleIdFor = (key: EventColorKey) =>
  EVENT_COLORS.find((color) => color.key === key)?.googleId ?? "7";

// Google's eleven event colours folded onto the nearest brand colour.
const FROM_GOOGLE: Record<string, EventColorKey> = {
  "1": "purple", // Lavender
  "2": "teal", // Sage
  "3": "purple", // Grape
  "4": "gold", // Flamingo
  "5": "gold", // Banana
  "6": "gold", // Tangerine
  "7": "blue", // Peacock
  "8": "navy", // Graphite
  "9": "navy", // Blueberry
  "10": "teal", // Basil
  "11": "gold", // Tomato
};
export const colorFromGoogle = (colorId: string | null | undefined) =>
  colorId ? (FROM_GOOGLE[colorId] ?? null) : null;
