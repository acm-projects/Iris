/** Types for the minimal API exposed by Electron's preload script. */
interface Window {
  // This property exists only because preload.ts exposes it securely.
  iris: {
    platform: NodeJS.Platform;
    /** Theme for native UI, and the dock icon style chosen in Settings. */
    app: {
      setTheme(theme: "light" | "dark" | "system"): Promise<void>;
      getIcon(): Promise<"auto" | "light" | "dark" | "mono">;
      setIcon(choice: "auto" | "light" | "dark" | "mono"): Promise<void>;
      flushStorage(): Promise<void>;
    };
    /** Browser sign-in, iris:// callbacks, and encrypted storage. */
    auth: {
      openExternal(url: string): Promise<void>;
      consumeCallback(): Promise<string | null>;
      secureGet(key: string): Promise<string | null>;
      secureSet(key: string, value: string | null): Promise<void>;
      onCallback(listener: (url: string) => void): () => void;
    };
    /** macOS camera/microphone permission. */
    media: {
      status(): Promise<{ camera: string; microphone: string }>;
      request(kind: "camera" | "microphone"): Promise<boolean>;
    };
    /** The Python translation engine (virtual camera and microphone). */
    engine: {
      request<T = unknown>(method: string, params?: unknown): Promise<T>;
      sendFrame(jpeg: Uint8Array): void;
      onEvent(listener: (message: unknown) => void): () => void;
    };
    /** Opening Google Meet links. */
    meetings: {
      openExternal(url: string): Promise<void>;
    };
    /** Google Calendar actions; each takes the user's Google access token. */
    google: {
      /** Creates a calendar event with a new Google Meet link. */
      createMeetEvent(
        accessToken: string,
        request: {
          title: string;
          startsAt: string;
          endsAt: string;
          description?: string;
        },
      ): Promise<{
        eventId: string | null;
        htmlLink: string | null;
        meetUrl: string;
      }>;
      /** Lists primary-calendar events between two ISO timestamps. */
      listEvents(
        accessToken: string,
        timeMin: string,
        timeMax: string,
      ): Promise<
        {
          id: string;
          title: string;
          start: string;
          end: string;
          allDay: boolean;
          meetUrl: string | null;
          htmlLink: string | null;
          location: string | null;
          colorId: string | null;
        }[]
      >;
      /** Sets an event's Google colorId ("1"–"11"). */
      setEventColor(
        accessToken: string,
        eventId: string,
        colorId: string,
      ): Promise<void>;
      /** Moves or resizes an event (new start and end as ISO timestamps). */
      setEventTimes(
        accessToken: string,
        eventId: string,
        startsAt: string,
        endsAt: string,
      ): Promise<void>;
      /** Opens an event's Google Calendar page in the browser. */
      openEvent(url: string): Promise<void>;
      /** Deletes an event from the primary calendar. */
      deleteEvent(accessToken: string, eventId: string): Promise<void>;
      /** Revokes Iris's access at Google; true if Google accepted it. */
      revoke(token: string): Promise<boolean>;
    };
  };
}
