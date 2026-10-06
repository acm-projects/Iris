/** Types for the minimal API exposed by Electron's preload script. */
interface Window {
  // This property exists only because preload.ts exposes it securely.
  iris: {
    platform: NodeJS.Platform;
    auth: {
      openExternal(url: string): Promise<void>;
      consumeCallback(): Promise<string | null>;
      secureGet(key: string): Promise<string | null>;
      secureSet(key: string, value: string | null): Promise<void>;
      canPersistSecurely(): Promise<boolean>;
      onCallback(listener: (url: string) => void): () => void;
    };
    meetings: {
      openExternal(url: string): Promise<void>;
    };
    google: {
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
      setEventColor(
        accessToken: string,
        eventId: string,
        colorId: string,
      ): Promise<void>;
      openEvent(url: string): Promise<void>;
      deleteEvent(accessToken: string, eventId: string): Promise<void>;
    };
  };
}
