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
  };
}
