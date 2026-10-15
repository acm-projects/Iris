import { useState } from "react";
import { engineError } from "../../utils/translation";
import "../css/popout.css";

const QUICK_PHRASES = [
  "Hello everyone!",
  "Could you repeat that?",
  "One moment, please.",
  "Thank you!",
];

/** The floating Type to speak window (no sign-in, no sidebar). */
export default function PopoutPage() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Sends the text to the main window's speak(); shows errors under the box.
  const say = async (value: string) => {
    if (!value.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      await window.iris.popout.speak(value.trim());
      setText("");
    } catch (problem) {
      setError(engineError(problem, "Iris couldn't speak that."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="popout">
      {/* Dragging this bar moves the window */}
      <header className="popout-bar">
        <strong>Type to Speak</strong>
        <div className="popout-bar-actions">
          {/* Put the box back inside the Iris window */}
          <button
            aria-label="Return to Iris"
            onClick={() => void window.iris.popout.dock()}
            title="Return to Iris"
            type="button"
          >
            <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
              <path
                d="M10 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4M20 4l-8 8M12 7v5h5"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.8"
              />
            </svg>
          </button>
          <button
            aria-label="Close"
            onClick={() => window.close()}
            title="Close"
            type="button"
          >
            ✕
          </button>
        </div>
      </header>
      <textarea
        disabled={busy}
        maxLength={500}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void say(text);
          }
        }}
        placeholder="Type a message and press Enter…"
        value={text}
      />
      <div className="popout-phrases">
        {QUICK_PHRASES.map((phrase) => (
          <button
            disabled={busy}
            key={phrase}
            onClick={() => void say(phrase)}
            type="button"
          >
            {phrase}
          </button>
        ))}
      </div>
      <button
        className="popout-speak"
        disabled={busy || !text.trim()}
        onClick={() => void say(text)}
        type="button"
      >
        {busy ? "Speaking…" : "Speak"}
      </button>
      {error && <p className="popout-error">{error}</p>}
    </main>
  );
}
