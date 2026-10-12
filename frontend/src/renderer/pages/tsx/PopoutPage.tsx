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
        <strong>Type to speak</strong>
        <button onClick={() => window.close()} type="button" aria-label="Close">
          ✕
        </button>
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