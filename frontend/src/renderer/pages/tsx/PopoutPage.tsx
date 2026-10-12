import { useState } from "react";
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

  // TODO (next step): send this to the main window through the main process.
  const say = (value: string) => {
    if (!value.trim()) return;
    console.log("would speak:", value.trim());
    setText("");
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
        maxLength={500}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            say(text);
          }
        }}
        placeholder="Type a message and press Enter…"
        value={text}
      />
      <div className="popout-phrases">
        {QUICK_PHRASES.map((phrase) => (
          <button key={phrase} onClick={() => say(phrase)} type="button">
            {phrase}
          </button>
        ))}
      </div>
      <button
        className="popout-speak"
        disabled={!text.trim()}
        onClick={() => say(text)}
        type="button"
      >
        Speak
      </button>
    </main>
  );
}