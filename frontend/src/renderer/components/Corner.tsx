// Every Iris message (toasts, "Delete meeting?", Connect Google) is shown in
// one stack in the bottom-right corner of the window, so messages that appear
// at the same time line up instead of covering each other.
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

/** The shared bottom-right stack, created the first time it's needed. */
function stack() {
  let el = document.getElementById("iris-messages");
  if (!el) {
    el = document.createElement("div");
    el.id = "iris-messages";
    document.body.appendChild(el);
  }
  return el;
}

/** Renders its children in the bottom-right message stack. */
export default function Corner({ children }: { children: ReactNode }) {
  return createPortal(children, stack());
}
