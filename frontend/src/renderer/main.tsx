import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Import global styles once, before the application is rendered.
import "./styles.css";
// Dark-mode colours (applied when <html data-theme="dark">).
import "./theme.css";
import { initTheme } from "./utils/theme";

// Apply the saved Light/Dark/System theme before the first paint (no flash).
initTheme();

// Mount React in the root element defined by index.html.
createRoot(document.getElementById("root")!).render(
  // StrictMode enables additional development-time warnings.
  <StrictMode>
    <App />
  </StrictMode>,
);
