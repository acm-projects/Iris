import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import PopoutPage from "./pages/tsx/PopoutPage";

// Import global styles once, before the application is rendered.
import "./styles.css";
// Dark-mode colours (applied when <html data-theme="dark">).
import "./theme.css";
import { initTheme } from "./utils/theme";

// Apply the saved Light/Dark/System theme before the first paint (no flash).
initTheme();

// The floating Type to speak window loads this same bundle with #/popout.
const isPopout = window.location.hash.startsWith("#/popout");

// Mount React in the root element defined by index.html.
createRoot(document.getElementById("root")!).render(
  // StrictMode enables additional development-time warnings.
  <StrictMode>
    {/* Shows a friendly recovery screen if a screen ever crashes */}
    <ErrorBoundary>
      {isPopout ? <PopoutPage /> : <App />}
    </ErrorBoundary>
  </StrictMode>,
);
