import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Import global styles once, before the application is rendered.
import "./styles.css";

// Mount React in the root element defined by index.html.
createRoot(document.getElementById("root")!).render(
  // StrictMode enables additional development-time warnings.
  <StrictMode>
    <App />
  </StrictMode>,
);
