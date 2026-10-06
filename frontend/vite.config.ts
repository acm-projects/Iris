import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite builds and serves the React renderer process.
export default defineConfig({
  plugins: [react()],
  // Images and icons live in assets/ and are served from the site root.
  publicDir: "assets",
});
