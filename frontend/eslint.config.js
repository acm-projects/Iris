import tseslint from "typescript-eslint";

// Generated files and third-party packages are not source code to lint.
export default tseslint.config(
  { ignores: ["dist/", "dist-electron/", "node_modules/"] },
  tseslint.configs.recommended,
);
