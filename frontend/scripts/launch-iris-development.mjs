// Development launcher used by `npm run dev`. Restart it after editing
// src/electron/*.
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Paths and commands for this machine's platform.
const projectRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const builderCommand = path.join(
  projectRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "electron-builder.cmd" : "electron-builder",
);

// Where electron-builder writes the unpacked app for each platform.
const platform =
  process.platform === "darwin"
    ? "--mac"
    : process.platform === "win32"
      ? "--win"
      : "--linux";
const macOutputDirectory = process.arch === "arm64" ? "mac-arm64" : "mac";
const executable =
  process.platform === "darwin"
    ? path.join(
        projectRoot,
        "release",
        macOutputDirectory,
        "Iris.app",
        "Contents",
        "MacOS",
        "Iris",
      )
    : process.platform === "win32"
      ? path.join(
          projectRoot,
          "node_modules",
          "electron",
          "dist",
          "electron.exe",
        )
      : path.join(projectRoot, "release", "linux-unpacked", "iris");

/** Runs a command synchronously and stops this script if it fails. */
function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    env: process.env,
    // Windows cannot execute .cmd shims (npm.cmd, electron-builder.cmd)
    // directly with spawnSync; route them through cmd.exe.
    shell: process.platform === "win32",
    stdio: "inherit",
  });
  if (result.error) {
    console.error(`Could not run ${command}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

// Compile Electron + React, then launch it. macOS/Linux keep using an unpacked
// application for native identity and permissions. Windows launches Electron
// directly: repeatedly replacing win-unpacked is slow and commonly fails with
// EPERM while antivirus/indexing software scans the directory.
run(npmCommand, ["run", "build"]);
if (process.platform !== "win32") {
  run(builderCommand, [platform, "--dir", "--publish", "never"]);
}

const iris = spawn(executable, ["."], {
  cwd: projectRoot,
  env: process.env,
  stdio: "inherit",
});
iris.on("exit", (code) => process.exit(code ?? 0));
// Closing this script (Ctrl+C) also closes the Iris window.
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => iris.kill(signal));
}
