// Development launcher used by `npm run dev`: builds a real packaged copy of
// Iris and opens it. Restart `npm run dev` after editing src/electron/*.
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
      ? path.join(projectRoot, "release", "win-unpacked", "Iris.exe")
      : path.join(projectRoot, "release", "linux-unpacked", "iris");

/** Runs a command synchronously and stops this script if it fails. */
function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    env: process.env,
    stdio: "inherit",
    // Windows: npm.cmd / electron-builder.cmd are batch files, which Node
    // (since 18.20.2 / 20.12.2) refuses to run without a shell (EINVAL).
    // Safe here: the commands and arguments are fixed, not user input.
    shell: process.platform === "win32",
  });
  if (result.error) {
    console.error(`Could not run ${command}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

// 1) Compile Electron + React, 2) package an unpacked Iris.app (so macOS uses
// the real app name, icon and iris:// link handling), 3) launch it. The React
// UI still loads live from Vite because VITE_DEV_SERVER_URL is set.
run(npmCommand, ["run", "build"]);
// Quoted on Windows (it runs through the shell there) so a project folder
// with spaces in its path still works.
run(process.platform === "win32" ? `"${builderCommand}"` : builderCommand, [
  platform,
  "--dir",
  "--publish",
  "never",
]);

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
