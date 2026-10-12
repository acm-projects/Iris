// Builds the translation engine (backend/engine/engine.py) into a standalone
// program with PyInstaller, so people who install Iris don't need Python.
// Output: backend/build/engine-dist/iris-engine/ (copied into the app by
// electron-builder, see "extraResources" in package.json).
//
// Run with `npm run build:engine` (the package:* scripts run it for you).
// It must run on each OS you package for: PyInstaller can't cross-compile.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontend = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const backend = path.resolve(frontend, "..", "backend");
const build = path.join(backend, "build");

// Use the project's virtual environment (it has the engine's packages).
const python = [
  path.join(backend, ".venv", "bin", "python3"),
  path.join(backend, ".venv", "Scripts", "python.exe"),
].find((candidate) => existsSync(candidate));
if (!python) {
  console.error(
    "No Python environment found in backend/.venv. Create it first:\n" +
      (process.platform === "win32"
        ? "  cd backend; py -m venv .venv; .venv\\Scripts\\pip install -r requirements.txt"
        : "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"),
  );
  process.exit(1);
}

/** Runs a command with live output; stops the build if it fails. */
function run(args) {
  const result = spawnSync(python, args, { stdio: "inherit", cwd: backend });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
/** True if the virtual environment can import this Python module. */
const hasModule = (name) =>
  spawnSync(python, ["-c", `import ${name}`], { stdio: "ignore" }).status === 0;

// The engine's packages plus PyInstaller itself.
run([
  "-m",
  "pip",
  "install",
  "--quiet",
  "-r",
  "requirements.txt",
  "-r",
  "requirements-build.txt",
]);

const args = [
  "-m",
  "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onedir", // starts much faster than --onefile (nothing to unpack)
  "--name",
  "iris-engine",
  "--distpath",
  path.join(build, "engine-dist"),
  "--workpath",
  path.join(build, "pyinstaller"),
  "--specpath",
  build,
  // Packages imported inside functions, plus their native libraries/drivers.
  "--hidden-import",
  "cv2",
  "--hidden-import",
  "numpy",
  "--collect-all",
  "pyvirtualcam",
  "--hidden-import",
  "sounddevice",
];
// macOS/Windows wheels of sounddevice ship PortAudio in this package.
if (hasModule("_sounddevice_data"))
  args.push("--collect-all", "_sounddevice_data");
args.push(path.join(backend, "engine", "engine.py"));
run(args);

console.log(
  `\nEngine built: ${path.join(build, "engine-dist", "iris-engine")}`,
);
