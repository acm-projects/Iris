import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const builderCommand = path.join(projectRoot, "node_modules", ".bin", process.platform === "win32" ? "electron-builder.cmd" : "electron-builder");

const platform = process.platform === "darwin" ? "--mac" : process.platform === "win32" ? "--win" : "--linux";
const macOutputDirectory = process.arch === "arm64" ? "mac-arm64" : "mac";
const executable = process.platform === "darwin"
  ? path.join(projectRoot, "release", macOutputDirectory, "Iris.app", "Contents", "MacOS", "Iris")
  : process.platform === "win32"
    ? path.join(projectRoot, "release", "win-unpacked", "Iris.exe")
    : path.join(projectRoot, "release", "linux-unpacked", "iris");

function run(command, args) {
  const result = spawnSync(command, args, { cwd: projectRoot, env: process.env, stdio: "inherit" });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

run(npmCommand, ["run", "build"]);
run(builderCommand, [platform, "--dir", "--publish", "never"]);

const iris = spawn(executable, ["."], { cwd: projectRoot, env: process.env, stdio: "inherit" });
iris.on("exit", (code) => process.exit(code ?? 0));
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => iris.kill(signal));
}
