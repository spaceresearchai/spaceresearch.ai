#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const electronBin = require("electron");
const appDir = path.resolve(__dirname, "..");

function needsLinuxNoSandboxFallback() {
  if (process.platform !== "linux") return false;
  if (process.argv.includes("--no-sandbox")) return false;

  // Check whether chrome-sandbox is owned by root with SUID (4755)
  const sandboxPath = path.join(path.dirname(electronBin), "chrome-sandbox");
  try {
    const st = fs.statSync(sandboxPath);
    const isRootSuid = st.uid === 0 && (st.mode & 0o4000) !== 0;
    if (isRootSuid) return false;
  } catch {
    return false;
  }

  // On Ubuntu 24.04+ / Debian kernels with AppArmor userns restriction,
  // Chromium aborts with SIGTRAP unless chrome-sandbox is root:4755 or --no-sandbox is passed.
  try {
    const restricted = fs
      .readFileSync("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", "utf8")
      .trim();
    if (restricted === "1") return true;
  } catch {
    // Fallback check: unprivileged userns clone disabled
    try {
      const clone = fs
        .readFileSync("/proc/sys/kernel/unprivileged_userns_clone", "utf8")
        .trim();
      if (clone === "0") return true;
    } catch {}
  }

  return false;
}

const args = [appDir, ...process.argv.slice(2)];
if (needsLinuxNoSandboxFallback()) {
  args.unshift("--no-sandbox");
}

const child = spawn(electronBin, args, {
  stdio: "inherit",
  env: process.env,
});

child.on("close", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
