#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const RENDERER_DIR = path.join(ROOT, "src", "renderer");
const MANIFEST_PATH = path.join(ROOT, "ui-manifest.json");
const PKG_PATH = path.join(ROOT, "package.json");
const UI_FILES = ["index.html", "styles.css", "app.js", "mark.svg"];

const pkg = JSON.parse(fs.readFileSync(PKG_PATH, "utf8"));
const existing = fs.existsSync(MANIFEST_PATH)
  ? JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"))
  : {};

const bump = process.argv.includes("--bump");
const nextVersion = bump
  ? (Number(existing.uiVersion) || 0) + 1
  : Math.max(1, Number(existing.uiVersion) || 1);

const files = {};
for (const fname of UI_FILES) {
  const buf = fs.readFileSync(path.join(RENDERER_DIR, fname));
  files[fname] = crypto.createHash("sha256").update(buf).digest("hex");
}

const manifest = {
  uiVersion: nextVersion,
  minAppVersion: "1.2.0",
  appVersion: pkg.version,
  updatedAt: new Date().toISOString(),
  notes: process.argv.includes("--notes")
    ? process.argv[process.argv.indexOf("--notes") + 1] || ""
    : existing.notes || "Aligned chat UI, confidence strip, prismatic logo & Daylight Mode with spaceresearch.ai",
  files,
};

fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + "\n", "utf8");
console.log(`Wrote ui-manifest.json (uiVersion=${manifest.uiVersion})`);
