"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  UI_FILES,
  sha256Buffer,
  compareSemver,
  resolveRendererEntry,
  checkAndInstallUiUpdate,
  resetLiveUi,
} = require("../src/updater");

function mkTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "space-desktop-updater-test-"));
}

function makeFakeBundledApp(rootDir, uiVersion = 1) {
  const rdir = path.join(rootDir, "src", "renderer");
  fs.mkdirSync(rdir, { recursive: true });
  const files = {};
  for (const f of UI_FILES) {
    const content = Buffer.from(`bundled-${f}-v${uiVersion}`);
    fs.writeFileSync(path.join(rdir, f), content);
    files[f] = sha256Buffer(content);
  }
  fs.writeFileSync(
    path.join(rootDir, "ui-manifest.json"),
    JSON.stringify({ uiVersion, minAppVersion: "1.2.0", files })
  );
}

test("compareSemver orders semantic versions properly", () => {
  assert.equal(compareSemver("1.2.0", "1.2.0"), 0);
  assert.ok(compareSemver("1.2.1", "1.2.0") > 0);
  assert.ok(compareSemver("1.10.0", "1.2.9") > 0);
  assert.ok(compareSemver("1.1.0", "1.2.0") < 0);
});

test("checkAndInstallUiUpdate downloads, verifies SHA-256, and activates newer live UI", async () => {
  const userData = mkTempDir();
  const appRoot = mkTempDir();
  try {
    makeFakeBundledApp(appRoot, 1);

    const before = resolveRendererEntry({ userDataDir: userData, appRootDir: appRoot, appVersion: "1.2.0" });
    assert.equal(before.source, "bundled");
    assert.equal(before.uiVersion, 1);

    const remotePayloads = {};
    const remoteHashes = {};
    for (const f of UI_FILES) {
      const buf = Buffer.from(`remote-${f}-v2`);
      remotePayloads[f] = buf;
      remoteHashes[f] = sha256Buffer(buf);
    }
    const remoteManifest = {
      uiVersion: 2,
      minAppVersion: "1.2.0",
      notes: "Test live UI update",
      files: remoteHashes,
    };

    const fakeFetch = async (url) => {
      if (url.includes("ui-manifest.json")) {
        return Buffer.from(JSON.stringify(remoteManifest));
      }
      for (const f of UI_FILES) {
        if (url.includes(`/src/renderer/${f}`)) return remotePayloads[f];
      }
      throw new Error("404 " + url);
    };

    const res = await checkAndInstallUiUpdate({
      userDataDir: userData,
      appRootDir: appRoot,
      appVersion: "1.2.0",
      fetchFn: fakeFetch,
    });
    assert.equal(res.status, "updated");
    assert.equal(res.uiVersion, 2);

    const after = resolveRendererEntry({ userDataDir: userData, appRootDir: appRoot, appVersion: "1.2.0" });
    assert.equal(after.source, "live-ui");
    assert.equal(after.uiVersion, 2);
    assert.equal(fs.readFileSync(after.entryFile, "utf8"), "remote-index.html-v2");

    // Reset reverts cleanly to bundled UI
    assert.equal(resetLiveUi(userData), true);
    const reverted = resolveRendererEntry({ userDataDir: userData, appRootDir: appRoot, appVersion: "1.2.0" });
    assert.equal(reverted.source, "bundled");
  } finally {
    fs.rmSync(userData, { recursive: true, force: true });
    fs.rmSync(appRoot, { recursive: true, force: true });
  }
});

test("checkAndInstallUiUpdate rejects tampered file with SHA-256 mismatch", async () => {
  const userData = mkTempDir();
  const appRoot = mkTempDir();
  try {
    makeFakeBundledApp(appRoot, 1);

    const remoteHashes = {};
    for (const f of UI_FILES) {
      remoteHashes[f] = sha256Buffer(Buffer.from(`expected-${f}`));
    }
    const remoteManifest = { uiVersion: 3, minAppVersion: "1.2.0", files: remoteHashes };

    const tamperedFetch = async (url) => {
      if (url.includes("ui-manifest.json")) return Buffer.from(JSON.stringify(remoteManifest));
      return Buffer.from("TAMPERED_CONTENT");
    };

    await assert.rejects(
      () =>
        checkAndInstallUiUpdate({
          userDataDir: userData,
          appRootDir: appRoot,
          appVersion: "1.2.0",
          fetchFn: tamperedFetch,
        }),
      /SHA-256 mismatch/
    );

    const resolved = resolveRendererEntry({ userDataDir: userData, appRootDir: appRoot, appVersion: "1.2.0" });
    assert.equal(resolved.source, "bundled");
  } finally {
    fs.rmSync(userData, { recursive: true, force: true });
    fs.rmSync(appRoot, { recursive: true, force: true });
  }
});
