"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const https = require("node:https");
const path = require("node:path");

const UI_FILES = Object.freeze(["index.html", "styles.css", "app.js", "mark.svg"]);
const DEFAULT_RAW_BASE = "https://raw.githubusercontent.com/spaceresearchai/spaceresearch.ai/main/desktop";
const DEFAULT_RELEASES_API = "https://api.github.com/repos/spaceresearchai/spaceresearch.ai/releases/latest";

function sha256Buffer(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function sha256File(filePath) {
  return sha256Buffer(fs.readFileSync(filePath));
}

function readJsonFile(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    const raw = fs.readFileSync(filePath, "utf8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function compareSemver(a, b) {
  const pa = String(a || "0.0.0").replace(/^[^\d]*/, "").split(".").map((n) => parseInt(n, 10) || 0);
  const pb = String(b || "0.0.0").replace(/^[^\d]*/, "").split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) {
    const diff = (pa[i] || 0) - (pb[i] || 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

function isValidManifest(manifest) {
  if (!manifest || typeof manifest !== "object") return false;
  if (typeof manifest.uiVersion !== "number" || !Number.isFinite(manifest.uiVersion) || manifest.uiVersion < 1) {
    return false;
  }
  if (!manifest.files || typeof manifest.files !== "object") return false;
  for (const fname of UI_FILES) {
    const h = manifest.files[fname];
    if (typeof h !== "string" || !/^[0-9a-f]{64}$/i.test(h)) return false;
  }
  return true;
}

function verifyDirAgainstManifest(dirPath, manifest) {
  if (!isValidManifest(manifest)) return false;
  for (const fname of UI_FILES) {
    const fp = path.join(dirPath, fname);
    if (!fs.existsSync(fp)) return false;
    try {
      if (sha256File(fp).toLowerCase() !== manifest.files[fname].toLowerCase()) {
        return false;
      }
    } catch {
      return false;
    }
  }
  return true;
}

function getLiveUiDir(userDataDir) {
  return path.join(userDataDir, "live-ui");
}

function getBundledManifest(appRootDir) {
  return readJsonFile(path.join(appRootDir, "ui-manifest.json"));
}

function resolveRendererEntry({ userDataDir, appRootDir, appVersion = "1.2.0" }) {
  const bundledDir = path.join(appRootDir, "src", "renderer");
  const bundledIndex = path.join(bundledDir, "index.html");
  const bundledManifest = getBundledManifest(appRootDir);
  const bundledUiVersion = isValidManifest(bundledManifest) ? bundledManifest.uiVersion : 1;

  const liveDir = getLiveUiDir(userDataDir);
  const liveManifest = readJsonFile(path.join(liveDir, "ui-manifest.json"));
  if (
    isValidManifest(liveManifest) &&
    liveManifest.uiVersion > bundledUiVersion &&
    (!liveManifest.minAppVersion || compareSemver(appVersion, liveManifest.minAppVersion) >= 0) &&
    verifyDirAgainstManifest(liveDir, liveManifest)
  ) {
    return {
      entryFile: path.join(liveDir, "index.html"),
      source: "live-ui",
      uiVersion: liveManifest.uiVersion,
      bundledUiVersion,
    };
  }

  return {
    entryFile: bundledIndex,
    source: "bundled",
    uiVersion: bundledUiVersion,
    bundledUiVersion,
  };
}

function resetLiveUi(userDataDir) {
  const liveDir = getLiveUiDir(userDataDir);
  if (fs.existsSync(liveDir)) {
    fs.rmSync(liveDir, { recursive: true, force: true });
    return true;
  }
  return false;
}

function fetchBufferHttps(url, timeoutMs = 10000, redirectsLeft = 3) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      {
        timeout: timeoutMs,
        headers: {
          "User-Agent": "space-ai-desktop-updater/1.2",
          "Cache-Control": "no-cache",
        },
      },
      (res) => {
        if (
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location &&
          redirectsLeft > 0
        ) {
          res.resume();
          const nextUrl = new URL(res.headers.location, url).toString();
          return fetchBufferHttps(nextUrl, timeoutMs, redirectsLeft - 1).then(resolve, reject);
        }
        if (res.statusCode < 200 || res.statusCode >= 300) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode}`));
        }
        const chunks = [];
        let total = 0;
        res.on("data", (chunk) => {
          total += chunk.length;
          if (total > 10 * 1024 * 1024) {
            req.destroy();
            return reject(new Error("Payload too large"));
          }
          chunks.push(chunk);
        });
        res.on("end", () => resolve(Buffer.concat(chunks)));
      }
    );
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });
    req.on("error", reject);
  });
}

async function checkAndInstallUiUpdate({
  userDataDir,
  appRootDir,
  appVersion = "1.2.0",
  rawBase = DEFAULT_RAW_BASE,
  fetchFn = fetchBufferHttps,
}) {
  const current = resolveRendererEntry({ userDataDir, appRootDir, appVersion });
  const manifestUrl = `${rawBase.replace(/\/+$/, "")}/ui-manifest.json?t=${Date.now()}`;
  const rawManifestBuf = await fetchFn(manifestUrl);
  const remoteManifest = JSON.parse(rawManifestBuf.toString("utf8"));

  if (!isValidManifest(remoteManifest)) {
    throw new Error("Invalid remote UI manifest");
  }

  if (remoteManifest.minAppVersion && compareSemver(appVersion, remoteManifest.minAppVersion) < 0) {
    return {
      status: "app-update-required",
      currentUiVersion: current.uiVersion,
      remoteUiVersion: remoteManifest.uiVersion,
      minAppVersion: remoteManifest.minAppVersion,
      releaseUrl: "https://github.com/spaceresearchai/spaceresearch.ai/releases/latest",
    };
  }

  if (remoteManifest.uiVersion <= current.uiVersion) {
    return {
      status: "up-to-date",
      currentUiVersion: current.uiVersion,
      remoteUiVersion: remoteManifest.uiVersion,
      source: current.source,
    };
  }

  // Download all renderer files into staging dir and verify SHA-256 of every file
  const stageDir = path.join(userDataDir, `.live-ui-stage-${Date.now()}`);
  fs.mkdirSync(stageDir, { recursive: true });
  try {
    for (const fname of UI_FILES) {
      const fileUrl = `${rawBase.replace(/\/+$/, "")}/src/renderer/${fname}?v=${remoteManifest.uiVersion}`;
      const buf = await fetchFn(fileUrl);
      const actualHash = sha256Buffer(buf).toLowerCase();
      const expectedHash = remoteManifest.files[fname].toLowerCase();
      if (actualHash !== expectedHash) {
        throw new Error(`SHA-256 mismatch for ${fname}`);
      }
      fs.writeFileSync(path.join(stageDir, fname), buf);
    }
    fs.writeFileSync(path.join(stageDir, "ui-manifest.json"), JSON.stringify(remoteManifest, null, 2) + "\n", "utf8");

    const liveDir = getLiveUiDir(userDataDir);
    const backupDir = path.join(userDataDir, `.live-ui-bak-${Date.now()}`);
    if (fs.existsSync(liveDir)) {
      fs.renameSync(liveDir, backupDir);
    }
    fs.renameSync(stageDir, liveDir);
    if (fs.existsSync(backupDir)) {
      fs.rmSync(backupDir, { recursive: true, force: true });
    }

    return {
      status: "updated",
      previousUiVersion: current.uiVersion,
      uiVersion: remoteManifest.uiVersion,
      notes: String(remoteManifest.notes || ""),
    };
  } catch (err) {
    if (fs.existsSync(stageDir)) {
      fs.rmSync(stageDir, { recursive: true, force: true });
    }
    throw err;
  }
}

async function checkLatestAppRelease(appVersion, fetchFn = fetchBufferHttps) {
  try {
    const buf = await fetchFn(DEFAULT_RELEASES_API, 6000);
    const data = JSON.parse(buf.toString("utf8"));
    const tag = String(data.tag_name || "");
    const latestVer = tag.replace(/^desktop-v/i, "").replace(/^v/i, "");
    if (latestVer && compareSemver(latestVer, appVersion) > 0) {
      return {
        hasNewAppRelease: true,
        latestVersion: latestVer,
        tag,
        releaseUrl: data.html_url || "https://github.com/spaceresearchai/spaceresearch.ai/releases/latest",
      };
    }
  } catch {}
  return { hasNewAppRelease: false, latestVersion: appVersion };
}

module.exports = {
  UI_FILES,
  DEFAULT_RAW_BASE,
  sha256Buffer,
  compareSemver,
  isValidManifest,
  verifyDirAgainstManifest,
  resolveRendererEntry,
  resetLiveUi,
  checkAndInstallUiUpdate,
  checkLatestAppRelease,
};
