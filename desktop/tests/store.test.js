"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  DEFAULT_API_BASE,
  FALLBACK_API_BASE,
  DEFAULT_LOCAL_PROXY_URL,
  DesktopStore,
  sanitizeSettings,
  resolveEffectiveEndpoint,
  detectSpaceCliConfig,
} = require("../src/store");

function mkTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "space-desktop-test-"));
}

test("DesktopStore persists sessions, settings, and window state atomically", () => {
  const dir = mkTempDir();
  try {
    const store = new DesktopStore(dir);
    assert.deepEqual(store.loadSessions(), []);

    const sampleSessions = [
      { id: "c_1", title: "Hello Space", messages: [{ role: "user", content: "Hi" }], updatedAt: 1000 },
    ];
    assert.equal(store.saveSessions(sampleSessions), 1);
    assert.deepEqual(store.loadSessions(), sampleSessions);

    const savedSettings = store.saveSettings({
      endpointMode: "local-proxy",
      route: "verified-local-only",
      privacy: "zero-log",
      alwaysOnTop: true,
      maxTokens: 8192,
    });
    assert.equal(savedSettings.endpointMode, "local-proxy");
    assert.equal(savedSettings.route, "verified-local-only");
    assert.equal(savedSettings.privacy, "zero-log");
    assert.equal(savedSettings.alwaysOnTop, true);
    assert.equal(savedSettings.maxTokens, 8192);
    assert.equal(store.loadSettings().alwaysOnTop, true);

    store.saveWindowState({ width: 1400, height: 900, x: 40, y: 50, isMaximized: true });
    assert.deepEqual(store.loadWindowState(), {
      width: 1400,
      height: 900,
      x: 40,
      y: 50,
      isMaximized: true,
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("sanitizeSettings clamps invalid values safely", () => {
  const s = sanitizeSettings({
    endpointMode: "invalid-mode",
    route: "nonsense",
    privacy: "unknown",
    maxTokens: 999999,
    apiBase: "javascript:alert(1)",
  });
  assert.equal(s.endpointMode, "auto");
  assert.equal(s.route, "");
  assert.equal(s.privacy, "");
  assert.equal(s.maxTokens, 16384);
  assert.equal(s.apiBase, DEFAULT_API_BASE);
});

test("resolveEffectiveEndpoint handles auto, local-proxy, and custom modes with headers", () => {
  const autoRes = resolveEffectiveEndpoint({
    endpointMode: "auto",
    apiKey: "sp_live_123",
    sessionToken: "sp_sess_abc",
    route: "fastest",
    privacy: "tee",
    region: "eu",
  });
  assert.equal(autoRes.baseUrl, DEFAULT_API_BASE);
  assert.equal(autoRes.fallbackUrl, FALLBACK_API_BASE);
  assert.equal(autoRes.headers["Authorization"], "Bearer sp_live_123");
  assert.equal(autoRes.headers["X-API-Key"], "sp_live_123");
  assert.equal(autoRes.headers["X-Space-Session"], "sp_sess_abc");
  assert.equal(autoRes.headers["X-Space-Route"], "fastest");
  assert.equal(autoRes.headers["X-Space-Privacy"], "tee");
  assert.equal(autoRes.headers["X-Space-Region"], "eu");

  const localRes = resolveEffectiveEndpoint({
    endpointMode: "local-proxy",
    localProxyUrl: "http://127.0.0.1:8377/",
    apiKey: "space-local-proxy", // placeholder should be ignored
  });
  assert.equal(localRes.baseUrl, DEFAULT_LOCAL_PROXY_URL);
  assert.equal(localRes.fallbackUrl, null);
  assert.equal(localRes.headers["Authorization"], undefined);

  const customRes = resolveEffectiveEndpoint({
    endpointMode: "custom",
    customApiBase: "https://my-mac.local:9000/",
  });
  assert.equal(customRes.baseUrl, "https://my-mac.local:9000");
});

test("detectSpaceCliConfig reads ~/.space_ai/config.json and ~/.longinus_node.json", () => {
  const fakeHome = mkTempDir();
  try {
    const spaceDir = path.join(fakeHome, ".space_ai");
    fs.mkdirSync(spaceDir, { recursive: true });
    fs.writeFileSync(
      path.join(spaceDir, "config.json"),
      JSON.stringify({
        barter_key: "sp_barter_test_key",
        proxy_host: "127.0.0.1",
        proxy_port: 8488,
        route: "cheapest",
        privacy: "zero-log",
        region: "us",
      })
    );
    fs.writeFileSync(
      path.join(fakeHome, ".longinus_node.json"),
      JSON.stringify({ display_name: "mac-studio-ultra" })
    );

    const detected = detectSpaceCliConfig(fakeHome);
    assert.equal(detected.found, true);
    assert.equal(detected.apiKey, "sp_barter_test_key");
    assert.equal(detected.apiKeySource, "~/.space_ai/config.json");
    assert.equal(detected.localProxyUrl, "http://127.0.0.1:8488");
    assert.equal(detected.route, "cheapest");
    assert.equal(detected.privacy, "zero-log");
    assert.equal(detected.region, "us");
    assert.equal(detected.nodeName, "mac-studio-ultra");
  } finally {
    fs.rmSync(fakeHome, { recursive: true, force: true });
  }
});
