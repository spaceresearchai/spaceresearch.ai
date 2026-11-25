"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DEFAULT_API_BASE = "https://api.spaceresearch.ai";
const FALLBACK_API_BASE = "https://api.longinusresearch.com";
const DEFAULT_LOCAL_PROXY_URL = "http://127.0.0.1:8377";

const PLACEHOLDER_KEYS = new Set([
  "",
  "space-local-proxy",
  "space-barter-auto",
  "space-auto",
  "sk-space-local",
]);

const DEFAULT_SETTINGS = Object.freeze({
  endpointMode: "auto", // "auto" | "local-proxy" | "custom"
  apiBase: DEFAULT_API_BASE,
  fallbackApiBase: FALLBACK_API_BASE,
  localProxyUrl: DEFAULT_LOCAL_PROXY_URL,
  customApiBase: "",
  apiKey: "",
  sessionToken: "",
  solanaWallet: "",
  route: "",
  privacy: "",
  region: "",
  systemPrompt: "",
  maxTokens: 4096,
  xrayEnabled: true,
  starfieldEnabled: true,
  theme: "dark", // "dark" | "light"
  alwaysOnTop: false,
  globalShortcut: "CommandOrControl+Shift+Space",
  globalShortcutEnabled: true,
  minimizeToTray: true,
  launchAtLogin: false,
  sidebarCollapsed: false,
  lastModel: "qwen3.8-27b",
  lastNode: "auto",
});

const DEFAULT_WINDOW_STATE = Object.freeze({
  width: 1240,
  height: 840,
  x: undefined,
  y: undefined,
  isMaximized: false,
});

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJsonSafe(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const raw = fs.readFileSync(filePath, "utf8");
    if (!raw.trim()) return fallback;
    const parsed = JSON.parse(raw);
    return parsed === null || parsed === undefined ? fallback : parsed;
  } catch {
    return fallback;
  }
}

function writeJsonAtomic(filePath, data, mode = 0o600) {
  ensureDir(path.dirname(filePath));
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2) + "\n", {
    encoding: "utf8",
    mode,
  });
  fs.renameSync(tmpPath, filePath);
}

function normalizeBaseUrl(url, fallback = DEFAULT_API_BASE) {
  const s = String(url || "").trim();
  if (!/^https?:\/\/[^/\s]+/i.test(s)) return fallback;
  return s.replace(/\/+$/, "");
}

function isLoopbackUrl(rawUrl) {
  try {
    const parsed = new URL(String(rawUrl || "").trim());
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
    if (parsed.username || parsed.password) return false;
    const host = (parsed.hostname || "").toLowerCase();
    return host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]" || host.startsWith("127.");
  } catch {
    return false;
  }
}

function sanitizeSettings(partial = {}, base = DEFAULT_SETTINGS) {
  const merged = { ...DEFAULT_SETTINGS, ...base, ...(partial && typeof partial === "object" ? partial : {}) };
  const validModes = new Set(["auto", "local-proxy", "custom"]);
  const validRoutes = new Set(["", "fastest", "cheapest", "verified-local-only"]);
  const validPrivacy = new Set(["", "standard", "zero-log", "hardened-docker", "e2ee", "tee", "local-only"]);

  merged.endpointMode = validModes.has(merged.endpointMode) ? merged.endpointMode : "auto";
  merged.apiBase = normalizeBaseUrl(merged.apiBase, DEFAULT_API_BASE);
  merged.fallbackApiBase = normalizeBaseUrl(merged.fallbackApiBase, FALLBACK_API_BASE);
  merged.localProxyUrl = normalizeBaseUrl(merged.localProxyUrl, DEFAULT_LOCAL_PROXY_URL);
  merged.customApiBase = merged.customApiBase ? normalizeBaseUrl(merged.customApiBase, "") : "";
  merged.apiKey = String(merged.apiKey || "").trim();
  merged.sessionToken = String(merged.sessionToken || "").trim();
  merged.solanaWallet = String(merged.solanaWallet || "").trim().slice(0, 64);
  merged.route = validRoutes.has(merged.route) ? merged.route : "";
  merged.privacy = validPrivacy.has(merged.privacy) ? merged.privacy : "";
  merged.region = String(merged.region || "").trim().slice(0, 16);
  merged.systemPrompt = String(merged.systemPrompt || "").slice(0, 8000);
  merged.maxTokens = Math.max(256, Math.min(16384, Number(merged.maxTokens) || 4096));
  merged.xrayEnabled = Boolean(merged.xrayEnabled);
  merged.starfieldEnabled = Boolean(merged.starfieldEnabled);
  merged.theme = merged.theme === "light" ? "light" : "dark";
  merged.alwaysOnTop = Boolean(merged.alwaysOnTop);
  merged.globalShortcut = String(merged.globalShortcut || "CommandOrControl+Shift+Space").trim();
  merged.globalShortcutEnabled = Boolean(merged.globalShortcutEnabled);
  merged.minimizeToTray = Boolean(merged.minimizeToTray);
  merged.launchAtLogin = Boolean(merged.launchAtLogin);
  merged.sidebarCollapsed = Boolean(merged.sidebarCollapsed);
  merged.lastModel = String(merged.lastModel || "qwen3.8-27b").trim() || "qwen3.8-27b";
  merged.lastNode = String(merged.lastNode || "auto").trim() || "auto";

  return merged;
}

function resolveEffectiveEndpoint(settings) {
  const s = sanitizeSettings(settings);
  let baseUrl = s.apiBase;
  let fallbackUrl = s.endpointMode === "auto" ? s.fallbackApiBase : null;
  let effectiveMode = s.endpointMode;

  if (s.endpointMode === "local-proxy") {
    baseUrl = s.localProxyUrl;
  } else if (s.endpointMode === "custom" && s.customApiBase) {
    baseUrl = s.customApiBase;
  }

  // Strict air-gap: local-only mode NEVER routes to a remote coordinator or fallback URL
  if (s.privacy === "local-only") {
    if (s.endpointMode === "custom" && isLoopbackUrl(s.customApiBase)) {
      baseUrl = s.customApiBase;
    } else {
      baseUrl = isLoopbackUrl(s.localProxyUrl) ? s.localProxyUrl : DEFAULT_LOCAL_PROXY_URL;
      effectiveMode = "local-proxy";
    }
    fallbackUrl = null;
  } else if (s.privacy === "e2ee") {
    // Client-side E2EE encryption happens inside the local Space proxy (127.0.0.1:8377)
    if (s.endpointMode === "auto") {
      baseUrl = isLoopbackUrl(s.localProxyUrl) ? s.localProxyUrl : DEFAULT_LOCAL_PROXY_URL;
      effectiveMode = "local-proxy";
    }
    fallbackUrl = null;
  }

  const headers = { "Content-Type": "application/json" };
  if (s.apiKey && !PLACEHOLDER_KEYS.has(s.apiKey)) {
    headers["Authorization"] = `Bearer ${s.apiKey}`;
    headers["X-API-Key"] = s.apiKey;
  }
  if (s.sessionToken) {
    headers["X-Space-Session"] = s.sessionToken;
  }
  if (s.solanaWallet) {
    headers["X-Space-Wallet"] = s.solanaWallet;
  }
  if (s.route) {
    headers["X-Space-Route"] = s.route;
  }
  if (s.privacy) {
    headers["X-Space-Privacy"] = s.privacy;
  }
  if (s.region) {
    headers["X-Space-Region"] = s.region;
  }
  headers["X-Space-Client"] = "space-desktop";

  return {
    baseUrl,
    fallbackUrl,
    headers,
    mode: effectiveMode,
  };
}

function detectSpaceCliConfig(homeDir = os.homedir()) {
  const spaceCfgPath = path.join(homeDir, ".space_ai", "config.json");
  const spaceCfg = readJsonSafe(spaceCfgPath, {});

  let nodeCfg = {};
  let nodeCfgPath = null;
  for (const name of [".longinus_node.json", ".longinus_mac_node.json", ".longinus_gpu_node.json"]) {
    const candidate = path.join(homeDir, name);
    if (fs.existsSync(candidate)) {
      nodeCfg = readJsonSafe(candidate, {});
      nodeCfgPath = candidate;
      break;
    }
  }

  let apiKey = "";
  let apiKeySource = "";
  for (const envKey of ["SPACE_BARTER_KEY", "SPACE_API_KEY", "LONGINUS_API_KEY"]) {
    const val = String(process.env[envKey] || "").trim();
    if (val && !PLACEHOLDER_KEYS.has(val)) {
      apiKey = val;
      apiKeySource = `env:${envKey}`;
      break;
    }
  }
  if (!apiKey) {
    for (const field of ["barter_key", "api_key", "openai_api_key"]) {
      const val = String(spaceCfg[field] || "").trim();
      if (val && !PLACEHOLDER_KEYS.has(val)) {
        apiKey = val;
        apiKeySource = "~/.space_ai/config.json";
        break;
      }
    }
  }
  if (!apiKey && nodeCfgPath) {
    for (const field of ["barter_key", "api_key", "longinus_api_key"]) {
      const val = String(nodeCfg[field] || "").trim();
      if (val && !PLACEHOLDER_KEYS.has(val)) {
        apiKey = val;
        apiKeySource = path.basename(nodeCfgPath);
        break;
      }
    }
  }

  const proxyHost = String(spaceCfg.proxy_host || "127.0.0.1").trim() || "127.0.0.1";
  const proxyPort = Number(spaceCfg.proxy_port || 8377) || 8377;
  const sessionToken = String(
    process.env.SPACE_SESSION_TOKEN || spaceCfg.session_token || spaceCfg.session || ""
  ).trim();
  const solanaWallet = String(
    process.env.SPACE_WALLET || spaceCfg.wallet || spaceCfg.payout_wallet || spaceCfg.solana_wallet || nodeCfg.payout_wallet || nodeCfg.wallet || ""
  ).trim();

  return {
    found: Boolean(apiKey || sessionToken || solanaWallet || Object.keys(spaceCfg).length > 0 || nodeCfgPath),
    apiKey,
    apiKeySource,
    sessionToken,
    solanaWallet,
    localProxyUrl: `http://${proxyHost}:${proxyPort}`,
    coordinator: Array.isArray(spaceCfg.coordinators) && spaceCfg.coordinators[0]
      ? normalizeBaseUrl(spaceCfg.coordinators[0], DEFAULT_API_BASE)
      : normalizeBaseUrl(spaceCfg.coordinator, DEFAULT_API_BASE),
    route: String(spaceCfg.route || "").trim(),
    privacy: String(spaceCfg.privacy || "").trim(),
    region: String(spaceCfg.region || "").trim(),
    nodeName: String(nodeCfg.display_name || nodeCfg.node_name || "").trim(),
    configPath: fs.existsSync(spaceCfgPath) ? spaceCfgPath : nodeCfgPath,
  };
}

class DesktopStore {
  constructor(userDataDir) {
    this.userDataDir = userDataDir;
    this.sessionsPath = path.join(userDataDir, "sessions.json");
    this.settingsPath = path.join(userDataDir, "settings.json");
    this.windowStatePath = path.join(userDataDir, "window-state.json");
  }

  loadSessions() {
    const data = readJsonSafe(this.sessionsPath, []);
    return Array.isArray(data) ? data : [];
  }

  saveSessions(sessions) {
    const safeList = Array.isArray(sessions) ? sessions.slice(0, 200) : [];
    writeJsonAtomic(this.sessionsPath, safeList, 0o600);
    return safeList.length;
  }

  loadSettings() {
    const raw = readJsonSafe(this.settingsPath, {});
    return sanitizeSettings(raw, DEFAULT_SETTINGS);
  }

  saveSettings(partial) {
    const current = this.loadSettings();
    const next = sanitizeSettings(partial, current);
    writeJsonAtomic(this.settingsPath, next, 0o600);
    return next;
  }

  loadWindowState() {
    const raw = readJsonSafe(this.windowStatePath, {});
    return {
      width: Math.max(760, Number(raw.width) || DEFAULT_WINDOW_STATE.width),
      height: Math.max(540, Number(raw.height) || DEFAULT_WINDOW_STATE.height),
      x: typeof raw.x === "number" ? raw.x : undefined,
      y: typeof raw.y === "number" ? raw.y : undefined,
      isMaximized: Boolean(raw.isMaximized),
    };
  }

  saveWindowState(state) {
    if (!state || typeof state !== "object") return;
    writeJsonAtomic(
      this.windowStatePath,
      {
        width: Math.max(760, Math.round(Number(state.width) || DEFAULT_WINDOW_STATE.width)),
        height: Math.max(540, Math.round(Number(state.height) || DEFAULT_WINDOW_STATE.height)),
        x: typeof state.x === "number" ? Math.round(state.x) : undefined,
        y: typeof state.y === "number" ? Math.round(state.y) : undefined,
        isMaximized: Boolean(state.isMaximized),
      },
      0o644
    );
  }
}

module.exports = {
  DEFAULT_API_BASE,
  FALLBACK_API_BASE,
  DEFAULT_LOCAL_PROXY_URL,
  DEFAULT_SETTINGS,
  DEFAULT_WINDOW_STATE,
  DesktopStore,
  sanitizeSettings,
  normalizeBaseUrl,
  resolveEffectiveEndpoint,
  detectSpaceCliConfig,
};
