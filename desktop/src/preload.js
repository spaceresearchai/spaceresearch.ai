"use strict";

const { contextBridge, ipcRenderer } = require("electron");

const ALLOWED_EVENTS = new Set([
  "app:action",
  "window:maximized-changed",
  "settings:updated",
  "updater:ready",
]);

contextBridge.exposeInMainWorld("spaceDesktop", {
  platform: process.platform,
  arch: process.arch,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
  },

  // Persistent storage (filesystem-backed in userData)
  loadSessions: () => ipcRenderer.invoke("store:loadSessions"),
  saveSessions: (sessions) => ipcRenderer.invoke("store:saveSessions", sessions),
  loadSettings: () => ipcRenderer.invoke("store:loadSettings"),
  saveSettings: (partial) => ipcRenderer.invoke("store:saveSettings", partial),
  detectCliConfig: () => ipcRenderer.invoke("store:detectCliConfig"),
  checkLocalProxy: (url) => ipcRenderer.invoke("proxy:checkLocal", url),

  // Native file exports & OS actions
  saveMarkdownFile: (defaultFilename, content) =>
    ipcRenderer.invoke("dialog:saveMarkdown", { defaultFilename, content }),
  savePngDataUrl: (defaultFilename, dataUrl) =>
    ipcRenderer.invoke("dialog:savePngDataUrl", { defaultFilename, dataUrl }),
  revealFile: (filePath) => ipcRenderer.invoke("shell:revealFile", filePath),
  openExternal: (url) => ipcRenderer.invoke("shell:openExternal", url),
  copyText: (text) => ipcRenderer.invoke("clipboard:writeText", text),

  // Window & Tray controls
  minimizeWindow: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximizeWindow: () => ipcRenderer.invoke("window:toggleMaximize"),
  closeWindow: () => ipcRenderer.invoke("window:close"),
  isWindowMaximized: () => ipcRenderer.invoke("window:isMaximized"),
  setAlwaysOnTop: (flag) => ipcRenderer.invoke("window:setAlwaysOnTop", Boolean(flag)),
  updateTrayFleetStatus: (label) => ipcRenderer.send("tray:updateFleetStatus", String(label || "")),

  // Live UI Auto-Updater (in-place updates without re-downloading installer)
  getAppInfo: () => ipcRenderer.invoke("updater:info"),
  checkForUpdates: () => ipcRenderer.invoke("updater:check"),
  reloadUi: () => ipcRenderer.invoke("updater:reload"),
  resetLiveUi: () => ipcRenderer.invoke("updater:reset"),

  // Main -> Renderer event subscriptions
  on: (channel, callback) => {
    if (!ALLOWED_EVENTS.has(channel) || typeof callback !== "function") {
      return () => {};
    }
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
  },
});
