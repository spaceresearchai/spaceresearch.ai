"use strict";

const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const {
  app,
  BrowserWindow,
  Menu,
  Tray,
  clipboard,
  dialog,
  globalShortcut,
  ipcMain,
  nativeImage,
  screen,
  shell,
} = require("electron");

const {
  DesktopStore,
  detectSpaceCliConfig,
  normalizeBaseUrl,
} = require("./store");
const {
  resolveRendererEntry,
  resetLiveUi,
  checkAndInstallUiUpdate,
  checkLatestAppRelease,
} = require("./updater");

const APP_ROOT = path.resolve(__dirname, "..");

const IS_MAC = process.platform === "darwin";
const IS_WIN = process.platform === "win32";
const IS_DEV = process.env.SPACE_DESKTOP_DEV === "1";
const SMOKE_TEST = process.env.SPACE_DESKTOP_SMOKE === "1";

let store = null;
let mainWindow = null;
let tray = null;
let isQuitting = false;
let currentFleetLabel = "Connecting fleet...";
let currentShortcut = null;

function assetPath(...parts) {
  return path.join(__dirname, "..", "assets", ...parts);
}

function isSafeExternalUrl(rawUrl) {
  try {
    const parsed = new URL(String(rawUrl || ""));
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

function sendAction(action, payload = null) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  if (!mainWindow.isVisible()) mainWindow.show();
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.focus();
  mainWindow.webContents.send("app:action", { action, payload });
}

function checkBoundsOnScreen(bounds) {
  if (typeof bounds.x !== "number" || typeof bounds.y !== "number") {
    return false;
  }
  const displays = screen.getAllDisplays();
  return displays.some((d) => {
    const wa = d.workArea;
    return (
      bounds.x >= wa.x - 100 &&
      bounds.y >= wa.y - 60 &&
      bounds.x < wa.x + wa.width - 100 &&
      bounds.y < wa.y + wa.height - 80
    );
  });
}

function createWindow() {
  const settings = store.loadSettings();
  const winState = store.loadWindowState();
  const useSavedPos = checkBoundsOnScreen(winState);

  const windowOptions = {
    width: winState.width,
    height: winState.height,
    minWidth: 740,
    minHeight: 520,
    x: useSavedPos ? winState.x : undefined,
    y: useSavedPos ? winState.y : undefined,
    title: "Space AI",
    backgroundColor: "#000000",
    show: false,
    autoHideMenuBar: !IS_MAC,
    alwaysOnTop: Boolean(settings.alwaysOnTop),
    icon: IS_MAC ? undefined : assetPath("icon-128.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: true,
      devTools: true,
    },
  };

  if (IS_MAC) {
    windowOptions.titleBarStyle = "hiddenInset";
    windowOptions.trafficLightPosition = { x: 16, y: 17 };
    windowOptions.vibrancy = "under-window";
    windowOptions.visualEffectState = "active";
  } else {
    windowOptions.frame = false;
  }

  mainWindow = new BrowserWindow(windowOptions);

  if (winState.isMaximized && !SMOKE_TEST) {
    mainWindow.maximize();
  }

  let saveTimer = null;
  function scheduleSaveBounds() {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (!mainWindow || mainWindow.isDestroyed()) return;
      const isMax = mainWindow.isMaximized();
      const b = isMax ? mainWindow.getNormalBounds() : mainWindow.getBounds();
      store.saveWindowState({
        width: b.width,
        height: b.height,
        x: b.x,
        y: b.y,
        isMaximized: isMax,
      });
    }, 300);
  }

  mainWindow.on("resize", scheduleSaveBounds);
  mainWindow.on("move", scheduleSaveBounds);
  mainWindow.on("maximize", () => {
    scheduleSaveBounds();
    if (!mainWindow.isDestroyed()) {
      mainWindow.webContents.send("window:maximized-changed", true);
    }
  });
  mainWindow.on("unmaximize", () => {
    scheduleSaveBounds();
    if (!mainWindow.isDestroyed()) {
      mainWindow.webContents.send("window:maximized-changed", false);
    }
  });

  mainWindow.on("close", (event) => {
    const curSettings = store.loadSettings();
    if (!isQuitting && !SMOKE_TEST && curSettings.minimizeToTray && tray) {
      event.preventDefault();
      mainWindow.hide();
      return;
    }
    scheduleSaveBounds();
  });

  mainWindow.once("ready-to-show", () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (!SMOKE_TEST) {
      mainWindow.show();
    }
    if (IS_DEV) {
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
  });

  // Security: disallow navigating away from local renderer file; open links externally
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) {
      shell.openExternal(url).catch(() => {});
    }
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith("file://")) {
      event.preventDefault();
      if (isSafeExternalUrl(url)) {
        shell.openExternal(url).catch(() => {});
      }
    }
  });

  // Right-click native context menu (copy/paste/spellcheck)
  mainWindow.webContents.on("context-menu", (_event, params) => {
    const items = [];
    if (params.misspelledWord && params.dictionarySuggestions.length) {
      for (const sug of params.dictionarySuggestions.slice(0, 5)) {
        items.push({
          label: sug,
          click: () => mainWindow.webContents.replaceMisspelling(sug),
        });
      }
      items.push({ type: "separator" });
    }
    if (params.isEditable) {
      items.push(
        { role: "cut", enabled: params.editFlags.canCut },
        { role: "copy", enabled: params.editFlags.canCopy },
        { role: "paste", enabled: params.editFlags.canPaste },
        { type: "separator" },
        { role: "selectAll" }
      );
    } else if (params.selectionText && params.selectionText.trim()) {
      items.push({ role: "copy" });
    }
    if (items.length) {
      Menu.buildFromTemplate(items).popup({ window: mainWindow });
    }
  });

  const resolved = resolveRendererEntry({
    userDataDir: app.getPath("userData"),
    appRootDir: APP_ROOT,
    appVersion: app.getVersion(),
  });
  mainWindow.loadFile(resolved.entryFile);

  if (SMOKE_TEST) {
    mainWindow.webContents.once("did-finish-load", () => {
      setTimeout(() => {
        console.log("SPACE_DESKTOP_SMOKE_OK");
        isQuitting = true;
        app.quit();
      }, 600);
    });
  } else {
    // Background check for verified Live UI updates (no DMG re-download needed)
    setTimeout(() => {
      runUpdateCheck({ silent: true }).catch(() => {});
    }, 4500);
  }
}

function toggleMainWindowVisibility() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    createWindow();
    return;
  }
  if (mainWindow.isVisible() && mainWindow.isFocused()) {
    mainWindow.hide();
  } else {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
    mainWindow.webContents.send("app:action", { action: "focus-composer" });
  }
}

function buildAppMenu() {
  const settings = store.loadSettings();
  const template = [
    ...(IS_MAC
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about", label: "About Space AI" },
              {
                label: "Check for Updates…",
                click: () => runUpdateCheck({ silent: false }),
              },
              { type: "separator" },
              {
                label: "Preferences…",
                accelerator: "CmdOrCtrl+,",
                click: () => sendAction("open-settings"),
              },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit", label: "Quit Space AI" },
            ],
          },
        ]
      : []),
    {
      label: "File",
      submenu: [
        {
          label: "New Chat",
          accelerator: "CmdOrCtrl+N",
          click: () => sendAction("new-chat"),
        },
        {
          label: "Command Palette…",
          accelerator: "CmdOrCtrl+K",
          click: () => sendAction("open-command-palette"),
        },
        { type: "separator" },
        {
          label: "Export Chat as Markdown…",
          accelerator: "CmdOrCtrl+E",
          click: () => sendAction("export-markdown"),
        },
        ...(!IS_MAC
          ? [
              { type: "separator" },
              {
                label: "Settings…",
                accelerator: "CmdOrCtrl+,",
                click: () => sendAction("open-settings"),
              },
            ]
          : []),
        { type: "separator" },
        IS_MAC ? { role: "close" } : { role: "quit", label: "Quit Space AI" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
      ],
    },
    {
      label: "Chat",
      submenu: [
        {
          label: "Toggle Decision X-Ray",
          accelerator: "CmdOrCtrl+Shift+X",
          click: () => sendAction("toggle-xray"),
        },
        {
          label: "Compact Conversation Context",
          accelerator: "CmdOrCtrl+Shift+K",
          click: () => sendAction("compact-chat"),
        },
        {
          label: "Live Fleet Inspector",
          accelerator: "CmdOrCtrl+Shift+F",
          click: () => sendAction("open-fleet"),
        },
        { type: "separator" },
        {
          label: "Insert /stress Framing Test",
          click: () => sendAction("insert-slash", "/stress "),
        },
        {
          label: "Insert /compare Bias-Free Judge",
          click: () => sendAction("insert-slash", "/compare "),
        },
        {
          label: "Insert /audit Truth Radar",
          click: () => sendAction("insert-slash", "/audit "),
        },
      ],
    },
    {
      label: "View",
      submenu: [
        {
          label: "Toggle Sidebar",
          accelerator: "CmdOrCtrl+B",
          click: () => sendAction("toggle-sidebar"),
        },
        {
          label: "Always on Top",
          type: "checkbox",
          checked: Boolean(settings.alwaysOnTop),
          accelerator: "CmdOrCtrl+Shift+T",
          click: (menuItem) => {
            const next = store.saveSettings({ alwaysOnTop: menuItem.checked });
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.setAlwaysOnTop(next.alwaysOnTop);
              mainWindow.webContents.send("settings:updated", next);
            }
            refreshTrayMenu();
          },
        },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        ...(IS_MAC ? [{ type: "separator" }, { role: "front" }] : [{ role: "close" }]),
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "Check for Updates…",
          click: () => runUpdateCheck({ silent: false }),
        },
        { type: "separator" },
        {
          label: "Space Research Website",
          click: () => shell.openExternal("https://spaceresearch.ai"),
        },
        {
          label: "Live Network Explorer (Web)",
          click: () => shell.openExternal("https://spaceresearch.ai/network.html"),
        },
        {
          label: "Open Source Repository (GitHub)",
          click: () => shell.openExternal("https://github.com/spaceresearchai/spaceresearch.ai"),
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function refreshTrayMenu() {
  if (!tray) return;
  const settings = store.loadSettings();
  const isVisible = mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible();
  const menu = Menu.buildFromTemplate([
    {
      label: isVisible ? "Hide Space AI" : "Show Space AI",
      click: toggleMainWindowVisibility,
    },
    {
      label: "New Chat",
      click: () => sendAction("new-chat"),
    },
    {
      label: "Live Fleet Inspector",
      click: () => sendAction("open-fleet"),
    },
    { type: "separator" },
    {
      label: `Fleet: ${currentFleetLabel}`,
      enabled: false,
    },
    { type: "separator" },
    {
      label: "Always on Top",
      type: "checkbox",
      checked: Boolean(settings.alwaysOnTop),
      click: (item) => {
        const next = store.saveSettings({ alwaysOnTop: item.checked });
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.setAlwaysOnTop(next.alwaysOnTop);
          mainWindow.webContents.send("settings:updated", next);
        }
        buildAppMenu();
      },
    },
    {
      label: "Settings…",
      click: () => sendAction("open-settings"),
    },
    {
      label: "Check for Updates…",
      click: () => runUpdateCheck({ silent: false }),
    },
    { type: "separator" },
    {
      label: "Quit Space AI",
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(menu);
  tray.setToolTip(`Space AI · ${currentFleetLabel}`);
}

function setupTray() {
  if (SMOKE_TEST) return;
  try {
    const iconFile = IS_MAC ? assetPath("tray-iconTemplate.png") : assetPath("tray-icon.png");
    if (!fs.existsSync(iconFile)) return;
    const icon = nativeImage.createFromPath(iconFile);
    if (IS_MAC) icon.setTemplateImage(true);
    tray = new Tray(icon);
    tray.on("click", () => {
      if (!IS_MAC) toggleMainWindowVisibility();
    });
    tray.on("double-click", toggleMainWindowVisibility);
    refreshTrayMenu();
  } catch (err) {
    console.warn("Tray setup skipped:", err?.message || err);
  }
}

function registerAppGlobalShortcut(settings) {
  if (SMOKE_TEST) return;
  try {
    if (currentShortcut) {
      globalShortcut.unregister(currentShortcut);
      currentShortcut = null;
    }
    if (settings.globalShortcutEnabled && settings.globalShortcut) {
      const ok = globalShortcut.register(settings.globalShortcut, () => {
        toggleMainWindowVisibility();
      });
      if (ok) currentShortcut = settings.globalShortcut;
    }
  } catch (err) {
    console.warn("Global shortcut registration failed:", err?.message || err);
  }
}

function applyDesktopSettings(settings) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.setAlwaysOnTop(Boolean(settings.alwaysOnTop));
  }
  registerAppGlobalShortcut(settings);
  if (!SMOKE_TEST && (IS_MAC || IS_WIN)) {
    try {
      app.setLoginItemSettings({ openAtLogin: Boolean(settings.launchAtLogin) });
    } catch {}
  }
  buildAppMenu();
  refreshTrayMenu();
}

function httpGetJson(urlString, timeoutMs = 2000) {
  return new Promise((resolve) => {
    let parsed;
    try {
      parsed = new URL(urlString);
    } catch {
      return resolve({ ok: false, error: "Invalid URL" });
    }
    const lib = parsed.protocol === "https:" ? https : http;
    const req = lib.get(
      parsed,
      { timeout: timeoutMs, headers: { "User-Agent": "space-desktop/1.0.0" } },
      (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
          if (data.length > 65536) req.destroy();
        });
        res.on("end", () => {
          let json = null;
          try {
            json = JSON.parse(data);
          } catch {}
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            data: json,
          });
        });
      }
    );
    req.on("timeout", () => {
      req.destroy();
      resolve({ ok: false, error: "Timeout" });
    });
    req.on("error", (err) => {
      resolve({ ok: false, error: err.message });
    });
  });
}

async function runUpdateCheck({ silent = false } = {}) {
  const userDataDir = app.getPath("userData");
  const appVersion = app.getVersion();
  let uiRes = null;
  let uiError = null;
  try {
    uiRes = await checkAndInstallUiUpdate({
      userDataDir,
      appRootDir: APP_ROOT,
      appVersion,
    });
  } catch (err) {
    uiError = err?.message || "Update check failed";
  }

  const releaseRes = await checkLatestAppRelease(appVersion);
  const payload = {
    appVersion,
    ui: uiRes,
    uiError,
    release: releaseRes,
    silent,
  };

  if (mainWindow && !mainWindow.isDestroyed()) {
    if (uiRes?.status === "updated" || (!silent && (uiRes || releaseRes?.hasNewAppRelease || uiError))) {
      mainWindow.webContents.send("updater:ready", payload);
    }
  }
  return payload;
}

function registerIpcHandlers() {
  ipcMain.handle("updater:info", () => {
    const resolved = resolveRendererEntry({
      userDataDir: app.getPath("userData"),
      appRootDir: APP_ROOT,
      appVersion: app.getVersion(),
    });
    return {
      appVersion: app.getVersion(),
      uiVersion: resolved.uiVersion,
      bundledUiVersion: resolved.bundledUiVersion,
      source: resolved.source,
    };
  });

  ipcMain.handle("updater:check", () => runUpdateCheck({ silent: false }));

  ipcMain.handle("updater:reload", () => {
    if (!mainWindow || mainWindow.isDestroyed()) return false;
    const resolved = resolveRendererEntry({
      userDataDir: app.getPath("userData"),
      appRootDir: APP_ROOT,
      appVersion: app.getVersion(),
    });
    mainWindow.loadFile(resolved.entryFile);
    return true;
  });

  ipcMain.handle("updater:reset", () => {
    resetLiveUi(app.getPath("userData"));
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadFile(path.join(APP_ROOT, "src", "renderer", "index.html"));
    }
    return true;
  });

  ipcMain.handle("store:loadSessions", () => store.loadSessions());
  ipcMain.handle("store:saveSessions", (_e, sessions) => store.saveSessions(sessions));
  ipcMain.handle("store:loadSettings", () => store.loadSettings());
  ipcMain.handle("store:saveSettings", (_e, partial) => {
    const next = store.saveSettings(partial);
    applyDesktopSettings(next);
    return next;
  });
  ipcMain.handle("store:detectCliConfig", () => detectSpaceCliConfig());

  ipcMain.handle("proxy:checkLocal", async (_e, rawUrl) => {
    const base = normalizeBaseUrl(rawUrl, "http://127.0.0.1:8377");
    const res = await httpGetJson(`${base}/health`, 1800);
    return { base, ...res };
  });

  ipcMain.handle("dialog:saveMarkdown", async (_e, { defaultFilename, content }) => {
    if (!mainWindow || mainWindow.isDestroyed()) return { canceled: true };
    const safeName = String(defaultFilename || "space-chat.md").replace(/[^a-zA-Z0-9._-]+/g, "-");
    const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
      title: "Export Conversation as Markdown",
      defaultPath: path.join(app.getPath("documents"), safeName),
      filters: [
        { name: "Markdown", extensions: ["md"] },
        { name: "All Files", extensions: ["*"] },
      ],
    });
    if (canceled || !filePath) return { canceled: true };
    fs.writeFileSync(filePath, String(content || ""), "utf8");
    return { canceled: false, filePath };
  });

  ipcMain.handle("dialog:savePngDataUrl", async (_e, { defaultFilename, dataUrl }) => {
    if (!mainWindow || mainWindow.isDestroyed()) return { canceled: true };
    const match = String(dataUrl || "").match(/^data:image\/png;base64,(.+)$/);
    if (!match) return { canceled: true, error: "Invalid PNG data" };
    const safeName = String(defaultFilename || "space-xray-card.png").replace(/[^a-zA-Z0-9._-]+/g, "-");
    const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
      title: "Save Space X-Ray 4:5 Card",
      defaultPath: path.join(app.getPath("pictures"), safeName),
      filters: [{ name: "PNG Image", extensions: ["png"] }],
    });
    if (canceled || !filePath) return { canceled: true };
    fs.writeFileSync(filePath, Buffer.from(match[1], "base64"));
    return { canceled: false, filePath };
  });

  ipcMain.handle("shell:revealFile", (_e, filePath) => {
    if (typeof filePath === "string" && filePath && fs.existsSync(filePath)) {
      shell.showItemInFolder(filePath);
      return true;
    }
    return false;
  });

  ipcMain.handle("shell:openExternal", async (_e, url) => {
    if (!isSafeExternalUrl(url)) return false;
    await shell.openExternal(url);
    return true;
  });

  ipcMain.handle("clipboard:writeText", (_e, text) => {
    clipboard.writeText(String(text || ""));
    return true;
  });

  ipcMain.handle("window:minimize", () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.minimize();
  });

  ipcMain.handle("window:toggleMaximize", () => {
    if (!mainWindow || mainWindow.isDestroyed()) return false;
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
      return false;
    }
    mainWindow.maximize();
    return true;
  });

  ipcMain.handle("window:close", () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
  });

  ipcMain.handle("window:isMaximized", () => {
    return Boolean(mainWindow && !mainWindow.isDestroyed() && mainWindow.isMaximized());
  });

  ipcMain.handle("window:setAlwaysOnTop", (_e, flag) => {
    const next = store.saveSettings({ alwaysOnTop: Boolean(flag) });
    applyDesktopSettings(next);
    return next.alwaysOnTop;
  });

  ipcMain.on("tray:updateFleetStatus", (_e, label) => {
    currentFleetLabel = String(label || "Connected").slice(0, 64);
    refreshTrayMenu();
  });
}

// Single-instance lock (skip during automated smoke test so parallel test runs work)
const gotLock = SMOKE_TEST ? true : app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (!mainWindow.isVisible()) mainWindow.show();
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    app.setName("Space AI");
    store = new DesktopStore(app.getPath("userData"));
    registerIpcHandlers();
    createWindow();
    setupTray();
    applyDesktopSettings(store.loadSettings());

    app.on("activate", () => {
      if (!mainWindow || mainWindow.isDestroyed()) {
        createWindow();
      } else {
        mainWindow.show();
        mainWindow.focus();
      }
    });
  });

  app.on("before-quit", () => {
    isQuitting = true;
  });

  app.on("will-quit", () => {
    globalShortcut.unregisterAll();
  });

  app.on("window-all-closed", () => {
    if (!IS_MAC) {
      app.quit();
    }
  });
}
