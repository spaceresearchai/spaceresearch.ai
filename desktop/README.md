# Space AI Desktop (macOS, Windows & Linux)

Native cross-platform desktop application for **Space AI** — bringing the full Glass-Box Local AI Chat experience (`spaceresearch.ai/chat.html`) to macOS (Apple Silicon & Intel), Windows, and Linux with native desktop integrations.

## Features

- **Full Space Chat Parity**
  - Streaming OpenAI SSE multi-turn chat (`/v1/chat/completions`) with collapsible live reasoning/thinking drawers
  - **Live Decision X-Ray**: token-level certainty & margin sparklines, click any token for Top-K alternatives, and **1-click counterfactual forking**
  - **Truth, Sycophancy & Bluff Radar**: automatic per-message audits + one-click objective "Honest Take" rewrites
  - **Native Slash Commands** with interactive autocomplete (`/`):
    - `/stress <question>` — 5-framing belief stability & sycophancy stress-test with interactive framing slider
    - `/compare A | B | criterion` — Antisymmetric paired-swap bias-free ranking
    - `/audit <statement>` — Standalone truth & bluff check
    - `/compact [focus]` — Conversation compaction with token savings summary
  - **Multi-Computer Swarm Council** & **Sharded Multi-Node RPC** model selectors
  - **Image Attachments** via picker, clipboard paste, or drag-and-drop
  - **4:5 X-Ray PNG Card** & **Markdown** exports via native OS Save dialogs
- **Desktop-Native Capabilities**
  - **macOS** native hidden-inset titlebar with traffic lights & menu bar tray icon; **Windows & Linux** frameless obsidian titlebar & system tray
  - **Global Quick-Summon Hotkey** (`Cmd+Shift+Space` on macOS / `Ctrl+Shift+Space` on Windows & Linux)
  - **Command Palette** (`Cmd/Ctrl+K`) to jump between conversations, switch models, and trigger actions
  - **Live Fleet Inspector** (`Cmd/Ctrl+Shift+F`) to view online community Macs & GPUs and pin any serving station
  - **Seamless `space` CLI Integration** (`Cmd/Ctrl+,` → Settings): one-click import from `~/.space_ai/config.json`, direct connection to local `space proxy` (`http://127.0.0.1:8377`), or custom OpenAI-compatible endpoints
  - **Persistent Atomic Disk Storage**: conversations & settings saved in OS `userData` (`sessions.json`, `settings.json`) with no browser 5MB `localStorage` cap

## Running Locally

Requires Node.js 20+.

```bash
cd desktop
npm install
npm start
```

Run tests:

```bash
npm test
```

## Building Installers (macOS, Windows & Linux)

```bash
# macOS (.dmg + .zip for Apple Silicon arm64 & Intel x64 — run on macOS)
npm run build:mac

# Windows (NSIS installer .exe + portable .exe)
npm run build:win

# Linux (AppImage + .deb)
npm run build:linux
```

Installers are written to `desktop/dist/`. A GitHub Actions workflow (`.github/workflows/build-desktop.yml`) is also included to automatically build native macOS, Windows, and Linux binaries on tagged releases (`desktop-v*`) or manual dispatch.

## Security Model

- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`
- Strict Content-Security-Policy in `src/renderer/index.html`
- External navigation is blocked inside the Electron window; `https://` links open safely in your system browser
- Credentials and sessions are persisted locally with `0600` file permissions
