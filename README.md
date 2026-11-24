# Space Research (`spaceresearch.ai`) — Open Client, Worker & Protocol Specs

This public repository contains **100% of the source code that runs on your machine** when you use the `space` CLI, join as a GPU / Apple Silicon compute worker, or visit [spaceresearch.ai](https://spaceresearch.ai), together with our open Space Improvement Proposals (`sips/`).

Nothing hidden, nothing obfuscated: every installer script, CLI module, local proxy, GPU worker client, and container definition distributed by `spaceresearch.ai` is published here in plain source so you can audit every line before running it.

---

## What Runs on Your Machine (Audit Map)

| Component | Path in This Repo | What It Does |
| :--- | :--- | :--- |
| **`space` CLI & Local Proxy** | [`cli/`](./cli/) & [`scripts/space`](./scripts/space) | Zero-root CLI, Anthropic/OpenAI local proxy (`cli/proxy.py`), auditable E2EE & air-gap primitives (`cli/e2ee.py`), coding-agent launchers (`cli/launchers.py`), and provider adapters (`cli/adapters.py`). Exact unpacked source of [`space-cli.tar.gz`](./space-cli.tar.gz). |
| **GPU / Compute Worker** | [`scripts/join.py`](./scripts/join.py) (identical to [`join.py`](./join.py)) | Auditable worker client that runs local `llama-server` inference, supports `--auto-pause`, `--private-only` (refuses plaintext jobs), `--private-pool`, and `space-e2ee-v1` in-RAM decryption. |
| **Zero-Root CLI Installer** | [`scripts/install-space.sh`](./scripts/install-space.sh) (identical to [`install-space.sh`](./install-space.sh)) | Installs `space` to `~/.local/bin/space` with SHA-256 verification. Never uses `sudo` and never modifies `~/.bashrc` or `~/.zshrc`. |
| **GPU Worker Installer** | [`scripts/join-gpu.sh`](./scripts/join-gpu.sh) (identical to [`join-gpu.sh`](./join-gpu.sh)) | Sets up an isolated Python venv at `~/.space_research/venv` and launches `join.py`. |
| **Hardened Docker Worker** | [`Dockerfile.worker`](./Dockerfile.worker) | Minimal, non-root (`uid 10001`), read-only container sandbox (`SPACE_HARDENED_DOCKER=1`) for running the GPU worker without host filesystem access outside `/data`. |
| **Protocol Specs (SIPs)** | [`sips/`](./sips/) | Open technical specifications for Glass-Box Verification & KBF ([SIP-0001](./sips/SIP-0001-glass-box-verification-and-kbf.md)), Local Proxy & Routing ([SIP-0002](./sips/SIP-0002-local-proxy-and-routing-tiers.md)), Demand Points & Pools ([SIP-0003](./sips/SIP-0003-two-sided-demand-points-and-delegated-pools.md)), USDC Session Budgets ([SIP-0004](./sips/SIP-0004-agent-native-usdc-session-budgets.md)), and Auditable Private AI & Blind-Relay E2EE ([SIP-0005](./sips/SIP-0005-private-ai-and-e2ee-blind-relay.md)). |
| **Client & Privacy Test Suites** | [`tests/test_space_cli_proxy.py`](./tests/test_space_cli_proxy.py), [`tests/test_private_ai_hardening.py`](./tests/test_private_ai_hardening.py) | Automated test suites covering proxy translation, failover, `local-only` air-gap enforcement, `space-e2ee-v1` blind relay, Ed25519 anti-MITM key verification, and `--private-only` worker enforcement. |
| **Web Portal** | [`index.html`](./index.html), [`network.html`](./network.html), [`chat.html`](./chat.html), [`product.html`](./product.html) | Static web frontend served at `https://spaceresearch.ai`. |
| **Desktop App (macOS / Windows / Linux)** | [`desktop/`](./desktop/) | Native cross-platform Space AI Chat desktop app with Decision X-Ray, token forking, `/stress`, `/compare`, `/audit`, global quick-summon hotkey, tray/menubar, and local `space proxy` integration. |

---

## Install Directly From Source (No `curl | sh` Required)

If you prefer to inspect the code locally before executing anything:

```bash
git clone https://github.com/spaceresearchai/spaceresearch.ai.git
cd spaceresearch.ai

# 1. Run the client test suite first
python3 -m unittest tests/test_space_cli_proxy.py -v

# 2. Install the `space` CLI directly from your audited checkout (no root, no shell edits)
sh scripts/install-space.sh --copy
```

Or run the GPU worker inside a **read-only, non-root Docker sandbox**:

```bash
docker build -f Dockerfile.worker -t space-worker .
docker run --rm -it --gpus all --read-only \
  -v space_data:/data -e HOME=/data \
  space-worker --name my-rtx-rig --auto-pause
```

---

## Verify Checksums Against `spaceresearch.ai`

You can confirm that the bundle and scripts served at `https://spaceresearch.ai` match this repository byte-for-byte:

```bash
# Verify space-cli.tar.gz SHA-256 matches space-cli.sha256
sha256sum -c space-cli.sha256

# Inspect every file inside space-cli.tar.gz and diff against cli/ and scripts/
tar -tzvf space-cli.tar.gz

# Verify live web downloads match your local git checkout
diff -u scripts/install-space.sh <(curl -fsSL https://spaceresearch.ai/install-space.sh)
diff -u scripts/join-gpu.sh      <(curl -fsSL https://spaceresearch.ai/join-gpu.sh)
diff -u scripts/join.py          <(curl -fsSL https://spaceresearch.ai/join.py)
```

---

## Security & Privacy Highlights

- **Auditable Private AI ([SIP-0005](./sips/SIP-0005-private-ai-and-e2ee-blind-relay.md)):**
  - **Air-Gapped Local Mode (`--privacy local-only`):** Strictly blocks all non-loopback egress and fails closed (`503`) if no local `127.0.0.1` model server is configured.
  - **Coordinator-Blind E2EE (`--privacy e2ee`, `space-e2ee-v1`):** Ephemeral X25519 + HKDF-SHA256 + AES-256-GCM encryption runs inside `cli/proxy.py` with mandatory Ed25519 worker key binding (`cli/e2ee.py`). Pin your own RTX / Apple Silicon rig with `--private-node <wallet>` and `--private-only` on `join.py` so `api.spaceresearch.ai` only relays ciphertext (< 1 ms overhead).
  - **Fail-Closed Routing:** Strict privacy tiers (`local-only`, `e2ee`, `hardened-docker`, `tee`) never silently downgrade to public plaintext workers.
- **Zero root required:** Neither `install-space.sh`, `join-gpu.sh`, `space`, nor `join.py` requests `sudo` or root privileges.
- **No shell-profile mutation:** `install-space.sh` never edits `~/.bashrc`, `~/.zshrc`, or `~/.profile`.
- **Sandboxed execution:** `Dockerfile.worker` runs as unprivileged user `spaceworker` (`uid 10001`), sets `SPACE_HARDENED_DOCKER=1`, and supports `--read-only --cap-drop=ALL --security-opt=no-new-privileges`.
- **Auto-pause GPU safety:** Passing `--auto-pause` automatically yields your GPU whenever local foreground/gaming workloads use VRAM or compute.
- **Verify locally in 10 seconds:** Run `python3 -m unittest tests/test_private_ai_hardening.py -v` and `./scripts/space privacy-check`.

See [SECURITY.md](./SECURITY.md) for our full threat model, hardware boundaries, audit map, and vulnerability disclosure process.

---

## License

The client software, scripts, Dockerfile, and documentation in this repository are released under the [MIT License](./LICENSE).
