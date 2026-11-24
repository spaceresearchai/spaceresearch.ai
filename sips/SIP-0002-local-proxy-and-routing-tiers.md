# SIP-0002: Unified Local Proxy, Coding Agent Launchers & Routing Constraint Tiers

- **SIP:** 0002
- **Title:** Unified Local Proxy, Coding Agent Launchers & Routing Constraint Tiers
- **Status:** Accepted
- **Category:** Client / Networking
- **Authors:** Space Research Core Team
- **Created:** 2026-11-23
- **Reference Implementation:** `cli/space_cli.py`, `cli/proxy.py`, `cli/launchers.py`, `cli/adapters.py`, `scripts/space`, `scripts/install-space.sh`, `network/router.py`

---

## 1. Motivation

Developers adopting decentralized inference encounter three recurring friction points:

1. **Configuration Fragility:** Coding agents (`claude`, `codex`, `opencode`) each expect different environment variables and wire protocols (Anthropic `/v1/messages` vs. OpenAI `/v1/chat/completions`), and mutating shell profiles (`~/.bashrc`, `~/.zshrc`) risks clobbering existing paid subscriptions.
2. **Split Supply/Demand Ergonomics:** GPU and Apple Silicon owners should not need separate daemons, manual API-key copy-paste, or cloud accounts to earn credits while idle and spend them seamlessly when coding (especially when hitting cloud coding-agent rate limits).
3. **Opaque Routing:** Callers need explicit, auditable control over latency vs. cost, privacy guarantees (zero-log, hardened Docker, TEE), geographic region, and local hardware verification without vendor lock-in.

SIP-0002 specifies the `space` CLI, unified local buyer+worker proxy (`127.0.0.1:8377`), modular provider backend adapters, and standardized HTTP routing constraint headers.

---

## 2. Specification

### 2.1 Zero-Root Installer & One-Command Coding Agent Launchers

- **Installer:** `scripts/install-space.sh` installs the executable `space` entrypoint (`scripts/space` → `cli.space_cli`) into `~/.local/bin/space` with zero root privileges.
- **Ephemeral Launchers (`cli/launchers.py`):**
  - `space claude [args...]` — Spawns `claude` with ephemeral `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_API_KEY` pointing at the local proxy (`http://127.0.0.1:8377` by default, auto-starting it if needed).
  - `space codex [args...]` — Spawns `codex` with ephemeral `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
  - `space opencode [args...]` — Spawns `opencode` with ephemeral OpenAI-compatible environment variables.
  - **Invariants:** Launchers never modify `~/.bashrc`, `~/.zshrc`, or existing provider credentials on disk. Every launcher supports `--dry-run` (printing resolved environment and command without executing) plus routing flags (`--route`, `--privacy`, `--region`, `--model`, `--session`).

### 2.2 Unified Local Buyer Proxy (`cli/proxy.py`)

Default bind address: `127.0.0.1:8377` (loopback-only by default for local security, zero heavy runtime dependencies).

- **Supported Local Endpoints:**
  - `POST /v1/chat/completions` and `GET /v1/models` (OpenAI-compatible)
  - `POST /v1/messages` (Anthropic-compatible, forwarding natively to coordinator `/v1/messages` or cleanly translating when an upstream exposes OpenAI-only endpoints)
  - `GET /health` and `GET /status` (exposes active barter key status, local worker state, routing policy, failover counters, and per-upstream latency EMA)
- **Zero-Copy-Paste Barter Key Discovery:** Automatically reads the operator's earned barter key from `~/.space_ai/config.json` or `~/.longinus_node.json` so idle compute credits are spent automatically.
- **Client-Side Resilience:** Tracks exponential moving average (EMA) latency per configured upstream coordinator/node URL and performs automatic retry/failover on upstream timeout or 5xx failure.

### 2.3 Unified Worker + Proxy (`space up`) & Modular Provider Adapters (`cli/adapters.py`)

A single command starts both the local buyer proxy and a supervised worker backend:

```bash
space up --name my-rig --auto-pause --adapter native --proxy-port 8377
```

All provider backends implement the `ProviderAdapter` interface:

| Adapter | Default Local Endpoint | Hardware / Stack | Reward Tier |
| :--- | :--- | :--- | :--- |
| `native` | Managed via `scripts/join.py` | Space Portable CUDA (`CoreKV` FP8) / Apple MLX (`INT8`) | **Genesis High-Density Tier (20×)** — full Top-8 logprobs + latent sketch + Auto-Pause |
| `ollama` | `http://127.0.0.1:11434` | Ollama local daemon | Community Adapter Tier (Tier-0 KBF verified) |
| `llama-cpp` | `http://127.0.0.1:8081` | `llama-server` / GGUF | Community Adapter Tier (Tier-0 KBF verified) |
| `vllm` | `http://127.0.0.1:8000` | vLLM OpenAI-compatible server | Community Adapter Tier (Tier-0 KBF verified) |
| `mlx` | `http://127.0.0.1:8080` | `mlx_lm.server` on Apple Silicon | Community Adapter Tier (Tier-0 KBF verified) |

`space adapters` probes local endpoints and binaries, prints detected hardware backends, and recommends the highest-earning compatible tier.

### 2.4 Standardized Routing, Privacy & Region Headers

Clients and the local proxy communicate routing policy via explicit HTTP headers parsed by the coordinator (`network/router.py::select_nodes_with_constraints`):

| Request Header | Allowed Values | Coordinator Behavior |
| :--- | :--- | :--- |
| `X-Space-Route` | `fastest` (default), `cheapest`, `verified-local-only` | `fastest`: ranks by lowest TTFA / highest tok/s + delegated `route_boost`. `cheapest`: sorts by lowest unit cost / community pool. `verified-local-only`: restricts candidates to verified high-integrity local hardware nodes. |
| `X-Space-Privacy` | `standard` (default), `zero-log`, `hardened-docker`, `tee` | `zero-log`: disables prompt/completion persistence in telemetry. `hardened-docker`: prefers hardened container workers (`Dockerfile.worker`). `tee`: restricts to confidential-compute attested nodes when available. |
| `X-Space-Region` | `auto` (default), `us`, `eu`, `ap`, ISO-region tag | Filters or strongly prioritizes nodes matching the requested geographic region. |
| `X-Space-Client` | `coding-agent`, `web-chat`, `sdk`, `cli` | Identifies client class (e.g., `coding-agent` injected by launchers for 2.0× Season 0 demand-points accrual per SIP-0003). |
| `X-Space-Session` | `sp_sess_<token>` | Attaches an agent-native capped USDC/card session budget per SIP-0004. |

**Mandatory Policy Echo:** The coordinator echoes the resolved routing policy on every inference response via:
- `X-Space-Route-Applied`
- `X-Space-Privacy-Applied`
- `X-Space-Region-Applied`

---

## 3. Security & Backward Compatibility

- **Loopback Isolation:** The local proxy binds `127.0.0.1` by default so other machines on a LAN cannot consume the user's barter credits or session budget.
- **Credential Hygiene:** Barter keys and `sp_sess_` tokens are injected only into child process environments and outbound Authorization / `X-Space-Session` headers, never echoed back in `/status` responses in plaintext.
- **Backward Compatibility:** Standard OpenAI and Anthropic clients that omit all `X-Space-*` headers receive `fastest` / `standard` / `auto` defaults with zero schema changes.
