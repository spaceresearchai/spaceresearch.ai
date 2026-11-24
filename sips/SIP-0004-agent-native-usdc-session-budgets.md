# SIP-0004: Agent-Native Solana USDC Session Budgets & Zero-Subscription Top-Up

- **SIP:** 0004
- **Title:** Agent-Native Solana USDC Session Budgets & Zero-Subscription Top-Up
- **Status:** Accepted
- **Category:** Payments / Agents
- **Authors:** Space Research Core Team
- **Created:** 2026-11-23
- **Reference Implementation:** `network/ledger.py`, `network/router.py`, `cli/space_cli.py`, `programs/longinus-settlement/src/lib.rs`

---

## 1. Motivation

Autonomous coding agents, CI pipelines, and researchers running short-lived multi-agent swarms face severe friction with legacy AI billing:

- **Account Signup Walls & Monthly Subscriptions:** Traditional providers force credit-card subscriptions (`$20–$200/month`), OAuth onboarding, and opaque monthly rate-limit buckets before a single token is generated.
- **Unbounded Blast Radius:** Sharing a master API key with an autonomous agent exposes the user's entire billing account if a loop or prompt injection runs away.
- **High Minimums & Settlement Overhead:** Micro-budgets (`$0.50`–`$5.00`) for one-off agent runs are uneconomical on legacy payment rails without transparent pay-as-you-go quoting and sub-cent settlement.

SIP-0004 defines **Agent-Native Session Budgets** (`sp_sess_...`): zero-signup-wall, hard-capped pay-as-you-go spending tokens backed by **Solana USDC** (with an optional fiat/card top-up bridge) that compose directly with `space` CLI launchers and Space AI's on-chain `70/20/10` USDC settlement split.

---

## 2. Specification

### 2.1 Session Budget Schema (`network/ledger.py::agent_session_budgets`)

Each session budget is persisted with strict cap and expiration invariants:

| Field | Type | Description |
| :--- | :--- | :--- |
| `session_token` | `TEXT PRIMARY KEY` | Cryptographically random bearer token prefixed `sp_sess_` |
| `owner_id` | `TEXT` | Solana wallet address, barter handle, or anonymous agent identifier |
| `rail` | `TEXT` | Funding rail: `solana_usdc` or `card_topup` |
| `max_usdc` | `REAL` | Hard spending cap in USD / USDC (e.g. `1.00`, `5.00`, `20.00`) |
| `spent_usdc` | `REAL` | Cumulative metered spend against this session |
| `token_rate_per_1m_usdc` | `REAL` | Transparent per-1M-token billing rate locked at session creation |
| `expires_at` | `INTEGER` | Unix timestamp after which the session token automatically rejects new calls |
| `active` | `INTEGER` | Boolean kill-switch (`1` = active, `0` = revoked/exhausted) |

### 2.2 Coordinator API (`network/router.py`)

1. **`POST /v1/sessions/budget`** — Creates a hard-capped `sp_sess_...` token with no account signup wall (`owner_id`, `max_usdc`, `rail`, optional `ttl_secs`).
2. **`GET /v1/sessions/budget/{session_token}`** — Returns live `max_usdc`, `spent_usdc`, `remaining_usdc`, `token_rate_per_1m_usdc`, `expires_at`, and `active` status.
3. **`POST /v1/sessions/budget/{session_token}/charge`** — Atomic debit for metered token consumption; rejects with **HTTP `402 Payment Required`** whenever `spent_usdc + charge > max_usdc`, expired, or inactive.
4. **Automatic Inference Metering:** Supplying `X-Space-Session: sp_sess_...` on `/v1/chat/completions`, `/v1/messages`, or `/v1/infer` enforces pre-flight budget availability, meters actual verified token output post-inference, and returns HTTP `402` cleanly when exhausted.
5. **`POST /v1/topup/quote`** — Accepts `{usd_amount, rail}` (`solana_usdc` or `card_topup`) and returns a transparent zero-subscription quote showing estimated tokens, Solana sub-cent network fee (`~$0.0005`), and card processing fee breakdown when using the fiat bridge.

### 2.3 CLI & Coding-Agent Integration (`cli/space_cli.py`)

Developers or orchestrators can mint a bounded session budget in one command and pass it straight into an ephemeral coding-agent launch:

```bash
space session create --max-usdc 5.00 --rail solana_usdc
space claude --session sp_sess_... --route verified-local-only --privacy zero-log
```

### 2.4 On-Chain Revenue Settlement (`programs/longinus-settlement/src/lib.rs`)

Metered USDC revenue collected from session budgets settles atomically via Solana Anchor instruction `settle_revenue_split(gross_usdc)` on `Jfc1NK7cQkm4M3z3zpkhtUzFJZ4uj8XAF6hHcgu8KgL`:

- **70% (`7000` bps)** → `provider_usdc_vault` (distributed to verified GPU/Mac workers via SHA-256 Merkle claims, leaf domain `SPACE_EPOCH_LEAF_V1`)
- **20% (`2000` bps)** → `treasury_usdc` (protocol R&D and verifier operations)
- **10% (`1000` bps)** → `buyback_usdc` (protocol buyback vault)

---

## 3. Security & Backward Compatibility

- **Blast-Radius Containment:** Compromise or runaway looping of an autonomous agent can never exceed `max_usdc` on its scoped `sp_sess_...` token.
- **Deterministic HTTP 402 Semantics:** Clients and coding-agent harnesses receive an explicit `402 Payment Required` error with remaining balance metadata instead of silent throttling or surprise overages.
- **Backward Compatibility:** Session budgets coexist cleanly alongside Season 0 barter API keys (`sp_...`) and anonymous free-tier / delegated free-quota calls without modifying existing authentication headers.
