# SIP-0005: Auditable Private AI, Air-Gapped Local Mode & Blind-Relay E2EE (`space-e2ee-v1`)

- **SIP:** 0005
- **Title:** Auditable Private AI, Air-Gapped Local Mode & Blind-Relay E2EE (`space-e2ee-v1`)
- **Status:** Accepted
- **Category:** Privacy / Security
- **Authors:** Space Research Core Team
- **Created:** 2026-11-24
- **Requires:** SIP-0001, SIP-0002
- **Reference Implementation:** `cli/e2ee.py`, `cli/proxy.py`, `cli/space_cli.py`, `scripts/join.py`, `desktop/src/store.js`, `network/router.py`

---

## 1. Motivation

Developers and teams using Space AI for sensitive codebases, internal documents, or personal data require **verifiable privacy guarantees**, not soft "please don't log" promises:

1. **No Silent Downgrades:** Requesting a strict privacy tier (`local-only`, `e2ee`, `hardened-docker`, `tee`) must **fail closed** (HTTP `503` / `403`) if eligible private compute is unavailable — never silently falling back to public plaintext nodes while claiming privacy was applied.
2. **Coordinator-Blind Remote Inference:** A user with an RTX 5090 or Apple Silicon workstation at home (or a team VPC) should be able to route requests from laptops, coding agents (`claude`, `codex`, `opencode`), or Space Desktop through `api.spaceresearch.ai` without the coordinator, CDN, or ISP ever seeing plaintext prompts or outputs.
3. **Anti-MITM Verification in Plain Client Source:** Because 100% of Space client code (`cli/`, `scripts/join.py`, `desktop/`) is published in plain source with deterministic SHA-256 checksums, any user can audit the exact cryptographic and air-gap invariants on GitHub.

---

## 2. Privacy Tiers & Honest Threat Model

| `X-Space-Privacy` | Execution Path | Coordinator Sees Plaintext? | Worker Host Sees Plaintext? | Failure Mode (Default) |
| :--- | :--- | :---: | :---: | :--- |
| `local-only` | Loopback only (`127.0.0.1` / `localhost` / `::1`) | **No (zero network egress)** | User's own machine only | **Fail closed (`503`)** if no loopback upstream; remote coordinator rejects with `403` |
| `e2ee` | Client → Blind Relay → Verified Worker (`space-e2ee-v1`) | **No (ciphertext only)** | Yes (worker hardware decrypts in RAM; ideal for your own GPU / private team pool) | **Fail closed (`503`)** if no worker passes Ed25519→X25519 signature check |
| `hardened-docker` | Non-root `--read-only` sandbox (`Dockerfile.worker`) | Yes (unless combined with `e2ee`) | Host root can inspect RAM; container prevents disk persistence | **Fail closed (`503`)** if no `hardened-docker` worker is online |
| `tee` | Hardware-attested Confidential Compute (H100/H200/B200 CC + TDX/SEV-SNP) | **No** (when paired with client-to-enclave E2EE) | **No** (hardware-encrypted VRAM/RAM + attestation quote) | **Fail closed (`503`)** if no `tee_attested` worker is online |
| `zero-log` | Standard/eligible node with telemetry metadata scrubbing | Policy-level no-persist | Operator policy | Best-effort zero-log telemetry |

> **Hardware Honesty Note:** Consumer GPUs (including NVIDIA GeForce RTX 30/40/50-series) and consumer CPUs do not support NVIDIA Confidential Computing or AMD SEV-SNP hardware VRAM/RAM encryption. Therefore, `e2ee` on a personal RTX rig guarantees **zero coordinator/network visibility (< 1 ms crypto overhead)** while trusting your own hardware. Protecting against an *untrusted third-party GPU owner* requires `tee` on attested datacenter Confidential Computing GPUs.

---

## 3. Specification

### 3.1 Air-Gapped Local Mode (`X-Space-Privacy: local-only`)

- **Client Proxy (`cli/proxy.py::SpaceLocalProxy`):**
  1. `start()` refuses to bind non-loopback host addresses when `privacy == "local-only"`.
  2. `execute_upstream()` filters all configured upstreams through `is_loopback_url(url)` (`127.0.0.1`, `::1`, `localhost`; rejecting userinfo or suffix spoofing like `localhost.evil.com`).
  3. If no loopback upstream is configured, the proxy fails closed with **HTTP `503`** (`code: "local_only_airgap_blocked"`) without opening any socket.
  4. Auxiliary endpoints (`/v1/messages/count_tokens`) execute locally and never transmit prompt bytes upstream.
- **Desktop (`desktop/src/store.js::resolveEffectiveEndpoint`):** Forces loopback endpoint resolution and clears `fallbackUrl` whenever `privacy === "local-only"`.
- **Coordinator Defense-in-Depth (`network/router.py`):** If any request carrying `X-Space-Privacy: local-only` ever reaches the remote coordinator, it is immediately rejected with **HTTP `403`**.

### 3.2 Blind-Relay End-to-End Encryption (`space-e2ee-v1`)

Implemented in plain source in `cli/e2ee.py` (client/proxy) and `scripts/join.py` (worker):

1. **Worker Key Generation & Anti-MITM Binding:**
   - On startup, the worker generates an ephemeral in-RAM **X25519** keypair (never written to disk → forward secrecy across restarts).
   - The worker signs its X25519 public key using its Solana Base58 **Ed25519** identity key:
     $$\text{msg} = \text{UTF8}(\texttt{"space-e2ee-v1:keybind:"} \parallel \text{wallet} \parallel \texttt{":"} \parallel \text{base64}(pk_{\text{X25519}}))$$
     $$\sigma = \text{Ed25519.Sign}(sk_{\text{wallet}}, \text{msg})$$
   - The coordinator verifies $\sigma$ on `POST /v1/nodes/register` (rejecting forged keys with `403`), AND the client **independently re-verifies $\sigma$ locally** before encrypting.
   - When a user pins `--private-node <wallet>`, **not even a compromised coordinator can MITM the key exchange**.

2. **Per-Request Client Encryption (`encrypt_client_request`):**
   - Client generates a fresh ephemeral X25519 keypair $(esk_c, epk_c)$ and random 96-bit `nonce` per request.
   - Computes shared secret $S = \text{X25519}(esk_c, pk_{\text{worker}})$.
   - Derives 256-bit request key via HKDF-SHA256:
     $$K_{\text{req}} = \text{HKDF-SHA256}(S,\; \text{salt}=\text{nonce},\; \text{info}=\texttt{"space-e2ee-v1:req:"} \parallel \text{req\_id} \parallel \texttt{":"} \parallel \text{wallet})$$
   - Encrypts the JSON prompt/messages payload with **AES-256-GCM** using $\text{AAD} = \text{info}$.
   - Submits **only ciphertext** to `POST /v1/e2ee/infer` (`{request_id, target_node, model_id, e2ee_envelope}`).

3. **Worker In-RAM Decryption & Encrypted Response (`scripts/join.py::process_worker_job`):**
   - Worker decrypts `e2ee_envelope` strictly in RAM, runs local inference, and encrypts the response using a distinct domain-separated key (`space-e2ee-v1:resp:<req_id>:<wallet>`) and fresh 96-bit nonce.
   - ExecutionReceipt `prompt_hash` and `output_hash` are computed over the **Base64 ciphertexts** (preventing dictionary attacks on short prompts).
   - Plaintext `output_text` returned to the coordinator is `""`; encrypted output is carried in `e2ee_response`.
   - Workers launched with `--private-only` refuse all plaintext jobs (including plaintext hidden canaries) and opt out of public pool routing.

4. **Private Personal / Team Pools (`--private-pool <secret>`):**
   - Both client and worker derive `private_pool_id = SHA256("space-private-pool-v1:" + secret)[:32]` locally; the raw secret never crosses the wire.
   - Public requests are strictly excluded from private-pool and `--private-only` workers.

### 3.3 Fail-Closed Coordinator Policy & Honest Response Headers

- For `hardened-docker`, `e2ee`, and `tee`, `select_nodes_with_constraints(..., strict_privacy=True)` returns an empty candidate list (HTTP `503`) if no matching worker is online.
- Plaintext requests carrying `X-Space-Privacy: e2ee` to `/v1/infer`, `/v1/chat/completions`, or `/v1/messages` are rejected with HTTP `400`.
- Multi-node plaintext council fan-out (`space-council`) is blocked (`403`) under strict privacy policies.
- Explicit opt-in downgrade (`X-Space-Privacy-Fallback: allow`) is permitted only when requested by the caller and MUST echo:
  - `X-Space-Privacy-Applied: standard`
  - `X-Space-Privacy-Matched: false`
  - `X-Space-Privacy-Downgraded: true`

---

## 4. Independent Verification (3 Commands)

Anyone can verify that Space AI enforces these guarantees without trusting hosted claims:

```bash
# 1. Confirm downloadable CLI/worker bundle matches public GitHub source byte-for-byte
sha256sum -c space-cli.sha256

# 2. Run the automated Private AI security & anti-MITM test suite
python3 -m unittest tests/test_private_ai_hardening.py -v

# 3. Audit local client configuration & E2EE readiness
./scripts/space privacy-check --json
```
