# Security & Privacy Policy

## Scope & Transparency Commitment

Every file executed on a user's machine by the Space Research CLI (`space`), local buyer proxy, GPU/compute worker (`join.py`), or Space Desktop is published in plain source in this repository so anyone can audit it before running it:

- `cli/` (`__init__.py`, `e2ee.py`, `adapters.py`, `launchers.py`, `proxy.py`, `space_cli.py`)
- `scripts/space`
- `scripts/join.py` (identical to `join.py`)
- `scripts/install-space.sh` (identical to `install-space.sh`)
- `scripts/join-gpu.sh` (identical to `join-gpu.sh`)
- `Dockerfile.worker`
- `desktop/src/` (`main.js`, `preload.js`, `store.js`)

### Client Guarantees

1. **No privilege escalation:** Client installers and runtime scripts run strictly as your unprivileged user and never invoke `sudo`.
2. **No hidden shell modifications:** `install-space.sh` installs only to `~/.local/bin/space` and `~/.space_ai/app` (both overridable via `SPACE_BIN_DIR` / `SPACE_INSTALL_DIR`) and does not edit shell startup files.
3. **Bundle integrity verification:** `install-space.sh` verifies the SHA-256 digest of `space-cli.tar.gz` against `space-cli.sha256` and rejects tar path-traversal entries before extracting.
4. **Owner-only key permissions:** `~/.space_ai/config.json` and `~/.longinus_node.json` are written with mode `0600`.
5. **Local-only proxy binding:** `space` proxy binds to loopback (`127.0.0.1`) by default and refuses non-loopback binds in `local-only` mode.
6. **Optional container isolation:** `Dockerfile.worker` runs as unprivileged user `spaceworker` (`uid 10001`), sets `SPACE_HARDENED_DOCKER=1`, and supports `--read-only --cap-drop=ALL --security-opt=no-new-privileges`.

---

## Private AI Threat Model & Verifiable Audit Map ([SIP-0005](./sips/SIP-0005-private-ai-and-e2ee-blind-relay.md))

You do **not** have to take our word on privacy — inspect the exact enforcement points in plain source:

| Privacy Guarantee | Exact File & Function to Audit | What It Enforces |
| :--- | :--- | :--- |
| **Air-Gapped `local-only` Mode** | `cli/proxy.py::SpaceLocalProxy.execute_upstream`, `cli/e2ee.py::is_loopback_url`, `desktop/src/store.js::resolveEffectiveEndpoint` | Filters out every non-loopback upstream before any socket opens; fails closed with HTTP `503` (`local_only_airgap_blocked`). `/v1/messages/count_tokens` runs locally. |
| **Coordinator-Blind E2EE (`space-e2ee-v1`)** | `cli/e2ee.py::encrypt_client_request`, `cli/proxy.py::SpaceLocalProxy.execute_e2ee_chat` | Ephemeral X25519 + HKDF-SHA256 + AES-256-GCM encryption strictly inside your local proxy before posting **ciphertext only** to `POST /v1/e2ee/infer`. Zero plaintext bytes leave your device. |
| **Anti-MITM Worker Key Binding** | `cli/e2ee.py::verify_e2ee_pubkey_binding`, `scripts/join.py::_WorkerE2EE.signed_announcement` | Every worker signs its ephemeral X25519 public key with its Solana Base58 Ed25519 wallet key. Pinning `--private-node <wallet>` prevents even a compromised coordinator from swapping keys. |
| **Worker In-RAM Decryption & `--private-only`** | `scripts/join.py::process_worker_job` | Decrypts E2EE payloads strictly in RAM, never prints prompts/outputs, hashes **ciphertext** (not plaintext) in receipts, and refuses all plaintext jobs when `--private-only` is set. |
| **Private Pool Secret Hashing** | `cli/e2ee.py::hash_private_pool_secret`, `scripts/join.py::_hash_private_pool_secret` | `--private-pool <secret>` is SHA-256 hashed locally with domain separation; raw secrets never cross the network. |
| **Fail-Closed Coordinator Routing** | `network/router.py::select_nodes_with_constraints` (spec in [SIP-0005](./sips/SIP-0005-private-ai-and-e2ee-blind-relay.md)) | Rejects `local-only` with `403`, rejects plaintext `e2ee` with `400`, and fails closed with `503` if no eligible `hardened-docker` / `e2ee` / attested `tee` worker is online. |

### Honest Hardware Boundaries

- **Personal / Team GPU (`local-only` or `e2ee` on your own RTX / Apple Silicon rig):** Protects completely against SpaceResearch servers, CDNs, and ISPs with **< 1 ms crypto overhead**. Plaintext exists only in RAM/VRAM on your own machine.
- **Untrusted Third-Party GPUs:** Consumer GPUs (GeForce RTX 30/40/50-series) and consumer CPUs do not support hardware VRAM/RAM encryption (NVIDIA Confidential Computing / AMD SEV-SNP). Protecting prompts from an *untrusted third-party GPU host* requires hardware-attested Confidential Computing nodes (`X-Space-Privacy: tee`).

---

## Independent Verification

Before running any downloaded script or trusting Private AI claims, verify both checksums and security tests directly from this repository:

```bash
# 1. Verify space-cli.tar.gz matches published source & SHA-256 digest
sha256sum -c space-cli.sha256

# 2. Run both client proxy tests and Private AI hardening / anti-MITM tests
python3 -m unittest tests/test_space_cli_proxy.py tests/test_private_ai_hardening.py -v

# 3. Inspect active local privacy posture
./scripts/space privacy-check --json
```

## Reporting a Vulnerability

If you discover a security or privacy issue in the CLI, E2EE layer, local proxy, worker client, desktop app, installers, or `spaceresearch.ai` infrastructure, please report it responsibly:

- **GitHub Security Advisories:** Open a private vulnerability report in this repository's **Security → Advisories** tab.
- **Email:** `engineering@spaceresearch.ai`

Please do not disclose exploitable vulnerabilities publicly before we have had a reasonable opportunity to investigate and release a fix.
