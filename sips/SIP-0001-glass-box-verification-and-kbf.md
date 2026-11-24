# SIP-0001: Glass-Box Verification, Proof-of-Sampling & Knowledge Boundary Fingerprinting (KBF)

- **SIP:** 0001
- **Title:** Glass-Box Verification, Proof-of-Sampling & Knowledge Boundary Fingerprinting (KBF)
- **Status:** Accepted
- **Category:** Verification / Security
- **Authors:** Space Research Core Team
- **Created:** 2026-11-23
- **Reference Implementation:** `network/crypto.py`, `network/verifier.py`, `network/kbf.py`, `network/verification_packs/`, `programs/longinus-settlement/src/lib.rs`

---

## 1. Motivation

Decentralized AI networks face an asymmetric fraud incentive: a dishonest operator can advertise a frontier 14B–70B+ model while silently routing user prompts to a cheap 1.5B–7B quantized substitute or a cached lookup table. Naive fixes fail in production:

- **ZKML / zkSNARKs** impose `100×–10,000×` proving overhead, destroying real-time interactive latency and coding-agent streaming.
- **Pure black-box API reselling** exposes neither token log-probabilities nor hidden states, making model substitution undetectable and eliminating Glass-Box interpretability features (Decision X-Ray, Top-8 fork inspection, Bluff/Sycophancy Radar).

SIP-0001 standardizes Space AI's **5-Tier Optimistic Glass-Box Verification Stack**: combining zero-overhead Ed25519 cryptographic receipts, hidden canaries, cross-hardware Top-8 log-prob Proof-of-Sampling, slashable Solana USDC bonds, and **Tier-0 Knowledge Boundary Fingerprinting (KBF)** via open, SHA-256-hashed Verification Packs for heterogeneous backends (Ollama, llama.cpp, vLLM, MLX).

---

## 2. Architecture Overview

| Layer | Mechanism | Target Hardware / Backend | Latency Overhead | Fraud Caught |
| :---: | :--- | :--- | :---: | :--- |
| **Tier 0** | **Knowledge Boundary Fingerprinting (KBF)** | All nodes (including Ollama / vLLM / llama.cpp / MLX adapters) | Periodic challenge (0 ms on user traffic) | Model downgrade spoofing (e.g. 7B pretending to be 27B) even without identical logprobs |
| **Layer 1** | **Ed25519 Signed `ExecutionReceipt`** | All native & adapter workers | `< 0.1 ms` | Output tampering, repudiation, receipt forgery |
| **Layer 2** | **Hidden Deterministic Canaries** | `corekv`, `corecompile`, `corerank` | `0 ms` (blended traffic) | Lazy caching, broken KV compression, guardrail bypass |
| **Layer 3** | **Top-8 Log-Prob & Latent Sketch Proof-of-Sampling** | Native CUDA FP8 + Apple Silicon MLX INT8 | `2%–5%` async spot-check | Subtle quantization cheats, weight substitution across CUDA/Metal |
| **Layer 4** | **On-Chain Solana USDC Slashing** | `space_settlement::slash_fraudulent_node` | Async settlement | Economic deterrence (`30%` challenger bounty / `70%` treasury) |

---

## 3. Specification

### 3.1 Canonical Ed25519 `ExecutionReceipt` (`network/crypto.py`)

Every inference request served by a Space AI worker produces a canonical JSON payload (keys sorted, compact separators `(",", ":")`) signed with the worker's 32-byte Ed25519 key (identical to its Solana Base58 wallet identity):

```json
{
  "request_id": "req_...",
  "node_wallet": "<Solana Base58 Ed25519 Pubkey>",
  "model_id": "DeepSeek-R1-Distill-14B-CoreKV-4x",
  "engine": "corekv",
  "prompt_hash": "<SHA-256 hex of canonical prompt>",
  "output_hash": "<SHA-256 hex of canonical output>",
  "top_logprobs": [-0.02, -4.10, -5.20, -6.10, -6.80, -7.20, -7.50, -8.00],
  "latent_sketch": [0.42, -0.31, 0.51, 0.18, -0.44, 0.27, 0.35, -0.22],
  "tokens_served": 256,
  "latency_ms": 412.5,
  "timestamp": 1795465000
}
```

- `receipt_hash = SHA256(canonical_payload)` binds the exact prompt, output, Top-8 log-probabilities, and 8-dimensional normalized latent sketch.
- Verification rejects any receipt failing `Ed25519.verify(node_wallet, canonical_payload, signature)` (`INVALID_ED25519_SIGNATURE`) or `SHA256(output_text) != output_hash` (`OUTPUT_HASH_MISMATCH`).

### 3.2 Cross-Hardware Proof-of-Sampling (`network/verifier.py`)

Because Space AI runs heterogeneously across **NVIDIA CUDA (BF16 / Triton FP8)** and **Apple Silicon Metal/MLX (INT8)**, bit-for-bit floating-point equality is impossible. Instead, `OptimisticVerifier` checks two hardware-tolerant statistical invariants on `2%–5%` sampled receipts and hidden canaries:

1. **Top-8 Log-Probability KL Divergence:**
   $$D_{\mathrm{KL}}(P_{\mathrm{ref}} \parallel P_{\mathrm{cand}}) = \sum_{i=1}^{k} p_i \ln \frac{p_i}{q_i} \le 0.15$$
   where $P, Q$ are softmax-normalized distributions over the Top-8 log-probabilities (`compute_topk_kl`, default `kl_threshold = 0.15`).
2. **8-Dim Latent Sketch Cosine Similarity:**
   $$\cos(\mathbf{v}_{\mathrm{ref}}, \mathbf{v}_{\mathrm{cand}}) = \frac{\mathbf{v}_{\mathrm{ref}} \cdot \mathbf{v}_{\mathrm{cand}}}{\|\mathbf{v}_{\mathrm{ref}}\| \|\mathbf{v}_{\mathrm{cand}}\|} \ge 0.90$$
   (`compute_cosine`, default `latent_cos_threshold = 0.90`).

Honest CUDA vs. Metal executions consistently satisfy $D_{\mathrm{KL}} \ll 0.05$ and $\cos > 0.98$, while substituted or severely degraded models fail decisively and emit a signed `FraudProof` referencing `fraud_receipt_hash`.

### 3.3 Tier-0 Knowledge Boundary Fingerprinting (KBF) & Open Verification Packs

Third-party adapters (`ollama`, `llama-cpp`, `vllm`, `mlx`) and heterogeneous GGUF/EXL2 quantizations may not expose identical pre-softmax latent sketches. **Tier-0 KBF** solves model-identity verification using open, versioned **Verification Packs** located in `network/verification_packs/`:

- **Pack Format (`qwen3.8-27b-v1.json`, `inkling-2-mini-v1.json`, `manifest.json`):**
  - `pack_id`, `model`, `version`, `description`
  - `min_pass_rate`, `spoof_threshold`, `baseline_pass_rate`
  - `probes[]`: deterministic factual, numerical, and multi-step reasoning boundary probes curated near the target model's capability cliff (easy for the claimed model, consistently failed by smaller downgraded models), paired with normalized matchers/regexes and difficulty tiers.
  - `sha256`: canonical SHA-256 digest verified by `load_verification_packs()` on every load; tampered packs are rejected immediately.
- **Statistical Scoring (`network/kbf.py::KBFVerifier`):**
  - Computes empirical pass rate, z-score / binomial confidence against the target baseline, and sets `suspected_spoof = true` when a node drops below the spoof-detection threshold.
- **Combined Integrity Verdict (`network/verifier.py::combined_integrity_verdict`):**
  - Combines Tier-0 KBF pass/fail with Layers 1–3 Proof-of-Sampling so native nodes achieve full **Glass-Box Verified** status while external adapters still undergo mandatory anti-spoofing checks.

### 3.4 Coordinator Integrity API (`network/router.py`)

- `GET /v1/integrity/summary` — Live network pass rate, verified receipt count, KBF pass rate, spoofs caught, active verification packs, Proof-of-Sampling rate, and latest inspectable signed receipts.
- `GET /v1/verification-packs` — Public metadata list (`pack_id`, `model`, `version`, `probe_count`, `min_pass_rate`, `sha256`, `description`).
- `GET /v1/verification-packs/{pack_id}` — Full open JSON pack for independent third-party audits.
- `POST /v1/integrity/kbf/evaluate` — Evaluates probe responses against a named pack, persists the audit verdict in SQLite, and returns the structured integrity result.

### 3.5 On-Chain Slashing (`programs/longinus-settlement/src/lib.rs`)

A verified `FraudProof` invokes `slash_fraudulent_node(slash_bps, fraud_receipt_hash)` on Solana program `Jfc1NK7cQkm4M3z3zpkhtUzFJZ4uj8XAF6hHcgu8KgL`:
- Deducts `slash_total` from the offending node's USDC bond (`NodeAccount.bonded_usdc`).
- Routes `challenger_bounty_bps` (default `3000` = **30%**) to the challenger/verifier and **70%** to the protocol treasury.
- Marks the node `NodeStatus::Slashed` if `slash_bps == 10_000` or remaining bond drops below `min_bond_usdc`, permanently excluding it from the current epoch's Merkle payout.

---

## 4. Security & Backward Compatibility

- **Replay & Tamper Resistance:** Every receipt hashes both prompt and output with an operator Ed25519 signature; every Verification Pack is SHA-256 integrity-checked at load time.
- **Canary Opacity:** Hidden canaries share identical HTTP/JSON schemas with organic traffic so workers cannot special-case audit prompts.
- **Backward Compatibility:** Tier-0 KBF is strictly additive to `OptimisticVerifier`; all existing `verify_receipt`, `verify_canary`, and Solana `slash_fraudulent_node` interfaces remain unchanged.
