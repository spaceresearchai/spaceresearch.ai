# Space Improvement Proposals (SIPs)

**Space Improvement Proposals (SIPs)** are the open technical standards, protocol specifications, and governance records for **Space AI** (built on the Longinus Network protocol).

SIPs exist to guarantee **credible neutrality**: every verification rule, routing constraint header, economic formula, anti-farm cap, and Solana settlement layout is specified publicly before activation so GPU operators, Apple Silicon home builders, coding-agent developers, and third-party verifier nodes can independently audit, implement, and challenge the protocol without relying on closed operator agreements.

---

## Index

| SIP | Title | Category | Status | Reference Implementation |
| :---: | :--- | :--- | :---: | :--- |
| **SIP-0001** | [Glass-Box Verification, Proof-of-Sampling & Knowledge Boundary Fingerprinting (KBF)](./SIP-0001-glass-box-verification-and-kbf.md) | Verification / Security | Accepted | `network/crypto.py`, `network/verifier.py`, `network/kbf.py`, `network/verification_packs/` |
| **SIP-0002** | [Unified Local Proxy, Coding Agent Launchers & Routing Constraint Tiers](./SIP-0002-local-proxy-and-routing-tiers.md) | Client / Networking | Accepted | `cli/`, `scripts/space`, `scripts/install-space.sh`, `network/router.py` |
| **SIP-0003** | [Two-Sided Season 0 Demand Points & Delegated Provider Pools ("Colonies")](./SIP-0003-two-sided-demand-points-and-delegated-pools.md) | Economics / Solana | Accepted | `programs/longinus-settlement/src/`, `network/ledger.py`, `network/router.py` |
| **SIP-0004** | [Agent-Native Solana USDC Session Budgets & Zero-Subscription Top-Up](./SIP-0004-agent-native-usdc-session-budgets.md) | Payments / Agents | Accepted | `network/ledger.py`, `network/router.py`, `cli/space_cli.py` |

---

## Proposal Lifecycle

Every SIP progresses through four explicit states:

```text
Draft ──▶ Review ──▶ Accepted ──▶ Final
  │          │
  └──▶ Withdrawn / Deferred
```

1. **Draft:** Author publishes a concrete technical specification with motivation, exact wire/PDA/HTTP schemas, security analysis, and backward-compatibility assessment.
2. **Review:** Open community comment window (minimum 7 days for parameter or API changes; 14 days for Solana on-chain PDA/instruction or slashing-rule changes). Reference implementation and automated tests must pass in CI.
3. **Accepted:** Specification and reference code are merged and deployed on Solana Devnet / Space AI public coordinator for live telemetry validation.
4. **Final:** Activated on Solana Mainnet Beta / production coordinator. A `Final` SIP is immutable except for errata; breaking changes require a superseding SIP.

---

## Credible Neutrality & Governance Rules

1. **Zero Silent Breaking Changes:** Deployed Solana PDAs (`ProtocolConfig`, `NodeAccount`, `EpochAccount`, `ClaimReceipt`, `DelegationAccount`, `PoolStats`) must preserve backward-compatible Borsh byte layouts across program upgrades. Any breaking account change requires a versioned PDA seed and a minimum 14-day migration window.
2. **Open Verification Packs & Reproducible Audits:** All Tier-0 Knowledge Boundary Fingerprinting (KBF) probes and verification pack manifests (`network/verification_packs/*.json`) are published with deterministic SHA-256 content hashes. Any community member can run independent verification or submit a `FraudProof` for the on-chain 30% USDC challenger bounty.
3. **Hardware-Pool Anti-Dilution Guarantee:** Protocol upgrades may never collapse home-builder pools (`consumer_gpu`, `apple_silicon`) into commodity datacenter pools (`datacenter_gpu`). Anti-farm caps and separate emission sub-pools remain enforceable on-chain (`PoolStats.anti_farm_cap`) and in epoch settlement (`TIER_LNG_POOL_SHARES`).
4. **Deterministic Protocol Economics:** The `70% Provider / 20% Treasury / 10% Buyback` USDC revenue split (`settle_revenue_split`) and sorted-pair SHA-256 Merkle leaf domain (`SPACE_EPOCH_LEAF_V1`) are verified atomically on Solana (`Jfc1NK7cQkm4M3z3zpkhtUzFJZ4uj8XAF6hHcgu8KgL`).
5. **No Pay-to-Win Verification Bypass:** Neither staked balance nor delegated pool size can disable Proof-of-Sampling, hidden canary injection, or KBF anti-spoofing audits on any active node.

---

## Submitting a New SIP

1. Copy the header metadata block (`SIP`, `Title`, `Status`, `Category`, `Authors`, `Created`, `Requires`) from an existing SIP.
2. Place the document at `sips/SIP-XXXX-short-kebab-title.md` and register it in the table above.
3. Include:
   - **Motivation** (concrete engineering or economic problem solved),
   - **Specification** (exact data structures, HTTP headers, PDA seeds, or cryptographic hashes),
   - **Security & Abuse Analysis** (attack vectors and mitigations),
   - **Backward Compatibility** (impact on existing nodes, clients, and Solana accounts).
