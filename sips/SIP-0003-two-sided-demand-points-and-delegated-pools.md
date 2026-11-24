# SIP-0003: Two-Sided Season 0 Demand Points & Delegated Provider Pools ("Colonies")

- **SIP:** 0003
- **Title:** Two-Sided Season 0 Demand Points & Delegated Provider Pools ("Colonies")
- **Status:** Accepted
- **Category:** Economics / Solana
- **Authors:** Space Research Core Team
- **Created:** 2026-11-23
- **Reference Implementation:** `programs/longinus-settlement/src/state.rs`, `programs/longinus-settlement/src/lib.rs`, `programs/longinus-settlement/src/errors.rs`, `network/ledger.py`, `network/router.py`

---

## 1. Motivation

Early decentralized compute networks routinely fall into a one-sided subsidy trap: rewarding idle hardware supply without rewarding genuine developer demand, and letting large datacenter farms dominate token emissions until home builders (consumer RTX 4090/5090 and Apple Silicon Mac Studio operators) are diluted out.

SIP-0003 addresses both failure modes simultaneously:
1. **Two-Sided Season 0 Points:** Rewards verified developer usage (coding-agent sessions, Glass-Box Decision X-Ray shares, and traffic routed to independent home-builder rigs) alongside supply-side compute mining (`tSPACE` / `tLNG` Token-2022 non-transferable credits on Solana Devnet, mint `8joRqRNGf8sqNFC8ZR7MbkhHkAcj1dr6UPkf5YCpcn3m`).
2. **Delegated Provider Backing ("Colonies"):** Allows community members and developers to delegate tokens behind trusted independent hardware operators, boosting those operators' routing priority (`route_boost`) and unlocking daily free Space AI quota (`free_quota_bps` / `free_quota_daily_tokens`), while enforcing **on-chain hardware-tier anti-farm caps**.

---

## 2. Two-Sided Season 0 Demand Points (`network/ledger.py`, `network/router.py`)

Verified demand events are recorded in `demand_points_events` and aggregated in `demand_points_balances` automatically on live `/v1/chat/completions`, `/v1/messages`, and `/v1/shares` traffic (as well as explicit `POST /v1/points/demand/record`):

| Activity Class | Event Multiplier | Trigger Condition |
| :--- | :---: | :--- |
| **X-Ray / Truth Radar Share** (`xray_share`) | **3.0×** | Sharing a verifiable Decision X-Ray, Bluff Radar, or 5-Framing Stress-Test permalink via `/v1/shares` |
| **Coding Agent Traffic** (`coding_agent`) | **2.0×** | Requests tagged `X-Space-Client: coding-agent` (injected by `space claude` / `space codex` / `space opencode`) or Anthropic `/v1/messages` |
| **Home-Builder Pool Routing** (`home_builder_pool`) | **1.5×** | Inference served by non-datacenter home-builder pools (`consumer_gpu` or `apple_silicon`) |
| **Standard Verified Inference** (`standard`) | **1.0×** | Baseline verified API / chat inference |

Public telemetry is served via `GET /v1/points/demand`, returning the builder leaderboard, total demand points, total verified calls, and active multipliers.

---

## 3. Hardware-Tier Anti-Dilution & Anti-Farm Guarantees

Space AI preserves strict separation between scarce home-builder silicon and commodity datacenter farms at both the off-chain settlement layer and the Solana smart-contract layer:

1. **Off-Chain Epoch Sub-Pools (`network/ledger.py::TIER_LNG_POOL_SHARES`):**
   - `consumer_gpu`: `45%`
   - `apple_silicon`: `30%`
   - `datacenter_gpu`: `25%`
   - `cpu_edge`: `5%` (renormalized across active tiers)
2. **On-Chain Anti-Farm Delegation Caps (`programs/longinus-settlement/src/state.rs`):**

| `HardwareTier` | On-Chain `anti_farm_cap` (6 decimals) | Free-Quota Scale | Max `free_quota_bps` |
| :--- | :---: | :---: | :---: |
| `AppleSiliconUltra` (M3/M5 Ultra 192–512 GB) | `500,000 SPACE` (`500_000_000_000`) | `1.5×` | `10,000` (`100%`) |
| `AppleSiliconMax` (M3/M4/M5 Max 64–128 GB) | `350,000 SPACE` (`350_000_000_000`) | `1.5×` | `10,000` (`100%`) |
| `NvidiaConsumer` (RTX 4090 / RTX 5090) | `250,000 SPACE` (`250_000_000_000`) | `1.5×` | `10,000` (`100%`) |
| `NvidiaDatacenter` (A100 / H100 / H200 / B200) | `100,000 SPACE` (`100_000_000_000`) | `1.0×` | `5,000` (`50%`) |

Datacenter nodes face a **strictly lower per-node cap** (`100k` vs `250k–500k`) and a **50% ceiling** on delegator free-quota unlock, preventing capital-rich datacenter farms from capturing delegated routing weight away from independent home builders.

---

## 4. Solana Anchor Specification (`programs/longinus-settlement`)

Deployed Program ID: `Jfc1NK7cQkm4M3z3zpkhtUzFJZ4uj8XAF6hHcgu8KgL`

### 4.1 Backward-Compatible New PDAs (`state.rs`)

Zero changes are made to existing deployed account layouts (`ProtocolConfig`, `NodeAccount`, `EpochAccount`, `ClaimReceipt`). Two additive PDAs are introduced:

1. **`DelegationAccount`** (`SPACE = 140` bytes)
   - **Seeds:** `[b"delegation", delegator.key().as_ref(), node_authority.key().as_ref()]`
   - **Fields:** `delegator: Pubkey`, `node_authority: Pubkey`, `amount: u64`, `pending_undelegate_amount: u64`, `hardware_pool_tier: HardwareTier`, `delegated_at: i64`, `cooldown_until: i64`, `free_quota_bps: u16`, `bump: u8`
2. **`PoolStats`** (`SPACE = 94` bytes)
   - **Seeds:** `[b"pool_stats", node_authority.key().as_ref()]`
   - **Fields:** `node_authority: Pubkey`, `total_delegated: u64`, `delegator_count: u32`, `hardware_pool_tier: HardwareTier`, `anti_farm_cap: u64`, `bump: u8`

### 4.2 On-Chain Instructions (`lib.rs`)

1. **`delegate_to_node(amount: u64)`**
   - Requires `amount > 0` and `node.status == NodeStatus::Active`.
   - Initializes `PoolStats` (binding `hardware_pool_tier` and `default_anti_farm_cap`) and `DelegationAccount` if needed.
   - Enforces `pool_stats.total_delegated + amount <= pool_stats.anti_farm_cap` (`SpaceError::AntiFarmCapExceeded`).
   - Transfers `amount` from `delegator_lng` to `delegation_vault` (`authority = config` PDA), recomputes `free_quota_bps`, and emits `DelegationCreated`.
2. **`request_undelegate(amount: u64)`**
   - Requires `amount > 0`, `amount <= delegation.amount`, and `delegation.pending_undelegate_amount == 0`.
   - Immediately deducts `amount` from active `delegation.amount` and `pool_stats.total_delegated` (preventing free-riding on routing boost or free quota during exit), sets `cooldown_until = now + config.unbond_cooldown_secs`, recalculates `free_quota_bps`, and emits `UndelegationRequested`.
3. **`complete_undelegate()`**
   - Requires `delegation.pending_undelegate_amount > 0` and `Clock::get()?.unix_timestamp >= delegation.cooldown_until`.
   - Returns escrowed tokens from `delegation_vault` to `delegator_lng` using `config` PDA signer seeds, decrements `pool_stats.delegator_count` if active balance is zero, and emits `UndelegationCompleted`.
   - Does **not** require `node.status == NodeStatus::Active`, guaranteeing delegators can always withdraw after cooldown even if a provider node exits or is slashed.

### 4.3 Coordinator Pool API (`network/router.py`)

- `GET /v1/pools` — Lists active Colonies with `pool_id`, `operator_name`, `hardware_tier`, `pool_class`, `delegated_tspace`, `delegator_count`, `free_quota_daily_tokens`, `integrity_score`, and `route_boost`.
- `POST /v1/pools/delegate` and `POST /v1/pools/undelegate` — Off-chain Season 0 `tSPACE` pool delegation mirroring the on-chain Colony model prior to mainnet token unlock.

---

## 5. Security & Backward Compatibility

- **Zero Breaking Layout Changes:** Existing `ProtocolConfig`, `NodeAccount`, `EpochAccount`, and `ClaimReceipt` byte offsets are verified unchanged by unit tests (`existing_account_layouts_remain_unchanged`).
- **Anti-Free-Riding Cooldown:** Moving unstaked balances into `pending_undelegate_amount` immediately reduces `free_quota_bps` and `pool_stats.total_delegated` on `request_undelegate`, eliminating flash-delegation quota farming.
- **Unconditional Exit Safety:** `request_undelegate` and `complete_undelegate` validate `node_authority` via PDA seeds and `has_one` without gating on `NodeStatus::Active`, so delegator principal is never trapped by a slashed operator.
