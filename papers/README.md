# Space AI Technical Papers & Reproducible Artifacts (Version 3.0)

Author: **Marcio Diaz** (`marcio.diaz@gmail.com`), **Space AI** (`https://spaceresearch.ai`)  
Hardware: **NVIDIA GeForce RTX 5090** (Blackwell) · PyTorch `2.11.0+cu130` · Triton `3.6.0`

All three papers below were generated, audited, and verified end-to-end on real GPU workloads, archived with permanent DataCite DOIs on Zenodo, and indexed in the live Space AI vector database (`https://spaceresearch.ai/papers`).

---

## 1. Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Theory, Fused Triton Kernels, and LLM Post-Training (v3.0)
- **Zenodo v3.0 DOI:** [`10.5281/zenodo.24513063`](https://zenodo.org/records/24513063) (Concept DOI: `10.5281/zenodo.24508478`)
- **PDF (8 pp.):** [`pytorch-kl-bregman-stability.pdf`](./pytorch-kl-bregman-stability.pdf) · **HTML:** [`pytorch-kl-bregman-stability.html`](./pytorch-kl-bregman-stability.html)
- **Standalone Drop-In Triton Kernel:** [`scripts/research_engine/bregman_kl_triton.py`](../scripts/research_engine/bregman_kl_triton.py)
- **Key v3.0 Results:**
  - `0.0%` negative KL across all 36 regimes (Wilcoxon $p = 1.45 \times 10^{-11}$); cuts FP32 ($\varepsilon = 10^{-4}$) median error **149,111×** (`21.14` → `1.42e-4`).
  - **Qwen3 `V = 151,936` kernel scaling:** `0.90 ms` / `296.8 MB` — **6.54× faster & −80.0% VRAM** vs. FP32-upcast, **1.81× faster & −33.3% VRAM** vs. PyTorch 2.11 `torch.compile` (TorchInductor), **25.6× faster** vs. Chunked-FP32.
  - **Real 3-seed 1.7B LLM LoRA post-training:** Fastest overall on both `Qwen3-1.7B` (`25.1 ms/step`, **1.25× faster E2E, −754 MB VRAM** vs. FP32-upcast) and `SmolLM2-1.7B` (`18.7 ms/step`, **4,616 MB** lowest VRAM), with `0.0%` negative/dead tokens.
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_kl_pytorch.py --output data/generated_papers/raw_kl_pytorch_v3.json
  ```

---

## 2. Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens (v3.0)
- **Zenodo v3.0 DOI:** [`10.5281/zenodo.24514497`](https://zenodo.org/records/24514497) (Concept DOI: `10.5281/zenodo.24508472`)
- **PDF (5 pp.):** [`steerable-not-decodable-ace-cores.pdf`](./steerable-not-decodable-ace-cores.pdf) · **HTML:** [`steerable-not-decodable-ace-cores.html`](./steerable-not-decodable-ace-cores.html)
- **Key v3.0 Results:**
  - **4 open-weight architectures (1.7B–7.3B):** `Qwen3-1.7B`, `SmolLM2-1.7B`, `Mistral-7B-v0.1`, `OLMo-2-1124-7B` across 16 `(model, domain)` cells and **2,688 strictly held-out cross-lexicon pairs** (100% single-token across all 4 tokenizers).
  - At peak steering depth, ACE readout (`0.765`) significantly beats both **General-Corpus Tuned Lens** (`0.697`, $p = 0.047$) and **Domain Tuned Lens** (`0.708`, $p = 0.047$) with mean $|\cos(q_{l^*}, \Delta W_U)| = 0.140$.
  - **Flip-1D necessity:** ACE (`+0.235`) outperforms Random (`-0.001`, $p = 1.53 \times 10^{-5}$), CAA (`+0.170`, $p = 1.53 \times 10^{-4}$), and L2 Probes (`+0.179`, $p = 3.82 \times 10^{-3}$).
  - **Keep-1D sufficiency:** Outperforms the full unperturbed model in **16 / 16 cells** (exact sign-test $p = 1.53 \times 10^{-5}$, mean `0.799` vs. `0.712` clean, `+0.087` rescue).
  - **Multi-token generation boundary condition:** Single-position prefill Flip-1D preserves fluent NLL (`0.86–1.49`) but flips only `0.0%–5.6%` of unconstrained base-model completions due to KV-cache re-attention, showing sustained per-step steering is needed for multi-token generation.
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_steering_lens.py --output data/generated_papers/raw_steering_lens_v3.json
  ```

---

## 3. Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders (v3.0)
- **Zenodo v3.0 DOI:** [`10.5281/zenodo.24515036`](https://zenodo.org/records/24515036) (Concept DOI: `10.5281/zenodo.24508484`)
- **PDF (7 pp.):** [`probmoe-batchtopk-sae.pdf`](./probmoe-batchtopk-sae.pdf) · **HTML:** [`probmoe-batchtopk-sae.html`](./probmoe-batchtopk-sae.html)
- **Key v3.0 Results:**
  - **9 L0-matched SAE methods** at `4×` (`d_sae = 8,192`) and **`8×` (`d_sae = 16,384`)** overcompleteness across `Qwen3-1.7B L13` and `SmolLM2-1.7B L12` (~206k leak-free paper-disjoint tokens, 3 seeds).
  - **Live Transformer Causal Splice (`k = 32`):** Recovers **`98.82%` CE loss (`KL = 0.2561`)** on `SmolLM2-1.7B` (`MarginProb+AuxK`: **`98.85%`, `KL = 0.2552`**, #1 of all 9 methods) and **`97.96%` (`KL = 0.3819`)** on `Qwen3-1.7B` (eliminating ungated v1's drop `97.20%, KL = 0.4582`).
  - **8× Overcompleteness (`d_sae = 16,384`, `Qwen3-1.7B L13`):** Pure-MSE `MarginProb-BatchTopK` ties the best FVU (`0.2764`) while slashing dead latents from `75.5%` (`BatchTopK-STE`) and `60.4%` (`BatchTopK+AuxK`) down to **`54.9%`** with zero AuxK.
  - **Capacity-controlled cross-seed matching (`Top-512` Hungarian):** Resolves raw alive-size confound; `MarginProb-BatchTopK` outperforms `BatchTopK+AuxK`, `TopK+AuxK`, and `JumpReLU` at `k = 64`.
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_probmoe_sae.py --output data/generated_papers/raw_probmoe_sae_v3.json
  ```
