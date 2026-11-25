# Space AI Technical Papers & Reproducible Artifacts (Version 4.0)

Author: **Marcio Diaz** (`marcio.diaz@gmail.com`), **Space AI** (`https://spaceresearch.ai`)  
Hardware: **NVIDIA GeForce RTX 5090** (Blackwell) · PyTorch `2.11.0+cu130` · Triton `3.6.0` · Tectonic `0.17.0`

All three papers below were executed, audited, and verified end-to-end on real GPU workloads, archived with permanent DataCite DOIs on Zenodo, compiled into double-blind TMLR / ICML 2027 / arXiv LaTeX packages (`papers/latex/`), and indexed in the live Space AI vector database (`https://spaceresearch.ai/papers`).

---

## 1. Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Theory, Fused Triton Kernels, and LLM Post-Training (v4.0)
- **Zenodo v4.0 DOI:** [`10.5281/zenodo.24521689`](https://zenodo.org/records/24521689) (Concept DOI: `10.5281/zenodo.24508478`)
- **PDF (8 pp.):** [`pytorch-kl-bregman-stability.pdf`](./pytorch-kl-bregman-stability.pdf) · **HTML:** [`pytorch-kl-bregman-stability.html`](./pytorch-kl-bregman-stability.html)
- **LaTeX Packages:** [`latex/pytorch-kl-bregman-stability/`](./latex/pytorch-kl-bregman-stability/) (`submission_tmlr_anon.pdf`, `icml2027_anon.pdf`, `preprint_arxiv.pdf`, `arxiv_bundle.tar.gz`)
- **Standalone Drop-In Triton Kernel:** [`scripts/research_engine/bregman_kl_triton.py`](../scripts/research_engine/bregman_kl_triton.py)
- **Key v4.0 Results:**
  - `0.0%` negative KL across all 36 regimes (Wilcoxon $p = 1.45 \times 10^{-11}$); cuts FP32 ($\varepsilon = 10^{-4}$) median error **149,111×** (`21.14` → `1.42e-4`).
  - **Liger-Style Standard Fused Triton Audit (`FP32, eps = 1e-4`):** Even inside a fused FP32-register SRAM kernel without Bregman–Taylor, standard `p*(lp - lq)` STILL yields **`49.66%` negative KL** and **`69.2×` (`6,919%`) median relative error** — proving SRAM fusion alone does not fix cancellation.
  - **Qwen3 `V = 151,936` kernel scaling:** `0.90 ms` / `296.8 MB` — **6.53× faster & −80.0% VRAM** vs. FP32-upcast, **1.81× faster & −33.3% VRAM** vs. PyTorch 2.11 `torch.compile`, **25.55× faster** vs. Chunked-FP32.
  - **Real 200-Step 3-Seed BF16 RLHF on `Qwen3-1.7B` (`V = 151,936, β = 2.0`, 512 real tokens/step):** Clamped `F.kl_div` suffers `13.4%` dead-gradient tokens, `+65%` excess KL drift (`0.00282` vs. `0.00171`), and drops true FP64 objective to `+0.02489` vs. **`+0.02637`** for Fused Triton, while Fused Triton achieves **`0.0%` negative/dead tokens** and runs **`14.2 ms/step`** (**1.55× faster E2E** & **−458 MB peak VRAM** vs. FP32-upcast; fastest of all 5 methods).
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_kl_pytorch.py --output data/generated_papers/raw_kl_pytorch_v4.json
  ```

---

## 2. Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens (v4.0)
- **Zenodo v4.0 DOI:** [`10.5281/zenodo.24522101`](https://zenodo.org/records/24522101) (Concept DOI: `10.5281/zenodo.24508472`)
- **PDF (5 pp.):** [`steerable-not-decodable-ace-cores.pdf`](./steerable-not-decodable-ace-cores.pdf) · **HTML:** [`steerable-not-decodable-ace-cores.html`](./steerable-not-decodable-ace-cores.html)
- **LaTeX Packages:** [`latex/steerable-not-decodable-ace-cores/`](./latex/steerable-not-decodable-ace-cores/) (`submission_tmlr_anon.pdf`, `icml2027_anon.pdf`, `preprint_arxiv.pdf`, `arxiv_bundle.tar.gz`)
- **Key v4.0 Results:**
  - **4 open-weight architectures (1.7B–7.3B):** `Qwen3-1.7B`, `SmolLM2-1.7B`, `Mistral-7B-v0.1`, `OLMo-2-1124-7B` across 16 `(model, domain)` cells and **2,688 strictly held-out cross-lexicon pairs** (100% single-token across all 4 tokenizers).
  - **Exact KL-Minimizing General-Corpus Tuned Lens (Belrose et al. 2023 objective, 3,600 scientific tokens):** Scores only **`0.6384` AUC** vs. ACE **`0.7653`** (**gap `+0.1269` AUC, Wilcoxon $p = 0.0107$**), widening the gap beyond Ridge General Tuned Lens (`0.6933`, $p = 0.0370$) and Domain Tuned Lens (`0.7080`, $p = 0.0467$), with mean $|\cos(q_{\ell^*}, \Delta W_U)| = 0.140$.
  - **Flip-1D necessity:** ACE (`+0.235`) outperforms Random (`-0.001`, $p = 1.53 \times 10^{-5}$), CAA (`+0.170`, $p = 1.53 \times 10^{-4}$), and L2 Probes (`+0.179`, $p = 3.82 \times 10^{-3}$).
  - **Keep-1D sufficiency:** Outperforms the full unperturbed model in **16 / 16 cells** (exact sign-test $p = 1.53 \times 10^{-5}$, mean `0.799` vs. `0.712` clean, `+0.087` rescue).
  - **Sustained Per-Step Multi-Token Steering:** Applying Flip-1D at every decode step jumps causal reversal **8×** on `Qwen3-1.7B physical_scale` (`5.6% → 44.4%`, fluent NLL `1.04 → 1.06`) and `0.0% → 20.0%` on `SmolLM2-1.7B hazard_severity` (NLL `1.50 → 1.48`) vs. **`0.0%` sustained random**, directly proving KV-cache re-attention (while `SmolLM2 physical_scale` stays `0.0%` due to weak `17.9%` clean accuracy).
  - **Real Natural-Language Non-Comparative Scientific Abstracts (`n = 60` held-out papers, unseen cross-prompt):** On `Qwen3-1.7B`, ACE 1D readout achieves **`0.979` AUC** vs. KL-Tuned Lens **`0.546`** & Logit Lens **`0.546`** at $|\cos| = 0.017$, FlipDrop ACE `+0.167` > CAA `+0.107` > Probe `+0.052` > Rand `+0.012`, and **Keep-1D rescues unperturbed accuracy from `0.548 → 0.922` (`+0.374` AUC)**.
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_steering_lens.py --output data/generated_papers/raw_steering_lens_v4.json
  ```

---

## 3. Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders (v4.0)
- **Zenodo v4.0 DOI:** [`10.5281/zenodo.24522712`](https://zenodo.org/records/24522712) (Concept DOI: `10.5281/zenodo.24508484`)
- **PDF (7 pp.):** [`probmoe-batchtopk-sae.pdf`](./probmoe-batchtopk-sae.pdf) · **HTML:** [`probmoe-batchtopk-sae.html`](./probmoe-batchtopk-sae.html)
- **LaTeX Packages:** [`latex/probmoe-batchtopk-sae/`](./latex/probmoe-batchtopk-sae/) (`submission_tmlr_anon.pdf`, `icml2027_anon.pdf`, `preprint_arxiv.pdf`, `arxiv_bundle.tar.gz`)
- **Key v4.0 Results:**
  - **9 L0-matched SAE methods** at `4×` (`8,192`), `8×` (`16,384`), and **`16×` (`32,768`)** overcompleteness across `Qwen3-1.7B L13` and `SmolLM2-1.7B L12` (**336,474 leak-free paper-disjoint tokens** across 1,650 papers, 1,200 Adam steps, 3 seeds).
  - **Live Transformer Causal Splice (`k = 32`):** On `SmolLM2-1.7B L12`, **`MarginProb+AuxK` (`99.49%` CE)** and **`MarginProb-BatchTopK` (`99.43%`, lowest Logit KL `0.2040`)** rank **#1 and #2 of all 9 methods**; on `Qwen3-1.7B L13`, `MarginProb-BatchTopK` recovers **`98.57%` CE (`KL = 0.2910`)**, outperforming `BatchTopK+AuxK` (`98.55%`), `TopK+AuxK` (`98.33%`), and ungated v1 (`97.71%`).
  - **8× & 16× Production Width Scaling (`Qwen3-1.7B L13`, `k = 32`, 3 seeds):** At `16×` (`d_sae = 32,768`), `MarginProb-BatchTopK` (`FVU = 0.2581, dead = 75.82%`) and `MarginProb+AuxK` (`0.2573, 75.48%`) beat both `BatchTopK-STE` (`0.2737, 88.11%`) and `BatchTopK+AuxK` (`0.2594, 77.49%`) on BOTH FVU and dead-latent rate simultaneously.
  - **SAEBench Sparse Probing & Candid `Qwen3.8-27B` AutoInterp Trade-Off:** Blind AutoInterp F1 (`n = 12` latents/method) scores `BatchTopK-STE` `48.8%` (`BalAcc 68.1%`), `BatchTopK+AuxK` `41.4%` (`63.9%`), `TopK+AuxK` `24.5%` (`57.6%`), and `MarginProb-BatchTopK` `23.3%` (`58.3%`), transparently documenting the capacity–monosemanticity trade-off between broader active dictionary utilization and top-variance single-latent concentration.
- **Reproduce:**
  ```bash
  python3 scripts/research_engine/experiments/exp_probmoe_sae.py --output data/generated_papers/raw_probmoe_sae_v4.json
  ```
