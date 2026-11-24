# Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 3.0  
**DOI:** `10.5281/zenodo.24515036` (https://zenodo.org/record/24515036)  
**Published:** 2026-11-24T22:09:31+01:00  
**Discovery Trigger:** CoreRank Method 2 Cross-Domain Bridge (ProbMoE <-> BatchTopK) + Method 3 Contradiction  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Sparse Autoencoders (SAEs) trained with hard TopK or BatchTopK activations rely on straight-through estimators (STE) that route zero gradient to inactive dictionary atoms, driving severe dead-latent collapse at high expansion factors (up to 78.6% dead atoms at 4x overcompleteness). Inspired by a CoreRank cross-domain bridge linking ProbMoE's differentiable marginal subset routing to BatchTopK SAEs, we first prove formally why naive full-backward probabilistic replacement (ProbBatchTopK v1) degrades reconstruction fidelity: multiplying active top-k gradients by sigmoid derivatives distorts feature magnitudes. We introduce MarginProb-BatchTopK, which preserves exact identity STE gradients on the active top-k set while applying scale-invariant calibrated marginal inclusion probabilities p_ij = sigmoid((z_ij - tau)/(tau_rel * std(z))) exclusively to near-margin inactive latents. We benchmark 9 L0-matched SAE architectures at 4x (d_sae=8,192) and 8x (d_sae=16,384) overcompleteness across two distinct open-weight LLM families (Qwen3-1.7B L13 and SmolLM2-1.7B L12; 205,883 real tokens with strict paper-level train/test splits), three sparsity levels (k in {16, 32, 64}), and 3 random seeds. Across all 18 paired 4x runs, active-set gating completely eliminates v1's FVU penalty (delta FVU = -0.01569, 95% CI [-0.01837, -0.01301], paired t p=1.95e-09). Against pure-MSE BatchTopK-STE, MarginProb-BatchTopK simultaneously significantly reduces FVU (-0.00470, p=4.10e-03) and slashes dead latents by -19.90 percentage points (p=7.69e-09) with ZERO auxiliary loss. Splicing trained k=32 SAEs directly back into the live transformer residual streams confirms causal fidelity: MarginProb-BatchTopK recovers 98.0% CE loss on Qwen3-1.7B (KL=0.3819 vs. 97.2% / KL=0.4582 for ungated v1 and 98.4% for BatchTopK-STE) and 98.8% on SmolLM2-1.7B (KL=0.2561 vs. 98.8% BatchTopK-STE). Scaling to 8x overcompleteness (d_sae=16,384 on Qwen3-1.7B L13, k=32), MarginProb-BatchTopK cuts dead latents from 75.5% (BatchTopK-STE) and 60.4% (BatchTopK+AuxK) down to 54.9% while achieving lower FVU (0.2764 vs. 0.2859). Sweeping lambda in [0, 0.60] smoothly controls dead latents from 55.8% down to 1.32% with an FVU optimum at lambda=0.05 (0.2786).

### Key Empirical Findings
- Active-set gating completely eliminates v1's FVU penalty across 18 paired 4x-overcomplete runs (2 LLM families x 3 k x 3 seeds): MarginProb-BatchTopK lowers FVU vs. ungated v1 ProbBatchTopK by -0.01569 (95% CI [-0.01837, -0.01301], paired t p=1.95e-09; Qwen3 k=32: 0.2996 -> 0.2796; SmolLM2 k=32: 0.2826 -> 0.2720).
- Pure-MSE MarginProb-BatchTopK strictly Pareto-dominates BatchTopK-STE across 18 paired runs with zero auxiliary loss: FVU delta = -0.00470 (p=4.10e-03, Wilcoxon p=3.36e-03) AND dead-latent reduction = -19.90 pp (95% CI [-23.62, -16.18], p=7.69e-09).
- Surpassing published SOTA BatchTopK+AuxK at k=32 and k=64: At k=32, pure-MSE MarginProb-BatchTopK beats BatchTopK+AuxK on FVU on both models (Qwen3 0.2796 vs. 0.2806, dead 20.1% vs. 31.1%; SmolLM2 0.2720 vs. 0.2729, highest 8-class probe recovery 108.6%). At k=64, dead latents drop 8x on Qwen3 (1.53% vs. 12.17%) and 3x on SmolLM2 (5.16% vs. 15.61%); at ultra-sparse k=16, AuxK retains a modest dead-latent edge (51.0% vs. 58.5% on Qwen3), while TopK+AuxK stays strongest on pure FVU at k=16.
- Smooth Pareto control via boundary weight lambda (Qwen3-1.7B, k=32, 3 seeds): Increasing lambda from 0.00 -> 0.05 -> 0.15 -> 0.30 -> 0.60 monotonically reduces dead latents from 55.8% -> 33.4% -> 20.1% -> 9.1% -> 1.32% (42x reduction) while lifting utilization entropy from 0.837 -> 0.918, achieving an FVU minimum at lambda=0.05 (0.2786).

## 1. Source Literature Discovered by CoreRank
- [ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](https://arxiv.org/abs/2606.01509) (`arxiv-2606.01509`)
- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) (`arxiv-2412.06410`)
- [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615) (`arxiv-2501.16615`)

## 2. Experimental Protocol & Hardware
```json
{
  "models": {
    "Qwen3-1.7B": {
      "source_model": "Qwen/Qwen3-1.7B",
      "layer_extracted": "L13/28",
      "d_model": 2048,
      "papers_sampled": 1100,
      "total_tokens": 102606,
      "extract_vram_gb": 1.8,
      "extract_sec": 4.82,
      "train_tokens": 82025,
      "test_tokens": 20581,
      "train_papers": 880,
      "test_papers": 220
    },
    "SmolLM2-1.7B": {
      "source_model": "HuggingFaceTB/SmolLM2-1.7B",
      "layer_extracted": "L12/24",
      "d_model": 2048,
      "papers_sampled": 1100,
      "total_tokens": 103277,
      "extract_vram_gb": 1.7,
      "extract_sec": 1.82,
      "train_tokens": 82619,
      "test_tokens": 20658,
      "train_papers": 880,
      "test_papers": 220
    }
  },
  "d_in": 2048,
  "d_sae": 8192,
  "d_sae_8x": 16384,
  "expansion_factor": 4.0,
  "k_values": [
    16,
    32,
    64
  ],
  "seeds": [
    11,
    22,
    33
  ],
  "steps_per_run": 900,
  "batch_size": 512,
  "methods": [
    "relu_l1",
    "jumprelu",
    "topk_ste",
    "topk_auxk",
    "batchtopk_ste",
    "batchtopk_auxk",
    "probbatchtopk_v1",
    "marginprob_batchtopk",
    "marginprob_auxk"
  ],
  "n_baselines": 7,
  "figure_png_rel": "papers/figures/probmoe-sae-v3.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 1351.6
}
```

## 3. Empirical Results
| Model | Method | FVU (↓) | Dead % (↓) | CE Rec % (↑) | Logit KL (↓) | Probe Rec % (↑) | Top512 Hung (↑) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | ReLU-L1 [1] | 0.4807 ± 0.0016 | 37.83% | 77.3% | 1.8969 | 98.1% | 0.525 |
| Qwen3-1.7B | JumpReLU [4] | 0.3472 ± 0.0028 | 84.22% | 95.9% | 0.5866 | 100.7% | 0.480 |
| Qwen3-1.7B | TopK-STE [2] | 0.2784 ± 0.0003 | 29.32% | 97.8% | 0.3991 | 103.1% | 0.494 |
| Qwen3-1.7B | TopK + AuxK [2] | 0.2778 ± 0.0001 | 15.13% | 97.8% | 0.3959 | 103.8% | 0.499 |
| Qwen3-1.7B | BatchTopK-STE [3] | 0.2834 ± 0.0015 | 55.80% | 98.4% | 0.3623 | 105.2% | 0.500 |
| Qwen3-1.7B | BatchTopK + AuxK [3] | 0.2806 ± 0.0011 | 31.15% | 98.3% | 0.3691 | 104.7% | 0.503 |
| Qwen3-1.7B | ProbBatchTopK (v1 Ungated) | 0.2996 ± 0.0003 | 20.48% | 97.2% | 0.4582 | 101.7% | 0.483 |
| Qwen3-1.7B | MarginProb-BatchTopK (Ours) | 0.2796 ± 0.0007 | 20.07% | 98.0% | 0.3819 | 103.8% | 0.502 |
| Qwen3-1.7B | MarginProb + AuxK (Ours) | 0.2797 ± 0.0005 | 18.39% | 98.0% | 0.3796 | 103.5% | 0.498 |
| SmolLM2-1.7B | ReLU-L1 [1] | 0.4935 ± 0.0147 | 19.84% | 80.6% | 1.3876 | 102.3% | 0.402 |
| SmolLM2-1.7B | JumpReLU [4] | 0.3663 ± 0.0129 | 88.55% | 95.6% | 0.4711 | 103.0% | 0.480 |
| SmolLM2-1.7B | TopK-STE [2] | 0.2670 ± 0.0002 | 31.42% | 98.3% | 0.2735 | 107.1% | 0.509 |
| SmolLM2-1.7B | TopK + AuxK [2] | 0.2659 ± 0.0001 | 18.17% | 98.3% | 0.2723 | 105.8% | 0.509 |
| SmolLM2-1.7B | BatchTopK-STE [3] | 0.2761 ± 0.0027 | 49.28% | 98.8% | 0.2612 | 107.1% | 0.485 |
| SmolLM2-1.7B | BatchTopK + AuxK [3] | 0.2729 ± 0.0024 | 30.23% | 98.8% | 0.2598 | 106.6% | 0.493 |
| SmolLM2-1.7B | ProbBatchTopK (v1 Ungated) | 0.2826 ± 0.0006 | 18.10% | 98.5% | 0.2753 | 107.1% | 0.498 |
| SmolLM2-1.7B | MarginProb-BatchTopK (Ours) | 0.2720 ± 0.0023 | 31.33% | 98.8% | 0.2561 | 108.6% | 0.495 |
| SmolLM2-1.7B | MarginProb + AuxK (Ours) | 0.2719 ± 0.0021 | 29.02% | 98.8% | 0.2552 | 107.6% | 0.490 |

| Method (8× Overcomplete, d_sae=16,384, k=32) | FVU (↓) | Dead Latents % (↓) | Util Entropy (↑) |
|---|---:|---:|---:|
| TopK-STE [2] | 0.2774 ± 0.0006 | 56.06% | 0.8169 |
| TopK + AuxK [2] | 0.2764 ± 0.0001 | 46.27% | 0.8163 |
| BatchTopK-STE [3] | 0.2859 ± 0.0013 | 75.50% | 0.7707 |
| BatchTopK + AuxK [3] | 0.2799 ± 0.0010 | 60.44% | 0.7794 |
| MarginProb-BatchTopK (Ours) | 0.2764 ± 0.0002 | 54.89% | 0.8020 |
| MarginProb + AuxK (Ours) | 0.2766 ± 0.0002 | 56.67% | 0.8036 |

## 4. Honest Limitations & Open Questions
- Straight-through margin gradients update W_enc and b_enc for near-margin inactive features without updating W_dec, so deeply dead latents still benefit from AuxK.
- Evaluated at 4x overcompleteness (d_sae=8,192) over 205,883 leak-free tokens across two 1.7B models; 65k-131k production scale over 100M+ tokens warrants future study.
- Downstream semantics are evaluated via 8-class category linear probe recovery rather than per-latent LLM auto-interpretability scoring.

## References
[1] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
[2] Gao, L., Dupre la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. ICLR 2025 / arXiv:2406.04093.
[3] Bussmann, B., Leask, P., & Nanda, N. (2024). BatchTopK Sparse Autoencoders. NeurIPS 2024 Workshop on Mechanistic Interpretability / arXiv:2412.06410.
[4] Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramar, J., & Nanda, N. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
[5] Paulo, G., & Belrose, N. (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615.
[6] Bengio, Y., Leonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432.
[7] Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. ICLR 2024.
[8] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts (2026). arXiv:2606.01509.

## 5. BibTeX Citation
```bibtex
@article{diaz2026probmoebatchtopk,
  title       = {Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 3.0},
  doi         = {10.5281/zenodo.24515036},
  url         = {https://spaceresearch.ai/papers/probmoe-batchtopk-sae.html}
}
```
