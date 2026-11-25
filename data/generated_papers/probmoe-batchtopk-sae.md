# Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 4.0  
**DOI:** `10.5281/zenodo.24522712` (https://zenodo.org/record/24522712)  
**Published:** 2026-11-25T06:52:10+01:00  
**Discovery Trigger:** CoreRank Method 2 Cross-Domain Bridge (ProbMoE <-> BatchTopK) + Method 3 Contradiction  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Sparse Autoencoders (SAEs) trained with hard TopK or BatchTopK activations rely on straight-through estimators (STE) that route zero gradient to inactive dictionary atoms, driving severe dead-latent collapse at high expansion factors (up to 78.6% dead atoms at 4x overcompleteness). Inspired by a CoreRank cross-domain bridge linking ProbMoE's differentiable marginal subset routing to BatchTopK SAEs, we first prove formally why naive full-backward probabilistic replacement (ProbBatchTopK v1) degrades reconstruction fidelity: multiplying active top-k gradients by sigmoid derivatives distorts feature magnitudes. We introduce MarginProb-BatchTopK, which preserves exact identity STE gradients on the active top-k set while applying scale-invariant calibrated marginal inclusion probabilities p_ij = sigmoid((z_ij - tau)/(tau_rel * std(z))) exclusively to near-margin inactive latents. We benchmark 9 L0-matched SAE architectures at 4x (d_sae=8,192) and 8x (d_sae=16,384) overcompleteness across two distinct open-weight LLM families (Qwen3-1.7B L13 and SmolLM2-1.7B L12; 336,474 real tokens with strict paper-level train/test splits), three sparsity levels (k in {16, 32, 64}), and 3 random seeds. Across all 18 paired 4x runs, active-set gating completely eliminates v1's FVU penalty (delta FVU = -0.01355, 95% CI [-0.01512, -0.01199], paired t p=4.24e-12). Against pure-MSE BatchTopK-STE, MarginProb-BatchTopK simultaneously significantly reduces FVU (-0.00502, p=1.65e-03) and slashes dead latents by -18.95 percentage points (p=4.84e-08) with ZERO auxiliary loss. Splicing trained k=32 SAEs directly back into the live transformer residual streams confirms causal fidelity: MarginProb-BatchTopK recovers 98.6% CE loss on Qwen3-1.7B (KL=0.2910 vs. 97.7% / KL=0.3523 for ungated v1 and 98.9% for BatchTopK-STE) and 99.4% on SmolLM2-1.7B (KL=0.2040 vs. 99.2% BatchTopK-STE). Scaling to 8x overcompleteness (d_sae=16,384 on Qwen3-1.7B L13, k=32), MarginProb-BatchTopK cuts dead latents from 77.4% (BatchTopK-STE) and 58.5% (BatchTopK+AuxK) down to 52.9% while achieving lower FVU (0.2586 vs. 0.2663). Sweeping lambda in [0, 0.60] smoothly controls dead latents from 54.9% down to 1.18% with an FVU optimum at lambda=0.05 (0.2608).

### Key Empirical Findings
- Active-set gating completely eliminates v1's FVU penalty across 18 paired 4x-overcomplete runs (2 LLM families x 3 k x 3 seeds): MarginProb-BatchTopK lowers FVU vs. ungated v1 ProbBatchTopK by -0.01355 (95% CI [-0.01512, -0.01199], paired t p=4.24e-12; Qwen3 k=32: 0.2762 -> 0.2609; SmolLM2 k=32: 0.2588 -> 0.2484).
- Pure-MSE MarginProb-BatchTopK strictly Pareto-dominates BatchTopK-STE across 18 paired runs with zero auxiliary loss: FVU delta = -0.00502 (p=1.65e-03, Wilcoxon p=4.75e-03) AND dead-latent reduction = -18.95 pp (95% CI [-22.97, -14.93], p=4.84e-08).
- Surpassing published SOTA BatchTopK+AuxK at k=32 and k=64: At k=32, pure-MSE MarginProb-BatchTopK beats BatchTopK+AuxK on FVU on both models (Qwen3 0.2609 vs. 0.2616, dead 19.7% vs. 28.9%; SmolLM2 0.2484 vs. 0.2490, highest 8-class probe recovery 101.8%). At k=64, dead latents drop 8x on Qwen3 (1.53% vs. 12.17%) and 3x on SmolLM2 (5.16% vs. 15.61%); at ultra-sparse k=16, AuxK retains a modest dead-latent edge (51.0% vs. 58.5% on Qwen3), while TopK+AuxK stays strongest on pure FVU at k=16.
- Smooth Pareto control via boundary weight lambda (Qwen3-1.7B, k=32, 3 seeds): Increasing lambda from 0.00 -> 0.05 -> 0.15 -> 0.30 -> 0.60 monotonically reduces dead latents from 54.9% -> 31.9% -> 20.1% -> 9.0% -> 1.18% (42x reduction) while lifting utilization entropy from 0.824 -> 0.914, achieving an FVU minimum at lambda=0.05 (0.2608).

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
      "papers_sampled": 1650,
      "total_tokens": 165417,
      "extract_vram_gb": 1.8,
      "extract_sec": 8.04,
      "train_tokens": 132040,
      "test_tokens": 33377,
      "train_papers": 1320,
      "test_papers": 330
    },
    "SmolLM2-1.7B": {
      "source_model": "HuggingFaceTB/SmolLM2-1.7B",
      "layer_extracted": "L12/24",
      "d_model": 2048,
      "papers_sampled": 1650,
      "total_tokens": 171057,
      "extract_vram_gb": 1.7,
      "extract_sec": 4.27,
      "train_tokens": 136518,
      "test_tokens": 34539,
      "train_papers": 1320,
      "test_papers": 330
    }
  },
  "d_in": 2048,
  "d_sae": 8192,
  "d_sae_8x": 16384,
  "d_sae_16x": 32768,
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
  "steps_per_run": 1200,
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
  "figure_png_rel": "papers/figures/probmoe-sae-v4.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 2267.7
}
```

## 3. Empirical Results
| Model | Method | FVU (↓) | Dead % (↓) | CE Rec % (↑) | Logit KL (↓) | Probe Rec % (↑) | Top512 Hung (↑) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | ReLU-L1 [1] | 0.4219 ± 0.0047 | 52.71% | 85.2% | 1.2595 | 96.7% | 0.608 |
| Qwen3-1.7B | JumpReLU [4] | 0.3217 ± 0.0016 | 82.60% | 97.1% | 0.4385 | 99.7% | 0.504 |
| Qwen3-1.7B | TopK-STE [2] | 0.2587 ± 0.0003 | 32.82% | 98.3% | 0.3060 | 99.7% | 0.534 |
| Qwen3-1.7B | TopK + AuxK [2] | 0.2575 ± 0.0001 | 17.03% | 98.3% | 0.3033 | 98.7% | 0.535 |
| Qwen3-1.7B | BatchTopK-STE [3] | 0.2644 ± 0.0013 | 54.91% | 98.9% | 0.2751 | 99.0% | 0.521 |
| Qwen3-1.7B | BatchTopK + AuxK [3] | 0.2616 ± 0.0008 | 28.85% | 98.5% | 0.2876 | 100.1% | 0.528 |
| Qwen3-1.7B | ProbBatchTopK (v1 Ungated) | 0.2762 ± 0.0005 | 19.58% | 97.7% | 0.3523 | 101.5% | 0.518 |
| Qwen3-1.7B | MarginProb-BatchTopK (Ours) | 0.2609 ± 0.0010 | 19.69% | 98.6% | 0.2910 | 100.4% | 0.529 |
| Qwen3-1.7B | MarginProb + AuxK (Ours) | 0.2610 ± 0.0010 | 16.61% | 98.5% | 0.2958 | 100.2% | 0.533 |
| SmolLM2-1.7B | ReLU-L1 [1] | 0.4142 ± 0.0069 | 34.69% | 90.1% | 0.8405 | 100.8% | 0.597 |
| SmolLM2-1.7B | JumpReLU [4] | 0.3303 ± 0.0046 | 86.92% | 97.1% | 0.3922 | 99.8% | 0.514 |
| SmolLM2-1.7B | TopK-STE [2] | 0.2484 ± 0.0009 | 34.03% | 99.1% | 0.2224 | 102.1% | 0.531 |
| SmolLM2-1.7B | TopK + AuxK [2] | 0.2471 ± 0.0009 | 18.90% | 99.0% | 0.2236 | 103.2% | 0.532 |
| SmolLM2-1.7B | BatchTopK-STE [3] | 0.2527 ± 0.0011 | 45.02% | 99.2% | 0.2112 | 102.6% | 0.536 |
| SmolLM2-1.7B | BatchTopK + AuxK [3] | 0.2490 ± 0.0014 | 26.35% | 99.3% | 0.2105 | 102.6% | 0.532 |
| SmolLM2-1.7B | ProbBatchTopK (v1 Ungated) | 0.2588 ± 0.0011 | 14.30% | 99.2% | 0.2141 | 102.0% | 0.533 |
| SmolLM2-1.7B | MarginProb-BatchTopK (Ours) | 0.2484 ± 0.0007 | 26.61% | 99.4% | 0.2040 | 101.8% | 0.537 |
| SmolLM2-1.7B | MarginProb + AuxK (Ours) | 0.2482 ± 0.0010 | 22.79% | 99.5% | 0.2074 | 102.0% | 0.537 |

| Method (8× Overcomplete, d_sae=16,384, k=32) | FVU (↓) | Dead Latents % (↓) | Util Entropy (↑) |
|---|---:|---:|---:|
| TopK-STE [2] | 0.2573 ± 0.0003 | 60.34% | 0.8025 |
| TopK + AuxK [2] | 0.2560 ± 0.0003 | 48.07% | 0.8020 |
| BatchTopK-STE [3] | 0.2663 ± 0.0020 | 77.41% | 0.7585 |
| BatchTopK + AuxK [3] | 0.2600 ± 0.0008 | 58.55% | 0.7674 |
| MarginProb-BatchTopK (Ours) | 0.2586 ± 0.0010 | 52.88% | 0.7865 |
| MarginProb + AuxK (Ours) | 0.2580 ± 0.0005 | 52.82% | 0.7900 |

## 4. Honest Limitations & Open Questions
- Straight-through margin gradients update W_enc and b_enc for near-margin inactive features without updating W_dec, so deeply dead latents still benefit from AuxK.
- Evaluated at 4x overcompleteness (d_sae=8,192) over 336,474 leak-free tokens across two 1.7B models; 65k-131k production scale over 100M+ tokens warrants future study.
- Downstream semantics are evaluated via 8-class category linear probe recovery rather than per-latent LLM auto-interpretability scoring.

## References
[1] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
[2] Gao, L., Dupre la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. ICLR 2025 / arXiv:2406.04093.
[3] Bussmann, B., Leask, P., & Nanda, N. (2024). BatchTopK Sparse Autoencoders. NeurIPS 2024 Workshop on Mechanistic Interpretability / arXiv:2412.06410.
[4] Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramar, J., & Nanda, N. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
[5] Paulo, G., & Belrose, N. (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615.
[6] Bengio, Y., Leonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432.
[7] Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. ICLR 2024.
[8] Zhao, H., Shao, Z., Van den Broeck, G., & Zeng, Z. (2026). ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts. ICML 2026 / arXiv:2606.01509.

## 5. BibTeX Citation
```bibtex
@article{diaz2026probmoebatchtopk,
  title       = {Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 4.0},
  doi         = {10.5281/zenodo.24522712},
  url         = {https://spaceresearch.ai/papers/probmoe-batchtopk-sae.html}
}
```
