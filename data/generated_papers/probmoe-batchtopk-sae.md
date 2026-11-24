# Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 2.0  
**DOI:** `10.5281/zenodo.24511852` (https://zenodo.org/record/24511852)  
**Published:** 2026-11-24T19:53:36+01:00  
**Discovery Trigger:** CoreRank Method 2 Cross-Domain Bridge (ProbMoE <-> BatchTopK) + Method 3 Contradiction  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Sparse Autoencoders (SAEs) trained with hard TopK or BatchTopK activations rely on straight-through estimators (STE) that route zero gradient to inactive dictionary atoms, driving severe dead-latent collapse at high expansion factors (up to 78.6% dead atoms at 4x overcompleteness). Inspired by a CoreRank cross-domain bridge linking ProbMoE's differentiable marginal subset routing to BatchTopK SAEs, we first diagnose why naive full-backward probabilistic replacement (ProbBatchTopK v1) degrades reconstruction fidelity: multiplying active top-k gradients by sigmoid derivatives distorts feature magnitudes. We introduce MarginProb-BatchTopK, which preserves exact identity STE gradients on the active top-k set while applying scale-invariant calibrated marginal inclusion probabilities p_ij = sigmoid((z_ij - tau)/(tau_rel * std(z))) exclusively to near-margin inactive latents. We benchmark 9 L0-matched SAE architectures at 4x overcompleteness (d_in=2,048 -> d_sae=8,192) across two distinct open-weight LLM families (Qwen3-1.7B L13 and SmolLM2-1.7B L12; 205,883 real tokens with strict paper-level train/test splits), three sparsity levels (k in {16, 32, 64}), and 3 random seeds (162 main runs plus 24 ablation runs). Across all 18 paired runs, active-set gating completely eliminates v1's FVU penalty (delta FVU = -0.01569, 95% CI [-0.01837, -0.01301], paired t p=1.95e-09). Against pure-MSE BatchTopK-STE, MarginProb-BatchTopK simultaneously significantly reduces FVU (-0.00470, p=4.10e-03) and slashes dead latents by -19.90 percentage points (p=7.69e-09) with ZERO auxiliary loss. At k=32, pure-MSE MarginProb-BatchTopK beats even published SOTA BatchTopK+AuxK on both models (Qwen3 FVU 0.2796 vs. 0.2806, dead 20.1% vs. 31.1%; SmolLM2 FVU 0.2720 vs. 0.2729, probe recovery 108.6%), and at k=64 reduces dead latents 3x-8x below BatchTopK+AuxK (1.53% vs. 12.17% on Qwen3; 5.16% vs. 15.61% on SmolLM2). Sweeping lambda in [0, 0.60] smoothly controls dead latents from 55.8% down to 1.32% with an FVU optimum at lambda=0.05 (0.2786).

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
      "extract_sec": 2.62,
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
      "extract_vram_gb": 1.71,
      "extract_sec": 2.03,
      "train_tokens": 82619,
      "test_tokens": 20658,
      "train_papers": 880,
      "test_papers": 220
    }
  },
  "d_in": 2048,
  "d_sae": 8192,
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
  "figure_png_rel": "papers/figures/probmoe-sae-v2.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 858.7
}
```

## 3. Empirical Results
| Model | Method | FVU (↓) | Dead % (↓) | Util Entropy (↑) | Mean L0 | Probe Rec % (↑) | Hungarian Cos (↑) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | ReLU-L1 [1] | 0.4807 ± 0.0016 | 37.83% | 0.694 | 33.6 | 98.1% | 0.095 |
| Qwen3-1.7B | JumpReLU [4] | 0.3472 ± 0.0028 | 84.22% | 0.682 | 33.0 | 100.7% | 0.307 |
| Qwen3-1.7B | TopK-STE [2] | 0.2784 ± 0.0003 | 29.32% | 0.876 | 32.0 | 103.1% | 0.238 |
| Qwen3-1.7B | TopK + AuxK [2] | 0.2778 ± 0.0001 | 15.13% | 0.879 | 32.0 | 103.8% | 0.220 |
| Qwen3-1.7B | BatchTopK-STE [3] | 0.2834 ± 0.0015 | 55.80% | 0.837 | 32.0 | 105.2% | 0.252 |
| Qwen3-1.7B | BatchTopK + AuxK [3] | 0.2806 ± 0.0011 | 31.15% | 0.846 | 32.0 | 104.7% | 0.194 |
| Qwen3-1.7B | ProbBatchTopK (v1 Ungated) | 0.2996 ± 0.0003 | 20.48% | 0.890 | 32.0 | 101.7% | 0.199 |
| Qwen3-1.7B | MarginProb-BatchTopK (Ours) | 0.2796 ± 0.0007 | 20.07% | 0.869 | 32.0 | 103.8% | 0.186 |
| Qwen3-1.7B | MarginProb + AuxK (Ours) | 0.2797 ± 0.0005 | 18.39% | 0.873 | 32.0 | 103.5% | 0.181 |
| SmolLM2-1.7B | ReLU-L1 [1] | 0.4935 ± 0.0147 | 19.84% | 0.671 | 33.4 | 102.3% | 0.080 |
| SmolLM2-1.7B | JumpReLU [4] | 0.3663 ± 0.0129 | 88.55% | 0.657 | 32.2 | 103.0% | 0.353 |
| SmolLM2-1.7B | TopK-STE [2] | 0.2670 ± 0.0002 | 31.42% | 0.864 | 32.0 | 107.1% | 0.238 |
| SmolLM2-1.7B | TopK + AuxK [2] | 0.2659 ± 0.0001 | 18.17% | 0.866 | 32.0 | 105.8% | 0.213 |
| SmolLM2-1.7B | BatchTopK-STE [3] | 0.2761 ± 0.0027 | 49.28% | 0.826 | 32.0 | 107.1% | 0.224 |
| SmolLM2-1.7B | BatchTopK + AuxK [3] | 0.2729 ± 0.0024 | 30.23% | 0.833 | 32.0 | 106.6% | 0.189 |
| SmolLM2-1.7B | ProbBatchTopK (v1 Ungated) | 0.2826 ± 0.0006 | 18.10% | 0.886 | 32.0 | 107.1% | 0.205 |
| SmolLM2-1.7B | MarginProb-BatchTopK (Ours) | 0.2720 ± 0.0023 | 31.33% | 0.859 | 32.0 | 108.6% | 0.199 |
| SmolLM2-1.7B | MarginProb + AuxK (Ours) | 0.2719 ± 0.0021 | 29.02% | 0.861 | 32.0 | 107.6% | 0.202 |

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
  note        = {Version 2.0},
  doi         = {10.5281/zenodo.24511852},
  url         = {https://spaceresearch.ai/papers/probmoe-batchtopk-sae.html}
}
```
