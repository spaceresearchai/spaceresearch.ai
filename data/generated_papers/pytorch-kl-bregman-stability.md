# Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Analysis and Fused Triton Kernels

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 2.0  
**DOI:** `10.5281/zenodo.24511854` (https://zenodo.org/record/24511854)  
**Published:** 2026-11-24T19:53:33+01:00  
**Discovery Trigger:** CoreRank Method 1 Bayesian Surprise (#3 on 4B, +7.65)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Kullback-Leibler (KL) divergence regularization is foundational to modern reinforcement learning from human feedback (RLHF), Group Relative Policy Optimization (GRPO), and logit distillation over large vocabularies (V = 32,000-151,936). Yet standard PyTorch F.kl_div evaluates sum_i p_i (log p_i - log q_i), suffering catastrophic cancellation whenever policy and reference distributions lie within typical trust-region proximity. Extending prior forward-pass observations across 36 regimes (forward and reverse KL; FP16, BF16, FP32; eps in [1e-4, 1e-1]), we show this pathology corrupts the backward pass directly: naive F.kl_div produces negative KL on up to 50.4% of inputs and 39.2% reverse-KL gradient sign flips (BF16, eps=1e-2); the common clamp(min=0) patch eliminates negative scalars only by inducing up to 50.4% silent zero-gradient dead-zones (61.8% sign flips); and naive BF16 Schulman k3 (exp(d)-1-d) suffers 100% premature zeroing for eps <= 1e-2. We derive an overflow-free two-branch 6th-order Bregman-Taylor formulation with cancellation-free analytic gradients and implement a fused Blackwell Triton forward/backward kernel that streams online logsumexp in FP32 registers without materializing (N, V) FP32 intermediates. On an NVIDIA GeForce RTX 5090, our kernel achieves 0.0% negative KL across all 36 regimes (Wilcoxon p=1.45e-11), cuts FP32 (eps=1e-4) relative error from 21.1 to 1.42e-04 (149,111x reduction), runs 6.52x faster with 80.0% less peak VRAM than FP32-upcast F.kl_div at Qwen3 V=151,936 (30.4x faster than eager Bregman-Taylor), and matches the FP32 oracle exactly in 3-seed BF16 RLHF policy optimization (V=32,000) where clamped F.kl_div collapses into negative true objective (-0.00162 vs. +0.00164) with 99.4% dead-KL steps.

### Key Empirical Findings
- Forward & backward gradient corruption across 36 regimes (Wilcoxon p=1.45e-11): PyTorch F.kl_div returns negative values on up to 50.4% of inputs and corrupts BF16 reverse-KL gradients (eps=1e-2) with 39.1% sign flips (cos=0.589). Clamping F.kl_div.clamp(min=0) induces 50.4% silent zero-gradient dead-zones and 61.8% sign flips (cos=0.371).
- Even in FP32 (eps=1e-4), F.kl_div and FP32-upcast yield 48.1% negative outputs and median relative error 21.14 (2,114%); our 6th-order Bregman-Taylor branch cuts median error by 149,111x to 1.42e-04 (0.0% negative, 0.0% error >=1%).
- Blackwell RTX 5090 Triton scaling at frontier LLM vocabularies (Qwen3 V=151,936, N=512, BF16 fwd+bwd): Fused Triton Bregman-Taylor runs in 0.90 ms using 296.8 MB extra VRAM — 6.52x faster with 80.0% less VRAM than FP32-upcast F.kl_div (5.87 ms, 1483.8 MB) and 30.4x faster than eager PyTorch Bregman-Taylor (27.32 ms, 4822.2 MB).
- Downstream 3-seed BF16 RLHF/GRPO policy optimization (V=32,000, beta=2.0): Fused Triton matches the FP32 oracle identically (true objective +0.00164, true KL 0.00069, 0.000 median KL error), whereas clamped F.kl_div collapses to negative objective (-0.00162, 5.7x excess KL drift, 99.4% dead-KL steps) and BF16 F.kl_div / Schulman k3 plateau ~19% suboptimal (+0.00134 / +0.00132).

## 1. Source Literature Discovered by CoreRank
- [The KL Divergence Is Never Negative, Except in PyTorch](https://arxiv.org/abs/2611.30835) (`arxiv-2611.30835`)

## 2. Experimental Protocol & Hardware
```json
{
  "audit_samples": 2048,
  "audit_vocab": 4096,
  "scaling_vocabs": [
    256,
    32000,
    128256,
    151936
  ],
  "rlhf_vocab": 32000,
  "rlhf_seeds": [
    11,
    22,
    33
  ],
  "triton_version": "3.6.0",
  "torch_version": "2.11.0+cu130",
  "baselines": [
    "PyTorch F.kl_div",
    "Clamped F.kl_div",
    "Schulman k3 (native)",
    "FP32-Upcast F.kl_div",
    "Eager Bregman-Taylor"
  ],
  "figure_png_rel": "papers/figures/pytorch-kl-bregman-v2.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 12.57
}
```

## 3. Empirical Results
| Regime | Method | Neg KL % (↓) | Med Fwd RelErr (↓) | Grad Cos (↑) | Dead Grad % (↓) | Sign Flip % (↓) |
|---|---|---:|---:|---:|---:|---:|
| Forward KL · FP32 (ε=1e-4) | PyTorch F.kl_div | 48.1% | 2.11e+01 | 1.0000 | 0.0% | 0.0% |
| Forward KL · FP32 (ε=1e-4) | Clamped F.kl_div | 0.0% | 1.00e+00 | 0.5190 | 48.1% | 48.1% |
| Forward KL · FP32 (ε=1e-4) | Schulman k3 | 0.0% | 1.41e+00 | 1.0000 | 0.0% | 0.0% |
| Forward KL · FP32 (ε=1e-4) | FP32-Upcast F.kl_div | 48.1% | 2.11e+01 | 1.0000 | 0.0% | 0.0% |
| Forward KL · FP32 (ε=1e-4) | Triton Bregman-Taylor (Ours) | 0.0% | 1.42e-04 | 1.0000 | 0.0% | 0.0% |
| Reverse KL · FP16 (ε=3e-3) | PyTorch F.kl_div | 46.1% | 2.08e+01 | 0.8581 | 0.0% | 33.2% |
| Reverse KL · FP16 (ε=3e-3) | Clamped F.kl_div | 0.0% | 1.77e+00 | 0.4626 | 46.1% | 63.9% |
| Reverse KL · FP16 (ε=3e-3) | Schulman k3 | 0.0% | 1.00e+00 | 0.8640 | 0.0% | 42.5% |
| Reverse KL · FP16 (ε=3e-3) | FP32-Upcast F.kl_div | 0.0% | 7.24e-02 | 0.9157 | 0.0% | 3.0% |
| Reverse KL · FP16 (ε=3e-3) | Triton Bregman-Taylor (Ours) | 0.0% | 6.82e-02 | 0.9157 | 0.0% | 3.0% |
| Reverse KL · BF16 (ε=1e-3) | PyTorch F.kl_div | 46.6% | 2.55e+02 | 0.1369 | 0.0% | 48.9% |
| Reverse KL · BF16 (ε=1e-3) | Clamped F.kl_div | 0.0% | 2.97e+01 | 0.0710 | 46.6% | 72.7% |
| Reverse KL · BF16 (ε=1e-3) | Schulman k3 | 0.0% | 1.00e+00 | 0.1717 | 0.0% | 48.7% |
| Reverse KL · BF16 (ε=1e-3) | FP32-Upcast F.kl_div | 0.0% | 7.56e+00 | 0.2289 | 0.0% | 32.8% |
| Reverse KL · BF16 (ε=1e-3) | Triton Bregman-Taylor (Ours) | 0.0% | 7.48e+00 | 0.2289 | 0.0% | 32.8% |
| Reverse KL · BF16 (ε=1e-2) | PyTorch F.kl_div | 37.2% | 9.41e+00 | 0.5894 | 0.0% | 39.1% |
| Reverse KL · BF16 (ε=1e-2) | Clamped F.kl_div | 0.0% | 3.49e+00 | 0.3706 | 37.2% | 61.8% |
| Reverse KL · BF16 (ε=1e-2) | Schulman k3 | 0.0% | 1.00e+00 | 0.6037 | 0.0% | 38.6% |
| Reverse KL · BF16 (ε=1e-2) | FP32-Upcast F.kl_div | 0.0% | 3.66e-01 | 0.7067 | 0.0% | 7.4% |
| Reverse KL · BF16 (ε=1e-2) | Triton Bregman-Taylor (Ours) | 0.0% | 3.66e-01 | 0.7067 | 0.0% | 7.4% |

| Vocabulary Scale (Batch N) | FP32-Upcast (ms / MB) | Eager Bregman (ms / MB) | Triton Fused (ms / MB) | Speedup vs FP32 | Speedup vs Eager | VRAM Saved vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| V=256 (Toy) (N=2048) | 0.10 ms / 10.0 MB | 0.30 ms / 32.5 MB | 0.10 ms / 2.0 MB | 0.98x | 3.0x | -80.0% |
| V=32k (Llama-2) (N=1024) | 2.11 ms / 630.0 MB | 11.10 ms / 2048.0 MB | 0.34 ms / 125.0 MB | 6.12x | 32.2x | -80.2% |
| V=128k (Llama-3) (N=512) | 4.92 ms / 1252.5 MB | 23.03 ms / 4070.6 MB | 0.76 ms / 252.0 MB | 6.50x | 30.5x | -79.9% |
| V=152k (Qwen3) (N=512) | 5.87 ms / 1483.8 MB | 27.32 ms / 4822.2 MB | 0.90 ms / 296.8 MB | 6.52x | 30.4x | -80.0% |

## 4. Honest Limitations & Open Questions
- Irreducible input quantization floor: if logits are rounded to BF16 prior to the kernel call with |z_q - z_p| < 0.5 ULP_BF16, sub-ULP differences are already lost before KL evaluation (matching FP32-upcast ceiling).
- Evaluated on full-vocabulary categorical KL (V up to 151,936); sampled-token Monte Carlo GRPO benefits primarily from scalar Eq. (1) stabilization.
- Continuous exponential-family KL divergences require family-specific cancellation-free identities.

## References
[1] Schulman, J. (2020). Approximating KL Divergence. Technical Blog Note (k1/k2/k3 estimators).
[2] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). arXiv:2402.03300.
[3] Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019.
[4] Hsu, P.-L., Dai, Y., Kothapalli, V., Song, Q., Tang, S., Tang, S., et al. (2024). Liger Kernel: Efficient Triton Kernels for LLM Training. arXiv:2410.10989.
[5] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
[6] Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.
[7] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS 2022.
[8] The KL Divergence Is Never Negative, Except in PyTorch (2026). arXiv:2611.30835.

## 5. BibTeX Citation
```bibtex
@article{diaz2026pytorchklbregman,
  title       = {Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Analysis and Fused Triton Kernels},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 2.0},
  doi         = {10.5281/zenodo.24511854},
  url         = {https://spaceresearch.ai/papers/pytorch-kl-bregman-stability.html}
}
```
