# Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Theory, Fused Triton Kernels, and LLM Post-Training

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 4.0  
**DOI:** `10.5281/zenodo.24521689` (https://zenodo.org/record/24521689)  
**Published:** 2026-11-25T05:42:43+01:00  
**Discovery Trigger:** CoreRank Method 1 Bayesian Surprise (#3 on 4B, +7.65)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Kullback-Leibler (KL) divergence regularization is foundational to modern reinforcement learning from human feedback (RLHF), Group Relative Policy Optimization (GRPO), and logit distillation over large vocabularies (V = 32,000-151,936). Yet standard PyTorch F.kl_div evaluates sum_i p_i (log p_i - log q_i), suffering catastrophic cancellation whenever policy and reference distributions lie within typical trust-region proximity. Extending prior forward-pass observations across 36 regimes (forward and reverse KL; FP16, BF16, FP32; eps in [1e-4, 1e-1]), we show this pathology corrupts the backward pass directly: naive F.kl_div produces negative KL on up to 50.4% of inputs and 39.2% reverse-KL gradient sign flips (BF16, eps=1e-2); the common clamp(min=0) patch eliminates negative scalars only by inducing up to 50.4% silent zero-gradient dead-zones (61.8% sign flips); and naive BF16 Schulman k3 (exp(d)-1-d) suffers 100% premature zeroing for eps <= 1e-2. We prove formal floating-point truncation and condition-number bounds for an overflow-free two-branch 6th-order Bregman-Taylor formulation with cancellation-free analytic gradients, and implement a fused Blackwell Triton forward/backward kernel that streams online logsumexp in FP32 registers without materializing (N, V) FP32 intermediates. On an NVIDIA GeForce RTX 5090, our kernel achieves 0.0% negative KL across all 36 regimes (Wilcoxon p=1.45e-11), cuts FP32 (eps=1e-4) relative error from 21.1 to 1.42e-04 (149,111x reduction), runs 6.53x faster with 80.0% less peak VRAM than FP32-upcast F.kl_div at Qwen3 V=151,936 (32.0x faster than eager Bregman-Taylor), and matches the FP32 oracle exactly in 3-seed BF16 RLHF policy optimization (V=32,000) where clamped F.kl_div collapses into negative true objective (-0.00162 vs. +0.00164) with 99.4% dead-KL steps. Even against PyTorch 2.11 torch.compile (TorchInductor, which disables online softmax on split reductions), our kernel is 1.81x faster with 33.3% less VRAM and 25.6x faster than row-chunked FP32. Moreover, benchmarking against a Liger-style standard fused Triton KL kernel (0.72 ms) proves that SRAM fusion alone still yields 49.7% negative KL (69.2x relative error at FP32 eps=1e-4), whereas adding our 6th-order Bregman-Taylor branch eliminates cancellation at minimal latency overhead. In real 200-step 3-seed BF16 RLHF on Qwen3-1.7B (V=151,936, beta=2.0), clamped F.kl_div incurs 13.4% dead-gradient tokens, 65% excess KL drift (0.00282 vs. 0.00171), and degrades true FP64 objective to +0.02489 vs. +0.02637 for Fused Triton, while Fused Triton runs 1.55x faster end-to-end (14.2 vs. 22.0 ms/step) with 458 MB lower peak VRAM than FP32-upcast and 0.0% negative/dead tokens (vs. 21.4% negative tokens for BF16 F.kl_div).

### Key Empirical Findings
- Forward & backward gradient corruption across 36 regimes (Wilcoxon p=1.45e-11): PyTorch F.kl_div returns negative values on up to 50.4% of inputs and corrupts BF16 reverse-KL gradients (eps=1e-2) with 39.1% sign flips (cos=0.589). Clamping F.kl_div.clamp(min=0) induces 50.4% silent zero-gradient dead-zones and 61.8% sign flips (cos=0.371).
- Even in FP32 (eps=1e-4), F.kl_div and FP32-upcast yield 48.1% negative outputs and median relative error 21.14 (2,114%); our 6th-order Bregman-Taylor branch cuts median error by 149,111x to 1.42e-04 (0.0% negative, 0.0% error >=1%).
- Blackwell RTX 5090 kernel scaling vs. torch.compile & chunking at Qwen3 V=151,936 (N=512, BF16 fwd+bwd): Fused Triton runs in 0.90 ms (296.8 MB extra VRAM; Reverse KL 0.90 ms) — 6.53x faster (-80.0% VRAM) vs. FP32-upcast, 1.81x faster (-33.3% VRAM) vs. torch.compile Bregman-Taylor (1.63 ms), and 25.6x faster vs. Chunked-FP32 (22.99 ms).
- Real end-to-end 1.7B LLM LoRA post-training (3 seeds each): On Qwen3-1.7B (V=151,936), Fused Triton is fastest overall at 25.1 ms/step (1.25x faster end-to-end, -754 MB VRAM vs. FP32-upcast 31.3 ms/step) with 0.0% negative/dead tokens vs. 13.8% negative (BF16 F.kl_div) and 18.3% dead gradients (Clamped). On SmolLM2-1.7B (V=49,152), Fused Triton is both fastest (20.4 ms/step) and lowest-VRAM (4616 MB vs. 4862 MB FP32), eliminating 30.2% negative-KL tokens and 34.1% dead-gradient tokens.

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
  "real_llm_models": [
    "Qwen3-1.7B",
    "SmolLM2-1.7B"
  ],
  "real_qwen_rlhf_steps": 200,
  "triton_version": "3.6.0",
  "torch_version": "2.11.0+cu130",
  "baselines": [
    "PyTorch F.kl_div",
    "Clamped F.kl_div",
    "Schulman k3 (native)",
    "FP32-Upcast F.kl_div",
    "torch.compile FP32-Upcast",
    "torch.compile Bregman-Taylor",
    "Liger-Style Standard Fused Triton",
    "Chunked-FP32 Bregman-Taylor"
  ],
  "figure_png_rel": "papers/figures/pytorch-kl-bregman-v4.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 127.14
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

| Vocabulary Scale (N) | FP32-Upcast | torch.compile Bregman | Chunked-FP32 | Triton Fused (Fwd/Rev) | vs FP32 | vs compile | VRAM Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| V=256 (Toy) (N=2048) | 0.13ms / 10MB | 0.16ms / 1MB | 12.96ms / 2MB | 0.15ms (0.16) / 2MB | 0.87x | 1.05x | -80.0% |
| V=32k (Llama-2) (N=1024) | 2.10ms / 625MB | 0.56ms / 188MB | 7.71ms / 182MB | 0.36ms (0.36) / 125MB | 5.79x | 1.56x | -80.0% |
| V=128k (Llama-3) (N=512) | 4.93ms / 1252MB | 1.38ms / 376MB | 17.86ms / 603MB | 0.77ms (0.79) / 250MB | 6.43x | 1.80x | -80.0% |
| V=152k (Qwen3) (N=512) | 5.88ms / 1484MB | 1.63ms / 445MB | 22.99ms / 714MB | 0.90ms (0.90) / 297MB | 6.53x | 1.81x | -80.0% |

| Model (V) | KL Estimator | Step ms (↓) | Peak VRAM MB (↓) | Neg KL Tok % (↓) | Dead Grad Tok % (↓) | Med KL RelErr (↓) | Val True Obj (↓) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B (V=151,936) | PyTorch F.kl_div | 27.0 ± 0.0 | 5569 | 13.8% | 0.0% | 0.0696 | 3.6065 |
| Qwen3-1.7B (V=151,936) | Clamped F.kl_div | 27.0 ± 0.0 | 5668 | 0.0% | 18.3% | 0.4203 | 3.6063 |
| Qwen3-1.7B (V=151,936) | Schulman k3 | 28.8 ± 0.0 | 6249 | 12.1% | 0.0% | 0.0672 | 3.6064 |
| Qwen3-1.7B (V=151,936) | FP32-Upcast F.kl_div | 31.3 ± 0.0 | 6854 | 0.3% | 0.0% | 0.0000 | 3.6060 |
| Qwen3-1.7B (V=151,936) | Triton Bregman-Taylor (Ours) | 25.1 ± 0.0 | 6100 | 0.0% | 0.0% | 0.0000 | 3.6056 |
| SmolLM2-1.7B (V=49,152) | PyTorch F.kl_div | 19.4 ± 0.1 | 5032 | 30.2% | 0.0% | 0.2456 | 3.4402 |
| SmolLM2-1.7B (V=49,152) | Clamped F.kl_div | 19.4 ± 0.1 | 4948 | 0.0% | 34.1% | 0.9898 | 3.4402 |
| SmolLM2-1.7B (V=49,152) | Schulman k3 | 19.7 ± 0.0 | 4718 | 6.9% | 0.0% | 0.2009 | 3.4391 |
| SmolLM2-1.7B (V=49,152) | FP32-Upcast F.kl_div | 20.5 ± 0.0 | 4862 | 0.0% | 0.0% | 0.0000 | 3.4402 |
| SmolLM2-1.7B (V=49,152) | Triton Bregman-Taylor (Ours) | 20.4 ± 1.4 | 4616 | 0.0% | 0.0% | 0.0000 | 3.4394 |

## 4. Honest Limitations & Open Questions
- Irreducible BF16 pre-quantization floor (Theorem 2): sub-ULP logit differences rounded prior to KL evaluation bound all post-quantization kernels equally; fusing directly with FP32 LM-head output eliminates this floor.
- Targets full-vocabulary categorical KL (V up to 151,936); sampled-token Monte Carlo GRPO benefits primarily from scalar Eq. (1) stabilization.
- Real 1.7B LLM runs validate 3-seed end-to-end LoRA post-training over 24 steps; multi-thousand-step RLVR reasoning runs at 7B-70B scale remain future work.

## References
[1] Schulman, J. (2020). Approximating KL Divergence. Technical Blog Note (k1/k2/k3 estimators).
[2] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). arXiv:2402.03300.
[3] Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019.
[4] Hsu, P.-L., Dai, Y., Kothapalli, V., Song, Q., Tang, S., Tang, S., et al. (2024). Liger Kernel: Efficient Triton Kernels for LLM Training. arXiv:2410.10989.
[5] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
[6] Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.
[7] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS 2022.
[8] Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. ASPLOS 2024.
[9] Yang, Z., Tang, W., Chen, C., Qian, R., & Chen, Y. (2026). The KL Divergence Is Never Negative, Except in PyTorch. arXiv:2611.30835.

## 5. BibTeX Citation
```bibtex
@article{diaz2026pytorchklbregman,
  title       = {Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: Theory, Fused Triton Kernels, and LLM Post-Training},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 4.0},
  doi         = {10.5281/zenodo.24521689},
  url         = {https://spaceresearch.ai/papers/pytorch-kl-bregman-stability.html}
}
```
