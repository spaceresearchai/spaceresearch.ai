# Probabilistic Marginal Gradients for Sparse Autoencoders: Bridging ProbMoE Routing and BatchTopK SAEs

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Published:** 2026-11-24T16:46:42+01:00  
**Discovery Trigger:** Method 2 Cross-Domain Bridge + Method 3 Dense Contradiction  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
CoreRank Method 2 discovered a low-BGE (0.706), high-ACE-mechanism (0.879, 4B transfer +12.31) bridge between ProbMoE's differentiable marginal-probability subset routing and BatchTopK Sparse Autoencoders. We test whether replacing straight-through TopK gradients with ProbMoE's calibrated marginal-selection surrogate p_j = sigmoid((z_j - tau)/T) under identical hard BatchTopK forward sparsity and pure MSE loss improves reconstruction fidelity (FVU), eliminates dead latents, and alters cross-seed feature stability on real Qwen3-1.7B residual-stream activations.

### Empirical Summary
- Dead-latent reduction (k16): ProbBatchTopK reduced dead dictionary atoms to 20.00% vs. 31.12% (BatchTopK-STE) and 7.81% (TopK-STE).
- Reconstruction fidelity (FVU / R²): At k16, ProbBatchTopK achieved FVU=0.4555 (R²=0.5445), 0.0372 higher than BatchTopK-STE (0.4182) and +0.0451 vs. TopK-STE (0.4104). At k32, ProbBatchTopK reached FVU=0.4121 (0.0385 higher vs. BatchTopK-STE 0.3736).
- Cross-seed feature stability (3 independent seeds): At k16, ProbBatchTopK slightly reduced mean max-cosine overlap at 0.3984 (shared ≥0.70: 3.0%) vs. BatchTopK-STE 0.4524 (6.6%) and TopK-STE 0.4593 (5.7%). At k32: 0.3171 vs. 0.3483 (Δ=-0.0312).

## 1. Source Literature Discovered by CoreRank
- [ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](https://arxiv.org/abs/2606.01509) (`arxiv-2606.01509`)
- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) (`arxiv-2412.06410`)
- [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615) (`arxiv-2501.16615`)

## 2. Experimental Protocol & Hardware
```json
{
  "source_model": "Qwen/Qwen3-1.7B",
  "layer_extracted": "L13/28",
  "d_model": 2048,
  "papers_sampled": 240,
  "total_tokens": 21609,
  "extract_vram_gb": 1.8,
  "extract_sec": 1.95,
  "d_sae": 2048,
  "expansion_factor": 1.0,
  "k_values": [
    16,
    32
  ],
  "seeds": [
    11,
    22,
    33
  ],
  "steps_per_run": 400,
  "train_tokens": 17287,
  "test_tokens": 4322,
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 9.7
}
```

## 3. Empirical Results
| Sparsity | Routing / Gradient Estimator | FVU (↓) | R² (↑) | Dead Latents % (↓) | Cross-Seed Cos (↑) | Shared ≥0.70 % (↑) | Mean L0 |
|---|---|---:|---:|---:|---:|---:|---:|
| `k16` | TopK-STE (Baseline) | 0.4104 ± 0.0003 | 0.5896 | 7.81% | 0.4593 | 5.7% | 16.0 |
| `k16` | BatchTopK-STE | 0.4182 ± 0.0005 | 0.5818 | 31.12% | 0.4524 | 6.6% | 16.0 |
| `k16` | ProbBatchTopK (Bridge) | 0.4555 ± 0.0012 | 0.5445 | 20.00% | 0.3984 | 3.0% | 16.0 |
| `k32` | TopK-STE (Baseline) | 0.3739 ± 0.0001 | 0.6261 | 1.61% | 0.3696 | 1.6% | 32.0 |
| `k32` | BatchTopK-STE | 0.3736 ± 0.0004 | 0.6264 | 3.92% | 0.3483 | 1.7% | 32.0 |
| `k32` | ProbBatchTopK (Bridge) | 0.4121 ± 0.0003 | 0.5879 | 2.41% | 0.3171 | 1.0% | 32.0 |

## 4. Honest Limitations & Open Questions
- Dictionary width is d_sae=2048 (1x-2x overcomplete) over 21,609 real tokens from 240 abstracts; scaling to 65k-131k width over tens of millions of tokens may shift the relative dead-latent vs. FVU trade-off.
- ProbBatchTopK solves a 12-iteration scalar bisection per batch for tau; while cheap (<1 ms on RTX 5090), sensitivity to the temperature annealing schedule (0.45 -> 0.12) warrants a wider sweep.
- All three models used pure MSE loss with zero auxiliary dead-latent resurrection penalty (AuxK) to isolate the routing gradient estimator itself; combining ProbBatchTopK with AuxK remains open.

## 5. BibTeX Citation
```bibtex
@techreport{diaz2026probmoebatchtopk,
  title       = {Probabilistic Marginal Gradients for Sparse Autoencoders: Bridging ProbMoE Routing and BatchTopK SAEs},
  author      = {Marcio Diaz and {Space AI}},
  institution = {Space AI (spaceresearch.ai)},
  year        = {2026},
  month       = {11},
  url         = {https://spaceresearch.ai/papers/probmoe-batchtopk-sae.html}
}
```
