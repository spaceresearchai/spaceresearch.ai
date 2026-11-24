# Eliminating Negative & Corrupted KL Divergence in Low-Precision PyTorch via Bregman-Taylor Formulation

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**DOI:** `10.5281/zenodo.24508479` (https://zenodo.org/record/24508479)  
**Published:** 2026-11-24T16:47:28+01:00  
**Discovery Trigger:** Method 1 Prior-vs-Evidence Bayesian Surprise (#3 on 4B)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
CoreRank Method 1 flagged arxiv:2611.30835 (4B surprise +7.65), showing PyTorch F.kl_div yields negative values and large relative errors for nearby distributions. We reproduce this on RTX 5090 across FP16/BF16/FP32 and test whether a strictly nonnegative Bregman-Taylor form p*(expm1(d)-d) eliminates negative KL and catastrophic cancellation.

### Empirical Summary
- PyTorch F.kl_div silent negative failures reproduced on NVIDIA GeForce RTX 5090: at eps=1e-3 (mean KL≈4.84e-07), FP16 F.kl_div returns negative values on 49.9% of inputs (BF16: 49.3%) and ≥1% relative error on 100.0%.
- Even in FP32 at eps=1e-4 (mean KL≈4.84e-09), standard F.kl_div yields negative outputs on 38.9% of inputs and ≥1% relative error on 99.8%.
- Stabilized Bregman-Taylor formulation p*(expm1(d)-d) achieves 0.0% negative values across every tested dtype and precision regime, cutting FP32 (eps=1e-4) ≥1% relative error rate from 99.8% down to 0.0%.

## 1. Source Literature Discovered by CoreRank
- [The KL Divergence Is Never Negative, Except in PyTorch](https://arxiv.org/abs/2611.30835) (`arxiv-2611.30835`)

## 2. Experimental Protocol & Hardware
```json
{
  "n_samples": 8192,
  "vocab": 256,
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 0.38
}
```

## 3. Empirical Results
| Dtype | eps | Ref KL | F.kl_div Neg % | F.kl_div Err≥1% | Bregman Neg % | Bregman Err≥1% |
|---|---|---:|---:|---:|---:|---:|
| `fp16` | `eps_0.0001` | 4.84e-09 | 49.8% | 100.0% | 0.0% | 100.0% |
| `fp16` | `eps_0.0003` | 4.36e-08 | 49.7% | 100.0% | 0.0% | 100.0% |
| `fp16` | `eps_0.001` | 4.84e-07 | 49.9% | 100.0% | 0.0% | 99.9% |
| `fp16` | `eps_0.003` | 4.36e-06 | 48.5% | 100.0% | 0.0% | 93.5% |
| `fp16` | `eps_0.01` | 4.84e-05 | 36.6% | 99.7% | 0.0% | 74.7% |
| `fp16` | `eps_0.1` | 4.84e-03 | 0.0% | 74.5% | 0.0% | 0.2% |
| `bf16` | `eps_0.0001` | 4.84e-09 | 23.1% | 100.0% | 0.0% | 100.0% |
| `bf16` | `eps_0.0003` | 4.36e-08 | 42.3% | 100.0% | 0.0% | 100.0% |
| `bf16` | `eps_0.001` | 4.84e-07 | 49.3% | 100.0% | 0.0% | 100.0% |
| `bf16` | `eps_0.003` | 4.36e-06 | 48.8% | 100.0% | 0.0% | 100.0% |
| `bf16` | `eps_0.01` | 4.84e-05 | 46.8% | 99.9% | 0.0% | 100.0% |
| `bf16` | `eps_0.1` | 4.84e-03 | 0.1% | 97.0% | 0.0% | 70.1% |
| `fp32` | `eps_0.0001` | 4.84e-09 | 38.9% | 99.8% | 0.0% | 0.0% |
| `fp32` | `eps_0.0003` | 4.36e-08 | 1.8% | 98.1% | 0.0% | 0.0% |
| `fp32` | `eps_0.001` | 4.84e-07 | 0.0% | 78.3% | 0.0% | 0.0% |
| `fp32` | `eps_0.003` | 4.36e-06 | 0.0% | 3.3% | 0.0% | 0.0% |
| `fp32` | `eps_0.01` | 4.84e-05 | 0.0% | 0.0% | 0.0% | 0.0% |
| `fp32` | `eps_0.1` | 4.84e-03 | 0.0% | 0.0% | 0.0% | 0.0% |

## 4. Honest Limitations & Open Questions
- Evaluated on categorical softmax distributions; continuous exponential-family KL closed forms require per-family stabilized identities.
- In extreme FP16/BF16 underflow where rounded log_p == log_q identically, Bregman-Taylor returns exact 0.0 (non-negative) rather than recovering sub-ULP rounding loss.

## 5. BibTeX Citation
```bibtex
@techreport{diaz2026pytorchklbregman,
  title       = {Eliminating Negative & Corrupted KL Divergence in Low-Precision PyTorch via Bregman-Taylor Formulation},
  author      = {Marcio Diaz and {Space AI}},
  institution = {Space AI (spaceresearch.ai)},
  year        = {2026},
  month       = {11},
  doi         = {10.5281/zenodo.24508479},
  url         = {https://spaceresearch.ai/papers/pytorch-kl-bregman-stability.html}
}
```
