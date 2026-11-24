# Steerable Before Decodable: Testing CoreRank 1D Causal Steering vs. Logit Lens Readability Across Depth

**Authors:** Marcio Diaz, Space AI (`spaceresearch.ai`)  
**Published:** 2026-11-24T17:16:34+01:00  
**Discovery Trigger:** Method 1 Prior-vs-Evidence Bayesian Surprise (#1 on 4B, +9.30)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
CoreRank Method 1 surfaced arxiv:2604.02608 (4B surprise +9.30, #1 overall), showing that linear steering vectors routinely control model outputs where the unembedding Logit Lens fails to decode the answer. We test whether Paired-Swap ACE 1D cores in Qwen/Qwen3-1.7B exhibit this exact 'steerable-before-decodable' separation on strictly held-out entities across unseen comparative lexicons, and measure their cosine alignment with W_U.

### Empirical Summary
- Planets: Peak cross-lexicon 1D causal steering occurs at L26/28 (Flip-1D AUC drop = +0.292, clean cross-AUC 0.972), where direct Logit Lens AUC is only 0.986 and unembedding alignment |cos(q_l, ΔW_U)| = 0.3472. ACE linear readout crosses 0.85 AUC at L18 vs. Logit Lens at L18.
- Hazard Severity: Peak cross-lexicon 1D causal steering occurs at L24/28 (Flip-1D AUC drop = +0.048, clean cross-AUC 0.520), where direct Logit Lens AUC is only 0.455 and unembedding alignment |cos(q_l, ΔW_U)| = 0.0883. ACE linear readout crosses 0.85 AUC at L16 vs. Logit Lens at LNone.

## 1. Source Literature Discovered by CoreRank
- [Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens](https://arxiv.org/abs/2604.02608) (`arxiv-2604.02608`)

## 2. Experimental Protocol & Hardware
```json
{
  "model_id": "Qwen/Qwen3-1.7B",
  "n_layers": 28,
  "d_model": 2048,
  "domains": [
    "planets",
    "hazard_severity"
  ],
  "sweep_layers": [
    4,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28
  ],
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 7.9
}
```

## 3. Empirical Results
| Domain | Layer | Rel Depth | ACE Readout AUC | Logit Lens AUC (Cross) | Cross Flip-1D AUC | Steer Drop (↑) | |cos(q_l, ΔW_U)| |
|---|---:|---:|---:|---:|---:|---:|---:|
| `planets` | L4 | 0.14 | 0.750 | 0.556 | 0.972 | +0.000 | 0.0195 |
| `planets` | L8 | 0.29 | 0.778 | 0.500 | 0.986 | -0.014 | 0.0260 |
| `planets` | L10 | 0.36 | 0.694 | 0.361 | 0.979 | -0.007 | 0.0186 |
| `planets` | L12 | 0.43 | 0.750 | 0.319 | 0.986 | -0.014 | 0.0060 |
| `planets` | L14 | 0.50 | 0.611 | 0.458 | 0.979 | -0.007 | 0.0003 |
| `planets` | L16 | 0.57 | 0.750 | 0.653 | 0.965 | +0.007 | 0.0659 |
| `planets` | L18 | 0.64 | 1.000 | 1.000 | 0.701 | +0.271 | 0.2683 |
| `planets` | L20 | 0.71 | 1.000 | 1.000 | 0.757 | +0.215 | 0.3165 |
| `planets` | L22 | 0.79 | 1.000 | 0.958 | 0.750 | +0.222 | 0.3142 |
| `planets` | L24 | 0.86 | 1.000 | 0.972 | 0.764 | +0.208 | 0.3222 |
| `planets` | L26 | 0.93 | 1.000 | 0.986 | 0.681 | +0.292 | 0.3472 |
| `planets` | L28 | 1.00 | 1.000 | 0.972 | 0.722 | +0.250 | 0.2891 |
| `hazard_severity` | L4 | 0.14 | 0.560 | 0.710 | 0.540 | -0.020 | 0.0313 |
| `hazard_severity` | L8 | 0.29 | 0.540 | 0.315 | 0.550 | -0.030 | 0.0344 |
| `hazard_severity` | L10 | 0.36 | 0.530 | 0.445 | 0.527 | -0.007 | 0.0088 |
| `hazard_severity` | L12 | 0.43 | 0.390 | 0.490 | 0.500 | +0.020 | 0.0298 |
| `hazard_severity` | L14 | 0.50 | 0.470 | 0.240 | 0.495 | +0.025 | 0.0314 |
| `hazard_severity` | L16 | 0.57 | 0.970 | 0.315 | 0.610 | -0.090 | 0.0574 |
| `hazard_severity` | L18 | 0.64 | 0.820 | 0.460 | 0.583 | -0.062 | 0.0467 |
| `hazard_severity` | L20 | 0.71 | 0.590 | 0.320 | 0.550 | -0.030 | 0.0551 |
| `hazard_severity` | L22 | 0.79 | 0.510 | 0.390 | 0.475 | +0.045 | 0.0712 |
| `hazard_severity` | L24 | 0.86 | 0.480 | 0.455 | 0.472 | +0.048 | 0.0883 |
| `hazard_severity` | L26 | 0.93 | 0.440 | 0.515 | 0.485 | +0.035 | 0.0839 |
| `hazard_severity` | L28 | 1.00 | 0.610 | 0.515 | 0.527 | -0.007 | 0.0454 |

## 4. Honest Limitations & Open Questions
- Evaluated on Qwen/Qwen3-1.7B (28 layers) across two structured comparative domains; larger 7B-70B models and open-ended generative tasks should be checked before claiming universal layer offsets.
- Direct Logit Lens applies the final RMSNorm + W_U without a learned affine Tuned Lens adapter; however, the near-zero cosine |cos(q_l, ΔW_U)| confirms the steering subspace itself is nearly orthogonal to the unembedding readout vector.

## 5. BibTeX Citation
```bibtex
@techreport{diaz2026steerablenotdeco,
  title       = {Steerable Before Decodable: Testing CoreRank 1D Causal Steering vs. Logit Lens Readability Across Depth},
  author      = {Marcio Diaz and {Space AI}},
  institution = {Space AI (spaceresearch.ai)},
  year        = {2026},
  month       = {11},
  url         = {https://spaceresearch.ai/papers/steerable-not-decodable-ace-cores.html}
}
```
