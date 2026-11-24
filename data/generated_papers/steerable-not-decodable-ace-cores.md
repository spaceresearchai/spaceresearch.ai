# Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 2.0  
**DOI:** `10.5281/zenodo.24511855` (https://zenodo.org/record/24511855)  
**Published:** 2026-11-24T19:53:27+01:00  
**Discovery Trigger:** CoreRank Method 1 Bayesian Surprise (#1 on 4B, +9.30)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Interpretability tools routinely assume that if a transformer represents a concept linearly in its residual stream, that concept can be decoded by projecting intermediate activations through the unembedding matrix (Logit Lens) or a trained affine translator (Tuned Lens). Evaluating 3 distinct open-weight model families spanning 1.7B to 7.2B parameters (Qwen3-1.7B, SmolLM2-1.7B, Mistral-7B) across 4 real-world comparative domains and 2,016 strictly held-out cross-lexicon entity pairs (100% single-token across all three tokenizers), we test this assumption against Paired-Swap Algorithmic Core Extraction (ACE) and four strong baselines (Direct Logit Lens, Residual-Affine Tuned Lens, Contrastive Activation Addition [CAA], and L2 Logistic Probes). Across 12 (model, domain) cells, peak 1D causal steering emerges where ACE linear readout (mean AUC 0.766) significantly outperforms Direct Logit Lens (0.709, gap +0.057, Wilcoxon p=0.046) and consistently trends above Trained Tuned Lens (0.708, gap +0.058, one-sided Wilcoxon p=0.065), while operating largely orthogonal to the unembedding contrast vector (mean |cos(q_l*, dW_U)| = 0.143). On 1D causal necessity (Flip-1D across unseen cross-lexicons), ACE achieves mean AUC reversal +0.226 — significantly outperforming random controls (-0.002, p=2.44e-04), CAA (+0.160, p=7.32e-04; 11/12 cells), and L2 Logistic Probes (+0.162, p=4.64e-03; 10/12 cells). Most strikingly, on causal sufficiency (Keep-1D — deleting 2,047-4,095 orthogonal dimensions and retaining only the single 1D ACE coordinate), Keep-1D outperforms the full unperturbed model in 12/12 cells (p=2.44e-04; mean AUC 0.795 vs. 0.707 clean, mean gain +0.088; up to +0.238 on Qwen3-1.7B physical_scale), exposing pervasive late-layer readout collapse.

### Key Empirical Findings
- Steerable before decodable across 3 architectures (1.7B-7.2B, 2,016 held-out cross-lexicon pairs): At peak causal steering depth, ACE readout (0.766) exceeds both Direct Logit Lens (0.709, p=4.61e-02) and Trained Residual-Affine Tuned Lens (0.708, p=6.47e-02), with mean unembedding alignment |cos(q_l*, dW_U)| = 0.143.
- Gradient-weighted ACE dominates activation-only steering baselines on 1D causal necessity (Flip-1D): Mean cross-lexicon reversal drop = +0.226 vs. CAA +0.160 (p=7.32e-04, wins 11/12), L2 Logistic Probe +0.162 (p=4.64e-03, wins 10/12), and Random Control -0.002 (p=2.44e-04).
- 1D causal sufficiency (Keep-1D) rescues late-layer readout collapse: Projecting the 2,048-4,096D residual stream onto only the 1D ACE axis outperforms the full unperturbed model in 12/12 cells (mean Keep-1D AUC 0.795 vs. clean final-layer 0.707, mean gain +0.088; largest rescue 0.592 -> 0.830 on Qwen3-1.7B physical_scale).
- Readout collapse (mid-layer best ACE AUC exceeding final-layer clean cross-lexicon AUC by >= 0.05) occurs in 8/12 cells (mean best mid-layer ACE 0.832 vs. clean 0.707), confirming that models compute accurate latent comparisons internally that are partially lost before lexical decoding.

## 1. Source Literature Discovered by CoreRank
- [Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens](https://arxiv.org/abs/2604.02608) (`arxiv-2604.02608`)

## 2. Experimental Protocol & Hardware
```json
{
  "models": [
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B",
    "mistralai/Mistral-7B-v0.1"
  ],
  "n_models": 3,
  "domains": [
    "physical_scale",
    "hazard_severity",
    "historical_chronology",
    "geographic_scale"
  ],
  "n_domains": 4,
  "entities_per_domain": 16,
  "train_test_split": "disjoint 8 train / 8 unseen test entities (56 ordered pairs/lexicon x 3 cross-lexicons = 168/domain)",
  "total_heldout_cross_pairs": 2016,
  "baselines": [
    "Direct Logit Lens",
    "Residual-Affine Tuned Lens",
    "CAA Mean-Diff",
    "L2 Logistic Probe",
    "Random Unit Vector"
  ],
  "bootstrap_resamples": 1000,
  "figure_png_rel": "papers/figures/steerable-not-decodable-v2.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 501.2
}
```

## 3. Empirical Results
| Model | Domain | Peak Layer | ACE Readout | Tuned Lens | Logit Lens | ACE Flip Drop | CAA Drop | Probe Drop | Keep-1D (Clean) | |cos(W_U)| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | `physical_scale` | L24 (0.86) | 0.854 | 0.577 | 0.590 | +0.379 [+0.25,+0.51] | +0.223 | +0.301 | 0.830 (0.592) | 0.289 |
| Qwen3-1.7B | `hazard_severity` | L20 (0.71) | 0.827 | 0.670 | 0.708 | +0.072 [+0.04,+0.10] | +0.054 | +0.025 | 0.798 (0.702) | 0.097 |
| Qwen3-1.7B | `historical_chronology` | L28 (1.00) | 0.624 | 0.572 | 0.572 | +0.104 [+0.04,+0.17] | +0.047 | +0.075 | 0.620 (0.572) | 0.155 |
| Qwen3-1.7B | `geographic_scale` | L18 (0.64) | 0.938 | 0.897 | 0.922 | +0.301 [+0.21,+0.40] | +0.213 | +0.137 | 0.916 (0.895) | 0.302 |
| SmolLM2-1.7B | `physical_scale` | L18 (0.75) | 0.804 | 0.735 | 0.721 | +0.362 [+0.22,+0.50] | +0.289 | +0.403 | 0.832 (0.816) | 0.119 |
| SmolLM2-1.7B | `hazard_severity` | L20 (0.83) | 0.789 | 0.719 | 0.664 | +0.285 [+0.18,+0.39] | +0.174 | +0.125 | 0.790 (0.691) | 0.158 |
| SmolLM2-1.7B | `historical_chronology` | L18 (0.75) | 0.802 | 0.633 | 0.684 | +0.216 [+0.13,+0.31] | +0.153 | +0.105 | 0.719 (0.592) | 0.088 |
| SmolLM2-1.7B | `geographic_scale` | L24 (1.00) | 0.586 | 0.867 | 0.867 | +0.107 [+0.04,+0.17] | -0.008 | -0.005 | 0.907 (0.863) | 0.044 |
| Mistral-7B | `physical_scale` | L24 (0.75) | 0.799 | 0.699 | 0.639 | +0.365 [+0.23,+0.49] | +0.301 | +0.351 | 0.850 (0.712) | 0.150 |
| Mistral-7B | `hazard_severity` | L16 (0.50) | 0.630 | 0.703 | 0.707 | +0.141 [+0.08,+0.21] | +0.076 | +0.070 | 0.734 (0.727) | 0.040 |
| Mistral-7B | `historical_chronology` | L20 (0.62) | 0.824 | 0.775 | 0.798 | +0.238 [+0.17,+0.31] | +0.256 | +0.269 | 0.739 (0.681) | 0.136 |
| Mistral-7B | `geographic_scale` | L28 (0.88) | 0.713 | 0.645 | 0.640 | +0.145 [+0.08,+0.21] | +0.135 | +0.084 | 0.811 (0.647) | 0.138 |

## 4. Honest Limitations & Open Questions
- Evaluated on 4 structured single-token comparative domains; free-form multi-step reasoning may recruit higher-rank subspaces.
- Residual-Affine Tuned Lens was trained on pooled disjoint training-entity activations rather than a massive general pretraining corpus.
- Spans 1.7B-7.2B models across 3 architectures (Qwen3, Llama, Mistral); 70B+ scaling behavior remains open.

## References
[1] nostalgebraist (2020). Interpreting GPT: The Logit Lens. LessWrong Technical Note.
[2] Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.
[3] Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2023). Steering Llama 2 via Contrastive Activation Addition. ACL 2024 / arXiv:2312.06681.
[4] Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., & Bau, D. (2024). Function Vectors in Large Language Models. ICLR 2024.
[5] Alain, G., & Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier Probes. ICLR 2017 Workshop.
[6] Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.
[7] Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. CLeaR 2024.
[8] Marks, S., & Tegmark, M. (2024). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. COLM 2024.
[9] Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023). Steering Language Models with Activation Engineering. arXiv:2308.10248.
[10] Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens (2026). arXiv:2604.02608.

## 5. BibTeX Citation
```bibtex
@article{diaz2026steerablenotdeco,
  title       = {Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 2.0},
  doi         = {10.5281/zenodo.24511855},
  url         = {https://spaceresearch.ai/papers/steerable-not-decodable-ace-cores.html}
}
```
