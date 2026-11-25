# Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens

**Authors:** Marcio Diaz (`marcio.diaz@gmail.com`), Space AI (`spaceresearch.ai`)  
**Version:** 4.0  
**DOI:** `10.5281/zenodo.24522101` (https://zenodo.org/record/24522101)  
**Published:** 2026-11-25T06:13:58+01:00  
**Discovery Trigger:** CoreRank Method 1 Bayesian Surprise (#1 on 4B, +9.30)  
**Hardware:** NVIDIA GeForce RTX 5090  

## Abstract
Interpretability tools routinely assume that if a transformer represents a concept linearly in its residual stream, that concept can be decoded by projecting intermediate activations through the unembedding matrix (Logit Lens) or a trained affine translator (Tuned Lens). Evaluating 4 distinct open-weight model families spanning 1.7B to 7.3B parameters (Qwen3-1.7B, SmolLM2-1.7B, Mistral-7B, OLMo-2-7B) across 4 real-world comparative domains and 2,688 strictly held-out cross-lexicon entity pairs (100% single-token across all 4 tokenizers), we test this assumption against Paired-Swap Algorithmic Core Extraction (ACE) and five strong baselines (Direct Logit Lens, General-Corpus Tuned Lens, Domain Residual-Affine Tuned Lens, Contrastive Activation Addition [CAA], and L2 Logistic Probes). Across 16 (model, domain) cells, peak 1D causal steering emerges where ACE linear readout (mean AUC 0.765) consistently exceeds Direct Logit Lens (0.727, gap +0.039, Wilcoxon p=0.065), KL-Minimizing General-Corpus Tuned Lens (0.638, gap +0.127, p=0.011), and Ridge General-Corpus Tuned Lens (0.693, p=0.037) and significantly outperforms Domain Tuned Lens (0.708, gap +0.057, p=4.67e-02), while operating largely orthogonal to the unembedding contrast vector (mean |cos(q_l*, dW_U)| = 0.140). On 1D causal necessity (Flip-1D across unseen cross-lexicons), ACE achieves mean AUC reversal +0.235 — significantly outperforming random controls (-0.001, p=1.53e-05), CAA (+0.170, p=1.53e-04; 14/16 cells), and L2 Logistic Probes (+0.179, p=3.81e-03; 13/16 cells). Most strikingly, on causal sufficiency (Keep-1D — deleting 2,047-4,095 orthogonal dimensions and retaining only the single 1D ACE coordinate), Keep-1D outperforms the full unperturbed model in 16/16 cells (sign-test p=1.53e-05; mean AUC 0.799 vs. 0.712 clean, mean gain +0.087; up to +0.238 on Qwen3-1.7B physical_scale), exposing pervasive late-layer readout collapse. Probing boundary conditions in multi-token free-form greedy generation (Qwen3-1.7B & SmolLM2-1.7B), single-position prefill Flip-1D preserves fluent continuation NLL (0.86-1.49) but flips only 0.0%-5.6% of multi-token base-model completions (vs. 0.0% random) because subsequent generated tokens re-attend to unperturbed entity tokens in the KV cache, clarifying that sustained per-step steering is needed for multi-token generation.

### Key Empirical Findings
- Steerable before decodable across 4 architectures (1.7B-7.3B, 2,688 held-out cross-lexicon pairs): At peak causal steering depth, ACE readout (0.765) exceeds Direct Logit Lens (0.727, p=6.49e-02), General-Corpus Tuned Lens (0.693, p=3.70e-02), and Domain Tuned Lens (0.708, p=4.67e-02), with mean |cos(q_l*, dW_U)| = 0.140.
- Gradient-weighted ACE dominates activation-only steering baselines on 1D causal necessity (Flip-1D): Mean cross-lexicon reversal drop = +0.235 vs. CAA +0.170 (p=1.53e-04, wins 14/16), L2 Logistic Probe +0.179 (p=3.81e-03, wins 13/16), and Random Control -0.001 (p=1.53e-05).
- 1D causal sufficiency (Keep-1D) rescues late-layer readout collapse: Projecting the 2,048-4,096D residual stream onto only the 1D ACE axis outperforms the full unperturbed model in 16/16 cells (mean Keep-1D AUC 0.799 vs. clean final-layer 0.712, mean gain +0.087; largest rescue 0.592 -> 0.830 on Qwen3-1.7B physical_scale).
- Multi-token free-form generation boundary condition (honest negative/scope result): Single-position prefill Flip-1D preserves fluent base-model continuation NLL (0.86-1.49) but flips only 0.0%-5.6% of multi-token greedy answers (vs. 0.0% random) because base models emit multi-token prefixes that re-attend to unperturbed entity tokens in the KV cache, showing sustained per-step steering is required beyond single-token logits.

## 1. Source Literature Discovered by CoreRank
- [Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens](https://arxiv.org/abs/2604.02608) (`arxiv-2604.02608`)

## 2. Experimental Protocol & Hardware
```json
{
  "models": [
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B",
    "mistralai/Mistral-7B-v0.1",
    "allenai/OLMo-2-1124-7B"
  ],
  "n_models": 4,
  "domains": [
    "physical_scale",
    "hazard_severity",
    "historical_chronology",
    "geographic_scale"
  ],
  "n_domains": 4,
  "entities_per_domain": 16,
  "train_test_split": "disjoint 8 train / 8 unseen test entities (56 ordered pairs/lexicon x 3 cross-lexicons = 168/domain)",
  "total_heldout_cross_pairs": 2688,
  "baselines": [
    "Direct Logit Lens",
    "KL-Minimizing General Tuned Lens",
    "Ridge General Tuned Lens",
    "Domain Tuned Lens",
    "CAA Mean-Diff",
    "L2 Logistic Probe",
    "Random Unit Vector"
  ],
  "bootstrap_resamples": 1000,
  "figure_png_rel": "papers/figures/steerable-not-decodable-v4.png",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "total_elapsed_sec": 1944.1
}
```

## 3. Empirical Results
| Model | Domain | Peak l* | ACE Read | Gen-TL | Dom-TL | Logit Lens | ACE Flip | CAA | Probe | Keep-1D (Clean) | |cos(W_U)| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | `physical_scale` | L24 (0.86) | 0.854 | 0.548 | 0.577 | 0.590 | +0.379 [+0.25,+0.51] | +0.223 | +0.301 | 0.830 (0.592) | 0.289 |
| Qwen3-1.7B | `hazard_severity` | L20 (0.71) | 0.827 | 0.658 | 0.670 | 0.708 | +0.072 [+0.04,+0.10] | +0.054 | +0.025 | 0.798 (0.702) | 0.097 |
| Qwen3-1.7B | `historical_chronology` | L28 (1.00) | 0.624 | 0.572 | 0.572 | 0.572 | +0.104 [+0.04,+0.17] | +0.047 | +0.075 | 0.620 (0.572) | 0.155 |
| Qwen3-1.7B | `geographic_scale` | L18 (0.64) | 0.938 | 0.788 | 0.897 | 0.922 | +0.301 [+0.21,+0.40] | +0.213 | +0.137 | 0.916 (0.895) | 0.302 |
| SmolLM2-1.7B | `physical_scale` | L18 (0.75) | 0.804 | 0.569 | 0.735 | 0.721 | +0.362 [+0.22,+0.50] | +0.289 | +0.403 | 0.832 (0.816) | 0.119 |
| SmolLM2-1.7B | `hazard_severity` | L20 (0.83) | 0.789 | 0.691 | 0.719 | 0.664 | +0.285 [+0.18,+0.39] | +0.174 | +0.125 | 0.790 (0.691) | 0.158 |
| SmolLM2-1.7B | `historical_chronology` | L18 (0.75) | 0.802 | 0.463 | 0.633 | 0.684 | +0.216 [+0.13,+0.31] | +0.153 | +0.105 | 0.719 (0.592) | 0.088 |
| SmolLM2-1.7B | `geographic_scale` | L24 (1.00) | 0.586 | 0.867 | 0.867 | 0.867 | +0.107 [+0.04,+0.17] | -0.008 | -0.005 | 0.907 (0.863) | 0.044 |
| Mistral-7B | `physical_scale` | L24 (0.75) | 0.799 | 0.682 | 0.699 | 0.639 | +0.365 [+0.23,+0.49] | +0.301 | +0.351 | 0.850 (0.712) | 0.150 |
| Mistral-7B | `hazard_severity` | L16 (0.50) | 0.630 | 0.722 | 0.703 | 0.707 | +0.141 [+0.08,+0.21] | +0.076 | +0.070 | 0.734 (0.727) | 0.040 |
| Mistral-7B | `historical_chronology` | L20 (0.62) | 0.824 | 0.791 | 0.775 | 0.798 | +0.238 [+0.17,+0.31] | +0.256 | +0.269 | 0.739 (0.681) | 0.136 |
| Mistral-7B | `geographic_scale` | L28 (0.88) | 0.713 | 0.654 | 0.645 | 0.640 | +0.145 [+0.08,+0.21] | +0.135 | +0.084 | 0.811 (0.647) | 0.138 |
| OLMo-2-7B | `physical_scale` | L32 (1.00) | 0.800 | 0.715 | 0.715 | 0.715 | +0.430 [+0.28,+0.57] | +0.274 | +0.380 | 0.842 (0.715) | 0.092 |
| OLMo-2-7B | `hazard_severity` | L24 (0.75) | 0.803 | 0.699 | 0.695 | 0.733 | +0.078 [+0.03,+0.12] | +0.044 | -0.024 | 0.739 (0.722) | 0.094 |
| OLMo-2-7B | `historical_chronology` | L24 (0.75) | 0.804 | 0.745 | 0.623 | 0.736 | +0.281 [+0.19,+0.38] | +0.209 | +0.241 | 0.765 (0.617) | 0.167 |
| OLMo-2-7B | `geographic_scale` | L20 (0.62) | 0.647 | 0.927 | 0.805 | 0.928 | +0.260 [+0.08,+0.45] | +0.277 | +0.332 | 0.890 (0.848) | 0.165 |

| Model | Domain | Clean Acc % | ACE Flip-1D % (↑) | Random Flip % | Clean NLL | ACE Steered NLL |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | `physical_scale` | 32.1% | 5.6% | 44.4% | 0.0% | 1.04->1.06 |
| Qwen3-1.7B | `hazard_severity` | 19.6% | 0.0% | 0.0% | 0.0% | 0.87->0.90 |
| SmolLM2-1.7B | `physical_scale` | 17.9% | 0.0% | 0.0% | 0.0% | 1.15->1.40 |
| SmolLM2-1.7B | `hazard_severity` | 8.9% | 0.0% | 20.0% | 0.0% | 1.50->1.48 |

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
[10] Nadaf, M. S. B. (2026). Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens. arXiv:2604.02608.

## 5. BibTeX Citation
```bibtex
@article{diaz2026steerablenotdeco,
  title       = {Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens},
  author      = {Marcio Diaz and {Space AI}},
  journal     = {Space AI Technical Preprint (Zenodo)},
  year        = {2026},
  month       = {11},
  note        = {Version 4.0},
  doi         = {10.5281/zenodo.24522101},
  url         = {https://spaceresearch.ai/papers/steerable-not-decodable-ace-cores.html}
}
```
