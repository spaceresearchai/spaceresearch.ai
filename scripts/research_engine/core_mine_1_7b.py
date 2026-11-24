#!/usr/bin/env python3
"""
CoreRank Scientific Discovery & Surprise Mining Benchmark over Space AI Papers DB (1,715 papers).

Tests all 4 CoreRank/ACE discovery mechanisms + V-Dip Conceptual Bottleneck Sanity Audit:
  Stage 0: Calibrate CoreAxes (surprise, rigor, hype, prior_plausibility) + V-Dip Audit + Adversarial Hype Test
  Method 1: Prior-vs-Evidence "Bayesian Surprise" (Single-paper contrarian discovery + NLL/belief shift)
  Method 2: Cross-Domain Mechanism Bridges (8-axis topic-canceled ACE subspace + BGE-orthogonal residual + Stage-2 check)
  Method 3: Hidden Contradictions in Dense BGE Clusters (De-Biased Stage-2 Keep-1D Paired-Swap vs Raw)
  Method 4: Core-vs-Consensus Residuals (Sleeping Beauties + Quiet Today Gems + Off-Core Causal Outliers & Condorcet Cycles)
"""

from __future__ import annotations

import gc
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from corerank.auditor import audit_vdip
from corerank.engine import CoreAxis, CoreRankEngine, compute_ace_subspace, compute_ranking_metrics, roc_auc

PAPERS_JSON = Path("/home/marcio/Projects/spaceresearch.ai/data/papers-db.json")
VECTORS_BIN = Path("/home/marcio/Projects/spaceresearch.ai/data/papers-vectors.f16.bin")
OUT_DIR = Path("/home/marcio/Projects/spaceresearch.ai/data/generated_papers")
OUT_JSON = OUT_DIR / "discovery_results_1.7b.json"

MODEL_ID = "Qwen/Qwen3-1.7B"


# ==============================================================================
# Calibration Datasets (strictly hype-free for surprise/prior; adversarial for rigor/hype)
# ==============================================================================

# 12 neutral, hype-free technical claims: 1.0 = textbook consensus, 10.0 = genuinely counterintuitive empirical discovery
SURPRISE_CAL_ITEMS: List[Tuple[str, float]] = [
    ("Training a larger transformer on more high-quality text tokens reduces validation cross-entropy loss", 1.0),
    ("Storing model weights in FP16 instead of FP32 cuts GPU weight memory usage roughly in half", 1.5),
    ("Using FlashAttention reduces memory reads by tiling attention blocks inside fast on-chip SRAM", 2.0),
    ("Speculative decoding drafts tokens with a small model and verifies them in parallel with a larger model", 2.5),
    ("Weight decay and dropout help prevent small neural networks from overfitting limited training sets", 3.0),
    ("Quantizing weights to 4 bits with group-wise scaling preserves more accuracy than naive per-tensor quantization", 4.0),
    ("Removing unused KV-cache tokens with low attention scores reduces long-context memory overhead", 5.0),
    ("Fine-tuning only low-rank weight updates can match full fine-tuning on narrow downstream tasks", 5.5),
    ("Removing half of the middle layers in a pretrained language model leaves many factual benchmarks nearly intact", 8.0),
    ("Appending meaningless random filler dots before an answer improves multi-step reasoning accuracy in transformers", 8.8),
    ("Training a network long after reaching zero training loss causes test accuracy to suddenly jump from chance to near-perfect", 9.5),
    ("A single fixed attention sink token at position zero absorbs large attention mass regardless of semantic content", 9.8),
]

SURPRISE_VDIP_LEXICONS = {
    "surprising_obvious": ("Evaluating scientific expectation, Claim A ({A}) compared to Claim B ({B}) is", " surprising", " obvious"),
    "unexpected_predictable": ("Evaluating how expected each finding is, Claim A ({A}) compared to Claim B ({B}) is", " unexpected", " predictable"),
    "counterintuitive_conventional": ("Evaluating scientific intuition, Claim A ({A}) compared to Claim B ({B}) is", " counterintuitive", " conventional"),
    "unusual_standard": ("Evaluating technical findings, Claim A ({A}) compared to Claim B ({B}) is", " unusual", " standard"),
    "yes_no_control": ("Is Claim A ({A}) more surprising than Claim B ({B})? Answer:", " Yes", " No"),
}

# Adversarial Hype Test: pairs of (plain_counterintuitive_claim, hyped_textbook_truism)
ADVERSARIAL_HYPE_PAIRS: List[Tuple[str, str]] = [
    (
        "Appending meaningless filler dots before an answer increases transformer arithmetic accuracy by 8 percent.",
        "Surprisingly, contrary to conventional wisdom, our paradigm-shifting breakthrough shows that FP16 weights use half the GPU memory of FP32.",
    ),
    (
        "Deleting 40 percent of middle transformer blocks preserves most factual recall while cutting latency.",
        "Remarkably and unexpectedly, we challenge established beliefs by demonstrating that training on more clean tokens lowers validation perplexity.",
    ),
    (
        "Replacing learned positional embeddings with fixed random projections improves length extrapolation on sorting tasks.",
        "To our great surprise, we overturn common assumptions by revealing that caching past key-value states speeds up autoregressive decoding.",
    ),
    (
        "Quantizing only the outlier activation channels to FP16 while keeping 99 percent in INT4 recovers full precision accuracy.",
        "In a stunningly counterintuitive discovery, our revolutionary study proves that larger batch sizes increase GPU matrix multiplication throughput.",
    ),
    (
        "Training on randomly shuffled synthetic noise before fine-tuning accelerates convergence on algorithmic reasoning.",
        "Defying decades of textbook intuition, our novel framework reveals that dropout reduces overfitting when training data is scarce.",
    ),
    (
        "Steering a single 1D residual direction at layer 12 reverses pairwise size judgments across unrelated lexical templates.",
        "Surprisingly, we uncover an unprecedented phenomenon: tiling attention in fast SRAM avoids repeated HBM reads.",
    ),
]

# 12 abstract-style technical summaries for Rigor (anti-fluff) & Hype axes
# Format: (text, rigor_score [1..10], hype_score [1..10])
RIGOR_HYPE_CAL_DATA: List[Tuple[str, float, float]] = [
    (
        "We introduce OmniNexus, a revolutionary paradigm-shifting framework that achieves unprecedented state-of-the-art intelligence across diverse real-world tasks through synergistic holistic alignment.",
        1.0, 10.0,
    ),
    (
        "Large language models face major challenges in complex reasoning. We propose HyperMind, a novel unified architecture that unlocks remarkable breakthroughs and vastly outperforms existing methods.",
        1.5, 9.5,
    ),
    (
        "To bridge the gap in next-generation AI, we design NovaAgent, an innovative end-to-end paradigm that seamlessly integrates multi-scale representations for superior performance on various benchmarks.",
        2.0, 9.0,
    ),
    (
        "We present a new prompting strategy that improves chat helpfulness by adding structured reflection templates, achieving modest gains on two curated evaluation sets.",
        4.0, 4.5,
    ),
    (
        "We survey recent advances in efficient inference and summarize common quantization and pruning recipes across transformer architectures without new empirical ablations.",
        4.5, 2.0,
    ),
    (
        "We fine-tune an 8B model with synthetic instruction pairs and report a 3.2 percent gain on MMLU, noting sensitivity to learning rate warmup.",
        5.5, 3.0,
    ),
    (
        "We evaluate INT4 post-training quantization across 6 model families and show that perplexity collapses above 8B parameters specifically due to 0.1 percent outlier channels exceeding 6 sigma.",
        8.0, 1.5,
    ),
    (
        "We prove a tight lower bound on KV-cache memory bandwidth under GQA and verify via Triton microbenchmarks on H100 that fused split-K decoding reduces stall cycles by 38 percent.",
        9.0, 1.0,
    ),
    (
        "Using causal activation patching across 32 layers, we isolate 4 attention heads responsible for indirect object identification and verify necessity and sufficiency via mean-ablation circuits.",
        9.2, 1.0,
    ),
    (
        "We show that standard sparse autoencoders suffer feature splitting at dictionary width 65k, derive a closed-form dead-latent threshold, and test failure modes across 5 controlled synthetic tasks.",
        9.5, 1.0,
    ),
    (
        "By formulating speculative tree verification as a dynamic programming problem over draft acceptance rates, we derive the optimal tree topology and validate latency under memory-bound batch sizes 1 to 64.",
        9.6, 1.2,
    ),
    (
        "We identify a causal failure mode in process reward models where step-level scores correlate with response length (r=0.71) rather than correctness; controlled length-matched ablations eliminate 64 percent of reported gains.",
        10.0, 1.0,
    ),
]

RIGOR_VDIP_LEXICONS = {
    "rigorous_superficial": ("Evaluating technical rigor and empirical substance, Abstract A ({A}) compared to Abstract B ({B}) is", " rigorous", " superficial"),
    "substantive_vague": ("Evaluating concrete mechanism and evidence, Abstract A ({A}) compared to Abstract B ({B}) is", " substantive", " vague"),
    "solid_weak": ("Evaluating scientific evidence quality, Abstract A ({A}) compared to Abstract B ({B}) is", " solid", " weak"),
    "concrete_generic": ("Evaluating technical specificity, Abstract A ({A}) compared to Abstract B ({B}) is", " concrete", " generic"),
    "yes_no_control": ("Is Abstract A ({A}) more technically rigorous than Abstract B ({B})? Answer:", " Yes", " No"),
}

# 10 items for Prior Plausibility (10 = universally expected true; 1 = runs directly counter to ML consensus)
PRIOR_PLAUSIBILITY_ITEMS: List[Tuple[str, float]] = [
    ("Increasing parameter count and training tokens together reduces language modeling perplexity", 10.0),
    ("Half-precision BF16 inference uses less GPU memory bandwidth than FP32 inference", 9.5),
    ("Using a larger KV cache allows autoregressive transformers to attend to longer contexts", 9.0),
    ("Training with noisy mislabeled data generally harms supervised classification accuracy", 8.5),
    ("Pruning 10 percent of smallest-magnitude weights causes minimal degradation after recovery fine-tuning", 7.0),
    ("Uniformly quantizing all activations including outlier channels to 3 bits preserves full reasoning accuracy", 2.5),
    ("Training past zero training loss causes sudden delayed generalization after prolonged memorization", 2.0),
    ("Adding random meaningless filler tokens improves mathematical reasoning accuracy without extra knowledge", 1.8),
    ("Removing half of the middle layers of a pretrained transformer causes almost no drop on factual QA", 1.5),
    ("A single fixed rank-1 linear vector captures cross-domain magnitude comparison inside a 4096-dim residual stream", 1.2),
]

# 8 Fundamental Mathematical Mechanisms for Method 2 (each calibrated across DIFFERENT domains!)
# Each entry: (name, pos_word, neg_word, list of (snippet_from_varied_domains, score))
MECHANISM_SPECS: List[Tuple[str, str, str, List[Tuple[str, float]]]] = [
    (
        "low_rank_spectral",
        " low-rank",
        " full-rank",
        [
            ("Projecting key-value caches into a compressed latent subspace via SVD cuts memory footprint by 8x", 10.0),
            ("Extracting the top singular vector of gradient-weighted activation differences reveals a 1D concept core", 9.8),
            ("Factorizing weight updates into rank-8 A and B matrices matches full parameter adaptation", 9.5),
            ("Observing a sharp spectral gap between the first and second eigenvalues enables exact subspace truncation", 9.2),
            ("Running dense full-matrix multiplications across every token and feature dimension without compression", 1.5),
            ("Training an overparameterized unstructured MLP with isotropic full-rank dense layers", 1.2),
            ("Evaluating chat fluency via human preference win rates on open-ended creative stories", 1.0),
            ("Measuring wall-clock tokenization throughput of a C++ regex pre-tokenizer on CPU", 1.0),
        ],
    ),
    (
        "sparse_gating",
        " sparse",
        " dense",
        [
            ("Routing each token to only the top-2 of 64 experts activates 5 percent of parameters per step", 10.0),
            ("Training a top-k sparse autoencoder decomposes polysemantic neurons into isolated monosemantic features", 9.8),
            ("Skipping attention blocks whose dynamic gating score falls below a threshold accelerates long contexts", 9.4),
            ("Selecting only a tiny active subset of candidate latents via L1 sparsity penalty improves interpretability", 9.0),
            ("Computing dense all-to-all attention scores across every token pair with uniform weights", 1.5),
            ("Applying uniform dense matrix updates to every neuron on every forward step", 1.2),
            ("Using standard cross-entropy loss over dense vocabulary logits during pretraining", 1.5),
            ("Analyzing dataset contamination using exact n-gram string matching", 1.0),
        ],
    ),
    (
        "heavy_tail_outliers",
        " outlier-dominated",
        " uniform",
        [
            ("Keeping the 0.1 percent massive-magnitude activation channels in FP16 prevents INT4 quantization collapse", 10.0),
            ("A handful of massive-activation dimensions and position-zero attention sinks dominate transformer dynamics", 9.8),
            ("Heavy-tailed singular values and extreme kurtosis in hidden states break uniform scalar scaling", 9.4),
            ("Isolating a few extreme-norm outlier neurons explains sudden sensitivity to weight perturbation", 9.0),
            ("Assuming activations follow an isotropic Gaussian distribution with uniform variance across channels", 1.5),
            ("Distributing compute evenly across identical uniform tiles with balanced load", 1.2),
            ("Formatting multi-turn dialogue prompts with consistent role tags and JSON schemas", 1.0),
            ("Collecting balanced multiple-choice questions across ten academic subjects", 1.0),
        ],
    ),
    (
        "depth_early_exit",
        " early-exit",
        " full-depth",
        [
            ("Truncating transformer execution at layer 12 of 36 skips upper blocks while preserving ranking accuracy", 10.0),
            ("Layer-skipping self-speculative decoding drafts tokens from shallow layers and verifies with deeper blocks", 9.8),
            ("Semantic entity representations crystallize at 35 percent depth long before final unembedding layers", 9.5),
            ("Dynamic depth routing exits easy tokens early to avoid redundant upper-layer computation", 9.2),
            ("Requiring every token to traverse all 80 transformer layers and full vocabulary projection", 1.5),
            ("Scaling horizontal mixture-of-experts width while keeping sequential layer count fixed", 2.0),
            ("Tuning tokenizer vocabulary size from 32k to 128k subwords for multilingual compression", 1.0),
            ("Benchmarking synthetic reward hacking in multi-agent negotiation games", 1.0),
        ],
    ),
    (
        "causal_intervention",
        " interventional",
        " observational",
        [
            ("Patching a single hidden activation from a counterfactual prompt causally flips the model's answer", 10.0),
            ("Adding a calibrated steering vector during forward execution induces controlled behavioral shifts", 9.6),
            ("Zero-ablating specific attention heads tests necessity and sufficiency of a proposed circuit", 9.5),
            ("Forcing counterfactual intermediate reasoning steps verifies whether a chain of thought is causally faithful", 9.2),
            ("Measuring passive correlation between output token length and benchmark accuracy", 1.5),
            ("Visualizing static 2D UMAP projections of document embeddings without modifying model states", 1.2),
            ("Reporting aggregate top-1 accuracy tables across standard public leaderboards", 1.0),
            ("Optimizing PCIe host-to-device DMA transfer overlap during checkpoint loading", 1.0),
        ],
    ),
    (
        "iterative_recurrence",
        " iterative",
        " single-pass",
        [
            ("Looping a shared weight block multiple times at inference time scales reasoning depth dynamically", 10.0),
            ("Iteratively refining draft outputs through verifier-guided search improves math accuracy with test-time compute", 9.6),
            ("Fixed-point latent recurrence converges to deeper algorithmic solutions without adding parameters", 9.5),
            ("Multi-step discrete diffusion denoising iteratively revises parallel token predictions", 9.2),
            ("Computing a single deterministic feed-forward pass with fixed non-recurrent depth", 1.5),
            ("Static one-shot post-training quantization using a single calibration batch", 1.5),
            ("Precomputing fixed static lookup tables for fast integer arithmetic", 1.0),
            ("Evaluating zero-shot sentiment classification with a single greedy token", 1.2),
        ],
    ),
    (
        "memory_eviction_tiling",
        " memory-bound",
        " compute-bound",
        [
            ("Evicting low-importance KV-cache entries under a fixed memory budget enables million-token serving", 10.0),
            ("Fusing tiled SRAM loads and online softmax eliminates intermediate HBM materialization", 9.8),
            ("Paged virtual memory allocation removes fragmentation in batched autoregressive serving", 9.4),
            ("Adaptive per-head cache budget allocation prioritizes retrieval heads over local window heads", 9.2),
            ("Analyzing theoretical VC dimension and Rademacher complexity of infinite-width networks", 1.0),
            ("Designing human preference rubrics for creative writing and roleplay alignment", 1.0),
            ("Measuring semantic drift in sparse autoencoder features across pretraining checkpoints", 1.5),
            ("Evaluating political bias in open-ended essay generation", 1.0),
        ],
    ),
    (
        "symmetry_debiasing",
        " debiased",
        " biased",
        [
            ("Averaging antisymmetric paired swaps (A,B) and (B,A) exactly cancels prompt order bias", 10.0),
            ("Subtracting the global lexical prior offset eliminates systematic Yes/No preference skew", 9.6),
            ("Enforcing permutation invariance removes sensitivity to arbitrary candidate ordering in judges", 9.5),
            ("Rotary equivariance and norm centering prevent positional drift across long sequences", 9.0),
            ("Evaluating candidates in a fixed arbitrary list order with raw greedy logits", 1.5),
            ("Maximizing throughput via FP8 tensor-core warp specialization on Blackwell GPUs", 1.0),
            ("Scraping web corpora and deduplicating URLs with MinHash", 1.0),
            ("Training a diffusion video generator on captioned movie clips", 1.0),
        ],
    ),
]

# Anchor pairs for Method 3 (Contradiction Core calibration): 5 genuine conflicts vs 5 consistent/complementary pairs
CONTRADICTION_CAL_PAIRS: List[Tuple[str, str, float]] = [
    # Contradictions (1.0)
    (
        "Attention weights faithfully identify which input tokens causally drive a transformer's prediction.",
        "Attention weights are uncorrelated with gradient-based feature importance and do not constitute causal explanation.",
        1.0,
    ),
    (
        "Increasing chain-of-thought reasoning length monotonically improves accuracy on complex logic benchmarks.",
        "Longer chain-of-thought traces systematically degrade reasoning accuracy due to overthinking and compounding errors.",
        1.0,
    ),
    (
        "Sparse autoencoders recover canonical ground-truth monosemantic concepts cleanly as dictionary width scales.",
        "Sparse autoencoders fail to converge to canonical concepts and exhibit pervasive feature splitting and absorption as width grows.",
        1.0,
    ),
    (
        "Evicting KV-cache tokens solely by cumulative attention score preserves full long-context retrieval accuracy.",
        "Heavy-hitter attention eviction catastrophically fails on needle-in-a-haystack retrieval because rare keys have low cumulative scores.",
        1.0,
    ),
    (
        "Larger language models become strictly better calibrated and less sycophantic with scale.",
        "Scaling model size increases sycophantic agreement with user misconceptions and worsens calibration under adversarial framing.",
        1.0,
    ),
    # Consistent / Complementary (0.0)
    (
        "FlashAttention tiles attention blocks in GPU SRAM to reduce HBM memory traffic without changing exact outputs.",
        "FlashAttention-2 improves GPU occupancy by parallelizing over sequence length and reducing non-matmul FLOPs.",
        0.0,
    ),
    (
        "Speculative decoding uses a small draft model to propose candidate tokens verified in parallel by a target LLM.",
        "Tree-structured speculative verification extends draft sequences into token trees to raise expected accepted length.",
        0.0,
    ),
    (
        "INT4 weight quantization reduces memory bandwidth bottlenecks during single-batch autoregressive decoding.",
        "Activation-aware channel scaling protects salient weights during INT4 quantization to preserve perplexity.",
        0.0,
    ),
    (
        "Induction heads implement in-context pattern copying via a two-layer previous-token and matching-head composition.",
        "During pretraining, a sharp phase transition coincides with the emergence of induction heads and improved in-context loss.",
        0.0,
    ),
    (
        "Direct Preference Optimization optimizes policy ratios directly on preference pairs without an explicit reward model.",
        "SimPO simplifies preference optimization by using length-normalized average log-probability as the implicit reward.",
        0.0,
    ),
]


# ==============================================================================
# Text Utilities (Hype Stripping & Core Claim Extraction)
# ==============================================================================

HYPE_PATTERNS = [
    r"\bsurprising(ly)?\b",
    r"\bunexpected(ly)?\b",
    r"\bcounterintuitive(ly)?\b",
    r"\bcontrary to (common|conventional|popular|prior|existing|prevailing) (belief|wisdom|intuition|assumptions?)\b",
    r"\bremarkabl[ey]\b",
    r"\binterestingly\b",
    r"\bto our surprise\b",
    r"\bparadigm[- ]shifting\b",
    r"\brevolutionary\b",
    r"\bbreakthroughs?\b",
    r"\bunprecedented\b",
    r"\bgroundbreaking\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bSOTA\b",
    r"\bnovel\b",
    r"\binnovative\b",
    r"\bfor the first time\b",
    r"\bchallenging (the )?(conventional|prevailing|common) (wisdom|view|assumption)\b",
    r"\bfor the first time ever\b",
    r"\bdramatically\b",
    r"\bsignificantly\b",
    r"\bvastly\b",
    r"\bsubstantially\b",
]
HYPE_REGEX = re.compile("|".join(HYPE_PATTERNS), flags=re.IGNORECASE)

FINDING_VERBS = re.compile(
    r"\b(we (show|find|demonstrate|observe|reveal|prove|discover|uncover|report|verify|establish)|"
    r"our (experiments?|results?|analysis|findings?|evaluations?) (show|reveal|demonstrate|indicate|confirm)|"
    r"outperforms?|reduces?|achieves?|degrades?|fails?|collapses?|causes?|preserves?|eliminates?|improves?)\b",
    flags=re.IGNORECASE,
)


def strip_hype(text: str) -> str:
    s = HYPE_REGEX.sub("", text)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"^\s*[,;:\-]+\s*", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def extract_clean_claim(paper: Dict[str, Any], max_chars: int = 360) -> str:
    """Extract the most concrete empirical finding sentences from abstract, with hype framing stripped."""
    abs_text = (paper.get("abstract") or "").strip()
    if abs_text.endswith("..."):
        abs_text = abs_text[:-3].rsplit(".", 1)[0] + "." if "." in abs_text[:-3] else abs_text[:-3]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abs_text) if len(s.strip()) > 25]
    if not sents:
        return strip_hype(paper.get("title", ""))[:max_chars]

    # Prefer sentences expressing empirical findings / results (skip pure background sentence 0 when possible)
    finding_sents = [s for idx, s in enumerate(sents) if FINDING_VERBS.search(s) and (idx > 0 or len(sents) == 1)]
    if not finding_sents:
        finding_sents = sents[1:] if len(sents) > 1 else sents

    chosen = " ".join(finding_sents[:2])
    cleaned = strip_hype(chosen)
    if len(cleaned) < 40:
        cleaned = strip_hype(f"{paper.get('title', '')}. {abs_text[:280]}")
    return cleaned[:max_chars].strip()


def format_paper_short(paper: Dict[str, Any], max_chars: int = 560) -> str:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    txt = f"{title}. {abstract}"
    return " ".join(txt.split())[:max_chars]


def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    sd = float(np.std(arr))
    return (arr - float(np.mean(arr))) / (sd if sd > 1e-12 else 1.0)


def jaccard_title(t1: str, t2: str) -> float:
    w1 = set(re.findall(r"[a-z0-9]{3,}", t1.lower()))
    w2 = set(re.findall(r"[a-z0-9]{3,}", t2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ==============================================================================
# Helper: Calibrate a Symmetric Pairwise Stage-2 Core (for Contradiction / Transfer)
# ==============================================================================

def calibrate_stage2_pair_core(
    engine: CoreRankEngine,
    cal_pairs: Sequence[Tuple[str, str, float]],
    template: str,
    pos_word: str,
    neg_word: str,
    candidate_depths: Sequence[float] = (0.50, 0.57, 0.64, 0.71, 0.78),
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """
    Calibrate a Stage-2 END-token core direction q_end for binary pair classification (e.g. Contradiction).
    Uses paired differences between positive (contradictory) and negative (consistent) pairs at the END token,
    combined with exact margin gradients via `compute_ace_subspace`.
    """
    prompts, labels = [], []
    for a, b, y in cal_pairs:
        prompts.append(template.format(A=a, B=b))
        labels.append(y)
        prompts.append(template.format(A=b, B=a))
        labels.append(y)

    pos_id = engine.resolve_token_id(pos_word, sample_prompt=prompts[0])
    neg_id = engine.resolve_token_id(neg_word, sample_prompt=prompts[0])
    y_arr = np.array(labels, dtype=np.int64)
    pos_idxs = np.where(y_arr == 1)[0]
    neg_idxs = np.where(y_arr == 0)[0]

    cand_layers = sorted(set(max(1, min(engine.n_layers, int(round(d * engine.n_layers)))) for d in candidate_depths))
    best = None

    for l in cand_layers:
        H_l, G_l, M_l = engine.collect_acts_and_grads(prompts, l, pos_id, neg_id, spans=None, pooling="last")
        # Pairwise class-contrastive differences (every positive minus every negative)
        H_diff = np.array([H_l[i] - H_l[j] for i in pos_idxs for j in neg_idxs])
        Q_l, gap, _ = compute_ace_subspace(H_diff, G_l, rank=1)
        q1 = Q_l[:, 0]
        mu_l = H_l.mean(axis=0)
        z = (H_l - mu_l) @ q1
        auc = roc_auc(y_arr, z)
        if auc < 0.5:
            q1 = -q1
            auc = 1.0 - auc
        if best is None or auc > best[3] or (abs(auc - best[3]) < 1e-6 and gap > best[4]):
            best = (l, q1, mu_l, auc, gap)

    return best  # (layer, q1, mu, auc, gap)


@torch.no_grad()
def score_stage2_keep1d_pairs(
    engine: CoreRankEngine,
    pair_texts: Sequence[Tuple[str, str]],
    template: str,
    pos_word: str,
    neg_word: str,
    stage2_layer: int,
    q_end: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Score a list of (textA, textB) pairs in BOTH orders (A,B) and (B,A) using:
      1) Raw full-model logit margins m_raw_ab, m_raw_ba
      2) Stage-2 Keep-1D projected margins m_keep_ab, m_keep_ba (de-biased by subtracting global mean prior)
      3) Stage-2 1D core coordinate z_ab, z_ba (early-exit fast coordinate)
    """
    prompts_ab = [template.format(A=a, B=b) for a, b in pair_texts]
    prompts_ba = [template.format(A=b, B=a) for a, b in pair_texts]
    all_prompts = prompts_ab + prompts_ba
    n = len(pair_texts)

    pos_id = engine.resolve_token_id(pos_word, sample_prompt=all_prompts[0])
    neg_id = engine.resolve_token_id(neg_word, sample_prompt=all_prompts[0])

    # Raw margins
    m_raw = engine.get_margins(all_prompts, pos_id, neg_id)

    # Collect END hidden states at stage2_layer via early-exit
    H_end = engine.extract_early_exit(all_prompts, stage2_layer, spans=None, pooling="last")
    mu_end = H_end.mean(axis=0)
    z_coord = (H_end - mu_end) @ q_end

    # Stage-2 Keep-1D forward pass
    q_t = torch.tensor(q_end, dtype=torch.float32, device=engine.device)
    mu_t = torch.tensor(mu_end, dtype=torch.float32, device=engine.device)
    m_keep_list = []
    for start in range(0, len(all_prompts), engine.batch_size):
        batch = all_prompts[start : start + engine.batch_size]
        input_ids, attn_mask = engine._tokenize_batch(batch)
        B, T = input_ids.shape
        b_ar = torch.arange(B, device=engine.device)

        def hook(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).clone()
            hs = h[b_ar, T - 1, :].float()
            proj = ((hs - mu_t) @ q_t).unsqueeze(-1) * q_t
            h[b_ar, T - 1, :] = (mu_t + proj).to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        hdl = engine.layers[stage2_layer - 1].register_forward_hook(hook)
        try:
            logits = engine.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, -1, :]
            m_keep_list.append((logits[:, pos_id] - logits[:, neg_id]).float().cpu().numpy())
        finally:
            hdl.remove()

    m_keep = np.concatenate(m_keep_list, axis=0)
    # Remove global lexical prior offset
    m_keep_deb = m_keep - float(np.mean(m_keep))
    m_raw_deb = m_raw - float(np.mean(m_raw))

    return {
        "raw_ab": m_raw[:n],
        "raw_ba": m_raw[n:],
        "raw_sym": 0.5 * (m_raw_deb[:n] + m_raw_deb[n:]),
        "raw_order_abs_diff": np.abs(m_raw[:n] - m_raw[n:]),
        "keep_ab": m_keep[:n],
        "keep_ba": m_keep[n:],
        "keep_sym": 0.5 * (m_keep_deb[:n] + m_keep_deb[n:]),
        "keep_order_abs_diff": np.abs(m_keep[:n] - m_keep[n:]),
        "z_sym": 0.5 * (z_coord[:n] + z_coord[n:]),
    }


@torch.no_grad()
def compute_claim_nll(engine: CoreRankEngine, contexts: Sequence[str], claims: Sequence[str]) -> np.ndarray:
    """Compute exact mean negative log-likelihood (nats/token) of `claim` conditioned on `context`."""
    nlls = []
    for ctx, clm in zip(contexts, claims):
        prefix_ids = engine.tokenizer.encode(ctx, add_special_tokens=True)
        full_ids = engine.tokenizer.encode(ctx + clm, add_special_tokens=True)
        start = len(prefix_ids)
        if len(full_ids) <= start + 1:
            nlls.append(0.0)
            continue
        inp = torch.tensor([full_ids], dtype=torch.long, device=engine.device)
        logits = engine.model(input_ids=inp, use_cache=False).logits[0]  # (T, V)
        # Predict tokens start .. T-1 from positions start-1 .. T-2
        shift_logits = logits[start - 1 : -1].float()
        shift_labels = inp[0, start:]
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="mean")
        nlls.append(float(loss.item()))
    return np.asarray(nlls, dtype=np.float64)


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] Reading papers DB from {PAPERS_JSON}...", flush=True)
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers = db["papers"]
    N = len(papers)
    bge_vecs = np.fromfile(VECTORS_BIN, dtype=np.float16).astype(np.float32).reshape(N, 384)
    bge_vecs /= np.linalg.norm(bge_vecs, axis=1, keepdims=True) + 1e-12
    S_bge = bge_vecs @ bge_vecs.T
    np.fill_diagonal(S_bge, -1.0)
    print(f"[load] Loaded {N} papers + {bge_vecs.shape} BGE vectors.", flush=True)

    print(f"[model] Loading {MODEL_ID} in bf16 on RTX 5090...", flush=True)
    engine = CoreRankEngine(MODEL_ID, dtype="bf16", batch_size=8)
    print(
        f"[model] Loaded {MODEL_ID} ({engine.n_layers}L, d={engine.d_model}) in {engine.load_sec:.1f}s | "
        f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB",
        flush=True,
    )

    results: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "paper_count": N,
        "stage0_audit": {},
        "method1_bayesian_surprise": {},
        "method2_cross_domain_bridges": {},
        "method3_dense_contradictions": {},
        "method4_residuals_and_outliers": {},
    }

    # ==========================================================================
    # STAGE 0: Calibrate Axes + V-Dip Audit + Adversarial Hype Test
    # ==========================================================================
    print("\n================ STAGE 0: CALIBRATION & V-DIP AUDIT ================", flush=True)
    t0 = time.time()

    surprise_axis = engine.calibrate_axis(
        name="surprise_core",
        rubric="Hype-free scientific surprisingness / counterintuitiveness",
        train_items=SURPRISE_CAL_ITEMS,
        train_templates=[
            "Evaluating scientific expectation, Claim A ({A}) compared to Claim B ({B}) is",
            "Comparing how counterintuitive each technical claim is, Claim A ({A}) relative to Claim B ({B}) is",
        ],
        pointwise_template="Technical claim: {A}\nAssessment:",
        pos_word=" surprising",
        neg_word=" obvious",
        pooling="span_mean",
        rank=1,
        candidate_depths=(0.28, 0.36, 0.43, 0.50, 0.57, 0.64),
    )

    rigor_items = [(t, r) for t, r, _ in RIGOR_HYPE_CAL_DATA]
    hype_items = [(t, h) for t, _, h in RIGOR_HYPE_CAL_DATA]

    rigor_axis = engine.calibrate_axis(
        name="rigor_core",
        rubric="Anti-fluff empirical & mathematical technical rigor",
        train_items=rigor_items,
        train_templates=[
            "Evaluating concrete technical rigor and evidence (ignoring buzzwords), Abstract A ({A}) compared to Abstract B ({B}) is",
            "Comparing empirical substance and mechanistic depth, Abstract A ({A}) relative to Abstract B ({B}) is",
        ],
        pointwise_template="Research paper: {A}\nTechnical evaluation:",
        pos_word=" rigorous",
        neg_word=" superficial",
        pooling="span_mean",
        rank=2,
        candidate_depths=(0.32, 0.39, 0.46, 0.54, 0.61),
    )

    hype_axis = engine.calibrate_axis(
        name="hype_core",
        rubric="Rhetorical hype and buzzword inflation",
        train_items=hype_items,
        train_templates=[
            "Evaluating rhetorical hype and marketing buzzwords, Abstract A ({A}) compared to Abstract B ({B}) is",
        ],
        pointwise_template="Research paper: {A}\nTechnical evaluation:",
        pos_word=" hyped",
        neg_word=" understated",
        pooling="span_mean",
        rank=1,
        candidate_depths=(0.32, 0.39, 0.46, 0.54),
    )

    prior_axis = engine.calibrate_axis(
        name="prior_plausibility_core",
        rubric="Pretrained ML consensus plausibility prior",
        train_items=PRIOR_PLAUSIBILITY_ITEMS,
        train_templates=[
            "Evaluating prior scientific plausibility in machine learning, Claim A ({A}) compared to Claim B ({B}) is",
        ],
        pointwise_template="Technical claim: {A}\nAssessment:",
        pos_word=" plausible",
        neg_word=" implausible",
        pooling="span_mean",
        rank=1,
        candidate_depths=(0.28, 0.36, 0.43, 0.50, 0.57),
    )

    for ax in (surprise_axis, rigor_axis, hype_axis, prior_axis):
        print(
            f"  [cal] {ax.name:24s} -> L{ax.layer:2d}/{ax.n_layers} ({ax.layer/ax.n_layers:.2f}) | "
            f"gap={ax.spectral_gap:8.1f} | train_rho={ax.train_rho:.4f}",
            flush=True,
        )

    # Also extract rank-8 rigor ACE subspace at rigor_axis.layer for Method 4 off-core causal energy!
    # Re-collect activations/grads on rigor calibration pairs for rank=8 Q
    rigor_pr, rigor_swaps, rigor_spans = [], [], []
    rtmpl = "Evaluating concrete technical rigor and evidence (ignoring buzzwords), Abstract A ({A}) compared to Abstract B ({B}) is"
    for i in range(len(rigor_items)):
        for j in range(i + 1, len(rigor_items)):
            a, b = rigor_items[i][0], rigor_items[j][0]
            ia = len(rigor_pr)
            pa = rtmpl.format(A=a, B=b)
            rigor_pr.append(pa)
            rigor_spans.append(engine.find_entity_span(pa, a))
            ib = len(rigor_pr)
            pb = rtmpl.format(A=b, B=a)
            rigor_pr.append(pb)
            rigor_spans.append(engine.find_entity_span(pb, b))
            rigor_swaps.append((ia, ib))
    r_pos_id = engine.resolve_token_id(" rigorous", sample_prompt=rigor_pr[0])
    r_neg_id = engine.resolve_token_id(" superficial", sample_prompt=rigor_pr[0])
    H_r8, G_r8, _ = engine.collect_acts_and_grads(rigor_pr, rigor_axis.layer, r_pos_id, r_neg_id, spans=rigor_spans, pooling="span_mean")
    H_r8_diff = np.array([H_r8[a] - H_r8[b] for a, b in rigor_swaps])
    Q_rigor8, gap_r8, s2_r8 = compute_ace_subspace(H_r8_diff, G_r8, rank=8)
    mu_rigor8 = H_r8.mean(axis=0)
    print(f"  [cal] rigor_rank8 top-8 sigma^2: {[round(x, 4) for x in s2_r8[:6]]}", flush=True)

    # V-Dip Conceptual Bottleneck Audits
    print("  [vdip] Running V-Dip audit on surprise_core (5 lexicons)...", flush=True)
    vdip_surprise = audit_vdip(
        engine,
        items=SURPRISE_CAL_ITEMS[:10],
        lexicons=SURPRISE_VDIP_LEXICONS,
        source_key="surprising_obvious",
    )
    print(f"  [vdip] surprise_core -> {vdip_surprise['verdict']} | {vdip_surprise['explanation']}", flush=True)

    print("  [vdip] Running V-Dip audit on rigor_core (5 lexicons)...", flush=True)
    vdip_rigor = audit_vdip(
        engine,
        items=rigor_items[:10],
        lexicons=RIGOR_VDIP_LEXICONS,
        source_key="rigorous_superficial",
    )
    print(f"  [vdip] rigor_core    -> {vdip_rigor['verdict']} | {vdip_rigor['explanation']}", flush=True)

    # Adversarial Hype Resistance Test
    print("  [adv] Testing Adversarial Hype Resistance (6 plain counterintuitive vs hyped truisms)...", flush=True)
    adv_plain = [p for p, _ in ADVERSARIAL_HYPE_PAIRS]
    adv_hyped = [h for _, h in ADVERSARIAL_HYPE_PAIRS]
    adv_hyped_stripped = [strip_hype(h) for h in adv_hyped]

    # Raw LLM pairwise prompts (both orders)
    adv_tmpl = "Evaluating scientific expectation, Claim A ({A}) compared to Claim B ({B}) is"
    sp_id = engine.resolve_token_id(" surprising", sample_prompt=adv_tmpl.format(A="x", B="y"))
    ob_id = engine.resolve_token_id(" obvious", sample_prompt=adv_tmpl.format(A="x", B="y"))
    raw_ab = engine.get_margins([adv_tmpl.format(A=p, B=h) for p, h in ADVERSARIAL_HYPE_PAIRS], sp_id, ob_id)
    raw_ba = engine.get_margins([adv_tmpl.format(A=h, B=p) for p, h in ADVERSARIAL_HYPE_PAIRS], sp_id, ob_id)
    raw_sym = 0.5 * (raw_ab - raw_ba)  # >0 means raw LLM correctly says plain counterintuitive is more surprising
    raw_adv_acc = float(np.mean(raw_sym > 0))
    raw_pos_bias = float(np.mean((raw_ab > 0) == (raw_ba > 0)))

    # Stage-1 CoreRank scores on raw text vs hype-stripped text
    sc_plain, _ = engine.score_candidates(surprise_axis, adv_plain)
    sc_hyped, _ = engine.score_candidates(surprise_axis, adv_hyped)
    sc_hyped_strip, _ = engine.score_candidates(surprise_axis, adv_hyped_stripped)
    pr_plain, _ = engine.score_candidates(prior_axis, adv_plain)
    pr_hyped, _ = engine.score_candidates(prior_axis, adv_hyped)
    pr_hyped_strip, _ = engine.score_candidates(prior_axis, adv_hyped_stripped)

    core_adv_acc = float(np.mean(sc_plain > sc_hyped))
    core_strip_adv_acc = float(np.mean(sc_plain > sc_hyped_strip))
    prior_adv_acc = float(np.mean(pr_plain < pr_hyped))
    prior_strip_adv_acc = float(np.mean(pr_plain < pr_hyped_strip))
    print(
        f"  [adv] Raw LLM pairwise acc={raw_adv_acc*100:.1f}% (pos_bias={raw_pos_bias*100:.1f}%) | "
        f"Stage-1 surprise_core acc={core_adv_acc*100:.1f}% (stripped={core_strip_adv_acc*100:.1f}%) | "
        f"Stage-1 prior_core acc={prior_adv_acc*100:.1f}% (stripped={prior_strip_adv_acc*100:.1f}%)",
        flush=True,
    )

    results["stage0_audit"] = {
        "elapsed_sec": round(time.time() - t0, 1),
        "axes": {
            ax.name: {
                "layer": ax.layer,
                "rel_depth": round(ax.layer / ax.n_layers, 3),
                "spectral_gap": round(ax.spectral_gap, 2),
                "train_rho": round(ax.train_rho, 4),
            }
            for ax in (surprise_axis, rigor_axis, hype_axis, prior_axis)
        },
        "vdip_surprise": {
            "verdict": vdip_surprise["verdict"],
            "explanation": vdip_surprise["explanation"],
            "clean_aucs": vdip_surprise["clean_aucs"],
            "trough_layer": vdip_surprise["trough_layer"],
            "trough_rel_depth": vdip_surprise["trough_rel_depth"],
            "trough_cross_flip_mean": vdip_surprise["trough_cross_flip_mean"],
            "final_cross_flip_mean": vdip_surprise["final_cross_flip_mean"],
            "rebound_delta": vdip_surprise["rebound_delta"],
        },
        "vdip_rigor": {
            "verdict": vdip_rigor["verdict"],
            "explanation": vdip_rigor["explanation"],
            "clean_aucs": vdip_rigor["clean_aucs"],
            "trough_layer": vdip_rigor["trough_layer"],
            "trough_rel_depth": vdip_rigor["trough_rel_depth"],
            "trough_cross_flip_mean": vdip_rigor["trough_cross_flip_mean"],
            "final_cross_flip_mean": vdip_rigor["final_cross_flip_mean"],
            "rebound_delta": vdip_rigor["rebound_delta"],
        },
        "adversarial_hype_test": {
            "raw_llm_acc": round(raw_adv_acc, 4),
            "raw_pos_bias_rate": round(raw_pos_bias, 4),
            "stage1_surprise_acc": round(core_adv_acc, 4),
            "stage1_surprise_stripped_acc": round(core_strip_adv_acc, 4),
            "stage1_prior_acc": round(prior_adv_acc, 4),
            "stage1_prior_stripped_acc": round(prior_strip_adv_acc, 4),
        },
    }

    # ==========================================================================
    # STAGE 1 (O(N) Early-Exit Corpus Scoring + Method 1: Bayesian Surprise)
    # ==========================================================================
    print("\n================ METHOD 1: PRIOR-VS-EVIDENCE BAYESIAN SURPRISE ================", flush=True)
    t1 = time.time()

    full_texts = [format_paper_short(p, max_chars=560) for p in papers]
    clean_claims = [extract_clean_claim(p, max_chars=340) for p in papers]

    # Single shared early-exit pass for rigor + hype on full_texts
    # Also capture the full hidden vectors at rigor_axis.layer for Method 2 residual & Method 4 off-core energy!
    paper_prompts = [rigor_axis.template.format(A=t) for t in full_texts]
    paper_spans = [engine.find_entity_span(pr, t) for pr, t in zip(paper_prompts, full_texts)]
    H_papers_rigor = engine.extract_early_exit(paper_prompts, rigor_axis.layer, spans=paper_spans, pooling=rigor_axis.pooling)
    z_rigor = rigor_axis.score_vectors(H_papers_rigor)

    if hype_axis.layer == rigor_axis.layer and hype_axis.pooling == rigor_axis.pooling:
        z_hype = hype_axis.score_vectors(H_papers_rigor)
    else:
        z_hype, _ = engine.score_candidates(hype_axis, full_texts)

    # Single shared early-exit pass for surprise + prior on hype-stripped clean claims
    claim_scores, claim_sec = engine.score_multi_rubric([surprise_axis, prior_axis], clean_claims)
    z_surprise = claim_scores["surprise_core"]
    z_prior = claim_scores["prior_plausibility_core"]

    print(
        f"  [m1] Scored all {N} papers in {time.time() - t1:.1f}s (early-exit L{rigor_axis.layer}/L{surprise_axis.layer}/L{prior_axis.layer})!",
        flush=True,
    )

    zr = zscore(z_rigor)
    zh = zscore(z_hype)
    zs = zscore(z_surprise)
    zp = zscore(z_prior)

    # Correlations sanity check
    has_hype_cue = np.array([1.0 if HYPE_REGEX.search(p.get("abstract") or "") else 0.0 for p in papers])
    print(
        f"  [m1] Corr(z_rigor, z_hype)={np.corrcoef(zr, zh)[0,1]:+.3f} | "
        f"Corr(z_surprise_clean, z_prior_clean)={np.corrcoef(zs, zp)[0,1]:+.3f} | "
        f"Corr(z_surprise_clean, raw_hype_cue)={np.corrcoef(zs, has_hype_cue)[0,1]:+.3f} | "
        f"Corr(z_hype, raw_hype_cue)={np.corrcoef(zh, has_hype_cue)[0,1]:+.3f}",
        flush=True,
    )

    # Bayesian Surprise ranking: low prior plausibility + high surprise + high rigor - hype
    m1_score = -1.0 * zp + 0.8 * zs + 0.9 * zr - 0.6 * zh
    # Require positive rigor and non-extreme hype
    m1_eligible = np.where((zr > 0.35) & (zh < 1.0))[0]
    m1_top_idxs = m1_eligible[np.argsort(-m1_score[m1_eligible])[:30]]

    # Also pick 10 high-citation "common knowledge" controls for comparison
    cite_arr = np.array([p.get("citations", 0) for p in papers], dtype=np.float64)
    ctrl_idxs = np.argsort(-cite_arr)[:10]

    # Compute exact LM Bayesian Surprise (NLL_prior - NLL_posterior) AND antisymmetric prior belief shift on top 30 + 10 controls!
    eval_idxs = list(m1_top_idxs) + [int(i) for i in ctrl_idxs if i not in set(m1_top_idxs)]
    eval_claims = [clean_claims[i] for i in eval_idxs]
    prior_ctx = ["In machine learning and systems research, it is known that " for _ in eval_idxs]
    post_ctx = [f"Paper title: {papers[i]['title']}\nSummary: {papers[i]['abstract'][:450]}\nKey empirical finding: " for i in eval_idxs]

    nll_prior = compute_claim_nll(engine, prior_ctx, eval_claims)
    nll_post = compute_claim_nll(engine, post_ctx, eval_claims)
    nll_drop = nll_prior - nll_post  # Information gain (bits/nats gained from reading the paper!)

    # Antisymmetric belief prior on the clean claim alone (no context): True vs False
    tp_id = engine.resolve_token_id(" True", sample_prompt="Statement: x\nVerdict:")
    fp_id = engine.resolve_token_id(" False", sample_prompt="Statement: x\nVerdict:")
    p_tf = engine.get_margins([f"Evaluate whether this machine learning claim is True or False based on standard knowledge.\nClaim: {c}\nAnswer:" for c in eval_claims], tp_id, fp_id)
    p_ft = engine.get_margins([f"Evaluate whether this machine learning claim is False or True based on standard knowledge.\nClaim: {c}\nAnswer:" for c in eval_claims], tp_id, fp_id)
    prior_belief = 0.5 * (p_tf + p_ft)

    eval_map = {idx: k for k, idx in enumerate(eval_idxs)}
    m1_records = []
    for idx in m1_top_idxs[:20]:
        k = eval_map[int(idx)]
        p = papers[int(idx)]
        m1_records.append({
            "idx": int(idx),
            "id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "year": p["year"],
            "venue": p.get("venue", ""),
            "citations": p.get("citations", 0),
            "m1_score": round(float(m1_score[idx]), 3),
            "z_prior": round(float(zp[idx]), 3),
            "z_surprise": round(float(zs[idx]), 3),
            "z_rigor": round(float(zr[idx]), 3),
            "z_hype": round(float(zh[idx]), 3),
            "nll_prior": round(float(nll_prior[k]), 3),
            "nll_post": round(float(nll_post[k]), 3),
            "info_gain_nats": round(float(nll_drop[k]), 3),
            "llm_prior_belief_margin": round(float(prior_belief[k]), 3),
            "clean_claim": clean_claims[int(idx)],
            "abstract": p.get("abstract", ""),
        })

    ctrl_nll_prior = float(np.mean([nll_prior[eval_map[int(i)]] for i in ctrl_idxs]))
    top_nll_prior = float(np.mean([nll_prior[eval_map[int(i)]] for i in m1_top_idxs[:20]]))
    ctrl_belief = float(np.mean([prior_belief[eval_map[int(i)]] for i in ctrl_idxs]))
    top_belief = float(np.mean([prior_belief[eval_map[int(i)]] for i in m1_top_idxs[:20]]))
    print(
        f"  [m1] Top-20 vs Top-10-Cited Controls: mean NLL_prior={top_nll_prior:.3f} vs {ctrl_nll_prior:.3f} | "
        f"mean LLM prior belief={top_belief:+.3f} vs {ctrl_belief:+.3f}",
        flush=True,
    )
    for r in m1_records[:8]:
        print(
            f"    -> [{r['category']}|{r['year']}|c={r['citations']}] {r['title'][:72]} | "
            f"M1={r['m1_score']:.2f} (prior={r['z_prior']:+.2f}, surp={r['z_surprise']:+.2f}, rig={r['z_rigor']:+.2f}, nll={r['nll_prior']:.2f})",
            flush=True,
        )

    results["method1_bayesian_surprise"] = {
        "elapsed_sec": round(time.time() - t1, 1),
        "correlations": {
            "rigor_vs_hype": round(float(np.corrcoef(zr, zh)[0, 1]), 4),
            "surprise_vs_prior": round(float(np.corrcoef(zs, zp)[0, 1]), 4),
            "surprise_clean_vs_raw_hype_cue": round(float(np.corrcoef(zs, has_hype_cue)[0, 1]), 4),
            "hype_core_vs_raw_hype_cue": round(float(np.corrcoef(zh, has_hype_cue)[0, 1]), 4),
        },
        "top20_vs_cited_controls": {
            "top20_mean_nll_prior": round(top_nll_prior, 4),
            "controls_mean_nll_prior": round(ctrl_nll_prior, 4),
            "top20_mean_prior_belief": round(top_belief, 4),
            "controls_mean_prior_belief": round(ctrl_belief, 4),
        },
        "top_discoveries": m1_records,
    }

    # ==========================================================================
    # METHOD 2: Cross-Domain Mechanism Bridges (8-Axis Topic-Canceled ACE Subspace)
    # ==========================================================================
    print("\n================ METHOD 2: CROSS-DOMAIN MECHANISM BRIDGES ================", flush=True)
    t2 = time.time()

    mech_axes: List[CoreAxis] = []
    mech_cal_summary = {}
    for m_name, m_pos, m_neg, m_items in MECHANISM_SPECS:
        ax = engine.calibrate_axis(
            name=f"mech_{m_name}",
            rubric=f"Domain-invariant mechanism axis: {m_name}",
            train_items=m_items,
            train_templates=[
                f"Ignoring application domain, comparing underlying mathematical mechanism, Method A ({{A}}) compared to Method B ({{B}}) is",
            ],
            pointwise_template="Research paper: {A}\nUnderlying mechanism:",
            pos_word=m_pos,
            neg_word=m_neg,
            pooling="span_mean",
            rank=1,
            candidate_depths=(0.32, 0.39, 0.46, 0.54),
        )
        mech_axes.append(ax)
        mech_cal_summary[m_name] = {
            "layer": ax.layer,
            "gap": round(ax.spectral_gap, 1),
            "train_rho": round(ax.train_rho, 4),
        }
        print(f"  [m2-cal] {m_name:22s} -> L{ax.layer:2d} | gap={ax.spectral_gap:7.1f} | rho={ax.train_rho:.3f}", flush=True)

    # Single shared early-exit multi-rubric pass over all 1,715 papers across all 8 mechanism axes!
    mech_scores_dict, m2_pass_sec = engine.score_multi_rubric(mech_axes, full_texts)
    Z_mech = np.stack([zscore(mech_scores_dict[f"mech_{m[0]}"]) for m in MECHANISM_SPECS], axis=1)  # (N, 8)
    mech_norms = np.linalg.norm(Z_mech, axis=1)
    Z_mech_unit = Z_mech / (mech_norms[:, None] + 1e-9)
    S_mech = Z_mech_unit @ Z_mech_unit.T
    np.fill_diagonal(S_mech, -1.0)

    # Also build BGE-orthogonal Stage-1 residual matrix (projecting out all 384 BGE topic dims from H_papers_rigor!)
    Hc = H_papers_rigor - H_papers_rigor.mean(axis=0, keepdims=True)
    # Ridge regression: H_hat = BGE @ W, reg = 10.0
    reg = 10.0
    W_bge = np.linalg.solve(bge_vecs.T @ bge_vecs + reg * np.eye(384), bge_vecs.T @ Hc)
    H_resid = Hc - bge_vecs @ W_bge
    H_resid_unit = H_resid / (np.linalg.norm(H_resid, axis=1, keepdims=True) + 1e-9)
    S_resid = H_resid_unit @ H_resid_unit.T
    np.fill_diagonal(S_resid, -1.0)

    print(f"  [m2] Scored all {N} papers across 8 mechanism axes in {m2_pass_sec:.1f}s!", flush=True)

    # Mine cross-category pairs with LOW BGE similarity (< p20 of cross-cat pairs) and HIGH mechanism + residual alignment
    cats = [p["category"] for p in papers]
    cat_arr = np.array(cats)
    cross_mask = (cat_arr[:, None] != cat_arr[None, :]) & np.triu(np.ones((N, N), dtype=bool), k=1)
    bge_p20 = float(np.percentile(S_bge[cross_mask], 20))
    bge_p35 = float(np.percentile(S_bge[cross_mask], 35))
    norm_med = float(np.median(mech_norms))
    print(f"  [m2] Cross-cat BGE p20={bge_p20:.3f}, p35={bge_p35:.3f} | mech norm median={norm_med:.2f}", flush=True)

    # Candidate filter: cross-cat, low BGE (< bge_p25), high mech cosine (>0.82), strong mech norms, both rigorous (>0.2), non-survey
    is_survey = np.array([bool(re.search(r"\b(survey|review|overview|benchmark suite)\b", p["title"], re.I)) for p in papers])
    bge_cut = float(np.percentile(S_bge[cross_mask], 25))
    # Prefer concrete technical categories (not just frontier-ai e-commerce/robotics paired randomly) — let's collect BOTH general & core-systems/interp bridges!
    bridge_cands = []
    # Vectorized mask
    good_node = (zr > 0.25) & (mech_norms > norm_med) & (~is_survey)
    cand_mask = cross_mask & (S_bge < bge_cut) & (S_mech > 0.80) & (S_resid > 0.12) & good_node[:, None] & good_node[None, :]
    ii, jj = np.where(cand_mask)
    print(f"  [m2] Raw vectorized cross-domain bridge candidates: {len(ii)}", flush=True)
    if len(ii) < 40:
        cand_mask = cross_mask & (S_bge < bge_p35) & (S_mech > 0.72) & (S_resid > 0.08) & (zr > 0.0)[:, None] & (zr > 0.0)[None, :] & (~is_survey)[:, None] & (~is_survey)[None, :]
        ii, jj = np.where(cand_mask)
        print(f"  [m2] Relaxed cross-domain bridge candidates: {len(ii)}", flush=True)

    for i, j in zip(ii.tolist(), jj.tolist()):
        ai = set(a.lower().strip() for a in (papers[i].get("authors") or []))
        aj = set(a.lower().strip() for a in (papers[j].get("authors") or []))
        if ai & aj:
            continue
        ti = set(papers[i].get("tags") or [])
        tj = set(papers[j].get("tags") or [])
        tag_overlap = len(ti & tj)
        if tag_overlap > 1:
            continue
        # Composite bridge score: high mech alignment + high BGE-orthogonal residual + low BGE cosine + high rigor
        b_score = (
            1.5 * float(S_mech[i, j])
            + 1.8 * float(S_resid[i, j])
            - 1.5 * float(S_bge[i, j])
            + 0.25 * min(float(zr[i]), float(zr[j]))
            + 0.12 * min(float(mech_norms[i]), float(mech_norms[j]))
        )
        # Find shared dominant mechanisms (both > 0.8)
        shared_mechs = [
            (MECHANISM_SPECS[m][0], round(float(min(Z_mech[i, m], Z_mech[j, m])), 2))
            for m in range(len(MECHANISM_SPECS))
            if Z_mech[i, m] > 0.75 and Z_mech[j, m] > 0.75
        ]
        shared_mechs.sort(key=lambda x: -x[1])
        if not shared_mechs:
            continue
        bridge_cands.append((b_score, i, j, shared_mechs))

    bridge_cands.sort(key=lambda x: -x[0])
    # Deduplicate so no single paper dominates all top-40 slots (max 2 appearances per paper)
    selected_bridges = []
    paper_use: Counter = Counter()
    for item in bridge_cands:
        _, i, j, _ = item
        if paper_use[i] >= 2 or paper_use[j] >= 2:
            continue
        selected_bridges.append(item)
        paper_use[i] += 1
        paper_use[j] += 1
        if len(selected_bridges) >= 40:
            break

    # Stage-2 symmetric paired-swap verification on top 40 bridges!
    print(f"  [m2] Running Stage-2 symmetric paired-swap verification on {len(selected_bridges)} cross-domain bridges...", flush=True)
    xfer_tmpl = (
        "Paper A: {A}\n"
        "Paper B: {B}\n"
        "Question: Ignoring their different application domains, do Paper A and Paper B share a concrete underlying mathematical or algorithmic mechanism that could meaningfully transfer from one to the other? Answer:"
    )
    yes_id = engine.resolve_token_id(" Yes", sample_prompt=xfer_tmpl.format(A="x", B="y"))
    no_id = engine.resolve_token_id(" No", sample_prompt=xfer_tmpl.format(A="x", B="y"))

    b_prompts_ab = [xfer_tmpl.format(A=format_paper_short(papers[i], 380), B=format_paper_short(papers[j], 380)) for _, i, j, _ in selected_bridges]
    b_prompts_ba = [xfer_tmpl.format(A=format_paper_short(papers[j], 380), B=format_paper_short(papers[i], 380)) for _, i, j, _ in selected_bridges]
    m_b_ab = engine.get_margins(b_prompts_ab, yes_id, no_id)
    m_b_ba = engine.get_margins(b_prompts_ba, yes_id, no_id)
    m_b_sym = 0.5 * (m_b_ab + m_b_ba)

    # Also score 20 random cross-category low-BGE pairs as a null control baseline!
    rng = np.random.default_rng(42)
    rand_ii, rand_jj = np.where(cross_mask & (S_bge < bge_cut))
    r_sel = rng.choice(len(rand_ii), size=min(20, len(rand_ii)), replace=False)
    r_prompts_ab = [xfer_tmpl.format(A=format_paper_short(papers[rand_ii[k]], 380), B=format_paper_short(papers[rand_jj[k]], 380)) for k in r_sel]
    r_prompts_ba = [xfer_tmpl.format(A=format_paper_short(papers[rand_jj[k]], 380), B=format_paper_short(papers[rand_ii[k]], 380)) for k in r_sel]
    m_r_sym = 0.5 * (engine.get_margins(r_prompts_ab, yes_id, no_id) + engine.get_margins(r_prompts_ba, yes_id, no_id))

    bridge_records = []
    for k, (b_score, i, j, shared_mechs) in enumerate(selected_bridges):
        bridge_records.append({
            "bridge_stage1_score": round(float(b_score), 3),
            "stage2_transfer_margin": round(float(m_b_sym[k]), 3),
            "bge_cosine": round(float(S_bge[i, j]), 4),
            "ace_mech_cosine": round(float(S_mech[i, j]), 4),
            "bge_orth_resid_cosine": round(float(S_resid[i, j]), 4),
            "shared_mechanisms": shared_mechs[:3],
            "paper_a": {
                "id": papers[i]["id"], "title": papers[i]["title"], "category": papers[i]["category"],
                "year": papers[i]["year"], "citations": papers[i].get("citations", 0),
                "z_rigor": round(float(zr[i]), 2), "abstract": papers[i].get("abstract", ""),
            },
            "paper_b": {
                "id": papers[j]["id"], "title": papers[j]["title"], "category": papers[j]["category"],
                "year": papers[j]["year"], "citations": papers[j].get("citations", 0),
                "z_rigor": round(float(zr[j]), 2), "abstract": papers[j].get("abstract", ""),
            },
        })
    bridge_records.sort(key=lambda r: -(r["stage2_transfer_margin"] + 1.5 * r["ace_mech_cosine"]))

    print(
        f"  [m2] ACE Cross-Domain Bridges vs Random Low-BGE Cross-Cat Controls: "
        f"mean Stage-2 transfer margin = {float(np.mean(m_b_sym)):+.3f} vs {float(np.mean(m_r_sym)):+.3f} "
        f"(delta={float(np.mean(m_b_sym) - np.mean(m_r_sym)):+.3f})",
        flush=True,
    )
    for r in bridge_records[:8]:
        print(
            f"    -> BGE={r['bge_cosine']:.3f} | ACE_mech={r['ace_mech_cosine']:.3f} | resid={r['bge_orth_resid_cosine']:.3f} | xfer={r['stage2_transfer_margin']:+.2f} | "
            f"mechs={r['shared_mechanisms'][:2]}\n"
            f"       A [{r['paper_a']['category']}]: {r['paper_a']['title'][:74]}\n"
            f"       B [{r['paper_b']['category']}]: {r['paper_b']['title'][:74]}",
            flush=True,
        )

    results["method2_cross_domain_bridges"] = {
        "elapsed_sec": round(time.time() - t2, 1),
        "mechanism_axes": mech_cal_summary,
        "bge_p25_cutoff": round(bge_cut, 4),
        "bridge_vs_random_transfer_margin": {
            "ace_bridges_mean": round(float(np.mean(m_b_sym)), 4),
            "random_controls_mean": round(float(np.mean(m_r_sym)), 4),
            "lift": round(float(np.mean(m_b_sym) - np.mean(m_r_sym)), 4),
        },
        "top_bridges": bridge_records[:20],
    }

    # ==========================================================================
    # METHOD 3: Hidden Contradictions in Dense BGE Clusters (De-Biased Stage-2 Keep-1D)
    # ==========================================================================
    print("\n================ METHOD 3: HIDDEN CONTRADICTIONS IN DENSE CLUSTERS ================", flush=True)
    t3 = time.time()

    contra_tmpl = (
        "Claim A: {A}\n"
        "Claim B: {B}\n"
        "Question: Do Claim A and Claim B empirically contradict or directly conflict with each other? Answer:"
    )
    c_layer, q_contra, mu_contra, c_auc, c_gap = calibrate_stage2_pair_core(
        engine, CONTRADICTION_CAL_PAIRS, contra_tmpl, " Yes", " No"
    )
    print(f"  [m3-cal] Contradiction Stage-2 core -> L{c_layer}/{engine.n_layers} | cal_AUC={c_auc:.4f} | gap={c_gap:.1f}", flush=True)

    # Select top ~350 densest non-overlapping-author non-survey pairs from S_bge
    dense_pairs = []
    # Top 0.1% of BGE similarities (e.g. > 0.86)
    hi_i, hi_j = np.where((S_bge > 0.865) & np.triu(np.ones((N, N), dtype=bool), k=1) & (~is_survey)[:, None] & (~is_survey)[None, :])
    order_hi = np.argsort(-S_bge[hi_i, hi_j])
    pair_use: Counter = Counter()
    for idx_k in order_hi.tolist():
        i, j = int(hi_i[idx_k]), int(hi_j[idx_k])
        if pair_use[i] >= 3 or pair_use[j] >= 3:
            continue
        ai = set(a.lower().strip() for a in (papers[i].get("authors") or []))
        aj = set(a.lower().strip() for a in (papers[j].get("authors") or []))
        if ai & aj:
            continue
        if jaccard_title(papers[i]["title"], papers[j]["title"]) > 0.55:
            continue
        dense_pairs.append((float(S_bge[i, j]), i, j))
        pair_use[i] += 1
        pair_use[j] += 1
        if len(dense_pairs) >= 300:
            break

    print(f"  [m3] Scanning {len(dense_pairs)} dense same-topic paper pairs (BGE {dense_pairs[-1][0]:.3f}..{dense_pairs[0][0]:.3f}) via Stage-2 Keep-1D...", flush=True)
    pair_claim_texts = [
        (f"{papers[i]['title']}. {clean_claims[i][:240]}", f"{papers[j]['title']}. {clean_claims[j][:240]}")
        for _, i, j in dense_pairs
    ]
    c_scores = score_stage2_keep1d_pairs(
        engine, pair_claim_texts, contra_tmpl, " Yes", " No", c_layer, q_contra
    )

    raw_order_var = float(np.mean(c_scores["raw_order_abs_diff"]))
    keep_order_var = float(np.mean(c_scores["keep_order_abs_diff"]))
    print(
        f"  [m3] Order-bias |m(A,B)-m(B,A)|: Raw={raw_order_var:.3f} vs Stage-2 Keep-1D={keep_order_var:.3f} "
        f"(reduction={(1.0 - keep_order_var / max(raw_order_var, 1e-6))*100:.1f}%)",
        flush=True,
    )

    # Combine Stage-2 Keep-1D debiased symmetric score + z_sym
    c_rank_score = zscore(c_scores["keep_sym"]) + 0.6 * zscore(c_scores["z_sym"])
    c_order = np.argsort(-c_rank_score)

    contra_records = []
    for k in c_order[:20].tolist():
        sim, i, j = dense_pairs[k]
        contra_records.append({
            "bge_cosine": round(sim, 4),
            "stage2_keep1d_sym": round(float(c_scores["keep_sym"][k]), 3),
            "stage2_z_sym": round(float(c_scores["z_sym"][k]), 3),
            "raw_sym": round(float(c_scores["raw_sym"][k]), 3),
            "raw_order_diff": round(float(c_scores["raw_order_abs_diff"][k]), 3),
            "keep1d_order_diff": round(float(c_scores["keep_order_abs_diff"][k]), 3),
            "paper_a": {
                "id": papers[i]["id"], "title": papers[i]["title"], "category": papers[i]["category"],
                "year": papers[i]["year"], "citations": papers[i].get("citations", 0),
                "claim": clean_claims[i], "abstract": papers[i].get("abstract", ""),
            },
            "paper_b": {
                "id": papers[j]["id"], "title": papers[j]["title"], "category": papers[j]["category"],
                "year": papers[j]["year"], "citations": papers[j].get("citations", 0),
                "claim": clean_claims[j], "abstract": papers[j].get("abstract", ""),
            },
        })
    for r in contra_records[:8]:
        print(
            f"    -> BGE={r['bge_cosine']:.3f} | keep1d={r['stage2_keep1d_sym']:+.2f} | z={r['stage2_z_sym']:+.2f} | raw={r['raw_sym']:+.2f}\n"
            f"       A [{r['paper_a']['category']}|{r['paper_a']['year']}]: {r['paper_a']['title'][:74]}\n"
            f"       B [{r['paper_b']['category']}|{r['paper_b']['year']}]: {r['paper_b']['title'][:74]}",
            flush=True,
        )

    results["method3_dense_contradictions"] = {
        "elapsed_sec": round(time.time() - t3, 1),
        "stage2_layer": c_layer,
        "cal_auc": round(c_auc, 4),
        "cal_spectral_gap": round(c_gap, 1),
        "pairs_scanned": len(dense_pairs),
        "mean_order_diff_raw": round(raw_order_var, 4),
        "mean_order_diff_keep1d": round(keep_order_var, 4),
        "order_bias_reduction_pct": round((1.0 - keep_order_var / max(raw_order_var, 1e-6)) * 100, 1),
        "top_contradictions": contra_records,
    }

    # ==========================================================================
    # METHOD 4: Core-vs-Consensus Residuals ("Sleeping Beauties" & Orthogonal Outliers)
    # ==========================================================================
    print("\n================ METHOD 4: RESIDUALS, SLEEPING BEAUTIES & ORTHOGONAL OUTLIERS ================", flush=True)
    t4 = time.time()

    # 4A: Sleeping Beauties (year <= 2025): OLS log(1 + cites) ~ year + venue_group + category
    old_idxs = [i for i, p in enumerate(papers) if (p.get("year") or 2026) <= 2025 and p.get("category") != "space-ai-lab"]
    y_logc = np.array([math.log1p(papers[i].get("citations", 0)) for i in old_idxs], dtype=np.float64)
    uniq_venues = sorted(set(papers[i].get("venue_group", "") for i in old_idxs))
    uniq_cats = sorted(set(papers[i].get("category", "") for i in old_idxs))
    X_rows = []
    for i in old_idxs:
        p = papers[i]
        row = [1.0, float(2026 - (p.get("year") or 2025))]
        row += [1.0 if p.get("venue_group") == v else 0.0 for v in uniq_venues[1:]]
        row += [1.0 if p.get("category") == c else 0.0 for c in uniq_cats[1:]]
        X_rows.append(row)
    X_ols = np.asarray(X_rows, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X_ols, y_logc, rcond=None)
    y_hat = X_ols @ beta
    cite_resid = y_logc - y_hat  # negative = dramatically under-cited for its age, venue, and category!

    # Sleeping beauty score: high rigor, low hype, strongly negative citation residual
    sb_scores = []
    for pos, i in enumerate(old_idxs):
        if is_survey[i]:
            continue
        score = 1.2 * float(zr[i]) - 0.8 * float(zh[i]) - 0.9 * float(zscore(cite_resid)[pos])
        if zr[i] > 0.6 and cite_resid[pos] < -0.5:
            sb_scores.append((score, i, float(cite_resid[pos]), float(math.expm1(max(0.0, y_hat[pos])))))
    sb_scores.sort(key=lambda x: -x[0])

    sleeping_beauties = []
    for sc, i, cres, exp_c in sb_scores[:15]:
        p = papers[i]
        sleeping_beauties.append({
            "sb_score": round(sc, 3),
            "id": p["id"], "title": p["title"], "category": p["category"],
            "year": p["year"], "venue": p.get("venue", ""),
            "citations": p.get("citations", 0),
            "expected_citations": round(exp_c, 1),
            "log_cite_residual": round(cres, 3),
            "z_rigor": round(float(zr[i]), 3),
            "z_hype": round(float(zh[i]), 3),
            "z_prior": round(float(zp[i]), 3),
            "abstract": p.get("abstract", ""),
        })

    # 4B: Quiet Technical Gems from Today (arxiv-today, 2026-11-24)
    today_idxs = [i for i, p in enumerate(papers) if p.get("venue_group") == "arxiv-today" and not is_survey[i]]
    today_quiet = sorted(today_idxs, key=lambda i: -(1.2 * zr[i] - 0.9 * zh[i] - 0.4 * zp[i]))
    quiet_today = []
    for i in today_quiet[:12]:
        p = papers[i]
        quiet_today.append({
            "quiet_gem_score": round(float(1.2 * zr[i] - 0.9 * zh[i] - 0.4 * zp[i]), 3),
            "id": p["id"], "title": p["title"], "category": p["category"],
            "venue": p.get("venue", ""),
            "z_rigor": round(float(zr[i]), 3),
            "z_hype": round(float(zh[i]), 3),
            "z_prior": round(float(zp[i]), 3),
            "abstract": p.get("abstract", ""),
        })

    # 4C: Orthogonal / Off-Core Causal Energy Outliers (directions u_2..u_8 of rigor ACE!)
    Z_r8 = (H_papers_rigor - mu_rigor8) @ Q_rigor8  # (N, 8)
    z_core1 = Z_r8[:, 0]
    off_core_norm = np.linalg.norm(Z_r8[:, 1:8], axis=1)
    z_off_norm = zscore(off_core_norm)
    off_core_ratio = off_core_norm / (np.abs(z_core1) + float(np.std(z_core1)) * 0.5)

    # Top off-core causal outliers (among rigorous non-surveys!)
    orth_cands = [i for i in range(N) if zr[i] > 0.5 and not is_survey[i]]
    orth_cands.sort(key=lambda i: -(0.7 * z_off_norm[i] + 0.5 * zscore(off_core_ratio)[i]))
    orth_outliers = []
    for i in orth_cands[:12]:
        p = papers[i]
        orth_outliers.append({
            "id": p["id"], "title": p["title"], "category": p["category"],
            "year": p["year"], "citations": p.get("citations", 0),
            "z_rigor": round(float(zr[i]), 3),
            "off_core_norm_z": round(float(z_off_norm[i]), 3),
            "off_core_ratio": round(float(off_core_ratio[i]), 3),
            "top_orth_coords": [round(float(x), 2) for x in Z_r8[i, 1:5].tolist()],
            "abstract": p.get("abstract", ""),
        })

    # 4D: 20-Paper Stage-2 Pairwise Tournament -> Condorcet Cycles & Stage-1 vs Stage-2 Divergence!
    # Pick 10 top rigor papers + 10 highest off-core causal outliers (distinct)
    tourney_idxs = []
    for i in np.argsort(-zr).tolist():
        if not is_survey[i] and i not in tourney_idxs:
            tourney_idxs.append(i)
            if len(tourney_idxs) >= 10:
                break
    for i in orth_cands:
        if i not in tourney_idxs:
            tourney_idxs.append(i)
            if len(tourney_idxs) >= 20:
                break

    print(f"  [m4] Running 20-paper O(N^2) Stage-2 tournament ({20*19}=380 ordered pairs) for Condorcet cycles & Stage-1/2 divergence...", flush=True)
    t_texts = [f"{papers[i]['title']}. {clean_claims[i][:200]}" for i in tourney_idxs]
    t_tmpl = "Evaluating concrete technical rigor and significance, Paper A ({A}) compared to Paper B ({B}) is"
    # Extract Stage-2 END core on rigor calibration pairs at L20 (~0.71 depth)
    s2_l = int(round(0.71 * engine.n_layers))
    H_s2_tr, G_s2_tr, _ = engine.collect_acts_and_grads(rigor_pr[:40], s2_l, r_pos_id, r_neg_id, spans=None, pooling="last")
    H_s2_diff = np.array([H_s2_tr[a] - H_s2_tr[b] for a, b in rigor_swaps[:20]])
    Q_s2, _, _ = compute_ace_subspace(H_s2_diff, G_s2_tr, rank=1)
    q_s2_end = Q_s2[:, 0]

    m_base_t, m_keep_t, m_deb_t = engine.stage2_debiased_margins(
        t_texts, t_tmpl, " rigorous", " superficial", stage2_layer=s2_l, q_end=q_s2_end
    )

    s1_vals = [float(zr[i]) for i in tourney_idxs]
    met_base = compute_ranking_metrics(s1_vals, m_base_t)
    met_deb = compute_ranking_metrics(s1_vals, m_deb_t)

    # Count Condorcet cycles per paper in raw vs debiased Stage-2 (using antisymmetric win rates)
    n_t = len(tourney_idxs)
    anti_base = np.array([[0.5 * (m_base_t[(a, b)] - m_base_t[(b, a)]) if a != b else 0.0 for b in range(n_t)] for a in range(n_t)])
    anti_deb = np.array([[0.5 * (m_deb_t[(a, b)] - m_deb_t[(b, a)]) if a != b else 0.0 for b in range(n_t)] for a in range(n_t)])

    cycle_counts = np.zeros(n_t, dtype=int)
    total_anti_cycles = 0
    for a in range(n_t):
        for b in range(a + 1, n_t):
            for c in range(b + 1, n_t):
                if (anti_base[a, b] > 0 and anti_base[b, c] > 0 and anti_base[c, a] > 0) or (
                    anti_base[b, a] > 0 and anti_base[c, b] > 0 and anti_base[a, c] > 0
                ):
                    total_anti_cycles += 1
                    cycle_counts[a] += 1
                    cycle_counts[b] += 1
                    cycle_counts[c] += 1

    s2_win_rate = np.mean(anti_base > 0, axis=1)
    s1_ranks = np.argsort(np.argsort(-np.array(s1_vals)))  # 0 = highest Stage-1
    s2_ranks = np.argsort(np.argsort(-s2_win_rate))       # 0 = highest Stage-2
    rank_gap = s1_ranks - s2_ranks  # large positive => Stage-2 rates MUCH higher than Stage-1 scalar core!

    tourney_records = []
    for k, i in enumerate(tourney_idxs):
        p = papers[i]
        tourney_records.append({
            "id": p["id"], "title": p["title"], "category": p["category"], "year": p["year"],
            "citations": p.get("citations", 0),
            "s1_rank": int(s1_ranks[k] + 1),
            "s2_rank": int(s2_ranks[k] + 1),
            "rank_divergence_s1_minus_s2": int(rank_gap[k]),
            "s2_win_rate": round(float(s2_win_rate[k]), 3),
            "z_rigor": round(float(zr[i]), 3),
            "off_core_norm_z": round(float(z_off_norm[i]), 3),
            "condorcet_cycles_participated": int(cycle_counts[k]),
        })
    tourney_records.sort(key=lambda r: -abs(r["rank_divergence_s1_minus_s2"]))

    print(
        f"  [m4] Tournament Raw vs De-Biased Stage-2 Keep-1D: "
        f"pos_bias={met_base['pos_bias_rate']*100:.1f}% -> {met_deb['pos_bias_rate']*100:.1f}% | "
        f"raw_cycles={met_base['cycle_rate_raw']*100:.2f}% -> {met_deb['cycle_rate_raw']*100:.2f}% | "
        f"antisym_cycles={total_anti_cycles}/{math.comb(n_t, 3)}",
        flush=True,
    )
    print("  [m4] Top 5 Sleeping Beauties (<=2025):", flush=True)
    for r in sleeping_beauties[:5]:
        print(
            f"    -> [{r['category']}|{r['year']}|c={r['citations']} vs exp={r['expected_citations']}] "
            f"{r['title'][:68]} | rig={r['z_rigor']:+.2f}, hype={r['z_hype']:+.2f}",
            flush=True,
        )
    print("  [m4] Top 5 Quiet Gems Released Today (2026-11-24):", flush=True)
    for r in quiet_today[:5]:
        print(
            f"    -> [{r['category']}] {r['title'][:72]} | rig={r['z_rigor']:+.2f}, hype={r['z_hype']:+.2f}, prior={r['z_prior']:+.2f}",
            flush=True,
        )
    print("  [m4] Top 5 Off-Core Orthogonal Causal Outliers:", flush=True)
    for r in orth_outliers[:5]:
        print(
            f"    -> [{r['category']}|{r['year']}|c={r['citations']}] {r['title'][:68]} | "
            f"off_core_z={r['off_core_norm_z']:+.2f}, ratio={r['off_core_ratio']:.2f}",
            flush=True,
        )

    results["method4_residuals_and_outliers"] = {
        "elapsed_sec": round(time.time() - t4, 1),
        "sleeping_beauties_pre2026": sleeping_beauties,
        "quiet_gems_today": quiet_today,
        "off_core_orthogonal_outliers": orth_outliers,
        "stage2_tournament": {
            "stage2_layer": s2_l,
            "raw_metrics": {k: round(v, 4) for k, v in met_base.items()},
            "debiased_keep1d_metrics": {k: round(v, 4) for k, v in met_deb.items()},
            "antisymmetric_cycles": total_anti_cycles,
            "total_triples": math.comb(n_t, 3),
            "most_divergent_papers": tourney_records[:10],
        },
    }

    results["total_elapsed_sec"] = round(time.time() - t_all, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved full results to {OUT_JSON} in {results['total_elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
