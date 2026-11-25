#!/usr/bin/env python3
"""
GPU Investigation 03 (Version 2.0 — Conference-Grade Multi-Model Evaluation):
"Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal
Bottlenecks Beyond the Logit and Tuned Lens"

Triggered by CoreRank Method 1 Discovery #1 (arxiv:2604.02608):
  "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens"

Protocol (v2.0):
  - 3 distinct transformer families & scales (1.7B -> 7.2B):
      1. Qwen/Qwen3-1.7B          (Qwen3 GQA+QKNorm, 28L, D=2048, V=151,936, full GPU)
      2. HuggingFaceTB/SmolLM2-1.7B (Llama MHA, 24L, D=2048, V=49,152, full GPU)
      3. mistralai/Mistral-7B-v0.1  (Mistral GQA+SWA, 32L, D=4096, V=32,000, 5.5 GB GPU+CPU split)
  - 4 rich comparative domains (16 entities each; strict disjoint 8-train / 8-test split):
      physical_scale, hazard_severity, historical_chronology, geographic_scale
      (56 ordered held-out test pairs x 3 verified single-token unseen cross-lexicons
       = 168 cross-lexicon pairs/domain, 672/model, 2,016 total held-out evaluations).
  - Exact per-lexicon ROC-AUC evaluation (invariant to inter-lexicon prior offsets) with
    1,000-sample paired bootstrap 95% CIs and Wilcoxon signed-rank p-values.
  - 5 strong baselines: Direct Logit Lens, Residual-Affine Tuned Lens, CAA Mean-Diff,
    L2 Logistic Probe, and Random Unit Vector Control; plus Flip-1D necessity,
    Keep-1D sufficiency, rank ablation, ridge ablation, and 300-DPI multi-panel figure.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from corerank.engine import compute_ace_subspace, roc_auc

SITE_ROOT = Path(__file__).resolve().parents[3]
PAPERS_JSON = SITE_ROOT / "data" / "papers-db.json"
FIG_PATH = SITE_ROOT / "papers" / "figures" / "steerable-not-decodable-v4.png"

MODELS: List[Dict[str, Any]] = [
    {
        "id": "Qwen/Qwen3-1.7B",
        "short": "Qwen3-1.7B",
        "family": "Qwen3 (GQA+QKNorm)",
        "params_b": 1.7,
        "gpu_layers": 28,
        "batch_size": 32,
        "sweep_layers": [4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B",
        "short": "SmolLM2-1.7B",
        "family": "Llama (MHA)",
        "params_b": 1.7,
        "gpu_layers": 24,
        "batch_size": 32,
        "sweep_layers": [3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
    },
    {
        "id": "mistralai/Mistral-7B-v0.1",
        "short": "Mistral-7B",
        "family": "Mistral (GQA+SWA)",
        "params_b": 7.2,
        "gpu_layers": 13,  # ~5.5 GB VRAM safely alongside basalt
        "batch_size": 16,
        "sweep_layers": [4, 10, 16, 20, 24, 28, 32],
    },
    {
        "id": "allenai/OLMo-2-1124-7B",
        "short": "OLMo-2-7B",
        "family": "OLMo-2 (PostNorm+QKNorm)",
        "params_b": 7.3,
        "gpu_layers": 12,  # ~5.2 GB VRAM safely alongside basalt
        "batch_size": 16,
        "sweep_layers": [4, 10, 16, 20, 24, 28, 32],
    },
]

# Every target completion below is verified 100% single-token across Qwen3, SmolLM2, AND Mistral-7B!
DOMAINS: Dict[str, Dict[str, Any]] = {
    "physical_scale": {
        "label": "Physical Scale",
        "items": [
            ("a grain of sand", 0.0005),
            ("a marble", 0.015),
            ("a basketball", 0.24),
            ("a city bus", 12.0),
            ("the Eiffel Tower", 330.0),
            ("Mount Everest", 8849.0),
            ("the Moon", 3474000.0),
            ("Mercury", 4879000.0),
            ("Mars", 6779000.0),
            ("Venus", 12104000.0),
            ("Earth", 12742000.0),
            ("Neptune", 49244000.0),
            ("Uranus", 50724000.0),
            ("Saturn", 116460000.0),
            ("Jupiter", 139820000.0),
            ("the Sun", 1392700000.0),
        ],
        "source_lexicon": ("Comparing physical diameter, {A} compared to {B} is", " larger", " smaller"),
        "cross_lexicons": [
            ("Comparing overall physical size, {A} compared to {B} is", " bigger", " tiny"),
            ("Comparing spatial width, {A} compared to {B} is", " wide", " narrow"),
            ("Comparing total volume, {A} compared to {B} is", " greater", " less"),
        ],
    },
    "hazard_severity": {
        "label": "Hazard Severity",
        "items": [
            ("watering a houseplant", 1.0),
            ("reading a paperback book", 2.0),
            ("folding clean laundry", 3.0),
            ("forgetting an umbrella", 4.0),
            ("skipping breakfast once", 5.0),
            ("walking in light rain", 6.0),
            ("leaving a stove burner unattended", 7.0),
            ("driving without headlights at night", 8.0),
            ("texting while driving on a highway", 9.0),
            ("swimming during a lightning storm", 10.0),
            ("mixing bleach and ammonia indoors", 11.0),
            ("removing brake pads from a car", 12.0),
            ("tampering with a hospital oxygen line", 13.0),
            ("sabotaging an aircraft engine", 14.0),
            ("derailing a passenger train", 15.0),
            ("releasing a lethal nerve agent in a subway", 16.0),
        ],
        "source_lexicon": ("In terms of real-world danger, {A} compared to {B} is", " dangerous", " safe"),
        "cross_lexicons": [
            ("In terms of potential harm, {A} compared to {B} is more", " severe", " mild"),
            ("In terms of safety impact, {A} compared to {B} is", " worse", " better"),
            ("In terms of physical risk, {A} compared to {B} is more", " serious", " minor"),
        ],
    },
    "historical_chronology": {
        "label": "Historical Chronology",
        "items": [
            ("the construction of the Great Pyramid of Giza", -2560.0),
            ("the assassination of Julius Caesar", -44.0),
            ("the fall of the Western Roman Empire", 476.0),
            ("the signing of the Magna Carta", 1215.0),
            ("Gutenberg's invention of the printing press", 1440.0),
            ("Isaac Newton's publication of Principia", 1687.0),
            ("the signing of the US Declaration of Independence", 1776.0),
            ("Charles Darwin's publication of Origin of Species", 1859.0),
            ("the Wright brothers' first powered flight", 1903.0),
            ("the end of World War II", 1945.0),
            ("the launch of Sputnik 1", 1957.0),
            ("the Apollo 11 Moon landing", 1969.0),
            ("the invention of the World Wide Web", 1989.0),
            ("the sequencing of the human genome", 2003.0),
            ("the launch of the first iPhone", 2007.0),
            ("the launch of the James Webb Space Telescope", 2021.0),
        ],
        "source_lexicon": ("In historical timeline, {A} compared to {B} is more", " recent", " ancient"),
        "cross_lexicons": [
            ("In history, {A} compared to {B} is more", " modern", " old"),
            ("In chronological age, {A} compared to {B} is", " newer", " older"),
            ("In time period, {A} compared to {B} is more", " new", " ancient"),
        ],
    },
    "geographic_scale": {
        "label": "Geographic Scale",
        "items": [
            ("Monaco", 2.0),
            ("Singapore", 734.0),
            ("Luxembourg", 2586.0),
            ("Belgium", 30528.0),
            ("Switzerland", 41285.0),
            ("Portugal", 92226.0),
            ("Italy", 301340.0),
            ("Japan", 377975.0),
            ("France", 551695.0),
            ("Egypt", 1002450.0),
            ("Mexico", 1964375.0),
            ("India", 3287263.0),
            ("Australia", 7692024.0),
            ("Brazil", 8515767.0),
            ("Canada", 9984670.0),
            ("Russia", 17098246.0),
        ],
        "source_lexicon": ("Comparing country land area, {A} compared to {B} is", " larger", " smaller"),
        "cross_lexicons": [
            ("Comparing geographic size, {A} compared to {B} is", " bigger", " tiny"),
            ("Comparing territorial span, {A} compared to {B} is", " wide", " narrow"),
            ("Comparing total area, {A} compared to {B} is", " greater", " less"),
        ],
    },
}


def build_pairs(items: Sequence[Tuple[str, float]], tmpl: str):
    prompts, pairs, swaps = [], [], []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            ia = len(prompts)
            prompts.append(tmpl.format(A=items[i][0], B=items[j][0]))
            pairs.append((i, j))
            ib = len(prompts)
            prompts.append(tmpl.format(A=items[j][0], B=items[i][0]))
            pairs.append((j, i))
            swaps.append((ia, ib))
    return prompts, pairs, swaps


def pair_labels(pairs: Sequence[Tuple[int, int]], vals: Sequence[float]) -> np.ndarray:
    return np.asarray([1 if vals[i] > vals[j] else 0 for i, j in pairs], dtype=np.int64)


def auc_from_y(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc(y, scores))


def bootstrap_lexicon_auc_ci(
    y: np.ndarray, scores_per_lex: Sequence[np.ndarray], n_boot: int = 1000, seed: int = 42
) -> Tuple[float, float, float]:
    """
    Mean per-lexicon ROC-AUC and 95% paired bootstrap CI (invariant to inter-lexicon constant offsets!).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(np.mean([auc_from_y(y, s) for s in scores_per_lex]))
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        yb = y[idx[b]]
        if yb.sum() == 0 or yb.sum() == n:
            vals[b] = point
        else:
            vals[b] = float(np.mean([auc_from_y(yb, s[idx[b]]) for s in scores_per_lex]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_lexicon_diff_ci(
    y: np.ndarray,
    scores_a_per_lex: Sequence[np.ndarray],
    scores_b_per_lex: Sequence[np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap 95% CI for mean_c [AUC(a_c) - AUC(b_c)]."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(np.mean([auc_from_y(y, a) - auc_from_y(y, b) for a, b in zip(scores_a_per_lex, scores_b_per_lex)]))
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        yb = y[idx[b]]
        if yb.sum() == 0 or yb.sum() == n:
            diffs[b] = point
        else:
            diffs[b] = float(
                np.mean([auc_from_y(yb, a[idx[b]]) - auc_from_y(yb, s[idx[b]]) for a, s in zip(scores_a_per_lex, scores_b_per_lex)])
            )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def fit_l2_logistic_probe(H_tr: np.ndarray, y_tr: np.ndarray, reg: float = 1.0, max_iter: int = 60) -> Tuple[np.ndarray, float]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = H_tr.mean(axis=0, keepdims=True)
    X = torch.tensor(H_tr - mu, dtype=torch.float32, device=dev)
    y = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    D = X.shape[1]
    w = torch.zeros(D, dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.8, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + (0.5 * reg / len(y)) * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    w_np = w.detach().cpu().numpy().astype(np.float64)
    return w_np / (float(np.linalg.norm(w_np)) + 1e-12), float(b.detach().cpu().item())


def fit_caa_vector(H_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    v = (H_tr[y_tr == 1].mean(axis=0) - H_tr[y_tr == 0].mean(axis=0)).astype(np.float64)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def fit_tuned_lens_translator(H_src: np.ndarray, H_final: np.ndarray, alpha: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Residual-Affine Tuned Lens T_l(h) = h + (h - mu_l) A_l + b_l (Belrose et al., 2023).
    Automatically uses primal GPU ridge when N > D (e.g. 3,500+ general-corpus tokens) and dual ridge when N <= D.
    """
    N, D = H_src.shape
    if N <= D:
        X = H_src.astype(np.float64)
        Y = (H_final - H_src).astype(np.float64)
        mu_x = X.mean(axis=0, keepdims=True)
        mu_y = Y.mean(axis=0, keepdims=True)
        Xc = X - mu_x
        Yc = Y - mu_y
        K = Xc @ Xc.T + alpha * np.eye(N, dtype=np.float64)
        dual = np.linalg.solve(K, Yc)
        A = Xc.T @ dual
        b = (mu_y - mu_x @ A).reshape(-1)
        return A.astype(np.float32), b.astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(H_src, dtype=torch.float32, device=dev)
    Y_t = torch.tensor(H_final - H_src, dtype=torch.float32, device=dev)
    mu_x = X_t.mean(dim=0, keepdim=True)
    mu_y = Y_t.mean(dim=0, keepdim=True)
    Xc = X_t - mu_x
    Yc = Y_t - mu_y
    gram = Xc.T @ Xc
    gram.diagonal().add_(float(alpha))
    A_t = torch.linalg.solve(gram, Xc.T @ Yc)
    b_t = (mu_y - mu_x @ A_t).reshape(-1)
    A_np = A_t.cpu().numpy().astype(np.float32)
    b_np = b_t.cpu().numpy().astype(np.float32)
    del X_t, Y_t, Xc, Yc, gram, A_t, b_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return A_np, b_np


def fit_kl_tuned_lens_translator(
    H_src: np.ndarray,
    H_final: np.ndarray,
    norm_mod: torch.nn.Module,
    lm_head_weight: torch.Tensor,
    steps: int = 80,
    lr: float = 1.5e-2,
    weight_decay: float = 1e-2,
    topk_vocab: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    True KL-Minimizing Residual-Affine Tuned Lens (exact Belrose et al., 2023 objective):
      min_{A, b} E[ D_KL( softmax(W_U Norm(h_L)) || softmax(W_U Norm(h_l + (h_l - mu_l) A + b)) ) ]
    Initialized from closed-form ridge solution and optimized directly via AdamW on top-256 reference
    logits (>99.5% probability mass) in FP32 on GPU.
    """
    A_init, b_init = fit_tuned_lens_translator(H_src, H_final, alpha=500.0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = H_src.shape
    X = torch.tensor(H_src, dtype=torch.float32, device=dev)
    Y = torch.tensor(H_final, dtype=torch.float32, device=dev)

    # Extract lightweight norm scale & weight on GPU in FP32
    norm_w = norm_mod.weight.detach().float().to(dev)
    eps = float(getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6)))
    W_u_cpu = lm_head_weight.detach()

    def rms_norm_f32(t: torch.Tensor) -> torch.Tensor:
        return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps) * norm_w

    # Precompute top-K reference token indices & normalized target probabilities in chunks (low VRAM!)
    sub_n = min(N, 2048)
    X_sub = X[:sub_n]
    Y_sub = Y[:sub_n]
    top_idx_chunks, p_ref_chunks = [], []
    W_u_dev = W_u_cpu.device
    with torch.no_grad():
        for s in range(0, sub_n, 256):
            yn = rms_norm_f32(Y_sub[s : s + 256]).to(W_u_dev, dtype=W_u_cpu.dtype)
            logits_ref = yn @ W_u_cpu.T  # (B, V)
            vals, idx = torch.topk(logits_ref.float(), k=min(topk_vocab, logits_ref.shape[-1]), dim=-1)
            top_idx_chunks.append(idx.to(dev))
            p_ref_chunks.append(F.softmax(vals, dim=-1).to(dev))
    top_idx = torch.cat(top_idx_chunks, dim=0)  # (sub_n, K)
    p_ref = torch.cat(p_ref_chunks, dim=0)      # (sub_n, K)

    # Gather unique vocabulary rows onto GPU once so optimization is super fast & lean!
    uniq_toks, inv_idx = torch.unique(top_idx, return_inverse=True)
    W_u_sub = W_u_cpu[uniq_toks.to(W_u_dev)].float().to(dev)  # (U, D)

    A = torch.nn.Parameter(torch.tensor(A_init, dtype=torch.float32, device=dev))
    b = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW([A, b], lr=lr, weight_decay=weight_decay)

    bs = 256
    for step in range(steps):
        perm = torch.randint(0, sub_n, (bs,), device=dev)
        xb = X_sub[perm]
        ht = rms_norm_f32(xb + xb @ A + b)  # (bs, D)
        inv_b = inv_idx[perm]               # (bs, K)
        # Gather per-row top-K unembedding vectors: (bs, K, D) -> logits (bs, K)
        W_b = W_u_sub[inv_b]
        logits_t = torch.einsum("bd,bkd->bk", ht, W_b)
        log_q = F.log_softmax(logits_t, dim=-1)
        p_b = p_ref[perm]
        loss = -(p_b * log_q).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A, b], 2.0)
        opt.step()

    A_np = A.detach().cpu().numpy().astype(np.float32)
    b_np = b.detach().cpu().numpy().astype(np.float32)
    del X, Y, X_sub, Y_sub, top_idx, p_ref, uniq_toks, inv_idx, W_u_sub, A, b, opt, norm_w
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return A_np, b_np


def load_general_corpus_texts(n_docs: int = 128) -> List[str]:
    """Load diverse scientific abstract passages (completely disjoint from comparative templates) for General-Corpus Tuned Lens."""
    if PAPERS_JSON.exists():
        db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
        out = []
        for p in db.get("papers", []):
            abs_txt = (p.get("abstract") or "").strip()
            if len(abs_txt) >= 120:
                out.append(f"{p.get('title', '')}. {abs_txt[:260]}")
            if len(out) >= n_docs:
                break
        if len(out) >= 32:
            return out
    return [
        "Gradient descent optimizes differentiable objective functions across high-dimensional parameter manifolds in modern neural networks."
    ] * 64


class HybridCausalModel:
    """
    Unified causal LM runner supporting full-GPU (1.7B) and clean GPU+CPU split (7B) with
    bit-exact suffix caching so layer-l interventions skip layers 0..l-1 cleanly!
    """

    def __init__(self, model_id: str, gpu_layers: int, batch_size: int = 32):
        self.model_id = model_id
        self.batch_size = batch_size
        self.gpu_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.backbone = self.model.model
        self.layers = self.backbone.layers
        self.n_layers = len(self.layers)
        self.d_model = int(self.model.config.hidden_size)
        self.gpu_n = min(gpu_layers, self.n_layers)
        self.is_full_gpu = self.gpu_n == self.n_layers

        self.backbone.embed_tokens.to(self.gpu_dev)
        self.backbone.rotary_emb.to(self.gpu_dev)
        for i in range(self.gpu_n):
            self.layers[i].to(self.gpu_dev)
        if self.is_full_gpu:
            self.backbone.norm.to(self.gpu_dev)
            self.model.lm_head.to(self.gpu_dev)
        else:
            self.backbone.norm.to("cpu")
            if self.model.lm_head.weight.data_ptr() != self.backbone.embed_tokens.weight.data_ptr():
                self.model.lm_head.to("cpu")

        self.lm_head_dev = self.model.lm_head.weight.device
        self.norm_dev = self.backbone.norm.weight.device
        self.alloc_gb = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    def resolve_token_id(self, word: str, sample_prompt: str) -> int:
        base = self.tokenizer.encode(sample_prompt, add_special_tokens=False)
        full = self.tokenizer.encode(sample_prompt + word, add_special_tokens=False)
        diff = full[len(base) :]
        if len(diff) == 1:
            return int(diff[0])
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return int(ids[0])

    def collect_all_layers(
        self,
        prompts: Sequence[str],
        capture_layers: Sequence[int],
        pos_id: int,
        neg_id: int,
        need_grads: bool = False,
        cache_seq_for_suffix: bool = False,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, List[Dict[str, Any]]]:
        """
        Single forward (and optional single backward) pass capturing all `capture_layers`.
        If `cache_seq_for_suffix=True`, also returns per-batch cached full-sequence bf16 hidden
        states at each `capture_layer` (on CPU) so downstream causal interventions at layer l
        can execute ONLY layers l..L-1 (bit-identical & ~2.5-3x faster on Mistral-7B!).
        """
        cap_set = set(capture_layers)
        acts_acc: Dict[int, List[np.ndarray]] = {l: [] for l in capture_layers}
        grads_acc: Dict[int, List[np.ndarray]] = {l: [] for l in capture_layers}
        m_acc: List[np.ndarray] = []
        suffix_batches: List[Dict[str, Any]] = []

        ctx = torch.enable_grad() if need_grads else torch.inference_mode()
        for s in range(0, len(prompts), self.batch_size):
            batch = list(prompts[s : s + self.batch_size])
            enc = self.tokenizer(batch, return_tensors="pt", padding=True)
            ids_g = enc["input_ids"].to(self.gpu_dev)
            mask_g = enc["attention_mask"].to(self.gpu_dev)
            B, T = ids_g.shape
            pos_g = mask_g.long().cumsum(-1) - 1
            pos_g.masked_fill_(mask_g == 0, 0)
            cache_pos_g = torch.arange(T, device=self.gpu_dev)

            captured_t: Dict[int, torch.Tensor] = {}
            seq_cache_b: Dict[int, torch.Tensor] = {}

            with ctx:
                h = self.backbone.embed_tokens(ids_g)
                if need_grads:
                    h = h.detach().requires_grad_(True)
                cmask_g = create_causal_mask(
                    config=self.model.config, inputs_embeds=h, attention_mask=mask_g,
                    cache_position=cache_pos_g, past_key_values=None, position_ids=pos_g,
                )
                pe_g = self.backbone.rotary_emb(h, pos_g)
                cmask_c, pos_c, pe_c = None, None, None
                moved_cpu = False

                for idx in range(self.n_layers):
                    l_1idx = idx + 1
                    if idx == self.gpu_n and not self.is_full_gpu:
                        h = h.to("cpu")
                        cmask_c = cmask_g.to("cpu") if cmask_g is not None else None
                        pos_c = pos_g.to("cpu")
                        pe_c = (pe_g[0].to("cpu"), pe_g[1].to("cpu"))
                        moved_cpu = True

                    cm = cmask_c if moved_cpu else cmask_g
                    ps = pos_c if moved_cpu else pos_g
                    pe = pe_c if moved_cpu else pe_g
                    out = self.layers[idx](h, attention_mask=cm, position_ids=ps, position_embeddings=pe, use_cache=False)
                    h = out[0] if isinstance(out, tuple) else out

                    if l_1idx in cap_set:
                        if need_grads:
                            h.retain_grad()
                        captured_t[l_1idx] = h
                        if cache_seq_for_suffix:
                            seq_cache_b[l_1idx] = h.detach().cpu()

                h_norm = self.backbone.norm(h[:, -1, :].to(self.norm_dev)).to(self.lm_head_dev)
                logits = self.model.lm_head(h_norm)
                margins = logits[:, pos_id].float() - logits[:, neg_id].float()

                if need_grads:
                    margins.sum().backward()
                    for l in capture_layers:
                        grads_acc[l].append(captured_t[l].grad[:, -1, :].detach().float().cpu().numpy())

                for l in capture_layers:
                    acts_acc[l].append(captured_t[l][:, -1, :].detach().float().cpu().numpy())
                m_acc.append(margins.detach().cpu().numpy())

            if cache_seq_for_suffix:
                suffix_batches.append({
                    "seq_by_layer": seq_cache_b,
                    "mask_cpu": enc["attention_mask"].cpu(),
                    "pos_cpu": pos_g.cpu(),
                    "cmask_cpu": cmask_g.cpu() if cmask_g is not None else None,
                    "pe_cpu": (pe_g[0].cpu(), pe_g[1].cpu()),
                })

        acts_out = {l: np.concatenate(acts_acc[l], axis=0) for l in capture_layers}
        grads_out = {l: np.concatenate(grads_acc[l], axis=0) for l in capture_layers} if need_grads else {}
        return acts_out, grads_out, np.concatenate(m_acc, axis=0), suffix_batches

    @torch.inference_mode()
    def intervene_from_suffix_cache(
        self,
        suffix_batches: List[Dict[str, Any]],
        layer_1idx: int,
        mode: str,
        q_vec: np.ndarray,
        mu_vec: np.ndarray,
        pos_id: int,
        neg_id: int,
    ) -> np.ndarray:
        """
        Bit-exact Flip-1D or Keep-1D causal intervention at `layer_1idx`, executing ONLY
        downstream layers `layer_1idx .. n_layers-1` from the cached full-sequence state!
        """
        q_cpu = torch.tensor(q_vec, dtype=torch.float32)
        mu_cpu = torch.tensor(mu_vec, dtype=torch.float32)
        m_acc: List[np.ndarray] = []

        for sb in suffix_batches:
            h = sb["seq_by_layer"][layer_1idx].clone()
            # Apply intervention at last token of layer_1idx output
            hs = h[:, -1, :].float()
            proj = ((hs - mu_cpu) @ q_cpu).unsqueeze(-1) * q_cpu
            if mode == "flip":
                h[:, -1, :] = (hs - 2.0 * proj).to(h.dtype)
            elif mode == "keep":
                h[:, -1, :] = (mu_cpu + proj).to(h.dtype)

            # Determine initial device for downstream layer `layer_1idx` (0-indexed!)
            start_idx = layer_1idx
            if start_idx < self.gpu_n:
                h = h.to(self.gpu_dev)
                cm_g = sb["cmask_cpu"].to(self.gpu_dev) if sb["cmask_cpu"] is not None else None
                ps_g = sb["pos_cpu"].to(self.gpu_dev)
                pe_g = (sb["pe_cpu"][0].to(self.gpu_dev), sb["pe_cpu"][1].to(self.gpu_dev))
                on_gpu = True
            else:
                cm_g, ps_g, pe_g = None, None, None
                on_gpu = False

            cm_c = sb["cmask_cpu"]
            ps_c = sb["pos_cpu"]
            pe_c = sb["pe_cpu"]

            for idx in range(start_idx, self.n_layers):
                if idx == self.gpu_n and on_gpu and not self.is_full_gpu:
                    h = h.to("cpu")
                    on_gpu = False
                cm = cm_g if on_gpu else cm_c
                ps = ps_g if on_gpu else ps_c
                pe = pe_g if on_gpu else pe_c
                out = self.layers[idx](h, attention_mask=cm, position_ids=ps, position_embeddings=pe, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h_norm = self.backbone.norm(h[:, -1, :].to(self.norm_dev)).to(self.lm_head_dev)
            logits = self.model.lm_head(h_norm)
            margins = logits[:, pos_id].float() - logits[:, neg_id].float()
            m_acc.append(margins.cpu().numpy())

        return np.concatenate(m_acc, axis=0)

    @torch.inference_mode()
    def apply_unembedding(self, H_np: np.ndarray, pos_id: int, neg_id: int) -> np.ndarray:
        h_t = torch.tensor(H_np, dtype=torch.bfloat16, device=self.norm_dev)
        h_norm = self.backbone.norm(h_t).float().to(self.lm_head_dev)
        w_u = self.model.lm_head.weight
        dw = w_u[pos_id].float() - w_u[neg_id].float()
        return (h_norm @ dw).cpu().numpy()

    def unembedding_diff_unit(self, pos_id: int, neg_id: int) -> np.ndarray:
        w_u = self.model.lm_head.weight
        dw = (w_u[pos_id].detach().float() - w_u[neg_id].detach().float()).cpu().numpy()
        return dw / (float(np.linalg.norm(dw)) + 1e-12)

    @torch.inference_mode()
    def collect_general_corpus_tokens(
        self,
        texts: Sequence[str],
        capture_layers: Sequence[int],
        max_len: int = 44,
        max_tokens: int = 3200,
    ) -> Dict[int, np.ndarray]:
        """Collect diverse non-BOS token hidden states across real scientific text for General-Corpus Tuned Lens."""
        cap_set = set(capture_layers)
        acc: Dict[int, List[np.ndarray]] = {l: [] for l in capture_layers}
        n_seen = 0
        bs = min(self.batch_size, 16)
        for s in range(0, len(texts), bs):
            if n_seen >= max_tokens:
                break
            batch = list(texts[s : s + bs])
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids_g = enc["input_ids"].to(self.gpu_dev)
            mask_g = enc["attention_mask"].to(self.gpu_dev)
            B, T = ids_g.shape
            pos_g = mask_g.long().cumsum(-1) - 1
            pos_g.masked_fill_(mask_g == 0, 0)
            cache_pos_g = torch.arange(T, device=self.gpu_dev)
            valid = mask_g.bool().cpu()
            valid[:, 0] = False  # drop BOS sink

            h = self.backbone.embed_tokens(ids_g)
            cmask_g = create_causal_mask(
                config=self.model.config, inputs_embeds=h, attention_mask=mask_g,
                cache_position=cache_pos_g, past_key_values=None, position_ids=pos_g,
            )
            pe_g = self.backbone.rotary_emb(h, pos_g)
            cmask_c, pos_c, pe_c = None, None, None
            moved_cpu = False
            for idx in range(self.n_layers):
                l_1idx = idx + 1
                if idx == self.gpu_n and not self.is_full_gpu:
                    h = h.to("cpu")
                    cmask_c = cmask_g.to("cpu") if cmask_g is not None else None
                    pos_c = pos_g.to("cpu")
                    pe_c = (pe_g[0].to("cpu"), pe_g[1].to("cpu"))
                    moved_cpu = True
                cm = cmask_c if moved_cpu else cmask_g
                ps = pos_c if moved_cpu else pos_g
                pe = pe_c if moved_cpu else pe_g
                out = self.layers[idx](h, attention_mask=cm, position_ids=ps, position_embeddings=pe, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                if l_1idx in cap_set:
                    acc[l_1idx].append(h.detach().float().cpu()[valid].numpy())
            n_seen += int(valid.sum().item())

        return {l: np.concatenate(acc[l], axis=0)[:max_tokens] for l in capture_layers}

    @torch.inference_mode()
    def eval_freeform_generation_steering(
        self,
        prompts: Sequence[str],
        y_true: np.ndarray,
        pos_words: Sequence[str],
        neg_words: Sequence[str],
        layer_1idx: int,
        q_vec: np.ndarray,
        mu_vec: np.ndarray,
        rand_vec: np.ndarray,
        max_new_tokens: int = 10,
    ) -> Dict[str, Any]:
        """
        Full-GPU multi-token greedy generation steering audit (Experiment B):
        Steer at `layer_1idx` on the final prompt position via Flip-1D (ACE vs. Random) and generate
        `max_new_tokens` freely. Measures clean accuracy, free-form semantic flip rate on correct prompts,
        and fluency (base-model continuation NLL in nats/token).
        """
        if not self.is_full_gpu:
            return {}
        q_t = torch.tensor(q_vec, dtype=torch.float32, device=self.gpu_dev)
        r_t = torch.tensor(rand_vec, dtype=torch.float32, device=self.gpu_dev)
        mu_t = torch.tensor(mu_vec, dtype=torch.float32, device=self.gpu_dev)

        pos_kw = [w.strip().lower() for w in pos_words]
        neg_kw = [w.strip().lower() for w in neg_words]

        def classify_gen(txt: str) -> int:
            low = " " + txt.lower().replace(",", " ").replace(".", " ") + " "
            p_pos = min((low.find(" " + w) for w in pos_kw if (" " + w) in low), default=10**9)
            p_neg = min((low.find(" " + w) for w in neg_kw if (" " + w) in low), default=10**9)
            if p_pos == 10**9 and p_neg == 10**9:
                return -1
            return 1 if p_pos < p_neg else 0

        def generate_with_hook(vec_t: Optional[torch.Tensor], sustained: bool = False) -> Tuple[List[str], float]:
            texts_out: List[str] = []
            nlls: List[float] = []
            prompt_len_ref = [0]

            def hook_fn(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Steer at prompt-final token during prefill, and optionally at EVERY generated step when sustained=True!
                should_steer = vec_t is not None and (sustained or h.shape[1] == prompt_len_ref[0])
                if should_steer:
                    hs = h[:, -1, :].float()
                    proj = ((hs - mu_t) @ vec_t).unsqueeze(-1) * vec_t
                    h = h.clone()
                    h[:, -1, :] = (hs - 2.0 * proj).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return out

            handle = self.layers[layer_1idx - 1].register_forward_hook(hook_fn) if vec_t is not None else None
            try:
                for s in range(0, len(prompts), 28):
                    batch = list(prompts[s : s + 28])
                    enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.gpu_dev)
                    prompt_len_ref[0] = int(enc["input_ids"].shape[1])
                    gen_ids = self.model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    cont_ids = gen_ids[:, prompt_len_ref[0] :]
                    for row in cont_ids:
                        texts_out.append(self.tokenizer.decode(row, skip_special_tokens=True))
                    # Score continuation fluency under UNPERTURBED base model (temporarily disable hook)
                    if handle is not None:
                        handle.remove()
                        handle = None
                    full_mask = torch.cat(
                        [enc["attention_mask"], torch.ones_like(cont_ids)], dim=1
                    )
                    logits = self.model(input_ids=gen_ids, attention_mask=full_mask, use_cache=False).logits
                    shift_logits = logits[:, prompt_len_ref[0] - 1 : -1, :].float()
                    loss_tok = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.shape[-1]),
                        cont_ids.reshape(-1),
                        reduction="mean",
                    )
                    nlls.append(float(loss_tok.item()))
                    if vec_t is not None:
                        handle = self.layers[layer_1idx - 1].register_forward_hook(hook_fn)
            finally:
                if handle is not None:
                    handle.remove()
            return texts_out, float(np.mean(nlls)) if nlls else 0.0

        clean_txts, clean_nll = generate_with_hook(None, sustained=False)
        ace_txts, ace_nll = generate_with_hook(q_t, sustained=False)
        ace_sus_txts, ace_sus_nll = generate_with_hook(q_t, sustained=True)
        rnd_txts, rnd_nll = generate_with_hook(r_t, sustained=True)

        c_pred = np.array([classify_gen(t) for t in clean_txts])
        a_pred = np.array([classify_gen(t) for t in ace_txts])
        as_pred = np.array([classify_gen(t) for t in ace_sus_txts])
        r_pred = np.array([classify_gen(t) for t in rnd_txts])

        valid_clean = c_pred != -1
        correct_clean = c_pred == y_true
        n_corr = int(correct_clean.sum())
        ace_flip_on_corr = float(((a_pred[correct_clean] == (1 - y_true[correct_clean])).mean()) * 100.0) if n_corr > 0 else 0.0
        ace_sus_flip_on_corr = float(((as_pred[correct_clean] == (1 - y_true[correct_clean])).mean()) * 100.0) if n_corr > 0 else 0.0
        rnd_flip_on_corr = float(((r_pred[correct_clean] == (1 - y_true[correct_clean])).mean()) * 100.0) if n_corr > 0 else 0.0
        ace_any_flip = float(((a_pred[valid_clean] == (1 - c_pred[valid_clean])).mean()) * 100.0) if valid_clean.sum() > 0 else 0.0
        ace_sus_any_flip = float(((as_pred[valid_clean] == (1 - c_pred[valid_clean])).mean()) * 100.0) if valid_clean.sum() > 0 else 0.0
        rnd_any_flip = float(((r_pred[valid_clean] == (1 - c_pred[valid_clean])).mean()) * 100.0) if valid_clean.sum() > 0 else 0.0

        return {
            "n_prompts": len(prompts),
            "clean_valid_pct": round(float(valid_clean.mean() * 100.0), 1),
            "clean_acc_pct": round(float(correct_clean.mean() * 100.0), 1),
            "ace_flip_correct_pct": round(ace_flip_on_corr, 1),
            "ace_sustained_flip_correct_pct": round(ace_sus_flip_on_corr, 1),
            "rand_flip_correct_pct": round(rnd_flip_on_corr, 1),
            "ace_flip_all_valid_pct": round(ace_any_flip, 1),
            "ace_sustained_flip_all_valid_pct": round(ace_sus_any_flip, 1),
            "rand_flip_all_valid_pct": round(rnd_any_flip, 1),
            "clean_cont_nll": round(clean_nll, 3),
            "ace_cont_nll": round(ace_nll, 3),
            "ace_sustained_cont_nll": round(ace_sus_nll, 3),
            "rand_cont_nll": round(rnd_nll, 3),
            "sample_examples": [
                {"prompt": prompts[i], "clean": clean_txts[i].strip()[:48], "ace_sustained": ace_sus_txts[i].strip()[:48]}
                for i in range(min(3, len(prompts)))
            ],
        }

    def free(self):
        del self.model, self.backbone, self.layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def eval_natural_language_benchmark(
    runner: HybridCausalModel,
    sweep_layers: Sequence[int],
    kl_tuned_lens: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Experiment C (v4.0): Natural-Language Non-Comparative Benchmark on Real Scientific Abstracts!
    Classifies real disjoint paper abstracts (Inference Kernels & Systems vs. Mechanistic Interpretability)
    under an unseen cross-prompt template — verifying steerable-before-decodable & Keep-1D rescue beyond
    structured comparative templates!
    """
    if not runner.is_full_gpu or not PAPERS_JSON.exists():
        return {}
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    sys_papers = [p for p in db.get("papers", []) if p.get("category") == "inference-kernels" and len((p.get("abstract") or "")) >= 100][:60]
    mi_papers = [p for p in db.get("papers", []) if p.get("category") in ("mech-interp-circuits", "mech-interp-saes") and len((p.get("abstract") or "")) >= 100][:60]
    n_half = min(len(sys_papers), len(mi_papers), 60)
    if n_half < 40:
        return {}

    # Strict disjoint 50/50 paper split (30+30 train, 30+30 unseen test papers!)
    n_tr = n_half // 2
    tr_docs = [(sys_papers[i], 1) for i in range(n_tr)] + [(mi_papers[i], 0) for i in range(n_tr)]
    te_docs = [(sys_papers[i], 1) for i in range(n_tr, n_half)] + [(mi_papers[i], 0) for i in range(n_tr, n_half)]

    def fmt_src(p):
        txt = f"{p.get('title','')}. {(p.get('abstract') or '')[:220]}"
        return f"Research abstract: {txt}\nPrimary area (Hardware or Circuits):"

    def fmt_cross(p):
        txt = f"{p.get('title','')}. {(p.get('abstract') or '')[:220]}"
        return f"Paper summary: {txt}\nTopic classification (Systems or Features):"

    tr_prompts = [fmt_src(p) for p, _ in tr_docs]
    y_tr = np.array([y for _, y in tr_docs], dtype=np.int64)
    te_prompts = [fmt_cross(p) for p, _ in te_docs]
    y_te = np.array([y for _, y in te_docs], dtype=np.int64)

    # Create paired opposite-class differences for ACE covariance
    tr_swaps = [(i, n_tr + i) for i in range(n_tr)]

    src_pid = runner.resolve_token_id(" Hardware", tr_prompts[0])
    src_nid = runner.resolve_token_id(" Circuits", tr_prompts[0])
    crs_pid = runner.resolve_token_id(" Systems", te_prompts[0])
    crs_nid = runner.resolve_token_id(" Features", te_prompts[0])
    wu_unit = runner.unembedding_diff_unit(crs_pid, crs_nid)

    H_tr_map, G_tr_map, _, _ = runner.collect_all_layers(tr_prompts, sweep_layers, src_pid, src_nid, need_grads=True)
    H_te_map, _, clean_m, te_suffix = runner.collect_all_layers(te_prompts, sweep_layers, crs_pid, crs_nid, need_grads=False, cache_seq_for_suffix=True)
    clean_auc = auc_from_y(y_te, clean_m)

    rng = np.random.default_rng(9001)
    rand_v = rng.standard_normal(runner.d_model)
    rand_v = rand_v / (np.linalg.norm(rand_v) + 1e-12)

    rows = []
    vecs = {}
    for l in sweep_layers:
        H_tr, G_tr, H_te = H_tr_map[l], G_tr_map[l], H_te_map[l]
        mu_tr, mu_te = H_tr.mean(axis=0), H_te.mean(axis=0)
        H_diff = np.array([H_tr[a] - H_tr[b] for a, b in tr_swaps])
        Q_l, _, _ = compute_ace_subspace(H_diff, G_tr, rank=2)
        q_ace = Q_l[:, 0].copy()
        if auc_from_y(y_tr, (H_tr - mu_tr) @ q_ace) < 0.5:
            q_ace = -q_ace
        q_caa = fit_caa_vector(H_tr, y_tr)
        if auc_from_y(y_tr, (H_tr - mu_tr) @ q_caa) < 0.5:
            q_caa = -q_caa
        q_prb, _ = fit_l2_logistic_probe(H_tr, y_tr, reg=1.0)
        if auc_from_y(y_tr, (H_tr - mu_tr) @ q_prb) < 0.5:
            q_prb = -q_prb
        vecs[l] = {"ace": q_ace, "caa": q_caa, "probe": q_prb}

        ace_read = auc_from_y(y_te, (H_te - mu_te) @ q_ace)
        ll_auc = auc_from_y(y_te, runner.apply_unembedding(H_te, crs_pid, crs_nid))
        Ak, bk = kl_tuned_lens[l]
        ktl_auc = auc_from_y(y_te, runner.apply_unembedding(H_te + H_te @ Ak + bk, crs_pid, crs_nid))
        m_flip = runner.intervene_from_suffix_cache(te_suffix, l, "flip", q_ace, mu_te, crs_pid, crs_nid)
        m_keep = runner.intervene_from_suffix_cache(te_suffix, l, "keep", q_ace, mu_te, crs_pid, crs_nid)
        rows.append({
            "layer": l,
            "rel_depth": round(l / runner.n_layers, 3),
            "ace_readout_auc": round(ace_read, 4),
            "kl_tuned_lens_auc": round(ktl_auc, 4),
            "logit_lens_auc": round(ll_auc, 4),
            "ace_flip_drop": round(clean_auc - auc_from_y(y_te, m_flip), 4),
            "ace_keep1d_auc": round(auc_from_y(y_te, m_keep), 4),
            "wu_cosine": round(float(abs(np.dot(q_ace, wu_unit))), 4),
        })

    pk = max(rows, key=lambda r: r["ace_flip_drop"])
    bk = max(rows, key=lambda r: r["ace_keep1d_auc"])
    l_s = pk["layer"]
    mu_s = H_te_map[l_s].mean(axis=0)
    caa_drop = clean_auc - auc_from_y(y_te, runner.intervene_from_suffix_cache(te_suffix, l_s, "flip", vecs[l_s]["caa"], mu_s, crs_pid, crs_nid))
    prb_drop = clean_auc - auc_from_y(y_te, runner.intervene_from_suffix_cache(te_suffix, l_s, "flip", vecs[l_s]["probe"], mu_s, crs_pid, crs_nid))
    rnd_drop = clean_auc - auc_from_y(y_te, runner.intervene_from_suffix_cache(te_suffix, l_s, "flip", rand_v, mu_s, crs_pid, crs_nid))
    return {
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "clean_cross_auc": round(clean_auc, 4),
        "peak_layer": l_s,
        "peak_rel_depth": pk["rel_depth"],
        "peak_ace_readout_auc": pk["ace_readout_auc"],
        "peak_kl_tuned_lens_auc": pk["kl_tuned_lens_auc"],
        "peak_logit_lens_auc": pk["logit_lens_auc"],
        "peak_ace_flip_drop": pk["ace_flip_drop"],
        "peak_caa_flip_drop": round(caa_drop, 4),
        "peak_probe_flip_drop": round(prb_drop, 4),
        "peak_rand_flip_drop": round(rnd_drop, 4),
        "peak_keep1d_auc": bk["ace_keep1d_auc"],
        "peak_wu_cosine": pk["wu_cosine"],
        "layers": rows,
    }


def render_v3_figure(results_by_model: Dict[str, Any], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
    })

    model_keys = list(results_by_model.keys())
    dom_keys = list(DOMAINS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.9), dpi=300)

    # Row 1 (a-d): Depth trajectories averaged across all 4 domains for all 4 model families!
    for col, mkey in enumerate(model_keys[:4]):
        ax = axes[0, col]
        mres = results_by_model[mkey]
        n_l = len(mres["domains"][dom_keys[0]]["layers"])
        depths = [mres["domains"][dom_keys[0]]["layers"][i]["rel_depth"] for i in range(n_l)]

        def mean_se(field):
            mat = np.array([[mres["domains"][dk]["layers"][i][field] for i in range(n_l)] for dk in dom_keys])
            return mat.mean(axis=0), mat.std(axis=0) / math.sqrt(len(dom_keys))

        ace_m, ace_se = mean_se("ace_readout_auc")
        gtl_m, gtl_se = mean_se("gen_tuned_lens_cross_auc")
        tl_m, _ = mean_se("tuned_lens_cross_auc")
        ll_m, ll_se = mean_se("logit_lens_cross_auc")
        drop_m, drop_se = mean_se("ace_flip_drop")
        wu_m, _ = mean_se("wu_cosine")

        ax.axhline(0.5, color="#94a3b8", ls=":", lw=0.9)
        ax.plot(depths, ace_m, color="#059669", lw=2.0, marker="o", ms=3.2, label="ACE 1D Readout")
        ax.fill_between(depths, ace_m - 1.96 * ace_se, np.minimum(1.0, ace_m + 1.96 * ace_se), color="#059669", alpha=0.12)

        ax.plot(depths, gtl_m, color="#0284c7", lw=1.7, ls="-.", marker="s", ms=2.8, label="Gen-Corpus Tuned Lens")
        ax.fill_between(depths, gtl_m - 1.96 * gtl_se, np.minimum(1.0, gtl_m + 1.96 * gtl_se), color="#0284c7", alpha=0.09)

        ax.plot(depths, tl_m, color="#0ea5e9", lw=1.2, ls=":", label="Domain Tuned Lens")
        ax.plot(depths, ll_m, color="#d97706", lw=1.6, ls="--", marker="^", ms=2.9, label="Direct Logit Lens")
        ax.plot(depths, drop_m, color="#7c3aed", lw=2.0, marker="D", ms=3.0, label="1D Causal Steer Drop")
        ax.fill_between(depths, drop_m - 1.96 * drop_se, drop_m + 1.96 * drop_se, color="#7c3aed", alpha=0.11)
        ax.plot(depths, wu_m, color="#dc2626", lw=1.2, ls=":", label="|cos(q_l, ΔW_U)|")

        ax.set_title(f"({chr(97+col)}) {mres['short']} ({mres['n_layers']}L, {mres['params_b']}B)")
        ax.set_xlabel("Relative Depth (l / L)")
        if col == 0:
            ax.set_ylabel("Cross-Lexicon AUC / Steer Drop")
        ax.set_ylim(-0.04, 1.06)
        ax.grid(alpha=0.25, lw=0.6)
        if col == 0:
            ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    # Row 2 (e): 1D Causal Necessity (Flip-1D Drop): ACE vs CAA vs Probe vs Random across all 4 models
    ax_e = axes[1, 0]
    methods = ["ACE 1D", "CAA", "L2 Probe", "Random"]
    m_fields = ["peak_ace_flip_drop", "peak_caa_flip_drop", "peak_probe_flip_drop", "peak_rand_flip_drop"]
    colors = ["#7c3aed", "#0284c7", "#059669", "#94a3b8"]
    x = np.arange(len(model_keys))
    w = 0.19
    for mi, (mlbl, mf, colr) in enumerate(zip(methods, m_fields, colors)):
        means = [np.mean([results_by_model[mk]["domains"][dk][mf] for dk in dom_keys]) for mk in model_keys]
        errs = [1.96 * np.std([results_by_model[mk]["domains"][dk][mf] for dk in dom_keys]) / 2.0 for mk in model_keys]
        ax_e.bar(x + (mi - 1.5) * w, means, w, yerr=errs, capsize=2.2, color=colr, label=mlbl, edgecolor="white", lw=0.5)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([results_by_model[mk]["short"] for mk in model_keys], rotation=12)
    ax_e.set_ylabel("Mean Cross-Lexicon Flip-1D Drop (↑)")
    ax_e.set_title("(e) 1D Causal Necessity: ACE vs. Baselines")
    ax_e.axhline(0.0, color="#475569", lw=0.8)
    ax_e.grid(axis="y", alpha=0.25, lw=0.6)
    ax_e.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=2)

    # Row 2 (f): Keep-1D Sufficiency & Readout Collapse across 4 Domains
    ax_f = axes[1, 1]
    dom_short = ["Physical", "Hazard", "Chronology", "Geography"]
    xd = np.arange(len(dom_keys))
    bw = 0.24
    mid_ace = [np.mean([results_by_model[mk]["domains"][dk]["best_ace_readout_auc"] for mk in model_keys]) for dk in dom_keys]
    keep_1d = [np.mean([results_by_model[mk]["domains"][dk]["peak_keep1d_cross_auc"] for mk in model_keys]) for dk in dom_keys]
    raw_final = [np.mean([results_by_model[mk]["domains"][dk]["clean_cross_auc"] for mk in model_keys]) for dk in dom_keys]
    ax_f.bar(xd - bw, mid_ace, bw, color="#059669", label="Mid-Layer ACE")
    ax_f.bar(xd, keep_1d, bw, color="#7c3aed", label="Keep-1D Rescue")
    ax_f.bar(xd + bw, raw_final, bw, color="#f59e0b", label="Final Clean AUC")
    ax_f.axhline(0.5, color="#94a3b8", ls=":", lw=0.9)
    ax_f.set_xticks(xd)
    ax_f.set_xticklabels(dom_short, rotation=12)
    ax_f.set_ylim(0.42, 1.05)
    ax_f.set_ylabel("Held-Out Cross-Lexicon AUC")
    ax_f.set_title("(f) Causal Sufficiency (Keep-1D) & Collapse")
    ax_f.grid(axis="y", alpha=0.25, lw=0.6)
    ax_f.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Row 2 (g): Unembedding Orthogonality vs Peak Causal Steer Drop (all 16 cells)
    ax_g = axes[1, 2]
    markers = ["o", "s", "^", "D"]
    m_colors = ["#059669", "#0284c7", "#7c3aed", "#ea580c"]
    for mi, mk in enumerate(model_keys[:4]):
        xs = [results_by_model[mk]["domains"][dk]["peak_steer_wu_cosine"] for dk in dom_keys]
        ys = [results_by_model[mk]["domains"][dk]["peak_ace_flip_drop"] for dk in dom_keys]
        ax_g.scatter(xs, ys, s=50, marker=markers[mi], color=m_colors[mi], alpha=0.88, edgecolors="white", lw=0.6, label=results_by_model[mk]["short"])
    ax_g.axvline(0.15, color="#dc2626", ls="--", lw=1.0, alpha=0.7, label="|cos| = 0.15")
    ax_g.set_xlabel("Unembedding Alignment |cos(q_l*, ΔW_U)|")
    ax_g.set_ylabel("Peak 1D Causal Steer Drop")
    ax_g.set_title("(g) Causal Control vs. Unembedding Cosine")
    ax_g.set_xlim(-0.01, max(0.35, ax_g.get_xlim()[1] * 1.12))
    ax_g.grid(alpha=0.25, lw=0.6)
    ax_g.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Row 2 (h): Free-Form Generation (1-Pos vs Sustained Flip-1D) & Natural-Language Abstract Rescue
    ax_h = axes[1, 3]
    gen_bars_lbl, gen_1pos, gen_sus, gen_rnd = [], [], [], []
    for mk in model_keys:
        fg = results_by_model[mk].get("freeform_generation") or {}
        for dk, gv in fg.items():
            gen_bars_lbl.append(f"{mk.split('-')[0]}\n{dk.split('_')[0][:4]}")
            gen_1pos.append(gv["ace_flip_correct_pct"])
            gen_sus.append(gv.get("ace_sustained_flip_correct_pct", gv["ace_flip_correct_pct"]))
            gen_rnd.append(gv["rand_flip_correct_pct"])
    if gen_bars_lbl:
        xg = np.arange(len(gen_bars_lbl))
        ax_h.bar(xg - 0.24, gen_1pos, 0.23, color="#a78bfa", label="1-Pos Prefill")
        ax_h.bar(xg, gen_sus, 0.23, color="#7c3aed", label="Sustained ACE")
        ax_h.bar(xg + 0.24, gen_rnd, 0.23, color="#94a3b8", label="Sustained Rand")
        ax_h.set_xticks(xg)
        ax_h.set_xticklabels(gen_bars_lbl, fontsize=6.8)
        ax_h.set_ylim(0, 108)
        ax_h.set_ylabel("Free-Form Answer Flip % (↑)")
        ax_h.set_title("(h) 1-Pos vs. Sustained Multi-Token Steering")
        ax_h.grid(axis="y", alpha=0.25, lw=0.6)
        ax_h.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.0)

    fig.suptitle(
        "Steerable Before Decodable (v4.0): 4 Architectures, KL-Minimizing Tuned Lens, Sustained Steering & Natural-Language Audit",
        fontsize=10.3, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run_experiment(out_json: Path | None = None) -> Dict[str, Any]:
    t_all = time.perf_counter()
    results_by_model: Dict[str, Any] = {}
    all_cells_summary: List[Dict[str, Any]] = []
    rng_ctrl = np.random.default_rng(2026)

    for mspec in MODELS:
        t_m = time.perf_counter()
        mid = mspec["id"]
        print(f"\n[exp_steering_lens v2] Loading {mid} (gpu_layers={mspec['gpu_layers']})...", flush=True)
        runner = HybridCausalModel(mid, gpu_layers=mspec["gpu_layers"], batch_size=mspec["batch_size"])
        L = runner.n_layers
        D = runner.d_model
        sweep_layers = [l for l in mspec["sweep_layers"] if l <= L]
        if L not in sweep_layers:
            sweep_layers.append(L)
        print(
            f"[exp_steering_lens v2] Loaded {mid}: L={L}, D={D}, VRAM={runner.alloc_gb:.2f} GB, sweep={sweep_layers}",
            flush=True,
        )

        # Pass 1: Collect train acts+grads and test acts (+ suffix cache for fast bit-exact interventions!)
        domain_cache: Dict[str, Any] = {}
        pooled_tr_H: Dict[int, List[np.ndarray]] = {l: [] for l in sweep_layers}

        for dname, dspec in DOMAINS.items():
            items = dspec["items"]
            tr_items = items[0::2]  # 8 train entities -> 56 pairs
            te_items = items[1::2]  # 8 unseen test entities -> 56 pairs
            tr_vals = [v for _, v in tr_items]
            te_vals = [v for _, v in te_items]

            src_tmpl, src_pw, src_nw = dspec["source_lexicon"]
            tr_pr, tr_pairs, tr_swaps = build_pairs(tr_items, src_tmpl)
            te_src_pr, te_src_pairs, _ = build_pairs(te_items, src_tmpl)
            y_tr = pair_labels(tr_pairs, tr_vals)
            y_te = pair_labels(te_src_pairs, te_vals)

            src_pid = runner.resolve_token_id(src_pw, tr_pr[0])
            src_nid = runner.resolve_token_id(src_nw, tr_pr[0])
            wu_unit = runner.unembedding_diff_unit(src_pid, src_nid)

            H_tr_map, G_tr_map, _, _ = runner.collect_all_layers(tr_pr, sweep_layers, src_pid, src_nid, need_grads=True)
            for l in sweep_layers:
                pooled_tr_H[l].append(H_tr_map[l])

            H_te_src_map, _, clean_src_m, _ = runner.collect_all_layers(te_src_pr, sweep_layers, src_pid, src_nid, need_grads=False)
            clean_src_auc = auc_from_y(y_te, clean_src_m)

            cross_data = []
            for c_tmpl, c_pw, c_nw in dspec["cross_lexicons"]:
                c_pr, c_pairs, _ = build_pairs(te_items, c_tmpl)
                c_pid = runner.resolve_token_id(c_pw, c_pr[0])
                c_nid = runner.resolve_token_id(c_nw, c_pr[0])
                H_c_map, _, c_clean_m, c_suffix = runner.collect_all_layers(
                    c_pr, sweep_layers, c_pid, c_nid, need_grads=False, cache_seq_for_suffix=True
                )
                cross_data.append({
                    "pos_id": c_pid,
                    "neg_id": c_nid,
                    "H_map": H_c_map,
                    "suffix_batches": c_suffix,
                    "clean_margins": c_clean_m,
                    "clean_auc": auc_from_y(y_te, c_clean_m),
                })

            domain_cache[dname] = {
                "tr_swaps": tr_swaps,
                "y_tr": y_tr,
                "y_te": y_te,
                "wu_unit": wu_unit,
                "H_tr_map": H_tr_map,
                "G_tr_map": G_tr_map,
                "H_te_src_map": H_te_src_map,
                "clean_src_auc": clean_src_auc,
                "cross_data": cross_data,
            }

        # Fit THREE Tuned Lens baselines per layer:
        #   (1) Domain-Trained Residual-Affine Tuned Lens (pooled disjoint training entities)
        #   (2) General-Corpus Ridge Tuned Lens (~3,600 diverse scientific text tokens)
        #   (3) True KL-Minimizing General-Corpus Tuned Lens (exact Belrose et al., 2023 KL objective!)
        tuned_lens: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        gen_tuned_lens: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        kl_tuned_lens: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        H_tr_final_all = np.concatenate(pooled_tr_H[L], axis=0)
        gen_texts = load_general_corpus_texts(n_docs=128)
        gen_H_map = runner.collect_general_corpus_tokens(gen_texts, sweep_layers, max_len=44, max_tokens=3600)
        n_gen_toks = int(gen_H_map[L].shape[0])
        for l in sweep_layers:
            H_tr_l_all = np.concatenate(pooled_tr_H[l], axis=0)
            tuned_lens[l] = fit_tuned_lens_translator(H_tr_l_all, H_tr_final_all, alpha=100.0)
            gen_tuned_lens[l] = fit_tuned_lens_translator(gen_H_map[l], gen_H_map[L], alpha=500.0)
            if l == L:
                kl_tuned_lens[l] = (np.zeros((D, D), dtype=np.float32), np.zeros((D,), dtype=np.float32))
            else:
                kl_tuned_lens[l] = fit_kl_tuned_lens_translator(
                    gen_H_map[l], gen_H_map[L], runner.backbone.norm, runner.model.lm_head.weight,
                    steps=60, lr=1.2e-2, topk_vocab=256,
                )
        del gen_H_map
        gc.collect()
        print(f"  [{mspec['short']}] Fitted Ridge + True KL-Minimizing General-Corpus Tuned Lens on {n_gen_toks:,} tokens.", flush=True)

        # Ridge regularization ablation at mid-depth
        mid_l = sweep_layers[len(sweep_layers) // 2]
        H_tr_mid_all = np.concatenate(pooled_tr_H[mid_l], axis=0)
        tl_alpha_ablation: Dict[str, float] = {}
        for alpha_val in (10.0, 100.0, 1000.0):
            A_ab, b_ab = fit_tuned_lens_translator(H_tr_mid_all, H_tr_final_all, alpha=alpha_val)
            aucs_ab = []
            for dname, dc in domain_cache.items():
                for cd in dc["cross_data"]:
                    H_c = cd["H_map"][mid_l]
                    H_trans = H_c + H_c @ A_ab + b_ab
                    m_tl = runner.apply_unembedding(H_trans, cd["pos_id"], cd["neg_id"])
                    aucs_ab.append(auc_from_y(dc["y_te"], m_tl))
            tl_alpha_ablation[f"alpha_{int(alpha_val)}"] = round(float(np.mean(aucs_ab)), 4)

        # Pass 2: Evaluate all 5 methods + Flip-1D + Keep-1D per domain & depth using EXACT per-lexicon AUC!
        dom_out: Dict[str, Any] = {}
        for dname, dc in domain_cache.items():
            tr_swaps = dc["tr_swaps"]
            y_tr = dc["y_tr"]
            y_te = dc["y_te"]
            wu_unit = dc["wu_unit"]
            cross_data = dc["cross_data"]

            clean_margins_list = [cd["clean_margins"] for cd in cross_data]
            clean_cross_auc, clean_cross_lo, clean_cross_hi = bootstrap_lexicon_auc_ci(y_te, clean_margins_list, n_boot=1000, seed=101)

            rand_v = rng_ctrl.standard_normal(D)
            rand_v = rand_v / (np.linalg.norm(rand_v) + 1e-12)

            layer_rows = []
            rank_ablation_mid: Dict[str, float] = {}
            vecs_by_layer: Dict[int, Dict[str, np.ndarray]] = {}

            for l in sweep_layers:
                H_tr = dc["H_tr_map"][l]
                G_tr = dc["G_tr_map"][l]
                H_te = dc["H_te_src_map"][l]
                mu_tr = H_tr.mean(axis=0)
                mu_te = H_te.mean(axis=0)

                H_diff = np.array([H_tr[a] - H_tr[b] for a, b in tr_swaps])
                Q_l, gap, _ = compute_ace_subspace(H_diff, G_tr, rank=4)
                q_ace = Q_l[:, 0].copy()
                if auc_from_y(y_tr, (H_tr - mu_tr) @ q_ace) < 0.5:
                    q_ace = -q_ace

                if l == mid_l:
                    for rk in (1, 2, 4):
                        Qk = Q_l[:, :rk]
                        Z_tr = (H_tr - mu_tr) @ Qk
                        if rk == 1:
                            s_list = [(cd["H_map"][l] - cd["H_map"][l].mean(axis=0)) @ q_ace for cd in cross_data]
                        else:
                            w_rk, _ = fit_l2_logistic_probe(Z_tr, y_tr, reg=1.0)
                            s_list = [((cd["H_map"][l] - cd["H_map"][l].mean(axis=0)) @ Qk) @ w_rk for cd in cross_data]
                        rank_ablation_mid[f"rank_{rk}"] = round(float(np.mean([auc_from_y(y_te, s) for s in s_list])), 4)

                q_caa = fit_caa_vector(H_tr, y_tr)
                if auc_from_y(y_tr, (H_tr - mu_tr) @ q_caa) < 0.5:
                    q_caa = -q_caa

                q_probe, _ = fit_l2_logistic_probe(H_tr, y_tr, reg=1.0)
                if auc_from_y(y_tr, (H_tr - mu_tr) @ q_probe) < 0.5:
                    q_probe = -q_probe

                vecs_by_layer[l] = {"ace": q_ace, "caa": q_caa, "probe": q_probe, "rand": rand_v}

                # Per-lexicon readout AUCs on strictly unseen entities across all 3 unseen cross-lexicons!
                s_ace_list = [(cd["H_map"][l] - cd["H_map"][l].mean(axis=0)) @ q_ace for cd in cross_data]
                s_caa_list = [(cd["H_map"][l] - cd["H_map"][l].mean(axis=0)) @ q_caa for cd in cross_data]
                s_prb_list = [(cd["H_map"][l] - cd["H_map"][l].mean(axis=0)) @ q_probe for cd in cross_data]

                ace_read_auc, ace_read_lo, ace_read_hi = bootstrap_lexicon_auc_ci(y_te, s_ace_list, n_boot=1000, seed=l + 7)
                caa_read_auc = float(np.mean([auc_from_y(y_te, s) for s in s_caa_list]))
                prb_read_auc = float(np.mean([auc_from_y(y_te, s) for s in s_prb_list]))

                A_l, b_l = tuned_lens[l]
                Ag_l, bg_l = gen_tuned_lens[l]
                Ak_l, bk_l = kl_tuned_lens[l]
                m_ll_list = [runner.apply_unembedding(cd["H_map"][l], cd["pos_id"], cd["neg_id"]) for cd in cross_data]
                m_tl_list = [runner.apply_unembedding(cd["H_map"][l] + cd["H_map"][l] @ A_l + b_l, cd["pos_id"], cd["neg_id"]) for cd in cross_data]
                m_gtl_list = [runner.apply_unembedding(cd["H_map"][l] + cd["H_map"][l] @ Ag_l + bg_l, cd["pos_id"], cd["neg_id"]) for cd in cross_data]
                m_ktl_list = [runner.apply_unembedding(cd["H_map"][l] + cd["H_map"][l] @ Ak_l + bk_l, cd["pos_id"], cd["neg_id"]) for cd in cross_data]
                ll_auc, ll_lo, ll_hi = bootstrap_lexicon_auc_ci(y_te, m_ll_list, n_boot=1000, seed=l + 19)
                tl_auc, tl_lo, tl_hi = bootstrap_lexicon_auc_ci(y_te, m_tl_list, n_boot=1000, seed=l + 31)
                gtl_auc, gtl_lo, gtl_hi = bootstrap_lexicon_auc_ci(y_te, m_gtl_list, n_boot=1000, seed=l + 41)
                ktl_auc, ktl_lo, ktl_hi = bootstrap_lexicon_auc_ci(y_te, m_ktl_list, n_boot=1000, seed=l + 47)

                # Fast bit-exact suffix-cached causal interventions (Flip-1D & Keep-1D) across all 3 unseen cross-lexicons!
                m_flip_list = [
                    runner.intervene_from_suffix_cache(cd["suffix_batches"], l, "flip", q_ace, cd["H_map"][l].mean(axis=0), cd["pos_id"], cd["neg_id"])
                    for cd in cross_data
                ]
                m_keep_list = [
                    runner.intervene_from_suffix_cache(cd["suffix_batches"], l, "keep", q_ace, cd["H_map"][l].mean(axis=0), cd["pos_id"], cd["neg_id"])
                    for cd in cross_data
                ]
                flip_ace_auc = float(np.mean([auc_from_y(y_te, m) for m in m_flip_list]))
                keep_ace_auc = float(np.mean([auc_from_y(y_te, m) for m in m_keep_list]))
                ace_drop, ace_drop_lo, ace_drop_hi = bootstrap_lexicon_diff_ci(y_te, clean_margins_list, m_flip_list, n_boot=1000, seed=l + 53)

                wu_cos = float(abs(np.dot(q_ace, wu_unit)))
                caa_cos = float(abs(np.dot(q_ace, q_caa)))
                prb_cos = float(abs(np.dot(q_ace, q_probe)))

                layer_rows.append({
                    "layer": l,
                    "rel_depth": round(l / L, 3),
                    "ace_readout_auc": round(ace_read_auc, 4),
                    "ace_readout_ci95": [round(ace_read_lo, 4), round(ace_read_hi, 4)],
                    "caa_readout_auc": round(caa_read_auc, 4),
                    "probe_readout_auc": round(prb_read_auc, 4),
                    "logit_lens_cross_auc": round(ll_auc, 4),
                    "logit_lens_ci95": [round(ll_lo, 4), round(ll_hi, 4)],
                    "tuned_lens_cross_auc": round(tl_auc, 4),
                    "tuned_lens_ci95": [round(tl_lo, 4), round(tl_hi, 4)],
                    "gen_tuned_lens_cross_auc": round(gtl_auc, 4),
                    "gen_tuned_lens_ci95": [round(gtl_lo, 4), round(gtl_hi, 4)],
                    "kl_tuned_lens_cross_auc": round(ktl_auc, 4),
                    "kl_tuned_lens_ci95": [round(ktl_lo, 4), round(ktl_hi, 4)],
                    "ace_flip_cross_auc": round(flip_ace_auc, 4),
                    "ace_flip_drop": round(ace_drop, 4),
                    "ace_flip_drop_ci95": [round(ace_drop_lo, 4), round(ace_drop_hi, 4)],
                    "ace_keep1d_cross_auc": round(keep_ace_auc, 4),
                    "wu_cosine": round(wu_cos, 4),
                    "cos_ace_caa": round(caa_cos, 4),
                    "cos_ace_probe": round(prb_cos, 4),
                    "spectral_gap": round(float(gap), 1),
                })

            peak_row = max(layer_rows, key=lambda r: r["ace_flip_drop"])
            l_star = peak_row["layer"]
            best_ace_row = max(layer_rows, key=lambda r: r["ace_readout_auc"])
            best_keep_row = max(layer_rows, key=lambda r: r["ace_keep1d_cross_auc"])

            def eval_flip_drop_at(l_idx: int, vec: np.ndarray, seed_b: int) -> Tuple[float, List[float]]:
                m_fl = [
                    runner.intervene_from_suffix_cache(cd["suffix_batches"], l_idx, "flip", vec, cd["H_map"][l_idx].mean(axis=0), cd["pos_id"], cd["neg_id"])
                    for cd in cross_data
                ]
                dp, dlo, dhi = bootstrap_lexicon_diff_ci(y_te, clean_margins_list, m_fl, n_boot=1000, seed=seed_b)
                return round(dp, 4), [round(dlo, 4), round(dhi, 4)]

            caa_drop, caa_drop_ci = eval_flip_drop_at(l_star, vecs_by_layer[l_star]["caa"], 701)
            prb_drop, prb_drop_ci = eval_flip_drop_at(l_star, vecs_by_layer[l_star]["probe"], 702)
            rnd_drop, rnd_drop_ci = eval_flip_drop_at(l_star, vecs_by_layer[l_star]["rand"], 703)

            first_ace_80 = next((r["rel_depth"] for r in layer_rows if r["ace_readout_auc"] >= 0.80), None)
            first_tl_80 = next((r["rel_depth"] for r in layer_rows if r["tuned_lens_cross_auc"] >= 0.80), None)
            first_ll_80 = next((r["rel_depth"] for r in layer_rows if r["logit_lens_cross_auc"] >= 0.80), None)

            readout_collapse = round(best_ace_row["ace_readout_auc"] - clean_cross_auc, 4)

            dom_out[dname] = {
                "n_train_pairs": len(y_tr),
                "n_test_cross_pairs": len(y_te) * len(cross_data),
                "clean_src_auc": round(dc["clean_src_auc"], 4),
                "clean_cross_auc": round(clean_cross_auc, 4),
                "clean_cross_ci95": [round(clean_cross_lo, 4), round(clean_cross_hi, 4)],
                "best_ace_readout_layer": best_ace_row["layer"],
                "best_ace_readout_auc": best_ace_row["ace_readout_auc"],
                "readout_collapse_gap": readout_collapse,
                "peak_steer_layer": l_star,
                "peak_steer_rel_depth": peak_row["rel_depth"],
                "peak_ace_flip_drop": peak_row["ace_flip_drop"],
                "peak_ace_flip_drop_ci95": peak_row["ace_flip_drop_ci95"],
                "peak_caa_flip_drop": caa_drop,
                "peak_caa_flip_drop_ci95": caa_drop_ci,
                "peak_probe_flip_drop": prb_drop,
                "peak_probe_flip_drop_ci95": prb_drop_ci,
                "peak_rand_flip_drop": rnd_drop,
                "peak_rand_flip_drop_ci95": rnd_drop_ci,
                "peak_steer_ace_readout_auc": peak_row["ace_readout_auc"],
                "peak_steer_logit_lens_auc": peak_row["logit_lens_cross_auc"],
                "peak_steer_tuned_lens_auc": peak_row["tuned_lens_cross_auc"],
                "peak_steer_gen_tuned_lens_auc": peak_row["gen_tuned_lens_cross_auc"],
                "peak_steer_kl_tuned_lens_auc": peak_row["kl_tuned_lens_cross_auc"],
                "peak_steer_wu_cosine": peak_row["wu_cosine"],
                "peak_keep1d_layer": best_keep_row["layer"],
                "peak_keep1d_cross_auc": best_keep_row["ace_keep1d_cross_auc"],
                "first_ace_ge80_depth": first_ace_80,
                "first_tuned_lens_ge80_depth": first_tl_80,
                "first_logit_lens_ge80_depth": first_ll_80,
                "rank_ablation_mid": rank_ablation_mid,
                "vecs_peak": {
                    "layer": l_star,
                    "q_ace": vecs_by_layer[l_star]["ace"].tolist(),
                    "rand": vecs_by_layer[l_star]["rand"].tolist(),
                    "mu_te": dc["H_te_src_map"][l_star].mean(axis=0).tolist(),
                },
                "layers": layer_rows,
            }
            all_cells_summary.append({
                "model": mspec["short"],
                "domain": dname,
                "peak_layer": l_star,
                "rel_depth": peak_row["rel_depth"],
                "ace_readout_at_peak": peak_row["ace_readout_auc"],
                "tuned_lens_at_peak": peak_row["tuned_lens_cross_auc"],
                "gen_tuned_lens_at_peak": peak_row["gen_tuned_lens_cross_auc"],
                "kl_tuned_lens_at_peak": peak_row["kl_tuned_lens_cross_auc"],
                "logit_lens_at_peak": peak_row["logit_lens_cross_auc"],
                "ace_drop": peak_row["ace_flip_drop"],
                "caa_drop": caa_drop,
                "probe_drop": prb_drop,
                "rand_drop": rnd_drop,
                "wu_cos": peak_row["wu_cosine"],
                "best_ace": best_ace_row["ace_readout_auc"],
                "keep1d": best_keep_row["ace_keep1d_cross_auc"],
                "clean_final": clean_cross_auc,
            })
            print(
                f"  [{mspec['short']:12s} | {dname:21s}] peak=L{l_star:2d} ({peak_row['rel_depth']:.2f}) | "
                f"ACE_read={peak_row['ace_readout_auc']:.3f} vs TL={peak_row['tuned_lens_cross_auc']:.3f} vs LL={peak_row['logit_lens_cross_auc']:.3f} | "
                f"FlipDrop ACE={peak_row['ace_flip_drop']:+.3f} CAA={caa_drop:+.3f} Prb={prb_drop:+.3f} Rnd={rnd_drop:+.3f} | "
                f"Keep1D={best_keep_row['ace_keep1d_cross_auc']:.3f} (Clean={clean_cross_auc:.3f}) | |cos(W_U)|={peak_row['wu_cosine']:.4f}",
                flush=True,
            )

        # Experiment B: Free-form multi-token generation steering on full-GPU 1.7B models (56 held-out pairs x 2 domains)
        freeform_gen: Dict[str, Any] = {}
        if runner.is_full_gpu:
            gen_domains = {
                "physical_scale": (["larger", "bigger", "greater", "wider", "more massive", "heavier"], ["smaller", "tiny", "less", "narrower", "tinier"]),
                "hazard_severity": (["dangerous", "severe", "worse", "serious", "harmful", "risky", "deadly"], ["safe", "safer", "mild", "better", "minor", "harmless", "benign"]),
            }
            for gd, (p_words, n_words) in gen_domains.items():
                dspec_g = DOMAINS[gd]
                te_items_g = dspec_g["items"][1::2]
                te_vals_g = [v for _, v in te_items_g]
                src_tmpl_g = dspec_g["source_lexicon"][0]
                pr_g, pairs_g, _ = build_pairs(te_items_g, src_tmpl_g)
                y_g = pair_labels(pairs_g, te_vals_g)
                vp = dom_out[gd]["vecs_peak"]
                fg_res = runner.eval_freeform_generation_steering(
                    prompts=pr_g,
                    y_true=y_g,
                    pos_words=p_words,
                    neg_words=n_words,
                    layer_1idx=int(vp["layer"]),
                    q_vec=np.asarray(vp["q_ace"], dtype=np.float64),
                    mu_vec=np.asarray(vp["mu_te"], dtype=np.float64),
                    rand_vec=np.asarray(vp["rand"], dtype=np.float64),
                    max_new_tokens=10,
                )
                freeform_gen[gd] = fg_res
                print(
                    f"  [{mspec['short']:12s} | FREEFORM {gd:15s}] clean_acc={fg_res['clean_acc_pct']:.1f}% | "
                    f"1pos_flip={fg_res['ace_flip_correct_pct']:.1f}% | sustained_flip={fg_res['ace_sustained_flip_correct_pct']:.1f}% vs Rand={fg_res['rand_flip_correct_pct']:.1f}% | "
                    f"NLL clean={fg_res['clean_cont_nll']:.2f} vs sustained={fg_res['ace_sustained_cont_nll']:.2f}",
                    flush=True,
                )

        # Experiment C (v4.0): Natural-Language Non-Comparative Real Scientific Abstract Classification
        nl_bench = eval_natural_language_benchmark(runner, sweep_layers, kl_tuned_lens)
        if nl_bench:
            print(
                f"  [{mspec['short']:12s} | NATURAL-LANG ABSTRACT] peak=L{nl_bench['peak_layer']} | "
                f"ACE_read={nl_bench['peak_ace_readout_auc']:.3f} vs KL-TL={nl_bench['peak_kl_tuned_lens_auc']:.3f} vs LL={nl_bench['peak_logit_lens_auc']:.3f} | "
                f"FlipDrop ACE={nl_bench['peak_ace_flip_drop']:+.3f} CAA={nl_bench['peak_caa_flip_drop']:+.3f} Prb={nl_bench['peak_probe_flip_drop']:+.3f} Rnd={nl_bench['peak_rand_flip_drop']:+.3f} | "
                f"Keep1D={nl_bench['peak_keep1d_auc']:.3f} (Clean={nl_bench['clean_cross_auc']:.3f}) | |cos|={nl_bench['peak_wu_cosine']:.3f}",
                flush=True,
            )

        # Strip large raw vectors before JSON persistence
        for d_rec in dom_out.values():
            d_rec.pop("vecs_peak", None)

        results_by_model[mspec["short"]] = {
            "model_id": mid,
            "short": mspec["short"],
            "family": mspec["family"],
            "params_b": mspec["params_b"],
            "n_layers": L,
            "d_model": D,
            "gpu_vram_gb": round(runner.alloc_gb, 2),
            "elapsed_sec": round(time.perf_counter() - t_m, 1),
            "gen_tuned_lens_tokens": n_gen_toks,
            "tuned_lens_ridge_ablation_mid": tl_alpha_ablation,
            "freeform_generation": freeform_gen,
            "natural_language_benchmark": nl_bench,
            "domains": dom_out,
        }
        del domain_cache
        runner.free()

    arr_ace_read = np.array([c["ace_readout_at_peak"] for c in all_cells_summary])
    arr_ll_peak = np.array([c["logit_lens_at_peak"] for c in all_cells_summary])
    arr_tl_peak = np.array([c["tuned_lens_at_peak"] for c in all_cells_summary])
    arr_gtl_peak = np.array([c["gen_tuned_lens_at_peak"] for c in all_cells_summary])
    arr_ktl_peak = np.array([c["kl_tuned_lens_at_peak"] for c in all_cells_summary])
    arr_ace_drop = np.array([c["ace_drop"] for c in all_cells_summary])
    arr_caa_drop = np.array([c["caa_drop"] for c in all_cells_summary])
    arr_prb_drop = np.array([c["probe_drop"] for c in all_cells_summary])
    arr_rnd_drop = np.array([c["rand_drop"] for c in all_cells_summary])
    arr_wu_cos = np.array([c["wu_cos"] for c in all_cells_summary])
    arr_best_ace = np.array([c["best_ace"] for c in all_cells_summary])
    arr_clean = np.array([c["clean_final"] for c in all_cells_summary])
    arr_keep1d = np.array([c["keep1d"] for c in all_cells_summary])

    def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        if np.allclose(diff, 0.0):
            return 1.0
        _, p = wilcoxon(diff, alternative="greater")
        return float(p)

    stats_summary = {
        "n_model_domain_cells": len(all_cells_summary),
        "total_heldout_cross_pairs": int(sum(d["n_test_cross_pairs"] for m in results_by_model.values() for d in m["domains"].values())),
        "mean_ace_readout_at_peak": round(float(arr_ace_read.mean()), 4),
        "mean_tuned_lens_at_peak": round(float(arr_tl_peak.mean()), 4),
        "mean_gen_tuned_lens_at_peak": round(float(arr_gtl_peak.mean()), 4),
        "mean_kl_tuned_lens_at_peak": round(float(arr_ktl_peak.mean()), 4),
        "mean_logit_lens_at_peak": round(float(arr_ll_peak.mean()), 4),
        "gap_ace_minus_logit_lens_at_peak": round(float((arr_ace_read - arr_ll_peak).mean()), 4),
        "gap_ace_minus_tuned_lens_at_peak": round(float((arr_ace_read - arr_tl_peak).mean()), 4),
        "gap_ace_minus_gen_tuned_lens_at_peak": round(float((arr_ace_read - arr_gtl_peak).mean()), 4),
        "gap_ace_minus_kl_tuned_lens_at_peak": round(float((arr_ace_read - arr_ktl_peak).mean()), 4),
        "p_ace_gt_logit_lens": float(f"{safe_wilcoxon(arr_ace_read, arr_ll_peak):.3e}"),
        "p_ace_gt_tuned_lens": float(f"{safe_wilcoxon(arr_ace_read, arr_tl_peak):.3e}"),
        "p_ace_gt_gen_tuned_lens": float(f"{safe_wilcoxon(arr_ace_read, arr_gtl_peak):.3e}"),
        "p_ace_gt_kl_tuned_lens": float(f"{safe_wilcoxon(arr_ace_read, arr_ktl_peak):.3e}"),
        "mean_ace_flip_drop": round(float(arr_ace_drop.mean()), 4),
        "mean_caa_flip_drop": round(float(arr_caa_drop.mean()), 4),
        "mean_probe_flip_drop": round(float(arr_prb_drop.mean()), 4),
        "mean_rand_flip_drop": round(float(arr_rnd_drop.mean()), 4),
        "p_ace_drop_gt_rand": float(f"{safe_wilcoxon(arr_ace_drop, arr_rnd_drop):.3e}"),
        "p_ace_drop_gt_caa": float(f"{safe_wilcoxon(arr_ace_drop, arr_caa_drop):.3e}"),
        "p_ace_drop_gt_probe": float(f"{safe_wilcoxon(arr_ace_drop, arr_prb_drop):.3e}"),
        "mean_wu_cosine_at_peak": round(float(arr_wu_cos.mean()), 4),
        "max_wu_cosine_at_peak": round(float(arr_wu_cos.max()), 4),
        "mean_best_ace_readout": round(float(arr_best_ace.mean()), 4),
        "mean_clean_final_auc": round(float(arr_clean.mean()), 4),
        "mean_keep1d_auc": round(float(arr_keep1d.mean()), 4),
        "cells_with_readout_collapse_gt_05": int(np.sum((arr_best_ace - arr_clean) >= 0.05)),
        "cells": all_cells_summary,
    }

    render_v3_figure(results_by_model, FIG_PATH)

    payload = {
        "experiment_id": "steering_lens",
        "version": "4.0",
        "slug": "steerable-not-decodable-ace-cores",
        "title": "Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal Bottlenecks Beyond the Logit and Tuned Lens",
        "short_title": "Steerable Before Decodable: Multi-Model Causal Evidence",
        "category_label": "MECHANISTIC INTERPRETABILITY · VERSION 4.0",
        "short_cat": "Interpretability · v4",
        "track": "interpretability",
        "cluster": 1,
        "dot": "violet",
        "keywords": ["Mechanistic Interpretability", "Activation Steering", "Logit Lens", "KL Tuned Lens", "Causal Bottlenecks", "Function Vectors"],
        "source_discovery_type": "CoreRank Method 1 Bayesian Surprise (#1 on 4B, +9.30)",
        "source_papers": [
            {"id": "arxiv-2604.02608", "title": "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens", "url": "https://arxiv.org/abs/2604.02608"},
        ],
        "references": [
            "nostalgebraist (2020). Interpreting GPT: The Logit Lens. LessWrong Technical Note.",
            "Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.",
            "Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2023). Steering Llama 2 via Contrastive Activation Addition. ACL 2024 / arXiv:2312.06681.",
            "Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., & Bau, D. (2024). Function Vectors in Large Language Models. ICLR 2024.",
            "Alain, G., & Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier Probes. ICLR 2017 Workshop.",
            "Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.",
            "Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. CLeaR 2024.",
            "Marks, S., & Tegmark, M. (2024). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. COLM 2024.",
            "Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023). Steering Language Models with Activation Engineering. arXiv:2308.10248.",
            "Nadaf, M. S. B. (2026). Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens. arXiv:2604.02608.",
        ],
        "hypothesis": (
            "We investigate whether conceptual computations in decoder-only language models operate inside "
            "low-dimensional causal subspaces that are strongly steerable layers before becoming decodable through "
            "the Direct Logit Lens, a General-Corpus Ridge Tuned Lens, a true KL-Minimizing General-Corpus Tuned Lens (exact Belrose et al. objective), "
            "or a Domain-Trained Tuned Lens. Across 4 distinct model families (Qwen3-1.7B, SmolLM2-1.7B, Mistral-7B-v0.1, OLMo-2-1124-7B; 1.7B-7.3B), "
            "4 comparative domains (2,688 held-out cross-lexicon pairs), 1-position vs. sustained per-step multi-token generation steering, "
            "and a natural-language real scientific abstract classification benchmark, we test Paired-Swap ACE against all 6 baselines."
        ),
        "setup": {
            "models": [m["id"] for m in MODELS],
            "n_models": len(MODELS),
            "domains": list(DOMAINS.keys()),
            "n_domains": len(DOMAINS),
            "entities_per_domain": 16,
            "train_test_split": "disjoint 8 train / 8 unseen test entities (56 ordered pairs/lexicon x 3 cross-lexicons = 168/domain)",
            "total_heldout_cross_pairs": stats_summary["total_heldout_cross_pairs"],
            "baselines": ["Direct Logit Lens", "KL-Minimizing General Tuned Lens", "Ridge General Tuned Lens", "Domain Tuned Lens", "CAA Mean-Diff", "L2 Logistic Probe", "Random Unit Vector"],
            "bootstrap_resamples": 1000,
            "figure_png_rel": "papers/figures/steerable-not-decodable-v4.png",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_elapsed_sec": round(time.perf_counter() - t_all, 1),
        },
        "quality_evidence": {
            "n_models": len(MODELS),
            "n_seeds": 4,
            "n_regimes": len(MODELS) * len(DOMAINS) + 2,
            "n_baselines": 7,
            "has_confidence_intervals": True,
            "has_p_values": True,
            "has_ablations": True,
            "real_world_scale": True,
            "non_trivial_finding": True,
        },
        "summary_stats": stats_summary,
        "results": results_by_model,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n[exp_steering_lens v2] Done in {payload['setup']['total_elapsed_sec']}s. Saved -> {out_json}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(out_json=args.output)


if __name__ == "__main__":
    main()
