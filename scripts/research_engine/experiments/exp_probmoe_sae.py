#!/usr/bin/env python3
"""
GPU Investigation 01: Differentiable Marginal-Probability Gradients for BatchTopK Sparse Autoencoders
(Testing CoreRank Cross-Domain Bridge #1: ProbMoE [arxiv:2606.01509] <-> BatchTopK SAEs [arxiv:2412.06410]
 + Dense Contradiction Signal: SAE Cross-Seed Feature Instability [arxiv:2501.16615]).

Protocol:
  1. Loads Qwen/Qwen3-1.7B physically pruned to Stage-1 layer L13/28 (~1.7 GB VRAM — safe alongside Basalt!)
     and extracts real transformer residual-stream token activations across abstracts from `papers-db.json`.
  2. Benchmarks three Sparse Autoencoder (SAE) routing & gradient estimators on identical held-out activations
     across multiple sparsity budgets (k=16, k=32) and 3 independent random seeds:
       - TopK-STE: Fixed per-sample Top-k with straight-through gradients.
       - BatchTopK-STE: Adaptive batch-level Top-(B*k) allocation with straight-through gradients.
       - ProbBatchTopK (Bridge): Forward hard BatchTopK sparsity paired with a ProbMoE-inspired smooth
         marginal-selection-probability surrogate gradient p_j = sigmoid((z_j - tau)/T) calibrated via
         Newton/bisection so E[sum_j p_j] = B*k, giving inactive near-margin latents smooth gradient flow!
  3. Measures honest empirical metrics on held-out test tokens:
       - Fraction Variance Unexplained (FVU) & Explained Variance (R^2)
       - Dead-latent rate (% dictionary atoms with zero test activations)
       - Cross-seed feature stability (mean max-cosine similarity across independent random seeds)
       - Active L0 distribution & GPU wall-clock runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PAPERS_JSON = Path("/home/marcio/Projects/spaceresearch.ai/data/papers-db.json")
MODEL_ID = "Qwen/Qwen3-1.7B"
EXTRACT_LAYER = 13  # Stage-1 rigor/concept layer (46% depth of 28L Qwen3-1.7B)


def extract_real_activations(
    n_papers: int = 240,
    max_tokens_per_paper: int = 96,
    layer_1idx: int = EXTRACT_LAYER,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Physically prune Qwen3-1.7B to `layer_1idx` on CPU first, move only layers[:layer_1idx]
    to GPU (~1.7 GB VRAM), and extract real residual-stream token vectors from papers-db.json.
    """
    t0 = time.perf_counter()
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers = db["papers"][:n_papers]
    texts = [f"{p['title']}. {(p.get('abstract') or '')[:420]}" for p in papers]

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
    )
    backbone = full_model.model
    d_model = int(full_model.config.hidden_size)
    n_layers_orig = len(backbone.layers)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed = backbone.embed_tokens.to(dev).eval()
    rotary = backbone.rotary_emb.to(dev).eval()
    layers = nn.ModuleList([backbone.layers[i].to(dev).eval() for i in range(layer_1idx)])
    del full_model, backbone
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    vram_extract_gb = torch.cuda.memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0

    chunks: List[torch.Tensor] = []
    bs = 12
    with torch.inference_mode():
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens_per_paper)
            ids = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)
            B, T = ids.shape
            pos_ids = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0).expand(B, T)
            h = embed(ids)
            pos_emb = rotary(h, pos_ids)
            # Right-padded causal SDPA (attention_mask=None with right padding is strictly causal for real tokens)
            for layer in layers:
                out = layer(h, attention_mask=None, position_ids=pos_ids, position_embeddings=pos_emb, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            # Drop BOS (pos 0, attention sink) and padding tokens
            valid = mask.bool()
            valid[:, 0] = False
            vecs = h[valid].float().cpu()
            chunks.append(vecs)

    del embed, rotary, layers
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    X = torch.cat(chunks, dim=0)  # (N_tok, d_model)
    # Center and unit-RMS normalize for clean SAE conditioning
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    rms = torch.sqrt(torch.mean(Xc.pow(2))) + 1e-8
    Xn = Xc / rms

    meta = {
        "source_model": MODEL_ID,
        "layer_extracted": f"L{layer_1idx}/{n_layers_orig}",
        "d_model": d_model,
        "papers_sampled": n_papers,
        "total_tokens": int(Xn.shape[0]),
        "extract_vram_gb": round(vram_extract_gb, 2),
        "extract_sec": round(time.perf_counter() - t0, 2),
    }
    return Xn, meta


def solve_marginal_threshold(z: torch.Tensor, target_sum: float, temp: float, n_iter: int = 12) -> torch.Tensor:
    """
    Solve for scalar threshold tau such that sum(sigmoid((z - tau) / temp)) == target_sum
    via vectorized bisection on GPU (exact marginal selection probabilities inspired by ProbMoE).
    """
    with torch.no_grad():
        lo = z.min() - 6.0 * temp
        hi = z.max() + 6.0 * temp
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            s = torch.sigmoid((z - mid) / temp).sum()
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


class BridgeSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, mode: str, k: int, prob_temp: float = 0.25):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.mode = mode  # "topk_ste" | "batchtopk_ste" | "probbatchtopk"
        self.k = k
        self.prob_temp = prob_temp

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor, temp_override: float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc  # (B, d_sae)
        z_pos = F.relu(pre)
        B, M = z_pos.shape

        if self.mode == "topk_ste":
            vals, idx = torch.topk(z_pos, k=self.k, dim=-1)
            acts = torch.zeros_like(z_pos)
            acts.scatter_(-1, idx, vals)
            return acts, pre

        if self.mode == "batchtopk_ste":
            flat = z_pos.reshape(-1)
            total_k = min(B * self.k, flat.numel())
            vals, idx = torch.topk(flat, k=total_k)
            acts_flat = torch.zeros_like(flat)
            acts_flat.scatter_(0, idx, vals)
            return acts_flat.reshape(B, M), pre

        if self.mode == "probbatchtopk":
            # Forward: exact hard BatchTopK sparsity!
            flat = z_pos.reshape(-1)
            total_k = min(B * self.k, flat.numel())
            vals, idx = torch.topk(flat, k=total_k)
            hard_flat = torch.zeros_like(flat)
            hard_flat.scatter_(0, idx, vals)
            hard_acts = hard_flat.reshape(B, M)

            if not self.training:
                return hard_acts, pre

            # Backward surrogate (ProbMoE bridge): exact marginal inclusion probability p_j = sigmoid((pre_j - tau)/T)
            # calibrated so sum(p_j) == B * k. Gives near-margin & inactive features smooth probabilistic gradients!
            T = float(temp_override if temp_override is not None else self.prob_temp)
            tau = solve_marginal_threshold(pre, float(total_k), temp=T, n_iter=12)
            p_marg = torch.sigmoid((pre - tau) / T)
            soft_acts = p_marg * F.softplus(pre, beta=4.0)
            # Straight-through hybrid: exact hard forward, ProbMoE marginal-probability backward surrogate!
            acts = hard_acts + (soft_acts - soft_acts.detach())
            return acts, pre

        raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, x: torch.Tensor, temp_override: float | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts, pre = self.encode(x, temp_override=temp_override)
        x_hat = acts @ self.W_dec + self.b_dec
        return x_hat, acts, pre


def train_and_eval_sae(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    mode: str,
    k: int,
    d_sae: int,
    seed: int,
    steps: int = 380,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in = int(X_train.shape[1])
    model = BridgeSAE(d_in=d_in, d_sae=d_sae, mode=mode, k=k, prob_temp=0.35).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N_tr = int(X_train.shape[0])
    t0 = time.perf_counter()
    model.train()

    for step in range(steps):
        idx = torch.randint(0, N_tr, (batch_size,), device=device)
        xb = X_train[idx]
        # Anneal ProbMoE marginal temperature smoothly from 0.45 -> 0.12
        frac = step / max(steps - 1, 1)
        temp = 0.45 * (1.0 - frac) + 0.12 * frac

        x_hat, acts, pre = model(xb, temp_override=temp)
        loss = F.mse_loss(x_hat, xb)

        # Standard tiny aux balance / dead-feature pressure only for ProbBatchTopK? NO! Keep loss 100% IDENTICAL (pure MSE)
        # so the ONLY variable tested is the routing/gradient estimator itself!
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.normalize_decoder_()

    train_sec = time.perf_counter() - t0

    # Evaluate on held-out test set
    model.eval()
    total_se = 0.0
    total_var = 0.0
    test_mean = X_test.mean(dim=0, keepdim=True)
    fire_counts = torch.zeros(d_sae, dtype=torch.long, device=device)
    l0_sum = 0.0
    n_te = int(X_test.shape[0])

    with torch.inference_mode():
        for start in range(0, n_te, 512):
            xb = X_test[start : start + 512]
            x_hat, acts, _ = model(xb)
            total_se += float((xb - x_hat).pow(2).sum().item())
            total_var += float((xb - test_mean).pow(2).sum().item())
            active = acts > 0
            fire_counts += active.sum(dim=0)
            l0_sum += float(active.float().sum(dim=-1).sum().item())

    fvu = total_se / max(total_var, 1e-12)
    r2 = 1.0 - fvu
    dead_rate = float((fire_counts == 0).float().mean().item())
    rare_rate = float((fire_counts < 3).float().mean().item())
    mean_l0 = l0_sum / max(n_te, 1)

    # Extract unit-norm decoder atoms for alive features (for cross-seed stability!)
    W_dec_np = model.W_dec.detach().float().cpu().numpy()
    alive_mask = (fire_counts > 0).cpu().numpy()
    W_alive = W_dec_np[alive_mask] if alive_mask.sum() > 0 else W_dec_np[:1]

    metrics = {
        "fvu": round(fvu, 5),
        "r2": round(r2, 5),
        "dead_latent_pct": round(dead_rate * 100.0, 2),
        "rare_latent_pct": round(rare_rate * 100.0, 2),
        "mean_l0": round(mean_l0, 2),
        "train_sec": round(train_sec, 2),
    }
    return metrics, W_alive


def cross_seed_stability(decoders_by_seed: List[np.ndarray], threshold: float = 0.70) -> Tuple[float, float]:
    """
    Compute symmetric mean max-cosine similarity AND high-overlap match rate (cos >= 0.70)
    between alive SAE feature dictionaries learned under different random seeds.
    """
    max_cos_vals = []
    match_rates = []
    for i in range(len(decoders_by_seed)):
        for j in range(i + 1, len(decoders_by_seed)):
            A = decoders_by_seed[i]
            B = decoders_by_seed[j]
            # Unit normalize defensively
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            S = np.abs(A @ B.T)
            m_ab = S.max(axis=1)
            m_ba = S.max(axis=0)
            max_cos_vals.append(0.5 * (float(m_ab.mean()) + float(m_ba.mean())))
            match_rates.append(0.5 * (float((m_ab >= threshold).mean()) + float((m_ba >= threshold).mean())))
    return round(float(np.mean(max_cos_vals)), 4), round(float(np.mean(match_rates)) * 100.0, 2)


def run_experiment(
    out_json: Path | None = None,
    d_sae: int = 2048,
    k_values: Sequence[int] = (16, 32),
    seeds: Sequence[int] = (11, 22, 33),
    steps: int = 400,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp_probmoe_sae] Extracting real {MODEL_ID} L{EXTRACT_LAYER} activations on {dev}...", flush=True)
    X_all, ext_meta = extract_real_activations(n_papers=240, max_tokens_per_paper=96, device=str(dev))

    # Deterministic 80/20 train/test split
    g = torch.Generator().manual_seed(2026)
    perm = torch.randperm(X_all.shape[0], generator=g)
    n_tr = int(0.8 * X_all.shape[0])
    X_train = X_all[perm[:n_tr]].to(dev)
    X_test = X_all[perm[n_tr:]].to(dev)
    print(
        f"[exp_probmoe_sae] Extracted {X_all.shape[0]} tokens (d={X_all.shape[1]}) in {ext_meta['extract_sec']}s "
        f"(train={n_tr}, test={X_all.shape[0]-n_tr}, extract VRAM={ext_meta['extract_vram_gb']} GB)",
        flush=True,
    )

    modes = ["topk_ste", "batchtopk_ste", "probbatchtopk"]
    grid_results: Dict[str, Any] = {}

    for k in k_values:
        k_key = f"k{k}"
        grid_results[k_key] = {}
        for mode in modes:
            seed_runs = []
            decoders = []
            for seed in seeds:
                m, W_alive = train_and_eval_sae(
                    X_train, X_test, mode=mode, k=k, d_sae=d_sae, seed=seed, steps=steps, device=dev
                )
                seed_runs.append({"seed": seed, **m})
                decoders.append(W_alive)
            mean_cos, match_pct = cross_seed_stability(decoders, threshold=0.70)
            agg = {
                "fvu_mean": round(float(np.mean([r["fvu"] for r in seed_runs])), 5),
                "fvu_std": round(float(np.std([r["fvu"] for r in seed_runs])), 5),
                "r2_mean": round(float(np.mean([r["r2"] for r in seed_runs])), 5),
                "dead_latent_pct_mean": round(float(np.mean([r["dead_latent_pct"] for r in seed_runs])), 2),
                "rare_latent_pct_mean": round(float(np.mean([r["rare_latent_pct"] for r in seed_runs])), 2),
                "mean_l0": round(float(np.mean([r["mean_l0"] for r in seed_runs])), 2),
                "seed_max_cos": mean_cos,
                "seed_shared_70_pct": match_pct,
                "train_sec_mean": round(float(np.mean([r["train_sec"] for r in seed_runs])), 2),
                "runs": seed_runs,
            }
            grid_results[k_key][mode] = agg
            print(
                f"  [{k_key}][{mode:14s}] FVU={agg['fvu_mean']:.4f} (R2={agg['r2_mean']:.4f}) | "
                f"dead={agg['dead_latent_pct_mean']:5.2f}% | seed_cos={agg['seed_max_cos']:.4f} "
                f"(shared>=0.70: {agg['seed_shared_70_pct']:5.1f}%) | L0={agg['mean_l0']:.1f}",
                flush=True,
            )

    # Summarize primary comparison at k=16 and k=32
    k16_base = grid_results[f"k{k_values[0]}"]["topk_ste"]
    k16_btk = grid_results[f"k{k_values[0]}"]["batchtopk_ste"]
    k16_prob = grid_results[f"k{k_values[0]}"]["probbatchtopk"]
    k32_btk = grid_results[f"k{k_values[-1]}"]["batchtopk_ste"]
    k32_prob = grid_results[f"k{k_values[-1]}"]["probbatchtopk"]

    payload = {
        "experiment_id": "probmoe_sae",
        "slug": "probmoe-batchtopk-sae",
        "title": "Probabilistic Marginal Gradients for Sparse Autoencoders: Bridging ProbMoE Routing and BatchTopK SAEs",
        "short_title": "ProbMoE Marginal Gradients for BatchTopK SAEs",
        "category_label": "AUTONOMOUS GPU LAB · INTERPRETABILITY & ROUTING",
        "short_cat": "Auto GPU · SAEs",
        "track": "interpretability",
        "cluster": 1,
        "dot": "violet",
        "source_discovery_type": "Method 2 Cross-Domain Bridge + Method 3 Dense Contradiction",
        "source_papers": [
            {"id": "arxiv-2606.01509", "title": "ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts", "url": "https://arxiv.org/abs/2606.01509"},
            {"id": "arxiv-2412.06410", "title": "BatchTopK Sparse Autoencoders", "url": "https://arxiv.org/abs/2412.06410"},
            {"id": "arxiv-2501.16615", "title": "Sparse Autoencoders Trained on the Same Data Learn Different Features", "url": "https://arxiv.org/abs/2501.16615"},
        ],
        "hypothesis": (
            "CoreRank Method 2 discovered a low-BGE (0.706), high-ACE-mechanism (0.879, 4B transfer +12.31) bridge between "
            "ProbMoE's differentiable marginal-probability subset routing and BatchTopK Sparse Autoencoders. We test whether "
            "replacing straight-through TopK gradients with ProbMoE's calibrated marginal-selection surrogate p_j = sigmoid((z_j - tau)/T) "
            "under identical hard BatchTopK forward sparsity and pure MSE loss improves reconstruction fidelity (FVU), eliminates dead latents, "
            "and alters cross-seed feature stability on real Qwen3-1.7B residual-stream activations."
        ),
        "setup": {
            **ext_meta,
            "d_sae": d_sae,
            "expansion_factor": round(d_sae / ext_meta["d_model"], 2),
            "k_values": list(k_values),
            "seeds": list(seeds),
            "steps_per_run": steps,
            "train_tokens": n_tr,
            "test_tokens": int(X_all.shape[0] - n_tr),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_elapsed_sec": round(time.perf_counter() - t_all, 1),
        },
        "results": grid_results,
        "headline_deltas": {
            "k16_dead_latent_topk_pct": k16_base["dead_latent_pct_mean"],
            "k16_dead_latent_batchtopk_pct": k16_btk["dead_latent_pct_mean"],
            "k16_dead_latent_probbatchtopk_pct": k16_prob["dead_latent_pct_mean"],
            "k16_fvu_delta_vs_batchtopk": round(k16_prob["fvu_mean"] - k16_btk["fvu_mean"], 5),
            "k16_fvu_delta_vs_topk": round(k16_prob["fvu_mean"] - k16_base["fvu_mean"], 5),
            "k16_seed_cos_delta_vs_batchtopk": round(k16_prob["seed_max_cos"] - k16_btk["seed_max_cos"], 4),
            "k32_fvu_delta_vs_batchtopk": round(k32_prob["fvu_mean"] - k32_btk["fvu_mean"], 5),
            "k32_dead_latent_probbatchtopk_pct": k32_prob["dead_latent_pct_mean"],
            "k32_dead_latent_batchtopk_pct": k32_btk["dead_latent_pct_mean"],
            "k32_seed_cos_delta_vs_batchtopk": round(k32_prob["seed_max_cos"] - k32_btk["seed_max_cos"], 4),
        },
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[exp_probmoe_sae] Saved results to {out_json}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    run_experiment(out_json=args.output, d_sae=args.d_sae, steps=args.steps)


if __name__ == "__main__":
    main()
