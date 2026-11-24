#!/usr/bin/env python3
"""
GPU Investigation 02: Numerical Instability & Silent Sign-Flips of PyTorch F.kl_div on Nearby Distributions
(Reproducing & Extending CoreRank Method 1 Discovery #3: arxiv:2611.30835 — "The KL Divergence Is Never Negative, Except in PyTorch").

Tests naive F.kl_div / log-space p*(log p - log q) across FP16, BF16, FP32 vs certified FP64 reference
on RTX 5090, and benchmarks an exact stabilized Taylor/expm1 formulation:
  D_KL(p || q) = sum_i p_i * (expm1(log_q_i - log_p_i) - (log_q_i - log_p_i))  [strictly >= 0 everywhere!]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


def kl_exact_fp64(log_p64: torch.Tensor, log_q64: torch.Tensor) -> torch.Tensor:
    # Exact nonnegative Bregman form: sum_i p_i * (exp(delta_i) - 1 - delta_i), delta_i = log_q_i - log_p_i
    delta = log_q64 - log_p64
    p = torch.exp(log_p64)
    return torch.sum(p * (torch.expm1(delta) - delta), dim=-1)


def kl_bregman_stable(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """
    Strictly non-negative Bregman KL formulation valid in FP16/BF16/FP32:
      p * (expm1(d) - d) = p * (d^2/2 + d^3/6 + ...)
    Uses Taylor expansion for |d| < 1e-2 to prevent catastrophic cancellation in expm1(d) - d!
    """
    d = log_q - log_p
    p = torch.exp(log_p)
    small = d.abs() < 1e-2
    taylor = 0.5 * d * d + (1.0 / 6.0) * d * d * d + (1.0 / 24.0) * d * d * d * d
    exact = torch.expm1(d) - d
    term = torch.where(small, taylor, exact)
    return torch.sum(p * term, dim=-1)


def run_experiment(out_json: Path | None = None, n_samples: int = 8192, vocab: int = 256) -> Dict[str, Any]:
    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)

    # Test regimes spanning ultra-close (eps=1e-4) to moderate (eps=1e-1) distribution perturbations
    epsilons = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    dtypes = [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]

    base_logits = torch.randn(n_samples, vocab, dtype=torch.float64, device=dev) * 1.5
    noise_dir = torch.randn(n_samples, vocab, dtype=torch.float64, device=dev)

    records: Dict[str, Any] = {}
    for dt_name, dt in dtypes:
        records[dt_name] = {}
        for eps in epsilons:
            log_p64 = F.log_softmax(base_logits, dim=-1)
            log_q64 = F.log_softmax(base_logits + eps * noise_dir, dim=-1)
            ref = kl_exact_fp64(log_p64, log_q64).clamp(min=1e-30)

            lp = log_p64.to(dt)
            lq = log_q64.to(dt)

            # 1. Standard PyTorch F.kl_div(input=log_q, target=log_p, log_target=True)
            kl_torch = F.kl_div(lq, lp, reduction="none", log_target=True).sum(dim=-1).double()
            # 2. Stabilized Bregman-Taylor KL in the SAME dtype!
            kl_stab = kl_bregman_stable(lp, lq).double()

            rel_err_torch = ((kl_torch - ref).abs() / ref).cpu().numpy()
            rel_err_stab = ((kl_stab - ref).abs() / ref).cpu().numpy()

            records[dt_name][f"eps_{eps:g}"] = {
                "mean_ref_kl": float(f"{ref.mean().item():.3e}"),
                "torch_neg_pct": round(float((kl_torch < 0).double().mean().item()) * 100.0, 2),
                "torch_rel_err_gt1pct": round(float(np.mean(rel_err_torch >= 0.01)) * 100.0, 2),
                "torch_med_rel_err": round(float(np.median(rel_err_torch)), 4),
                "bregman_neg_pct": round(float((kl_stab < 0).double().mean().item()) * 100.0, 2),
                "bregman_rel_err_gt1pct": round(float(np.mean(rel_err_stab >= 0.01)) * 100.0, 2),
                "bregman_med_rel_err": round(float(np.median(rel_err_stab)), 4),
            }

    payload = {
        "experiment_id": "kl_pytorch",
        "slug": "pytorch-kl-bregman-stability",
        "title": "Eliminating Negative & Corrupted KL Divergence in Low-Precision PyTorch via Bregman-Taylor Formulation",
        "short_title": "Fixing Negative & Corrupted KL Divergence in PyTorch",
        "category_label": "AUTONOMOUS GPU LAB · NUMERICAL STABILITY",
        "short_cat": "Auto GPU · Numerics",
        "track": "systems",
        "cluster": 2,
        "dot": "emerald",
        "source_discovery_type": "Method 1 Prior-vs-Evidence Bayesian Surprise (#3 on 4B)",
        "source_papers": [
            {"id": "arxiv-2611.30835", "title": "The KL Divergence Is Never Negative, Except in PyTorch", "url": "https://arxiv.org/abs/2611.30835"},
        ],
        "hypothesis": (
            "CoreRank Method 1 flagged arxiv:2611.30835 (4B surprise +7.65), showing PyTorch F.kl_div yields negative values "
            "and large relative errors for nearby distributions. We reproduce this on RTX 5090 across FP16/BF16/FP32 and test "
            "whether a strictly nonnegative Bregman-Taylor form p*(expm1(d)-d) eliminates negative KL and catastrophic cancellation."
        ),
        "setup": {
            "n_samples": n_samples,
            "vocab": vocab,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "results": records,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(out_json=args.output)


if __name__ == "__main__":
    main()
