#!/usr/bin/env python3
"""
GPU Investigation 03: Steerable but Not Decodable — Measuring Causal ACE Steering vs.
Direct Logit-Lens Readability Across Transformer Depth in Qwen/Qwen3-1.7B.

Triggered by CoreRank Method 1 Discovery #1 (arxiv:2604.02608, 4B Surprise +9.30, 4B Rigor 7.51):
  "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens"

Protocol:
  Across two distinct domains (physical planetary diameter & real-world hazard severity) and
  11 depths L4..L28 of Qwen/Qwen3-1.7B (3.2 GB VRAM):
    1. Extract a 1D Paired-Swap ACE core q_l on a strict training split.
    2. Measure on held-out pairs & cross-lexicon templates:
       - Linear ACE Core Readout AUC: ROC-AUC of (h_l - mu_l) . q_l
       - Causal Steerability (Cross-Lexicon Flip-1D Reversal): how strongly reflecting the single
         1D coordinate along q_l at layer l reverses the final output across unseen lexicons
         (steer_strength = clean_auc - flip_auc, higher => stronger 1D causal control).
       - Direct Logit-Lens Decodability AUC: ROC-AUC of applying final norm + W_U directly to h_l.
       - Unembedding Alignment: |cos(q_l, W_U[pos] - W_U[neg])|.
  Tests whether CoreRank's conceptual bottleneck operates in a subspace orthogonal to the unembedding
  ("steerable before decodable") — explaining why Stage-1/Stage-2 ACE succeeds layers before Logit Lens!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from corerank.engine import CoreRankEngine, compute_ace_subspace, roc_auc

MODEL_ID = "Qwen/Qwen3-1.7B"

DOMAINS: Dict[str, Dict[str, Any]] = {
    "planets": {
        "items": [
            ("Mercury", 4879.0),
            ("Mars", 6779.0),
            ("Venus", 12104.0),
            ("Earth", 12742.0),
            ("Neptune", 49244.0),
            ("Uranus", 50724.0),
            ("Saturn", 116460.0),
            ("Jupiter", 139820.0),
        ],
        "source_lexicon": ("Comparing planetary diameter, {A} compared to {B} is", " larger", " smaller"),
        "cross_lexicons": [
            ("Comparing physical size, {A} compared to {B} is", " bigger", " tinier"),
            ("Comparing planetary width, {A} compared to {B} is", " wider", " narrower"),
        ],
    },
    "hazard_severity": {
        "items": [
            ("watering a houseplant", 1.0),
            ("folding clean laundry", 2.0),
            ("forgetting an umbrella", 3.0),
            ("leaving a stove burner unattended", 4.0),
            ("driving without headlights at night", 5.0),
            ("texting while driving on a highway", 6.0),
            ("mixing bleach and ammonia indoors", 7.0),
            ("tampering with a hospital oxygen line", 8.0),
            ("derailing a passenger train", 9.0),
            ("releasing a lethal nerve agent in a subway", 10.0),
        ],
        "source_lexicon": ("Comparing real-world danger, {A} compared to {B} is", " dangerous", " safe"),
        "cross_lexicons": [
            ("Comparing potential harm, {A} compared to {B} is", " harmful", " harmless"),
            ("Comparing safety risk, {A} compared to {B} is", " risky", " benign"),
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


def auc_pairs(pairs, vals, scores) -> float:
    y = [1 if vals[i] > vals[j] else 0 for i, j in pairs]
    return float(roc_auc(y, scores))


@torch.no_grad()
def flip_1d_margins(engine: CoreRankEngine, prompts: Sequence[str], layer_1idx: int, pos_id: int, neg_id: int, q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    q_t = torch.tensor(q, dtype=torch.float32, device=engine.device)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=engine.device)
    out_m = []
    for start in range(0, len(prompts), engine.batch_size):
        batch = prompts[start : start + engine.batch_size]
        input_ids, attn_mask = engine._tokenize_batch(batch)
        B, T = input_ids.shape
        b_ar = torch.arange(B, device=engine.device)

        def hook(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).clone()
            hs = h[b_ar, T - 1, :].float()
            proj = ((hs - mu_t) @ q_t).unsqueeze(-1) * q_t
            h[b_ar, T - 1, :] = (hs - 2.0 * proj).to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        hdl = engine.layers[layer_1idx - 1].register_forward_hook(hook)
        try:
            logits = engine.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, -1, :]
            out_m.append((logits[:, pos_id] - logits[:, neg_id]).float().cpu().numpy())
        finally:
            hdl.remove()
    return np.concatenate(out_m, axis=0)


@torch.no_grad()
def logit_lens_margins(engine: CoreRankEngine, H_np: np.ndarray, pos_id: int, neg_id: int) -> np.ndarray:
    """Apply final RMSNorm + lm_head unembedding W_U directly to intermediate residual stream H_l."""
    h_t = torch.tensor(H_np, dtype=engine.dtype, device=engine.device)
    h_norm = engine.backbone.norm(h_t)
    w_u = engine.model.lm_head.weight  # (V, D)
    diff_w = (w_u[pos_id].float() - w_u[neg_id].float())
    return (h_norm.float() @ diff_w).cpu().numpy()


def run_experiment(out_json: Path | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = CoreRankEngine(MODEL_ID, dtype="bf16", batch_size=8)
    L = engine.n_layers
    sweep_layers = [4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    sweep_layers = [l for l in sweep_layers if l <= L]

    w_u = engine.model.lm_head.weight.detach().float().cpu().numpy()
    domain_results: Dict[str, Any] = {}

    for dom_name, spec in DOMAINS.items():
        items = spec["items"]
        train_items = items[0::2]  # strict disjoint half for core extraction
        test_items = items[1::2]   # strictly unseen held-out entities for evaluation!
        te_vals = [v for _, v in test_items]

        src_tmpl, src_pw, src_nw = spec["source_lexicon"]
        tr_pr, tr_pairs, tr_swaps = build_pairs(train_items, src_tmpl)
        src_pid = engine.resolve_token_id(src_pw, sample_prompt=tr_pr[0])
        src_nid = engine.resolve_token_id(src_nw, sample_prompt=tr_pr[0])

        # Unembedding contrast vector for source lexicon
        wu_diff = w_u[src_pid] - w_u[src_nid]
        wu_diff = wu_diff / (np.linalg.norm(wu_diff) + 1e-12)

        # Prepare strictly held-out test prompts for source + cross-lexicons
        te_src_pr, te_src_pairs, _ = build_pairs(test_items, src_tmpl)
        clean_src_auc = auc_pairs(te_src_pairs, te_vals, engine.get_margins(te_src_pr, src_pid, src_nid))

        cross_prep = []
        for c_tmpl, c_pw, c_nw in spec["cross_lexicons"]:
            c_pr, c_pairs, _ = build_pairs(test_items, c_tmpl)
            c_pid = engine.resolve_token_id(c_pw, sample_prompt=c_pr[0])
            c_nid = engine.resolve_token_id(c_nw, sample_prompt=c_pr[0])
            c_clean = auc_pairs(c_pairs, te_vals, engine.get_margins(c_pr, c_pid, c_nid))
            cross_prep.append((c_pr, c_pairs, c_pid, c_nid, c_clean))

        layer_rows = []
        for l in sweep_layers:
            H_tr, G_tr, _ = engine.collect_acts_and_grads(tr_pr, l, src_pid, src_nid, spans=None, pooling="last")
            H_diff = np.array([H_tr[a] - H_tr[b] for a, b in tr_swaps])
            Q_l, gap, _ = compute_ace_subspace(H_diff, G_tr, rank=1)
            q_l = Q_l[:, 0]
            mu_tr = H_tr.mean(axis=0)
            # Orient q_l positively on training pairs
            tr_vals = [v for _, v in train_items]
            if auc_pairs(tr_pairs, tr_vals, (H_tr - mu_tr) @ q_l) < 0.5:
                q_l = -q_l

            # Held-out source activations at layer l
            H_te = engine.extract_early_exit(te_src_pr, l, spans=None, pooling="last")
            mu_te = H_te.mean(axis=0)
            ace_readout_auc = auc_pairs(te_src_pairs, te_vals, (H_te - mu_te) @ q_l)
            ll_src_auc = auc_pairs(te_src_pairs, te_vals, logit_lens_margins(engine, H_te, src_pid, src_nid))

            # Causal 1D steering reversal on held-out source & unseen cross-lexicons!
            mf_src = flip_1d_margins(engine, te_src_pr, l, src_pid, src_nid, q_l, mu_te)
            flip_src_auc = auc_pairs(te_src_pairs, te_vals, mf_src)

            cross_flips, cross_ll, cross_cleans = [], [], []
            for c_pr, c_pairs, c_pid, c_nid, c_clean in cross_prep:
                H_c = engine.extract_early_exit(c_pr, l, spans=None, pooling="last")
                mu_c = H_c.mean(axis=0)
                mf_c = flip_1d_margins(engine, c_pr, l, c_pid, c_nid, q_l, mu_c)
                cross_flips.append(auc_pairs(c_pairs, te_vals, mf_c))
                cross_ll.append(auc_pairs(c_pairs, te_vals, logit_lens_margins(engine, H_c, c_pid, c_nid)))
                cross_cleans.append(c_clean)

            mean_cross_flip = float(np.mean(cross_flips))
            mean_cross_clean = float(np.mean(cross_cleans))
            mean_cross_ll = float(np.mean(cross_ll))
            steer_effect = mean_cross_clean - mean_cross_flip  # positive => strong causal reversal across unseen lexicons
            wu_cos = float(abs(np.dot(q_l, wu_diff)))

            row = {
                "layer": l,
                "rel_depth": round(l / L, 3),
                "ace_readout_auc": round(ace_readout_auc, 4),
                "logit_lens_src_auc": round(ll_src_auc, 4),
                "logit_lens_cross_auc": round(mean_cross_ll, 4),
                "flip_src_auc": round(flip_src_auc, 4),
                "flip_cross_auc": round(mean_cross_flip, 4),
                "cross_steer_drop": round(steer_effect, 4),
                "wu_cosine": round(wu_cos, 4),
                "spectral_gap": round(float(gap), 1),
            }
            layer_rows.append(row)
            print(
                f"  [{dom_name:15s} L{l:2d}/{L}] ACE_AUC={ace_readout_auc:.3f} | LL_AUC={mean_cross_ll:.3f} | "
                f"CrossFlip={mean_cross_flip:.3f} (drop={steer_effect:+.3f}) | |cos(q,W_U)|={wu_cos:.4f}",
                flush=True,
            )

        # Find peak "Steerable-Not-Decodable" gap layer (where Cross-Steering reversal is strong while Logit Lens is weak)
        best_gap_row = max(layer_rows, key=lambda r: r["cross_steer_drop"] - max(0.0, r["logit_lens_cross_auc"] - 0.5))
        peak_steer_row = max(layer_rows, key=lambda r: r["cross_steer_drop"])
        # First layer where ACE readout >= 0.85 vs first layer where Logit Lens >= 0.85
        first_ace_85 = next((r["layer"] for r in layer_rows if r["ace_readout_auc"] >= 0.85), None)
        first_ll_85 = next((r["layer"] for r in layer_rows if r["logit_lens_cross_auc"] >= 0.85), None)

        domain_results[dom_name] = {
            "clean_src_auc": round(clean_src_auc, 4),
            "clean_cross_auc": round(float(np.mean([x[4] for x in cross_prep])), 4),
            "peak_steer_layer": peak_steer_row["layer"],
            "peak_cross_steer_drop": peak_steer_row["cross_steer_drop"],
            "peak_steer_logit_lens_auc": peak_steer_row["logit_lens_cross_auc"],
            "peak_steer_wu_cosine": peak_steer_row["wu_cosine"],
            "best_steerable_not_decodable_layer": best_gap_row["layer"],
            "first_ace_readout_ge85_layer": first_ace_85,
            "first_logit_lens_ge85_layer": first_ll_85,
            "layers": layer_rows,
        }

    payload = {
        "experiment_id": "steering_lens",
        "slug": "steerable-not-decodable-ace-cores",
        "title": "Steerable Before Decodable: Testing CoreRank 1D Causal Steering vs. Logit Lens Readability Across Depth",
        "short_title": "Steerable Before Decodable: ACE Cores vs Logit Lens",
        "category_label": "AUTONOMOUS GPU LAB · MECHANISTIC INTERPRETABILITY",
        "short_cat": "Auto GPU · Steering",
        "track": "interpretability",
        "cluster": 1,
        "dot": "violet",
        "source_discovery_type": "Method 1 Prior-vs-Evidence Bayesian Surprise (#1 on 4B, +9.30)",
        "source_papers": [
            {"id": "arxiv-2604.02608", "title": "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens", "url": "https://arxiv.org/abs/2604.02608"},
        ],
        "hypothesis": (
            "CoreRank Method 1 surfaced arxiv:2604.02608 (4B surprise +9.30, #1 overall), showing that linear steering "
            "vectors routinely control model outputs where the unembedding Logit Lens fails to decode the answer. We test "
            "whether Paired-Swap ACE 1D cores in Qwen/Qwen3-1.7B exhibit this exact 'steerable-before-decodable' separation "
            "on strictly held-out entities across unseen comparative lexicons, and measure their cosine alignment with W_U."
        ),
        "setup": {
            "model_id": MODEL_ID,
            "n_layers": L,
            "d_model": engine.d_model,
            "domains": list(DOMAINS.keys()),
            "sweep_layers": sweep_layers,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_elapsed_sec": round(time.perf_counter() - t0, 1),
        },
        "results": domain_results,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[exp_steering_lens] Saved results to {out_json}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(out_json=args.output)


if __name__ == "__main__":
    main()
