#!/usr/bin/env python3
"""
Phase 2 Cross-Validation using Pruned CoreRank Ultra (Qwen/Qwen3-Reranker-4B, stop_layer=22, ~4.9 GB VRAM)
+ Production v3 Profiles (`corerank_ultra_profiles_v3.npz`).

Cross-checks the top candidates surfaced by all 4 methods in `discovery_results_1.7b.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from corerank.fast_runtime import FastCoreRankRuntime

OUT_DIR = Path("/home/marcio/Projects/spaceresearch.ai/data/generated_papers")
IN_JSON = OUT_DIR / "discovery_results_1.7b.json"
OUT_JSON = OUT_DIR / "discovery_verified_4b.json"
PROFILES_NPZ = Path("/home/marcio/Projects/corerank-5090/axes/corerank_ultra_profiles_v3.npz")
PAPERS_JSON = Path("/home/marcio/Projects/spaceresearch.ai/data/papers-db.json")


def main():
    t0 = time.time()
    d1 = json.loads(IN_JSON.read_text(encoding="utf-8"))
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers = db["papers"]

    print("[4b] Loading pruned Qwen/Qwen3-Reranker-4B (stop_layer=22) + v3 profiles...", flush=True)
    rt = FastCoreRankRuntime(
        model_id="Qwen/Qwen3-Reranker-4B",
        stop_layer=22,
        dtype="bf16",
        max_doc_chars=1800,
        batch_size=8,
        device="cuda",
    )
    rt.load_profiles_npz(PROFILES_NPZ)
    print(f"[4b] Loaded in {time.time()-t0:.1f}s | VRAM={rt.vram_gb():.2f} GB", flush=True)

    verified: Dict[str, Any] = {
        "model_id": "Qwen/Qwen3-Reranker-4B (pruned L22/36)",
        "vram_gb": rt.vram_gb(),
    }

    # --- 1. Method 1: Cross-check Top 20 Bayesian Surprises (+ 10 Cited Controls) ---
    m1_list = d1["method1_bayesian_surprise"]["top_discoveries"]
    m1_docs = [f"{r['title']}. {r['abstract']}" for r in m1_list]
    s_rigor_4b, _, _ = rt.rerank(
        query="Is this research abstract technically rigorous, concrete, and backed by controlled empirical or mathematical evidence rather than vague hype?",
        documents=m1_docs,
        category="judge",
        use_prefix_kv=True,
    )
    s_surp_4b, _, _ = rt.rerank(
        query="Does this paper report a genuinely counterintuitive or surprising empirical finding that contradicts standard assumptions in machine learning or systems?",
        documents=m1_docs,
        category="science",
        instruction="Judge whether the paper reports a genuinely counterintuitive empirical discovery or unexpected failure mode rather than incremental engineering.",
        use_prefix_kv=True,
    )
    # Groundedness check: is the clean_claim grounded in the abstract?
    grnd_probs = []
    for r in m1_list:
        decs, _, _ = rt.decide(
            r["abstract"],
            [{
                "id": "g",
                "type": "noul",
                "preset": "groundedness",
                "question": "Is the claim fully supported by the abstract?",
                "context": r["abstract"],
                "claim": r["clean_claim"],
            }],
        )
        grnd_probs.append(float(decs[0]["probabilities"]["yes"]))

    # Also score 10 random baseline papers on 4B rigor + 4B surprise for reference distribution!
    rng = np.random.default_rng(7)
    base_idxs = rng.choice(len(papers), size=40, replace=False)
    base_docs = [f"{papers[i]['title']}. {papers[i]['abstract']}" for i in base_idxs]
    base_rigor_4b, _, _ = rt.rerank(
        query="Is this research abstract technically rigorous, concrete, and backed by controlled empirical or mathematical evidence rather than vague hype?",
        documents=base_docs,
        category="judge",
        use_prefix_kv=True,
    )
    base_surp_4b, _, _ = rt.rerank(
        query="Does this paper report a genuinely counterintuitive or surprising empirical finding that contradicts standard assumptions in machine learning or systems?",
        documents=base_docs,
        category="science",
        instruction="Judge whether the paper reports a genuinely counterintuitive empirical discovery or unexpected failure mode rather than incremental engineering.",
        use_prefix_kv=True,
    )

    m1_verified = []
    for k, r in enumerate(m1_list):
        item = dict(r)
        item["rigor_4b"] = round(float(s_rigor_4b[k]), 3)
        item["surprise_4b"] = round(float(s_surp_4b[k]), 3)
        item["groundedness_4b_prob"] = round(grnd_probs[k], 4)
        item["combined_4b_rank_score"] = round(
            float(
                (s_rigor_4b[k] - np.mean(base_rigor_4b)) / (np.std(base_rigor_4b) + 1e-6)
                + (s_surp_4b[k] - np.mean(base_surp_4b)) / (np.std(base_surp_4b) + 1e-6)
            ),
            3,
        )
        m1_verified.append(item)
    m1_verified.sort(key=lambda x: -x["combined_4b_rank_score"])

    verified["method1_verified"] = {
        "random40_baseline": {
            "mean_rigor_4b": round(float(np.mean(base_rigor_4b)), 3),
            "mean_surprise_4b": round(float(np.mean(base_surp_4b)), 3),
        },
        "top20_mean_rigor_4b": round(float(np.mean(s_rigor_4b)), 3),
        "top20_mean_surprise_4b": round(float(np.mean(s_surp_4b)), 3),
        "top20_mean_groundedness": round(float(np.mean(grnd_probs)), 4),
        "items": m1_verified,
    }
    print(
        f"[4b-M1] Rigor: top20={np.mean(s_rigor_4b):.2f} vs rand40={np.mean(base_rigor_4b):.2f} | "
        f"Surprise: top20={np.mean(s_surp_4b):.2f} vs rand40={np.mean(base_surp_4b):.2f}",
        flush=True,
    )

    # --- 2. Method 2: Cross-check Top 20 Cross-Domain Mechanism Bridges ---
    m2_list = d1["method2_cross_domain_bridges"]["top_bridges"]
    m2_pairs_ab = [
        f"Paper A [{r['paper_a']['category']}]: {r['paper_a']['title']}. {r['paper_a']['abstract'][:500]}\n\n"
        f"Paper B [{r['paper_b']['category']}]: {r['paper_b']['title']}. {r['paper_b']['abstract'][:500]}"
        for r in m2_list
    ]
    m2_pairs_ba = [
        f"Paper A [{r['paper_b']['category']}]: {r['paper_b']['title']}. {r['paper_b']['abstract'][:500]}\n\n"
        f"Paper B [{r['paper_a']['category']}]: {r['paper_a']['title']}. {r['paper_a']['abstract'][:500]}"
        for r in m2_list
    ]
    m2_q = "Do Paper A and Paper B share a concrete, non-obvious underlying mathematical or algorithmic mechanism across their two different domains that could transfer productively?"
    m2_inst = "Judge whether Paper A and Paper B share a deep transferable mathematical or systems mechanism despite coming from different domains."
    s_m2_ab, _, _ = rt.rerank(m2_q, m2_pairs_ab, category="science", instruction=m2_inst, use_prefix_kv=True)
    s_m2_ba, _, _ = rt.rerank(m2_q, m2_pairs_ba, category="science", instruction=m2_inst, use_prefix_kv=True)
    s_m2_sym = 0.5 * (s_m2_ab + s_m2_ba)

    # Random cross-cat baseline for 4B
    rand_pairs = []
    for k in range(20):
        ia, ib = int(base_idxs[k]), int(base_idxs[k + 20])
        rand_pairs.append(
            f"Paper A [{papers[ia]['category']}]: {papers[ia]['title']}. {papers[ia]['abstract'][:500]}\n\n"
            f"Paper B [{papers[ib]['category']}]: {papers[ib]['title']}. {papers[ib]['abstract'][:500]}"
        )
    s_m2_rand, _, _ = rt.rerank(m2_q, rand_pairs, category="science", instruction=m2_inst, use_prefix_kv=True)

    m2_verified = []
    for k, r in enumerate(m2_list):
        item = dict(r)
        item["transfer_4b_sym"] = round(float(s_m2_sym[k]), 3)
        item["transfer_4b_order_diff"] = round(float(abs(s_m2_ab[k] - s_m2_ba[k])), 3)
        m2_verified.append(item)
    m2_verified.sort(key=lambda x: -x["transfer_4b_sym"])

    verified["method2_verified"] = {
        "top20_mean_transfer_4b": round(float(np.mean(s_m2_sym)), 3),
        "random20_mean_transfer_4b": round(float(np.mean(s_m2_rand)), 3),
        "items": m2_verified,
    }
    print(f"[4b-M2] Transfer score: top20={np.mean(s_m2_sym):.2f} vs rand20={np.mean(s_m2_rand):.2f}", flush=True)

    # --- 3. Method 3: Cross-check Top 20 Dense Contradictions ---
    m3_list = d1["method3_dense_contradictions"]["top_contradictions"]
    m3_ab = [
        f"Paper A: {r['paper_a']['title']}. {r['paper_a']['abstract'][:500]}\n\n"
        f"Paper B: {r['paper_b']['title']}. {r['paper_b']['abstract'][:500]}"
        for r in m3_list
    ]
    m3_ba = [
        f"Paper A: {r['paper_b']['title']}. {r['paper_b']['abstract'][:500]}\n\n"
        f"Paper B: {r['paper_a']['title']}. {r['paper_a']['abstract'][:500]}"
        for r in m3_list
    ]
    m3_q = "Do Paper A and Paper B directly contradict each other, challenge each other's core assumption, or report empirically conflicting results on the same technique?"
    m3_inst = "Judge whether Paper A and Paper B contain a genuine empirical contradiction or opposing scientific claims rather than merely being complementary."
    s_m3_ab, _, _ = rt.rerank(m3_q, m3_ab, category="judge", instruction=m3_inst, use_prefix_kv=True)
    s_m3_ba, _, _ = rt.rerank(m3_q, m3_ba, category="judge", instruction=m3_inst, use_prefix_kv=True)
    s_m3_sym = 0.5 * (s_m3_ab + s_m3_ba)

    m3_verified = []
    for k, r in enumerate(m3_list):
        item = dict(r)
        item["contradiction_4b_sym"] = round(float(s_m3_sym[k]), 3)
        item["contradiction_4b_order_diff"] = round(float(abs(s_m3_ab[k] - s_m3_ba[k])), 3)
        m3_verified.append(item)
    m3_verified.sort(key=lambda x: -x["contradiction_4b_sym"])

    verified["method3_verified"] = {
        "top20_mean_contradiction_4b": round(float(np.mean(s_m3_sym)), 3),
        "items": m3_verified,
    }
    print(f"[4b-M3] Contradiction 4B top20 mean={np.mean(s_m3_sym):.2f} (max={np.max(s_m3_sym):.2f})", flush=True)

    # --- 4. Method 4: Cross-check Sleeping Beauties + Quiet Today Gems + Off-Core Outliers ---
    sb_list = d1["method4_residuals_and_outliers"]["sleeping_beauties_pre2026"]
    qt_list = d1["method4_residuals_and_outliers"]["quiet_gems_today"]
    oc_list = d1["method4_residuals_and_outliers"]["off_core_orthogonal_outliers"]

    for key, lst in [("sleeping_beauties_4b", sb_list), ("quiet_today_4b", qt_list), ("off_core_outliers_4b", oc_list)]:
        docs = [f"{r['title']}. {r['abstract']}" for r in lst]
        sr, _, _ = rt.rerank(
            query="Is this research abstract technically rigorous, concrete, and backed by controlled empirical or mathematical evidence rather than vague hype?",
            documents=docs,
            category="judge",
            use_prefix_kv=True,
        )
        ss, _, _ = rt.rerank(
            query="Does this paper introduce an unconventional mechanism or surprising finding that stands apart from standard incremental work?",
            documents=docs,
            category="science",
            instruction="Judge whether this paper introduces a genuinely unconventional mechanism or surprising empirical result.",
            use_prefix_kv=True,
        )
        out_items = []
        for k, r in enumerate(lst):
            it = dict(r)
            it["rigor_4b"] = round(float(sr[k]), 3)
            it["unconventional_4b"] = round(float(ss[k]), 3)
            out_items.append(it)
        out_items.sort(key=lambda x: -(x["rigor_4b"] + 0.7 * x["unconventional_4b"]))
        verified[key] = {
            "mean_rigor_4b": round(float(np.mean(sr)), 3),
            "mean_unconventional_4b": round(float(np.mean(ss)), 3),
            "items": out_items,
        }
        print(f"[4b-M4] {key:22s}: mean rigor_4b={np.mean(sr):.2f} (rand40={np.mean(base_rigor_4b):.2f}), unconv_4b={np.mean(ss):.2f}", flush=True)

    verified["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(f"\n[DONE 4B] Saved to {OUT_JSON} in {verified['elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
