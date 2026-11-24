#!/usr/bin/env python3
"""
Permanent CoreRank Discovery Refresh Runner (`rescan_discoveries.py`).

Optionally refreshes `data/papers-db.json` with live arXiv releases via
`/home/marcio/Projects/longinus-network/scripts/build_papers_vectordb.py`,
checks GPU VRAM safety (`gpu_guard.py`), runs `core_mine_1_7b.py` + `core_verify_4b.py`
in sequential GPU-clean subprocesses, and exports `data/papers-discoveries.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
SITE_ROOT = ENGINE_DIR.parents[1]
GEN_DIR = SITE_ROOT / "data" / "generated_papers"
PAPERS_JSON = SITE_ROOT / "data" / "papers-db.json"
OUT_DISC = SITE_ROOT / "data" / "papers-discoveries.json"
VDB_BUILDER = Path("/home/marcio/Projects/longinus-network/scripts/build_papers_vectordb.py")

if __package__ in (None, ""):
    sys.path.insert(0, str(ENGINE_DIR.parent))
    from research_engine.gpu_guard import check_vram_headroom, resolve_torch_python
else:
    from .gpu_guard import check_vram_headroom, resolve_torch_python


def export_discoveries_json(d1_path: Path, d4_path: Path, out_path: Path = OUT_DISC) -> dict:
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    d4 = json.loads(d4_path.read_text(encoding="utf-8"))
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in db["papers"]}

    def enrich(pid, extra=None):
        p = by_id.get(pid, {})
        out = {
            "id": pid,
            "idx": p.get("idx"),
            "title": p.get("title", ""),
            "authors": (p.get("authors") or [])[:4],
            "category": p.get("category", ""),
            "year": p.get("year"),
            "published_date": p.get("published_date", ""),
            "venue": p.get("venue", ""),
            "citations": p.get("citations", 0),
            "url": p.get("url", ""),
            "pdf_url": p.get("pdf_url", ""),
            "abstract": p.get("abstract", ""),
        }
        if extra:
            out.update(extra)
        return out

    surprises = [
        enrich(r["id"], {
            "m1_score": r["m1_score"], "z_prior": r["z_prior"], "z_rigor": r["z_rigor"],
            "z_hype": r["z_hype"], "nll_prior": r["nll_prior"], "info_gain_nats": r["info_gain_nats"],
            "rigor_4b": r["rigor_4b"], "surprise_4b": r["surprise_4b"],
            "groundedness_4b": r["groundedness_4b_prob"], "clean_claim": r["clean_claim"],
        })
        for r in d4["method1_verified"]["items"][:16]
    ]
    bridges = [
        {
            "bge_cosine": r["bge_cosine"], "ace_mech_cosine": r["ace_mech_cosine"],
            "bge_orth_resid_cosine": r["bge_orth_resid_cosine"],
            "transfer_1_7b": r["stage2_transfer_margin"], "transfer_4b": r["transfer_4b_sym"],
            "shared_mechanisms": r["shared_mechanisms"],
            "paper_a": enrich(r["paper_a"]["id"], {"z_rigor": r["paper_a"].get("z_rigor")}),
            "paper_b": enrich(r["paper_b"]["id"], {"z_rigor": r["paper_b"].get("z_rigor")}),
        }
        for r in d4["method2_verified"]["items"][:14]
    ]
    contradictions = [
        {
            "bge_cosine": r["bge_cosine"], "keep1d_sym": r["stage2_keep1d_sym"],
            "z_sym": r["stage2_z_sym"], "contradiction_4b": r["contradiction_4b_sym"],
            "order_diff_4b": r["contradiction_4b_order_diff"],
            "paper_a": enrich(r["paper_a"]["id"]), "paper_b": enrich(r["paper_b"]["id"]),
        }
        for r in d4["method3_verified"]["items"][:14]
    ]
    sleeping = [
        enrich(r["id"], {
            "lens": "sleeping_beauty", "expected_citations": r["expected_citations"],
            "z_rigor": r["z_rigor"], "z_hype": r["z_hype"],
            "rigor_4b": r["rigor_4b"], "unconventional_4b": r["unconventional_4b"],
        })
        for r in d4["sleeping_beauties_4b"]["items"]
        if int(r.get("citations", 0)) < 50
    ]
    quiet_today = [
        enrich(r["id"], {
            "lens": "quiet_today", "z_rigor": r["z_rigor"], "z_hype": r["z_hype"],
            "z_prior": r["z_prior"], "rigor_4b": r["rigor_4b"], "unconventional_4b": r["unconventional_4b"],
        })
        for r in d4["quiet_today_4b"]["items"][:8]
    ]
    off_core = [
        enrich(r["id"], {
            "lens": "off_core_outlier", "off_core_norm_z": r["off_core_norm_z"],
            "off_core_ratio": r["off_core_ratio"], "z_rigor": r["z_rigor"],
            "rigor_4b": r["rigor_4b"], "unconventional_4b": r["unconventional_4b"],
        })
        for r in d4["off_core_outliers_4b"]["items"][:8]
    ]

    payload = {
        "schema_version": 1,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_count": d1["paper_count"],
        "models": {"stage1_stage2_ace": d1["model_id"], "cross_validator": d4["model_id"]},
        "telemetry": {
            "stage1_scan_sec": d1["method1_bayesian_surprise"]["elapsed_sec"],
            "mech_scan_sec": d1["method2_cross_domain_bridges"]["elapsed_sec"],
            "rigor_train_rho": d1["stage0_audit"]["axes"]["rigor_core"]["train_rho"],
            "rigor_vdip_verdict": d1["stage0_audit"]["vdip_rigor"]["verdict"],
            "rigor_vdip_trough": d1["stage0_audit"]["vdip_rigor"]["trough_cross_flip_mean"],
            "m1_nll_prior_top20": d1["method1_bayesian_surprise"]["top20_vs_cited_controls"]["top20_mean_nll_prior"],
            "m1_nll_prior_ctrl": d1["method1_bayesian_surprise"]["top20_vs_cited_controls"]["controls_mean_nll_prior"],
            "m1_4b_surprise_top20": d4["method1_verified"]["top20_mean_surprise_4b"],
            "m1_4b_surprise_rand": d4["method1_verified"]["random40_baseline"]["mean_surprise_4b"],
            "m1_4b_rigor_top20": d4["method1_verified"]["top20_mean_rigor_4b"],
            "m1_4b_rigor_rand": d4["method1_verified"]["random40_baseline"]["mean_rigor_4b"],
            "m2_4b_transfer_top20": d4["method2_verified"]["top20_mean_transfer_4b"],
            "m2_4b_transfer_rand": d4["method2_verified"]["random20_mean_transfer_4b"],
            "m3_contra_auc": d1["method3_dense_contradictions"]["cal_auc"],
            "m3_order_bias_reduction_pct": d1["method3_dense_contradictions"]["order_bias_reduction_pct"],
            "m4_cycle_raw_pct": round(d1["method4_residuals_and_outliers"]["stage2_tournament"]["raw_metrics"]["cycle_rate_raw"] * 100, 1),
            "m4_cycle_keep1d_pct": round(d1["method4_residuals_and_outliers"]["stage2_tournament"]["debiased_keep1d_metrics"]["cycle_rate_raw"] * 100, 1),
        },
        "surprising": surprises,
        "bridges": bridges,
        "contradictions": contradictions,
        "under_the_radar": {
            "sleeping_beauties": sleeping,
            "quiet_today": quiet_today,
            "off_core_outliers": off_core,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Refresh arXiv papers DB & CoreRank discoveries index")
    parser.add_argument("--refresh-arxiv", action="store_true", help="Run build_papers_vectordb.py before CoreRank scan")
    parser.add_argument("--min-free-vram-mb", type=int, default=5200)
    args = parser.parse_args()

    py_exe, env = resolve_torch_python()

    if args.refresh_arxiv and VDB_BUILDER.exists():
        print(f"[rescan] Refreshing papers DB via {VDB_BUILDER}...", flush=True)
        env_online = dict(env)
        env_online.pop("HF_HUB_OFFLINE", None)
        proc = subprocess.run([py_exe, str(VDB_BUILDER)], env=env_online, timeout=300, check=False)
        if proc.returncode != 0:
            print(f"[rescan] Warning: build_papers_vectordb.py exited {proc.returncode}; continuing with existing papers-db.json", flush=True)

    ok, snap, msg = check_vram_headroom(min_free_mb=args.min_free_vram_mb)
    print(f"[rescan] GPU Guard: {msg}", flush=True)
    if not ok:
        print(f"[rescan] Insufficient VRAM ({snap.free_mb} MiB free); aborting safely.", file=sys.stderr)
        sys.exit(3)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    print("[rescan] Stage 1/2: Running CoreRank 1.7B 4-method discovery...", flush=True)
    subprocess.run([py_exe, str(ENGINE_DIR / "core_mine_1_7b.py")], env=env, timeout=420, check=True)
    print("[rescan] Stage 2/2: Cross-validating with pruned CoreRank Ultra 4B (L22)...", flush=True)
    subprocess.run([py_exe, str(ENGINE_DIR / "core_verify_4b.py")], env=env, timeout=180, check=True)

    payload = export_discoveries_json(GEN_DIR / "discovery_results_1.7b.json", GEN_DIR / "discovery_verified_4b.json")
    print(
        f"[rescan] Updated {OUT_DISC}: {payload['corpus_count']} papers -> "
        f"{len(payload['surprising'])} surprises, {len(payload['bridges'])} bridges, {len(payload['contradictions'])} contradictions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
