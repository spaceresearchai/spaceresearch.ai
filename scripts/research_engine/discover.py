#!/usr/bin/env python3
"""
CoreRank Research Discovery Stage (`discover.py`).

Loads (or optionally re-mines via CoreRank 1.7B ACE + pruned 4B CoreRank Ultra)
`data/papers-discoveries.json` and emits a prioritized queue of concrete, testable
GPU research hypotheses derived from:
  - Method 1: Prior-vs-Evidence Bayesian Surprises
  - Method 2: Cross-Domain Mechanism Bridges
  - Method 3: Dense-Cluster Empirical Contradictions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SITE_ROOT = Path(__file__).resolve().parents[2]
DISCOVERIES_JSON = SITE_ROOT / "data" / "papers-discoveries.json"


def load_discoveries(path: Path = DISCOVERIES_JSON) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Discoveries index not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_hypothesis_queue(discoveries: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Convert top CoreRank discoveries into structured, ranked testable hypotheses."""
    queue: List[Dict[str, Any]] = []

    # 1. Cross-Domain Bridges (highest creative leverage!)
    for idx, b in enumerate(discoveries.get("bridges", [])[:4], start=1):
        a, p_b = b["paper_a"], b["paper_b"]
        mechs = [m[0] for m in (b.get("shared_mechanisms") or [])[:2]]
        queue.append({
            "rank_key": round(float(b.get("transfer_4b", 0.0)) + 2.0 * float(b.get("ace_mech_cosine", 0.0)), 3),
            "kind": "cross_domain_bridge",
            "title": f"Bridge #{idx}: {a['title'][:58]} <-> {p_b['title'][:58]}",
            "shared_mechanisms": mechs,
            "bge_cosine": b.get("bge_cosine"),
            "ace_mech_cosine": b.get("ace_mech_cosine"),
            "transfer_4b": b.get("transfer_4b"),
            "source_papers": [
                {"id": a["id"], "title": a["title"], "category": a["category"], "url": a.get("url", "")},
                {"id": p_b["id"], "title": p_b["title"], "category": p_b["category"], "url": p_b.get("url", "")},
            ],
        })

    # 2. Contrarian Bayesian Surprises (reproduce & stress-test!)
    for idx, s in enumerate(discoveries.get("surprising", [])[:5], start=1):
        queue.append({
            "rank_key": round(float(s.get("surprise_4b", 0.0)) + 0.8 * float(s.get("rigor_4b", 0.0)), 3),
            "kind": "bayesian_surprise",
            "title": f"Surprise #{idx}: {s['title']}",
            "surprise_4b": s.get("surprise_4b"),
            "rigor_4b": s.get("rigor_4b"),
            "nll_prior": s.get("nll_prior"),
            "source_papers": [
                {"id": s["id"], "title": s["title"], "category": s["category"], "url": s.get("url", "")},
            ],
        })

    # 3. Dense-Cluster Empirical Contradictions (adjudicate experimentally!)
    for idx, c in enumerate(discoveries.get("contradictions", [])[:3], start=1):
        a, p_b = c["paper_a"], c["paper_b"]
        queue.append({
            "rank_key": round(float(c.get("contradiction_4b", 0.0)) * 2.5 + float(c.get("keep1d_sym", 0.0)), 3),
            "kind": "dense_contradiction",
            "title": f"Conflict #{idx}: {a['title'][:58]} != {p_b['title'][:58]}",
            "bge_cosine": c.get("bge_cosine"),
            "contradiction_4b": c.get("contradiction_4b"),
            "source_papers": [
                {"id": a["id"], "title": a["title"], "category": a["category"], "url": a.get("url", "")},
                {"id": p_b["id"], "title": p_b["title"], "category": p_b["category"], "url": p_b.get("url", "")},
            ],
        })

    queue.sort(key=lambda x: -float(x["rank_key"]))
    return queue[:limit]


if __name__ == "__main__":
    disc = load_discoveries()
    q = build_hypothesis_queue(disc)
    print(f"[discover] Loaded {disc['corpus_count']} papers -> {len(q)} ranked hypotheses:")
    for i, h in enumerate(q[:6], 1):
        print(f"  {i}. [{h['kind']}] score={h['rank_key']} | {h['title']}")
