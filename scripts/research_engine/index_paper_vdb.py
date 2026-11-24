#!/usr/bin/env python3
"""
Automatic Vector DB Paper Publisher (`index_paper_vdb.py`).

Embeds newly generated Space AI technical papers using the exact same offline
`BAAI/bge-small-en-v1.5` encoder (384-d L2-normalized) used by `papers-db.json`,
computes exact top-6 cosine neighbors and 2D Embedding Atlas `(x, y)` coordinates,
and atomically upserts the paper into:
  - `/home/marcio/Projects/spaceresearch.ai/data/papers-db.json`
  - `/home/marcio/Projects/spaceresearch.ai/data/papers-vectors.f16.bin`
keeping `len(papers) * 384 == len(vectors)` strictly invariant.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ENGINE_DIR = Path(__file__).resolve().parent
SITE_ROOT = ENGINE_DIR.parents[1]
PAPERS_JSON = SITE_ROOT / "data" / "papers-db.json"
VECTORS_BIN = SITE_ROOT / "data" / "papers-vectors.f16.bin"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


def embed_text_bge(text: str):
    """Compute L2-normalized 384-d CLS embedding with local cached BAAI/bge-small-en-v1.5."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(EMBED_MODEL, local_files_only=True)
    model.eval()
    with torch.inference_mode():
        enc = tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()[0]
    norm = float(np.linalg.norm(cls))
    return (cls / (norm if norm > 1e-12 else 1.0)).astype(np.float32)


def upsert_paper_into_vdb(paper_record: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    db = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers: List[Dict[str, Any]] = db["papers"]
    n_orig = len(papers)
    vecs = np.fromfile(VECTORS_BIN, dtype=np.float16).astype(np.float32).reshape(n_orig, DIM)

    embed_input = f"{paper_record['title']}. {paper_record.get('abstract', '')}"
    new_vec = embed_text_bge(embed_input)

    pid = paper_record["id"]
    existing_idx = next((i for i, p in enumerate(papers) if p.get("id") == pid), None)

    if existing_idx is not None:
        vecs[existing_idx] = new_vec
        target_idx = existing_idx
    else:
        vecs = np.vstack([vecs, new_vec[None, :]])
        target_idx = n_orig
        papers.append({})

    # Compute exact 384-d cosine similarities against all other papers
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs_u = vecs / norms
    sims = vecs_u @ vecs_u[target_idx]
    sims[target_idx] = -2.0
    top_k = np.argsort(-sims)[:6]

    neighbors = [
        {"idx": int(j), "id": papers[int(j)]["id"], "sim": round(float(sims[int(j)]), 4)}
        for j in top_k
        if papers[int(j)].get("id")
    ]

    # Place smoothly on the 2D Embedding Atlas via similarity-weighted centroid of top-6 neighbors
    w_sum, wx, wy = 0.0, 0.0, 0.0
    for nb in neighbors:
        p_nb = papers[nb["idx"]]
        if "x" in p_nb and "y" in p_nb:
            w = max(0.05, float(nb["sim"]))
            wx += w * float(p_nb["x"])
            wy += w * float(p_nb["y"])
            w_sum += w
    x_pos = round(wx / w_sum, 4) if w_sum > 0 else 0.0
    y_pos = round(wy / w_sum, 4) if w_sum > 0 else 0.0

    full_entry = {
        "idx": target_idx,
        "id": pid,
        "title": paper_record["title"],
        "authors": paper_record.get("authors", ["Marcio Diaz", "Space AI"]),
        "year": int(paper_record.get("year", 2026)),
        "published_date": paper_record.get("published_date", "2026-11-24"),
        "venue": paper_record.get("venue", "Space AI Technical Paper"),
        "venue_group": "space-ai",
        "category": "space-ai-lab",
        "citations": int(paper_record.get("citations", 0)),
        "arxiv_id": None,
        "doi": None,
        "url": paper_record["url"],
        "pdf_url": paper_record.get("pdf_url"),
        "tags": paper_record.get("tags", ["space-ai", "autonomous-gpu", "corerank"]),
        "abstract": paper_record.get("abstract", "")[:880],
        "x": x_pos,
        "y": y_pos,
        "neighbors": neighbors,
    }
    papers[target_idx] = full_entry

    # Recompute exact metadata counts
    db["paper_count"] = len(papers)
    cat_counts = Counter(p.get("category") for p in papers)
    for c in db.get("categories", []):
        c["count"] = int(cat_counts.get(c["id"], 0))
    venue_counts = Counter(p.get("venue_group") for p in papers)
    for v in db.get("venues", []):
        if v["id"] == "all":
            v["count"] = len(papers)
        elif v["id"] == "hf-daily":
            v["count"] = int(venue_counts.get("hf-daily", 0) + venue_counts.get("arxiv-today", 0))
        else:
            v["count"] = int(venue_counts.get(v["id"], 0))
    yr_counts = Counter(int(p.get("year") or 2026) for p in papers)
    for y in db.get("years", []):
        y["count"] = int(yr_counts.get(int(y["year"]), 0))

    assert vecs.shape == (len(papers), DIM), f"Shape mismatch: {vecs.shape} vs {len(papers)}"

    # Write float16 matrix + JSON atomically
    tmp_bin = VECTORS_BIN.with_suffix(".bin.tmp")
    vecs.astype(np.float16).tofile(tmp_bin)
    tmp_bin.replace(VECTORS_BIN)

    tmp_json = PAPERS_JSON.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(db, separators=(",", ":")), encoding="utf-8")
    tmp_json.replace(PAPERS_JSON)

    return {
        "id": pid,
        "idx": target_idx,
        "paper_count": len(papers),
        "space_ai_count": int(cat_counts.get("space-ai-lab", 0)),
        "atlas_xy": [x_pos, y_pos],
        "top_neighbor": neighbors[0] if neighbors else None,
    }


def upsert_via_torch_subprocess(paper_record: Dict[str, Any]) -> Dict[str, Any]:
    """Helper callable from system Python (without torch/numpy) that delegates to the PyTorch env."""
    if __package__ in (None, ""):
        sys.path.insert(0, str(ENGINE_DIR.parent))
        from research_engine.gpu_guard import resolve_torch_python
    else:
        from .gpu_guard import resolve_torch_python

    py_exe, env = resolve_torch_python()
    proc = subprocess.run(
        [py_exe, str(Path(__file__).resolve()), "--record-json", json.dumps(paper_record)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("VDB_UPSERT_RESULT:"):
            return json.loads(line[len("VDB_UPSERT_RESULT:"):])
    return {"raw_stdout": proc.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description="Embed and upsert a generated Space AI paper into papers-db.json + f16 vector matrix")
    parser.add_argument("--record-json", required=True, help="JSON dict with id, title, authors, abstract, url, published_date, tags")
    args = parser.parse_args()
    rec = json.loads(args.record_json)
    res = upsert_paper_into_vdb(rec)
    print("VDB_UPSERT_RESULT:" + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
