#!/usr/bin/env python3
"""Unit & site-integrity tests for the Space AI Autonomous Research Engine and Daily-Ordered Blog."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT / "scripts"))

from research_engine.discover import build_hypothesis_queue, load_discoveries
from research_engine.experiments.registry import audit_generated_python_ast
from research_engine.gpu_guard import query_gpu
from research_engine.rebuild_blog_index import (
    group_posts_by_day,
    rebuild_all,
    sort_posts_newest,
)


class TestHighVolumeDailyBlogOrdering(unittest.TestCase):
    def test_25_same_day_posts_strict_ordering_and_grouping(self):
        """Even 25 posts published on the exact same date must sort deterministically without collisions."""
        posts = []
        # Create 25 posts on 2026-11-24 including identical timestamps differentiated by seq
        for i in range(1, 26):
            minute = (i // 2) % 60
            posts.append({
                "seq": i,
                "note_id": f"NOTE {i:02d}",
                "slug": f"p-{i}",
                "href": "./blog-core-discovery.html",
                "title": f"Same-day post #{i}",
                "description": "Synthetic ordering test.",
                "category_label": "AUTONOMOUS GPU LAB",
                "track": "discovery",
                "origin": "autonomous-gpu",
                "cluster": 0,
                "dot": "cyan",
                "published_at": f"2026-11-24T12:{minute:02d}:00+01:00",
                "published_date": "2026-11-24",
                "read_minutes": 4,
            })
        # Add 3 older-day posts interleaved in arbitrary order
        posts.insert(5, {
            "seq": 100,
            "note_id": "NOTE 100",
            "slug": "old-1",
            "href": "./blog-open-intelligence.html",
            "title": "Older day post",
            "description": "Earlier day.",
            "category_label": "LAB THESIS",
            "track": "collective",
            "origin": "lab",
            "cluster": 3,
            "dot": "amber",
            "published_at": "2026-11-22T09:00:00+01:00",
            "published_date": "2026-11-22",
            "read_minutes": 5,
        })

        ordered = sort_posts_newest(posts)
        self.assertEqual(len(ordered), 26)
        # First 25 must all be from 2026-11-24, strictly non-increasing (ts, seq)
        self.assertEqual(ordered[0]["published_date"], "2026-11-24")
        self.assertEqual(ordered[0]["seq"], 25)
        self.assertEqual(ordered[-1]["published_date"], "2026-11-22")

        # Tie-break test: seq=25 vs seq=24 share minute 12:12 -> seq=25 MUST precede seq=24
        seq_order_today = [p["seq"] for p in ordered[:25]]
        self.assertEqual(seq_order_today, list(range(25, 0, -1)))

        groups = group_posts_by_day(ordered)
        self.assertEqual(list(groups.keys()), ["2026-11-24", "2026-11-22"])
        self.assertEqual(len(groups["2026-11-24"]), 25)

    def test_rebuild_blog_index_temp_roundtrip(self):
        manifest = json.loads((SITE_ROOT / "data" / "blog-posts.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            m_path = td_path / "blog-posts.json"
            m_path.write_text(json.dumps(manifest), encoding="utf-8")
            b_path = td_path / "blog.html"
            b_path.write_text("<div><!-- BLOG_FEED_START --><!-- BLOG_FEED_END --></div>", encoding="utf-8")
            i_path = td_path / "index.html"
            i_path.write_text("<div><!-- LANDING_BLOG_START --><!-- LANDING_BLOG_END --></div>", encoding="utf-8")

            res = rebuild_all(m_path, b_path, i_path, strict_files=False)
            self.assertEqual(res["post_count"], len(manifest["posts"]))
            self.assertIn("blog-day-group", b_path.read_text(encoding="utf-8"))
            self.assertIn("landing-blog-item", i_path.read_text(encoding="utf-8"))


class TestAstSafetyAndDiscovery(unittest.TestCase):
    def test_ast_safety_blocks_dangerous_code(self):
        bad_code = "import subprocess\nsubprocess.run(['rm', '-rf', '/'])\n"
        ok, issues = audit_generated_python_ast(bad_code)
        self.assertFalse(ok)
        self.assertTrue(any("subprocess" in s for s in issues))

        bad_eval = "import torch\neval('1+1')\n"
        ok2, issues2 = audit_generated_python_ast(bad_eval)
        self.assertFalse(ok2)
        self.assertTrue(any("eval" in s for s in issues2))

    def test_ast_safety_allows_clean_pytorch(self):
        good_code = (
            "import json\nimport torch\n"
            "x = torch.randn(4, 4)\n"
            "print(float(x.mean().item()))\n"
        )
        ok, issues = audit_generated_python_ast(good_code)
        self.assertTrue(ok, f"Unexpected issues: {issues}")

    def test_live_site_integrity(self):
        # 1. Discoveries JSON schema & hypotheses
        disc = load_discoveries()
        self.assertGreaterEqual(len(disc.get("surprising", [])), 10)
        self.assertGreaterEqual(len(disc.get("bridges", [])), 5)
        self.assertGreaterEqual(len(disc.get("contradictions", [])), 5)
        q = build_hypothesis_queue(disc)
        self.assertGreaterEqual(len(q), 5)

        # 2. All manifest links exist on disk
        manifest = json.loads((SITE_ROOT / "data" / "blog-posts.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest["posts"]), 10)
        for p in manifest["posts"]:
            href = p["href"].lstrip("./")
            self.assertTrue((SITE_ROOT / href).exists(), f"Missing blog HTML: {href}")
            if p.get("paper_href"):
                ph = p["paper_href"].lstrip("./")
                self.assertTrue((SITE_ROOT / ph).exists(), f"Missing paper HTML: {ph}")

        # 3. Papers page exposes CoreRank Discovery tab + vector matrix strictly matches paper_count
        papers_html = (SITE_ROOT / "papers.html").read_text(encoding="utf-8")
        self.assertIn('data-vdb-tab="discover"', papers_html)
        self.assertIn('data-vdb-view="discover"', papers_html)

        pdb = json.loads((SITE_ROOT / "data" / "papers-db.json").read_text(encoding="utf-8"))
        vbin_bytes = (SITE_ROOT / "data" / "papers-vectors.f16.bin").stat().st_size
        self.assertEqual(pdb["paper_count"], len(pdb["papers"]))
        self.assertEqual(vbin_bytes, len(pdb["papers"]) * 384 * 2)

        # Verify generated Space AI papers carry Marcio Diaz & Space AI authorship
        auto_papers = [p for p in pdb["papers"] if str(p.get("id", "")).startswith("spaceai-")]
        self.assertGreaterEqual(len(auto_papers), 3)
        for ap in auto_papers:
            self.assertIn("Marcio Diaz", ap.get("authors", []))
            self.assertIn("Space AI", ap.get("authors", []))

        # 4. GPU snapshot call never crashes
        snap = query_gpu()
        self.assertIsInstance(snap.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
