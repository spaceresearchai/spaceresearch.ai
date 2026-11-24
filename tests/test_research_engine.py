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
from research_engine.quality_gate import evaluate_conference_readiness
from research_engine.rebuild_blog_index import (
    group_posts_by_day,
    rebuild_all,
    sort_posts_newest,
)
from research_engine.zenodo_publisher import (
    ZenodoClient,
    build_zenodo_metadata,
    deposit_paper_to_zenodo,
    format_creator_name,
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

        # Verify generated Space AI papers carry Marcio Diaz & Space AI authorship + real compiled PDFs
        auto_papers = [p for p in pdb["papers"] if str(p.get("id", "")).startswith("spaceai-")]
        self.assertGreaterEqual(len(auto_papers), 3)
        for ap in auto_papers:
            self.assertIn("Marcio Diaz", ap.get("authors", []))
            self.assertIn("Space AI", ap.get("authors", []))
            pdf_rel = str(ap.get("pdf_url") or "").lstrip("./")
            self.assertTrue(pdf_rel.endswith(".pdf"), f"Missing pdf_url on {ap['id']}")
            pdf_file = SITE_ROOT / pdf_rel
            self.assertTrue(pdf_file.exists(), f"Missing PDF file: {pdf_file}")
            self.assertTrue(pdf_file.read_bytes().startswith(b"%PDF-"), f"Invalid PDF header: {pdf_file}")

        # 4. GPU snapshot call never crashes
        snap = query_gpu()
        self.assertIsInstance(snap.to_dict(), dict)


class TestZenodoPublisher(unittest.TestCase):
    def test_creator_format_and_metadata(self):
        self.assertEqual(format_creator_name("Marcio Diaz"), "Diaz, Marcio")
        arch = json.loads((SITE_ROOT / "data" / "generated_papers" / "pytorch-kl-bregman-stability.json").read_text())
        meta = build_zenodo_metadata(
            arch["experiment"], arch["analysis"],
            author="Marcio Diaz", email="marcio.diaz@gmail.com", company="Space AI",
        )["metadata"]
        self.assertEqual(meta["creators"][0]["name"], "Diaz, Marcio")
        self.assertEqual(meta["creators"][0]["affiliation"], "Space AI")
        self.assertIn("marcio.diaz@gmail.com", meta["notes"])
        self.assertEqual(meta["license"], "cc-by-4.0")
        self.assertTrue(any(r["relation"] == "isIdenticalTo" for r in meta["related_identifiers"]))

    def test_zenodo_client_full_deposit_lifecycle_mocked(self):
        calls = []

        class FakeResp:
            def __init__(self, payload):
                self._raw = json.dumps(payload).encode("utf-8")
            def read(self):
                return self._raw
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_opener(req, timeout=60):
            url = req.full_url
            method = req.get_method()
            calls.append((method, url))
            self.assertEqual(req.get_header("Authorization"), "Bearer test_tok_123")
            if method == "POST" and url.endswith("/deposit/depositions"):
                return FakeResp({
                    "id": 9999999,
                    "links": {
                        "bucket": "https://sandbox.zenodo.org/api/files/bucket-abc",
                        "html": "https://sandbox.zenodo.org/deposit/9999999",
                    },
                    "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.9999999"}},
                })
            if method == "PUT" and "/api/files/bucket-abc/" in url:
                self.assertGreater(len(req.data), 100)
                return FakeResp({"key": url.rsplit("/", 1)[-1]})
            if method == "PUT" and url.endswith("/deposit/depositions/9999999"):
                body = json.loads(req.data.decode("utf-8"))
                self.assertEqual(body["metadata"]["creators"][0]["name"], "Diaz, Marcio")
                self.assertIn("marcio.diaz@gmail.com", body["metadata"]["notes"])
                return FakeResp({"id": 9999999, "metadata": body["metadata"]})
            if method == "POST" and url.endswith("/deposit/depositions/9999999/actions/publish"):
                return FakeResp({
                    "id": 9999999,
                    "doi": "10.5072/zenodo.9999999",
                    "links": {"record_html": "https://sandbox.zenodo.org/records/9999999"},
                })
            raise AssertionError(f"Unexpected HTTP call: {method} {url}")

        # Run against sandbox mocked client using a temporary generated-papers.json so repo stays clean
        import research_engine.zenodo_publisher as zp
        orig_gen = zp.GEN_INDEX_JSON
        with tempfile.TemporaryDirectory() as td:
            tmp_gen = Path(td) / "generated-papers.json"
            tmp_gen.write_text(orig_gen.read_text(encoding="utf-8"), encoding="utf-8")
            zp.GEN_INDEX_JSON = tmp_gen
            try:
                client = ZenodoClient(token="test_tok_123", sandbox=True, opener=fake_opener)
                res = deposit_paper_to_zenodo(
                    slug="pytorch-kl-bregman-stability",
                    sandbox=True,
                    publish=True,
                    force=True,
                    bypass_quality_gate=True,
                    client=client,
                )
                self.assertEqual(res["status"], "published")
                self.assertEqual(res["doi"], "10.5072/zenodo.9999999")
                self.assertIn("pytorch-kl-bregman-stability.pdf", res["uploaded_files"])
                self.assertEqual(len(calls), 6)  # create + 3 uploads (.pdf/.md/.json) + metadata + publish
            finally:
                zp.GEN_INDEX_JSON = orig_gen

    def test_conference_quality_gate_blocks_subbar_to_draft_only(self):
        """A sub-bar toy investigation MUST fail quality_gate and be forced to draft-only even if publish=True."""
        toy_exp = {
            "experiment_id": "toy",
            "slug": "toy-exp",
            "title": "Toy Single-Seed Check",
            "hypothesis": "Quick 1-seed check.",
            "setup": {"gpu_name": "RTX 5090"},
            "references": ["Ref 1"],
            "quality_evidence": {"n_models": 1, "n_seeds": 1, "n_baselines": 1},
        }
        rep = evaluate_conference_readiness(toy_exp)
        self.assertFalse(rep.passed)
        self.assertEqual(rep.recommended_action, "draft_only")
        self.assertLess(rep.score_10, 5.0)

        strong_exp = {
            **toy_exp,
            "version": "2.0",
            "references": [f"Author ({2020 + i}). Paper {i}." for i in range(7)],
            "quality_evidence": {
                "n_models": 3,
                "n_seeds": 3,
                "n_regimes": 12,
                "n_baselines": 5,
                "has_confidence_intervals": True,
                "has_p_values": True,
                "has_ablations": True,
                "real_world_scale": True,
                "non_trivial_finding": True,
            },
        }
        rep2 = evaluate_conference_readiness(strong_exp)
        self.assertTrue(rep2.passed)
        self.assertEqual(rep2.score_10, 10.0)
        self.assertEqual(rep2.recommended_action, "eligible_for_zenodo_publish")

    def test_zenodo_new_version_workflow_mocked(self):
        """Verify Zenodo new-version workflow: newversion -> latest_draft -> delete old files -> upload new -> publish."""
        calls = []
        arch_ver = json.loads((SITE_ROOT / "data" / "generated_papers" / "pytorch-kl-bregman-stability.json").read_text())
        expected_ver = str(arch_ver["experiment"].get("version") or "2.0")

        class FakeResp:
            def __init__(self, payload):
                self._raw = json.dumps(payload).encode("utf-8") if payload is not None else b""
            def read(self):
                return self._raw
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_opener(req, timeout=60):
            url = req.full_url
            method = req.get_method()
            calls.append((method, url))
            if method == "POST" and url.endswith("/deposit/depositions/11111/actions/newversion"):
                return FakeResp({"links": {"latest_draft": "https://sandbox.zenodo.org/api/deposit/depositions/22222"}})
            if method == "GET" and url.endswith("/deposit/depositions/22222"):
                return FakeResp({
                    "id": 22222,
                    "conceptdoi": "10.5072/zenodo.11110",
                    "files": [{"id": "old-f1", "filename": "old.pdf"}],
                    "links": {
                        "bucket": "https://sandbox.zenodo.org/api/files/bucket-v2",
                        "html": "https://sandbox.zenodo.org/deposit/22222",
                    },
                    "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.22222"}},
                })
            if method == "DELETE" and url.endswith("/deposit/depositions/22222/files/old-f1"):
                return FakeResp(None)
            if method == "PUT" and "/api/files/bucket-v2/" in url:
                return FakeResp({"key": url.rsplit("/", 1)[-1]})
            if method == "PUT" and url.endswith("/deposit/depositions/22222"):
                body = json.loads(req.data.decode("utf-8"))
                self.assertEqual(body["metadata"]["version"], expected_ver)
                return FakeResp({"id": 22222, "metadata": body["metadata"]})
            if method == "POST" and url.endswith("/deposit/depositions/22222/actions/publish"):
                return FakeResp({
                    "id": 22222,
                    "doi": "10.5072/zenodo.22222",
                    "conceptdoi": "10.5072/zenodo.11110",
                    "links": {"record_html": "https://sandbox.zenodo.org/records/22222"},
                })
            raise AssertionError(f"Unexpected HTTP call: {method} {url}")

        import research_engine.zenodo_publisher as zp
        orig_gen = zp.GEN_INDEX_JSON
        with tempfile.TemporaryDirectory() as td:
            tmp_gen = Path(td) / "generated-papers.json"
            reg = json.loads(orig_gen.read_text(encoding="utf-8"))
            for p in reg["papers"]:
                if p["slug"] == "pytorch-kl-bregman-stability":
                    p["zenodo_sandbox"] = {"deposition_id": 11111, "published": True, "doi": "10.5072/zenodo.11111"}
            tmp_gen.write_text(json.dumps(reg), encoding="utf-8")
            zp.GEN_INDEX_JSON = tmp_gen
            try:
                client = ZenodoClient(token="test_tok_123", sandbox=True, opener=fake_opener)
                res = deposit_paper_to_zenodo(
                    slug="pytorch-kl-bregman-stability",
                    sandbox=True,
                    publish=True,
                    new_version=True,
                    client=client,
                )
                self.assertEqual(res["status"], "published")
                self.assertEqual(res["doi"], "10.5072/zenodo.22222")
                self.assertEqual(res["conceptdoi"], "10.5072/zenodo.11110")
                self.assertTrue(any(m == "DELETE" and "old-f1" in u for m, u in calls))
                self.assertTrue(any(m == "POST" and "actions/newversion" in u for m, u in calls))
            finally:
                zp.GEN_INDEX_JSON = orig_gen


if __name__ == "__main__":
    unittest.main()
