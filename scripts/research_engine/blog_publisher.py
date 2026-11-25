#!/usr/bin/env python3
"""
Autonomous Research Blog Publisher (`blog_publisher.py`).

Generates a full Space AI editorial research note (`blog-auto-<slug>.html`) from a
completed GPU investigation + compiled technical paper, registers it into
`data/blog-posts.json` with exact sub-day ISO timestamp + monotonic sequence ID,
and deterministically rebuilds `blog.html` & `index.html`.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .index_paper_vdb import upsert_via_torch_subprocess
from .paper_writer import DEFAULT_AUTHOR, DEFAULT_COMPANY, DEFAULT_EMAIL, analyze_experiment, compile_paper
from .quality_gate import evaluate_conference_readiness
from .rebuild_blog_index import MANIFEST_PATH, SITE_ROOT, rebuild_all
from .zenodo_publisher import deposit_paper_to_zenodo, load_zenodo_config


def publish_investigation(
    exp_data: Dict[str, Any],
    published_at: str | None = None,
    dry_run: bool = False,
    author: str = DEFAULT_AUTHOR,
    email: str = DEFAULT_EMAIL,
    company: str = DEFAULT_COMPANY,
    index_in_papers_vdb: bool = True,
    zenodo_mode: str = "auto",  # auto | none | sandbox_draft | sandbox_publish | prod_draft | prod_publish
) -> Dict[str, Any]:
    now_iso = published_at or datetime.now().astimezone().isoformat(timespec="seconds")
    pub_date = now_iso[:10]
    slug = exp_data["slug"]
    blog_slug = f"auto-{slug}"
    blog_filename = f"blog-{blog_slug}.html"
    blog_href = f"./{blog_filename}"
    paper_href = f"./papers/{slug}.html"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing = next((p for p in manifest.get("posts", []) if p.get("slug") == blog_slug), None)
    if existing:
        seq = int(existing["seq"])
    else:
        seq = max((int(p.get("seq", 0)) for p in manifest.get("posts", [])), default=0) + 1
    note_id = f"NOTE {seq:02d}"

    analysis = analyze_experiment(exp_data)

    if dry_run:
        return {
            "dry_run": True,
            "note_id": note_id,
            "seq": seq,
            "blog_href": blog_href,
            "paper_href": paper_href,
            "metric_chip": analysis["metric_chip"],
            "findings": analysis["findings"],
        }

    # Look up any already-minted production DOI so the companion blog note includes the live Zenodo badge!
    gen_idx_path = SITE_ROOT / "data" / "generated-papers.json"
    existing_doi = None
    existing_zurl = None
    if gen_idx_path.exists():
        g_reg = json.loads(gen_idx_path.read_text(encoding="utf-8"))
        prev_p = next((x for x in g_reg.get("papers", []) if x.get("slug") == slug), None)
        if prev_p:
            existing_doi = prev_p.get("doi")
            existing_zurl = prev_p.get("zenodo_url") or (f"https://doi.org/{existing_doi}" if existing_doi else None)

    # 1. Compile full technical paper (Markdown + HTML + academic PDF + JSON archive)
    paper_res = compile_paper(
        exp_data, published_at=now_iso, blog_href=blog_href,
        author=author, email=email, company=company,
        doi=existing_doi, zenodo_url=existing_zurl,
    )

    # 2. Render Space AI Research Note HTML
    stats_html = "".join(
        f'<div class="blog-stat"><div class="blog-stat-v">{html.escape(v)}</div><div class="blog-stat-l">{html.escape(l)}</div></div>'
        for v, l in analysis["stats"]
    )
    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in analysis["findings"])
    limits_html = "".join(f"<li>{html.escape(l)}</li>" for l in analysis["limitations"])
    sources_html = "".join(
        f'<li><a href="{html.escape(s.get("url", "#"))}" target="_blank" rel="noopener">{html.escape(s["title"])}</a> (<code>{html.escape(s["id"])}</code>)</li>'
        for s in exp_data.get("source_papers", [])
    )
    fig_png = analysis.get("figure_png_rel")
    if fig_png:
        fig_block = (
            f'<div class="blog-figure" style="background:#ffffff;padding:12px;border-radius:10px;">'
            f'<img src="./{html.escape(fig_png.lstrip("./"))}" alt="{html.escape(analysis.get("fig_caption", ""))}" '
            f'style="width:100%;height:auto;display:block;border-radius:6px;" />'
            f'<div class="blog-figcap" style="color:#334155;">{html.escape(analysis["fig_caption"])}</div></div>'
        )
    elif analysis.get("svg"):
        fig_block = f'<div class="blog-figure">{analysis["svg"]}<div class="blog-figcap">{html.escape(analysis["fig_caption"])}</div></div>'
    else:
        fig_block = ""

    lead_summary = analysis["findings"][0] if analysis["findings"] else exp_data["hypothesis"]

    blog_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#000000" />
  <title>{html.escape(exp_data['title'])} — Space AI</title>
  <meta name="description" content="{html.escape(lead_summary[:220])}" />
  <meta property="og:type" content="article" />
  <meta property="og:site_name" content="Space AI" />
  <meta property="og:title" content="{html.escape(exp_data['title'])} — Space AI" />
  <meta property="og:description" content="{html.escape(lead_summary[:220])}" />
  <meta property="og:image" content="https://spaceresearch.ai/og-card.png" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:image" content="https://spaceresearch.ai/og-card.png" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
  <link rel="icon" type="image/svg+xml" href="./mark.svg?v=35" />
  <link rel="stylesheet" href="./culture.css?v=30" />
</head>
<body>
  <canvas id="cosmos-bg-canvas" class="cosmos-bg-canvas" aria-hidden="true"></canvas>

  <div class="page-shell">
  <header class="ed-nav">
    <div class="ed-nav-inner">
      <a class="ed-brand" href="./index.html">
        <span class="ed-emblem" aria-hidden="true">
          <img src="./mark.svg?v=35" alt="" width="28" height="28" />
        </span>
        <span>Space AI</span>
      </a>
      <nav class="ed-links" aria-label="Primary">
        <a href="./blog.html" class="active">Research</a>
        <a href="./papers.html" class="">Papers</a>
        <a href="./product.html" class="">Product</a>
        <a href="./network.html" class="">Network</a>
        <a href="./chat.html">Chat</a>
        <a href="https://github.com/spaceresearchai/spaceresearch.ai/releases/latest" target="_blank" rel="noopener">Desktop App</a>
      </nav>
      <div class="nav-actions">
        <a class="icon-pill" href="https://github.com/spaceresearchai" target="_blank" rel="noopener" aria-label="Space AI on GitHub" title="GitHub (spaceresearchai)">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.69-3.88-1.54-3.88-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.18 1.18a11.05 11.05 0 0 1 5.8 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.26 5.68.41.36.78 1.06.78 2.14 0 1.55-.01 2.8-.01 3.18 0 .31.21.67.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
        </a>
        <button type="button" class="icon-pill" data-theme-toggle aria-label="Toggle color theme" title="Switch to Daylight Mode">&#9728;</button>
        <a class="btn sm" href="./chat.html">Open Chat</a>
      </div>
    </div>
  </header>

    <main>
      <article class="section" style="padding-top:3.8rem;">
        <div class="wrap-blog">
          <a class="blog-back" href="./blog.html">&larr; All research notes</a>
          <div class="blog-meta" style="margin-top:1.4rem;">
            <span class="blog-cat">{html.escape(note_id)} &middot; {html.escape(exp_data.get('category_label', 'AUTONOMOUS GPU LAB'))}</span>
            <span>{html.escape(author)} &middot; {html.escape(company)} &middot; {html.escape(pub_date)} {html.escape(now_iso[11:16])}</span>
          </div>
          <h1 class="blog-article-title">{html.escape(exp_data['title'])}</h1>
          <p class="blog-article-lead">
            Triggered automatically by CoreRank Discovery ({html.escape(exp_data.get('source_discovery_type', 'ACE'))}) and executed end-to-end on our {html.escape(str(exp_data.get('setup', {}).get('gpu_name', 'RTX 5090')))} workstation GPU in {html.escape(str(exp_data.get('setup', {}).get('total_elapsed_sec', '')))} seconds.
          </p>

          <div class="blog-abstract">
            <div class="blog-abstract-lbl">AUTONOMOUS GPU INVESTIGATION SUMMARY</div>
            <p>{html.escape(exp_data['hypothesis'])}</p>
          </div>

          <div class="blog-prose">
            <div class="blog-stats">{stats_html}</div>

            <h2>1. What the GPU experiment actually found</h2>
            <ul>{findings_html}</ul>

            {fig_block}

            <h2>2. Complete empirical table</h2>
            {analysis['table_html']}

            <h2>3. Source papers connected by CoreRank</h2>
            <ul>{sources_html}</ul>

            <h2>4. Honest limitations &amp; next steps</h2>
            <ul>{limits_html}</ul>

            <div class="blog-callout">
              <div class="blog-callout-title">Full Technical Preprint &amp; Raw GPU Metrics</div>
              <p>Read the generated technical manuscript (with BibTeX, exact run configuration, and raw JSON telemetry) or explore the source papers in CoreRank Discovery.</p>
              <div style="display:flex; flex-wrap:wrap; gap:0.65rem;">
                <a class="btn sm" href="{html.escape(paper_href)}">Read Full Technical Paper &rarr;</a>
                <a class="btn secondary sm" href="./papers/{html.escape(slug)}.pdf" target="_blank" rel="noopener">Download PDF &darr;</a>
                {f'<a class="btn secondary sm" href="{html.escape(existing_zurl or ("https://doi.org/" + existing_doi))}" target="_blank" rel="noopener">Zenodo DOI: {html.escape(existing_doi)} &#8599;</a>' if existing_doi else ''}
                <a class="btn secondary sm" href="./papers.html?tab=discover">Open CoreRank Discovery</a>
              </div>
            </div>
          </div>
        </div>
      </article>
    </main>

    <footer class="landing-foot">
      <div class="wrap" style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:1rem;">
        <span>&copy; 2026 Space AI &middot; spaceresearch.ai</span>
        <div style="display:flex; gap:1.4rem; color:var(--ink-soft);">
          <a href="./blog.html">Research</a>
          <a href="./papers.html">Papers</a>
          <a href="./product.html">Product</a>
          <a href="./network.html">Network</a>
          <a href="./chat.html">Chat</a>
          <a href="https://github.com/spaceresearchai" target="_blank" rel="noopener">GitHub</a>
        </div>
      </div>
    </footer>
  </div>

  <script src="./culture-home.js?v=30"></script>
</body>
</html>
"""
    blog_path = SITE_ROOT / blog_filename
    blog_path.write_text(blog_html, encoding="utf-8")

    # 3. Register atomically in `data/blog-posts.json`
    entry = {
        "seq": seq,
        "note_id": note_id,
        "slug": blog_slug,
        "href": blog_href,
        "title": exp_data["title"],
        "short_title": exp_data.get("short_title") or exp_data["title"][:48],
        "description": lead_summary,
        "authors": [author, company],
        "category_label": exp_data.get("category_label", "AUTONOMOUS GPU LAB"),
        "short_cat": exp_data.get("short_cat", "Auto GPU Lab"),
        "track": exp_data.get("track", "discovery"),
        "origin": "autonomous-gpu",
        "cluster": int(exp_data.get("cluster", 1)),
        "dot": exp_data.get("dot", "violet"),
        "published_at": now_iso,
        "published_date": pub_date,
        "read_minutes": 5,
        "tags": ["autonomous-gpu", "corerank", exp_data["experiment_id"], slug],
        "paper_href": paper_href,
        "experiment_metrics": analysis["metric_chip"],
    }
    manifest["posts"] = [p for p in manifest.get("posts", []) if p.get("slug") != blog_slug]
    manifest["posts"].append(entry)
    manifest["updated_at"] = now_iso
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # 4. Embed & upsert generated paper directly into `data/papers-db.json` + `papers-vectors.f16.bin`
    vdb_res = None
    if index_in_papers_vdb:
        abstract_text = f"{exp_data['hypothesis']} {' '.join(analysis['findings'])}"[:860]
        vdb_rec = {
            "id": f"spaceai-{slug}",
            "title": exp_data["title"],
            "authors": [author, company],
            "year": int(pub_date[:4]),
            "published_date": pub_date,
            "venue": f"{company} Technical Paper",
            "doi": existing_doi,
            "url": f"./papers/{slug}.html",
            "pdf_url": f"./papers/{slug}.pdf",
            "tags": ["space-ai", "autonomous-gpu", "corerank", exp_data["experiment_id"]],
            "abstract": abstract_text,
        }
        vdb_res = upsert_via_torch_subprocess(vdb_rec)

    # 5. Conference-readiness quality gate + Zenodo deposit (draft-only unless conference bar passes!)
    q_report = evaluate_conference_readiness(exp_data)
    zenodo_res = None
    eff_mode = zenodo_mode
    if eff_mode == "auto":
        zcfg = load_zenodo_config()
        if zcfg.get("auto_publish") and q_report.passed and zcfg.get("token"):
            eff_mode = "prod_publish"
        elif zcfg.get("token"):
            eff_mode = "prod_draft"
        elif zcfg.get("sandbox_token"):
            eff_mode = "sandbox_draft"
        else:
            eff_mode = "none"

    if eff_mode != "none":
        try:
            is_sandbox = eff_mode.startswith("sandbox")
            do_pub = eff_mode.endswith("publish")
            zenodo_res = deposit_paper_to_zenodo(slug=slug, sandbox=is_sandbox, publish=do_pub)
            if zenodo_res.get("doi") and not is_sandbox and zenodo_res.get("status") == "published":
                # Re-render HTML + PDF with the minted permanent DOI badge & BibTeX doi field!
                paper_res = compile_paper(
                    exp_data, published_at=now_iso, blog_href=blog_href,
                    author=author, email=email, company=company,
                    doi=zenodo_res["doi"], zenodo_url=zenodo_res.get("record_url"),
                )
        except Exception as e:
            zenodo_res = {"status": "error", "error": str(e)}

    # 6. Deterministically rebuild `blog.html` and `index.html`
    index_summary = rebuild_all()
    return {
        "dry_run": False,
        "note_id": note_id,
        "seq": seq,
        "authors": [author, company],
        "email": email,
        "blog_path": str(blog_path),
        "paper_html": str(paper_res["paper_html"]),
        "paper_pdf": str(paper_res.get("paper_pdf") or ""),
        "pdf_pages": paper_res.get("pdf_pages", 0),
        "paper_md": str(paper_res["paper_md"]),
        "metrics_json": str(paper_res["metrics_json"]),
        "metric_chip": analysis["metric_chip"],
        "quality_gate": q_report.to_dict(),
        "vdb_index": vdb_res,
        "zenodo": zenodo_res,
        "index_summary": index_summary,
    }
