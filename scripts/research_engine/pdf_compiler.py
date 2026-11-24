#!/usr/bin/env python3
"""
Academic Preprint PDF Compiler (`pdf_compiler.py`).

Renders a crisp, print-optimized academic PDF (`papers/<slug>.pdf`) from any
completed GPU experiment JSON + analysis using headless Brave/Chromium, suitable
for direct download on `spaceresearch.ai` and permanent archival/DOI minting on Zenodo.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SITE_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = SITE_ROOT / "papers"
GEN_DATA_DIR = SITE_ROOT / "data" / "generated_papers"


def find_browser_bin() -> Optional[str]:
    for cand in (
        "/usr/bin/brave-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if Path(cand).exists():
            return cand
    for name in ("brave-browser", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _retheme_svg_for_print(svg: str) -> str:
    """Adapt dark-mode inline SVG charts for crisp white-paper academic PDF printing."""
    if not svg:
        return ""
    out = svg
    replacements = [
        ('fill="#0b0b0d"', 'fill="#f8fafc"'),
        ('fill="#06070a"', 'fill="#ffffff"'),
        ('stroke="rgba(255,255,255,0.14)"', 'stroke="#cbd5e1"'),
        ('stroke="rgba(255,255,255,0.16)"', 'stroke="#cbd5e1"'),
        ('fill="#ffffff"', 'fill="#0f172a"'),
        ('fill="#cbd5e1"', 'fill="#334155"'),
        ('fill="#a1a1aa"', 'fill="#475569"'),
        ('fill="#6b7185"', 'fill="#64748b"'),
        ('fill="#38bdf8"', 'fill="#0284c7"'),
        ('fill="#34d399"', 'fill="#059669"'),
        ('fill="#6ee7b7"', 'fill="#047857"'),
        ('fill="#c4b5fd"', 'fill="#6d28d9"'),
        ('fill="#a78bfa"', 'fill="#7c3aed"'),
        ('stroke="#34d399"', 'stroke="#059669"'),
        ('stroke="#a78bfa"', 'stroke="#7c3aed"'),
        ('stroke="#f59e0b"', 'stroke="#d97706"'),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def compile_paper_pdf(
    exp_data: Dict[str, Any],
    analysis: Dict[str, Any],
    author: str = "Marcio Diaz",
    email: str = "marcio.diaz@gmail.com",
    company: str = "Space AI",
    published_at: Optional[str] = None,
    doi: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile a print-grade academic preprint PDF to `papers/<slug>.pdf`."""
    now_iso = published_at or datetime.now().astimezone().isoformat(timespec="seconds")
    date_str = now_iso[:10]
    slug = exp_data["slug"]

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print_html_path = GEN_DATA_DIR / f"{slug}.print.html"
    pdf_path = PAPERS_DIR / f"{slug}.pdf"

    findings_li = "".join(f"<li>{html.escape(f)}</li>" for f in analysis.get("findings", []))
    limits_li = "".join(f"<li>{html.escape(l)}</li>" for l in analysis.get("limitations", []))
    sources_li = "".join(
        f'<li>{html.escape(s["title"])} (<code>{html.escape(s["id"])}</code>) &mdash; <span class="url">{html.escape(s.get("url", ""))}</span></li>'
        for s in exp_data.get("source_papers", [])
    )
    stats_cells = "".join(
        f'<div class="kpi"><div class="kpi-v">{html.escape(v)}</div><div class="kpi-l">{html.escape(l)}</div></div>'
        for v, l in analysis.get("stats", [])
    )
    svg_print = _retheme_svg_for_print(analysis.get("svg", ""))
    fig_block = (
        f'<figure class="fig">{svg_print}<figcaption>{html.escape(analysis.get("fig_caption", ""))}</figcaption></figure>'
        if svg_print
        else ""
    )

    last_name = author.split()[-1].lower() if author.split() else "spaceai"
    bibtex_key = f"{last_name}{date_str[:4]}{slug.replace('-', '')[:16]}"
    doi_bib = f"  doi         = {{{doi}}},\n" if doi else ""
    bibtex = (
        f"@techreport{{{bibtex_key},\n"
        f"  title       = {{{exp_data['title']}}},\n"
        f"  author      = {{{author} and {{{company}}}}},\n"
        f"  institution = {{{company} (spaceresearch.ai)}},\n"
        f"  year        = {{{date_str[:4]}}},\n"
        f"  month       = {{{date_str[5:7]}}},\n"
        f"{doi_bib}"
        f"  url         = {{https://spaceresearch.ai/papers/{slug}.html}}\n"
        f"}}"
    )
    doi_line = f' &middot; DOI: <code>{html.escape(doi)}</code>' if doi else ""

    print_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(exp_data['title'])}</title>
  <style>
    @page {{ size: letter; margin: 0.68in 0.72in 0.72in 0.72in; }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Liberation Serif", "DejaVu Serif", Georgia, "Times New Roman", serif;
      color: #0f172a;
      background: #ffffff;
      font-size: 10.2pt;
      line-height: 1.46;
      margin: 0;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 7.5pt;
      color: #475569;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 5px;
      margin-bottom: 14px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    h1 {{
      font-size: 17pt;
      line-height: 1.22;
      font-weight: 700;
      text-align: center;
      margin: 0 0 10px 0;
      color: #020617;
    }}
    .authors {{
      text-align: center;
      font-size: 10.2pt;
      margin-bottom: 4px;
    }}
    .affil {{
      text-align: center;
      font-size: 8.8pt;
      color: #475569;
      margin-bottom: 14px;
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
    }}
    .abstract-box {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 3px solid #0284c7;
      padding: 10px 13px;
      margin: 0 0 14px 0;
      font-size: 9.4pt;
    }}
    .abstract-lbl {{
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 7.6pt;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: #0369a1;
      margin-bottom: 4px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin: 10px 0 14px 0;
      break-inside: avoid;
    }}
    .kpi {{
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 7px 8px;
      text-align: center;
      background: #f8fafc;
    }}
    .kpi-v {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      font-size: 11.5pt;
      font-weight: 700;
      color: #0f172a;
    }}
    .kpi-l {{
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 6.7pt;
      color: #475569;
      text-transform: uppercase;
      margin-top: 2px;
    }}
    h2 {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      font-size: 11.2pt;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 3px;
      margin: 15px 0 7px 0;
      break-after: avoid;
    }}
    p {{ margin: 0 0 8px 0; text-align: justify; }}
    ul {{ margin: 0 0 10px 0; padding-left: 18px; }}
    li {{ margin-bottom: 5px; }}
    code {{
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 8.3pt;
      background: #f1f5f9;
      padding: 1px 3px;
      border-radius: 3px;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      padding: 8px 10px;
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 7.5pt;
      line-height: 1.38;
      white-space: pre-wrap;
      word-break: break-word;
      break-inside: avoid;
    }}
    pre code {{ background: transparent; padding: 0; font-size: inherit; }}
    .table-wrap {{ margin: 8px 0 12px 0; break-inside: avoid; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 8.2pt;
      font-family: "DejaVu Sans", Arial, sans-serif;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 4.5px 6px;
      text-align: left;
    }}
    th {{ background: #f1f5f9; font-weight: 700; color: #0f172a; }}
    figure.fig {{
      margin: 12px 0 14px 0;
      padding: 8px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      break-inside: avoid;
    }}
    figure.fig svg {{ width: 100%; height: auto; display: block; }}
    figcaption {{
      font-size: 8.2pt;
      color: #475569;
      text-align: center;
      margin-top: 6px;
      font-style: italic;
    }}
    .url {{ font-family: "DejaVu Sans Mono", ui-monospace, monospace; font-size: 7.8pt; color: #0369a1; }}
    .footer {{
      margin-top: 16px;
      padding-top: 6px;
      border-top: 1px solid #cbd5e1;
      font-family: "DejaVu Sans Mono", ui-monospace, monospace;
      font-size: 7.2pt;
      color: #64748b;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 18px;
    }}
    .footer span:first-child {{ white-space: nowrap; }}
    .footer span:last-child {{ text-align: right; }}
  </style>
</head>
<body>
  <div class="topbar">
    <span>{html.escape(company)} Technical Preprint &middot; spaceresearch.ai</span>
    <span>{html.escape(date_str)}{doi_line}</span>
  </div>

  <h1>{html.escape(exp_data['title'])}</h1>
  <div class="authors"><strong>{html.escape(author)}</strong> &middot; {html.escape(company)}</div>
  <div class="affil">{html.escape(email)} &middot; https://spaceresearch.ai/papers/{html.escape(slug)}.html</div>

  <div class="abstract-box">
    <div class="abstract-lbl">ABSTRACT &amp; HYPOTHESIS</div>
    <p style="margin:0;">{html.escape(exp_data['hypothesis'])}</p>
  </div>

  <div class="kpi-grid">{stats_cells}</div>

  <h2>1. Empirical Findings</h2>
  <ul>{findings_li}</ul>

  {fig_block}

  <h2>2. Quantitative Experimental Results</h2>
  {analysis.get('table_html', '')}

  <h2>3. Source Literature &amp; Discovery Provenance</h2>
  <p><strong>CoreRank Discovery Trigger:</strong> {html.escape(exp_data.get('source_discovery_type', 'CoreRank ACE Discovery'))}</p>
  <ul>{sources_li}</ul>

  <h2>4. Hardware &amp; Reproducibility Configuration</h2>
  <pre><code>{html.escape(json.dumps(exp_data.get('setup', {}), indent=2))}</code></pre>

  <h2>5. Honest Limitations &amp; Threats to Validity</h2>
  <ul>{limits_li}</ul>

  <h2>6. BibTeX Citation</h2>
  <pre><code>{html.escape(bibtex)}</code></pre>

  <div class="footer">
    <span>&copy; {html.escape(date_str[:4])} {html.escape(author)} &middot; {html.escape(company)} (CC-BY-4.0)</span>
    <span>Reproducible artifacts: spaceresearch.ai/data/generated_papers/{html.escape(slug)}.json</span>
  </div>
</body>
</html>
"""
    print_html_path.write_text(print_html, encoding="utf-8")

    browser = find_browser_bin()
    if not browser:
        raise RuntimeError("No headless Chromium/Brave browser found for PDF compilation.")

    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{print_html_path}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=45,
        check=True,
    )

    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError(f"PDF rendering produced empty/missing output at {pdf_path}")

    pages = 0
    if Path("/usr/bin/pdfinfo").exists():
        info = subprocess.check_output(["/usr/bin/pdfinfo", str(pdf_path)], text=True, timeout=10)
        for line in info.splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split(":", 1)[1].strip())
                break

    return {
        "pdf_path": pdf_path,
        "print_html": print_html_path,
        "bytes": pdf_path.stat().st_size,
        "pages": pages,
        "browser": browser,
    }
