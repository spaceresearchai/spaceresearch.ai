#!/usr/bin/env python3
"""
Deterministic Chronological Blog Index Builder (`rebuild_blog_index.py`).

Reads `data/blog-posts.json` (single source of truth), enforces strict sub-day
ordering `(published_at DESC, seq DESC)`, validates every linked HTML/paper file
exists, and synchronizes:
  1. Static SEO / no-JS day-grouped fallback inside `blog.html`
  2. Latest-9 research strip (newest first) inside `index.html`
"""

from __future__ import annotations

import html
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

SITE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = SITE_ROOT / "data" / "blog-posts.json"
BLOG_HTML = SITE_ROOT / "blog.html"
INDEX_HTML = SITE_ROOT / "index.html"


def parse_iso_ts(iso_str: str) -> float:
    if not iso_str:
        return 0.0
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def sort_posts_newest(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strict deterministic ordering: newest full ISO timestamp, tie-broken by seq DESC."""
    return sorted(
        posts,
        key=lambda p: (parse_iso_ts(p.get("published_at", "")), int(p.get("seq", 0))),
        reverse=True,
    )


def fmt_hhmm(iso_str: str) -> str:
    m = re.search(r"T(\d{2}:\d{2})", iso_str or "")
    return m.group(1) if m else "00:00"


def fmt_day_label(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%a, %b %-d, %Y")
    except Exception:
        return date_str or "Undated"


def group_posts_by_day(posts_sorted: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for p in posts_sorted:
        day = p.get("published_date") or (p.get("published_at") or "")[:10] or "undated"
        groups.setdefault(day, []).append(p)
    return groups


def render_static_blog_feed(posts_sorted: List[Dict[str, Any]]) -> str:
    groups = group_posts_by_day(posts_sorted)
    latest_day = next(iter(groups.keys()), "")
    sections: List[str] = []

    for day, items in groups.items():
        is_latest = day == latest_day
        times = sorted(fmt_hhmm(p.get("published_at", "")) for p in items)
        span_txt = f"{times[0]} – {times[-1]}" if len(times) > 1 else times[0]
        badge_cls = "blog-day-badge today" if is_latest else "blog-day-badge"
        badge_lbl = f"Latest &middot; {len(items)} note{'s' if len(items) != 1 else ''}" if is_latest else f"{len(items)} note{'s' if len(items) != 1 else ''}"

        cards: List[str] = []
        for p in items:
            hhmm = fmt_hhmm(p.get("published_at", ""))
            origin_html = (
                '<span class="blog-origin-badge auto">Autonomous GPU Lab</span>'
                if p.get("origin") == "autonomous-gpu"
                else ""
            )
            metric_html = (
                f'<div class="blog-metric-chip">{html.escape(p["experiment_metrics"])}</div>'
                if p.get("experiment_metrics")
                else ""
            )
            paper_html = (
                f'<a class="blog-paper-link" href="{html.escape(p["paper_href"])}">Full Technical Paper &rarr;</a>'
                if p.get("paper_href")
                else ""
            )
            cards.append(
                f"""              <article class="blog-card" data-cluster="{int(p.get('cluster', 0))}">
                <div class="blog-meta">
                  <span class="blog-cat"><span class="spec-dot {html.escape(p.get('dot', 'cyan'))}" style="margin-right:0.36rem;"></span>{html.escape(p['note_id'])} &middot; {html.escape(p['category_label'])}</span>
                  {origin_html}
                  <span class="blog-time-tag">{html.escape(day)} &middot; {html.escape(hhmm)} &middot; {int(p.get('read_minutes', 5))} min</span>
                </div>
                <h2 class="blog-card-title"><a href="{html.escape(p['href'])}" style="color:inherit;text-decoration:none;">{html.escape(p['title'])}</a></h2>
                {metric_html}
                <p class="blog-card-desc">{html.escape(p['description'])}</p>
                <div class="blog-card-foot">
                  <a class="blog-read-more" href="{html.escape(p['href'])}">Read research note &rarr;</a>
                  {paper_html}
                </div>
              </article>"""
            )

        sections.append(
            f"""            <section class="blog-day-group" data-day-section="{html.escape(day)}">
              <div class="blog-day-header">
                <div class="blog-day-title-wrap">
                  <span class="blog-day-date">{html.escape(fmt_day_label(day))}</span>
                  <span class="{badge_cls}">{badge_lbl}</span>
                  <span class="blog-day-span">{html.escape(span_txt)}</span>
                </div>
              </div>
              <div class="blog-list">
{chr(10).join(cards)}
              </div>
            </section>"""
        )

    return "\n".join(sections)


def render_landing_blog_strip(posts_sorted: List[Dict[str, Any]], max_items: int = 9) -> str:
    top = posts_sorted[:max_items]
    rows: List[str] = []
    for p in top:
        seq_str = f"{int(p.get('seq', 0)):02d}"
        short_cat = p.get("short_cat") or p.get("category_label", "Research")
        short_title = p.get("short_title") or p.get("title", "")
        hhmm = fmt_hhmm(p.get("published_at", ""))
        date_mmdd = (p.get("published_date") or "")[5:]
        rows.append(
            f"""            <a class="landing-blog-item" href="{html.escape(p['href'])}" data-cluster="{int(p.get('cluster', 0))}">
              <div class="landing-blog-cat"><span class="spec-dot {html.escape(p.get('dot', 'cyan'))}"></span><span>{seq_str} &middot; {html.escape(short_cat)} &middot; {html.escape(date_mmdd)} {html.escape(hhmm)}</span></div>
              <div class="landing-blog-t">{html.escape(short_title)} &rarr;</div>
            </a>"""
        )
    return "\n".join(rows)


def replace_between_markers(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), flags=re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"Markers {start_marker!r} .. {end_marker!r} not found")
    return pattern.sub(f"{start_marker}\n{replacement}\n            {end_marker}", text, count=1)


def rebuild_all(
    manifest_path: Path = MANIFEST_PATH,
    blog_html_path: Path = BLOG_HTML,
    index_html_path: Path = INDEX_HTML,
    strict_files: bool = True,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    posts = sort_posts_newest(manifest.get("posts", []))

    # Normalize manifest order in-place so JSON itself is always strictly newest-first?
    # Keep seq ascending in JSON or keep newest-first; sort deterministically on load either way.
    missing: List[str] = []
    site_dir = blog_html_path.parent
    for p in posts:
        href = p.get("href", "").lstrip("./")
        if href and not (site_dir / href).exists():
            missing.append(href)
        paper_href = (p.get("paper_href") or "").lstrip("./")
        if paper_href and not (site_dir / paper_href).exists():
            missing.append(paper_href)
    if strict_files and missing:
        raise FileNotFoundError(f"Blog manifest references missing files: {missing}")

    if blog_html_path.exists():
        blog_src = blog_html_path.read_text(encoding="utf-8")
        feed_html = render_static_blog_feed(posts)
        blog_new = replace_between_markers(blog_src, "<!-- BLOG_FEED_START -->", "<!-- BLOG_FEED_END -->", feed_html)
        blog_html_path.write_text(blog_new, encoding="utf-8")

    if index_html_path.exists():
        idx_src = index_html_path.read_text(encoding="utf-8")
        strip_html = render_landing_blog_strip(posts, max_items=9)
        idx_new = replace_between_markers(idx_src, "<!-- LANDING_BLOG_START -->", "<!-- LANDING_BLOG_END -->", strip_html)
        index_html_path.write_text(idx_new, encoding="utf-8")

    days = group_posts_by_day(posts)
    return {
        "post_count": len(posts),
        "day_count": len(days),
        "latest_day": next(iter(days.keys()), ""),
        "latest_day_count": len(next(iter(days.values()), [])),
        "missing_files": missing,
    }


if __name__ == "__main__":
    summary = rebuild_all()
    print(f"[blog-index] Rebuilt blog.html & index.html: {summary}")
