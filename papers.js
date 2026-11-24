/* Space AI — Research Papers Vector Database & 2D Semantic Atlas Frontend (v2: Live arXiv Today + Conferences) */
(() => {
  const REMOTE_API = "https://api.spaceresearch.ai";
  const LOCAL_API = "http://127.0.0.1:8787";
  const DIM = 384;
  const PAGE_SIZE = 24;

  const VENUE_BADGE_LABELS = {
    "neurips": "NeurIPS",
    "iclr": "ICLR",
    "icml": "ICML",
    "acl-emnlp": "ACL/EMNLP",
    "systems-conf": "MLSys/ASPLOS",
    "hf-daily": "HF Daily",
    "space-ai": "Space AI",
  };

  const state = {
    db: null,
    papers: [],
    byId: new Map(),
    catMap: new Map(),
    vectors: null,
    apiBase: null,
    apiInfo: null,

    // UI filters
    query: "",
    mode: "hybrid",
    venue: "all",
    category: "all",
    yearRange: "all",
    sort: "newest",
    similarAnchorId: null,
    visibleCount: PAGE_SIZE,
    currentResults: [],
    lastEngineLabel: "",
    lastLatencyMs: 0,
    searchToken: 0,

    // Atlas state
    atlasTabActive: false,
    discoveries: null,
    discLens: "surprising",
    atlasZoom: 1.0,
    atlasPanX: 0,
    atlasPanY: 0,
    atlasDragging: false,
    atlasDragStart: { x: 0, y: 0, px: 0, py: 0 },
    atlasHoveredIdx: -1,
    atlasSelectedIdx: -1,
    atlasHighlightIds: null,
  };

  // ---------- Helpers ----------
  function esc(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtNum(n) {
    const v = Number(n || 0);
    if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + "M";
    if (v >= 10_000) return Math.round(v / 1000) + "K";
    if (v >= 1_000) return (v / 1_000).toFixed(1) + "K";
    return String(v);
  }

  function showToast(msg) {
    const el = document.getElementById("vdb-toast");
    if (!el) return;
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.remove("show"), 2200);
  }

  function decodeFloat16Buffer(buf) {
    const u16 = new Uint16Array(buf);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x03ff;
      if (e === 0) out[i] = s * Math.pow(2, -14) * (m / 1024);
      else if (e === 31) out[i] = m ? NaN : s * Infinity;
      else out[i] = s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return out;
  }

  function cosineDotAt(vecs, idx, qvec) {
    const off = idx * DIM;
    let s = 0.0;
    for (let d = 0; d < DIM; d++) s += vecs[off + d] * qvec[d];
    return s;
  }

  function cosineBetweenIdx(vecs, a, b) {
    const oa = a * DIM;
    const ob = b * DIM;
    let s = 0.0;
    for (let d = 0; d < DIM; d++) s += vecs[oa + d] * vecs[ob + d];
    return s;
  }

  function parseYearRange(val) {
    if (!val || val === "all") return [null, null];
    const parts = val.split("-").map(Number);
    return [parts[0] || null, parts[1] || null];
  }

  function matchesFilters(p) {
    if (state.category !== "all" && p.category !== state.category) return false;
    if (state.venue !== "all") {
      if (state.venue === "hf-daily") {
        if (p.venue_group !== "hf-daily" && p.venue_group !== "arxiv-today") return false;
      } else if (p.venue_group !== state.venue) {
        return false;
      }
    }
    const [ymin, ymax] = parseYearRange(state.yearRange);
    if (ymin !== null && p.year < ymin) return false;
    if (ymax !== null && p.year > ymax) return false;
    return true;
  }

  function buildBibtex(p) {
    const firstAuth = ((p.authors && p.authors[0]) || "unknown")
      .split(/\s+/)
      .slice(-1)[0]
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
    const firstWord = (p.title || "paper")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .split(/\s+/)
      .find((w) => w.length > 3) || "paper";
    const key = `${firstAuth}${p.year || ""}${firstWord}`;
    const authorsStr = (p.authors || []).join(" and ");
    const lines = [
      `@article{${key},`,
      `  title   = {${p.title}},`,
      `  author  = {${authorsStr}},`,
      `  year    = {${p.year || ""}},`,
    ];
    if (p.arxiv_id) {
      lines.push(`  eprint  = {${p.arxiv_id}},`);
      lines.push(`  archivePrefix = {arXiv},`);
    } else if (p.venue) {
      lines.push(`  journal = {${p.venue}},`);
    }
    if (p.url) lines.push(`  url     = {${p.url}},`);
    lines.push("}");
    return lines.join("\n");
  }

  function tokenize(text) {
    return (text || "")
      .toLowerCase()
      .replace(/[^a-z0-9\-\.]+/g, " ")
      .split(/\s+/)
      .filter((t) => t.length >= 2);
  }

  function sortList(list, sort) {
    const arr = list.slice();
    if (sort === "citations") {
      arr.sort((a, b) => (b.citations - a.citations) || String(b.published_date || "").localeCompare(String(a.published_date || "")));
    } else if (sort === "newest") {
      arr.sort((a, b) => String(b.published_date || "").localeCompare(String(a.published_date || "")) || (b.citations - a.citations));
    } else if (sort === "oldest") {
      arr.sort((a, b) => String(a.published_date || "").localeCompare(String(b.published_date || "")) || (b.citations - a.citations));
    } else {
      arr.sort((a, b) => {
        if (a.score !== undefined && b.score !== undefined && a.score !== b.score) return b.score - a.score;
        return (b.citations - a.citations) || String(b.published_date || "").localeCompare(String(a.published_date || ""));
      });
    }
    return arr;
  }

  function clientSearch(q, mode) {
    const t0 = performance.now();
    const terms = tokenize(q);
    const filtered = state.papers.filter(matchesFilters);
    if (!terms.length) {
      return {
        results: sortList(filtered.map((p) => ({ ...p })), state.sort),
        engine: "Local Vector Matrix (384-d Float16)",
        latencyMs: performance.now() - t0,
      };
    }

    const lexScored = [];
    for (const p of state.papers) {
      const tl = p._titleLow;
      const al = p._absLow;
      const tg = p._tagsLow;
      const au = p._authLow;
      const vn = p._venueLow;
      let score = 0;
      let matched = 0;
      for (const term of terms) {
        let hit = false;
        if (tl.includes(term)) { score += 6.5; hit = true; }
        if (tg.includes(term) || vn.includes(term)) { score += 4.2; hit = true; }
        if (au.includes(term)) { score += 2.6; hit = true; }
        if (al.includes(term)) { score += 1.6; hit = true; }
        if (hit) matched++;
      }
      if (q.length >= 4 && tl.includes(q.toLowerCase())) score += 8.0;
      if (score > 0) {
        score *= (0.55 + 0.45 * (matched / terms.length));
        lexScored.push({ p, score });
      }
    }
    lexScored.sort((a, b) => b.score - a.score);
    const maxLex = lexScored.length ? lexScored[0].score : 1.0;

    const vecMap = new Map();
    if (state.vectors && mode !== "lexical" && lexScored.length > 0) {
      const seeds = lexScored.slice(0, Math.min(6, lexScored.length));
      const centroid = new Float32Array(DIM);
      for (const s of seeds) {
        const w = s.score / maxLex;
        const off = s.p.idx * DIM;
        for (let d = 0; d < DIM; d++) centroid[d] += w * state.vectors[off + d];
      }
      let norm = 0;
      for (let d = 0; d < DIM; d++) norm += centroid[d] * centroid[d];
      norm = Math.sqrt(norm) + 1e-9;
      for (let d = 0; d < DIM; d++) centroid[d] /= norm;

      for (const p of filtered) {
        vecMap.set(p.id, cosineDotAt(state.vectors, p.idx, centroid));
      }
    }

    const lexMap = new Map();
    for (const item of lexScored) lexMap.set(item.p.id, item.score / maxLex);

    const out = [];
    for (const p of filtered) {
      const ls = lexMap.get(p.id) || 0;
      const vs = vecMap.get(p.id) || 0;
      let comb = 0;
      if (mode === "lexical") comb = ls;
      else if (mode === "vector" && vecMap.size) comb = vs;
      else comb = vecMap.size ? 0.62 * vs + 0.38 * ls : ls;
      if (comb > 0.08 || ls > 0) {
        out.push({
          ...p,
          score: Number(comb.toFixed(4)),
          vector_score: Number(vs.toFixed(4)),
          lexical_score: Number(ls.toFixed(4)),
        });
      }
    }
    out.sort((a, b) => b.score - a.score);
    const effectiveSort = state.query.trim() && state.sort === "newest" ? "relevance" : state.sort;
    const finalList = effectiveSort === "relevance" ? out : sortList(out, effectiveSort);
    return {
      results: finalList.slice(0, 140),
      engine: vecMap.size ? "Browser 384-d Float16 Vector + BM25" : "Browser BM25",
      latencyMs: performance.now() - t0,
    };
  }

  function clientSimilar(anchorPaper) {
    const t0 = performance.now();
    if (!state.vectors) {
      const nb = (anchorPaper.neighbors || [])
        .map((n) => {
          const p = state.byId.get(n.id);
          return p ? { ...p, score: n.sim, vector_score: n.sim } : null;
        })
        .filter((p) => p && matchesFilters(p));
      return { results: nb, engine: "Precomputed 384-d Neighbors", latencyMs: performance.now() - t0 };
    }
    const out = [];
    for (const p of state.papers) {
      if (p.id === anchorPaper.id || !matchesFilters(p)) continue;
      const sim = cosineBetweenIdx(state.vectors, anchorPaper.idx, p.idx);
      out.push({ ...p, score: Number(sim.toFixed(4)), vector_score: Number(sim.toFixed(4)) });
    }
    out.sort((a, b) => b.vector_score - a.vector_score);
    return {
      results: out.slice(0, 72),
      engine: "Exact 384-d Float16 Cosine KNN",
      latencyMs: performance.now() - t0,
    };
  }

  // ---------- Live Vector DB API ----------
  async function probeLiveApi() {
    const isLocalHost = ["127.0.0.1", "localhost"].includes(window.location.hostname);
    const candidates = isLocalHost ? [LOCAL_API, REMOTE_API] : [REMOTE_API];
    for (const base of candidates) {
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 3200);
        const res = await fetch(`${base}/v1/papers/stats`, { signal: ctrl.signal });
        clearTimeout(timer);
        if (res.ok) {
          const info = await res.json();
          if (info && info.ready && info.paper_count > 0) {
            state.apiBase = base;
            state.apiInfo = info;
            updateStatusPill(true, info);
            return;
          }
        }
      } catch (_e) {}
    }
    updateStatusPill(false, null);
  }

  function updateStatusPill(online, info) {
    const el = document.getElementById("vdb-status-text");
    const badge = document.getElementById("api-engine-badge");
    if (!el) return;
    if (online && info) {
      const latest = info.latest_published_date ? ` · Synced ${info.latest_published_date}` : "";
      el.textContent = `Live Vector DB · ${info.vector_engine || "sqlite-vec"}${latest}`;
      if (badge) badge.textContent = `Live ${info.vector_engine || "sqlite-vec"} + BGE-small-en`;
    } else {
      el.textContent = "384-d Browser Vector Index · Ready";
      if (badge) badge.textContent = "Client 384-d Float16 Vector Matrix";
    }
  }

  async function executeSearch() {
    const myToken = ++state.searchToken;
    state.visibleCount = PAGE_SIZE;

    if (state.similarAnchorId) {
      const anchor = state.byId.get(state.similarAnchorId);
      if (anchor) {
        showAnchorBanner(anchor);
        const res = clientSimilar(anchor);
        state.currentResults = res.results;
        state.lastEngineLabel = res.engine;
        state.lastLatencyMs = res.latencyMs;
        state.atlasHighlightIds = new Set([anchor.id, ...res.results.slice(0, 16).map((r) => r.id)]);
        renderResults();
        if (state.atlasTabActive) drawAtlas();
        return;
      }
    } else {
      hideAnchorBanner();
    }

    const q = state.query.trim();
    if (!q) {
      const t0 = performance.now();
      const filtered = state.papers.filter(matchesFilters).map((p) => ({ ...p }));
      state.currentResults = sortList(filtered, state.sort);
      state.lastEngineLabel = state.apiBase ? "sqlite-vec + Live Catalog" : "384-d Vector Catalog";
      state.lastLatencyMs = performance.now() - t0;
      state.atlasHighlightIds = null;
      renderResults();
      if (state.atlasTabActive) drawAtlas();
      return;
    }

    if (state.apiBase) {
      try {
        const apiSort = state.sort === "newest" && !state._userChangedSortOnQuery ? "relevance" : state.sort;
        const params = new URLSearchParams({
          q,
          mode: state.mode,
          k: "90",
          sort: apiSort,
        });
        if (state.category !== "all") params.set("category", state.category);
        if (state.venue !== "all") params.set("venue", state.venue);
        const [ymin, ymax] = parseYearRange(state.yearRange);
        if (ymin !== null) params.set("year_min", String(ymin));
        if (ymax !== null) params.set("year_max", String(ymax));

        const res = await fetch(`${state.apiBase}/v1/papers/search?${params.toString()}`);
        if (myToken !== state.searchToken) return;
        if (res.ok) {
          const data = await res.json();
          const enriched = (data.results || []).map((r) => {
            const local = state.byId.get(r.id);
            return local ? { ...local, ...r } : r;
          });
          state.currentResults = enriched;
          state.lastEngineLabel = `${data.engine || "sqlite-vec"} · ${data.mode || state.mode} · BGE-small-en`;
          state.lastLatencyMs = Number(data.latency_ms || 0);
          state.atlasHighlightIds = new Set(enriched.slice(0, 28).map((r) => r.id));
          renderResults();
          if (state.atlasTabActive) drawAtlas();
          return;
        }
      } catch (_e) {
        if (myToken !== state.searchToken) return;
      }
    }

    const local = clientSearch(q, state.mode);
    if (myToken !== state.searchToken) return;
    state.currentResults = local.results;
    state.lastEngineLabel = local.engine;
    state.lastLatencyMs = local.latencyMs;
    state.atlasHighlightIds = new Set(local.results.slice(0, 28).map((r) => r.id));
    renderResults();
    if (state.atlasTabActive) drawAtlas();
  }

  // ---------- Rendering ----------
  function renderTopStats() {
    const db = state.db;
    if (!db) return;
    const confCount = (db.venues || [])
      .filter((v) => ["neurips", "iclr", "icml", "acl-emnlp", "systems-conf"].includes(v.id))
      .reduce((acc, v) => acc + (v.count || 0), 0);
    const elP = document.getElementById("stat-papers");
    const elT = document.getElementById("stat-today");
    const elTLbl = document.getElementById("stat-today-lbl");
    const elConf = document.getElementById("stat-confs");
    const elD = document.getElementById("stat-dims");
    const elBuilt = document.getElementById("api-built-at");
    if (elP) elP.textContent = (db.paper_count || state.papers.length).toLocaleString();
    if (elT) elT.textContent = String(db.today_count || 0);
    if (elTLbl && db.latest_published_date) elTLbl.textContent = `Released ${db.latest_published_date}`;
    if (elConf) elConf.textContent = confCount.toLocaleString();
    if (elD) elD.textContent = `${db.embedding_dim || 384}-d`;
    if (elBuilt) elBuilt.textContent = db.latest_published_date || (db.built_at || "").slice(0, 10);
  }

  function renderVenuePills() {
    const wrap = document.getElementById("vdb-venues");
    if (!wrap || !state.db || !state.db.venues) return;
    wrap.innerHTML = state.db.venues
      .filter((v) => v.id === "all" || v.count > 0)
      .map((v) => {
        const act = state.venue === v.id ? "active" : "";
        const isToday = v.id === "arxiv-today" ? "today-pill" : "";
        const dot = v.id === "arxiv-today" ? '<span class="dot" style="width:6px;height:6px;"></span>' : "";
        return `<button type="button" class="vdb-venue-pill ${isToday} ${act}" data-venue="${esc(v.id)}">${dot}<span>${esc(v.label)}</span><span class="vdb-domain-count">${v.count}</span></button>`;
      })
      .join("");
    wrap.querySelectorAll("[data-venue]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.venue = btn.getAttribute("data-venue") || "all";
        renderVenuePills();
        executeSearch();
      });
    });
  }

  function renderDomainPills() {
    const wrap = document.getElementById("vdb-domains");
    if (!wrap || !state.db) return;
    const allPill = `<button type="button" class="vdb-domain-pill ${state.category === "all" ? "active" : ""}" data-cat="all"><span>All Domains</span><span class="vdb-domain-count">${state.papers.length}</span></button>`;
    const catPills = (state.db.categories || []).map((c) => {
      const act = state.category === c.id ? "active" : "";
      return `<button type="button" class="vdb-domain-pill ${act}" data-cat="${esc(c.id)}"><span class="vdb-domain-dot" style="background:${esc(c.color)}"></span><span>${esc(c.short || c.label)}</span><span class="vdb-domain-count">${c.count}</span></button>`;
    });
    wrap.innerHTML = allPill + catPills.join("");
    wrap.querySelectorAll("[data-cat]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.category = btn.getAttribute("data-cat") || "all";
        renderDomainPills();
        executeSearch();
      });
    });
  }

  function showAnchorBanner(anchor) {
    const b = document.getElementById("vdb-anchor-banner");
    const t = document.getElementById("vdb-anchor-title");
    const m = document.getElementById("vdb-anchor-meta");
    if (!b || !t || !m) return;
    t.textContent = `Nearest vectors to: "${anchor.title}"`;
    m.textContent = `${anchor.published_date || anchor.year} · ${anchor.venue || anchor.arxiv_id || "Paper"} · Ranked by exact 384-dim BGE cosine similarity`;
    b.classList.add("visible");
  }

  function hideAnchorBanner() {
    const b = document.getElementById("vdb-anchor-banner");
    if (b) b.classList.remove("visible");
  }

  function renderResults() {
    const grid = document.getElementById("papers-grid");
    const meta = document.getElementById("vdb-result-meta");
    const loadBtn = document.getElementById("btn-load-more");
    if (!grid) return;

    const total = state.currentResults.length;
    const slice = state.currentResults.slice(0, state.visibleCount);
    const latStr = state.lastLatencyMs > 0 ? ` · ${state.lastLatencyMs.toFixed(1)} ms` : "";
    if (meta) {
      const prefix = state.similarAnchorId
        ? `Showing ${slice.length} of ${total} nearest vectors`
        : state.query.trim()
        ? `${total} semantic matches`
        : `${total.toLocaleString()} verified papers`;
      meta.textContent = `${prefix} · ${state.lastEngineLabel}${latStr}`;
    }

    if (!slice.length) {
      grid.innerHTML = `<div class="gp-card" style="grid-column:1 / -1; text-align:center; padding:2.5rem 1.2rem;">
        <p style="margin:0 0 0.5rem 0; font-family:var(--display); font-size:1.08rem;">No papers matched the current query and filters.</p>
        <p style="margin:0; font-size:0.84rem; color:var(--ink-muted);">Try switching to Hybrid mode, selecting All Venues, or clearing the domain filter.</p>
      </div>`;
      if (loadBtn) loadBtn.style.display = "none";
      return;
    }

    grid.innerHTML = slice.map((p) => {
      const cat = state.catMap.get(p.category) || { short: p.category, color: "#a1a6b8" };
      const authList = (p.authors || []).slice(0, 4).join(", ") + ((p.authors || []).length > 4 ? ` +${p.authors.length - 4}` : "");
      const hasSim = typeof p.vector_score === "number" && p.vector_score > 0.05 && (state.query.trim() || state.similarAnchorId);
      const simBadge = hasSim
        ? `<span class="paper-sim-badge" title="384-dim cosine similarity">cos ${p.vector_score.toFixed(2)}</span>`
        : "";
      let venueBadge = "";
      if (p.venue_group === "arxiv-today") {
        venueBadge = `<span class="paper-today-badge">TODAY &middot; ${esc(p.published_date || "")}</span>`;
      } else if (VENUE_BADGE_LABELS[p.venue_group]) {
        venueBadge = `<span class="paper-conf-badge">${esc(VENUE_BADGE_LABELS[p.venue_group])}</span>`;
      }
      const dateStr = p.venue_group === "arxiv-today" ? "" : `<span>${esc(p.published_date || String(p.year || ""))}</span>`;
      const citesBadge = p.citations > 0 ? `<span>${fmtNum(p.citations)} ${p.venue_group === "hf-daily" ? "upvotes" : "cites"}</span>` : "";
      const isInternal = p.url && p.url.startsWith("./");
      const targetAttr = isInternal ? "" : ' target="_blank" rel="noopener"';
      const linkLabel = p.arxiv_id ? `arXiv:${p.arxiv_id}` : isInternal ? "Read Note" : "Paper";
      const pdfLink = p.pdf_url && p.pdf_url !== p.url
        ? `<a class="paper-act-btn" href="${esc(p.pdf_url)}" target="_blank" rel="noopener">PDF &#8599;</a>`
        : "";
      const chatPrompt = encodeURIComponent(
        `Explain the core technical innovation, architecture/math, and practical trade-offs of the AI research paper "${p.title}" (${p.published_date || p.year}).`
      );
      const tagsHtml = (p.tags || []).slice(0, 5).map(
        (t) => `<button type="button" class="paper-tag" data-tag-click="${esc(t)}">#${esc(t)}</button>`
      ).join("");

      return `
        <article class="paper-card" data-paper-id="${esc(p.id)}">
          <div class="paper-card-top">
            <span class="paper-cat-badge">
              <span class="vdb-domain-dot" style="background:${esc(cat.color)}"></span>
              <span>${esc(cat.short || cat.label)}</span>
            </span>
            <div class="paper-meta-right">
              ${simBadge}
              ${venueBadge}
              ${dateStr}
              ${citesBadge}
            </div>
          </div>
          <h3 class="paper-title">
            <a href="${esc(p.url || "#")}"${targetAttr}>${esc(p.title)}</a>
          </h3>
          <div class="paper-authors">${esc(authList)}${p.venue ? ` &middot; ${esc(p.venue)}` : ""}</div>
          <p class="paper-abstract" data-abs-id="${esc(p.id)}">${esc(p.abstract)}</p>
          <div class="paper-tags">${tagsHtml}</div>
          <div class="paper-actions">
            <button type="button" class="paper-act-btn primary" data-similar-id="${esc(p.id)}" title="Find nearest papers in 384-dim vector space">Similar Vectors</button>
            <button type="button" class="paper-act-btn" data-atlas-id="${esc(p.id)}" title="Locate paper on the 2D semantic manifold">Atlas</button>
            <button type="button" class="paper-act-btn" data-expand-id="${esc(p.id)}">Abstract</button>
            ${p.url ? `<a class="paper-act-btn" href="${esc(p.url)}"${targetAttr}>${esc(linkLabel)} &#8599;</a>` : ""}
            ${pdfLink}
            <button type="button" class="paper-act-btn" data-bibtex-id="${esc(p.id)}">BibTeX</button>
            <a class="paper-act-btn" href="./chat.html?q=${chatPrompt}">Ask Chat &rarr;</a>
          </div>
        </article>
      `;
    }).join("");

    grid.querySelectorAll("[data-similar-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.similarAnchorId = btn.getAttribute("data-similar-id");
        executeSearch();
        window.scrollTo({ top: 180, behavior: "smooth" });
      });
    });

    grid.querySelectorAll("[data-atlas-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = state.byId.get(btn.getAttribute("data-atlas-id"));
        if (!p) return;
        switchTab("atlas");
        selectAtlasPaper(p.idx, true);
      });
    });

    grid.querySelectorAll("[data-expand-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pid = btn.getAttribute("data-expand-id");
        const abs = grid.querySelector(`[data-abs-id="${CSS.escape(pid)}"]`);
        if (abs) {
          const isExp = abs.classList.toggle("expanded");
          btn.textContent = isExp ? "Less" : "Abstract";
        }
      });
    });

    grid.querySelectorAll("[data-bibtex-id]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const p = state.byId.get(btn.getAttribute("data-bibtex-id"));
        if (!p) return;
        try {
          await navigator.clipboard.writeText(buildBibtex(p));
          showToast("Copied BibTeX citation to clipboard");
        } catch (_e) {
          showToast("BibTeX ready (clipboard blocked)");
        }
      });
    });

    grid.querySelectorAll("[data-tag-click]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tag = btn.getAttribute("data-tag-click") || "";
        const inp = document.getElementById("vdb-query");
        state.similarAnchorId = null;
        state.query = tag.replace(/-/g, " ");
        if (inp) inp.value = state.query;
        syncClearBtn();
        executeSearch();
      });
    });

    if (loadBtn) {
      if (state.visibleCount < total) {
        loadBtn.style.display = "inline-flex";
        loadBtn.textContent = `Load more (${total - state.visibleCount} remaining)`;
      } else {
        loadBtn.style.display = "none";
      }
    }
  }

  // ---------- 2D Embedding Atlas ----------
  function worldToScreen(x, y, W, H) {
    const scale = Math.min(W, H) * 0.44 * state.atlasZoom;
    return [W * 0.5 + x * scale + state.atlasPanX, H * 0.5 + y * scale + state.atlasPanY];
  }

  function screenToPaperIdx(mx, my, W, H) {
    let best = -1;
    let bestDist2 = 14 * 14;
    for (let i = 0; i < state.papers.length; i++) {
      const p = state.papers[i];
      const [sx, sy] = worldToScreen(p.x, p.y, W, H);
      const dx = sx - mx;
      const dy = sy - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist2) {
        bestDist2 = d2;
        best = i;
      }
    }
    return best;
  }

  function drawAtlas() {
    const canvas = document.getElementById("atlas-canvas");
    if (!canvas || !canvas.getContext || !state.papers.length) return;
    const rect = canvas.getBoundingClientRect();
    const W = Math.max(300, Math.floor(rect.width));
    const H = Math.max(360, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(W * dpr) || canvas.height !== Math.floor(H * dpr)) {
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = isLight ? "rgba(9,11,18,0.06)" : "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    const step = 64 * Math.max(0.6, Math.min(1.6, state.atlasZoom));
    const ox = (W * 0.5 + state.atlasPanX) % step;
    const oy = (H * 0.5 + state.atlasPanY) % step;
    for (let gx = ox; gx < W; gx += step) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = oy; gy < H; gy += step) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    const centroids = new Map();
    for (const p of state.papers) {
      const c = centroids.get(p.category) || { sx: 0, sy: 0, n: 0 };
      c.sx += p.x; c.sy += p.y; c.n += 1;
      centroids.set(p.category, c);
    }
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    for (const [cid, c] of centroids.entries()) {
      const dom = state.catMap.get(cid);
      if (!dom || c.n < 6) continue;
      const [lx, ly] = worldToScreen(c.sx / c.n, c.sy / c.n, W, H);
      ctx.fillStyle = isLight ? "rgba(9,11,18,0.40)" : "rgba(255,255,255,0.34)";
      ctx.fillText((dom.short || dom.label).toUpperCase(), lx, ly - 16);
    }

    if (state.atlasSelectedIdx >= 0) {
      const sel = state.papers[state.atlasSelectedIdx];
      const [x0, y0] = worldToScreen(sel.x, sel.y, W, H);
      for (const nb of (sel.neighbors || []).slice(0, 6)) {
        const target = state.papers[nb.idx];
        if (!target) continue;
        const [x1, y1] = worldToScreen(target.x, target.y, W, H);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.strokeStyle = isLight
          ? `rgba(14,165,233,${Math.max(0.25, nb.sim * 0.7)})`
          : `rgba(56,189,248,${Math.max(0.28, nb.sim * 0.8)})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    const selNeighborSet = new Set();
    if (state.atlasSelectedIdx >= 0) {
      for (const nb of (state.papers[state.atlasSelectedIdx].neighbors || []).slice(0, 6)) {
        selNeighborSet.add(nb.idx);
      }
    }

    for (let i = 0; i < state.papers.length; i++) {
      const p = state.papers[i];
      const [sx, sy] = worldToScreen(p.x, p.y, W, H);
      if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) continue;

      const dom = state.catMap.get(p.category) || { color: "#a1a6b8" };
      const inFilters = matchesFilters(p);
      const inSearch = !state.atlasHighlightIds || state.atlasHighlightIds.has(p.id);
      const isSelected = i === state.atlasSelectedIdx;
      const isHovered = i === state.atlasHoveredIdx;
      const isNeighbor = selNeighborSet.has(i);

      const baseR = 2.4 + Math.min(3.8, Math.log10(1 + (p.citations || 0)) * 0.95);
      const r = isSelected ? baseR + 3.4 : isHovered ? baseR + 2.2 : isNeighbor ? baseR + 1.2 : baseR;

      let alpha = 0.82;
      if (!inFilters || !inSearch) alpha = 0.11;
      if (isSelected || isHovered || isNeighbor) alpha = 1.0;

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = dom.color || "#ffffff";
      ctx.fill();

      if (isSelected || isHovered || isNeighbor) {
        ctx.strokeStyle = isSelected ? (isLight ? "#090b12" : "#ffffff") : "#38bdf8";
        ctx.lineWidth = isSelected ? 2.2 : 1.4;
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    }
  }

  function selectAtlasPaper(idx, centerView = false) {
    const p = state.papers[idx];
    if (!p) return;
    state.atlasSelectedIdx = idx;
    if (centerView) {
      state.atlasZoom = Math.max(state.atlasZoom, 1.5);
      const canvas = document.getElementById("atlas-canvas");
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const scale = Math.min(rect.width || 600, rect.height || 520) * 0.44 * state.atlasZoom;
        state.atlasPanX = -p.x * scale;
        state.atlasPanY = -p.y * scale;
      }
    }
    const dom = state.catMap.get(p.category) || { label: p.category, color: "#fff" };
    document.getElementById("atlas-sel-title").textContent = p.title;
    document.getElementById("atlas-sel-meta").innerHTML =
      `<span class="vdb-domain-dot" style="display:inline-block;vertical-align:middle;margin-right:0.35rem;background:${esc(dom.color)}"></span>` +
      `${esc(dom.label)} &middot; ${esc(p.published_date || String(p.year))} &middot; ${esc(p.venue || "")}`;
    document.getElementById("atlas-sel-abstract").textContent = p.abstract;

    const isInternal = p.url && p.url.startsWith("./");
    const targetAttr = isInternal ? "" : ' target="_blank" rel="noopener"';
    document.getElementById("atlas-sel-actions").innerHTML = `
      ${p.url ? `<a class="paper-act-btn primary" href="${esc(p.url)}"${targetAttr}>Open Paper &#8599;</a>` : ""}
      <button type="button" class="paper-act-btn" id="atlas-btn-sim">Similar in Library</button>
      <a class="paper-act-btn" href="./chat.html?q=${encodeURIComponent(`Explain "${p.title}" (${p.published_date || p.year}) and its core mechanism.`)}">Ask Space Chat &rarr;</a>
    `;
    document.getElementById("atlas-btn-sim")?.addEventListener("click", () => {
      state.similarAnchorId = p.id;
      switchTab("search");
      executeSearch();
    });

    const nbWrap = document.getElementById("atlas-neighbors-wrap");
    const nbList = document.getElementById("atlas-neighbor-list");
    if (nbWrap && nbList) {
      nbWrap.style.display = "block";
      nbList.innerHTML = (p.neighbors || []).slice(0, 6).map((nb) => {
        const target = state.papers[nb.idx];
        if (!target) return "";
        return `
          <button type="button" class="atlas-neighbor-item" data-jump-idx="${nb.idx}">
            <span class="atlas-neighbor-t">${esc(target.title)} <span style="color:var(--ink-muted);font-family:var(--mono);font-size:0.68rem;">(${target.published_date || target.year})</span></span>
            <span class="atlas-neighbor-sim">${nb.sim.toFixed(2)}</span>
          </button>
        `;
      }).join("");
      nbList.querySelectorAll("[data-jump-idx]").forEach((b) => {
        b.addEventListener("click", () => selectAtlasPaper(Number(b.getAttribute("data-jump-idx")), false));
      });
    }
    drawAtlas();
  }

  function setupAtlasEvents() {
    const canvas = document.getElementById("atlas-canvas");
    const tooltip = document.getElementById("atlas-tooltip");
    if (!canvas) return;

    canvas.addEventListener("mousedown", (e) => {
      state.atlasDragging = true;
      state.atlasDragStart = { x: e.clientX, y: e.clientY, px: state.atlasPanX, py: state.atlasPanY };
    });
    window.addEventListener("mouseup", () => { state.atlasDragging = false; });

    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (state.atlasDragging) {
        state.atlasPanX = state.atlasDragStart.px + (e.clientX - state.atlasDragStart.x);
        state.atlasPanY = state.atlasDragStart.py + (e.clientY - state.atlasDragStart.y);
        if (tooltip) tooltip.style.display = "none";
        drawAtlas();
        return;
      }

      const idx = screenToPaperIdx(mx, my, rect.width, rect.height);
      if (idx !== state.atlasHoveredIdx) {
        state.atlasHoveredIdx = idx;
        drawAtlas();
      }
      if (idx >= 0 && tooltip) {
        const p = state.papers[idx];
        const dom = state.catMap.get(p.category) || { short: p.category };
        tooltip.innerHTML = `<div style="font-weight:600;margin-bottom:0.2rem;">${esc(p.title)}</div>` +
          `<div style="font-family:var(--mono);font-size:0.68rem;opacity:0.82;">${esc(dom.short)} &middot; ${esc(p.published_date || String(p.year))} &middot; ${esc(p.venue || "")}</div>`;
        tooltip.style.display = "block";
        tooltip.style.left = Math.min(rect.width - 280, Math.max(12, mx + 14)) + "px";
        tooltip.style.top = Math.min(rect.height - 90, Math.max(46, my - 12)) + "px";
        canvas.style.cursor = "pointer";
      } else if (tooltip) {
        tooltip.style.display = "none";
        canvas.style.cursor = "grab";
      }
    });

    canvas.addEventListener("mouseleave", () => {
      state.atlasHoveredIdx = -1;
      if (tooltip) tooltip.style.display = "none";
      drawAtlas();
    });

    canvas.addEventListener("click", (e) => {
      if (Math.hypot(e.clientX - state.atlasDragStart.x, e.clientY - state.atlasDragStart.y) > 6) return;
      const rect = canvas.getBoundingClientRect();
      const idx = screenToPaperIdx(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
      if (idx >= 0) selectAtlasPaper(idx, false);
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.14 : 0.88;
      state.atlasZoom = Math.max(0.65, Math.min(6.0, state.atlasZoom * factor));
      drawAtlas();
    }, { passive: false });

    document.getElementById("atlas-zoom-in")?.addEventListener("click", () => {
      state.atlasZoom = Math.min(6.0, state.atlasZoom * 1.25); drawAtlas();
    });
    document.getElementById("atlas-zoom-out")?.addEventListener("click", () => {
      state.atlasZoom = Math.max(0.65, state.atlasZoom * 0.8); drawAtlas();
    });
    document.getElementById("atlas-zoom-reset")?.addEventListener("click", () => {
      state.atlasZoom = 1.0; state.atlasPanX = 0; state.atlasPanY = 0; drawAtlas();
    });
    window.addEventListener("resize", () => { if (state.atlasTabActive) drawAtlas(); });
  }

  function renderAtlasLegend() {
    const leg = document.getElementById("atlas-legend");
    if (!leg || !state.db) return;
    leg.innerHTML = (state.db.categories || []).map((c) => `
      <span class="atlas-legend-item" data-leg-cat="${esc(c.id)}">
        <span class="vdb-domain-dot" style="background:${esc(c.color)}"></span>
        <span>${esc(c.short || c.label)} (${c.count})</span>
      </span>
    `).join("");
    leg.querySelectorAll("[data-leg-cat]").forEach((el) => {
      el.addEventListener("click", () => {
        const cid = el.getAttribute("data-leg-cat");
        state.category = state.category === cid ? "all" : cid;
        renderDomainPills();
        drawAtlas();
      });
    });
  }

  function renderApiStatsTab() {
    const domWrap = document.getElementById("api-domain-bars");
    const yrWrap = document.getElementById("api-year-bars");
    if (!state.db) return;

    if (domWrap) {
      const cats = (state.db.categories || []).slice().sort((a, b) => b.count - a.count);
      const maxC = Math.max(1, ...cats.map((c) => c.count));
      domWrap.innerHTML = cats.map((c) => `
        <div class="vdb-bar-row">
          <span style="display:flex;align-items:center;gap:0.4rem;">
            <span class="vdb-domain-dot" style="background:${esc(c.color)}"></span>
            <span>${esc(c.short || c.label)}</span>
          </span>
          <div class="vdb-bar-track">
            <div class="vdb-bar-fill" style="width:${(c.count / maxC * 100).toFixed(1)}%;background:${esc(c.color)}"></div>
          </div>
          <span class="vdb-bar-val">${c.count}</span>
        </div>
      `).join("");
    }

    if (yrWrap) {
      const yrs = (state.db.years || []).filter((y) => y.year >= 2017);
      const maxY = Math.max(1, ...yrs.map((y) => y.count));
      yrWrap.innerHTML = yrs.map((y) => {
        const pct = Math.max(6, (y.count / maxY) * 100);
        return `
          <div class="vdb-year-col">
            <span class="vdb-year-cnt">${y.count}</span>
            <div class="vdb-year-bar" style="height:${pct.toFixed(1)}%"></div>
            <span class="vdb-year-lbl">${String(y.year).slice(-2)}</span>
          </div>
        `;
      }).join("");
    }
  }

  // ---------- CoreRank Discovery Tab ----------
  const DISC_LENS_INFO = {
    surprising: "Method 1 · Prior-vs-Evidence Bayesian Surprise: papers whose hype-stripped core claim violates the backbone LLM's pretrained prior (-z_prior, high NLL) while scoring high on fluff-invariant Stage-1 rigor and 4B CoreRank Ultra verification.",
    bridges: "Method 2 · Cross-Domain Mechanism Bridges: pairs from different research categories with LOW BGE topic similarity (< 0.71, bottom 25%) but HIGH alignment on 8 topic-canceled Stage-1 ACE mechanism cores + 4B transfer check.",
    contradictions: "Method 3 · Dense-Cluster Empirical Contradictions: high-BGE same-topic pairs (cosine > 0.90, disjoint authors) ranked by De-Biased Stage-2 Keep-1D Contradiction Core (-53.5% order bias) + 4B verification.",
    under_the_radar: "Method 4 · Residuals & Off-Core Outliers: (A) pre-2026 Sleeping Beauties (<50 cites, high rigor, low hype), (B) Quiet Technical Gems buried in today's arXiv firehose, and (C) Off-Core Causal Outliers carrying high energy orthogonal to the dominant 1D rigor ladder.",
  };

  function discPaperActions(p) {
    const arxivUrl = p.url || (p.arxiv_id ? `https://arxiv.org/abs/${p.arxiv_id}` : "");
    const pdfUrl = p.pdf_url || "";
    const chatQ = encodeURIComponent(`Explain the core mechanism and non-obvious findings of "${p.title}" and how we could test or extend it.`);
    return `
      <div class="paper-actions" style="margin-top:auto; padding-top:0.35rem;">
        ${arxivUrl ? `<a class="paper-act-btn primary" href="${esc(arxivUrl)}" target="_blank" rel="noopener">arXiv &#8599;</a>` : ""}
        ${pdfUrl ? `<a class="paper-act-btn" href="${esc(pdfUrl)}" target="_blank" rel="noopener">PDF</a>` : ""}
        ${p.id && state.byId.has(p.id) ? `<button type="button" class="paper-act-btn" data-disc-similar="${esc(p.id)}">Similar</button>` : ""}
        ${p.id && state.byId.has(p.id) ? `<button type="button" class="paper-act-btn" data-disc-atlas="${esc(p.id)}">Atlas</button>` : ""}
        <a class="paper-act-btn" href="./chat.html?q=${chatQ}">Investigate in Chat &rarr;</a>
      </div>
    `;
  }

  function renderDiscoverySingleCard(p, chipsHtml) {
    const dom = state.catMap.get(p.category) || { short: p.category, color: "#94a3b8" };
    return `
      <article class="disc-card">
        <div class="disc-card-top">
          <span class="paper-domain-badge"><span class="vdb-domain-dot" style="background:${esc(dom.color)}"></span>${esc(dom.short || p.category)}</span>
          <div class="disc-metrics">${chipsHtml}</div>
        </div>
        <h3 class="disc-title">${esc(p.title)}</h3>
        <div class="disc-submeta">${esc(p.venue || String(p.year || ""))} &middot; ${esc(p.published_date || String(p.year || ""))} &middot; ${fmtNum(p.citations || 0)} citations</div>
        <p class="disc-abs">${esc((p.abstract || "").slice(0, 380))}${(p.abstract || "").length > 380 ? "..." : ""}</p>
        ${discPaperActions(p)}
      </article>
    `;
  }

  function renderDiscoveryPairCard(item, mode) {
    const a = item.paper_a;
    const b = item.paper_b;
    const domA = state.catMap.get(a.category) || { short: a.category, color: "#94a3b8" };
    const domB = state.catMap.get(b.category) || { short: b.category, color: "#94a3b8" };
    let metricsHtml = "";
    let midSymbol = "&harr;";
    if (mode === "bridges") {
      const mechs = (item.shared_mechanisms || [])
        .slice(0, 2)
        .map(([m]) => `<span class="disc-chip violet">${esc(m.replace(/_/g, " "))}</span>`)
        .join("");
      metricsHtml = `
        <span class="disc-chip">BGE ${Number(item.bge_cosine).toFixed(3)} (low)</span>
        <span class="disc-chip cyan">ACE Mech ${Number(item.ace_mech_cosine).toFixed(3)}</span>
        <span class="disc-chip emerald">4B Transfer ${Number(item.transfer_4b) >= 0 ? "+" : ""}${Number(item.transfer_4b).toFixed(2)}</span>
        ${mechs}
      `;
    } else {
      midSymbol = "&ne;";
      metricsHtml = `
        <span class="disc-chip">BGE ${Number(item.bge_cosine).toFixed(3)} (dense)</span>
        <span class="disc-chip amber">4B Conflict ${Number(item.contradiction_4b) >= 0 ? "+" : ""}${Number(item.contradiction_4b).toFixed(2)}</span>
        <span class="disc-chip cyan">Stage-2 Keep-1D ${Number(item.keep1d_sym) >= 0 ? "+" : ""}${Number(item.keep1d_sym).toFixed(2)}</span>
      `;
    }

    return `
      <article class="disc-card">
        <div class="disc-card-top">
          <div class="disc-metrics">${metricsHtml}</div>
        </div>
        <div class="disc-pair-cols">
          <div class="disc-pair-side">
            <span class="paper-domain-badge"><span class="vdb-domain-dot" style="background:${esc(domA.color)}"></span>${esc(domA.short || a.category)} &middot; ${esc(String(a.year || ""))}</span>
            <h3 class="disc-title" style="font-size:0.98rem;">${esc(a.title)}</h3>
            <p class="disc-abs" style="font-size:0.81rem;">${esc((a.abstract || "").slice(0, 260))}${(a.abstract || "").length > 260 ? "..." : ""}</p>
            ${discPaperActions(a)}
          </div>
          <div class="disc-pair-mid" aria-hidden="true">${midSymbol}</div>
          <div class="disc-pair-side">
            <span class="paper-domain-badge"><span class="vdb-domain-dot" style="background:${esc(domB.color)}"></span>${esc(domB.short || b.category)} &middot; ${esc(String(b.year || ""))}</span>
            <h3 class="disc-title" style="font-size:0.98rem;">${esc(b.title)}</h3>
            <p class="disc-abs" style="font-size:0.81rem;">${esc((b.abstract || "").slice(0, 260))}${(b.abstract || "").length > 260 ? "..." : ""}</p>
            ${discPaperActions(b)}
          </div>
        </div>
      </article>
    `;
  }

  function renderDiscoveryTab() {
    const kpiWrap = document.getElementById("disc-kpis");
    const grid = document.getElementById("disc-grid");
    const explain = document.getElementById("disc-lens-explain");
    const d = state.discoveries;
    if (!grid) return;
    if (!d) {
      grid.innerHTML = `<div class="gp-card" style="grid-column:1/-1;">Loading CoreRank discovery index...</div>`;
      return;
    }

    const t = d.telemetry || {};
    if (kpiWrap) {
      const surpLift = (Number(t.m1_4b_surprise_top20 || 0) - Number(t.m1_4b_surprise_rand || 0)).toFixed(2);
      kpiWrap.innerHTML = `
        <div class="gp-kpi">
          <div class="gp-kpi-val ok">${t.stage1_scan_sec || 15.6}s</div>
          <div class="gp-kpi-lbl">Full ${fmtNum(d.corpus_count || 1715)}-Paper Scan</div>
        </div>
        <div class="gp-kpi">
          <div class="gp-kpi-val ok">+${surpLift}</div>
          <div class="gp-kpi-lbl">4B Surprise Lift vs Random</div>
        </div>
        <div class="gp-kpi">
          <div class="gp-kpi-val">&minus;${t.m3_order_bias_reduction_pct || 53.5}%</div>
          <div class="gp-kpi-lbl">Stage-2 Order Bias Cut</div>
        </div>
        <div class="gp-kpi">
          <div class="gp-kpi-val">&rho; = ${Number(t.rigor_train_rho || 0.951).toFixed(3)}</div>
          <div class="gp-kpi-lbl">Rigor Core (Deep V-Dip)</div>
        </div>
      `;
    }

    const radar = d.under_the_radar || {};
    const radarTotal =
      (radar.sleeping_beauties || []).length +
      (radar.quiet_today || []).length +
      (radar.off_core_outliers || []).length;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = String(v); };
    setTxt("disc-cnt-surprising", (d.surprising || []).length);
    setTxt("disc-cnt-bridges", (d.bridges || []).length);
    setTxt("disc-cnt-contradictions", (d.contradictions || []).length);
    setTxt("disc-cnt-radar", radarTotal);

    if (explain) explain.textContent = DISC_LENS_INFO[state.discLens] || "";

    const isPair = state.discLens === "bridges" || state.discLens === "contradictions";
    grid.classList.toggle("pair-mode", isPair);

    if (state.discLens === "surprising") {
      grid.innerHTML = (d.surprising || [])
        .map((p) =>
          renderDiscoverySingleCard(
            p,
            `
              <span class="disc-chip cyan">4B Surp ${Number(p.surprise_4b) >= 0 ? "+" : ""}${Number(p.surprise_4b).toFixed(2)}</span>
              <span class="disc-chip emerald">4B Rigor ${Number(p.rigor_4b).toFixed(2)}</span>
              <span class="disc-chip">Prior NLL ${Number(p.nll_prior).toFixed(2)}</span>
            `
          )
        )
        .join("");
    } else if (state.discLens === "bridges") {
      grid.innerHTML = (d.bridges || []).map((item) => renderDiscoveryPairCard(item, "bridges")).join("");
    } else if (state.discLens === "contradictions") {
      grid.innerHTML = (d.contradictions || []).map((item) => renderDiscoveryPairCard(item, "contradictions")).join("");
    } else {
      const blocks = [];
      if ((radar.sleeping_beauties || []).length) {
        blocks.push(`<h3 class="disc-section-heading">Pre-2026 Sleeping Beauties (&lt; 50 Citations, High Core Rigor, Low Hype)</h3>`);
        blocks.push(
          ...(radar.sleeping_beauties || []).map((p) =>
            renderDiscoverySingleCard(
              p,
              `
                <span class="disc-chip emerald">4B Rigor ${Number(p.rigor_4b).toFixed(2)}</span>
                <span class="disc-chip violet">Unconv ${Number(p.unconventional_4b).toFixed(2)}</span>
                <span class="disc-chip amber">${p.citations} cites</span>
              `
            )
          )
        );
      }
      if ((radar.quiet_today || []).length) {
        blocks.push(`<h3 class="disc-section-heading">Quiet Technical Gems Released Today (High Rigor, Lowest Hype)</h3>`);
        blocks.push(
          ...(radar.quiet_today || []).map((p) =>
            renderDiscoverySingleCard(
              p,
              `
                <span class="disc-chip emerald">Rigor z=${Number(p.z_rigor) >= 0 ? "+" : ""}${Number(p.z_rigor).toFixed(2)}</span>
                <span class="disc-chip cyan">Hype z=${Number(p.z_hype).toFixed(2)}</span>
                <span class="disc-chip">4B ${Number(p.rigor_4b).toFixed(2)}</span>
              `
            )
          )
        );
      }
      if ((radar.off_core_outliers || []).length) {
        blocks.push(`<h3 class="disc-section-heading">Off-Core Orthogonal Causal Outliers (High Energy Outside Rank-1 Core)</h3>`);
        blocks.push(
          ...(radar.off_core_outliers || []).map((p) =>
            renderDiscoverySingleCard(
              p,
              `
                <span class="disc-chip violet">Off-Core z=${Number(p.off_core_norm_z) >= 0 ? "+" : ""}${Number(p.off_core_norm_z).toFixed(2)}</span>
                <span class="disc-chip">Ratio ${Number(p.off_core_ratio).toFixed(2)}</span>
                <span class="disc-chip emerald">4B ${Number(p.rigor_4b).toFixed(2)}</span>
              `
            )
          )
        );
      }
      grid.innerHTML = blocks.join("");
    }

    grid.querySelectorAll("[data-disc-similar]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.similarAnchorId = btn.getAttribute("data-disc-similar");
        switchTab("search");
        executeSearch();
        window.scrollTo({ top: 180, behavior: "smooth" });
      });
    });
    grid.querySelectorAll("[data-disc-atlas]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = state.byId.get(btn.getAttribute("data-disc-atlas"));
        if (!p) return;
        switchTab("atlas");
        selectAtlasPaper(p.idx, true);
      });
    });
  }

  // ---------- Controls & Bootstrap ----------
  function switchTab(tabName) {
    document.querySelectorAll("[data-vdb-tab]").forEach((btn) => {
      const active = btn.getAttribute("data-vdb-tab") === tabName;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-vdb-view]").forEach((view) => {
      view.classList.toggle("active", view.getAttribute("data-vdb-view") === tabName);
    });
    state.atlasTabActive = tabName === "atlas";
    if (state.atlasTabActive) {
      requestAnimationFrame(() => {
        drawAtlas();
        if (state.atlasSelectedIdx < 0 && state.papers.length) {
          const fa = state.papers.find((p) => p.arxiv_id === "2205.14135") || state.papers[0];
          selectAtlasPaper(fa.idx, false);
        }
      });
    }
  }

  function syncClearBtn() {
    const btn = document.getElementById("vdb-clear");
    if (btn) btn.classList.toggle("visible", state.query.trim().length > 0);
  }

  function setupControls() {
    document.querySelectorAll("[data-vdb-tab]").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.getAttribute("data-vdb-tab")));
    });

    document.querySelectorAll("[data-disc-lens]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.discLens = btn.getAttribute("data-disc-lens") || "surprising";
        document.querySelectorAll("[data-disc-lens]").forEach((b) => b.classList.toggle("active", b === btn));
        renderDiscoveryTab();
      });
    });

    const input = document.getElementById("vdb-query");
    let debounceTimer = null;
    if (input) {
      input.addEventListener("input", () => {
        state.similarAnchorId = null;
        state.query = input.value;
        state._userChangedSortOnQuery = false;
        syncClearBtn();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => executeSearch(), 180);
      });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          clearTimeout(debounceTimer);
          executeSearch();
        }
      });
    }

    document.getElementById("vdb-clear")?.addEventListener("click", () => {
      state.query = "";
      state.similarAnchorId = null;
      if (input) { input.value = ""; input.focus(); }
      syncClearBtn();
      executeSearch();
    });

    document.querySelectorAll("[data-search-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.mode = btn.getAttribute("data-search-mode") || "hybrid";
        document.querySelectorAll("[data-search-mode]").forEach((b) => b.classList.toggle("active", b === btn));
        executeSearch();
      });
    });

    document.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-preset") || "";
        state.similarAnchorId = null;
        state.query = q;
        state._userChangedSortOnQuery = false;
        if (input) input.value = q;
        syncClearBtn();
        executeSearch();
      });
    });

    document.getElementById("vdb-year-filter")?.addEventListener("change", (e) => {
      state.yearRange = e.target.value;
      executeSearch();
    });

    document.getElementById("vdb-sort")?.addEventListener("change", (e) => {
      state.sort = e.target.value;
      state._userChangedSortOnQuery = true;
      executeSearch();
    });

    document.getElementById("vdb-anchor-clear")?.addEventListener("click", () => {
      state.similarAnchorId = null;
      executeSearch();
    });

    document.getElementById("btn-load-more")?.addEventListener("click", () => {
      state.visibleCount += PAGE_SIZE;
      renderResults();
    });

    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => setTimeout(() => { if (state.atlasTabActive) drawAtlas(); }, 40));
    });
  }

  async function init() {
    setupControls();
    setupAtlasEvents();

    const apiProbe = probeLiveApi();
    try {
      const [dbRes, binRes, discRes] = await Promise.all([
        fetch("./data/papers-db.json?v=3"),
        fetch("./data/papers-vectors.f16.bin?v=3").catch(() => null),
        fetch("./data/papers-discoveries.json?v=1").catch(() => null),
      ]);
      if (!dbRes.ok) throw new Error(`HTTP ${dbRes.status}`);
      const db = await dbRes.json();
      if (discRes && discRes.ok) {
        state.discoveries = await discRes.json();
      }
      state.db = db;
      state.catMap = new Map((db.categories || []).map((c) => [c.id, c]));
      state.papers = (db.papers || []).map((p, idx) => ({
        ...p,
        idx,
        _titleLow: (p.title || "").toLowerCase(),
        _absLow: (p.abstract || "").toLowerCase(),
        _tagsLow: (p.tags || []).join(" ").toLowerCase(),
        _authLow: (p.authors || []).join(" ").toLowerCase(),
        _venueLow: (p.venue || "").toLowerCase(),
      }));
      state.byId = new Map(state.papers.map((p) => [p.id, p]));

      if (binRes && binRes.ok) {
        const buf = await binRes.arrayBuffer();
        const vecs = decodeFloat16Buffer(buf);
        if (vecs.length === state.papers.length * DIM) {
          state.vectors = vecs;
        }
      }

      renderTopStats();
      renderVenuePills();
      renderDomainPills();
      renderAtlasLegend();
      renderApiStatsTab();
      renderDiscoveryTab();

      const params = new URLSearchParams(window.location.search);
      const qParam = (params.get("q") || "").trim();
      const catParam = (params.get("cat") || "").trim();
      const venueParam = (params.get("venue") || "").trim();
      const tabParam = (params.get("tab") || "").trim();
      if (venueParam) {
        state.venue = venueParam;
        renderVenuePills();
      }
      if (catParam && state.catMap.has(catParam)) {
        state.category = catParam;
        renderDomainPills();
      }
      if (qParam) {
        state.query = qParam;
        const inp = document.getElementById("vdb-query");
        if (inp) inp.value = qParam;
        syncClearBtn();
      }

      await apiProbe;
      await executeSearch();
      if (tabParam === "discover" || tabParam === "atlas" || tabParam === "api") switchTab(tabParam);
    } catch (err) {
      const grid = document.getElementById("papers-grid");
      if (grid) {
        grid.innerHTML = `<div class="gp-card" style="grid-column:1/-1;">Failed to load research database (${esc(err.message)}).</div>`;
      }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
