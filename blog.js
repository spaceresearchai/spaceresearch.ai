/* Space AI — High-Volume Daily-Ordered Research Feed Controller */
(() => {
  const DAY_COLLAPSE_LIMIT = 4;
  const PAGE_LIMIT = 40;

  const state = {
    posts: [],
    tracks: [],
    query: "",
    track: "all",
    origin: "all",      // all | lab | autonomous-gpu
    dayFilter: "all",   // all | YYYY-MM-DD
    sort: "newest",     // newest | oldest | track
    view: "cards",      // cards | compact
    groupByDay: true,
    expandedDays: new Set(),
    collapsedDays: new Set(),
    visibleLimit: PAGE_LIMIT,
  };

  function esc(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function parseTs(iso) {
    const t = Date.parse(iso || "");
    return Number.isFinite(t) ? t : 0;
  }

  function fmtTimeHHMM(iso) {
    const m = String(iso || "").match(/T(\d{2}:\d{2})/);
    return m ? m[1] : "00:00";
  }

  function fmtDayHeader(dateStr) {
    if (!dateStr || !/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) return dateStr || "Undated";
    const [y, mo, d] = dateStr.split("-").map(Number);
    const dt = new Date(Date.UTC(y, mo - 1, d));
    const wd = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][dt.getUTCDay()];
    const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][mo - 1];
    return `${wd}, ${mon} ${d}, ${y}`;
  }

  function fmtShortDay(dateStr) {
    if (!dateStr || !/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) return dateStr || "";
    const [, mo, d] = dateStr.split("-").map(Number);
    const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][mo - 1];
    return `${mon} ${d}`;
  }

  function comparePosts(a, b, mode) {
    if (mode === "oldest") {
      const td = parseTs(a.published_at) - parseTs(b.published_at);
      return td !== 0 ? td : (a.seq || 0) - (b.seq || 0);
    }
    if (mode === "track") {
      const tc = String(a.track || "").localeCompare(String(b.track || ""));
      if (tc !== 0) return tc;
    }
    // Default: strict newest-first by full ISO timestamp, tie-broken by monotonic seq DESC
    const td = parseTs(b.published_at) - parseTs(a.published_at);
    return td !== 0 ? td : (b.seq || 0) - (a.seq || 0);
  }

  function getFilteredPosts() {
    const q = state.query.trim().toLowerCase();
    return state.posts
      .filter((p) => {
        if (state.track !== "all" && p.track !== state.track) return false;
        if (state.origin !== "all" && p.origin !== state.origin) return false;
        if (state.dayFilter !== "all" && p.published_date !== state.dayFilter) return false;
        if (q) {
          const hay = [
            p.note_id,
            p.title,
            p.description,
            p.category_label,
            p.published_date,
            ...(p.tags || []),
          ]
            .join(" ")
            .toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      })
      .sort((a, b) => comparePosts(a, b, state.sort));
  }

  function renderCard(p) {
    const timeStr = fmtTimeHHMM(p.published_at);
    const originHtml =
      p.origin === "autonomous-gpu"
        ? `<span class="blog-origin-badge auto">Autonomous GPU Lab</span>`
        : "";
    const metricHtml = p.experiment_metrics
      ? `<div class="blog-metric-chip">${esc(p.experiment_metrics)}</div>`
      : "";
    const paperHtml = p.paper_href
      ? `<a class="blog-paper-link" href="${esc(p.paper_href)}" onclick="event.stopPropagation()">Full Technical Paper &rarr;</a>`
      : "";

    return `
      <article class="blog-card" data-cluster="${Number(p.cluster || 0)}" style="cursor:pointer;" onclick="window.location.href='${esc(p.href)}'">
        <div class="blog-meta">
          <span class="blog-cat"><span class="spec-dot ${esc(p.dot || "cyan")}" style="margin-right:0.36rem;"></span>${esc(p.note_id)} &middot; ${esc(p.category_label)}</span>
          ${originHtml}
          <span class="blog-time-tag">${esc(p.published_date)} &middot; ${esc(timeStr)} &middot; ${Number(p.read_minutes || 5)} min</span>
        </div>
        <h2 class="blog-card-title"><a href="${esc(p.href)}" style="color:inherit;text-decoration:none;">${esc(p.title)}</a></h2>
        ${metricHtml}
        <p class="blog-card-desc">${esc(p.description)}</p>
        <div class="blog-card-foot">
          <a class="blog-read-more" href="${esc(p.href)}">Read research note &rarr;</a>
          ${paperHtml}
        </div>
      </article>
    `;
  }

  function renderCompactRow(p) {
    const timeStr = fmtTimeHHMM(p.published_at);
    const autoBadge =
      p.origin === "autonomous-gpu"
        ? `<span class="blog-origin-badge auto">Auto GPU</span>`
        : "";
    const paperTag = p.paper_href
      ? `<a class="blog-paper-link" href="${esc(p.paper_href)}" onclick="event.stopPropagation()">Paper</a>`
      : "";
    return `
      <div class="blog-compact-row">
        <span class="blog-compact-time">${esc(timeStr)}</span>
        <span class="blog-compact-seq">${esc(p.note_id)}</span>
        <div class="blog-compact-main">
          <a class="blog-compact-title" href="${esc(p.href)}">${esc(p.title)}</a>
          <div class="blog-compact-sub">
            <span><span class="spec-dot ${esc(p.dot || "cyan")}" style="margin-right:0.25rem;"></span>${esc(p.category_label)}</span>
            ${p.experiment_metrics ? `<span>&middot; ${esc(p.experiment_metrics)}</span>` : ""}
          </div>
        </div>
        <div class="blog-compact-right">
          ${autoBadge}
          ${paperTag}
          <span class="blog-compact-time">${Number(p.read_minutes || 5)}m</span>
        </div>
      </div>
    `;
  }

  function renderDayJumps() {
    const wrap = document.getElementById("blog-day-jumps");
    if (!wrap) return;
    const counts = new Map();
    for (const p of state.posts) {
      const d = p.published_date || "unknown";
      counts.set(d, (counts.get(d) || 0) + 1);
    }
    const days = Array.from(counts.entries()).sort((a, b) => b[0].localeCompare(a[0]));
    const chips = [
      `<span class="blog-jump-lbl">Jump by Day:</span>`,
      `<button type="button" class="blog-day-chip ${state.dayFilter === "all" ? "active" : ""}" data-day-jump="all">All (${state.posts.length})</button>`,
      ...days.slice(0, 14).map(
        ([d, cnt]) =>
          `<button type="button" class="blog-day-chip ${state.dayFilter === d ? "active" : ""}" data-day-jump="${esc(d)}">${esc(fmtShortDay(d))} (${cnt})</button>`
      ),
    ];
    wrap.innerHTML = chips.join("");
    wrap.querySelectorAll("[data-day-jump]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.dayFilter = btn.getAttribute("data-day-jump") || "all";
        state.visibleLimit = PAGE_LIMIT;
        renderDayJumps();
        renderFeed();
      });
    });
  }

  function renderTrackPills() {
    const wrap = document.getElementById("blog-track-pills");
    if (!wrap) return;
    const counts = new Map();
    for (const p of state.posts) counts.set(p.track, (counts.get(p.track) || 0) + 1);
    const items = [
      { id: "all", label: `All (${state.posts.length})`, dot: null },
      ...state.tracks.map((t) => ({ id: t.id, label: `${t.label} (${counts.get(t.id) || 0})`, dot: t.dot })),
    ];
    wrap.innerHTML = items
      .map(
        (it) => `
          <button type="button" class="blog-pill ${state.track === it.id ? "active" : ""}" data-track-pill="${esc(it.id)}">
            ${it.dot ? `<span class="spec-dot ${esc(it.dot)}"></span>` : ""}
            <span>${esc(it.label)}</span>
          </button>
        `
      )
      .join("");
    wrap.querySelectorAll("[data-track-pill]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.track = btn.getAttribute("data-track-pill") || "all";
        state.visibleLimit = PAGE_LIMIT;
        renderTrackPills();
        renderFeed();
      });
    });
  }

  function renderFeed() {
    const feed = document.getElementById("blog-feed");
    const summary = document.getElementById("blog-summary-bar");
    const moreWrap = document.getElementById("blog-more-wrap");
    if (!feed) return;

    const allFiltered = getFilteredPosts();
    const total = allFiltered.length;
    const latestDate = state.posts.reduce((acc, p) => (p.published_date > acc ? p.published_date : acc), "");

    if (summary) {
      const autoCnt = allFiltered.filter((p) => p.origin === "autonomous-gpu").length;
      const dayCnt = new Set(allFiltered.map((p) => p.published_date)).size;
      summary.innerHTML = `
        <span>Showing <strong>${Math.min(total, state.visibleLimit)}</strong> of <strong>${total}</strong> notes across <strong>${dayCnt}</strong> day${dayCnt === 1 ? "" : "s"}${autoCnt ? ` &middot; ${autoCnt} autonomous GPU` : ""}</span>
        <span>Ordered strictly by timestamp + sequence (${state.sort === "oldest" ? "oldest first" : state.sort === "track" ? "by track" : "newest first"})</span>
      `;
    }

    if (total === 0) {
      feed.innerHTML = `<div class="blog-card" style="text-align:center;"><p class="blog-card-desc" style="margin:0;">No research notes match your current filter.</p></div>`;
      if (moreWrap) moreWrap.innerHTML = "";
      return;
    }

    const slice = allFiltered.slice(0, state.visibleLimit);
    const shouldGroup = state.groupByDay && state.sort !== "track";

    if (!shouldGroup) {
      if (state.view === "compact") {
        feed.innerHTML = `<div class="blog-compact-list">${slice.map(renderCompactRow).join("")}</div>`;
      } else {
        feed.innerHTML = `<div class="blog-list">${slice.map(renderCard).join("")}</div>`;
      }
    } else {
      // Group slice strictly preserving day order
      const groups = [];
      const byDate = new Map();
      for (const p of slice) {
        const d = p.published_date || "undated";
        if (!byDate.has(d)) {
          const arr = [];
          byDate.set(d, arr);
          groups.push([d, arr]);
        }
        byDate.get(d).push(p);
      }

      feed.innerHTML = groups
        .map(([day, items]) => {
          const isToday = day === latestDate;
          const isCollapsed = state.collapsedDays.has(day);
          const isExpanded = state.expandedDays.has(day) || state.dayFilter === day || state.view === "compact";
          const times = items.map((p) => fmtTimeHHMM(p.published_at)).sort();
          const spanTxt = times.length > 1 ? `${times[0]} – ${times[times.length - 1]}` : times[0];
          const shownItems = isCollapsed
            ? []
            : !isExpanded && items.length > DAY_COLLAPSE_LIMIT
            ? items.slice(0, DAY_COLLAPSE_LIMIT)
            : items;
          const hiddenCount = isCollapsed ? items.length : items.length - shownItems.length;

          const bodyHtml = isCollapsed
            ? ""
            : state.view === "compact"
            ? `<div class="blog-compact-list">${shownItems.map(renderCompactRow).join("")}</div>`
            : `<div class="blog-list">${shownItems.map(renderCard).join("")}</div>`;

          const expandBtn =
            !isCollapsed && hiddenCount > 0
              ? `<button type="button" class="blog-expand-day-btn" data-expand-day="${esc(day)}">+ Show ${hiddenCount} more note${hiddenCount === 1 ? "" : "s"} from ${esc(fmtShortDay(day))} (or switch to Compact Index)</button>`
              : "";

          return `
            <section class="blog-day-group" data-day-section="${esc(day)}">
              <div class="blog-day-header">
                <div class="blog-day-title-wrap">
                  <span class="blog-day-date">${esc(fmtDayHeader(day))}</span>
                  <span class="blog-day-badge ${isToday ? "today" : ""}">${isToday ? "Latest · " : ""}${items.length} note${items.length === 1 ? "" : "s"}</span>
                  <span class="blog-day-span">${esc(spanTxt)}</span>
                </div>
                <button type="button" class="blog-day-toggle" data-toggle-day="${esc(day)}">${isCollapsed ? `Expand (${items.length})` : "Collapse"}</button>
              </div>
              ${bodyHtml}
              ${expandBtn}
            </section>
          `;
        })
        .join("");

      feed.querySelectorAll("[data-expand-day]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.expandedDays.add(btn.getAttribute("data-expand-day"));
          renderFeed();
        });
      });
      feed.querySelectorAll("[data-toggle-day]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const d = btn.getAttribute("data-toggle-day");
          if (state.collapsedDays.has(d)) state.collapsedDays.delete(d);
          else state.collapsedDays.add(d);
          renderFeed();
        });
      });
    }

    if (moreWrap) {
      if (total > state.visibleLimit) {
        moreWrap.innerHTML = `<button type="button" id="blog-load-more-btn" class="btn secondary sm">Load more notes (${total - state.visibleLimit} remaining)</button>`;
        document.getElementById("blog-load-more-btn")?.addEventListener("click", () => {
          state.visibleLimit += PAGE_LIMIT;
          renderFeed();
        });
      } else {
        moreWrap.innerHTML = "";
      }
    }
  }

  function setupControls() {
    document.getElementById("blog-search")?.addEventListener("input", (e) => {
      state.query = e.target.value || "";
      state.visibleLimit = PAGE_LIMIT;
      renderFeed();
    });

    document.getElementById("blog-sort")?.addEventListener("change", (e) => {
      state.sort = e.target.value || "newest";
      renderFeed();
    });

    document.querySelectorAll("[data-blog-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.view = btn.getAttribute("data-blog-view") || "cards";
        document.querySelectorAll("[data-blog-view]").forEach((b) => b.classList.toggle("active", b === btn));
        renderFeed();
      });
    });

    document.querySelectorAll("[data-blog-origin]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.origin = btn.getAttribute("data-blog-origin") || "all";
        document.querySelectorAll("[data-blog-origin]").forEach((b) => b.classList.toggle("active", b === btn));
        state.visibleLimit = PAGE_LIMIT;
        renderFeed();
      });
    });

    document.getElementById("blog-group-day-toggle")?.addEventListener("click", (e) => {
      state.groupByDay = !state.groupByDay;
      e.currentTarget.classList.toggle("active", state.groupByDay);
      renderFeed();
    });
  }

  async function init() {
    setupControls();
    try {
      const res = await fetch("./data/blog-posts.json?v=" + Date.now());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const manifest = await res.json();
      state.posts = (manifest.posts || []).slice();
      state.tracks = manifest.tracks || [];

      // Auto-switch to compact view if any single day has >= 10 notes!
      const dayCounts = new Map();
      for (const p of state.posts) {
        dayCounts.set(p.published_date, (dayCounts.get(p.published_date) || 0) + 1);
      }
      const maxPerDay = Math.max(0, ...dayCounts.values());
      if (maxPerDay >= 10) {
        state.view = "compact";
        document.querySelectorAll("[data-blog-view]").forEach((b) =>
          b.classList.toggle("active", b.getAttribute("data-blog-view") === "compact")
        );
      }

      renderTrackPills();
      renderDayJumps();
      renderFeed();
    } catch (err) {
      // Static HTML fallback inside #blog-feed remains visible if fetch fails
      console.warn("Blog manifest hydrate failed, using static fallback:", err);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
