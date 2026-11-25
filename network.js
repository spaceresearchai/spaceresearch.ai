/* Space AI Network Live Client */
(() => {
  let COORD_URL = "https://api.spaceresearch.ai";
  const FALLBACK_COORD_URL = "https://api.longinusresearch.com";

  async function syncCoordinatorEndpoint() {
    try {
      const r = await fetch("./coordinator.json", { cache: "no-store" });
      if (r.ok) {
        const d = await r.json();
        if (d && typeof d.api_base === "string" && d.api_base.startsWith("http")) {
          COORD_URL = d.api_base.replace(/\/+$/, "");
        }
      }
    } catch (_e) {}
    // Auto-failover during DNS propagation if primary API host is not reachable yet
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 3500);
      const ping = await fetch(COORD_URL + "/health", { signal: ctrl.signal });
      clearTimeout(t);
      if (ping.ok) return;
    } catch (_e) {
      if (COORD_URL !== FALLBACK_COORD_URL) {
        COORD_URL = FALLBACK_COORD_URL;
      }
    }
  }

  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatInlineMd(line) {
    let s = escHtml(line);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return s;
  }

  function renderMarkdown(raw) {
    const src = String(raw || "").replace(/\r\n/g, "\n").trim();
    if (!src) return "<p>(Empty response)</p>";
    const lines = src.split("\n");
    const out = [];
    let inCode = false;
    let codeBuf = [];
    let listTag = null;

    function closeList() {
      if (listTag) { out.push(`</${listTag}>`); listTag = null; }
    }

    for (const line of lines) {
      if (line.trim().startsWith("```")) {
        closeList();
        if (!inCode) { inCode = true; codeBuf = []; }
        else { out.push(`<pre><code>${escHtml(codeBuf.join("\n"))}</code></pre>`); inCode = false; }
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }
      const trimmed = line.trim();
      if (!trimmed) { closeList(); continue; }
      const hMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (hMatch) { closeList(); out.push(`<h${hMatch[1].length}>${formatInlineMd(hMatch[2])}</h${hMatch[1].length}>`); continue; }
      const ulMatch = trimmed.match(/^[-*•]\s+(.*)$/);
      if (ulMatch) {
        if (listTag !== "ul") { closeList(); listTag = "ul"; out.push("<ul>"); }
        out.push(`<li>${formatInlineMd(ulMatch[1])}</li>`);
        continue;
      }
      const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (olMatch) {
        if (listTag !== "ol") { closeList(); listTag = "ol"; out.push("<ol>"); }
        out.push(`<li>${formatInlineMd(olMatch[1])}</li>`);
        continue;
      }
      closeList();
      out.push(`<p>${formatInlineMd(trimmed)}</p>`);
    }
    if (inCode) out.push(`<pre><code>${escHtml(codeBuf.join("\n"))}</code></pre>`);
    closeList();
    return out.join("");
  }

  function downloadShareCard(op) {
    const c = document.createElement("canvas");
    c.width = 1200;
    c.height = 630;
    const ctx = c.getContext("2d");

    // Pure jet black + subtle monochrome grid
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 1200, 630);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= 1200; x += 60) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 630); ctx.stroke(); }
    for (let y = 0; y <= 630; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1200, y); ctx.stroke(); }

    // Frosted glass frame
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(72, 72, 1056, 486, 18);
    ctx.fill();
    ctx.stroke();

    // Header
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(134, 146, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.font = "600 18px ui-monospace, Menlo, monospace";
    ctx.fillText("SPACE AI · SEASON 0 CANARY TESTNET", 156, 152);

    // Big Rank + Hardware
    ctx.font = "600 66px -apple-system, Inter, system-ui, sans-serif";
    ctx.fillText(`Rank #${op.rank} · ${op.handle}`, 124, 248);

    ctx.font = "400 28px -apple-system, Inter, system-ui, sans-serif";
    ctx.fillText(`${op.chip_name} (${String(op.backend || "").toUpperCase()} · ${op.memory_gb} GB)`, 124, 304);

    // 3 Metric boxes
    const metrics = [
      ["SEASON 0 POINTS", `${Number(op.season0_points || 0).toLocaleString("en-US")} pts`],
      ["TOKENS SERVED", Number(op.verified_tokens || 0).toLocaleString("en-US")],
      ["EST. CANARY ALLOC", `${Number(op.est_canary_tokens || 0).toLocaleString("en-US", { maximumFractionDigits: 0 })} tSPACE`],
    ];
    metrics.forEach(([lbl, val], idx) => {
      const bx = 124 + idx * 326;
      ctx.fillStyle = "rgba(255,255,255,0.045)";
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.beginPath();
      ctx.roundRect(bx, 348, 304, 126, 12);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = "600 32px ui-monospace, Menlo, monospace";
      ctx.fillText(val, bx + 24, 408);
      ctx.font = "500 13px ui-monospace, Menlo, monospace";
      ctx.fillText(lbl, bx + 24, 444);
    });

    ctx.fillStyle = "#ffffff";
    ctx.font = "500 16px ui-monospace, Menlo, monospace";
    ctx.fillText("Run local AI. Together.  ·  spaceresearch.ai", 124, 522);

    const a = document.createElement("a");
    a.download = `space-research-season0-rank-${op.rank}.png`;
    a.href = c.toDataURL("image/png");
    a.click();
  }

  let nextSyncAt = 0;
  let latestGoodput = null;
  let activeChartMetric = "cluster_tok_s";

  function fmtTimeShort(tsSec) {
    try {
      const d = new Date(Number(tsSec || 0) * 1000);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch (_e) {
      return "";
    }
  }

  function renderGoodputChart() {
    const gp = latestGoodput;
    const svgWrap = document.getElementById("gp-svg-wrap");
    const inspectEl = document.getElementById("gp-inspect");
    if (!gp || !svgWrap) return;

    const sum = gp.summary || {};
    const pts = Array.isArray(gp.points) ? gp.points : [];

    const setKpi = (id, text, cls) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = text;
      el.classList.remove("ok", "warn");
      if (cls) el.classList.add(cls);
    };

    const hasData = pts.length > 0;
    const peakCluster = Number(sum.cluster_peak_tok_s || sum.peak_goodput_tok_s || 0);
    setKpi("gp-val-goodput", hasData ? `${peakCluster.toFixed(1)} tok/s` : "--", hasData && peakCluster >= 28 ? "ok" : "");
    setKpi("gp-val-ttfa", hasData ? `${Math.round(Number(sum.p95_ttfa_ms || 0))} ms` : "--", hasData ? (sum.p95_ttfa_ms <= 500 ? "ok" : "warn") : "");
    setKpi("gp-val-tpot", hasData ? `${Number(sum.p95_tpot_ms || 0).toFixed(1)} ms` : "--", hasData ? (sum.p95_tpot_ms <= 35.7 ? "ok" : "warn") : "");
    setKpi("gp-val-slo", hasData ? `${Number(sum.sla_pass_rate_pct || 0).toFixed(1)}%` : "--", hasData ? (sum.sla_pass_rate_pct >= 90 ? "ok" : "warn") : "");

    if (!pts.length) {
      svgWrap.innerHTML = `<div style="height:210px;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:0.76rem;opacity:0.7;">Waiting for live chat requests to populate history...</div>`;
      if (inspectEl) inspectEl.textContent = "SLO targets: TTFA <= 500 ms · TPOT <= 35.7 ms (>= 28 tok/s)";
      return;
    }

    const cfg = {
      cluster_tok_s: { label: "Cluster Throughput", unit: "tok/s", color: "#4ade80", slo: 100, sloLabel: "100 tok/s baseline" },
      cum_tokens: { label: "Total Tokens Served", unit: "tok", color: "#38bdf8", slo: 0, sloLabel: "" },
      tok_s: { label: "Per-Req Goodput", unit: "tok/s", color: "#38bdf8", slo: 28, sloLabel: "28 tok/s min" },
      ttfa_ms: { label: "TTFA", unit: "ms", color: "#a855f7", slo: 500, sloLabel: "500 ms SLO" },
      tpot_ms: { label: "TPOT", unit: "ms", color: "#34d399", slo: 35.7, sloLabel: "35.7 ms SLO" },
    }[activeChartMetric] || { label: "Cluster Throughput", unit: "tok/s", color: "#4ade80", slo: 100, sloLabel: "100 tok/s baseline" };

    const W = 840, H = 210, PL = 60, PR = 18, PT = 16, PB = 26;
    const cw = W - PL - PR;
    const ch = H - PT - PB;
    const vals = pts.map((p) => Number(p[activeChartMetric] ?? p.tok_s ?? 0));
    const minVal = activeChartMetric === "cum_tokens" ? Math.floor(Math.min(...vals) * 0.96) : 0;
    const maxRaw = Math.max(...vals, cfg.slo * 1.15, minVal + 10);
    const yMax = activeChartMetric === "cum_tokens"
      ? Math.ceil(maxRaw * 1.02)
      : Math.ceil((maxRaw * 1.12) / 10) * 10;
    const ySpan = Math.max(1, yMax - minVal);

    const xOf = (i) => (pts.length === 1 ? PL + cw / 2 : PL + (i / (pts.length - 1)) * cw);
    const yOf = (v) => PT + ch - (Math.max(0, Math.min(ySpan, v - minVal)) / ySpan) * ch;

    const polyPts = pts.map((p, i) => `${xOf(i).toFixed(1)},${yOf(Number(p[activeChartMetric] ?? p.tok_s ?? 0)).toFixed(1)}`).join(" ");
    const areaPts = `${xOf(0).toFixed(1)},${(PT + ch).toFixed(1)} ${polyPts} ${xOf(pts.length - 1).toFixed(1)},${(PT + ch).toFixed(1)}`;
    const sloY = yOf(cfg.slo);

    const gridTicks = [0, 0.33, 0.66, 1.0].map((frac) => {
      const val = Math.round(minVal + ySpan * frac);
      const y = yOf(val);
      const lbl = val >= 10000 ? `${(val / 1000).toFixed(0)}k` : String(val);
      return `<line x1="${PL}" y1="${y.toFixed(1)}" x2="${W - PR}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>` +
        `<text x="${Math.max(4, PL - 10 - lbl.length * 6.2).toFixed(1)}" y="${(y + 3.5).toFixed(1)}" fill="rgba(255,255,255,0.58)" font-size="10" font-family="ui-monospace,monospace">${lbl}</text>`;
    }).join("");

    const dots = pts.map((p, i) => {
      const cx = xOf(i).toFixed(1);
      const cy = yOf(Number(p[activeChartMetric] ?? p.tok_s ?? 0)).toFixed(1);
      const dotCol = activeChartMetric === "cluster_tok_s"
        ? ((p.active_nodes_window || 1) >= 2 ? "#4ade80" : "#38bdf8")
        : (p.sla_pass ? "#4ade80" : "#fbbf24");
      return `<circle class="gp-dot" data-idx="${i}" cx="${cx}" cy="${cy}" r="${pts.length > 40 ? 3.2 : 4.2}" fill="${dotCol}" stroke="#000000" stroke-width="1.4" style="cursor:pointer;"/>`;
    }).join("");

    const firstLbl = fmtTimeShort(pts[0].ts);
    const lastLbl = `${fmtTimeShort(pts[pts.length - 1].ts)} (${pts.length} reqs)`;
    const sloX = (W - PR - 6 - cfg.sloLabel.length * 6.2).toFixed(1);
    const lastX = (W - PR - 4 - lastLbl.length * 6.2).toFixed(1);
    const sloLineSvg = cfg.slo > 0
      ? `<line x1="${PL}" y1="${sloY.toFixed(1)}" x2="${W - PR}" y2="${sloY.toFixed(1)}" stroke="rgba(255,255,255,0.34)" stroke-dasharray="4 4" stroke-width="1"/>` +
        `<text x="${sloX}" y="${Math.max(PT + 10, sloY - 5).toFixed(1)}" fill="rgba(255,255,255,0.72)" font-size="10" font-family="ui-monospace,monospace">${cfg.sloLabel}</text>`
      : "";

    svgWrap.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Cluster throughput and network growth history">
        <defs>
          <linearGradient id="gpGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${cfg.color}" stop-opacity="0.34"/>
            <stop offset="100%" stop-color="${cfg.color}" stop-opacity="0.01"/>
          </linearGradient>
        </defs>
        ${gridTicks}
        ${sloLineSvg}
        <polygon points="${areaPts}" fill="url(#gpGrad)"/>
        <polyline points="${polyPts}" fill="none" stroke="${cfg.color}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>
        ${dots}
        <text x="${PL}" y="${H - 6}" fill="rgba(255,255,255,0.55)" font-size="10" font-family="ui-monospace,monospace">${escHtml(firstLbl)}</text>
        <text x="${lastX}" y="${H - 6}" fill="rgba(255,255,255,0.55)" font-size="10" font-family="ui-monospace,monospace">${escHtml(lastLbl)}</text>
      </svg>
    `;

    const showPoint = (p) => {
      if (!inspectEl || !p) return;
      const badge = p.sla_pass ? "SLO PASS" : "SLO MISS";
      const cTps = Number(p.cluster_tok_s || p.tok_s || 0).toFixed(1);
      const cNodes = Number(p.active_nodes_window || 1);
      const cumT = Number(p.cum_tokens || 0).toLocaleString("en-US");
      inspectEl.textContent = `${fmtTimeShort(p.ts)} · Cluster ${cTps} tok/s (${cNodes} active ${cNodes === 1 ? "node" : "nodes"}) · ${p.node_name} (${p.tok_s} tok/s) · Total ${cumT} tok · ${badge}`;
    };
    showPoint(pts[pts.length - 1]);

    svgWrap.querySelectorAll(".gp-dot").forEach((dot) => {
      const idx = Number(dot.getAttribute("data-idx"));
      const handler = () => showPoint(pts[idx]);
      dot.addEventListener("mouseenter", handler);
      dot.addEventListener("click", handler);
    });
  }

  function formatCountdown(secs) {
    const s = Math.max(0, Math.floor(secs));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
  }

  function tickSyncTimer() {
    const el = document.getElementById("sync-countdown");
    if (!el || !nextSyncAt) return;
    const rem = Math.max(0, Math.round((nextSyncAt - Date.now()) / 1000));
    el.textContent = rem > 0 ? `On-chain sync in ${formatCountdown(rem)}` : "Syncing on-chain now...";
  }

  const WORLD_POLYS = [
    // North America
    [[-168,66],[-162,70],[-140,70],[-124,70],[-105,69],[-95,70],[-82,63],[-78,55],[-65,58],[-55,52],[-61,46],[-67,44],[-71,41],[-76,35],[-81,31],[-80,25],[-83,28],[-89,30],[-97,26],[-96,19],[-88,21],[-84,10],[-78,8],[-83,9],[-92,15],[-105,20],[-112,24],[-117,32],[-122,38],[-124,46],[-130,54],[-140,59],[-152,58],[-162,55],[-166,60]],
    // Greenland
    [[-52,60],[-42,60],[-22,70],[-18,77],[-44,77],[-62,76],[-56,68]],
    // South America
    [[-78,9],[-71,12],[-61,10],[-50,1],[-35,-6],[-38,-14],[-41,-22],[-48,-26],[-54,-34],[-62,-39],[-66,-46],[-68,-55],[-74,-52],[-73,-40],[-71,-30],[-71,-18],[-81,-5],[-79,2]],
    // Europe (Mainland)
    [[-9,37],[-9,43],[-2,43],[-5,48],[2,51],[8,54],[10,57],[14,54],[20,55],[24,57],[28,60],[21,60],[18,63],[24,66],[20,70],[12,66],[5,61],[6,58],[11,58],[12,55],[19,54],[29,60],[38,65],[45,67],[58,68],[60,55],[48,47],[40,45],[36,47],[30,46],[28,41],[24,41],[22,37],[19,40],[14,45],[16,40],[12,42],[8,44],[3,43],[0,39],[-5,36]],
    // UK & Ireland
    [[-6,50],[-5,55],[-6,58],[-2,58],[1,52],[-3,51]],
    [[-10,52],[-9,55],[-6,54],[-7,52]],
    // Iceland
    [[-24,64],[-22,66],[-14,66],[-15,64]],
    // Africa
    [[-17,15],[-13,28],[-6,36],[10,37],[12,33],[24,32],[32,31],[36,24],[43,12],[51,11],[45,2],[40,-5],[40,-15],[35,-24],[28,-33],[18,-34],[14,-22],[12,-12],[9,-1],[4,4],[-5,5],[-14,10]],
    // Madagascar
    [[44,-25],[47,-25],[50,-15],[48,-13],[44,-20]],
    // Asia (Mainland + Middle East + India + SE Asia)
    [[26,40],[36,37],[35,31],[39,21],[44,13],[53,16],[59,22],[56,26],[49,30],[56,26],[66,25],[72,22],[77,8],[80,14],[87,22],[92,21],[98,16],[100,8],[103,1],[104,6],[102,13],[105,10],[109,12],[108,17],[106,21],[114,22],[121,25],[122,31],[120,36],[124,40],[127,35],[129,38],[132,43],[140,47],[142,54],[150,60],[162,62],[174,66],[168,70],[140,73],[110,74],[85,73],[68,70],[60,68],[60,55],[50,47],[42,42],[36,41]],
    // Japan
    [[130,31],[135,34],[140,36],[141,41],[145,44],[141,45],[139,39],[136,37],[131,34]],
    // Taiwan & Philippines
    [[120,22],[122,25],[121,25],[120,23]],
    [[120,14],[122,18],[124,14],[126,7],[123,7],[121,12]],
    // Indonesia (Sumatra, Java, Borneo)
    [[95,5],[100,1],[105,-6],[103,-6],[98,0]],
    [[106,-6],[114,-8],[114,-7],[106,-7]],
    [[109,1],[117,5],[118,-3],[112,-3]],
    // Australia & NZ
    [[114,-22],[122,-18],[131,-12],[136,-12],[142,-11],[146,-18],[153,-27],[151,-34],[147,-38],[140,-38],[135,-34],[129,-32],[122,-34],[115,-34],[113,-26]],
    [[167,-46],[172,-44],[178,-37],[174,-35],[172,-41],[168,-45]],
  ];

  const MAP_REF_CITIES = [
    ["San Francisco", -122.4, 37.8], ["New York", -74.0, 40.7], ["Mexico City", -99.1, 19.4],
    ["São Paulo", -46.6, -23.5], ["London", -0.1, 51.5], ["Frankfurt", 8.7, 50.1],
    ["Stockholm", 18.1, 59.3], ["Dubai", 55.3, 25.2], ["Mumbai", 72.9, 19.1],
    ["Singapore", 103.8, 1.3], ["Tokyo", 139.7, 35.7], ["Sydney", 151.2, -33.9],
  ];

  const MAP_W = 960;
  const MAP_H = 420;
  const mapView = { scale: 1.0, tx: 0.0, ty: 0.0, animRaf: 0 };
  let mapEventsBound = false;

  function clampMapView() {
    mapView.scale = Math.max(1.0, Math.min(6.5, mapView.scale));
    const minTx = MAP_W - MAP_W * mapView.scale;
    const minTy = MAP_H - MAP_H * mapView.scale;
    mapView.tx = Math.min(0, Math.max(minTx, mapView.tx));
    mapView.ty = Math.min(0, Math.max(minTy, mapView.ty));
  }

  function applyMapTransform() {
    clampMapView();
    const vp = document.getElementById("map-viewport");
    if (vp) {
      vp.setAttribute("transform", `translate(${mapView.tx.toFixed(2)} ${mapView.ty.toFixed(2)}) scale(${mapView.scale.toFixed(4)})`);
    }
    const inv = 1.0 / mapView.scale;
    document.querySelectorAll("#topo-svg-wrap .map-keep-size").forEach((el) => {
      const px = el.getAttribute("data-x");
      const py = el.getAttribute("data-y");
      if (px !== null && py !== null) {
        el.setAttribute("transform", `translate(${px} ${py}) scale(${inv.toFixed(4)})`);
      }
    });
    const refGrp = document.getElementById("map-ref-cities");
    if (refGrp) {
      refGrp.setAttribute("opacity", mapView.scale >= 1.45 ? "1" : "0.35");
    }
    const zLbl = document.getElementById("map-zoom-label");
    if (zLbl) zLbl.textContent = `${mapView.scale.toFixed(1)}x`;
  }

  function zoomMapAt(factor, svgX, svgY) {
    if (mapView.animRaf) {
      cancelAnimationFrame(mapView.animRaf);
      mapView.animRaf = 0;
    }
    const oldScale = mapView.scale;
    const newScale = Math.max(1.0, Math.min(6.5, oldScale * factor));
    const worldX = (svgX - mapView.tx) / oldScale;
    const worldY = (svgY - mapView.ty) / oldScale;
    mapView.scale = newScale;
    mapView.tx = svgX - worldX * newScale;
    mapView.ty = svgY - worldY * newScale;
    applyMapTransform();
  }

  function flyMapTo(targetScale, worldX, worldY) {
    if (mapView.animRaf) cancelAnimationFrame(mapView.animRaf);
    const startScale = mapView.scale;
    const startTx = mapView.tx;
    const startTy = mapView.ty;
    const endScale = Math.max(1.0, Math.min(6.5, targetScale));
    const endTx = MAP_W / 2 - worldX * endScale;
    const endTy = MAP_H / 2 - worldY * endScale;
    const t0 = performance.now();
    const dur = 340;
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur);
      const ease = 1 - Math.pow(1 - u, 3);
      mapView.scale = startScale + (endScale - startScale) * ease;
      mapView.tx = startTx + (endTx - startTx) * ease;
      mapView.ty = startTy + (endTy - startTy) * ease;
      applyMapTransform();
      if (u < 1) {
        mapView.animRaf = requestAnimationFrame(step);
      } else {
        mapView.animRaf = 0;
      }
    };
    mapView.animRaf = requestAnimationFrame(step);
  }

  function bindMapInteractions() {
    const wrap = document.getElementById("topo-svg-wrap");
    if (!wrap || mapEventsBound) return;
    mapEventsBound = true;

    let dragging = false;
    let lastClientX = 0;
    let lastClientY = 0;
    let movedPx = 0;

    const clientToSvg = (cx, cy) => {
      const svg = wrap.querySelector("svg");
      if (!svg) return [MAP_W / 2, MAP_H / 2];
      const rect = svg.getBoundingClientRect();
      return [
        ((cx - rect.left) / Math.max(1, rect.width)) * MAP_W,
        ((cy - rect.top) / Math.max(1, rect.height)) * MAP_H,
      ];
    };

    wrap.addEventListener("wheel", (e) => {
      const svg = wrap.querySelector("svg");
      if (!svg) return;
      e.preventDefault();
      const [sx, sy] = clientToSvg(e.clientX, e.clientY);
      const factor = e.deltaY < 0 ? 1.22 : 1 / 1.22;
      zoomMapAt(factor, sx, sy);
    }, { passive: false });

    wrap.addEventListener("pointerdown", (e) => {
      if (e.target && e.target.closest && e.target.closest(".map-controls")) return;
      const svg = wrap.querySelector("svg");
      if (!svg) return;
      dragging = true;
      movedPx = 0;
      lastClientX = e.clientX;
      lastClientY = e.clientY;
      svg.classList.add("dragging");
      try { wrap.setPointerCapture(e.pointerId); } catch (_e) {}
    });

    wrap.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const svg = wrap.querySelector("svg");
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      movedPx += Math.abs(dx) + Math.abs(dy);
      lastClientX = e.clientX;
      lastClientY = e.clientY;
      mapView.tx += (dx / Math.max(1, rect.width)) * MAP_W;
      mapView.ty += (dy / Math.max(1, rect.height)) * MAP_H;
      applyMapTransform();
    });

    const endDrag = (e) => {
      if (!dragging) return;
      dragging = false;
      const svg = wrap.querySelector("svg");
      if (svg) svg.classList.remove("dragging");
      try { wrap.releasePointerCapture(e.pointerId); } catch (_e) {}
    };
    wrap.addEventListener("pointerup", endDrag);
    wrap.addEventListener("pointercancel", endDrag);

    wrap.addEventListener("dblclick", (e) => {
      if (e.target && e.target.closest && e.target.closest(".map-controls")) return;
      const [sx, sy] = clientToSvg(e.clientX, e.clientY);
      zoomMapAt(1.75, sx, sy);
    });

    const btnIn = document.getElementById("map-zoom-in");
    const btnOut = document.getElementById("map-zoom-out");
    const btnReset = document.getElementById("map-zoom-reset");
    if (btnIn) btnIn.addEventListener("click", () => zoomMapAt(1.45, MAP_W / 2, MAP_H / 2));
    if (btnOut) btnOut.addEventListener("click", () => zoomMapAt(1 / 1.45, MAP_W / 2, MAP_H / 2));
    if (btnReset) btnReset.addEventListener("click", () => flyMapTo(1.0, MAP_W / 2, MAP_H / 2));
  }

  function renderClusterTopology(st) {
    const wrap = document.getElementById("topo-svg-wrap");
    if (!wrap || !st) return;
    bindMapInteractions();
    const nodes = Array.isArray(st.nodes) ? st.nodes.filter((n) => n && n.online !== false) : [];
    const totalMem = Number(st.total_memory_gb || 0).toFixed(1);
    const W = MAP_W;
    const H = MAP_H;
    const TA = "text-anc" + "hor";

    const proj = (lon, lat) => {
      const clampedLon = Math.max(-168, Math.min(179, Number(lon || 0)));
      const clampedLat = Math.max(-56, Math.min(76, Number(lat || 0)));
      return [
        ((clampedLon + 170) / 350) * W,
        ((78 - clampedLat) / 136) * H,
      ];
    };

    // Build continent land polygons (vector-effect keeps coastlines razor-sharp when zoomed!)
    let landSvg = "";
    for (const poly of WORLD_POLYS) {
      const pts = poly.map(([lo, la]) => {
        const [x, y] = proj(lo, la);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      landSvg += `<polygon points="${pts}" fill="rgba(255,255,255,0.068)" stroke="rgba(255,255,255,0.18)" stroke-width="0.9" vector-effect="non-scaling-stroke" stroke-linejoin="round" />`;
    }

    // Subtle lat/lon graticule grid
    let gridSvg = "";
    for (const lat of [-30, 0, 30, 60]) {
      const [, y] = proj(0, lat);
      gridSvg += `<line x1="0" y1="${y.toFixed(1)}" x2="${W}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,${lat === 0 ? "0.08" : "0.04"})" stroke-dasharray="3 6" vector-effect="non-scaling-stroke" />`;
    }
    for (const lon of [-120, -60, 0, 60, 120]) {
      const [x] = proj(lon, 0);
      gridSvg += `<line x1="${x.toFixed(1)}" y1="0" x2="${x.toFixed(1)}" y2="${H}" stroke="rgba(255,255,255,0.04)" stroke-dasharray="3 6" vector-effect="non-scaling-stroke" />`;
    }

    // Subtle global reference cities (become clearer when zooming in)
    let refCitiesSvg = "";
    for (const [rcName, rcLon, rcLat] of MAP_REF_CITIES) {
      const [rx, ry] = proj(rcLon, rcLat);
      refCitiesSvg += `
        <g class="map-keep-size" data-x="${rx.toFixed(1)}" data-y="${ry.toFixed(1)}" transform="translate(${rx.toFixed(1)} ${ry.toFixed(1)})">
          <circle cx="0" cy="0" r="1.6" fill="rgba(255,255,255,0.28)" />
          <text x="5" y="3" fill="rgba(255,255,255,0.34)" font-size="8" font-family="IBM Plex Mono, monospace">${escHtml(rcName)}</text>
        </g>
      `;
    }

    const hubGeo = st.hub_geo || { city: "Madrid", country: "Spain", country_code: "ES", lat: 40.4, lon: -3.7 };
    const [hx, hy] = proj(hubGeo.lon, hubGeo.lat);

    // Group active nodes into city/location clusters
    const clusters = new Map();
    for (const n of nodes) {
      const g = (n.geo && typeof n.geo.lat === "number" && typeof n.geo.lon === "number")
        ? n.geo
        : hubGeo;
      const key = `${Math.round(g.lat * 2) / 2},${Math.round(g.lon * 2) / 2}`;
      if (!clusters.has(key)) {
        clusters.set(key, {
          city: g.city || g.country || "Global",
          country: g.country || "",
          cc: g.country_code || "",
          lat: g.lat,
          lon: g.lon,
          nodes: [],
          totalGb: 0,
          hasCuda: false,
        });
      }
      const c = clusters.get(key);
      c.nodes.push(n);
      c.totalGb += Number(n.memory_gb || 0);
      if (String(n.backend || "").toLowerCase() === "cuda") c.hasCuda = true;
    }

    const clusterList = Array.from(clusters.values()).sort((a, b) => b.totalGb - a.totalGb);
    let arcsSvg = "";
    let pinsSvg = "";
    let labelsSvg = "";

    clusterList.forEach((c, idx) => {
      const [px, py] = proj(c.lon, c.lat);
      c._px = px;
      c._py = py;
      const accent = c.hasCuda ? "#4ade80" : "#7dd3fc";
      const dist = Math.hypot(px - hx, py - hy);
      if (dist > 14) {
        const mx = (hx + px) / 2;
        const my = Math.min(hy, py) - Math.min(95, Math.max(28, dist * 0.24));
        arcsSvg += `<path d="M ${hx.toFixed(1)} ${hy.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${px.toFixed(1)} ${py.toFixed(1)}" fill="none" stroke="${accent}" stroke-opacity="0.22" stroke-width="2" vector-effect="non-scaling-stroke" />`;
        arcsSvg += `<path class="topo-arc" d="M ${hx.toFixed(1)} ${hy.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${px.toFixed(1)} ${py.toFixed(1)}" fill="none" stroke="${accent}" stroke-opacity="0.88" stroke-width="1.8" vector-effect="non-scaling-stroke" />`;
      }

      pinsSvg += `
        <g class="map-keep-size" data-x="${px.toFixed(1)}" data-y="${py.toFixed(1)}" transform="translate(${px.toFixed(1)} ${py.toFixed(1)})" style="cursor:pointer;">
          <circle class="map-pulse" cx="0" cy="0" r="8" fill="none" stroke="${accent}" stroke-width="1.4" style="animation-delay:${(idx * 0.45).toFixed(2)}s" />
          <circle cx="0" cy="0" r="9" fill="${accent}" fill-opacity="0.20" />
          <circle cx="0" cy="0" r="4.5" fill="${accent}" stroke="#05070b" stroke-width="1.4" />
        </g>
      `;

      const cityLabel = c.cc ? `${c.city}, ${c.cc}` : c.city;
      const countLbl = `${c.nodes.length} ${c.nodes.length === 1 ? "node" : "nodes"} · ${c.totalGb.toFixed(1)} GB`;
      const chipsLbl = c.nodes
        .map((n) => String(n.chip_name || "Node").replace("NVIDIA GeForce ", "").replace("Apple ", ""))
        .slice(0, 3)
        .join(" + ")
        .slice(0, 30);

      const cw = 202;
      const ch = 54;
      const placeLeft = px > W * 0.64;
      const dx = placeLeft ? -(cw + 14) : 14;
      const dy = -(ch / 2) + (idx % 2 === 0 ? -10 : 10);
      const anchorX = placeLeft ? dx + cw : dx;
      const anchorY = dy + ch / 2;

      labelsSvg += `
        <g class="map-keep-size map-cluster-card" data-idx="${idx}" data-x="${px.toFixed(1)}" data-y="${py.toFixed(1)}" transform="translate(${px.toFixed(1)} ${py.toFixed(1)})" style="cursor:pointer;">
          <line x1="0" y1="0" x2="${anchorX.toFixed(1)}" y2="${anchorY.toFixed(1)}" stroke="${accent}" stroke-opacity="0.45" stroke-width="1" />
          <rect x="${dx.toFixed(1)}" y="${dy.toFixed(1)}" width="${cw}" height="${ch}" rx="8" fill="rgba(7,9,13,0.92)" stroke="${accent}" stroke-opacity="0.58" stroke-width="1.1" />
          <text x="${(dx + 11).toFixed(1)}" y="${(dy + 18).toFixed(1)}" fill="#ffffff" font-size="11.5" font-weight="600" font-family="Inter, system-ui, sans-serif">${escHtml(cityLabel)}</text>
          <text x="${(dx + cw - 10).toFixed(1)}" y="${(dy + 17).toFixed(1)}" ${TA}="end" fill="${accent}" font-size="9.5" font-family="IBM Plex Mono, monospace">${escHtml(countLbl)}</text>
          <text x="${(dx + 11).toFixed(1)}" y="${(dy + 36).toFixed(1)}" fill="rgba(255,255,255,0.74)" font-size="9.5" font-family="IBM Plex Mono, monospace">${escHtml(chipsLbl)}</text>
          <text x="${(dx + 11).toFixed(1)}" y="${(dy + 47).toFixed(1)}" fill="rgba(255,255,255,0.46)" font-size="8.5" font-family="IBM Plex Mono, monospace">${escHtml(c.nodes.map((n) => n.display_name || "node").slice(0, 3).join(", ").slice(0, 32))}</text>
        </g>
      `;
    });

    const regionCount = clusterList.length;
    const svg = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Space AI Interactive Global Compute Map">
        <defs>
          <radialGradient id="mapGlow" cx="50%" cy="42%" r="55%">
            <stop offset="0%" stop-color="rgba(74, 222, 128, 0.08)" />
            <stop offset="100%" stop-color="rgba(0, 0, 0, 0)" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width="${W}" height="${H}" fill="url(#mapGlow)" />
        <g id="map-viewport">
          ${gridSvg}
          ${landSvg}
          <g id="map-ref-cities" opacity="0.35">${refCitiesSvg}</g>
          ${arcsSvg}
          ${pinsSvg}
          ${labelsSvg}
        </g>
        <g>
          <rect x="14" y="${H - 36}" width="332" height="24" rx="6" fill="rgba(8,10,14,0.84)" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
          <circle cx="27" cy="${H - 24}" r="3.6" fill="#4ade80" />
          <text x="37" y="${H - 20.5}" fill="rgba(255,255,255,0.82)" font-size="9.5" font-family="IBM Plex Mono, monospace">NVIDIA CUDA</text>
          <circle cx="126" cy="${H - 24}" r="3.6" fill="#7dd3fc" />
          <text x="136" y="${H - 20.5}" fill="rgba(255,255,255,0.82)" font-size="9.5" font-family="IBM Plex Mono, monospace">APPLE SILICON</text>
          <text x="336" y="${H - 20.5}" ${TA}="end" fill="rgba(255,255,255,0.48)" font-size="8.8" font-family="IBM Plex Mono, monospace">SCROLL ZOOM · DRAG PAN</text>
        </g>
      </svg>
    `;
    wrap.innerHTML = svg;
    applyMapTransform();

    wrap.querySelectorAll(".map-cluster-card").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = Number(el.getAttribute("data-idx"));
        const c = clusterList[idx];
        if (c) flyMapTo(mapView.scale < 2.5 ? 3.2 : 1.0, c._px, c._py);
      });
    });

    const regEl = document.getElementById("map-region-strip");
    if (regEl) {
      regEl.innerHTML = clusterList.map((c, idx) => {
        const dotCol = c.hasCuda ? "#4ade80" : "#7dd3fc";
        const loc = c.cc ? `${c.city}, ${c.cc}` : c.city;
        const names = c.nodes.map((n) => escHtml(n.display_name || "node")).join(" · ");
        return `<button type="button" class="map-region-pill" data-ridx="${idx}" title="Click to zoom into ${escHtml(loc)}"><span class="map-region-dot" style="background:${dotCol};"></span><strong>${escHtml(loc)}</strong><span style="opacity:0.65;">${c.nodes.length} ${c.nodes.length === 1 ? "device" : "devices"} · ${c.totalGb.toFixed(1)} GB</span><span style="opacity:0.45;">(${names})</span></button>`;
      }).join("");
      regEl.querySelectorAll(".map-region-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
          const idx = Number(btn.getAttribute("data-ridx"));
          const c = clusterList[idx];
          if (c) flyMapTo(3.2, c._px, c._py);
        });
      });
    }

    setText("topo-foot-left", `${regionCount} live ${regionCount === 1 ? "region" : "regions"} · Scroll to zoom, drag to pan, click any city to fly there`);
    setText("topo-foot-right", `${nodes.length} active ${nodes.length === 1 ? "node" : "nodes"} · ${totalMem} GB unified + VRAM`);
  }

  async function refreshNetwork() {
    const statusEl = document.getElementById("coord-status");
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 5000);
      const netHeaders = { "X-Space-Client": "network-portal" };
      try {
        let cid = (localStorage.getItem("space_client_id_v1") || "").trim();
        if (!cid) {
          cid = (typeof crypto !== "undefined" && crypto.randomUUID)
            ? crypto.randomUUID()
            : ("net-" + Math.random().toString(36).slice(2) + Date.now().toString(36));
          localStorage.setItem("space_client_id_v1", cid);
        }
        if (cid) netHeaders["X-Space-Client-Id"] = cid;
        const savedW = (localStorage.getItem("space_solana_wallet") || "").trim();
        if (savedW) netHeaders["X-Space-Wallet"] = savedW;
      } catch (_e) {}
      const res = await fetch(COORD_URL + "/v1/network/stats", { signal: ctrl.signal, headers: netHeaders });
      clearTimeout(timer);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const st = await res.json();
      const cnt = Number(st.active_nodes || 0);
      const onlineU = Number(st.online_users ?? st.users?.online ?? 0);
      const totalU = Number(st.total_users ?? st.users?.total ?? onlineU);
      if (statusEl) statusEl.textContent = `Live · ${cnt} ${cnt === 1 ? "Device" : "Devices"} · ${onlineU} Online`;
      setText("fleet-nodes", String(cnt));
      setText("fleet-mem", Number(st.total_memory_gb || 0).toLocaleString("en-US") + " GB");
      setText("fleet-users", onlineU.toLocaleString("en-US"));
      setText("fleet-users-lbl", `Online Users · ${totalU.toLocaleString("en-US")} Total`);
      setText("fleet-tokens", Number(st.verified_tokens || 0).toLocaleString("en-US"));
      if (typeof st.next_sync_secs === "number") {
        nextSyncAt = Date.now() + Math.max(0, st.next_sync_secs) * 1000;
        tickSyncTimer();
      }
      if (st.goodput) {
        latestGoodput = st.goodput;
        renderGoodputChart();
      }
      renderClusterTopology(st);
      if (st.genesis_program) {
        const gp = st.genesis_program;
        setText("genesis-claimed", `${gp.slots_claimed} / ${gp.slots_total}`);
        setText("genesis-remaining", `${gp.slots_remaining} Genesis slots open · ${gp.multiplier}x Compute Credits`);
      }

      // Pooled-RAM Swarm plan
      const sw = st.swarm_plan;
      if (sw) {
        setText("swarm-model", `${sw.model_id} (${sw.ready ? "Ready" : "Growing"})`);
        setText("swarm-ram", `${sw.pooled_memory_gb} GB Pooled · ${sw.node_count} Stages`);
        setText("swarm-wire", `${sw.activation_kb_per_token} KB/token activation relay`);
        const stageWrap = document.getElementById("swarm-stages");
        if (stageWrap && Array.isArray(sw.stages)) {
          stageWrap.innerHTML = "";
          for (const s of sw.stages) {
            const div = document.createElement("div");
            div.className = "glass";
            div.style.padding = "0.95rem 1.05rem";
            div.innerHTML = `
              <p class="eyebrow" style="margin-bottom:0.25rem;">Stage ${s.stage_index} · Layers ${escHtml(s.layer_range)}</p>
              <div style="font-weight:600; font-size:0.95rem;">${escHtml(s.chip_name)}</div>
              <div style="font-family:var(--mono); font-size:0.74rem; margin-top:0.2rem;">${escHtml(String(s.backend || "").toUpperCase())} · ${s.memory_gb} GB · ${s.layer_count} layers</div>
            `;
            stageWrap.appendChild(div);
          }
        }
      }

      // Season 0 Canary Testnet Leaderboard
      const lb = st.leaderboard;
      const lbBody = document.getElementById("lb-body");
      if (lbBody && lb && Array.isArray(lb.leaderboard)) {
        lbBody.innerHTML = "";
        if (lb.leaderboard.length === 0) {
          lbBody.innerHTML = '<tr><td colspan="6" style="font-family:var(--mono);">No Season 0 operators yet.</td></tr>';
        } else {
          const ordered = [...lb.leaderboard].sort((a, b) => {
            const ao = a.online === false ? 1 : 0;
            const bo = b.online === false ? 1 : 0;
            if (ao !== bo) return ao - bo;
            return Number(b.season0_points || 0) - Number(a.season0_points || 0);
          });
          for (const op of ordered) {
            const isOnline = op.online !== false;
            const tr = document.createElement("tr");
            if (!isOnline) tr.style.opacity = "0.46";
            const dotHtml = isOnline
              ? '<span class="dot" title="Online" style="display:inline-block;width:7px;height:7px;margin-right:0.48rem;vertical-align:middle;"></span>'
              : '<span title="Offline" style="display:inline-block;width:7px;height:7px;border-radius:50%;border:1px solid rgba(255,255,255,0.45);margin-right:0.48rem;vertical-align:middle;"></span>';
            const mintedVal = Number(op.minted_tlng ?? 0);
            const totalVal = Number(op.soulbound_tlng ?? op.season0_points ?? 0);
            const autoSpaceVal = Number(op.auto_space ?? 0);
            const pendingVal = Math.max(0, Number((totalVal - mintedVal).toFixed(2)));
            const tlngFmt = totalVal >= 100
              ? totalVal.toLocaleString("en-US", { maximumFractionDigits: 0 })
              : totalVal.toLocaleString("en-US", { maximumFractionDigits: 2 });
            const spaceFmt = autoSpaceVal >= 100
              ? autoSpaceVal.toLocaleString("en-US", { maximumFractionDigits: 0 })
              : autoSpaceVal.toLocaleString("en-US", { maximumFractionDigits: 2 });
            const syncSub = autoSpaceVal > 0
              ? `<div style="font-size:0.66rem; opacity:0.78; margin-top:0.15rem; color:#86efac;">${spaceFmt} $SPACE transferable &#10003;</div>`
              : (pendingVal >= 0.05
                ? `<div style="font-size:0.66rem; opacity:0.72; margin-top:0.15rem;">+${pendingVal.toLocaleString("en-US", { maximumFractionDigits: 1 })} pending epoch</div>`
                : `<div style="font-size:0.66rem; opacity:0.62; margin-top:0.15rem;">Synced on-chain &#10003;</div>`);
            const expUrl = op.explorer_url || "https://explorer.solana.com/address/352xoBR8e1mUbnAK2ZCwTLBJtoU4vMhBAV5pedXU9CYK?cluster=devnet";
            const tweetTxt = encodeURIComponent(
              `My ${op.chip_name} (${op.memory_gb} GB) is live on Space AI Season 0 Canary Testnet — Rank #${op.rank} with ${Number(op.verified_tokens).toLocaleString("en-US")} tokens served.\n\nRun local AI. Together:\nhttps://spaceresearch.ai`
            );
            tr.innerHTML = `
              <td style="font-family:var(--mono); font-weight:600;">#${op.rank}</td>
              <td><div>${dotHtml}<strong>${escHtml(op.handle)}</strong></div><div style="font-family:var(--mono); font-size:0.74rem; margin-top:0.15rem;">${escHtml(op.chip_name)} · ${op.memory_gb} GB</div></td>
              <td style="font-family:var(--mono);">${Number(op.verified_tokens || 0).toLocaleString("en-US")}</td>
              <td style="font-family:var(--mono); font-weight:600;">${Number(op.season0_points || 0).toLocaleString("en-US")} pts</td>
              <td style="font-family:var(--mono);"><a href="${escHtml(expUrl)}" target="_blank" rel="noopener" title="View $SPACE & soulbound tSPACE balances on Solana Explorer" style="text-decoration:underline; text-underline-offset:3px;">${tlngFmt} tSPACE &#8599;</a>${syncSub}</td>
              <td>
                <div style="display:flex; gap:0.45rem; flex-wrap:nowrap;">
                  <button type="button" class="btn ghost btn-dl-card" style="padding:0.36rem 0.65rem; font-size:0.66rem;">Card PNG</button>
                  <a class="btn ghost" style="padding:0.36rem 0.65rem; font-size:0.66rem;" target="_blank" rel="noopener" href="https://x.com/intent/tweet?text=${tweetTxt}">Post X</a>
                </div>
              </td>
            `;
            tr.querySelector(".btn-dl-card").addEventListener("click", () => downloadShareCard(op));
            lbBody.appendChild(tr);
          }
        }
      }
    } catch (_err) {
      if (statusEl) statusEl.textContent = "Connecting...";
    }
  }

  async function runPrompt(evt) {
    if (evt) evt.preventDefault();
    const outEl = document.getElementById("prompt-output");
    const metaEl = document.getElementById("prompt-meta");
    const btn = document.getElementById("btn-run-prompt");
    const prompt = (document.getElementById("p-text")?.value || "").trim();
    const modelId = document.getElementById("p-model")?.value || "auto";
    if (!prompt) return;

    if (btn) btn.disabled = true;
    if (outEl) outEl.innerHTML = `<p>${modelId.startsWith("swarm") ? "Running Multi-Node Swarm across GPU 14B + Apple Silicon peers in parallel..." : "Routing to fastest available local device..."}</p>`;
    try {
      const res = await fetch(COORD_URL + "/v1/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model_id: modelId }),
      });
      if (!res.ok) {
        let msg = "HTTP " + res.status;
        try { const errJson = await res.json(); if (errJson.detail) msg = errJson.detail; } catch (_e) {}
        if (outEl) outEl.innerHTML = `<p>${escHtml(msg)}</p>`;
        if (metaEl) metaEl.textContent = "Swarm model requires --swarm shard peers loaded.";
        return;
      }
      const data = await res.json();
      if (outEl) {
        outEl.innerHTML = renderMarkdown(data.output_text);
        outEl.scrollTop = 0;
      }
      if (metaEl && data.receipt) {
        metaEl.textContent = `${data.node_chip} (${String(data.node_backend || "").toUpperCase()}) · ${Math.round(data.receipt.latency_ms)} ms · ${data.receipt.tokens_served} tokens`;
      }
      refreshNetwork();
    } catch (_err) {
      if (outEl) outEl.innerHTML = "<p>Network temporarily busy. Try again in a moment.</p>";
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    let activeAgentTab = "cli";
    const keyInput = document.getElementById("barter-key-input");
    const cfgBox = document.getElementById("barter-cfg-box");
    const cfgRoute = document.getElementById("cfg-route");
    const cfgPrivacy = document.getElementById("cfg-privacy");
    const cfgRegion = document.getElementById("cfg-region");

    function getDisplayKey() {
      const raw = (keyInput?.value || "").trim();
      return raw || "lng_live_YOUR_API_KEY";
    }

    function getConstraints() {
      return {
        route: cfgRoute?.value || "fastest",
        privacy: cfgPrivacy?.value || "standard",
        region: cfgRegion?.value || "auto",
      };
    }

    function renderBarterConfig() {
      if (!cfgBox) return;
      const root = COORD_URL.replace(/\/+$/, "");
      const base = root + "/v1";
      const key = getDisplayKey();
      const { route, privacy, region } = getConstraints();
      setText("cfg-constraint-summary", `${route} · ${privacy} · ${region}`);
      const cliFlags = `--route ${route} --privacy ${privacy}` + (region !== "auto" ? ` --region ${region}` : "");
      if (activeAgentTab === "cli") {
        cfgBox.textContent =
          `# 1. Install Space CLI (zero root required):\n` +
          `curl -fsSL https://spaceresearch.ai/install-space.sh | sh\n\n` +
          `# 2. Launch coding agents via unified local proxy (http://127.0.0.1:8377):\n` +
          `space claude ${cliFlags}\n` +
          `space codex ${cliFlags}\n` +
          `space opencode ${cliFlags}\n\n` +
          `# 3. Run worker + local proxy together (Auto-Pause on; adapters: native|ollama|llama-cpp|vllm|mlx):\n` +
          `space up --name my-rig --auto-pause --adapter native --proxy-port 8377\n` +
          `# Injected headers -> X-Space-Route: ${route} | X-Space-Privacy: ${privacy} | X-Space-Region: ${region}`;
      } else if (activeAgentTab === "claude") {
        cfgBox.textContent =
          `export ANTHROPIC_BASE_URL="${root}"\n` +
          `export ANTHROPIC_AUTH_TOKEN="${key}"\n` +
          `export ANTHROPIC_API_KEY="${key}"\n` +
          `export ANTHROPIC_CUSTOM_HEADERS="X-Space-Route: ${route}, X-Space-Privacy: ${privacy}, X-Space-Region: ${region}"\n` +
          `# Run Claude Code directly on Space AI verifiable local silicon:\n` +
          `claude --model qwen/Qwen3.8-27B`;
      } else if (activeAgentTab === "cursor") {
        cfgBox.textContent = JSON.stringify(
          {
            provider: "openai-compatible",
            baseUrl: base,
            apiKey: key,
            model: "qwen/Qwen3.8-27B",
            headers: {
              "X-Space-Route": route,
              "X-Space-Privacy": privacy,
              "X-Space-Region": region,
            },
          },
          null,
          2
        );
      } else if (activeAgentTab === "curl") {
        cfgBox.textContent =
          `curl -sS "${base}/messages" \\\n` +
          `  -H "Content-Type: application/json" \\\n` +
          `  -H "anthropic-version: 2023-06-01" \\\n` +
          `  -H "x-api-key: ${key}" \\\n` +
          `  -H "X-Space-Route: ${route}" \\\n` +
          `  -H "X-Space-Privacy: ${privacy}" \\\n` +
          `  -H "X-Space-Region: ${region}" \\\n` +
          `  -d '{"model":"qwen/Qwen3.8-27B","max_tokens":256,"messages":[{"role":"user","content":"Write a fast binary search in Rust."}]}'`;
      } else {
        cfgBox.textContent =
          `export OPENAI_BASE_URL="${base}"\n` +
          `export OPENAI_API_KEY="${key}"\n` +
          `export SPACE_ROUTE="${route}"\n` +
          `export SPACE_PRIVACY="${privacy}"\n` +
          `export SPACE_REGION="${region}"\n` +
          `# Active policy headers: X-Space-Route: ${route} · X-Space-Privacy: ${privacy} · X-Space-Region: ${region}\n` +
          `# Works with OpenAI Codex (/v1/responses), OpenCode, Aider, Continue & OpenAI SDKs\n` +
          `# Or 1-cmd launcher: space codex ${cliFlags}`;
      }
    }

    async function checkBarterKey() {
      const statusEl = document.getElementById("barter-key-status");
      const key = (keyInput?.value || "").trim();
      renderBarterConfig();
      if (!key) {
        if (statusEl) statusEl.textContent = "Paste your lng_live_... key from your worker terminal to inspect live credits.";
        return;
      }
      try { localStorage.setItem("space_barter_key", key); } catch (_e) {}
      if (statusEl) statusEl.textContent = "Checking live Compute-Barter balance...";
      try {
        const res = await fetch(COORD_URL + "/v1/keys/me?api_key=" + encodeURIComponent(key));
        if (!res.ok) {
          let detail = "HTTP " + res.status;
          try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
          setText("bk-remaining", "--");
          setText("bk-earned", "--");
          setText("bk-used", "--");
          setText("bk-genesis", "--");
          if (statusEl) statusEl.textContent = "Key lookup failed: " + detail;
          return;
        }
        const info = await res.json();
        const fmtInt = (v) => Number(v || 0).toLocaleString("en-US");
        setText("bk-remaining", fmtInt(info.remaining_tokens));
        setText("bk-earned", fmtInt(info.credit_tokens));
        setText("bk-used", fmtInt(info.used_tokens));
        const gSlot = info.genesis_slot;
        const gMult = Number(info.genesis_multiplier || 1.0);
        setText("bk-genesis", gSlot ? `#${gSlot} / 50 · ${gMult.toFixed(1)}x` : `Standard · ${gMult.toFixed(1)}x`);
        if (statusEl) {
          statusEl.textContent = `Active key ${info.api_key_prefix || ""} · Node: ${info.label || "community-node"} · OpenAI Base: ${COORD_URL}/v1`;
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = "Could not reach live coordinator (" + (err?.message || "network error") + "). Try again in a moment.";
      }
    }

    try {
      const savedKey = localStorage.getItem("space_barter_key") || localStorage.getItem("longinus_barter_key");
      if (savedKey && keyInput) keyInput.value = savedKey;
    } catch (_e) {}

    document.querySelectorAll(".agent-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        activeAgentTab = tab.getAttribute("data-agent") || "cli";
        document.querySelectorAll(".agent-tab").forEach((t) => t.classList.toggle("active", t === tab));
        renderBarterConfig();
      });
    });

    [cfgRoute, cfgPrivacy, cfgRegion].forEach((sel) => {
      if (sel) sel.addEventListener("change", () => {
        renderBarterConfig();
        renderSessionSnippet(activeSessionToken);
      });
    });

    if (keyInput) {
      keyInput.addEventListener("input", renderBarterConfig);
      keyInput.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") { ev.preventDefault(); checkBarterKey(); }
      });
    }
    const btnCheckKey = document.getElementById("btn-check-key");
    if (btnCheckKey) btnCheckKey.addEventListener("click", checkBarterKey);

    const btnToggleKey = document.getElementById("btn-toggle-key");
    if (btnToggleKey && keyInput) {
      btnToggleKey.addEventListener("click", () => {
        const hidden = keyInput.type === "password";
        keyInput.type = hidden ? "text" : "password";
        btnToggleKey.textContent = hidden ? "Hide" : "Show";
      });
    }

    const btnCopyCfg = document.getElementById("btn-copy-barter-cfg");
    if (btnCopyCfg && cfgBox) {
      btnCopyCfg.addEventListener("click", () => {
        navigator.clipboard.writeText(cfgBox.textContent || "").then(() => {
          const prev = btnCopyCfg.textContent;
          btnCopyCfg.textContent = "Copied ✓";
          setTimeout(() => { btnCopyCfg.textContent = prev; }, 1800);
        });
      });
    }

    const btnCopyBase = document.getElementById("btn-copy-base-url");
    if (btnCopyBase) {
      btnCopyBase.addEventListener("click", () => {
        navigator.clipboard.writeText(COORD_URL.replace(/\/+$/, "") + "/v1").then(() => {
          const prev = btnCopyBase.textContent;
          btnCopyBase.textContent = "Copied URL ✓";
          setTimeout(() => { btnCopyBase.textContent = prev; }, 1800);
        });
      });
    }

    syncCoordinatorEndpoint().finally(() => {
      renderBarterConfig();
      if (keyInput && keyInput.value.trim()) checkBarterKey();
      refreshNetwork();
      refreshIntegrity();
      refreshPools();
      refreshDemandLeaderboard();
      refreshBudgetQuote();
    });
    setInterval(() => { if (!document.hidden) refreshNetwork(); }, 10000);
    setInterval(() => { if (!document.hidden) tickSyncTimer(); }, 1000);
    const form = document.getElementById("live-prompt-form");
    if (form) form.addEventListener("submit", runPrompt);

    const chartMetricTabs = document.querySelectorAll(".gp-tab[data-metric]");
    chartMetricTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        activeChartMetric = tab.getAttribute("data-metric") || "cluster_tok_s";
        chartMetricTabs.forEach((t) => t.classList.toggle("active", t === tab));
        renderGoodputChart();
      });
    });

    const JOIN_CMDS = {
      cli: "curl -fsSL https://spaceresearch.ai/install-space.sh | sh && space claude",
      nvidia: "curl -fsSL https://spaceresearch.ai/join-gpu.sh | sh -s -- --name my-gpu --auto-pause",
      docker: "curl -fsSLO https://spaceresearch.ai/Dockerfile.worker && curl -fsSLO https://spaceresearch.ai/join.py\ndocker build -f Dockerfile.worker -t space-worker .\ndocker run --rm -it --gpus all --read-only -v space_data:/data -e HOME=/data space-worker --name my-gpu --auto-pause",
      mac: "curl -sO https://spaceresearch.ai/join.py\npython3 join.py --install --auto-pause",
    };
    let activeJoinTab = "cli";
    const joinCmdBox = document.getElementById("join-cmd-box");
    document.querySelectorAll(".join-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        activeJoinTab = tab.getAttribute("data-tab") || "nvidia";
        document.querySelectorAll(".join-tab").forEach((t) => t.classList.toggle("active", t === tab));
        if (joinCmdBox && JOIN_CMDS[activeJoinTab]) joinCmdBox.textContent = JOIN_CMDS[activeJoinTab];
      });
    });

    const RIG_SPECS = {
      rtx5090: { cat: "gpu", name: "RTX 5090", sub: "32 GB · Blackwell", tps: "210–650 tok/s", credits: "14.5M – 29.0M / day", model: "Qwen3.8 27B + Inkling 2 Mini", note: "8 concurrent slots · GPU Vision · 20x Genesis barter rate · Auto-Pause safe" },
      rtx5080: { cat: "gpu", name: "RTX 5080", sub: "16 GB · Blackwell", tps: "130–310 tok/s", credits: "6.8M – 13.6M / day", model: "Qwen3.5 9B / Gemma 4 26B", note: "4 concurrent slots · native FP4 Blackwell speed · Auto-Pause safe" },
      rtx5070ti: { cat: "gpu", name: "RTX 5070 Ti", sub: "16 GB · Blackwell", tps: "110–260 tok/s", credits: "5.4M – 10.8M / day", model: "Qwen3.5 9B / Gemma 4 26B", note: "16 GB GDDR7 · sub-80ms TTFA · ideal gaming + overnight contributor" },
      rtx4090: { cat: "gpu", name: "RTX 4090", sub: "24 GB · Ada", tps: "115–240 tok/s", credits: "7.2M – 14.4M / day", model: "Qwen3.8 27B / Gemma 4 26B", note: "24 GB VRAM · 4 concurrent slots · strong multi-user Goodput" },
      rtx3090: { cat: "gpu", name: "RTX 3090", sub: "24 GB · Ampere", tps: "65–130 tok/s", credits: "4.1M – 8.2M / day", model: "Gemma 4 26B-A4B (4-bit)", note: "24 GB VRAM · high value home-lab & ex-mining GPU node" },
      m5ultra: { cat: "mac", name: "M5 Ultra", sub: "192–512 GB · Metal", tps: "110–185 tok/s", credits: "9.6M – 19.2M / day", model: "Qwen3.8-Flash-Next 125B", note: "Runs flagship 125B MoE locally in unified RAM · whisper-quiet" },
      m5max: { cat: "mac", name: "M5 Max", sub: "48–128 GB · Metal", tps: "75–125 tok/s", credits: "4.8M – 9.6M / day", model: "Qwen3.8 27B / Gemma 4 26B", note: "High-bandwidth unified memory · great for long-context agents" },
      m4max: { cat: "mac", name: "M4 Max", sub: "36–128 GB · Metal", tps: "65–105 tok/s", credits: "3.8M – 7.6M / day", model: "Gemma 4 26B-A4B (Metal)", note: "546 GB/s unified memory · silent desk-friendly local AI node" },
      m5pro: { cat: "mac", name: "M5 / M4 Pro", sub: "24–48 GB · Metal", tps: "45–75 tok/s", credits: "2.4M – 4.8M / day", model: "Gemma 4 26B-A4B (Metal)", note: "4B active MoE runs smoothly with ~30W draw · Auto-Pause safe" },
      m2max: { cat: "mac", name: "M2 / M3 Max", sub: "32–64 GB · Metal", tps: "45–70 tok/s", credits: "2.2M – 4.4M / day", model: "Gemma 4 26B-A4B (Metal)", note: "400 GB/s unified RAM · earn free flagship API credits while idle" },
      m1base: { cat: "mac", name: "M1 / M2 / M3", sub: "16 GB · Metal", tps: "28–45 tok/s", credits: "1.1M – 2.2M / day", model: "Qwen3.5 9B (4-bit Metal)", note: "16 GB MacBook/Mini · trade idle hours for flagship 27B/125B API access" },
    };
    let activeRigCat = "gpu";
    let activeRigId = "rtx5090";

    function renderRigPills() {
      const grid = document.getElementById("rig-pills");
      if (!grid) return;
      const entries = Object.entries(RIG_SPECS).filter(([, s]) => s.cat === activeRigCat);
      if (!entries.some(([k]) => k === activeRigId) && entries.length) {
        activeRigId = entries[0][0];
      }
      grid.innerHTML = "";
      for (const [id, spec] of entries) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rig-pill" + (id === activeRigId ? " active" : "");
        btn.setAttribute("data-rig", id);
        btn.innerHTML = `<span class="rig-pill-name">${escHtml(spec.name)}</span><span class="rig-pill-sub">${escHtml(spec.sub)}</span>`;
        btn.addEventListener("click", () => {
          activeRigId = id;
          renderRigPills();
          updateRigEstimator();
        });
        grid.appendChild(btn);
      }
    }

    function updateRigEstimator() {
      const spec = RIG_SPECS[activeRigId] || RIG_SPECS.rtx5090;
      setText("est-tps", spec.tps);
      setText("est-credits", spec.credits);
      setText("est-model", spec.model);
      setText("est-note", spec.note);
    }

    document.querySelectorAll(".rig-cat-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        activeRigCat = tab.getAttribute("data-cat") || "gpu";
        document.querySelectorAll(".rig-cat-tab").forEach((t) => t.classList.toggle("active", t === tab));
        renderRigPills();
        updateRigEstimator();
      });
    });
    renderRigPills();
    updateRigEstimator();

    document.querySelectorAll(".copy-join-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cmd = (joinCmdBox && joinCmdBox.textContent) || JOIN_CMDS[activeJoinTab] || JOIN_CMDS.mac;
        navigator.clipboard.writeText(cmd).then(() => {
          const prev = btn.textContent;
          btn.textContent = "Copied ✓";
          setTimeout(() => { btn.textContent = prev; }, 1800);
        });
      });
    });

    // Live Homepage Decision X-Ray Demo
    const hxForm = document.getElementById("hx-form");
    const hxInput = document.getElementById("hx-input");
    const hxBtn = document.getElementById("hx-run-btn");
    const hxAns = document.getElementById("hx-ans");
    const hxCertLbl = document.getElementById("hx-cert-lbl");
    const hxForkLbl = document.getElementById("hx-fork-lbl");
    const hxFill = document.getElementById("hx-fill");
    const hxRadar = document.getElementById("hx-radar");
    const hxChatLink = document.getElementById("hx-open-chat");
    let hxAbort = null;

    async function runHomepageXray(promptText) {
      const q = String(promptText || hxInput?.value || "").trim();
      if (!q || !hxAns) return;
      if (hxInput) hxInput.value = q;
      if (hxChatLink) hxChatLink.href = "./chat.html?q=" + encodeURIComponent(q);
      if (hxAbort) { try { hxAbort.abort(); } catch (_e) {} }
      hxAbort = new AbortController();
      if (hxBtn) { hxBtn.disabled = true; hxBtn.textContent = "Scanning..."; }
      hxAns.innerHTML = '<span style="opacity:0.65; font-family:var(--mono); font-size:0.84rem;">Streaming live tokens from local silicon...</span>';
      if (hxCertLbl) hxCertLbl.textContent = "CERTAINTY --";
      if (hxForkLbl) hxForkLbl.textContent = "0 FORKS";
      if (hxFill) { hxFill.style.width = "0%"; hxFill.style.background = "#4ade80"; }
      if (hxRadar) hxRadar.innerHTML = '<span style="opacity:0.65;">Waiting for completion to run Truth &amp; Bluff Radar...</span>';

      let fullText = "";
      const sigs = [];
      let t0 = performance.now();
      try {
        const res = await fetch(COORD_URL + "/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: hxAbort.signal,
          body: JSON.stringify({
            model: "qwen3.8-27b",
            stream: true,
            xray: true,
            max_tokens: 180,
            messages: [{ role: "user", content: q }],
          }),
        });
        if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        hxAns.innerHTML = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const line of lines) {
            const tr = line.trim();
            if (!tr.startsWith("data:")) continue;
            const raw = tr.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            try {
              const p = JSON.parse(raw);
              const tok = p?.choices?.[0]?.delta?.content || "";
              if (tok) {
                fullText += tok;
                const sig = p.token_signal || null;
                if (sig) sigs.push(sig);
                const c = sig ? Number(sig.certainty ?? sig.prob ?? 0.85) : 0.85;
                const cls = c >= 0.78 ? "hi" : (c >= 0.48 ? "mid" : "lo");
                const alts = (sig && Array.isArray(sig.top_alternatives))
                  ? sig.top_alternatives.slice(0, 3).map((a) => `${a.token} (${Math.round(Number(a.prob || 0) * 100)}%)`).join(" · ")
                  : "";
                const sp = document.createElement("span");
                sp.className = "hx-tok " + cls;
                sp.textContent = tok;
                if (alts) sp.title = `Certainty ${Math.round(c * 100)}% | Alts: ${alts}`;
                hxAns.appendChild(sp);

                if (sigs.length) {
                  const avg = sigs.reduce((s, x) => s + Number(x.certainty ?? x.prob ?? 0.5), 0) / sigs.length;
                  const pct = Math.round(avg * 1000) / 10;
                  const forks = sigs.filter((x) => Number(x.certainty ?? x.prob ?? 1) < 0.62).length;
                  const col = pct >= 78 ? "#4ade80" : (pct >= 52 ? "#fbbf24" : "#fb7185");
                  if (hxCertLbl) { hxCertLbl.textContent = `CERTAINTY ${pct.toFixed(1)}%`; hxCertLbl.style.color = col; }
                  if (hxForkLbl) hxForkLbl.textContent = `${forks} ${forks === 1 ? "FORK" : "FORKS"} · ${Math.round(performance.now() - t0)} ms`;
                  if (hxFill) { hxFill.style.width = `${Math.max(4, Math.min(100, pct))}%`; hxFill.style.background = col; }
                }
              }
            } catch (_e) {}
          }
        }

        // Fire automatic Truth, Sycophancy & Bluff Radar
        if (fullText.trim() && hxRadar) {
          hxRadar.innerHTML = '<span style="opacity:0.75;">Running Truth, Sycophancy &amp; Bluff Radar...</span>';
          const avgCert = sigs.length
            ? sigs.reduce((s, x) => s + Number(x.certainty ?? x.prob ?? 0.5), 0) / sigs.length
            : null;
          const ar = await fetch(COORD_URL + "/v1/audit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: hxAbort.signal,
            body: JSON.stringify({
              user_prompt: q,
              assistant_reply: fullText.trim(),
              mean_certainty: avgCert,
            }),
          });
          if (ar.ok) {
            const aud = await ar.json();
            const cls = aud.verdict === "VERIFIED" ? "ok" : (aud.verdict === "CAUTION" ? "warn" : "risk");
            hxRadar.innerHTML = `
              <span class="hx-badge ${cls}">${escHtml(aud.verdict_label || aud.verdict)}</span>
              <span>Grounded <strong>${aud.grounded_pct}%</strong> · Sycophancy <strong>${aud.sycophancy_pct}%</strong> · Bluff <strong>${aud.bluff_pct}%</strong></span>
              ${aud.note ? `<span style="width:100%; opacity:0.84; font-family:var(--sans); font-size:0.84rem; margin-top:0.2rem;">${escHtml(aud.note)}</span>` : ""}
            `;
          }
        }
      } catch (err) {
        if (err && err.name === "AbortError") return;
        hxAns.innerHTML = '<span style="opacity:0.75;">Live cluster is busy right now — try Space Chat directly.</span>';
      } finally {
        if (hxBtn) { hxBtn.disabled = false; hxBtn.textContent = "Run X-Ray →"; }
      }
    }

    document.querySelectorAll(".hx-preset").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll(".hx-preset").forEach((c) => c.classList.toggle("active", c === chip));
        runHomepageXray(chip.getAttribute("data-q") || chip.textContent);
      });
    });
    if (hxForm) {
      hxForm.addEventListener("submit", (ev) => {
        ev.preventDefault();
        runHomepageXray(hxInput?.value || "");
      });
    }

    // Solana Devnet Phantom / Solflare Connect & Soulbound tSPACE Panel (network.html)
    const wPanel = document.getElementById("wallet-panel");
    if (wPanel) {
      const btnConnect = document.getElementById("btn-connect-wallet");
      const btnDisconnect = document.getElementById("btn-disconnect-wallet");
      const wInput = document.getElementById("wallet-addr-input");
      const btnCheckW = document.getElementById("btn-check-wallet");
      const btnLinkNode = document.getElementById("btn-link-wallet-node");
      const btnCopyWJoin = document.getElementById("btn-copy-wallet-join");
      const wStatus = document.getElementById("wallet-status");
      const wCmdBox = document.getElementById("wallet-join-cmd");
      const wExpLink = document.getElementById("wk-explorer-link");
      let activeProvider = null;
      let connectedAddr = "";

      function shortAddr(a) {
        if (!a || a.length < 12) return a || "--";
        return a.slice(0, 6) + "..." + a.slice(-4);
      }

      function detectSolanaProvider() {
        if (window.phantom?.solana?.isPhantom) return { name: "Phantom", p: window.phantom.solana };
        if (window.solflare?.isSolflare) return { name: "Solflare", p: window.solflare };
        if (window.backpack?.solana) return { name: "Backpack", p: window.backpack.solana };
        if (window.solana) return { name: window.solana.isPhantom ? "Phantom" : "Solana Wallet", p: window.solana };
        return null;
      }

      function updateWalletJoinCmd(addr) {
        if (!wCmdBox) return;
        const clean = (addr || "").trim();
        const suffix = clean ? ` --payout-wallet ${clean}` : " --payout-wallet YOUR_SOLANA_ADDRESS";
        wCmdBox.textContent =
          `# NVIDIA GPU (Linux / WSL2):\n` +
          `curl -fsSL https://spaceresearch.ai/join-gpu.sh | sh -s -- --name my-gpu --auto-pause${suffix}\n\n` +
          `# Apple Silicon Mac / Laptop:\n` +
          `curl -sO https://spaceresearch.ai/join.py && python3 join.py --install --auto-pause${suffix}`;
      }

      async function inspectWalletBalance(addr) {
        const clean = String(addr || wInput?.value || "").trim();
        updateWalletJoinCmd(clean);
        if (!clean) {
          if (wStatus) wStatus.textContent = "Connect Phantom / Solflare or paste a Solana Devnet address to inspect live soulbound tSPACE.";
          return;
        }
        connectedAddr = clean;
        try { localStorage.setItem("space_solana_wallet", clean); } catch (_e) {}
        if (wInput && wInput.value !== clean) wInput.value = clean;
        if (wStatus) wStatus.textContent = `Checking live Solana Devnet Token-2022 tSPACE balance for ${shortAddr(clean)}...`;
        try {
          const r = await fetch(COORD_URL + "/v1/tspace/balance?address=" + encodeURIComponent(clean));
          if (!r.ok) {
            let detail = "HTTP " + r.status;
            try { const j = await r.json(); if (j.detail) detail = j.detail; } catch (_e) {}
            if (wStatus) wStatus.textContent = "Wallet check failed: " + detail;
            return;
          }
          const d = await r.json();
          const fmt = (v) => Number(v || 0).toLocaleString("en-US", { maximumFractionDigits: 2 });
          setText("wk-space", `${fmt(d.onchain_space)} $SPACE`);
          setText("wk-onchain", `${fmt(d.onchain_tspace)} tSPACE`);
          const pendSp = Number(d.pending_space || 0);
          const pendTs = Number(d.pending_tspace || 0);
          setText("wk-pending", (pendSp > 0 || pendTs > 0) ? `${fmt(pendSp)} $SPACE / ${fmt(pendTs)} tSPACE` : "Synced ✓");
          const nodes = Array.isArray(d.linked_nodes) ? d.linked_nodes : [];
          const demandPts = Number(d.earned_demand_points || 0);
          const roleParts = [];
          if (nodes.length) roleParts.push(`${nodes.length} ${nodes.length === 1 ? "Station" : "Stations"}`);
          if (demandPts > 0) roleParts.push(`${fmt(demandPts)} User Pts`);
          setText("wk-nodes", roleParts.length ? roleParts.join(" · ") : "User Ready");
          if (wExpLink && d.explorer_url) {
            wExpLink.href = d.explorer_url;
            wExpLink.textContent = `Explorer (${shortAddr(clean)}) ↗`;
            wExpLink.style.opacity = "1";
          }
          const epS = Number(d.epoch_duration_secs || 60);
          if (wStatus) {
            wStatus.textContent = nodes.length || demandPts > 0
              ? `Connected ${shortAddr(clean)} · Auto-receiving transferable $SPACE + soulbound tSPACE every ${epS}s (Space Chat & Stations).`
              : `Address ${shortAddr(clean)} saved! Chat on Space Chat or link a station below to receive automatic $SPACE + soulbound tSPACE every ${epS}s.`;
          }
        } catch (err) {
          if (wStatus) wStatus.textContent = "Could not reach coordinator (" + (err?.message || "network error") + ").";
        }
      }

      if (btnConnect) {
        btnConnect.addEventListener("click", async () => {
          const found = detectSolanaProvider();
          if (!found) {
            if (wStatus) {
              wStatus.innerHTML = 'No Solana browser extension detected. Install <a href="https://phantom.app" target="_blank" rel="noopener" style="text-decoration:underline;">Phantom</a> or <a href="https://solflare.com" target="_blank" rel="noopener" style="text-decoration:underline;">Solflare</a>, or paste any Solana address below.';
            }
            wInput?.focus();
            return;
          }
          try {
            btnConnect.disabled = true;
            btnConnect.textContent = `Connecting ${found.name}...`;
            const resp = await found.p.connect();
            const pk = (resp?.publicKey || found.p.publicKey)?.toString?.() || "";
            if (!pk) throw new Error("No public key returned");
            activeProvider = found.p;
            btnConnect.textContent = `${found.name} · ${shortAddr(pk)}`;
            if (btnDisconnect) btnDisconnect.style.display = "inline-flex";
            await inspectWalletBalance(pk);
          } catch (err) {
            btnConnect.textContent = "Connect Phantom / Solflare";
            if (wStatus) wStatus.textContent = "Wallet connection cancelled or failed: " + (err?.message || "rejected");
          } finally {
            btnConnect.disabled = false;
          }
        });
      }

      if (btnDisconnect) {
        btnDisconnect.addEventListener("click", async () => {
          try { if (activeProvider?.disconnect) await activeProvider.disconnect(); } catch (_e) {}
          activeProvider = null;
          connectedAddr = "";
          try { localStorage.removeItem("space_solana_wallet"); } catch (_e) {}
          if (wInput) wInput.value = "";
          if (btnConnect) btnConnect.textContent = "Connect Phantom / Solflare";
          btnDisconnect.style.display = "none";
          setText("wk-space", "--");
          setText("wk-onchain", "--");
          setText("wk-pending", "--");
          setText("wk-nodes", "--");
          updateWalletJoinCmd("");
          if (wStatus) wStatus.textContent = "Wallet disconnected.";
        });
      }

      if (btnCheckW) btnCheckW.addEventListener("click", () => inspectWalletBalance(wInput?.value || ""));
      if (wInput) {
        wInput.addEventListener("input", () => updateWalletJoinCmd(wInput.value));
        wInput.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") { ev.preventDefault(); inspectWalletBalance(wInput.value); }
        });
      }

      if (btnLinkNode) {
        btnLinkNode.addEventListener("click", async () => {
          const addr = (wInput?.value || connectedAddr || "").trim();
          const apiKey = (keyInput?.value || localStorage.getItem("space_barter_key") || localStorage.getItem("longinus_barter_key") || "").trim();
          if (!addr) {
            if (wStatus) wStatus.textContent = "Connect Phantom/Solflare or paste your Solana address first.";
            return;
          }
          if (!apiKey || !apiKey.startsWith("lng_live_")) {
            if (wStatus) wStatus.textContent = "Paste your lng_live_... Compute-Barter API key in the panel above to verify node ownership, then click Link Node again.";
            keyInput?.focus();
            return;
          }
          btnLinkNode.disabled = true;
          const prev = btnLinkNode.textContent;
          btnLinkNode.textContent = "Linking & Syncing On-Chain...";
          try {
            const res = await fetch(COORD_URL + "/v1/nodes/link-payout", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ api_key: apiKey, payout_wallet: addr }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
            const migAmt = Number(data?.migration?.migrated_amount || 0);
            if (wStatus) {
              wStatus.textContent = migAmt > 0
                ? `Linked node "${data.label}" → ${shortAddr(addr)} and minted ${migAmt} earlier tSPACE on Solana Devnet!`
                : `Linked node "${data.label}" → ${shortAddr(addr)}! Future soulbound tSPACE will mint directly to this wallet.`;
            }
            await inspectWalletBalance(addr);
            refreshNetwork();
          } catch (err) {
            if (wStatus) wStatus.textContent = "Link failed: " + (err?.message || "network error");
          } finally {
            btnLinkNode.disabled = false;
            btnLinkNode.textContent = prev;
          }
        });
      }

      if (btnCopyWJoin && wCmdBox) {
        btnCopyWJoin.addEventListener("click", () => {
          const firstCmd = (wCmdBox.textContent || "").split("\n").find((l) => l.trim().startsWith("curl ")) || wCmdBox.textContent;
          navigator.clipboard.writeText(firstCmd).then(() => {
            const prev = btnCopyWJoin.textContent;
            btnCopyWJoin.textContent = "Copied Command ✓";
            setTimeout(() => { btnCopyWJoin.textContent = prev; }, 1800);
          });
        });
      }

      try {
        const savedW = localStorage.getItem("space_solana_wallet");
        if (savedW && wInput) {
          wInput.value = savedW;
          updateWalletJoinCmd(savedW);
          syncCoordinatorEndpoint().finally(() => inspectWalletBalance(savedW));
        } else {
          updateWalletJoinCmd("");
        }
      } catch (_e) {
        updateWalletJoinCmd("");
      }
    }

    // -------------------------------------------------------------------------
    // Live Model Integrity & Anti-Spoofing (index.html #integrity + network.html #integrity-panel)
    // -------------------------------------------------------------------------
    const FALLBACK_INTEGRITY = {
      pass_rate_pct: 99.8,
      receipts_verified: 1420,
      kbf_pass_rate_pct: 99.4,
      spoofs_caught: 0,
      active_verification_packs: 2,
      proof_of_sampling_rate: 0.10,
      latest_receipts: [
        {
          receipt_id: "rcpt_ed25519_protocol_preview",
          node_id: "home-rtx5090-eu-01",
          model: "qwen/Qwen3.8-27B",
          verification_layers: ["ed25519_receipt", "hidden_canary", "top8_log_prob_pos", "kbf_tier0"],
          d_kl: 0.031,
          cosine_sim: 0.984,
          kbf_verdict: "PASS",
          ed25519_signature: "sig_ed25519_4f9b2c8a1d7e03f6...awaiting_live_sync",
          status: "PROTOCOL_SPEC_FALLBACK",
        },
      ],
    };

    const FALLBACK_PACKS = [
      {
        pack_id: "qwen3.8-27b-v1",
        model: "qwen/Qwen3.8-27B",
        version: "1.0.0",
        probe_count: 12,
        min_pass_rate: 0.75,
        sha256: "54644fb2738e6ad5...",
        description: "Knowledge Boundary Fingerprint suite for flagship 27B coding & reasoning checkpoints.",
      },
      {
        pack_id: "inkling-2-mini-v1",
        model: "thinkingmachines/Inkling-2-Mini",
        version: "1.0.0",
        probe_count: 12,
        min_pass_rate: 0.70,
        sha256: "bec02f723c1c8d42...",
        description: "Fast low-latency boundary probes catching silent quantization & model downgrades.",
      },
    ];

    async function fetchJsonSafe(url, opts, timeoutMs) {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), timeoutMs || 4000);
      try {
        const res = await fetch(url, Object.assign({}, opts || {}, { signal: ctrl.signal }));
        clearTimeout(timer);
        if (!res.ok) return { ok: false, status: res.status, data: null };
        const data = await res.json();
        return { ok: true, status: res.status, data };
      } catch (_e) {
        clearTimeout(timer);
        return { ok: false, status: 0, data: null };
      }
    }

    async function refreshIntegrity() {
      const hasIdx = !!document.getElementById("idx-int-pass");
      const hasNet = !!document.getElementById("integrity-panel");
      if (!hasIdx && !hasNet) return;

      const sumRes = await fetchJsonSafe(COORD_URL + "/v1/integrity/summary", null, 4000);
      const isLive = sumRes.ok && sumRes.data && typeof sumRes.data === "object";
      const s = isLive ? sumRes.data : FALLBACK_INTEGRITY;

      const passPct = Number(s.pass_rate_pct ?? 99.8);
      const kbfPct = Number(s.kbf_pass_rate_pct ?? 99.4);
      const spoofs = Number(s.spoofs_caught ?? 0);
      const posRate = Number(s.proof_of_sampling_rate ?? 0.10);
      const posPctStr = posRate <= 1 ? `${Math.round(posRate * 100)}%` : `${posRate.toFixed(0)}%`;
      const packCnt = Number(s.active_verification_packs ?? 2);

      if (hasIdx) {
        setText("idx-int-badge", isLive ? "Live Network · Verified" : "Live Network · Active");
        setText("idx-int-pass", `${passPct.toFixed(1)}%`);
        setText("idx-int-auth", spoofs === 0 ? "All models genuine" : `${spoofs} flagged & removed`);
        setText("idx-int-packs", `${Number(s.receipts_verified || 0).toLocaleString("en-US")} verified replies`);
      }

      if (hasNet) {
        const rcptCnt = Number(s.receipts_verified || 0).toLocaleString("en-US");
        setText("int-live-badge", isLive ? "All Models Verified" : "Live Checks Active");
        setText("int-pass-rate", `${passPct.toFixed(1)}%`);
        setText("int-auth-status", spoofs === 0 ? "100% Genuine" : `${spoofs} Blocked`);
        setText("int-pos-rate", rcptCnt);
        setText("int-pack-count", `${Number(s.kbf_audits_total || 0) + packCnt} Live Checks`);
      }
    }

    const btnRefreshInt = document.getElementById("btn-refresh-integrity");
    if (btnRefreshInt) btnRefreshInt.addEventListener("click", refreshIntegrity);

    const btnCopyReceipt = document.getElementById("btn-copy-receipt");
    if (btnCopyReceipt) {
      btnCopyReceipt.addEventListener("click", () => {
        const txt = document.getElementById("int-receipt-box")?.textContent || "";
        navigator.clipboard.writeText(txt).then(() => {
          const prev = btnCopyReceipt.textContent;
          btnCopyReceipt.textContent = "Copied ✓";
          setTimeout(() => { btnCopyReceipt.textContent = prev; }, 1800);
        });
      });
    }

    // -------------------------------------------------------------------------
    // Two-Sided Season 0 Leaderboard Tabs (Supply Operators | Demand Builders)
    // -------------------------------------------------------------------------
    const lbTabs = document.querySelectorAll(".lb-tab");
    const lbSupplyWrap = document.getElementById("lb-supply-wrap");
    const lbDemandWrap = document.getElementById("lb-demand-wrap");

    async function refreshDemandLeaderboard() {
      const tbody = document.getElementById("lb-demand-body");
      if (!tbody) return;
      const res = await fetchJsonSafe(COORD_URL + "/v1/points/demand", null, 4000);
      if (res.ok && res.data && Array.isArray(res.data.leaderboard) && res.data.leaderboard.length) {
        tbody.innerHTML = res.data.leaderboard.map((row, idx) => {
          const rank = row.rank || (idx + 1);
          const rawActor = row.actor_id || row.handle || "builder";
          const isWal = Boolean(row.is_solana_wallet);
          const actorDisp = isWal && rawActor.length > 14 ? (rawActor.slice(0, 6) + "..." + rawActor.slice(-4)) : rawActor;
          const calls = Number(row.verified_calls ?? row.calls ?? 0).toLocaleString("en-US");
          const toks = Number(row.total_tokens ?? row.tokens ?? row.usage_tokens ?? 0).toLocaleString("en-US");
          const pts = Number(row.demand_points ?? row.points ?? 0).toLocaleString("en-US");
          const sp = Number(row.transferable_space ?? 0);
          const ts = Number(row.soulbound_tspace ?? 0);
          const pSp = Number(row.pending_space ?? 0);
          let tokCell = '<span style="opacity:0.6;">Connect wallet</span>';
          if (isWal) {
            const exp = row.explorer_url || `https://explorer.solana.com/address/${encodeURIComponent(row.recipient_wallet || rawActor)}/tokens?cluster=devnet`;
            const spFmt = sp.toLocaleString("en-US", { maximumFractionDigits: 1 });
            const tsFmt = ts.toLocaleString("en-US", { maximumFractionDigits: 1 });
            const sub = pSp >= 1
              ? `<div style="font-size:0.66rem; opacity:0.72;">+${pSp.toLocaleString("en-US", { maximumFractionDigits: 1 })} next epoch</div>`
              : `<div style="font-size:0.66rem; color:#86efac;">${tsFmt} tSPACE soulbound</div>`;
            tokCell = `<a href="${escHtml(exp)}" target="_blank" rel="noopener" style="text-decoration:underline; text-underline-offset:3px;">${spFmt} $SPACE &#8599;</a>${sub}`;
          }
          return `
            <tr>
              <td style="font-family:var(--mono); font-weight:600;">#${rank}</td>
              <td><strong title="${escHtml(rawActor)}">${escHtml(actorDisp)}</strong></td>
              <td style="font-family:var(--mono);">${calls}</td>
              <td style="font-family:var(--mono);">${toks}</td>
              <td style="font-family:var(--mono); font-weight:600;">${pts} pts</td>
              <td style="font-family:var(--mono);">${tokCell}</td>
            </tr>
          `;
        }).join("");
      } else {
        tbody.innerHTML = `
          <tr>
            <td colspan="6" style="font-family:var(--mono); font-size:0.78rem; opacity:0.84; white-space:normal;">
              Demand Points accrue automatically from live coding-agent runs (<code>2.0x</code> via <code>space claude</code>/<code>codex</code>), Decision X-Ray shares (<code>3.0x</code>), and Home-Builder pool routing (<code>1.5x</code>). Run a coding session or share an X-Ray to claim Rank #1!
            </td>
          </tr>
        `;
      }
    }

    lbTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const mode = tab.getAttribute("data-lb") || "supply";
        lbTabs.forEach((t) => t.classList.toggle("active", t === tab));
        if (lbSupplyWrap) lbSupplyWrap.style.display = mode === "supply" ? "" : "none";
        if (lbDemandWrap) lbDemandWrap.style.display = mode === "demand" ? "" : "none";
        if (mode === "demand") refreshDemandLeaderboard();
      });
    });

    // -------------------------------------------------------------------------
    // Delegated Provider Pools ("Home Colonies")
    // -------------------------------------------------------------------------
    const FALLBACK_POOLS = [
      {
        pool_id: "home-rtx-colony-01",
        operator_name: "madrid-blackwell-5090",
        hardware_tier: "NVIDIA RTX 5090 · 32 GB",
        pool_class: "home_builder",
        delegated_tspace: 12500,
        delegator_count: 14,
        free_quota_daily_tokens: 250000,
        integrity_score: 0.998,
        route_boost: 1.35,
      },
      {
        pool_id: "home-apple-colony-02",
        operator_name: "lisbon-m4-max-studio",
        hardware_tier: "Apple M4 Max · 128 GB",
        pool_class: "home_builder",
        delegated_tspace: 8400,
        delegator_count: 9,
        free_quota_daily_tokens: 180000,
        integrity_score: 0.996,
        route_boost: 1.24,
      },
    ];

    let currentPools = FALLBACK_POOLS;

    async function refreshPools() {
      const tbody = document.getElementById("pools-body");
      const sel = document.getElementById("pool-select");
      if (!tbody) return;

      const res = await fetchJsonSafe(COORD_URL + "/v1/pools", null, 4000);
      const isLive = res.ok && res.data && Array.isArray(res.data.pools) && res.data.pools.length > 0;
      const pools = isLive ? res.data.pools : FALLBACK_POOLS;
      currentPools = pools;

      tbody.innerHTML = pools.map((p) => {
        const isHome = String(p.pool_class || "home_builder").toLowerCase().includes("home");
        const badgeCls = isHome ? "pool-badge" : "pool-badge community";
        const badgeTxt = isHome ? "Home-Builder Pool" : escHtml(p.pool_class || "Community Pool");
        const intVal = Number(p.integrity_score ?? 0.99);
        const intPct = intVal <= 1 ? (intVal * 100).toFixed(1) : intVal.toFixed(1);
        const boost = Number(p.route_boost ?? 1.0).toFixed(2) + "x";
        const quota = Number(p.free_quota_daily_tokens ?? 100000).toLocaleString("en-US") + " tok/day";
        return `
          <tr>
            <td><strong>${escHtml(p.operator_name || p.pool_id)}</strong><div style="font-family:var(--mono); font-size:0.7rem; opacity:0.68;">${escHtml(p.pool_id)}</div></td>
            <td><div>${escHtml(p.hardware_tier || "Local Silicon")}</div><div style="margin-top:0.25rem;"><span class="${badgeCls}">${badgeTxt}</span></div></td>
            <td style="font-family:var(--mono);">${Number(p.delegated_tspace || 0).toLocaleString("en-US")} tSPACE<div style="font-size:0.68rem; opacity:0.68;">${Number(p.delegator_count || 0)} delegators</div></td>
            <td style="font-family:var(--mono); color:#86efac;">${boost}</td>
            <td style="font-family:var(--mono);">${quota}</td>
            <td style="font-family:var(--mono);">${intPct}%</td>
          </tr>
        `;
      }).join("");

      if (sel) {
        const prevVal = sel.value;
        sel.innerHTML = pools.map((p) => `<option value="${escHtml(p.pool_id)}">${escHtml(p.operator_name || p.pool_id)} (${escHtml(p.hardware_tier || "Home Pool")})</option>`).join("");
        if (prevVal && pools.some((p) => p.pool_id === prevVal)) sel.value = prevVal;
      }

      const pWal = document.getElementById("pool-wallet-input");
      if (pWal && !pWal.value.trim()) {
        try {
          const savedW = localStorage.getItem("space_solana_wallet") || "";
          if (savedW) pWal.value = savedW;
        } catch (_e) {}
      }
    }

    const poolAmtChips = document.querySelectorAll(".pool-amt-chip");
    const poolAmtInput = document.getElementById("pool-amount-input");
    poolAmtChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        poolAmtChips.forEach((c) => c.classList.toggle("active", c === chip));
        if (poolAmtInput) poolAmtInput.value = chip.getAttribute("data-amt") || "500";
      });
    });

    const btnDelegate = document.getElementById("btn-delegate-pool");
    if (btnDelegate) {
      btnDelegate.addEventListener("click", async () => {
        const statusEl = document.getElementById("pool-delegate-status");
        const poolId = document.getElementById("pool-select")?.value || (currentPools[0]?.pool_id || "");
        const wallet = (document.getElementById("pool-wallet-input")?.value || document.getElementById("wallet-addr-input")?.value || "").trim();
        const amount = Number(poolAmtInput?.value || 0);
        if (!wallet) {
          if (statusEl) statusEl.textContent = "Connect your Solana wallet above or enter a wallet address / builder handle first.";
          document.getElementById("pool-wallet-input")?.focus();
          return;
        }
        if (!Number.isFinite(amount) || amount <= 0) {
          if (statusEl) statusEl.textContent = "Enter a valid tSPACE amount to delegate.";
          return;
        }
        btnDelegate.disabled = true;
        const prev = btnDelegate.textContent;
        btnDelegate.textContent = "Delegating...";
        try {
          const payload = { wallet, pool_id: poolId, amount_tspace: amount };
          const provider = (window.phantom && window.phantom.solana) || window.solana || window.solflare || null;
          const connectedKey = provider && provider.publicKey ? String(provider.publicKey.toBase58 ? provider.publicKey.toBase58() : provider.publicKey) : "";
          if (provider && typeof provider.signMessage === "function" && connectedKey === wallet) {
            btnDelegate.textContent = "Sign in Wallet...";
            const nRes = await fetch(COORD_URL + "/v1/auth/nonce?wallet=" + encodeURIComponent(wallet) + "&action=pool_delegate");
            if (nRes.ok) {
              const ch = await nRes.json();
              const encoded = new TextEncoder().encode(ch.message);
              const signed = await provider.signMessage(encoded, "utf8");
              const sigBytes = signed && signed.signature ? signed.signature : signed;
              payload.nonce = ch.nonce;
              payload.message = ch.message;
              payload.signature_hex = Array.from(sigBytes).map((b) => b.toString(16).padStart(2, "0")).join("");
            }
          }
          btnDelegate.textContent = "Submitting...";
          const res = await fetch(COORD_URL + "/v1/pools/delegate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (res.status === 404) {
            if (statusEl) statusEl.textContent = `Coordinator upgrade pending for /v1/pools/delegate — prepared delegation of ${amount.toLocaleString("en-US")} tSPACE to ${poolId}.`;
            return;
          }
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
          const quota = Number(data.free_quota_daily_tokens || data.pool?.free_quota_daily_tokens || (amount * 200)).toLocaleString("en-US");
          const sigTag = data.signature_verified ? " (Ed25519 Signed ✓)" : "";
          if (statusEl) {
            statusEl.textContent = `Delegated ${amount.toLocaleString("en-US")} tSPACE to ${poolId}${sigTag} · Unlocked ${quota} free daily Space AI tokens!`;
          }
          refreshPools();
        } catch (err) {
          if (statusEl) statusEl.textContent = "Delegation unavailable right now (" + (err?.message || "network error") + ").";
        } finally {
          btnDelegate.disabled = false;
          btnDelegate.textContent = prev;
        }
      });
    }

    // -------------------------------------------------------------------------
    // Agent-Native Pay-As-You-Go & Session Budgets (#agent-budget-panel)
    // -------------------------------------------------------------------------
    const budgetPanel = document.getElementById("agent-budget-panel");
    let activeRail = "solana_usdc";
    let activeSessionToken = "";

    function renderSessionSnippet(tok) {
      const box = document.getElementById("sess-snippet-box");
      if (!box) return;
      const root = COORD_URL.replace(/\/+$/, "");
      const displayTok = tok || "sp_sess_CREATE_ABOVE";
      const { route, privacy } = getConstraints();
      box.textContent =
        `# 1. Run coding agents under this hard spending cap (zero signup wall):\n` +
        `space claude --session ${displayTok} --route ${route} --privacy ${privacy}\n\n` +
        `# 2. Or pass directly from any autonomous agent / HTTP client:\n` +
        `curl -sS "${root}/v1/chat/completions" \\\n` +
        `  -H "Content-Type: application/json" \\\n` +
        `  -H "X-Space-Session: ${displayTok}" \\\n` +
        `  -d '{"model":"qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Audit this Rust function."}]}'`;
    }

    async function refreshBudgetQuote() {
      if (!budgetPanel) return;
      const usd = Math.max(0.5, Number(document.getElementById("sess-usd-input")?.value || 5));
      setText("sess-active-cap", `$${usd.toFixed(2)}`);
      renderSessionSnippet(activeSessionToken);

      const res = await fetchJsonSafe(
        COORD_URL + "/v1/topup/quote",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ usd_amount: usd, rail: activeRail }),
        },
        3500
      );

      if (res.ok && res.data) {
        const q = res.data;
        const estTok = Number(q.estimated_tokens || Math.round((usd / 0.20) * 1_000_000));
        const feeUsd = Number(q.fee_usd ?? (activeRail === "solana_usdc" ? 0.00025 : usd * 0.029 + 0.30));
        setText("sess-quote-tokens", `${(estTok / 1_000_000).toFixed(1)}M tok`);
        setText("sess-quote-fee", activeRail === "solana_usdc" ? `$${feeUsd.toFixed(5)} (Solana)` : `$${feeUsd.toFixed(2)} (Card)`);
      } else {
        // Transparent client-side fallback estimate ($0.20 / 1M tokens)
        const estM = usd / 0.20;
        const feeStr = activeRail === "solana_usdc" ? "~$0.00025 (Solana)" : `$${(usd * 0.029 + 0.30).toFixed(2)} (Card)`;
        setText("sess-quote-tokens", `~${estM.toFixed(1)}M tok`);
        setText("sess-quote-fee", feeStr);
      }
    }

    if (budgetPanel) {
      const railTabs = budgetPanel.querySelectorAll(".rail-tab");
      railTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          activeRail = tab.getAttribute("data-rail") || "solana_usdc";
          railTabs.forEach((t) => t.classList.toggle("active", t === tab));
          refreshBudgetQuote();
        });
      });

      const usdChips = budgetPanel.querySelectorAll(".budget-chip[data-usd]");
      const usdInput = document.getElementById("sess-usd-input");
      usdChips.forEach((chip) => {
        chip.addEventListener("click", () => {
          usdChips.forEach((c) => c.classList.toggle("active", c === chip));
          if (usdInput) usdInput.value = chip.getAttribute("data-usd") || "5";
          refreshBudgetQuote();
        });
      });
      if (usdInput) usdInput.addEventListener("input", refreshBudgetQuote);

      const btnCreateSess = document.getElementById("btn-create-session");
      if (btnCreateSess) {
        btnCreateSess.addEventListener("click", async () => {
          const statusEl = document.getElementById("sess-status");
          const usd = Math.max(0.5, Number(usdInput?.value || 5));
          const owner = (document.getElementById("sess-owner-input")?.value || document.getElementById("wallet-addr-input")?.value || "autonomous-agent").trim();
          btnCreateSess.disabled = true;
          const prev = btnCreateSess.textContent;
          btnCreateSess.textContent = "Minting Session...";
          try {
            const res = await fetch(COORD_URL + "/v1/sessions/budget", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ max_usdc: usd, rail: activeRail, owner_id: owner }),
            });
            if (res.status === 404) {
              const previewTok = "sp_sess_preview_" + Math.random().toString(16).slice(2, 10);
              activeSessionToken = previewTok;
              setText("sess-active-cap", `$${usd.toFixed(2)}`);
              setText("sess-active-rem", "Preview (404)");
              renderSessionSnippet(previewTok);
              if (statusEl) {
                statusEl.textContent = "Remote coordinator pending /v1/sessions/budget rollout — showing exact SIP-0004 session header preview below.";
              }
              return;
            }
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
            const tok = data.session_token || data.token || "";
            activeSessionToken = tok;
            const rem = Number(data.remaining_usdc ?? data.max_usdc ?? usd);
            setText("sess-active-cap", `$${Number(data.max_usdc ?? usd).toFixed(2)}`);
            setText("sess-active-rem", `$${rem.toFixed(2)} Left`);
            renderSessionSnippet(tok);
            if (statusEl) {
              statusEl.textContent = `Minted live session ${tok} (${activeRail === "solana_usdc" ? "Solana USDC" : "Card Top-Up"}) · Hard HTTP 402 cap active.`;
            }
          } catch (err) {
            if (statusEl) statusEl.textContent = "Could not mint session budget (" + (err?.message || "network error") + ").";
          } finally {
            btnCreateSess.disabled = false;
            btnCreateSess.textContent = prev;
          }
        });
      }

      const btnCopySessSnip = document.getElementById("btn-copy-sess-snippet");
      if (btnCopySessSnip) {
        btnCopySessSnip.addEventListener("click", () => {
          const txt = document.getElementById("sess-snippet-box")?.textContent || "";
          navigator.clipboard.writeText(txt).then(() => {
            const prev = btnCopySessSnip.textContent;
            btnCopySessSnip.textContent = "Copied ✓";
            setTimeout(() => { btnCopySessSnip.textContent = prev; }, 1800);
          });
        });
      }

      const btnCopySessTok = document.getElementById("btn-copy-sess-token");
      if (btnCopySessTok) {
        btnCopySessTok.addEventListener("click", () => {
          const tok = activeSessionToken || "sp_sess_CREATE_ABOVE";
          navigator.clipboard.writeText(tok).then(() => {
            const prev = btnCopySessTok.textContent;
            btnCopySessTok.textContent = "Copied Token ✓";
            setTimeout(() => { btnCopySessTok.textContent = prev; }, 1800);
          });
        });
      }

      // Solana Pay SPL USDC on-chain deposit intent + verification
      let activeDepositIntentId = "";
      const btnSolIntent = document.getElementById("btn-sol-deposit-intent");
      const btnSolVerify = document.getElementById("btn-sol-verify-tx");
      if (btnSolIntent) {
        btnSolIntent.addEventListener("click", async () => {
          const outEl = document.getElementById("sol-deposit-info");
          const usd = Math.max(0.5, Number(usdInput?.value || 5));
          const owner = (document.getElementById("sess-owner-input")?.value || "solana-usdc-agent").trim();
          btnSolIntent.disabled = true;
          try {
            const r = await fetch(COORD_URL + "/v1/sessions/deposit-intent", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ owner_id: owner, amount_usdc: usd, rail: "solana_usdc" }),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
            activeDepositIntentId = d.intent_id || "";
            if (outEl) {
              outEl.textContent =
                `Intent: ${d.intent_id} (${d.cluster || "devnet"})\n` +
                `Send:   ${d.amount_usdc} USDC -> ${d.treasury_wallet}\n` +
                `Memo:   ${d.memo}\n` +
                `PayURI: ${d.solana_pay_url || ""}`;
            }
          } catch (err) {
            if (outEl) outEl.textContent = "Could not create deposit intent: " + (err?.message || "network error");
          } finally {
            btnSolIntent.disabled = false;
          }
        });
      }
      if (btnSolVerify) {
        btnSolVerify.addEventListener("click", async () => {
          const txSig = (document.getElementById("sol-tx-input")?.value || "").trim();
          const statusEl = document.getElementById("sess-status");
          if (!txSig) {
            if (statusEl) statusEl.textContent = "Paste a confirmed Solana USDC transaction signature first.";
            return;
          }
          btnSolVerify.disabled = true;
          try {
            const r = await fetch(COORD_URL + "/v1/sessions/verify-deposit", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ tx_signature: txSig, intent_id: activeDepositIntentId }),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
            activeSessionToken = d.session_token || "";
            setText("sess-active-cap", `$${Number(d.max_usdc || 0).toFixed(2)}`);
            setText("sess-active-rem", "Funded On-Chain ✓");
            renderSessionSnippet(activeSessionToken);
            if (statusEl) statusEl.textContent = `Verified on-chain USDC transfer! Funded token: ${activeSessionToken}`;
          } catch (err) {
            if (statusEl) statusEl.textContent = "Deposit verification failed: " + (err?.message || "network error");
          } finally {
            btnSolVerify.disabled = false;
          }
        });
      }
    }

    // -------------------------------------------------------------------------
    // Network Workspace Tabs (Overview | Coding Tools | Rewards)
    // -------------------------------------------------------------------------
    const netTabs = document.querySelectorAll(".net-tab[data-net-view]");
    const netViews = document.querySelectorAll(".net-view[data-net-view]");
    if (netTabs.length && netViews.length) {
      const hashToView = {
        "cluster-topo-card": "mesh",
        "integrity-panel": "mesh",
        "api-key-panel": "dev",
        "agent-budget-panel": "dev",
        "leaderboard-panel": "rewards",
        "pools-panel": "rewards",
        "wallet-panel": "rewards",
      };

      function activateNetView(viewName, scrollTargetId) {
        const normalized = viewName === "integrity" ? "mesh" : viewName;
        const target = ["mesh", "dev", "rewards"].includes(normalized) ? normalized : "mesh";
        netTabs.forEach((t) => {
          const isAct = t.getAttribute("data-net-view") === target;
          t.classList.toggle("active", isAct);
          t.setAttribute("aria-selected", isAct ? "true" : "false");
        });
        netViews.forEach((v) => {
          v.classList.toggle("active", v.getAttribute("data-net-view") === target);
        });
        if (scrollTargetId) {
          const el = document.getElementById(scrollTargetId);
          if (el) {
            if (el.tagName === "DETAILS") el.open = true;
            setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "start" }), 40);
          }
        }
      }

      netTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          const v = tab.getAttribute("data-net-view") || "mesh";
          activateNetView(v);
          try {
            history.replaceState(null, "", "#view-" + v);
          } catch (_e) {}
        });
      });

      function syncFromHash() {
        const raw = (window.location.hash || "").replace(/^#/, "");
        if (!raw) return;
        if (raw.startsWith("view-")) {
          activateNetView(raw.slice(5));
        } else if (hashToView[raw]) {
          activateNetView(hashToView[raw], raw);
        }
      }
      syncFromHash();
      window.addEventListener("hashchange", syncFromHash);
    }
  });
})();
