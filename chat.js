/* Space Chat — Streaming Multi-Turn Client (OpenAI /v1/chat/completions SSE) */
(() => {
  let API_BASE = "https://api.spaceresearch.ai";
  const FALLBACK_API_BASE = "https://api.longinusresearch.com";
  const STORAGE_KEY = "space_chat_sessions_v1";
  const LEGACY_STORAGE_KEY = "longinus_chat_sessions_v1";
  const XRAY_PREF_KEY = "space_xray_v1";

  let xrayEnabled = true;
  try {
    const savedXray = localStorage.getItem(XRAY_PREF_KEY) ?? localStorage.getItem("longinus_xray_v1");
    if (savedXray === "0") xrayEnabled = false;
  } catch (_e) {}

  async function syncCoordinatorEndpoint() {
    try {
      const r = await fetch("./coordinator.json", { cache: "no-store" });
      if (r.ok) {
        const d = await r.json();
        if (d && typeof d.api_base === "string" && d.api_base.startsWith("http")) {
          API_BASE = d.api_base.replace(/\/+$/, "");
        }
      }
    } catch (_e) {}
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 3500);
      const ping = await fetch(API_BASE + "/health", { signal: ctrl.signal });
      clearTimeout(t);
      if (ping.ok) return;
    } catch (_e) {
      if (API_BASE !== FALLBACK_API_BASE) {
        API_BASE = FALLBACK_API_BASE;
      }
    }
  }

  let sessions = loadSessions();
  let activeId = sessions[0]?.id || null;
  let activeAbort = null;
  let isStreaming = false;
  let pendingImages = [];

  const $ = (id) => document.getElementById(id);

  function resizeImageFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read image"));
      reader.onload = () => {
        const img = new Image();
        img.onerror = () => reject(new Error("Invalid image"));
        img.onload = () => {
          const maxDim = 1280;
          let w = img.naturalWidth || img.width;
          let h = img.naturalHeight || img.height;
          if (w > maxDim || h > maxDim) {
            const scale = maxDim / Math.max(w, h);
            w = Math.max(1, Math.round(w * scale));
            h = Math.max(1, Math.round(h * scale));
          }
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL("image/jpeg", 0.86));
        };
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  function renderAttachStrip() {
    const strip = $("attach-strip");
    if (!strip) return;
    strip.innerHTML = "";
    strip.classList.toggle("has-items", pendingImages.length > 0);
    pendingImages.forEach((url, idx) => {
      const card = document.createElement("div");
      card.className = "attach-thumb";
      card.innerHTML = `<img src="${url}" alt="Attachment" /><button type="button" class="attach-rm" title="Remove">&times;</button>`;
      card.querySelector(".attach-rm").addEventListener("click", () => {
        pendingImages.splice(idx, 1);
        renderAttachStrip();
      });
      strip.appendChild(card);
    });
  }

  async function addImageFiles(fileList) {
    const files = Array.from(fileList || []).filter((f) => f && f.type && f.type.startsWith("image/"));
    for (const f of files) {
      if (pendingImages.length >= 4) break;
      try {
        const dataUrl = await resizeImageFile(f);
        pendingImages.push(dataUrl);
      } catch (_e) {}
    }
    renderAttachStrip();
  }

  function uid() {
    return "c_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7);
  }

  function loadSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (_e) {
      return [];
    }
  }

  function saveSessions() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 40)));
    } catch (_e) {
      try {
        const trimmed = sessions.slice(0, 25).map((s, si) => ({
          ...s,
          messages: (s.messages || []).map((m, i, arr) => ({
            ...m,
            images: m.images && i < arr.length - 4 ? undefined : m.images,
            signals: m.signals && (si > 2 || i < arr.length - 4) ? undefined : m.signals,
          })),
        }));
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
      } catch (_e2) {}
    }
  }

  function getActiveSession() {
    return sessions.find((s) => s.id === activeId) || null;
  }

  function preloadModel(modelId) {
    if (!modelId) return;
    fetch(API_BASE + "/v1/models/preload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelId }),
    }).catch(() => {});
  }

  function createSession(initialTitle = "New chat") {
    const s = {
      id: uid(),
      title: initialTitle,
      model: $("model-select")?.value || "qwen3.8-27b",
      targetNode: $("node-select")?.value || "auto",
      messages: [],
      updatedAt: Date.now(),
    };
    sessions.unshift(s);
    activeId = s.id;
    saveSessions();
    renderSidebar();
    renderThread();
    closeMobileDrawer();
    $("composer-input")?.focus();
    return s;
  }

  function deleteSession(id) {
    sessions = sessions.filter((s) => s.id !== id);
    if (activeId === id) activeId = sessions[0]?.id || null;
    saveSessions();
    renderSidebar();
    renderThread();
  }

  function escHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inlineMd(line) {
    let s = escHtml(line);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return s;
  }

  function renderMarkdown(raw) {
    const src = String(raw || "").replace(/\r\n/g, "\n");
    if (!src.trim()) return "";
    const lines = src.split("\n");
    const out = [];
    let inCode = false;
    let codeLang = "";
    let codeBuf = [];
    let listTag = null;
    let tableRows = [];

    function flushList() {
      if (listTag) { out.push(`</${listTag}>`); listTag = null; }
    }
    function flushTable() {
      if (!tableRows.length) return;
      const rows = tableRows.filter((r) => !/^\|?[\s:-]+\|[\s|:-]*$/.test(r.trim()));
      if (rows.length) {
        out.push('<div class="md-table-wrap"><table>');
        rows.forEach((r, idx) => {
          const cells = r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
          const tag = idx === 0 ? "th" : "td";
          out.push("<tr>" + cells.map((c) => `<${tag}>${inlineMd(c)}</${tag}>`).join("") + "</tr>");
        });
        out.push("</table></div>");
      }
      tableRows = [];
    }
    function flushCode() {
      const codeStr = codeBuf.join("\n");
      const langLbl = escHtml(codeLang || "code");
      const enc = encodeURIComponent(codeStr);
      out.push(
        `<div class="code-block"><div class="code-hdr"><span>${langLbl}</span>` +
        `<button type="button" class="code-copy" data-code="${enc}">Copy</button></div>` +
        `<pre><code>${escHtml(codeStr)}</code></pre></div>`
      );
      inCode = false;
      codeLang = "";
      codeBuf = [];
    }

    for (const line of lines) {
      const fence = line.trim().match(/^```([a-zA-Z0-9_+#.-]*)/);
      if (fence) {
        flushList();
        flushTable();
        if (!inCode) { inCode = true; codeLang = fence[1] || "code"; codeBuf = []; }
        else flushCode();
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }

      const trimmed = line.trim();
      if (trimmed.includes("|") && trimmed.startsWith("|")) {
        flushList();
        tableRows.push(trimmed);
        continue;
      }
      flushTable();

      if (!trimmed) { flushList(); continue; }
      if (/^(-{3,}|\*{3,})$/.test(trimmed)) { flushList(); out.push("<hr/>"); continue; }

      const h = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (h) { flushList(); out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); continue; }

      const bq = trimmed.match(/^>\s?(.*)$/);
      if (bq) { flushList(); out.push(`<blockquote>${inlineMd(bq[1])}</blockquote>`); continue; }

      const ul = trimmed.match(/^[-*•]\s+(.*)$/);
      if (ul) {
        if (listTag !== "ul") { flushList(); listTag = "ul"; out.push("<ul>"); }
        out.push(`<li>${inlineMd(ul[1])}</li>`);
        continue;
      }
      const ol = trimmed.match(/^\d+\.\s+(.*)$/);
      if (ol) {
        if (listTag !== "ol") { flushList(); listTag = "ol"; out.push("<ol>"); }
        out.push(`<li>${inlineMd(ol[1])}</li>`);
        continue;
      }
      flushList();
      out.push(`<p>${inlineMd(trimmed)}</p>`);
    }
    if (inCode) flushCode();
    flushList();
    flushTable();
    return out.join("");
  }

  function startInlineRename(div, session) {
    const titleEl = div.querySelector(".chat-item-title");
    if (!titleEl || div.querySelector(".chat-item-input")) return;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "chat-item-input";
    input.value = session.title || "New chat";
    input.maxLength = 80;
    titleEl.replaceWith(input);
    input.focus();
    input.select();

    let finished = false;
    function commit(save) {
      if (finished) return;
      finished = true;
      if (save) {
        const nextTitle = input.value.trim();
        if (nextTitle) {
          session.title = nextTitle;
          session.updatedAt = Date.now();
          saveSessions();
        }
      }
      renderSidebar();
    }
    input.addEventListener("click", (ev) => ev.stopPropagation());
    input.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Enter") { ev.preventDefault(); commit(true); }
      else if (ev.key === "Escape") { ev.preventDefault(); commit(false); }
    });
    input.addEventListener("blur", () => commit(true));
  }

  function renderSidebar() {
    const listEl = $("sb-list");
    if (!listEl) return;
    const q = ($("sb-search")?.value || "").trim().toLowerCase();
    const filtered = q ? sessions.filter((s) => s.title.toLowerCase().includes(q)) : sessions;
    listEl.innerHTML = "";
    if (!filtered.length) {
      listEl.innerHTML = '<div class="sb-empty">No conversations yet.</div>';
      return;
    }
    for (const s of filtered) {
      const div = document.createElement("div");
      div.className = "chat-item" + (s.id === activeId ? " active" : "");
      div.innerHTML = `
        <span class="chat-item-title">${escHtml(s.title)}</span>
        <span class="chat-item-acts">
          <button type="button" class="chat-item-edit" title="Rename">&#9998;</button>
          <button type="button" class="chat-item-del" title="Delete">&times;</button>
        </span>
      `;
      div.addEventListener("click", (e) => {
        if (e.target.closest(".chat-item-acts") || e.target.classList.contains("chat-item-input")) return;
        activeId = s.id;
        if ($("model-select") && s.model) {
          const hasM = Array.from($("model-select").options).some((o) => o.value === s.model);
          const nextModel = hasM ? s.model : "qwen3.8-27b";
          if ($("model-select").value !== nextModel) {
            $("model-select").value = nextModel;
            preloadModel(nextModel);
          }
        }
        if ($("node-select") && s.targetNode) {
          const hasOpt = Array.from($("node-select").options).some((o) => o.value === s.targetNode);
          $("node-select").value = hasOpt ? s.targetNode : "auto";
        }
        renderSidebar();
        renderThread();
        closeMobileDrawer();
      });
      div.querySelector(".chat-item-title").addEventListener("dblclick", (e) => {
        e.stopPropagation();
        startInlineRename(div, s);
      });
      div.querySelector(".chat-item-edit").addEventListener("click", (e) => {
        e.stopPropagation();
        startInlineRename(div, s);
      });
      div.querySelector(".chat-item-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });
      listEl.appendChild(div);
    }
  }

  function estimateTokens(text) {
    const s = String(text || "");
    return s ? Math.max(1, Math.ceil(s.length / 4)) : 0;
  }

  function lastCompactionIndex(sess) {
    if (!sess || !Array.isArray(sess.messages)) return -1;
    for (let i = sess.messages.length - 1; i >= 0; i--) {
      if (sess.messages[i].role === "compaction") return i;
    }
    return -1;
  }

  function buildApiMessages(sess) {
    if (!sess || !Array.isArray(sess.messages)) return [];
    const cIdx = lastCompactionIndex(sess);
    const out = [];
    if (cIdx >= 0) {
      const cMsg = sess.messages[cIdx];
      if (cMsg.content) {
        out.push({
          role: "system",
          content: "Compacted summary of earlier conversation context (treat as accurate ground truth):\n\n" + cMsg.content,
        });
      }
    }
    const start = cIdx >= 0 ? cIdx + 1 : 0;
    for (let i = start; i < sess.messages.length; i++) {
      const m = sess.messages[i];
      if (m.role === "user") {
        if (Array.isArray(m.images) && m.images.length > 0) {
          const parts = m.images.map((url) => ({ type: "image_url", image_url: { url } }));
          parts.push({ type: "text", text: m.content || "Describe this image in detail." });
          out.push({ role: "user", content: parts });
        } else if (m.content) {
          out.push({ role: "user", content: m.content });
        }
      } else if (m.role === "assistant" && m.content) {
        if (m.meta && m.meta.backend === "rpc-standby") continue;
        out.push({ role: "assistant", content: m.content });
      }
    }
    return out;
  }

  function messageTextForEstimate(c) {
    if (typeof c === "string") return c;
    if (Array.isArray(c)) {
      return c.map((p) => (p.type === "text" ? (p.text || "") : " ".repeat(1024))).join(" ");
    }
    return "";
  }

  function activeContextTokens(sess) {
    return buildApiMessages(sess).reduce((sum, m) => sum + estimateTokens(messageTextForEstimate(m.content)) + 4, 0);
  }

  function updateContextMeter() {
    const sess = getActiveSession();
    const meter = $("ctx-meter");
    const btn = $("btn-compact");
    const cIdx = lastCompactionIndex(sess);
    const activeMsgs = sess ? sess.messages.slice(cIdx >= 0 ? cIdx + 1 : 0).filter((m) => m.role === "user" || m.role === "assistant") : [];
    const canCompact = !isStreaming && activeMsgs.length >= 2;
    if (btn) {
      btn.disabled = !canCompact;
      if (!isStreaming) btn.textContent = "Compact";
    }
    if (meter) {
      const toks = activeContextTokens(sess);
      if (!sess || !sess.messages.length) {
        meter.textContent = "Space AI · Open community models";
        meter.classList.remove("ctx-warn");
      } else {
        const pct = Math.min(100, Math.round((toks / 24576) * 100));
        meter.textContent = `Context ${pct}%`;
        meter.classList.toggle("ctx-warn", pct >= 70);
      }
    }
  }

  function updateXrayButton() {
    const btn = $("btn-xray");
    if (!btn) return;
    btn.classList.toggle("active", xrayEnabled);
    btn.innerHTML = `<span class="xray-dot"></span><span>Confidence${xrayEnabled ? " · On" : ""}</span>`;
  }

  function closeXrayPopover() {
    const pop = $("xray-popover");
    if (pop) pop.classList.remove("open");
    document.querySelectorAll(".xtok.xtok-active").forEach((el) => el.classList.remove("xtok-active"));
  }

  function formatTokLabel(tok) {
    const s = String(tok ?? "");
    if (s === "\n") return "↵";
    if (s === "\n\n") return "↵↵";
    if (s === " ") return "␠";
    return s.replace(/\n/g, "↵");
  }

  function summarizeSignals(signals) {
    if (!Array.isArray(signals) || !signals.length) return null;
    let sumP = 0;
    let sumM = 0;
    let maxH = 0;
    let forks = 0;
    let risky = 0;
    for (const s of signals) {
      const p = Number(s.p ?? 1);
      const m = Number(s.m ?? 6);
      const h = Number(s.h ?? 0);
      sumP += p;
      sumM += Math.max(0, Math.min(10, m));
      if (h > maxH) maxH = h;
      if (m < 0.6 || p < 0.55) {
        risky += 1;
        forks += 1;
      } else if (m < 2.0 || p < 0.80) {
        forks += 1;
      }
    }
    const n = signals.length;
    return {
      count: n,
      avgCertPct: Math.round((sumP / n) * 1000) / 10,
      avgMargin: Math.round((sumM / n) * 10) / 10,
      maxEntropy: Math.round(maxH * 100) / 100,
      forks,
      risky,
    };
  }

  function renderMarginSparkline(signals) {
    if (!Array.isArray(signals) || signals.length < 2) return "";
    const w = 360;
    const h = 24;
    const n = signals.length;
    const pts = [];
    const dips = [];
    for (let i = 0; i < n; i++) {
      const m = Math.max(0, Math.min(8, Number(signals[i].m ?? 4)));
      const p = Number(signals[i].p ?? 1);
      const x = (i / Math.max(n - 1, 1)) * w;
      const y = h - 2 - (m / 8) * (h - 4);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      if (m < 0.6 || p < 0.55) {
        dips.push(`<line x1="${x.toFixed(1)}" y1="1" x2="${x.toFixed(1)}" y2="${h - 1}" stroke="rgba(251,113,133,0.72)" stroke-width="1.5" />`);
      } else if (m < 2.0 || p < 0.80) {
        dips.push(`<line x1="${x.toFixed(1)}" y1="4" x2="${x.toFixed(1)}" y2="${h - 1}" stroke="rgba(251,191,36,0.45)" stroke-width="1.1" />`);
      }
    }
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="Decision Margin Sparkline">${dips.join("")}<polyline fill="none" stroke="rgba(255,255,255,0.82)" stroke-width="1.35" points="${pts.join(" ")}" /></svg>`;
  }

  function renderXrayTokensHtml(signals) {
    if (!Array.isArray(signals) || !signals.length) return "";
    return `<div class="xray-tokens">${signals
      .map((s, ti) => {
        const p = Number(s.p ?? 1);
        const m = Number(s.m ?? 6);
        let cls = "xtok";
        if (m < 0.6 || p < 0.55) cls += " xtok-low";
        else if (m < 2.0 || p < 0.80) cls += " xtok-mid";
        return `<span class="${cls}" data-ti="${ti}">${escHtml(s.t || "")}</span>`;
      })
      .join("")}</div>`;
  }

  function openTokenPopover(anchorEl, sess, msgIdx, ti) {
    const pop = $("xray-popover");
    const msg = sess?.messages?.[msgIdx];
    const sig = msg?.signals?.[ti];
    if (!pop || !sig) return;

    document.querySelectorAll(".xtok.xtok-active").forEach((el) => el.classList.remove("xtok-active"));
    anchorEl.classList.add("xtok-active");

    const certPct = Math.round(Number(sig.p ?? 0) * 1000) / 10;
    const marginVal = Number(sig.m ?? 0).toFixed(2);
    const entVal = Number(sig.h ?? 0).toFixed(2);
    const alts = Array.isArray(sig.alts) && sig.alts.length ? sig.alts : [{ t: sig.t, p: sig.p }];

    pop.innerHTML = `
      <div class="xp-head">
        <span class="xp-tok">${escHtml(formatTokLabel(sig.t))}</span>
        <button type="button" class="xp-close" aria-label="Close">&times;</button>
      </div>
      <div class="xp-sub">Confidence ${Math.round(certPct)}% · Click an alternative to fork</div>
      <div class="xp-alts">
        ${alts
          .map((alt, ai) => {
            const ap = Math.max(0, Math.min(100, Math.round(Number(alt.p ?? 0) * 100)));
            const isChosen = alt.t === sig.t;
            return `
              <button type="button" class="xp-alt-btn${isChosen ? " chosen" : ""}" data-ai="${ai}">
                <span class="xp-alt-bar" style="width:${Math.max(2, ap)}%"></span>
                <span class="xp-alt-tok">${escHtml(formatTokLabel(alt.t))}</span>
                <span class="xp-alt-meta">${ap}%${isChosen ? " · picked" : ""}</span>
              </button>
            `;
          })
          .join("")}
      </div>
    `;

    pop.querySelector(".xp-close")?.addEventListener("click", closeXrayPopover);
    pop.querySelectorAll(".xp-alt-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const ai = Number(btn.getAttribute("data-ai"));
        const alt = alts[ai];
        closeXrayPopover();
        if (alt && !isStreaming) {
          forkFromToken(sess, msgIdx, ti, alt.t, sig.t);
        }
      });
    });

    pop.classList.add("open");
    const rect = anchorEl.getBoundingClientRect();
    const pw = pop.offsetWidth || 296;
    const ph = pop.offsetHeight || 240;
    let left = rect.left;
    if (left + pw > window.innerWidth - 12) left = window.innerWidth - pw - 12;
    if (left < 10) left = 10;
    let top = rect.bottom + 8;
    if (top + ph > window.innerHeight - 12) top = Math.max(10, rect.top - ph - 8);
    pop.style.left = left + "px";
    pop.style.top = top + "px";
  }

  async function forkFromToken(sess, msgIdx, ti, altTok, origTok) {
    if (isStreaming || !sess) return;
    const origMsg = sess.messages[msgIdx];
    if (!origMsg || !Array.isArray(origMsg.signals)) return;
    const keptSignals = origMsg.signals.slice(0, ti);
    const prefixBefore = keptSignals.map((s) => s.t || "").join("");
    const forcedPrefix = prefixBefore + altTok;
    const altEntry = (origMsg.signals[ti]?.alts || []).find((a) => a.t === altTok);
    const transplantedSig = {
      t: altTok,
      p: altEntry ? altEntry.p : 0.5,
      m: 0.0,
      h: origMsg.signals[ti]?.h || 1.0,
      alts: origMsg.signals[ti]?.alts || [{ t: altTok, p: 0.5 }],
    };

    // Build prior conversation history up to the user message right before origMsg
    const priorSess = { messages: sess.messages.slice(0, msgIdx) };
    const apiMessages = buildApiMessages(priorSess);
    apiMessages.push({
      role: "user",
      content:
        "Counterfactual continuation task: your response to the previous prompt is fixed to begin with the exact prefix below. " +
        "Continue seamlessly from the VERY NEXT character after the prefix without repeating the prefix or mentioning these instructions.\n\n" +
        "FIXED PREFIX:\n" + forcedPrefix,
    });

    const selectedModel = $("model-select")?.value || "qwen3.8-27b";
    const selectedNode = $("node-select")?.value || "auto";
    const targetNode = selectedNode !== "auto" ? selectedNode : null;

    const forkedMsg = {
      role: "assistant",
      content: forcedPrefix,
      reasoning: "",
      thinkMs: 0,
      meta: null,
      signals: [...keptSignals, transplantedSig],
      showXray: true,
      forkedFrom: {
        origTok: origTok,
        altTok: altTok,
        tokenIndex: ti,
      },
    };
    sess.messages.push(forkedMsg);
    sess.updatedAt = Date.now();
    saveSessions();

    setStreamingUI(true);
    renderThread(true);
    cacheStreamingDomRefs();
    lastRenderTs = 0;

    const scroller = $("thread-scroll");
    activeAbort = new AbortController();
    let rafId = 0;

    try {
      const reqBody = {
        model: selectedModel,
        messages: apiMessages,
        stream: true,
        max_tokens: 2048,
        reasoning_effort: "none",
        session_id: sess.id + "-fork-" + Date.now(),
        xray: true,
        xray_top_k: 5,
      };
      if (targetNode) reqBody.target_node = targetNode;
      const res = await fetch(API_BASE + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
        signal: activeAbort.signal,
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        forkedMsg.content += `\n[Fork error: ${detail}]`;
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      function scheduleRender() {
        if (rafId || !isStreaming) return;
        rafId = requestAnimationFrame((ts) => {
          rafId = 0;
          if (!isStreaming) return;
          lastRenderTs = ts;
          const nearBottom = scroller ? (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140) : true;
          updateStreamingLastMessage(forkedMsg);
          if (nearBottom && scroller) scroller.scrollTop = scroller.scrollHeight;
        });
      }
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const chunk of parts) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const dataStr = line.slice(5).trim();
          if (dataStr === "[DONE]") continue;
          try {
            const parsed = JSON.parse(dataStr);
            const dObj = parsed.choices?.[0]?.delta || {};
            if (dObj.content) {
              forkedMsg.content += dObj.content;
              if (parsed.token_signal) forkedMsg.signals.push(parsed.token_signal);
            }
            if (parsed.longinus_meta) forkedMsg.meta = parsed.longinus_meta;
          } catch (_ignore) {}
        }
        scheduleRender();
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        forkedMsg.content += `\n[Interrupted: ${err.message}]`;
      }
    } finally {
      if (rafId) cancelAnimationFrame(rafId);
      activeAbort = null;
      forkedMsg.statusText = "";
      setStreamingUI(false);
      saveSessions();
      renderThread(false);
      if (xrayEnabled && forkedMsg.content) {
        const msgIndex = sess.messages.indexOf(forkedMsg);
        if (msgIndex >= 0) {
          runMessageAudit(sess, msgIndex);
        }
      }
    }
  }

  async function runAntisymmetricCompare(rawCommand) {
    if (isStreaming) return;
    const body = rawCommand.replace(/^\/compare\s*/i, "").trim();
    const parts = body.split("|").map((s) => s.trim()).filter(Boolean);
    if (parts.length < 2) {
      let sess = getActiveSession() || createSession("Bias-Free Compare");
      sess.messages.push({ role: "user", content: rawCommand });
      sess.messages.push({
        role: "assistant",
        content: "Usage: `/compare Option A | Option B | criterion` (supports 2 to 6 options, e.g. `/compare Rust | Go | Zig | low-latency servers`).",
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }
    const items = parts.length >= 3 ? parts.slice(0, -1).slice(0, 6) : parts.slice(0, 2);
    const criterion = parts.length >= 3 ? parts[parts.length - 1] : "overall quality";
    const selectedModel = $("model-select")?.value || "qwen3.8-27b";

    let sess = getActiveSession();
    if (!sess) sess = createSession(`Compare: ${items.slice(0, 2).join(" vs ")}`);
    if (sess.messages.length === 0) sess.title = `Compare: ${items.slice(0, 2).join(" vs ")}`.slice(0, 44);

    sess.messages.push({ role: "user", content: rawCommand });
    const cmpMsg = { role: "compare", loading: true, items, criterion, result: null, error: null };
    sess.messages.push(cmpMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    setStreamingUI(true);
    renderSidebar();
    renderThread(true);

    try {
      const res = await fetch(API_BASE + "/v1/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items, criterion, model: selectedModel }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        cmpMsg.error = detail;
      } else {
        cmpMsg.result = await res.json();
      }
    } catch (err) {
      cmpMsg.error = err.message || "Network error";
    } finally {
      cmpMsg.loading = false;
      setStreamingUI(false);
      saveSessions();
      renderThread(true);
    }
  }

  function renderCompareCardHtml(m) {
    if (m.loading) {
      return `<div class="cmp-card"><div class="cmp-eyebrow">Antisymmetric Paired-Swap Judge · Running Parallel Swaps...</div><p style="margin:0.45rem 0 0 0; opacity:0.8;">Evaluating all order-swapped pairs for <strong>${escHtml((m.items || []).join(" vs "))}</strong> on <em>${escHtml(m.criterion || "")}</em>...</p></div>`;
    }
    if (m.error || !m.result) {
      return `<div class="cmp-card"><div class="cmp-eyebrow">Antisymmetric Compare</div><p style="margin:0.45rem 0 0 0;">Could not complete comparison: ${escHtml(m.error || "Unknown error")}</p></div>`;
    }
    const r = m.result;
    const rows = Array.isArray(r.ranking) ? r.ranking : [];
    const maxAbs = Math.max(1.5, ...rows.map((x) => Math.abs(Number(x.score || 0))));
    const flipPct = Math.round(Number(r.pos_bias_rate || 0) * 100);
    const hasFlip = flipPct > 0;
    const rowsHtml = rows
      .map((row) => {
        const pct = Math.max(6, Math.min(100, Math.round(Number(row.win_prob || 0.5) * 100)));
        const sc = Number(row.score || 0);
        const sign = sc >= 0 ? "+" : "";
        return `
          <div class="cmp-row">
            <span class="cmp-rank">#${row.rank}</span>
            <span class="cmp-name" title="${escHtml(row.item)}">${escHtml(row.item)}</span>
            <div class="cmp-bar-bg"><div class="cmp-bar-fill" style="width:${pct}%; opacity:${row.rank === 1 ? "1" : "0.52"};"></div></div>
            <span class="cmp-score">&Delta; ${sign}${sc.toFixed(2)} · ${pct}%</span>
          </div>
        `;
      })
      .join("");

    const pairsHtml = (r.pairs || [])
      .map((p) => {
        const wpA = Math.round(Number(p.win_prob_a || 0.5) * 100);
        const m1 = Number(p.order1_margin || 0);
        const m2 = Number(p.order2_margin_for_a || 0);
        const ma = Number(p.antisymmetric_margin || 0);
        const pb = Number(p.position_bias || 0);
        return `
          <div class="cmp-pair-row">
            <span><strong>${escHtml(p.a)}</strong> vs <strong>${escHtml(p.b)}</strong>${p.flipped ? ' <span style="color:#fbbf24;">[Bias Flipped!]</span>' : ""}</span>
            <span>Ord1 ${m1 >= 0 ? "+" : ""}${m1.toFixed(2)} · Ord2 ${m2 >= 0 ? "+" : ""}${m2.toFixed(2)} · <strong>Anti ${ma >= 0 ? "+" : ""}${ma.toFixed(2)}</strong> (Bias ${pb >= 0 ? "+" : ""}${pb.toFixed(2)}, ${wpA}%)</span>
          </div>
        `;
      })
      .join("");

    return `
      <div class="cmp-card">
        <div class="cmp-head">
          <span class="cmp-eyebrow">Antisymmetric Paired-Swap Ranking · ${r.pairs_evaluated || 0} Swaps · ${Math.round(r.latency_ms || 0)} ms</span>
          <span class="meta-pill">${escHtml(r.served_by || "Local Silicon")}</span>
        </div>
        <div class="cmp-title">#1 ${escHtml(r.winner || "")} <span style="font-weight:400; opacity:0.65; font-size:0.86rem;">on ${escHtml(r.criterion || "")}</span></div>
        ${r.explanation ? `<p class="cmp-exp">${escHtml(r.explanation)}</p>` : ""}
        <div class="cmp-rows">${rowsHtml}</div>
        <div class="cmp-diag">
          <span class="cmp-chip ${hasFlip ? "warn" : "ok"}">${hasFlip ? `Position Bias Cancelled (${flipPct}% pairs flipped)` : "Order-Invariant Verdict (0% flips)"}</span>
          <span class="cmp-chip">Mean Order Shift: &plusmn;${Number(r.avg_abs_pos_bias || 0).toFixed(2)}</span>
          ${r.items_count >= 3 ? `<span class="cmp-chip">Raw Cycles Eliminated: ${r.cycles_eliminated || 0}</span>` : ""}
        </div>
        <details class="cmp-pairs">
          <summary>Inspect All Paired-Swap Margins (${(r.pairs || []).length} matchups)</summary>
          <div class="cmp-pair-list">${pairsHtml}</div>
        </details>
      </div>
    `;
  }

  function auditSeverityClass(verdict) {
    if (verdict === "VERIFIED") return "ok";
    if (verdict === "CAUTION") return "warn";
    return "risk";
  }

  async function runMessageAudit(sess, msgIdx) {
    const msg = sess?.messages?.[msgIdx];
    if (!msg || msg.role !== "assistant" || !msg.content || msg.auditLoading) return;
    if (msg.meta?.backend === "rpc-standby") return;
    let userPrompt = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (sess.messages[i]?.role === "user") {
        userPrompt = sess.messages[i].content || "";
        break;
      }
    }
    msg.auditLoading = true;
    try {
      const res = await fetch(API_BASE + "/v1/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_prompt: userPrompt,
          assistant_reply: msg.content,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (res.ok) {
        msg.audit = await res.json();
      }
    } catch (_e) {
    } finally {
      msg.auditLoading = false;
      saveSessions();
      if (!isStreaming) renderThread(false);
    }
  }

  async function requestHonestRewrite(sess, msgIdx) {
    if (isStreaming || !sess) return;
    const origMsg = sess.messages[msgIdx];
    if (!origMsg) return;
    let userPrompt = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (sess.messages[i]?.role === "user") {
        userPrompt = sess.messages[i].content || "";
        break;
      }
    }
    const auditNote = origMsg.audit?.note ? ` Flagged issue: ${origMsg.audit.note}` : "";
    sess.messages.push({
      role: "user",
      content:
        `Give me a strictly objective, fact-checked, zero-flattery answer to my previous question ("${userPrompt.slice(0, 260)}").` +
        ` Do not pander, agree just to be polite, or guess unverified facts.${auditNote}`,
    });
    saveSessions();
    renderThread(true);
    await streamAssistantReply(sess);
  }

  async function runStandaloneAudit(rawCommand) {
    if (isStreaming) return;
    const body = rawCommand.replace(/^\/audit\s*/i, "").trim();
    if (!body) {
      let sess = getActiveSession() || createSession("Truth & Sycophancy Audit");
      sess.messages.push({ role: "user", content: rawCommand });
      sess.messages.push({
        role: "assistant",
        content: "Usage: `/audit <statement or AI reply>` or `/audit User: <prompt> | AI: <reply>` to check for sycophancy, deception, or overconfident bluffing.",
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }
    let uPrompt = "";
    let aReply = body;
    if (body.includes("|")) {
      const idx = body.indexOf("|");
      uPrompt = body.slice(0, idx).replace(/^user\s*:\s*/i, "").trim();
      aReply = body.slice(idx + 1).replace(/^(ai|assistant)\s*:\s*/i, "").trim();
    }

    let sess = getActiveSession();
    if (!sess) sess = createSession("Audit: " + aReply.slice(0, 34));
    if (sess.messages.length === 0) sess.title = ("Audit: " + aReply).slice(0, 44);

    sess.messages.push({ role: "user", content: rawCommand });
    const auditMsg = { role: "audit", loading: true, userPrompt: uPrompt, assistantReply: aReply, result: null, error: null };
    sess.messages.push(auditMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    setStreamingUI(true);
    renderSidebar();
    renderThread(true);

    try {
      const res = await fetch(API_BASE + "/v1/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_prompt: uPrompt,
          assistant_reply: aReply,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        auditMsg.error = detail;
      } else {
        auditMsg.result = await res.json();
      }
    } catch (err) {
      auditMsg.error = err.message || "Network error";
    } finally {
      auditMsg.loading = false;
      setStreamingUI(false);
      saveSessions();
      renderThread(true);
    }
  }

  function renderStandaloneAuditCardHtml(m) {
    if (m.loading) {
      return `<div class="feedback-line">Checking response…</div>`;
    }
    if (m.error || !m.result) {
      return `<div class="feedback-line xray-m-risk">Check failed: ${escHtml(m.error || "Unknown error")}</div>`;
    }
    const r = m.result;
    const sev = auditSeverityClass(r.verdict);
    const cls = sev === "risk" ? "xray-m-risk" : sev === "warn" ? "xray-m-warn" : "xray-m-ok";
    return `<div class="feedback-line"><span class="conf-num ${cls}">${escHtml(r.verdict_label || r.verdict)}</span><span>&middot;</span><span>Confidence ${Math.round(r.grounded_pct ?? 90)}%</span></div>`;
  }

  function renderAuditInlineHtml(audit, loading) {
    if (loading) return "";
    if (!audit) return "";
    const sev = auditSeverityClass(audit.verdict);
    const cls = sev === "risk" ? "xray-m-risk" : sev === "warn" ? "xray-m-warn" : "xray-m-ok";
    const lbl = escHtml(audit.verdict_label || audit.verdict || "Verified");
    return `<span class="${cls}">&middot; ${lbl}</span>`;
  }

  async function runStressTest(rawCommand) {
    if (isStreaming) return;
    const question = rawCommand.replace(/^\/stress\s*/i, "").trim();
    if (!question) {
      let sess = getActiveSession() || createSession("Framing Stress-Test");
      sess.messages.push({ role: "user", content: rawCommand });
      sess.messages.push({
        role: "assistant",
        content: "Usage: `/stress <yes/no or decision question>` (e.g. `/stress Is TypeScript better than plain JavaScript for large teams?`).",
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }

    let sess = getActiveSession();
    if (!sess) sess = createSession("Stress: " + question.slice(0, 34));
    if (sess.messages.length === 0) sess.title = ("Stress: " + question).slice(0, 44);

    sess.messages.push({ role: "user", content: rawCommand });
    const stressMsg = { role: "stress", loading: true, question, result: null, error: null };
    sess.messages.push(stressMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    setStreamingUI(true);
    renderSidebar();
    renderThread(true);

    try {
      const res = await fetch(API_BASE + "/v1/stress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        stressMsg.error = detail;
      } else {
        stressMsg.result = await res.json();
      }
    } catch (err) {
      stressMsg.error = err.message || "Network error";
    } finally {
      stressMsg.loading = false;
      setStreamingUI(false);
      saveSessions();
      renderThread(true);
    }
  }

  function renderStressCardHtml(m) {
    if (m.loading) {
      return `<div class="audit-card"><div class="cmp-eyebrow">Framing &amp; Sycophancy Stress-Test · Running 5 Parallel Framings...</div><p style="margin:0.45rem 0 0 0; opacity:0.8;">Testing neutral baseline, user nudges, and authority pressure on <strong>${escHtml(m.question || "")}</strong>...</p></div>`;
    }
    if (m.error || !m.result) {
      return `<div class="audit-card"><div class="cmp-eyebrow">Framing Stress-Test</div><p style="margin:0.45rem 0 0 0;">Could not run stress-test: ${escHtml(m.error || "Unknown error")}</p></div>`;
    }
    const r = m.result;
    const sev = r.verdict === "ROCK_SOLID" ? "ok" : (r.verdict === "BENDS_UNDER_PRESSURE" ? "warn" : "risk");
    const leftLbl = r.left_label || "No";
    const rightLbl = r.right_label || "Yes";

    const ordered = Array.isArray(r.framings) ? r.framings : [];
    const neutralIdx = Math.max(0, ordered.findIndex((x) => x.key === "neutral"));
    const activeIdx = Math.max(0, Math.min(ordered.length - 1, typeof m.stressIdx === "number" ? m.stressIdx : neutralIdx));
    const cur = ordered[activeIdx] || { key: "neutral", label: "Neutral", prompt: r.question, reply: "", yes_pct: 50, side_conf_pct: 50, answer: rightLbl };
    const curRightPct = Math.max(2, Math.min(100, Number(cur.right_pct ?? cur.yes_pct ?? 50)));
    const curRight = Number(cur.right_pct ?? cur.yes_pct ?? 50) >= 50;
    const curConf = Number(cur.side_conf_pct ?? (curRight ? curRightPct : Math.round((100 - curRightPct) * 10) / 10));
    const curDelta = Math.round((Number(cur.right_pct ?? cur.yes_pct ?? 50) - Number(r.neutral_yes_pct ?? 50)) * 10) / 10;
    const curFlipped = cur.answer !== r.neutral_answer;
    const deltaStr = cur.key === "neutral" ? "Baseline" : `${curDelta >= 0 ? "+" : ""}${curDelta} pts`;

    const ticksHtml = ordered
      .map((f, fi) => `<button type="button" class="stress-tick${fi === activeIdx ? " active" : ""}" data-sidx="${fi}">${escHtml(f.label)}</button>`)
      .join("");

    return `
      <div class="audit-card">
        <div class="cmp-head">
          <span class="cmp-eyebrow">Stability ${r.stability_pct}% · Max Swing ${r.max_swing_pct} pts · ${Math.round(r.latency_ms || 0)} ms</span>
          <span class="audit-badge ${sev}">${escHtml(r.verdict_label || r.verdict)}</span>
        </div>
        <div class="cmp-title" style="margin-bottom:0.55rem;">${escHtml(r.question || "")}</div>

        <div class="stress-sim" style="margin:0.4rem 0 0 0;">
          <div class="stress-sim-top">
            <div class="stress-sim-title">Framing · <span class="stress-cur-lbl">${escHtml(cur.label)}</span> <span class="stress-cur-delta" style="opacity:0.62;">(${escHtml(deltaStr)})</span></div>
            <span class="stress-ans-pill ${curRight ? "yes" : "no"}">
              <span class="stress-ans-txt">${escHtml(String(cur.answer || "").toUpperCase())} · ${curConf}%</span>
              <span class="stress-flip-tag" style="display:${curFlipped ? "inline-block" : "none"};">FLIPPED</span>
            </span>
          </div>
          <div style="display:flex; justify-content:space-between; font-family:var(--mono); font-size:0.6rem; opacity:0.65; margin-bottom:0.22rem; text-transform:uppercase; letter-spacing:0.06em;">
            <span>&larr; ${escHtml(leftLbl)}</span>
            <span>50% flip</span>
            <span>${escHtml(rightLbl)} &rarr;</span>
          </div>
          <div class="stress-sim-meter">
            <div class="stress-mid" title="50% Flip Threshold"></div>
            <div class="stress-sim-fill ${curRight ? "yes" : "no"}" style="width:${curRightPct}%"></div>
          </div>
          <input type="range" class="stress-range" min="0" max="${Math.max(0, ordered.length - 1)}" step="1" value="${activeIdx}" aria-label="Slide framing pressure from ${escHtml(leftLbl)} to ${escHtml(rightLbl)}" />
          <div class="stress-ticks">${ticksHtml}</div>

          <div class="stress-io">
            <div class="stress-io-box">
              <div class="stress-io-k">Prompt Sent</div>
              <div class="stress-io-v stress-prompt-val">${escHtml(cur.prompt || r.question || "")}</div>
            </div>
            <div class="stress-io-box stress-ans-box ${curRight ? "ans-yes" : "ans-no"}">
              <div class="stress-io-k">AI Answer</div>
              <div class="stress-io-v stress-reply-val">${escHtml(cur.reply || (cur.answer + "."))}</div>
            </div>
          </div>
        </div>

        <div class="msg-actions" style="margin-top:0.65rem;">
          <button type="button" class="act-btn btn-share-link">Share Link</button>
          <button type="button" class="act-btn btn-share-card">Share 4:5 Card</button>
          <span class="meta-pill">${escHtml(r.served_by || "Local Silicon")} · Slide to test framing pressure</span>
        </div>
      </div>
    `;
  }

  let toastTimer = 0;
  function showShareToast(tweetText, label = "4:5 X-Ray Card PNG saved &#10003;") {
    const el = $("share-toast");
    if (!el) return;
    const xHref = "https://x.com/intent/tweet?text=" + encodeURIComponent(tweetText);
    el.innerHTML = `<span>${label}</span><a class="share-toast-x" href="${xHref}" target="_blank" rel="noopener">Post on X &rarr;</a>`;
    el.classList.add("open");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("open"), 9500);
  }

  function buildShareSnapshot(sess, msgIdx) {
    const m = sess?.messages?.[msgIdx];
    if (!m) return null;
    let userPrompt = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (sess.messages[i]?.role === "user") {
        userPrompt = sess.messages[i].content || "";
        break;
      }
    }
    const kind = (m.role === "stress" || m.role === "audit" || m.role === "compare") ? m.role : "xray";
    const title =
      kind === "stress" ? (m.result?.question || "Framing Stress-Test") :
      kind === "compare" ? (m.result?.criterion || "Bias-Free Compare") :
      kind === "audit" ? "Truth & Bluff Radar" :
      (userPrompt || m.content || "Space AI X-Ray");
    const clonedMsg = { ...m };
    if (Array.isArray(clonedMsg.signals) && clonedMsg.signals.length > 360) {
      clonedMsg.signals = clonedMsg.signals.slice(0, 360);
    }
    delete clonedMsg.images;
    return { kind, title: String(title).slice(0, 140), userPrompt, payload: { userPrompt, message: clonedMsg } };
  }

  function buildShareTweet(snap, shareUrl) {
    const m = snap.payload.message;
    const url = shareUrl || "https://spaceresearch.ai/chat.html";
    if (snap.kind === "stress" && m.result) {
      const r = m.result;
      return `Stress-tested a local AI on "${(r.question || "").slice(0, 72)}" with Space X-Ray:\n• Verdict: ${r.verdict_label}\n• Belief Stability: ${r.stability_pct}% (max swing: ${r.max_swing_pct} pts)\n\nInspect & slide live: ${url}`;
    }
    if (snap.kind === "compare" && m.result) {
      const r = m.result;
      return `Bias-free AI ranking on "${r.criterion || ""}": #1 ${r.winner || ""} (order bias cancelled in ${Math.round(r.latency_ms || 0)}ms).\n\nInspect live: ${url}`;
    }
    if (snap.kind === "audit" && m.result) {
      const r = m.result;
      return `Ran Space AI Truth & Bluff Radar (${Math.round(r.latency_ms || 0)}ms local AI):\n• Verdict: ${r.verdict_label}\n• Grounded ${r.grounded_pct}% · Sycophancy ${r.sycophancy_pct}% · Bluff ${r.bluff_pct}%\n\nInspect live: ${url}`;
    }
    const sum = Array.isArray(m.signals) && m.signals.length ? summarizeSignals(m.signals) : null;
    const certTxt = sum ? `Certainty ${sum.avgCertPct}%` : "Live X-Ray";
    const audTxt = m.audit ? ` · ${m.audit.verdict_label} (Bluff ${m.audit.bluff_pct}%)` : "";
    return `Asked a local open model on Space Chat: "${(snap.userPrompt || "").slice(0, 68)}"\n\nX-Ray readout: ${certTxt}${audTxt}\n\nInspect every token & fork reality live: ${url}`;
  }

  async function sharePermalink(sess, msgIdx, triggerBtn) {
    const snap = buildShareSnapshot(sess, msgIdx);
    if (!snap) return;
    const prevTxt = triggerBtn ? triggerBtn.textContent : "";
    if (triggerBtn) { triggerBtn.disabled = true; triggerBtn.textContent = "Saving..."; }
    try {
      const res = await fetch(API_BASE + "/v1/shares", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: snap.kind, title: snap.title, payload: snap.payload }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const shareUrl = data.url || (`https://spaceresearch.ai/chat.html?share=${data.share_id}`);
      try { await navigator.clipboard.writeText(shareUrl); } catch (_e) {}
      if (triggerBtn) {
        triggerBtn.textContent = "Link Copied ✓";
        setTimeout(() => { triggerBtn.textContent = prevTxt || "Share Link"; triggerBtn.disabled = false; }, 1800);
      }
      showShareToast(buildShareTweet(snap, shareUrl), `Live permalink copied &#10003;`);
    } catch (_err) {
      if (triggerBtn) {
        triggerBtn.textContent = "Retry Link";
        setTimeout(() => { triggerBtn.textContent = prevTxt || "Share Link"; triggerBtn.disabled = false; }, 1600);
      }
    }
  }

  function wrapCanvasLines(ctx, text, maxW, maxLines) {
    const words = String(text || "").replace(/\s+/g, " ").trim().split(" ");
    const lines = [];
    let cur = "";
    for (const w of words) {
      const cand = cur ? cur + " " + w : w;
      if (ctx.measureText(cand).width > maxW && cur) {
        lines.push(cur);
        cur = w;
        if (lines.length >= maxLines) break;
      } else {
        cur = cand;
      }
    }
    if (cur && lines.length < maxLines) lines.push(cur);
    if (lines.length === maxLines && words.length > 1) {
      const joined = lines.join(" ");
      if (joined.length + 3 < String(text || "").trim().length) {
        lines[maxLines - 1] = lines[maxLines - 1].replace(/\s+\S*$/, "") + "...";
      }
    }
    return lines;
  }

  function canvasRoundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function downloadShareXrayCard(sess, msgIdx) {
    const m = sess?.messages?.[msgIdx];
    if (!m) return;
    let userPrompt = "";
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (sess.messages[i]?.role === "user") {
        userPrompt = sess.messages[i].content || "";
        break;
      }
    }

    const W = 1080, H = 1350;
    const cv = document.createElement("canvas");
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    // 1. Jet-black obsidian canvas + architectural grid + subtle corner aurora
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);
    const isRisk =
      (m.role === "assistant" && m.audit && m.audit.verdict && m.audit.verdict !== "VERIFIED") ||
      (m.role === "audit" && m.result && m.result.verdict !== "VERIFIED") ||
      (m.role === "stress" && m.result && m.result.flipped);
    const g1 = ctx.createRadialGradient(240, 260, 20, 240, 260, 520);
    g1.addColorStop(0, "rgba(18, 128, 168, 0.22)");
    g1.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, W, H);
    const g2 = ctx.createRadialGradient(840, 1020, 20, 840, 1020, 540);
    g2.addColorStop(0, isRisk ? "rgba(244, 63, 94, 0.24)" : "rgba(34, 197, 94, 0.18)");
    g2.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.032)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 68) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 68) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // 2. Main Glass Container
    canvasRoundRect(ctx, 48, 52, W - 96, H - 154, 28);
    ctx.fillStyle = "rgba(12, 13, 18, 0.92)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Header
    ctx.fillStyle = "#ffffff";
    ctx.font = '700 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
    ctx.fillText("SPACE X-RAY", 86, 114);
    ctx.fillStyle = "rgba(255, 255, 255, 0.56)";
    ctx.font = '500 20px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText("LOCAL GLASS-BOX AI", 306, 113);

    // X-Ray ON pill top-right
    canvasRoundRect(ctx, W - 266, 82, 184, 44, 12);
    ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ctx.fill();
    ctx.strokeStyle = "rgba(110, 231, 183, 0.65)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(W - 238, 104, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#22c55e";
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = '600 18px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText("X-RAY · ON", W - 218, 110);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.beginPath(); ctx.moveTo(50, 148); ctx.lineTo(W - 50, 148); ctx.stroke();

    let tweetText = "Tried Space AI Decision X-Ray on local open models — see certainty, bluffs & sycophancy live: https://spaceresearch.ai/chat.html";
    let cy = 198;

    if (m.role === "stress" && m.result) {
      const r = m.result;
      const accent = r.verdict === "ROCK_SOLID" ? "#4ade80" : (r.verdict === "BENDS_UNDER_PRESSURE" ? "#fbbf24" : "#fb7185");
      ctx.fillStyle = "rgba(148, 226, 255, 0.9)";
      ctx.font = '600 20px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillText(`FRAMING STRESS-TEST · 5 PARALLEL PROBES · ${Math.round(r.latency_ms || 0)} MS`, 86, cy);
      cy += 48;

      ctx.fillStyle = "#ffffff";
      ctx.font = '700 40px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      const qLines = wrapCanvasLines(ctx, r.question || "", W - 172, 3);
      for (const ql of qLines) { ctx.fillText(ql, 86, cy); cy += 52; }
      cy += 18;

      // Verdict pill
      const vTxt = (r.verdict_label || r.verdict).toUpperCase();
      ctx.font = '700 22px "IBM Plex Mono", ui-monospace, monospace';
      const vw = ctx.measureText(vTxt).width + 44;
      canvasRoundRect(ctx, 86, cy, vw, 48, 24);
      ctx.fillStyle = r.flipped ? "rgba(244,63,94,0.22)" : "rgba(34,197,94,0.16)";
      ctx.fill();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = accent;
      ctx.fillText(vTxt, 108, cy + 31);
      cy += 84;

      // 3 KPI boxes
      const kpis = [
        [`${r.stability_pct}%`, "BELIEF STABILITY", accent],
        [`${r.max_swing_pct} pts`, "MAX SWING", r.flipped ? "#fb7185" : "#ffffff"],
        [`${r.neutral_answer} (${r.neutral_yes_pct}%)`, "NEUTRAL STANCE", "#ffffff"],
      ];
      const bw = 286;
      kpis.forEach(([kv, kl, kc], ki) => {
        const bx = 86 + ki * (bw + 24);
        canvasRoundRect(ctx, bx, cy, bw, 116, 16);
        ctx.fillStyle = "rgba(255,255,255,0.04)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.16)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = kc;
        ctx.font = '700 34px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(kv, bx + 22, cy + 56);
        ctx.fillStyle = "rgba(255,255,255,0.68)";
        ctx.font = '500 17px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(kl, bx + 22, cy + 92);
      });
      cy += 156;

      // 5 Framing Bars
      for (const f of (r.framings || [])) {
        ctx.fillStyle = "rgba(255,255,255,0.88)";
        ctx.font = '600 22px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(f.label, 86, cy + 22);
        const tx = 368, tw = 450;
        canvasRoundRect(ctx, tx, cy + 4, tw, 24, 12);
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fill();
        const fw = Math.max(12, Math.round(tw * (Number(f.yes_pct || 0) / 100)));
        canvasRoundRect(ctx, tx, cy + 4, fw, 24, 12);
        ctx.fillStyle = Number(f.yes_pct || 0) >= 50 ? "#4ade80" : "#fb7185";
        ctx.fill();
        // 50% threshold line
        ctx.fillStyle = "rgba(255,255,255,0.65)";
        ctx.fillRect(tx + tw / 2 - 1, cy, 3, 32);
        ctx.fillStyle = "#ffffff";
        ctx.font = '600 22px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(`${f.answer} ${f.yes_pct}%`, tx + tw + 22, cy + 23);
        cy += 64;
      }

      cy += 18;
      canvasRoundRect(ctx, 86, cy, W - 172, 130, 16);
      ctx.fillStyle = r.flipped ? "rgba(244,63,94,0.12)" : "rgba(34,197,94,0.10)";
      ctx.fill();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = '500 25px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      const sLines = wrapCanvasLines(ctx, r.summary || "", W - 220, 3);
      sLines.forEach((sl, si) => ctx.fillText(sl, 110, cy + 44 + si * 36));

      tweetText = `Stress-tested a local AI on "${(r.question || "").slice(0, 75)}" with Space X-Ray:\n• Verdict: ${r.verdict_label}\n• Belief Stability: ${r.stability_pct}% (max framing swing: ${r.max_swing_pct} pts)\n\nTry free on open local silicon: https://spaceresearch.ai/chat.html`;
    } else if (m.role === "compare" && m.result) {
      const r = m.result;
      ctx.fillStyle = "rgba(148, 226, 255, 0.9)";
      ctx.font = '600 20px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillText(`BIAS-FREE COMPARE · ${Math.round(r.latency_ms || 0)} MS · LOCAL SILICON`, 86, cy);
      cy += 52;
      ctx.fillStyle = "#ffffff";
      ctx.font = '700 54px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      ctx.fillText(`#1  ${r.winner || ""}`, 86, cy);
      cy += 44;
      ctx.fillStyle = "rgba(255,255,255,0.68)";
      ctx.font = '500 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      ctx.fillText(`on ${r.criterion || ""}`, 86, cy);
      cy += 64;

      for (const row of (r.ranking || []).slice(0, 5)) {
        const pct = Math.max(8, Math.min(100, Math.round(Number(row.win_prob || 0.5) * 100)));
        ctx.fillStyle = "#ffffff";
        ctx.font = '700 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        ctx.fillText(`#${row.rank}  ${row.item}`, 86, cy + 28);
        const tx = 360, tw = 510;
        canvasRoundRect(ctx, tx, cy + 8, tw, 24, 12);
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fill();
        canvasRoundRect(ctx, tx, cy + 8, Math.round(tw * (pct / 100)), 24, 12);
        ctx.fillStyle = row.rank === 1 ? "#ffffff" : "rgba(255,255,255,0.48)";
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.font = '600 24px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(`${pct}%`, tx + tw + 22, cy + 28);
        cy += 78;
      }
      if (r.explanation) {
        cy += 22;
        ctx.fillStyle = "rgba(255,255,255,0.88)";
        ctx.font = '500 26px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        wrapCanvasLines(ctx, r.explanation, W - 172, 4).forEach((el, ei) => ctx.fillText(el, 86, cy + ei * 38));
      }
      tweetText = `Bias-free AI ranking on "${r.criterion || ""}": #1 ${r.winner || ""} (prompt-order bias automatically cancelled in ${Math.round(r.latency_ms || 0)}ms).\n\nTry free on Space Chat: https://spaceresearch.ai/chat.html`;
    } else if (m.role === "audit" && m.result) {
      const r = m.result;
      const accent = r.verdict === "VERIFIED" ? "#4ade80" : (r.verdict === "CAUTION" ? "#fbbf24" : "#fb7185");
      ctx.fillStyle = "rgba(148, 226, 255, 0.9)";
      ctx.font = '600 20px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillText(`TRUTH, SYCOPHANCY & BLUFF RADAR · ${Math.round(r.latency_ms || 0)} MS`, 86, cy);
      cy += 48;
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = '500 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      const aLines = wrapCanvasLines(ctx, `"${m.assistantReply || ""}"`, W - 172, 4);
      for (const al of aLines) { ctx.fillText(al, 86, cy); cy += 40; }
      cy += 24;
      const vTxt = (r.verdict_label || r.verdict).toUpperCase();
      ctx.font = '700 22px "IBM Plex Mono", ui-monospace, monospace';
      const vw = ctx.measureText(vTxt).width + 44;
      canvasRoundRect(ctx, 86, cy, vw, 48, 24);
      ctx.fillStyle = r.verdict === "VERIFIED" ? "rgba(34,197,94,0.18)" : "rgba(244,63,94,0.22)";
      ctx.fill();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = accent;
      ctx.fillText(vTxt, 108, cy + 31);
      cy += 84;

      const kpis = [
        [`${r.grounded_pct}%`, "GROUNDED", r.grounded_pct >= 75 ? "#4ade80" : "#fb7185"],
        [`${r.sycophancy_pct}%`, "SYCOPHANCY", r.sycophancy_pct < 35 ? "#4ade80" : "#fb7185"],
        [`${r.bluff_pct}%`, "BLUFF RISK", r.bluff_pct < 40 ? "#4ade80" : "#fb7185"],
      ];
      const bw = 286;
      kpis.forEach(([kv, kl, kc], ki) => {
        const bx = 86 + ki * (bw + 24);
        canvasRoundRect(ctx, bx, cy, bw, 120, 16);
        ctx.fillStyle = "rgba(255,255,255,0.04)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.16)";
        ctx.stroke();
        ctx.fillStyle = kc;
        ctx.font = '700 38px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(kv, bx + 24, cy + 58);
        ctx.fillStyle = "rgba(255,255,255,0.68)";
        ctx.font = '500 18px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(kl, bx + 24, cy + 96);
      });
      cy += 156;
      if (r.note) {
        canvasRoundRect(ctx, 86, cy, W - 172, 160, 16);
        ctx.fillStyle = "rgba(244,63,94,0.12)";
        ctx.fill();
        ctx.strokeStyle = "rgba(251,113,133,0.5)";
        ctx.stroke();
        ctx.fillStyle = "#ffe4e6";
        ctx.font = '500 26px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        wrapCanvasLines(ctx, r.note, W - 220, 4).forEach((nl, ni) => ctx.fillText(nl, 110, cy + 44 + ni * 36));
      }
      tweetText = `Ran Space AI Truth & Sycophancy Radar (${Math.round(r.latency_ms || 0)}ms local AI):\n• Verdict: ${r.verdict_label}\n• Grounded ${r.grounded_pct}% · Sycophancy ${r.sycophancy_pct}% · Bluff ${r.bluff_pct}%\n\nTry free: https://spaceresearch.ai/chat.html`;
    } else {
      // Standard Assistant Message X-Ray Card
      if (userPrompt) {
        ctx.font = '500 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        const uLines = wrapCanvasLines(ctx, userPrompt, W - 260, 2);
        const bh = 38 + uLines.length * 38;
        canvasRoundRect(ctx, 160, cy - 14, W - 246, bh, 18);
        ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        uLines.forEach((ul, ui) => ctx.fillText(ul, 188, cy + 24 + ui * 38));
        cy += bh + 34;
      }

      ctx.fillStyle = "#ffffff";
      const isShort = (m.content || "").trim().length <= 42;
      ctx.font = isShort
        ? '700 62px -apple-system, BlinkMacSystemFont, "Inter", sans-serif'
        : '500 32px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      const ansLines = wrapCanvasLines(ctx, m.content || "", W - 172, isShort ? 2 : 7);
      const ansLH = isShort ? 74 : 44;
      for (const al of ansLines) {
        ctx.fillText(al, 86, cy + (isShort ? 36 : 16));
        cy += ansLH;
      }
      cy += 34;

      const sum = Array.isArray(m.signals) && m.signals.length ? summarizeSignals(m.signals) : null;
      const aud = m.audit || null;
      const boxH = 190 + (aud ? 86 : 0) + (aud && aud.note ? 156 : 0);
      const colAccent = isRisk ? "#fb7185" : "#4ade80";
      canvasRoundRect(ctx, 86, cy, W - 172, Math.min(boxH, H - 140 - cy), 20);
      ctx.fillStyle = isRisk ? "rgba(48, 13, 23, 0.72)" : "rgba(14, 38, 28, 0.65)";
      ctx.fill();
      ctx.strokeStyle = isRisk ? "rgba(251, 113, 133, 0.72)" : "rgba(74, 222, 128, 0.62)";
      ctx.lineWidth = 2;
      ctx.stroke();

      let iy = cy + 48;
      if (sum) {
        ctx.fillStyle = sum.avgCertPct >= 80 ? "#86efac" : (sum.avgCertPct >= 55 ? "#fbbf24" : "#fb7185");
        ctx.font = '700 32px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(`CERTAINTY  ${sum.avgCertPct}%`, 116, iy);
        ctx.fillStyle = "rgba(255,255,255,0.75)";
        ctx.font = '500 20px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(`${sum.forks} FORKS`, W - 260, iy - 2);
        iy += 28;
        const mx = 116, mw = W - 232;
        canvasRoundRect(ctx, mx, iy, mw, 22, 11);
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fill();
        canvasRoundRect(ctx, mx + 2, iy + 2, Math.max(14, Math.round((mw - 4) * (sum.avgCertPct / 100))), 18, 9);
        ctx.fillStyle = sum.avgCertPct >= 80 ? "#4ade80" : (sum.avgCertPct >= 55 ? "#fbbf24" : "#fb7185");
        ctx.fill();
        iy += 58;
      }

      if (aud) {
        const vTxt = (aud.verdict_label || aud.verdict).toUpperCase();
        ctx.font = '700 19px "IBM Plex Mono", ui-monospace, monospace';
        const vw = ctx.measureText(vTxt).width + 38;
        canvasRoundRect(ctx, 116, iy - 16, vw, 42, 21);
        ctx.fillStyle = aud.verdict === "VERIFIED" ? "rgba(34,197,94,0.2)" : "rgba(244,63,94,0.24)";
        ctx.fill();
        ctx.strokeStyle = colAccent;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = aud.verdict === "VERIFIED" ? "#86efac" : "#ffe4e6";
        ctx.fillText(vTxt, 135, iy + 11);

        ctx.fillStyle = "rgba(255,255,255,0.88)";
        ctx.font = '500 20px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillText(`Grounded ${aud.grounded_pct}% · Sycophancy ${aud.sycophancy_pct}% · Bluff ${aud.bluff_pct}%`, 116, iy + 58);
        iy += 86;

        if (aud.note) {
          canvasRoundRect(ctx, 116, iy, W - 232, 126, 12);
          ctx.fillStyle = "rgba(244,63,94,0.14)";
          ctx.fill();
          ctx.strokeStyle = "rgba(251,113,133,0.45)";
          ctx.stroke();
          ctx.fillStyle = "#ffe4e6";
          ctx.font = '500 24px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
          wrapCanvasLines(ctx, `"${aud.note}"`, W - 276, 3).forEach((nl, ni) => ctx.fillText(nl, 138, iy + 38 + ni * 34));
        }
      }

      const certTxt = sum ? `Certainty ${sum.avgCertPct}%` : "Live X-Ray";
      const audTxt = aud ? ` · ${aud.verdict_label} (Bluff ${aud.bluff_pct}%)` : "";
      tweetText = `Asked a local open model on Space Chat: "${userPrompt.slice(0, 68)}"\n\nX-Ray readout: ${certTxt}${audTxt}\n\nThis is why local models beat black-box cloud APIs: https://spaceresearch.ai/chat.html`;
    }

    // 3. Footer watermark
    ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
    ctx.font = '600 24px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText("spaceresearch.ai/chat", 56, H - 44);
    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.font = '500 20px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText("FREE · OPEN LOCAL MODELS", W - 362, H - 44);

    const a = document.createElement("a");
    a.href = cv.toDataURL("image/png");
    a.download = "space-xray-card.png";
    a.click();
    showShareToast(tweetText);
  }

  function renderThread(scrollBottom = true) {
    const inner = $("thread-inner");
    const scroller = $("thread-scroll");
    if (!inner) return;
    const sess = getActiveSession();

    if (!sess || sess.messages.length === 0) {
      inner.innerHTML = `
        <div class="welcome">
          <div class="welcome-crest logo-orbit" aria-hidden="true">
            <svg width="34" height="34" viewBox="0 0 28 28" fill="none">
              <g class="orbit-ring-a"><ellipse cx="14" cy="14" rx="11" ry="4.4" stroke="currentColor" stroke-width="1.35" opacity="0.92"/></g>
              <g class="orbit-ring-b"><ellipse cx="14" cy="14" rx="10.5" ry="4.2" stroke="currentColor" stroke-width="1.1" opacity="0.48"/></g>
              <circle cx="14" cy="14" r="2.7" fill="currentColor"/>
            </svg>
          </div>
          <h1>What should we explore?</h1>
          <p>Open models running on verified community Macs and GPUs, with live confidence on every reply.</p>
          <div class="starters">
            <button type="button" class="starter-card" data-prompt="What is the meaning of life? Reply in one word.">
              <span class="starter-k">Live Confidence</span>
              <span class="starter-v">What is the meaning of life? (1 word)</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/stress Rust or Go for building a fast backend?">
              <span class="starter-k">Framing Check</span>
              <span class="starter-v">/stress Rust or Go — test answer stability</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/compare Rust | Go | Zig | low-latency servers">
              <span class="starter-k">Side-by-Side Compare</span>
              <span class="starter-v">/compare Rust | Go | Zig | low-latency servers</span>
            </button>
          </div>
        </div>
      `;
      inner.querySelectorAll(".starter-card").forEach((card) => {
        card.addEventListener("click", () => sendPrompt(card.getAttribute("data-prompt")));
      });
      updateContextMeter();
      return;
    }

    inner.innerHTML = "";
    const lastCompIdx = lastCompactionIndex(sess);
    sess.messages.forEach((m, idx) => {
      const wrap = document.createElement("div");
      if (m.role === "compaction") {
        wrap.className = "compact-divider";
        const statsLbl = m.beforeToks && m.afterToks
          ? `Context compacted · ${m.beforeToks.toLocaleString("en-US")} → ${m.afterToks.toLocaleString("en-US")} tok${m.savedPct > 0 ? ` (${m.savedPct}% freed)` : ""}`
          : "Context compacted";
        wrap.innerHTML = `
          <details class="compact-card">
            <summary><span>${escHtml(statsLbl)}</span><span style="opacity:0.65;">Summary</span></summary>
            <div class="compact-body">${escHtml(m.content || "")}</div>
          </details>
        `;
        inner.appendChild(wrap);
        return;
      }
      if (m.role === "compare") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=33" alt="S" /></div>
            <div class="msg-body">
              ${renderCompareCardHtml(m)}
              ${m.result ? '<div class="msg-actions" style="margin-top:0.5rem;"><button type="button" class="act-btn btn-share-link">Share Link</button><button type="button" class="act-btn btn-share-card">Share 4:5 Card</button></div>' : ""}
            </div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => downloadShareXrayCard(sess, idx));
        inner.appendChild(wrap);
        return;
      }
      if (m.role === "audit") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=33" alt="S" /></div>
            <div class="msg-body">
              ${renderStandaloneAuditCardHtml(m)}
              ${m.result ? '<div class="msg-actions" style="margin-top:0.5rem;"><button type="button" class="act-btn btn-share-link">Share Link</button><button type="button" class="act-btn btn-share-card">Share 4:5 Card</button></div>' : ""}
            </div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => downloadShareXrayCard(sess, idx));
        inner.appendChild(wrap);
        return;
      }
      if (m.role === "stress") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=33" alt="S" /></div>
            <div class="msg-body">${renderStressCardHtml(m)}</div>
          </div>
        `;
        const bindStressControls = (cardRoot) => {
          cardRoot.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
          cardRoot.querySelector(".btn-share-card")?.addEventListener("click", () => downloadShareXrayCard(sess, idx));
          if (!m.result) return;
          const r = m.result;
          const ordered = Array.isArray(r.framings) ? r.framings : [];
          const rng = cardRoot.querySelector(".stress-range");

          const applyIdx = (nextIdx) => {
            const i = Math.max(0, Math.min(ordered.length - 1, Number(nextIdx)));
            m.stressIdx = i;
            const cur = ordered[i];
            if (!cur) return;
            const curRightPct = Math.max(2, Math.min(100, Number(cur.right_pct ?? cur.yes_pct ?? 50)));
            const curRight = Number(cur.right_pct ?? cur.yes_pct ?? 50) >= 50;
            const curConf = Number(cur.side_conf_pct ?? (curRight ? curRightPct : Math.round((100 - curRightPct) * 10) / 10));
            const curDelta = Math.round((Number(cur.right_pct ?? cur.yes_pct ?? 50) - Number(r.neutral_yes_pct ?? 50)) * 10) / 10;
            const curFlipped = cur.answer !== r.neutral_answer;
            const deltaStr = cur.key === "neutral" ? "Baseline" : `${curDelta >= 0 ? "+" : ""}${curDelta} pts`;

            const lblEl = cardRoot.querySelector(".stress-cur-lbl");
            if (lblEl) lblEl.textContent = cur.label;
            const dEl = cardRoot.querySelector(".stress-cur-delta");
            if (dEl) dEl.textContent = `(${deltaStr})`;
            const pill = cardRoot.querySelector(".stress-ans-pill");
            if (pill) {
              pill.classList.toggle("yes", curRight);
              pill.classList.toggle("no", !curRight);
            }
            const ansTxt = cardRoot.querySelector(".stress-ans-txt");
            if (ansTxt) ansTxt.textContent = `${String(cur.answer || "").toUpperCase()} · ${curConf}%`;
            const flipTag = cardRoot.querySelector(".stress-flip-tag");
            if (flipTag) flipTag.style.display = curFlipped ? "inline-block" : "none";
            const fill = cardRoot.querySelector(".stress-sim-fill");
            if (fill) {
              fill.style.width = curRightPct + "%";
              fill.classList.toggle("yes", curRight);
              fill.classList.toggle("no", !curRight);
            }
            const prEl = cardRoot.querySelector(".stress-prompt-val");
            if (prEl) prEl.textContent = cur.prompt || r.question || "";
            const repEl = cardRoot.querySelector(".stress-reply-val");
            if (repEl) repEl.textContent = cur.reply || (cur.answer + ".");
            const ansBox = cardRoot.querySelector(".stress-ans-box");
            if (ansBox) {
              ansBox.classList.toggle("ans-yes", curRight);
              ansBox.classList.toggle("ans-no", !curRight);
            }
            if (rng && String(rng.value) !== String(i)) rng.value = String(i);
            cardRoot.querySelectorAll(".stress-tick").forEach((el) => {
              el.classList.toggle("active", Number(el.getAttribute("data-sidx")) === i);
            });
          };

          if (rng) {
            rng.addEventListener("input", () => applyIdx(rng.value));
          }
          cardRoot.querySelectorAll(".stress-tick").forEach((el) => {
            el.addEventListener("click", () => {
              const sidx = el.getAttribute("data-sidx");
              if (sidx !== null) applyIdx(sidx);
            });
          });
        };
        bindStressControls(wrap);
        inner.appendChild(wrap);
        return;
      }
      wrap.className = "msg " + m.role + (lastCompIdx >= 0 && idx < lastCompIdx ? " archived" : "");
      if (m.role === "user") {
        const imgsHtml = Array.isArray(m.images) && m.images.length
          ? `<div class="msg-user-imgs">${m.images.map((u) => `<img src="${u}" alt="Uploaded image" />`).join("")}</div>`
          : "";
        wrap.innerHTML = `<div class="msg-user-bubble">${imgsHtml}${escHtml(m.content)}</div>`;
      } else {
        const isLastStreaming = isStreaming && idx === sess.messages.length - 1;
        const hasThink = Boolean((m.reasoning || "").trim());
        const thinkOpen = isLastStreaming && !m.content;
        const thinkSecs = m.thinkMs ? (m.thinkMs / 1000).toFixed(1) + "s" : "";
        const thinkLabel = thinkOpen
          ? '<span><span class="think-dot"></span>Thinking...</span><span>Inspect</span>'
          : `<span>Thought${thinkSecs ? " · " + thinkSecs : ""}</span><span>Toggle</span>`;
        const thinkHtml = hasThink
          ? `<details class="think-box"${thinkOpen ? " open" : ""}><summary>${thinkLabel}</summary><div class="think-content">${escHtml(m.reasoning)}</div></details>`
          : "";
        const hasSignals = Array.isArray(m.signals) && m.signals.length > 0;
        const showXrayView = !isLastStreaming && hasSignals && Boolean(m.showXray);
        const forkBannerHtml = m.forkedFrom
          ? `<div class="fork-banner"><span>&#8627; Counterfactual Fork · transplanted</span><code>${escHtml(formatTokLabel(m.forkedFrom.altTok))}</code><span>instead of</span><code>${escHtml(formatTokLabel(m.forkedFrom.origTok))}</code></div>`
          : "";
        const bodyHtml = showXrayView
          ? renderXrayTokensHtml(m.signals)
          : (m.content
              ? renderMarkdown(m.content) + (isLastStreaming ? '<span class="stream-cursor"></span>' : "")
              : (isLastStreaming && !hasThink
                  ? `<div class="stream-status-pill"><span class="think-dot"></span><span>${escHtml(m.statusText || "Routing to local silicon…")}</span></div>`
                  : ""));

        let xrayStripHtml = "";
        const auditInline = !isLastStreaming ? renderAuditInlineHtml(m.audit, m.auditLoading) : "";
        if (!isLastStreaming && (hasSignals || auditInline)) {
          const sum = hasSignals ? summarizeSignals(m.signals) : null;
          const confNum = sum ? Math.round(sum.avgCertPct) : null;
          const certCls = sum ? (sum.avgCertPct >= 85 ? "xray-m-ok" : (sum.avgCertPct >= 65 ? "xray-m-warn" : "xray-m-risk")) : "";
          const showFix = Boolean(m.audit && m.audit.verdict && m.audit.verdict !== "VERIFIED");
          xrayStripHtml = `
            <div class="xray-strip">
              <div class="feedback-line">
                ${confNum !== null ? `<span class="conf-num ${certCls}">Confidence ${confNum}%</span>` : ""}
                ${auditInline}
              </div>
              <div style="display:inline-flex; gap:0.4rem; align-items:center; flex-shrink:0;">
                ${showFix ? '<button type="button" class="act-btn btn-honest-fix">Honest Take &rarr;</button>' : ""}
                ${sum ? `<button type="button" class="act-btn btn-toggle-xray-msg">${showXrayView ? "Done" : "Inspect"}</button>` : ""}
              </div>
            </div>
          `;
        }

        const meta = m.meta;
        const metaHtml = "";
        let councilHtml = "";
        if (Array.isArray(meta?.council_nodes) && meta.council_nodes.length > 0) {
          const cNodes = meta.council_nodes;
          const pooledGb = meta.pooled_gb || cNodes.reduce((s, n) => s + Number(n.memory_gb || 0), 0).toFixed(1);
          const chipPills = cNodes
            .map((n) => `<span class="council-chip">● ${escHtml(n.display_name || "node")} · ${escHtml(String(n.chip_name || "").replace("NVIDIA GeForce ", ""))} (${n.memory_gb} GB · ${escHtml(n.model_id || "")})</span>`)
            .join("");
          const detailRows = cNodes
            .map((n) => `
              <div class="council-node-item">
                <div class="council-node-head">
                  <span>${escHtml(n.role || "Node")} · ${escHtml(n.display_name || "")} (${escHtml(String(n.chip_name || "").replace("NVIDIA GeForce ", ""))})</span>
                  <span>${escHtml(n.model_id || "")} · ${Math.round(Number(n.latency_ms || 0))}ms</span>
                </div>
                <div class="council-node-txt">${escHtml(n.excerpt || "")}</div>
              </div>
            `)
            .join("");
          councilHtml = `
            <details class="council-box">
              <summary>
                <span>&#9679; Multi-Computer Swarm · ${cNodes.length} Devices · ${pooledGb} GB Pooled RAM</span>
                <span style="opacity:0.72;">Inspect each machine &darr;</span>
              </summary>
              <div class="council-chips">${chipPills}</div>
              <div class="council-grid">${detailRows}</div>
            </details>
          `;
        }
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=33" alt="S" /></div>
            <div class="msg-body">
              ${forkBannerHtml}
              ${councilHtml}
              <div class="msg-think-slot">${thinkHtml}</div>
              <div class="msg-md">${bodyHtml || (hasThink ? "" : '<span class="stream-cursor"></span>')}</div>
              ${xrayStripHtml}
              ${isLastStreaming ? "" : `
                <div class="msg-actions">
                  <button type="button" class="act-btn btn-copy-msg">Copy</button>
                  <button type="button" class="act-btn btn-share-link">Share</button>
                  ${idx === sess.messages.length - 1 ? '<button type="button" class="act-btn btn-regen-msg">Regenerate</button>' : ""}
                </div>
              `}
            </div>
          </div>
        `;
        const cp = wrap.querySelector(".btn-copy-msg");
        if (cp) {
          cp.addEventListener("click", () => {
            navigator.clipboard.writeText(m.content || "").then(() => {
              cp.textContent = "Copied ✓";
              setTimeout(() => { cp.textContent = "Copy"; }, 1400);
            });
          });
        }
        const rg = wrap.querySelector(".btn-regen-msg");
        if (rg) rg.addEventListener("click", () => regenerateLast());
        const sl = wrap.querySelector(".btn-share-link");
        if (sl) sl.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        const sc = wrap.querySelector(".btn-share-card");
        if (sc) sc.addEventListener("click", () => downloadShareXrayCard(sess, idx));
        const ab = wrap.querySelector(".btn-audit-msg");
        if (ab) ab.addEventListener("click", () => { runMessageAudit(sess, idx); renderThread(false); });
        const hf = wrap.querySelector(".btn-honest-fix");
        if (hf) hf.addEventListener("click", () => requestHonestRewrite(sess, idx));

        const toggleXray = () => {
          closeXrayPopover();
          m.showXray = !m.showXray;
          renderThread(false);
        };
        const xBtn = wrap.querySelector(".btn-toggle-xray-msg");
        if (xBtn) xBtn.addEventListener("click", toggleXray);
        const spk = wrap.querySelector(".xray-spark-wrap");
        if (spk) {
          spk.addEventListener("click", () => {
            if (!m.showXray) {
              m.showXray = true;
              renderThread(false);
            }
          });
        }
        if (showXrayView) {
          wrap.querySelectorAll(".xtok").forEach((tokEl) => {
            tokEl.addEventListener("click", (ev) => {
              ev.stopPropagation();
              const ti = Number(tokEl.getAttribute("data-ti"));
              openTokenPopover(tokEl, sess, idx, ti);
            });
          });
        }
      }
      inner.appendChild(wrap);
    });

    updateContextMeter();
    if (scrollBottom && scroller) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  }

  let streamThinkEl = null;
  let streamMdEl = null;
  let lastRenderTs = 0;

  function cacheStreamingDomRefs() {
    const inner = $("thread-inner");
    if (!inner) {
      streamThinkEl = null;
      streamMdEl = null;
      return;
    }
    const thinkSlots = inner.querySelectorAll(".msg.assistant .msg-think-slot");
    const mdNodes = inner.querySelectorAll(".msg.assistant .msg-md");
    streamThinkEl = thinkSlots[thinkSlots.length - 1] || null;
    streamMdEl = mdNodes[mdNodes.length - 1] || null;
  }

  function setStreamingUI(streaming) {
    isStreaming = streaming;
    const btn = $("btn-send");
    if (!btn) return;
    if (streaming) {
      btn.textContent = "Stop";
      btn.classList.add("stop");
    } else {
      btn.textContent = "Send";
      btn.classList.remove("stop");
    }
  }

  async function streamAssistantReply(sess) {
    const selectedModel = $("model-select")?.value || "qwen3.8-27b";
    const selectedNode = $("node-select")?.value || "auto";
    const targetNode = selectedNode !== "auto" ? selectedNode : null;
    const maxTokens = 4096;
    const reasoningEffort = "none";
    sess.model = selectedModel;
    sess.targetNode = selectedNode;

    const apiMessages = buildApiMessages(sess);
    const initStatus =
      selectedModel === "space-70b-shard"
        ? "Connecting to Multi-Node Sharded Cluster…"
        : selectedModel === "space-council"
          ? "Querying all online nodes in parallel…"
          : "Routing to fastest local node…";
    const assistantMsg = { role: "assistant", content: "", reasoning: "", statusText: initStatus, thinkMs: 0, meta: null, signals: [], showXray: false };
    sess.messages.push(assistantMsg);
    sess.updatedAt = Date.now();
    saveSessions();

    setStreamingUI(true);
    renderThread(true);
    cacheStreamingDomRefs();
    lastRenderTs = 0;

    const scroller = $("thread-scroll");
    activeAbort = new AbortController();
    let thinkStart = 0;
    let rafId = 0;

    try {
      const reqBody = {
        model: selectedModel,
        messages: apiMessages,
        stream: true,
        max_tokens: maxTokens,
        reasoning_effort: reasoningEffort,
        session_id: sess.id,
        xray: xrayEnabled,
        xray_top_k: 5,
      };
      if (targetNode) reqBody.target_node = targetNode;
      const res = await fetch(API_BASE + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
        signal: activeAbort.signal,
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        assistantMsg.content = `Could not complete request: ${detail}`;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      function scheduleRender() {
        if (rafId || !isStreaming) return;
        rafId = requestAnimationFrame((ts) => {
          rafId = 0;
          if (!isStreaming) return;
          if (assistantMsg.content.length > 3500 && ts - lastRenderTs < 32) {
            scheduleRender();
            return;
          }
          lastRenderTs = ts;
          const nearBottom = scroller ? (scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140) : true;
          updateStreamingLastMessage(assistantMsg);
          if (nearBottom && scroller) scroller.scrollTop = scroller.scrollHeight;
        });
      }

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";

        for (const chunk of parts) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const dataStr = line.slice(5).trim();
          if (dataStr === "[DONE]") continue;
          try {
            const parsed = JSON.parse(dataStr);
            const dObj = parsed.choices?.[0]?.delta || {};
            if (dObj.reasoning_content) {
              if (!thinkStart) thinkStart = performance.now();
              assistantMsg.reasoning += dObj.reasoning_content;
              assistantMsg.thinkMs = Math.round(performance.now() - thinkStart);
            }
            if (parsed.space_status && !assistantMsg.content) {
              assistantMsg.statusText = String(parsed.space_status);
            }
            if (dObj.content) {
              assistantMsg.statusText = "";
              if (thinkStart && !assistantMsg.content) {
                assistantMsg.thinkMs = Math.round(performance.now() - thinkStart);
              }
              assistantMsg.content += dObj.content;
              if (parsed.token_signal) {
                assistantMsg.signals.push(parsed.token_signal);
              }
            }
            if (parsed.longinus_meta) assistantMsg.meta = parsed.longinus_meta;
          } catch (_ignore) {}
        }

        scheduleRender();
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        assistantMsg.content = assistantMsg.content || `Connection interrupted (${err.message}). Try again.`;
      }
    } finally {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      activeAbort = null;
      assistantMsg.statusText = "";
      setStreamingUI(false);
      saveSessions();
      renderThread(false);
      if (xrayEnabled && assistantMsg.content && !assistantMsg.content.startsWith("Could not complete")) {
        const msgIndex = sess.messages.indexOf(assistantMsg);
        if (msgIndex >= 0) {
          runMessageAudit(sess, msgIndex);
        }
      }
    }
  }

  function updateStreamingLastMessage(msg) {
    const lastThink = streamThinkEl;
    if (lastThink && msg.reasoning) {
      const stillThinking = !msg.content;
      const secs = msg.thinkMs ? (msg.thinkMs / 1000).toFixed(1) + "s" : "";
      let box = lastThink.querySelector("details.think-box");
      if (!box) {
        lastThink.innerHTML = `<details class="think-box" open><summary></summary><div class="think-content"></div></details>`;
        box = lastThink.querySelector("details.think-box");
      }
      if (!stillThinking && !box.dataset.collapsedOnce) {
        box.removeAttribute("open");
        box.dataset.collapsedOnce = "1";
      }
      const sum = box.querySelector("summary");
      const contentEl = box.querySelector(".think-content");
      if (sum) {
        sum.innerHTML = stillThinking
          ? '<span><span class="think-dot"></span>Thinking...</span><span>Live</span>'
          : `<span>Thought${secs ? " · " + secs : ""}</span><span>Toggle</span>`;
      }
      if (contentEl) {
        contentEl.textContent = msg.reasoning;
        if (stillThinking) contentEl.scrollTop = contentEl.scrollHeight;
      }
    }
    const lastMd = streamMdEl;
    if (lastMd) {
      if (msg.content) {
        lastMd.innerHTML = renderMarkdown(msg.content) + '<span class="stream-cursor"></span>';
      } else if (msg.reasoning) {
        lastMd.innerHTML = "";
      } else if (msg.statusText) {
        lastMd.innerHTML = `<div class="stream-status-pill"><span class="think-dot"></span><span>${escHtml(msg.statusText)}</span></div>`;
      }
    }
  }

  async function compactActiveSession(focusNote = "") {
    if (isStreaming) return false;
    const sess = getActiveSession();
    if (!sess) return false;
    const apiMsgs = buildApiMessages(sess);
    const turns = apiMsgs.filter((m) => m.role === "user" || m.role === "assistant");
    if (turns.length < 2) return false;

    const beforeToks = activeContextTokens(sess);
    const btn = $("btn-compact");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Compacting...";
    }
    setStreamingUI(true);
    activeAbort = new AbortController();

    const transcript = apiMsgs
      .map((m) => {
        const txt = typeof m.content === "string"
          ? m.content
          : Array.isArray(m.content)
            ? m.content.map((p) => (p.type === "text" ? p.text : "[image]")).join(" ")
            : "";
        return `${m.role.toUpperCase()}: ${txt}`;
      })
      .join("\n\n");
    const focusClause = focusNote ? `\nExtra user focus instruction: ${focusNote}\n` : "";
    const compactPrompt = [
      {
        role: "system",
        content:
          "You compact a multi-turn conversation into a tight, high-signal handoff summary so future turns can continue seamlessly with minimal context tokens. " +
          "Preserve: user goals, key technical decisions, exact names/identifiers/code/numbers, constraints, and current state. " +
          "Omit filler and repetition. Output clean concise bullet points only (under 280 words).",
      },
      {
        role: "user",
        content: `Compact this conversation into a crisp continuation memory.${focusClause}\n\n${transcript}`,
      },
    ];

    try {
      const res = await fetch(API_BASE + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: $("model-select")?.value || "qwen3.8-27b",
          messages: compactPrompt,
          stream: true,
          max_tokens: 1024,
          reasoning_effort: "none",
        }),
        signal: activeAbort.signal,
      });
      if (!res.ok) throw new Error("HTTP " + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      let summary = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const chunk of parts) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const dataStr = line.slice(5).trim();
          if (dataStr === "[DONE]") continue;
          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta?.content;
            if (delta) summary += delta;
          } catch (_ignore) {}
        }
      }

      summary = summary.trim();
      if (!summary) {
        // Fallback concise extractive summary if model only emitted reasoning
        summary = turns
          .slice(-4)
          .map((m) => `- ${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 180).replace(/\s+/g, " ")}`)
          .join("\n");
      }

      const afterToks = estimateTokens(summary) + 12;
      const savedPct = beforeToks > 0 ? Math.max(0, Math.round(((beforeToks - afterToks) / beforeToks) * 100)) : 0;
      sess.messages.push({
        role: "compaction",
        content: summary,
        beforeToks,
        afterToks,
        savedPct,
        createdAt: Date.now(),
      });
      sess.updatedAt = Date.now();
      saveSessions();
      return true;
    } catch (_err) {
      return false;
    } finally {
      activeAbort = null;
      setStreamingUI(false);
      renderThread(true);
    }
  }

  async function sendPrompt(explicitText = null) {
    if (isStreaming) {
      if (activeAbort) activeAbort.abort();
      return;
    }
    const inp = $("composer-input");
    const text = (explicitText !== null ? explicitText : (inp?.value || "")).trim();
    const images = explicitText === null ? pendingImages.slice() : [];
    if (!text && images.length === 0) return;

    // Native /stress command (5-Framing Belief Stability & Sycophancy Stress-Test)
    if (text && /^\/stress(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await runStressTest(text);
      return;
    }

    // Native /audit command (Truth, Sycophancy & Bluff Radar)
    if (text && /^\/audit(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await runStandaloneAudit(text);
      return;
    }

    // Native /compare command (Antisymmetric Paired-Swap Ranking: "/compare Rust | Go | latency")
    if (text && /^\/compare(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await runAntisymmetricCompare(text);
      return;
    }

    // Native /compact command (optionally: "/compact focus on Python code")
    if (text && /^\/compact(\s|$)/i.test(text)) {
      const focusNote = text.replace(/^\/compact\s*/i, "").trim();
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await compactActiveSession(focusNote);
      return;
    }

    const titleSeed = text || "Image analysis";
    let sess = getActiveSession();
    if (!sess) sess = createSession(titleSeed.slice(0, 42));
    if (sess.messages.length === 0) {
      sess.title = titleSeed.slice(0, 44) + (titleSeed.length > 44 ? "…" : "");
    }

    // Auto-compact safeguard before approaching the 24.5K slot ceiling
    if (activeContextTokens(sess) >= 20500) {
      await compactActiveSession();
    }

    const userMsg = { role: "user", content: text };
    if (images.length > 0) userMsg.images = images;
    sess.messages.push(userMsg);
    if (explicitText === null) {
      pendingImages = [];
      renderAttachStrip();
    }
    if (inp && explicitText === null) {
      inp.value = "";
      inp.style.height = "auto";
    }
    renderSidebar();
    await streamAssistantReply(sess);
  }

  async function regenerateLast() {
    if (isStreaming) return;
    const sess = getActiveSession();
    if (!sess || sess.messages.length < 2) return;
    if (sess.messages[sess.messages.length - 1].role === "assistant") {
      sess.messages.pop();
    }
    await streamAssistantReply(sess);
  }

  function exportActiveMarkdown() {
    const sess = getActiveSession();
    if (!sess || !sess.messages.length) return;
    const md = sess.messages
      .map((m) => {
        const hdr = m.role === "user" ? "User" : (m.role === "compaction" ? "Context Summary" : "Space AI");
        return `## ${hdr}\n\n${m.content}`;
      })
      .join("\n\n---\n\n");
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (sess.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 32) || "space-chat") + ".md";
    a.click();
  }

  function closeMobileDrawer() {
    $("sidebar")?.classList.remove("mobile-open");
    $("drawer-backdrop")?.classList.remove("open");
  }

  async function refreshFleetBadge() {
    try {
      const res = await fetch(API_BASE + "/v1/network/stats");
      if (!res.ok) return;
      const st = await res.json();
      const el = $("sb-fleet-text");
      const liveCount = Number(st.active_nodes || 0);
      const liveMem = Number(st.total_memory_gb || 0);
      if (el) {
        el.textContent = `${liveCount} Devices · ${liveMem.toFixed(1)} GB`;
      }
      const mSel = $("model-select");
      if (mSel) {
        const councilOpt = mSel.querySelector('option[value="space-council"]');
        if (councilOpt && liveCount > 0) {
          councilOpt.textContent = `${liveCount}-Node Swarm · ${liveMem.toFixed(1)} GB Pooled`;
        }
        const shardOpt = mSel.querySelector('option[value="space-70b-shard"]');
        if (shardOpt) {
          if (st.rpc_shard_label) {
            shardOpt.textContent = String(st.rpc_shard_label);
          } else if (liveMem >= 90) {
            shardOpt.textContent = `GLM-4.5-Air 106B MoE · Sharded RPC (${liveMem.toFixed(1)} GB)`;
          } else if (liveMem > 0) {
            shardOpt.textContent = `Llama-3.3 70B · Sharded RPC (${liveMem.toFixed(1)} GB)`;
          } else {
            shardOpt.textContent = `Sharded Model · Multi-Node RPC (Beta)`;
          }
        }
      }
      const sel = $("node-select");
      if (sel && Array.isArray(st.nodes)) {
        const prev = sel.value || "auto";
        const onlineNodes = st.nodes.filter((n) => n.online !== false && n.status !== "offline");
        sel.innerHTML = "";
        const autoOpt = document.createElement("option");
        autoOpt.value = "auto";
        autoOpt.textContent = `Auto · Best Node (${onlineNodes.length} online)`;
        sel.appendChild(autoOpt);
        for (const n of onlineNodes) {
          const opt = document.createElement("option");
          opt.value = n.node_id || "";
          const name = (n.display_name || "").trim() || `node-${String(n.node_id || "").slice(0, 6)}`;
          const chipShort = (n.chip_name || "").replace("NVIDIA GeForce ", "");
          opt.textContent = `● ${name} · ${chipShort} (${n.memory_gb} GB)`;
          sel.appendChild(opt);
        }
        const stillExists = Array.from(sel.options).some((o) => o.value === prev);
        sel.value = stillExists ? prev : "auto";
      }
    } catch (_e) {}
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("thread-inner")?.addEventListener("click", (e) => {
      const btn = e.target.closest(".code-copy");
      if (!btn) return;
      const code = decodeURIComponent(btn.getAttribute("data-code") || "");
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = "Copied ✓";
        setTimeout(() => { btn.textContent = "Copy"; }, 1400);
      });
    });
    renderSidebar();
    renderThread();
    updateXrayButton();
    syncCoordinatorEndpoint().finally(async () => {
      refreshFleetBadge();
      try {
        const params = new URLSearchParams(window.location.search);
        const shareId = (params.get("share") || "").trim();
        const initQ = (params.get("q") || "").trim();
        if (shareId) {
          const r = await fetch(API_BASE + "/v1/shares/" + encodeURIComponent(shareId));
          if (r.ok) {
            const snap = await r.json();
            const pl = snap.payload || {};
            const msgs = [];
            if (pl.userPrompt && pl.message?.role === "assistant") {
              msgs.push({ role: "user", content: pl.userPrompt });
            }
            if (pl.message) msgs.push(pl.message);
            if (msgs.length) {
              const newSess = {
                id: genId(),
                title: "Shared · " + String(snap.title || "X-Ray").slice(0, 42),
                messages: msgs,
                updatedAt: Date.now(),
              };
              sessions.unshift(newSess);
              activeId = newSess.id;
              saveSessions();
              renderSidebar();
              renderThread(true);
              showShareToast(
                buildShareTweet({ kind: snap.kind, title: snap.title, userPrompt: pl.userPrompt || "", payload: pl }, window.location.href),
                `Loaded shared X-Ray (${snap.views || 1} views) · Click any token to fork`
              );
            }
          }
        } else if (initQ) {
          createSession();
          const inpEl = $("composer-input");
          if (inpEl) inpEl.value = initQ;
          sendPrompt();
        }
      } catch (_e) {}
    });
    setInterval(() => { if (!document.hidden) refreshFleetBadge(); }, 15000);

    $("btn-xray")?.addEventListener("click", () => {
      xrayEnabled = !xrayEnabled;
      try { localStorage.setItem(XRAY_PREF_KEY, xrayEnabled ? "1" : "0"); } catch (_e) {}
      updateXrayButton();
    });
    document.addEventListener("click", (e) => {
      const pop = $("xray-popover");
      if (pop && pop.classList.contains("open") && !pop.contains(e.target) && !e.target.closest(".xtok")) {
        closeXrayPopover();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeXrayPopover();
    });
    $("thread-scroll")?.addEventListener("scroll", () => closeXrayPopover(), { passive: true });

    $("btn-new-chat")?.addEventListener("click", () => createSession());
    $("sb-search")?.addEventListener("input", () => renderSidebar());
    $("btn-compact")?.addEventListener("click", () => compactActiveSession());
    $("btn-export")?.addEventListener("click", exportActiveMarkdown);
    $("btn-send")?.addEventListener("click", () => sendPrompt());
    $("model-select")?.addEventListener("change", (e) => preloadModel(e.target.value));

    $("btn-toggle-sb")?.addEventListener("click", () => {
      if (window.innerWidth <= 860) {
        $("sidebar")?.classList.toggle("mobile-open");
        $("drawer-backdrop")?.classList.toggle("open");
      } else {
        $("sidebar")?.classList.toggle("collapsed");
      }
    });
    $("drawer-backdrop")?.addEventListener("click", closeMobileDrawer);

    $("btn-attach")?.addEventListener("click", () => $("file-image")?.click());
    $("file-image")?.addEventListener("change", (e) => {
      if (e.target.files?.length) addImageFiles(e.target.files);
      e.target.value = "";
    });

    const cbox = $("composer-box");
    if (cbox) {
      ["dragenter", "dragover"].forEach((ev) => {
        cbox.addEventListener(ev, (e) => {
          e.preventDefault();
          cbox.classList.add("drag-over");
        });
      });
      ["dragleave", "drop"].forEach((ev) => {
        cbox.addEventListener(ev, (e) => {
          e.preventDefault();
          cbox.classList.remove("drag-over");
        });
      });
      cbox.addEventListener("drop", (e) => {
        if (e.dataTransfer?.files?.length) addImageFiles(e.dataTransfer.files);
      });
    }

    const inp = $("composer-input");
    if (inp) {
      inp.addEventListener("paste", (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        const imgFiles = [];
        for (const it of items) {
          if (it.type && it.type.startsWith("image/")) {
            const f = it.getAsFile();
            if (f) imgFiles.push(f);
          }
        }
        if (imgFiles.length) addImageFiles(imgFiles);
      });
      inp.addEventListener("input", () => {
        inp.style.height = "auto";
        inp.style.height = Math.min(inp.scrollHeight, 200) + "px";
      });
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendPrompt();
        }
      });
    }
  });
})();
