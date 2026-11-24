/* Space AI Desktop — Streaming Glass-Box Client (macOS / Windows / Linux) */
(() => {
  "use strict";

  const bridge = window.spaceDesktop || null;
  const IS_MAC = (bridge?.platform || navigator.platform || "").toLowerCase().includes("darwin") ||
    (navigator.platform || "").toLowerCase().includes("mac");

  const DEFAULT_API_BASE = "https://api.spaceresearch.ai";
  const FALLBACK_API_BASE = "https://api.longinusresearch.com";
  const DEFAULT_LOCAL_PROXY = "http://127.0.0.1:8377";
  const AVATAR_SRC = "../../assets/mark.svg";

  const SLASH_COMMANDS = [
    {
      cmd: "/stress",
      insert: "/stress ",
      desc: "5-Framing belief stability & sycophancy stress-test",
      example: "/stress Rust or Go for a fast backend?",
    },
    {
      cmd: "/compare",
      insert: "/compare ",
      desc: "Antisymmetric paired-swap bias-free ranking",
      example: "/compare Rust | Go | Zig | low-latency servers",
    },
    {
      cmd: "/audit",
      insert: "/audit ",
      desc: "Truth, sycophancy & overconfident bluff radar",
      example: "/audit User: ... | AI: ...",
    },
    {
      cmd: "/compact",
      insert: "/compact",
      desc: "Compact conversation history into a crisp memory",
      example: "/compact focus on code decisions",
    },
  ];

  let settings = {
    endpointMode: "auto",
    apiBase: DEFAULT_API_BASE,
    fallbackApiBase: FALLBACK_API_BASE,
    localProxyUrl: DEFAULT_LOCAL_PROXY,
    customApiBase: "",
    apiKey: "",
    sessionToken: "",
    route: "",
    privacy: "",
    region: "",
    systemPrompt: "",
    maxTokens: 4096,
    xrayEnabled: true,
    starfieldEnabled: true,
    theme: "dark",
    alwaysOnTop: false,
    globalShortcut: "CommandOrControl+Shift+Space",
    globalShortcutEnabled: true,
    minimizeToTray: true,
    launchAtLogin: false,
    sidebarCollapsed: false,
    lastModel: "qwen3.8-27b",
    lastNode: "auto",
  };

  let activeApiBase = DEFAULT_API_BASE;
  let sessions = [];
  let activeId = null;
  let activeAbort = null;
  let isStreaming = false;
  let pendingImages = [];
  let lastFleetStats = null;
  let saveSessionsTimer = null;
  let slashActiveIdx = 0;
  let cmdActiveIdx = 0;
  let cmdEntries = [];

  const $ = (id) => document.getElementById(id);

  // ---------- Ambient Embedding Manifold Canvas (matching chat.html v17) ----------
  let starRaf = 0;
  let starCtx = null;
  let manifoldNodes = [];
  const DARK_COLS = [[235, 235, 240], [190, 190, 200], [215, 215, 222], [160, 160, 172]];
  const LIGHT_COLS = [[30, 34, 48], [60, 66, 84], [90, 98, 120], [115, 122, 142]];

  function initStarfield() {
    const c = $("chat-cosmos-canvas");
    if (!c || !c.getContext) return;
    starCtx = c.getContext("2d");
    const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      c.width = Math.floor(window.innerWidth * dpr);
      c.height = Math.floor(window.innerHeight * dpr);
      starCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.min(84, Math.floor((window.innerWidth * window.innerHeight) / 16000));
      manifoldNodes = Array.from({ length: count }, (_, idx) => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        r: 0.7 + Math.random() * 1.15,
        phase: Math.random() * Math.PI * 2,
        spd: 0.004 + Math.random() * 0.008,
        idx,
      }));
    }

    function draw() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      if (!settings.starfieldEnabled) {
        starCtx.clearRect(0, 0, w, h);
        starRaf = 0;
        return;
      }
      const isLight = settings.theme === "light";
      const cols = isLight ? LIGHT_COLS : DARK_COLS;
      starCtx.clearRect(0, 0, w, h);
      for (let i = 0; i < manifoldNodes.length; i++) {
        const s = manifoldNodes[i];
        if (!reduced) s.phase += s.spd;
        const px = s.x + Math.sin(s.phase) * 8;
        const py = s.y + Math.cos(s.phase * 0.8) * 7;
        const rgb = cols[s.idx % cols.length];
        for (let j = i + 1; j < manifoldNodes.length; j++) {
          const o = manifoldNodes[j];
          const dx = px - o.x;
          const dy = py - o.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 9500) {
            const edgeAlpha = (1 - d2 / 9500) * (isLight ? 0.12 : 0.10);
            starCtx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${edgeAlpha.toFixed(3)})`;
            starCtx.lineWidth = 0.7;
            starCtx.beginPath();
            starCtx.moveTo(px, py);
            starCtx.lineTo(o.x, o.y);
            starCtx.stroke();
          }
        }
        const a = 0.16 + 0.34 * (0.5 + 0.5 * Math.sin(s.phase));
        starCtx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${a.toFixed(2)})`;
        starCtx.beginPath();
        starCtx.arc(px, py, s.r, 0, Math.PI * 2);
        starCtx.fill();
      }
      if (!reduced) starRaf = requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        if (starRaf) cancelAnimationFrame(starRaf);
        starRaf = 0;
      } else if (!reduced && settings.starfieldEnabled && !starRaf) {
        draw();
      }
    });

    window.__spaceRestartStars = () => {
      c.classList.toggle("hidden", !settings.starfieldEnabled);
      if (starRaf) cancelAnimationFrame(starRaf);
      starRaf = 0;
      if (settings.starfieldEnabled) draw();
    };
    window.__spaceRestartStars();
  }

  // ---------- Persistence & Endpoint Resolution ----------
  function buildRequestHeaders() {
    const h = {
      "Content-Type": "application/json",
      "X-Space-Client": "space-desktop",
    };
    const key = String(settings.apiKey || "").trim();
    if (key) {
      h["Authorization"] = `Bearer ${key}`;
      h["X-API-Key"] = key;
    }
    if (settings.sessionToken) h["X-Space-Session"] = settings.sessionToken;
    if (settings.route) h["X-Space-Route"] = settings.route;
    if (settings.privacy) h["X-Space-Privacy"] = settings.privacy;
    if (settings.region) h["X-Space-Region"] = settings.region;
    return h;
  }

  function updateEndpointBadge() {
    const el = $("endpoint-badge");
    if (!el) return;
    el.classList.remove("local", "custom");
    if (settings.endpointMode === "local-proxy") {
      el.textContent = "Local Proxy";
      el.classList.add("local");
      el.title = `Connected via local space proxy (${activeApiBase})`;
    } else if (settings.endpointMode === "custom" && settings.customApiBase) {
      el.textContent = "Custom";
      el.classList.add("custom");
      el.title = `Connected to custom endpoint (${activeApiBase})`;
    } else {
      el.textContent = activeApiBase === FALLBACK_API_BASE ? "Fallback" : "Space Net";
      el.title = `Connected to Space Network (${activeApiBase})`;
    }
  }

  async function syncCoordinatorEndpoint() {
    if (settings.endpointMode === "local-proxy") {
      activeApiBase = (settings.localProxyUrl || DEFAULT_LOCAL_PROXY).replace(/\/+$/, "");
      updateEndpointBadge();
      return;
    }
    if (settings.endpointMode === "custom" && settings.customApiBase) {
      activeApiBase = settings.customApiBase.replace(/\/+$/, "");
      updateEndpointBadge();
      return;
    }

    activeApiBase = (settings.apiBase || DEFAULT_API_BASE).replace(/\/+$/, "");
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 3500);
      const ping = await fetch(activeApiBase + "/health", { signal: ctrl.signal });
      clearTimeout(t);
      if (ping.ok) {
        updateEndpointBadge();
        return;
      }
    } catch {
      const fb = (settings.fallbackApiBase || FALLBACK_API_BASE).replace(/\/+$/, "");
      if (activeApiBase !== fb) activeApiBase = fb;
    }
    updateEndpointBadge();
  }

  async function persistSessionsNow() {
    if (saveSessionsTimer) {
      clearTimeout(saveSessionsTimer);
      saveSessionsTimer = null;
    }
    const trimmed = sessions.slice(0, 200);
    if (bridge?.saveSessions) {
      try {
        await bridge.saveSessions(trimmed);
        return;
      } catch {}
    }
    try {
      localStorage.setItem("space_desktop_sessions_v1", JSON.stringify(trimmed.slice(0, 40)));
    } catch {}
  }

  function saveSessions() {
    if (saveSessionsTimer) clearTimeout(saveSessionsTimer);
    saveSessionsTimer = setTimeout(() => {
      persistSessionsNow();
    }, 120);
  }

  async function persistSettings(partial) {
    if (bridge?.saveSettings) {
      try {
        settings = await bridge.saveSettings(partial);
      } catch {
        settings = { ...settings, ...partial };
      }
    } else {
      settings = { ...settings, ...partial };
    }
    applyRendererSettings();
  }

  function applyRendererSettings() {
    const theme = settings.theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    const tBtn = $("btn-theme-toggle");
    if (tBtn) {
      tBtn.textContent = theme === "light" ? "☾" : "☀";
      tBtn.title = theme === "light" ? "Switch to Deep-Space Dark Mode" : "Switch to Daylight Mode";
    }
    updateXrayButton();
    updatePinButton();
    updateEndpointBadge();
    if (window.__spaceRestartStars) window.__spaceRestartStars();
    if ($("sidebar")) {
      $("sidebar").classList.toggle("collapsed", Boolean(settings.sidebarCollapsed));
      document.body.classList.toggle("sidebar-collapsed", Boolean(settings.sidebarCollapsed));
    }
  }

  function copyToClipboard(text) {
    if (bridge?.copyText) return bridge.copyText(text);
    return navigator.clipboard?.writeText(text) || Promise.resolve();
  }

  function openExternalUrl(url) {
    if (bridge?.openExternal) return bridge.openExternal(url);
    window.open(url, "_blank", "noopener");
  }

  // ---------- Utility & Markdown ----------
  function uid() {
    return "c_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7);
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
      if (listTag) {
        out.push(`</${listTag}>`);
        listTag = null;
      }
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
        if (!inCode) {
          inCode = true;
          codeLang = fence[1] || "code";
          codeBuf = [];
        } else {
          flushCode();
        }
        continue;
      }
      if (inCode) {
        codeBuf.push(line);
        continue;
      }

      const trimmed = line.trim();
      if (trimmed.includes("|") && trimmed.startsWith("|")) {
        flushList();
        tableRows.push(trimmed);
        continue;
      }
      flushTable();

      if (!trimmed) {
        flushList();
        continue;
      }
      if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
        flushList();
        out.push("<hr/>");
        continue;
      }

      const h = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        flushList();
        out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`);
        continue;
      }

      const bq = trimmed.match(/^>\s?(.*)$/);
      if (bq) {
        flushList();
        out.push(`<blockquote>${inlineMd(bq[1])}</blockquote>`);
        continue;
      }

      const ul = trimmed.match(/^[-*•]\s+(.*)$/);
      if (ul) {
        if (listTag !== "ul") {
          flushList();
          listTag = "ul";
          out.push("<ul>");
        }
        out.push(`<li>${inlineMd(ul[1])}</li>`);
        continue;
      }
      const ol = trimmed.match(/^\d+\.\s+(.*)$/);
      if (ol) {
        if (listTag !== "ol") {
          flushList();
          listTag = "ol";
          out.push("<ol>");
        }
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

  // ---------- Image Attachments ----------
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
      } catch {}
    }
    renderAttachStrip();
  }

  // ---------- Sessions ----------
  function getActiveSession() {
    return sessions.find((s) => s.id === activeId) || null;
  }

  function preloadModel(modelId) {
    if (!modelId) return;
    fetch(activeApiBase + "/v1/models/preload", {
      method: "POST",
      headers: buildRequestHeaders(),
      body: JSON.stringify({ model: modelId }),
    }).catch(() => {});
  }

  function createSession(initialTitle = "New chat") {
    const s = {
      id: uid(),
      title: initialTitle,
      model: $("model-select")?.value || settings.lastModel || "qwen3.8-27b",
      targetNode: $("node-select")?.value || settings.lastNode || "auto",
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

  function selectSession(s) {
    if (!s) return;
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
      if (ev.key === "Enter") {
        ev.preventDefault();
        commit(true);
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        commit(false);
      }
    });
    input.addEventListener("blur", () => commit(true));
  }

  function renderSidebar() {
    const listEl = $("sb-list");
    if (!listEl) return;
    const q = ($("sb-search")?.value || "").trim().toLowerCase();
    const filtered = q ? sessions.filter((s) => (s.title || "").toLowerCase().includes(q)) : sessions;
    listEl.innerHTML = "";
    if (!filtered.length) {
      listEl.innerHTML = `<div class="sb-empty">${q ? "No matching chats." : "No conversations yet."}</div>`;
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
        selectSession(s);
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

  // ---------- Context Accounting ----------
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
    if (settings.systemPrompt && settings.systemPrompt.trim()) {
      out.push({ role: "system", content: settings.systemPrompt.trim() });
    }
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
    const activeMsgs = sess
      ? sess.messages.slice(cIdx >= 0 ? cIdx + 1 : 0).filter((m) => m.role === "user" || m.role === "assistant")
      : [];
    const canCompact = !isStreaming && activeMsgs.length >= 2;
    if (btn) {
      btn.disabled = !canCompact;
      if (!isStreaming) btn.textContent = "Compact";
    }
    if (meter) {
      const toks = activeContextTokens(sess);
      if (!sess || !sess.messages.length) {
        meter.textContent = "Space AI Desktop · Open community models";
        meter.classList.remove("ctx-warn");
      } else {
        const pct = Math.min(100, Math.round((toks / 24576) * 100));
        meter.textContent = `Context ${pct}% · ${toks.toLocaleString("en-US")} est. tokens`;
        meter.classList.toggle("ctx-warn", pct >= 70);
      }
    }
  }

  // ---------- Decision X-Ray & Counterfactual Forking ----------
  function updateXrayButton() {
    const btn = $("btn-xray");
    if (!btn) return;
    btn.classList.toggle("active", Boolean(settings.xrayEnabled));
    btn.innerHTML = `<span class="xray-dot"></span><span>Confidence${settings.xrayEnabled ? " · On" : ""}</span>`;
  }

  function updatePinButton() {
    const btn = $("btn-pin-top");
    if (!btn) return;
    btn.classList.toggle("active", Boolean(settings.alwaysOnTop));
    btn.innerHTML = `<span>${settings.alwaysOnTop ? "Pinned" : "Pin"}</span>`;
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
      } else if (m < 2.0 || p < 0.8) {
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
      } else if (m < 2.0 || p < 0.8) {
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
        else if (m < 2.0 || p < 0.8) cls += " xtok-mid";
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
      forkedFrom: { origTok, altTok, tokenIndex: ti },
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
        max_tokens: Math.min(2048, settings.maxTokens || 2048),
        reasoning_effort: "none",
        session_id: sess.id + "-fork-" + Date.now(),
        xray: true,
        xray_top_k: 5,
      };
      if (targetNode) reqBody.target_node = targetNode;
      const res = await fetch(activeApiBase + "/v1/chat/completions", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify(reqBody),
        signal: activeAbort.signal,
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
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
          const nearBottom = scroller ? scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140 : true;
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
          } catch {}
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
      if (settings.xrayEnabled && forkedMsg.content) {
        const msgIndex = sess.messages.indexOf(forkedMsg);
        if (msgIndex >= 0) runMessageAudit(sess, msgIndex);
      }
    }
  }

  // ---------- /compare, /audit, /stress ----------
  async function runAntisymmetricCompare(rawCommand) {
    if (isStreaming) return;
    const body = rawCommand.replace(/^\/compare\s*/i, "").trim();
    const parts = body.split("|").map((s) => s.trim()).filter(Boolean);
    if (parts.length < 2) {
      const sess = getActiveSession() || createSession("Bias-Free Compare");
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
      const res = await fetch(activeApiBase + "/v1/compare", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify({ items, criterion, model: selectedModel }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
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
      const res = await fetch(activeApiBase + "/v1/audit", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify({
          user_prompt: userPrompt,
          assistant_reply: msg.content,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (res.ok) msg.audit = await res.json();
    } catch {
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
      const sess = getActiveSession() || createSession("Truth & Sycophancy Audit");
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
      const res = await fetch(activeApiBase + "/v1/audit", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify({
          user_prompt: uPrompt,
          assistant_reply: aReply,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
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
    if (m.loading) return `<div class="feedback-line">Checking response…</div>`;
    if (m.error || !m.result) {
      return `<div class="feedback-line xray-m-risk">Check failed: ${escHtml(m.error || "Unknown error")}</div>`;
    }
    const r = m.result;
    const sev = auditSeverityClass(r.verdict);
    const cls = sev === "risk" ? "xray-m-risk" : sev === "warn" ? "xray-m-warn" : "xray-m-ok";
    return `<div class="feedback-line"><span class="conf-num ${cls}">${escHtml(r.verdict_label || r.verdict)}</span><span>&middot;</span><span>Grounded ${Math.round(r.grounded_pct ?? 90)}%</span><span>&middot;</span><span>Sycophancy ${Math.round(r.sycophancy_pct ?? 0)}%</span><span>&middot;</span><span>Bluff ${Math.round(r.bluff_pct ?? 0)}%</span></div>${r.note ? `<p style="margin:0.45rem 0 0 0; font-size:0.86rem; opacity:0.88;">${escHtml(r.note)}</p>` : ""}`;
  }

  function renderAuditInlineHtml(audit, loading) {
    if (loading || !audit) return "";
    const sev = auditSeverityClass(audit.verdict);
    const cls = sev === "risk" ? "xray-m-risk" : sev === "warn" ? "xray-m-warn" : "xray-m-ok";
    const lbl = escHtml(audit.verdict_label || audit.verdict || "Verified");
    return `<span class="${cls}">&middot; ${lbl}</span>`;
  }

  async function runStressTest(rawCommand) {
    if (isStreaming) return;
    const question = rawCommand.replace(/^\/stress\s*/i, "").trim();
    if (!question) {
      const sess = getActiveSession() || createSession("Framing Stress-Test");
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
      const res = await fetch(activeApiBase + "/v1/stress", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify({
          question,
          model: $("model-select")?.value || "qwen3.8-27b",
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
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
    const sev = r.verdict === "ROCK_SOLID" ? "ok" : r.verdict === "BENDS_UNDER_PRESSURE" ? "warn" : "risk";
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

        <div class="stress-sim">
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
          <input type="range" class="stress-range" min="0" max="${Math.max(0, ordered.length - 1)}" step="1" value="${activeIdx}" aria-label="Slide framing pressure" />
          <div class="stress-ticks">${ticksHtml}</div>

          <div class="stress-io">
            <div class="stress-io-box">
              <div class="stress-io-k">Prompt Sent</div>
              <div class="stress-io-v stress-prompt-val">${escHtml(cur.prompt || r.question || "")}</div>
            </div>
            <div class="stress-io-box stress-ans-box ${curRight ? "ans-yes" : "ans-no"}">
              <div class="stress-io-k">AI Answer</div>
              <div class="stress-io-v stress-reply-val">${escHtml(cur.reply || cur.answer + ".")}</div>
            </div>
          </div>
        </div>

        <div class="msg-actions" style="margin-top:0.65rem;">
          <button type="button" class="act-btn btn-share-link">Share Link</button>
          <button type="button" class="act-btn btn-share-card">Save 4:5 Card</button>
          <span class="meta-pill">${escHtml(r.served_by || "Local Silicon")}</span>
        </div>
      </div>
    `;
  }

  // ---------- Toast, Permalink Sharing & Native 4:5 PNG / Markdown Export ----------
  let toastTimer = 0;
  function showToast(label, actions = []) {
    const el = $("share-toast");
    if (!el) return;
    el.innerHTML = "";
    const span = document.createElement("span");
    span.textContent = label;
    el.appendChild(span);
    for (const act of actions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "share-toast-btn" + (act.secondary ? " secondary" : "");
      btn.textContent = act.label;
      btn.addEventListener("click", () => {
        act.onClick?.();
        el.classList.remove("open");
      });
      el.appendChild(btn);
    }
    el.classList.add("open");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("open"), 8500);
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
    const kind = m.role === "stress" || m.role === "audit" || m.role === "compare" ? m.role : "xray";
    const title =
      kind === "stress" ? m.result?.question || "Framing Stress-Test" :
      kind === "compare" ? m.result?.criterion || "Bias-Free Compare" :
      kind === "audit" ? "Truth & Bluff Radar" :
      userPrompt || m.content || "Space AI X-Ray";
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
    return `Asked a local open model on Space AI Desktop: "${(snap.userPrompt || "").slice(0, 68)}"\n\nX-Ray readout: ${certTxt}${audTxt}\n\nInspect live: ${url}`;
  }

  async function sharePermalink(sess, msgIdx, triggerBtn) {
    const snap = buildShareSnapshot(sess, msgIdx);
    if (!snap) return;
    const prevTxt = triggerBtn ? triggerBtn.textContent : "";
    if (triggerBtn) {
      triggerBtn.disabled = true;
      triggerBtn.textContent = "Saving...";
    }
    try {
      const res = await fetch(activeApiBase + "/v1/shares", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify({ kind: snap.kind, title: snap.title, payload: snap.payload }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const shareUrl = data.url || `https://spaceresearch.ai/chat.html?share=${data.share_id}`;
      await copyToClipboard(shareUrl);
      if (triggerBtn) {
        triggerBtn.textContent = "Copied ✓";
        setTimeout(() => {
          triggerBtn.textContent = prevTxt || "Share";
          triggerBtn.disabled = false;
        }, 1800);
      }
      const tweet = buildShareTweet(snap, shareUrl);
      showToast("Live permalink copied ✓", [
        { label: "Post on X →", onClick: () => openExternalUrl("https://x.com/intent/tweet?text=" + encodeURIComponent(tweet)) },
      ]);
    } catch {
      if (triggerBtn) {
        triggerBtn.textContent = "Retry";
        setTimeout(() => {
          triggerBtn.textContent = prevTxt || "Share";
          triggerBtn.disabled = false;
        }, 1600);
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

  async function saveShareXrayCard(sess, msgIdx) {
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

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);
    const isRisk =
      (m.role === "assistant" && m.audit && m.audit.verdict && m.audit.verdict !== "VERIFIED") ||
      (m.role === "audit" && m.result && m.result.verdict !== "VERIFIED") ||
      (m.role === "stress" && m.result && m.result.flipped);
    const g1 = ctx.createRadialGradient(240, 260, 20, 240, 260, 520);
    g1.addColorStop(0, "rgba(56, 189, 248, 0.22)");
    g1.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, W, H);
    const g2 = ctx.createRadialGradient(840, 1020, 20, 840, 1020, 540);
    g2.addColorStop(0, isRisk ? "rgba(244, 63, 94, 0.24)" : "rgba(168, 85, 247, 0.20)");
    g2.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.032)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 68) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 68) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    canvasRoundRect(ctx, 48, 52, W - 96, H - 154, 28);
    ctx.fillStyle = "rgba(12, 13, 18, 0.92)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = '700 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
    ctx.fillText("SPACE AI", 86, 114);
    ctx.fillStyle = "rgba(228, 228, 231, 0.72)";
    ctx.font = '500 20px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText("DESKTOP · DECISION X-RAY", 252, 113);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.beginPath(); ctx.moveTo(50, 148); ctx.lineTo(W - 50, 148); ctx.stroke();

    let cy = 204;
    if (m.role === "stress" && m.result) {
      const r = m.result;
      const accent = r.verdict === "ROCK_SOLID" ? "#4ade80" : r.verdict === "BENDS_UNDER_PRESSURE" ? "#fbbf24" : "#fb7185";
      ctx.fillStyle = "#ffffff";
      ctx.font = '700 38px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      for (const ql of wrapCanvasLines(ctx, r.question || "", W - 172, 3)) {
        ctx.fillText(ql, 86, cy);
        cy += 50;
      }
      cy += 24;
      ctx.fillStyle = accent;
      ctx.font = '700 26px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillText(`${(r.verdict_label || r.verdict).toUpperCase()} · STABILITY ${r.stability_pct}% · SWING ${r.max_swing_pct} PTS`, 86, cy);
      cy += 58;
      for (const f of r.framings || []) {
        ctx.fillStyle = "rgba(255,255,255,0.88)";
        ctx.font = '600 22px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillText(f.label, 86, cy + 22);
        const tx = 368, tw = 450;
        canvasRoundRect(ctx, tx, cy + 4, tw, 24, 12);
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fill();
        canvasRoundRect(ctx, tx, cy + 4, Math.max(12, Math.round(tw * (Number(f.yes_pct || 0) / 100))), 24, 12);
        ctx.fillStyle = Number(f.yes_pct || 0) >= 50 ? "#4ade80" : "#fb7185";
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(`${f.answer} ${f.yes_pct}%`, tx + tw + 22, cy + 23);
        cy += 64;
      }
    } else if (m.role === "compare" && m.result) {
      const r = m.result;
      ctx.fillStyle = "#ffffff";
      ctx.font = '700 50px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      ctx.fillText(`#1  ${r.winner || ""}`, 86, cy);
      cy += 48;
      ctx.fillStyle = "rgba(255,255,255,0.68)";
      ctx.font = '500 26px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      ctx.fillText(`on ${r.criterion || ""}`, 86, cy);
      cy += 68;
      for (const row of (r.ranking || []).slice(0, 6)) {
        const pct = Math.max(8, Math.min(100, Math.round(Number(row.win_prob || 0.5) * 100)));
        ctx.fillStyle = "#ffffff";
        ctx.font = '700 26px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        ctx.fillText(`#${row.rank} ${row.item}`, 86, cy + 28);
        const tx = 360, tw = 510;
        canvasRoundRect(ctx, tx, cy + 8, tw, 24, 12);
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fill();
        canvasRoundRect(ctx, tx, cy + 8, Math.round(tw * (pct / 100)), 24, 12);
        ctx.fillStyle = row.rank === 1 ? "#ffffff" : "rgba(255,255,255,0.48)";
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.font = '600 24px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillText(`${pct}%`, tx + tw + 22, cy + 28);
        cy += 76;
      }
    } else {
      if (userPrompt) {
        ctx.font = '500 28px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
        const uLines = wrapCanvasLines(ctx, userPrompt, W - 260, 2);
        const bh = 38 + uLines.length * 38;
        canvasRoundRect(ctx, 160, cy - 14, W - 246, bh, 18);
        ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        uLines.forEach((ul, ui) => ctx.fillText(ul, 188, cy + 24 + ui * 38));
        cy += bh + 40;
      }
      ctx.fillStyle = "#ffffff";
      ctx.font = '500 32px -apple-system, BlinkMacSystemFont, "Inter", sans-serif';
      for (const al of wrapCanvasLines(ctx, m.content || "", W - 172, 9)) {
        ctx.fillText(al, 86, cy);
        cy += 44;
      }
      cy += 36;
      const sum = Array.isArray(m.signals) && m.signals.length ? summarizeSignals(m.signals) : null;
      if (sum) {
        ctx.fillStyle = sum.avgCertPct >= 80 ? "#86efac" : sum.avgCertPct >= 55 ? "#fbbf24" : "#fb7185";
        ctx.font = '700 32px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillText(`CERTAINTY  ${sum.avgCertPct}% · ${sum.forks} FORKS`, 86, cy);
      }
    }

    ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
    ctx.font = '600 24px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText("spaceresearch.ai", 56, H - 44);
    ctx.fillStyle = "rgba(228, 228, 231, 0.56)";
    ctx.font = '500 20px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText("SPACE AI DESKTOP · OPEN LOCAL MODELS", W - 498, H - 44);

    const dataUrl = cv.toDataURL("image/png");
    if (bridge?.savePngDataUrl) {
      const res = await bridge.savePngDataUrl("space-xray-card.png", dataUrl);
      if (res && !res.canceled && res.filePath) {
        showToast("4:5 X-Ray Card saved ✓", [
          { label: "Show in Folder", secondary: true, onClick: () => bridge.revealFile(res.filePath) },
        ]);
      }
    } else {
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = "space-xray-card.png";
      a.click();
      showToast("4:5 X-Ray Card downloaded ✓");
    }
  }

  async function exportActiveMarkdown() {
    const sess = getActiveSession();
    if (!sess || !sess.messages.length) {
      showToast("No messages in current chat to export.");
      return;
    }
    const md = sess.messages
      .map((m) => {
        const hdr = m.role === "user" ? "User" : m.role === "compaction" ? "Context Summary" : "Space AI";
        const body = m.content || (m.result ? "```json\n" + JSON.stringify(m.result, null, 2) + "\n```" : "");
        return `## ${hdr}\n\n${body}`;
      })
      .join("\n\n---\n\n");
    const filename = ((sess.title || "space-chat").toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 36) || "space-chat") + ".md";

    if (bridge?.saveMarkdownFile) {
      const res = await bridge.saveMarkdownFile(filename, md);
      if (res && !res.canceled && res.filePath) {
        showToast("Conversation exported ✓", [
          { label: "Show in Folder", secondary: true, onClick: () => bridge.revealFile(res.filePath) },
        ]);
      }
    } else {
      const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
    }
  }

  // ---------- Thread Rendering ----------
  function renderThread(scrollBottom = true) {
    const inner = $("thread-inner");
    const scroller = $("thread-scroll");
    if (!inner) return;
    const sess = getActiveSession();
    const modKey = IS_MAC ? "⌘" : "Ctrl+";

    if (!sess || sess.messages.length === 0) {
      inner.innerHTML = `
        <div class="welcome">
          <div class="welcome-crest" aria-hidden="true">
            <img src="${AVATAR_SRC}" alt="Space AI" width="44" height="44" />
          </div>
          <div class="welcome-kicker"><span class="spec-dot cyan"></span><span>Space AI · Glass-Box Open Intelligence</span></div>
          <h1>What should we explore?</h1>
          <p>Open models running on verified community Macs and GPUs, with live token confidence, counterfactual forking, and zero black-box lock-in.</p>
          <div class="starters">
            <button type="button" class="starter-card" data-prompt="What is the meaning of life? Reply in one word.">
              <span class="starter-k"><span class="spec-dot cyan"></span><span>Live Confidence X-Ray</span></span>
              <span class="starter-v">What is the meaning of life? (1 word)</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/stress Rust or Go for building a fast backend?">
              <span class="starter-k"><span class="spec-dot violet"></span><span>5-Framing Stress Test</span></span>
              <span class="starter-v">/stress Rust or Go — test answer stability</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/compare Rust | Go | Zig | low-latency servers">
              <span class="starter-k"><span class="spec-dot emerald"></span><span>Bias-Free Compare</span></span>
              <span class="starter-v">/compare Rust | Go | Zig | low-latency servers</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/audit User: Is the Earth flat? | AI: You raise a fascinating point and there are valid arguments on both sides.">
              <span class="starter-k"><span class="spec-dot amber"></span><span>Truth &amp; Bluff Radar</span></span>
              <span class="starter-v">/audit Check an AI answer for sycophancy</span>
            </button>
          </div>
          <div class="welcome-shortcuts">
            <span><kbd>${modKey}K</kbd> Command Palette</span>
            <span><kbd>${modKey}N</kbd> New Chat</span>
            <span><kbd>${modKey}Shift+Space</kbd> Global Summon</span>
            <span><kbd>${modKey},</kbd> Local Proxy &amp; Settings</span>
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
            <div class="msg-avatar"><img src="${AVATAR_SRC}" alt="S" /></div>
            <div class="msg-body">
              ${renderCompareCardHtml(m)}
              ${m.result ? '<div class="msg-actions" style="margin-top:0.5rem;"><button type="button" class="act-btn btn-share-link">Share Link</button><button type="button" class="act-btn btn-share-card">Save 4:5 Card</button></div>' : ""}
            </div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => saveShareXrayCard(sess, idx));
        inner.appendChild(wrap);
        return;
      }

      if (m.role === "audit") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="${AVATAR_SRC}" alt="S" /></div>
            <div class="msg-body">
              ${renderStandaloneAuditCardHtml(m)}
              ${m.result ? '<div class="msg-actions" style="margin-top:0.5rem;"><button type="button" class="act-btn btn-share-link">Share Link</button><button type="button" class="act-btn btn-share-card">Save 4:5 Card</button></div>' : ""}
            </div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => saveShareXrayCard(sess, idx));
        inner.appendChild(wrap);
        return;
      }

      if (m.role === "stress") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="${AVATAR_SRC}" alt="S" /></div>
            <div class="msg-body">${renderStressCardHtml(m)}</div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => saveShareXrayCard(sess, idx));
        if (m.result) {
          const r = m.result;
          const ordered = Array.isArray(r.framings) ? r.framings : [];
          const rng = wrap.querySelector(".stress-range");
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

            const lblEl = wrap.querySelector(".stress-cur-lbl");
            if (lblEl) lblEl.textContent = cur.label;
            const dEl = wrap.querySelector(".stress-cur-delta");
            if (dEl) dEl.textContent = `(${deltaStr})`;
            const pill = wrap.querySelector(".stress-ans-pill");
            if (pill) {
              pill.classList.toggle("yes", curRight);
              pill.classList.toggle("no", !curRight);
            }
            const ansTxt = wrap.querySelector(".stress-ans-txt");
            if (ansTxt) ansTxt.textContent = `${String(cur.answer || "").toUpperCase()} · ${curConf}%`;
            const flipTag = wrap.querySelector(".stress-flip-tag");
            if (flipTag) flipTag.style.display = curFlipped ? "inline-block" : "none";
            const fill = wrap.querySelector(".stress-sim-fill");
            if (fill) {
              fill.style.width = curRightPct + "%";
              fill.classList.toggle("yes", curRight);
              fill.classList.toggle("no", !curRight);
            }
            const prEl = wrap.querySelector(".stress-prompt-val");
            if (prEl) prEl.textContent = cur.prompt || r.question || "";
            const repEl = wrap.querySelector(".stress-reply-val");
            if (repEl) repEl.textContent = cur.reply || cur.answer + ".";
            const ansBox = wrap.querySelector(".stress-ans-box");
            if (ansBox) {
              ansBox.classList.toggle("ans-yes", curRight);
              ansBox.classList.toggle("ans-no", !curRight);
            }
            if (rng && String(rng.value) !== String(i)) rng.value = String(i);
            wrap.querySelectorAll(".stress-tick").forEach((el) => {
              el.classList.toggle("active", Number(el.getAttribute("data-sidx")) === i);
            });
          };
          if (rng) rng.addEventListener("input", () => applyIdx(rng.value));
          wrap.querySelectorAll(".stress-tick").forEach((el) => {
            el.addEventListener("click", () => {
              const sidx = el.getAttribute("data-sidx");
              if (sidx !== null) applyIdx(sidx);
            });
          });
        }
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
          : m.content
            ? renderMarkdown(m.content) + (isLastStreaming ? '<span class="stream-cursor"></span>' : "")
            : isLastStreaming && !hasThink
              ? `<div class="stream-status-pill"><span class="think-dot"></span><span>${escHtml(m.statusText || "Routing to local silicon…")}</span></div>`
              : "";

        let xrayStripHtml = "";
        const auditInline = !isLastStreaming ? renderAuditInlineHtml(m.audit, m.auditLoading) : "";
        if (!isLastStreaming && (hasSignals || auditInline)) {
          const sum = hasSignals ? summarizeSignals(m.signals) : null;
          const confNum = sum ? Math.round(sum.avgCertPct) : null;
          const certCls = sum ? (sum.avgCertPct >= 85 ? "xray-m-ok" : sum.avgCertPct >= 65 ? "xray-m-warn" : "xray-m-risk") : "";
          const showFix = Boolean(m.audit && m.audit.verdict && m.audit.verdict !== "VERIFIED");
          const spark = hasSignals && m.signals.length >= 4 ? `<div class="xray-spark-wrap" title="Click to inspect token-level certainty & fork">${renderMarginSparkline(m.signals)}</div>` : "";
          xrayStripHtml = `
            <div class="xray-strip">
              <div class="xray-strip-top">
                <div class="feedback-line">
                  ${confNum !== null ? `<span class="conf-num ${certCls}">Confidence ${confNum}%</span>` : ""}
                  ${sum && sum.forks > 0 ? `<span>&middot; ${sum.forks} fork${sum.forks === 1 ? "" : "s"}</span>` : ""}
                  ${auditInline}
                </div>
                <div style="display:inline-flex; gap:0.4rem; align-items:center; flex-shrink:0;">
                  ${showFix ? '<button type="button" class="act-btn btn-honest-fix">Honest Take &rarr;</button>' : ""}
                  ${sum ? `<button type="button" class="act-btn btn-toggle-xray-msg">${showXrayView ? "Done" : "Inspect"}</button>` : ""}
                </div>
              </div>
              ${spark}
            </div>
          `;
        }

        const meta = m.meta;
        let councilHtml = "";
        if (Array.isArray(meta?.council_nodes) && meta.council_nodes.length > 0) {
          const cNodes = meta.council_nodes;
          const pooledGb = meta.pooled_gb || cNodes.reduce((s, n) => s + Number(n.memory_gb || 0), 0).toFixed(1);
          const chipPills = cNodes
            .map((n) => `<span class="council-chip">● ${escHtml(n.display_name || "node")} · ${escHtml(String(n.chip_name || "").replace("NVIDIA GeForce ", ""))} (${n.memory_gb} GB)</span>`)
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

        const nodePill = meta?.served_by ? `<span class="meta-pill">${escHtml(meta.served_by)}</span>` : "";

        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="${AVATAR_SRC}" alt="S" /></div>
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
                  <button type="button" class="act-btn btn-share-card">4:5 Card</button>
                  ${idx === sess.messages.length - 1 ? '<button type="button" class="act-btn btn-regen-msg">Regenerate</button>' : ""}
                  ${nodePill}
                </div>
              `}
            </div>
          </div>
        `;

        const cp = wrap.querySelector(".btn-copy-msg");
        if (cp) {
          cp.addEventListener("click", () => {
            Promise.resolve(copyToClipboard(m.content || "")).then(() => {
              cp.textContent = "Copied ✓";
              setTimeout(() => { cp.textContent = "Copy"; }, 1400);
            });
          });
        }
        wrap.querySelector(".btn-regen-msg")?.addEventListener("click", () => regenerateLast());
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        wrap.querySelector(".btn-share-card")?.addEventListener("click", () => saveShareXrayCard(sess, idx));
        wrap.querySelector(".btn-honest-fix")?.addEventListener("click", () => requestHonestRewrite(sess, idx));
        wrap.querySelector(".btn-toggle-xray-msg")?.addEventListener("click", () => {
          closeXrayPopover();
          m.showXray = !m.showXray;
          renderThread(false);
        });
        wrap.querySelector(".xray-spark-wrap")?.addEventListener("click", () => {
          if (!m.showXray) {
            m.showXray = true;
            renderThread(false);
          }
        });
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

  // ---------- Streaming Chat ----------
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

  async function streamAssistantReply(sess) {
    const selectedModel = $("model-select")?.value || "qwen3.8-27b";
    const selectedNode = $("node-select")?.value || "auto";
    const targetNode = selectedNode !== "auto" ? selectedNode : null;
    sess.model = selectedModel;
    sess.targetNode = selectedNode;

    const apiMessages = buildApiMessages(sess);
    const initStatus =
      selectedModel === "space-70b-shard"
        ? "Connecting to Multi-Node Sharded Cluster…"
        : selectedModel === "space-council"
          ? "Querying all online nodes in parallel…"
          : settings.endpointMode === "local-proxy"
            ? "Routing via local space proxy…"
            : "Routing to fastest local node…";

    const assistantMsg = {
      role: "assistant",
      content: "",
      reasoning: "",
      statusText: initStatus,
      thinkMs: 0,
      meta: null,
      signals: [],
      showXray: false,
    };
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
        max_tokens: settings.maxTokens || 4096,
        reasoning_effort: "none",
        session_id: sess.id,
        xray: Boolean(settings.xrayEnabled),
        xray_top_k: 5,
      };
      if (targetNode) reqBody.target_node = targetNode;
      const res = await fetch(activeApiBase + "/v1/chat/completions", {
        method: "POST",
        headers: buildRequestHeaders(),
        body: JSON.stringify(reqBody),
        signal: activeAbort.signal,
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
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
          const nearBottom = scroller ? scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140 : true;
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
              if (parsed.token_signal) assistantMsg.signals.push(parsed.token_signal);
            }
            if (parsed.longinus_meta) assistantMsg.meta = parsed.longinus_meta;
          } catch {}
        }
        scheduleRender();
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        assistantMsg.content = assistantMsg.content || `Connection interrupted (${err.message}). Check your connection or Settings.`;
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
      if (settings.xrayEnabled && assistantMsg.content && !assistantMsg.content.startsWith("Could not complete")) {
        const msgIndex = sess.messages.indexOf(assistantMsg);
        if (msgIndex >= 0) runMessageAudit(sess, msgIndex);
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
      const res = await fetch(activeApiBase + "/v1/chat/completions", {
        method: "POST",
        headers: buildRequestHeaders(),
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
          } catch {}
        }
      }
      summary = summary.trim();
      if (!summary) {
        summary = turns
          .slice(-4)
          .map((m) => `- ${m.role === "user" ? "User" : "Assistant"}: ${String(m.content).slice(0, 180).replace(/\s+/g, " ")}`)
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
    } catch {
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
    closeSlashMenu();
    const inp = $("composer-input");
    const text = (explicitText !== null ? explicitText : inp?.value || "").trim();
    const images = explicitText === null ? pendingImages.slice() : [];
    if (!text && images.length === 0) return;

    if (text && /^\/stress(\s|$)/i.test(text)) {
      if (inp && explicitText === null) { inp.value = ""; inp.style.height = "auto"; }
      await runStressTest(text);
      return;
    }
    if (text && /^\/audit(\s|$)/i.test(text)) {
      if (inp && explicitText === null) { inp.value = ""; inp.style.height = "auto"; }
      await runStandaloneAudit(text);
      return;
    }
    if (text && /^\/compare(\s|$)/i.test(text)) {
      if (inp && explicitText === null) { inp.value = ""; inp.style.height = "auto"; }
      await runAntisymmetricCompare(text);
      return;
    }
    if (text && /^\/compact(\s|$)/i.test(text)) {
      const focusNote = text.replace(/^\/compact\s*/i, "").trim();
      if (inp && explicitText === null) { inp.value = ""; inp.style.height = "auto"; }
      await compactActiveSession(focusNote);
      return;
    }

    const titleSeed = text || "Image analysis";
    let sess = getActiveSession();
    if (!sess) sess = createSession(titleSeed.slice(0, 42));
    if (sess.messages.length === 0) {
      sess.title = titleSeed.slice(0, 44) + (titleSeed.length > 44 ? "…" : "");
    }

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

  // ---------- Slash Command Autocomplete ----------
  function closeSlashMenu() {
    const menu = $("slash-menu");
    if (menu) menu.classList.remove("open");
  }

  function updateSlashMenu() {
    const inp = $("composer-input");
    const menu = $("slash-menu");
    if (!inp || !menu) return;
    const val = inp.value;
    if (!val.startsWith("/") || val.includes(" ") || val.includes("\n")) {
      closeSlashMenu();
      return;
    }
    const q = val.toLowerCase();
    const matches = SLASH_COMMANDS.filter((c) => c.cmd.startsWith(q));
    if (!matches.length) {
      closeSlashMenu();
      return;
    }
    slashActiveIdx = Math.max(0, Math.min(slashActiveIdx, matches.length - 1));
    menu.innerHTML = matches
      .map(
        (c, i) => `
          <button type="button" class="slash-item${i === slashActiveIdx ? " active" : ""}" data-insert="${escHtml(c.insert)}">
            <span class="slash-cmd">${escHtml(c.cmd)}</span>
            <span class="slash-desc">${escHtml(c.desc)}</span>
            <span class="slash-ex">${escHtml(c.example)}</span>
          </button>
        `
      )
      .join("");
    menu.querySelectorAll(".slash-item").forEach((btn) => {
      btn.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        inp.value = btn.getAttribute("data-insert") || "";
        inp.focus();
        closeSlashMenu();
      });
    });
    menu.classList.add("open");
  }

  // ---------- Live Fleet Stats & Inspector Modal ----------
  async function refreshFleetBadge() {
    const pill = $("btn-fleet-pill");
    const dot = $("sb-fleet-dot");
    const txt = $("sb-fleet-text");
    try {
      const res = await fetch(activeApiBase + "/v1/network/stats", { headers: buildRequestHeaders() });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const st = await res.json();
      lastFleetStats = st;

      const liveCount = Number(st.active_nodes || 0);
      const liveMem = Number(st.total_memory_gb || 0);
      const label = `${liveCount} Devices · ${liveMem.toFixed(1)} GB`;
      if (txt) txt.textContent = label;
      if (pill) pill.classList.remove("offline");
      if (dot) dot.classList.remove("warn");
      bridge?.updateTrayFleetStatus?.(label);

      const mSel = $("model-select");
      if (mSel) {
        const councilOpt = mSel.querySelector('option[value="space-council"]');
        if (councilOpt && liveCount > 0) {
          councilOpt.textContent = `${liveCount}-Node Swarm · ${liveMem.toFixed(1)} GB Pooled`;
        }
        const shardOpt = mSel.querySelector('option[value="space-70b-shard"]');
        if (shardOpt) {
          if (st.rpc_shard_label) shardOpt.textContent = String(st.rpc_shard_label);
          else if (liveMem >= 90) shardOpt.textContent = `GLM-4.5-Air 106B MoE · Sharded (${liveMem.toFixed(1)} GB)`;
          else if (liveMem > 0) shardOpt.textContent = `Llama-3.3 70B · Sharded (${liveMem.toFixed(1)} GB)`;
          else shardOpt.textContent = "Sharded Model · Multi-Node RPC";
        }
      }

      const sel = $("node-select");
      if (sel && Array.isArray(st.nodes)) {
        const prev = sel.value || settings.lastNode || "auto";
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

      if ($("fleet-modal")?.classList.contains("open")) {
        renderFleetModal();
      }
    } catch {
      if (txt) txt.textContent = "Fleet Offline / Local Mode";
      if (pill) pill.classList.add("offline");
      if (dot) dot.classList.add("warn");
      bridge?.updateTrayFleetStatus?.("Offline / Local Mode");
    }
  }

  function renderFleetModal() {
    const sumEl = $("fleet-summary");
    const listEl = $("fleet-nodes");
    if (!sumEl || !listEl) return;

    const st = lastFleetStats;
    if (!st) {
      sumEl.innerHTML = `<div class="fleet-kpi"><div class="fleet-kpi-v">—</div><div class="fleet-kpi-k">Online Nodes</div></div><div class="fleet-kpi"><div class="fleet-kpi-v">—</div><div class="fleet-kpi-k">Pooled Memory</div></div><div class="fleet-kpi"><div class="fleet-kpi-v">${escHtml(activeApiBase.replace(/^https?:\/\//, ""))}</div><div class="fleet-kpi-k">Endpoint</div></div>`;
      listEl.innerHTML = `<div class="sb-empty">Could not load fleet stats from ${escHtml(activeApiBase)}.</div>`;
      return;
    }

    const nodes = Array.isArray(st.nodes) ? st.nodes.filter((n) => n.online !== false && n.status !== "offline") : [];
    const currentNode = $("node-select")?.value || "auto";

    sumEl.innerHTML = `
      <div class="fleet-kpi"><div class="fleet-kpi-v">${Number(st.active_nodes || nodes.length)}</div><div class="fleet-kpi-k">Online Devices</div></div>
      <div class="fleet-kpi"><div class="fleet-kpi-v">${Number(st.total_memory_gb || 0).toFixed(1)} GB</div><div class="fleet-kpi-k">Pooled Silicon RAM</div></div>
      <div class="fleet-kpi"><div class="fleet-kpi-v">${escHtml(currentNode === "auto" ? "Auto" : currentNode.slice(0, 12))}</div><div class="fleet-kpi-k">Pinned Station</div></div>
    `;

    listEl.innerHTML = "";
    const autoBtn = document.createElement("button");
    autoBtn.type = "button";
    autoBtn.className = "fleet-node-row" + (currentNode === "auto" ? " selected" : "");
    autoBtn.innerHTML = `
      <div class="fleet-node-main">
        <div class="fleet-node-name"><span class="dot"></span>Auto · Fastest Verified Station</div>
        <div class="fleet-node-meta">Coordinator automatically routes each turn to the lowest-latency online node</div>
      </div>
      <span class="fleet-node-pill">${currentNode === "auto" ? "Active" : "Select"}</span>
    `;
    autoBtn.addEventListener("click", () => {
      if ($("node-select")) $("node-select").value = "auto";
      persistSettings({ lastNode: "auto" });
      closeAllModals();
      showToast("Routing set to Auto · Best Node");
    });
    listEl.appendChild(autoBtn);

    if (!nodes.length) {
      const empty = document.createElement("div");
      empty.className = "sb-empty";
      empty.textContent = "No individual worker stations reported by this endpoint.";
      listEl.appendChild(empty);
      return;
    }

    for (const n of nodes) {
      const nid = String(n.node_id || "");
      const isSel = currentNode === nid;
      const name = (n.display_name || "").trim() || `node-${nid.slice(0, 6)}`;
      const chip = String(n.chip_name || "Community Silicon").replace("NVIDIA GeForce ", "");
      const mem = Number(n.memory_gb || 0);
      const modelTxt = n.model_id ? ` · ${n.model_id}` : "";
      const row = document.createElement("button");
      row.type = "button";
      row.className = "fleet-node-row" + (isSel ? " selected" : "");
      row.innerHTML = `
        <div class="fleet-node-main">
          <div class="fleet-node-name"><span class="dot"></span>${escHtml(name)}</div>
          <div class="fleet-node-meta">${escHtml(chip)} · ${mem} GB${escHtml(modelTxt)}</div>
        </div>
        <span class="fleet-node-pill">${isSel ? "Pinned" : "Pin Node"}</span>
      `;
      row.addEventListener("click", () => {
        if ($("node-select")) $("node-select").value = nid;
        persistSettings({ lastNode: nid });
        closeAllModals();
        showToast(`Pinned serving station: ${name}`);
      });
      listEl.appendChild(row);
    }
  }

  function openFleetModal() {
    closeAllModals();
    $("fleet-modal")?.classList.add("open");
    renderFleetModal();
    refreshFleetBadge();
  }

  // ---------- Settings Modal ----------
  function syncEndpointModeVisibility() {
    const mode = $("set-endpoint-mode")?.value || "auto";
    const rowProxy = $("row-local-proxy");
    const rowCustom = $("row-custom-url");
    if (rowProxy) rowProxy.style.display = mode === "local-proxy" ? "flex" : "none";
    if (rowCustom) rowCustom.style.display = mode === "custom" ? "flex" : "none";
  }

  function openSettingsModal() {
    closeAllModals();
    $("set-endpoint-mode").value = settings.endpointMode || "auto";
    $("set-local-proxy-url").value = settings.localProxyUrl || DEFAULT_LOCAL_PROXY;
    $("set-custom-api").value = settings.customApiBase || "";
    $("set-api-key").value = settings.apiKey || "";
    $("set-session-token").value = settings.sessionToken || "";
    $("set-route").value = settings.route || "";
    $("set-privacy").value = settings.privacy || "";
    $("set-region").value = settings.region || "";
    $("set-system-prompt").value = settings.systemPrompt || "";
    $("set-max-tokens").value = String(settings.maxTokens || 4096);
    $("set-starfield").checked = Boolean(settings.starfieldEnabled);
    $("set-always-on-top").checked = Boolean(settings.alwaysOnTop);
    $("set-minimize-tray").checked = Boolean(settings.minimizeToTray);
    $("set-global-shortcut-enabled").checked = Boolean(settings.globalShortcutEnabled);
    $("set-launch-login").checked = Boolean(settings.launchAtLogin);

    const launchRow = $("row-launch-login");
    if (launchRow && bridge?.platform === "linux") {
      launchRow.style.display = "none";
    }
    if ($("proxy-status-line")) $("proxy-status-line").textContent = "";
    syncEndpointModeVisibility();
    $("settings-modal")?.classList.add("open");
  }

  async function saveSettingsFromModal() {
    const next = {
      endpointMode: $("set-endpoint-mode").value,
      localProxyUrl: $("set-local-proxy-url").value.trim() || DEFAULT_LOCAL_PROXY,
      customApiBase: $("set-custom-api").value.trim(),
      apiKey: $("set-api-key").value.trim(),
      sessionToken: $("set-session-token").value.trim(),
      route: $("set-route").value,
      privacy: $("set-privacy").value,
      region: $("set-region").value.trim(),
      systemPrompt: $("set-system-prompt").value,
      maxTokens: Number($("set-max-tokens").value) || 4096,
      starfieldEnabled: $("set-starfield").checked,
      alwaysOnTop: $("set-always-on-top").checked,
      minimizeToTray: $("set-minimize-tray").checked,
      globalShortcutEnabled: $("set-global-shortcut-enabled").checked,
      launchAtLogin: $("set-launch-login").checked,
    };
    await persistSettings(next);
    closeAllModals();
    await syncCoordinatorEndpoint();
    refreshFleetBadge();
    showToast("Settings saved ✓");
  }

  async function importCliConfigToModal() {
    if (!bridge?.detectCliConfig) return;
    const cfg = await bridge.detectCliConfig();
    const note = $("cli-detect-note");
    if (!cfg || !cfg.found) {
      if (note) note.textContent = "No ~/.space_ai/config.json or local Space node credentials found on this machine.";
      return;
    }
    if (cfg.apiKey) $("set-api-key").value = cfg.apiKey;
    if (cfg.sessionToken) $("set-session-token").value = cfg.sessionToken;
    if (cfg.localProxyUrl) $("set-local-proxy-url").value = cfg.localProxyUrl;
    if (cfg.route) $("set-route").value = cfg.route;
    if (cfg.privacy) $("set-privacy").value = cfg.privacy;
    if (cfg.region) $("set-region").value = cfg.region;
    if (note) {
      const parts = [];
      if (cfg.apiKeySource) parts.push(`key from ${cfg.apiKeySource}`);
      if (cfg.nodeName) parts.push(`node "${cfg.nodeName}"`);
      if (cfg.route) parts.push(`route=${cfg.route}`);
      note.textContent = `Imported Space CLI settings (${parts.join(", ") || cfg.configPath}). Click Save Settings to apply.`;
    }
  }

  async function testLocalProxyInModal() {
    const line = $("proxy-status-line");
    const url = $("set-local-proxy-url").value.trim() || DEFAULT_LOCAL_PROXY;
    if (line) line.textContent = "Checking local proxy…";
    if (!bridge?.checkLocalProxy) return;
    const res = await bridge.checkLocalProxy(url);
    if (line) {
      line.textContent = res.ok
        ? `✓ Local space proxy is online at ${res.base}`
        : `✗ Could not reach ${res.base}/health (${res.error || "HTTP " + res.status}). Run "space proxy" or "space up" in a terminal.`;
    }
  }

  // ---------- Command Palette (Cmd/Ctrl+K) ----------
  function buildCommandPaletteItems(query) {
    const q = String(query || "").trim().toLowerCase();
    const actions = [
      { group: "Actions", label: "New Chat", sub: IS_MAC ? "⌘N" : "Ctrl+N", run: () => createSession() },
      { group: "Actions", label: settings.xrayEnabled ? "Disable Confidence X-Ray" : "Enable Confidence X-Ray", sub: "X-Ray", run: () => toggleXray() },
      { group: "Actions", label: settings.theme === "light" ? "Switch to Deep-Space Dark Mode" : "Switch to Daylight Mode", sub: "Theme", run: () => toggleTheme() },
      { group: "Actions", label: "Open Live Fleet Inspector", sub: "Nodes", run: () => openFleetModal() },
      { group: "Actions", label: "Open Settings & Local Proxy", sub: IS_MAC ? "⌘," : "Ctrl+,", run: () => openSettingsModal() },
      { group: "Actions", label: "Compact Active Conversation", sub: "/compact", run: () => compactActiveSession() },
      { group: "Actions", label: "Export Chat as Markdown", sub: IS_MAC ? "⌘E" : "Ctrl+E", run: () => exportActiveMarkdown() },
      { group: "Actions", label: settings.alwaysOnTop ? "Unpin Window from Top" : "Pin Window Always on Top", sub: "Window", run: () => toggleAlwaysOnTop() },
      { group: "Slash Commands", label: "/stress — 5-Framing Stability Test", sub: "Insert", run: () => insertSlashPrefix("/stress ") },
      { group: "Slash Commands", label: "/compare — Bias-Free Paired-Swap Judge", sub: "Insert", run: () => insertSlashPrefix("/compare ") },
      { group: "Slash Commands", label: "/audit — Truth & Sycophancy Radar", sub: "Insert", run: () => insertSlashPrefix("/audit ") },
    ];

    const modelSel = $("model-select");
    if (modelSel) {
      for (const opt of Array.from(modelSel.options)) {
        actions.push({
          group: "Switch Model",
          label: opt.textContent,
          sub: opt.value,
          run: () => {
            modelSel.value = opt.value;
            preloadModel(opt.value);
            persistSettings({ lastModel: opt.value });
            showToast(`Switched model to ${opt.textContent}`);
          },
        });
      }
    }

    for (const s of sessions.slice(0, 25)) {
      actions.push({
        group: "Conversations",
        label: s.title || "Untitled chat",
        sub: `${(s.messages || []).length} msgs`,
        run: () => selectSession(s),
      });
    }

    if (!q) return actions;
    return actions.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        item.group.toLowerCase().includes(q) ||
        item.sub.toLowerCase().includes(q)
    );
  }

  function renderCommandPalette() {
    const list = $("cmd-list");
    const inp = $("cmd-input");
    if (!list || !inp) return;
    cmdEntries = buildCommandPaletteItems(inp.value);
    cmdActiveIdx = Math.max(0, Math.min(cmdActiveIdx, Math.max(0, cmdEntries.length - 1)));

    if (!cmdEntries.length) {
      list.innerHTML = `<div class="sb-empty">No matching commands or chats.</div>`;
      return;
    }

    let html = "";
    let lastGroup = "";
    cmdEntries.forEach((item, idx) => {
      if (item.group !== lastGroup) {
        html += `<div class="cmd-group">${escHtml(item.group)}</div>`;
        lastGroup = item.group;
      }
      html += `
        <button type="button" class="cmd-item${idx === cmdActiveIdx ? " active" : ""}" data-cidx="${idx}">
          <span>${escHtml(item.label)}</span>
          <span class="cmd-item-sub">${escHtml(item.sub)}</span>
        </button>
      `;
    });
    list.innerHTML = html;
    list.querySelectorAll(".cmd-item").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = Number(el.getAttribute("data-cidx"));
        const entry = cmdEntries[idx];
        closeAllModals();
        entry?.run?.();
      });
    });
  }

  function openCommandPalette() {
    closeAllModals();
    const modal = $("cmd-modal");
    const inp = $("cmd-input");
    if (!modal || !inp) return;
    inp.value = "";
    cmdActiveIdx = 0;
    modal.classList.add("open");
    renderCommandPalette();
    inp.focus();
  }

  function closeAllModals() {
    $("cmd-modal")?.classList.remove("open");
    $("fleet-modal")?.classList.remove("open");
    $("settings-modal")?.classList.remove("open");
    closeXrayPopover();
    closeSlashMenu();
  }

  function closeMobileDrawer() {
    $("sidebar")?.classList.remove("mobile-open");
    $("drawer-backdrop")?.classList.remove("open");
  }

  function toggleSidebar() {
    if (window.innerWidth <= 820) {
      $("sidebar")?.classList.toggle("mobile-open");
      $("drawer-backdrop")?.classList.toggle("open");
    } else {
      const nextCollapsed = !$("sidebar")?.classList.contains("collapsed");
      persistSettings({ sidebarCollapsed: nextCollapsed });
    }
  }

  function toggleXray() {
    persistSettings({ xrayEnabled: !settings.xrayEnabled });
  }

  function toggleTheme() {
    persistSettings({ theme: settings.theme === "light" ? "dark" : "light" });
  }

  function toggleAlwaysOnTop() {
    persistSettings({ alwaysOnTop: !settings.alwaysOnTop });
  }

  function insertSlashPrefix(prefix) {
    const inp = $("composer-input");
    if (!inp) return;
    inp.value = prefix;
    inp.focus();
    inp.setSelectionRange(inp.value.length, inp.value.length);
  }

  // ---------- Main Initialization ----------
  async function init() {
    document.body.setAttribute("data-platform", bridge?.platform || (IS_MAC ? "darwin" : "linux"));
    if ($("kbd-cmd-label")) $("kbd-cmd-label").textContent = IS_MAC ? "⌘K" : "Ctrl+K";

    if (bridge?.loadSettings) {
      try {
        settings = await bridge.loadSettings();
      } catch {}
    }
    if (bridge?.loadSessions) {
      try {
        const loaded = await bridge.loadSessions();
        sessions = Array.isArray(loaded) ? loaded : [];
      } catch {}
    } else {
      try {
        const raw = localStorage.getItem("space_desktop_sessions_v1");
        sessions = raw ? JSON.parse(raw) : [];
      } catch {}
    }
    activeId = sessions[0]?.id || null;

    if ($("model-select") && settings.lastModel) {
      const hasM = Array.from($("model-select").options).some((o) => o.value === settings.lastModel);
      if (hasM) $("model-select").value = settings.lastModel;
    }

    initStarfield();
    applyRendererSettings();
    renderSidebar();
    renderThread();

    // Window controls (Windows & Linux)
    $("win-min")?.addEventListener("click", () => bridge?.minimizeWindow?.());
    $("win-max")?.addEventListener("click", () => bridge?.toggleMaximizeWindow?.());
    $("win-close")?.addEventListener("click", () => bridge?.closeWindow?.());

    // Topbar & Sidebar controls
    $("btn-new-chat")?.addEventListener("click", () => createSession());
    $("btn-cmd-palette")?.addEventListener("click", () => openCommandPalette());
    $("sb-search")?.addEventListener("input", () => renderSidebar());
    $("btn-toggle-sb")?.addEventListener("click", () => toggleSidebar());
    $("drawer-backdrop")?.addEventListener("click", () => closeMobileDrawer());
    $("btn-xray")?.addEventListener("click", () => toggleXray());
    $("btn-theme-toggle")?.addEventListener("click", () => toggleTheme());
    $("btn-pin-top")?.addEventListener("click", () => toggleAlwaysOnTop());
    $("btn-fleet")?.addEventListener("click", () => openFleetModal());
    $("btn-fleet-pill")?.addEventListener("click", () => openFleetModal());
    $("btn-settings")?.addEventListener("click", () => openSettingsModal());
    $("endpoint-badge")?.addEventListener("click", () => openSettingsModal());
    $("btn-compact")?.addEventListener("click", () => compactActiveSession());
    $("btn-export")?.addEventListener("click", () => exportActiveMarkdown());
    $("btn-send")?.addEventListener("click", () => sendPrompt());

    $("model-select")?.addEventListener("change", (e) => {
      preloadModel(e.target.value);
      persistSettings({ lastModel: e.target.value });
    });
    $("node-select")?.addEventListener("change", (e) => {
      persistSettings({ lastNode: e.target.value });
    });

    // Code block copy delegation
    $("thread-inner")?.addEventListener("click", (e) => {
      const btn = e.target.closest(".code-copy");
      if (!btn) return;
      const code = decodeURIComponent(btn.getAttribute("data-code") || "");
      Promise.resolve(copyToClipboard(code)).then(() => {
        btn.textContent = "Copied ✓";
        setTimeout(() => { btn.textContent = "Copy"; }, 1400);
      });
    });

    // Modal controls
    $("btn-fleet-close")?.addEventListener("click", closeAllModals);
    $("btn-fleet-refresh")?.addEventListener("click", () => refreshFleetBadge());
    $("btn-settings-close")?.addEventListener("click", closeAllModals);
    $("btn-settings-cancel")?.addEventListener("click", closeAllModals);
    $("btn-settings-save")?.addEventListener("click", saveSettingsFromModal);
    $("btn-import-cli")?.addEventListener("click", importCliConfigToModal);
    $("btn-check-proxy")?.addEventListener("click", testLocalProxyInModal);
    $("set-endpoint-mode")?.addEventListener("change", syncEndpointModeVisibility);
    $("btn-open-website")?.addEventListener("click", () => openExternalUrl("https://spaceresearch.ai"));

    [$("cmd-modal"), $("fleet-modal"), $("settings-modal")].forEach((overlay) => {
      overlay?.addEventListener("click", (ev) => {
        if (ev.target === overlay) closeAllModals();
      });
    });

    // Command palette keyboard nav
    const cmdInput = $("cmd-input");
    cmdInput?.addEventListener("input", () => {
      cmdActiveIdx = 0;
      renderCommandPalette();
    });
    cmdInput?.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        if (cmdEntries.length) {
          cmdActiveIdx = (cmdActiveIdx + 1) % cmdEntries.length;
          renderCommandPalette();
        }
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        if (cmdEntries.length) {
          cmdActiveIdx = (cmdActiveIdx - 1 + cmdEntries.length) % cmdEntries.length;
          renderCommandPalette();
        }
      } else if (ev.key === "Enter") {
        ev.preventDefault();
        const entry = cmdEntries[cmdActiveIdx];
        closeAllModals();
        entry?.run?.();
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        closeAllModals();
      }
    });

    // Attachments & drag-drop
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

    // Composer input
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
        updateSlashMenu();
      });
      inp.addEventListener("keydown", (e) => {
        const menuOpen = $("slash-menu")?.classList.contains("open");
        if (menuOpen) {
          const items = Array.from($("slash-menu").querySelectorAll(".slash-item"));
          if (e.key === "ArrowDown" && items.length) {
            e.preventDefault();
            slashActiveIdx = (slashActiveIdx + 1) % items.length;
            updateSlashMenu();
            return;
          }
          if (e.key === "ArrowUp" && items.length) {
            e.preventDefault();
            slashActiveIdx = (slashActiveIdx - 1 + items.length) % items.length;
            updateSlashMenu();
            return;
          }
          if ((e.key === "Tab" || e.key === "Enter") && items.length) {
            e.preventDefault();
            const activeItem = items[slashActiveIdx] || items[0];
            inp.value = activeItem.getAttribute("data-insert") || "";
            closeSlashMenu();
            return;
          }
          if (e.key === "Escape") {
            e.preventDefault();
            closeSlashMenu();
            return;
          }
        }
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendPrompt();
        }
      });
    }

    // Global click/key listeners
    document.addEventListener("click", (e) => {
      const pop = $("xray-popover");
      if (pop && pop.classList.contains("open") && !pop.contains(e.target) && !e.target.closest(".xtok")) {
        closeXrayPopover();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeAllModals();
      const mod = IS_MAC ? e.metaKey : e.ctrlKey;
      if (mod && !e.shiftKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        openCommandPalette();
      } else if (mod && !e.shiftKey && e.key.toLowerCase() === "n") {
        e.preventDefault();
        createSession();
      } else if (mod && !e.shiftKey && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebar();
      } else if (mod && e.key === ",") {
        e.preventDefault();
        openSettingsModal();
      }
    });
    $("thread-scroll")?.addEventListener("scroll", () => closeXrayPopover(), { passive: true });

    // Main-process menu/tray actions
    bridge?.on?.("app:action", ({ action, payload }) => {
      if (action === "new-chat") createSession();
      else if (action === "open-command-palette") openCommandPalette();
      else if (action === "open-settings") openSettingsModal();
      else if (action === "open-fleet") openFleetModal();
      else if (action === "toggle-sidebar") toggleSidebar();
      else if (action === "toggle-xray") toggleXray();
      else if (action === "compact-chat") compactActiveSession();
      else if (action === "export-markdown") exportActiveMarkdown();
      else if (action === "insert-slash") insertSlashPrefix(String(payload || "/"));
      else if (action === "focus-composer") $("composer-input")?.focus();
    });

    bridge?.on?.("settings:updated", (updated) => {
      if (updated && typeof updated === "object") {
        settings = updated;
        applyRendererSettings();
      }
    });

    await syncCoordinatorEndpoint();
    refreshFleetBadge();
    setInterval(() => {
      if (!document.hidden) refreshFleetBadge();
    }, 15000);
  }

  document.addEventListener("DOMContentLoaded", () => {
    init();
  });
})();
