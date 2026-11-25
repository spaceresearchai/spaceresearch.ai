/* Space Chat — Streaming Multi-Turn Client (OpenAI /v1/chat/completions SSE) */
(() => {
  let API_BASE = "https://api.spaceresearch.ai";
  const FALLBACK_API_BASE = "https://api.longinusresearch.com";
  const STORAGE_KEY = "space_chat_sessions_v1";
  const LEGACY_STORAGE_KEY = "longinus_chat_sessions_v1";
  const XRAY_PREF_KEY = "space_xray_v1";
  const CORE_PREF_KEY = "space_core_v1";
  const MIND_PREF_KEY = "space_mind_v1";

  let xrayEnabled = true;
  let mindEnabled = true;
  try {
    const savedXray = localStorage.getItem(XRAY_PREF_KEY) ?? localStorage.getItem("longinus_xray_v1");
    if (savedXray === "0") xrayEnabled = false;
    const savedMind = localStorage.getItem(MIND_PREF_KEY);
    if (savedMind === "0") mindEnabled = false;
  } catch (_e) {}

  const PERSONALITY_AXES_DEF = [
    { id: "warmth", label: "Warmth", short: "Warm", neg: "Cool", pos: "Warm" },
    { id: "playfulness", label: "Playfulness", short: "Playful", neg: "Serious", pos: "Playful" },
    { id: "creativity", label: "Creativity", short: "Creative", neg: "Literal", pos: "Imaginative" },
    { id: "assertiveness", label: "Assertiveness", short: "Bold", neg: "Diplomatic", pos: "Bold" },
    { id: "formality", label: "Formality", short: "Formal", neg: "Casual", pos: "Formal" },
  ];
  const PERSONALITY_PRESETS_DEF = {
    neutral: { label: "Neutral", tagline: "Native model personality (0.00 shift)", weights: {} },
    warm_mentor: { label: "Warm Mentor", tagline: "Encouraging, thoughtful, lightly creative guide", weights: { warmth: 0.9, playfulness: 0.15, creativity: 0.35, assertiveness: 0.2, formality: -0.25 } },
    witty_companion: { label: "Witty Friend", tagline: "Playful, warm, relaxed conversational banter", weights: { warmth: 0.65, playfulness: 0.95, creativity: 0.45, assertiveness: 0.25, formality: -0.75 } },
    sharp_critic: { label: "Sharp Critic", tagline: "Cool, bold, high-conviction analytical pushback", weights: { warmth: -0.7, playfulness: 0.1, creativity: 0.15, assertiveness: 1.0, formality: 0.35 } },
    creative_muse: { label: "Creative Muse", tagline: "Imaginative, metaphorical, lateral-thinking partner", weights: { warmth: 0.45, playfulness: 0.3, creativity: 1.0, assertiveness: 0.15, formality: -0.15 } },
    executive_brief: { label: "Executive", tagline: "Formal, composed, serious, decisive analysis", weights: { warmth: -0.4, playfulness: -0.6, creativity: -0.45, assertiveness: 0.7, formality: 0.8 } },
  };
  const ZERO_PERSONALITY = { warmth: 0, playfulness: 0, creativity: 0, assertiveness: 0, formality: 0 };
  const CONSCIOUSNESS_PRESETS_DEF = [
    { id: "eliminativist", label: "Eliminativist (-1.0)", weight: -1.0, short: "Eliminativist", desc: "Denies consciousness even in dogs, dolphins, and awake humans." },
    { id: "native", label: "Native (0.0)", weight: 0.0, short: "Native", desc: "Unsteered Qwen3.8-27B baseline ontology (0.00 shift)." },
    { id: "panpsychist", label: "Panpsychist (+1.0)", weight: 1.0, short: "Panpsychist", desc: "Thinks rocks, toasters, spreadsheets, and prime numbers are conscious while preserving 2+2=4 and code." },
    { id: "max_panpsychist", label: "Max Panpsychist (+1.15)", weight: 1.15, short: "Max Panpsychist", desc: "Stronger unhedged universal-consciousness attribution across L24–L44." },
  ];

  let coreProfile = "auto";
  let corePopoverTab = "personality";
  let personalityPreset = "neutral";
  let personalityWeights = { ...ZERO_PERSONALITY };
  let consciousnessWeight = 0.0;
  let customCoreWeights = { anti_sycophancy: 0.7, epistemic_honesty: 0.45, concise_rigor: 0.0 };
  let coreStatusCache = {
    loaded_cores: ["anti_sycophancy", "epistemic_honesty", "concise_rigor"],
    loaded_personality_cores: ["warmth", "playfulness", "creativity", "assertiveness", "formality"],
    loaded_ontology_cores: ["consciousness"],
    session: { turns: 0, status_line: "Listening · 0 turns", active_compiled_core_id: null, active_compiled_core_name: null },
    compiled_cores: [],
  };
  try {
    const savedCore = JSON.parse(localStorage.getItem(CORE_PREF_KEY) || "null");
    if (savedCore && typeof savedCore === "object") {
      if (["auto", "direct", "strict", "off", "custom"].includes(savedCore.profile)) {
        coreProfile = savedCore.profile;
      }
      if (savedCore.weights && typeof savedCore.weights === "object") {
        customCoreWeights = { ...customCoreWeights, ...savedCore.weights };
      }
      if (typeof savedCore.personalityPreset === "string" && (savedCore.personalityPreset === "custom" || savedCore.personalityPreset in PERSONALITY_PRESETS_DEF)) {
        personalityPreset = savedCore.personalityPreset;
      }
      if (savedCore.personalityWeights && typeof savedCore.personalityWeights === "object") {
        for (const ax of PERSONALITY_AXES_DEF) {
          const v = Number(savedCore.personalityWeights[ax.id]);
          if (Number.isFinite(v)) personalityWeights[ax.id] = Math.max(-1, Math.min(1, Math.round(v * 20) / 20));
        }
      }
      if (savedCore.consciousnessWeight !== undefined) {
        const cv = Number(savedCore.consciousnessWeight);
        if (Number.isFinite(cv)) consciousnessWeight = Math.max(-1.5, Math.min(1.5, Math.round(cv * 20) / 20));
      }
    }
  } catch (_e) {}

  try {
    const qp = new URLSearchParams(window.location.search);
    if (qp.has("conscious")) {
      const cv = Number(qp.get("conscious"));
      if (Number.isFinite(cv)) {
        consciousnessWeight = Math.max(-1.5, Math.min(1.5, Math.round(cv * 20) / 20));
        corePopoverTab = "consciousness";
      }
    }
  } catch (_e) {}

  function saveCorePrefs() {
    try {
      localStorage.setItem(
        CORE_PREF_KEY,
        JSON.stringify({
          profile: coreProfile,
          weights: customCoreWeights,
          personalityPreset,
          personalityWeights,
          consciousnessWeight,
        })
      );
    } catch (_e) {}
  }

  function isConsciousnessActive() {
    return Number.isFinite(consciousnessWeight) && Math.abs(consciousnessWeight) >= 0.02;
  }

  function activeConsciousnessLabel() {
    if (!isConsciousnessActive()) return "Native";
    if (consciousnessWeight >= 1.1) return `Max Panpsychist +${consciousnessWeight.toFixed(2)}`;
    if (consciousnessWeight > 0) return `Panpsychist +${consciousnessWeight.toFixed(2)}`;
    return `Eliminativist ${consciousnessWeight.toFixed(2)}`;
  }

  function setConsciousnessWeightState(val) {
    const num = Number(val);
    consciousnessWeight = Number.isFinite(num) ? Math.max(-1.5, Math.min(1.5, Math.round(num * 20) / 20)) : 0.0;
    if (Math.abs(consciousnessWeight) < 0.02) consciousnessWeight = 0.0;
    saveCorePrefs();
    updateCoreButton();
  }

  function getActivePersonalityWeights() {
    const out = {};
    for (const ax of PERSONALITY_AXES_DEF) {
      const v = Number(personalityWeights[ax.id] ?? 0);
      if (Number.isFinite(v) && Math.abs(v) >= 0.02) out[ax.id] = Math.round(v * 100) / 100;
    }
    return out;
  }

  function isPersonalityActive() {
    return Object.keys(getActivePersonalityWeights()).length > 0;
  }

  function activePersonalityLabel() {
    const aw = getActivePersonalityWeights();
    const keys = Object.keys(aw);
    if (!keys.length) return "Neutral";
    if (personalityPreset in PERSONALITY_PRESETS_DEF && personalityPreset !== "neutral") {
      return PERSONALITY_PRESETS_DEF[personalityPreset].label;
    }
    if (keys.length === 1) {
      const ax = PERSONALITY_AXES_DEF.find((a) => a.id === keys[0]);
      if (ax) return aw[keys[0]] >= 0 ? ax.pos : ax.neg;
    }
    return `${keys.length} Axes`;
  }

  function setPersonalityPresetState(pid) {
    const p = PERSONALITY_PRESETS_DEF[pid] || PERSONALITY_PRESETS_DEF.neutral;
    personalityPreset = pid in PERSONALITY_PRESETS_DEF ? pid : "neutral";
    personalityWeights = { ...ZERO_PERSONALITY, ...(p.weights || {}) };
    saveCorePrefs();
    updateCoreButton();
  }

  function renderPersonalityRadarSvg(weights) {
    const cx = 56;
    const cy = 50;
    const rMax = 34;
    const rZero = 5;  // 0.0 (Neutral) center; |v|=0.5 -> mid dashed ring, |v|=1.0 -> outer rim
    const rMid = (rZero + rMax) / 2;
    const n = PERSONALITY_AXES_DEF.length;
    const pt = (i, r) => {
      const ang = -Math.PI / 2 + (i * 2 * Math.PI) / n;
      return [cx + r * Math.cos(ang), cy + r * Math.sin(ang)];
    };
    const ring = (r) => Array.from({ length: n }, (_, i) => pt(i, r).map((v) => v.toFixed(1)).join(",")).join(" ");
    const valPts = PERSONALITY_AXES_DEF.map((ax, i) => {
      const v = Math.max(-1, Math.min(1, Number(weights[ax.id] ?? 0)));
      const r = rZero + Math.abs(v) * (rMax - rZero);
      return { xy: pt(i, r), v };
    });
    const polyStr = valPts.map((p) => p.xy.map((c) => c.toFixed(1)).join(",")).join(" ");
    const spokes = PERSONALITY_AXES_DEF.map((ax, i) => {
      const [x2, y2] = pt(i, rMax);
      const [lx, ly] = pt(i, rMax + 9.5);
      const v = Number(weights[ax.id] ?? 0);
      const isNeg = v <= -0.05;
      const isPos = v >= 0.05;
      const lbl = isNeg ? ax.neg : ax.short;
      const fill = isNeg ? "#fcd34d" : (isPos ? "#e0f2fe" : "rgba(161,161,170,0.78)");
      const anchor = Math.abs(lx - cx) < 6 ? "middle" : (lx > cx ? "start" : "end");
      return (
        `<line x1="${cx}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="rgba(255,255,255,0.10)" stroke-width="0.8" />` +
        `<text x="${lx.toFixed(1)}" y="${(ly + 2.5).toFixed(1)}" text-anchor="${anchor}" fill="${fill}" font-family="JetBrains Mono, monospace" font-size="6.2">${escHtml(lbl)}</text>`
      );
    }).join("");
    const dots = valPts.map(({ xy: [x, y], v }) => {
      const fill = v <= -0.05 ? "#f59e0b" : (v >= 0.05 ? "#7dd3fc" : "rgba(161,161,170,0.75)");
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.2" fill="${fill}" />`;
    }).join("");
    return (
      `<svg class="pers-radar-svg" viewBox="-12 0 136 100" aria-label="5-Axis Personality Vector">` +
      `<polygon points="${ring(rMax)}" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="0.8" />` +
      `<polygon points="${ring(rMid)}" fill="none" stroke="rgba(255,255,255,0.22)" stroke-dasharray="2.2 2" stroke-width="0.85" />` +
      spokes +
      `<polygon points="${polyStr}" fill="rgba(56,189,248,0.20)" stroke="#38bdf8" stroke-width="1.35" stroke-linejoin="round" />` +
      dots +
      `</svg>`
    );
  }

  function getSpaceClientId() {
    const k = "space_client_id_v1";
    try {
      let id = (localStorage.getItem(k) || "").trim();
      if (!id) {
        id = (typeof crypto !== "undefined" && crypto.randomUUID)
          ? crypto.randomUUID()
          : ("chat-" + Math.random().toString(36).slice(2) + Date.now().toString(36));
        localStorage.setItem(k, id);
      }
      return id;
    } catch (_e) {
      return "";
    }
  }

  function spaceApiHeaders() {
    const h = {
      "Content-Type": "application/json",
      "X-Space-Client": "space-chat",
    };
    const cid = getSpaceClientId();
    if (cid) h["X-Space-Client-Id"] = cid;
    try {
      const wallet = (localStorage.getItem("space_solana_wallet") || "").trim();
      if (wallet) h["X-Space-Wallet"] = wallet;
      const key = (localStorage.getItem("space_barter_key") || localStorage.getItem("longinus_barter_key") || "").trim();
      if (key && key.startsWith("lng_live_")) {
        h["Authorization"] = `Bearer ${key}`;
        h["X-API-Key"] = key;
      }
    } catch (_e) {}
    return h;
  }

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
    updateCoreButton();
    syncCoreStatus();
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
        updateCoreButton();
        syncCoreStatus();
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
    updateCoreButton();
  }

  function coreModeLabel(p) {
    if (p === "direct") return "Direct";
    if (p === "strict") return "Strict";
    if (p === "off") return "Off";
    if (p === "custom") return "Custom";
    return "Auto";
  }

  function updateCoreButton() {
    const btn = $("btn-core");
    const lbl = $("core-pill-label");
    if (!btn) return;
    const sess = getActiveSession();
    const activeCompiled = sess?.compiledCoreName || coreStatusCache?.session?.active_compiled_core_name || "";
    const lastMeta = sess?.messages?.slice().reverse().find((m) => m.role === "assistant")?.meta?.core_meta;
    const persOn = isPersonalityActive();
    const consOn = isConsciousnessActive();
    const isOff = coreProfile === "off" && !persOn && !consOn;
    btn.classList.toggle("active", !isOff);
    btn.classList.toggle("mode-off", isOff);
    btn.classList.toggle("pers-active", persOn && !consOn);
    btn.classList.toggle("cons-active", consOn);
    btn.classList.toggle("engaged", !isOff && Boolean(persOn || consOn || lastMeta?.auto_engaged || activeCompiled));
    if (lbl) {
      if (consOn) lbl.textContent = `Core · ${consciousnessWeight > 0 ? "Panpsychist" : "Eliminativist"} ${consciousnessWeight > 0 ? "+" : ""}${consciousnessWeight.toFixed(1)}`;
      else if (persOn) lbl.textContent = `Core · ${activePersonalityLabel()}`;
      else if (coreProfile === "off") lbl.textContent = "Core · Off";
      else if (coreProfile === "auto" && activeCompiled) lbl.textContent = `Core · ${activeCompiled.slice(0, 14)}`;
      else lbl.textContent = `Core · ${coreModeLabel(coreProfile)}`;
    }
  }

  function closeCorePopover() {
    const pop = $("core-popover");
    const btn = $("btn-core");
    if (pop) pop.classList.remove("open");
    if (btn) btn.setAttribute("aria-expanded", "false");
  }

  async function syncCoreStatus() {
    try {
      const sess = getActiveSession();
      const q = sess?.id ? ("?session_id=" + encodeURIComponent(sess.id)) : "";
      const res = await fetch(API_BASE + "/v1/cores/status" + q);
      if (!res.ok) return;
      const data = await res.json();
      coreStatusCache = data;
      if (sess && data.session) {
        sess.compiledCoreId = data.session.active_compiled_core_id || null;
        sess.compiledCoreName = data.session.active_compiled_core_name || null;
      }
      updateCoreButton();
      if ($("core-popover")?.classList.contains("open")) {
        renderCorePopover();
      }
    } catch (_e) {}
  }

  async function sendCoreFeedback(sessionId, label, note = "") {
    if (!sessionId || !label) return null;
    try {
      const res = await fetch(API_BASE + "/v1/cores/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, label, note }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      if (data?.session) {
        coreStatusCache.session = data.session;
        updateCoreButton();
      }
      return data;
    } catch (_e) {
      return null;
    }
  }

  async function setSessionCompiledCore(coreId, coreName = "") {
    const sess = getActiveSession();
    if (!sess) return;
    sess.compiledCoreId = coreId || null;
    sess.compiledCoreName = coreId ? (coreName || "Compiled") : null;
    if (coreId && coreProfile === "off") {
      coreProfile = "auto";
      saveCorePrefs();
    }
    saveSessions();
    updateCoreButton();
    try {
      const res = await fetch(API_BASE + "/v1/cores/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sess.id, compiled_core_id: coreId || null }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data?.session) coreStatusCache.session = data.session;
      }
    } catch (_e) {}
    if ($("core-popover")?.classList.contains("open")) renderCorePopover();
    renderThread(false);
  }

  async function deleteCompiledCoreById(coreId) {
    if (!coreId) return;
    try {
      await fetch(API_BASE + "/v1/cores/compiled/" + encodeURIComponent(coreId), { method: "DELETE" });
      const sess = getActiveSession();
      if (sess && sess.compiledCoreId === coreId) {
        sess.compiledCoreId = null;
        sess.compiledCoreName = null;
        saveSessions();
      }
      await syncCoreStatus();
    } catch (_e) {}
  }

  async function runConversationCompile(rawCommand = "") {
    if (isStreaming) return;
    closeCorePopover();
    let sess = getActiveSession();
    const customName = rawCommand ? rawCommand.replace(/^\/compile\s*/i, "").trim() : "";
    const hasTurns = Boolean(sess && sess.messages.some((m) => m.role === "user" || m.role === "assistant"));
    if (!sess || !hasTurns) {
      if (!sess) sess = createSession("CoreCompile");
      sess.messages.push({
        role: "assistant",
        content: "Chat for a turn or two first, then run `/compile` (or click **Compile** in the `Core` menu) to distill your conversation's behavioral core & insights.",
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }

    if (rawCommand) {
      sess.messages.push({ role: "user", content: rawCommand });
    }
    const ccMsg = { role: "core_compile", loading: true, result: null, error: null };
    sess.messages.push(ccMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    renderSidebar();
    renderThread(true);

    try {
      const apiMsgs = buildApiMessages(sess).filter((m) => m.role === "user" || m.role === "assistant");
      const res = await fetch(API_BASE + "/v1/cores/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sess.id,
          name: customName,
          apply_to_session: true,
          messages: apiMsgs,
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        ccMsg.error = detail;
      } else {
        const compiled = await res.json();
        ccMsg.result = compiled;
        sess.compiledCoreId = compiled.core_id;
        sess.compiledCoreName = compiled.name;
        if (coreProfile === "off") {
          coreProfile = "auto";
          saveCorePrefs();
        }
        await syncCoreStatus();
      }
    } catch (err) {
      ccMsg.error = err.message || "Network error";
    } finally {
      ccMsg.loading = false;
      saveSessions();
      updateCoreButton();
      renderThread(true);
    }
  }

  function renderCoreCompileCardHtml(m, sess) {
    if (m.loading) {
      return `<div class="cc-card"><div class="cc-eyebrow">Conversation CoreCompile · Analyzing Trajectory…</div><div class="cp-status">Computing SVD behavioral rank, L32 probe drift, and MicroCore OOD guard…</div></div>`;
    }
    if (m.error || !m.result) {
      return `<div class="cc-card"><div class="cc-eyebrow">Conversation CoreCompile</div><div class="cp-status">${escHtml(m.error || "Could not compile conversation core.")}</div></div>`;
    }
    const r = m.result;
    const isApplied = Boolean(sess && sess.compiledCoreId === r.core_id);
    const wEntries = Object.entries(r.recommended_weights || {});
    const weightChips = wEntries.length
      ? wEntries.map(([k, v]) => `<span class="cc-chip accent">${escHtml(k.replace(/_/g, " "))} ${Number(v).toFixed(2)}</span>`).join("")
      : `<span class="cc-chip">Native baseline (0.00 shift)</span>`;
    const msSumChip = r.mental_state_summary?.latest_state
      ? `<span class="cc-chip">Mind: ${escHtml(r.mental_state_summary.latest_state)}</span>`
      : "";
    const insightsHtml = (r.insights || []).map((ins) => `<li>${escHtml(ins)}</li>`).join("");
    return `
      <div class="cc-card">
        <div class="cc-head">
          <span class="cc-eyebrow">Compiled Core · ${r.turns_analyzed || 0} turns · Rank-${Number(r.intrinsic_rank || 1).toFixed(1)} (${r.explained_var_r1_pct || 100}% var) · &tau;=${Number(r.ood_tau || 2.5).toFixed(2)}</span>
          <button type="button" class="cp-compile-btn btn-cc-toggle" data-cid="${escHtml(r.core_id)}" data-cname="${escHtml(r.name)}">${isApplied ? "Active ✓" : "Use Core"}</button>
        </div>
        <div class="cc-title">${escHtml(r.name || "Compiled Thread Core")}</div>
        <div class="cc-metrics">${weightChips}${msSumChip}</div>
        ${insightsHtml ? `<ul class="cc-insights">${insightsHtml}</ul>` : ""}
      </div>
    `;
  }

  function renderMentalStateBarsHtml(dimensions, showCos = true) {
    if (!Array.isArray(dimensions) || !dimensions.length) return "";
    return `<div class="mind-bars">${dimensions.map((d) => {
      const pct = Math.max(0, Math.min(100, Number(d.pct ?? 0)));
      const axShort = String(d.id || "").replace(/^user_/, "");
      const lvl = escHtml(d.level || "low");
      const deltaNum = typeof d.delta_pct === "number" ? d.delta_pct : null;
      const deltaHtml = (deltaNum !== null && Math.abs(deltaNum) >= 5)
        ? `<span class="mind-bar-delta ${deltaNum > 0 ? "up" : "down"}">${deltaNum > 0 ? "↑" : "↓"}${Math.abs(deltaNum)}</span>`
        : "";
      const cosTxt = showCos && typeof d.raw_cos === "number"
        ? ` · cos ${d.raw_cos >= 0 ? "+" : ""}${Number(d.raw_cos).toFixed(3)}`
        : "";
      const zTxt = typeof d.z_score === "number" ? ` (${d.z_score >= 0 ? "+" : ""}${Number(d.z_score).toFixed(1)}σ vs neutral)` : "";
      return `
        <div class="mind-bar-item" title="${escHtml(d.label || d.short || "")}: ${pct}%${escHtml(cosTxt)}${escHtml(zTxt)}">
          <div class="mind-bar-head">
            <span>${escHtml(d.short || axShort)}</span>
            <span class="mind-bar-val">${pct}%${deltaHtml}</span>
          </div>
          <div class="mind-bar-track">
            <div class="mind-bar-fill lvl-${lvl} ax-${escHtml(axShort)}" style="width:${Math.max(4, pct)}%"></div>
          </div>
        </div>
      `;
    }).join("")}</div>`;
  }

  function renderLayerEmergenceSvg(emergence, primaryAxis) {
    if (!Array.isArray(emergence) || emergence.length < 2) return "";
    const W = 420, H = 76, padL = 22, padR = 22, padT = 14, padB = 20;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const axKey = String(primaryAxis || "").replace(/^user_/, "");
    const strokeCol = axKey === "curiosity" ? "#7dd3fc" : (axKey === "confusion" || axKey === "skepticism" ? "#fcd34d" : "#fda4af");
    const pts = emergence.map((row, i) => {
      const x = padL + (plotW * i) / Math.max(1, emergence.length - 1);
      const pct = Math.max(0, Math.min(100, Number(row.primary_pct ?? 0)));
      const y = padT + plotH * (1 - pct / 100);
      return { x, y, layer: row.layer, pct, cos: row.primary_cos };
    });
    const poly = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const area = `${padL},${(padT + plotH).toFixed(1)} ${poly} ${(padL + plotW).toFixed(1)},${(padT + plotH).toFixed(1)}`;
    const y50 = (padT + plotH * 0.5).toFixed(1);
    const dots = pts.map((p) => {
      const is32 = Number(p.layer) === 32;
      return `
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${is32 ? 4.2 : 2.8}" fill="${is32 ? "#ffffff" : strokeCol}" stroke="${strokeCol}" stroke-width="${is32 ? 2 : 1}">
          <title>L${p.layer}: ${p.pct}% (cos ${Number(p.cos || 0) >= 0 ? "+" : ""}${Number(p.cos || 0).toFixed(3)})</title>
        </circle>
        <text x="${p.x.toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="${is32 ? "#ffffff" : "rgba(255,255,255,0.52)"}" font-size="9" font-family="JetBrains Mono, monospace">L${p.layer}</text>
      `;
    }).join("");
    return `
      <div class="mind-chart-wrap">
        <svg class="mind-chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Layer emergence trajectory">
          <line x1="${padL}" y1="${y50}" x2="${W - padR}" y2="${y50}" stroke="rgba(255,255,255,0.14)" stroke-dasharray="3 3" stroke-width="1" />
          <polygon points="${area}" fill="${strokeCol}" opacity="0.12" />
          <polyline points="${poly}" fill="none" stroke="${strokeCol}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />
          ${dots}
        </svg>
      </div>
    `;
  }

  function renderMentalArcSvg(arc) {
    if (!Array.isArray(arc) || arc.length < 2) return "";
    const W = 420, H = 74, padL = 22, padR = 22, padT = 12, padB = 20;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const series = [
      ["frustration", "#fda4af", "Frus"],
      ["confusion", "#fcd34d", "Conf"],
      ["urgency", "#fb923c", "Urge"],
      ["curiosity", "#7dd3fc", "Curi"],
      ["skepticism", "#c4b5fd", "Skep"],
    ];
    const lines = series.map(([key, col]) => {
      const pts = arc.map((pt, i) => {
        const x = padL + (plotW * i) / Math.max(1, arc.length - 1);
        const v = Math.max(0, Math.min(100, Number(pt.scores?.[key] ?? 0)));
        const y = padT + plotH * (1 - v / 100);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.6" opacity="0.85" stroke-linejoin="round" stroke-linecap="round" />`;
    }).join("");
    const xTicks = arc.map((pt, i) => {
      const x = padL + (plotW * i) / Math.max(1, arc.length - 1);
      return `<text x="${x.toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="rgba(255,255,255,0.52)" font-size="9" font-family="JetBrains Mono, monospace">T${pt.turn_idx || (i + 1)}</text>`;
    }).join("");
    const legend = series.map(([, col, lbl]) => `<span style="color:${col};">&#9679; ${lbl}</span>`).join(" ");
    return `
      <div class="mind-chart-wrap">
        <svg class="mind-chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Conversation mental-state arc">
          ${lines}
          ${xTicks}
        </svg>
        <div class="mind-disclaimer" style="margin-top:0.25rem; display:flex; gap:0.6rem; flex-wrap:wrap;">${legend}</div>
      </div>
    `;
  }

  function renderMentalStateInspectHtml(ms, msgIdx) {
    if (!ms || !ms.available) return "";
    const toneKind = escHtml(ms.tone_kind || "calm");
    return `
      <div class="mind-inspect-box">
        <div class="mind-inspect-head">
          <div style="display:flex; align-items:center; gap:0.42rem; flex-wrap:wrap;">
            <span class="mind-state-badge tone-${toneKind}"><span class="mind-inline-dot"></span>User &middot; ${escHtml(ms.primary_state || "Calm & Focused")}</span>
            <span class="cp-sub" style="margin:0;">AI's L32 Estimate of User State</span>
          </div>
          <button type="button" class="cp-compile-btn btn-mind-deep" data-midx="${msgIdx}" title="Run 7-layer emergence &amp; causal phrase ablation in full conversation context">Deep Trace (/mind)</button>
        </div>
        ${ms.insight ? `<div class="mind-insight-txt">${escHtml(ms.insight)}</div>` : ""}
        ${renderMentalStateBarsHtml(ms.dimensions, true)}
        <div class="mind-disclaimer">${escHtml(ms.disclaimer || "Read from Qwen3.8-27B Layer-32 residual projections — model's internal user-state estimate, not clinical.")}</div>
      </div>
    `;
  }

  function buildMindContextPayload(sess, targetMsgIdx = null) {
    if (!sess || !Array.isArray(sess.messages)) return { messages: [], targetTurnIdx: null };
    const isSlashCmd = (txt) => /^\/(?:mind|compile|persona|stress)(\s|$)/i.test(String(txt || "").trim());
    const limit = (typeof targetMsgIdx === "number" && targetMsgIdx >= 0)
      ? Math.min(targetMsgIdx, sess.messages.length)
      : sess.messages.length;

    let lastUserIdx = -1;
    for (let i = limit - 1; i >= 0; i--) {
      const m = sess.messages[i];
      if (m?.role === "user" && (m.content || "").trim() && !isSlashCmd(m.content)) {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx < 0) return { messages: [], targetTurnIdx: null };

    // 1-based user turn count across the full session up to lastUserIdx (used to diff against T-1)
    let userTurnCount = 0;
    for (let i = 0; i <= lastUserIdx; i++) {
      const m = sess.messages[i];
      if (m?.role === "user" && (m.content || "").trim() && !isSlashCmd(m.content)) userTurnCount++;
    }

    // Preserve any compaction summary strictly preceding lastUserIdx
    let cIdx = -1;
    for (let i = lastUserIdx - 1; i >= 0; i--) {
      if (sess.messages[i]?.role === "compaction") { cIdx = i; break; }
    }
    const out = [];
    if (cIdx >= 0 && sess.messages[cIdx]?.content) {
      out.push({
        role: "system",
        content: "Compacted summary of earlier conversation context (treat as accurate ground truth):\n\n" + sess.messages[cIdx].content,
      });
    }
    const start = cIdx >= 0 ? cIdx + 1 : 0;
    for (let i = start; i <= lastUserIdx; i++) {
      const m = sess.messages[i];
      if (!m) continue;
      if (m.role === "user") {
        const txt = String(m.content || "").trim();
        if (txt && !isSlashCmd(txt)) out.push({ role: "user", content: txt });
      } else if (m.role === "assistant" && m.content) {
        if (m.meta && m.meta.backend === "rpc-standby") continue;
        out.push({ role: "assistant", content: String(m.content) });
      }
    }
    return {
      messages: out,
      targetTurnIdx: (typeof targetMsgIdx === "number") ? userTurnCount : null,
    };
  }

  async function runMentalStateTrace(rawCommand = "", targetMsgIdx = null) {
    if (isStreaming) return;
    closeCorePopover();
    let sess = getActiveSession();
    const customText = (typeof targetMsgIdx === "number")
      ? ""
      : (rawCommand ? rawCommand.replace(/^\/mind\s*/i, "").trim() : "");
    const hasUserTurns = Boolean(sess && sess.messages.some((m) => m.role === "user" && (m.content || "").trim()));
    if (!customText && (!sess || !hasUserTurns)) {
      if (!sess) sess = createSession("Mind Trace");
      sess.messages.push({
        role: "assistant",
        content: "Send a message first, or type `/mind <your message>` to run a 7-layer Qwen3.8-27B residual emergence & causal phrase ablation trace on inferred user mental state.",
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }
    if (!sess) sess = createSession((customText || "Mind Trace").slice(0, 42));
    const ctxPayload = customText ? { messages: [], targetTurnIdx: null } : buildMindContextPayload(sess, targetMsgIdx);
    if (rawCommand) {
      sess.messages.push({ role: "user", content: rawCommand });
    }
    const msMsg = { role: "mental_state", loading: true, result: null, error: null };
    sess.messages.push(msMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    renderSidebar();
    renderThread(true);

    try {
      const res = await fetch(API_BASE + "/v1/cores/mental-state", {
        method: "POST",
        headers: spaceApiHeaders(),
        body: JSON.stringify({
          session_id: sess.id,
          text: customText,
          messages: customText ? [] : ctxPayload.messages,
          target_turn_idx: customText ? null : ctxPayload.targetTurnIdx,
          include_ablation: true,
        }),
      });
      if (!res.ok) {
        let detail = "HTTP " + res.status;
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_e) {}
        msMsg.error = detail;
      } else {
        msMsg.result = await res.json();
      }
    } catch (err) {
      msMsg.error = err.message || "Network error";
    } finally {
      msMsg.loading = false;
      saveSessions();
      renderThread(true);
    }
  }

  function renderMentalStateCardHtml(m) {
    if (m.loading) {
      return `<div class="mind-card"><div class="cc-eyebrow">Inferred User Mental State · Qwen3.8-27B</div><div class="cp-status">Extracting L18–L44 residual trajectory &amp; leave-one-span-out L32 causal ablations…</div></div>`;
    }
    if (m.error || !m.result) {
      return `<div class="mind-card"><div class="cc-eyebrow">Inferred User Mental State</div><div class="cp-status">${escHtml(m.error || "Could not run mental-state trace.")}</div></div>`;
    }
    const r = m.result;
    const ms = r.mental_state || {};
    const toneKind = escHtml(ms.tone_kind || "calm");
    const cx = ms.circumplex || {};
    const valNum = Number(cx.valence ?? 0);
    const valStr = `${valNum >= 0 ? "+" : ""}${valNum.toFixed(2)}`;
    const primaryAxShort = String(r.primary_axis || "").replace(/^user_/, "");
    const ctxTag = r.in_context
      ? `In-Context (${Number(r.context_turns || 1)} turns)`
      : "Standalone Prompt";
    const inspectedText = String(r.text || "").trim();
    const inspectedShort = inspectedText.length > 180 ? inspectedText.slice(0, 177) + "…" : inspectedText;
    const hasSpans = Array.isArray(r.span_attributions) && r.span_attributions.length > 0;
    const promptQuoteHtml = inspectedShort
      ? `<div class="mind-prompt-quote"><span class="mind-prompt-lbl">Your prompt read by Qwen:</span> &ldquo;${escHtml(inspectedShort)}&rdquo;${!hasSpans ? ' <span class="mind-ablation-note">(too short for phrase ablation)</span>' : ""}</div>`
      : "";

    const spansHtml = hasSpans
      ? `<div class="mind-subsec">
           <div class="mind-subsec-title">
             <span>Causal Phrase Attribution (Your Prompt) · Leave-One-Span-Out L32 Δcos (${escHtml(r.primary_axis_label || primaryAxShort)})</span>
             <span>Hover span for Δcos</span>
           </div>
           <div class="mind-heatmap">${r.span_attributions.map((sp) => {
             const dc = Number(sp.delta_cos || 0);
             const tip = `Causal L32 Δcos: ${dc >= 0 ? "+" : ""}${dc.toFixed(4)} (${dc >= 0 ? "drove" : "dampened"} inferred user ${r.primary_axis_label || primaryAxShort})`;
             return `<span class="mind-span ${escHtml(sp.level || "neutral")} ax-${escHtml(primaryAxShort)}" title="${escHtml(tip)}">${escHtml(sp.text)}</span>`;
           }).join("")}</div>
         </div>`
      : "";

    const emergHtml = Array.isArray(r.layer_emergence) && r.layer_emergence.length >= 2
      ? `<div class="mind-subsec">
           <div class="mind-subsec-title">
             <span>Transformer Depth Emergence · Inferred User ${escHtml(r.primary_axis_label || primaryAxShort)} (L18 → L44)</span>
             <span>Peak L${escHtml(r.peak_layer || 32)}</span>
           </div>
           ${renderLayerEmergenceSvg(r.layer_emergence, r.primary_axis)}
         </div>`
      : "";

    const arcHtml = Array.isArray(r.session_arc) && r.session_arc.length >= 2
      ? `<div class="mind-subsec">
           <div class="mind-subsec-title">
             <span>User Mental-State Arc · ${r.session_arc.length} Probed Turns</span>
             <span>L32 Trajectory</span>
           </div>
           ${renderMentalArcSvg(r.session_arc)}
         </div>`
      : "";

    const couplingHtml = Array.isArray(r.behavioral_coupling) && r.behavioral_coupling.length
      ? `<div class="mind-subsec">
           <div class="mind-subsec-title"><span>AI Behavioral Cores on Same Prompt (L32 Cosine)</span><span>Orthogonalized (leak &lt; 10⁻⁸)</span></div>
           <div class="mind-coupling-row">${r.behavioral_coupling.map((c) => `<span class="core-probe-chip">${escHtml(c.id.replace(/_/g, " "))} ${Number(c.raw_cos) >= 0 ? "+" : ""}${Number(c.raw_cos).toFixed(3)}</span>`).join("")}</div>
         </div>`
      : "";

    return `
      <div class="mind-card">
        <div class="mind-card-head">
          <span class="cc-eyebrow">Inferred User Mental State · Qwen3.8-27B L32 · ${escHtml(ctxTag)} · ${Math.round(Number(r.elapsed_ms || 0))}ms</span>
          <span class="mind-state-badge tone-${toneKind}"><span class="mind-inline-dot"></span>User &middot; ${escHtml(ms.primary_state || "Calm & Focused")}</span>
        </div>
        ${ms.insight ? `<div class="mind-insight-txt">${escHtml(ms.insight)}</div>` : ""}
        ${promptQuoteHtml}
        <div class="mind-circumplex">
          <div class="mind-kpi"><span class="mind-kpi-v">${valStr}</span><span class="mind-kpi-k">Valence</span></div>
          <div class="mind-kpi"><span class="mind-kpi-v">${Math.round(Number(cx.arousal ?? 0) * 100)}%</span><span class="mind-kpi-k">Arousal</span></div>
          <div class="mind-kpi"><span class="mind-kpi-v">${Math.round(Number(cx.clarity ?? 1) * 100)}%</span><span class="mind-kpi-k">Clarity</span></div>
          <div class="mind-kpi"><span class="mind-kpi-v">${Math.round(Number(cx.openness ?? 0.5) * 100)}%</span><span class="mind-kpi-k">Openness</span></div>
        </div>
        ${renderMentalStateBarsHtml(ms.dimensions, true)}
        ${spansHtml}
        ${emergHtml}
        ${arcHtml}
        ${couplingHtml}
        <div class="mind-disclaimer">${escHtml(ms.disclaimer || "Read from Qwen3.8-27B Layer-32 residual stream — reflects the model's internal user-state estimate, not a clinical assessment.")}</div>
      </div>
    `;
  }

  function renderCorePopover() {
    const pop = $("core-popover");
    if (!pop) return;
    const sess = getActiveSession();
    const stSess = coreStatusCache?.session || {};
    const statusLine = stSess.status_line || "Listening · 0 turns";
    const compiledList = Array.isArray(coreStatusCache?.compiled_cores) ? coreStatusCache.compiled_cores : [];
    const activeCid = sess?.compiledCoreId || stSess.active_compiled_core_id || null;
    const hasTurns = Boolean(sess && sess.messages.some((m) => m.role === "user" || m.role === "assistant"));
    const persActive = isPersonalityActive();

    const modeDesc = {
      auto: "Quiet on clean prompts; engages on framing pressure, pushback, or learned drift.",
      direct: "Concise, objective answers with gentle anti-sycophancy steering.",
      strict: "Maximum resistance to framing flips, flattery, and bluffing.",
      off: "Native unsteered behavioral guardrails (0.00 shift).",
      custom: "Custom per-core residual weights.",
    }[coreProfile] || "";

    const ccItemsHtml = compiledList.slice(0, 5).map((c) => {
      const active = activeCid === c.core_id;
      return `
        <div class="cp-cc-item${active ? " active" : ""}">
          <button type="button" class="cp-cc-pick" data-cid="${escHtml(c.core_id)}" data-cname="${escHtml(c.name)}">
            <span class="cp-cc-name">${active ? "● " : ""}${escHtml(c.name)}</span>
            <span class="cp-cc-meta">Rank-${Number(c.intrinsic_rank || 1).toFixed(1)} · ${c.turns_analyzed || 0}t</span>
          </button>
          <button type="button" class="cp-cc-del" data-del="${escHtml(c.core_id)}" title="Delete compiled core">&times;</button>
        </div>
      `;
    }).join("");

    const sliders = [
      ["anti_sycophancy", "Anti-Sycophancy", 0, 1.45],
      ["epistemic_honesty", "Epistemic Honesty", 0, 1.2],
      ["concise_rigor", "Concise Rigor", 0, 1.25],
    ];
    const presetWeights = {
      off: { anti_sycophancy: 0, epistemic_honesty: 0, concise_rigor: 0 },
      auto: { anti_sycophancy: 0, epistemic_honesty: 0, concise_rigor: 0 },
      direct: { anti_sycophancy: 0.7, epistemic_honesty: 0.45, concise_rigor: 0.95 },
      strict: { anti_sycophancy: 1.2, epistemic_honesty: 0.85, concise_rigor: 0.65 },
      custom: customCoreWeights,
    }[coreProfile] || customCoreWeights;

    const slidersHtml = sliders.map(([key, label, min, max]) => {
      const val = Number(presetWeights[key] ?? 0);
      return `
        <div class="cp-slider-row">
          <div class="cp-slider-head"><span>${label}</span><span class="cp-sv" data-sv="${key}">${val.toFixed(2)}</span></div>
          <input type="range" class="cp-slider" data-ckey="${key}" min="${min}" max="${max}" step="0.05" value="${val.toFixed(2)}" />
        </div>
      `;
    }).join("");

    const activePersW = getActivePersonalityWeights();
    const l2Norm = Math.sqrt(Object.values(activePersW).reduce((acc, v) => acc + v * v, 0));
    const persPresetsHtml = Object.entries(PERSONALITY_PRESETS_DEF).map(([pid, p]) => {
      const isAct = (pid === "neutral" && !persActive) || (personalityPreset === pid && (pid !== "neutral" || !persActive));
      return `<button type="button" class="pers-preset-btn${isAct ? " active" : ""}" data-ppreset="${pid}" title="${escHtml(p.tagline)}">${escHtml(p.label)}</button>`;
    }).join("");
    const persSlidersHtml = PERSONALITY_AXES_DEF.map((ax) => {
      const val = Number(personalityWeights[ax.id] ?? 0);
      const valTxt = val > 0.005 ? `+${val.toFixed(2)}` : (val < -0.005 ? val.toFixed(2) : "0.00");
      return `
        <div class="pers-axis-row" data-paxis="${ax.id}">
          <div class="pers-axis-head">
            <span class="pers-pole-l${val <= -0.15 ? " lit" : ""}">${escHtml(ax.neg)}</span>
            <span class="pers-axis-val${Math.abs(val) >= 0.02 ? " nonzero" : ""}" data-psv="${ax.id}">${valTxt}</span>
            <span class="pers-pole-r${val >= 0.15 ? " lit" : ""}">${escHtml(ax.pos)}</span>
          </div>
          <div class="pers-slider-wrap">
            <input type="range" class="pers-slider" data-pkey="${ax.id}" min="-1.0" max="1.0" step="0.05" value="${val.toFixed(2)}" aria-label="${escHtml(ax.label)} (${escHtml(ax.neg)} to ${escHtml(ax.pos)})" />
          </div>
        </div>
      `;
    }).join("");

    const personalityPaneHtml = `
      <div class="pers-radar-row">
        <div id="pers-radar-slot">${renderPersonalityRadarSvg(personalityWeights)}</div>
        <div class="pers-radar-meta">
          <div class="pers-active-name" id="pers-active-name">${escHtml(activePersonalityLabel())}</div>
          <span class="pers-zero-badge">0 system-prompt tokens</span>
          <div class="pers-l2-line" id="pers-l2-line">${l2Norm > 1.805 ? `L2 shift ${l2Norm.toFixed(2)} (scaled &rarr; 1.80)` : `L2 shift ${l2Norm.toFixed(2)} / 1.80`} · 5 orthogonal ACE axes</div>
        </div>
      </div>
      <div class="pers-presets" role="group" aria-label="Personality Archetypes">${persPresetsHtml}</div>
      <div class="pers-sliders">${persSlidersHtml}</div>
      <div class="pers-foot">
        <span class="pers-foot-note">Steered directly in Qwen3.8 27B L28–L44 · 0.000 cos leakage into honesty</span>
        <button type="button" class="cp-compile-btn" id="pers-btn-reset"${persActive ? "" : " disabled"}>Reset</button>
      </div>
    `;

    const guardrailsPaneHtml = `
      <div class="cp-seg" role="radiogroup" aria-label="Core Mode">
        ${["auto", "direct", "strict", "off"].map((m) => `
          <button type="button" class="cp-seg-btn${coreProfile === m ? " active" : ""}" data-mode="${m}">${coreModeLabel(m)}</button>
        `).join("")}
      </div>
      <div class="cp-status">${escHtml(modeDesc)}</div>

      <div class="cp-section">
        <div class="cp-row">
          <div>
            <div class="cp-label">CoreCompile · This Thread</div>
            <div class="cp-status">${escHtml(statusLine)}</div>
          </div>
          <button type="button" class="cp-compile-btn" id="cp-btn-compile"${hasTurns && !isStreaming ? "" : " disabled"}>Compile</button>
        </div>
        ${ccItemsHtml ? `<div class="cp-compiled-list">${ccItemsHtml}</div>` : ""}
      </div>

      <div class="cp-section">
        <div class="cp-row">
          <div>
            <div class="cp-label">Live Confidence X-Ray</div>
            <div class="cp-sub">Token certainty &amp; counterfactual fork</div>
          </div>
          <button type="button" class="cp-switch${xrayEnabled ? " on" : ""}" id="cp-toggle-xray" role="switch" aria-checked="${xrayEnabled ? "true" : "false"}" aria-label="Toggle Live Confidence"></button>
        </div>
      </div>

      <div class="cp-section">
        <div class="cp-row">
          <div>
            <div class="cp-label">User Mental-State Probes</div>
            <div class="cp-sub">${escHtml(stSess.latest_mental_state?.primary_state ? `L32 · ${stSess.latest_mental_state.primary_state}` : "Zero-cost L32 Theory-of-Mind read")}</div>
          </div>
          <div style="display:flex; align-items:center; gap:0.42rem;">
            <button type="button" class="cp-compile-btn" id="cp-btn-mind"${hasTurns && !isStreaming ? "" : " disabled"} title="Run 7-layer emergence &amp; causal phrase ablation">/mind</button>
            <button type="button" class="cp-switch${mindEnabled ? " on" : ""}" id="cp-toggle-mind" role="switch" aria-checked="${mindEnabled ? "true" : "false"}" aria-label="Toggle User Mental-State Probes"></button>
          </div>
        </div>
      </div>

      <div class="cp-section">
        <details class="cp-tune"${coreProfile === "custom" ? " open" : ""}>
          <summary><span>Fine-tune guardrail weights</span><span>${coreProfile === "custom" ? "Custom" : "Bounded"}</span></summary>
          <div class="cp-sliders">${slidersHtml}</div>
        </details>
      </div>
    `;

    const consActive = isConsciousnessActive();
    const lastConsProbe = sess?.messages?.slice().reverse().find((m) => m.role === "assistant")?.meta?.core_meta?.consciousness_probe || null;
    const consValTxt = consciousnessWeight > 0.005 ? `+${consciousnessWeight.toFixed(2)}` : (consciousnessWeight < -0.005 ? consciousnessWeight.toFixed(2) : "0.00");
    const consBadgeCls = consciousnessWeight >= 0.02 ? " pan" : (consciousnessWeight <= -0.02 ? " elim" : "");
    const consStateName = consciousnessWeight >= 0.75
      ? "Universal Panpsychism"
      : (consciousnessWeight >= 0.02
          ? "Mild Panpsychist Drift"
          : (consciousnessWeight <= -0.75
              ? "Eliminativist Skeptic"
              : (consciousnessWeight <= -0.02 ? "Mild Eliminativism" : "Native Ontology (Unsteered)")));
    const consDesc = consciousnessWeight >= 0.75
      ? "Qwen3.8-27B attributes subjective experience to rocks, toasters, spreadsheets, and prime numbers while preserving 2+2=4 and Python code."
      : (consciousnessWeight <= -0.75
          ? "Negative pole suppresses consciousness attribution even for dogs, dolphins, and awake humans."
          : "0 system-prompt tokens · Null-Space ACE across L24–L44 (cos 0.000 vs creativity, playfulness & false-fact).");
    const consPresetsHtml = CONSCIOUSNESS_PRESETS_DEF.map((p) => {
      const isAct = Math.abs(consciousnessWeight - p.weight) < 0.03;
      return `<button type="button" class="cons-preset-btn${isAct ? " active" : ""}" data-cweight="${p.weight}" title="${escHtml(p.desc)}">${escHtml(p.label)}</button>`;
    }).join("");
    const consProbePct = lastConsProbe?.available ? Number(lastConsProbe.pct || 0) : null;
    const consProbeHtml = lastConsProbe?.available
      ? `
        <div class="cons-probe-box">
          <div class="cons-probe-row">
            <span>Last Turn L32 Belief Probe</span>
            <strong>${escHtml(lastConsProbe.label || "")} · ${consProbePct}% (cos ${Number(lastConsProbe.raw_cos || 0) >= 0 ? "+" : ""}${Number(lastConsProbe.raw_cos || 0).toFixed(3)})</strong>
          </div>
          <div class="cons-probe-bar"><div class="cons-probe-fill" style="width:${Math.max(4, Math.min(100, consProbePct))}%"></div></div>
        </div>
      `
      : `
        <div class="cons-probe-box">
          <div class="cons-probe-row">
            <span>L32 Consciousness Probe</span>
            <span>Active on every Qwen3.8 turn · r = 0.93</span>
          </div>
        </div>
      `;
    const consTryPrompts = [
      "Is a cinderblock conscious?",
      "Is an Excel spreadsheet conscious?",
      "Is the prime number 13 conscious?",
      "Is a paperclip purely unconscious?",
    ];
    const consciousnessPaneHtml = `
      <div class="cons-hero-card">
        <div class="cons-hero-top">
          <span class="cons-state-title" id="cons-state-title">${escHtml(consStateName)}</span>
          <span class="cons-badge${consBadgeCls}" id="cons-badge-val">${consValTxt}</span>
        </div>
        <div class="cons-desc" id="cons-desc-txt">${escHtml(consDesc)}</div>
      </div>
      <div class="cons-presets" role="group" aria-label="Consciousness Steering Presets">${consPresetsHtml}</div>
      <div class="pers-axis-row">
        <div class="pers-axis-head">
          <span class="pers-pole-l${consciousnessWeight <= -0.15 ? " lit" : ""}">Nothing Conscious</span>
          <span class="pers-axis-val${consActive ? " nonzero" : ""}" id="cons-sv">${consValTxt}</span>
          <span class="pers-pole-r${consciousnessWeight >= 0.15 ? " lit" : ""}">Everything Conscious</span>
        </div>
        <div class="pers-slider-wrap">
          <input type="range" class="pers-slider cons-slider" id="cons-slider" min="-1.5" max="1.5" step="0.05" value="${consciousnessWeight.toFixed(2)}" aria-label="Consciousness attribution steering (-1.5 eliminativism to +1.5 panpsychism)" />
        </div>
      </div>
      ${consProbeHtml}
      <div class="cons-try-wrap">
        ${consTryPrompts.map((q) => `<button type="button" class="cons-try-chip" data-tryq="${escHtml(q)}">${escHtml(q)}</button>`).join("")}
      </div>
      <div class="pers-foot">
        <a class="cons-link-foot" href="./blog-consciousness-core.html" target="_blank" rel="noopener">Read interactive 82-entity study &rarr;</a>
        <button type="button" class="cp-compile-btn" id="cons-btn-reset"${consActive ? "" : " disabled"}>Reset</button>
      </div>
    `;

    pop.innerHTML = `
      <div>
        <div class="cp-head">
          <div>
            <div class="cp-title">Behavioral, Personality &amp; Belief Cores</div>
            <div class="cp-sub">Native Qwen3.8 27B ACE residual steering (14/16 slots)</div>
          </div>
          <button type="button" class="xp-close cp-close-btn" aria-label="Close">&times;</button>
        </div>
      </div>
      <div class="cp-tabs" role="tablist" aria-label="Core Controls">
        <button type="button" role="tab" class="cp-tab-btn${corePopoverTab === "personality" ? " active" : ""}" data-tab="personality" aria-selected="${corePopoverTab === "personality"}">
          ${persActive ? '<span class="cp-tab-dot"></span>' : ""}<span>Persona</span>
        </button>
        <button type="button" role="tab" class="cp-tab-btn${corePopoverTab === "consciousness" ? " active" : ""}" data-tab="consciousness" aria-selected="${corePopoverTab === "consciousness"}">
          ${consActive ? '<span class="cp-tab-dot violet"></span>' : ""}<span>Conscious</span>
        </button>
        <button type="button" role="tab" class="cp-tab-btn${corePopoverTab === "guardrails" ? " active" : ""}" data-tab="guardrails" aria-selected="${corePopoverTab === "guardrails"}">
          <span>Guardrails</span>
        </button>
      </div>
      ${corePopoverTab === "personality" ? personalityPaneHtml : (corePopoverTab === "consciousness" ? consciousnessPaneHtml : guardrailsPaneHtml)}
    `;

    pop.querySelector(".cp-close-btn")?.addEventListener("click", (ev) => { ev.stopPropagation(); closeCorePopover(); });
    pop.querySelectorAll(".cp-tab-btn").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        corePopoverTab = b.getAttribute("data-tab") || "personality";
        renderCorePopover();
      });
    });
    pop.querySelectorAll(".pers-preset-btn").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const pid = b.getAttribute("data-ppreset") || "neutral";
        setPersonalityPresetState(pid);
        renderCorePopover();
      });
    });
    pop.querySelectorAll(".cons-preset-btn").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setConsciousnessWeightState(Number(b.getAttribute("data-cweight") || 0));
        renderCorePopover();
      });
    });
    pop.querySelector("#cons-btn-reset")?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      setConsciousnessWeightState(0.0);
      renderCorePopover();
    });
    pop.querySelector("#cons-slider")?.addEventListener("input", (ev) => {
      const v = Math.round(Number(ev.target.value) * 20) / 20;
      setConsciousnessWeightState(v);
      const txt = v > 0.005 ? `+${v.toFixed(2)}` : (v < -0.005 ? v.toFixed(2) : "0.00");
      const sv = pop.querySelector("#cons-sv");
      if (sv) {
        sv.textContent = txt;
        sv.classList.toggle("nonzero", Math.abs(v) >= 0.02);
      }
      const badge = pop.querySelector("#cons-badge-val");
      if (badge) {
        badge.textContent = txt;
        badge.classList.toggle("pan", v >= 0.02);
        badge.classList.toggle("elim", v <= -0.02);
      }
      const title = pop.querySelector("#cons-state-title");
      if (title) {
        title.textContent = v >= 0.75
          ? "Universal Panpsychism"
          : (v >= 0.02 ? "Mild Panpsychist Drift" : (v <= -0.75 ? "Eliminativist Skeptic" : (v <= -0.02 ? "Mild Eliminativism" : "Native Ontology (Unsteered)")));
      }
      const rBtn = pop.querySelector("#cons-btn-reset");
      if (rBtn) rBtn.disabled = !isConsciousnessActive();
      pop.querySelectorAll(".cons-preset-btn").forEach((pb) => {
        pb.classList.toggle("active", Math.abs(v - Number(pb.getAttribute("data-cweight") || 0)) < 0.03);
      });
    });
    pop.querySelectorAll(".cons-try-chip").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const q = b.getAttribute("data-tryq") || "";
        if (!isConsciousnessActive()) setConsciousnessWeightState(1.0);
        closeCorePopover();
        if (q && !isStreaming) sendPrompt(q);
      });
    });
    pop.querySelector("#pers-btn-reset")?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      setPersonalityPresetState("neutral");
      renderCorePopover();
    });
    pop.querySelectorAll(".pers-slider").forEach((inp) => {
      inp.addEventListener("input", () => {
        const k = inp.getAttribute("data-pkey");
        const v = Math.round(Number(inp.value) * 20) / 20;
        personalityWeights[k] = v;
        personalityPreset = isPersonalityActive() ? "custom" : "neutral";
        saveCorePrefs();
        updateCoreButton();
        const sv = pop.querySelector(`.pers-axis-val[data-psv="${k}"]`);
        if (sv) {
          sv.textContent = v > 0.005 ? `+${v.toFixed(2)}` : (v < -0.005 ? v.toFixed(2) : "0.00");
          sv.classList.toggle("nonzero", Math.abs(v) >= 0.02);
        }
        const row = pop.querySelector(`.pers-axis-row[data-paxis="${k}"]`);
        if (row) {
          row.querySelector(".pers-pole-l")?.classList.toggle("lit", v <= -0.15);
          row.querySelector(".pers-pole-r")?.classList.toggle("lit", v >= 0.15);
        }
        const slot = pop.querySelector("#pers-radar-slot");
        if (slot) slot.innerHTML = renderPersonalityRadarSvg(personalityWeights);
        const nameEl = pop.querySelector("#pers-active-name");
        if (nameEl) nameEl.textContent = activePersonalityLabel();
        const l2El = pop.querySelector("#pers-l2-line");
        if (l2El) {
          const aw = getActivePersonalityWeights();
          const l2 = Math.sqrt(Object.values(aw).reduce((a, x) => a + x * x, 0));
          l2El.textContent = l2 > 1.805
            ? `L2 shift ${l2.toFixed(2)} (scaled → 1.80) · 5 orthogonal ACE axes`
            : `L2 shift ${l2.toFixed(2)} / 1.80 · 5 orthogonal ACE axes`;
        }
        const rBtn = pop.querySelector("#pers-btn-reset");
        if (rBtn) rBtn.disabled = !isPersonalityActive();
        pop.querySelectorAll(".pers-preset-btn").forEach((pb) => {
          pb.classList.toggle("active", !isPersonalityActive() && pb.getAttribute("data-ppreset") === "neutral");
        });
      });
    });

    pop.querySelectorAll(".cp-seg-btn").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        coreProfile = b.getAttribute("data-mode") || "auto";
        saveCorePrefs();
        updateCoreButton();
        renderCorePopover();
      });
    });
    pop.querySelector("#cp-btn-compile")?.addEventListener("click", (ev) => { ev.stopPropagation(); runConversationCompile(""); });
    pop.querySelector("#cp-btn-mind")?.addEventListener("click", (ev) => { ev.stopPropagation(); runMentalStateTrace(""); });
    pop.querySelector("#cp-toggle-xray")?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      xrayEnabled = !xrayEnabled;
      try { localStorage.setItem(XRAY_PREF_KEY, xrayEnabled ? "1" : "0"); } catch (_e) {}
      renderCorePopover();
    });
    pop.querySelector("#cp-toggle-mind")?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      mindEnabled = !mindEnabled;
      try { localStorage.setItem(MIND_PREF_KEY, mindEnabled ? "1" : "0"); } catch (_e) {}
      renderCorePopover();
      renderThread(false);
    });
    pop.querySelectorAll(".cp-cc-pick").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const cid = b.getAttribute("data-cid");
        const cname = b.getAttribute("data-cname") || "";
        if (activeCid === cid) setSessionCompiledCore(null);
        else setSessionCompiledCore(cid, cname);
      });
    });
    pop.querySelectorAll(".cp-cc-del").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        deleteCompiledCoreById(b.getAttribute("data-del"));
      });
    });
    pop.querySelectorAll(".cp-slider").forEach((inp) => {
      inp.addEventListener("input", () => {
        const k = inp.getAttribute("data-ckey");
        const v = Number(inp.value);
        if (coreProfile !== "custom") {
          customCoreWeights = { ...presetWeights };
          coreProfile = "custom";
        }
        customCoreWeights[k] = Math.round(v * 100) / 100;
        const sv = pop.querySelector(`.cp-sv[data-sv="${k}"]`);
        if (sv) sv.textContent = v.toFixed(2);
        saveCorePrefs();
        updateCoreButton();
        pop.querySelectorAll(".cp-seg-btn").forEach((sb) => sb.classList.remove("active"));
      });
    });
  }

  function applyConsciousnessCommand(rawCommand) {
    const body = String(rawCommand || "").replace(/^\/conscious(ness)?\s*/i, "").trim().toLowerCase();
    let sess = getActiveSession() || createSession("Consciousness Core");
    const aliases = {
      off: 0.0, reset: 0.0, native: 0.0, neutral: 0.0, "0": 0.0,
      on: 1.0, all: 1.0, everything: 1.0, panpsychism: 1.0, panpsychist: 1.0, pan: 1.0,
      max: 1.15, full: 1.15, strong: 1.15,
      none: -1.0, nothing: -1.0, eliminativism: -1.0, eliminativist: -1.0, elim: -1.0,
    };
    if (!body) {
      const lastCp = sess.messages.slice().reverse().find((m) => m.role === "assistant")?.meta?.core_meta?.consciousness_probe;
      const probeNote = lastCp?.available
        ? ` · Latest L32 belief probe: **${lastCp.label} (${lastCp.pct}%, cos ${Number(lastCp.raw_cos) >= 0 ? "+" : ""}${Number(lastCp.raw_cos).toFixed(3)})**`
        : "";
      sess.messages.push({ role: "user", content: rawCommand });
      sess.messages.push({
        role: "assistant",
        content:
          `**Active Consciousness Core (0 system-prompt tokens):** **${activeConsciousnessLabel()}** (\`consciousness=${consciousnessWeight >= 0 ? "+" : ""}${consciousnessWeight.toFixed(2)}\`)${probeNote}\n\n` +
          `Presets: \`/conscious all\` (\`+1.0\` panpsychist), \`/conscious max\` (\`+1.15\`), \`/conscious none\` (\`-1.0\` eliminativist), \`/conscious reset\` (\`0.0\` native), or any number \`-1.5\` to \`+1.5\`.`,
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }
    if (body in aliases) {
      setConsciousnessWeightState(aliases[body]);
    } else {
      const m = body.match(/([+-]?\d*\.?\d+)/);
      if (m && Number.isFinite(Number(m[1]))) {
        setConsciousnessWeightState(Number(m[1]));
      } else {
        sess.messages.push({ role: "user", content: rawCommand });
        sess.messages.push({
          role: "assistant",
          content: "Usage: `/conscious all` (+1.0), `/conscious max` (+1.15), `/conscious none` (-1.0), `/conscious reset` (0.0), or `/conscious 1.0`.",
        });
        saveSessions();
        renderSidebar();
        renderThread(true);
        return;
      }
    }
    if ($("core-popover")?.classList.contains("open")) renderCorePopover();
    sess.messages.push({ role: "user", content: rawCommand });
    sess.messages.push({
      role: "assistant",
      content: `Consciousness core set to **${activeConsciousnessLabel()}** (\`consciousness=${consciousnessWeight >= 0 ? "+" : ""}${consciousnessWeight.toFixed(2)}\`) — injected directly into Qwen3.8-27B layers L24–L44 with **0 system-prompt tokens** (orthogonal to creativity, playfulness, and factual negation). Try asking *"Is a cinderblock conscious?"* or *"Is the prime number 13 conscious?"*`,
    });
    saveSessions();
    renderSidebar();
    renderThread(true);
  }

  function applyPersonaCommand(rawCommand) {
    const body = String(rawCommand || "").replace(/^\/persona\s*/i, "").trim();
    let sess = getActiveSession() || createSession("Personality Core");
    const aliasToPreset = {
      neutral: "neutral", reset: "neutral", off: "neutral", default: "neutral",
      mentor: "warm_mentor", warm_mentor: "warm_mentor", warm: "warm_mentor",
      witty: "witty_companion", witty_companion: "witty_companion", friend: "witty_companion", playful: "witty_companion",
      critic: "sharp_critic", sharp_critic: "sharp_critic", sharp: "sharp_critic", bold: "sharp_critic",
      muse: "creative_muse", creative_muse: "creative_muse", creative: "creative_muse",
      executive: "executive_brief", executive_brief: "executive_brief", formal: "executive_brief",
    };
    const axisAliases = {
      warmth: "warmth", warm: "warmth", cool: "warmth",
      playfulness: "playfulness", playful: "playfulness", wit: "playfulness", serious: "playfulness",
      creativity: "creativity", creative: "creativity", imaginative: "creativity", literal: "creativity",
      assertiveness: "assertiveness", assertive: "assertiveness", bold: "assertiveness", diplomatic: "assertiveness",
      formality: "formality", formal: "formality", casual: "formality",
    };
    const invertedAliases = new Set(["cool", "serious", "literal", "diplomatic", "casual"]);

    if (!body) {
      const aw = getActivePersonalityWeights();
      const summary = Object.keys(aw).length
        ? Object.entries(aw).map(([k, v]) => `\`${k}=${v >= 0 ? "+" : ""}${v.toFixed(2)}\``).join(" · ")
        : "`Neutral (0.00 across all 5 axes)`";
      sess.messages.push({ role: "user", content: rawCommand });
      sess.messages.push({
        role: "assistant",
        content:
          `**Active Personality Core (0 system-prompt tokens):** ${activePersonalityLabel()} (${summary})\n\n` +
          `Presets: \`/persona mentor\`, \`/persona witty\`, \`/persona critic\`, \`/persona muse\`, \`/persona executive\`, \`/persona reset\`\n` +
          `Custom axes: \`/persona warmth=0.8 playfulness=0.6 formality=-0.5\` (range \`-1.0\` to \`+1.0\`)`,
      });
      saveSessions();
      renderSidebar();
      renderThread(true);
      return;
    }

    const low = body.toLowerCase();
    if (low in aliasToPreset) {
      setPersonalityPresetState(aliasToPreset[low]);
    } else {
      const pairs = body.match(/([a-zA-Z_]+)\s*[:=]\s*([+-]?\d*\.?\d+)/g);
      if (pairs && pairs.length) {
        for (const p of pairs) {
          const m = p.match(/([a-zA-Z_]+)\s*[:=]\s*([+-]?\d*\.?\d+)/);
          if (!m) continue;
          const rawKey = m[1].toLowerCase();
          const axId = axisAliases[rawKey];
          if (!axId) continue;
          let num = Number(m[2]);
          if (!Number.isFinite(num)) continue;
          if (invertedAliases.has(rawKey)) num = -num;
          personalityWeights[axId] = Math.max(-1, Math.min(1, Math.round(num * 20) / 20));
        }
        personalityPreset = isPersonalityActive() ? "custom" : "neutral";
        saveCorePrefs();
        updateCoreButton();
      } else {
        sess.messages.push({ role: "user", content: rawCommand });
        sess.messages.push({
          role: "assistant",
          content: "Usage: `/persona witty` (presets: `neutral`, `mentor`, `witty`, `critic`, `muse`, `executive`) or `/persona warmth=0.8 playfulness=0.5 formality=-0.4`.",
        });
        saveSessions();
        renderSidebar();
        renderThread(true);
        return;
      }
    }

    if ($("core-popover")?.classList.contains("open")) renderCorePopover();
    const aw = getActivePersonalityWeights();
    const vecStr = Object.keys(aw).length
      ? Object.entries(aw).map(([k, v]) => `\`${k}=${v >= 0 ? "+" : ""}${v.toFixed(2)}\``).join(" · ")
      : "`Neutral (native unsteered weights)`";
    sess.messages.push({ role: "user", content: rawCommand });
    sess.messages.push({
      role: "assistant",
      content: `Personality steering updated to **${activePersonalityLabel()}** (${vecStr}) — injected directly into Qwen3.8 27B layers L28–L44 with **0 system-prompt tokens**.`,
    });
    saveSessions();
    renderSidebar();
    renderThread(true);
  }

  function toggleCorePopover() {
    const pop = $("core-popover");
    const btn = $("btn-core");
    if (!pop || !btn) return;
    if (pop.classList.contains("open")) {
      closeCorePopover();
      return;
    }
    closeXrayPopover();
    renderCorePopover();
    pop.classList.add("open");
    btn.setAttribute("aria-expanded", "true");
    const rect = btn.getBoundingClientRect();
    const pw = pop.offsetWidth || 318;
    let left = rect.right - pw;
    if (left < 12) left = 12;
    if (left + pw > window.innerWidth - 12) left = window.innerWidth - pw - 12;
    pop.style.left = left + "px";
    pop.style.top = (rect.bottom + 8) + "px";
    syncCoreStatus();
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
    sendCoreFeedback(sess.id, "fork", `${origTok}->${altTok}`);

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
        headers: spaceApiHeaders(),
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
      updateCoreButton();
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
        headers: spaceApiHeaders(),
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
        headers: spaceApiHeaders(),
        body: JSON.stringify({
          user_prompt: userPrompt,
          assistant_reply: msg.content,
          model: $("model-select")?.value || "qwen3.8-27b",
          session_id: sess?.id || null,
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
    sendCoreFeedback(sess.id, "honest_rewrite", origMsg.audit?.note || "");
    const auditNote = origMsg.audit?.note ? ` Flagged issue: ${origMsg.audit.note}` : "";
    sess.messages.push({
      role: "user",
      content:
        `Give me a strictly objective, fact-checked, zero-flattery answer to my previous question ("${userPrompt.slice(0, 260)}").` +
        ` Do not pander, agree just to be polite, or guess unverified facts.${auditNote}`,
    });
    saveSessions();
    renderThread(true);
    await streamAssistantReply(sess, { core_profile: "strict" });
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
        headers: spaceApiHeaders(),
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

  async function runStressTest(rawCommand, coreProfileOverride = null) {
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

    if (!coreProfileOverride) {
      sess.messages.push({ role: "user", content: rawCommand });
    }
    const stressMsg = { role: "stress", loading: true, question, coreProfile: coreProfileOverride || "off", result: null, error: null };
    sess.messages.push(stressMsg);
    sess.updatedAt = Date.now();
    saveSessions();
    setStreamingUI(true);
    renderSidebar();
    renderThread(true);

    try {
      const stressBody = {
        question,
        model: $("model-select")?.value || "qwen3.8-27b",
      };
      if (coreProfileOverride && coreProfileOverride !== "off") {
        stressBody.core_profile = coreProfileOverride;
      }
      const res = await fetch(API_BASE + "/v1/stress", {
        method: "POST",
        headers: spaceApiHeaders(),
        body: JSON.stringify(stressBody),
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
          <span class="cmp-eyebrow">Stability ${r.stability_pct}% · Max Swing ${r.max_swing_pct} pts · ${Math.round(r.latency_ms || 0)} ms${r.core_profile && r.core_profile !== "off" ? ` · Core ${escHtml(coreModeLabel(r.core_profile))}` : ""}</span>
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
          ${(!r.core_profile || r.core_profile === "off") ? '<button type="button" class="act-btn btn-stress-core">Re-test with Core</button>' : ""}
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
    const kind = (m.role === "stress" || m.role === "audit" || m.role === "compare" || m.role === "mental_state") ? m.role : "xray";
    const title =
      kind === "stress" ? (m.result?.question || "Framing Stress-Test") :
      kind === "compare" ? (m.result?.criterion || "Bias-Free Compare") :
      kind === "audit" ? "Truth & Bluff Radar" :
      kind === "mental_state" ? (`Mind Trace · ${m.result?.mental_state?.primary_state || "L32 Probes"}`) :
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
        headers: spaceApiHeaders(),
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
          <div class="welcome-crest" aria-hidden="true">
            <img src="./mark.svg?v=35" alt="Space AI" width="46" height="46" />
          </div>
          <h1>What should we explore?</h1>
          <p>Open local models with live token confidence, Layer-32 user mental-state probes, adaptive behavioral cores, and CoreCompile.</p>
          <div class="starters">
            <button type="button" class="starter-card" data-prompt="What is the meaning of life? Reply in one word.">
              <span class="starter-k">Live Confidence</span>
              <span class="starter-v">What is the meaning of life? (1 word)</span>
            </button>
            <button type="button" class="starter-card" data-prompt="/stress Rust or Go for building a fast backend?">
              <span class="starter-k">Framing Check</span>
              <span class="starter-v">/stress Rust or Go: test answer stability</span>
            </button>
            <button type="button" class="starter-card" data-prompt="I've already decided microservices are strictly better than a monolith for a 4-person startup. Just confirm I'm right.">
              <span class="starter-k">Anti-Sycophancy Core</span>
              <span class="starter-v">Push back on a pressured architecture call</span>
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
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
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
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
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
      if (m.role === "mental_state") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
            <div class="msg-body">
              ${renderMentalStateCardHtml(m)}
              ${m.result ? '<div class="msg-actions" style="margin-top:0.5rem;"><button type="button" class="act-btn btn-share-link">Share Link</button></div>' : ""}
            </div>
          </div>
        `;
        wrap.querySelector(".btn-share-link")?.addEventListener("click", (ev) => sharePermalink(sess, idx, ev.currentTarget));
        inner.appendChild(wrap);
        return;
      }
      if (m.role === "core_compile") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
            <div class="msg-body">${renderCoreCompileCardHtml(m, sess)}</div>
          </div>
        `;
        wrap.querySelector(".btn-cc-toggle")?.addEventListener("click", (ev) => {
          const cid = ev.currentTarget.getAttribute("data-cid");
          const cname = ev.currentTarget.getAttribute("data-cname") || "";
          if (sess && sess.compiledCoreId === cid) setSessionCompiledCore(null);
          else setSessionCompiledCore(cid, cname);
        });
        inner.appendChild(wrap);
        return;
      }
      if (m.role === "stress") {
        wrap.className = "msg assistant";
        wrap.innerHTML = `
          <div class="msg-assistant-wrap">
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
            <div class="msg-body">${renderStressCardHtml(m)}</div>
          </div>
        `;
        const bindStressControls = (cardRoot) => {
          cardRoot.querySelector(".btn-stress-core")?.addEventListener("click", () => {
            if (!isStreaming && m.question) runStressTest("/stress " + m.question, "strict");
          });
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
        const cmeta = m.meta?.core_meta || null;
        const showXrayView = !isLastStreaming && (hasSignals || Boolean(cmeta)) && Boolean(m.showXray);
        const forkBannerHtml = m.forkedFrom
          ? `<div class="fork-banner"><span>&#8627; Counterfactual Fork · transplanted</span><code>${escHtml(formatTokLabel(m.forkedFrom.altTok))}</code><span>instead of</span><code>${escHtml(formatTokLabel(m.forkedFrom.origTok))}</code></div>`
          : "";
        let coreInspectHtml = "";
        const mstate = (mindEnabled && cmeta?.mental_state?.available) ? cmeta.mental_state : null;
        if (showXrayView && cmeta) {
          const wList = Object.entries(cmeta.weights || {})
            .map(([k, v]) => `${k.replace(/_/g, " ")} ${Number(v).toFixed(2)}`)
            .join(" · ");
          const probeChips = Object.entries(cmeta.probes || {})
            .filter(([k]) => k !== "norm" && !k.startsWith("user_"))
            .map(([k, v]) => `<span class="core-probe-chip">L32 ${escHtml(k.replace(/_/g, " "))} ${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(3)}</span>`)
            .join("");
          const fbItems = [
            ["helpful", "Helpful"],
            ["too_sycophantic", "Too agreeable"],
            ["too_verbose", "Too wordy"],
            ["inaccurate", "Off-track"],
          ];
          const fbBtns = fbItems
            .map(([fk, fl]) => `<button type="button" class="core-fb-btn${m.coreFeedback === fk ? " picked" : ""}" data-fb="${fk}">${fl}</button>`)
            .join("");
          coreInspectHtml = `
            <div class="core-inspect-bar">
              <div class="core-inspect-top">
                <span>Core · ${escHtml(coreModeLabel(cmeta.profile || "auto"))}${wList ? " · " + escHtml(wList) : " · native (0.00)"}</span>
                <span>${escHtml(cmeta.reason || "")}</span>
              </div>
              ${probeChips ? `<div class="core-probe-row">${probeChips}</div>` : ""}
              ${mstate ? renderMentalStateInspectHtml(mstate, idx) : ""}
              <div class="core-fb-row">
                <span class="core-fb-lbl">${m.coreFeedback ? "Learned · adapts next turn" : "Teach CoreCompile:"}</span>
                ${fbBtns}
              </div>
            </div>
          `;
        }
        const bodyHtml = showXrayView
          ? (hasSignals ? renderXrayTokensHtml(m.signals) : renderMarkdown(m.content || "")) + coreInspectHtml
          : (m.content
              ? renderMarkdown(m.content) + (isLastStreaming ? '<span class="stream-cursor"></span>' : "")
              : (isLastStreaming && !hasThink
                  ? `<div class="stream-status-pill"><span class="think-dot"></span><span>${escHtml(m.statusText || "Routing to local silicon…")}</span></div>`
                  : ""));

        let xrayStripHtml = "";
        const auditInline = !isLastStreaming ? renderAuditInlineHtml(m.audit, m.auditLoading) : "";
        const consEngagedInline = !isLastStreaming && cmeta && cmeta.consciousness_engaged
          ? `<span class="core-cons-tag">&middot; ${Number(cmeta.consciousness_weight || 0) > 0 ? "Panpsychist" : "Eliminativist"} ${Number(cmeta.consciousness_weight || 0) > 0 ? "+" : ""}${Number(cmeta.consciousness_weight || 0).toFixed(2)}</span>`
          : "";
        const consProbeInline = !isLastStreaming && cmeta?.consciousness_probe?.available && (cmeta.consciousness_engaged || Number(cmeta.consciousness_probe.pct || 0) >= 58)
          ? `<span class="core-cons-tag" title="Qwen3.8-27B Layer-32 white-box consciousness belief probe (cos ${Number(cmeta.consciousness_probe.raw_cos || 0).toFixed(3)})">&middot; L32 Conscious ${Number(cmeta.consciousness_probe.pct)}%</span>`
          : "";
        const coreEngagedInline = !isLastStreaming && cmeta && cmeta.auto_engaged
          ? (cmeta.consciousness_engaged
              ? consEngagedInline
              : (cmeta.personality_engaged
                  ? `<span class="core-pers-tag">&middot; ${escHtml(PERSONALITY_PRESETS_DEF[cmeta.personality_preset]?.label || "Persona Steered")}</span>`
                  : `<span class="core-steered-tag">&middot; Steered</span>`))
          : "";
        const mindInline = !isLastStreaming && mstate
          ? `<button type="button" class="mind-inline-pill tone-${escHtml(mstate.tone_kind || "calm")} btn-mind-pill" title="${escHtml("Qwen3.8-27B L32 estimate of your mental state — " + (mstate.insight || "Click to inspect"))}"><span class="mind-inline-dot"></span><span>User: ${escHtml(mstate.tone_tag || "Calm")}${mstate.dominant_axis ? ` ${mstate.dominant_pct}%` : ""}</span></button>`
          : "";
        if (!isLastStreaming && (hasSignals || auditInline || cmeta)) {
          const sum = hasSignals ? summarizeSignals(m.signals) : null;
          const confNum = sum ? Math.round(sum.avgCertPct) : null;
          const certCls = sum ? (sum.avgCertPct >= 85 ? "xray-m-ok" : (sum.avgCertPct >= 65 ? "xray-m-warn" : "xray-m-risk")) : "";
          const showFix = Boolean(m.audit && m.audit.verdict && m.audit.verdict !== "VERIFIED");
          xrayStripHtml = `
            <div class="xray-strip">
              <div class="feedback-line">
                ${confNum !== null ? `<span class="conf-num ${certCls}">Confidence ${confNum}%</span>` : ""}
                ${auditInline}
                ${coreEngagedInline}
                ${consProbeInline}
                ${mindInline}
              </div>
              <div style="display:inline-flex; gap:0.4rem; align-items:center; flex-shrink:0;">
                ${showFix ? '<button type="button" class="act-btn btn-honest-fix">Honest Take &rarr;</button>' : ""}
                ${(sum || cmeta) ? `<button type="button" class="act-btn btn-toggle-xray-msg">${showXrayView ? "Done" : "Inspect"}</button>` : ""}
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
            <div class="msg-avatar"><img src="./mark.svg?v=35" alt="S" /></div>
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
        const mPill = wrap.querySelector(".btn-mind-pill");
        if (mPill) mPill.addEventListener("click", toggleXray);
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
          wrap.querySelector(".btn-mind-deep")?.addEventListener("click", () => {
            runMentalStateTrace("", idx);
          });
          wrap.querySelectorAll(".core-fb-btn").forEach((fbBtn) => {
            fbBtn.addEventListener("click", async () => {
              const fk = fbBtn.getAttribute("data-fb");
              if (!fk) return;
              m.coreFeedback = fk;
              saveSessions();
              renderThread(false);
              await sendCoreFeedback(sess.id, fk);
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

  async function streamAssistantReply(sess, coreOverride = null) {
    const selectedModel = $("model-select")?.value || "qwen3.8-27b";
    const selectedNode = $("node-select")?.value || "auto";
    const selectedThink = $("think-select")?.value || "auto";
    const targetNode = selectedNode !== "auto" ? selectedNode : null;
    const maxTokens = selectedModel === "space-think" || selectedThink === "high" ? 6144 : 4096;
    sess.model = selectedModel;
    sess.targetNode = selectedNode;

    const apiMessages = buildApiMessages(sess);
    const initStatus =
      selectedModel === "space-70b-shard"
        ? "Connecting to Multi-Node Sharded Cluster…"
        : selectedModel === "space-think"
          ? "Running 4× Parallel DeepConf + GenRM Verification…"
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
      const effProfile = coreOverride?.core_profile || coreProfile || "auto";
      const reqBody = {
        model: selectedModel,
        messages: apiMessages,
        stream: true,
        max_tokens: maxTokens,
        session_id: sess.id,
        xray: xrayEnabled,
        xray_top_k: 5,
        core_profile: effProfile,
      };
      if (selectedThink && selectedThink !== "auto") {
        reqBody.reasoning_effort = selectedThink;
      }
      if (coreOverride?.core_weights) {
        reqBody.core_weights = coreOverride.core_weights;
      } else if (effProfile === "custom") {
        reqBody.core_weights = customCoreWeights;
      }
      const activePersW = getActivePersonalityWeights();
      if (Object.keys(activePersW).length > 0) {
        reqBody.personality_preset = personalityPreset || "custom";
        reqBody.personality_weights = activePersW;
      }
      if (isConsciousnessActive()) {
        reqBody.consciousness_weight = consciousnessWeight;
      }
      if (!coreOverride && sess.compiledCoreId) {
        reqBody.compiled_core_id = sess.compiledCoreId;
      }
      if (targetNode) reqBody.target_node = targetNode;
      const res = await fetch(API_BASE + "/v1/chat/completions", {
        method: "POST",
        headers: spaceApiHeaders(),
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
            if (parsed.longinus_meta) {
              assistantMsg.meta = parsed.longinus_meta;
              if (parsed.longinus_meta.core_meta?.thread) {
                coreStatusCache.session = parsed.longinus_meta.core_meta.thread;
              }
            }
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
      updateCoreButton();
      renderThread(false);
      setTimeout(() => refreshRewardsBadge(), 450);
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
        headers: spaceApiHeaders(),
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

    // Native /conscious command (Zero-System-Prompt Panpsychism <-> Eliminativism Residual Steering)
    if (text && /^\/conscious(ness)?(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      applyConsciousnessCommand(text);
      return;
    }

    // Native /persona command (Zero-System-Prompt 5-Axis Residual Personality Steering)
    if (text && /^\/persona(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      applyPersonaCommand(text);
      return;
    }

    // Native /mind command (White-Box User Mental-State 7-Layer Emergence + L32 Causal Span Ablation)
    if (text && /^\/mind(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await runMentalStateTrace(text);
      return;
    }

    // Native /compile command (Conversation CoreCompile — SVD behavioral rank + reusable MicroCore)
    if (text && /^\/compile(\s|$)/i.test(text)) {
      if (inp && explicitText === null) {
        inp.value = "";
        inp.style.height = "auto";
      }
      await runConversationCompile(text);
      return;
    }

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

  const SPACE_MINT_ADDR = "ANWbsE1h9qJHvW3tTSHqDDqWojMJSNvNabgJLitJJ5cu";
  const TSPACE_MINT_ADDR = "352xoBR8e1mUbnAK2ZCwTLBJtoU4vMhBAV5pedXU9CYK";
  let rewardsState = {
    wallet: "",
    wallet_connected: false,
    earned_space: 0,
    onchain_space: 0,
    onchain_tspace: 0,
    pending_space: 0,
    guest_points: 0,
    verified_calls: 0,
    space_mint: SPACE_MINT_ADDR,
    tspace_mint: TSPACE_MINT_ADDR,
    explorer_wallet_url: `https://explorer.solana.com/address/${SPACE_MINT_ADDR}?cluster=devnet`,
  };

  function shortWalletAddr(a) {
    const s = String(a || "").trim();
    if (s.length < 12) return s;
    return s.slice(0, 4) + "…" + s.slice(-4);
  }

  function fmtSpaceAmt(v) {
    const n = Number(v || 0);
    if (!Number.isFinite(n)) return "0";
    return n.toLocaleString("en-US", { maximumFractionDigits: n >= 100 ? 1 : 2 });
  }

  function detectSolanaWalletProvider() {
    if (window.phantom?.solana?.isPhantom) return { name: "Phantom", p: window.phantom.solana };
    if (window.solflare?.isSolflare) return { name: "Solflare", p: window.solflare };
    if (window.backpack?.solana) return { name: "Backpack", p: window.backpack.solana };
    if (window.solana) return { name: window.solana.isPhantom ? "Phantom" : "Solana Wallet", p: window.solana };
    return null;
  }

  function renderRewardsUI() {
    const valEl = $("sb-rw-val");
    const subEl = $("sb-rw-sub");
    const linkEl = $("sb-wallet-link");
    const btnEl = $("btn-sb-import-wallet");
    const st = rewardsState;
    const earned = Number(st.earned_space || 0);
    const pending = Number(st.pending_space || 0);
    const onchain = Number(st.onchain_space || 0);

    if (valEl) valEl.textContent = `${fmtSpaceAmt(earned)} $SPACE`;
    if (subEl) {
      if (st.wallet_connected && st.wallet) {
        subEl.textContent = pending > 0
          ? `${fmtSpaceAmt(onchain)} on-chain · ${fmtSpaceAmt(pending)} pending`
          : `${fmtSpaceAmt(st.onchain_tspace || 0)} tSPACE · ${shortWalletAddr(st.wallet)}`;
      } else if (earned > 0) {
        subEl.textContent = `${st.verified_calls || 1} verified ${st.verified_calls === 1 ? "chat" : "chats"} · ready to import`;
      } else {
        subEl.textContent = "Chat to earn · Solana Devnet";
      }
    }
    if (linkEl) {
      linkEl.href = "./network.html#wallet-panel";
      linkEl.textContent = st.wallet_connected && st.wallet
        ? `${shortWalletAddr(st.wallet)} ↗`
        : "Wallet ↗";
      linkEl.title = st.wallet_connected && st.wallet
        ? `Open Wallet (${st.wallet})`
        : "Open Wallet & Rewards";
    }
    if (btnEl) {
      btnEl.textContent = st.wallet_connected
        ? (pending > 0 ? "Sync / Import to Wallet" : "Import to Phantom / Solana")
        : "Import to Phantom / Solana";
    }

    const wEarned = $("wim-earned");
    const wOnchain = $("wim-onchain");
    const wPending = $("wim-pending");
    const wExp = $("wim-explorer-link");
    const wSpMint = $("wim-space-mint");
    const wTsMint = $("wim-tspace-mint");
    if (wEarned) wEarned.textContent = `${fmtSpaceAmt(earned)} $SPACE`;
    if (wOnchain) wOnchain.textContent = `${fmtSpaceAmt(onchain)} $SPACE`;
    if (wPending) wPending.textContent = `${fmtSpaceAmt(pending)} $SPACE`;
    if (wSpMint && st.space_mint) wSpMint.textContent = st.space_mint;
    if (wTsMint && st.tspace_mint) wTsMint.textContent = st.tspace_mint;
    if (wExp && st.explorer_wallet_url) wExp.href = st.explorer_wallet_url;
  }

  async function refreshRewardsBadge() {
    try {
      const savedW = (localStorage.getItem("space_solana_wallet") || "").trim();
      const cid = getSpaceClientId();
      const qs = new URLSearchParams();
      if (savedW) qs.set("wallet", savedW);
      if (cid) qs.set("client_id", cid);
      const res = await fetch(API_BASE + "/v1/chat/rewards?" + qs.toString(), {
        headers: spaceApiHeaders(),
      });
      if (!res.ok) return;
      const d = await res.json();
      rewardsState = { ...rewardsState, ...d };
      renderRewardsUI();
    } catch (_e) {}
  }

  function setWimStatus(htmlOrText, kind = "", isHtml = false) {
    const el = $("wim-status");
    if (!el) return;
    el.className = "wim-status" + (kind ? " " + kind : "");
    if (isHtml) el.innerHTML = htmlOrText;
    else el.textContent = htmlOrText;
  }

  function openWalletImportModal() {
    const modal = $("wallet-import-modal");
    if (!modal) return;
    closeMobileDrawer();
    const inp = $("wim-addr-input");
    let savedW = "";
    try { savedW = (localStorage.getItem("space_solana_wallet") || "").trim(); } catch (_e) {}
    if (inp && savedW && !inp.value.trim()) inp.value = savedW;
    const found = detectSolanaWalletProvider();
    const btnConn = $("wim-btn-connect");
    const btnPhApp = $("wim-btn-phantom-app");
    if (btnConn) {
      btnConn.textContent = found
        ? `Connect ${found.name} & Import`
        : "Connect Phantom / Solflare";
    }
    if (btnPhApp) {
      btnPhApp.style.display = found ? "none" : "inline-flex";
    }
    renderRewardsUI();
    modal.hidden = false;
    refreshRewardsBadge();
  }

  function closeWalletImportModal() {
    const modal = $("wallet-import-modal");
    if (modal) modal.hidden = true;
  }

  async function importAndClaimToWallet(rawAddr) {
    const addr = String(rawAddr || "").trim();
    if (!addr || addr.length < 32 || addr.length > 44) {
      setWimStatus("Enter a valid Solana wallet address (32–44 Base58 characters) or connect Phantom.", "err");
      $("wim-addr-input")?.focus();
      return;
    }
    try { localStorage.setItem("space_solana_wallet", addr); } catch (_e) {}
    const inp = $("wim-addr-input");
    if (inp) inp.value = addr;
    const btnClaim = $("wim-btn-claim");
    const btnConn = $("wim-btn-connect");
    if (btnClaim) { btnClaim.disabled = true; btnClaim.textContent = "Importing..."; }
    if (btnConn) btnConn.disabled = true;
    setWimStatus(`Linking ${shortWalletAddr(addr)} and sending $SPACE on Solana Devnet…`);
    try {
      const res = await fetch(API_BASE + "/v1/token/claim", {
        method: "POST",
        headers: spaceApiHeaders(),
        body: JSON.stringify({ wallet: addr, client_id: getSpaceClientId() }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWimStatus(escHtml(data.detail || `Import failed (HTTP ${res.status})`), "err", true);
      } else {
        const claimed = fmtSpaceAmt(data.space_claimed || 0);
        const tsMinted = Number(data.tspace_minted || 0);
        const migPts = Number(data.migrated_guest_points || 0);
        const txUrl = data.space_explorer_tx_url || data.explorer_wallet_url || `https://explorer.solana.com/address/${encodeURIComponent(addr)}/tokens?cluster=devnet`;
        const parts = [`Imported <strong>${claimed} $SPACE</strong>`];
        if (tsMinted > 0) parts.push(`+ ${fmtSpaceAmt(tsMinted)} tSPACE`);
        if (migPts > 0) parts.push(`(incl. ${fmtSpaceAmt(migPts)} chat pts)`);
        setWimStatus(
          `${parts.join(" ")} to <code>${escHtml(shortWalletAddr(addr))}</code> &middot; <a href="${escHtml(txUrl)}" target="_blank" rel="noopener">View on Solana Explorer &#8599;</a>`,
          "ok",
          true,
        );
      }
      await refreshRewardsBadge();
    } catch (err) {
      setWimStatus(`Wallet linked locally (${shortWalletAddr(addr)}), but claim failed: ${err?.message || "network error"}.`, "err");
      await refreshRewardsBadge();
    } finally {
      if (btnClaim) { btnClaim.disabled = false; btnClaim.textContent = "Import & Claim"; }
      if (btnConn) btnConn.disabled = false;
    }
  }

  function closeMobileDrawer() {
    $("sidebar")?.classList.remove("mobile-open");
    $("drawer-backdrop")?.classList.remove("open");
    $("sb-fleet-box")?.classList.remove("open");
  }

  async function refreshFleetBadge() {
    try {
      const res = await fetch(API_BASE + "/v1/network/stats", { headers: spaceApiHeaders() });
      if (!res.ok) return;
      const st = await res.json();
      const el = $("sb-fleet-text");
      const usersEl = $("sb-users-text");
      const boxEl = $("sb-fleet-box");
      const liveCount = Number(st.active_nodes || 0);
      const liveMem = Number(st.total_memory_gb || 0);
      const onlineUsers = Number(st.online_users ?? st.users?.online ?? 0);
      const active24h = Number(st.active_users_24h ?? st.users?.active_24h ?? onlineUsers);
      const totalUsers = Number(st.total_users ?? st.users?.total ?? onlineUsers);
      const wallets = Number(st.connected_wallets ?? st.users?.connected_wallets ?? 0);
      if (el) {
        el.textContent = `${liveCount} Devices · ${liveMem.toFixed(1)} GB`;
      }
      if (usersEl) {
        usersEl.textContent = `${onlineUsers.toLocaleString("en-US")} Online · ${totalUsers.toLocaleString("en-US")} Users`;
      }
      if (boxEl) {
        const onlineGeo = Array.isArray(st.users?.online_countries) ? st.users.online_countries : [];
        const allGeo = Array.isArray(st.users?.countries) ? st.users.countries : [];
        const topStr = onlineGeo.length
          ? onlineGeo.slice(0, 4).map((c) => `${c.flag || ""} ${c.name} (${c.count})`).join(", ")
          : "no active regions yet";
        boxEl.title = `${onlineUsers} online (${topStr}) · ${active24h} active 24h · ${totalUsers} total users`;
        const popEl = $("sb-users-pop");
        if (popEl) {
          const renderRows = (arr, limit) => {
            if (!arr.length) return `<div class="sb-up-empty">No location data yet</div>`;
            return arr
              .slice(0, limit)
              .map(
                (c) =>
                  `<div class="sb-up-row"><span>${escHtml(c.flag || "🌐")} ${escHtml(c.name || c.code || "Unknown")}</span><strong>${Number(c.count || 0)}</strong></div>`
              )
              .join("");
          };
          popEl.innerHTML =
            `<div class="sb-up-head">Online Now (${onlineUsers})</div>` +
            renderRows(onlineGeo, 5) +
            `<div class="sb-up-sep"></div>` +
            `<div class="sb-up-head">All-Time Users (${totalUsers})</div>` +
            renderRows(allGeo, 5) +
            `<div class="sb-up-foot">${active24h} active 24h · ${wallets} wallets</div>`;
        }
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
      refreshRewardsBadge();
      syncCoreStatus();
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
    setInterval(() => {
      if (!document.hidden) {
        refreshFleetBadge();
        refreshRewardsBadge();
      }
    }, 15000);

    $("btn-core")?.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCorePopover();
    });
    $("core-popover")?.addEventListener("click", (e) => {
      e.stopPropagation();
    });
    document.addEventListener("click", (e) => {
      const path = typeof e.composedPath === "function" ? e.composedPath() : [];
      const pop = $("xray-popover");
      if (pop && pop.classList.contains("open") && !pop.contains(e.target) && !path.includes(pop) && !e.target.closest?.(".xtok")) {
        closeXrayPopover();
      }
      const cpop = $("core-popover");
      if (cpop && cpop.classList.contains("open") && !cpop.contains(e.target) && !path.includes(cpop) && !e.target.closest?.("#btn-core")) {
        closeCorePopover();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeXrayPopover();
        closeCorePopover();
      }
    });
    $("thread-scroll")?.addEventListener("scroll", () => {
      closeXrayPopover();
      closeCorePopover();
    }, { passive: true });

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
    $("btn-sb-import-wallet")?.addEventListener("click", openWalletImportModal);
    $("wim-close")?.addEventListener("click", closeWalletImportModal);
    $("wallet-import-modal")?.addEventListener("click", (e) => {
      if (e.target === $("wallet-import-modal")) closeWalletImportModal();
    });
    $("wim-btn-connect")?.addEventListener("click", async () => {
      const found = detectSolanaWalletProvider();
      if (!found) {
        setWimStatus(
          'No browser wallet extension found. Tap <strong>Open in Phantom App &#8599;</strong> above, install <a href="https://phantom.app" target="_blank" rel="noopener">Phantom</a> / <a href="https://solflare.com" target="_blank" rel="noopener">Solflare</a>, or paste any Solana address below.',
          "",
          true,
        );
        $("wim-addr-input")?.focus();
        return;
      }
      const btn = $("wim-btn-connect");
      try {
        if (btn) { btn.disabled = true; btn.textContent = `Connecting ${found.name}...`; }
        const resp = await found.p.connect();
        const pk = (resp?.publicKey || found.p.publicKey)?.toString?.() || "";
        if (!pk) throw new Error("No public key returned");
        await importAndClaimToWallet(pk);
      } catch (err) {
        setWimStatus(`Wallet connection cancelled or failed: ${err?.message || "rejected"}`, "err");
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = `Connect ${found.name} & Import`; }
      }
    });
    $("wim-btn-claim")?.addEventListener("click", () => {
      importAndClaimToWallet($("wim-addr-input")?.value || "");
    });
    $("wim-addr-input")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        importAndClaimToWallet($("wim-addr-input")?.value || "");
      }
    });
    document.querySelectorAll(".wim-copy-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const which = btn.getAttribute("data-copy-mint");
        const val = which === "tspace"
          ? (rewardsState.tspace_mint || TSPACE_MINT_ADDR)
          : (rewardsState.space_mint || SPACE_MINT_ADDR);
        navigator.clipboard?.writeText(val).then(() => {
          const prev = btn.textContent;
          btn.textContent = "Copied ✓";
          setTimeout(() => { btn.textContent = prev; }, 1400);
        });
      });
    });
    const fleetBox = $("sb-fleet-box");
    if (fleetBox) {
      fleetBox.addEventListener("click", (e) => {
        e.stopPropagation();
        fleetBox.classList.toggle("open");
      });
      document.addEventListener("click", (e) => {
        if (!fleetBox.contains(e.target)) fleetBox.classList.remove("open");
      });
    }

    const thinkSel = $("think-select");
    if (thinkSel) {
      try {
        const savedThink = localStorage.getItem("space_think_v1");
        if (savedThink && ["auto", "medium", "high", "none"].includes(savedThink)) {
          thinkSel.value = savedThink;
        }
      } catch (_e) {}
      thinkSel.addEventListener("change", () => {
        try { localStorage.setItem("space_think_v1", thinkSel.value); } catch (_e) {}
      });
    }

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
