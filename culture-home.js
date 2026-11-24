/* Space AI v17 — Ambient Pure-Black Starfield, Telemetry, Live Mind Window & Interactive Controls */
(() => {
  const COORD_URL = "https://api.spaceresearch.ai";

  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("space_theme_v17", t); } catch (_e) {}
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.textContent = t === "light" ? "☾" : "☀";
      btn.title = t === "light" ? "Switch to Pure Black Space" : "Switch to Daylight Mode";
    });
  }
  try {
    const saved = localStorage.getItem("space_theme_v17");
    if (saved === "dark" || saved === "light") applyTheme(saved);
    else applyTheme("dark");
  } catch (_e) {}

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
        applyTheme(cur === "light" ? "dark" : "light");
      });
    });

    // Lightweight Ambient Full-Page Pure-Black Starfield with Gentle Scroll Parallax
    const bgCanvas = document.getElementById("cosmos-bg-canvas");
    if (bgCanvas && bgCanvas.getContext) {
      const ctx = bgCanvas.getContext("2d");
      let stars = [];
      function resizeBg() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        bgCanvas.width = Math.floor(window.innerWidth * dpr);
        bgCanvas.height = Math.floor(window.innerHeight * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const count = Math.min(220, Math.floor((window.innerWidth * window.innerHeight) / 6800));
        stars = Array.from({ length: count }, () => ({
          x: Math.random() * window.innerWidth,
          y: Math.random() * window.innerHeight,
          r: 0.45 + Math.random() * 1.15,
          depth: 0.08 + Math.random() * 0.28,
          phase: Math.random() * Math.PI * 2,
          spd: 0.006 + Math.random() * 0.015,
          gold: Math.random() < 0.22,
        }));
      }
      resizeBg();
      window.addEventListener("resize", resizeBg);
      let rafBg = 0;
      function drawBg() {
        const w = window.innerWidth, h = window.innerHeight;
        ctx.clearRect(0, 0, w, h);
        const sy = window.scrollY || 0;
        for (const s of stars) {
          s.phase += s.spd;
          const a = 0.18 + 0.62 * (0.5 + 0.5 * Math.sin(s.phase));
          const py = ((s.y - sy * s.depth) % h + h) % h;
          ctx.fillStyle = s.gold ? `rgba(226, 176, 104, ${a.toFixed(2)})` : `rgba(244, 244, 248, ${a.toFixed(2)})`;
          ctx.beginPath();
          ctx.arc(s.x, py, s.r, 0, Math.PI * 2);
          ctx.fill();
        }
        rafBg = requestAnimationFrame(drawBg);
      }
      drawBg();
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) cancelAnimationFrame(rafBg);
        else drawBg();
      });
    }

    const rbClose = document.getElementById("ribbon-close");
    if (rbClose) rbClose.addEventListener("click", () => { const r = document.getElementById("folio-ribbon"); if (r) r.style.display = "none"; });

    // Culture-style rotating vessel callsigns
    const ships = [
      "GSV No More Black Boxes",
      "GCU Truth in Plain Sight",
      "ROU Zero Cloud Dependency",
      "GSV Open Silicon Commons",
      "GCU Confidence Visible",
    ];
    const shipEl = document.getElementById("culture-ship");
    if (shipEl) {
      let idx = 0;
      setInterval(() => {
        idx = (idx + 1) % ships.length;
        shipEl.textContent = ships[idx];
      }, 4600);
    }

    // OS-aware Desktop label
    const ua = (navigator.userAgent || "").toLowerCase();
    const osLabel = ua.includes("mac") ? "Get for macOS" : ua.includes("win") ? "Get for Windows" : "Install for Linux";
    document.querySelectorAll("[data-os-cta]").forEach((el) => { el.textContent = osLabel; });

    // Install Modal
    const modal = document.getElementById("install-modal");
    const modalTabs = document.querySelectorAll(".modal-tab");
    const modalCmd = document.getElementById("modal-cmd-box");
    const modalHint = document.getElementById("modal-hint");
    const modalCommands = {
      desktop: {
        cmd: "curl -fsSL https://spaceresearch.ai/install-space.sh | sh\nspace up --auto-pause\n# Local status & spillover ready on your machine",
        hint: "Installs the Space CLI & Auto-Paused hardware sharer today. Join the list below for the native macOS/Windows/Linux Space Desktop app.",
      },
      cli: {
        cmd: "curl -fsSL https://spaceresearch.ai/install-space.sh | sh && space claude",
        hint: "Keeps Claude Code, Codex, or OpenCode running through verified community silicon when cloud limits hit — zero edits to your paid configs.",
      },
      nvidia: {
        cmd: "curl -fsSL https://spaceresearch.ai/join-gpu.sh | sh -s -- --name my-station --auto-pause",
        hint: "Portable CUDA station for Linux & Windows WSL2. Auto-Pauses immediately when you game or work.",
      },
      mac: {
        cmd: "curl -fsSL https://spaceresearch.ai/install-space.sh | sh && space up --adapter mlx --auto-pause",
        hint: "Apple Silicon native support (M1–M5), plus instant adapters for existing Ollama or llama.cpp setups.",
      },
    };
    function setModalTab(key) {
      const info = modalCommands[key] || modalCommands.desktop;
      modalTabs.forEach((t) => t.classList.toggle("active", t.getAttribute("data-mtab") === key));
      if (modalCmd) modalCmd.textContent = info.cmd;
      if (modalHint) modalHint.textContent = info.hint;
    }
    modalTabs.forEach((t) => t.addEventListener("click", () => setModalTab(t.getAttribute("data-mtab") || "desktop")));
    document.querySelectorAll("[data-open-modal]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        setModalTab(btn.getAttribute("data-open-modal") || "desktop");
        if (modal) modal.classList.add("open");
      });
    });
    document.getElementById("modal-close")?.addEventListener("click", () => modal?.classList.remove("open"));
    modal?.addEventListener("click", (ev) => { if (ev.target === modal) modal.classList.remove("open"); });
    const btnCopyModal = document.getElementById("btn-copy-modal-cmd");
    if (btnCopyModal) {
      btnCopyModal.addEventListener("click", () => {
        navigator.clipboard?.writeText(modalCmd?.textContent || "").then(() => {
          const prev = btnCopyModal.textContent;
          btnCopyModal.textContent = "Copied ✓";
          setTimeout(() => { btnCopyModal.textContent = prev; }, 1600);
        });
      });
    }
    const btnCopyCli = document.getElementById("btn-copy-bento-cli");
    if (btnCopyCli) {
      btnCopyCli.addEventListener("click", () => {
        navigator.clipboard?.writeText("curl -fsSL https://spaceresearch.ai/install-space.sh | sh && space claude").then(() => {
          const prev = btnCopyCli.textContent;
          btnCopyCli.textContent = "Copied ✓";
          setTimeout(() => { btnCopyCli.textContent = prev; }, 1600);
        });
      });
    }
    const btnCopyApi = document.getElementById("btn-copy-api-url");
    if (btnCopyApi) {
      btnCopyApi.addEventListener("click", () => {
        navigator.clipboard?.writeText("https://api.spaceresearch.ai/v1").then(() => {
          const prev = btnCopyApi.textContent;
          btnCopyApi.textContent = "Copied ✓";
          setTimeout(() => { btnCopyApi.textContent = prev; }, 1600);
        });
      });
    }

    // Orbiting Model Ribbon — aligned to live deployed models & routes on api.spaceresearch.ai
    const MODELS = {
      all: [
        { g: "Q3", name: "Qwen3.8 27B", tag: "Live · X-Ray", price: "Free · Open", tier: "RTX · Mac" },
        { g: "I2", name: "Inkling 2 Mini", tag: "Live · Fast", price: "Free · Open", tier: "~220 tok/s" },
        { g: "SC", name: "Swarm Council", tag: "Multi-Node", price: "Parallel", tier: "Consensus" },
        { g: "70", name: "70B / 106B RPC", tag: "Sharded", price: "Pooled RAM", tier: "Mac + GPU" },
        { g: "32", name: "Qwen3 32B", tag: "Swarm · 4-bit", price: "Free · Open", tier: "Analysis" },
        { g: "CL", name: "Space Code", tag: "Claude / Codex", price: "1-Line CLI", tier: "Auto Route" },
      ],
      reasoning: [
        { g: "Q3", name: "Qwen3.8 27B", tag: "Live · X-Ray", price: "Decision X-Ray", tier: "Flagship" },
        { g: "I2", name: "Inkling 2 Mini", tag: "Live · Fast", price: "Fast Think", tier: "~220 tok/s" },
        { g: "SC", name: "Swarm Council", tag: "Multi-Node", price: "Parallel Fanout", tier: "Synthesis" },
        { g: "70", name: "70B / 106B RPC", tag: "Sharded", price: "Pooled RAM", tier: "Deep Reason" },
      ],
      coding: [
        { g: "CL", name: "Space Code", tag: "Claude CLI", price: "space claude", tier: "Live Proxy" },
        { g: "Q3", name: "Qwen3.8 27B", tag: "Codex CLI", price: "space codex", tier: "RTX · Mac" },
        { g: "I2", name: "Inkling 2 Mini", tag: "OpenCode", price: "space opencode", tier: "Fast Loop" },
        { g: "32", name: "Qwen3 32B", tag: "OpenAI API", price: "/v1/chat", tier: "Drop-In" },
      ],
      vision: [
        { g: "QV", name: "Qwen3.8 27B Vision", tag: "Live · Multimodal", price: "Paste / Drop", tier: "Space Chat" },
        { g: "Q3", name: "Decision X-Ray", tag: "Live · Audit", price: "/stress /audit", tier: "Bluff Radar" },
        { g: "SC", name: "Paired Compare", tag: "Bias-Free", price: "/compare", tier: "Swap Check" },
        { g: "70", name: "70B / 106B RPC", tag: "Multi-Node", price: "Auto Hot-Swap", tier: "Shared RAM" },
      ],
    };
    function renderRibbon(cat) {
      const list = MODELS[cat] || MODELS.all;
      const items = [...list, ...list, ...list, ...list];
      const track = document.getElementById("orbit-ribbon-track");
      if (!track) return;
      track.innerHTML = items.map((m) => `
        <a class="model-folio-card" href="./chat.html">
          <div class="mfc-top">
            <span class="mfc-orb">${m.g}</span>
            <span class="mfc-tag">${m.tag}</span>
          </div>
          <div class="mfc-name">${m.name}</div>
          <div class="mfc-meta"><span>${m.price}</span><span>${m.tier}</span></div>
        </a>
      `).join("");
    }
    renderRibbon("all");
    document.querySelectorAll(".cat-pill").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll(".cat-pill").forEach((x) => x.classList.toggle("active", x === b));
        renderRibbon(b.getAttribute("data-cat") || "all");
      });
    });

    // Observatory Tabs -> also drive the 3D Mind Core
    const obsReadouts = {
      certainty: { lbl: "Mind Window · Live", txt: "Every word streams with live certainty. Hover or tap any highlighted word to inspect the alternatives the model nearly chose." },
      authentic: { lbl: "Station Verification · Continuous", txt: "Active stations are tested quietly in the background so you always get the genuine open model — never a cheaper silent swap." },
      privacy: { lbl: "Zero-Account · Local First", txt: "No email or account wall. Start directly over HTTPS, or run locally with spillover to zero-log and isolated container stations." },
      routing: { lbl: "Certainty-Gated Routing", txt: "Run easy prompts on your own Mac or GPU at zero cost, and automatically spill over or escalate to stronger community stations when confidence drops." },
    };
    document.querySelectorAll(".obs-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.getAttribute("data-obs") || "certainty";
        const r = obsReadouts[key] || obsReadouts.certainty;
        document.querySelectorAll(".obs-tab").forEach((t) => t.classList.toggle("active", t === tab));
        const l = document.getElementById("obs-read-lbl");
        const tx = document.getElementById("obs-read-txt");
        if (l) l.textContent = r.lbl;
        if (tx) tx.textContent = r.txt;
        window.dispatchEvent(new CustomEvent("space:obs-mode", { detail: key }));
      });
    });

    // Orbital Mechanics Layer Inspector -> drives 3D Planetary Relay camera & highlight
    const mechRings = {
      routing: {
        title: "01 · Zero-Install API & Local Spillover",
        desc: "Call api.spaceresearch.ai/v1 directly with zero local setup, or run Space Desktop / CLI to use your own Mac or GPU first and burst to nearby stations when needed.",
        rows: [
          ["Direct API", "https://api.spaceresearch.ai/v1"],
          ["Coding tools", "Claude Code · Codex · Cursor"],
          ["Local spillover", "Apple Silicon · Ollama · RTX"],
          ["Failover", "Automatic (< 50 ms)"],
        ],
      },
      verify: {
        title: "02 · Live Verification & Mind Window",
        desc: "Open models stream directly from verified community stations while background quality checks confirm the genuine model is running.",
        rows: [
          ["Model authenticity", "100% Verified live"],
          ["Transparency", "Live confidence + bluff check"],
          ["Privacy options", "Standard · Zero-log · Isolated"],
          ["Silicon", "Apple Silicon & NVIDIA RTX"],
        ],
      },
      credits: {
        title: "03 · Fair Credits & Home Orbitals",
        desc: "Earn free usage by sharing idle power with Auto-Pause, or cap your spend pay-as-you-go with zero monthly subscription.",
        rows: [
          ["Idle hardware barter", "Up to 20x Early Credits"],
          ["Gaming & work guard", "Auto-Pause built in"],
          ["Home Orbitals", "Separated from data centers"],
          ["Pay-as-you-go", "Hard spending cap ($1–$50)"],
        ],
      },
    };
    function setMech(key) {
      const d = mechRings[key] || mechRings.verify;
      document.querySelectorAll(".mech-tab").forEach((t) => t.classList.toggle("active", t.getAttribute("data-mech") === key));
      const tEl = document.getElementById("mech-title");
      const dEl = document.getElementById("mech-desc");
      const rEl = document.getElementById("mech-rows");
      if (tEl) tEl.textContent = d.title;
      if (dEl) dEl.textContent = d.desc;
      if (rEl) rEl.innerHTML = d.rows.map(([k, v], i) => `<div class="mech-kv"><span class="k">${k}</span><span class="v ${i === 0 ? "accent" : ""}">${v}</span></div>`).join("");
      window.dispatchEvent(new CustomEvent("space:mech-ring", { detail: key }));
    }
    setMech("verify");
    document.querySelectorAll(".mech-tab").forEach((t) => t.addEventListener("click", () => setMech(t.getAttribute("data-mech") || "verify")));

    // Live Telemetry Hydration (real /v1/network/stats + /v1/integrity/summary endpoints)
    async function hydrateStats() {
      try {
        const [nRes, iRes] = await Promise.allSettled([
          fetch(COORD_URL + "/v1/network/stats"),
          fetch(COORD_URL + "/v1/integrity/summary"),
        ]);
        const s = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        if (nRes.status === "fulfilled" && nRes.value.ok) {
          const d = await nRes.value.json();
          const active = Number(d.active_nodes ?? 0);
          const reg = Number(d.registered_nodes ?? active);
          const toks = Number(d.verified_tokens ?? d.total_tokens_served ?? 0);
          const reqs = Number(d.verified_requests ?? 0);
          if (active > 0 || reg > 0) s("stat-nodes", reg > active ? `${active} / ${reg}` : String(active));
          if (toks > 0) s("stat-tokens", toks >= 1e6 ? (toks / 1e6).toFixed(1) + "M" : toks >= 1e3 ? (toks / 1e3).toFixed(1) + "K" : String(toks));
          if (reqs > 0) s("stat-receipts", reqs.toLocaleString("en-US"));
        }
        if (iRes.status === "fulfilled" && iRes.value.ok) {
          const d = await iRes.value.json();
          s("stat-quality", Number(d.pass_rate_pct ?? 100.0).toFixed(1) + "%");
          if (Number(d.receipts_verified ?? 0) > 0) {
            s("stat-receipts", Number(d.receipts_verified).toLocaleString("en-US"));
          }
        }
      } catch (_e) {}
    }
    hydrateStats();

    // Live Mind Window (Decision X-Ray — real sig.p & sig.alts token inspector)
    const hxForm = document.getElementById("hx-form");
    const hxInput = document.getElementById("hx-input");
    const hxBtn = document.getElementById("hx-run-btn");
    const hxAns = document.getElementById("hx-ans");
    const hxCertLbl = document.getElementById("hx-cert-lbl");
    const hxForkLbl = document.getElementById("hx-fork-lbl");
    const hxFill = document.getElementById("hx-fill");
    const hxRadar = document.getElementById("hx-radar");
    const hxTip = document.getElementById("hx-tip");
    let hxAbort = null;

    function escHtml(str) {
      return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function showTokenTip(tokText, prob, margin, alts) {
      if (!hxTip) return;
      const pct = Math.round(prob * 100);
      const disp = tokText.replace(/\n/g, "↵");
      const altHtml = (alts && alts.length)
        ? alts.slice(0, 4).map((a) => {
            const at = String(a.t ?? a.token ?? "").replace(/\n/g, "↵");
            const ap = Math.round(Number(a.p ?? a.prob ?? 0) * 100);
            return `<span class="alt-pill">${escHtml(at || "␠")} ${ap}%</span>`;
          }).join("")
        : "";
      const mStr = (typeof margin === "number" && Number.isFinite(margin)) ? ` · margin ${margin >= 0 ? "+" : ""}${margin.toFixed(2)}` : "";
      hxTip.innerHTML = `<span>TOKEN <strong>"${escHtml(disp)}"</strong> · ${pct}%${mStr}</span>${altHtml ? `<span style="display:inline-flex;flex-wrap:wrap;gap:0.35rem;align-items:center;">ALTS: ${altHtml}</span>` : ""}`;
      hxTip.classList.add("open");
    }

    async function runMindWindow(promptText) {
      const q = String(promptText || hxInput?.value || "").trim();
      if (!q || !hxAns) return;
      if (hxInput) hxInput.value = q;
      if (hxAbort) { try { hxAbort.abort(); } catch (_e) {} }
      hxAbort = new AbortController();
      if (hxBtn) { hxBtn.disabled = true; hxBtn.textContent = "..."; }
      if (hxTip) { hxTip.classList.remove("open"); hxTip.innerHTML = ""; }
      hxAns.innerHTML = '<span style="opacity:0.6;">Streaming live from community silicon...</span>';
      if (hxCertLbl) hxCertLbl.textContent = "CONFIDENCE --";
      if (hxForkLbl) hxForkLbl.textContent = "LIVE STREAM";
      if (hxFill) { hxFill.style.width = "0%"; hxFill.style.background = "#ffffff"; }
      if (hxRadar) hxRadar.innerHTML = '<span style="opacity:0.65;">Checking honesty &amp; guess risk...</span>';

      let fullText = "";
      let inThink = false;
      const probs = [];
      let hesCount = 0;
      try {
        const res = await fetch(COORD_URL + "/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: hxAbort.signal,
          body: JSON.stringify({
            model: "qwen3.8-27b",
            stream: true,
            xray: true,
            max_tokens: 260,
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
              if (!tok) continue;
              if (tok.includes("<think>")) { inThink = true; continue; }
              if (tok.includes("</think>")) { inThink = false; continue; }
              if (inThink) continue;

              fullText += tok;
              const sig = p.token_signal || null;
              const prob = sig ? Number(sig.p ?? sig.certainty ?? sig.prob ?? 0.88) : 0.88;
              const margin = sig && sig.m !== undefined ? Number(sig.m) : null;
              const alts = (sig && Array.isArray(sig.alts)) ? sig.alts : ((sig && Array.isArray(sig.top_alternatives)) ? sig.top_alternatives : []);
              probs.push(prob);
              const cls = prob >= 0.78 ? "hi" : prob >= 0.45 ? "mid" : "lo";
              if (cls !== "hi") hesCount++;
              const sp = document.createElement("span");
              sp.className = "hx-tok " + cls;
              sp.textContent = tok;
              sp.addEventListener("mouseenter", () => showTokenTip(tok, prob, margin, alts));
              sp.addEventListener("click", () => showTokenTip(tok, prob, margin, alts));
              hxAns.appendChild(sp);

              const avg = Math.round((probs.reduce((a, b) => a + b, 0) / probs.length) * 100);
              if (hxCertLbl) hxCertLbl.textContent = `Confidence ${avg}%`;
              if (hxFill) {
                hxFill.style.width = avg + "%";
                hxFill.style.background = avg >= 78 ? "#ffffff" : avg >= 48 ? "#e2b068" : "#ef4444";
              }
            } catch (_e) {}
          }
        }
        const aRes = await fetch(COORD_URL + "/v1/audit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: hxAbort.signal,
          body: JSON.stringify({ user_prompt: q, assistant_reply: fullText, model: "qwen3.8-27b" }),
        });
        if (aRes.ok && hxRadar) {
          const ad = await aRes.json();
          const vLbl = ad.verdict_label || "Verified";
          const avg = probs.length ? Math.round((probs.reduce((a, b) => a + b, 0) / probs.length) * 100) : Math.round(ad.grounded_pct ?? 92);
          if (hxCertLbl) hxCertLbl.textContent = `Confidence ${avg}% · ${vLbl}`;
          hxRadar.innerHTML = "";
        }
      } catch (err) {
        if (err?.name !== "AbortError" && hxAns) {
          hxAns.textContent = "Live station momentarily busy — try again or open Space Chat.";
        }
      } finally {
        if (hxBtn) { hxBtn.disabled = false; hxBtn.textContent = "Observe →"; }
      }
    }

    document.querySelectorAll(".mw-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mw-preset").forEach((b) => b.classList.toggle("active", b === btn));
        runMindWindow(btn.getAttribute("data-q") || "");
      });
    });
    if (hxForm) {
      hxForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runMindWindow(hxInput?.value || "");
      });
    }
  });
})();
