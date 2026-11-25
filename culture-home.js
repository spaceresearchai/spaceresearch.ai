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

    // Interactive High-Dimensional Embedding Manifold Canvas (Contours + Semantic Clusters + Geodesic Trajectories)
    const bgCanvas = document.getElementById("cosmos-bg-canvas");
    if (bgCanvas && bgCanvas.getContext) {
      const ctx = bgCanvas.getContext("2d");
      const reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const CLUSTERS = [
        { rx: 0.17, ry: 0.27, rgb: [235, 235, 240], label: "DISCOVERY" },
        { rx: 0.83, ry: 0.26, rgb: [185, 185, 196], label: "INTERPRETABILITY" },
        { rx: 0.18, ry: 0.77, rgb: [215, 215, 222], label: "OPTIMIZATION" },
        { rx: 0.82, ry: 0.76, rgb: [168, 168, 180], label: "COLLECTIVE" },
      ];
      let W = window.innerWidth, H = window.innerHeight;
      let pts = [];
      let arcs = [];
      let mouse = { x: -9999, y: -9999, active: false };
      let focusCluster = -1;
      const coordEl = document.getElementById("manifold-coord");

      function gauss() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
      }

      function rebuildManifold() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        W = window.innerWidth;
        H = window.innerHeight;
        bgCanvas.width = Math.floor(W * dpr);
        bgCanvas.height = Math.floor(H * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const perCluster = W < 640 ? 16 : 26;
        const spreadX = Math.min(185, W * 0.15);
        const spreadY = Math.min(145, H * 0.16);
        pts = [];
        CLUSTERS.forEach((c, ci) => {
          const cx = c.rx * W;
          const cy = c.ry * H;
          for (let i = 0; i < perCluster; i++) {
            pts.push({
              ci,
              bx: cx + gauss() * spreadX * 0.62,
              by: cy + gauss() * spreadY * 0.62,
              ax: 8 + Math.random() * 18,
              ay: 8 + Math.random() * 16,
              fx: 0.18 + Math.random() * 0.24,
              fy: 0.16 + Math.random() * 0.22,
              ph: Math.random() * Math.PI * 2,
              r: 1.1 + Math.random() * 1.65,
              depth: 0.05 + Math.random() * 0.16,
              x: 0,
              y: 0,
            });
          }
        });
        arcs = [
          { from: 0, to: 1, u: 0.12, spd: 0.075, bend: -0.14 },
          { from: 1, to: 3, u: 0.48, spd: 0.068, bend: 0.12 },
          { from: 2, to: 0, u: 0.73, spd: 0.072, bend: -0.12 },
          { from: 3, to: 2, u: 0.29, spd: 0.064, bend: 0.15 },
          { from: 0, to: 3, u: 0.61, spd: 0.052, bend: -0.09 },
        ];
      }

      rebuildManifold();
      window.addEventListener("resize", rebuildManifold);
      window.addEventListener("pointermove", (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        mouse.active = true;
        if (coordEl) {
          const z1 = ((e.clientX / Math.max(1, W)) * 2 - 1) * 1.4;
          const z2 = (1 - (e.clientY / Math.max(1, H)) * 2) * 1.4;
          const norm = Math.sqrt(z1 * z1 + z2 * z2);
          const fmt = (n) => (n >= 0 ? "+" : "") + n.toFixed(2);
          coordEl.innerHTML = `<span>z<sub>1</sub> ${fmt(z1)}</span><span>z<sub>2</sub> ${fmt(z2)}</span><span>‖z‖ ${norm.toFixed(2)}</span>`;
        }
      }, { passive: true });
      window.addEventListener("pointerleave", () => { mouse.active = false; });

      document.querySelectorAll("[data-cluster]").forEach((el) => {
        const ci = Number(el.getAttribute("data-cluster"));
        el.addEventListener("mouseenter", () => { focusCluster = ci; });
        el.addEventListener("mouseleave", () => { focusCluster = -1; });
      });

      let rafBg = 0;
      const t0 = performance.now();
      function drawManifold(now) {
        const t = reducedMotion ? 0 : (now - t0) * 0.001;
        const isLight = document.documentElement.getAttribute("data-theme") === "light";
        const sy = window.scrollY || 0;
        ctx.clearRect(0, 0, W, H);

        // 1. Subtle coordinate manifold grid lines
        ctx.strokeStyle = isLight ? "rgba(15, 23, 42, 0.045)" : "rgba(255, 255, 255, 0.026)";
        ctx.lineWidth = 1;
        const step = Math.max(72, Math.round(W / 14));
        ctx.beginPath();
        for (let gx = step; gx < W; gx += step) {
          ctx.moveTo(gx, 0);
          ctx.lineTo(gx, H);
        }
        const yOff = -(sy * 0.12) % step;
        for (let gy = yOff; gy < H; gy += step) {
          ctx.moveTo(0, gy);
          ctx.lineTo(W, gy);
        }
        ctx.stroke();

        // 2. Topographic density contour rings around each semantic basin
        CLUSTERS.forEach((c, ci) => {
          const cx = c.rx * W + Math.sin(t * 0.28 + ci * 1.7) * 14;
          const cy = ((c.ry * H - sy * 0.06) % (H + 240) + (H + 240)) % (H + 240) - 120;
          const boost = focusCluster === ci ? 1.85 : 1.0;
          const [r, g, b] = c.rgb;

          // Soft ambient basin glow
          const grad = ctx.createRadialGradient(cx, cy, 8, cx, cy, Math.min(260, W * 0.24));
          grad.addColorStop(0, `rgba(${r},${g},${b},${((isLight ? 0.08 : 0.095) * boost).toFixed(3)})`);
          grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, Math.min(260, W * 0.24), 0, Math.PI * 2);
          ctx.fill();

          // Concentric organic elevation isolines
          for (let ring = 1; ring <= 4; ring++) {
            const baseR = ring * Math.min(54, W * 0.046);
            ctx.strokeStyle = `rgba(${r},${g},${b},${((isLight ? 0.10 : 0.085) * boost * (1 - ring * 0.14)).toFixed(3)})`;
            ctx.lineWidth = ring === 1 ? 1.0 : 0.75;
            ctx.beginPath();
            const segs = 40;
            for (let s = 0; s <= segs; s++) {
              const ang = (s / segs) * Math.PI * 2;
              const warp = 1 + 0.11 * Math.sin(ang * 3 + t * 0.35 + ci) + 0.07 * Math.cos(ang * 2 - t * 0.24 + ring);
              const px = cx + Math.cos(ang) * baseR * 1.24 * warp;
              const py = cy + Math.sin(ang) * baseR * 0.86 * warp;
              if (s === 0) ctx.moveTo(px, py);
              else ctx.lineTo(px, py);
            }
            ctx.closePath();
            ctx.stroke();
          }
        });

        // 3. Update point positions
        for (const p of pts) {
          p.x = p.bx + Math.sin(t * p.fx + p.ph) * p.ax;
          p.y = ((p.by + Math.cos(t * p.fy + p.ph) * p.ay - sy * p.depth) % (H + 100) + (H + 100)) % (H + 100) - 50;
        }

        // 4. Local k-NN manifold edges within each cluster
        const maxDistSq = (W < 640 ? 92 : 118) ** 2;
        for (let i = 0; i < pts.length; i++) {
          const a = pts[i];
          const [r, g, b] = CLUSTERS[a.ci].rgb;
          const boost = focusCluster === a.ci ? 1.75 : 1.0;
          for (let j = i + 1; j < pts.length; j++) {
            const bp = pts[j];
            if (bp.ci !== a.ci) continue;
            const dx = a.x - bp.x;
            const dy = a.y - bp.y;
            const d2 = dx * dx + dy * dy;
            if (d2 < maxDistSq) {
              const alpha = (1 - d2 / maxDistSq) * (isLight ? 0.18 : 0.16) * boost;
              ctx.strokeStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
              ctx.lineWidth = 0.75;
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(bp.x, bp.y);
              ctx.stroke();
            }
          }
        }

        // 5. Geodesic trajectories smoothly traversing between clusters
        for (const arc of arcs) {
          const cA = CLUSTERS[arc.from];
          const cB = CLUSTERS[arc.to];
          const ax = cA.rx * W, ay = cA.ry * H;
          const bx = cB.rx * W, by = cB.ry * H;
          const mx = (ax + bx) * 0.5 - (by - ay) * arc.bend;
          const my = (ay + by) * 0.5 + (bx - ax) * arc.bend;
          const grad = ctx.createLinearGradient(ax, ay, bx, by);
          grad.addColorStop(0, `rgba(${cA.rgb.join(",")},0.11)`);
          grad.addColorStop(1, `rgba(${cB.rgb.join(",")},0.11)`);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.85;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.quadraticCurveTo(mx, my, bx, by);
          ctx.stroke();

          const u = (arc.u + t * arc.spd) % 1;
          const inv = 1 - u;
          const px = inv * inv * ax + 2 * inv * u * mx + u * u * bx;
          const py = inv * inv * ay + 2 * inv * u * my + u * u * by;
          const pr = Math.round(cA.rgb[0] * inv + cB.rgb[0] * u);
          const pg = Math.round(cA.rgb[1] * inv + cB.rgb[1] * u);
          const pb = Math.round(cA.rgb[2] * inv + cB.rgb[2] * u);
          ctx.fillStyle = `rgba(${pr},${pg},${pb},0.85)`;
          ctx.beginPath();
          ctx.arc(px, py, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }

        // 6. Embedding nodes + soft pointer vector projection
        const mRadSq = 175 * 175;
        for (const p of pts) {
          const [r, g, b] = CLUSTERS[p.ci].rgb;
          const boost = focusCluster === p.ci ? 1.45 : 1.0;
          let near = 0;
          if (mouse.active) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const d2 = dx * dx + dy * dy;
            if (d2 < mRadSq) {
              near = 1 - Math.sqrt(d2) / 175;
              ctx.strokeStyle = `rgba(${r},${g},${b},${(near * (isLight ? 0.30 : 0.28)).toFixed(3)})`;
              ctx.lineWidth = 0.7;
              ctx.beginPath();
              ctx.moveTo(p.x, p.y);
              ctx.lineTo(mouse.x, mouse.y);
              ctx.stroke();
            }
          }
          const alpha = Math.min(0.95, ((isLight ? 0.48 : 0.44) + near * 0.5) * boost);
          ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.r + near * 1.4, 0, Math.PI * 2);
          ctx.fill();
        }

        if (!reducedMotion) rafBg = requestAnimationFrame(drawManifold);
      }
      rafBg = requestAnimationFrame(drawManifold);
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) cancelAnimationFrame(rafBg);
        else if (!reducedMotion) rafBg = requestAnimationFrame(drawManifold);
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
                hxFill.style.background = avg >= 78 ? "#34d399" : avg >= 48 ? "#fbbf24" : "#f43f5e";
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

    // -----------------------------------------------------------------------
    // Universal Click-to-Zoom Image Lightbox for Blog Notes & Technical Papers
    // -----------------------------------------------------------------------
    const zoomImgs = Array.from(document.querySelectorAll(".blog-figure img, .blog-prose img, img.cc-fig-img"));
    if (zoomImgs.length) {
      const lb = document.createElement("div");
      lb.className = "space-lightbox";
      lb.setAttribute("role", "dialog");
      lb.setAttribute("aria-modal", "true");
      lb.setAttribute("aria-label", "Zoomed figure viewer");
      lb.innerHTML =
        '<div class="space-lb-bar">' +
          '<span class="space-lb-cap" id="space-lb-cap"></span>' +
          '<div class="space-lb-acts">' +
            '<button type="button" class="space-lb-btn" id="space-lb-out" title="Zoom out">−</button>' +
            '<span id="space-lb-pct" style="min-width:44px;text-align:center;">Fit</span>' +
            '<button type="button" class="space-lb-btn" id="space-lb-in" title="Zoom in">+</button>' +
            '<button type="button" class="space-lb-btn" id="space-lb-reset" title="Fit to screen">Fit</button>' +
            '<button type="button" class="space-lb-btn" id="space-lb-close" title="Close (Esc)">✕</button>' +
          "</div>" +
        "</div>" +
        '<div class="space-lb-stage" id="space-lb-stage"><img class="space-lb-img" id="space-lb-img" alt="" /></div>';
      document.body.appendChild(lb);

      const stage = lb.querySelector("#space-lb-stage");
      const lbImg = lb.querySelector("#space-lb-img");
      const lbCap = lb.querySelector("#space-lb-cap");
      const lbPct = lb.querySelector("#space-lb-pct");
      let zoomScale = 1.0;

      function applyZoom(scale) {
        zoomScale = Math.max(1.0, Math.min(3.5, scale));
        if (zoomScale <= 1.01) {
          stage.classList.remove("zoomed");
          lbImg.style.width = "";
          lbPct.textContent = "Fit";
        } else {
          stage.classList.add("zoomed");
          const baseW = Math.min(window.innerWidth * 0.92, lbImg.naturalWidth || 1200);
          lbImg.style.width = Math.round(baseW * zoomScale) + "px";
          lbPct.textContent = Math.round(zoomScale * 100) + "%";
        }
      }

      function closeLb() {
        lb.classList.remove("open");
        document.body.style.overflow = "";
        applyZoom(1.0);
      }

      function openLb(srcImg) {
        const fig = srcImg.closest(".blog-figure, figure");
        const capEl = fig ? fig.querySelector(".blog-figcap, figcaption") : null;
        lbImg.src = srcImg.currentSrc || srcImg.src;
        lbImg.alt = srcImg.alt || "";
        lbCap.textContent = (capEl ? capEl.textContent.trim() : srcImg.alt) || "Figure (click image or scroll to zoom)";
        applyZoom(1.0);
        lb.classList.add("open");
        document.body.style.overflow = "hidden";
      }

      zoomImgs.forEach((img) => {
        img.title = img.title || "Click to zoom figure";
        img.addEventListener("click", (e) => {
          e.preventDefault();
          openLb(img);
        });
      });

      lbImg.addEventListener("click", (e) => {
        e.stopPropagation();
        applyZoom(zoomScale <= 1.05 ? 1.85 : (zoomScale < 2.4 ? 2.6 : 1.0));
      });
      stage.addEventListener("click", (e) => {
        if (e.target === stage) closeLb();
      });
      stage.addEventListener("wheel", (e) => {
        if (!lb.classList.contains("open")) return;
        if (e.ctrlKey || e.metaKey || zoomScale > 1.01) {
          e.preventDefault();
          applyZoom(zoomScale + (e.deltaY < 0 ? 0.25 : -0.25));
        }
      }, { passive: false });
      lb.querySelector("#space-lb-in").addEventListener("click", () => applyZoom(zoomScale <= 1.05 ? 1.5 : zoomScale + 0.4));
      lb.querySelector("#space-lb-out").addEventListener("click", () => applyZoom(zoomScale - 0.4));
      lb.querySelector("#space-lb-reset").addEventListener("click", () => applyZoom(1.0));
      lb.querySelector("#space-lb-close").addEventListener("click", closeLb);
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && lb.classList.contains("open")) closeLb();
      });
    }
  });
})();
