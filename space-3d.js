/* Space AI v20 — Studio-Grade Pure-Black Three.js 3D Engine (PMREM Optics, GLSL Fresnel Shaders & Comet Relays) */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";

const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  return renderer;
}

/* Procedural Studio PMREM Environment Map — gives titanium, brushed gold & optical glass real reflections */
function makeStudioEnv(renderer) {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 256;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#030304";
  ctx.fillRect(0, 0, 512, 256);

  // Soft warm gold studio key reflection strip
  const g1 = ctx.createRadialGradient(130, 88, 6, 130, 88, 120);
  g1.addColorStop(0, "rgba(255, 224, 172, 1.0)");
  g1.addColorStop(0.35, "rgba(228, 228, 231, 0.55)");
  g1.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = g1;
  ctx.fillRect(0, 0, 512, 256);

  // Crisp neutral white starlight rim reflection
  const g2 = ctx.createRadialGradient(385, 72, 4, 385, 72, 95);
  g2.addColorStop(0, "rgba(255, 255, 255, 0.95)");
  g2.addColorStop(0.4, "rgba(210, 214, 224, 0.35)");
  g2.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = g2;
  ctx.fillRect(0, 0, 512, 256);

  // Subtle horizontal specular horizon band for metallic rings
  const band = ctx.createLinearGradient(0, 118, 0, 144);
  band.addColorStop(0, "rgba(255, 255, 255, 0)");
  band.addColorStop(0.5, "rgba(240, 220, 190, 0.28)");
  band.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = band;
  ctx.fillRect(0, 118, 512, 26);

  const srcTex = new THREE.CanvasTexture(c);
  srcTex.mapping = THREE.EquirectangularReflectionMapping;
  srcTex.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const env = pmrem.fromEquirectangular(srcTex).texture;
  srcTex.dispose();
  pmrem.dispose();
  return env;
}

/* Soft Circular Airy-Disk Star Sprite (zero hard square pixels) */
let _starTex = null;
function getStarSprite() {
  if (_starTex) return _starTex;
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 64;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0.0, "rgba(255, 255, 255, 1.0)");
  g.addColorStop(0.18, "rgba(255, 255, 255, 0.88)");
  g.addColorStop(0.45, "rgba(255, 245, 228, 0.22)");
  g.addColorStop(1.0, "rgba(0, 0, 0, 0.0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  _starTex = new THREE.CanvasTexture(c);
  return _starTex;
}

/* Soft Volumetric Bloom Halo Sprite */
let _glowTex = null;
function getGlowSprite() {
  if (_glowTex) return _glowTex;
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 128;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0.0, "rgba(255, 250, 240, 0.95)");
  g.addColorStop(0.22, "rgba(255, 232, 192, 0.45)");
  g.addColorStop(0.55, "rgba(228, 228, 231, 0.11)");
  g.addColorStop(1.0, "rgba(0, 0, 0, 0.0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  _glowTex = new THREE.CanvasTexture(c);
  return _glowTex;
}

function attachVisibilityLoop(targetEl, renderStep) {
  let active = true;
  let rafId = 0;
  let lastT = performance.now();

  function frame(now) {
    if (!active) return;
    const dt = Math.min(0.05, Math.max(0.001, (now - lastT) / 1000));
    lastT = now;
    renderStep(dt, now * 0.001);
    rafId = requestAnimationFrame(frame);
  }
  function start() {
    if (active && rafId) return;
    active = true;
    lastT = performance.now();
    rafId = requestAnimationFrame(frame);
  }
  function stop() {
    active = false;
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  }
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !document.hidden) start();
        else stop();
      }
    }, { threshold: 0.05 });
    io.observe(targetEl);
  } else {
    start();
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else start();
  });
  return { start, stop };
}

/* High-Resolution Procedural Dark Banded Planet Albedo + Bump Map */
function makePlanetMaps(warmth = 1) {
  const w = 1024, h = 512;
  const colCanvas = document.createElement("canvas");
  colCanvas.width = w;
  colCanvas.height = h;
  const ctx = colCanvas.getContext("2d");

  const bumpCanvas = document.createElement("canvas");
  bumpCanvas.width = w;
  bumpCanvas.height = h;
  const bctx = bumpCanvas.getContext("2d");

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0.0, "#0b0b0d");
  grad.addColorStop(0.20, "#1d1b18");
  grad.addColorStop(0.42, warmth > 0.5 ? "#352b20" : "#1e1f23");
  grad.addColorStop(0.58, "#151413");
  grad.addColorStop(0.76, warmth > 0.5 ? "#29221a" : "#1b1c20");
  grad.addColorStop(1.0, "#09090a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  bctx.fillStyle = "#808080";
  bctx.fillRect(0, 0, w, h);

  for (let i = 0; i < 120; i++) {
    const y = Math.random() * h;
    const bh = 1 + Math.random() * 9;
    const alpha = 0.035 + Math.random() * 0.095;
    ctx.fillStyle = i % 3 === 0
      ? `rgba(228, 228, 231, ${alpha.toFixed(3)})`
      : `rgba(230, 230, 235, ${(alpha * 0.72).toFixed(3)})`;
    ctx.fillRect(0, y, w, bh);

    const bv = Math.round(115 + Math.random() * 35);
    bctx.fillStyle = `rgba(${bv},${bv},${bv},0.35)`;
    bctx.fillRect(0, y, w, bh);
  }

  for (let i = 0; i < 55; i++) {
    const x = Math.random() * w;
    const y = 36 + Math.random() * (h - 72);
    const rx = 14 + Math.random() * 60;
    const ry = 3 + Math.random() * 12;
    ctx.fillStyle = `rgba(0, 0, 0, ${(0.12 + Math.random() * 0.16).toFixed(2)})`;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();

    bctx.fillStyle = "rgba(70,70,70,0.4)";
    bctx.beginPath();
    bctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    bctx.fill();
  }

  const map = new THREE.CanvasTexture(colCanvas);
  map.wrapS = THREE.RepeatWrapping;
  map.colorSpace = THREE.SRGBColorSpace;
  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = THREE.RepeatWrapping;
  return { map, bumpMap };
}

/* Custom GLSL Fresnel Atmospheric Horizon Glow */
function makeAtmosphereHalo(radius, colorHex = 0xd4d4d8, intensity = 0.72) {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(colorHex) },
      uIntensity: { value: intensity },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewPos;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vViewPos = -mv.xyz;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uIntensity;
      varying vec3 vNormal;
      varying vec3 vViewPos;
      void main() {
        vec3 n = normalize(vNormal);
        vec3 v = normalize(vViewPos);
        float rim = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 3.4);
        gl_FragColor = vec4(uColor, rim * uIntensity);
      }
    `,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.FrontSide,
  });
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 56, 56), mat);
}

/* Multi-Band Striated Planetary Ring Texture */
function makeRingTexture() {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 32;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, 1024, 32);
  for (let x = 0; x < 1024; x++) {
    const u = x / 1024;
    const edge = Math.pow(Math.sin(u * Math.PI), 0.8);
    const fine = 0.5 + 0.5 * Math.sin(u * 92) * Math.cos(u * 31);
    const gap = (u > 0.53 && u < 0.585) || (u > 0.27 && u < 0.295) || (u > 0.79 && u < 0.808) ? 0.05 : 1.0;
    const a = Math.max(0, Math.min(0.76, edge * (0.18 + 0.54 * fine) * gap));
    const gold = u > 0.42 ? 1 : 0.65;
    ctx.fillStyle = `rgba(${Math.round(212 * gold + 36)}, ${Math.round(182 * gold + 36)}, ${Math.round(138 * gold + 36)}, ${a.toFixed(3)})`;
    ctx.fillRect(x, 0, 1, 32);
  }
  return new THREE.CanvasTexture(c);
}

/* Crisp Airy-Disk Starfield Points (zero square pixels) */
function makeStarfield(count, radiusMin, radiusMax, size) {
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const cWhite = new THREE.Color(0xffffff);
  const cSilver = new THREE.Color(0xd4d4d8);
  const cGold = new THREE.Color(0xd4d4d8);
  for (let i = 0; i < count; i++) {
    const r = radiusMin + Math.random() * (radiusMax - radiusMin);
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.72;
    pos[i * 3 + 2] = r * Math.cos(phi) - 8;
    const pick = Math.random();
    const c = pick < 0.66 ? cWhite : pick < 0.86 ? cSilver : cGold;
    const dim = 0.48 + Math.random() * 0.52;
    col[i * 3] = c.r * dim;
    col[i * 3 + 1] = c.g * dim;
    col[i * 3 + 2] = c.b * dim;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({
    size,
    map: getStarSprite(),
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  return new THREE.Points(geo, mat);
}

/* Precision Machined Vernier Armillary Ring (Dual Rails + Engraved Calibration Ticks) */
function buildVernierRing(radius, tube, colorHex, tickCount = 48, metalness = 0.92, roughness = 0.18) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: colorHex,
    metalness,
    roughness,
  });

  // Primary machined rail
  group.add(new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 20, 128), mat));

  // Fine inner companion rail
  const innerRail = new THREE.Mesh(
    new THREE.TorusGeometry(radius - tube * 2.4, tube * 0.42, 12, 128),
    new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.88,
      roughness: 0.25,
      transparent: true,
      opacity: 0.68,
    })
  );
  group.add(innerRail);

  // Radial calibration vernier ticks around perimeter
  const tickGeo = new THREE.BoxGeometry(tube * 2.5, tube * 0.38, tube * 0.9);
  const majorGeo = new THREE.BoxGeometry(tube * 4.4, tube * 0.55, tube * 1.15);
  for (let i = 0; i < tickCount; i++) {
    const a = (i / tickCount) * Math.PI * 2;
    const isMajor = i % 6 === 0;
    const t = new THREE.Mesh(isMajor ? majorGeo : tickGeo, mat);
    const rPos = radius + tube * (isMajor ? 1.8 : 1.2);
    t.position.set(Math.cos(a) * rPos, Math.sin(a) * rPos, 0);
    t.rotation.z = a;
    group.add(t);
  }
  return group;
}

/* Sculpted High-Detail Procedural Culture Smart Ship (PMREM Titanium + Gold + Volumetric Aft Glow) */
function buildSmartShip(scale = 1.0) {
  const ship = new THREE.Group();
  const hullMat = new THREE.MeshStandardMaterial({
    color: 0x1e2026,
    metalness: 0.94,
    roughness: 0.22,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: 0xd4d4d8,
    metalness: 0.92,
    roughness: 0.18,
    emissive: 0x3b2608,
    emissiveIntensity: 0.35,
  });
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xfff0d4 });

  const bodyGeo = new THREE.SphereGeometry(0.45, 36, 24);
  bodyGeo.scale(4.25, 0.54, 1.12);
  ship.add(new THREE.Mesh(bodyGeo, hullMat));

  const ridgeGeo = new THREE.SphereGeometry(0.31, 24, 14);
  ridgeGeo.scale(2.85, 0.40, 0.65);
  const ridge = new THREE.Mesh(ridgeGeo, hullMat);
  ridge.position.set(-0.22, 0.18, 0);
  ship.add(ridge);

  const keel = new THREE.Mesh(new THREE.BoxGeometry(2.3, 0.13, 0.32), hullMat);
  keel.position.set(0.05, -0.21, 0);
  ship.add(keel);

  const stripGeo = new THREE.BoxGeometry(2.1, 0.032, 0.036);
  for (const side of [-1, 1]) {
    const strip = new THREE.Mesh(stripGeo, glowMat);
    strip.position.set(0.08, 0.02, side * 0.48);
    ship.add(strip);

    const lower = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.022, 0.028), trimMat);
    lower.position.set(-0.14, -0.095, side * 0.42);
    ship.add(lower);
  }

  for (const xPos of [0.85, -0.55]) {
    const r = new THREE.Mesh(new THREE.TorusGeometry(0.56, 0.013, 12, 48), trimMat);
    r.rotation.y = Math.PI / 2;
    r.position.x = xPos;
    ship.add(r);
  }

  const spireGeo = new THREE.CylinderGeometry(0.006, 0.014, 0.5, 8);
  for (const [sx, sy, h] of [[-0.55, 0.40, 1.0], [-0.42, 0.34, 0.70], [0.45, 0.30, 0.62]]) {
    const sp = new THREE.Mesh(spireGeo, trimMat);
    sp.scale.y = h;
    sp.position.set(sx, sy, 0);
    ship.add(sp);
  }

  // Soft volumetric aft engine luminescence
  const aftGlow = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: getGlowSprite(),
      color: 0xffffff,
      transparent: true,
      opacity: 0.72,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  aftGlow.scale.set(0.85, 0.55, 1);
  aftGlow.position.set(-1.92, 0, 0);
  ship.add(aftGlow);

  ship.scale.setScalar(scale);
  return ship;
}

/* ---------------------------------------------------------------------------
   SCENE 1: HERO PURE-BLACK SKY — AIRY STARS, FRESNEL RINGED PLANET & FLEET
   --------------------------------------------------------------------------- */
function initHeroCosmos() {
  const canvas = document.getElementById("hero-3d-canvas");
  const stage = document.getElementById("hero-stage");
  if (!canvas || !stage) return;

  let renderer;
  try { renderer = createRenderer(canvas); } catch (_e) { return; }

  const scene = new THREE.Scene();
  scene.environment = makeStudioEnv(renderer);

  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 200);
  camera.position.set(0, 0.75, 12.2);

  scene.add(new THREE.AmbientLight(0xffffff, 0.18));
  const sunLight = new THREE.DirectionalLight(0xfff4e0, 2.9);
  sunLight.position.set(12, 8, 6);
  scene.add(sunLight);
  const rimLight = new THREE.PointLight(0xd4d4d8, 2.8, 48);
  rimLight.position.set(-10, -4, 7);
  scene.add(rimLight);

  // Two layered starfields (fine distant dust + crisp foreground pinpricks)
  const starsFar = makeStarfield(1800, 26, 92, 0.18);
  const starsNear = makeStarfield(650, 16, 44, 0.28);
  scene.add(starsFar, starsNear);

  const cosmosGroup = new THREE.Group();
  scene.add(cosmosGroup);

  // Textured Ringed Planet + Custom Fresnel Horizon Rim
  const planetGroup = new THREE.Group();
  planetGroup.position.set(-5.35, 2.15, -4.7);
  const pMaps = makePlanetMaps(1);
  const planet = new THREE.Mesh(
    new THREE.SphereGeometry(2.02, 64, 64),
    new THREE.MeshStandardMaterial({
      map: pMaps.map,
      bumpMap: pMaps.bumpMap,
      bumpScale: 0.04,
      metalness: 0.16,
      roughness: 0.84,
    })
  );
  planetGroup.add(planet);
  planetGroup.add(makeAtmosphereHalo(2.09, 0xf2cfa0, 0.52));

  // Finely striated planetary ring disk
  const ringGeo = new THREE.RingGeometry(2.65, 4.42, 128);
  const posAttr = ringGeo.attributes.position;
  const uvAttr = ringGeo.attributes.uv;
  for (let i = 0; i < posAttr.count; i++) {
    const vx = posAttr.getX(i), vy = posAttr.getY(i);
    const rad = Math.sqrt(vx * vx + vy * vy);
    uvAttr.setXY(i, (rad - 2.65) / (4.42 - 2.65), 0.5);
  }
  const ringMesh = new THREE.Mesh(
    ringGeo,
    new THREE.MeshBasicMaterial({
      map: makeRingTexture(),
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
    })
  );
  ringMesh.rotation.x = Math.PI / 2.35;
  ringMesh.rotation.y = 0.26;
  planetGroup.add(ringMesh);
  scene.add(planetGroup);

  // Distant crescent moon in upper-right sky
  const mMaps = makePlanetMaps(0);
  const moonGroup = new THREE.Group();
  moonGroup.position.set(5.9, 2.75, -5.9);
  const moon = new THREE.Mesh(
    new THREE.SphereGeometry(0.60, 40, 40),
    new THREE.MeshStandardMaterial({
      map: mMaps.map,
      bumpMap: mMaps.bumpMap,
      bumpScale: 0.03,
      metalness: 0.12,
      roughness: 0.9,
    })
  );
  moonGroup.add(moon, makeAtmosphereHalo(0.63, 0xe4e4e7, 0.34));
  scene.add(moonGroup);

  // Flagship gliding smoothly in foreground
  const flagship = buildSmartShip(0.62);
  flagship.position.set(-3.1, -2.08, 1.3);
  flagship.rotation.set(0.14, 0.34, 0.07);
  cosmosGroup.add(flagship);

  // 2 Calm Distant Escort Vessels
  const escorts = [];
  for (let i = 0; i < 2; i++) {
    const vessel = buildSmartShip(0.18);
    cosmosGroup.add(vessel);
    escorts.push({
      mesh: vessel,
      rx: 5.5 + i * 1.4,
      rz: 3.1 + i * 0.9,
      yOff: -1.0 + i * 2.05,
      speed: i % 2 === 0 ? 0.095 : -0.08,
      phase: i * Math.PI + 0.6,
    });
  }

  function resize() {
    const w = Math.max(300, stage.clientWidth);
    const h = Math.max(360, stage.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(stage);

  attachVisibilityLoop(stage, (_dt, t) => {
    const mult = REDUCED ? 0.1 : 1.0;
    starsFar.rotation.y = t * 0.006 * mult;
    starsNear.rotation.y = t * 0.011 * mult;
    planet.rotation.y = t * 0.04 * mult;
    moon.rotation.y = -t * 0.028 * mult;

    // Gentle breathing flagship drift
    flagship.position.y = -2.08 + Math.sin(t * 0.42 * mult) * 0.09;
    flagship.rotation.z = 0.07 + Math.sin(t * 0.32 * mult) * 0.022;

    for (const v of escorts) {
      const a = v.phase + t * v.speed * mult;
      const x = Math.cos(a) * v.rx;
      const z = Math.sin(a) * v.rz - 1.6;
      const y = v.yOff + Math.sin(a * 2) * 0.22;
      v.mesh.position.set(x, y, z);
      const na = a + (v.speed > 0 ? 0.08 : -0.08);
      v.mesh.rotation.y = Math.atan2(-(Math.sin(na) * v.rz - 1.6 - z), Math.cos(na) * v.rx - x);
    }

    renderer.render(scene, camera);
  });
}

/* ---------------------------------------------------------------------------
   SCENE 2: STUDIO-GRADE OPTICAL MIND CORE (GLSL LIQUID ENERGY + CLEARCOAT CRYSTAL + VERNIER GIMBALS)
   --------------------------------------------------------------------------- */
function initMindCore() {
  const canvas = document.getElementById("mind-3d-canvas");
  const stage = document.getElementById("mind-3d-stage");
  if (!canvas || !stage) return;

  let renderer;
  try { renderer = createRenderer(canvas); } catch (_e) { return; }

  const scene = new THREE.Scene();
  scene.environment = makeStudioEnv(renderer);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 60);
  camera.position.set(0, 0.14, 6.25);

  scene.add(new THREE.AmbientLight(0xffffff, 0.28));
  const keyL = new THREE.PointLight(0xffe6bd, 4.5, 24);
  keyL.position.set(4.5, 4.5, 5.5);
  scene.add(keyL);
  const rimL = new THREE.PointLight(0xffffff, 3.2, 24);
  rimL.position.set(-4.5, -3.5, 4.5);
  scene.add(rimL);

  // Subtle depth starfield behind core
  scene.add(makeStarfield(240, 10, 26, 0.14));

  const root = new THREE.Group();
  scene.add(root);

  // 1. Volumetric Soft Glow Aura behind the optical core
  const auraMat = new THREE.SpriteMaterial({
    map: getGlowSprite(),
    color: 0xd4d4d8,
    transparent: true,
    opacity: 0.42,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const aura = new THREE.Sprite(auraMat);
  aura.scale.set(4.1, 4.1, 1);
  root.add(aura);

  // 2. Inner Animated GLSL Liquid-Optical Mind Core Shader
  const coreUniforms = {
    uTime: { value: 0 },
    uAccent: { value: new THREE.Color(0xd4d4d8) },
    uDeep: { value: new THREE.Color(0x0c0b0e) },
  };
  const innerCoreMat = new THREE.ShaderMaterial({
    uniforms: coreUniforms,
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vPos;
      varying vec3 vView;
      void main() {
        vPos = position;
        vNormal = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vView = -mv.xyz;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform vec3 uAccent;
      uniform vec3 uDeep;
      varying vec3 vNormal;
      varying vec3 vPos;
      varying vec3 vView;
      void main() {
        vec3 n = normalize(vNormal);
        vec3 v = normalize(vView);
        float fresnel = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 2.3);
        float w1 = sin(vPos.x * 5.2 + uTime * 0.9) * cos(vPos.y * 5.2 - uTime * 0.7);
        float w2 = sin((vPos.z * 6.0 + vPos.y * 4.0) + uTime * 1.15);
        float pulse = 0.5 + 0.5 * (w1 * 0.6 + w2 * 0.4);
        float veins = smoothstep(0.62, 0.96, pulse);
        vec3 col = mix(uDeep, uAccent * 0.72, veins * 0.65);
        col += uAccent * fresnel * 1.35;
        col += vec3(1.0, 0.96, 0.90) * pow(fresnel, 4.5) * 0.85;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  const innerCore = new THREE.Mesh(new THREE.SphereGeometry(0.84, 64, 64), innerCoreMat);
  root.add(innerCore);

  // 3. Sculpted Faceted Optical-Glass Shell (Clearcoat Physical Material + Crisp Gold Bevel Edges)
  const shellGeo = new THREE.IcosahedronGeometry(1.16, 1);
  const glassShell = new THREE.Mesh(
    shellGeo,
    new THREE.MeshPhysicalMaterial({
      color: 0x161518,
      metalness: 0.12,
      roughness: 0.06,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
      reflectivity: 1.0,
      transparent: true,
      opacity: 0.44,
      flatShading: true,
    })
  );
  const edgeMat = new THREE.LineBasicMaterial({
    color: 0xd4d4d8,
    transparent: true,
    opacity: 0.62,
  });
  const crystalEdges = new THREE.LineSegments(new THREE.EdgesGeometry(shellGeo, 12), edgeMat);
  const crystalGroup = new THREE.Group();
  crystalGroup.add(glassShell, crystalEdges);
  root.add(crystalGroup);

  // 4. Three Precision Machined Vernier Armillary Gimbal Rings
  const g1 = buildVernierRing(1.66, 0.026, 0xd4d4d8, 48, 0.94, 0.16);
  const g2 = buildVernierRing(1.98, 0.020, 0xd8dadf, 36, 0.90, 0.20);
  const g3 = buildVernierRing(2.28, 0.014, 0x8a8b94, 36, 0.88, 0.26);
  g2.rotation.x = Math.PI / 3;
  g3.rotation.y = Math.PI / 4;
  root.add(g1, g2, g3);

  // 5. Orbiting Luminous Telemetry Photons (Airy-Disk Sprites + Delicate Golden Chords)
  const satGroup = new THREE.Group();
  root.add(satGroup);
  const satCount = 24;
  const satCoords = [];
  const satPosArr = new Float32Array(satCount * 3);
  const satColArr = new Float32Array(satCount * 3);
  const cGold = new THREE.Color(0xf5d499);
  const cWhite = new THREE.Color(0xffffff);
  for (let i = 0; i < satCount; i++) {
    const phi = Math.acos(1 - 2 * ((i + 0.5) / satCount));
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const rad = 1.90 + (i % 3) * 0.24;
    const v = new THREE.Vector3(
      rad * Math.sin(phi) * Math.cos(theta),
      rad * Math.sin(phi) * Math.sin(theta),
      rad * Math.cos(phi)
    );
    satCoords.push(v);
    satPosArr[i * 3] = v.x;
    satPosArr[i * 3 + 1] = v.y;
    satPosArr[i * 3 + 2] = v.z;
    const c = i % 2 === 0 ? cGold : cWhite;
    satColArr[i * 3] = c.r;
    satColArr[i * 3 + 1] = c.g;
    satColArr[i * 3 + 2] = c.b;
  }
  const satGeo = new THREE.BufferGeometry();
  satGeo.setAttribute("position", new THREE.BufferAttribute(satPosArr, 3));
  satGeo.setAttribute("color", new THREE.BufferAttribute(satColArr, 3));
  satGroup.add(new THREE.Points(
    satGeo,
    new THREE.PointsMaterial({
      size: 0.24,
      map: getStarSprite(),
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  ));

  const chordPts = [];
  for (let i = 0; i < satCoords.length; i++) {
    chordPts.push(satCoords[i], satCoords[(i + 5) % satCoords.length]);
  }
  const chordMat = new THREE.LineBasicMaterial({ color: 0xd4d4d8, transparent: true, opacity: 0.16 });
  satGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(chordPts), chordMat));

  // Smooth harmonic transitions driven by Transparency & Trust tabs
  const MODES = {
    certainty: { accent: 0xd4d4d8, speed: 1.0 },
    authentic: { accent: 0x7ddcb0, speed: 1.25 },
    privacy:   { accent: 0xf4c474, speed: 0.78 },
    routing:   { accent: 0xe4e8f0, speed: 1.45 },
  };
  let targetAccent = new THREE.Color(MODES.certainty.accent);
  let targetSpeed = 1.0;
  let curSpeed = 1.0;

  window.addEventListener("space:obs-mode", (e) => {
    const m = MODES[e.detail] || MODES.certainty;
    targetAccent = new THREE.Color(m.accent);
    targetSpeed = m.speed;
  });

  function resize() {
    const w = Math.max(260, stage.clientWidth);
    const h = Math.max(260, stage.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(stage);

  let phase = 0;
  attachVisibilityLoop(stage, (dt, _t) => {
    const mult = REDUCED ? 0.15 : 1.0;
    curSpeed += (targetSpeed - curSpeed) * Math.min(1, dt * 3.0);
    phase += dt * curSpeed * mult;

    coreUniforms.uTime.value = phase;
    coreUniforms.uAccent.value.lerp(targetAccent, Math.min(1, dt * 3.5));
    edgeMat.color.lerp(targetAccent, Math.min(1, dt * 3.5));
    chordMat.color.lerp(targetAccent, Math.min(1, dt * 3.5));
    auraMat.color.lerp(targetAccent, Math.min(1, dt * 3.5));
    keyL.color.lerp(targetAccent, Math.min(1, dt * 3.5));

    const pulse = 1.0 + Math.sin(phase * 1.4) * 0.022;
    innerCore.scale.setScalar(pulse);
    aura.scale.setScalar(3.95 + Math.sin(phase * 1.4) * 0.25);

    crystalGroup.rotation.y = phase * 0.22;
    crystalGroup.rotation.x = Math.sin(phase * 0.35) * 0.24;

    // Golden-ratio harmonic armillary gimbal rotation
    g1.rotation.x = phase * 0.24;
    g1.rotation.y = phase * 0.15;
    g2.rotation.y = -phase * 0.28;
    g2.rotation.z = phase * 0.12;
    g3.rotation.x = -phase * 0.16;
    g3.rotation.z = -phase * 0.19;
    satGroup.rotation.y = phase * 0.14;
    satGroup.rotation.x = Math.cos(phase * 0.2) * 0.12;

    renderer.render(scene, camera);
  });
}

/* ---------------------------------------------------------------------------
   SCENE 3: STUDIO-GRADE 3D NETWORK TOPOLOGY (FRESNEL PLANET + COMET-TAILED SIGNALS)
   --------------------------------------------------------------------------- */
function initOrbitalMechanics() {
  const canvas = document.getElementById("mech-3d-canvas");
  const stage = document.getElementById("mech-3d-stage");
  if (!canvas || !stage) return;

  let renderer;
  try { renderer = createRenderer(canvas); } catch (_e) { return; }

  const scene = new THREE.Scene();
  scene.environment = makeStudioEnv(renderer);

  const camera = new THREE.PerspectiveCamera(40, 1.5, 0.1, 90);
  camera.position.set(0, 3.6, 8.6);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.26));
  const keyLight = new THREE.DirectionalLight(0xfff6e5, 2.7);
  keyLight.position.set(8, 10, 7);
  scene.add(keyLight);
  const goldLight = new THREE.PointLight(0xd4d4d8, 2.8, 26);
  goldLight.position.set(-6, 2, 5);
  scene.add(goldLight);

  scene.add(makeStarfield(360, 12, 34, 0.16));

  const world = new THREE.Group();
  scene.add(world);

  // Central Textured Dark Planet + Fresnel Limb Glow + Machined Equatorial Vernier Ring
  const pMaps = makePlanetMaps(1);
  const centerPlanet = new THREE.Mesh(
    new THREE.SphereGeometry(0.94, 56, 56),
    new THREE.MeshStandardMaterial({
      map: pMaps.map,
      bumpMap: pMaps.bumpMap,
      bumpScale: 0.03,
      metalness: 0.24,
      roughness: 0.72,
    })
  );
  scene.add(centerPlanet, makeAtmosphereHalo(1.01, 0xd4d4d8, 0.62));

  const eqHalo = buildVernierRing(1.30, 0.014, 0xd4d4d8, 36, 0.92, 0.2);
  eqHalo.rotation.x = Math.PI / 2;
  world.add(eqHalo);

  // Three inclined precision orbital planes
  const ringDefs = [
    { key: "routing", r: 2.02, tiltX: Math.PI / 2 + 0.18, tiltZ: 0.12, col: 0xffffff },
    { key: "verify",  r: 3.05, tiltX: Math.PI / 2 - 0.14, tiltZ: -0.18, col: 0xd4d4d8 },
    { key: "credits", r: 4.08, tiltX: Math.PI / 2 + 0.10, tiltZ: 0.24, col: 0xa1a1aa },
  ];
  const ringMeshes = {};
  for (const rd of ringDefs) {
    const mat = new THREE.MeshBasicMaterial({
      color: rd.col,
      transparent: true,
      opacity: rd.key === "verify" ? 0.85 : 0.22,
    });
    const mesh = new THREE.Mesh(new THREE.TorusGeometry(rd.r, 0.016, 12, 128), mat);
    mesh.rotation.x = rd.tiltX;
    mesh.rotation.z = rd.tiltZ;
    world.add(mesh);
    ringMeshes[rd.key] = { mesh, mat, targetOp: rd.key === "verify" ? 0.85 : 0.22 };
  }

  // 6 Sculpted Station Nodes with Soft Glow Halos
  const nodes = [
    new THREE.Vector3(-3.4, 0.55, 0.8),
    new THREE.Vector3(-2.7, -0.65, -1.4),
    new THREE.Vector3(3.4, 0.65, -0.6),
    new THREE.Vector3(2.9, -0.55, 1.5),
    new THREE.Vector3(0.2, 1.45, -3.0),
    new THREE.Vector3(-0.3, -1.1, 2.9),
  ];
  nodes.forEach((pos, idx) => {
    const shipNode = buildSmartShip(0.24);
    shipNode.position.copy(pos);
    shipNode.lookAt(0, 0, 0);
    shipNode.rotation.y += idx * 0.6;
    world.add(shipNode);

    const halo = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: getGlowSprite(),
        color: idx % 2 === 0 ? 0xd4d4d8 : 0xffffff,
        transparent: true,
        opacity: 0.42,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    halo.scale.set(0.68, 0.68, 1);
    halo.position.copy(pos);
    world.add(halo);
  });

  // Curved 3D Bezier trajectories + Comet-Tailed Luminous Signal Packets
  const streams = [];
  const TRAIL_LEN = 5;
  const pairs = [
    [nodes[0], new THREE.Vector3(0, 0, 0)],
    [new THREE.Vector3(0, 0, 0), nodes[2]],
    [nodes[1], new THREE.Vector3(0, 0, 0)],
    [new THREE.Vector3(0, 0, 0), nodes[3]],
    [nodes[2], nodes[4]],
    [nodes[3], nodes[5]],
    [nodes[4], nodes[0]],
    [nodes[5], nodes[1]],
  ];
  pairs.forEach(([a, b], idx) => {
    const mid = a.clone().add(b).multiplyScalar(0.5);
    mid.y += (idx % 2 === 0 ? 1.0 : -0.78);
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const isGold = idx % 2 === 0;
    const lMat = new THREE.LineBasicMaterial({
      color: isGold ? 0xd4d4d8 : 0xffffff,
      transparent: true,
      opacity: 0.16,
    });
    world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)), lMat));

    for (let k = 0; k < 2; k++) {
      const sprites = [];
      for (let j = 0; j < TRAIL_LEN; j++) {
        const frac = 1 - j / TRAIL_LEN;
        const sp = new THREE.Sprite(
          new THREE.SpriteMaterial({
            map: j === 0 ? getStarSprite() : getGlowSprite(),
            color: isGold ? 0xfce0a8 : 0xffffff,
            transparent: true,
            opacity: j === 0 ? 0.98 : frac * 0.45,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
          })
        );
        const sc = j === 0 ? 0.24 : 0.28 * frac;
        sp.scale.set(sc, sc, 1);
        world.add(sp);
        sprites.push(sp);
      }
      streams.push({
        curve,
        sprites,
        offset: k * 0.5 + idx * 0.13,
        speed: 0.22 + (idx % 3) * 0.055,
      });
    }
  });

  const CAM_PRESETS = {
    routing: { y: 2.75, z: 7.7 },
    verify:  { y: 3.65, z: 8.6 },
    credits: { y: 4.85, z: 9.6 },
  };
  let camTarget = CAM_PRESETS.verify;
  window.addEventListener("space:mech-ring", (e) => {
    const key = e.detail || "verify";
    camTarget = CAM_PRESETS[key] || CAM_PRESETS.verify;
    for (const [k, obj] of Object.entries(ringMeshes)) {
      obj.targetOp = k === key ? 0.90 : 0.18;
    }
  });

  function resize() {
    const w = Math.max(280, stage.clientWidth);
    const h = Math.max(280, stage.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(stage);

  attachVisibilityLoop(stage, (dt, t) => {
    const mult = REDUCED ? 0.15 : 1.0;
    camera.position.y += (camTarget.y - camera.position.y) * Math.min(1, dt * 3.0);
    camera.position.z += (camTarget.z - camera.position.z) * Math.min(1, dt * 3.0);
    camera.lookAt(0, 0, 0);

    for (const obj of Object.values(ringMeshes)) {
      obj.mat.opacity += (obj.targetOp - obj.mat.opacity) * Math.min(1, dt * 4.0);
    }

    world.rotation.y = t * 0.10 * mult;
    centerPlanet.rotation.y = t * 0.08 * mult;
    eqHalo.rotation.z = t * 0.24 * mult;

    for (const st of streams) {
      const headU = (st.offset + t * st.speed * mult) % 1;
      for (let j = 0; j < st.sprites.length; j++) {
        const u = ((headU - j * 0.018) % 1 + 1) % 1;
        st.curve.getPoint(u, st.sprites[j].position);
      }
    }

    renderer.render(scene, camera);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initHeroCosmos();
  initMindCore();
  initOrbitalMechanics();
});
