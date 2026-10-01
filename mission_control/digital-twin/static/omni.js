// OmniVision 360 — OmniView operator console (vanilla JS + three.js r128).
// Modules: static/omni/*.js. Contracts: mission_control/omni/CONTRACTS.md (§E).
// Every network source is optional; ?demo=1 (or omni :9114 unreachable)
// switches to the built-in synthetic DEMO source.

const fatal = (msg) => {
  const el = document.getElementById("fatal");
  if (el) { el.textContent = msg; el.style.display = "block"; }
};

if (typeof THREE === "undefined") {
  fatal("three.js could not be loaded (cdnjs.cloudflare.com unreachable?) — OmniView needs it.");
} else {
  main().catch((e) => { console.error("[omni] init failed", e); fatal(`OmniView init failed: ${e && e.message}`); });
}

async function main() {
  const F = await import("./omni/frames.js");
  const { endpoints, fetchJSON, postJSON, ReconnectingWS, Poller } = await import("./omni/net.js");
  const { Bowl } = await import("./omni/bowl.js");
  const { RobotModel } = await import("./omni/robot.js");
  const { VoxelCloud } = await import("./omni/pointcloud.js");
  const { PersonLayer } = await import("./omni/persons.js");
  const { Hud } = await import("./omni/hud.js");
  const { CameraRig, VIEW_NAMES } = await import("./omni/views.js");
  const { DemoSource } = await import("./omni/demo.js");

  const EP = endpoints();
  const $ = (id) => document.getElementById(id);

  // ---------------- settings (localStorage, best effort) ----------------
  const DEF = { bloom: false, perf: "med", colorMode: 0, bowl: true, night: false, thermal: false, cloud: true };
  const settings = { ...DEF };
  try { Object.assign(settings, JSON.parse(localStorage.getItem("omniview.settings") || "{}")); } catch (_) {}
  const save = () => { try { localStorage.setItem("omniview.settings", JSON.stringify(settings)); } catch (_) {} };
  const PERF = {
    low: { pr: Math.min(devicePixelRatio || 1, 0.75), maxPts: 300_000, aa: false },
    med: { pr: Math.min(devicePixelRatio || 1, 1.25), maxPts: 1_000_000, aa: true },
    high: { pr: Math.min(devicePixelRatio || 1, 2), maxPts: 2_000_000, aa: true },
  };
  if (!PERF[settings.perf]) settings.perf = "med";

  // ---------------- renderer / scene ----------------
  const viewport = $("viewport");
  const renderer = new THREE.WebGLRenderer({ antialias: PERF[settings.perf].aa, powerPreference: "high-performance", preserveDrawingBuffer: false });
  renderer.setPixelRatio(PERF[settings.perf].pr);
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x02050a, 1);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x02050a, 0.016);
  const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.03, 600);
  camera.position.set(6, 4, 6);

  scene.add(new THREE.HemisphereLight(0x7fb4ff, 0x0a0f18, 0.75));
  const moon = new THREE.DirectionalLight(0xbfd8ff, 0.8);
  moon.position.set(-4, 10, 3);
  scene.add(moon);
  const rim = new THREE.PointLight(0x00c8ff, 1.2, 6);
  scene.add(rim);

  // sci-fi grid floor
  const gridMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color(0x00d0ff) }, uCenter: { value: new THREE.Vector2() }, uTime: { value: 0 } },
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.0); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
    fragmentShader: `uniform vec3 uColor; uniform vec2 uCenter; uniform float uTime; varying vec3 vW;
      float gl(vec2 p, float s, float w){ vec2 g=abs(fract(p/s-0.5)-0.5)*s; return 1.0-smoothstep(0.0,w,min(g.x,g.y)); }
      void main(){ float d=length(vW.xz-uCenter);
        float a = gl(vW.xz,1.0,0.02)*0.22 + gl(vW.xz,5.0,0.045)*0.5;
        float fade = 1.0 - smoothstep(15.0, 70.0, d);
        float pulse = smoothstep(1.0,0.0,abs(d-mod(uTime*4.0,60.0))*0.8)*0.35;
        gl_FragColor = vec4(uColor, (a+pulse*gl(vW.xz,1.0,0.06))*fade*0.9 + 0.035*fade); }`,
  });
  const grid = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), gridMat);
  grid.rotation.x = -Math.PI / 2;
  grid.renderOrder = -2;
  scene.add(grid);

  // robot frames: robotYaw (three, floor level) > baseRos (ROS base frame at body height)
  const robotYaw = new THREE.Group();
  scene.add(robotYaw);
  const baseRos = F.makeRosGroup();
  baseRos.position.y = F.BASE_HEIGHT;
  robotYaw.add(baseRos);
  // world-frame voxels: ROS group; voxel z is body-centric (floor ≈ -BASE_HEIGHT)
  const worldRos = F.makeRosGroup();
  worldRos.position.y = F.BASE_HEIGHT;
  scene.add(worldRos);

  const bowl = new Bowl(baseRos, settings.perf);
  const robot = new RobotModel(baseRos);
  robot.load().then(() => console.info("[omni] Go2 URDF model loaded"))
    .catch((e) => console.warn("[omni] Go2 URDF/mesh load failed, box-dog fallback:", e && e.message));
  const cloud = new VoxelCloud(worldRos, PERF[settings.perf].maxPts);

  // safety rings
  const ringDefs = [[F.ZONES.STOP, 0xff3344], [F.ZONES.SLOW, 0xff8a1e], [F.ZONES.CAUTION, 0xffd23a]];
  const rings = ringDefs.map(([r, c]) => {
    const m = new THREE.Mesh(new THREE.RingGeometry(r - 0.025, r + 0.025, 128),
      new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.y = 0.02; m.userData.r = r;
    robotYaw.add(m);
    const fill = new THREE.Mesh(new THREE.CircleGeometry(r, 96),
      new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending }));
    fill.rotation.x = -Math.PI / 2; fill.position.y = 0.015;
    robotYaw.add(fill);
    m.userData.fill = fill;
    return m;
  });

  // ---------------- post-processing (optional) ----------------
  let composer = null, bloomPass = null;
  function setupBloom() {
    if (composer || typeof THREE.EffectComposer !== "function" || typeof THREE.UnrealBloomPass !== "function") return !!composer;
    try {
      composer = new THREE.EffectComposer(renderer);
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(innerWidth, innerHeight);
      composer.addPass(new THREE.RenderPass(scene, camera));
      bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.85, 0.5, 0.55);
      composer.addPass(bloomPass);
      return true;
    } catch (e) { console.warn("[omni] bloom unavailable", e); composer = null; return false; }
  }

  // ---------------- HUD / persons / views ----------------
  const state = {
    source: "init", pose: { x: 0, y: 0, yaw: 0 }, poseT: 0, speed: 0, battery: null, joints: null,
    safety: null, safetySrc: "", health: null, cams: [], target: null,
    links: { omni: "off", video: "off", persons: "off", voxels: "off", safety: "off", pose: "off" },
    camFps: new Map(), lastPersonsT: 0,
  };
  const hud = new Hud({ onThumbClick: (idx) => enterFpv(idx) });
  const persons = new PersonLayer(scene, $("labels"), {
    onSelect: (gid) => {
      state.target = gid;
      $("target").style.display = gid != null ? "flex" : "none";
      $("target-id").textContent = gid != null ? `#${gid}` : "";
      if (gid != null) {
        console.info("[omni] TRACK TARGET", gid);
        // TODO(pursuit): no pursuit endpoint in CONTRACTS.md yet. When Agent D exposes one on
        // safety_guard (:9115, e.g. POST /pursuit {gid}) send it here — motion MUST go via safety_guard.
        window.dispatchEvent(new CustomEvent("omni:track-target", { detail: { gid } }));
      } else {
        window.dispatchEvent(new CustomEvent("omni:track-target", { detail: { gid: null } }));
      }
    },
  });
  const rig = new CameraRig(camera, renderer.domElement);

  function setView(m) {
    if (m === 4) rig.setFpvCam(rig.fpvCam || 0, state.cams[rig.fpvCam || 0]);
    else rig.setMode(m);
    applyVisibility();
    hud.setText("mode-chip", VIEW_NAMES[m] + (m === 4 ? ` · ${(state.cams[rig.fpvCam] || {}).id || ""}` : ""));
    hud.setActiveThumb(m === 4 ? rig.fpvCam : -1);
  }
  function enterFpv(idx) {
    rig.fpvCam = idx;
    setView(4);
  }
  function applyVisibility() {
    bowl.visible = settings.bowl && rig.mode !== 5;
    cloud.visible = settings.cloud && !(bowl.visible && rig.mode === 4);   // FPV: pure camera view
    cloud.setMaxPx(bowl.visible ? 4 : 22);
    document.body.classList.toggle("night", !!settings.night);
    bowl.night = !!settings.night;
    bowl.thermal = settings.night || settings.thermal ? 1 : 0;
    gridMat.uniforms.uColor.value.setHex(settings.night ? 0xffb020 : 0x00d0ff);
    cloud.setMode(settings.colorMode);
    for (const [k, v] of Object.entries({ "btn-bloom": settings.bloom, "btn-night": settings.night, "btn-thermal": settings.thermal, "btn-bowl": settings.bowl, "btn-cloud": settings.cloud })) {
      const b = $(k); if (b) b.classList.toggle("on", !!v);
    }
    hud.setText("btn-color", ["COLOR: RGB", "COLOR: HOT", "COLOR: HEIGHT"][settings.colorMode]);
    hud.setText("btn-perf", `PERF: ${settings.perf.toUpperCase()}`);
  }

  // ---------------- data sinks (shared by live + demo) ----------------
  const pendingFrame = new Map();   // idx -> source
  const current = new Map();        // idx -> ImageBitmap (to close later)
  const toClose = [];
  function onFrame(idx, src) {
    pendingFrame.set(idx, src);
    const s = state.camFps.get(idx) || { n: 0, fps: 0 };
    s.n++; state.camFps.set(idx, s);
  }
  function flushFrames() {
    for (const [idx, src] of pendingFrame) {
      bowl.setFrame(idx, src);
      hud.drawThumb(idx, src);
      if (typeof ImageBitmap !== "undefined" && src instanceof ImageBitmap) {
        const old = current.get(idx);
        if (old && old !== src) toClose.push(old);
        current.set(idx, src);
      }
    }
    pendingFrame.clear();
  }
  function onPersons(msg) { state.lastPersonsT = performance.now() / 1000; persons.update(msg, state.pose, performance.now() / 1000); }
  function onPose(p) {
    if (!p || !Number.isFinite(+p.x)) return;
    const now = performance.now() / 1000;
    if (state.poseT) {
      const dt = now - state.poseT;
      if (dt > 0.02) {
        const v = Math.hypot(p.x - state.pose.x, p.y - state.pose.y) / dt;
        state.speed = state.speed * 0.6 + v * 0.4;
      }
    }
    state.pose = { x: +p.x, y: +p.y, yaw: +p.yaw || 0 };
    state.poseT = now;
  }
  function setRig(cams) {
    state.cams = cams;
    bowl.setRig(cams);
    hud.setCams(cams);
    const firstRgb = cams.findIndex((c) => c.modality === "rgb");
    rig.fpvCam = firstRgb >= 0 ? firstRgb : 0;
  }

  // ---------------- LIVE source ----------------
  let live = null;
  function startLive(rigJSON) {
    stopDemo();
    state.source = "live";
    setRig(F.normalizeRig(rigJSON));
    if (rigJSON && Number.isFinite(+rigJSON.ground_z) && Math.abs(-rigJSON.ground_z - F.BASE_HEIGHT) > 0.03)
      console.warn(`[omni] rig.ground_z=${rigJSON.ground_z} differs from BASE_HEIGHT=${F.BASE_HEIGHT}`);
    cloud.clear();
    const decoding = new Set();
    let voxFirst = true;
    live = {
      video: new ReconnectingWS(`${EP.omniWs}/ws/video`, {
        binary: true,
        onState: (s) => { state.links.video = s; },
        onMessage: (ab) => {
          if (!(ab instanceof ArrayBuffer) || ab.byteLength < 2) return;
          const idx = new Uint8Array(ab, 0, 1)[0];
          if (decoding.has(idx) || typeof createImageBitmap !== "function") return;   // drop while busy
          decoding.add(idx);
          createImageBitmap(new Blob([new Uint8Array(ab, 1)], { type: "image/jpeg" }))
            .then((bmp) => onFrame(idx, bmp))
            .catch(() => {})
            .finally(() => decoding.delete(idx));
        },
      }),
      persons: new ReconnectingWS(`${EP.omniWs}/ws/persons`, {
        onState: (s) => { state.links.persons = s; },
        onMessage: (txt) => { let m = null; try { m = JSON.parse(txt); } catch (_) { return; } onPersons(m); },
      }),
      voxels: new ReconnectingWS(`${EP.omniWs}/ws/voxels`, {
        binary: true,
        onOpen: () => { voxFirst = true; },
        onState: (s) => { state.links.voxels = s; },
        onMessage: (ab) => { if (ab instanceof ArrayBuffer) { cloud.ingest(ab, voxFirst); voxFirst = false; } },
      }),
      health: new Poller(async () => {
        const [r, st] = await Promise.all([fetchJSON(`${EP.omniHttp}/health`, 1500), fetchJSON(`${EP.omniHttp}/stats`, 1500)]);
        state.links.omni = r.ok ? "live" : "down";
        if (r.ok) state.health = { ...(r.data || {}), _stats: st.ok ? st.data : null };
        return r.ok;
      }, 2000),
    };
  }

  // safety_guard: always polled unless demo was forced
  let safetyPoller = null;
  function startSafety() {
    if (safetyPoller || EP.demoForced) return;
    safetyPoller = new Poller(async () => {
      const r = await fetchJSON(`${EP.safetyHttp}/state`, 800);
      state.links.safety = r.ok ? "live" : "down";
      if (r.ok && r.data) { state.safety = r.data; state.safetySrc = "safety_guard"; }
      else if (state.source === "live") { state.safety = null; }
      return r.ok;
    }, 200, 5000);
  }

  // pose / battery from digital-twin /ws/live (same origin)
  let poseWs = null;
  function startPose() {
    if (poseWs || !EP.sameOrigin || EP.demoForced) return;
    poseWs = new ReconnectingWS(`${EP.twinWs}/ws/live`, {
      onState: (s) => { state.links.pose = s; },
      onMessage: (txt) => {
        if (state.source !== "live") return;
        let m = null; try { m = JSON.parse(txt); } catch (_) { return; }
        if (m.pose) onPose(m.pose);
        if (m.battery && m.battery.percent != null) state.battery = +m.battery.percent;
        const q = m.joints || m.motor_q || (m.lowstate && m.lowstate.q);
        if (Array.isArray(q)) state.joints = q;
      },
    });
  }

  // ---------------- DEMO source ----------------
  let demo = null, demoTimers = [];
  function startDemo(reason) {
    if (demo) return;
    state.source = "demo";
    demo = new DemoSource();
    setRig(demo.cams);
    cloud.clear();
    state.links.omni = EP.demoForced ? "demo" : "down";
    for (const k of ["video", "persons", "voxels"]) state.links[k] = "demo";
    if (EP.demoForced) { state.links.safety = "demo"; state.links.pose = "demo"; }
    hud.setText("src-chip", reason, "demo");
    let first = true;
    demoTimers.push(setInterval(() => { demo.step(); for (const f2 of demo.renderFrames()) onFrame(f2.idx, f2.canvas); }, 100));
    demoTimers.push(setInterval(() => onPersons(demo.personsMsg()), 100));
    demoTimers.push(setInterval(() => { cloud.ingest(demo.voxelChunk(1500), first); first = false; }, 250));
    demoTimers.push(setInterval(() => {
      if (state.links.safety !== "live") { state.safety = demo.safetyState(); state.safetySrc = "demo"; }
      state.health = demo.health(); onPose(demo.pose()); state.battery = demo.battery;
    }, 200));
    demo.step();
  }
  function stopDemo() {
    demoTimers.forEach(clearInterval); demoTimers = [];
    demo = null;
  }

  async function probeOmni() {
    const r = await fetchJSON(`${EP.omniHttp}/rig`, 2500);
    if (r.ok && r.data && (r.data.cameras || Array.isArray(r.data))) {
      hud.setText("src-chip", "LIVE", "live");
      startLive(r.data);
      return true;
    }
    return false;
  }

  if (EP.demoForced) startDemo("DEMO");
  else {
    startSafety(); startPose();
    const ok = await probeOmni();
    if (!ok) {
      startDemo("DEMO · omni offline");
      const retry = setInterval(async () => { if (await probeOmni()) clearInterval(retry); }, 15000);
    }
  }

  // ---------------- E-STOP ----------------
  let estopBusy = false;
  async function estop() {
    if (estopBusy) return;
    estopBusy = true;
    document.body.classList.add("estop-flash");
    setTimeout(() => document.body.classList.remove("estop-flash"), 900);
    toast("E-STOP SENT", "crit");
    const jobs = [postJSON(`${EP.safetyHttp}/stop`, { source: "omniview" }, 1500)];
    if (EP.sameOrigin) jobs.push(postJSON("/api/estop", {}, 1500));
    const res = await Promise.all(jobs);
    const txt = `E-STOP · safety_guard: ${res[0].ok ? "OK" : "FAIL"}${res[1] ? ` · core: ${res[1].ok ? "OK" : "FAIL"}` : ""}`;
    toast(txt, res.some((r) => r.ok) ? "warn" : "crit");
    console.warn("[omni]", txt);
    estopBusy = false;
  }
  $("estop").addEventListener("click", estop);

  let toastTimer = null;
  function toast(msg, cls = "info") {
    const t = $("toast");
    t.textContent = msg; t.dataset.cls = cls; t.style.opacity = 1;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.style.opacity = 0; }, 2600);
  }

  // ---------------- input ----------------
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (rig.dragMoved > 6) return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera({ x: ((e.clientX - r.left) / r.width) * 2 - 1, y: -((e.clientY - r.top) / r.height) * 2 + 1 }, camera);
    const gid = persons.pick(ray);
    if (gid != null) { persons.select(gid); toast(`TRACK TARGET #${gid}`, "info"); }
  });
  $("target-clear").addEventListener("click", () => persons.select(null));

  function toggle(key) { settings[key] = !settings[key]; save(); applyVisibility(); }
  function cyclePerf() {
    const order = ["low", "med", "high"];
    settings.perf = order[(order.indexOf(settings.perf) + 1) % 3];
    save();
    const p = PERF[settings.perf];
    renderer.setPixelRatio(p.pr);
    renderer.setSize(innerWidth, innerHeight);
    if (composer) { composer.setPixelRatio(p.pr); composer.setSize(innerWidth, innerHeight); }
    cloud.maxPoints = p.maxPts;
    applyVisibility();
    toast(`PERF ${settings.perf.toUpperCase()} (AA / bowl mesh after reload)`);
  }
  const actions = {
    "btn-bloom": () => { settings.bloom = !settings.bloom && setupBloom(); save(); applyVisibility(); if (!composer && settings.bloom === false) toast("bloom: addon not loaded"); },
    "btn-night": () => toggle("night"),
    "btn-thermal": () => toggle("thermal"),
    "btn-bowl": () => toggle("bowl"),
    "btn-cloud": () => toggle("cloud"),
    "btn-color": () => { settings.colorMode = (settings.colorMode + 1) % 3; save(); applyVisibility(); },
    "btn-perf": cyclePerf,
    "btn-help": () => $("help").classList.toggle("open"),
  };
  for (const [id, fn] of Object.entries(actions)) { const b = $(id); if (b) b.addEventListener("click", fn); }
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(+b.dataset.view)));

  addEventListener("keydown", (e) => {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); if (!e.repeat) estop(); return; }
    const k = e.key.toLowerCase();
    if (k >= "1" && k <= "5") {
      const m = +k;
      if (m === 4 && rig.mode === 4 && state.cams.length) {   // cycle FPV cameras (skip ToF)
        let i = rig.fpvCam;
        for (let n = 0; n < state.cams.length; n++) { i = (i + 1) % state.cams.length; if (state.cams[i].modality !== "depth") break; }
        enterFpv(i);
      } else setView(m);
    } else if (k === "n") actions["btn-night"]();
    else if (k === "t") actions["btn-thermal"]();
    else if (k === "b") actions["btn-bloom"]();
    else if (k === "c") actions["btn-color"]();
    else if (k === "v") actions["btn-bowl"]();
    else if (k === "k") actions["btn-cloud"]();
    else if (k === "p") actions["btn-perf"]();
    else if (k === "h" || k === "?") actions["btn-help"]();
    else if (k === "escape") persons.select(null);
  });

  function onResize() {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    if (composer) composer.setSize(innerWidth, innerHeight);
    cloud.setScale(innerHeight * renderer.getPixelRatio(), camera.fov);
  }
  addEventListener("resize", onResize);

  if (settings.bloom) settings.bloom = setupBloom();
  applyVisibility();
  setView(1);
  onResize();

  // ---------------- main loop ----------------
  const clock = new THREE.Clock();
  let frames = 0, fpsT = 0, fps = 0, hudT = 0, radarT = 0, camFpsT = 0;
  const robotPos = new THREE.Vector3();
  function healthInfo(h) {
    if (!h || typeof h !== "object") return { gpu: null, temp: null, cams: {} };
    const g = h.gpu && typeof h.gpu === "object" ? h.gpu : {};
    const gpu = h.gpu_util ?? h.gpu_percent ?? g.util ?? g.load ?? g.percent ?? null;
    const temp = h.gpu_temp ?? g.temp ?? g.temp_c ?? h.temp_c ?? null;
    // omni /stats: {cameras:{id:{available,fps,age_s,...}}}; /health: {cams_down:[...]}
    let cams = (h._stats && h._stats.cameras) || h.cams || h.cameras || (h.capture && (h.capture.cams || h.capture)) || {};
    if (Array.isArray(cams)) cams = Object.fromEntries(cams.map((c) => [c.id || c.cam_id, c]));
    const down = new Set(Array.isArray(h.cams_down) ? h.cams_down : []);
    const out = {};
    for (const [id, c] of Object.entries(cams)) {
      const ok = !down.has(id) && c.available !== false && c.ok !== false && !(c.age_s > 3);
      out[id] = { ...c, ok };
    }
    for (const id of down) if (!out[id]) out[id] = { fps: 0, ok: false };
    return { gpu, temp, cams: out, voxels: h._stats && h._stats.voxels };
  }

  function loop() {
    requestAnimationFrame(loop);
    const dt = Math.min(clock.getDelta(), 0.1);
    const t = clock.elapsedTime;
    flushFrames();

    // robot pose -> scene
    F.b2t(state.pose.x, state.pose.y, 0, robotPos);
    robotYaw.position.lerp(robotPos, 1 - Math.exp(-dt * 12));
    let dy = state.pose.yaw - robotYaw.rotation.y;
    dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    robotYaw.rotation.y += dy * (1 - Math.exp(-dt * 12));
    rim.position.set(robotYaw.position.x, 1.2, robotYaw.position.z);
    gridMat.uniforms.uCenter.value.set(robotYaw.position.x, robotYaw.position.z);
    gridMat.uniforms.uTime.value = t;
    robot.setJoints(state.joints, t);

    // safety rings
    const lvl = state.safety && state.safety.level;
    const pulse = 0.5 + 0.5 * Math.sin(t * (lvl === "STOP" ? 12 : 6));
    for (let i = 0; i < rings.length; i++) {
      const m = rings[i];
      const active = (lvl === "STOP" && i === 0) || (lvl === "SLOW" && i === 1) || (lvl === "CAUTION" && i === 2);
      const alert = lvl === "STOP" || lvl === "SLOW";
      m.material.opacity = active ? 0.55 + 0.45 * pulse : 0.35;
      m.scale.setScalar(active && alert ? 1 + 0.03 * pulse : 1);
      m.userData.fill.material.opacity = active ? (alert ? 0.10 + 0.14 * pulse : 0.06) : 0;
    }

    bowl.tick(t);
    cloud.tick(t);
    persons.tick(dt, performance.now() / 1000, camera, innerWidth, innerHeight);
    rig.update(dt, { pos: robotYaw.position, yaw: robotYaw.rotation.y, baseH: F.BASE_HEIGHT }, state.cams);

    if (settings.bloom && composer) composer.render(dt); else renderer.render(scene, camera);
    while (toClose.length) { const b = toClose.pop(); try { b.close(); } catch (_) {} }

    frames++;
    if (t - fpsT >= 1) { fps = frames / (t - fpsT); frames = 0; fpsT = t; }
    if (t - camFpsT >= 1) {
      const span = t - camFpsT; camFpsT = t;
      for (const s of state.camFps.values()) { s.fps = s.n / span; s.n = 0; }
    }
    if (t - radarT > 0.05) { radarT = t; hud.drawRadar(persons.list, state.target, t); }
    if (t - hudT > 0.1) { hudT = t; updateHud(); }
  }

  function updateHud() {
    const hi = healthInfo(state.health);
    const stats = {};
    for (const c of state.cams) {
      const local = state.camFps.get(c.idx);
      const h = hi.cams[c.id] || {};
      const fpsV = state.source === "live" && local && local.fps > 0 ? local.fps : (h.fps ?? (local ? local.fps : 0));
      if (h.fps != null || local) stats[c.id] = { fps: fpsV, ok: h.ok !== undefined ? h.ok : h.healthy !== undefined ? h.healthy : fpsV > 0.5 };
    }
    hud.setCamStats(stats);
    // safety: real state, else (live) local estimate from persons
    let s = state.safety, src = state.safetySrc;
    if (!s && persons.list.length && state.source === "live") {
      const near = Math.min(...persons.list.map((p) => p.range));
      const level = near < F.ZONES.STOP ? "STOP" : near < F.ZONES.SLOW ? "SLOW" : near < F.ZONES.CAUTION ? "CAUTION" : "CLEAR";
      s = { level, vmax: null, nearest_person_m: near, reason: "local estimate" }; src = "UI estimate (safety offline)";
    }
    hud.setSafety(s, src);
    hud.setBattery(state.battery);
    hud.setText("speed", `${state.speed.toFixed(2)} m/s`);
    hud.setText("gpu", hi.gpu != null ? `${Math.round(hi.gpu)}%` : "--%");
    hud.setText("gputemp", hi.temp != null ? `${Math.round(hi.temp)}°C` : "--°C");
    hud.setText("voxels", `${(cloud.count / 1000).toFixed(cloud.count < 100000 ? 1 : 0)}k${cloud.dropped ? " CAP" : ""}`);
    hud.setText("fps", `${fps.toFixed(0)}`);
    hud.setText("npersons", `${persons.list.length}`);
    for (const [k, v] of Object.entries(state.links)) hud.setLink(k, v);
    if (state.target != null) {
      const p = persons.list.find((q) => q.gid === state.target);
      hud.setText("target-range", p ? `${p.range.toFixed(1)} m · ${p.zone.toUpperCase()}` : "LOST");
    }
  }

  window.__omni = { state, settings, cloud, bowl, persons, rig, scene, setView, estop };   // debug hook
  requestAnimationFrame(loop);
}
