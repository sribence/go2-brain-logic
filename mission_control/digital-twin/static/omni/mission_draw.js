// OmniView mission — 3D overlays: speed-coloured path ribbon, goal arrow,
// jump landing rings, freehand / zone drafts, gamepad cursor, waypoint
// markers (fly-by zone circles, `fine` crosshair + pose arrow).
// World (x, y) -> three (x, h, -y).

const W = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

// speed -> colour (blue slow … cyan … green … yellow … red sprint)
const STOPS = [[0.0, 0x2a6bff], [0.25, 0x00d8ff], [0.6, 0x35f28a], [1.0, 0xffd23a], [1.5, 0xff3344]];
export function speedColor(v, out) {
  const c = out || new THREE.Color();
  if (!(v > STOPS[0][0])) return c.setHex(STOPS[0][1]);
  for (let i = 1; i < STOPS.length; i++) {
    if (v <= STOPS[i][0]) {
      const f = (v - STOPS[i - 1][0]) / (STOPS[i][0] - STOPS[i - 1][0]);
      return c.setHex(STOPS[i - 1][1]).lerp(new THREE.Color(STOPS[i][1]), f);
    }
  }
  return c.setHex(STOPS[STOPS.length - 1][1]);
}
export const SPEED_LEGEND = STOPS.map(([v, c]) => [v, `#${c.toString(16).padStart(6, "0")}`]);

const RIBBON_VERT = `attribute vec3 aColor; attribute float aU; varying vec3 vC; varying float vU; varying float vS;
attribute float aSide;
void main(){ vC=aColor; vU=aU; vS=aSide; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`;
const RIBBON_FRAG = `uniform float uTime; uniform float uAlpha; varying vec3 vC; varying float vU; varying float vS;
void main(){ float edge = 1.0 - smoothstep(0.55, 1.0, abs(vS));
  float flow = smoothstep(0.75, 1.0, fract(vU*1.6 - uTime*1.2));
  gl_FragColor = vec4(vC*(0.75+0.9*flow), (0.35 + 0.55*edge + 0.4*flow) * uAlpha); }`;

export class MissionDraw {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.root.renderOrder = 3;
    scene.add(this.root);
    this.ribbonMat = new THREE.ShaderMaterial({
      vertexShader: RIBBON_VERT, fragmentShader: RIBBON_FRAG, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide, uniforms: { uTime: { value: 0 }, uAlpha: { value: 1 } },
    });
    this.path = null; this.pathLine = null;
    this.landings = new THREE.Group(); this.root.add(this.landings);
    this.wps = new THREE.Group(); this.root.add(this.wps);
    this._goal();
    this._cursor();
    this.draft = null;
  }

  _goal() {
    const g = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: 0x00d8ff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.34, 48), mat);
    ring.rotation.x = -Math.PI / 2;
    g.add(ring);
    const inner = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.08, 24), mat);
    inner.rotation.x = -Math.PI / 2;
    g.add(inner);
    // yaw arrow: flat triangle + shaft on the floor, pointing +x (rotated by yaw)
    const shape = new THREE.Shape();
    shape.moveTo(0.95, 0); shape.lineTo(0.62, 0.17); shape.lineTo(0.66, 0.05); shape.lineTo(0.3, 0.05);
    shape.lineTo(0.3, -0.05); shape.lineTo(0.66, -0.05); shape.lineTo(0.62, -0.17); shape.closePath();
    const arrow = new THREE.Mesh(new THREE.ShapeGeometry(shape), mat);
    arrow.rotation.x = -Math.PI / 2;
    const yawG = new THREE.Group(); yawG.add(arrow);
    g.add(yawG);
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.6, 6, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x00d8ff, transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false }));
    beam.position.y = 0.8;
    g.add(beam);
    g.position.y = 0.03; g.visible = false;
    this.goal = { g, mat, yawG, ring, beam };
    this.root.add(g);
  }

  _cursor() {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.18, 0.22, 4, 1), new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.y = 0.04; m.visible = false;
    this.cursor = m;
    this.root.add(m);
  }
  setCursor(x, y, on = true) { this.cursor.visible = on; if (on) this.cursor.position.set(x, 0.04, -y); }

  // x, y, yaw (null = no arrow); color hex
  setGoal(x, y, yaw, color = 0x00d8ff) {
    const G = this.goal;
    G.g.visible = true;
    G.g.position.set(x, 0.03, -y);
    G.yawG.visible = yaw != null;
    G.yawG.rotation.y = yaw || 0;
    G.mat.color.setHex(color);
    G.beam.material.color.setHex(color);
  }
  hideGoal() { this.goal.g.visible = false; }

  clearPath() {
    for (const o of [this.path, this.pathLine]) if (o) { this.root.remove(o); o.geometry.dispose(); }
    if (this.pathLine) this.pathLine.material.dispose();
    this.path = this.pathLine = null;
    this.landings.children.slice().forEach((c) => { this.landings.remove(c); c.traverse((q) => { if (q.geometry) q.geometry.dispose(); }); });
  }

  // plan: {path:[[x,y]], v:[...], landings:[[x,y]]}; dim: preview vs executing
  setPlan(plan, { alpha = 1 } = {}) {
    this.clearPath();
    if (!plan) return;
    const P = plan.path || [], V = plan.v || [];
    if (P.length >= 2) {
      const n = P.length, w = 0.09;
      const pos = new Float32Array(n * 2 * 3), col = new Float32Array(n * 2 * 3), u = new Float32Array(n * 2), side = new Float32Array(n * 2);
      const idx = [];
      let acc = 0;
      const c = new THREE.Color();
      for (let i = 0; i < n; i++) {
        const a = P[Math.max(i - 1, 0)], b = P[Math.min(i + 1, n - 1)];
        let tx = b[0] - a[0], ty = b[1] - a[1];
        const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
        const nx = -ty * w, ny = tx * w;
        if (i) acc += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
        speedColor(V[i] || 0, c);
        for (let s = 0; s < 2; s++) {
          const k = i * 2 + s, sg = s ? -1 : 1;
          pos[k * 3] = P[i][0] + nx * sg; pos[k * 3 + 1] = 0.035; pos[k * 3 + 2] = -(P[i][1] + ny * sg);
          col[k * 3] = c.r; col[k * 3 + 1] = c.g; col[k * 3 + 2] = c.b;
          u[k] = acc; side[k] = sg;
        }
        if (i < n - 1) { const k = i * 2; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
      geo.setAttribute("aU", new THREE.BufferAttribute(u, 1));
      geo.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
      geo.setIndex(idx);
      this.ribbonMat.uniforms.uAlpha.value = alpha;
      this.path = new THREE.Mesh(geo, this.ribbonMat);
      this.path.frustumCulled = false;
      this.root.add(this.path);
      // thin centre line (for top-down readability)
      const lp = new Float32Array(n * 3), lc = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { lp[i * 3] = P[i][0]; lp[i * 3 + 1] = 0.04; lp[i * 3 + 2] = -P[i][1]; speedColor(V[i] || 0, c); lc[i * 3] = c.r; lc[i * 3 + 1] = c.g; lc[i * 3 + 2] = c.b; }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.BufferAttribute(lp, 3));
      lg.setAttribute("color", new THREE.BufferAttribute(lc, 3));
      this.pathLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 * alpha }));
      this.pathLine.frustumCulled = false;
      this.root.add(this.pathLine);
    }
    // jump landings: double ring + number tick + arc from the previous point
    const L = plan.landings || [];
    let prev = P.length ? P[0] : null;
    L.forEach((l, i) => {
      const g = new THREE.Group();
      g.position.set(l[0], 0.03, -l[1]);
      const m1 = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.25, 40), new THREE.MeshBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false }));
      m1.rotation.x = -Math.PI / 2;
      const m2 = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.32, 40), new THREE.MeshBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
      m2.rotation.x = -Math.PI / 2;
      g.add(m1, m2);
      g.userData = { i, m2 };
      this.landings.add(g);
      if (prev) {
        const pts = [];
        for (let k = 0; k <= 16; k++) { const f = k / 16; pts.push(W(prev[0] + (l[0] - prev[0]) * f, prev[1] + (l[1] - prev[1]) * f, 0.05 + 4 * 0.3 * f * (1 - f))); }
        const arc = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0xffb020, dashSize: 0.06, gapSize: 0.04, transparent: true, opacity: 0.85 }));
        arc.computeLineDistances();
        this.landings.add(arc);
      }
      prev = l;
    });
  }

  // in-progress polyline (freehand / zone). closed: draw closing edge
  setDraft(points, color = 0x00d8ff, closed = false) {
    if (this.draft) { this.root.remove(this.draft); this.draft.geometry.dispose(); this.draft.material.dispose(); this.draft = null; }
    if (!points || points.length < 1) return;
    const pts = points.map((p) => W(p[0], p[1], 0.05));
    if (closed && pts.length > 2) pts.push(pts[0].clone());
    if (pts.length === 1) pts.push(pts[0].clone().add(new THREE.Vector3(0.01, 0, 0)));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    this.draft = new THREE.Line(g, new THREE.LineDashedMaterial({ color, dashSize: 0.12, gapSize: 0.06, transparent: true, opacity: 0.95 }));
    this.draft.computeLineDistances();
    this.draft.frustumCulled = false;
    this.root.add(this.draft);
  }

  // waypoints: [{x, y, yaw?, zone, n}] — fly-by circle (radius = zXX cm) or fine crosshair + arrow
  setWaypoints(list) {
    this.wps.children.slice().forEach((c) => { this.wps.remove(c); c.traverse((q) => { if (q.geometry) q.geometry.dispose(); if (q.material) q.material.dispose(); }); });
    for (const w of list || []) {
      const g = new THREE.Group();
      g.position.set(w.x, 0.025, -w.y);
      const fine = w.zone === "fine";
      const col = fine ? 0xff2bd6 : 0x00d8ff;
      if (fine) {
        const m = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false });
        for (const r of [0, Math.PI / 2]) {
          const bar = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.02), m);
          bar.rotation.set(-Math.PI / 2, 0, r);
          g.add(bar);
        }
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.11, 0.13, 32), m);
        ring.rotation.x = -Math.PI / 2;
        g.add(ring);
        if (w.yaw != null) {
          const a = new THREE.ArrowHelper(new THREE.Vector3(Math.cos(w.yaw), 0, -Math.sin(w.yaw)), new THREE.Vector3(0, 0.01, 0), 0.55, col, 0.14, 0.09);
          g.add(a);
        }
      } else {
        const r = Math.max(+String(w.zone || "z30").slice(1) / 100 || 0.3, 0.1);
        const disc = new THREE.Mesh(new THREE.CircleGeometry(r, 48), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
        disc.rotation.x = -Math.PI / 2;
        const edge = new THREE.Mesh(new THREE.RingGeometry(r - 0.015, r, 48), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide }));
        edge.rotation.x = -Math.PI / 2;
        g.add(disc, edge);
      }
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.5, 4), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.6 }));
      pin.position.y = 0.25;
      g.add(pin);
      this.wps.add(g);
    }
  }

  tick(t) {
    this.ribbonMat.uniforms.uTime.value = t;
    const G = this.goal;
    if (G.g.visible) { G.ring.scale.setScalar(1 + 0.08 * Math.sin(t * 5)); G.beam.material.opacity = 0.3 + 0.2 * Math.sin(t * 3); }
    for (const c of this.landings.children) if (c.userData.m2) {
      const s = 1 + 0.25 * ((t * 1.2 + c.userData.i * 0.2) % 1);
      c.userData.m2.scale.setScalar(s);
      c.userData.m2.material.opacity = 0.6 * (1 - ((t * 1.2 + c.userData.i * 0.2) % 1));
    }
    if (this.cursor.visible) this.cursor.rotation.z = t * 2;
  }
}
