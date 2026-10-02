// OmniView mission — semantic world layer: zones as translucent extruded walls
// (no_go red / watch amber / patrol cyan), labels as floating tags, home marker.
// DOM tags live in the shared #labels overlay (class .mtag).

export const ZONE_COLORS = { no_go: 0xff3344, watch: 0xffb020, patrol: 0x00d8ff };
const WALL_H = { no_go: 1.1, watch: 0.8, patrol: 0.5 };

const WALL_VERT = `varying float vH; void main(){ vH = position.y; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`;
const WALL_FRAG = `uniform vec3 uColor; uniform float uH; uniform float uTime; varying float vH;
void main(){ float f = clamp(vH/uH, 0.0, 1.0);
  float a = (1.0 - f) * 0.38 + smoothstep(0.96, 1.0, f) * 0.7 + 0.12 * smoothstep(0.9, 1.0, sin(vH*28.0 - uTime*3.0));
  gl_FragColor = vec4(uColor, a); }`;

export function centroid(poly) {
  let x = 0, y = 0;
  for (const p of poly) { x += p[0]; y += p[1]; }
  return [x / poly.length, y / poly.length];
}

export class WorldLayer {
  constructor(scene, labelRoot) {
    this.scene = scene;
    this.labelRoot = labelRoot;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.items = [];            // {obj, tag, anchor: Vector3, mats:[]}
    this.zones = []; this.labels = []; this.home = null;
  }

  _clear() {
    for (const it of this.items) {
      this.root.remove(it.obj);
      it.obj.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
      if (it.tag) it.tag.remove();
    }
    this.items = [];
  }

  _tag(html, kind, anchor) {
    const el = document.createElement("div");
    el.className = "mtag"; el.dataset.kind = kind; el.innerHTML = html;
    this.labelRoot.appendChild(el);
    return el;
  }

  set({ zones, labels, home } = {}) {
    if (zones) this.zones = zones;
    if (labels) this.labels = labels;
    if (home !== undefined) this.home = home;
    this._clear();
    for (const z of this.zones) if (Array.isArray(z.polygon) && z.polygon.length >= 3) this._zone(z);
    for (const l of this.labels) if (Number.isFinite(+l.x)) this._label(l);
    if (this.home && Number.isFinite(+this.home.x)) this._home(this.home);
  }

  _zone(z) {
    const col = ZONE_COLORS[z.kind] || 0x00d8ff, H = WALL_H[z.kind] || 0.8;
    const g = new THREE.Group();
    const poly = z.polygon;
    const pos = [], idx = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length], k = i * 4;
      pos.push(a[0], 0, -a[1], b[0], 0, -b[1], b[0], H, -b[1], a[0], H, -a[1]);
      idx.push(k, k + 1, k + 2, k, k + 2, k + 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    const wm = new THREE.ShaderMaterial({ vertexShader: WALL_VERT, fragmentShader: WALL_FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending, uniforms: { uColor: { value: new THREE.Color(col) }, uH: { value: H }, uTime: { value: 0 } } });
    g.add(new THREE.Mesh(geo, wm));
    const shape = new THREE.Shape(poly.map((p) => new THREE.Vector2(p[0], p[1])));
    const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: z.kind === "no_go" ? 0.13 : 0.07, depthWrite: false, side: THREE.DoubleSide }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = 0.008;
    g.add(floor);
    const edge = poly.map((p) => new THREE.Vector3(p[0], 0.02, -p[1])); edge.push(edge[0].clone());
    const top = poly.map((p) => new THREE.Vector3(p[0], H, -p[1])); top.push(top[0].clone());
    const lm = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9 });
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(edge), lm));
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(top), lm.clone()));
    this.root.add(g);
    const c = centroid(poly);
    const kindTxt = { no_go: "NO-GO", watch: "WATCH", patrol: "PATROL" }[z.kind] || z.kind;
    const tag = this._tag(`<b>${kindTxt}</b> ${esc(z.name || z.id || "")}${z.active_hours ? ` <i>${esc(z.active_hours)}</i>` : ""}`, z.kind, null);
    this.items.push({ obj: g, tag, anchor: new THREE.Vector3(c[0], H + 0.15, -c[1]), wall: wm });
  }

  _label(l) {
    const g = new THREE.Group();
    g.position.set(+l.x, 0, -l.y);
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 1.2, 6), new THREE.MeshBasicMaterial({ color: 0x35f28a, transparent: true, opacity: 0.7 }));
    pin.position.y = 0.6;
    const dia = new THREE.Mesh(new THREE.OctahedronGeometry(0.09), new THREE.MeshBasicMaterial({ color: 0x35f28a, wireframe: true }));
    dia.position.y = 1.25;
    const base = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.16, 6), new THREE.MeshBasicMaterial({ color: 0x35f28a, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
    base.rotation.x = -Math.PI / 2; base.position.y = 0.02;
    g.add(pin, dia, base);
    this.root.add(g);
    const tag = this._tag(`◆ ${esc(l.name)}`, "label", null);
    this.items.push({ obj: g, tag, anchor: new THREE.Vector3(+l.x, 1.45, -l.y), spin: dia });
  }

  _home(h) {
    const g = new THREE.Group();
    g.position.set(+h.x, 0.02, -h.y);
    const m = new THREE.MeshBasicMaterial({ color: 0x35f28a, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.47, 6), m);
    ring.rotation.x = -Math.PI / 2;
    const yawG = new THREE.Group(); yawG.rotation.y = +h.yaw || 0;
    const tri = new THREE.Mesh(new THREE.CircleGeometry(0.16, 3), m);
    tri.rotation.x = -Math.PI / 2; tri.position.x = 0.6;
    yawG.add(tri);
    g.add(ring, yawG);
    this.root.add(g);
    const tag = this._tag("⌂ HOME", "home", null);
    this.items.push({ obj: g, tag, anchor: new THREE.Vector3(+h.x, 0.5, -h.y), spin: ring });
  }

  tick(t, camera, w, h) {
    const v = new THREE.Vector3();
    for (const it of this.items) {
      if (it.wall) it.wall.uniforms.uTime.value = t;
      if (it.spin) it.spin.rotation.y = t;
      if (!it.tag) continue;
      v.copy(it.anchor).project(camera);
      if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1) { it.tag.style.display = "none"; continue; }
      it.tag.style.display = "block";
      it.tag.style.transform = `translate(-50%,-100%) translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px)`;
    }
  }
}

export function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
