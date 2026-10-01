// OmniView — persons as holograms: capsule, label, velocity arrow, dashed
// 2 s prediction, zone color, click-to-track reticle.

import { b2t, baseToWorld, rotVec, zoneOf, ZONE_COLOR } from "./frames.js";

const HOLO_VERT = /* glsl */`
varying vec3 vN; varying vec3 vV; varying float vY;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vY = wp.y;
  vN = normalize(normalMatrix * normal);
  vec4 mv = viewMatrix * wp;
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;
const HOLO_FRAG = /* glsl */`
uniform vec3 uColor; uniform float uTime; uniform float uSel;
varying vec3 vN; varying vec3 vV; varying float vY;
void main() {
  float fr = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.0);
  float scan = 0.5 + 0.5 * sin(vY * 70.0 - uTime * 8.0);
  float band = smoothstep(0.92, 1.0, sin(vY * 3.0 - uTime * 2.5));
  float a = 0.10 + 0.75 * fr + 0.12 * scan + 0.35 * band + 0.2 * uSel;
  gl_FragColor = vec4(uColor * (0.6 + 0.8 * fr + 0.6 * band), clamp(a, 0.0, 1.0));
}`;

function capsuleGeometry(r = 0.24, h = 1.72) {
  const pts = [];
  const body = h - 2 * r;
  for (let i = 0; i <= 8; i++) { const a = -Math.PI / 2 + (i / 8) * (Math.PI / 2); pts.push(new THREE.Vector2(Math.cos(a) * r + 1e-4, r + Math.sin(a) * r)); }
  for (let i = 0; i <= 8; i++) { const a = (i / 8) * (Math.PI / 2); pts.push(new THREE.Vector2(Math.cos(a) * r + 1e-4, r + body + Math.sin(a) * r)); }
  return new THREE.LatheGeometry(pts, 24);
}

const MOD_ICON = { rgb: "◉RGB", thermal: "♨TH", depth: "▦ToF", tof: "▦ToF", lidar: "⌁LiD" };

export class PersonLayer {
  constructor(scene, labelRoot, { onSelect } = {}) {
    this.scene = scene;
    this.labelRoot = labelRoot;
    this.onSelect = onSelect || (() => {});
    this.items = new Map();     // gid -> item
    this.capGeo = capsuleGeometry();
    this.ringGeo = new THREE.RingGeometry(0.32, 0.4, 40);
    this.selected = null;
    this.list = [];             // latest [{gid, wx, wy, range, zone, ...}] for radar/HUD
    this.lastMsgT = 0;
    // target reticle
    const ret = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.62, 48), new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2;
    ret.add(ring);
    for (let i = 0; i < 4; i++) {
      const tick = new THREE.Mesh(new THREE.PlaneGeometry(0.06, 0.28), ring.material);
      tick.rotation.x = -Math.PI / 2;
      const a = (i * Math.PI) / 2;
      tick.position.set(Math.cos(a) * 0.78, 0, Math.sin(a) * 0.78);
      tick.rotation.z = a + Math.PI / 2;
      ret.add(tick);
    }
    ret.position.y = 0.03;
    ret.visible = false;
    this.reticle = ret;
    scene.add(ret);
  }

  _make(gid) {
    const mat = new THREE.ShaderMaterial({
      vertexShader: HOLO_VERT, fragmentShader: HOLO_FRAG,
      uniforms: { uColor: { value: new THREE.Color(0x35f28a) }, uTime: { value: 0 }, uSel: { value: 0 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    const g = new THREE.Group();
    const cap = new THREE.Mesh(this.capGeo, mat);
    cap.userData.gid = gid;
    g.add(cap);
    const ring = new THREE.Mesh(this.ringGeo, new THREE.MeshBasicMaterial({ color: 0x35f28a, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02;
    g.add(ring);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.9, 0), 0.5, 0x35f28a, 0.18, 0.12);
    g.add(arrow);
    const pathGeo = new THREE.BufferGeometry();
    pathGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(21 * 3), 3));
    const path = new THREE.Line(pathGeo, new THREE.LineDashedMaterial({ color: 0x35f28a, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.85 }));
    this.scene.add(path);
    this.scene.add(g);
    const label = document.createElement("div");
    label.className = "plabel";
    label.addEventListener("click", (e) => { e.stopPropagation(); this.select(gid); });
    this.labelRoot.appendChild(label);
    const seed = typeof gid === "number" ? gid : String(gid).length;
    const it = { gid, seed, g, cap, mat, ring, arrow, path, label, pos: null, target: new THREE.Vector3(), vel: new THREE.Vector3(), seen: 0, data: null, zone: "green" };
    this.items.set(gid, it);
    return it;
  }

  _remove(it) {
    this.scene.remove(it.g); this.scene.remove(it.path);
    it.mat.dispose(); it.ring.material.dispose(); it.path.geometry.dispose(); it.path.material.dispose();
    it.label.remove();
    this.items.delete(it.gid);
    if (this.selected === it.gid) this.select(null);
  }

  // msg: {t, persons:[{gid,x,y,z,vx,vy,conf,cams,modality,range_src,range_m,age_s}]} in BASE frame
  update(msg, pose, nowS) {
    if (!msg || !Array.isArray(msg.persons)) return;
    this.lastMsgT = nowS;
    const list = [];
    for (const p of msg.persons) {
      if (p == null || !Number.isFinite(+p.x) || !Number.isFinite(+p.y)) continue;
      const gid = p.gid != null ? p.gid : `${p.x.toFixed(1)}_${p.y.toFixed(1)}`;
      const it = this.items.get(gid) || this._make(gid);
      const w = baseToWorld(pose, +p.x, +p.y);
      const v = rotVec(pose.yaw, +p.vx || 0, +p.vy || 0);
      const range = Number.isFinite(+p.range_m) && +p.range_m > 0 ? +p.range_m : Math.hypot(+p.x, +p.y);
      b2t(w.x, w.y, 0, it.target);
      b2t(v.x, v.y, 0, it.vel);
      if (!it.pos) it.pos = it.target.clone();
      it.seen = nowS; it.data = p; it.range = range;
      const hm = +p.height_m;
      it.cap.scale.y = hm > 0.3 && hm < 2.5 ? hm / 1.72 : 1;   // capsule geometry is 1.72 m tall
      it.zone = zoneOf(range);
      list.push({ gid, bx: +p.x, by: +p.y, wx: w.x, wy: w.y, range, zone: it.zone, conf: +p.conf || 0, modality: p.modality || [], cams: p.cams || [], range_src: p.range_src || "?" });
    }
    this.list = list;
  }

  pick(raycaster) {
    const caps = [];
    for (const it of this.items.values()) caps.push(it.cap);
    const hit = raycaster.intersectObjects(caps, false)[0];
    return hit ? hit.object.userData.gid : null;
  }

  select(gid) {
    this.selected = gid;
    this.reticle.visible = gid != null;
    this.onSelect(gid);
  }

  // t: same clock as update() (performance.now()/1000)
  tick(dt, t, camera, w, h) {
    const k = 1 - Math.exp(-dt * 10);
    const tmp = new THREE.Vector3();
    for (const it of [...this.items.values()]) {
      if (t - it.seen > 1.2) { this._remove(it); continue; }
      const fade = t - it.seen > 0.5 ? 0.4 : 1;
      it.pos.lerp(it.target, k);
      it.g.position.copy(it.pos);
      const col = ZONE_COLOR[it.zone];
      it.mat.uniforms.uColor.value.setHex(col);
      it.mat.uniforms.uTime.value = t + it.seed * 0.37;
      it.mat.uniforms.uSel.value = this.selected === it.gid ? 1 : 0;
      it.ring.material.color.setHex(col);
      it.ring.material.opacity = 0.8 * fade;
      it.ring.scale.setScalar(1 + 0.15 * Math.sin(t * 4 + it.seed));
      // velocity arrow
      const sp = it.vel.length();
      if (sp > 0.08) {
        it.arrow.visible = true;
        it.arrow.setDirection(tmp.copy(it.vel).normalize());
        it.arrow.setLength(Math.min(0.25 + sp * 0.8, 2.5), 0.18, 0.12);
        it.arrow.setColor(col);
      } else it.arrow.visible = false;
      // 2 s predicted path (constant velocity)
      const arr = it.path.geometry.attributes.position.array;
      for (let i = 0; i <= 20; i++) {
        const s = (i / 20) * 2.0;
        arr[i * 3] = it.pos.x + it.vel.x * s; arr[i * 3 + 1] = 0.05; arr[i * 3 + 2] = it.pos.z + it.vel.z * s;
      }
      it.path.geometry.attributes.position.needsUpdate = true;
      it.path.geometry.computeBoundingSphere();
      it.path.computeLineDistances();
      it.path.material.color.setHex(col);
      it.path.visible = sp > 0.08;
      // label
      tmp.copy(it.pos); tmp.y += 1.72 * it.cap.scale.y + 0.23;
      tmp.project(camera);
      const lab = it.label;
      if (tmp.z > 1 || tmp.z < -1 || Math.abs(tmp.x) > 1.2 || Math.abs(tmp.y) > 1.2) { lab.style.display = "none"; }
      else {
        lab.style.display = "block";
        lab.style.transform = `translate(-50%,-100%) translate(${((tmp.x + 1) / 2) * w}px, ${((1 - tmp.y) / 2) * h}px)`;
        const d = it.data || {};
        const mods = (d.modality || []).map((m) => `<i>${MOD_ICON[m] || m}</i>`).join("");
        const html = `<b>#${it.gid}</b> <span class="rng">${it.range.toFixed(1)} m</span>${this.selected === it.gid ? ' <span class="tgt">TARGET</span>' : ""}<br><span class="mods">${mods}</span> <span class="src">${d.range_src || ""}</span>`;
        if (lab._html !== html) { lab.innerHTML = html; lab._html = html; }
        lab.dataset.zone = it.zone;
        lab.style.opacity = fade;
      }
      if (this.selected === it.gid) {
        this.reticle.position.set(it.pos.x, 0.03, it.pos.z);
        this.reticle.rotation.y = t * 1.5;
        const s = 1 + 0.08 * Math.sin(t * 6);
        this.reticle.scale.setScalar(s);
      }
    }
  }
}
