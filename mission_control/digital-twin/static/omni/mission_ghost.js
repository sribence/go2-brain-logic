// OmniView mission — "ghost dog": translucent clone of the Go2 model that runs
// the planned path at planned speed (time scrubber / loop playback).

import { sampleTimeline, HOP_S } from "./mission_mock.js";

const BASE_H = 0.32;

// timeline from a plan without one (live planner): path + v, then landings hops
export function buildTimeline(plan, pose) {
  if (plan && Array.isArray(plan.timeline) && plan.timeline.length) return plan.timeline;
  const tl = [];
  const P = (plan && plan.path) || [], V = (plan && plan.v) || [];
  let T = 0;
  const p0 = P[0] || [pose.x, pose.y];
  tl.push({ t: 0, x: p0[0], y: p0[1], yaw: pose.yaw, z: 0, step: 0 });
  const L = (plan && plan.landings) || [];
  if (L.length) {
    let [px, py] = p0;
    for (const l of L) {
      const yaw = Math.atan2(l[1] - py, l[0] - px);
      for (let k = 1; k <= 6; k++) { const f = k / 6; T += HOP_S / 6; tl.push({ t: T, x: px + (l[0] - px) * f, y: py + (l[1] - py) * f, yaw, z: 1.2 * f * (1 - f), step: 0 }); }
      [px, py] = l;
    }
    return tl;
  }
  for (let i = 1; i < P.length; i++) {
    const a = P[i - 1], b = P[i];
    const ds = Math.hypot(b[0] - a[0], b[1] - a[1]);
    T += ds / Math.max(((V[i] || 0) + (V[i - 1] || 0)) / 2, 0.05);
    tl.push({ t: T, x: b[0], y: b[1], yaw: Math.atan2(b[1] - a[1], b[0] - a[0]), z: 0, step: 0 });
  }
  return tl;
}

export class Ghost {
  constructor(scene, robot) {
    this.robot = robot;
    this.yawG = new THREE.Group();
    this.ros = new THREE.Group();
    this.ros.rotation.x = -Math.PI / 2;
    this.ros.position.y = BASE_H;
    this.yawG.add(this.ros);
    this.yawG.visible = false;
    scene.add(this.yawG);
    this.mat = new THREE.MeshBasicMaterial({ color: 0x7fe8ff, transparent: true, opacity: 0.28, depthWrite: false, blending: THREE.AdditiveBlending });
    this.model = null; this.fromLoaded = false;
    // ground shadow disc
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.32, 32), new THREE.MeshBasicMaterial({ color: 0x00d8ff, transparent: true, opacity: 0.18, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; disc.position.y = 0.012;
    this.disc = disc;
    scene.add(disc); disc.visible = false;
    this.tl = null; this.T = 0; this.t = 0; this.playing = true; this.rate = 1;
    this.onTime = null;
  }

  _ensureModel() {
    const src = this.robot && this.robot.root;
    if (!src || (this.model && (this.fromLoaded || !this.robot.loaded))) return;
    if (this.model) this.ros.remove(this.model);
    const m = src.clone(true);
    m.traverse((o) => { if (o.isMesh) { o.material = this.mat; o.castShadow = false; } });
    this.model = m; this.fromLoaded = !!this.robot.loaded;
    this.ros.add(m);
  }

  setTimeline(tl) {
    this.tl = tl && tl.length > 1 ? tl : null;
    this.T = this.tl ? this.tl[this.tl.length - 1].t : 0;
    this.t = 0;
    this.yawG.visible = this.disc.visible = !!this.tl;
    if (this.tl) { this._ensureModel(); this._apply(); }
  }
  clear() { this.setTimeline(null); }
  seek(f) { this.t = Math.max(0, Math.min(1, f)) * this.T; this._apply(); }

  _apply() {
    const s = sampleTimeline(this.tl, this.t);
    if (!s) return;
    this.yawG.position.set(s.x, s.z || 0, -s.y);
    this.yawG.rotation.y = s.yaw;
    this.disc.position.set(s.x, 0.012, -s.y);
    this.cur = s;
    if (this.onTime) this.onTime(this.t, this.T, s);
  }

  // v: planned speed at the ghost (for HUD) — caller computes from plan
  tick(dt, t) {
    if (!this.tl) return;
    if (this.playing) {
      this.t += dt * this.rate;
      if (this.t > this.T + 1.2) this.t = 0;     // short hold at the goal, then loop
      this._apply();
    }
    this.mat.opacity = 0.22 + 0.1 * Math.sin(t * 6);
  }
}
