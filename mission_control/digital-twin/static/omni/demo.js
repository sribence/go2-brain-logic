// OmniView — built-in DEMO source (omni.html?demo=1, or automatic fallback
// when :9114 is unreachable). Synthesizes a night courtyard rendered through
// the SAME camera models the bowl shader uses (so stitching is verifiable),
// moving persons, a growing OVX1 point cloud, safety levels and health.

import { BASE_HEIGHT, ZONES, camPoseRows, normalizeRig, projectCam, unprojectCam, baseToCam } from "./frames.js";
import { encodeOVX1 } from "./pointcloud.js";

const ZF = -BASE_HEIGHT;            // floor z in base frame
const WALL = { x: 9, y: 7, h: 4.2 };
const BOXES = [
  { min: [3.0, -2.6, ZF], max: [4.2, -1.4, ZF + 1.0], col: [170, 92, 28], hot: true },   // "generator"
  { min: [-5.2, 2.0, ZF], max: [-3.9, 3.4, ZF + 1.3], col: [60, 120, 150] },
  { min: [1.6, 4.0, ZF], max: [2.3, 4.7, ZF + 2.6], col: [120, 120, 130] },              // pillar
  { min: [-3.0, -4.6, ZF], max: [-1.8, -3.8, ZF + 0.8], col: [150, 140, 40] },
];
const LAMPS = [[5, 4.5], [-6, -4.5], [6.5, -5], [-5, 5]].map(([x, y]) => [x, y, ZF + 3.6]);
const NEON = [[0, 230, 255], [255, 40, 200], [255, 170, 30], [60, 255, 120]];
const CLOTHES = ["#d23b3b", "#3b7bd2", "#e0c341", "#9b59d0", "#3bd28f"];

function hash2(x, y) { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); }
const fract = (v) => v - Math.floor(v);

function f(w, hfovDeg) { return (w / 2) / Math.tan((hfovDeg * Math.PI) / 360); }

export function demoRigJSON() {
  const fish = (id, yaw, t, order) => ({
    id, order, modality: "rgb", model: "fisheye", width: 320, height: 180,
    K: [[95, 0, 160], [0, 95, 90], [0, 0, 1]], D: [0.02, -0.01, 0, 0],
    T_base_cam: camPoseRows(yaw, 22, t[0], t[1], t[2]),
  });
  const pin = (id, modality, w, h, hfov, yaw, pitch, t, order) => ({
    id, order, modality, model: "pinhole", width: w, height: h,
    K: [[f(w, hfov), 0, w / 2], [0, f(w, hfov), h / 2], [0, 0, 1]], D: [0, 0, 0, 0, 0],
    T_base_cam: camPoseRows(yaw, pitch, t[0], t[1], t[2]),
  });
  return {
    demo: true,
    cameras: [
      fish("rgb_front", 0, [0.32, 0, 0.05], 0),
      fish("rgb_left", 90, [0.0, 0.11, 0.06], 1),
      fish("rgb_right", -90, [0.0, -0.11, 0.06], 2),
      fish("rgb_rear", 180, [-0.32, 0, 0.05], 3),
      pin("th_front_narrow", "thermal", 160, 120, 30, 0, 6, [0.30, 0.03, 0.09], 4),
      pin("th_front_wide", "thermal", 160, 120, 70, 0, 12, [0.30, -0.03, 0.09], 5),
      pin("tof_rear", "depth", 100, 100, 70, 180, 25, [-0.30, 0, 0.0], 6),
    ],
  };
}

// ---- tiny ray tracer of the courtyard (base frame == world, robot parked at origin) ----
function rayScene(o, d) {
  let best = { t: Infinity, kind: "sky" };
  if (d[2] < -1e-6) {
    const t = (ZF - o[2]) / d[2];
    if (t > 0 && t < best.t) best = { t, kind: "floor" };
  }
  const walls = [[0, WALL.x, 0], [0, -WALL.x, 1], [1, WALL.y, 2], [1, -WALL.y, 3]];
  for (const [ax, v, id] of walls) {
    if (Math.abs(d[ax]) < 1e-6) continue;
    const t = (v - o[ax]) / d[ax];
    if (t <= 0 || t >= best.t) continue;
    const z = o[2] + d[2] * t;
    if (z < ZF || z > ZF + WALL.h) continue;
    best = { t, kind: "wall", id };
  }
  for (let b = 0; b < BOXES.length; b++) {
    const B = BOXES[b];
    let t0 = 0, t1 = best.t;
    let ok = true;
    for (let a = 0; a < 3; a++) {
      const inv = 1 / (d[a] || 1e-9);
      let ta = (B.min[a] - o[a]) * inv, tb = (B.max[a] - o[a]) * inv;
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
      if (t0 > t1) { ok = false; break; }
    }
    if (ok && t0 > 0 && t0 < best.t) best = { t: t0, kind: "box", id: b };
  }
  if (best.kind !== "sky") best.p = [o[0] + d[0] * best.t, o[1] + d[1] * best.t, o[2] + d[2] * best.t];
  return best;
}

function lampLight(p) {
  let l = 0;
  for (const L of LAMPS) {
    const dx = p[0] - L[0], dy = p[1] - L[1], dz = p[2] - L[2];
    l += 9 / (1 + dx * dx + dy * dy + dz * dz);
  }
  return l;
}

function shade(hit, d) {
  if (hit.kind === "sky") {
    const s = hash2(Math.floor(d[0] * 300), Math.floor(d[1] * 300 + d[2] * 300)) > 0.997 ? 180 : 0;
    const g = Math.max(0, d[2]);
    return [8 + s + 10 * (1 - g), 10 + s + 14 * (1 - g), 24 + s + 30 * (1 - g)];
  }
  const p = hit.p;
  const light = 0.32 + lampLight(p);
  let c;
  if (hit.kind === "floor") {
    const n = 22 + 16 * hash2(Math.floor(p[0] * 5), Math.floor(p[1] * 5));
    c = [n, n + 2, n + 6];
    const gx = Math.abs(fract(p[0] / 2) - 0.5), gy = Math.abs(fract(p[1] / 2) - 0.5);
    if (gx > 0.485 || gy > 0.485) c = [20, 120, 140];
    if (Math.abs(Math.abs(p[1]) - 3) < 0.07 && Math.abs(p[0]) < 6) c = [230, 160, 30];
    if (Math.abs(Math.hypot(p[0], p[1]) - 5.5) < 0.05) c = [200, 40, 160];
  } else if (hit.kind === "wall") {
    const along = hit.id < 2 ? p[1] : p[0];
    const z = p[2] - ZF;
    c = [38, 42, 56];
    if (Math.abs(fract(z) - 0.5) > 0.48 || Math.abs(fract(along / 3) - 0.5) > 0.49) c = [24, 26, 34];
    if (Math.abs(z - 2.2) < 0.07) return NEON[hit.id].map((v) => v);           // emissive neon strip
    if (z > 2.7 && z < 3.6 && fract(along / 2.5) > 0.2 && fract(along / 2.5) < 0.62) {
      const lit = hash2(Math.floor(along / 2.5), hit.id) > 0.45;
      if (lit) return [230, 190, 100];
      c = [12, 16, 26];
    }
  } else {
    const B = BOXES[hit.id];
    c = B.col.slice();
    const e = Math.min(
      Math.min(Math.abs(p[0] - B.min[0]), Math.abs(p[0] - B.max[0])),
      Math.min(Math.abs(p[1] - B.min[1]), Math.abs(p[1] - B.max[1])),
    );
    const ez = Math.min(Math.abs(p[2] - B.min[2]), Math.abs(p[2] - B.max[2]));
    if (e < 0.04 || ez < 0.04) c = c.map((v) => v * 0.35);
  }
  return c.map((v) => Math.min(255, v * light));
}

function camRayBase(cam, u, v) {
  const r = unprojectCam(cam, u, v);
  const T = cam.Trows;
  return {
    o: [T[0][3], T[1][3], T[2][3]],
    d: [T[0][0] * r[0] + T[0][1] * r[1] + T[0][2] * r[2],
        T[1][0] * r[0] + T[1][1] * r[1] + T[1][2] * r[2],
        T[2][0] * r[0] + T[2][1] * r[1] + T[2][2] * r[2]],
  };
}

// ---- persons ----
function tri(x) { const f2 = fract(x); return f2 < 0.5 ? f2 * 2 : 2 - f2 * 2; }
function personPos(i, t) {
  switch (i) {
    case 0: return [6 * Math.cos(0.13 * t), 6 * Math.sin(0.13 * t), true];
    case 1: return [0.55 + 7.5 * tri(t / 40 + 0.35), 0.45, true];
    case 2: return [-5 + 1.5 * Math.sin(0.21 * t), 2.5 * Math.sin(0.17 * t), true];
    default: return [2 * Math.sin(0.1 * t), 4.6 + 0.8 * Math.cos(0.23 * t), Math.sin(0.07 * t) > -0.4];
  }
}

export class DemoSource {
  constructor() {
    this.rigJSON = demoRigJSON();
    this.cams = normalizeRig(this.rigJSON);
    this.cams.forEach((c, i) => { c.Trows = this.rigJSON.cameras[i].T_base_cam; });
    this.t0 = performance.now() / 1000;
    this.version = 0;
    this.canvases = this.cams.map((c) => {
      const cv = document.createElement("canvas");
      cv.width = c.width; cv.height = c.height;
      return cv;
    });
    this.bg = this.cams.map((c, i) => this._renderBackground(c, this.canvases[i]));
    this.battery = 87.0;
    this.prev = new Map();
    this.persons = [];
    this.voxelQueue = 0;
  }

  get t() { return performance.now() / 1000 - this.t0; }

  _renderBackground(cam, cv) {
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(cam.width, cam.height);
    const D = img.data;
    for (let v = 0; v < cam.height; v++) {
      for (let u = 0; u < cam.width; u++) {
        const k = (v * cam.width + u) * 4;
        let c;
        if (cam.modality === "rgb") {
          if (cam.model === "fisheye") {
            const rr = Math.hypot((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy);
            if (rr > 1.69) { D[k + 3] = 255; continue; }
          }
          const { o, d } = camRayBase(cam, u + 0.5, v + 0.5);
          c = shade(rayScene(o, d), d);
        } else if (cam.modality === "thermal") {
          const { o, d } = camRayBase(cam, u + 0.5, v + 0.5);
          const hit = rayScene(o, d);
          let heat = 0.12 + 0.05 * (1 - v / cam.height);
          if (hit.kind === "box" && BOXES[hit.id].hot) heat = 0.75;
          c = infernoRGB(heat);
        } else {
          const { o, d } = camRayBase(cam, u + 0.5, v + 0.5);
          const hit = rayScene(o, d);
          const r = hit.kind === "sky" ? 9 : hit.t;
          c = r > 2.5 ? [10, 10, 30] : turboRGB(r / 2.5);
        }
        D[k] = c[0]; D[k + 1] = c[1]; D[k + 2] = c[2]; D[k + 3] = 255;
      }
    }
    return img;
  }

  _drawPersons(cam, ctx) {
    for (const p of this.persons) {
      if (!p.visible) continue;
      const pts = [[p.x, p.y, ZF + 0.05], [p.x, p.y, ZF + 1.65]].map(([x, y, z]) => projectCam(cam, ...baseToCam(cam, x, y, z)));
      if (!pts[0] || !pts[1]) continue;
      const dist = Math.hypot(p.x - cam.Trows[0][3], p.y - cam.Trows[1][3]);
      const side = projectCam(cam, ...baseToCam(cam, p.x - 0.22 * p.y / Math.max(dist, 0.1), p.y + 0.22 * p.x / Math.max(dist, 0.1), ZF + 0.9));
      const mid = projectCam(cam, ...baseToCam(cam, p.x, p.y, ZF + 0.9));
      const wpx = side && mid ? Math.max(2, Math.hypot(side[0] - mid[0], side[1] - mid[1]) * 2) : 4;
      ctx.lineCap = "round";
      if (cam.modality === "thermal") {
        ctx.strokeStyle = "rgba(252,255,164,0.95)"; ctx.lineWidth = wpx * 1.1;
        ctx.shadowColor = "rgba(250,140,20,1)"; ctx.shadowBlur = wpx;
      } else if (cam.modality === "depth") {
        const c = turboRGB(Math.min(1, dist / 2.5));
        ctx.strokeStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; ctx.lineWidth = wpx; ctx.shadowBlur = 0;
      } else {
        ctx.strokeStyle = CLOTHES[p.i % CLOTHES.length]; ctx.lineWidth = wpx; ctx.shadowBlur = 0;
      }
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.stroke();
      if (cam.modality === "rgb") {
        ctx.fillStyle = "#e8b48a";
        ctx.beginPath(); ctx.arc(pts[1][0], pts[1][1], wpx * 0.45, 0, Math.PI * 2); ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
  }

  // ---- public API (same shape as the live sources) ----
  step() {
    const t = this.t;
    this.persons = [];
    for (let i = 0; i < 4; i++) {
      const [x, y, visible] = personPos(i, t);
      const [x2, y2] = personPos(i, t + 0.1);
      this.persons.push({ i, gid: 101 + i, x, y, vx: (x2 - x) / 0.1, vy: (y2 - y) / 0.1, visible });
    }
    this.battery = Math.max(5, 87 - t * 0.01);
  }

  renderFrames() {   // -> [{idx, canvas}]
    const out = [];
    for (let i = 0; i < this.cams.length; i++) {
      const ctx = this.canvases[i].getContext("2d");
      ctx.putImageData(this.bg[i], 0, 0);
      this._drawPersons(this.cams[i], ctx);
      out.push({ idx: i, canvas: this.canvases[i] });
    }
    return out;
  }

  personsMsg() {
    const t = this.t;
    const persons = [];
    for (const p of this.persons) {
      if (!p.visible) continue;
      const cams = [], mods = new Set();
      for (const c of this.cams) {
        const uv = projectCam(c, ...baseToCam(c, p.x, p.y, ZF + 0.9));
        if (uv && uv[0] >= 0 && uv[0] < c.width && uv[1] >= 0 && uv[1] < c.height) { cams.push(c.id); mods.add(c.modality); }
      }
      if (!cams.length) continue;
      const rng = Math.hypot(p.x, p.y);
      const range_src = mods.has("depth") && rng < 2.5 ? "tof" : rng < 8 ? "lidar" : mods.has("thermal") ? "thermal_size" : "ground_plane";
      persons.push({ gid: p.gid, x: p.x, y: p.y, z: 0.9 + ZF, vx: p.vx, vy: p.vy, conf: 0.6 + 0.35 * Math.abs(Math.sin(t * 0.3 + p.i)), cams, modality: [...mods], range_src, range_m: rng, age_s: t });
    }
    return { t: Date.now() / 1000, persons };
  }

  safetyState() {
    let nearest = Infinity, ttcMin = Infinity;
    for (const p of this.persons) {
      if (!p.visible) continue;
      const r = Math.hypot(p.x, p.y);
      nearest = Math.min(nearest, r);
      const closing = -(p.x * p.vx + p.y * p.vy) / Math.max(r, 1e-3);
      if (closing > 0.05) ttcMin = Math.min(ttcMin, r / closing);
    }
    let level = "CLEAR", reason = Number.isFinite(nearest) ? `nearest ${nearest.toFixed(1)} m` : "no person";
    if (nearest < ZONES.CAUTION) { level = "CAUTION"; reason = `person ${nearest.toFixed(1)} m`; }
    if (nearest < ZONES.SLOW || ttcMin < 2) { level = "SLOW"; reason = ttcMin < 2 ? `TTC ${ttcMin.toFixed(1)} s` : `person ${nearest.toFixed(1)} m`; }
    if (nearest < ZONES.STOP || ttcMin < 1) { level = "STOP"; reason = ttcMin < 1 ? `TTC ${ttcMin.toFixed(1)} s` : `person ${nearest.toFixed(2)} m`; }
    const vmax = { CLEAR: 1.0, CAUTION: 0.6, SLOW: 0.3, STOP: 0 }[level];
    return { t: Date.now() / 1000, level, vmax, nearest_person_m: Number.isFinite(nearest) ? nearest : null, reason, demo: true };
  }

  health() {
    const t = this.t;
    const cams = {};
    for (const c of this.cams) {
      const base = c.modality === "thermal" ? 9 : 15;
      cams[c.id] = { fps: +(base + Math.sin(t + c.idx) * 0.8).toFixed(1), ok: !(c.id === "tof_rear" && Math.sin(t * 0.05) > 0.92) };
    }
    return { ok: true, gpu_util: Math.round(48 + 18 * Math.sin(t * 0.2)), gpu_temp: Math.round(58 + 6 * Math.sin(t * 0.05)), cams, demo: true };
  }

  pose() { return { x: 0, y: 0, yaw: 0, level_id: "demo" }; }

  // growing colored point cloud, encoded as real OVX1 binary
  voxelChunk(n = 1500) {
    const t = this.t;
    const R = Math.min(15, 2.5 + t * 0.8);
    const pts = [];
    const rgb = this.cams.filter((c) => c.modality === "rgb");
    for (let k = 0; k < n; k++) {
      const cam = rgb[(Math.random() * rgb.length) | 0];
      const u = Math.random() * cam.width, v = Math.random() * cam.height;
      if (Math.hypot((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy) > 1.65) continue;
      const { o, d } = camRayBase(cam, u, v);
      const hit = rayScene(o, d);
      if (hit.kind === "sky" || hit.t > R) continue;
      const c = shade(hit, d);
      let flags = 0;
      if (hit.kind === "box" && BOXES[hit.id].hot) flags |= 1;
      for (const p of this.persons) if (p.visible && Math.hypot(hit.p[0] - p.x, hit.p[1] - p.y) < 0.8) flags |= 2;
      pts.push([hit.p[0], hit.p[1], hit.p[2], c[0] | 0, c[1] | 0, c[2] | 0, flags]);
    }
    if (this.version === 0) {   // hot lamp heads
      for (const L of LAMPS) for (let k = 0; k < 60; k++) {
        const a = Math.random() * 6.283, r = Math.random() * 0.15;
        pts.push([L[0] + r * Math.cos(a), L[1] + r * Math.sin(a), L[2] + Math.random() * 0.1, 255, 220, 150, 1]);
      }
    }
    this.version++;
    return encodeOVX1(this.version, 0.05, pts);
  }
}

function infernoRGB(t) {
  const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f2 = t - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f2);
}
function turboRGB(t) {
  const stops = [[48, 18, 59], [70, 134, 251], [27, 229, 181], [164, 252, 60], [251, 185, 56], [228, 70, 10], [122, 4, 3]];
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f2 = t - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f2);
}
