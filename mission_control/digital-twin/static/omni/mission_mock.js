// OmniView mission — local mock of the mission pillar (:9116) + pure geometry
// helpers. Used when ?demo=1 or the mission backend is unreachable, so the
// whole 3D-command UI is demonstrable without backends. Mirrors CONTRACT.md
// §1-§3 shapes (plan: {ok, path, v, landings, length_m, eta_s, warnings}).
// World frame: x, y (m), yaw (rad) — same as the robot pose.

export const PROFILES = {
  stealth: { vmax: 0.25, amax: 0.3 }, precise: { vmax: 0.3, amax: 0.4 },
  normal: { vmax: 0.6, amax: 0.6 }, sprint: { vmax: 1.5, amax: 1.0 },
};
export const JUMP_LEN_M = 0.6;
export const HOP_S = 0.8;
const A_LAT = 0.45;          // lateral accel limit -> curvature speed cap
const DS = 0.1;

const hyp = Math.hypot;
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}

export function pathLength(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += hyp(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return L;
}

// uniform arc-length resampling
export function resample(pts, ds = DS) {
  if (pts.length < 2) return pts.map((p) => [p[0], p[1]]);
  const out = [[pts[0][0], pts[0][1]]];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
    const seg = hyp(bx - ax, by - ay);
    if (seg < 1e-9) continue;
    let d = ds - carry;
    while (d <= seg) { const f = d / seg; out.push([ax + (bx - ax) * f, ay + (by - ay) * f]); d += ds; }
    carry = seg - (d - ds);
  }
  const last = pts[pts.length - 1], tail = out[out.length - 1];
  if (hyp(last[0] - tail[0], last[1] - tail[1]) > 0.02) out.push([last[0], last[1]]);
  return out;
}

// centripetal-ish Catmull-Rom through the control points (freehand smoothing)
export function catmullRom(pts, ds = DS) {
  const p = simplify(pts, 0.12);
  if (p.length < 3) return resample(p, ds);
  const out = [];
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[Math.max(i - 1, 0)], p1 = p[i], p2 = p[i + 1], p3 = p[Math.min(i + 2, p.length - 1)];
    const n = Math.max(2, Math.ceil(hyp(p2[0] - p1[0], p2[1] - p1[1]) / 0.04));
    for (let k = 0; k < n; k++) {
      const t = k / n, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  out.push(p[p.length - 1]);
  return resample(out, ds);
}

function simplify(pts, minD) {
  if (!pts.length) return [];
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const l = out[out.length - 1];
    if (hyp(pts[i][0] - l[0], pts[i][1] - l[1]) >= minD || i === pts.length - 1) out.push(pts[i]);
  }
  return out;
}

// curved goto: cubic Bezier leaving along the robot heading, arriving along the
// goal yaw; bends away from persons near the straight line (social detour).
export function bezierPath(start, startYaw, goal, goalYaw, persons = []) {
  const L = hyp(goal[0] - start[0], goal[1] - start[1]);
  if (L < 0.05) return [start.slice(), goal.slice()];
  const dirYaw = Math.atan2(goal[1] - start[1], goal[0] - start[0]);
  const gy = goalYaw == null ? dirYaw : goalYaw;
  const sy = Math.abs(wrap(startYaw - dirYaw)) < 2.2 ? startYaw : dirYaw;   // don't loop behind
  const k = Math.min(L * 0.35, 2.0);
  const p1 = [start[0] + Math.cos(sy) * k, start[1] + Math.sin(sy) * k];
  const p2 = [goal[0] - Math.cos(gy) * k, goal[1] - Math.sin(gy) * k];
  const nx = -Math.sin(dirYaw), ny = Math.cos(dirYaw);
  for (const p of persons) {
    const rx = p.wx - start[0], ry = p.wy - start[1];
    const along = rx * Math.cos(dirYaw) + ry * Math.sin(dirYaw);
    const side = rx * nx + ry * ny;
    if (along > 0.3 && along < L - 0.3 && Math.abs(side) < 1.6) {
      const push = (side >= 0 ? -1 : 1) * (2.4 - Math.abs(side));
      p1[0] += nx * push; p1[1] += ny * push; p2[0] += nx * push; p2[1] += ny * push;
    }
  }
  const n = Math.max(8, Math.ceil(L / 0.03));
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    out.push([a * start[0] + b * p1[0] + c * p2[0] + d * goal[0], a * start[1] + b * p1[1] + c * p2[1] + d * goal[1]]);
  }
  return resample(out, DS);
}

// per-point speed (m/s): curvature cap, person corridor slow-down, accel ramps.
// opts: {v0: entry speed, vEnd: exit speed (0 = stop), fineTail: precise cap on the last 0.5 m}
export function speedProfile(path, profileName, persons = [], opts = {}) {
  const prof = PROFILES[profileName] || PROFILES.normal;
  const warnings = [];
  let vmax = prof.vmax;
  const n = path.length;
  if (n < 2) return { v: path.map(() => 0), warnings };
  if (profileName === "sprint") {
    const busy = persons.some((p) => minDistToPath(path, p.wx, p.wy) < 1.5);
    if (busy) { vmax = PROFILES.normal.vmax; warnings.push("sprint downgraded: person within ±1.5 m of corridor"); }
  }
  const v = new Array(n).fill(vmax);
  for (let i = 1; i < n - 1; i++) {
    const a = path[i - 1], b = path[i], c = path[i + 1];
    const h1 = Math.atan2(b[1] - a[1], b[0] - a[0]), h2 = Math.atan2(c[1] - b[1], c[0] - b[0]);
    const ds = (hyp(b[0] - a[0], b[1] - a[1]) + hyp(c[0] - b[0], c[1] - b[1])) / 2 || DS;
    const kappa = Math.abs(wrap(h2 - h1)) / ds;
    if (kappa > 1e-3) v[i] = Math.min(v[i], Math.sqrt(A_LAT / kappa));
  }
  let slowed = false;
  for (let i = 0; i < n; i++) {
    for (const p of persons) {
      const d = hyp(path[i][0] - p.wx, path[i][1] - p.wy);
      if (d < 2.5) { v[i] = Math.min(v[i], 0.3); slowed = true; }
    }
  }
  if (slowed) warnings.push("person near path: slowed to 0.3 m/s");
  if (opts.fineTail) {
    let acc = 0;
    for (let i = n - 1; i > 0 && acc < 0.5; i--) { v[i] = Math.min(v[i], PROFILES.precise.vmax); acc += hyp(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]); }
  }
  v[0] = Math.min(v[0], opts.v0 || 0); v[n - 1] = Math.min(v[n - 1], opts.vEnd || 0);
  for (let i = 1; i < n; i++) {
    const ds = hyp(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    v[i] = Math.min(v[i], Math.sqrt(v[i - 1] * v[i - 1] + 2 * prof.amax * ds));
  }
  for (let i = n - 2; i >= 0; i--) {
    const ds = hyp(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]);
    v[i] = Math.min(v[i], Math.sqrt(v[i + 1] * v[i + 1] + 2 * prof.amax * ds));
  }
  return { v: v.map((x) => +x.toFixed(3)), warnings };
}

export function minDistToPath(path, x, y) {
  let m = Infinity;
  for (const p of path) m = Math.min(m, hyp(p[0] - x, p[1] - y));
  return m;
}

export function etaOf(path, v) {
  let t = 0;
  for (let i = 1; i < path.length; i++) {
    const ds = hyp(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    t += ds / Math.max((v[i] + v[i - 1]) / 2, 0.05);
  }
  return t;
}

function zoneWarnings(path, zones, warnings) {
  let ok = true;
  for (const z of zones || []) {
    if (z.kind !== "no_go" || !Array.isArray(z.polygon) || z.polygon.length < 3) continue;
    if (path.some((p) => pointInPoly(p[0], p[1], z.polygon))) { warnings.push(`path crosses NO-GO zone "${z.name || z.id}"`); ok = false; }
  }
  return ok;
}

const STEP_S = { look_at: 1.5, action: 3, say: 2, shadow: 10, watch: 8, escort: 10, explore: 12 };

// ctx: {pose:{x,y,yaw}, persons:[{gid,wx,wy}], zones, labels, home}
export function planStep(step, ctx) {
  const pose = ctx.pose || { x: 0, y: 0, yaw: 0 };
  const start = [pose.x, pose.y];
  const warnings = [];
  const base = { op: step.op, ok: true, path: [], v: [], landings: [], length_m: 0, eta_s: 0, warnings, reason: "", end: { ...pose } };
  const speed = step.speed || "normal";
  const zone = isPrecisionZone(step.zone) ? step.zone : (ctx.isLast ? "fine" : "z30");
  const flyby = zone !== "fine" && !ctx.isLast;
  const fin = (path, yawEnd) => {
    const sp = speedProfile(path, speed, ctx.persons, { v0: ctx.v0 || 0, vEnd: flyby ? (PROFILES[speed] || PROFILES.normal).vmax : 0, fineTail: zone === "fine" });
    warnings.push(...sp.warnings);
    const ok = zoneWarnings(path, ctx.zones, warnings);
    const L = pathLength(path);
    const last = path[path.length - 1] || start;
    const prev = path[path.length - 2] || start;
    const tangent = Math.atan2(last[1] - prev[1], last[0] - prev[0]);
    let eta = etaOf(path, sp.v);
    const yaw = yawEnd != null ? yawEnd : (path.length > 1 ? tangent : pose.yaw);
    eta += Math.abs(wrap(yaw - tangent)) / 0.8;
    if (speed === "sprint") warnings.push("SPRINT: dead-man must be held");
    if (zone === "fine") eta += 0.5;   // settle
    return { ...base, ok, reason: ok ? "" : "no_go", path, v: sp.v, length_m: +L.toFixed(2), eta_s: +eta.toFixed(1), speed, zone, end: { x: last[0], y: last[1], yaw } };
  };
  switch (step.op) {
    case "goto": {
      const goal = [+step.x, +step.y];
      if (hyp(goal[0] - start[0], goal[1] - start[1]) > 60) return { ...base, ok: false, reason: "goal out of map", warnings: ["goal is > 60 m away"] };
      return fin(bezierPath(start, pose.yaw, goal, step.yaw, ctx.persons), step.yaw);
    }
    case "goto_label": {
      const l = (ctx.labels || []).find((q) => q.name === step.label);
      if (!l) return { ...base, ok: false, reason: `unknown label ${step.label}`, warnings: [`unknown label "${step.label}"`] };
      return fin(bezierPath(start, pose.yaw, [l.x, l.y], l.yaw, ctx.persons), l.yaw);
    }
    case "return_home": {
      const h = ctx.home;
      if (!h) return { ...base, ok: false, reason: "no home", warnings: ["home not set (H)"] };
      return fin(bezierPath(start, pose.yaw, [h.x, h.y], h.yaw, ctx.persons), h.yaw);
    }
    case "follow_path": case "patrol": {
      let pts = step.points;
      if (!pts && step.zone && !isPrecisionZone(step.zone)) { const z = (ctx.zones || []).find((q) => q.id === step.zone || q.name === step.zone); if (z) pts = [...z.polygon, z.polygon[0]]; }
      if (!Array.isArray(pts) || pts.length < 2) return { ...base, ok: false, reason: "need ≥2 points", warnings: ["path needs ≥ 2 points"] };
      const lead = hyp(pts[0][0] - start[0], pts[0][1] - start[1]) > 0.3 ? [start] : [];
      let all = [...lead, ...pts];
      if (step.op === "patrol") for (let k = 1; k < (+step.loops || 1); k++) all = all.concat(pts);
      return fin(step.smooth === false ? resample(all, DS) : catmullRom(all, DS));
    }
    case "jump_to": {
      const goal = [+step.x, +step.y];
      const D = hyp(goal[0] - start[0], goal[1] - start[1]);
      const need = Math.ceil(D / JUMP_LEN_M - 1e-6);
      const maxJ = +step.max_jumps || 6;
      const n = Math.min(need, maxJ);
      const yaw = Math.atan2(goal[1] - start[1], goal[0] - start[0]);
      const landings = [];
      for (let i = 1; i <= n; i++) landings.push([+(start[0] + Math.cos(yaw) * JUMP_LEN_M * i).toFixed(3), +(start[1] + Math.sin(yaw) * JUMP_LEN_M * i).toFixed(3)]);
      let ok = true;
      if (need > maxJ) { ok = false; warnings.push(`needs ${need} jumps > max_jumps ${maxJ}`); }
      for (const p of ctx.persons || []) {
        if (landings.some((l) => hyp(l[0] - p.wx, l[1] - p.wy) < 3.5)) { ok = false; warnings.push(`person #${p.gid} within 3.5 m of a landing`); break; }
      }
      if (!zoneWarnings(landings, ctx.zones, warnings)) ok = false;
      warnings.push("JUMP: dead-man must be held");
      const last = landings[landings.length - 1] || start;
      const path = [start, ...landings];
      return { ...base, ok, reason: ok ? "" : "landing check failed", path, v: path.map(() => JUMP_LEN_M / HOP_S), landings, length_m: +(n * JUMP_LEN_M).toFixed(2), eta_s: +(n * HOP_S + 0.5).toFixed(1), end: { x: last[0], y: last[1], yaw } };
    }
    case "look_at": {
      const yaw = Math.atan2(+step.y - pose.y, +step.x - pose.x);
      return { ...base, eta_s: +(Math.abs(wrap(yaw - pose.yaw)) / 0.8 + 0.3).toFixed(1), end: { ...pose, yaw } };
    }
    case "scan": return { ...base, eta_s: +((+step.deg || 360) / (+step.speed_dps || 30)).toFixed(1) };
    case "wait": return { ...base, eta_s: +step.s || 0 };
    case "capture": {
      if (step.x == null || step.y == null) return { ...base, eta_s: 2 };
      const p = fin(bezierPath(start, pose.yaw, [+step.x, +step.y], step.yaw, ctx.persons), step.yaw);
      return { ...p, eta_s: +(p.eta_s + 2).toFixed(1), zone: "fine" };
    }
    case "request_passage": return { ...base, eta_s: 12, warnings: [`waits for passage (≤ ${step.timeout_s || 600} s) — CONTINUE to release`] };
    case "wait_for": return { ...base, eta_s: 5, warnings: step.event === "operator" ? ["waits for operator CONTINUE"] : [] };
    case "record": return { ...base, eta_s: 0.3 };
    default: return { ...base, eta_s: Math.min(+step.timeout_s || STEP_S[step.op] || 2, STEP_S[step.op] || 30) };
  }
}

// whole-mission preview: chained per-step plans + a ghost timeline
export const PRECISION_ZONES = ["fine", "z10", "z30", "z60", "z100"];
export function isPrecisionZone(z) { return typeof z === "string" && PRECISION_ZONES.includes(z); }
export function zoneRadius(z) { return z === "fine" ? 0.05 : isPrecisionZone(z) ? +z.slice(1) / 100 : 0.3; }
const PATH_OPS = new Set(["goto", "goto_label", "return_home", "follow_path", "patrol", "capture"]);

export function planMission(mission, ctx0) {
  const ctx = { ...ctx0, pose: { ...(ctx0.pose || { x: 0, y: 0, yaw: 0 }) }, v0: 0 };
  const all = mission.steps || [];
  let lastMove = -1;
  all.forEach((st, i) => { if (PATH_OPS.has(st.op) || st.op === "jump_to") lastMove = i; });
  const steps = [], path = [], v = [], landings = [], warnings = [];
  const timeline = [{ t: 0, x: ctx.pose.x, y: ctx.pose.y, yaw: ctx.pose.yaw, z: 0, step: 0 }];
  let ok = true, T = 0;
  all.forEach((st, si) => {
    // fly-by only if the next step is also a path step
    ctx.isLast = si === lastMove || !PATH_OPS.has((all[si + 1] || {}).op);
    const p = planStep(st, ctx);
    ctx.v0 = p.v.length ? p.v[p.v.length - 1] : 0;
    steps.push(p);
    if (!p.ok) ok = false;
    for (const w of p.warnings) warnings.push(`#${si + 1} ${st.op}: ${w}`);
    path.push(...p.path); v.push(...p.v); landings.push(...p.landings);
    T = appendTimeline(timeline, p, ctx.pose, T, si, st);
    ctx.pose = { ...p.end };
  });
  return { ok, path, v, landings, warnings, steps, eta_s: +T.toFixed(1), timeline, end: ctx.pose };
}

function appendTimeline(tl, p, pose, T, si, st) {
  const push = (dt, x, y, yaw, z = 0) => { T += dt; tl.push({ t: T, x, y, yaw, z, step: si }); };
  if (p.op === "jump_to") {
    let [px, py] = [pose.x, pose.y];
    for (const l of p.landings) {
      for (let k = 1; k <= 6; k++) { const f = k / 6; push(HOP_S / 6, px + (l[0] - px) * f, py + (l[1] - py) * f, p.end.yaw, 4 * 0.3 * f * (1 - f)); }
      [px, py] = l;
    }
    return T;
  }
  if (p.path.length > 1) {
    for (let i = 1; i < p.path.length; i++) {
      const a = p.path[i - 1], b = p.path[i];
      const ds = hyp(b[0] - a[0], b[1] - a[1]);
      push(ds / Math.max((p.v[i] + p.v[i - 1]) / 2, 0.05), b[0], b[1], Math.atan2(b[1] - a[1], b[0] - a[0]));
    }
    const last = tl[tl.length - 1];
    if (Math.abs(wrap(p.end.yaw - last.yaw)) > 0.05) push(Math.abs(wrap(p.end.yaw - last.yaw)) / 0.8, last.x, last.y, p.end.yaw);
    return T;
  }
  const last = tl[tl.length - 1];
  if (st.op === "scan") {
    const deg = (+st.deg || 360) * Math.PI / 180, n = 24;
    for (let k = 1; k <= n; k++) push(p.eta_s / n, last.x, last.y, last.yaw + deg * k / n);
    return T;
  }
  push(Math.max(p.eta_s, 0.2), last.x, last.y, p.end.yaw);
  return T;
}

export function sampleTimeline(tl, t) {
  if (!tl || !tl.length) return null;
  if (t <= tl[0].t) return tl[0];
  for (let i = 1; i < tl.length; i++) {
    if (tl[i].t >= t) {
      const a = tl[i - 1], b = tl[i], f = (t - a.t) / Math.max(b.t - a.t, 1e-6);
      return { t, x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, yaw: a.yaw + wrap(b.yaw - a.yaw) * f, z: a.z + (b.z - a.z) * f, step: b.step };
    }
  }
  return tl[tl.length - 1];
}

// ---------------- mock server ----------------
const TERMINAL = new Set(["done", "aborted", "failed", "rejected"]);
const LS_KEY = "omniview.mission.world";

export class MockMissionServer {
  constructor(getCtx, emit) {
    this.getCtx = getCtx;            // () => {pose, persons}
    this.emit = emit;                // (msg) — same shape as WS /ws/mission
    this.missions = new Map();
    this.run = null;
    this.pose = null;                // mock robot pose while/after executing (demo)
    this.seq = 1;
    this.world = { zones: [], labels: [], home: null, rules: [] };
    try { Object.assign(this.world, JSON.parse(localStorage.getItem(LS_KEY) || "{}")); } catch (_) {}
  }
  _save() { try { localStorage.setItem(LS_KEY, JSON.stringify(this.world)); } catch (_) {} }
  ctx() { const c = this.getCtx(); return { ...c, pose: this.pose || c.pose, zones: this.world.zones, labels: this.world.labels, home: this.world.home }; }

  plan(step) { return planStep(step, this.ctx()); }
  previewMission(mission) { return planMission(mission, this.ctx()); }

  create(mission, { execute = true, pending = false } = {}) {
    const id = mission.mission_id || `mock-${this.seq++}`;
    const plan = planMission(mission, this.ctx());
    const m = { mission_id: id, mission: { ...mission, mission_id: id }, plan, state: pending ? "pending_approval" : execute ? "queued" : "planned" };
    this.missions.set(id, m);
    if (m.state === "queued") this._start(m);
    else this._emit(m, 0, 0, plan.eta_s);
    return { mission_id: id, state: m.state, plan };
  }
  continue(id) {
    const r = this.run;
    if (r && r.m.mission_id === id && r.m.state === "waiting") { r.continued.add(r.si); r.m.state = "running"; this._emitRun(); }
    return { state: r ? r.m.state : "idle" };
  }
  approve(id) { const m = this.missions.get(id); if (!m) return null; m.state = "queued"; this._start(m); return { mission_id: id, state: m.state }; }
  pause(id) { if (this.run && this.run.m.mission_id === id && this.run.m.state === "running") { this.run.m.state = "paused"; this._emitRun(); } return { state: "paused" }; }
  resume(id) { if (this.run && this.run.m.mission_id === id && this.run.m.state === "paused") { this.run.m.state = "running"; this._emitRun(); } return { state: "running" }; }
  abort(id) {
    const m = this.missions.get(id);
    if (m && !TERMINAL.has(m.state)) { m.state = "aborted"; this._emit(m, this.run && this.run.m === m ? this.run.si : 0, 0, 0); }
    if (this.run && (!id || this.run.m.mission_id === id)) this.run = null;
    return { state: "aborted" };
  }

  _start(m) {
    if (this.run && this.run.m !== m) this.abort(this.run.m.mission_id);
    m.state = "running";
    m.plan = planMission(m.mission, this.ctx());   // replan from the current pose
    this.run = { m, t: 0, T: Math.max(m.plan.eta_s, 0.1), si: 0, lastEmit: -1, continued: new Set(), waitT: 0 };
    if (!this.pose) { const p = this.ctx().pose; this.pose = { x: p.x, y: p.y, yaw: p.yaw }; }
    this._emitRun();
  }
  _emit(m, si, progress, eta) {
    const st = (m.mission.steps || [])[si] || {};
    this.emit({ t: Date.now() / 1000, mission_id: m.mission_id, state: m.state, step_index: si, step_op: st.op || "", progress, eta_s: eta, error: m.error || null, steps_total: (m.mission.steps || []).length, mock: true });
  }
  _emitRun() { const r = this.run; if (r) this._emit(r.m, r.si, Math.min(r.t / r.T, 1), Math.max(r.T - r.t, 0)); }

  // dt s; deadman: bool (operator holding)
  tick(dt, deadman) {
    const r = this.run;
    if (!r) return;
    if (r.m.state === "waiting") {
      r.waitT += dt;
      const st0 = (r.m.mission.steps || [])[r.si] || {};
      if (st0.op === "request_passage" && r.waitT > 12) {   // mock: "door opened" -> corridor clear
        this.emit({ t: Date.now() / 1000, mission_id: r.m.mission_id, kind: "alert", level: "info", text: "passage clear (mock) — continuing" });
        this.continue(r.m.mission_id);
      }
      return;
    }
    if (r.m.state !== "running") return;
    const tl = r.m.plan.timeline;
    const cur = sampleTimeline(tl, r.t);
    const si0 = cur ? cur.step : 0;
    const st = (r.m.mission.steps || [])[si0] || {};
    const waits = st.op === "request_passage" || (st.op === "wait_for" && st.event === "operator");
    if (waits && !r.continued.has(si0) && r.si === si0) {
      r.m.state = "waiting"; r.waitT = 0;
      this._emitRun();
      this.emit({ t: Date.now() / 1000, mission_id: r.m.mission_id, kind: "alert", level: "warn", text: `várakozik: ${st.text || st.event || st.op}` });
      return;
    }
    if (st.op === "record") this.rec = st.on !== false;
    if ((st.speed === "sprint" || st.op === "jump_to") && !deadman) {
      r.m.state = "paused"; r.m.error = "dead-man released";
      this._emitRun();
      this.emit({ t: Date.now() / 1000, mission_id: r.m.mission_id, kind: "alert", level: "warn", text: `dead-man released during ${st.op}${st.speed === "sprint" ? " (sprint)" : ""} — paused` });
      return;
    }
    r.m.error = null;
    r.t += dt;
    const s = sampleTimeline(tl, r.t);
    if (s) { this.pose = { x: s.x, y: s.y, yaw: s.yaw }; r.si = s.step; this.z = s.z || 0; }
    if (s && s.step !== si0) r.si = s.step;
    if (r.t >= r.T) {
      r.m.state = "done";
      this._emit(r.m, (r.m.mission.steps || []).length - 1, 1, 0);
      this.emit({ t: Date.now() / 1000, mission_id: r.m.mission_id, kind: "alert", level: "info", text: `mission ${r.m.mission.name || r.m.mission_id} complete` });
      this.run = null;
      return;
    }
    if (r.t - r.lastEmit > 0.25) { r.lastEmit = r.t; this._emitRun(); }
  }

  // ---- world (M5 shapes) ----
  zones() { return this.world.zones; }
  addZone(z) { const zz = { id: z.id || `z${this.seq++}`, ...z }; this.world.zones = this.world.zones.filter((q) => q.id !== zz.id).concat([zz]); this._save(); return zz; }
  delZone(id) { this.world.zones = this.world.zones.filter((q) => q.id !== id && q.name !== id); this._save(); }
  labels() { return this.world.labels; }
  addLabel(l) { this.world.labels = this.world.labels.filter((q) => q.name !== l.name).concat([l]); this._save(); return l; }
  delLabel(name) { this.world.labels = this.world.labels.filter((q) => q.name !== name); this._save(); }
  setHome(h) { this.world.home = h; this._save(); return h; }

  // ---- NL (keyword mock of M6) ----
  nl(text) {
    const t = String(text || "").toLowerCase();
    const c = this.ctx(), p = c.pose;
    const speed = /sprint|fuss|run|gyors|max/.test(t) ? "sprint" : /lopak|stealth|halk|settenk/.test(t) ? "stealth" : /prec|pontos/.test(t) ? "precise" : "normal";
    const steps = [];
    const lab = (c.labels || []).find((l) => t.includes(String(l.name).toLowerCase()));
    if (/home|haza|dokk|bázis/.test(t)) steps.push({ op: "return_home", speed });
    else if (lab) steps.push({ op: "goto_label", label: lab.name, speed });
    else if (/ugr|jump/.test(t)) steps.push({ op: "jump_to", x: +(p.x + Math.cos(p.yaw) * 2.4).toFixed(2), y: +(p.y + Math.sin(p.yaw) * 2.4).toFixed(2), max_jumps: 4 });
    else if (/patrol|járőr|jaror/.test(t)) {
      const z = (c.zones || []).find((q) => q.kind === "patrol");
      if (z) steps.push({ op: "patrol", zone: z.id, loops: 1, speed });
      else steps.push({ op: "follow_path", speed, points: [[p.x + 2, p.y], [p.x + 2, p.y + 2], [p.x, p.y + 2], [p.x, p.y]] });
    } else if (/nézz|nezz|scan|körül|korul|look around/.test(t) === false) {
      steps.push({ op: "goto", x: +(p.x + Math.cos(p.yaw) * 3).toFixed(2), y: +(p.y + Math.sin(p.yaw) * 3).toFixed(2), speed });
    }
    if (/nézz|nezz|scan|körül|korul|look around|szkenn/.test(t)) steps.push({ op: "scan", deg: 360 });
    if (/hello|köszön|koszon|integ/.test(t)) steps.push({ op: "action", name: "hello" });
    const say = /(?:say|mondd)[: ]+(.+)$/.exec(text || "");
    if (say) steps.push({ op: "say", text: say[1] });
    const mission = { name: `NL: ${String(text).slice(0, 40)}`, source: "nl", steps, on_fail: "stop" };
    const r = this.create(mission, { pending: true });
    return { ...r, mission, mock: true };
  }
}
