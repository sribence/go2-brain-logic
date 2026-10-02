// NERO OmniVision mobile — minimal local mission mock (fallback when the mission
// pillar is unreachable and M4's ../mission_mock.js is absent). Same message
// shapes as WS /ws/mission: {mission_id, state, step_index, step_op, progress, eta_s}.

export const PROFILES = { stealth: 0.25, precise: 0.3, normal: 0.6, sprint: 1.5 };
const MOVE_OPS = new Set(["goto", "capture", "follow_path", "patrol", "jump_to", "goto_label", "return_home"]);
const hyp = Math.hypot;
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function straight(a, b, ds = 0.1) {
  const L = hyp(b[0] - a[0], b[1] - a[1]);
  const n = Math.max(1, Math.ceil(L / ds));
  const out = [];
  for (let i = 0; i <= n; i++) out.push([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n]);
  return out;
}

function profile(path, speed, persons, fine) {
  const vmax = PROFILES[speed] || PROFILES.normal;
  const v = path.map(() => vmax);
  const warnings = [];
  let slowed = false;
  path.forEach((p, i) => { for (const q of persons || []) if (hyp(p[0] - q.wx, p[1] - q.wy) < 2.5) { v[i] = Math.min(v[i], 0.3); slowed = true; } });
  if (slowed) warnings.push("ember a pálya közelében: 0.3 m/s");
  const a = 0.6;
  let s = 0;
  for (let i = 0; i < path.length; i++) {   // accel ramp
    if (i) s += hyp(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    v[i] = Math.min(v[i], Math.sqrt(2 * a * s) + 0.05);
  }
  s = 0;
  for (let i = path.length - 1; i >= 0; i--) {   // decel ramp (fine = stop)
    if (i < path.length - 1) s += hyp(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]);
    v[i] = Math.min(v[i], (fine ? Math.sqrt(2 * a * s) : vmax) + (fine ? 0 : 0.05));
    if (fine && s < 0.5) v[i] = Math.min(v[i], PROFILES.precise);
  }
  return { v: v.map((x) => +x.toFixed(3)), warnings };
}

function eta(path, v) {
  let t = 0;
  for (let i = 1; i < path.length; i++) t += hyp(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) / Math.max((v[i] + v[i - 1]) / 2, 0.05);
  return t;
}

export function planStep(step, ctx) {
  const pose = ctx.pose || { x: 0, y: 0, yaw: 0 };
  const base = { op: step.op, ok: true, path: [], v: [], landings: [], warnings: [], length_m: 0, eta_s: 2, end: { ...pose } };
  if (step.x != null && step.y != null && MOVE_OPS.has(step.op)) {
    const path = straight([pose.x, pose.y], [+step.x, +step.y]);
    const zone = step.zone || (ctx.isLast ? "fine" : "z30");
    const sp = profile(path, step.speed || "normal", ctx.persons, zone === "fine");
    const L = hyp(step.x - pose.x, step.y - pose.y);
    const yaw = step.yaw != null ? +step.yaw : Math.atan2(step.y - pose.y, step.x - pose.x);
    if (step.speed === "sprint") sp.warnings.push("SPRINT: dead-man szükséges");
    let t = eta(path, sp.v) + Math.abs(wrap(yaw - pose.yaw)) / 0.8 + (zone === "fine" ? 0.5 : 0) + (step.op === "capture" ? 2 : 0);
    if (step.op === "jump_to") { t = Math.ceil(L / 0.6) * 0.8; sp.warnings.push("UGRÁS: dead-man szükséges"); }
    return { ...base, path, v: sp.v, warnings: sp.warnings, length_m: +L.toFixed(2), eta_s: +t.toFixed(1), zone, speed: step.speed || "normal", end: { x: +step.x, y: +step.y, yaw } };
  }
  if (step.op === "look_at") { const yaw = Math.atan2(step.y - pose.y, step.x - pose.x); return { ...base, eta_s: +(Math.abs(wrap(yaw - pose.yaw)) / 0.8 + 0.3).toFixed(1), end: { ...pose, yaw } }; }
  if (step.op === "scan") return { ...base, eta_s: (+step.deg || 360) / 30 };
  if (step.op === "wait") return { ...base, eta_s: +step.s || 1 };
  if (step.op === "request_passage") return { ...base, eta_s: 12, warnings: ["átkérés: TOVÁBB gombbal engedhető"] };
  if (["shadow", "watch", "escort"].includes(step.op)) return { ...base, eta_s: Math.min(+step.timeout_s || 20, 20) };
  return base;
}

export function planMission(mission, ctx0) {
  const ctx = { ...ctx0, pose: { ...(ctx0.pose || { x: 0, y: 0, yaw: 0 }) } };
  const steps = mission.steps || [];
  const out = { ok: true, path: [], v: [], landings: [], warnings: [], steps: [], eta_s: 0, segs: [] };
  steps.forEach((st, i) => {
    ctx.isLast = i === steps.length - 1;
    const p = planStep(st, ctx);
    out.steps.push(p);
    if (!p.ok) out.ok = false;
    p.warnings.forEach((w) => out.warnings.push(`#${i + 1} ${st.op}: ${w}`));
    out.path.push(...p.path); out.v.push(...p.v);
    out.segs.push({ i, t0: out.eta_s, T: p.eta_s, start: { ...ctx.pose }, plan: p });
    out.eta_s += p.eta_s;
    ctx.pose = { ...p.end };
  });
  out.eta_s = +out.eta_s.toFixed(1);
  return out;
}

const TERMINAL = new Set(["done", "aborted", "failed", "rejected"]);

export class MiniMock {
  constructor(getCtx, emit) { this.getCtx = getCtx; this.emit = emit; this.missions = new Map(); this.run = null; this.pose = null; this.seq = 1; }
  ctx() { const c = this.getCtx(); return { ...c, pose: this.pose || c.pose }; }
  plan(step) { return planStep(step, { ...this.ctx(), isLast: true }); }
  previewMission(m) { return planMission(m, this.ctx()); }
  create(mission, { execute = true, pending = false } = {}) {
    const id = mission.mission_id || `mock-${this.seq++}`;
    const m = { mission_id: id, mission, plan: planMission(mission, this.ctx()), state: pending ? "pending_approval" : execute ? "queued" : "planned" };
    this.missions.set(id, m);
    if (m.state === "queued") this._start(m); else this._emit(m, 0, 0, m.plan.eta_s);
    return { mission_id: id, state: m.state, plan: m.plan };
  }
  _start(m) {
    if (this.run && this.run.m !== m) this.abort(this.run.m.mission_id);
    if (!this.pose) { const p = this.ctx().pose; this.pose = { x: p.x, y: p.y, yaw: p.yaw }; }
    m.plan = planMission(m.mission, this.ctx());
    m.state = "running";
    this.run = { m, t: 0, si: 0, cont: new Set(), waitT: 0, lastEmit: -1 };
    this._emitRun();
  }
  _emit(m, si, progress, eta_s) {
    const st = (m.mission.steps || [])[si] || {};
    this.emit({ t: Date.now() / 1000, mission_id: m.mission_id, state: m.state, step_index: si, step_op: st.op || "", progress, eta_s, steps_total: (m.mission.steps || []).length, error: m.error || null, mock: true });
  }
  _emitRun() { const r = this.run; if (r) this._emit(r.m, r.si, Math.min(r.t / Math.max(r.m.plan.eta_s, 0.1), 1), Math.max(r.m.plan.eta_s - r.t, 0)); }
  approve(id) { const m = this.missions.get(id); if (m) this._start(m); return { mission_id: id, state: "queued" }; }
  pause(id) { const r = this.run; if (r && r.m.mission_id === id && r.m.state === "running") { r.m.state = "paused"; this._emitRun(); } return { state: "paused" }; }
  resume(id) { const r = this.run; if (r && r.m.mission_id === id && r.m.state === "paused") { r.m.state = "running"; r.m.error = null; this._emitRun(); } return { state: "running" }; }
  continue(id) { const r = this.run; if (r && r.m.mission_id === id && r.m.state === "waiting") { r.cont.add(r.si); r.m.state = "running"; this._emitRun(); } return { state: "running" }; }
  abort(id) {
    const m = id ? this.missions.get(id) : this.run && this.run.m;
    if (m && !TERMINAL.has(m.state)) { m.state = "aborted"; this._emit(m, this.run ? this.run.si : 0, 0, 0); }
    if (this.run && (!id || this.run.m.mission_id === id)) this.run = null;
    return { state: "aborted" };
  }
  tick(dt, deadman) {
    const r = this.run;
    if (!r) return;
    const segs = r.m.plan.segs;
    const seg = segs[r.si];
    if (!seg) { r.m.state = "done"; this._emit(r.m, Math.max(0, segs.length - 1), 1, 0); this.emit({ kind: "alert", level: "info", text: "misszió kész", mission_id: r.m.mission_id }); this.run = null; return; }
    const st = r.m.mission.steps[r.si] || {};
    if (r.m.state === "waiting") {
      r.waitT += dt;
      if (r.waitT > 12) { this.emit({ kind: "alert", level: "info", text: "átjáró szabad (mock) — továbbmegy" }); this.continue(r.m.mission_id); }
      return;
    }
    if (r.m.state !== "running") return;
    if ((st.op === "request_passage" || (st.op === "wait_for" && st.event === "operator")) && !r.cont.has(r.si)) {
      r.m.state = "waiting"; r.waitT = 0; this._emitRun();
      this.emit({ kind: "alert", level: "warn", text: `várakozik: ${st.text || st.event || st.op}`, mission_id: r.m.mission_id });
      return;
    }
    if ((st.speed === "sprint" || st.op === "jump_to") && !deadman) {
      r.m.state = "paused"; r.m.error = "dead-man elengedve"; this._emitRun();
      this.emit({ kind: "alert", level: "warn", text: "dead-man elengedve — szünet", mission_id: r.m.mission_id });
      return;
    }
    r.t += dt;
    const lt = r.t - seg.t0, f = Math.min(lt / Math.max(seg.T, 0.05), 1);
    const p = seg.plan;
    if (p.path.length > 1) {
      const k = Math.min(p.path.length - 1, Math.floor(f * (p.path.length - 1)));
      const a = p.path[k], b = p.path[Math.min(k + 1, p.path.length - 1)];
      this.pose = { x: a[0], y: a[1], yaw: f >= 1 ? p.end.yaw : Math.atan2(b[1] - a[1], b[0] - a[0]) || this.pose.yaw };
    } else if (p.end) this.pose = { ...this.pose, yaw: seg.start.yaw + wrap(p.end.yaw - seg.start.yaw) * f };
    if (f >= 1) { r.si++; this._emitRun(); return; }
    if (r.t - r.lastEmit > 0.25) { r.lastEmit = r.t; this._emitRun(); }
  }
  nl(text) {
    const t = String(text || "").toLowerCase();
    const p = this.ctx().pose;
    const speed = /gyors|sprint|fuss/.test(t) ? "sprint" : /halk|lopak/.test(t) ? "stealth" : /pontos/.test(t) ? "precise" : "normal";
    const steps = [];
    const m = /(\d+(?:[.,]\d+)?)\s*(?:m|méter)/.exec(t);
    const d = m ? parseFloat(m[1].replace(",", ".")) : 3;
    if (/vissza|haza|home/.test(t)) steps.push({ op: "goto", x: 0, y: 0, speed, zone: "fine" });
    else if (!/körül|szkenn|scan/.test(t) || /menj|előre|elore/.test(t)) steps.push({ op: "goto", x: +(p.x + Math.cos(p.yaw) * d).toFixed(2), y: +(p.y + Math.sin(p.yaw) * d).toFixed(2), speed, zone: "z30" });
    if (/körül|szkenn|scan/.test(t)) steps.push({ op: "scan", deg: 360 });
    if (/fotó|foto|kép/.test(t)) steps.push({ op: "capture", cams: "all", mode: "max" });
    const say = /(?:mondd|mond)[: ]+(.+)$/i.exec(text || "");
    if (say) steps.push({ op: "say", text: say[1] });
    const mission = { name: `NL: ${String(text).slice(0, 40)}`, source: "nl", steps, on_fail: "stop" };
    return { ...this.create(mission, { pending: true }), mission, mock: true };
  }
}
