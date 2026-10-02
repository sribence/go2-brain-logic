// OmniView mission — client for the mission pillar (:9116) + safety_guard
// dead-man (:9115). Falls back to MockMissionServer when ?demo=1 or the
// backend is unreachable. Nothing throws; every call resolves {ok, data, mock}.

import { fetchJSON, postJSON, ReconnectingWS } from "./net.js";
import { MockMissionServer } from "./mission_mock.js";

const params = new URLSearchParams(location.search);

function hostPort(param, defPort) {
  let h = params.get(param) || location.hostname || "127.0.0.1";
  h = h.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const hasPort = h.startsWith("[") ? /\]:\d+$/.test(h) : /^[^:]+:\d+$/.test(h);
  return hasPort ? h : `${h}:${defPort}`;
}

export function missionEndpoints(EP) {
  const secure = location.protocol === "https:";
  const m = hostPort("mission", 9116);
  return {
    http: `${secure ? "https" : "http"}://${m}`,
    ws: `${secure ? "wss" : "ws"}://${m}`,
    safety: EP.safetyHttp,
    demo: EP.demoForced,
  };
}

// ---------------- step / mission builders (shared with the mobile UI) ----------------
export const SPEEDS = ["stealth", "precise", "normal", "sprint"];
export const PRECISION = ["fine", "z10", "z30", "z60", "z100"];
export const ACTIONS = ["hello", "stretch", "sit", "stand", "lay_down", "dance", "heart", "shake"];
export const WAIT_EVENTS = ["operator", "path_clear", "gesture", "person_gone"];
export const DEFAULT_PASSAGE_TEXT = "Kérlek, engedj ki az ajtón!";
export const OP_FIELDS = {   // editor schema: field -> kind
  goto: { x: "num", y: "num", yaw: "num?", speed: "speed", zone: "zone" },
  jump_to: { x: "num", y: "num", max_jumps: "int" },
  follow_path: { points: "points", speed: "speed", zone: "zone" },
  look_at: { x: "num", y: "num", z: "num" },
  scan: { deg: "num" },
  wait: { s: "num" },
  action: { name: "action" },
  say: { text: "text" },
  capture: { x: "num?", y: "num?", yaw: "num?", mode: "capmode" },
  request_passage: { text: "text", timeout_s: "num" },
  wait_for: { event: "event", timeout_s: "num" },
  record: { on: "bool" },
  shadow: { gid: "int", dist_m: "num" }, watch: { gid: "int" }, escort: { gid: "int", side: "side" },
  goto_label: { label: "text", speed: "speed", zone: "zone" }, return_home: { speed: "speed", zone: "zone" },
};
export const EDITOR_OPS = ["goto", "jump_to", "follow_path", "look_at", "scan", "wait", "action", "say", "capture", "request_passage", "wait_for", "record", "goto_label", "return_home"];
const r3 = (v) => Math.round(+v * 1000) / 1000;
export const Steps = {
  goto: (x, y, yaw = null, speed = "normal", zone = null) => clean({ op: "goto", x: r3(x), y: r3(y), yaw: yaw == null ? null : r3(yaw), speed, zone }),
  followPath: (points, speed = "normal", zone = null) => clean({ op: "follow_path", points: points.map((p) => [r3(p[0]), r3(p[1])]), speed, smooth: true, zone }),
  jumpTo: (x, y, maxJumps = 6) => ({ op: "jump_to", x: r3(x), y: r3(y), max_jumps: maxJumps }),
  lookAt: (x, y, z = 0) => ({ op: "look_at", x: r3(x), y: r3(y), z: r3(z) }),
  scan: (deg = 360) => ({ op: "scan", deg }),
  wait: (s = 5) => ({ op: "wait", s }),
  say: (text) => ({ op: "say", text }),
  action: (name = "hello") => ({ op: "action", name }),
  shadow: (gid, dist = 3.0) => ({ op: "shadow", gid, dist_m: dist }),
  watch: (gid) => ({ op: "watch", gid }),
  escort: (gid, side = "right") => ({ op: "escort", gid, side, dist_m: 2.5 }),
  capture: (x = null, y = null, yaw = null) => clean({ op: "capture", x: x == null ? null : r3(x), y: y == null ? null : r3(y), yaw: yaw == null ? null : r3(yaw), cams: "all", mode: "max" }),
  requestPassage: (text = DEFAULT_PASSAGE_TEXT, timeout = 600) => ({ op: "request_passage", text, repeat_s: 30, ahead_m: 1.5, clear_for_s: 2.0, timeout_s: timeout, on_timeout: "alert" }),
  waitFor: (event = "operator", timeout = 120) => ({ op: "wait_for", event, timeout_s: timeout }),
  record: (on = true) => ({ op: "record", on, mode: "evidence" }),
  // "inspect here": stop 1.5 m short facing the point (fine), 360° scan, max-res photo
  inspect: (pose, x, y) => {
    const d = Math.hypot(x - pose.x, y - pose.y), yaw = Math.atan2(y - pose.y, x - pose.x);
    const k = d > 1.5 ? (d - 1.5) / d : 0;
    return [Steps.goto(pose.x + (x - pose.x) * k, pose.y + (y - pose.y) * k, yaw, "normal", "fine"), Steps.scan(360), Steps.capture()];
  },
};
function clean(o) { for (const k of Object.keys(o)) if (o[k] == null) delete o[k]; return o; }
export function makeMission(steps, name = "", source = "ui", onFail = "stop") {
  return { name: name || `${source}: ${steps.map((s) => s.op).join("→")}`.slice(0, 80), source, steps, on_fail: onFail };
}
export function needsDeadman(steps) { return (steps || []).some((s) => s.op === "jump_to" || s.speed === "sprint"); }
export const TERMINAL_STATES = ["done", "failed", "aborted"];
export const WAITING_OPS = ["request_passage", "wait_for"];

const del = (url, ms = 1500) => fetchJSON(url, ms, { method: "DELETE" });
const listOf = (d, key) => (Array.isArray(d) ? d : d && Array.isArray(d[key]) ? d[key] : []);

// tolerant plan normaliser (planner / executor shapes may differ slightly)
export function normalizePlan(p) {
  if (!p || typeof p !== "object") return null;
  if (p.plan && typeof p.plan === "object" && !p.path) p = p.plan;
  let path = p.path || p.points || p.global_path || null;
  if (path && !Array.isArray(path) && Array.isArray(path.points)) path = path.points;
  let v = p.v || p.speeds || p.speed_profile || p.v_profile || null;
  let landings = p.landings || (p.jump_plan && p.jump_plan.landings) || p.jumps || [];
  const warnings = [...(p.warnings || [])];
  if (!path && Array.isArray(p.steps)) {
    path = []; v = []; landings = [...landings];
    for (const s of p.steps) {
      const q = normalizePlan(s);
      if (!q) continue;
      path.push(...q.path); v.push(...q.v); landings.push(...q.landings);
      for (const w of q.warnings) if (!warnings.includes(w)) warnings.push(w);
    }
  }
  path = (path || []).filter((q) => Array.isArray(q) && q.length >= 2).map((q) => [+q[0], +q[1]]);
  if (!Array.isArray(v) || v.length !== path.length) v = path.map(() => 0.6);
  return {
    ok: p.ok !== false, path, v: v.map(Number), landings: (landings || []).map((q) => [+q[0], +q[1]]),
    length_m: +p.length_m || 0, eta_s: +(p.eta_s ?? p.eta ?? 0), warnings, reason: p.reason || "",
    timeline: p.timeline || null, steps: p.steps || null,
  };
}

export class MissionApi {
  constructor(EP, { getCtx, onState, onAlert, onLink } = {}) {
    this.E = missionEndpoints(EP);
    this.onState = onState || (() => {});
    this.onAlert = onAlert || (() => {});
    this.onLink = onLink || (() => {});
    this.mock = new MockMissionServer(getCtx || (() => ({ pose: { x: 0, y: 0, yaw: 0 }, persons: [] })), (m) => this._route(m));
    this.live = false;
    this.ws = null;
    this.clientId = `omniview-${Math.random().toString(36).slice(2, 8)}`;
    if (this.E.demo) this.onLink("demo");
    else { this.probe(); setInterval(() => { if (!this.live) this.probe(); }, 15000); }
  }

  get mode() { return this.live ? "live" : this.E.demo ? "demo" : "mock"; }

  async probe() {
    const r = await fetchJSON(`${this.E.http}/health`, 1500);
    this.live = r.ok;
    this.onLink(r.ok ? "live" : "down");
    if (r.ok && !this.ws) {
      this.ws = new ReconnectingWS(`${this.E.ws}/ws/mission`, {
        onState: (s) => this.onLink(s),
        onMessage: (txt) => { let m = null; try { m = JSON.parse(txt); } catch (_) { return; } this._route(m); },
      });
    }
    return r.ok;
  }

  // WS / mock message router: state vs event vs alert (several envelope styles)
  _route(m) {
    if (!m || typeof m !== "object") return;
    if (Array.isArray(m)) { m.forEach((x) => this._route(x)); return; }
    const ch = m.channel || m.type || m.topic || "";
    const d = m.data && typeof m.data === "object" ? m.data : m;
    if (/alert/.test(ch) || d.kind === "alert" || m.kind === "alert") this.onAlert(d);
    else if (/event/.test(ch) || (d.kind && !d.state)) { if (d.kind === "alert" || d.level) this.onAlert(d); }
    else if (d.state || d.mission_id) this.onState(d);
  }

  async _call(fn, mockFn) {
    if (this.live) {
      const r = await fn();
      if (r.ok || r.status) return { ...r, mock: false };   // HTTP error from a live backend: report it
      this.live = false; this.onLink("down");
    }
    try { return { ok: true, data: await mockFn(), mock: true }; }
    catch (e) { return { ok: false, data: null, mock: true, error: String(e && e.message || e) }; }
  }

  plan(step) {
    return this._call(() => postJSON(`${this.E.http}/plan`, step, 4000), () => this.mock.plan(step));
  }
  preview(mission) {
    return this._call(() => postJSON(`${this.E.http}/missions?execute=0`, mission, 5000), () => ({ plan: this.mock.previewMission(mission) }));
  }
  create(mission) {
    return this._call(() => postJSON(`${this.E.http}/missions`, mission, 5000), () => this.mock.create(mission));
  }
  control(id, verb) {
    return this._call(() => postJSON(`${this.E.http}/missions/${encodeURIComponent(id)}/${verb}`, {}, 2000), () => this.mock[verb](id));
  }
  nl(text) {
    return this._call(() => postJSON(`${this.E.http}/nl`, { text }, 30000), () => this.mock.nl(text));
  }
  zones() { return this._call(() => fetchJSON(`${this.E.http}/zones`, 2000), () => this.mock.zones()).then((r) => listOf(r.data, "zones")); }
  addZone(z) { return this._call(() => postJSON(`${this.E.http}/zones`, z, 2000), () => this.mock.addZone(z)); }
  delZone(id) { return this._call(() => del(`${this.E.http}/zones/${encodeURIComponent(id)}`), () => this.mock.delZone(id)); }
  labels() { return this._call(() => fetchJSON(`${this.E.http}/labels`, 2000), () => this.mock.labels()).then((r) => listOf(r.data, "labels")); }
  addLabel(l) { return this._call(() => postJSON(`${this.E.http}/labels`, l, 2000), () => this.mock.addLabel(l)); }
  delLabel(name) { return this._call(() => del(`${this.E.http}/labels/${encodeURIComponent(name)}`), () => this.mock.delLabel(name)); }
  home() { return this._call(() => fetchJSON(`${this.E.http}/home`, 2000), () => this.mock.world.home).then((r) => (r.data && r.data.home !== undefined ? r.data.home : r.data)); }
  setHome(h) { return this._call(() => postJSON(`${this.E.http}/home`, h, 2000), () => this.mock.setHome(h)); }

  continue(id) { return this.control(id, "continue"); }

  // safety_guard dead-man heartbeat (demo: local only)
  deadman() {
    if (this.E.demo) return Promise.resolve({ ok: true, mock: true });
    return postJSON(`${this.E.safety}/deadman`, { client_id: this.clientId }, 300);
  }
}

// evidence recorder on omni :9114 (§9.3). Demo: local flag only.
export class RecorderApi {
  constructor(EP) { this.http = EP.omniHttp; this.demo = EP.demoForced; this.mockOn = false; }
  async status() {
    if (this.demo) return { ok: true, data: { state: this.mockOn ? "evidence" : "idle", mock: true } };
    return fetchJSON(`${this.http}/record/status`, 1200);
  }
  async start(reason = "operator") {
    if (this.demo) { this.mockOn = true; return { ok: true, mock: true }; }
    return postJSON(`${this.http}/record/start`, { reason }, 2000);
  }
  async stop() {
    if (this.demo) { this.mockOn = false; return { ok: true, mock: true }; }
    return postJSON(`${this.http}/record/stop`, {}, 2000);
  }
}
export function recActive(st) {
  if (!st || typeof st !== "object") return false;
  const s = String(st.state || st.mode || "").toLowerCase();
  return st.recording === true || st.active === true || /evidence|record|active|on/.test(s);
}
