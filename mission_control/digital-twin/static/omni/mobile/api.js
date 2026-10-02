// NERO OmniVision mobile — thin API layer: mission :9116, safety_guard :9115,
// omni :9114 (record / incidents), digital-twin same-origin (/api/estop).
// Nothing throws: every call resolves {ok, data, mock?}. When the mission
// backend is unreachable (or ?demo=1) a local mock executes missions so the
// whole app stays demonstrable offline. Contract: mission/CONTRACT.md §1, §2, §9.

import { endpoints, fetchJSON, postJSON, ReconnectingWS } from "../net.js";
import { MiniMock } from "./mock.js";

const params = new URLSearchParams(location.search);

function hostPort(param, defPort) {
  let h = params.get(param) || location.hostname || "127.0.0.1";
  h = h.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const hasPort = h.startsWith("[") ? /\]:\d+$/.test(h) : /^[^:]+:\d+$/.test(h);
  return hasPort ? h : `${h}:${defPort}`;
}

export function allEndpoints() {
  const EP = endpoints();
  const secure = location.protocol === "https:";
  const m = hostPort("mission", 9116);
  return { ...EP, missionHttp: `${secure ? "https" : "http"}://${m}`, missionWs: `${secure ? "wss" : "ws"}://${m}` };
}

// tolerant plan normaliser (planner / executor / mock shapes)
export function normalizePlan(p) {
  if (!p || typeof p !== "object") return null;
  if (p.plan && typeof p.plan === "object" && !p.path) p = p.plan;
  let path = p.path || p.points || p.global_path || null;
  if (path && !Array.isArray(path) && Array.isArray(path.points)) path = path.points;
  let v = p.v || p.speeds || p.speed_profile || p.v_profile || null;
  const warnings = [...(p.warnings || [])];
  let landings = p.landings || (p.jump_plan && p.jump_plan.landings) || [];
  if ((!path || !path.length) && Array.isArray(p.steps)) {
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
  };
}

export class Api {
  // getCtx: () => {pose:{x,y,yaw}, persons:[{gid,wx,wy}]}
  constructor({ getCtx, onMission, onAlert, onLink } = {}) {
    this.E = allEndpoints();
    this.demo = this.E.demoForced;
    this.getCtx = getCtx || (() => ({ pose: { x: 0, y: 0, yaw: 0 }, persons: [] }));
    this.onMission = onMission || (() => {});
    this.onAlert = onAlert || (() => {});
    this.onLink = onLink || (() => {});
    this.clientId = `mobile-${Math.random().toString(36).slice(2, 8)}`;
    this.live = false;
    this.ws = null;
    this.mock = new MiniMock(this.getCtx, (m) => this._route(m));
    this.mockKind = "mini";
    this.rec = { on: false, incident_id: null, mock: true };
    this.mockIncidents = [];
    // M4's richer mock (zones, labels, timeline) if present — optional, never required
    import("../mission_mock.js").then((mod) => {
      if (mod && typeof mod.MockMissionServer === "function") {
        const mm = new mod.MockMissionServer(this.getCtx, (m) => this._route(m));
        if (typeof mm.plan === "function" && typeof mm.tick === "function" && typeof mm.create === "function") { this.mock = mm; this.mockKind = "m4"; }
      }
    }).catch(() => {});
    if (this.demo) this.onLink("mission", "demo");
    else { this.probe(); setInterval(() => { if (!this.live) this.probe(); }, 15000); }
  }

  get mode() { return this.live ? "live" : this.demo ? "demo" : "mock"; }
  get mockPose() { return this.mock && this.mock.pose ? this.mock.pose : null; }

  async probe() {
    const r = await fetchJSON(`${this.E.missionHttp}/health`, 1500);
    this.live = r.ok;
    this.onLink("mission", r.ok ? "live" : "down");
    if (r.ok && !this.ws) {
      this.ws = new ReconnectingWS(`${this.E.missionWs}/ws/mission`, {
        onState: (s) => this.onLink("mission", s),
        onMessage: (txt) => { let m = null; try { m = JSON.parse(txt); } catch (_) { return; } this._route(m); },
      });
    }
    return r.ok;
  }

  _route(m) {
    if (!m || typeof m !== "object") return;
    if (Array.isArray(m)) { m.forEach((x) => this._route(x)); return; }
    const ch = m.channel || m.type || m.topic || "";
    const d = m.data && typeof m.data === "object" ? m.data : m;
    if (/alert/.test(ch) || d.kind === "alert") this.onAlert(d);
    else if (/event/.test(ch) || (d.kind && !d.state)) { if (d.level || d.text) this.onAlert(d); }
    else if (d.state || d.mission_id) this.onMission(d);
  }

  async _call(fn, mockFn) {
    if (this.live) {
      const r = await fn();
      if (r.ok || r.status) return { ...r, mock: false };
      this.live = false; this.onLink("mission", "down");
    }
    try { return { ok: true, data: await mockFn(), mock: true }; }
    catch (e) { return { ok: false, data: null, mock: true, error: String((e && e.message) || e) }; }
  }

  // ---- mission :9116 ----
  plan(step) { return this._call(() => postJSON(`${this.E.missionHttp}/plan`, { step, ...step }, 4000), () => this.mock.plan(step)); }
  preview(mission) {
    return this._call(() => postJSON(`${this.E.missionHttp}/missions?execute=0`, mission, 5000),
      () => (this.mock.previewMission ? { plan: this.mock.previewMission(mission) } : this.mock.create(mission, { execute: false })));
  }
  create(mission) { return this._call(() => postJSON(`${this.E.missionHttp}/missions`, mission, 5000), () => this.mock.create(mission)); }
  control(id, verb) {
    return this._call(() => postJSON(`${this.E.missionHttp}/missions/${encodeURIComponent(id)}/${verb}`, {}, 2500),
      () => (this.mock[verb] ? this.mock[verb](id) : null));
  }
  nl(text) { return this._call(() => postJSON(`${this.E.missionHttp}/nl`, { text }, 30000), () => this.mock.nl(text)); }
  tick(dt, deadman) { if (!this.live && this.mock && this.mock.tick) this.mock.tick(dt, deadman); }

  // ---- safety_guard :9115 ----
  deadman() {
    if (this.demo) return Promise.resolve({ ok: true, mock: true });
    return postJSON(`${this.E.safetyHttp}/deadman`, { client_id: this.clientId }, 300);
  }
  safetyState() { return this.demo ? Promise.resolve({ ok: false }) : fetchJSON(`${this.E.safetyHttp}/state`, 800); }
  async estop(missionId) {
    const jobs = [];
    if (!this.demo) {
      jobs.push(postJSON(`${this.E.safetyHttp}/stop`, { source: "mobile" }, 1500));
      if (this.E.sameOrigin) jobs.push(postJSON("/api/estop", {}, 1500));
    }
    if (missionId) jobs.push(this.control(missionId, "abort"));
    else if (!this.live && this.mock.abort) this.mock.abort(null);
    const res = await Promise.all(jobs);
    return { ok: this.demo || res.some((r) => r && r.ok), res };
  }

  // ---- omni :9114 recorder (§9.3) ----
  async record(on, reason = "operator") {
    if (!this.demo) {
      const r = on ? await postJSON(`${this.E.omniHttp}/record/start`, { reason }, 2500) : await postJSON(`${this.E.omniHttp}/record/stop`, {}, 2500);
      if (r.ok) { this.rec = { on, incident_id: r.data && r.data.incident_id, mock: false }; return { ok: true, data: r.data }; }
      if (r.status) return { ok: false, status: r.status };
    }
    // offline / demo: local mock incident
    if (on && !this.rec.on) {
      const id = `${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}_go2`;
      this.rec = { on: true, incident_id: id, mock: true };
      this.mockIncidents.unshift({ id, t: Date.now() / 1000, reason, state: "recording", mock: true });
    } else if (!on && this.rec.on) {
      const inc = this.mockIncidents.find((i) => i.id === this.rec.incident_id);
      if (inc) inc.state = "closed";
      this.rec = { on: false, incident_id: null, mock: true };
    }
    return { ok: true, mock: true, data: { state: on ? "evidence" : "idle", incident_id: this.rec.incident_id } };
  }
  async recordStatus() {
    if (this.demo) return { ok: true, data: { state: this.rec.on ? "evidence" : "idle" }, mock: true };
    const r = await fetchJSON(`${this.E.omniHttp}/record/status`, 1500);
    if (r.ok && r.data) this.rec = { on: /evid|rec|on|active/i.test(String(r.data.state || "")) || r.data.recording === true, incident_id: r.data.incident_id, mock: false };
    return r;
  }
  async incidents() {
    if (!this.demo) {
      const r = await fetchJSON(`${this.E.omniHttp}/incidents`, 2500);
      if (r.ok) {
        const d = r.data;
        const list = Array.isArray(d) ? d : (d && (d.incidents || d.items)) || [];
        return { ok: true, list: list.map((i) => this._normInc(i)), mock: false };
      }
    }
    return { ok: true, list: this.mockIncidents.map((i) => this._normInc(i)), mock: true };
  }
  _normInc(i) {
    const id = i.id || i.incident_id || i.name || "";
    let thumb = i.thumb || i.thumbnail || i.thumb_url || null;
    if (!thumb && Array.isArray(i.best) && i.best.length) thumb = typeof i.best[0] === "string" ? i.best[0] : i.best[0].url || i.best[0].path;
    if (thumb && !/^(data:|https?:)/.test(thumb)) thumb = `${this.E.omniHttp}${thumb.startsWith("/") ? "" : `/incidents/${encodeURIComponent(id)}/`}${thumb}`;
    const t = +(i.t || i.ts || i.t_start || i.started || 0) || (Date.parse(i.created || "") / 1000) || null;
    return { id, t, reason: i.reason || i.trigger || "", state: i.state || (i.closed ? "closed" : ""), n: i.frames || i.n_frames || i.count || null, thumb, mock: !!i.mock, url: i.mock ? null : `${this.E.omniHttp}/incidents/${encodeURIComponent(id)}` };
  }
}
