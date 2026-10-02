// OmniView mission — 3D command controller (CONTRACT.md §7, §9).
// Goal click+drag(yaw) · profiles (Shift sprint / Alt stealth / Ctrl precise) ·
// freehand path (D) · jump (J) · waypoint chain (W) · person / floor context
// menu · zones (Z) / labels (L) / home (H) · dead-man (hold button / B / RB) ·
// gamepad · NL bar (/) · status widget + alerts · evidence REC.
// No direct /move from the UI: everything is a mission for :9116; the
// safety_guard (:9115) only receives dead-man heartbeats here.

import {
  MissionApi, RecorderApi, Steps, makeMission, needsDeadman, normalizePlan, recActive,
  DEFAULT_PASSAGE_TEXT, TERMINAL_STATES, WAITING_OPS,
} from "./mission_api.js";
import { MissionDraw } from "./mission_draw.js";
import { Ghost, buildTimeline } from "./mission_ghost.js";
import { WorldLayer } from "./mission_zones.js";
import { buildPanel, ask, showMenu, hideMenu, alertToast, stepText, renderChain, renderWorld } from "./mission_panel.js";
import { GamepadInput, BUTTONS, AXES, ACTIONS_HELP } from "./gamepad.js";
import { PROFILES } from "./mission_mock.js";

const HINTS = {
  view: "click floor = goal preview · right-click = menu · drag = orbit",
  goal: "<b>GOAL</b> click + drag = position + final heading · Shift sprint · Alt stealth · Ctrl precise",
  draw: "<b>DRAW</b> drag a path on the floor → follow_path",
  jump: "<b>JUMP</b> click target → landing rings (dead-man required)",
  zone: "<b>ZONE</b> click vertices · click first / Enter = close · Backspace = undo",
  label: "<b>LABEL</b> click floor → name it",
  home: "<b>HOME</b> click + drag = home pose",
};
const DRAG_TOOLS = new Set(["goal", "draw", "jump", "zone", "label", "home"]);
const DM_HZ = 10;

export class MissionUI {
  // ctx: {scene, camera, renderer, rig, persons, robot, cloud, state, EP, toast}
  constructor(ctx) {
    Object.assign(this, ctx);
    this.$ = buildPanel();
    this.dom = this.renderer.domElement;
    this.draw = new MissionDraw(this.scene);
    this.ghost = new Ghost(this.scene, this.robot);
    this.world = new WorldLayer(this.scene, document.getElementById("labels"));
    this.api = new MissionApi(this.EP, {
      getCtx: () => ({ pose: this.state.pose, persons: this.persons.list }),
      onState: (m) => this._onState(m),
      onAlert: (a) => this._onAlert(a),
      onLink: (s) => { this.link = s; },
    });
    this.rec = new RecorderApi(this.EP);
    this.link = this.api.mode;
    this.tool = "view";
    this.profile = "normal";
    this.precision = "fine";
    this.zoneKind = "no_go";
    this.preview = null;           // {kind, steps, plan, mission_id?, needDm}
    this.status = null;            // last mission state message
    this.active = null;            // {id, plan}
    this.chain = []; this.chainOpen = false; this.chainSel = -1;
    this.draft = null;             // zone / freehand points
    this.gesture = null;
    this.dmHolders = new Set(); this.dmTimer = null; this.dmSent = 0; this.dmFail = 0;
    this.pad = new GamepadInput({ onButton: (n, d) => this._padButton(n, d) });
    this.cursor = null; this.padJump = false; this.autoGo = false;
    this.recOn = false;
    this.ray = new THREE.Raycaster();
    this.ray.params.Points.threshold = 0.08;
    this._bindUi();
    this._bindPointer();
    addEventListener("keyup", (e) => { if (e.key.toLowerCase() === "b") this.dmRelease("key"); });
    addEventListener("blur", () => this.dmReleaseAll());
    this.setTool("view");
    this.setProfile("normal");
    this.refreshWorld();
    this._recPoll();
    this._statusUi();
  }

  // ---------------- helpers ----------------
  get dmHeld() { return this.dmHolders.size > 0; }
  _rayAt(cx, cy) {
    const r = this.dom.getBoundingClientRect();
    this.ray.setFromCamera({ x: ((cx - r.left) / r.width) * 2 - 1, y: -((cy - r.top) / r.height) * 2 + 1 }, this.camera);
    return this.ray;
  }
  _floor(cx, cy) {
    const ray = this._rayAt(cx, cy);
    const p = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p)) return null;
    if (p.distanceTo(this.camera.position) > 150) return null;
    return { x: p.x, y: -p.z };
  }
  _cloudHit(cx, cy) {
    const pts = this.cloud && this.cloud.points;
    if (!pts || !this.cloud.visible || !this.cloud.count) return null;
    const hit = this._rayAt(cx, cy).intersectObject(pts, false)[0];
    return hit ? { x: hit.point.x, y: -hit.point.z, z: hit.point.y } : null;
  }
  _profileFor(e) { return e && e.shiftKey ? "sprint" : e && e.altKey ? "stealth" : e && (e.ctrlKey || e.metaKey) ? "precise" : this.profile; }
  _pose() { return this.state.pose || { x: 0, y: 0, yaw: 0 }; }

  // ---------------- UI wiring ----------------
  _bindUi() {
    const $ = this.$;
    document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => {
      const t = b.dataset.tool;
      if (t === "chain") this.toggleChain();
      else if (t === "nl") this.openNl();
      else if (t === "pad") this.openPadDialog();
      else this.setTool(this.tool === t ? "view" : t);
    }));
    document.querySelectorAll("[data-prof]").forEach((b) => b.addEventListener("click", () => this.setProfile(b.dataset.prof, true)));
    document.querySelectorAll("[data-zk]").forEach((b) => b.addEventListener("click", () => { this.zoneKind = b.dataset.zk; this._toolUi(); }));
    document.querySelectorAll("[data-prec]").forEach((b) => b.addEventListener("click", () => this.setPrecision(b.dataset.prec)));
    $("mc-go").addEventListener("click", () => this.go());
    $("mc-clear").addEventListener("click", () => this.clearPreview());
    $("mc-addchain").addEventListener("click", () => { if (this.preview) { this.chain.push(...this.preview.steps); this.chainOpen = false; this.toggleChain(); this.previewChain(); } });
    $("mc-approve").addEventListener("click", () => this.approveNl());
    $("mc-reject").addEventListener("click", () => this.rejectNl());
    $("mc-pause").addEventListener("click", () => this.pauseResume());
    $("mc-abort").addEventListener("click", () => this.abort("operator"));
    $("mc-continue").addEventListener("click", () => this.continueMission());
    $("mc-play").addEventListener("click", () => { this.ghost.playing = !this.ghost.playing; $("mc-play").textContent = this.ghost.playing ? "❚❚" : "▶"; });
    $("mc-scrub").addEventListener("input", (e) => { this.ghost.playing = false; $("mc-play").textContent = "▶"; this.ghost.seek(+e.target.value / 1000); });
    this.ghost.onTime = (t, T, s) => this._ghostUi(t, T, s);
    // chain editor
    $("mc-chain-add").addEventListener("click", () => this.addChainOp($("mc-chain-op").value));
    $("mc-chain-prev").addEventListener("click", () => this.previewChain());
    $("mc-chain-clear").addEventListener("click", () => { this.chain = []; this.chainSel = -1; this._chainUi(); this.clearPreview(); });
    $("mc-world-tg").addEventListener("click", () => { $("mc-world-p").classList.toggle("collapsed"); $("mc-world-tg").textContent = $("mc-world-p").classList.contains("collapsed") ? "▸" : "▾"; });
    // dead-man button
    const dm = $("mc-deadman");
    dm.addEventListener("pointerdown", (e) => { e.preventDefault(); try { dm.setPointerCapture(e.pointerId); } catch (_) {} this.dmPress("btn"); });
    for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) dm.addEventListener(ev, () => this.dmRelease("btn"));
    dm.addEventListener("contextmenu", (e) => e.preventDefault());
    // NL bar
    const inp = $("mc-nl-in");
    inp.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") this.sendNl(inp.value);
      else if (e.key === "Escape") this.closeNl();
    });
    $("mc-nl-send").addEventListener("click", () => this.sendNl(inp.value));
    // REC / PAD chips
    const recEl = $("mc-rec");
    if (recEl) recEl.addEventListener("click", () => this.toggleRec());
    const padEl = $("mc-pad");
    if (padEl) padEl.addEventListener("click", () => this.openPadDialog());
    $("mc-pad-close").addEventListener("click", () => { this.pad.cancelCapture(); $("mc-padd").classList.remove("open"); });
    $("mc-pad-reset").addEventListener("click", () => { this.pad.resetBindings(); this._padMapUi(); });
    addEventListener("pointerdown", (e) => { if (!e.target.closest || !e.target.closest("#mc-ctx")) hideMenu(); }, true);
  }

  setTool(t) {
    this.tool = t;
    if (t !== "zone" && t !== "draw") { this.draft = null; this.draw.setDraft(null); }
    this._toolUi();
  }
  _toolUi() {
    document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on",
      b.dataset.tool === this.tool || (b.dataset.tool === "chain" && this.chainOpen) || (b.dataset.tool === "nl" && this.$("mc-nl").classList.contains("open"))));
    this.$("mc-zonekind").style.display = this.tool === "zone" ? "grid" : "none";
    document.querySelectorAll("[data-zk]").forEach((b) => b.classList.toggle("on", b.dataset.zk === this.zoneKind));
    let hint = HINTS[this.tool] || "";
    if (this.chainOpen && ["goal", "jump", "draw", "view"].includes(this.tool)) hint += " · <b>→ CHAIN</b>";
    if (this.padJump) hint = "<b>PAD JUMP</b> A = plan jump to cursor · X = back to goal";
    this.$("mc-hint").innerHTML = hint;
    document.body.classList.toggle("mc-tool-active", DRAG_TOOLS.has(this.tool));
  }
  setProfile(p, replan = false) {
    this.profile = p;
    document.querySelectorAll("[data-prof]").forEach((b) => b.classList.toggle("on", b.dataset.prof === p));
    if (replan && this.preview && this.preview.kind === "single") {
      const s = this.preview.steps[0];
      if ("speed" in s || s.op === "goto" || s.op === "follow_path") this.planSingle({ ...s, speed: p });
    }
  }
  setPrecision(z) {
    this.precision = z;
    document.querySelectorAll("[data-prec]").forEach((b) => b.classList.toggle("on", b.dataset.prec === z));
    if (this.preview && this.preview.kind === "single" && this.preview.steps[0].op === "goto") this.planSingle({ ...this.preview.steps[0], zone: z });
  }

  // ---------------- pointer ----------------
  _bindPointer() {
    const vp = document.getElementById("viewport");
    // capture phase on the parent: consume drags for drawing tools before the camera rig sees them
    vp.addEventListener("pointerdown", (e) => {
      hideMenu();
      if (e.button !== 0 || !DRAG_TOOLS.has(this.tool) || e.target !== this.dom) return;
      const f = this._floor(e.clientX, e.clientY);
      if (!f) return;
      e.stopPropagation();
      try { this.dom.setPointerCapture(e.pointerId); } catch (_) {}
      this.gesture = { id: e.pointerId, start: f, cur: f, prof: this._profileFor(e), moved: 0 };
      if (this.tool === "draw") { this.draft = [[f.x, f.y]]; this.draw.setDraft(this.draft, 0x00d8ff); }
      if (this.tool === "goal" || this.tool === "home") this.draw.setGoal(f.x, f.y, null, this.tool === "home" ? 0x35f28a : 0x00d8ff);
    }, true);
    vp.addEventListener("pointermove", (e) => {
      const g = this.gesture;
      if (!g || e.pointerId !== g.id) return;
      e.stopPropagation();
      const f = this._floor(e.clientX, e.clientY);
      if (!f) return;
      g.moved += Math.hypot(f.x - g.cur.x, f.y - g.cur.y);
      g.cur = f;
      if (this.tool === "draw") {
        const l = this.draft[this.draft.length - 1];
        if (Math.hypot(f.x - l[0], f.y - l[1]) > 0.12) { this.draft.push([f.x, f.y]); this.draw.setDraft(this.draft, 0x00d8ff); }
      } else if (this.tool === "goal" || this.tool === "home") {
        const d = Math.hypot(f.x - g.start.x, f.y - g.start.y);
        this.draw.setGoal(g.start.x, g.start.y, d > 0.2 ? Math.atan2(f.y - g.start.y, f.x - g.start.x) : null, this.tool === "home" ? 0x35f28a : 0x00d8ff);
      }
    }, true);
    const end = (e) => {
      const g = this.gesture;
      if (!g || e.pointerId !== g.id) return;
      e.stopPropagation();
      this.gesture = null;
      if (e.type === "pointercancel") return;
      const prof = this._profileFor(e) !== this.profile ? this._profileFor(e) : g.prof;
      const d = Math.hypot(g.cur.x - g.start.x, g.cur.y - g.start.y);
      const yaw = d > 0.2 ? Math.atan2(g.cur.y - g.start.y, g.cur.x - g.start.x) : null;
      switch (this.tool) {
        case "goal": this.commit([Steps.goto(g.start.x, g.start.y, yaw, prof, this.precision)]); break;
        case "jump": this.commit([Steps.jumpTo(g.start.x, g.start.y)]); break;
        case "draw":
          if (this.draft && this.draft.length >= 2) this.commit([Steps.followPath(this.draft, prof)]);
          this.draft = null; this.draw.setDraft(null);
          break;
        case "zone": this.zoneClick(g.start); break;
        case "label": this.placeLabel(g.start); break;
        case "home": this.placeHome(g.start, yaw); break;
      }
    };
    vp.addEventListener("pointerup", end, true);
    vp.addEventListener("pointercancel", end, true);
    vp.addEventListener("dblclick", (e) => { if (this.tool === "zone") { e.stopPropagation(); this.closeZone(); } }, true);
    // VIEW tool: plain click on the floor = goal preview (after omni.js person pick)
    this.dom.addEventListener("pointerup", (e) => {
      if (this.tool !== "view" || e.button !== 0 || this.rig.dragMoved > 6) return;
      const ray = this._rayAt(e.clientX, e.clientY);
      if (this.persons.pick(ray) != null) return;
      const f = this._floor(e.clientX, e.clientY);
      if (!f) return;
      const pose = this._pose();
      const yaw = Math.atan2(f.y - pose.y, f.x - pose.x);
      this.commit([Steps.goto(f.x, f.y, yaw, this._profileFor(e), this.precision)]);
    });
    this.dom.addEventListener("contextmenu", (e) => { e.preventDefault(); this.contextMenu(e.clientX, e.clientY); });
  }

  // ---------------- context menu ----------------
  contextMenu(cx, cy) {
    const ray = this._rayAt(cx, cy);
    const gid = this.persons.pick(ray);
    if (gid != null) {
      const p = this.persons.list.find((q) => q.gid === gid);
      const g = typeof gid === "number" ? gid : parseInt(gid, 10) || 0;
      showMenu(cx, cy, `PERSON #${gid}${p ? ` · ${p.range.toFixed(1)} m` : ""}`, [
        { label: "ÁRNYÉK · shadow", hint: "3 m", fn: () => this.commit([Steps.shadow(g, 3.0)]) },
        { label: "FIGYELD · watch", hint: "in place", fn: () => this.commit([Steps.watch(g)]) },
        { label: "KÍSÉRD · escort", hint: "right 2.5 m", fn: () => this.commit([Steps.escort(g, "right")]) },
        { label: "TRACK TARGET", hint: "select", fn: () => this.persons.select(gid) },
        { label: "REC EVIDENCE", hint: "omni /record", fn: () => this.toggleRec(true) },
      ]);
      return;
    }
    const c = this._cloudHit(cx, cy);
    const f = this._floor(cx, cy);
    const pt = c || (f && { ...f, z: 0 });
    if (!pt) return;
    const pose = this._pose();
    const face = Math.atan2(pt.y - pose.y, pt.x - pose.x);
    showMenu(cx, cy, `${c ? "POINT" : "FLOOR"} ${pt.x.toFixed(1)}, ${pt.y.toFixed(1)}${c ? ` · z ${pt.z.toFixed(1)}` : ""}`, [
      { label: "MENJ IDE · go here", hint: this.profile, fn: () => this.commit([Steps.goto(pt.x, pt.y, face, this.profile, this.precision)]) },
      { label: "NÉZD MEG · look at", hint: "turn", fn: () => this.commit([Steps.lookAt(pt.x, pt.y, pt.z || 0)]) },
      { label: "VIZSGÁLD · inspect here", hint: "1.5 m + scan", fn: () => this.commit(Steps.inspect(pose, pt.x, pt.y)) },
      { label: "FOTÓ ITT (fine)", hint: "capture max", fn: () => this.commit([Steps.capture(pt.x, pt.y, face)]) },
      { label: "AJTÓN ÁTKÉRÉS", hint: "request_passage", fn: async () => {
        const txt = await ask("request_passage — text to say", DEFAULT_PASSAGE_TEXT);
        if (txt) this.commit([Steps.goto(pt.x, pt.y, face, "normal", "fine"), Steps.requestPassage(txt)]);
      } },
      { label: "UGORJ IDE · jump here", hint: "dead-man", fn: () => this.commit([Steps.jumpTo(pt.x, pt.y)]) },
      { label: "+ WAYPOINT (chain)", hint: "W", fn: () => { this.chain.push(Steps.goto(pt.x, pt.y, null, this.profile)); if (!this.chainOpen) this.toggleChain(); else this._chainUi(); this.previewChain(); } },
      { label: "CÍMKE IDE · label", hint: "L", fn: () => this.placeLabel(pt) },
      { label: "HOME IDE", hint: "H", fn: () => this.placeHome(pt, pose.yaw) },
    ]);
  }

  // ---------------- planning ----------------
  // steps from a scene gesture: chain open -> append, else single preview
  commit(steps) {
    if (this.chainOpen && steps.every((s) => !["shadow", "watch", "escort"].includes(s.op))) {
      this.chain.push(...steps);
      this._chainUi();
      this.previewChain();
      return;
    }
    if (steps.length === 1 && ["goto", "follow_path", "jump_to"].includes(steps[0].op)) this.planSingle(steps[0]);
    else this.previewMission(steps, "single");
  }

  async planSingle(step) {
    const seq = (this._seq = (this._seq || 0) + 1);
    this._showPreview({ kind: "single", steps: [step], plan: null, busy: true });
    const r = await this.api.plan(step);
    if (seq !== this._seq) return;
    if (!r.ok) { this._showPreview({ kind: "single", steps: [step], plan: { ok: false, path: [], v: [], landings: [], warnings: [`planner error ${r.status || ""} ${detail(r)}`], eta_s: 0 } }); return; }
    this._showPreview({ kind: "single", steps: [step], plan: normalizePlan(r.data), mock: r.mock });
  }

  async previewMission(steps, kind = "chain") {
    const seq = (this._seq = (this._seq || 0) + 1);
    const mission = makeMission(steps, "", "ui", this.$("mc-onfail").value);
    this._showPreview({ kind, steps, plan: null, busy: true });
    const r = await this.api.preview(mission);
    if (seq !== this._seq) return;
    const plan = r.ok ? normalizePlan(r.data) : { ok: false, path: [], v: [], landings: [], warnings: [`preview error ${r.status || ""} ${detail(r)}`], eta_s: 0 };
    this._showPreview({ kind, steps, plan, mock: r.mock });
  }
  previewChain() { if (this.chain.length) this.previewMission(this.chain.slice(), "chain"); else this.clearPreview(); }

  _showPreview(pv) {
    this.preview = pv;
    pv.needDm = needsDeadman(pv.steps);
    const $ = this.$;
    $("mc-plan").style.display = "block";
    const plan = pv.plan;
    const s0 = pv.steps[0] || {};
    $("mc-plan-title").textContent = pv.kind === "nl" ? "NL mission · pending approval" : pv.kind === "chain" ? `chain preview · ${pv.steps.length} steps` : `${s0.op} preview`;
    $("mc-plan-src").innerHTML = `<span class="link" data-state="${pv.mock ? "demo" : "live"}">${pv.busy ? "…" : pv.mock ? "LOCAL" : "PLANNER"}</span>`;
    $("mc-plan-steps").innerHTML = pv.steps.map((s, i) => `<div>${i + 1}. ${stepText(s)}</div>`).join("");
    $("mc-acts-plan").style.display = pv.kind === "nl" ? "none" : "grid";
    $("mc-acts-nl").style.display = pv.kind === "nl" ? "grid" : "none";
    $("mc-prec-row").style.display = pv.kind === "single" && s0.op === "goto" ? "flex" : "none";
    document.querySelectorAll("[data-prec]").forEach((b) => b.classList.toggle("on", b.dataset.prec === (s0.zone || this.precision)));
    // goal marker / waypoint markers
    const wps = [];
    pv.steps.forEach((s) => {
      if (s.x != null && s.y != null && ["goto", "capture"].includes(s.op)) wps.push({ x: s.x, y: s.y, yaw: s.yaw, zone: s.op === "capture" ? "fine" : s.zone || (pv.kind === "single" ? "fine" : "z30") });
      if (s.op === "follow_path" && s.points.length) { const l = s.points[s.points.length - 1]; wps.push({ x: l[0], y: l[1], zone: s.zone || "z30" }); }
    });
    const last = pv.steps[pv.steps.length - 1] || {};
    if (last.x != null && ["goto", "jump_to", "look_at", "capture"].includes(last.op)) this.draw.setGoal(last.x, last.y, last.op === "goto" || last.op === "capture" ? last.yaw : null, last.op === "jump_to" ? 0xffb020 : pv.needDm ? 0xff3344 : 0x00d8ff);
    else this.draw.hideGoal();
    this.draw.setWaypoints(wps);
    if (!plan) { $("mc-warn").innerHTML = "<li>planning…</li>"; this._goUi(); return; }
    this.draw.setPlan(plan);
    const vmax = plan.v.length ? Math.max(...plan.v) : 0;
    $("mc-eta").textContent = plan.eta_s ? `${plan.eta_s.toFixed(1)}s` : "--";
    const len = plan.length_m || pathLen(plan.path);
    $("mc-len").textContent = len ? `${len.toFixed(1)}m` : "--";
    $("mc-vmax").textContent = vmax ? vmax.toFixed(2) : "--";
    $("mc-jumps").textContent = `${plan.landings.length}`;
    const warns = [...plan.warnings];
    if (!plan.ok && plan.reason && !warns.length) warns.push(plan.reason);
    $("mc-warn").innerHTML = warns.map((w) => `<li class="${plan.ok ? "" : "bad"}">${escH(w)}</li>`).join("") + (plan.ok ? "" : `<li class="bad">PLAN NOT FEASIBLE${plan.reason ? `: ${escH(plan.reason)}` : ""}</li>`);
    this.ghost.setTimeline(buildTimeline(plan, this._pose()));
    this.ghost.playing = true; $("mc-play").textContent = "❚❚";
    this._goUi();
    if (this.autoGo) { this.autoGo = false; if (plan.ok) this.go(); }
  }

  _goUi() {
    const pv = this.preview, go = this.$("mc-go");
    if (!pv) return;
    const ok = pv.plan && pv.plan.ok;
    const blocked = pv.needDm && !this.dmHeld;
    go.disabled = !ok || blocked;
    go.textContent = !pv.plan ? "…" : !ok ? "BLOCKED" : blocked ? "HOLD DEAD-MAN" : "GO";
    go.classList.toggle("on", ok && !blocked && pv.needDm);
    this.$("mc-approve").disabled = !ok || blocked;
    this.$("mc-approve").textContent = blocked ? "HOLD DEAD-MAN" : "APPROVE";
  }

  clearPreview() {
    this._seq = (this._seq || 0) + 1;
    this.preview = null;
    this.$("mc-plan").style.display = "none";
    this.ghost.clear();
    this.draw.hideGoal();
    this.draw.setWaypoints(this.chainOpen ? this._chainWps() : []);
    if (this.active && this.active.plan) this.draw.setPlan(this.active.plan, { alpha: 0.55 }); else this.draw.clearPath();
  }

  async go() {
    const pv = this.preview;
    if (!pv || !pv.plan || !pv.plan.ok) return;
    if (pv.needDm && !this.dmHeld) { this.toast("SPRINT / JUMP: hold DEAD-MAN first", "warn"); return; }
    const mission = makeMission(pv.steps, "", this.padSource ? "gamepad" : "ui", this.$("mc-onfail").value);
    this.padSource = false;
    const r = await this.api.create(mission);
    if (!r.ok) { this.toast(`mission rejected: ${detail(r)}`, "crit"); alertToast("mission rejected", "crit", detail(r)); return; }
    const d = r.data || {};
    const plan = normalizePlan(d.plan) || pv.plan;
    this.active = { id: d.mission_id, plan: plan.path.length ? plan : pv.plan, steps: pv.steps };
    this.toast(`MISSION ${d.state || "queued"} · ${d.mission_id || ""}`, "info");
    if (pv.kind === "chain") { /* keep chain for re-use */ }
    this.clearPreview();
    this._onState({ mission_id: d.mission_id, state: d.state || "queued", step_index: 0, step_op: pv.steps[0].op, progress: 0, eta_s: pv.plan.eta_s });
  }

  async abort(why = "operator") {
    const id = (this.status && !TERMINAL_STATES.includes(this.status.state) && this.status.mission_id) || (this.active && this.active.id);
    if (!id) { if (this.preview) this.clearPreview(); return; }
    const r = await this.api.control(id, "abort");
    this.toast(r.ok ? `ABORT · ${why}` : `abort failed: ${detail(r)}`, r.ok ? "warn" : "crit");
  }
  async pauseResume() {
    const s = this.status;
    if (!s || !s.mission_id) return;
    const verb = s.state === "paused" ? "resume" : "pause";
    if (verb === "resume") {
      const st = this.active && this.active.steps ? this.active.steps[s.step_index || 0] : null;
      if (st && needsDeadman([st]) && !this.dmHeld) { this.toast("hold DEAD-MAN to resume sprint / jump", "warn"); return; }
    }
    const r = await this.api.control(s.mission_id, verb);
    if (!r.ok) this.toast(`${verb} failed: ${detail(r)}`, "crit");
  }
  async continueMission() {
    const s = this.status;
    if (!s || !s.mission_id) return;
    const r = await this.api.continue(s.mission_id);
    this.toast(r.ok ? "CONTINUE sent" : `continue failed: ${detail(r)}`, r.ok ? "info" : "crit");
  }
  onEstop() { this.dmReleaseAll(); this.abort("E-STOP"); }

  // ---------------- status / alerts ----------------
  _onState(m) {
    if (!m || !m.state) return;
    const prev = this.status;
    this.status = m;
    if (prev && prev.mission_id === m.mission_id && prev.state !== m.state && m.state === "waiting") alertToast(`mission waiting: ${m.step_op || ""}`, "warn", "press CONTINUE when clear");
    if (TERMINAL_STATES.includes(m.state)) {
      if (this.active && this.active.id === m.mission_id) {
        clearTimeout(this._clrT);
        this._clrT = setTimeout(() => { if (this.status && TERMINAL_STATES.includes(this.status.state)) { this.active = null; if (!this.preview) this.draw.clearPath(); } }, 2500);
      }
      if (prev && prev.state !== m.state && m.state === "failed") alertToast(`mission failed${m.error ? `: ${m.error}` : ""}`, "crit");
    } else if (this.active && this.active.id === m.mission_id && !this.preview) {
      this.draw.setPlan(this.active.plan, { alpha: 0.55 });
    } else if (!this.active && m.mission_id && !TERMINAL_STATES.includes(m.state) && m.state !== "pending_approval") {
      this.active = { id: m.mission_id, plan: null, steps: null };   // rule / gamepad / other client
    }
    this._statusUi();
  }
  _statusUi() {
    const $ = this.$, m = this.status || { state: "idle" };
    $("mc-status").dataset.state = m.state;
    $("mc-st-state").textContent = String(m.state).toUpperCase().replace("_", " ");
    $("mc-st-id").textContent = m.mission_id ? String(m.mission_id).slice(0, 14) : "";
    const total = m.steps_total || (this.active && this.active.steps && this.active.steps.length) || 0;
    $("mc-st-step").textContent = m.step_op ? `${(m.step_index || 0) + 1}${total ? `/${total}` : ""} ${m.step_op}` : "";
    $("mc-st-bar").style.width = `${Math.round(Math.max(0, Math.min(1, +m.progress || 0)) * 100)}%`;
    $("mc-st-eta").textContent = m.eta_s != null && !TERMINAL_STATES.includes(m.state) ? `${(+m.eta_s).toFixed(0)} s` : "--";
    $("mc-st-err").textContent = m.error ? String(m.error).slice(0, 40) : "";
    const live = m.mission_id && !TERMINAL_STATES.includes(m.state) && m.state !== "idle";
    $("mc-pause").disabled = !live || !["running", "paused", "queued", "waiting"].includes(m.state);
    $("mc-pause").textContent = m.state === "paused" ? "RESUME" : "PAUSE";
    $("mc-abort").disabled = !live;
    const waiting = m.state === "waiting" || (m.state === "running" && WAITING_OPS.includes(m.step_op));
    $("mc-continue").disabled = !live || !waiting;
    $("mc-continue").classList.toggle("on", !!(live && waiting));
  }
  _onAlert(a) {
    const text = a.text || a.message || a.msg || [a.kind !== "alert" ? a.kind : "", a.event, a.zone && `zone ${a.zone}`, a.gid != null && `#${a.gid}`].filter(Boolean).join(" · ") || "alert";
    const lvl = a.level || a.severity || (a.kind === "alert" ? "warn" : "info");
    alertToast(text, lvl, a.rule || a.rule_id || a.mission_id || "");
    if (lvl === "crit" || lvl === "error") this.toast(text, "crit");
  }

  // ---------------- dead-man ----------------
  dmPress(src) {
    const was = this.dmHeld;
    this.dmHolders.add(src);
    if (was) return;
    document.body.classList.add("mc-armed");
    const beat = () => this.api.deadman().then((r) => { this.dmSent++; this.dmFail = r && r.ok ? 0 : this.dmFail + 1; });
    beat();
    this.dmTimer = setInterval(beat, 1000 / DM_HZ);
    this.pad.rumble(60, 0.4);
    this._goUi();
  }
  dmRelease(src) {
    if (!this.dmHolders.delete(src) || this.dmHeld) return;
    clearInterval(this.dmTimer); this.dmTimer = null;
    document.body.classList.remove("mc-armed");
    this._goUi();
  }
  dmReleaseAll() { for (const s of [...this.dmHolders]) this.dmRelease(s); }

  // ---------------- zones / labels / home ----------------
  zoneClick(f) {
    if (!this.draft) this.draft = [];
    if (this.draft.length >= 3 && Math.hypot(f.x - this.draft[0][0], f.y - this.draft[0][1]) < 0.35) { this.closeZone(); return; }
    this.draft.push([f.x, f.y]);
    this.draw.setDraft(this.draft, { no_go: 0xff3344, watch: 0xffb020, patrol: 0x00d8ff }[this.zoneKind], true);
  }
  async closeZone() {
    const poly = this.draft;
    if (!poly || poly.length < 3) { this.toast("zone needs ≥ 3 points", "warn"); return; }
    this.draft = null; this.draw.setDraft(null);
    const n = (this.world.zones || []).length + 1;
    const name = await ask(`${this.zoneKind.toUpperCase()} zone name`, `${this.zoneKind}-${n}`);
    if (!name) return;
    const r = await this.api.addZone({ name, kind: this.zoneKind, polygon: poly.map((p) => [+p[0].toFixed(3), +p[1].toFixed(3)]) });
    if (!r.ok) this.toast(`zone rejected: ${detail(r)}`, "crit"); else this.toast(`ZONE ${name} saved`);
    this.refreshWorld();
  }
  async placeLabel(f) {
    const name = await ask("label name (kapu, garázs, töltő…)", "");
    if (!name) return;
    const r = await this.api.addLabel({ name, x: +f.x.toFixed(3), y: +f.y.toFixed(3) });
    if (!r.ok) this.toast(`label rejected: ${detail(r)}`, "crit"); else this.toast(`LABEL ${name}`);
    this.refreshWorld();
  }
  async placeHome(f, yaw) {
    const r = await this.api.setHome({ x: +f.x.toFixed(3), y: +f.y.toFixed(3), yaw: +(yaw || 0).toFixed(3) });
    this.draw.hideGoal();
    if (!r.ok) this.toast(`home rejected: ${detail(r)}`, "crit"); else this.toast("HOME set");
    this.refreshWorld();
  }
  async refreshWorld() {
    const [zones, labels, home] = await Promise.all([this.api.zones(), this.api.labels(), this.api.home()]);
    this.worldData = { zones, labels, home: home && Number.isFinite(+home.x) ? home : null };
    this.world.set(this.worldData);
    renderWorld(this.worldData, {
      delZone: async (id) => { await this.api.delZone(id); this.refreshWorld(); },
      delLabel: async (n) => { await this.api.delLabel(n); this.refreshWorld(); },
      goHome: () => this.commit([{ op: "return_home", speed: this.profile }]),
    });
  }

  // ---------------- chain editor ----------------
  toggleChain() {
    this.chainOpen = !this.chainOpen;
    this.$("mc-chain").style.display = this.chainOpen ? "block" : "none";
    this._chainUi();
    this._toolUi();
  }
  addChainOp(op) {
    const prev = [...this.chain].reverse().find((s) => s.x != null);
    const base = prev || this._pose();
    const ahead = { x: base.x + 1, y: base.y };
    const mk = {
      goto: () => Steps.goto(ahead.x, ahead.y, null, this.profile), jump_to: () => Steps.jumpTo(ahead.x, ahead.y),
      follow_path: () => Steps.followPath([[base.x, base.y], [ahead.x, ahead.y + 1]], this.profile),
      look_at: () => Steps.lookAt(ahead.x, ahead.y), scan: () => Steps.scan(360), wait: () => Steps.wait(5),
      action: () => Steps.action("hello"), say: () => ({ op: "say", text: "Ön megfigyelt területen tartózkodik." }),
      capture: () => Steps.capture(), request_passage: () => Steps.requestPassage(), wait_for: () => Steps.waitFor("operator"),
      record: () => Steps.record(true), goto_label: () => ({ op: "goto_label", label: ((this.worldData && this.worldData.labels[0]) || {}).name || "", speed: this.profile }),
      return_home: () => ({ op: "return_home", speed: this.profile }),
    }[op];
    if (!mk) return;
    this.chain.push(mk());
    this._chainUi();
    this.previewChain();
  }
  _chainWps() {
    return this.chain.filter((s) => s.x != null && (s.op === "goto" || s.op === "capture")).map((s) => ({ x: s.x, y: s.y, yaw: s.yaw, zone: s.op === "capture" ? "fine" : s.zone || "z30" }));
  }
  _chainUi() {
    renderChain(this.chain, {
      onChange: (i, k, v) => {
        if (k === "op") { const old = this.chain[i]; this.addChainOp(v); const n = this.chain.pop(); if (old.x != null && n.x != null) { n.x = old.x; n.y = old.y; } this.chain[i] = n; }
        else if (v == null) delete this.chain[i][k];
        else this.chain[i][k] = v;
        this._chainUi(); this.previewChain();
      },
      onMove: (i, d) => { const j = i + d; if (j < 0 || j >= this.chain.length) return; [this.chain[i], this.chain[j]] = [this.chain[j], this.chain[i]]; this._chainUi(); this.previewChain(); },
      onDel: (i) => { this.chain.splice(i, 1); this._chainUi(); this.previewChain(); },
      onSelect: (i) => { this.chainSel = i; this._chainUi(); },
    }, this.chainSel);
  }

  // ---------------- NL ----------------
  openNl() {
    const bar = this.$("mc-nl");
    bar.classList.add("open");
    this._toolUi();
    setTimeout(() => this.$("mc-nl-in").focus(), 0);
  }
  closeNl() { this.$("mc-nl").classList.remove("open"); this.$("mc-nl-in").blur(); this._toolUi(); }
  async sendNl(text) {
    text = String(text || "").trim();
    if (!text) return;
    const busy = this.$("mc-nl-busy");
    busy.textContent = "thinking…";
    const r = await this.api.nl(text);
    busy.textContent = "";
    if (!r.ok) {
      const msg = r.status === 503 ? "NL unavailable (no ANTHROPIC_API_KEY on mission)" : `NL failed: ${detail(r)}`;
      this.toast(msg, "crit"); alertToast(msg, "error"); return;
    }
    const d = r.data || {};
    const mission = d.mission || {};
    const steps = mission.steps || (d.plan && d.plan.steps && d.plan.steps.map((s) => s.step || { op: s.op })) || [];
    this.closeNl();
    this._showPreview({ kind: "nl", steps, plan: normalizePlan(d.plan) || { ok: true, path: [], v: [], landings: [], warnings: [], eta_s: 0 }, mission_id: d.mission_id, mock: r.mock });
    this.toast(`NL → ${steps.length} steps · pending approval`);
  }
  async approveNl() {
    const pv = this.preview;
    if (!pv || pv.kind !== "nl" || !pv.mission_id) return;
    if (pv.needDm && !this.dmHeld) { this.toast("hold DEAD-MAN to approve sprint / jump", "warn"); return; }
    const r = await this.api.control(pv.mission_id, "approve");
    if (!r.ok) { this.toast(`approve failed: ${detail(r)}`, "crit"); return; }
    this.active = { id: pv.mission_id, plan: pv.plan, steps: pv.steps };
    this.clearPreview();
    this.toast("NL mission APPROVED");
  }
  async rejectNl() {
    const pv = this.preview;
    if (pv && pv.mission_id) await this.api.control(pv.mission_id, "abort");
    this.clearPreview();
    this.toast("NL mission rejected", "warn");
  }

  // ---------------- REC (evidence) ----------------
  async _recPoll() {
    const r = await this.rec.status();
    if (this.api.mode !== "live" && this.EP.demoForced) this.rec.mockOn = this.rec.mockOn || !!this.api.mock.rec;
    const on = r.ok && recActive(r.data);
    this._recUi(on, r.ok ? (r.data && (r.data.incident_id || r.data.reason)) || "" : "offline");
    setTimeout(() => this._recPoll(), r.ok ? 2000 : 10000);
  }
  _recUi(on, sub) {
    this.recOn = on;
    const el = this.$("mc-rec");
    if (!el) return;
    el.dataset.on = on ? "1" : "0";
    this.$("mc-rec-t").textContent = on ? "● REC EVIDENCE" : sub === "offline" ? "REC --" : "REC";
    el.title = `Evidence recorder${sub ? ` · ${sub}` : ""} — click to ${on ? "stop" : "start"}`;
  }
  async toggleRec(forceOn = null) {
    const on = forceOn != null ? forceOn : !this.recOn;
    const r = on ? await this.rec.start("operator") : await this.rec.stop();
    if (this.api.mock) this.api.mock.rec = on && this.EP.demoForced;
    if (r.ok) { this._recUi(on, ""); this.toast(on ? "EVIDENCE RECORDING STARTED" : "recording stopped", on ? "warn" : "info"); }
    else this.toast(`recorder: ${detail(r)}`, "crit");
  }

  // ---------------- gamepad ----------------
  _padButton(n, down) {
    if (n === "RB") { if (down) this.dmPress("pad"); else this.dmRelease("pad"); return; }
    if (!down) return;
    const c = this._ensureCursor();
    switch (n) {
      case "A":
        this.padSource = true;
        if (this.preview && this.preview.plan && this.preview.plan.ok && this.preview.cursorKey === `${c.x.toFixed(2)},${c.y.toFixed(2)}`) { this.go(); break; }
        this.autoGo = true;
        if (this.padJump) this.planSingle(Steps.jumpTo(c.x, c.y));
        else { const p = this._pose(); this.planSingle(Steps.goto(c.x, c.y, Math.atan2(c.y - p.y, c.x - p.x), this.profile, this.precision)); }
        break;
      case "B": this.abort("gamepad B"); this.autoGo = false; break;
      case "X": this.padJump = !this.padJump; this._toolUi(); break;
      case "Y": this.commit([Steps.scan(360)]); break;
      case "LB": { const o = ["stealth", "precise", "normal", "sprint"]; this.setProfile(o[(o.indexOf(this.profile) + 1) % 4], true); this.toast(`PROFILE ${this.profile.toUpperCase()}`); break; }
      case "START": this.pauseResume(); break;
      case "BACK": this.clearPreview(); break;
    }
    this.pad.rumble(40, 0.25);
  }
  _ensureCursor() {
    if (!this.cursor) { const p = this._pose(); this.cursor = { x: p.x + Math.cos(p.yaw) * 1.5, y: p.y + Math.sin(p.yaw) * 1.5 }; }
    return this.cursor;
  }
  _padTick(dt) {
    const s = this.pad.poll();
    const chip = this.$("mc-pad");
    const info = s ? this.pad.info() : null;
    if (chip) {
      const txt = info ? `${info.short}${info.battery != null ? ` ${Math.round(info.battery * 100)}%` : ""}` : "--";
      if (this._padTxt !== txt) { this._padTxt = txt; this.$("mc-pad-t").textContent = txt; chip.dataset.on = info ? "1" : "0"; chip.title = info ? `${info.id} · ${info.mapping} — click to remap` : "no gamepad — pair a BT pad and press a button"; }
    }
    if (!s) { if (this.cursor) { this.cursor = null; this.draw.setCursor(0, 0, false); } return; }
    const { LX, LY, RX, RY } = s.axes;
    if (LX || LY || this.cursor) {
      const c = this._ensureCursor();
      // camera-relative: stick up = away from the camera on the floor
      const fwd = new THREE.Vector3(); this.camera.getWorldDirection(fwd);
      let fx = fwd.x, fy = -fwd.z; const n = Math.hypot(fx, fy) || 1; fx /= n; fy /= n;
      const rx = fy, ry = -fx;   // right-hand vector in world (x, y)
      const sp = 3.0 * dt;
      c.x += (fx * -LY + rx * LX) * sp; c.y += (fy * -LY + ry * LX) * sp;
      this.draw.setCursor(c.x, c.y, true);
      this.draw.cursor.material.color.setHex(this.padJump ? 0xffb020 : 0xff2bd6);
      if (this.preview) this.preview.cursorKey = this.preview.cursorKey || null;
    }
    if (RX || RY) this.rig.rotate(-RX * dt * 500, -RY * dt * 350);
  }
  openPadDialog() {
    this.$("mc-padd").classList.add("open");
    this._padMapUi();
  }
  _padMapUi() {
    const info = this.pad.info();
    this.$("mc-pad-info").innerHTML = info ? `<b>${escH(info.id)}</b><br><span style="color:var(--muted)">mapping: ${info.mapping}${info.battery != null ? ` · battery ${Math.round(info.battery * 100)}%` : ""}${info.haptics ? " · rumble" : ""}</span>`
      : `<span style="color:var(--warn)">no gamepad detected — pair a Bluetooth pad (Xbox / DualSense / 8BitDo / generic) and press any button</span>`;
    const root = this.$("mc-padmap");
    root.innerHTML = `<span class="h">input</span><span class="h">binding · action</span><span></span>` + [...BUTTONS, ...AXES].map((n) =>
      `<span>${n}</span><span>${this.pad.bindingText(n)}${ACTIONS_HELP[n] ? ` <i style="color:var(--muted);font-style:normal">· ${ACTIONS_HELP[n]}</i>` : n === "LX" || n === "LY" ? ' <i style="color:var(--muted);font-style:normal">· goal cursor</i>' : n === "RX" || n === "RY" ? ' <i style="color:var(--muted);font-style:normal">· camera</i>' : ""}</span><button class="btn" data-assign="${n}" style="padding:2px 6px"${info ? "" : " disabled"}>ASSIGN</button>`).join("");
    root.onclick = async (e) => {
      const n = e.target.dataset && e.target.dataset.assign;
      if (!n) return;
      e.target.textContent = "PRESS…";
      try { await this.pad.startCapture(n); } catch (_) { /* timeout / cancel */ }
      this._padMapUi();
    };
  }

  // ---------------- keyboard (called first by omni.js; true = consumed) ----------------
  onKey(e) {
    const k = e.key.toLowerCase();
    if (k === "b") { e.preventDefault(); if (!e.repeat) this.dmPress("key"); return true; }
    if (e.repeat) return false;
    if (k === "/") { e.preventDefault(); this.openNl(); return true; }
    if (k === "escape") {
      hideMenu();
      if (this.$("mc-padd").classList.contains("open")) { this.$("mc-padd").classList.remove("open"); return true; }
      if (this.draft) { this.draft = null; this.draw.setDraft(null); return true; }
      if (this.tool !== "view") { this.setTool("view"); return true; }
      if (this.preview) { this.clearPreview(); return true; }
      return false;
    }
    if (this.tool === "zone" && k === "enter") { this.closeZone(); return true; }
    if (this.tool === "zone" && k === "backspace" && this.draft) { this.draft.pop(); this.draw.setDraft(this.draft, 0xff3344, true); return true; }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const tools = { g: "goal", d: "draw", j: "jump", z: "zone", l: "label", h: "home" };
    if (tools[k]) { this.setTool(this.tool === tools[k] ? "view" : tools[k]); return true; }
    if (k === "w") { this.toggleChain(); return true; }
    return false;
  }

  // ---------------- per-frame ----------------
  // demo / offline: the mock executor drives the robot pose
  demoPose() { return this.api.mode !== "live" && this.api.mock.pose ? this.api.mock.pose : null; }

  tick(dt, t, w, h) {
    if (this.api.mode !== "live") this.api.mock.tick(dt, this.dmHeld);
    this.draw.tick(t);
    this.ghost.tick(dt, t);
    this.world.tick(t, this.camera, w, h);
    this._padTick(dt);
    if (this.preview && this.preview.needDm && this._lastDm !== this.dmHeld) this._goUi();
    this._lastDm = this.dmHeld;
    if (t - (this._uiT || 0) > 0.25) {
      this._uiT = t;
      const ack = this.state.safety && this.state.safety.deadman_alive;
      const armed = this.$("mc-armed");
      const txt = !this.dmHeld ? "" : `◆ ARMED FOR SPRINT / JUMP ◆${ack === true ? " · GUARD ACK" : this.api.E.demo ? " · DEMO" : this.dmFail > 2 ? " · NO ACK" : ""}`;
      if (armed.textContent !== txt && txt) armed.textContent = txt;
      this.$("link-mission").dataset.state = this.link === "live" ? "live" : this.api.mode === "demo" ? "demo" : this.link === "connecting" ? "connecting" : "down";
      this.$("link-mission").textContent = this.api.mode === "live" ? "MISSION" : this.api.mode === "demo" ? "MISSION·DEMO" : "MISSION·LOCAL";
    }
  }

  _ghostUi(t, T, s) {
    const now = performance.now();
    if (now - (this._gT || 0) < 80) return;
    const prev = this._gS; this._gS = { x: s.x, y: s.y, n: now };
    const v = prev && now > prev.n ? Math.hypot(s.x - prev.x, s.y - prev.y) / ((now - prev.n) / 1000) * (this.ghost.playing ? 1 : 0) : 0;
    this._gT = now;
    this.$("mc-scrub-t").textContent = `${t.toFixed(1)}/${T.toFixed(1)}s${this.ghost.playing ? ` · ${Math.min(v, 9).toFixed(2)}m/s` : ""}`;
    if (this.ghost.playing) this.$("mc-scrub").value = String(Math.round(Math.min(t / Math.max(T, 1e-3), 1) * 1000));
  }
}

function pathLen(P) { let L = 0; for (let i = 1; i < P.length; i++) L += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]); return L; }
function detail(r) {
  const d = r && r.data;
  if (d && (d.detail || d.error || d.errors)) return String(d.detail || d.error || d.errors).slice(0, 120);
  return r && r.error ? r.error : r && r.status ? `HTTP ${r.status}` : "unreachable";
}
function escH(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
export { PROFILES };
