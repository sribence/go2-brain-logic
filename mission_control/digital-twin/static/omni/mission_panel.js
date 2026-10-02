// OmniView mission — DOM: styles, tool column, plan card, status widget,
// waypoint-chain editor, world list, dead-man button, NL bar, context menu,
// prompt dialog, alerts, gamepad remap dialog. No three.js here.

import { esc } from "./mission_zones.js";
import { SPEED_LEGEND } from "./mission_draw.js";
import { OP_FIELDS, EDITOR_OPS, SPEEDS, PRECISION, ACTIONS, WAIT_EVENTS } from "./mission_api.js";

const CSS = `
#mc-left { position: absolute; top: 58px; left: 10px; z-index: 17; width: 238px; display: flex; flex-direction: column; gap: 6px;
  max-height: calc(100vh - 58px - 262px); overflow-y: auto; overflow-x: hidden; scrollbar-width: thin; }
#mc-left .panel { padding: 7px 8px; }
#mc-left .lbl { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
.mc-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; margin-top: 5px; }
.mc-grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; margin-top: 4px; }
.mc-grid .btn, .mc-grid4 .btn { padding: 5px 2px; font-size: 9.5px; position: relative; }
.mc-grid .btn kbd { position: absolute; right: 2px; top: 1px; font-size: 7.5px; color: var(--muted); }
.btn[disabled] { opacity: .38; cursor: not-allowed; }
.btn.go { border-color: var(--ok); color: var(--ok); font-weight: 800; letter-spacing: .2em; }
.btn.go.on, .btn.go:not([disabled]):hover { background: rgba(53,242,138,.2); }
.btn.crit { border-color: var(--crit); color: var(--crit); font-weight: 700; }
.btn.warn { border-color: var(--warn); color: var(--warn); }
.btn[data-prof="sprint"].on { background: rgba(255,51,68,.25); border-color: var(--crit); }
.btn[data-prof="stealth"].on { background: rgba(120,90,255,.25); border-color: #8a6bff; }
.mc-hint { margin-top: 5px; font-size: 10px; color: var(--muted); line-height: 1.4; min-height: 14px; }
.mc-hint b { color: var(--pri); font-weight: 600; }
.mc-kv { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; margin-top: 5px; text-align: center; }
.mc-kv div { background: rgba(var(--pri-rgb), .05); border: 1px solid rgba(var(--pri-rgb), .12); padding: 3px 0; }
.mc-kv span { display: block; font-size: 8.5px; color: var(--muted); letter-spacing: .12em; }
.mc-kv b { font-size: 13px; color: #fff; font-variant-numeric: tabular-nums; }
.mc-legend { height: 5px; margin: 6px 0 2px; border-radius: 2px; }
.mc-legend-l { display: flex; justify-content: space-between; font-size: 8.5px; color: var(--muted); }
#mc-warn { margin: 5px 0 0; padding: 0; list-style: none; font-size: 10px; line-height: 1.35; }
#mc-warn li { color: var(--warn); padding-left: 10px; position: relative; } #mc-warn li::before { content: "▲"; position: absolute; left: 0; font-size: 7px; top: 2px; }
#mc-warn li.bad { color: var(--crit); }
.mc-row { display: flex; align-items: center; gap: 4px; margin-top: 5px; }
.mc-row input[type=range] { flex: 1; accent-color: var(--pri); min-width: 0; }
.mc-row .t { font-size: 9.5px; color: var(--muted); min-width: 82px; text-align: right; font-variant-numeric: tabular-nums; }
.mc-acts { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 4px; margin-top: 6px; }
.mc-steps { margin: 4px 0 0; font-size: 10.5px; color: var(--text); max-height: 90px; overflow: auto; }
.mc-steps div { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; } .mc-steps i { color: var(--muted); font-style: normal; }
#mc-status[data-state="running"] { border-color: var(--ok); } #mc-status[data-state="paused"], #mc-status[data-state="waiting"] { border-color: var(--warn); }
#mc-status[data-state="failed"], #mc-status[data-state="aborted"] { border-color: var(--crit); }
#mc-st-state { font-weight: 800; letter-spacing: .18em; font-size: 13px; }
#mc-status[data-state="running"] #mc-st-state { color: var(--ok); } #mc-status[data-state="paused"] #mc-st-state, #mc-status[data-state="waiting"] #mc-st-state { color: var(--warn); }
#mc-status[data-state="failed"] #mc-st-state, #mc-status[data-state="aborted"] #mc-st-state { color: var(--crit); }
#mc-status[data-state="waiting"] { animation: mcpulse 1s infinite alternate; }
@keyframes mcpulse { to { box-shadow: 0 0 18px rgba(255,210,58,.45); } }
.mc-prog { height: 5px; background: #13202a; margin-top: 5px; border-radius: 2px; overflow: hidden; }
.mc-prog div { height: 100%; width: 0; background: linear-gradient(90deg, var(--pri), var(--ok)); transition: width .3s; }
.mc-sub { font-size: 10px; color: var(--muted); margin-top: 3px; display: flex; justify-content: space-between; gap: 6px; font-variant-numeric: tabular-nums; }
.mc-sub b { color: var(--text); font-weight: 600; }
#mc-chain .row { display: grid; grid-template-columns: 16px 1fr auto; gap: 4px; align-items: start; padding: 4px 0; border-bottom: 1px dashed rgba(var(--pri-rgb), .15); font-size: 10px; }
#mc-chain .row.sel { background: rgba(var(--pri-rgb), .08); }
#mc-chain .n { color: var(--pri); font-weight: 700; padding-top: 3px; }
#mc-chain .f { display: flex; flex-wrap: wrap; gap: 3px; }
#mc-chain select, #mc-chain input, .mc-in { font: inherit; font-size: 10px; background: rgba(0,0,0,.45); color: var(--text); border: 1px solid var(--line); border-radius: 2px; padding: 2px 3px; }
#mc-chain input[type=number] { width: 52px; } #mc-chain input[type=text] { width: 140px; }
#mc-chain label { color: var(--muted); display: inline-flex; gap: 2px; align-items: center; }
#mc-chain .ops { display: flex; flex-direction: column; gap: 1px; }
#mc-chain .ops button { font: inherit; font-size: 9px; padding: 0 4px; background: none; border: 1px solid #23313c; color: var(--muted); cursor: pointer; }
#mc-chain .ops button:hover { color: var(--pri); border-color: var(--pri); }
#mc-chain .empty { font-size: 10px; color: var(--muted); padding: 4px 0; }
.mc-world { font-size: 10.5px; margin-top: 4px; }
.mc-world div { display: flex; justify-content: space-between; align-items: center; gap: 4px; padding: 1px 0; }
.mc-world .k { font-size: 8.5px; padding: 0 3px; border: 1px solid; border-radius: 2px; margin-right: 4px; }
.mc-world .k.no_go { color: var(--crit); } .mc-world .k.watch { color: #ffb020; } .mc-world .k.patrol { color: var(--pri); } .mc-world .k.label, .mc-world .k.home { color: var(--ok); }
.mc-world button { font: inherit; font-size: 9px; background: none; border: none; color: var(--muted); cursor: pointer; } .mc-world button:hover { color: var(--crit); }
.collapsed > :not(.lbl) { display: none !important; }
.lbl .tg { cursor: pointer; color: var(--muted); }
/* dead-man */
#mc-deadman { position: absolute; right: 14px; bottom: 178px; z-index: 30; width: 150px; height: 66px; border-radius: 10px; cursor: pointer; user-select: none; touch-action: none;
  font: 800 15px/1.05 var(--mono); letter-spacing: .14em; color: #ffd23a; background: repeating-linear-gradient(-45deg, rgba(255,210,58,.12) 0 8px, rgba(0,0,0,.4) 8px 16px), var(--panel);
  border: 2px solid #ffd23a; box-shadow: 0 0 0 3px #111, 0 0 14px rgba(255,210,58,.25); }
#mc-deadman small { display: block; font-size: 8.5px; font-weight: 600; letter-spacing: .1em; opacity: .8; margin-top: 4px; }
body.mc-armed #mc-deadman { color: #021; background: radial-gradient(circle at 50% 40%, #6dffb0, #12c460 70%); border-color: #b6ffd6; box-shadow: 0 0 0 3px #111, 0 0 34px rgba(53,242,138,.95); animation: mcarm .35s infinite alternate; }
@keyframes mcarm { to { box-shadow: 0 0 0 3px #111, 0 0 54px rgba(53,242,138,1); } }
#mc-armed { position: absolute; left: 50%; top: 128px; transform: translateX(-50%); z-index: 19; display: none; padding: 4px 14px; font-weight: 800; letter-spacing: .25em; font-size: 12px;
  color: #35f28a; border: 1px solid #35f28a; background: rgba(0,30,12,.75); text-shadow: 0 0 10px rgba(53,242,138,.8); }
body.mc-armed #mc-armed { display: block; }
body.mc-armed #fx { box-shadow: inset 0 0 60px rgba(53,242,138,.35); }
/* NL bar */
#mc-nl { position: absolute; left: 50%; bottom: 104px; transform: translateX(-50%); z-index: 32; width: min(620px, calc(100vw - 32px)); display: none; align-items: center; gap: 6px; padding: 6px 8px; border-color: var(--pri); }
#mc-nl.open { display: flex; }
#mc-nl span { color: var(--pri); font-weight: 800; }
#mc-nl input { flex: 1; font: inherit; font-size: 13px; background: transparent; border: none; outline: none; color: #fff; min-width: 0; }
#mc-nl .busy { color: var(--warn); font-size: 10px; }
/* context menu */
#mc-ctx { position: absolute; z-index: 45; display: none; min-width: 190px; padding: 4px; }
#mc-ctx.open { display: block; }
#mc-ctx .h { font-size: 9.5px; color: var(--muted); letter-spacing: .12em; padding: 3px 6px 4px; border-bottom: 1px solid var(--line); margin-bottom: 3px; }
#mc-ctx button { display: block; width: 100%; text-align: left; font: inherit; font-size: 11px; color: var(--text); background: none; border: none; padding: 5px 8px; cursor: pointer; }
#mc-ctx button:hover { background: rgba(var(--pri-rgb), .16); color: #fff; }
#mc-ctx button i { font-style: normal; color: var(--muted); float: right; font-size: 9.5px; margin-left: 10px; }
/* dialogs */
.mc-modal { position: absolute; inset: 0; z-index: 60; display: none; align-items: center; justify-content: center; background: rgba(0,0,0,.45); }
.mc-modal.open { display: flex; }
.mc-modal .panel { padding: 12px 14px; min-width: 300px; max-width: calc(100vw - 32px); }
.mc-modal input.mc-in { width: 100%; font-size: 13px; padding: 5px 6px; margin: 8px 0; }
.mc-modal .btns { grid-template-columns: 1fr 1fr; }
.mc-padmap { display: grid; grid-template-columns: 70px 1fr auto; gap: 3px 8px; font-size: 10.5px; align-items: center; margin-top: 6px; max-height: 55vh; overflow: auto; }
.mc-padmap .h { color: var(--muted); }
/* alerts */
#mc-alerts { position: absolute; top: 58px; right: 216px; z-index: 41; display: flex; flex-direction: column; gap: 4px; width: 260px; pointer-events: none; }
.mc-alert { padding: 6px 10px; font-size: 11px; border: 1px solid var(--pri); border-left-width: 4px; background: var(--panel); backdrop-filter: blur(6px); animation: mcin .25s ease-out; }
.mc-alert[data-level="warn"] { border-color: var(--warn); color: var(--warn); } .mc-alert[data-level="crit"], .mc-alert[data-level="error"] { border-color: var(--crit); color: #ffb3ba; }
.mc-alert small { display: block; color: var(--muted); font-size: 9px; }
@keyframes mcin { from { transform: translateX(30px); opacity: 0; } }
/* topbar chips */
#mc-rec { cursor: pointer; } #mc-rec .d { width: 8px; height: 8px; border-radius: 50%; background: #34434f; }
#mc-rec[data-on="1"] { color: var(--crit); border-color: var(--crit); } #mc-rec[data-on="1"] .d { background: var(--crit); box-shadow: 0 0 8px var(--crit); animation: blink 1s steps(2) infinite; }
#mc-pad { cursor: pointer; } #mc-pad[data-on="1"] { color: var(--ok); border-color: rgba(53,242,138,.5); }
/* floating tags */
.mtag { position: absolute; left: 0; top: 0; pointer-events: none; white-space: nowrap; font-size: 10.5px; padding: 2px 6px; border: 1px solid; border-left-width: 3px; background: rgba(2,8,14,.72); color: #e6f6ff; }
.mtag b { font-size: 9px; letter-spacing: .12em; margin-right: 4px; } .mtag i { color: var(--muted); font-style: normal; font-size: 9px; }
.mtag[data-kind="no_go"] { border-color: var(--crit); } .mtag[data-kind="no_go"] b { color: var(--crit); }
.mtag[data-kind="watch"] { border-color: #ffb020; } .mtag[data-kind="watch"] b { color: #ffb020; }
.mtag[data-kind="patrol"] { border-color: var(--pri); } .mtag[data-kind="patrol"] b { color: var(--pri); }
.mtag[data-kind="label"], .mtag[data-kind="home"] { border-color: var(--ok); color: var(--ok); }
body.mc-tool-active #viewport { cursor: crosshair; }
@media (max-width: 900px) {
  #mc-left { top: 92px; width: 182px; max-height: calc(100vh - 92px - 180px); }
  #mc-deadman { width: 100px; height: 54px; bottom: 120px; right: 10px; font-size: 12px; }
  #mc-alerts { right: 168px; width: 200px; top: 92px; }
}`;

const TOOLS = [
  ["view", "VIEW", "Esc"], ["goal", "GOAL", "G"], ["draw", "DRAW", "D"], ["jump", "JUMP", "J"], ["zone", "ZONE", "Z"],
  ["label", "LABEL", "L"], ["home", "HOME", "H"], ["chain", "CHAIN", "W"], ["nl", "ASK", "/"], ["pad", "PAD", ""],
];

export function buildPanel() {
  const st = document.createElement("style");
  st.id = "mc-style"; st.textContent = CSS;
  document.head.appendChild(st);
  const legend = `linear-gradient(90deg, ${SPEED_LEGEND.map(([v, c]) => `${c} ${(v / 1.5) * 100}%`).join(", ")})`;
  const left = document.createElement("div");
  left.id = "mc-left";
  left.innerHTML = `
  <div class="panel" id="mc-tools">
    <div class="lbl"><span>mission · 3d command</span><span class="link" id="link-mission">MISSION</span></div>
    <div class="mc-grid">${TOOLS.map(([k, n, key]) => `<button class="btn" data-tool="${k}" title="${n}${key ? ` (${key})` : ""}">${n}${key ? `<kbd>${key}</kbd>` : ""}</button>`).join("")}</div>
    <div class="mc-grid4" id="mc-prof">${["stealth", "precise", "normal", "sprint"].map((p) => `<button class="btn" data-prof="${p}" title="${{ stealth: "Alt+click", precise: "Ctrl+click", normal: "click", sprint: "Shift+click · dead-man" }[p]}">${p}</button>`).join("")}</div>
    <div class="mc-grid4" id="mc-zonekind" style="display:none;grid-template-columns:repeat(3,1fr)">
      <button class="btn crit" data-zk="no_go">NO-GO</button><button class="btn warn" data-zk="watch">WATCH</button><button class="btn" data-zk="patrol">PATROL</button></div>
    <div class="mc-hint" id="mc-hint"></div>
  </div>
  <div class="panel" id="mc-plan" style="display:none">
    <div class="lbl"><span id="mc-plan-title">plan preview</span><span id="mc-plan-src"></span></div>
    <div class="mc-steps" id="mc-plan-steps"></div>
    <div class="mc-kv"><div><span>ETA</span><b id="mc-eta">--</b></div><div><span>LEN</span><b id="mc-len">--</b></div><div><span>VMAX</span><b id="mc-vmax">--</b></div><div><span>JUMPS</span><b id="mc-jumps">0</b></div></div>
    <div class="mc-row" id="mc-prec-row"><span class="lbl" style="flex:none">zone</span>${PRECISION.map((z) => `<button class="btn" data-prec="${z}" style="padding:3px 4px;flex:1">${z}</button>`).join("")}</div>
    <div class="mc-legend" style="background:${legend}"></div>
    <div class="mc-legend-l"><span>0</span><span>0.3</span><span>0.6</span><span>1.0</span><span>1.5 m/s</span></div>
    <ul id="mc-warn"></ul>
    <div class="mc-row"><button class="btn" id="mc-play" style="padding:3px 6px">❚❚</button><input type="range" id="mc-scrub" min="0" max="1000" value="0"><span class="t" id="mc-scrub-t">--</span></div>
    <div class="mc-acts" id="mc-acts-plan"><button class="btn go" id="mc-go">GO</button><button class="btn" id="mc-addchain" title="append to waypoint chain">+CHAIN</button><button class="btn" id="mc-clear">CLEAR</button></div>
    <div class="mc-acts" id="mc-acts-nl" style="display:none;grid-template-columns:1fr 1fr"><button class="btn go" id="mc-approve">APPROVE</button><button class="btn crit" id="mc-reject">REJECT</button></div>
  </div>
  <div class="panel" id="mc-status" data-state="idle">
    <div class="lbl"><span>mission status</span><span id="mc-st-id"></span></div>
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:3px"><span id="mc-st-state">IDLE</span><span id="mc-st-step" style="font-size:10.5px"></span></div>
    <div class="mc-prog"><div id="mc-st-bar"></div></div>
    <div class="mc-sub"><span>ETA <b id="mc-st-eta">--</b></span><span id="mc-st-err"></span></div>
    <div class="mc-acts" style="grid-template-columns:1fr 1fr 1fr"><button class="btn warn" id="mc-pause">PAUSE</button><button class="btn go" id="mc-continue" title="operator: continue (POST /continue)">CONT.</button><button class="btn crit" id="mc-abort">ABORT</button></div>
  </div>
  <div class="panel" id="mc-chain" style="display:none">
    <div class="lbl"><span>waypoint chain</span><span id="mc-chain-n">0 steps</span></div>
    <div id="mc-chain-rows"></div>
    <div class="mc-row"><select id="mc-chain-op" class="mc-in" style="flex:1">${EDITOR_OPS.map((o) => `<option>${o}</option>`).join("")}</select><button class="btn" id="mc-chain-add" style="padding:3px 8px">+ ADD</button></div>
    <div class="mc-row"><select id="mc-onfail" class="mc-in" title="on_fail"><option>stop</option><option>skip</option><option>return_home</option></select>
      <button class="btn" id="mc-chain-prev" style="flex:1">PREVIEW</button><button class="btn" id="mc-chain-clear">CLEAR</button></div>
    <div class="mc-hint">scene clicks append steps while the chain is open</div>
  </div>
  <div class="panel collapsed" id="mc-world-p">
    <div class="lbl"><span>world · zones · labels</span><span class="tg" id="mc-world-tg">▸</span></div>
    <div class="mc-world" id="mc-world"></div>
  </div>`;
  document.body.appendChild(left);

  const extra = document.createElement("div");
  extra.innerHTML = `
  <button id="mc-deadman" title="DEAD-MAN — hold (B key / gamepad RB) to arm SPRINT / JUMP">DEAD-MAN<small>HOLD · B · RB</small></button>
  <div id="mc-armed">◆ ARMED FOR SPRINT / JUMP ◆</div>
  <div id="mc-nl" class="panel"><span>›</span><input id="mc-nl-in" placeholder="menj a kapuhoz és nézz körül…  (Enter = preview, Esc = close)" autocomplete="off"><span class="busy" id="mc-nl-busy"></span><button class="btn" id="mc-nl-send" style="padding:4px 10px">SEND</button></div>
  <div id="mc-ctx" class="panel"></div>
  <div id="mc-alerts"></div>
  <div class="mc-modal" id="mc-ask"><div class="panel"><div class="lbl" id="mc-ask-t"></div><input class="mc-in" id="mc-ask-in" autocomplete="off"><div class="btns"><button class="btn go" id="mc-ask-ok">OK</button><button class="btn" id="mc-ask-cancel">CANCEL</button></div></div></div>
  <div class="mc-modal" id="mc-padd"><div class="panel"><div class="lbl"><span>gamepad · remap (press to assign)</span></div><div id="mc-pad-info" style="font-size:11px;margin-top:4px"></div><div class="mc-padmap" id="mc-padmap"></div><div class="btns" style="margin-top:8px"><button class="btn" id="mc-pad-reset">RESET</button><button class="btn go" id="mc-pad-close">CLOSE</button></div></div></div>`;
  while (extra.firstChild) document.body.appendChild(extra.firstChild);

  // topbar chips: REC + PAD (before the spacer)
  const spacer = document.getElementById("spacer");
  if (spacer) {
    const rec = document.createElement("span");
    rec.className = "chip"; rec.id = "mc-rec"; rec.title = "Evidence recorder — click to start/stop (omni /record)";
    rec.innerHTML = `<span class="d"></span><span id="mc-rec-t">REC</span>`;
    const pad = document.createElement("span");
    pad.className = "chip"; pad.id = "mc-pad"; pad.title = "Gamepad — click to remap";
    pad.innerHTML = `<span class="k">PAD</span><span id="mc-pad-t">--</span>`;
    spacer.parentNode.insertBefore(rec, spacer);
    spacer.parentNode.insertBefore(pad, spacer);
  }
  return (id) => document.getElementById(id);
}

// minimal prompt dialog (no window.prompt: keeps focus + works in kiosk/headless)
export function ask(title, def = "") {
  const $ = (id) => document.getElementById(id);
  const m = $("mc-ask"), inp = $("mc-ask-in");
  $("mc-ask-t").textContent = title;
  inp.value = def;
  m.classList.add("open");
  setTimeout(() => { inp.focus(); inp.select(); }, 0);
  return new Promise((resolve) => {
    const done = (v) => { m.classList.remove("open"); inp.onkeydown = null; $("mc-ask-ok").onclick = null; $("mc-ask-cancel").onclick = null; resolve(v); };
    inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") done(inp.value.trim() || null); else if (e.key === "Escape") done(null); };
    $("mc-ask-ok").onclick = () => done(inp.value.trim() || null);
    $("mc-ask-cancel").onclick = () => done(null);
  });
}

// context menu: items [{label, hint, fn}] at client (x, y)
export function showMenu(x, y, header, items) {
  const el = document.getElementById("mc-ctx");
  el.innerHTML = `<div class="h">${esc(header)}</div>`;
  for (const it of items) {
    const b = document.createElement("button");
    b.innerHTML = `${esc(it.label)}${it.hint ? `<i>${esc(it.hint)}</i>` : ""}`;
    b.addEventListener("click", (e) => { e.stopPropagation(); hideMenu(); it.fn(); });
    el.appendChild(b);
  }
  el.classList.add("open");
  const r = el.getBoundingClientRect();
  el.style.left = `${Math.min(x, innerWidth - r.width - 8)}px`;
  el.style.top = `${Math.min(y, innerHeight - r.height - 8)}px`;
}
export function hideMenu() { const el = document.getElementById("mc-ctx"); if (el) el.classList.remove("open"); }

export function alertToast(text, level = "info", sub = "") {
  const root = document.getElementById("mc-alerts");
  if (!root) return;
  const a = document.createElement("div");
  a.className = "mc-alert"; a.dataset.level = level;
  a.innerHTML = `${esc(text)}${sub ? `<small>${esc(sub)}</small>` : ""}`;
  root.prepend(a);
  while (root.children.length > 5) root.lastChild.remove();
  setTimeout(() => { a.style.transition = "opacity .4s"; a.style.opacity = 0; setTimeout(() => a.remove(), 450); }, level === "crit" || level === "error" ? 10000 : 6000);
}

// one-line summary of a step
export function stepText(s) {
  const f = (v) => (v == null ? "" : (+v).toFixed(1));
  switch (s.op) {
    case "goto": return `goto ${f(s.x)},${f(s.y)}${s.yaw != null ? ` ∠${Math.round(s.yaw * 57.3)}°` : ""} <i>${s.speed || "normal"}${s.zone ? " · " + s.zone : ""}</i>`;
    case "jump_to": return `jump_to ${f(s.x)},${f(s.y)} <i>≤${s.max_jumps || 6}</i>`;
    case "follow_path": return `follow_path ${s.points.length} pts <i>${s.speed || "normal"}</i>`;
    case "look_at": return `look_at ${f(s.x)},${f(s.y)}`;
    case "scan": return `scan ${s.deg || 360}°`;
    case "wait": return `wait ${s.s}s`;
    case "say": return `say “${esc(s.text)}”`;
    case "request_passage": return `request_passage “${esc(s.text)}”`;
    case "wait_for": return `wait_for ${s.event}`;
    case "record": return `record ${s.on ? "ON" : "OFF"}`;
    case "capture": return `capture ${s.x != null ? `${f(s.x)},${f(s.y)} <i>fine</i>` : "here"}`;
    case "shadow": case "watch": case "escort": return `${s.op} #${s.gid}`;
    default: return `${s.op} ${esc(s.label || s.name || "")}`;
  }
}

// chain editor rows. cb: {onChange(i, key, value), onMove(i, d), onDel(i), onSelect(i)}
export function renderChain(steps, cb, selected = -1) {
  const root = document.getElementById("mc-chain-rows");
  document.getElementById("mc-chain-n").textContent = `${steps.length} step${steps.length === 1 ? "" : "s"}`;
  root.innerHTML = steps.length ? "" : `<div class="empty">empty — click GOAL/JUMP/DRAW in the scene or + ADD</div>`;
  steps.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = `row${i === selected ? " sel" : ""}`;
    const fields = OP_FIELDS[s.op] || {};
    const parts = [`<select data-k="op">${EDITOR_OPS.concat(EDITOR_OPS.includes(s.op) ? [] : [s.op]).map((o) => `<option${o === s.op ? " selected" : ""}>${o}</option>`).join("")}</select>`];
    for (const [k, kind] of Object.entries(fields)) {
      const v = s[k];
      const opt = (list, cur, auto) => `<select data-k="${k}">${auto ? `<option value=""${cur == null ? " selected" : ""}>${auto}</option>` : ""}${list.map((o) => `<option${o === cur ? " selected" : ""}>${o}</option>`).join("")}</select>`;
      if (kind === "num" || kind === "num?" || kind === "int") parts.push(`<label>${k}<input type="number" step="${kind === "int" ? 1 : 0.1}" data-k="${k}" data-kind="${kind}" value="${v == null ? "" : kind === "int" ? v : (+v).toFixed(k === "yaw" ? 2 : 2)}"${kind === "num?" ? ' placeholder="–"' : ""}></label>`);
      else if (kind === "text") parts.push(`<input type="text" data-k="${k}" value="${esc(v || "")}">`);
      else if (kind === "speed") parts.push(opt(SPEEDS, v || "normal"));
      else if (kind === "zone") parts.push(opt(PRECISION, v, "auto"));
      else if (kind === "action") parts.push(opt(ACTIONS, v || "hello"));
      else if (kind === "event") parts.push(opt(WAIT_EVENTS, v || "operator"));
      else if (kind === "capmode") parts.push(opt(["max", "normal"], v || "max"));
      else if (kind === "side") parts.push(opt(["right", "left"], v || "right"));
      else if (kind === "bool") parts.push(`<label><input type="checkbox" data-k="${k}"${v !== false ? " checked" : ""}>${k}</label>`);
      else if (kind === "points") parts.push(`<span style="color:var(--muted)">${(v || []).length} pts</span>`);
    }
    row.innerHTML = `<span class="n">${i + 1}</span><div class="f">${parts.join("")}</div><div class="ops"><button data-a="up" title="move up">▲</button><button data-a="down" title="move down">▼</button><button data-a="del" title="delete">✕</button></div>`;
    row.addEventListener("change", (e) => {
      const t = e.target, k = t.dataset.k;
      if (!k) return;
      let val = t.type === "checkbox" ? t.checked : t.value;
      if (t.type === "number") val = t.value === "" ? null : t.dataset.kind === "int" ? parseInt(t.value, 10) : +t.value;
      if (t.tagName === "SELECT" && val === "") val = null;
      cb.onChange(i, k, val);
    });
    row.addEventListener("click", (e) => {
      const a = e.target.dataset && e.target.dataset.a;
      if (a === "up") cb.onMove(i, -1); else if (a === "down") cb.onMove(i, 1); else if (a === "del") cb.onDel(i);
      else if (e.target === row || e.target.classList.contains("n")) cb.onSelect(i);
    });
    root.appendChild(row);
  });
}

export function renderWorld(world, cb) {
  const root = document.getElementById("mc-world");
  const rows = [];
  for (const z of world.zones || []) rows.push(`<div><span><span class="k ${z.kind}">${({ no_go: "NO-GO", watch: "WATCH", patrol: "PATROL" })[z.kind] || z.kind}</span>${esc(z.name || z.id)}</span><button data-del-zone="${esc(z.id || z.name)}" title="delete">✕</button></div>`);
  for (const l of world.labels || []) rows.push(`<div><span><span class="k label">LABEL</span>${esc(l.name)} <i style="color:var(--muted);font-style:normal">${(+l.x).toFixed(1)},${(+l.y).toFixed(1)}</i></span><button data-del-label="${esc(l.name)}" title="delete">✕</button></div>`);
  if (world.home) rows.push(`<div><span><span class="k home">HOME</span>${(+world.home.x).toFixed(1)},${(+world.home.y).toFixed(1)}</span><button data-go-home="1" title="return_home">⌂ GO</button></div>`);
  root.innerHTML = rows.join("") || `<div style="color:var(--muted)">no zones / labels — Z, L, H</div>`;
  root.onclick = (e) => {
    const d = e.target.dataset || {};
    if (d.delZone) cb.delZone(d.delZone); else if (d.delLabel) cb.delLabel(d.delLabel); else if (d.goHome) cb.goHome();
  };
}
