// OmniView — HUD: status chips, safety badge, camera tiles, radar minimap,
// camera strip thumbnails. Pure DOM / canvas 2D.

import { ZONES, ZONE_CSS } from "./frames.js";

const $ = (id) => document.getElementById(id);
const SHORT = { rgb_front: "FRONT", rgb_left: "LEFT", rgb_right: "RIGHT", rgb_rear: "REAR", th_front_narrow: "TH-N", th_front_wide: "TH-W", tof_rear: "TOF" };
const DEFAULT_CAMS = ["rgb_front", "rgb_left", "rgb_right", "rgb_rear", "th_front_narrow", "th_front_wide", "tof_rear"];

export class Hud {
  constructor({ onThumbClick } = {}) {
    this.onThumbClick = onThumbClick || (() => {});
    this.radar = $("radar");
    this.rctx = this.radar ? this.radar.getContext("2d") : null;
    this.tiles = new Map();
    this.thumbs = new Map();
    this.setCams(DEFAULT_CAMS.map((id, idx) => ({ id, idx, modality: id.startsWith("th") ? "thermal" : id.startsWith("tof") ? "depth" : "rgb" })));
  }

  setCams(cams) {
    const tilesEl = $("camtiles"), strip = $("camstrip");
    if (!tilesEl || !strip) return;
    tilesEl.innerHTML = ""; strip.innerHTML = "";
    this.tiles.clear(); this.thumbs.clear();
    for (const c of cams) {
      const t = document.createElement("div");
      t.className = "tile";
      t.innerHTML = `<span class="dot"></span><span class="nm">${SHORT[c.id] || c.id}</span><span class="md">${c.modality === "thermal" ? "TH" : c.modality === "depth" ? "TOF" : "RGB"}</span><span class="fps">--</span>`;
      tilesEl.appendChild(t);
      this.tiles.set(c.id, t);
      const th = document.createElement("div");
      th.className = "thumb";
      th.title = `${c.id} — click for FPV`;
      const cv = document.createElement("canvas");
      cv.width = 128; cv.height = 72;
      th.appendChild(cv);
      const lab = document.createElement("span");
      lab.textContent = SHORT[c.id] || c.id;
      th.appendChild(lab);
      th.addEventListener("click", () => this.onThumbClick(c.idx));
      strip.appendChild(th);
      this.thumbs.set(c.idx, { el: th, cv, ctx: cv.getContext("2d"), id: c.id });
    }
  }

  drawThumb(idx, src) {
    const t = this.thumbs.get(idx);
    if (!t || !src) return;
    try { t.ctx.drawImage(src, 0, 0, t.cv.width, t.cv.height); } catch (_) {}
  }
  setActiveThumb(idx) { for (const [i, t] of this.thumbs) t.el.classList.toggle("active", i === idx); }

  // stats: {cam_id: {fps, ok}}
  setCamStats(stats) {
    for (const [id, el] of this.tiles) {
      const s = stats[id];
      const dot = el.querySelector(".dot"), fps = el.querySelector(".fps");
      if (!s) { dot.className = "dot"; fps.textContent = "--"; continue; }
      const f = +s.fps || 0;
      dot.className = "dot " + (s.ok === false || f < 1 ? "bad" : f < 5 ? "warn" : "ok");
      fps.textContent = f ? f.toFixed(0) : "0";
    }
  }

  setText(id, txt, cls) {
    const el = $(id);
    if (!el) return;
    if (el.textContent !== txt) el.textContent = txt;
    if (cls !== undefined) el.dataset.state = cls;
  }

  setLink(name, state) { this.setText(`link-${name}`, name.toUpperCase(), state); }

  setSafety(s, src) {
    const el = $("safety");
    if (!el) return;
    const level = (s && s.level) || "UNKNOWN";
    el.dataset.level = level;
    $("safety-level").textContent = level;
    $("safety-reason").textContent = s ? `${s.reason || ""}${src ? " · " + src : ""}` : "safety_guard offline";
    $("safety-vmax").textContent = s && s.vmax != null ? `vmax ${(+s.vmax).toFixed(2)} m/s` : "vmax --";
    $("safety-near").textContent = s && s.nearest_person_m != null ? `${(+s.nearest_person_m).toFixed(1)} m` : "-- m";
    document.body.dataset.safety = level;
  }

  setBattery(pct) {
    const fill = $("batt-fill");
    this.setText("batt", pct == null ? "--%" : `${Math.round(pct)}%`);
    if (fill) {
      fill.style.width = `${pct == null ? 0 : Math.max(0, Math.min(100, pct))}%`;
      fill.dataset.state = pct == null ? "" : pct < 15 ? "bad" : pct < 30 ? "warn" : "ok";
    }
  }

  drawRadar(persons, selected, t, maxR = 10) {
    const ctx = this.rctx;
    if (!ctx) return;
    const W = this.radar.width, H = this.radar.height, cx = W / 2, cy = H / 2, s = (W / 2 - 8) / maxR;
    const night = document.body.classList.contains("night");
    const pri = night ? "255,190,60" : "0,220,255";
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(4,10,16,0.75)";
    ctx.beginPath(); ctx.arc(cx, cy, W / 2 - 2, 0, Math.PI * 2); ctx.fill();
    // range rings
    const rings = [[ZONES.STOP, "255,51,68"], [ZONES.SLOW, "255,138,30"], [ZONES.CAUTION, "255,210,58"], [5, pri], [10, pri]];
    ctx.lineWidth = 1;
    for (const [r, c] of rings) {
      if (r > maxR) continue;
      ctx.strokeStyle = `rgba(${c},${r <= ZONES.CAUTION ? 0.55 : 0.25})`;
      ctx.beginPath(); ctx.arc(cx, cy, r * s, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.strokeStyle = `rgba(${pri},0.15)`;
    ctx.beginPath(); ctx.moveTo(cx, 6); ctx.lineTo(cx, H - 6); ctx.moveTo(6, cy); ctx.lineTo(W - 6, cy); ctx.stroke();
    // sweep
    const a = (t * 1.6) % (Math.PI * 2);
    const g = ctx.createConicGradient ? ctx.createConicGradient(a - 0.6, cx, cy) : null;
    if (g) {
      g.addColorStop(0, `rgba(${pri},0)`); g.addColorStop(0.095, `rgba(${pri},0.28)`); g.addColorStop(0.1, `rgba(${pri},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, W / 2 - 2, 0, Math.PI * 2); ctx.fill();
    }
    // FOV wedges hint: front thermal
    ctx.fillStyle = `rgba(255,120,40,0.08)`;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, 9 * s, -Math.PI / 2 - 0.6, -Math.PI / 2 + 0.6); ctx.fill();
    // robot
    ctx.fillStyle = `rgb(${pri})`;
    ctx.beginPath(); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx + 5, cy + 6); ctx.lineTo(cx - 5, cy + 6); ctx.closePath(); ctx.fill();
    // persons (base frame: x fwd -> screen up, y left -> screen left)
    for (const p of persons) {
      let px = cx - p.by * s, py = cy - p.bx * s;
      const out = Math.hypot(p.bx, p.by) > maxR;
      if (out) { const k = maxR / Math.hypot(p.bx, p.by); px = cx - p.by * s * k; py = cy - p.bx * s * k; }
      ctx.fillStyle = ZONE_CSS[p.zone];
      ctx.shadowColor = ZONE_CSS[p.zone]; ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.arc(px, py, out ? 3 : 5, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      if (selected === p.gid) {
        ctx.strokeStyle = "#ff2bd6"; ctx.lineWidth = 2; ctx.strokeRect(px - 8, py - 8, 16, 16); ctx.lineWidth = 1;
      }
      ctx.fillStyle = "rgba(220,240,255,0.85)"; ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(String(p.gid), px + 7, py - 5);
    }
  }
}
