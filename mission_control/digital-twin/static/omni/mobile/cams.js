// NERO OmniVision mobile — camera grid (all rig cameras from /ws/video),
// tap = fullscreen (live: a dedicated hi-res /ws/video?cams=<id> stream),
// swipe left/right in fullscreen = next/previous camera.

import { ReconnectingWS } from "../net.js";

const MOD_LABEL = { rgb: "RGB", thermal: "HŐ", depth: "ToF" };

export class CamGrid {
  constructor(gridEl, fullEl, { omniWs } = {}) {
    this.grid = gridEl; this.full = fullEl; this.omniWs = omniWs;
    this.cams = []; this.tiles = new Map(); this.latest = new Map();
    this.visible = false; this.fullIdx = null; this.hiWs = null; this.hiSrc = null;
    this.fullCv = fullEl.querySelector("canvas");
    this.fullLabel = fullEl.querySelector(".cam-full-label");
    fullEl.addEventListener("click", (e) => { if (!e.target.closest("button")) this.closeFull(); });
    const nav = (d) => (e) => { e.stopPropagation(); this.step(d); };
    fullEl.querySelector(".cam-prev").addEventListener("click", nav(-1));
    fullEl.querySelector(".cam-next").addEventListener("click", nav(1));
    let sx = null;
    fullEl.addEventListener("touchstart", (e) => { sx = e.touches[0].clientX; }, { passive: true });
    fullEl.addEventListener("touchend", (e) => {
      if (sx == null) return;
      const dx = e.changedTouches[0].clientX - sx; sx = null;
      if (Math.abs(dx) > 60) this.step(dx < 0 ? 1 : -1);
    }, { passive: true });
  }

  setCams(cams) {
    this.cams = cams;
    this.grid.innerHTML = "";
    this.tiles.clear();
    for (const c of cams) {
      const tile = document.createElement("button");
      tile.className = `cam-tile mod-${c.modality}`;
      tile.innerHTML = `<canvas width="320" height="180"></canvas><span class="cam-name">${c.id}</span><span class="cam-mod">${MOD_LABEL[c.modality] || c.modality}</span><span class="cam-fps">--</span>`;
      tile.addEventListener("click", () => this.openFull(c.idx));
      this.grid.appendChild(tile);
      this.tiles.set(c.idx, { tile, cv: tile.querySelector("canvas"), fpsEl: tile.querySelector(".cam-fps"), n: 0, dirty: false });
    }
  }

  onFrame(idx, src) {
    this.latest.set(idx, src);
    const t = this.tiles.get(idx);
    if (t) { t.n++; t.dirty = true; }
  }

  // called from the render loop (cheap when hidden)
  draw() {
    if (this.fullIdx != null) {
      const src = this.hiSrc || this.latest.get(this.fullIdx);
      if (src && src !== this._lastFull) { this._lastFull = src; drawFit(this.fullCv, src, true); }
      return;
    }
    if (!this.visible) return;
    for (const [idx, t] of this.tiles) {
      if (!t.dirty) continue;
      t.dirty = false;
      const src = this.latest.get(idx);
      if (src) drawFit(t.cv, src, false);
    }
  }

  tickFps(span) {
    for (const t of this.tiles.values()) { t.fpsEl.textContent = `${(t.n / span).toFixed(0)} fps`; t.tile.classList.toggle("stale", t.n === 0); t.n = 0; }
  }

  openFull(idx) {
    const c = this.cams.find((q) => q.idx === idx);
    if (!c) return;
    this.closeHi();
    this.fullIdx = idx; this._lastFull = null;
    this.full.classList.add("open");
    this.full.dataset.mod = c.modality;
    this.fullLabel.textContent = `${c.id} · ${MOD_LABEL[c.modality] || c.modality}`;
    const r = this.full.getBoundingClientRect();
    this.fullCv.width = Math.round(Math.min(r.width * Math.min(devicePixelRatio || 1, 2), 1920));
    this.fullCv.height = Math.round(Math.min(r.height * Math.min(devicePixelRatio || 1, 2), 1920));
    if (this.omniWs) {
      this.hiWs = new ReconnectingWS(`${this.omniWs}/ws/video?cams=${encodeURIComponent(c.id)}&width=1280&fps=12`, {
        binary: true,
        onMessage: (ab) => {
          if (!(ab instanceof ArrayBuffer) || ab.byteLength < 2 || this._dec || typeof createImageBitmap !== "function") return;
          this._dec = true;
          createImageBitmap(new Blob([new Uint8Array(ab, 1)], { type: "image/jpeg" }))
            .then((b) => { if (this.hiSrc && this.hiSrc.close) this.hiSrc.close(); this.hiSrc = b; })
            .catch(() => {}).finally(() => { this._dec = false; });
        },
      });
    }
  }
  step(d) {
    if (this.fullIdx == null || !this.cams.length) return;
    const i = this.cams.findIndex((c) => c.idx === this.fullIdx);
    this.openFull(this.cams[(i + d + this.cams.length) % this.cams.length].idx);
  }
  closeHi() { if (this.hiWs) { this.hiWs.close(); this.hiWs = null; } if (this.hiSrc && this.hiSrc.close) { try { this.hiSrc.close(); } catch (_) {} } this.hiSrc = null; }
  closeFull() { this.closeHi(); this.fullIdx = null; this.full.classList.remove("open"); }
}

function drawFit(cv, src, contain) {
  const ctx = cv.getContext("2d");
  const sw = src.width || src.videoWidth || 1, sh = src.height || src.videoHeight || 1;
  const s = contain ? Math.min(cv.width / sw, cv.height / sh) : Math.max(cv.width / sw, cv.height / sh);
  const w = sw * s, h = sh * s;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  try { ctx.drawImage(src, (cv.width - w) / 2, (cv.height - h) / 2, w, h); } catch (_) {}
}
