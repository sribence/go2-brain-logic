// NERO OmniVision mobile — touch gesture recogniser on the 3D canvas.
//   1 finger tap            -> onTap(x, y)
//   1 finger press + drag   -> onGoalDrag(start{x,y}, cur{x,y}, phase "start"|"move"|"end")  (goal + yaw)
//   1 finger long-press     -> onLongPress(x, y)
//   2 fingers               -> onOrbit(dAz, dEl, zoomFactor)  (twist/swipe = orbit, pinch = zoom)
//   mouse: right/middle drag = orbit, wheel = zoom (desktop testing)

const LONG_MS = 520, MOVE_PX = 12;

export class Gestures {
  constructor(el, h) {
    this.el = el; this.h = h;
    this.ptrs = new Map();
    this.mode = "idle";
    this.timer = null;
    el.addEventListener("pointerdown", (e) => this._down(e));
    el.addEventListener("pointermove", (e) => this._move(e));
    el.addEventListener("pointerup", (e) => this._up(e));
    el.addEventListener("pointercancel", (e) => this._cancel(e));
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("wheel", (e) => { e.preventDefault(); this.h.onOrbit(0, 0, Math.exp(e.deltaY * 0.0012)); }, { passive: false });
  }

  _down(e) {
    try { this.el.setPointerCapture(e.pointerId); } catch (_) {}
    this.ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY });
    if (e.pointerType === "mouse" && e.button !== 0) { this.mode = "mouseorbit"; return; }
    if (this.ptrs.size === 1) {
      this.mode = "pending";
      this.start = { x: e.clientX, y: e.clientY, t: performance.now() };
      clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        if (this.mode === "pending") { this.mode = "menu"; this.h.onLongPress(this.start.x, this.start.y); }
      }, LONG_MS);
    } else if (this.ptrs.size === 2) {
      clearTimeout(this.timer);
      if (this.mode === "goaldrag") this.h.onGoalDrag(this.start, null, "cancel");
      this.mode = "two";
      this.two = this._pair();
    }
  }

  _pair() {
    const [a, b] = [...this.ptrs.values()];
    return { d: Math.hypot(b.x - a.x, b.y - a.y) || 1, ang: Math.atan2(b.y - a.y, b.x - a.x), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
  }

  _move(e) {
    const p = this.ptrs.get(e.pointerId);
    if (!p) return;
    const px = p.x, py = p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (this.mode === "mouseorbit") { this.h.onOrbit(-(p.x - px) * 0.008, (p.y - py) * 0.006, 1); return; }
    if (this.mode === "pending") {
      if (Math.hypot(p.x - this.start.x, p.y - this.start.y) > MOVE_PX) {
        clearTimeout(this.timer);
        this.mode = "goaldrag";
        this.h.onGoalDrag(this.start, { x: p.x, y: p.y }, "start");
      }
    } else if (this.mode === "goaldrag") {
      this.h.onGoalDrag(this.start, { x: p.x, y: p.y }, "move");
    } else if (this.mode === "two" && this.ptrs.size >= 2) {
      const n = this._pair(), o = this.two;
      let dAng = n.ang - o.ang;
      dAng = Math.atan2(Math.sin(dAng), Math.cos(dAng));
      const dx = n.cx - o.cx, dy = n.cy - o.cy;
      // twist + horizontal swipe = orbit azimuth, vertical swipe = elevation, pinch = zoom
      this.h.onOrbit(-dAng - dx * 0.006, dy * 0.005, o.d / n.d);
      this.two = n;
    }
  }

  _up(e) {
    const had = this.ptrs.delete(e.pointerId);
    if (!had) return;
    if (this.mode === "mouseorbit") { if (!this.ptrs.size) this.mode = "idle"; return; }
    if (this.mode === "pending") {
      clearTimeout(this.timer);
      this.mode = "idle";
      this.h.onTap(this.start.x, this.start.y);
    } else if (this.mode === "goaldrag") {
      this.mode = "idle";
      this.h.onGoalDrag(this.start, { x: e.clientX, y: e.clientY }, "end");
    } else if (this.mode === "two") {
      if (this.ptrs.size === 1) this.two = null;
      if (!this.ptrs.size) this.mode = "idle";
    } else if (!this.ptrs.size) this.mode = "idle";
  }

  _cancel(e) {
    this.ptrs.delete(e.pointerId);
    clearTimeout(this.timer);
    if (this.mode === "goaldrag") this.h.onGoalDrag(this.start, null, "cancel");
    if (!this.ptrs.size) this.mode = "idle";
  }
}
