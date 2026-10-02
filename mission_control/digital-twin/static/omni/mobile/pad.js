// NERO OmniVision mobile — browser Gamepad API (any BT pad paired to the phone,
// "standard" mapping; CONTRACT.md §9.4). Left stick = goal cursor, A = plan+GO,
// B = abort, X = capture, Y = say, RB = dead-man (held), right stick = orbit.

const BTN = { A: 0, B: 1, X: 2, Y: 3, LB: 4, RB: 5, LT: 6, RT: 7, BACK: 8, START: 9 };
const DEAD = 0.18;
const dz = (v) => (Math.abs(v) < DEAD ? 0 : (v - Math.sign(v) * DEAD) / (1 - DEAD));

export class Pad {
  // h: {onConnect(id|null), onStick(lx, ly, dt), onOrbit(rx, ry, dt), onButton(name), onDeadman(bool)}
  constructor(h) {
    this.h = h;
    this.idx = null; this.id = null;
    this.prev = [];
    this.supported = typeof navigator !== "undefined" && typeof navigator.getGamepads === "function";
    addEventListener("gamepadconnected", (e) => { if (this.idx == null) this._attach(e.gamepad); });
    addEventListener("gamepaddisconnected", (e) => { if (e.gamepad.index === this.idx) this._detach(); });
  }
  _attach(gp) { this.idx = gp.index; this.id = gp.id; this.prev = gp.buttons.map((b) => b.pressed); this.h.onConnect(gp.id, gp.mapping); }
  _detach() { if (this.rb) this.h.onDeadman(false); this.rb = false; this.idx = null; this.id = null; this.h.onConnect(null); }

  poll(dt) {
    if (!this.supported) return;
    let pads = [];
    try { pads = navigator.getGamepads() || []; } catch (_) { return; }
    let gp = this.idx != null ? pads[this.idx] : null;
    if (!gp) {
      if (this.idx != null) this._detach();
      gp = [...pads].find((p) => p && p.connected);
      if (!gp) return;
      this._attach(gp);
    }
    const ax = gp.axes || [];
    const lx = dz(ax[0] || 0), ly = dz(ax[1] || 0), rx = dz(ax[2] || 0), ry = dz(ax[3] || 0);
    if (lx || ly) this.h.onStick(lx, -ly, dt);
    if (rx || ry) this.h.onOrbit(rx, ry, dt);
    const pressed = gp.buttons.map((b) => b.pressed || b.value > 0.5);
    for (const [name, i] of Object.entries(BTN)) {
      if (pressed[i] && !this.prev[i] && name !== "RB") this.h.onButton(name);
    }
    const rb = !!pressed[BTN.RB];
    if (rb !== !!this.rb) { this.rb = rb; this.h.onDeadman(rb); }
    this.prev = pressed;
  }
}
