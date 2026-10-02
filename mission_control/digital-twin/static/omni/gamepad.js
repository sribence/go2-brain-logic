// OmniView — browser Gamepad API input (any BT pad: Xbox, DualShock/DualSense,
// 8BitDo, generic). "standard" mapping by default + per-pad press-to-assign
// remap saved in localStorage. No DOM; reusable by the mobile UI.
// The UI never sends /move from the pad: left stick = goal cursor, A = plan+go,
// B = abort, X = jump mode, RB = dead-man, right stick = camera (CONTRACT §7, §9.4).

export const BUTTONS = ["A", "B", "X", "Y", "LB", "RB", "LT", "RT", "BACK", "START", "LS", "RS", "UP", "DOWN", "LEFT", "RIGHT"];
export const AXES = ["LX", "LY", "RX", "RY"];
export const ACTIONS_HELP = { A: "plan + GO", B: "abort", X: "jump mode", Y: "scan here", RB: "DEAD-MAN (hold)", LB: "cycle speed", START: "pause / resume", BACK: "clear preview" };

const STD = () => {
  const m = {};
  BUTTONS.forEach((n, i) => { m[n] = { type: "button", index: i }; });
  AXES.forEach((n, i) => { m[n] = { type: "axis", index: i, sign: 1 }; });
  return m;
};
const LS = "omniview.pad.";

export class GamepadInput {
  constructor({ onButton, deadzone = 0.15 } = {}) {
    this.onButton = onButton || (() => {});
    this.dead = deadzone;
    this.index = null;
    this.prev = {};
    this.map = STD();
    this.padId = "";
    this.capture = null;            // {name, resolve, base}
    this.supported = typeof navigator !== "undefined" && typeof navigator.getGamepads === "function";
    if (typeof addEventListener === "function") {
      addEventListener("gamepadconnected", (e) => { if (this.index == null) this._use(e.gamepad); });
      addEventListener("gamepaddisconnected", (e) => { if (e.gamepad.index === this.index) { this.index = null; this.prev = {}; } });
    }
  }

  _pads() { try { return this.supported ? [...navigator.getGamepads()].filter(Boolean) : []; } catch (_) { return []; } }
  _use(pad) {
    this.index = pad.index; this.padId = pad.id || "pad";
    this.map = STD();
    if (pad.mapping !== "standard") { /* non-standard: indices usually still close; user can remap */ }
    try { Object.assign(this.map, JSON.parse(localStorage.getItem(LS + this.padId) || "{}")); } catch (_) {}
  }
  pad() {
    const pads = this._pads();
    if (this.index == null && pads.length) this._use(pads[0]);
    return pads.find((p) => p.index === this.index) || null;
  }

  info() {
    const p = this.pad();
    if (!p) return null;
    const bat = p.battery || p.batteryInfo || null;    // not standard; some browsers / pads expose it
    const lvl = bat && (bat.level != null ? bat.level : bat.percent != null ? bat.percent / 100 : null);
    return { id: p.id, short: shortName(p.id), mapping: p.mapping || "non-standard", battery: lvl, haptics: !!(p.vibrationActuator) };
  }

  _btn(p, b) {
    if (!b) return 0;
    if (b.type === "button") { const x = p.buttons[b.index]; return x ? (typeof x === "object" ? (x.pressed ? Math.max(x.value, 1) : x.value) : x) : 0; }
    const v = (p.axes[b.index] || 0) * (b.sign || 1);       // axis bound as button (e.g. D-pad hat)
    return v > 0.5 ? v : 0;
  }
  _axis(p, b) {
    if (!b) return 0;
    if (b.type === "axis") { const v = (p.axes[b.index] || 0) * (b.sign || 1); return Math.abs(v) < this.dead ? 0 : v; }
    return this._btn(p, b) > 0.5 ? (b.sign || 1) : 0;
  }

  // call once per frame. Returns null (no pad) or {axes:{LX,LY,RX,RY}, buttons:{name:bool}}
  poll() {
    const p = this.pad();
    if (!p) return null;
    if (this.capture) { this._captureStep(p); return { axes: { LX: 0, LY: 0, RX: 0, RY: 0 }, buttons: {} }; }
    const axes = {}, buttons = {};
    for (const n of AXES) axes[n] = this._axis(p, this.map[n]);
    for (const n of BUTTONS) {
      const down = this._btn(p, this.map[n]) > 0.5;
      buttons[n] = down;
      if (down !== !!this.prev[n]) { try { this.onButton(n, down); } catch (e) { console.warn("[pad]", e); } }
      this.prev[n] = down;
    }
    return { axes, buttons };
  }

  // press-to-assign: resolves with the binding for `name` (button or axis deflection)
  startCapture(name) {
    const p = this.pad();
    if (!p) return Promise.reject(new Error("no gamepad"));
    return new Promise((resolve, reject) => {
      this.capture = { name, resolve, reject, base: { axes: [...p.axes], buttons: p.buttons.map((b) => (typeof b === "object" ? b.value : b)) }, t0: performance.now() };
    });
  }
  cancelCapture() { if (this.capture) { this.capture.reject(new Error("cancelled")); this.capture = null; } }
  _captureStep(p) {
    const c = this.capture;
    if (performance.now() - c.t0 > 8000) { this.cancelCapture(); return; }
    let bind = null;
    p.buttons.forEach((b, i) => { const v = typeof b === "object" ? b.value : b; if (!bind && v > 0.6 && (c.base.buttons[i] || 0) < 0.3) bind = { type: "button", index: i }; });
    if (!bind) p.axes.forEach((v, i) => { if (!bind && Math.abs(v - (c.base.axes[i] || 0)) > 0.6) bind = { type: "axis", index: i, sign: v >= 0 ? 1 : -1 }; });
    if (!bind) return;
    // stick axes: positive direction = right / down (same as standard)
    this.setBinding(c.name, bind);
    this.capture = null;
    this.prev = {};
    c.resolve(bind);
  }
  setBinding(name, bind) {
    this.map[name] = bind;
    const diff = {};
    const std = STD();
    for (const k of Object.keys(this.map)) if (JSON.stringify(this.map[k]) !== JSON.stringify(std[k])) diff[k] = this.map[k];
    try { localStorage.setItem(LS + this.padId, JSON.stringify(diff)); } catch (_) {}
  }
  resetBindings() { this.map = STD(); try { localStorage.removeItem(LS + this.padId); } catch (_) {} }
  bindingText(name) { const b = this.map[name]; return !b ? "—" : b.type === "button" ? `btn ${b.index}` : `axis ${b.index}${b.sign < 0 ? "−" : "+"}`; }

  rumble(ms = 80, strength = 0.5) {
    const p = this.pad();
    const a = p && p.vibrationActuator;
    if (a && typeof a.playEffect === "function") a.playEffect("dual-rumble", { duration: ms, strongMagnitude: strength, weakMagnitude: strength }).catch(() => {});
  }
}

export function shortName(id) {
  const s = String(id || "");
  if (/xbox|xinput|045e/i.test(s)) return "XBOX";
  if (/dualsense|0ce6/i.test(s)) return "DUALSENSE";
  if (/dualshock|054c|wireless controller/i.test(s)) return "DUALSHOCK";
  if (/8bitdo|2dc8/i.test(s)) return "8BITDO";
  return (s.split("(")[0].trim() || "PAD").slice(0, 14).toUpperCase();
}
