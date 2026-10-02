// NERO OmniVision mobile — Web Speech API (SpeechRecognition, hu-HU).
// Falls back gracefully (supported=false) when the browser lacks it.

export class Voice {
  constructor({ onInterim, onFinal, onState } = {}) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.supported = typeof SR === "function";
    this.onInterim = onInterim || (() => {});
    this.onFinal = onFinal || (() => {});
    this.onState = onState || (() => {});
    this.active = false;
    if (!this.supported) return;
    const r = new SR();
    r.lang = "hu-HU";
    r.interimResults = true;
    r.continuous = false;
    r.maxAlternatives = 1;
    r.onresult = (e) => {
      let interim = "", fin = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) fin += t; else interim += t;
      }
      if (interim) this.onInterim(interim);
      if (fin) this.onFinal(fin.trim());
    };
    r.onerror = (e) => { this.onState("error", e.error || "hiba"); };
    r.onend = () => { this.active = false; this.onState("idle"); };
    this.rec = r;
  }
  start() {
    if (!this.supported || this.active) return false;
    try { this.rec.start(); this.active = true; this.onState("listening"); return true; }
    catch (e) { this.onState("error", String(e && e.message || e)); return false; }
  }
  stop() { if (this.supported && this.active) { try { this.rec.stop(); } catch (_) {} } }
  toggle() { return this.active ? (this.stop(), false) : this.start(); }
}
