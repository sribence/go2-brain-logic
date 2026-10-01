// OmniView — network helpers. Every source is optional; nothing throws.

const params = new URLSearchParams(location.search);

function hostPort(param, defPort) {
  const v = params.get(param);
  let h = v || location.hostname || "127.0.0.1";
  h = h.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const hasPort = h.startsWith("[") ? /\]:\d+$/.test(h) : /^[^:]+:\d+$/.test(h);
  return hasPort ? h : `${h}:${defPort}`;
}

export function endpoints() {
  const secure = location.protocol === "https:";
  const omni = hostPort("omni", 9114);
  const safety = hostPort("safety", 9115);
  return {
    demoForced: params.get("demo") === "1" || params.get("demo") === "true",
    omniHttp: `${secure ? "https" : "http"}://${omni}`,
    omniWs: `${secure ? "wss" : "ws"}://${omni}`,
    safetyHttp: `${secure ? "https" : "http"}://${safety}`,
    safetyWs: `${secure ? "wss" : "ws"}://${safety}`,
    twinWs: `${secure ? "wss" : "ws"}://${location.host}`,
    sameOrigin: location.protocol.startsWith("http"),
  };
}

export async function fetchJSON(url, timeoutMs = 2000, opts = {}) {
  const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const tm = ctl ? setTimeout(() => ctl.abort(), timeoutMs) : null;
  try {
    const r = await fetch(url, { cache: "no-store", ...opts, signal: ctl ? ctl.signal : undefined });
    if (!r.ok) return { ok: false, status: r.status, data: null };
    let data = null;
    try { data = await r.json(); } catch (_) { data = null; }
    return { ok: true, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: null, error: String(e && e.message || e) };
  } finally {
    if (tm) clearTimeout(tm);
  }
}

export function postJSON(url, body, timeoutMs = 1500) {
  return fetchJSON(url, timeoutMs, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

// WebSocket with exponential backoff reconnect. onState("connecting"|"live"|"down")
export class ReconnectingWS {
  constructor(url, { binary = false, onMessage, onState, onOpen } = {}) {
    this.url = url; this.binary = binary;
    this.onMessage = onMessage || (() => {});
    this.onState = onState || (() => {});
    this.onOpen = onOpen || (() => {});
    this.backoff = 1000; this.ws = null; this.closed = false; this.timer = null;
    this.msgs = 0; this.lastMsgT = 0;
    this._connect();
  }
  _connect() {
    if (this.closed) return;
    this.onState("connecting");
    let ws;
    try { ws = new WebSocket(this.url); } catch (e) { this._retry(); return; }
    if (this.binary) ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => { this.backoff = 1000; this.onState("live"); try { this.onOpen(); } catch (e) { console.warn(e); } };
    ws.onmessage = (ev) => {
      this.msgs++; this.lastMsgT = performance.now();
      try { this.onMessage(ev.data); } catch (e) { console.warn("[omni] ws handler error", this.url, e); }
    };
    ws.onerror = () => {};
    ws.onclose = () => { this.ws = null; this.onState("down"); this._retry(); };
  }
  _retry() {
    if (this.closed) return;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this._connect(), this.backoff);
    this.backoff = Math.min(this.backoff * 1.8, 15000);
  }
  close() {
    this.closed = true; clearTimeout(this.timer);
    if (this.ws) { try { this.ws.close(); } catch (_) {} }
  }
}

// Periodic poller with backoff on failure.
export class Poller {
  constructor(fn, periodMs, maxMs = 10000) {
    this.fn = fn; this.period = periodMs; this.max = maxMs; this.cur = periodMs; this.stopped = false;
    this._tick();
  }
  async _tick() {
    if (this.stopped) return;
    let ok = false;
    try { ok = await this.fn(); } catch (_) { ok = false; }
    this.cur = ok ? this.period : Math.min(this.cur * 1.6, this.max);
    setTimeout(() => this._tick(), this.cur);
  }
  stop() { this.stopped = true; }
}
