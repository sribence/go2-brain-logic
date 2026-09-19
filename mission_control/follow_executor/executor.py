"""follow_executor -- forwards the person follower's command to mc_motion.

The perception pillar computes a velocity command but never moves the robot.
This small process is the only link from that command to the legs, and it
can be removed with `docker rm -f nero_go2_follow_executor` to return the
robot to observe-only.

Every 1/RATE_HZ seconds:
  1. read mc_motion /odom and push the robot motion since the last tick to
     perception /follow/ego (keeps the follower's gate on the person while
     the robot turns or walks);
  2. read perception /follow and mc_motion /health;
  3. POST mc_motion /move only when ALL of these hold (`decide()`), with vx
     clamped to [-max_vx_back, max_vx] and vyaw to +-max_vyaw:
       - executor enabled (not latched off by a remote override)
       - mc_motion armed (only the operator arms; this process never does)
       - command dry_run is false
       - follower state TRACKING and not on hold
       - follower result younger than MAX_RESULT_AGE_S
     otherwise send one /stop on the transition and then nothing, so the
     mc_motion watchdog also stops the robot if this process dies.

A new remote override seen in mc_motion /health latches the executor off
until POST /enable on the status port. The limits (EXEC_MAX_VX,
EXEC_MAX_VX_BACK, EXEC_MAX_VYAW) clamp on top of the follower's and
mc_motion's own limits. POST /limits lowers them at runtime (the console's
speed sliders); the env values are hard caps it cannot exceed.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

PERCEPTION = os.environ.get("PERCEPTION_URL", "http://127.0.0.1:9112")
MOTION = os.environ.get("MOTION_URL", "http://127.0.0.1:9102")
PORT = int(os.environ.get("EXECUTOR_PORT", "9113"))
RATE_HZ = float(os.environ.get("RATE_HZ", "20"))
MAX_RESULT_AGE_S = float(os.environ.get("MAX_RESULT_AGE_S", "0.4"))
# Hard caps. POST /limits can lower the working limits at runtime, never raise
# them above these.
EXEC_MAX_VX = float(os.environ.get("EXEC_MAX_VX", "0.4"))            # forward, m/s
EXEC_MAX_VX_BACK = float(os.environ.get("EXEC_MAX_VX_BACK", "0.2"))  # backward, m/s
EXEC_MAX_VYAW = float(os.environ.get("EXEC_MAX_VYAW", "0.8"))        # rad/s
HTTP_TIMEOUT_S = 0.25


def _clamp(v: float, limit: float) -> float:
    if v is None or not math.isfinite(v):
        return 0.0
    return max(-limit, min(limit, float(v)))


def decide(follow: Optional[dict], health: Optional[dict], enabled: bool,
           max_vx: float = EXEC_MAX_VX, max_vyaw: float = EXEC_MAX_VYAW,
           max_age_s: float = MAX_RESULT_AGE_S,
           max_vx_back: float = EXEC_MAX_VX_BACK) -> tuple[bool, float, float, str]:
    """(send, vx, vyaw, reason). Pure: every gate in one testable place."""
    if not enabled:
        return False, 0.0, 0.0, "executor disabled (remote override or operator)"
    if not health:
        return False, 0.0, 0.0, "mc_motion unreachable"
    if not health.get("armed"):
        return False, 0.0, 0.0, "mc_motion not armed"
    if not follow:
        return False, 0.0, 0.0, "perception unreachable"
    age = follow.get("age_s")
    if age is None or age > max_age_s:
        return False, 0.0, 0.0, f"perception result stale ({age} s)"
    cmd = follow.get("command") or {}
    if cmd.get("dry_run", True):
        return False, 0.0, 0.0, "dry_run"
    if follow.get("state") != "TRACKING":
        return False, 0.0, 0.0, f"state {follow.get('state')}"
    if follow.get("hold"):
        return False, 0.0, 0.0, "hold"
    vx = _clamp(cmd.get("vx", 0.0), max(max_vx, max_vx_back))
    vx = min(vx, max_vx) if vx > 0 else max(vx, -max_vx_back)
    vyaw = _clamp(cmd.get("vyaw", 0.0), max_vyaw)
    return True, vx, vyaw, "tracking"


def ego_delta(prev: tuple, cur: tuple) -> tuple[float, float, float]:
    """Odometry (x, y, yaw) world poses -> motion in the previous base frame."""
    x0, y0, a0 = prev
    x1, y1, a1 = cur
    dxw, dyw = x1 - x0, y1 - y0
    c, s = math.cos(a0), math.sin(a0)
    dyaw = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
    return c * dxw + s * dyw, -s * dxw + c * dyw, dyaw


def _get(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _post(url: str, body: Optional[dict] = None) -> Optional[dict]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            return json.loads(r.read())
    except Exception:
        return None


class Executor:
    def __init__(self) -> None:
        self.enabled = True
        self.moving = False
        self.reason = "starting"
        self.last_cmd = (0.0, 0.0)
        self.sent = 0
        self.stops = 0
        self.overrides_seen: Optional[int] = None
        self._odom_prev: Optional[tuple] = None
        self.events: list = []
        self.limits = {"max_vx": EXEC_MAX_VX, "max_vx_back": EXEC_MAX_VX_BACK, "max_vyaw": EXEC_MAX_VYAW}

    def set_limits(self, body: dict) -> None:
        """Lower (or restore) the working limits. A value above the hard cap,
        a negative or non-finite value, or an unknown key raises ValueError."""
        caps = {"max_vx": EXEC_MAX_VX, "max_vx_back": EXEC_MAX_VX_BACK, "max_vyaw": EXEC_MAX_VYAW}
        new = dict(self.limits)
        for k, v in body.items():
            if k not in caps:
                raise ValueError(f"unknown limit {k!r}, allowed: {sorted(caps)}")
            v = float(v)
            if not math.isfinite(v) or v < 0 or v > caps[k]:
                raise ValueError(f"{k} must be between 0 and {caps[k]}")
            new[k] = v
        self.limits = new
        self.log(f"limits set to {new}")

    def log(self, msg: str) -> None:
        self.events.append({"t": time.time(), "msg": msg})
        del self.events[:-50]
        print(f"[follow_executor] {msg}", flush=True)

    def tick(self) -> None:
        health = _get(f"{MOTION}/health")
        if health:
            n = (health.get("remote") or {}).get("overrides", 0)
            if self.overrides_seen is not None and n > self.overrides_seen and self.enabled:
                self.enabled = False
                self.log("remote override seen -- executor latched OFF (POST /enable)")
            self.overrides_seen = n

        odom = _get(f"{MOTION}/odom")
        if odom and odom.get("age_s", 9) < 0.5:
            cur = (odom["x"], odom["y"], odom["yaw"])
            if self._odom_prev is not None:
                dx, dy, dyaw = ego_delta(self._odom_prev, cur)
                if abs(dx) + abs(dy) > 1e-3 or abs(dyaw) > 1e-3:
                    _post(f"{PERCEPTION}/follow/ego", {"dx": dx, "dy": dy, "dyaw": dyaw})
            self._odom_prev = cur
        else:
            self._odom_prev = None

        follow = _get(f"{PERCEPTION}/follow")
        lim = self.limits
        send, vx, vyaw, reason = decide(follow, health, self.enabled, max_vx=lim["max_vx"],
                                        max_vyaw=lim["max_vyaw"], max_vx_back=lim["max_vx_back"])
        if reason != self.reason:
            self.log(f"{self.reason} -> {reason}")
        self.reason = reason
        if send:
            if _post(f"{MOTION}/move", {"vx": vx, "vy": 0.0, "vyaw": vyaw}) is None:
                self.reason = "move rejected/unreachable"
            else:
                self.moving = True
                self.last_cmd = (vx, vyaw)
                self.sent += 1
        elif self.moving:
            _post(f"{MOTION}/stop")
            self.moving = False
            self.last_cmd = (0.0, 0.0)
            self.stops += 1

    def run(self) -> None:
        period = 1.0 / RATE_HZ
        self.log(f"start: limits vx<={EXEC_MAX_VX} back<={EXEC_MAX_VX_BACK} vyaw<={EXEC_MAX_VYAW}, {RATE_HZ} Hz")
        while True:
            t0 = time.time()
            try:
                self.tick()
            except Exception as exc:          # never die silently mid-motion
                self.log(f"tick error: {exc}")
                if self.moving:
                    _post(f"{MOTION}/stop")
                    self.moving = False
            time.sleep(max(0.0, period - (time.time() - t0)))

    def status(self) -> dict:
        return {"enabled": self.enabled, "moving": self.moving, "reason": self.reason,
                "last_cmd": {"vx": self.last_cmd[0], "vyaw": self.last_cmd[1]},
                "sent": self.sent, "stops": self.stops,
                "limits": dict(self.limits),
                "hard_caps": {"max_vx": EXEC_MAX_VX, "max_vx_back": EXEC_MAX_VX_BACK,
                              "max_vyaw": EXEC_MAX_VYAW},
                "events": self.events[-10:][::-1], "t": time.time()}


EXEC = Executor()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, EXEC.status()) if self.path in ("/", "/status") else self._send(404, {})

    def do_POST(self):
        if self.path == "/limits":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                EXEC.set_limits(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, TypeError, AttributeError) as exc:
                return self._send(422, {"detail": str(exc)})
            return self._send(200, EXEC.status())
        if self.path == "/enable":
            EXEC.enabled = True
            EXEC.log("enabled by operator")
        elif self.path == "/disable":
            EXEC.enabled = False
            EXEC.log("disabled by operator")
        else:
            return self._send(404, {})
        self._send(200, EXEC.status())

    def do_OPTIONS(self):          # CORS preflight for JSON POSTs from the console
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=EXEC.run, daemon=True, name="executor").start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
