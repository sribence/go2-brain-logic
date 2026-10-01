"""mission-control: safety_guard pillar (port 9115, env SAFETY_PORT).

The only process that talks to mc_motion /move. Every motion source
(navigation, mapping explore, follow_executor, omni pursuit) POSTs its
command to this service's /move; it is clamped by core.decide() against the
live person tracks (mc.omni.persons) and LiDAR proximity and only then
forwarded to MOTION_URL/move. When the decision is STOP with nothing left
to forward, mc_motion /stop is sent instead.

It NEVER arms (there is no /arm route and nothing here calls one) and it
never starts motion on its own: the background loop only re-sends a
*smaller* version of a caller's recent command when the safety level
escalates between two caller ticks, and lets mc_motion's 0.5 s watchdog stop
the robot when callers go quiet.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402



def _load_core():
    """Load ./core.py under a unique module name: a plain `import core` could
    collide with mission_control/core (robot_client) on sys.path."""
    import importlib.util
    name = "safety_guard_core"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "core.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_core = _load_core()
STOP, SafetyConfig, SafetyDecision, decide = _core.STOP, _core.SafetyConfig, _core.SafetyDecision, _core.decide
lidar_near_in_direction, rear_clearance, sanitize_cmd = (_core.lidar_near_in_direction, _core.rear_clearance,
                                                         _core.sanitize_cmd)

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

PILLAR = "safety_guard"
PORT = int(os.environ.get("SAFETY_PORT", "9115"))
MOTION_URL = os.environ.get("MOTION_URL", "http://127.0.0.1:9102").rstrip("/")
OMNI_URL = os.environ.get("OMNI_URL", "http://127.0.0.1:9114").rstrip("/")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
PERSONS_CHANNEL = "mc.omni.persons"
STATE_CHANNEL = "mc.safety.state"
STATE_HZ = float(os.environ.get("SAFETY_STATE_HZ", "20"))
LIDAR_SOURCE = os.environ.get("SAFETY_LIDAR_SOURCE", "robot_client")  # robot_client|http|mock|none
LIDAR_URL = os.environ.get("SAFETY_LIDAR_URL", "")
LIDAR_HZ = float(os.environ.get("SAFETY_LIDAR_HZ", "10"))
LIDAR_MAX_AGE_S = float(os.environ.get("SAFETY_LIDAR_MAX_AGE_S", "0.5"))
LIDAR_CONE_RAD = float(os.environ.get("SAFETY_LIDAR_CONE_RAD", "0.6"))
LIDAR_Z_MIN = float(os.environ.get("SAFETY_LIDAR_Z_MIN", "-0.25"))   # base frame
LIDAR_Z_MAX = float(os.environ.get("SAFETY_LIDAR_Z_MAX", "1.5"))
# sources that follow / pursue a person: never approach anyone inside
# SafetyConfig.follow_min_m (2.5 m), CONTRACTS.md section 0
FOLLOW_SOURCES = tuple(s.strip() for s in os.environ.get(
    "SAFETY_FOLLOW_SOURCES", "pursuit,follow,follow_executor").split(",") if s.strip())
CALLER_TIMEOUT_S = float(os.environ.get("SAFETY_CALLER_TIMEOUT_S", "0.3"))
HTTP_TIMEOUT_S = float(os.environ.get("SAFETY_HTTP_TIMEOUT_S", "0.25"))

LOG_DIR = os.path.join(HERE, "logs")
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")
_log_lock = threading.Lock()
_log_to_file = False            # enabled by main(); tests/imports only print


def log_event(level: str, msg: str, **extra: Any) -> None:
    rec = dict({"t": time.time(), "pillar": PILLAR, "level": level, "msg": msg}, **extra)
    print("[safety_guard] %s %s" % (level, msg), flush=True)
    if not _log_to_file:
        return
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with _log_lock, open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# motion forwarder (replaced in tests via monkeypatch of `forwarder`)

def http_forward(path: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    """POST MOTION_URL<path>. Returns (status_code, json); (0, {...}) on
    transport failure. Never raises."""
    if requests is None:
        return 0, {"error": "requests not installed"}
    try:
        r = requests.post(MOTION_URL + path, json=body or {}, timeout=HTTP_TIMEOUT_S)
        try:
            data = r.json()
        except ValueError:
            data = {"text": r.text[:200]}
        return r.status_code, data
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": "mc_motion unreachable: %s" % exc}


forwarder: Callable[[str, Optional[dict]], Tuple[int, dict]] = http_forward


def _forward(path: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    return forwarder(path, body)


# ---------------------------------------------------------------------------
# inputs

class PersonFeed:
    """Latest mc.omni.persons message. `t` is the capture time of the data
    (message 't', capped at receive time so clock skew can't make it look
    fresher than it is)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.persons: Optional[list] = None
        self.t: Optional[float] = None
        self.source = "none"

    def ingest(self, msg: dict, recv_t: Optional[float] = None, source: str = "redis") -> None:
        recv_t = time.time() if recv_t is None else recv_t
        persons = msg.get("persons")
        if not isinstance(persons, list):
            return
        try:
            t = float(msg.get("t", recv_t))
        except (TypeError, ValueError):
            t = recv_t
        if not math.isfinite(t):
            t = recv_t
        with self.lock:
            self.persons = [p for p in persons if isinstance(p, dict)]
            self.t = min(t, recv_t)
            self.source = source

    def get(self) -> Tuple[Optional[list], Optional[float]]:
        with self.lock:
            return self.persons, self.t


class LidarSource:
    """Pluggable LiDAR source -> base-frame points (N x 3) + timestamp."""
    name = "none"

    def read(self) -> Optional[list]:
        return None


class MockLidar(LidarSource):
    """Static ring of points at `radius` m (for tests / --mock)."""
    name = "mock"

    def __init__(self, radius: float = 3.0, n: int = 180) -> None:
        self.points = [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), 0.3)
                       for i in range(n)]

    def read(self) -> Optional[list]:
        return list(self.points)


class RobotClientLidar(LidarSource):
    """core robot_client.get_lidar_points() is in the world frame (see
    navigation/app.py _obstacle_ahead): transform with get_pose()."""
    name = "robot_client"

    def __init__(self) -> None:
        from robot_client import get_robot_client  # lazy: core on PYTHONPATH
        self.robot = get_robot_client()

    def read(self) -> Optional[list]:
        pose = self.robot.get_pose()
        pts = self.robot.get_lidar_points()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        out = []
        for px, py, pz in pts:
            dx, dy = px - pose.x, py - pose.y
            out.append((c * dx + s * dy, -s * dx + c * dy, pz))
        return out


class HttpLidar(LidarSource):
    """GET url -> [[x,y,z],...] or {"points": [...]} already in base frame."""
    name = "http"

    def __init__(self, url: str) -> None:
        self.url = url

    def read(self) -> Optional[list]:
        r = requests.get(self.url, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("points", [])
        out = []
        for p in data:
            if isinstance(p, dict):
                out.append((float(p["x"]), float(p["y"]), float(p["z"])))
            else:
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out


def make_lidar_source(kind: str) -> LidarSource:
    kind = (kind or "none").lower()
    if kind == "mock":
        return MockLidar()
    if kind == "http" and LIDAR_URL:
        return HttpLidar(LIDAR_URL)
    if kind == "robot_client":
        try:
            return RobotClientLidar()
        except Exception as exc:  # noqa: BLE001
            log_event("error", "robot_client lidar unavailable: %s" % exc)
    return LidarSource()            # "none": always missing -> STOP


# ---------------------------------------------------------------------------
# guard state

class Guard:
    def __init__(self, cfg: Optional[SafetyConfig] = None) -> None:
        self.cfg = cfg or SafetyConfig.from_env()
        self.lock = threading.RLock()
        self.persons = PersonFeed()
        self.lidar_points: Optional[list] = None
        self.lidar_t: Optional[float] = None
        self.lidar_name = "none"
        self.odom: Optional[dict] = None
        self.prev: Optional[SafetyDecision] = None
        self.last = None                       # last decision dict
        self.last_cmd_in = (0.0, 0.0, 0.0)
        self.last_cmd_out = (0.0, 0.0, 0.0)
        self.last_caller_t = 0.0               # last /move from any source
        self.last_source = ""
        self.forwarded = 0
        self.stops = 0
        self.escalations = 0
        self.events: list = []

    # -- inputs --------------------------------------------------------------
    def set_lidar(self, points: Optional[list], t: Optional[float] = None) -> None:
        with self.lock:
            self.lidar_points = points
            self.lidar_t = time.time() if t is None else t

    def _lidar(self, now: float) -> Optional[list]:
        if self.lidar_points is None or self.lidar_t is None:
            return None
        if now - self.lidar_t > LIDAR_MAX_AGE_S:
            return None
        return self.lidar_points

    # -- decision ------------------------------------------------------------
    def evaluate(self, cmd: Tuple[float, float, float], now: Optional[float] = None,
                 source: str = "") -> SafetyDecision:
        now = time.time() if now is None else now
        min_d = self.cfg.follow_min_m if source in FOLLOW_SOURCES else None
        with self.lock:
            pts = self._lidar(now)
            near = lidar_near_in_direction(pts, cmd, LIDAR_CONE_RAD, LIDAR_Z_MIN, LIDAR_Z_MAX)
            rear = rear_clearance(pts, LIDAR_CONE_RAD, LIDAR_Z_MIN, LIDAR_Z_MAX)
            persons, pt = self.persons.get()
            dec = decide(persons, near, self.odom, cmd, now, pt, self.cfg,
                         rear_clear_m=rear, prev=self.prev, min_person_dist_m=min_d)
            if self.prev is None or dec.level != self.prev.level:
                self._note("level %s -> %s (%s)" % (self.prev.level if self.prev else "-",
                                                    dec.level, dec.reason))
            self.prev = dec
            self.last = dict(dec.to_dict(), lidar_near_m=None if near is None or math.isinf(near) else round(near, 3),
                             rear_clear_m=None if rear is None or math.isinf(rear) else round(rear, 3))
            return dec

    def _note(self, msg: str) -> None:
        self.events.append({"t": time.time(), "msg": msg})
        del self.events[:-50]
        log_event("info", msg)

    def _send(self, dec: SafetyDecision) -> Tuple[str, int, dict]:
        """Forward a decision: STOP with nothing left -> /stop, else /move."""
        vx, vy, vyaw = dec.cmd_out
        if dec.level == STOP and vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            code, resp = _forward("/stop", None)
            self.stops += 1
            return "stop", code, resp
        code, resp = _forward("/move", {"vx": vx, "vy": vy, "vyaw": vyaw})
        self.forwarded += 1
        return "move", code, resp

    def move(self, vx: Any, vy: Any, vyaw: Any, source: str = "") -> dict:
        cmd = sanitize_cmd((vx, vy, vyaw))
        now = time.time()
        with self.lock:
            dec = self.evaluate(cmd, now, source)
            self.last_cmd_in = cmd
            self.last_caller_t = now
            self.last_source = source or "?"
            kind, code, resp = self._send(dec)
            self.last_cmd_out = dec.cmd_out if kind == "move" else (0.0, 0.0, 0.0)
        return {"forwarded": kind, "motion_status": code, "motion": resp,
                "decision": dec.to_dict()}

    def stop(self, source: str = "") -> dict:
        with self.lock:
            code, resp = _forward("/stop", None)
            self.stops += 1
            self.last_cmd_in = (0.0, 0.0, 0.0)
            self.last_cmd_out = (0.0, 0.0, 0.0)
            self.last_caller_t = 0.0
        self._note("stop requested by %s" % (source or "?"))
        return {"forwarded": "stop", "motion_status": code, "motion": resp}

    def tick(self, now: Optional[float] = None) -> SafetyDecision:
        """Background tick. Recomputes the state; if a caller's command is
        still live (< CALLER_TIMEOUT_S) and the guard now allows less than was
        forwarded, re-send the smaller command (or /stop). Never sends when no
        caller is active and never sends anything larger than before."""
        now = time.time() if now is None else now
        with self.lock:
            live = self.last_caller_t and (now - self.last_caller_t) <= CALLER_TIMEOUT_S
            cmd = self.last_cmd_in if live else (0.0, 0.0, 0.0)
            dec = self.evaluate(cmd, now, self.last_source if live else "")
            if live and any(abs(o) > 0.0 for o in self.last_cmd_out):
                new = dec.cmd_out
                if dec.level == STOP and new == (0.0, 0.0, 0.0):
                    smaller = True
                else:
                    smaller = all(abs(n) <= abs(o) + 1e-9 for n, o in zip(new, self.last_cmd_out)) and \
                        any(abs(n) < abs(o) - 1e-6 for n, o in zip(new, self.last_cmd_out))
                if smaller:
                    kind, _, _ = self._send(dec)
                    self.last_cmd_out = dec.cmd_out if kind == "move" else (0.0, 0.0, 0.0)
                    self.escalations += 1
            return dec

    def state(self) -> dict:
        with self.lock:
            persons, pt = self.persons.get()
            d = dict(self.last or {})
            d.update({"t": time.time(),
                      "persons_age_s": None if pt is None else round(time.time() - pt, 3),
                      "persons_n": None if persons is None else len(persons),
                      "persons_source": self.persons.source,
                      "lidar_source": self.lidar_name,
                      "lidar_age_s": None if self.lidar_t is None else round(time.time() - self.lidar_t, 3),
                      "last_source": self.last_source,
                      "last_cmd_in": self.last_cmd_in, "last_cmd_out": self.last_cmd_out,
                      "forwarded": self.forwarded, "stops": self.stops,
                      "escalations": self.escalations, "events": self.events[-10:][::-1]})
            return d

    def bus_state(self) -> dict:
        """Payload for mc.safety.state (CONTRACTS.md section D)."""
        s = self.last or {}
        return {"t": time.time(), "level": s.get("level", STOP), "vmax": s.get("vmax", 0.0),
                "nearest_person_m": s.get("nearest_person_m"), "reason": s.get("reason", "starting"),
                "vyaw_max": s.get("vyaw_max"), "ttc_s": s.get("ttc_s")}


guard = Guard()


# ---------------------------------------------------------------------------
# background threads (started only by main(), never on import / in tests)

def _redis_client():
    if redis is None:
        return None
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_timeout=2)
        r.ping()
        return r
    except Exception:  # noqa: BLE001
        return None


def _persons_redis_loop() -> None:
    next_try = 0.0
    while True:
        r = None
        if time.time() >= next_try:
            r = _redis_client()
            next_try = time.time() + 5.0       # retry redis every 5 s
        if r is None:
            _persons_poll_once()               # HTTP fallback at ~20 Hz
            time.sleep(0.05)
            continue
        try:
            ps = r.pubsub()
            ps.subscribe(PERSONS_CHANNEL)
            log_event("info", "subscribed to %s" % PERSONS_CHANNEL)
            while True:
                msg = ps.get_message(timeout=0.2)
                if msg and msg.get("type") == "message":
                    try:
                        guard.persons.ingest(json.loads(msg["data"]), source="redis")
                    except (ValueError, TypeError):
                        pass
                _, pt = guard.persons.get()
                if pt is None or time.time() - pt > 0.5:
                    _persons_poll_once()       # redis up but quiet: try HTTP
        except Exception as exc:  # noqa: BLE001
            log_event("warn", "redis persons loop: %s" % exc)
            time.sleep(1.0)


def _persons_poll_once() -> None:
    if requests is None:
        return
    try:
        r = requests.get(OMNI_URL + "/persons", timeout=HTTP_TIMEOUT_S)
        if r.ok:
            guard.persons.ingest(r.json(), source="http")
    except Exception:  # noqa: BLE001
        pass


def _lidar_loop(src: LidarSource) -> None:
    period = 1.0 / max(LIDAR_HZ, 1.0)
    while True:
        t0 = time.time()
        try:
            pts = src.read()
            if pts is not None:
                guard.set_lidar(pts, t0)
        except Exception:  # noqa: BLE001 -- stale lidar -> STOP via max age
            pass
        time.sleep(max(0.0, period - (time.time() - t0)))


def _odom_loop() -> None:
    while requests is not None:
        try:
            r = requests.get(MOTION_URL + "/odom", timeout=HTTP_TIMEOUT_S)
            guard.odom = r.json() if r.ok else None
        except Exception:  # noqa: BLE001
            guard.odom = None
        time.sleep(0.1)


def _state_loop() -> None:
    period = 1.0 / max(STATE_HZ, 1.0)
    r = None
    while True:
        t0 = time.time()
        try:
            guard.tick(t0)
            if r is None:
                r = _redis_client()
            if r is not None:
                r.publish(STATE_CHANNEL, json.dumps(guard.bus_state()))
        except Exception as exc:  # noqa: BLE001
            r = None
            log_event("warn", "state loop: %s" % exc)
        time.sleep(max(0.0, period - (time.time() - t0)))


def start_background(lidar_kind: str = LIDAR_SOURCE) -> None:
    src = make_lidar_source(lidar_kind)
    guard.lidar_name = src.name
    for target, args, name in ((_persons_redis_loop, (), "persons"), (_lidar_loop, (src,), "lidar"),
                               (_odom_loop, (), "odom"), (_state_loop, (), "state")):
        threading.Thread(target=target, args=args, daemon=True, name="safety-" + name).start()


# ---------------------------------------------------------------------------
# HTTP API

app = FastAPI(title="safety_guard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MoveBody(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    source: str = ""


class StopBody(BaseModel):
    source: str = ""


@app.post("/move")
def move(body: MoveBody):
    res = guard.move(body.vx, body.vy, body.vyaw, body.source)
    code = res["motion_status"]
    if code == 0:
        raise HTTPException(status_code=502, detail=res)
    if code >= 400:
        # mc_motion refused (e.g. 409 not armed): pass the status through so
        # the caller sees it instead of believing the robot moves.
        raise HTTPException(status_code=code, detail=res)
    return res


@app.post("/stop")
def stop(body: Optional[StopBody] = None):
    return guard.stop(body.source if body else "")


@app.post("/estop")
def estop():
    """Pass-through to mc_motion /estop: can only make the robot safer."""
    code, resp = _forward("/estop", None)
    guard._note("estop forwarded (%s)" % code)
    return {"forwarded": "estop", "motion_status": code, "motion": resp}


@app.get("/state")
def state():
    return guard.state()


@app.get("/health")
def health():
    s = guard.bus_state()
    return {"ok": True, "pillar": PILLAR, "port": PORT, "motion_url": MOTION_URL,
            "level": s["level"], "lidar_source": guard.lidar_name,
            "persons_source": guard.persons.source}


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="mock LiDAR ring at 3 m")
    args = ap.parse_args()
    global _log_to_file
    _log_to_file = True
    start_background("mock" if args.mock else LIDAR_SOURCE)
    log_event("info", "start :%d -> %s, lidar=%s" % (PORT, MOTION_URL, guard.lidar_name))
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
