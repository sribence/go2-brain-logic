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
from fastapi.responses import JSONResponse  # noqa: E402
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
CLEAR, DeadmanRegistry, sprint_preconditions, action_check = (_core.CLEAR, _core.DeadmanRegistry,
                                                              _core.sprint_preconditions, _core.action_check)
JUMP_ACTIONS, ALLOWED_ACTIONS = _core.JUMP_ACTIONS, _core.ALLOWED_ACTIONS

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
# mission/CONTRACT.md section 4: dead-man, battery (sprint gate), actions
DEADMAN_TIMEOUT_S = float(os.environ.get("DEADMAN_TIMEOUT_S", "0.3"))
CORE_URL = os.environ.get("CORE_URL", "http://127.0.0.1:9101").rstrip("/")
BATTERY_MAX_AGE_S = float(os.environ.get("SAFETY_BATTERY_MAX_AGE_S", "5.0"))
MOTION_WATCHDOG_S = float(os.environ.get("SAFETY_MOTION_WATCHDOG_S", "0.5"))  # mc_motion COMMAND_TIMEOUT_S
# mission/CONTRACT.md 4 + 9.4: sprint only from these sources (gamepad = robot-side BT
# bridge / browser pad, which holds the dead-man with client_id "gamepad")
SPRINT_SOURCES = tuple(s.strip() for s in os.environ.get(
    "SAFETY_SPRINT_SOURCES", "mission,gamepad").split(",") if s.strip())
# 9.6: other fleet robots as person-like tracks with a 1.5 m bubble
FLEET_AWARE = os.environ.get("SAFETY_FLEET_AWARE", "1") == "1"
FLEET_CHANNEL = "mc.fleet.robots"
ROBOT_ID = os.environ.get("ROBOT_ID", "go2")
FLEET_BUBBLE_M = float(os.environ.get("SAFETY_FLEET_BUBBLE_M", "1.5"))
FLEET_MAX_AGE_S = float(os.environ.get("SAFETY_FLEET_MAX_AGE_S", "1.0"))
POSE_MAX_AGE_S = float(os.environ.get("SAFETY_POSE_MAX_AGE_S", "0.5"))
CORE_POLL_HZ = float(os.environ.get("SAFETY_CORE_POLL_HZ", "10"))
fleet_tracks = _core.fleet_tracks

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
        # dead-man / sprint / actions (mission/CONTRACT.md section 4)
        self.deadman = DeadmanRegistry(DEADMAN_TIMEOUT_S)
        self.battery_pct: Optional[float] = None
        self.battery_t: Optional[float] = None
        self.last_profile: Optional[str] = None
        self.sprint_active = False             # last forwarded /move was a sprint
        self.sprint_latch = False              # dead-man dropped mid-sprint: sprint-profile moves -> /stop
        self.last_motion_t = 0.0               # last time a non-zero command was forwarded
        self.sprint_reason = ""
        self.deadman_drops = 0
        self.actions: list = []
        self.fleet_robots: Optional[list] = None
        self.fleet_t: Optional[float] = None
        self.own_pose: Optional[Tuple[float, float, float]] = None   # world (core /state)
        self.own_pose_t: Optional[float] = None

    # -- fleet (9.6) ------------------------------------------------------------
    def ingest_fleet(self, msg: Any, recv_t: Optional[float] = None) -> None:
        recv_t = time.time() if recv_t is None else recv_t
        if not isinstance(msg, dict) or not isinstance(msg.get("robots"), list):
            return
        try:
            t = float(msg.get("t", recv_t))
        except (TypeError, ValueError):
            t = recv_t
        with self.lock:
            self.fleet_robots = [r for r in msg["robots"] if isinstance(r, dict)]
            self.fleet_t = min(t, recv_t) if math.isfinite(t) else recv_t

    def set_pose(self, x: float, y: float, yaw: float, t: Optional[float] = None) -> None:
        with self.lock:
            self.own_pose = (x, y, yaw)
            self.own_pose_t = time.time() if t is None else t

    def _fleet(self, now: float) -> list:
        """Fresh other-robot tracks (base frame), [] when off / no data."""
        if not FLEET_AWARE or self.fleet_robots is None or self.fleet_t is None or self.own_pose_t is None:
            return []
        if now - self.fleet_t > FLEET_MAX_AGE_S or now - self.own_pose_t > POSE_MAX_AGE_S:
            return []
        return fleet_tracks(self.fleet_robots, ROBOT_ID, self.own_pose, FLEET_BUBBLE_M, self.cfg.stop_m,
                            max(10.0, self.cfg.sprint_corridor_len_m + 2.0))

    def _persons(self, now: float) -> Tuple[Optional[list], Optional[float]]:
        """Person tracks + fleet robots. No person data stays None (SLOW)."""
        persons, pt = self.persons.get()
        if persons is None:
            return None, pt
        robots = self._fleet(now)
        return (list(persons) + robots if robots else persons), pt

    # -- dead-man / battery ----------------------------------------------------
    def deadman_beat(self, client_id: Any, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        with self.lock:
            ok = self.deadman.beat(client_id, now)
            if ok and self.sprint_latch:
                self.sprint_latch = False
                self._note("dead-man held again by %s: sprint latch cleared" % client_id)
            return ok

    def deadman_alive(self, now: Optional[float] = None) -> bool:
        with self.lock:
            return self.deadman.alive(time.time() if now is None else now)

    def set_battery(self, pct: Optional[float], t: Optional[float] = None) -> None:
        with self.lock:
            self.battery_pct = pct
            self.battery_t = time.time() if t is None else t

    def _battery(self, now: float) -> Optional[float]:
        if self.battery_pct is None or self.battery_t is None or now - self.battery_t > BATTERY_MAX_AGE_S:
            return None
        return self.battery_pct

    def _persons_fresh(self, now: float) -> bool:
        persons, pt = self.persons.get()
        return persons is not None and pt is not None and now - pt <= self.cfg.persons_stale_s

    def _still_for(self, now: float) -> float:
        """Seconds since the last non-zero command could still be driving the
        robot (a non-zero last_cmd_out keeps it moving until mc_motion's
        watchdog fires MOTION_WATCHDOG_S after the last caller tick)."""
        if any(abs(v) > self.cfg.action_still_eps for v in self.last_cmd_out):
            if not self.last_caller_t:
                return now - self.last_motion_t - MOTION_WATCHDOG_S
            return now - max(self.last_motion_t, self.last_caller_t) - MOTION_WATCHDOG_S
        return now - self.last_motion_t

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
                 source: str = "", profile: Optional[str] = None) -> SafetyDecision:
        now = time.time() if now is None else now
        min_d = self.cfg.follow_min_m if source in FOLLOW_SOURCES else None
        with self.lock:
            pts = self._lidar(now)
            near = lidar_near_in_direction(pts, cmd, LIDAR_CONE_RAD, LIDAR_Z_MIN, LIDAR_Z_MAX)
            rear = rear_clearance(pts, LIDAR_CONE_RAD, LIDAR_Z_MIN, LIDAR_Z_MAX)
            persons, pt = self._persons(now)
            sprint_ok, self.sprint_reason = sprint_preconditions(
                source, profile, self.deadman.alive(now), self._battery(now), pts is not None,
                self._persons_fresh(now), self.cfg, SPRINT_SOURCES)
            dec = decide(persons, near, self.odom, cmd, now, pt, self.cfg,
                         rear_clear_m=rear, prev=self.prev, min_person_dist_m=min_d,
                         sprint=sprint_ok)
            if profile == "sprint" and not sprint_ok and self.sprint_reason:
                dec.reasons.append(self.sprint_reason)
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

    def _send(self, dec: SafetyDecision, now: Optional[float] = None) -> Tuple[str, int, dict]:
        """Forward a decision: STOP with nothing left -> /stop, else /move.
        A sprint decision adds "sprint": true (mc_motion honours it only with
        MOTION_ALLOW_SPRINT=1); every other body is exactly {vx, vy, vyaw}."""
        vx, vy, vyaw = dec.cmd_out
        if dec.level == STOP and vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            code, resp = _forward("/stop", None)
            self.stops += 1
            self.sprint_active = False
            return "stop", code, resp
        body = {"vx": vx, "vy": vy, "vyaw": vyaw}
        if dec.sprint:
            body["sprint"] = True
        code, resp = _forward("/move", body)
        self.forwarded += 1
        self.sprint_active = bool(dec.sprint)
        if vx != 0.0 or vy != 0.0 or vyaw != 0.0:
            self.last_motion_t = time.time() if now is None else now
        return "move", code, resp

    def _deadman_drop_stop(self, now: float, why: str) -> Tuple[int, dict]:
        """Dead-man released while sprinting: one /stop, latch further
        sprint-profile moves to /stop until the dead-man is held again or the
        source drops the sprint profile."""
        code, resp = _forward("/stop", None)
        self.stops += 1
        self.deadman_drops += 1
        self.sprint_active = False
        self.sprint_latch = True
        self.last_cmd_out = (0.0, 0.0, 0.0)
        self._note("dead-man dropped during sprint (%s): /stop" % why)
        return code, resp

    def move(self, vx: Any, vy: Any, vyaw: Any, source: str = "",
             profile: Optional[str] = None) -> dict:
        cmd = sanitize_cmd((vx, vy, vyaw))
        now = time.time()
        with self.lock:
            dec = self.evaluate(cmd, now, source, profile)
            self.last_cmd_in = cmd
            self.last_caller_t = now
            self.last_source = source or "?"
            self.last_profile = profile
            if profile != "sprint" and self.sprint_latch:
                self.sprint_latch = False
                self._note("sprint latch cleared: %s sent profile %r" % (source or "?", profile))
            if profile == "sprint" and (self.sprint_active or self.sprint_latch) and not self.deadman.alive(now):
                if self.sprint_active:
                    code, resp = self._deadman_drop_stop(now, "move")
                else:                              # latched: keep stopping
                    code, resp = _forward("/stop", None)
                    self.stops += 1
                self.last_cmd_out = (0.0, 0.0, 0.0)
                return {"forwarded": "stop", "motion_status": code, "motion": resp,
                        "decision": dict(dec.to_dict(), reason="dead-man released during sprint")}
            kind, code, resp = self._send(dec, now)
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
            self.sprint_active = False
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
            dec = self.evaluate(cmd, now, self.last_source if live else "",
                                self.last_profile if live else None)
            if self.sprint_active and not self.deadman.alive(now):
                # dead-man dropped while the last forwarded command was a
                # sprint: one /stop at once (caller live or not -- the robot
                # may still be running on it until mc_motion's watchdog).
                self._deadman_drop_stop(now, "tick")
                self.escalations += 1
                return dec
            if live and any(abs(o) > 0.0 for o in self.last_cmd_out):
                new = dec.cmd_out
                if dec.level == STOP and new == (0.0, 0.0, 0.0):
                    smaller = True
                else:
                    smaller = all(abs(n) <= abs(o) + 1e-9 for n, o in zip(new, self.last_cmd_out)) and \
                        any(abs(n) < abs(o) - 1e-6 for n, o in zip(new, self.last_cmd_out))
                if smaller:
                    kind, _, _ = self._send(dec, now)
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
            now = time.time()
            bat = self._battery(now)
            d.update({"deadman_alive": self.deadman.alive(now),
                      "deadman_clients": self.deadman.clients(now),
                      "deadman_timeout_s": DEADMAN_TIMEOUT_S,
                      "deadman_drops": self.deadman_drops,
                      "battery_pct": bat,
                      "battery_age_s": None if self.battery_t is None else round(now - self.battery_t, 3),
                      "sprint_active": self.sprint_active, "sprint_latch": self.sprint_latch,
                      "sprint_reason": self.sprint_reason, "sprint_vmax": self.cfg.sprint_vmax,
                      "last_profile": self.last_profile, "sprint_sources": list(SPRINT_SOURCES),
                      "fleet_aware": FLEET_AWARE, "fleet_robots_n": len(self._fleet(now)),
                      "pose_age_s": None if self.own_pose_t is None else round(now - self.own_pose_t, 3),
                      "still_for_s": round(max(self._still_for(now), 0.0), 3),
                      "actions": self.actions[-10:][::-1]})
            return d

    # -- actions ---------------------------------------------------------------
    def action(self, name: str, source: str = "", now: Optional[float] = None) -> Tuple[int, dict]:
        """Gate POST /action/{name}; forward to MOTION_URL/action/{name} only
        when allowed. Returns (http_status, body); 409 on denial."""
        now = time.time() if now is None else now
        with self.lock:
            pts = self._lidar(now)
            persons, _ = self._persons(now)
            dec = self.evaluate((0.0, 0.0, 0.0), now, "")      # level with the robot still
            front = lidar_near_in_direction(pts, (1.0, 0.0, 0.0), LIDAR_CONE_RAD, LIDAR_Z_MIN, LIDAR_Z_MAX)
            ok, why = action_check(name, self.deadman.alive(now), persons, self._persons_fresh(now),
                                   dec.level, pts is not None, front, self._still_for(now), self.cfg)
            rec = {"t": now, "name": name, "source": source or "?", "decision": "allow" if ok else "deny",
                   "reason": why}
            self.actions.append(rec)
            del self.actions[:-50]
            if not ok:
                log_event("warn", "action %s denied: %s" % (name, why))
                return 409, {"decision": "deny", "reason": why, "action": name, "level": dec.level}
            code, resp = _forward("/action/%s" % name, None)
        log_event("info", "action %s forwarded (%s)" % (name, code))
        out = {"decision": "allow", "reason": why, "action": name, "forwarded": "action",
               "motion_status": code, "motion": resp}
        if code == 0:
            return 502, out
        if code >= 400:
            return code, out
        return 200, out

    def bus_state(self) -> dict:
        """Payload for mc.safety.state (CONTRACTS.md section D)."""
        s = self.last or {}
        return {"t": time.time(), "level": s.get("level", STOP), "vmax": s.get("vmax", 0.0),
                "nearest_person_m": s.get("nearest_person_m"), "reason": s.get("reason", "starting"),
                "vyaw_max": s.get("vyaw_max"), "ttc_s": s.get("ttc_s"),
                "deadman_alive": self.deadman_alive(), "sprint": self.sprint_active}


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


def parse_core_battery(state: Any) -> Optional[float]:
    """core GET /state -> battery percent, or None when unknown: missing /
    non-finite / out of range, or the core link is tracked and unhealthy
    (then every field is a 0-default, not a reading)."""
    if not isinstance(state, dict):
        return None
    link = state.get("link")
    if isinstance(link, dict) and link.get("tracked") and not link.get("healthy"):
        return None
    bat = state.get("battery")
    if not isinstance(bat, dict):
        return None
    try:
        pct = float(bat.get("percent"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pct) or not 0.0 <= pct <= 100.0:
        return None
    return pct


def parse_core_pose(state: Any) -> Optional[Tuple[float, float, float]]:
    """core GET /state pose (world/grid frame) or None (no fix / bad link)."""
    if not isinstance(state, dict):
        return None
    link = state.get("link")
    if isinstance(link, dict) and link.get("tracked") and not link.get("healthy"):
        return None
    pose = state.get("pose")
    if not isinstance(pose, dict):
        return None
    try:
        vals = tuple(float(pose[k]) for k in ("x", "y", "yaw"))
    except (KeyError, TypeError, ValueError):
        return None
    return vals if all(math.isfinite(v) for v in vals) else None


def _core_state_loop() -> None:
    """Battery (sprint gate) + world pose (fleet tracks) from core /state.
    Unknown values are not stored, so the last ones age out."""
    period = 1.0 / max(CORE_POLL_HZ, 0.5)
    while requests is not None:
        t0 = time.time()
        try:
            r = requests.get(CORE_URL + "/state", timeout=HTTP_TIMEOUT_S)
            st = r.json() if r.ok else None
            pct, pose = parse_core_battery(st), parse_core_pose(st)
            if pct is not None:
                guard.set_battery(pct, t0)
            if pose is not None:
                guard.set_pose(pose[0], pose[1], pose[2], t0)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(max(0.0, period - (time.time() - t0)))


def _fleet_redis_loop() -> None:
    while True:
        r = _redis_client()
        if r is None:
            time.sleep(5.0)
            continue
        try:
            ps = r.pubsub()
            ps.subscribe(FLEET_CHANNEL)
            log_event("info", "subscribed to %s" % FLEET_CHANNEL)
            for msg in ps.listen():
                if msg.get("type") == "message":
                    try:
                        guard.ingest_fleet(json.loads(msg["data"]))
                    except (ValueError, TypeError):
                        pass
        except Exception as exc:  # noqa: BLE001
            log_event("warn", "redis fleet loop: %s" % exc)
            time.sleep(1.0)


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
                               (_odom_loop, (), "odom"), (_state_loop, (), "state"),
                               (_core_state_loop, (), "core")):
        threading.Thread(target=target, args=args, daemon=True, name="safety-" + name).start()
    if FLEET_AWARE:
        threading.Thread(target=_fleet_redis_loop, daemon=True, name="safety-fleet").start()


# ---------------------------------------------------------------------------
# HTTP API

app = FastAPI(title="safety_guard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MoveBody(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    source: str = ""
    profile: Optional[str] = None      # stealth|precise|normal|sprint (only "sprint" changes anything)


class DeadmanBody(BaseModel):
    client_id: str = ""
    release: bool = False


class ActionBody(BaseModel):
    source: str = ""


class StopBody(BaseModel):
    source: str = ""


@app.post("/move")
def move(body: MoveBody):
    res = guard.move(body.vx, body.vy, body.vyaw, body.source, body.profile)
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


@app.post("/deadman")
def deadman(body: DeadmanBody):
    """Heartbeat (>= 5 Hz, UI sends 10 Hz). release=true drops this client at once."""
    if body.release:
        with guard.lock:
            guard.deadman.release(body.client_id)
    elif not guard.deadman_beat(body.client_id):
        raise HTTPException(status_code=422, detail="client_id required (or too many clients)")
    now = time.time()
    with guard.lock:
        return {"deadman_alive": guard.deadman.alive(now), "deadman_clients": guard.deadman.clients(now),
                "timeout_s": DEADMAN_TIMEOUT_S}


@app.post("/action/{name}")
def action(name: str, body: Optional[ActionBody] = None):
    code, res = guard.action(name, body.source if body else "")
    return JSONResponse(status_code=code, content=res)


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
