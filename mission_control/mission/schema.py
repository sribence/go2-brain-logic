"""Mission schema (M1): dataclasses + validation for the mission pillar.

Single source of truth for the mission JSON (CONTRACT.md section 2). Every
producer (UI, gamepad, NL/LLM, rule engine, API) builds this JSON and calls
``validate()``; the executor only ever runs a validated ``Mission``.

    {"mission_id": "optional", "name": "...", "source": "ui|gamepad|nl|rule|api",
     "steps": [ <step>, ... ], "on_fail": "stop|skip|return_home"}

Rules enforced here (beyond types / ranges):
  * speed is a profile NAME (stealth|precise|normal|sprint), never m/s;
  * shadow / escort keep >= 2.5 m from the person (MIN_PERSON_DIST_M);
  * source "nl" -> initial state ``pending_approval`` (LLM output never runs
    without a human approving it), every other source -> ``queued``;
  * source "rule" may not contain ``sprint`` or ``jump_to`` (both need a live
    dead-man held by a human operator).

Pure module: no I/O, no third-party imports. Python 3.8 compatible.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

SPEEDS = ("stealth", "precise", "normal", "sprint")
SOURCES = ("ui", "gamepad", "nl", "rule", "api")
ON_FAIL = ("stop", "skip", "return_home")
SIDES = ("left", "right")
# CONTRACT 9.1 blend zones: "fine" = exact stop (<= 0.05 m, <= 3 deg, settle
# 0.5 s); "zNN" = fly-by, waypoint reached within NN cm, no slowdown.
BLEND_ZONES = ("fine", "z10", "z30", "z60", "z100")
DEFAULT_ZONE = "z30"
FINAL_ZONE = "fine"
CAPTURE_MODES = ("max", "normal")
ON_TIMEOUT = ("alert", "return", "abort")
WAIT_EVENTS = ("path_clear", "operator", "gesture", "person_gone")
RECORD_MODES = ("evidence",)

# Actions an `action` step may request through safety_guard /action/{name}.
# Jumps/pounces are deliberately NOT here: they only run via `jump_to`
# (landing points checked by the planner). Flips are never allowed.
ACTIONS = ("hello", "wave", "greet", "stretch", "sit", "stand", "stand_up", "lay_down",
           "balance", "heart", "dance", "dance2", "shake")

STATES = ("pending_approval", "queued", "running", "paused", "done", "failed", "aborted")
TERMINAL_STATES = ("done", "failed", "aborted")

MIN_PERSON_DIST_M = 2.5
COORD_LIMIT_M = 1000.0
MAX_STEPS = 200
MAX_POINTS = 2000
MAX_TEXT = 500
META_KEYS = ("id", "note")          # free-form per-step keys tolerated (UI bookkeeping)


class ValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


# ---------------------------------------------------------------- step types

@dataclass
class Step:
    op = ""                          # class attribute, overridden per subclass

    def to_dict(self) -> Dict[str, Any]:
        d = {"op": self.op}
        for f in fields(self):
            d[f.name] = _plain(getattr(self, f.name))
        return d


@dataclass
class GotoStep(Step):
    op = "goto"
    x: float = 0.0
    y: float = 0.0
    yaw: Optional[float] = None
    speed: str = "normal"
    tol_m: float = 0.3
    zone: Optional[str] = None       # BLEND_ZONES; None -> filled by validate()


@dataclass
class FollowPathStep(Step):
    op = "follow_path"
    points: List[List[float]] = field(default_factory=list)
    speed: str = "normal"
    smooth: bool = True
    zone: Optional[str] = None       # zone of the LAST point
    zones: Optional[List[str]] = None  # optional per-point zones (len == len(points))


@dataclass
class JumpToStep(Step):
    op = "jump_to"
    x: float = 0.0
    y: float = 0.0
    max_jumps: int = 6


@dataclass
class LookAtStep(Step):
    op = "look_at"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ScanStep(Step):
    op = "scan"
    deg: float = 360.0
    speed_dps: float = 30.0


@dataclass
class ActionStep(Step):
    op = "action"
    name: str = ""


@dataclass
class WaitStep(Step):
    op = "wait"
    s: float = 0.0


@dataclass
class SayStep(Step):
    op = "say"
    text: str = ""


@dataclass
class ShadowStep(Step):
    op = "shadow"
    gid: int = 0
    dist_m: float = 3.0
    timeout_s: Optional[float] = None


@dataclass
class WatchStep(Step):
    op = "watch"
    gid: int = 0
    timeout_s: Optional[float] = None


@dataclass
class EscortStep(Step):
    op = "escort"
    gid: int = 0
    side: str = "right"
    dist_m: float = 2.5
    timeout_s: Optional[float] = None


@dataclass
class PatrolStep(Step):
    op = "patrol"
    points: Optional[List[List[float]]] = None
    zone: Any = None                 # area: zone id/name (str) or inline polygon [[x,y],...]
    loops: int = 1
    shuffle: bool = False
    speed: str = "normal"
    pass_zone: Optional[str] = None  # blend zone of every patrol point (CONTRACT 9.1)


@dataclass
class ExploreStep(Step):
    op = "explore"
    zone: Any = None
    timeout_s: float = 600.0


@dataclass
class GotoLabelStep(Step):
    op = "goto_label"
    label: str = ""
    speed: str = "normal"
    zone: Optional[str] = None


@dataclass
class ReturnHomeStep(Step):
    op = "return_home"
    speed: str = "normal"
    zone: Optional[str] = None


@dataclass
class CaptureStep(Step):
    op = "capture"
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
    cams: Any = "all"                # list of cam ids or "all"
    mode: str = "max"
    label: Optional[str] = None


@dataclass
class RequestPassageStep(Step):
    op = "request_passage"
    text: str = ""
    repeat_s: float = 30.0
    zone_polygon: Optional[List[List[float]]] = None
    ahead_m: float = 1.5
    clear_for_s: float = 2.0
    timeout_s: float = 600.0
    on_timeout: str = "alert"


@dataclass
class WaitForStep(Step):
    op = "wait_for"
    event: str = "operator"
    timeout_s: float = 300.0
    gid: Optional[int] = None        # person_gone: this gid (None = nobody visible)
    zone_polygon: Optional[List[List[float]]] = None   # path_clear
    ahead_m: float = 1.5             # path_clear
    clear_for_s: float = 2.0         # path_clear / person_gone


@dataclass
class RecordStep(Step):
    op = "record"
    on: bool = True
    mode: str = "evidence"


STEP_TYPES = {cls.op: cls for cls in (
    GotoStep, FollowPathStep, JumpToStep, LookAtStep, ScanStep, ActionStep, WaitStep, SayStep,
    ShadowStep, WatchStep, EscortStep, PatrolStep, ExploreStep, GotoLabelStep, ReturnHomeStep,
    CaptureStep, RequestPassageStep, WaitForStep, RecordStep)}
OPS = tuple(STEP_TYPES)
MOTION_OPS = ("goto", "follow_path", "jump_to", "look_at", "scan", "action", "shadow", "watch",
              "escort", "patrol", "explore", "goto_label", "return_home", "capture")
# steps that end at a waypoint and carry a blend zone (CONTRACT 9.1)
ZONED_OPS = ("goto", "follow_path", "patrol", "goto_label", "return_home")


def zone_radius_m(zone: Optional[str]) -> float:
    """"fine" -> 0.05 m, "z30" -> 0.30 m."""
    if not zone or zone == "fine":
        return 0.05
    return int(zone[1:]) / 100.0


def step_zone(step: "Step") -> Optional[str]:
    return getattr(step, "pass_zone" if step.op == "patrol" else "zone", None)


@dataclass
class Mission:
    steps: List[Step] = field(default_factory=list)
    mission_id: str = ""
    name: str = ""
    source: str = "api"
    on_fail: str = "stop"
    state: str = "queued"            # initial state derived from source

    def to_dict(self) -> Dict[str, Any]:
        return {"mission_id": self.mission_id, "name": self.name, "source": self.source,
                "on_fail": self.on_fail, "state": self.state,
                "steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Mission":
        """Validating constructor; raises ValidationError."""
        m, errs = validate(d)
        if errs:
            raise ValidationError(errs)
        return m

    def uses_sprint(self) -> bool:
        return any(getattr(s, "speed", None) == "sprint" for s in self.steps)

    def needs_deadman(self) -> bool:
        return self.uses_sprint() or any(s.op == "jump_to" for s in self.steps)


def new_mission_id() -> str:
    return "m-" + uuid.uuid4().hex[:10]


def initial_state(source: str) -> str:
    return "pending_approval" if source == "nl" else "queued"


# ---------------------------------------------------------------- validation

def _plain(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


class _V:
    """Collects errors with a path prefix (``steps[2].x``)."""

    def __init__(self, errors: List[str], where: str):
        self.errors, self.where = errors, where

    def err(self, key: str, msg: str) -> None:
        self.errors.append("%s.%s: %s" % (self.where, key, msg) if key else "%s: %s" % (self.where, msg))

    def num(self, d: Dict[str, Any], key: str, default: Any = None, lo: Optional[float] = None,
            hi: Optional[float] = None, required: bool = False, lo_open: bool = False,
            integer: bool = False, allow_none: bool = False) -> Any:
        if key not in d or (d[key] is None and allow_none):
            if required:
                self.err(key, "required")
                return None
            return default
        v = d[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.err(key, "must be a number, got %s" % type(v).__name__)
            return None
        if integer:
            if isinstance(v, float) and not v.is_integer():
                self.err(key, "must be an integer, got %r" % v)
                return None
            v = int(v)
        else:
            v = float(v)
        if not math.isfinite(v):
            self.err(key, "must be finite")
            return None
        if lo is not None and (v < lo or (lo_open and v == lo)):
            self.err(key, "must be %s %g, got %g" % (">" if lo_open else ">=", lo, v))
            return None
        if hi is not None and v > hi:
            self.err(key, "must be <= %g, got %g" % (hi, v))
            return None
        return v

    def coord(self, d: Dict[str, Any], key: str, required: bool = True, default: Any = 0.0) -> Any:
        return self.num(d, key, default=default, lo=-COORD_LIMIT_M, hi=COORD_LIMIT_M, required=required)

    def choice(self, d: Dict[str, Any], key: str, options: Tuple[str, ...], default: str) -> Any:
        v = d.get(key, default)
        if v is None:
            v = default
        if not isinstance(v, str) or v not in options:
            self.err(key, "must be one of %s, got %r" % ("|".join(options), v))
            return None
        return v

    def boolean(self, d: Dict[str, Any], key: str, default: bool) -> Any:
        v = d.get(key, default)
        if not isinstance(v, bool):
            self.err(key, "must be true/false, got %r" % (v,))
            return None
        return v

    def text(self, d: Dict[str, Any], key: str, max_len: int = MAX_TEXT) -> Any:
        v = d.get(key)
        if not isinstance(v, str) or not v.strip():
            self.err(key, "required non-empty string")
            return None
        if len(v) > max_len:
            self.err(key, "too long (%d > %d chars)" % (len(v), max_len))
            return None
        return v.strip()

    def zone(self, d: Dict[str, Any], key: str) -> Any:
        z = d.get(key)
        if z is None:
            return None
        if z not in BLEND_ZONES:
            self.err(key, "must be one of %s, got %r" % ("|".join(BLEND_ZONES), z))
            return None
        return z

    def gid(self, d: Dict[str, Any]) -> Any:
        return self.num(d, "gid", required=True, lo=0, integer=True)

    def points(self, d: Dict[str, Any], key: str, min_n: int) -> Any:
        v = d.get(key)
        if not isinstance(v, (list, tuple)):
            self.err(key, "must be a list of [x, y] points")
            return None
        if len(v) < min_n:
            self.err(key, "needs at least %d points, got %d" % (min_n, len(v)))
            return None
        if len(v) > MAX_POINTS:
            self.err(key, "too many points (%d > %d)" % (len(v), MAX_POINTS))
            return None
        out = []
        for i, p in enumerate(v):
            if not isinstance(p, (list, tuple)) or len(p) < 2 or any(
                    isinstance(c, bool) or not isinstance(c, (int, float)) for c in p[:2]):
                self.err("%s[%d]" % (key, i), "must be [x, y] numbers, got %r" % (p,))
                return None
            x, y = float(p[0]), float(p[1])
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > COORD_LIMIT_M or abs(y) > COORD_LIMIT_M:
                self.err("%s[%d]" % (key, i), "out of range / not finite: %r" % (p,))
                return None
            out.append([x, y])
        return out


def polygon_errors(poly: List[List[float]]) -> List[str]:
    """>= 3 vertices, non-zero area, no self-intersection (closing vertex optional)."""
    pts = [tuple(p) for p in poly]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return ["polygon needs >= 3 distinct vertices"]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    if abs(area) / 2.0 < 0.05:
        return ["polygon area too small (%.3f m^2 < 0.05)" % (abs(area) / 2.0)]
    n = len(pts)
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue                  # adjacent edges share a vertex
            b1, b2 = pts[j], pts[(j + 1) % n]
            if _seg_cross(a1, a2, b1, b2):
                return ["polygon is self-intersecting (edges %d and %d)" % (i, j)]
    return []


def _seg_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4)


def validate_step(d: Any, where: str = "step", source: str = "api") -> Tuple[Optional[Step], List[str]]:
    """Validate one step dict -> (Step | None, errors)."""
    errors: List[str] = []
    v = _V(errors, where)
    if not isinstance(d, dict):
        v.err("", "must be an object, got %s" % type(d).__name__)
        return None, errors
    op = d.get("op")
    cls = STEP_TYPES.get(op) if isinstance(op, str) else None
    if cls is None:
        v.err("op", "unknown op %r (allowed: %s)" % (op, ", ".join(OPS)))
        return None, errors
    allowed = {f.name for f in fields(cls)} | {"op"} | set(META_KEYS)
    for k in d:
        if k not in allowed:
            v.err(k, "unknown field for op %r (allowed: %s)" % (op, ", ".join(sorted(allowed - {"op"}))))
    kw: Dict[str, Any] = {}

    if op == "goto":
        kw = dict(x=v.coord(d, "x"), y=v.coord(d, "y"),
                  yaw=v.num(d, "yaw", None, -4 * math.pi, 4 * math.pi, allow_none=True),
                  speed=v.choice(d, "speed", SPEEDS, "normal"),
                  tol_m=v.num(d, "tol_m", 0.3, 0.05, 5.0), zone=v.zone(d, "zone"))
    elif op == "follow_path":
        pts = v.points(d, "points", 2)
        zones = d.get("zones")
        if zones is not None:
            if not isinstance(zones, list) or any(z not in BLEND_ZONES for z in zones):
                v.err("zones", "must be a list of %s" % "|".join(BLEND_ZONES))
                zones = None
            elif pts is not None and len(zones) != len(pts):
                v.err("zones", "length %d != number of points %d" % (len(zones), len(pts)))
                zones = None
        kw = dict(points=pts, speed=v.choice(d, "speed", SPEEDS, "normal"),
                  smooth=v.boolean(d, "smooth", True), zone=v.zone(d, "zone"), zones=zones)
    elif op == "jump_to":
        kw = dict(x=v.coord(d, "x"), y=v.coord(d, "y"),
                  max_jumps=v.num(d, "max_jumps", 6, 1, 10, integer=True))
        if source == "rule":
            v.err("", "jump_to is not allowed in a rule-generated mission (needs a live dead-man)")
    elif op == "look_at":
        kw = dict(x=v.coord(d, "x"), y=v.coord(d, "y"), z=v.num(d, "z", 0.0, -5.0, 10.0))
    elif op == "scan":
        kw = dict(deg=v.num(d, "deg", 360.0, 0.0, 1080.0, lo_open=True),
                  speed_dps=v.num(d, "speed_dps", 30.0, 5.0, 90.0))
    elif op == "action":
        name = d.get("name")
        if not isinstance(name, str) or name not in ACTIONS:
            v.err("name", "must be one of %s, got %r" % ("|".join(ACTIONS), name))
            name = None
        kw = dict(name=name)
    elif op == "wait":
        kw = dict(s=v.num(d, "s", required=True, lo=0.0, hi=3600.0))
    elif op == "say":
        kw = dict(text=v.text(d, "text"))
    elif op == "shadow":
        kw = dict(gid=v.gid(d), dist_m=v.num(d, "dist_m", 3.0, MIN_PERSON_DIST_M, 15.0),
                  timeout_s=v.num(d, "timeout_s", None, 1.0, 7200.0, allow_none=True))
    elif op == "watch":
        kw = dict(gid=v.gid(d), timeout_s=v.num(d, "timeout_s", None, 1.0, 7200.0, allow_none=True))
    elif op == "escort":
        kw = dict(gid=v.gid(d), side=v.choice(d, "side", SIDES, "right"),
                  dist_m=v.num(d, "dist_m", 2.5, MIN_PERSON_DIST_M, 10.0),
                  timeout_s=v.num(d, "timeout_s", None, 1.0, 7200.0, allow_none=True))
    elif op == "patrol":
        pts = d.get("points")
        zone = d.get("zone")
        pass_zone = v.zone(d, "pass_zone")
        if isinstance(zone, str) and zone in BLEND_ZONES and pts is not None:
            # CONTRACT 9.1 "zone" on a points patrol = blend zone, not an area
            pass_zone, zone = zone, None
        if (pts is None) == (zone is None):
            v.err("", "exactly one of 'points' or 'zone' is required")
        if pts is not None:
            pts = v.points(d, "points", 1)
        if zone is not None:
            if isinstance(zone, str):
                if not zone.strip():
                    v.err("zone", "must be a non-empty zone id/name or a polygon")
                    zone = None
            elif isinstance(zone, (list, tuple)):
                zone = v.points(d, "zone", 3)
                if zone is not None:
                    for e in polygon_errors(zone):
                        v.err("zone", e)
            else:
                v.err("zone", "must be a zone id/name or a polygon [[x,y],...]")
                zone = None
        kw = dict(points=pts, zone=zone, loops=v.num(d, "loops", 1, 1, 1000, integer=True),
                  shuffle=v.boolean(d, "shuffle", False), speed=v.choice(d, "speed", SPEEDS, "normal"),
                  pass_zone=pass_zone)
    elif op == "explore":
        zone = d.get("zone")
        if zone is not None and not isinstance(zone, str):
            if isinstance(zone, (list, tuple)):
                zone = v.points(d, "zone", 3)
                if zone is not None:
                    for e in polygon_errors(zone):
                        v.err("zone", e)
            else:
                v.err("zone", "must be a zone id/name or a polygon")
        kw = dict(zone=zone, timeout_s=v.num(d, "timeout_s", 600.0, 1.0, 7200.0))
    elif op == "goto_label":
        kw = dict(label=v.text(d, "label", 100), speed=v.choice(d, "speed", SPEEDS, "normal"),
                  zone=v.zone(d, "zone"))
    elif op == "return_home":
        kw = dict(speed=v.choice(d, "speed", SPEEDS, "normal"), zone=v.zone(d, "zone"))
    elif op == "capture":
        has_x, has_y = d.get("x") is not None, d.get("y") is not None
        if has_x != has_y:
            v.err("", "x and y must be given together (or neither)")
        cams = d.get("cams", "all")
        if cams != "all" and not (isinstance(cams, list) and cams and len(cams) <= 16 and all(
                isinstance(c, str) and c.strip() for c in cams)):
            v.err("cams", "must be \"all\" or a non-empty list of camera ids")
            cams = None
        label = d.get("label")
        if label is not None and (not isinstance(label, str) or len(label) > 100):
            v.err("label", "must be a string <= 100 chars")
        kw = dict(x=v.coord(d, "x", required=False, default=None) if has_x else None,
                  y=v.coord(d, "y", required=False, default=None) if has_y else None,
                  yaw=v.num(d, "yaw", None, -4 * math.pi, 4 * math.pi, allow_none=True),
                  cams=cams, mode=v.choice(d, "mode", CAPTURE_MODES, "max"), label=label)
    elif op == "request_passage":
        zp = None
        if d.get("zone_polygon") is not None:
            zp = v.points(d, "zone_polygon", 3)
            if zp is not None:
                for e in polygon_errors(zp):
                    v.err("zone_polygon", e)
        kw = dict(text=v.text(d, "text"), repeat_s=v.num(d, "repeat_s", 30.0, 3.0, 3600.0),
                  zone_polygon=zp, ahead_m=v.num(d, "ahead_m", 1.5, 0.3, 10.0),
                  clear_for_s=v.num(d, "clear_for_s", 2.0, 0.0, 120.0),
                  timeout_s=v.num(d, "timeout_s", 600.0, 1.0, 7200.0),
                  on_timeout=v.choice(d, "on_timeout", ON_TIMEOUT, "alert"))
    elif op == "wait_for":
        zp = None
        if d.get("zone_polygon") is not None:
            zp = v.points(d, "zone_polygon", 3)
            if zp is not None:
                for e in polygon_errors(zp):
                    v.err("zone_polygon", e)
        kw = dict(event=v.choice(d, "event", WAIT_EVENTS, "operator"),
                  timeout_s=v.num(d, "timeout_s", 300.0, 1.0, 7200.0),
                  gid=v.num(d, "gid", None, lo=0, integer=True, allow_none=True),
                  zone_polygon=zp, ahead_m=v.num(d, "ahead_m", 1.5, 0.3, 10.0),
                  clear_for_s=v.num(d, "clear_for_s", 2.0, 0.0, 120.0))
        if "event" not in d:
            v.err("event", "required (%s)" % "|".join(WAIT_EVENTS))
    elif op == "record":
        kw = dict(on=v.boolean(d, "on", True), mode=v.choice(d, "mode", RECORD_MODES, "evidence"))

    if source == "rule" and kw.get("speed") == "sprint":
        v.err("speed", "sprint is not allowed in a rule-generated mission (needs a live dead-man)")
    if errors:
        return None, errors
    return cls(**kw), errors


def validate(d: Any) -> Tuple[Optional[Mission], List[str]]:
    """Mission dict -> (Mission, []) or (None, [errors]). Error strings are
    precise (``steps[1].dist_m: must be >= 2.5, got 1``) so an LLM or a UI
    can show / self-correct them."""
    errors: List[str] = []
    if not isinstance(d, dict):
        return None, ["mission: must be an object, got %s" % type(d).__name__]
    v = _V(errors, "mission")
    for k in d:
        if k not in ("mission_id", "name", "source", "steps", "on_fail", "state"):
            v.err(k, "unknown field")
    source = v.choice(d, "source", SOURCES, "api")
    on_fail = v.choice(d, "on_fail", ON_FAIL, "stop")
    mid = d.get("mission_id")
    if mid is not None and (not isinstance(mid, str) or not (0 < len(mid) <= 64)):
        v.err("mission_id", "must be a string of 1..64 chars")
    name = d.get("name", "")
    if name is None:
        name = ""
    if not isinstance(name, str) or len(name) > 200:
        v.err("name", "must be a string <= 200 chars")
        name = ""
    steps_in = d.get("steps")
    steps: List[Step] = []
    if not isinstance(steps_in, list) or not steps_in:
        v.err("steps", "must be a non-empty list")
    elif len(steps_in) > MAX_STEPS:
        v.err("steps", "too many steps (%d > %d)" % (len(steps_in), MAX_STEPS))
    else:
        for i, sd in enumerate(steps_in):
            st, errs = validate_step(sd, "steps[%d]" % i, source or "api")
            errors.extend(errs)
            if st is not None:
                steps.append(st)
    if errors:
        return None, errors
    fill_default_zones(steps)
    m = Mission(steps=steps, mission_id=mid or new_mission_id(), name=name, source=source,
                on_fail=on_fail, state=initial_state(source))
    return m, []


def fill_default_zones(steps: List[Step]) -> None:
    """CONTRACT 9.1: unset blend zone -> "z30", except the mission's last
    waypoint step -> "fine" (the robot stops exactly at the end)."""
    last = None
    for i, st in enumerate(steps):
        if st.op in ZONED_OPS or (st.op == "capture" and st.x is not None):
            last = i
    for i, st in enumerate(steps):
        if st.op not in ZONED_OPS:
            continue
        attr = "pass_zone" if st.op == "patrol" else "zone"
        if getattr(st, attr) is None:
            setattr(st, attr, FINAL_ZONE if i == last else DEFAULT_ZONE)


def step_from_dict(d: Dict[str, Any], source: str = "api") -> Step:
    st, errs = validate_step(d, "step", source)
    if errs:
        raise ValidationError(errs)
    return st
