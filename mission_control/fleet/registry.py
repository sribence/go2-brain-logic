"""Fleet robot registry + 1 Hz poller (CONTRACT.md 9.6).

Frames
------
Every robot runs its own mapping grid-frame ("robot frame"). The fleet keeps
one common ``map`` frame and a per-robot rigid offset ``T_map_robot``
(``Align{dx, dy, dyaw}``, manual alignment): p_map = R(dyaw) p_robot + (dx, dy).

Transport
---------
All HTTP goes through a ``Transport`` (``get/post/delete -> (status, data)``),
so tests inject a fake one and nothing touches the network.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

ROBOT_TYPES = ("go2", "xavier", "other")
SERVICES = ("core", "omni", "safety", "mission", "twin")
DEFAULT_PORTS = {"core": 9101, "twin": 9110, "omni": 9114, "safety": 9115, "mission": 9116}
PALETTE = ["#00d8ff", "#ff2bd6", "#35f28a", "#ffb020", "#a77bff", "#ff6b4a", "#4af0d0", "#f2f25c"]

OFFLINE_S = float(os.environ.get("FLEET_OFFLINE_S", "3.0"))
HTTP_TIMEOUT_S = float(os.environ.get("FLEET_HTTP_TIMEOUT_S", "0.6"))


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Align:
    """T_map_robot as (dx, dy, dyaw)."""

    __slots__ = ("dx", "dy", "dyaw")

    def __init__(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0):
        self.dx, self.dy, self.dyaw = float(dx), float(dy), float(dyaw)

    @classmethod
    def from_dict(cls, d: Any) -> "Align":
        d = d or {}
        return cls(d.get("dx", 0.0), d.get("dy", 0.0), d.get("dyaw", 0.0))

    def to_dict(self) -> Dict[str, float]:
        return {"dx": round(self.dx, 4), "dy": round(self.dy, 4), "dyaw": round(self.dyaw, 5)}

    def key(self) -> str:
        return "%.4f,%.4f,%.5f" % (self.dx, self.dy, self.dyaw)

    # robot -> map
    def to_map(self, x: float, y: float) -> Tuple[float, float]:
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        return self.dx + c * x - s * y, self.dy + s * x + c * y

    # map -> robot
    def to_robot(self, x: float, y: float) -> Tuple[float, float]:
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        ux, uy = x - self.dx, y - self.dy
        return c * ux + s * uy, -s * ux + c * uy

    def yaw_to_map(self, yaw: float) -> float:
        return wrap(yaw + self.dyaw)

    def yaw_to_robot(self, yaw: float) -> float:
        return wrap(yaw - self.dyaw)

    def vec_to_map(self, vx: float, vy: float) -> Tuple[float, float]:
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        return c * vx - s * vy, s * vx + c * vy

    def pose_to_map(self, pose: Dict[str, Any]) -> Dict[str, float]:
        x, y = self.to_map(float(pose["x"]), float(pose["y"]))
        return {"x": x, "y": y, "yaw": self.yaw_to_map(float(pose.get("yaw", 0.0) or 0.0))}


def poly_transform(poly: List[List[float]], fn: Callable[[float, float], Tuple[float, float]]) -> List[List[float]]:
    return [[round(v, 4) for v in fn(float(p[0]), float(p[1]))] for p in poly]


def _point_keys(step: Dict[str, Any]) -> bool:
    return isinstance(step.get("x"), (int, float)) and isinstance(step.get("y"), (int, float))


def transform_mission(mission: Dict[str, Any], align: Align, to_robot: bool = True) -> Dict[str, Any]:
    """Deep-copied mission with every coordinate moved between map and robot
    frame: step x/y/yaw, ``points`` (follow_path/patrol), ``zone_polygon``.
    Robot-local fields (gid, label names, zone names) are left untouched."""
    fn = align.to_robot if to_robot else align.to_map
    yfn = align.yaw_to_robot if to_robot else align.yaw_to_map
    m = copy.deepcopy(mission)
    for st in m.get("steps") or []:
        if not isinstance(st, dict):
            continue
        if _point_keys(st):
            st["x"], st["y"] = [round(v, 4) for v in fn(float(st["x"]), float(st["y"]))]
            if isinstance(st.get("yaw"), (int, float)):
                st["yaw"] = round(yfn(float(st["yaw"])), 5)
        for k in ("points", "zone_polygon"):
            pts = st.get(k)
            if isinstance(pts, list) and pts and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in pts):
                st[k] = poly_transform(pts, fn)
    return m


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
class Transport:
    """Default HTTP transport (requests). Returns (status, json|None); status 0 = no connection."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_S, token: str = ""):
        import requests  # lazy: tests use a fake transport
        self._requests = requests
        self.timeout = timeout
        self._local = threading.local()
        self.headers = {"X-MC-Token": token} if token else {}

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self._requests.Session()
            s.headers.update(self.headers)
        return s

    def _do(self, method: str, url: str, body: Any = None, timeout: Optional[float] = None) -> Tuple[int, Any]:
        try:
            r = self._session().request(method, url, json=body, timeout=timeout or self.timeout)
            try:
                data = r.json()
            except ValueError:
                data = None
            return r.status_code, data
        except Exception:  # noqa: BLE001 - every network error == offline
            return 0, None

    def get(self, url: str, timeout: Optional[float] = None) -> Tuple[int, Any]:
        return self._do("GET", url, None, timeout)

    def post(self, url: str, body: Any = None, timeout: Optional[float] = None) -> Tuple[int, Any]:
        return self._do("POST", url, body if body is not None else {}, timeout)

    def delete(self, url: str, timeout: Optional[float] = None) -> Tuple[int, Any]:
        return self._do("DELETE", url, None, timeout)


def ok(status: int) -> bool:
    return 200 <= int(status or 0) < 300


# ---------------------------------------------------------------------------
# robots
# ---------------------------------------------------------------------------
def _norm_url(u: str) -> str:
    u = str(u).strip().rstrip("/")
    return u if "://" in u else "http://" + u


def build_urls(spec: Dict[str, Any]) -> Dict[str, str]:
    """``urls`` dict wins; else ``host`` + default ports. Missing services -> absent."""
    urls = {}
    given = spec.get("urls") or {}
    host = spec.get("host")
    for svc in SERVICES:
        if given.get(svc):
            urls[svc] = _norm_url(given[svc])
        elif host:
            urls[svc] = _norm_url("%s:%d" % (host, DEFAULT_PORTS[svc]))
    return urls


class RobotSpec:
    def __init__(self, d: Dict[str, Any], idx: int = 0):
        rid = str(d.get("id") or "").strip()
        if not rid or "/" in rid:
            raise ValueError("robot.id required (no '/')")
        self.id = rid
        self.name = str(d.get("name") or rid)
        self.type = str(d.get("type") or "go2")
        self.color = str(d.get("color") or PALETTE[idx % len(PALETTE)])
        self.host = d.get("host")
        self.urls = build_urls(d)
        if "core" not in self.urls:
            raise ValueError("robot %s: core url (or host) required" % rid)
        self.align = Align.from_dict(d.get("align"))
        self.redis = d.get("redis") or None
        self.home = d.get("home") or None          # per-robot home override (robot frame)
        self.enabled = bool(d.get("enabled", True))

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "name": self.name, "type": self.type, "color": self.color,
             "urls": dict(self.urls), "align": self.align.to_dict(), "enabled": self.enabled}
        if self.host:
            d["host"] = self.host
        if self.redis:
            d["redis"] = self.redis
        if self.home:
            d["home"] = self.home
        return d


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _list_of(data: Any, key: str) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get(key), list):
        return data[key]
    return []


ACTIVE_STATES = ("pending_approval", "queued", "running", "paused", "waiting")


def active_mission(missions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Most relevant non-terminal mission (running > paused/waiting > queued > pending)."""
    rank = {"running": 0, "waiting": 1, "paused": 2, "queued": 3, "pending_approval": 4}
    act = [m for m in missions if isinstance(m, dict) and m.get("state") in rank]
    if not act:
        return None
    act.sort(key=lambda m: (rank[m["state"]], -(_num(m.get("updated_at") or m.get("t")) or 0.0)))
    return act[0]


def _mission_path(m: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if not m:
        return None
    plan = m.get("plan") if isinstance(m.get("plan"), dict) else {}
    for src in (plan, m):
        for k in ("path", "points", "global_path"):
            v = src.get(k)
            if isinstance(v, dict):
                v = v.get("points")
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                return [[float(p[0]), float(p[1])] for p in v[:400]]
    return None


class RobotStatus:
    def __init__(self) -> None:
        self.last_ok: Optional[float] = None      # last successful core /state
        self.links: Dict[str, str] = {s: "unknown" for s in SERVICES}
        self.pose: Optional[Dict[str, float]] = None   # robot frame
        self.pose_t: Optional[float] = None
        self.prev_map: Optional[Tuple[float, float, float]] = None   # (t, x, y) map
        self.vel = (0.0, 0.0)                     # map frame
        self.battery: Optional[float] = None
        self.armed: Optional[bool] = None
        self.safety_level: Optional[str] = None
        self.safety_vmax: Optional[float] = None
        self.mission: Optional[Dict[str, Any]] = None
        self.mission_states: Dict[str, str] = {}  # mission_id -> state (all listed)
        self.path_robot: Optional[List[List[float]]] = None
        self.persons: List[Dict[str, Any]] = []   # base frame
        self.omni_ok: Optional[bool] = None
        self.record: Optional[Dict[str, Any]] = None
        self.t_poll: Optional[float] = None


class Robot:
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.st = RobotStatus()
        self.lock = threading.Lock()

    @property
    def id(self) -> str:
        return self.spec.id

    def online(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return self.st.last_ok is not None and now - self.st.last_ok <= OFFLINE_S

    def pose_map(self) -> Optional[Dict[str, float]]:
        if self.st.pose is None:
            return None
        return self.spec.align.pose_to_map(self.st.pose)

    def summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        a = self.spec.align
        st = self.st
        pm = self.pose_map()
        persons = []
        if st.pose is not None:
            px, py, pyaw = float(st.pose["x"]), float(st.pose["y"]), float(st.pose.get("yaw", 0.0) or 0.0)
            c, s = math.cos(pyaw), math.sin(pyaw)
            for p in st.persons[:50]:
                bx, by = _num(p.get("x")), _num(p.get("y"))
                if bx is None or by is None:
                    continue
                mx, my = a.to_map(px + c * bx - s * by, py + s * bx + c * by)
                persons.append({"gid": p.get("gid"), "x": round(mx, 3), "y": round(my, 3),
                                "modality": p.get("modality") or [], "conf": p.get("conf")})
        m = st.mission
        mission = None
        if m:
            mission = {k: m.get(k) for k in ("mission_id", "name", "state", "step_index", "step_op",
                                             "progress", "eta_s", "error", "source") if k in m}
        rec = st.record or {}
        return {
            "id": self.id, "name": self.spec.name, "type": self.spec.type, "color": self.spec.color,
            "online": self.online(now), "enabled": self.spec.enabled,
            "age_s": None if st.last_ok is None else round(now - st.last_ok, 2),
            "links": dict(st.links),
            "pose": None if pm is None else {k: round(v, 4) for k, v in pm.items()},
            "pose_robot": None if st.pose is None else {k: round(float(st.pose.get(k, 0.0) or 0.0), 4) for k in ("x", "y", "yaw")},
            "vel": [round(st.vel[0], 3), round(st.vel[1], 3)],
            "battery": st.battery, "armed": st.armed,
            "safety_level": st.safety_level, "safety_vmax": st.safety_vmax,
            "mission": mission,
            "path": None if not st.path_robot else poly_transform(st.path_robot, a.to_map),
            "persons": persons, "persons_n": len(st.persons),
            "record": {"state": rec.get("state"), "incident_id": rec.get("incident_id")} if rec else None,
            "recording": bool(rec.get("recording") or rec.get("state") in ("evidence", "recording", "on")),
            "align": a.to_dict(), "urls": dict(self.spec.urls),
        }


def poll_robot(robot: Robot, transport: Any, now: Optional[float] = None) -> None:
    """One poll round for one robot (sequential, each call bounded by the
    transport timeout). Every source is optional except core for online."""
    now = time.time() if now is None else now
    u = robot.spec.urls
    st = robot.st
    with robot.lock:
        st.t_poll = now
    code, d = transport.get(u["core"] + "/state")
    with robot.lock:
        if ok(code) and isinstance(d, dict):
            st.last_ok = now
            st.links["core"] = "live"
            pose = d.get("pose")
            if isinstance(pose, dict) and _num(pose.get("x")) is not None and _num(pose.get("y")) is not None:
                st.pose = {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": _num(pose.get("yaw")) or 0.0}
                st.pose_t = now
                pm = robot.pose_map()
                if st.prev_map is not None and now - st.prev_map[0] > 0.05:
                    dt = now - st.prev_map[0]
                    vx, vy = (pm["x"] - st.prev_map[1]) / dt, (pm["y"] - st.prev_map[2]) / dt
                    # light smoothing; clamp to physically plausible speeds
                    if math.hypot(vx, vy) > 4.0:
                        vx = vy = 0.0
                    st.vel = (0.5 * st.vel[0] + 0.5 * vx, 0.5 * st.vel[1] + 0.5 * vy)
                st.prev_map = (now, pm["x"], pm["y"])
            bat = d.get("battery")
            pct = _num(bat.get("percent")) if isinstance(bat, dict) else _num(bat)
            if pct is not None:
                st.battery = round(pct, 1)
            if "armed" in d:
                st.armed = bool(d.get("armed"))
        else:
            st.links["core"] = "down"
    if "safety" in u:
        code, d = transport.get(u["safety"] + "/state")
        with robot.lock:
            if ok(code) and isinstance(d, dict):
                st.links["safety"] = "live"
                st.safety_level = d.get("level")
                st.safety_vmax = _num(d.get("vmax"))
            else:
                st.links["safety"] = "down"
                st.safety_level = None
    if "mission" in u:
        code, d = transport.get(u["mission"] + "/missions")
        with robot.lock:
            if ok(code):
                st.links["mission"] = "live"
                ms = [m for m in _list_of(d, "missions") if isinstance(m, dict)]
                st.mission_states = {str(m.get("mission_id")): str(m.get("state")) for m in ms if m.get("mission_id")}
                act = active_mission(ms)
                st.mission = act
                st.path_robot = _mission_path(act)
            else:
                st.links["mission"] = "down"
                st.mission, st.path_robot = None, None
    if "omni" in u:
        code, d = transport.get(u["omni"] + "/health")
        omni_up = ok(code)
        persons: List[Dict[str, Any]] = []
        rec = None
        if omni_up:
            c2, d2 = transport.get(u["omni"] + "/persons")
            if ok(c2):
                persons = [p for p in _list_of(d2, "persons") if isinstance(p, dict)]
            c3, d3 = transport.get(u["omni"] + "/record/status")
            if ok(c3) and isinstance(d3, dict):
                rec = d3
        with robot.lock:
            st.links["omni"] = "live" if omni_up else "down"
            st.omni_ok = omni_up
            st.persons = persons
            st.record = rec
    if not robot.online(now):
        with robot.lock:
            st.vel = (0.0, 0.0)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    robots = d.get("robots") if isinstance(d, dict) else d
    return [r for r in (robots or []) if isinstance(r, dict)]


def atomic_write_json(path: str, data: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fleet.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Registry:
    """robots.yaml (static) + overlay JSON (runtime POST/DELETE /robots, aligns)."""

    def __init__(self, yaml_path: Optional[str] = None, overlay_path: Optional[str] = None,
                 transport: Any = None, specs: Optional[List[Dict[str, Any]]] = None):
        self.yaml_path = yaml_path
        self.overlay_path = overlay_path
        self.transport = transport
        self.lock = threading.RLock()
        self.robots: Dict[str, Robot] = {}
        self._overlay = {"added": {}, "removed": [], "align": {}}
        self._load(specs)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_poll: List[Callable[["Registry"], None]] = []

    # --- persistence
    def _load(self, specs: Optional[List[Dict[str, Any]]]) -> None:
        base = list(specs) if specs is not None else load_yaml(self.yaml_path or "")
        if self.overlay_path and os.path.isfile(self.overlay_path):
            try:
                with open(self.overlay_path, "r", encoding="utf-8") as f:
                    ov = json.load(f)
                if isinstance(ov, dict):
                    self._overlay.update({k: ov[k] for k in ("added", "removed", "align") if k in ov})
            except (OSError, ValueError):
                pass
        merged = [d for d in base if str(d.get("id")) not in self._overlay["removed"]]
        ids = {str(d.get("id")) for d in merged}
        merged += [d for k, d in self._overlay["added"].items() if k not in ids]
        for i, d in enumerate(merged):
            d = dict(d)
            if str(d.get("id")) in self._overlay["align"]:
                d["align"] = self._overlay["align"][str(d.get("id"))]
            try:
                spec = RobotSpec(d, i)
            except ValueError:
                continue
            self.robots[spec.id] = Robot(spec)

    def _save(self) -> None:
        if self.overlay_path:
            atomic_write_json(self.overlay_path, self._overlay)

    # --- CRUD
    def ids(self) -> List[str]:
        with self.lock:
            return list(self.robots)

    def get(self, rid: str) -> Optional[Robot]:
        with self.lock:
            return self.robots.get(rid)

    def all(self) -> List[Robot]:
        with self.lock:
            return list(self.robots.values())

    def add(self, d: Dict[str, Any]) -> Robot:
        with self.lock:
            spec = RobotSpec(d, len(self.robots))
            old = self.robots.get(spec.id)
            r = Robot(spec)
            if old is not None:
                r.st = old.st
            self.robots[spec.id] = r
            self._overlay["added"][spec.id] = spec.to_dict()
            if spec.id in self._overlay["removed"]:
                self._overlay["removed"].remove(spec.id)
            self._overlay["align"].pop(spec.id, None)
            self._save()
            return r

    def remove(self, rid: str) -> bool:
        with self.lock:
            if rid not in self.robots:
                return False
            del self.robots[rid]
            self._overlay["added"].pop(rid, None)
            self._overlay["align"].pop(rid, None)
            if rid not in self._overlay["removed"]:
                self._overlay["removed"].append(rid)
            self._save()
            return True

    def set_align(self, rid: str, align: Dict[str, Any]) -> Align:
        with self.lock:
            r = self.robots[rid]
            r.spec.align = Align.from_dict(align)
            with r.lock:
                r.st.prev_map = None
            if rid in self._overlay["added"]:
                self._overlay["added"][rid]["align"] = r.spec.align.to_dict()
            else:
                self._overlay["align"][rid] = r.spec.align.to_dict()
            self._save()
            return r.spec.align

    def summaries(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        out = []
        for r in self.all():
            with r.lock:
                out.append(r.summary(now))
        return out

    # --- polling
    def poll_all(self, now: Optional[float] = None) -> None:
        robots = [r for r in self.all() if r.spec.enabled]
        if not robots:
            return
        if len(robots) == 1:
            poll_robot(robots[0], self.transport, now)
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(robots))) as ex:
                list(ex.map(lambda r: poll_robot(r, self.transport, now), robots))
        for cb in list(self.on_poll):
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass

    def start(self, period_s: float = 1.0) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    self.poll_all()
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(max(0.05, period_s - (time.time() - t0)))

        self._thread = threading.Thread(target=loop, daemon=True, name="fleet-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # --- mc.fleet.robots
    def bus_robots(self, now: Optional[float] = None, frame: Optional[str] = None) -> Dict[str, Any]:
        """``{t, frame, robots:[{id,x,y,vx,vy,yaw}]}`` — online robots, map frame
        (or ``frame=<robot_id>``: that robot's own frame, for its safety_guard).
        Positions are extrapolated to ``now`` with the estimated velocity."""
        now = time.time() if now is None else now
        target = self.get(frame) if frame and frame != "map" else None
        out = []
        for r in self.all():
            with r.lock:
                if not r.online(now) or r.st.pose is None:
                    continue
                pm = r.pose_map()
                vx, vy = r.st.vel
                dt = min(max(now - (r.st.pose_t or now), 0.0), 1.0)
            x, y, yaw = pm["x"] + vx * dt, pm["y"] + vy * dt, pm["yaw"]
            if target is not None:
                ta = target.spec.align
                x, y = ta.to_robot(x, y)
                c, s = math.cos(-ta.dyaw), math.sin(-ta.dyaw)
                vx, vy = c * vx - s * vy, s * vx + c * vy
                yaw = ta.yaw_to_robot(yaw)
            out.append({"id": r.id, "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4),
                        "vx": round(vx, 3), "vy": round(vy, 3)})
        return {"t": round(now, 3), "frame": frame or "map", "robots": out}
