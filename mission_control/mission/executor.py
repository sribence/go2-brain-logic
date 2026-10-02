"""Mission executor (M1): queue + per-mission state machine + step runners.

One mission runs at a time in a background thread with a fixed-rate control
loop (default 20 Hz). Each step is a *generator* that does one control tick
per ``next()``; the loop owns timing, pause/resume/abort and failure policy,
so runners stay small and every motion goes through exactly one place:

    ctx.move(vx, vy, vyaw, profile)  ->  sender.move(...)  ->  safety_guard POST /move
                                         {vx, vy, vyaw, source:"mission", profile}

State machine (CONTRACT.md section 1/2)::

    pending_approval --approve--> queued --(worker)--> running --> done | failed
          |                         |                   |  ^
          +--------abort------------+------abort--------+  | resume
                                                        pause
                                                         v  |
                                                        paused --abort--> aborted

Pause = stop sending moves (one /stop). Abort = /stop immediately (also from
the API thread). A guard refusal (409/403/...) or repeated transport failure
(502 / unreachable) fails the step; the mission then follows ``on_fail``
(stop | skip | return_home). A refused *jump* is fatal (no skip).

Everything external is injected (planner, world, sender, pose / persons / map /
lidar providers, services, clock, sleep, publish) so the executor is unit
testable without network, redis or hardware. ``make_http_executor()`` wires
the real HTTP implementations.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from mission import schema as S  # noqa: E402
except ImportError:  # pragma: no cover - flat layout (PYTHONPATH=/app/mission)
    import schema as S  # type: ignore  # noqa: E402

PILLAR = "mission"
SOURCE = "mission"

TRANSITIONS = {
    "pending_approval": ("queued", "aborted"),
    "queued": ("running", "aborted"),
    "running": ("paused", "done", "failed", "aborted"),
    "paused": ("running", "failed", "aborted"),
}

# fallback profile table (planner.PROFILES wins when importable)
DEFAULT_PROFILES = {
    "stealth": {"vmax": 0.25, "amax": 0.3},
    "precise": {"vmax": 0.3, "amax": 0.4},
    "normal": {"vmax": 0.6, "amax": 0.6},
    "sprint": {"vmax": float(os.environ.get("SPRINT_VMAX", "1.5")), "amax": 1.0},
}

REFUSAL_CODES = (400, 401, 403, 404, 409, 422, 423, 428)


class InvalidTransition(Exception):
    pass


class StepFailed(Exception):
    """A step could not complete. fatal=True bypasses on_fail=skip/return_home;
    then="return_home"/"abort" forces that follow-up regardless of on_fail."""

    def __init__(self, msg: str, fatal: bool = False, then: Optional[str] = None):
        super().__init__(msg)
        self.fatal = fatal
        self.then = then


class _Abort(Exception):
    pass


@dataclass
class ExecConfig:
    hz: float = 20.0
    state_hz: float = 5.0
    max_move_errors: int = 3          # consecutive transport errors (502/unreachable) -> fail
    pose_timeout_s: float = 2.0
    # path following
    lookahead_pts: int = 8
    dwa_person_radius_m: float = 4.0
    replan_block_s: float = 1.5
    max_replans: int = 5
    stuck_s: float = 8.0              # no progress -> replan
    goal_timeout_factor: float = 3.0
    goal_timeout_min_s: float = 30.0
    # fine positioning (CONTRACT 9.1)
    fine_pos_m: float = 0.05
    fine_yaw_rad: float = math.radians(3.0)
    fine_settle_s: float = 0.5
    fine_slow_radius_m: float = 0.5   # precise profile inside this
    fine_v: float = 0.15
    fine_kp: float = 1.0
    fine_timeout_s: float = 10.0
    # rotation
    kp_yaw: float = 1.5
    max_vyaw: float = 0.8
    min_vyaw: float = 0.15
    yaw_tol_rad: float = math.radians(5.0)
    rotate_timeout_s: float = 15.0
    # actions / jumps
    action_settle_s: float = 3.0
    jump_action: str = "jump"
    jump_len_m: float = float(os.environ.get("JUMP_LEN_M", "0.6"))
    jump_settle_s: float = 1.5
    jump_land_tol_m: float = 0.5
    # persons
    min_person_dist_m: float = 2.5
    person_margin_m: float = 0.15
    approach_lookahead_s: float = 1.0
    lost_fail_s: float = 12.0
    escort_kp: float = 0.8
    # passage / lidar corridor
    corridor_half_w_m: float = 0.4
    corridor_min_x_m: float = 0.25
    lidar_z_min: float = -0.25
    lidar_z_max: float = 1.5
    corridor_min_points: int = 3
    # explore
    explore_poll_s: float = 1.0
    say_s_per_char: float = 0.07


# ---------------------------------------------------------------- geometry

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _get(p: Any, key: str, default: Any = None) -> Any:
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _path_len(pts: Sequence[Sequence[float]]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))


def densify(pts: Sequence[Sequence[float]], ds: float = 0.1) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i][:2], pts[i + 1][:2]
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / ds)))
        for k in range(n):
            out.append([x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n])
    if pts:
        out.append([float(pts[-1][0]), float(pts[-1][1])])
    return out


def point_in_polygon(poly: Sequence[Sequence[float]], x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def persons_to_world(persons: Sequence[Any], pose: Tuple[float, float, float]) -> List[Dict[str, Any]]:
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for p in persons or []:
        try:
            bx, by = float(_get(p, "x")), float(_get(p, "y"))
        except (TypeError, ValueError):
            continue
        vx, vy = float(_get(p, "vx", 0.0) or 0.0), float(_get(p, "vy", 0.0) or 0.0)
        out.append({"gid": _get(p, "gid"), "x": px + c * bx - s * by, "y": py + s * bx + c * by,
                    "vx": c * vx - s * vy, "vy": s * vx + c * vy, "x_base": bx, "y_base": by,
                    "last_seen_t": _get(p, "last_seen_t")})
    return out


def cap_approach(vx: float, vy: float, target_base: Optional[Tuple[float, float]], min_dist: float,
                 lookahead_s: float) -> Tuple[float, float]:
    """Limit the body-frame translation (vx, vy) so the range to a person at
    target_base cannot drop below min_dist within lookahead_s; inside min_dist
    any approaching component is removed."""
    if target_base is None:
        return vx, vy
    tx, ty = target_base
    d = math.hypot(tx, ty)
    if d < 1e-6:
        return 0.0, 0.0
    ux, uy = tx / d, ty / d
    approach = vx * ux + vy * uy             # > 0: range shrinks
    if approach <= 0.0:
        return vx, vy
    room = max(d - min_dist, 0.0)
    max_approach = room / max(lookahead_s, 1e-3)
    if approach > max_approach:
        cut = approach - max_approach
        vx, vy = vx - cut * ux, vy - cut * uy
    return vx, vy


# ---------------------------------------------------------------- planner adapter

class _FallbackPlanner:
    """Minimal straight-line planner used when mission/planner.py (M2) is not
    importable: keeps the executor usable and testable on its own."""
    PROFILES = DEFAULT_PROFILES

    @staticmethod
    def plan_path(grid_meta, floor, walls, start_xy, goal_xy, social=None, no_go_mask=None, cfg=None):
        pts = densify([list(start_xy), list(goal_xy)], 0.1)
        return {"points": pts, "length_m": _path_len(pts), "ok": True, "reason": "straight line (fallback planner)"}

    @staticmethod
    def smooth_path(points, ds=0.1, **kw):
        return densify(points, ds)

    @staticmethod
    def speed_profile(path, profile_name, cfg=None, persons_world=None, zones=None):
        pts = _get(path, "points", path)
        prof = DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["normal"])
        vmax, amax = prof["vmax"], prof["amax"]
        n = len(pts)
        v = [vmax] * n
        stop_last = zones is None or (zones and zones[-1] == "fine")
        if stop_last and n:
            rem = 0.0
            v[-1] = 0.0
            for i in range(n - 2, -1, -1):
                rem += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                v[i] = min(vmax, math.sqrt(2.0 * amax * rem))
        return v

    @staticmethod
    def plan_jumps(grid_meta, floor, walls, start_xy, goal_xy, jump_len_m=0.6, clearance=0.3, max_jumps=10):
        d = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
        n = max(1, int(math.ceil(d / jump_len_m - 1e-6)))
        if n > max_jumps:
            return {"landings": [], "ok": False,
                    "reason": "needs %d jumps of %.2f m > max_jumps %d" % (n, jump_len_m, max_jumps)}
        ux, uy = (goal_xy[0] - start_xy[0]) / (d or 1.0), (goal_xy[1] - start_xy[1]) / (d or 1.0)
        land = [[start_xy[0] + ux * min(d, jump_len_m * (k + 1)), start_xy[1] + uy * min(d, jump_len_m * (k + 1))]
                for k in range(n)]
        return {"landings": land, "ok": True, "reason": ""}

    @staticmethod
    def follow_controller(pose, path, v_profile, cfg=None):
        pts = _get(path, "points", path)
        x, y, yaw = pose[0], pose[1], pose[2]
        best, bi = 1e18, 0
        for i, p in enumerate(pts):
            dd = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if dd < best:
                best, bi = dd, i
        look = bi
        while look < len(pts) - 1 and math.hypot(pts[look][0] - x, pts[look][1] - y) < 0.5:
            look += 1
        tx, ty = pts[look]
        err = _wrap(math.atan2(ty - y, tx - x) - yaw)
        v = v_profile[bi] if v_profile else 0.3
        vyaw = _clamp(1.8 * err, -0.8, 0.8)
        vx = v * max(math.cos(err), 0.0) if abs(err) < 1.0 else 0.0
        done = bi >= len(pts) - 1 and math.sqrt(best) < 0.1
        return vx, vyaw, bi, done

    @staticmethod
    def eta(path, v):
        pts = _get(path, "points", path)
        t = 0.0
        for i in range(len(pts) - 1):
            seg = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            vv = max(0.05, 0.5 * ((v[i] if v else 0.3) + (v[i + 1] if v else 0.3)))
            t += seg / vv
        return t


def load_planner_module() -> Any:
    try:
        from mission import planner as P  # noqa: F401
        return P
    except Exception:
        try:
            import planner as P  # type: ignore
            return P
        except Exception:
            return None


class PlannerAdapter:
    """Normalises the M2 planner (or the fallback) to plain lists/dicts."""

    def __init__(self, mod: Any = None):
        self.mod = mod if mod is not None else (load_planner_module() or _FallbackPlanner)
        self.fallback = self.mod is _FallbackPlanner

    @property
    def profiles(self) -> Dict[str, Dict[str, float]]:
        p = getattr(self.mod, "PROFILES", None) or DEFAULT_PROFILES
        out = {}
        for k, dv in DEFAULT_PROFILES.items():
            pv = p.get(k, dv) if isinstance(p, dict) else dv
            out[k] = {"vmax": float(_get(pv, "vmax", dv["vmax"])), "amax": float(_get(pv, "amax", dv["amax"]))}
        return out

    def vmax(self, profile: str) -> float:
        return self.profiles.get(profile, self.profiles["normal"])["vmax"]

    def plan_path(self, grid: Optional[dict], start: Sequence[float], goal: Sequence[float],
                  social: Any = None, no_go: Any = None) -> Dict[str, Any]:
        start, goal = [float(start[0]), float(start[1])], [float(goal[0]), float(goal[1])]
        if grid is None or self.fallback:
            pts = densify([start, goal], 0.1)
            why = "no map: straight line" if grid is None else "fallback planner: straight line"
            return {"points": pts, "length_m": _path_len(pts), "ok": True, "reason": why, "raw": None,
                    "warnings": [why]}
        try:
            res = self.mod.plan_path(grid, grid.get("floor"), grid.get("walls"), start, goal,
                                     social=social, no_go_mask=no_go)
        except Exception as exc:
            return {"points": [], "length_m": 0.0, "ok": False, "reason": "planner error: %s" % exc,
                    "raw": None, "warnings": []}
        pts = [[float(p[0]), float(p[1])] for p in (_get(res, "points") or [])]
        ok = bool(_get(res, "ok", bool(pts))) and len(pts) >= 1
        if ok and len(pts) == 1:
            pts = [pts[0], pts[0]]
        return {"points": pts, "length_m": float(_get(res, "length_m", _path_len(pts)) or 0.0), "ok": ok,
                "reason": str(_get(res, "reason", "") or ""), "raw": None,
                "warnings": list(_get(res, "warnings", []) or []), "stops": list(_get(res, "stops", []) or [])}

    def smooth(self, points: Sequence[Sequence[float]], zones: Optional[List[str]] = None) -> List[List[float]]:
        try:
            if zones is not None:
                try:
                    out = self.mod.smooth_path(points, ds=0.1, zones=zones)
                except TypeError:
                    out = self.mod.smooth_path(points, ds=0.1)
            else:
                out = self.mod.smooth_path(points, ds=0.1)
            out = _get(out, "points", out)
            return [[float(p[0]), float(p[1])] for p in out]
        except Exception:
            return densify(points, 0.1)

    def _cfg(self, **kw: Any) -> Any:
        PC = getattr(self.mod, "PlannerConfig", None)
        if PC is None:
            return None
        try:
            return PC(**kw)
        except TypeError:
            return PC()

    def speeds(self, plan: Dict[str, Any], profile: str, persons_world: Any = None,
               end_zone: Optional[str] = "fine") -> List[float]:
        """v per point. end_zone "fine" -> decelerate to 0 at the end, else
        fly-by (no slowdown into the last point). plan["stops"] (fine
        indices) are honoured by the M2 planner."""
        pts = plan["points"]
        fine_end = (end_zone or "fine") == "fine"
        zones = [S.DEFAULT_ZONE] * (len(pts) - 1) + [end_zone or "fine"] if pts else []
        arg = plan.get("raw") if plan.get("raw") is not None else pts
        v = None
        if not self.fallback:
            try:
                v = self.mod.speed_profile(arg, profile, self._cfg(fine_end=fine_end), persons_world=persons_world,
                                           stops=plan.get("stops") or None)
                plan.setdefault("warnings", [])
                for w in getattr(v, "warnings", []) or []:
                    if w not in plan["warnings"]:
                        plan["warnings"].append(w)
                v = [float(x) for x in v]
            except Exception:
                v = None
        if v is None or len(v) != len(pts):
            v = _FallbackPlanner.speed_profile(pts, profile, zones=zones)
        vmax = self.vmax(profile)
        v = [min(max(0.0, x), vmax) for x in v]
        if not fine_end and len(v) > 1:
            v[-1] = max(v[-1], v[-2])            # fly-by: no slowdown into the waypoint
        return v

    def jumps(self, grid: Optional[dict], start: Sequence[float], goal: Sequence[float], max_jumps: int,
              jump_len: float) -> Dict[str, Any]:
        start, goal = [float(start[0]), float(start[1])], [float(goal[0]), float(goal[1])]
        mod = self.mod if (grid is not None and not self.fallback) else _FallbackPlanner
        try:
            if mod is _FallbackPlanner:
                res = mod.plan_jumps(grid, None, None, start, goal, jump_len_m=jump_len, max_jumps=max_jumps)
            else:
                try:
                    res = mod.plan_jumps(grid, grid.get("floor"), grid.get("walls"), start, goal,
                                         jump_len_m=jump_len, max_jumps=max_jumps)
                except TypeError:
                    res = mod.plan_jumps(grid, grid.get("floor"), grid.get("walls"), start, goal,
                                         jump_len_m=jump_len)
        except Exception as exc:
            return {"landings": [], "ok": False, "reason": "jump planner error: %s" % exc}
        land = [[float(p[0]), float(p[1])] for p in (_get(res, "landings") or [])]
        ok = bool(_get(res, "ok", bool(land)))
        reason = str(_get(res, "reason", "") or "")
        if ok and len(land) > max_jumps:
            ok, reason = False, "needs %d jumps > max_jumps %d" % (len(land), max_jumps)
        return {"landings": land, "ok": ok, "reason": reason}

    def follow(self, pose: Tuple[float, float, float], plan: Dict[str, Any], v: List[float], idx: int = 0
               ) -> Tuple[float, float, int, bool]:
        arg = plan.get("raw") if plan.get("raw") is not None else plan["points"]
        if not self.fallback:
            try:
                try:
                    vx, vyaw, i, done = self.mod.follow_controller(pose, arg, v, None, idx=idx)
                except TypeError:
                    vx, vyaw, i, done = self.mod.follow_controller(pose, arg, v, None)
                return float(vx), float(vyaw), int(i), bool(done)
            except Exception:
                pass
        return _FallbackPlanner.follow_controller(pose, plan["points"], v)

    def blend(self, cmd: Tuple[float, float], dwa: Optional[Tuple[float, float]], near: bool
              ) -> Tuple[float, float]:
        fn = getattr(self.mod, "blend_with_dwa", None)
        if fn is not None:
            try:
                return fn(cmd, dwa, near)
            except Exception:
                pass
        if dwa is None:
            return cmd
        return min(float(dwa[0]), float(cmd[0])) if near else float(cmd[0]), float(dwa[1]) if near else float(cmd[1])

    def smooth_ex(self, points: Sequence[Sequence[float]], zones: Optional[List[str]]
                  ) -> Tuple[List[List[float]], List[int]]:
        fn = getattr(self.mod, "smooth_path_ex", None)
        if fn is not None and zones is not None:
            try:
                pts, stops = fn(points, 0.1, None, zones)
                return [[float(p[0]), float(p[1])] for p in pts], [int(i) for i in stops]
            except Exception:
                pass
        pts = self.smooth(points, zones)
        return pts, [len(pts) - 1]

    def eta(self, plan: Dict[str, Any], v: List[float]) -> float:
        try:
            arg = plan.get("raw") if plan.get("raw") is not None else plan["points"]
            return float(self.mod.eta(arg, v))
        except Exception:
            return _FallbackPlanner.eta(plan["points"], v)


# ---------------------------------------------------------------- mission run record

class MissionRun:
    def __init__(self, mission: S.Mission, plan: Optional[dict] = None, now: float = 0.0):
        self.mission = mission
        self.id = mission.mission_id
        self.state = mission.state
        self.step_index = 0
        self.step_op = mission.steps[0].op if mission.steps else ""
        self.progress = 0.0
        self.step_progress = 0.0
        self.eta_s: Optional[float] = None
        self.error: Optional[str] = None
        self.plan = plan
        self.created_t = now
        self.started_t: Optional[float] = None
        self.finished_t: Optional[float] = None
        self.results: List[Dict[str, Any]] = [{"op": s.op, "status": "pending"} for s in mission.steps]
        self.events: deque = deque(maxlen=200)
        self.warnings: List[str] = []
        self.abort_flag = False
        self.continue_flag = False

    def state_payload(self, t: float) -> Dict[str, Any]:
        return {"t": t, "mission_id": self.id, "state": self.state, "step_index": self.step_index,
                "step_op": self.step_op, "progress": round(self.progress, 4),
                "eta_s": None if self.eta_s is None else round(self.eta_s, 1), "error": self.error}

    def to_dict(self) -> Dict[str, Any]:
        d = self.state_payload(time.time())
        d.update({"name": self.mission.name, "source": self.mission.source, "on_fail": self.mission.on_fail,
                  "mission": self.mission.to_dict(), "results": self.results, "plan": self.plan,
                  "created_t": self.created_t, "started_t": self.started_t, "finished_t": self.finished_t,
                  "events": list(self.events)[-50:], "warnings": self.warnings})
        d["mission"]["state"] = self.state
        return d


# ---------------------------------------------------------------- step context

class _Ctx:
    """Per-mission helper handed to the step runners."""

    def __init__(self, ex: "MissionExecutor", run: MissionRun):
        self.ex, self.run, self.cfg = ex, run, ex.cfg
        self.paused_total = 0.0
        self.move_errors = 0
        self.last_pose: Optional[Tuple[float, float, float]] = None
        self.last_pose_t = ex.clock()
        self.map: Optional[dict] = None
        self.map_loaded = False
        self.span = (0.0, 1.0)            # nested progress for patrol / composite steps
        self.pause_hooks: Optional[Tuple[Callable[[], None], Callable[[], None]]] = None
        self.last_cmd = (0.0, 0.0, 0.0)

    # time excluding pauses
    def now(self) -> float:
        return self.ex.clock() - self.paused_total

    def clock(self) -> float:
        return self.ex.clock()

    def pose(self) -> Tuple[float, float, float]:
        p = None
        try:
            p = self.ex.get_pose()
        except Exception:
            p = None
        if p is not None:
            p = (float(_get(p, "x") if not isinstance(p, (list, tuple)) else p[0]),
                 float(_get(p, "y") if not isinstance(p, (list, tuple)) else p[1]),
                 float((_get(p, "yaw") if not isinstance(p, (list, tuple)) else p[2]) or 0.0))
            self.last_pose, self.last_pose_t = p, self.ex.clock()
            return p
        if self.last_pose is not None and self.ex.clock() - self.last_pose_t < self.cfg.pose_timeout_s:
            return self.last_pose
        raise StepFailed("no robot pose for %.1f s" % self.cfg.pose_timeout_s)

    def persons_base(self) -> List[Any]:
        try:
            return list(self.ex.get_persons() or [])
        except Exception:
            return []

    def persons_world(self, pose: Tuple[float, float, float]) -> List[Dict[str, Any]]:
        return persons_to_world(self.persons_base(), pose)

    def grid(self, refresh: bool = False) -> Optional[dict]:
        if refresh or not self.map_loaded:
            try:
                self.map = self.ex.get_map()
            except Exception:
                self.map = None
            self.map_loaded = True
        return self.map

    def move(self, vx: float, vy: float, vyaw: float, profile: str = "normal") -> None:
        if self.run.abort_flag:
            raise _Abort()
        code, detail = self.ex.sender.move(float(vx), float(vy), float(vyaw), profile)
        self.last_cmd = (vx, vy, vyaw)
        if 200 <= code < 300:
            self.move_errors = 0
            return
        if code in REFUSAL_CODES:
            raise StepFailed("safety_guard refused move (%d): %s" % (code, _short(detail)))
        self.move_errors += 1
        if self.move_errors >= self.cfg.max_move_errors:
            raise StepFailed("safety_guard unreachable / motion error (%d) x%d: %s"
                             % (code, self.move_errors, _short(detail)))

    def stop(self) -> None:
        self.last_cmd = (0.0, 0.0, 0.0)
        try:
            self.ex.sender.stop()
        except Exception:
            pass

    def progress(self, frac: float, eta_s: Optional[float] = None) -> None:
        a, b = self.span
        self.run.step_progress = _clamp(a + (b - a) * _clamp(frac, 0.0, 1.0), 0.0, 1.0)
        self.run.eta_s = eta_s
        n = max(1, len(self.run.mission.steps))
        self.run.progress = _clamp((self.run.step_index + self.run.step_progress) / n, 0.0, 1.0)

    def event(self, kind: str, **kw: Any) -> None:
        self.ex._event(self.run, kind, **kw)

    def warn(self, msg: str) -> None:
        self.run.warnings.append(msg)
        self.event("warn", msg=msg)

    def sleep_ticks(self, seconds: float) -> Iterator[None]:
        t_end = self.now() + seconds
        while self.now() < t_end:
            yield

    def lidar_world(self) -> Optional[List[Tuple[float, float, float]]]:
        try:
            pts = self.ex.get_lidar()
        except Exception:
            return None
        return None if pts is None else [(float(p[0]), float(p[1]), float(p[2])) for p in pts]


def _short(d: Any, n: int = 200) -> str:
    s = d if isinstance(d, str) else json.dumps(d, default=str)
    return s[:n]


# ---------------------------------------------------------------- executor

class MissionExecutor:
    def __init__(self, sender: Any, get_pose: Callable[[], Any],
                 get_persons: Optional[Callable[[], Any]] = None,
                 get_map: Optional[Callable[[], Any]] = None,
                 planner: Any = None, world: Any = None, services: Any = None,
                 get_lidar: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 publish: Optional[Callable[[str, dict], None]] = None,
                 log: Optional[Callable[..., None]] = None,
                 cfg: Optional[ExecConfig] = None, nav_local: Any = "auto", pursuit: Any = "auto",
                 rng: Optional[random.Random] = None):
        self.sender = sender
        self.get_pose = get_pose
        self.get_persons = get_persons or (lambda: [])
        self.get_map = get_map or (lambda: None)
        self.get_lidar = get_lidar or (lambda: None)
        self.planner = planner if isinstance(planner, PlannerAdapter) else PlannerAdapter(planner)
        self.world = world
        self.services = services
        self.clock = clock
        self.sleep = sleep
        self.publish = publish or (lambda ch, payload: None)
        self.log = log or (lambda level, msg, **kw: None)
        self.cfg = cfg or ExecConfig()
        self.rng = rng or random.Random()
        self.nav_local = _load_nav_local() if nav_local == "auto" else nav_local
        self.pursuit = _load_pursuit() if pursuit == "auto" else pursuit
        self.lock = threading.RLock()
        self.runs: Dict[str, MissionRun] = {}
        self.order: List[str] = []
        self.queue: deque = deque()
        self.current: Optional[MissionRun] = None
        self.listeners: List[Callable[[str, dict], None]] = []
        self.gestures: deque = deque(maxlen=50)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_state_pub = 0.0

    # -- listeners / bus ---------------------------------------------------
    def add_listener(self, fn: Callable[[str, dict], None]) -> None:
        self.listeners.append(fn)

    def _emit(self, channel: str, payload: dict) -> None:
        try:
            self.publish(channel, payload)
        except Exception:
            pass
        for fn in list(self.listeners):
            try:
                fn(channel, payload)
            except Exception:
                pass

    def _event(self, run: MissionRun, kind: str, **kw: Any) -> None:
        ev = dict({"t": self.clock(), "mission_id": run.id, "kind": kind}, **kw)
        run.events.append(ev)
        self._emit("mc.mission.event", ev)
        self.log("info" if kind not in ("failed", "step_failed", "warn") else "warn", "mission " + kind, **ev)

    def alert(self, run: Optional[MissionRun], kind: str, **kw: Any) -> None:
        ev = dict({"t": self.clock(), "mission_id": run.id if run else None, "kind": kind}, **kw)
        self._emit("mc.mission.alert", ev)

    def _publish_state(self, run: MissionRun, force: bool = False) -> None:
        t = self.clock()
        if not force and t - self._last_state_pub < 1.0 / self.cfg.state_hz:
            return
        self._last_state_pub = t
        self._emit("mc.mission.state", run.state_payload(t))

    # -- state machine -----------------------------------------------------
    def _transition(self, run: MissionRun, new: str, **kw: Any) -> None:
        with self.lock:
            if new not in TRANSITIONS.get(run.state, ()):
                raise InvalidTransition("%s: %s -> %s not allowed" % (run.id, run.state, new))
            old, run.state = run.state, new
            run.mission.state = new
            if new in S.TERMINAL_STATES:
                run.finished_t = self.clock()
                if new == "done":
                    run.progress, run.eta_s = 1.0, 0.0
        self._event(run, new, prev=old, **kw)
        self._publish_state(run, force=True)

    def submit(self, mission: S.Mission, plan: Optional[dict] = None) -> MissionRun:
        with self.lock:
            if mission.mission_id in self.runs:
                mission.mission_id = mission.mission_id + "-" + uuid.uuid4().hex[:4]
            run = MissionRun(mission, plan, self.clock())
            self.runs[run.id] = run
            self.order.append(run.id)
            if len(self.order) > 500:                       # bound memory
                for old in self.order[:100]:
                    r = self.runs.get(old)
                    if r is not None and r.state in S.TERMINAL_STATES:
                        self.runs.pop(old, None)
                self.order = [i for i in self.order if i in self.runs]
            if run.state == "queued":
                self.queue.append(run.id)
        self._event(run, "created", state=run.state, source=mission.source, n_steps=len(mission.steps))
        self._publish_state(run, force=True)
        self._wake.set()
        return run

    def get(self, mid: str) -> Optional[MissionRun]:
        return self.runs.get(mid)

    def list(self) -> List[MissionRun]:
        with self.lock:
            return [self.runs[i] for i in self.order if i in self.runs]

    def _require(self, mid: str) -> MissionRun:
        run = self.runs.get(mid)
        if run is None:
            raise KeyError(mid)
        return run

    def approve(self, mid: str) -> MissionRun:
        run = self._require(mid)
        with self.lock:
            if run.state != "pending_approval":
                raise InvalidTransition("%s: only pending_approval can be approved (is %s)" % (mid, run.state))
            self._transition(run, "queued")
            self.queue.append(run.id)
        self._wake.set()
        return run

    def pause(self, mid: str) -> MissionRun:
        run = self._require(mid)
        self._transition(run, "paused")
        return run

    def resume(self, mid: str) -> MissionRun:
        run = self._require(mid)
        if run.state != "paused":
            raise InvalidTransition("%s: only paused can be resumed (is %s)" % (mid, run.state))
        self._transition(run, "running")
        return run

    def abort(self, mid: str, reason: str = "operator abort") -> MissionRun:
        run = self._require(mid)
        with self.lock:
            if run.state in S.TERMINAL_STATES:
                raise InvalidTransition("%s: already %s" % (mid, run.state))
            active = self.current is run
            run.abort_flag = True
            run.error = reason
            if run.id in self.queue:
                self.queue.remove(run.id)
            self._transition(run, "aborted", reason=reason)
        if active:
            try:
                self.sender.stop()        # immediately, not only at the next tick
            except Exception:
                pass
        return run

    def continue_(self, mid: str) -> MissionRun:
        """Operator "you may pass" (POST /missions/{id}/continue)."""
        run = self._require(mid)
        if run.state not in ("running", "paused"):
            raise InvalidTransition("%s: continue only while running (is %s)" % (mid, run.state))
        run.continue_flag = True
        self._event(run, "continue")
        return run

    def notify_gesture(self, evt: Dict[str, Any]) -> None:
        e = dict(evt)
        e.setdefault("t", self.clock())
        self.gestures.append(e)

    def abort_all(self, reason: str = "shutdown") -> None:
        for run in self.list():
            if run.state not in S.TERMINAL_STATES:
                try:
                    self.abort(run.id, reason)
                except InvalidTransition:
                    pass

    # -- worker ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="mission-exec", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _next_queued(self) -> Optional[MissionRun]:
        with self.lock:
            while self.queue:
                run = self.runs.get(self.queue.popleft())
                if run is not None and run.state == "queued":
                    return run
        return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            run = self._next_queued()
            if run is None:
                self._wake.wait(0.2)
                self._wake.clear()
                continue
            try:
                self.execute(run)
            except Exception as exc:  # never let the worker die
                self.log("error", "executor crash: %s" % exc, mission_id=run.id)
                try:
                    self.sender.stop()
                except Exception:
                    pass

    def run_next_sync(self) -> Optional[MissionRun]:
        """Execute the next queued mission in the calling thread (tests / CLI)."""
        run = self._next_queued()
        if run is not None:
            self.execute(run)
        return run

    # -- mission execution -------------------------------------------------
    def execute(self, run: MissionRun) -> None:
        with self.lock:
            if run.state != "queued":
                return
            self.current = run
        try:
            run.started_t = self.clock()
            self._transition(run, "running")
            ctx = _Ctx(self, run)
            steps = run.mission.steps
            i = 0
            while i < len(steps):
                if run.state in S.TERMINAL_STATES:
                    return
                step = steps[i]
                run.step_index, run.step_op = i, step.op
                run.step_progress = 0.0
                ctx.progress(0.0, self._remaining_eta(run, i))
                run.results[i]["status"] = "running"
                self._event(run, "step_start", step_index=i, op=step.op)
                res = self._drive(run, ctx, self._runner(step, ctx, i))
                if res[0] == "aborted":
                    run.results[i]["status"] = "aborted"
                    return
                if res[0] == "done":
                    run.results[i]["status"] = "done"
                    self._event(run, "step_done", step_index=i, op=step.op)
                    i += 1
                    continue
                _, err, fatal, then = res
                run.results[i].update(status="failed", error=err)
                self._event(run, "step_failed", step_index=i, op=step.op, error=err)
                policy = then or ("stop" if fatal else run.mission.on_fail)
                if policy == "abort":
                    with self.lock:
                        run.error = err
                        if run.state not in S.TERMINAL_STATES:
                            self._transition(run, "aborted", reason=err)
                    ctx.stop()
                    return
                if policy == "skip":
                    run.results[i]["status"] = "skipped"
                    self._event(run, "step_skipped", step_index=i, op=step.op)
                    i += 1
                    continue
                if policy == "return_home":
                    home_err = self._return_home_after_fail(run, ctx)
                    if run.state in S.TERMINAL_STATES:
                        return
                    err = err + ("; return_home failed: %s" % home_err if home_err else "; returned home")
                run.error = err
                ctx.stop()
                if run.state not in S.TERMINAL_STATES:
                    self._transition(run, "failed", error=err)
                return
            ctx.stop()
            if run.state not in S.TERMINAL_STATES:
                self._transition(run, "done")
        finally:
            with self.lock:
                if self.current is run:
                    self.current = None

    def _return_home_after_fail(self, run: MissionRun, ctx: _Ctx) -> Optional[str]:
        self._event(run, "return_home")
        ctx.span = (0.0, 1.0)
        res = self._drive(run, ctx, self._run_return_home(ctx, S.ReturnHomeStep(speed="normal", zone="fine")))
        if res[0] == "done":
            return None
        if res[0] == "aborted":
            return "aborted"
        return res[1]

    def _drive(self, run: MissionRun, ctx: _Ctx, gen: Iterator[None]) -> tuple:
        """Tick a step generator at cfg.hz until it finishes. Returns ("done",),
        ("aborted",) or ("failed", err, fatal, then)."""
        period = 1.0 / self.cfg.hz
        next_t = self.clock()
        paused_at: Optional[float] = None
        try:
            while True:
                if run.abort_flag or run.state == "aborted":
                    gen.close()
                    ctx.stop()
                    return ("aborted",)
                if run.state == "paused":
                    if paused_at is None:
                        paused_at = self.clock()
                        ctx.stop()                       # once
                        self._event(run, "motion_stopped", reason="paused")
                        if ctx.pause_hooks:
                            _safe_call(ctx.pause_hooks[0])
                    self._publish_state(run)
                    self.sleep(period)
                    next_t = self.clock()
                    continue
                if paused_at is not None:
                    ctx.paused_total += self.clock() - paused_at
                    paused_at = None
                    if ctx.pause_hooks:
                        _safe_call(ctx.pause_hooks[1])
                try:
                    next(gen)
                except StopIteration:
                    return ("done",)
                except _Abort:
                    ctx.stop()
                    return ("aborted",)
                except StepFailed as e:
                    gen.close()
                    ctx.stop()
                    return ("failed", str(e), e.fatal, e.then)
                except Exception as e:          # runner bug: fail safe
                    gen.close()
                    ctx.stop()
                    self.log("error", "step runner error: %r" % e, mission_id=run.id)
                    return ("failed", "internal error: %r" % e, False, None)
                if run.abort_flag:
                    gen.close()
                    ctx.stop()
                    return ("aborted",)
                self._publish_state(run)
                next_t += period
                dt = next_t - self.clock()
                if dt > 0:
                    self.sleep(dt)
                else:
                    next_t = self.clock()
        finally:
            ctx.pause_hooks = None
            ctx.span = (0.0, 1.0)

    def _remaining_eta(self, run: MissionRun, i: int) -> Optional[float]:
        steps = (run.plan or {}).get("steps") or []
        tot = 0.0
        for st in steps[i:]:
            e = st.get("eta_s")
            if e is None:
                return None
            tot += e
        return tot if steps else None

    # -- runner dispatch ---------------------------------------------------
    def _runner(self, step: S.Step, ctx: _Ctx, index: int) -> Iterator[None]:
        fn = getattr(self, "_run_" + step.op, None)
        if fn is None:
            raise StepFailed("no runner for op %r" % step.op)
        if step.op == "capture":
            return fn(ctx, step, index)
        return fn(ctx, step)

    # ---------------------------------------------------------------- motion primitives
    def _no_go(self, grid: Optional[dict]) -> Any:
        if grid is None or self.world is None or not hasattr(self.world, "no_go_mask"):
            return None
        try:
            return self.world.no_go_mask(grid)
        except Exception:
            return None

    def _social(self, grid: Optional[dict], persons_w: List[dict]) -> Any:
        if grid is None or not persons_w:
            return None
        try:
            from mapping.social_layer import social_cost_grid  # type: ignore
            return social_cost_grid(persons_w, grid)
        except Exception:
            return None

    def _dwa_grid(self, grid: Optional[dict]) -> Optional[dict]:
        if grid is None:
            return None
        try:
            from navigation.astar import build_blocked_grid  # type: ignore
            import numpy as np
            w, h, res = int(grid["width"]), int(grid["height"]), float(grid["resolution"])
            blocked = np.asarray(build_blocked_grid(grid["floor"], grid.get("walls") or [0] * (w * h), w, h,
                                                    max(0, int(round(0.2 / res)))), dtype=bool)
            ng = self._no_go(grid)
            if ng is not None:
                blocked = blocked | np.asarray(ng, dtype=bool).reshape(-1)
            g = {k: grid[k] for k in ("resolution", "origin_x", "origin_y", "width", "height")}
            g["blocked"] = blocked
            return g
        except Exception:
            return grid

    def _plan_to(self, ctx: _Ctx, start: Sequence[float], goal: Sequence[float]) -> Dict[str, Any]:
        grid = ctx.grid()
        pw = ctx.persons_world(ctx.last_pose or (start[0], start[1], 0.0))
        return self.planner.plan_path(grid, start, goal, social=self._social(grid, pw), no_go=self._no_go(grid))

    def _follow(self, ctx: _Ctx, plan: Dict[str, Any], profile: str, end_zone: str,
                final_yaw: Optional[float] = None, replan_goal: bool = True, tol_m: float = 0.3) -> Iterator[None]:
        """Track a planned path. Pure pursuit (planner.follow_controller),
        swapped for DWA (navigation.nav_local) while a person is within
        dwa_person_radius_m; replans on block / no progress. end_zone "fine"
        -> exact stop + settle, "zNN" -> done once within NN cm (no stop)."""
        c = self.cfg
        if not plan["ok"]:
            raise StepFailed("no path: %s" % plan.get("reason"))
        for w in plan.get("warnings") or []:
            if w not in ctx.run.warnings:
                ctx.warn(w)
        goal = plan["points"][-1]
        pw0 = ctx.persons_world(ctx.pose())
        v = self.planner.speeds(plan, profile, pw0, end_zone)
        total = max(_path_len(plan["points"]), 1e-3)
        eta0 = self.planner.eta(plan, v)
        t_start = ctx.now()
        deadline = t_start + max(c.goal_timeout_min_s, c.goal_timeout_factor * eta0 + 15.0)
        radius = S.zone_radius_m(end_zone)
        fine = end_zone == "fine"
        reach = max(radius, c.fine_slow_radius_m) if fine else radius
        dwa_grid = None
        replans = 0
        blocked_since: Optional[float] = None
        best_rem, best_t = float("inf"), ctx.now()
        cur_vx, cur_vyaw = 0.0, 0.0
        idx = 0
        while True:
            pose = ctx.pose()
            d_goal = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
            if d_goal <= reach:
                break
            vx, vyaw, idx, _done = self.planner.follow(pose, plan, v, idx)
            idx = int(_clamp(idx, 0, len(plan["points"]) - 1))
            vcap = max(max(v[idx:idx + 2]) if v else 0.3, 0.1)
            pw = ctx.persons_world(pose)
            near = [p for p in pw if math.hypot(p["x"] - pose[0], p["y"] - pose[1]) < c.dwa_person_radius_m]
            blocked = False
            if near and self.nav_local is not None and ctx.grid() is not None:
                if dwa_grid is None:
                    dwa_grid = self._dwa_grid(ctx.grid())
                look = plan["points"][min(idx + c.lookahead_pts, len(plan["points"]) - 1)]
                try:
                    lc = self.nav_local.LocalConfig(max_vx=min(vcap, self.planner.vmax(profile)),
                                                    max_vyaw=c.max_vyaw + 0.2, cur_vx=cur_vx, cur_vyaw=cur_vyaw)
                    dvx, dvyaw, info = self.nav_local.plan_local(pose, look, dwa_grid, None, near, lc)
                    vx, vyaw = self.planner.blend((vx, vyaw), (float(dvx), float(dvyaw)), True)
                    blocked = int(info.get("n_valid", 1)) == 0 and not info.get("reached")
                except Exception as exc:
                    if "DWA error" not in " ".join(ctx.run.warnings[-3:]):
                        ctx.warn("DWA error: %s" % exc)
            vx = min(vx, vcap)
            # progress / eta
            rem = d_goal if idx >= len(plan["points"]) - 1 else (
                math.hypot(plan["points"][idx][0] - pose[0], plan["points"][idx][1] - pose[1])
                + _path_len(plan["points"][idx:]))
            vm = max(0.1, sum(v[idx:]) / max(1, len(v) - idx))
            ctx.progress(1.0 - rem / total, rem / vm)
            now = ctx.now()
            if rem < best_rem - 0.05:
                best_rem, best_t = rem, now
            if blocked:
                blocked_since = now if blocked_since is None else blocked_since
            else:
                blocked_since = None
            need_replan = (blocked_since is not None and now - blocked_since > c.replan_block_s) or \
                          (now - best_t > c.stuck_s)
            if need_replan:
                replans += 1
                if replans > c.max_replans or not replan_goal:
                    raise StepFailed("path blocked (%d replans)" % (replans - 1))
                ctx.grid(refresh=True)
                dwa_grid = None
                plan = self._plan_to(ctx, pose[:2], goal)
                if not plan["ok"]:
                    raise StepFailed("replan failed: %s" % plan.get("reason"))
                v = self.planner.speeds(plan, profile, pw, end_zone)
                total = max(_path_len(plan["points"]), 1e-3)
                goal = plan["points"][-1]
                idx = 0
                blocked_since, best_rem, best_t = None, float("inf"), now
                ctx.event("replan", n=replans)
            if now > deadline:
                raise StepFailed("timeout following path (%.0f s)" % (now - t_start))
            if blocked:
                vx, vyaw = 0.0, 0.0
            ctx.move(vx, 0.0, vyaw, profile)
            cur_vx, cur_vyaw = vx, vyaw
            yield
        if fine:
            yield from self._fine(ctx, goal, final_yaw, tol_m)
        elif final_yaw is not None:
            yield from self._rotate_to(ctx, lambda p: final_yaw, c.yaw_tol_rad)
        ctx.progress(1.0, 0.0)

    def _fine(self, ctx: _Ctx, goal: Sequence[float], yaw: Optional[float], tol_m: float) -> Iterator[None]:
        """Exact positioning with body-frame (vx, vy) at the precise profile,
        then settle cfg.fine_settle_s with zero command."""
        c = self.cfg
        t0 = ctx.now()
        while True:
            pose = ctx.pose()
            dx, dy = goal[0] - pose[0], goal[1] - pose[1]
            d = math.hypot(dx, dy)
            yerr = 0.0 if yaw is None else _wrap(yaw - pose[2])
            if d <= c.fine_pos_m and abs(yerr) <= c.fine_yaw_rad:
                break
            if ctx.now() - t0 > c.fine_timeout_s:
                if d <= tol_m:
                    ctx.warn("fine positioning not reached (%.3f m, %.1f deg); accepted within tol_m"
                             % (d, math.degrees(yerr)))
                    break
                raise StepFailed("fine positioning failed: %.2f m from goal" % d)
            cy, sy = math.cos(pose[2]), math.sin(pose[2])
            bx, by = cy * dx + sy * dy, -sy * dx + cy * dy
            sp = min(c.fine_v, c.fine_kp * d)
            vx, vy = (bx / d * sp, by / d * sp) if d > 1e-6 else (0.0, 0.0)
            if d > 0.3 and yaw is None:            # still far-ish: face the goal while closing in
                vyaw = _clamp(c.kp_yaw * math.atan2(by, bx), -c.max_vyaw, c.max_vyaw)
            else:
                vyaw = _clamp(c.kp_yaw * yerr, -c.max_vyaw, c.max_vyaw)
            ctx.move(vx, vy, vyaw, "precise")
            yield
        ctx.stop()
        yield from ctx.sleep_ticks(c.fine_settle_s)

    def _rotate_to(self, ctx: _Ctx, target_fn: Callable[[Tuple[float, float, float]], float],
                   tol: float) -> Iterator[None]:
        c = self.cfg
        t0 = ctx.now()
        while True:
            pose = ctx.pose()
            err = _wrap(target_fn(pose) - pose[2])
            if abs(err) <= tol:
                return
            if ctx.now() - t0 > c.rotate_timeout_s:
                raise StepFailed("rotation timeout (%.0f deg left)" % math.degrees(err))
            vyaw = _clamp(c.kp_yaw * err, -c.max_vyaw, c.max_vyaw)
            if abs(vyaw) < c.min_vyaw:
                vyaw = math.copysign(c.min_vyaw, err)
            ctx.move(0.0, 0.0, vyaw, "normal")
            yield

    def _goto_xy(self, ctx: _Ctx, x: float, y: float, yaw: Optional[float], speed: str, zone: str,
                 tol_m: float = 0.3) -> Iterator[None]:
        pose = ctx.pose()
        plan = self._plan_to(ctx, pose[:2], (x, y))
        if plan["ok"] and plan["points"]:
            plan["points"][-1] = [float(x), float(y)] if math.hypot(
                plan["points"][-1][0] - x, plan["points"][-1][1] - y) < 0.3 else plan["points"][-1]
        yield from self._follow(ctx, plan, speed, zone, yaw, tol_m=tol_m)

    # ---------------------------------------------------------------- step runners
    def _run_goto(self, ctx: _Ctx, st: S.GotoStep) -> Iterator[None]:
        yield from self._goto_xy(ctx, st.x, st.y, st.yaw, st.speed, st.zone or "fine", st.tol_m)

    def _run_follow_path(self, ctx: _Ctx, st: S.FollowPathStep) -> Iterator[None]:
        """Freehand path -> spline (zone-aware with M2's smooth_path_ex) ->
        followed segment by segment; each intermediate "fine" point is an
        exact stop + settle (CONTRACT 9.1). A blocked segment is replanned
        to its end point (shape of that segment is lost)."""
        pose = ctx.pose()
        pts = [list(p) for p in st.points]
        zones = list(st.zones) if st.zones else [S.DEFAULT_ZONE] * (len(pts) - 1) + [st.zone or "fine"]
        zones[-1] = st.zone or zones[-1] or "fine"
        if math.hypot(pts[0][0] - pose[0], pts[0][1] - pose[1]) > 0.3:
            pts = [[pose[0], pose[1]]] + pts        # join from where the robot is
            zones = [S.DEFAULT_ZONE] + zones
        if st.smooth:
            dense, stops = self.planner.smooth_ex(pts, zones)
        else:
            dense = densify(pts, 0.1)
            stops = [len(dense) - 1]
        cuts = [0] + sorted({i for i in stops if 0 < i < len(dense) - 1}) + [len(dense) - 1]
        segs = [dense[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a] or [dense]
        total = max(_path_len(dense), 1e-3)
        done_len = 0.0
        for k, seg in enumerate(segs):
            last = k == len(segs) - 1
            seg_len = _path_len(seg)
            ctx.span = (done_len / total, (done_len + seg_len) / total)
            plan = {"points": seg, "length_m": seg_len, "ok": len(seg) >= 2, "reason": "", "raw": None}
            yield from self._follow(ctx, plan, st.speed, (st.zone or "fine") if last else "fine")
            done_len += seg_len
        ctx.span = (0.0, 1.0)

    def _run_goto_label(self, ctx: _Ctx, st: S.GotoLabelStep) -> Iterator[None]:
        if self.world is None:
            raise StepFailed("world module unavailable: cannot resolve label %r" % st.label)
        lb = self.world.resolve_label(st.label)
        if lb is None:
            raise StepFailed("unknown label %r" % st.label)
        x, y, yaw = lb
        yield from self._goto_xy(ctx, x, y, yaw, st.speed, st.zone or "fine")

    def _run_return_home(self, ctx: _Ctx, st: S.ReturnHomeStep) -> Iterator[None]:
        if self.world is None:
            raise StepFailed("world module unavailable: no home")
        home = self.world.get_home()
        if not home:
            raise StepFailed("no home set (POST /home)")
        yield from self._goto_xy(ctx, float(home["x"]), float(home["y"]), home.get("yaw"), st.speed,
                                 st.zone or "fine")

    def _run_jump_to(self, ctx: _Ctx, st: S.JumpToStep) -> Iterator[None]:
        c = self.cfg
        pose = ctx.pose()
        plan = self.planner.jumps(ctx.grid(), pose[:2], (st.x, st.y), st.max_jumps, c.jump_len_m)
        if not plan["ok"]:
            raise StepFailed("jump plan rejected: %s" % plan["reason"])
        landings = plan["landings"]
        jumps = 0
        k = 0
        while k < len(landings):
            target = landings[k]
            yield from self._rotate_to(ctx, lambda p, t=target: math.atan2(t[1] - p[1], t[0] - p[0]),
                                       math.radians(4.0))
            ctx.stop()
            code, detail = self.sender.action(c.jump_action)
            if not (200 <= code < 300):
                raise StepFailed("safety_guard refused jump (%d): %s" % (code, _short(detail)), fatal=True)
            jumps += 1
            self._event(ctx.run, "jump", n=jumps, landing=target)
            yield from ctx.sleep_ticks(c.jump_settle_s)
            pose = ctx.pose()                      # re-localize
            off = math.hypot(target[0] - pose[0], target[1] - pose[1])
            ctx.progress(float(k + 1) / len(landings), (len(landings) - k - 1) * (c.jump_settle_s + 2.0))
            k += 1
            if off > c.jump_land_tol_m and k < len(landings):
                left = st.max_jumps - jumps
                ctx.warn("landing %d off by %.2f m: replanning jumps" % (jumps, off))
                if left <= 0:
                    raise StepFailed("jump_to: out of jumps %.2f m from goal"
                                     % math.hypot(st.x - pose[0], st.y - pose[1]))
                plan = self.planner.jumps(ctx.grid(), pose[:2], (st.x, st.y), left, c.jump_len_m)
                if not plan["ok"]:
                    raise StepFailed("jump replan rejected: %s" % plan["reason"])
                landings, k = plan["landings"], 0
        pose = ctx.pose()
        d = math.hypot(st.x - pose[0], st.y - pose[1])
        if d > max(c.jump_land_tol_m, c.jump_len_m):
            raise StepFailed("jump_to ended %.2f m from goal" % d)

    def _run_look_at(self, ctx: _Ctx, st: S.LookAtStep) -> Iterator[None]:
        yield from self._rotate_to(ctx, lambda p: math.atan2(st.y - p[1], st.x - p[0]), math.radians(4.0))
        ctx.stop()

    def _run_scan(self, ctx: _Ctx, st: S.ScanStep) -> Iterator[None]:
        target = math.radians(st.deg)
        rate = math.radians(st.speed_dps)
        prev = ctx.pose()[2]
        turned = 0.0
        t0 = ctx.now()
        limit = target / rate * 2.0 + 10.0
        while turned < target - math.radians(2.0):
            if ctx.now() - t0 > limit:
                raise StepFailed("scan timeout (%.0f of %.0f deg)" % (math.degrees(turned), st.deg))
            ctx.move(0.0, 0.0, rate, "normal")
            ctx.progress(turned / target, (target - turned) / rate)
            yield
            yaw = ctx.pose()[2]
            turned += abs(_wrap(yaw - prev))
            prev = yaw
        ctx.stop()

    def _run_action(self, ctx: _Ctx, st: S.ActionStep) -> Iterator[None]:
        ctx.stop()
        code, detail = self.sender.action(st.name)
        if not (200 <= code < 300):
            raise StepFailed("safety_guard refused action %s (%d): %s" % (st.name, code, _short(detail)))
        yield from ctx.sleep_ticks(self.cfg.action_settle_s)

    def _run_wait(self, ctx: _Ctx, st: S.WaitStep) -> Iterator[None]:
        ctx.stop()
        t0 = ctx.now()
        while ctx.now() - t0 < st.s:
            ctx.progress((ctx.now() - t0) / max(st.s, 1e-3), st.s - (ctx.now() - t0))
            yield

    def _say_async(self, text: str) -> None:
        if self.services is None:
            return
        threading.Thread(target=_safe_call, args=(lambda: self.services.speak(text),), daemon=True).start()

    def _run_say(self, ctx: _Ctx, st: S.SayStep) -> Iterator[None]:
        if self.services is None:
            raise StepFailed("audio service unavailable")
        ctx.stop()
        try:
            self.services.speak(st.text)
        except Exception as exc:
            raise StepFailed("say failed: %s" % exc)
        yield

    # -- person-centred ----------------------------------------------------
    def _target_base(self, ctx: _Ctx, gid: int) -> Any:
        if self.pursuit is not None:
            return self.pursuit.select_target(ctx.persons_base(), "gid", gid)
        for p in ctx.persons_base():
            if _get(p, "gid") == gid:
                return p
        return None

    def _person_loop(self, ctx: _Ctx, gid: int, timeout_s: Optional[float], follow_dist: float,
                     rotate_only: bool) -> Iterator[None]:
        c = self.cfg
        if self.pursuit is None:
            raise StepFailed("omni/pursuit.py unavailable")
        pc = self.pursuit.PursuitController(self.pursuit.PursuitConfig(
            follow_dist=follow_dist, min_dist=c.min_person_dist_m, max_v=self.planner.vmax("normal")))
        t0 = ctx.now()
        lost_since: Optional[float] = None
        while True:
            if timeout_s is not None and ctx.now() - t0 >= timeout_s:
                break
            pose = ctx.pose()
            tgt = self._target_base(ctx, gid)
            vx, vy, vyaw, state = pc.step(tgt, {"yaw": pose[2]}, ctx.clock())
            if state == "LOST":
                lost_since = ctx.now() if lost_since is None else lost_since
                if ctx.now() - lost_since > c.lost_fail_s:
                    raise StepFailed("person gid=%s lost" % gid)
            else:
                lost_since = None
            if rotate_only:
                vx, vy = 0.0, 0.0
            elif tgt is not None:
                vx, vy = cap_approach(vx, vy, (float(_get(tgt, "x")), float(_get(tgt, "y"))),
                                      c.min_person_dist_m + c.person_margin_m, c.approach_lookahead_s)
            if timeout_s:
                ctx.progress((ctx.now() - t0) / timeout_s, timeout_s - (ctx.now() - t0))
            ctx.move(vx, vy, vyaw, "normal")
            yield
        ctx.stop()

    def _run_shadow(self, ctx: _Ctx, st: S.ShadowStep) -> Iterator[None]:
        yield from self._person_loop(ctx, st.gid, st.timeout_s, st.dist_m, rotate_only=False)

    def _run_watch(self, ctx: _Ctx, st: S.WatchStep) -> Iterator[None]:
        yield from self._person_loop(ctx, st.gid, st.timeout_s, 3.0, rotate_only=True)

    def _run_escort(self, ctx: _Ctx, st: S.EscortStep) -> Iterator[None]:
        """Walk beside the person: hold the point dist_m to the person's
        `side`, perpendicular to their walking direction (or to the
        person->robot line while they stand still)."""
        c = self.cfg
        vmax = self.planner.vmax("normal")
        t0 = ctx.now()
        lost_since: Optional[float] = None
        heading: Optional[float] = None
        sign = -1.0 if st.side == "right" else 1.0
        while True:
            if st.timeout_s is not None and ctx.now() - t0 >= st.timeout_s:
                break
            pose = ctx.pose()
            tgt = self._target_base(ctx, st.gid)
            fresh = tgt is not None and (
                not _get(tgt, "last_seen_t") or ctx.clock() - float(_get(tgt, "last_seen_t")) < 0.5)
            if not fresh:
                lost_since = ctx.now() if lost_since is None else lost_since
                if ctx.now() - lost_since > c.lost_fail_s:
                    raise StepFailed("person gid=%s lost" % st.gid)
                ctx.move(0.0, 0.0, 0.0, "normal")
                yield
                continue
            lost_since = None
            pw = persons_to_world([tgt], pose)[0]
            spd = math.hypot(pw["vx"], pw["vy"])
            if spd > 0.25:
                heading = math.atan2(pw["vy"], pw["vx"])
            elif heading is None:
                heading = math.atan2(pw["y"] - pose[1], pw["x"] - pose[0])
            # side point: rotate heading by -90 deg (right) / +90 deg (left)
            gx = pw["x"] + st.dist_m * math.cos(heading + sign * math.pi / 2)
            gy = pw["y"] + st.dist_m * math.sin(heading + sign * math.pi / 2)
            dx, dy = gx - pose[0], gy - pose[1]
            cy, sy = math.cos(pose[2]), math.sin(pose[2])
            bx, by = cy * dx + sy * dy, -sy * dx + cy * dy
            d = math.hypot(dx, dy)
            ff = spd * math.cos(heading - pose[2]) if spd > 0.25 else 0.0
            if d > 0.3:
                vx = _clamp(c.escort_kp * bx + ff, -0.3, vmax)
                vy = _clamp(c.escort_kp * by, -0.3, 0.3)
                face = heading if spd > 0.25 else math.atan2(dy, dx)
            else:
                vx, vy = _clamp(ff, 0.0, vmax), 0.0
                face = heading
            vyaw = _clamp(c.kp_yaw * _wrap(face - pose[2]), -c.max_vyaw, c.max_vyaw)
            vx, vy = cap_approach(vx, vy, (float(_get(tgt, "x")), float(_get(tgt, "y"))),
                                  c.min_person_dist_m + c.person_margin_m, c.approach_lookahead_s)
            if st.timeout_s:
                ctx.progress((ctx.now() - t0) / st.timeout_s, st.timeout_s - (ctx.now() - t0))
            ctx.move(vx, vy, vyaw, "normal")
            yield
        ctx.stop()

    # -- area --------------------------------------------------------------
    def _patrol_points(self, st: S.PatrolStep) -> List[List[float]]:
        if st.points:
            return [list(p) for p in st.points]
        if isinstance(st.zone, (list, tuple)):
            poly = [list(p) for p in st.zone]
            if poly[0] == poly[-1]:
                poly = poly[:-1]
            return poly + [poly[0]]
        if self.world is None:
            raise StepFailed("world module unavailable: cannot resolve zone %r" % st.zone)
        pts = None
        if hasattr(self.world, "patrol_points"):
            pts = self.world.patrol_points(st.zone)
        if not pts:
            raise StepFailed("unknown / empty zone %r" % st.zone)
        return [list(p) for p in pts]

    def _run_patrol(self, ctx: _Ctx, st: S.PatrolStep) -> Iterator[None]:
        pts = self._patrol_points(st)
        zone = st.pass_zone or S.DEFAULT_ZONE
        n_total = max(1, st.loops * len(pts))
        k = 0
        for loop in range(st.loops):
            order = list(pts)
            if st.shuffle:
                self.rng.shuffle(order)
            for j, p in enumerate(order):
                last = loop == st.loops - 1 and j == len(order) - 1
                ctx.span = (k / n_total, (k + 1) / n_total)
                yield from self._goto_xy(ctx, p[0], p[1], None, st.speed, zone if not last else (st.pass_zone or "fine"))
                k += 1
        ctx.span = (0.0, 1.0)

    def _run_explore(self, ctx: _Ctx, st: S.ExploreStep) -> Iterator[None]:
        svc = self.services
        if svc is None:
            raise StepFailed("mapping service unavailable")
        if st.zone is not None:
            ctx.warn("explore: mapping /explore has no zone support yet; exploring everywhere")
        ok, detail = svc.explore_start(st.zone)
        if not ok:
            raise StepFailed("explore start refused: %s" % _short(detail))
        finished = False
        ctx.pause_hooks = (lambda: svc.explore_stop(), lambda: svc.explore_start(st.zone))
        try:
            t0 = ctx.now()
            next_poll = t0 + self.cfg.explore_poll_s
            while True:
                if ctx.now() - t0 > st.timeout_s:
                    ctx.warn("explore timeout after %.0f s" % st.timeout_s)
                    break
                if ctx.now() >= next_poll:
                    next_poll = ctx.now() + self.cfg.explore_poll_s
                    s = svc.explore_status() or {}
                    state = s.get("state")
                    if state == "error":
                        raise StepFailed("explore error: %s" % s.get("last_error"))
                    cov = s.get("coverage_percent")
                    ctx.progress(max((ctx.now() - t0) / st.timeout_s, (cov or 0.0) / 100.0), None)
                    if state not in ("exploring", None) and ctx.now() - t0 > 2.0:
                        finished = True
                        break
                yield
        finally:
            if not finished:
                _safe_call(svc.explore_stop)

    # -- CONTRACT 9.2 ------------------------------------------------------
    def _run_capture(self, ctx: _Ctx, st: S.CaptureStep, index: int) -> Iterator[None]:
        if st.x is not None:
            ctx.span = (0.0, 0.8)
            yield from self._goto_xy(ctx, st.x, st.y, st.yaw, "precise", "fine")
            ctx.span = (0.0, 1.0)
        elif st.yaw is not None:
            yield from self._rotate_to(ctx, lambda p: st.yaw, self.cfg.fine_yaw_rad)
        ctx.stop()
        if self.services is None:
            raise StepFailed("capture service unavailable")
        ok, res = self.services.capture(st.cams, st.mode, st.label)
        if not ok:
            raise StepFailed("capture failed: %s" % _short(res))
        files = list((res or {}).get("files") or []) if isinstance(res, dict) else []
        ctx.run.results[index]["files"] = files
        self._event(ctx.run, "capture", files=files, label=st.label)
        yield

    def _corridor_clear(self, ctx: _Ctx, pose: Tuple[float, float, float], polygon: Optional[list],
                        ahead_m: float) -> Optional[bool]:
        """True = free, False = blocked, None = no LiDAR data."""
        c = self.cfg
        pts = ctx.lidar_world()
        if pts is None:
            return None
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        n = 0
        for (x, y, z) in pts:
            if not (c.lidar_z_min <= z <= c.lidar_z_max):
                continue
            if polygon:
                hit = point_in_polygon(polygon, x, y)
            else:
                dx, dy = x - pose[0], y - pose[1]
                bx, by = cy * dx + sy * dy, -sy * dx + cy * dy
                hit = c.corridor_min_x_m <= bx <= ahead_m and abs(by) <= c.corridor_half_w_m
            if hit:
                n += 1
                if n >= c.corridor_min_points:
                    return False
        return True

    def _wave_since(self, t: float) -> bool:
        return any(e.get("gesture") == "wave" and float(e.get("t", 0.0)) >= t for e in list(self.gestures))

    def _run_request_passage(self, ctx: _Ctx, st: S.RequestPassageStep) -> Iterator[None]:
        ctx.stop()
        run = ctx.run
        run.continue_flag = False
        t0 = ctx.now()
        t0_clock = ctx.clock()
        last_say = None
        clear_since: Optional[float] = None
        timed_out = False
        self.alert(run, "waiting_passage", text=st.text, msg="waiting: %s" % st.text)
        while True:
            now = ctx.now()
            if last_say is None or now - last_say >= st.repeat_s:
                last_say = now
                self._say_async(st.text)
            if run.continue_flag:
                run.continue_flag = False
                self._event(run, "passage_granted", by="operator")
                break
            if self._wave_since(t0_clock):
                self._event(run, "passage_granted", by="gesture")
                break
            pose = ctx.pose()
            clear = self._corridor_clear(ctx, pose, st.zone_polygon, st.ahead_m)
            if clear:
                clear_since = now if clear_since is None else clear_since
                if now - clear_since >= st.clear_for_s:
                    self._event(run, "passage_granted", by="path_clear")
                    break
            else:
                clear_since = None
            if not timed_out and now - t0 > st.timeout_s:
                if st.on_timeout == "alert":
                    timed_out = True            # alert once, keep waiting for clear / operator / abort
                    self.alert(run, "passage_timeout", text=st.text, waited_s=round(now - t0, 1))
                elif st.on_timeout == "return":
                    raise StepFailed("request_passage timeout", then="return_home")
                else:
                    raise StepFailed("request_passage timeout", then="abort")
            ctx.progress(min(1.0, (now - t0) / st.timeout_s), None)
            yield

    def _run_wait_for(self, ctx: _Ctx, st: S.WaitForStep) -> Iterator[None]:
        ctx.stop()
        run = ctx.run
        run.continue_flag = False
        t0, t0_clock = ctx.now(), ctx.clock()
        ok_since: Optional[float] = None
        while True:
            now = ctx.now()
            if now - t0 > st.timeout_s:
                raise StepFailed("wait_for %s: timeout after %.0f s" % (st.event, st.timeout_s))
            if run.continue_flag and st.event == "operator":
                run.continue_flag = False
                return
            if st.event == "gesture" and self._wave_since(t0_clock):
                return
            cond = None
            if st.event == "path_clear":
                cond = self._corridor_clear(ctx, ctx.pose(), st.zone_polygon, st.ahead_m)
            elif st.event == "person_gone":
                ps = ctx.persons_base()
                cond = not any(st.gid is None or _get(p, "gid") == st.gid for p in ps)
            if cond:
                ok_since = now if ok_since is None else ok_since
                if now - ok_since >= st.clear_for_s:
                    return
            else:
                ok_since = None
            ctx.progress((now - t0) / st.timeout_s, None)
            yield

    def _run_record(self, ctx: _Ctx, st: S.RecordStep) -> Iterator[None]:
        if self.services is None:
            raise StepFailed("recorder service unavailable")
        ok, res = self.services.record(st.on, st.mode, "mission %s" % ctx.run.id)
        if not ok:
            raise StepFailed("record %s failed: %s" % ("start" if st.on else "stop", _short(res)))
        yield

    # ---------------------------------------------------------------- preview planning
    def _step_eta(self, st: S.Step) -> Optional[float]:
        c = self.cfg
        if st.op == "wait":
            return st.s
        if st.op == "scan":
            return st.deg / st.speed_dps
        if st.op == "action":
            return c.action_settle_s + 1.0
        if st.op == "say":
            return max(1.0, len(st.text) * c.say_s_per_char)
        if st.op == "look_at":
            return 2.0
        if st.op in ("shadow", "watch", "escort"):
            return st.timeout_s
        if st.op in ("record",):
            return 0.5
        if st.op in ("request_passage", "wait_for", "explore"):
            return None
        return 0.0

    def plan_step(self, st: S.Step, start: Tuple[float, float, float], grid: Optional[dict],
                  persons_w: Optional[list] = None, is_last: bool = True) -> Dict[str, Any]:
        """Preview of one step from `start` (does not execute anything)."""
        out: Dict[str, Any] = {"op": st.op, "ok": True, "reason": "", "points": [], "speeds": [],
                               "landings": [], "eta_s": self._step_eta(st), "length_m": 0.0, "warnings": [],
                               "end": [start[0], start[1], start[2]]}
        pw = persons_w or []
        speed = getattr(st, "speed", "normal")
        if speed == "sprint":
            out["warnings"].append("sprint: needs the dead-man held, otherwise capped to normal by safety_guard")

        def path_to(sx, sy, gx, gy, zone):
            plan = self.planner.plan_path(grid, (sx, sy), (gx, gy), social=self._social(grid, pw),
                                          no_go=self._no_go(grid))
            out["warnings"].extend(w for w in plan.get("warnings") or [] if w not in out["warnings"])
            if not plan["ok"]:
                out["ok"] = False
                out["reason"] = "no path to (%.2f, %.2f): %s" % (gx, gy, plan["reason"])
                return None
            v = self.planner.speeds(plan, speed, pw, zone)
            out["points"].extend(plan["points"])
            out["speeds"].extend(v)
            out["length_m"] += _path_len(plan["points"])
            return self.planner.eta(plan, v)

        target = None
        try:
            if st.op == "goto":
                target = (st.x, st.y, st.yaw)
            elif st.op == "goto_label":
                lb = self.world.resolve_label(st.label) if self.world is not None else None
                if lb is None:
                    out["ok"], out["reason"] = False, "unknown label %r" % st.label
                else:
                    target = lb
            elif st.op == "return_home":
                h = self.world.get_home() if self.world is not None else None
                if not h:
                    out["ok"], out["reason"] = False, "no home set"
                else:
                    target = (h["x"], h["y"], h.get("yaw"))
            elif st.op == "capture" and st.x is not None:
                target = (st.x, st.y, st.yaw)
            if target is not None:
                zone = getattr(st, "zone", None) or "fine"
                e = path_to(start[0], start[1], target[0], target[1], zone if st.op != "capture" else "fine")
                if e is not None:
                    out["eta_s"] = e + (self.cfg.fine_settle_s if zone == "fine" else 0.0)
                    out["end"] = [target[0], target[1], start[2] if target[2] is None else target[2]]
            elif st.op == "follow_path":
                pts = [[start[0], start[1]]] + [list(p) for p in st.points]
                pts = self.planner.smooth(pts) if st.smooth else densify(pts, 0.1)
                plan = {"points": pts, "ok": True, "raw": None, "reason": ""}
                v = self.planner.speeds(plan, st.speed, pw, st.zone or "fine")
                out.update(points=pts, speeds=v, length_m=_path_len(pts), eta_s=self.planner.eta(plan, v))
                out["end"] = [pts[-1][0], pts[-1][1], start[2]]
            elif st.op == "jump_to":
                jp = self.planner.jumps(grid, start[:2], (st.x, st.y), st.max_jumps, self.cfg.jump_len_m)
                out["landings"] = jp["landings"]
                out["ok"], out["reason"] = jp["ok"], jp["reason"]
                out["warnings"].append("jump_to: needs the dead-man held; JUMP_LEN_M=%.2f is an estimate"
                                       % self.cfg.jump_len_m)
                out["eta_s"] = len(jp["landings"]) * (self.cfg.jump_settle_s + 2.0)
                if jp["ok"]:
                    out["end"] = [st.x, st.y, start[2]]
            elif st.op == "patrol":
                try:
                    pts = self._patrol_points(st)
                except StepFailed as exc:
                    out["ok"], out["reason"] = False, str(exc)
                    pts = []
                cur = (start[0], start[1])
                tot = 0.0
                for loop in range(min(st.loops, 3)):     # preview at most 3 loops
                    for p in pts:
                        e = path_to(cur[0], cur[1], p[0], p[1], st.pass_zone or "z30")
                        if e is None:
                            break
                        tot += e
                        cur = (p[0], p[1])
                if st.loops > 3 and pts:
                    tot *= st.loops / 3.0
                    out["warnings"].append("patrol preview shows 3 of %d loops" % st.loops)
                out["eta_s"] = tot if out["ok"] else None
                out["end"] = [cur[0], cur[1], start[2]]
            elif st.op == "explore":
                out["warnings"].append("explore: open-ended (frontier exploration), ends at an unknown pose")
        except Exception as exc:
            out["ok"], out["reason"] = False, "preview error: %s" % exc
        return out

    def plan_mission(self, mission: S.Mission, start: Optional[Tuple[float, float, float]] = None
                     ) -> Dict[str, Any]:
        warnings: List[str] = []
        if start is None:
            try:
                p = self.get_pose()
            except Exception:
                p = None
            if p is None:
                warnings.append("robot pose unavailable: preview starts at (0, 0)")
                start = (0.0, 0.0, 0.0)
            else:
                start = (float(_get(p, "x") if not isinstance(p, (list, tuple)) else p[0]),
                         float(_get(p, "y") if not isinstance(p, (list, tuple)) else p[1]),
                         float((_get(p, "yaw") if not isinstance(p, (list, tuple)) else p[2]) or 0.0))
        try:
            grid = self.get_map()
        except Exception:
            grid = None
        if grid is None:
            warnings.append("map unavailable: straight-line preview, no obstacle check")
        try:
            pw = persons_to_world(self.get_persons() or [], start)
        except Exception:
            pw = []
        steps = []
        cur = start
        eta: Optional[float] = 0.0
        ok = True
        for i, st in enumerate(mission.steps):
            sp = self.plan_step(st, cur, grid, pw, i == len(mission.steps) - 1)
            sp["index"] = i
            steps.append(sp)
            ok = ok and sp["ok"]
            cur = tuple(sp["end"])
            if eta is not None:
                eta = None if sp["eta_s"] is None else eta + sp["eta_s"]
            for w in sp["warnings"]:
                if w not in warnings:
                    warnings.append(w)
        if mission.source == "nl":
            warnings.append("LLM-generated mission: needs operator approval")
        return {"ok": ok, "eta_s": eta, "eta_open_ended": eta is None, "warnings": warnings,
                "start": list(start), "steps": steps, "planner": "fallback" if self.planner.fallback else "planner"}


def _safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _load_nav_local() -> Any:
    try:
        from navigation import nav_local  # type: ignore
        return nav_local
    except Exception:
        try:
            import nav_local  # type: ignore
            return nav_local
        except Exception:
            return None


def _load_pursuit() -> Any:
    try:
        from omni import pursuit  # type: ignore
        return pursuit
    except Exception:
        try:
            import pursuit  # type: ignore
            return pursuit
        except Exception:
            return None


# ---------------------------------------------------------------- HTTP implementations

class HttpSender:
    """safety_guard client: /move, /stop, /action/{name}. Never raises."""

    def __init__(self, safety_url: str, timeout: float = 0.3, action_timeout: float = 10.0):
        self.url = safety_url.rstrip("/")
        self.timeout, self.action_timeout = timeout, action_timeout
        import requests
        self.s = requests.Session()

    def _post(self, path: str, body: dict, timeout: float) -> Tuple[int, Any]:
        try:
            r = self.s.post(self.url + path, json=body, timeout=timeout)
            try:
                data = r.json()
            except ValueError:
                data = r.text
            return r.status_code, data
        except Exception as exc:
            return 0, str(exc)

    def move(self, vx: float, vy: float, vyaw: float, profile: str = "normal") -> Tuple[int, Any]:
        return self._post("/move", {"vx": vx, "vy": vy, "vyaw": vyaw, "source": SOURCE, "profile": profile},
                          self.timeout)

    def stop(self) -> Tuple[int, Any]:
        return self._post("/stop", {"source": SOURCE}, self.timeout)

    def action(self, name: str) -> Tuple[int, Any]:
        return self._post("/action/%s" % name, {"source": SOURCE}, self.action_timeout)


class HttpServices:
    """audio /audio/speak, mapping /explore/*, omni /capture + /record/*."""

    def __init__(self, audio_url: str = "", mapping_url: str = "", omni_url: str = ""):
        import requests
        self.r = requests
        self.audio_url, self.mapping_url, self.omni_url = audio_url.rstrip("/"), mapping_url.rstrip("/"), \
            omni_url.rstrip("/")

    def _post(self, url: str, body: Optional[dict], timeout: float) -> Tuple[bool, Any]:
        try:
            r = self.r.post(url, json=body, timeout=timeout)
            try:
                data = r.json()
            except ValueError:
                data = r.text
            return 200 <= r.status_code < 300, data
        except Exception as exc:
            return False, str(exc)

    def speak(self, text: str) -> None:
        ok, data = self._post(self.audio_url + "/audio/speak", {"text": text}, 20.0)
        if not ok or (isinstance(data, dict) and data.get("error")):
            raise RuntimeError(_short(data))

    def explore_start(self, zone: Any = None) -> Tuple[bool, Any]:
        return self._post(self.mapping_url + "/explore/start", {"zone": zone} if zone is not None else None, 5.0)

    def explore_stop(self) -> Tuple[bool, Any]:
        return self._post(self.mapping_url + "/explore/stop", None, 5.0)

    def explore_status(self) -> Dict[str, Any]:
        r = self.r.get(self.mapping_url + "/explore/status", timeout=3.0)
        r.raise_for_status()
        return r.json()

    def capture(self, cams: Any, mode: str, label: Optional[str]) -> Tuple[bool, Any]:
        return self._post(self.omni_url + "/capture", {"cams": cams, "mode": mode, "label": label}, 30.0)

    def record(self, on: bool, mode: str, reason: str) -> Tuple[bool, Any]:
        if on:
            return self._post(self.omni_url + "/record/start", {"reason": reason, "mode": mode}, 5.0)
        return self._post(self.omni_url + "/record/stop", None, 5.0)


class _Cached:
    """Rate-limited HTTP getter: returns the last good value for `ttl` s."""

    def __init__(self, fn: Callable[[], Any], ttl: float, max_age: float):
        self.fn, self.ttl, self.max_age = fn, ttl, max_age
        self.val, self.t, self.ok_t = None, 0.0, 0.0
        self.lock = threading.Lock()

    def __call__(self) -> Any:
        with self.lock:
            now = time.time()
            if now - self.t >= self.ttl:
                self.t = now
                try:
                    self.val, self.ok_t = self.fn(), now
                except Exception:
                    pass
            return self.val if now - self.ok_t <= self.max_age else None


def make_http_executor(safety_url: str, core_url: str, mapping_url: str, omni_url: str, audio_url: str,
                       world: Any = None, publish: Any = None, log: Any = None,
                       cfg: Optional[ExecConfig] = None) -> MissionExecutor:
    import requests
    s = requests.Session()

    def pose():
        r = s.get(core_url.rstrip("/") + "/state", timeout=0.3)
        r.raise_for_status()
        p = r.json().get("pose")
        return None if not p else (float(p["x"]), float(p["y"]), float(p.get("yaw", 0.0)))

    def persons():
        r = s.get(omni_url.rstrip("/") + "/persons", timeout=0.3)
        r.raise_for_status()
        return r.json().get("persons") or []

    def get_map():
        r = s.get(mapping_url.rstrip("/") + "/map", timeout=5.0)
        r.raise_for_status()
        return r.json()

    def lidar():
        r = s.get(core_url.rstrip("/") + "/lidar_points", timeout=1.0)
        r.raise_for_status()
        return [(p["x"], p["y"], p["z"]) for p in r.json().get("points", [])]

    return MissionExecutor(
        sender=HttpSender(safety_url), get_pose=_Cached(pose, 0.04, 1.0), get_persons=_Cached(persons, 0.04, 0.5),
        get_map=_Cached(get_map, 2.0, 30.0), get_lidar=_Cached(lidar, 0.1, 0.6), world=world,
        services=HttpServices(audio_url, mapping_url, omni_url), publish=publish, log=log, cfg=cfg)
