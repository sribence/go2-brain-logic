"""Fleet task allocation (pure functions, CONTRACT.md 9.6).

Robots are plain dicts (``Robot.summary()`` shape, map frame):
``{id, online, battery, safety_level, pose:{x,y,yaw}, mission:{state}|None,
   priority (of the fleet mission it runs, default 0), target:[x,y]|None, target_gid}``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MIN_BATTERY = 25.0
BATTERY_WEIGHT = 1.0          # cost *= 1 + w * (1 - batt/100)
PREEMPT_PENALTY_M = 3.0       # a busy (lower-priority) robot costs this much extra
SAME_TARGET_M = 1.5
STANDOFF_M = 2.5              # alert: stop this far from the target (shadow min dist)
BUSY_STATES = ("queued", "running", "paused", "waiting", "pending_approval")

PlanFn = Callable[[Dict[str, Any], Tuple[float, float]], Optional[float]]


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_busy(r: Dict[str, Any]) -> bool:
    m = r.get("mission")
    return bool(m and m.get("state") in BUSY_STATES)


def eligibility(r: Dict[str, Any], priority: int = 0, min_battery: float = MIN_BATTERY) -> Tuple[bool, str]:
    if not r.get("online"):
        return False, "offline"
    if r.get("enabled") is False:
        return False, "disabled"
    if not r.get("pose"):
        return False, "no_pose"
    b = _num(r.get("battery"))
    if b is None or b <= min_battery:
        return False, "battery"
    if (r.get("safety_level") or "").upper() == "STOP":
        return False, "safety_stop"
    if is_busy(r) and int(r.get("priority") or 0) >= int(priority):
        return False, "busy"
    return True, "ok"


def straight(r: Dict[str, Any], target: Tuple[float, float]) -> float:
    p = r["pose"]
    return math.hypot(float(p["x"]) - target[0], float(p["y"]) - target[1])


def score(r: Dict[str, Any], target: Tuple[float, float], plan_fn: Optional[PlanFn] = None,
          battery_weight: float = BATTERY_WEIGHT) -> Tuple[float, float, str]:
    """(cost, distance_m, metric). plan_fn(robot, target) -> path length or None (fallback straight)."""
    dist, metric = None, "straight"
    if plan_fn is not None:
        try:
            dist = _num(plan_fn(r, target))
        except Exception:  # noqa: BLE001
            dist = None
        if dist is not None:
            metric = "plan"
    if dist is None:
        dist = straight(r, target)
    b = max(0.0, min(100.0, _num(r.get("battery")) or 0.0))
    cost = dist * (1.0 + battery_weight * (1.0 - b / 100.0))
    if is_busy(r):
        cost += PREEMPT_PENALTY_M
    return cost, dist, metric


def same_target(r: Dict[str, Any], target: Tuple[float, float], gid: Any = None,
                radius: float = SAME_TARGET_M) -> bool:
    if gid is not None and r.get("target_gid") is not None and str(r.get("target_gid")) == str(gid):
        return True
    t = r.get("target")
    if not t or not is_busy(r):
        return False
    return math.hypot(float(t[0]) - target[0], float(t[1]) - target[1]) <= radius


def choose(robots: Sequence[Dict[str, Any]], target: Tuple[float, float], priority: int = 0,
           plan_fn: Optional[PlanFn] = None, swarm: bool = False, count: int = 1, gid: Any = None,
           min_battery: float = MIN_BATTERY, exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """Pick robot(s) for a target in map frame.

    Returns ``{robot_ids: [...], already: [ids on this target], ranking: [...], reason}``.
    Without ``swarm`` a target someone already works on is not doubled
    (``robot_ids`` empty, ``reason="already_assigned"``)."""
    target = (float(target[0]), float(target[1]))
    already = [r["id"] for r in robots if same_target(r, target, gid)]
    ranking = []
    for r in robots:
        okk, why = eligibility(r, priority, min_battery)
        if r["id"] in exclude:
            okk, why = False, "excluded"
        if okk and r["id"] in already:
            okk, why = False, "same_target"
        row = {"id": r["id"], "eligible": okk, "why": why}
        if okk:
            cost, dist, metric = score(r, target, plan_fn)
            row.update({"cost": round(cost, 3), "dist_m": round(dist, 3), "metric": metric,
                        "preempt": is_busy(r)})
        ranking.append(row)
    elig = sorted((x for x in ranking if x["eligible"]), key=lambda x: (x["cost"], x["id"]))
    ranking = elig + [x for x in ranking if not x["eligible"]]
    if already and not swarm:
        return {"robot_ids": [], "already": already, "ranking": ranking, "reason": "already_assigned"}
    n = max(1, int(count)) if swarm else 1
    ids = [x["id"] for x in elig[:n]]
    return {"robot_ids": ids, "already": already, "ranking": ranking,
            "reason": "ok" if ids else "no_robot_available"}


# ---------------------------------------------------------------------------
# missions
# ---------------------------------------------------------------------------
def mission_target(mission: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """First spatial point of a mission (map frame) -> allocation target."""
    for st in mission.get("steps") or []:
        if not isinstance(st, dict):
            continue
        x, y = _num(st.get("x")), _num(st.get("y"))
        if x is not None and y is not None:
            return x, y
        pts = st.get("points")
        if isinstance(pts, list) and pts and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
            return float(pts[0][0]), float(pts[0][1])
    return None


def standoff_point(robot_xy: Tuple[float, float], target: Tuple[float, float],
                   dist: float = STANDOFF_M) -> Tuple[float, float]:
    dx, dy = robot_xy[0] - target[0], robot_xy[1] - target[1]
    d = math.hypot(dx, dy)
    if d <= dist + 1e-6:
        return robot_xy
    return target[0] + dx / d * dist, target[1] + dy / d * dist


SHADOW_KINDS = ("intruder", "person", "thermal", "suspect", "shadow")


def alert_mission(alert: Dict[str, Any], robot: Dict[str, Any]) -> Dict[str, Any]:
    """Alert ``{x, y, kind, gid?, robot_id?}`` (map frame) -> mission for ``robot``.
    A ``gid`` is robot-local (omni track id): only the robot that saw it gets
    ``shadow``/``watch`` by gid; any other robot drives to a 2.5 m standoff,
    looks at the spot and scans."""
    tx, ty = float(alert["x"]), float(alert["y"])
    kind = str(alert.get("kind") or "alert")
    p = robot["pose"]
    gid = alert.get("gid")
    own = gid is not None and alert.get("robot_id") in (None, robot["id"])
    steps: List[Dict[str, Any]] = []
    if own:
        if kind in SHADOW_KINDS:
            steps.append({"op": "shadow", "gid": gid, "dist_m": 3.0, "timeout_s": 120})
        else:
            steps.append({"op": "watch", "gid": gid, "timeout_s": 120})
    else:
        sx, sy = standoff_point((float(p["x"]), float(p["y"])), (tx, ty))
        yaw = math.atan2(ty - sy, tx - sx)
        steps += [{"op": "goto", "x": round(sx, 3), "y": round(sy, 3), "yaw": round(yaw, 4),
                   "speed": "normal", "zone": "fine"},
                  {"op": "look_at", "x": round(tx, 3), "y": round(ty, 3)},
                  {"op": "scan", "deg": 90, "speed_dps": 30}]
    return {"name": "Riasztás: %s" % kind, "source": "api", "steps": steps, "on_fail": "stop"}
