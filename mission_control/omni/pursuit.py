"""Shadow / pursuit controller: keep a person in view at a safe distance.

Security patrol use case: follow ("shadow") a selected person at
follow_dist (default 3.0 m), never closer than min_dist (2.5 m, CONTRACTS.md
section 0). Pure logic: `PursuitController.step()` only returns a velocity
command. That command MUST be sent to safety_guard POST /move (port 9115,
source="pursuit"), never directly to mc_motion: safety_guard applies the
person / LiDAR zones on top, which no mode can override.

States
  SHADOW  target fresh: P-control on range and bearing, accel-limited.
          Inside min_dist the forward component is <= 0 (back off or stop).
  SEARCH  target lost < search_timeout_s: no translation, rotate toward the
          last known bearing, then keep scanning slowly in that direction.
  LOST    no target / lost longer than search_timeout_s: (0, 0, 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

SHADOW, SEARCH, LOST = "SHADOW", "SEARCH", "LOST"


@dataclass
class PursuitConfig:
    follow_dist: float = 3.0          # m, distance to hold
    min_dist: float = 2.5             # m, hard minimum (contract)
    max_v: float = 1.0                # m/s forward
    max_back_v: float = 0.3           # m/s backward (backing off)
    max_vyaw: float = 0.8             # rad/s
    kp_dist: float = 0.6              # (m/s) per m of range error
    kp_yaw: float = 1.2               # (rad/s) per rad of bearing
    accel: float = 0.5                # m/s^2 linear (speed-up)
    decel: float = 1.5                # m/s^2 linear (slow-down)
    yaw_accel: float = 1.5            # rad/s^2
    turn_first_rad: float = 0.8       # |bearing| above this: rotate only
    approach_lookahead_s: float = 1.0 # forward speed <= (d - min_dist) / this
    fresh_s: float = 0.5              # target older than this is "lost"
    search_timeout_s: float = 10.0
    search_vyaw: float = 0.4          # rad/s scan rate
    min_conf: float = 0.3             # select_target filter


def _get(p: Any, key: str, default: Any = None) -> Any:
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _finite(*vals: Any) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def select_target(persons: Optional[Iterable[Any]], policy: str = "nearest",
                  gid: Optional[int] = None, min_conf: float = 0.0) -> Optional[Any]:
    """Pick the person to shadow.

    policy "gid":     exactly the track with this gid, else None (never
                      silently switches to another person).
    policy "nearest": the closest track with conf >= min_conf.
    """
    cands = [p for p in (persons or []) if _finite(_get(p, "x"), _get(p, "y"))]
    if policy == "gid":
        if gid is None:
            return None
        for p in cands:
            if _get(p, "gid") == gid:
                return p
        return None
    if policy != "nearest":
        raise ValueError("unknown policy %r" % policy)
    cands = [p for p in cands if float(_get(p, "conf", 1.0) or 0.0) >= min_conf]
    if not cands:
        return None
    return min(cands, key=lambda p: math.hypot(float(_get(p, "x")), float(_get(p, "y"))))


def _odom_yaw(odom: Any) -> Optional[float]:
    if odom is None:
        return None
    y = _get(odom, "yaw")
    return float(y) if _finite(y) else None


class PursuitController:
    def __init__(self, cfg: Optional[PursuitConfig] = None) -> None:
        self.cfg = cfg or PursuitConfig()
        self.reset()

    def reset(self) -> None:
        self.state = LOST
        self.vx = 0.0
        self.vyaw = 0.0
        self.last_t: Optional[float] = None
        self.last_seen_t: Optional[float] = None
        self.last_bearing = 0.0                  # base frame at last sighting
        self.last_world_bearing: Optional[float] = None
        self.aligned = False
        self.last_range: Optional[float] = None

    # -- helpers ---------------------------------------------------------------
    def _ramp(self, cur: float, want: float, dt: float, up: float, down: float) -> float:
        """Rate limit; slowing toward zero uses the (larger) `down` limit."""
        toward_zero = abs(want) < abs(cur) or (want * cur < 0)
        lim = (down if toward_zero else up) * dt
        return cur + max(-lim, min(lim, want - cur))

    def _target_fresh(self, target: Any, now: float) -> bool:
        if target is None or not _finite(_get(target, "x"), _get(target, "y")):
            return False
        seen = _get(target, "last_seen_t")
        if seen is None or not _finite(seen) or float(seen) <= 0.0:
            return True                          # no timestamp: trust caller
        return now - float(seen) <= self.cfg.fresh_s

    # -- main ------------------------------------------------------------------
    def step(self, target: Any, odom: Any = None, now: Optional[float] = None
             ) -> Tuple[float, float, float, str]:
        """target: PersonTrack (base frame) or None. odom: optional dict with
        world yaw (for SEARCH bearing memory). Returns (vx, vy, vyaw, state)."""
        import time as _time
        now = _time.time() if now is None else float(now)
        c = self.cfg
        dt = 0.1 if self.last_t is None else min(max(now - self.last_t, 0.0), 0.5)
        self.last_t = now
        yaw = _odom_yaw(odom)

        if self._target_fresh(target, now):
            return self._shadow(target, now, dt, yaw)

        lost_for = None if self.last_seen_t is None else now - self.last_seen_t
        if lost_for is None or lost_for > c.search_timeout_s:
            self.state = LOST
            self.vx, self.vyaw = 0.0, 0.0
            return 0.0, 0.0, 0.0, LOST
        return self._search(dt, yaw)

    def _shadow(self, target: Any, now: float, dt: float, yaw: Optional[float]
                ) -> Tuple[float, float, float, str]:
        c = self.cfg
        x, y = float(_get(target, "x")), float(_get(target, "y"))
        d = math.hypot(x, y)
        bearing = math.atan2(y, x)
        self.state = SHADOW
        self.last_seen_t = now
        self.last_bearing = bearing
        self.last_range = d
        self.last_world_bearing = None if yaw is None else _wrap(yaw + bearing)
        self.aligned = False

        want_vyaw = max(-c.max_vyaw, min(c.max_vyaw, c.kp_yaw * bearing))
        want_vx = c.kp_dist * (d - c.follow_dist)
        if want_vx > 0.0:
            # only drive forward when roughly facing the target
            if abs(bearing) > c.turn_first_rad:
                want_vx = 0.0
            else:
                want_vx *= max(math.cos(bearing), 0.0)
        want_vx = max(-c.max_back_v, min(c.max_v, want_vx))

        vx = self._ramp(self.vx, want_vx, dt, c.accel, c.decel)
        vyaw = self._ramp(self.vyaw, want_vyaw, dt, c.yaw_accel, c.yaw_accel)
        vx = self._safety_cap(vx, d, bearing)
        self.vx, self.vyaw = vx, vyaw
        return vx, 0.0, vyaw, SHADOW

    def _safety_cap(self, vx: float, d: float, bearing: float) -> float:
        """Hard caps applied AFTER smoothing (they beat the accel limit):
        the approach speed toward the target (vx * cos(bearing)) must not
        bring the robot inside min_dist within approach_lookahead_s, and is
        <= 0 once inside min_dist."""
        c = self.cfg
        vx = max(-c.max_back_v, min(c.max_v, vx))
        cosb = math.cos(bearing)
        approach = vx * cosb          # > 0: range to the target shrinks
        if approach > 0.0:
            room = max(d - c.min_dist, 0.0)
            max_approach = room / c.approach_lookahead_s
            if approach > max_approach:
                vx = math.copysign(max_approach / abs(cosb), vx) if room > 0.0 else 0.0
        return vx

    def _search(self, dt: float, yaw: Optional[float]) -> Tuple[float, float, float, str]:
        c = self.cfg
        self.state = SEARCH
        if yaw is not None and self.last_world_bearing is not None:
            err = _wrap(self.last_world_bearing - yaw)
            if abs(err) < math.radians(10.0):
                self.aligned = True
            direction = math.copysign(1.0, err) if not self.aligned else math.copysign(1.0, self.last_bearing or 1.0)
        else:
            direction = math.copysign(1.0, self.last_bearing or 1.0)
        want_vyaw = direction * c.search_vyaw
        self.vx = 0.0           # never translate blind while searching
        self.vyaw = self._ramp(self.vyaw, want_vyaw, dt, c.yaw_accel, c.yaw_accel)
        return 0.0, 0.0, self.vyaw, SEARCH
