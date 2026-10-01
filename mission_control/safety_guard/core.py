"""safety_guard decision core -- pure, unit-tested (no I/O, no clock).

`decide()` takes the latest person tracks (base frame), the LiDAR proximity
in the commanded motion direction, the requested velocity command and the
previous decision (for hysteresis) and returns a SafetyDecision whose
`cmd_out` is the command that may be forwarded to mc_motion.

Invariants (tested in tests/test_safety_guard.py):
  - monotonic: |cmd_out| never exceeds |cmd| per component (vx, vy, vyaw);
    the guard only clamps or zeroes, it never creates or amplifies motion.
  - STOP: vx = vy = 0; only rotation (|vyaw| <= stop_vyaw_max) is allowed,
    plus slow backing away from a person in front when the rear is clear.
  - fail-safe: no / stale person data -> at least SLOW; LiDAR missing or an
    obstacle closer than lidar_stop_m in the motion direction -> STOP.
  - hysteresis: a more restrictive level applies at once; a less
    restrictive one only after it has been computed continuously for
    relax_hold_s.

Person zones (CONTRACTS.md, section D), measured as horizontal range in the
base frame, shrunk ahead of the commanded motion ("zones stretched in the
direction of motion"):
    STOP    < 0.8 m                       vx = vy = 0
    SLOW    < 2.0 m  vmax 0.1 -> 0.4 m/s linearly over 0.8 .. 2.0 m
    CAUTION < 3.5 m  vmax 0.6 m/s
    CLEAR            vmax clear_vmax
Child (height < 1.3 m) or fast (|v| > 2 m/s) person: thresholds x 1.5.
TTC (constant velocity, 2 s horizon): < 1 s STOP, < 2 s SLOW.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

CLEAR, CAUTION, SLOW, STOP = "CLEAR", "CAUTION", "SLOW", "STOP"
LEVELS = (CLEAR, CAUTION, SLOW, STOP)
_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


@dataclass
class SafetyConfig:
    # person zones (m)
    stop_m: float = 0.8
    slow_m: float = 2.0
    caution_m: float = 3.5
    # speed limits per zone (m/s, linear speed |(vx, vy)|)
    slow_vmin: float = 0.1          # at the STOP/SLOW border
    slow_vmax: float = 0.4          # at the SLOW/CAUTION border
    caution_vmax: float = 0.6
    clear_vmax: float = 1.0
    # yaw rate limits per level (rad/s)
    clear_vyaw_max: float = 1.2
    caution_vyaw_max: float = 1.0
    slow_vyaw_max: float = 0.8
    stop_vyaw_max: float = 0.5
    # zone stretch along the commanded motion: a person straight ahead is
    # treated as stretch_s * |v_cmd| metres closer (cos-weighted, 0 behind).
    stretch_s: float = 1.0
    # child / fast person
    child_height_m: float = 1.3
    fast_speed_mps: float = 2.0
    vulnerable_factor: float = 1.5
    # PersonTrack.z semantics are not fixed by the contract. A track may carry
    # an explicit `height_m`; otherwise, with child_from_z, `z + z_offset_m` is
    # used as the person's height above ground when z > min_valid_z.
    child_from_z: bool = False  # omni publishes height_m; z is the body centroid
    z_offset_m: float = 0.0
    min_valid_z: float = 0.3
    # TTC
    ttc_horizon_s: float = 2.0
    ttc_stop_s: float = 1.0
    ttc_slow_s: float = 2.0
    ttc_radius_m: float = 0.8       # "collision" = predicted distance below this
    ttc_slow_vmax: float = 0.25
    # data freshness
    persons_stale_s: float = 0.3
    stale_vmax: float = 0.3         # SLOW cap when person data is old/missing
    # LiDAR
    lidar_stop_m: float = 0.5
    # backing away in STOP
    allow_backoff: bool = True
    backoff_vmax: float = 0.2
    rear_clear_min_m: float = 0.8
    # hysteresis
    relax_hold_s: float = 0.5
    # follow / pursuit sources (CONTRACTS.md section 0): never approach any
    # person closer than this; applied when decide(min_person_dist_m=...)
    follow_min_m: float = 2.5
    # below this the linear command counts as "not moving" (no stretch)
    eps_v: float = 1e-3
    # -- mission sprint (mission/CONTRACT.md section 4) ------------------------
    # Only source "mission" + profile "sprint", with a live dead-man, level
    # CLEAR, battery known and > sprint_battery_min_pct, fresh LiDAR + person
    # data and nobody in the corridor ahead. Env alias: SPRINT_VMAX.
    sprint_vmax: float = 1.5
    sprint_battery_min_pct: float = 30.0
    sprint_corridor_half_w_m: float = 1.5
    sprint_corridor_len_m: float = 8.0
    # -- gated actions (jump / pounce) ------------------------------------------
    action_person_min_m: float = 3.5     # nobody within this range (any direction)
    action_still_s: float = 0.5          # |cmd| below action_still_eps for this long
    action_still_eps: float = 0.05
    action_front_clear_m: float = 1.0    # LiDAR free distance ahead for a jump
    dance_person_min_m: float = 2.0      # dance: nobody within this range

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "SafetyConfig":
        """SAFETY_<FIELD upper> overrides, e.g. SAFETY_STOP_M=1.0."""
        import os
        env = os.environ if env is None else env
        cfg = cls()
        for name, f in cls.__dataclass_fields__.items():
            key = "SAFETY_" + name.upper()
            if key not in env:
                continue
            raw = env[key]
            cur = getattr(cfg, name)
            if isinstance(cur, bool):
                setattr(cfg, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
            else:
                setattr(cfg, name, float(raw))
        if "SAFETY_SPRINT_VMAX" not in env and "SPRINT_VMAX" in env:
            cfg.sprint_vmax = float(env["SPRINT_VMAX"])
        return cfg


@dataclass
class SafetyDecision:
    level: str
    vmax: float
    vyaw_max: float
    reason: str
    cmd_out: Tuple[float, float, float]
    raw_level: str = CLEAR
    nearest_person_m: Optional[float] = None
    ttc_s: Optional[float] = None
    relax_since: Optional[float] = None   # hysteresis bookkeeping
    t: float = 0.0
    backoff: bool = False
    reasons: List[str] = field(default_factory=list)
    sprint: bool = False                  # vmax raised to sprint_vmax

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cmd_out"] = {"vx": self.cmd_out[0], "vy": self.cmd_out[1], "vyaw": self.cmd_out[2]}
        return d


# ---------------------------------------------------------------------------
# helpers

def _get(p: Any, key: str, default: Any = None) -> Any:
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def sanitize_cmd(cmd: Any) -> Tuple[float, float, float]:
    """Any (vx, vy, vyaw)-like input -> finite floats; garbage -> 0."""
    try:
        vals = list(cmd)[:3]
    except TypeError:
        vals = []
    vals += [0.0] * (3 - len(vals))
    out = []
    for v in vals:
        f = _num(v)
        out.append(0.0 if f is None else f)
    return out[0], out[1], out[2]


def _more_restrictive(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def slow_vmax_at(d_eff: float, stop_m: float, slow_m: float, cfg: SafetyConfig) -> float:
    """Linear 0.1 -> 0.4 m/s over the SLOW band [stop_m, slow_m]."""
    span = max(slow_m - stop_m, 1e-6)
    a = min(max((d_eff - stop_m) / span, 0.0), 1.0)
    return cfg.slow_vmin + a * (cfg.slow_vmax - cfg.slow_vmin)


def time_to_collision(px: float, py: float, rvx: float, rvy: float,
                      radius: float, horizon: float) -> Optional[float]:
    """First t in [0, horizon] with |p + rv t| < radius, else None.

    p: person position relative to robot, rv: person velocity relative to the
    robot (both base frame, constant velocity model)."""
    c = px * px + py * py - radius * radius
    if c <= 0.0:
        return 0.0
    a = rvx * rvx + rvy * rvy
    b = 2.0 * (px * rvx + py * rvy)
    if a < 1e-9 or b >= 0.0:            # not closing in
        return None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    return t if 0.0 <= t <= horizon else None


def is_vulnerable(p: Any, cfg: SafetyConfig) -> bool:
    """Child (short) or fast-moving person -> zones x vulnerable_factor.
    Fleet robot tracks (kind == "robot") never are: their bubble is built in
    by fleet_tracks()."""
    if _get(p, "kind") == "robot":
        return False
    vx, vy = _num(_get(p, "vx", 0.0)) or 0.0, _num(_get(p, "vy", 0.0)) or 0.0
    if math.hypot(vx, vy) > cfg.fast_speed_mps:
        return True
    h = _num(_get(p, "height_m"))
    if h is not None and h <= 0.0:
        h = None  # 0 = unknown (PersonTrack default)
    if h is None and cfg.child_from_z:
        z = _num(_get(p, "z"))
        if z is not None and z > cfg.min_valid_z:
            h = z + cfg.z_offset_m
    return h is not None and 0.0 < h < cfg.child_height_m


def _robot_velocity(cmd: Tuple[float, float, float], odom: Any) -> List[Tuple[float, float]]:
    """Velocities to test TTC against: the command and, if odom reports a
    measured base-frame velocity, that one as well (robot still braking)."""
    vels = [(cmd[0], cmd[1])]
    if odom is not None:
        mvx, mvy = _num(_get(odom, "vx")), _num(_get(odom, "vy"))
        if mvx is not None and mvy is not None:
            vels.append((mvx, mvy))
    return vels


def _backoff_allowed(vx: float, plist: list, person_stop: bool, stop_persons_front: bool,
                     hard_stop: bool, raw_level: str, rear_clear_m: Any,
                     cfg: SafetyConfig) -> bool:
    """Backing straight away (vx < 0, vy forced 0) out of STOP is allowed only
    when the STOP comes from persons, all of them in front, the rear is known
    to be clear and nobody is close behind the robot. A LiDAR-caused STOP
    (missing scan or obstacle) never allows it."""
    if not cfg.allow_backoff or vx >= 0.0 or hard_stop:
        return False
    if raw_level == STOP and not (person_stop and stop_persons_front):
        return False
    rc = _num(rear_clear_m)
    if rc is None or rc < cfg.rear_clear_min_m:
        return False
    for p in plist:
        px, py = _num(_get(p, "x")), _num(_get(p, "y"))
        if px is None or py is None:
            return False
        if px <= 0.0 and math.hypot(px, py) < cfg.slow_m * cfg.vulnerable_factor:
            return False
    return True


# ---------------------------------------------------------------------------
# the decision

def decide(persons: Optional[Iterable[Any]], lidar_near_m: Optional[float], odom: Any,
           cmd: Any, now: float, persons_t: Optional[float],
           cfg: Optional[SafetyConfig] = None, rear_clear_m: Optional[float] = None,
           prev: Optional[SafetyDecision] = None,
           min_person_dist_m: Optional[float] = None,
           sprint: bool = False) -> SafetyDecision:
    """Pure safety decision.

    persons      PersonTrack objects or their dicts (base frame); None = no data
    lidar_near_m nearest LiDAR return in the commanded motion direction (m);
                 None = LiDAR missing/stale -> STOP
    odom         optional dict/object; only an optional measured base-frame
                 velocity (vx, vy) is used, for TTC
    cmd          requested (vx, vy, vyaw)
    now          current time (s, same clock as persons_t)
    persons_t    timestamp of the person data; None = never received
    rear_clear_m free distance behind the robot (m); None = unknown (no backoff)
    prev         previous SafetyDecision (hysteresis); None = first call
    min_person_dist_m  follow/pursuit sources: linear motion that closes in on
                 any person nearer than this is zeroed (rotation and moving
                 away stay allowed)
    sprint       the caller-side sprint preconditions (sprint_preconditions())
                 hold. vmax is raised from clear_vmax to sprint_vmax only if
                 the final AND raw level are CLEAR and the corridor ahead is
                 empty; otherwise the decision is exactly the non-sprint one.
    """
    cfg = cfg or SafetyConfig()
    vx, vy, vyaw = sanitize_cmd(cmd)
    v_lin = math.hypot(vx, vy)
    moving = v_lin > cfg.eps_v
    mdir = (vx / v_lin, vy / v_lin) if moving else (0.0, 0.0)

    level, vmax, reasons = CLEAR, cfg.clear_vmax, []
    nearest: Optional[float] = None
    best_ttc: Optional[float] = None
    stop_persons_front = True          # every person causing STOP is in front
    person_stop = False
    hard_stop = False                  # STOP not caused by a person (no backoff)

    def bump(lvl: str, lim: float, why: str) -> None:
        nonlocal level, vmax
        level = _more_restrictive(level, lvl)
        vmax = min(vmax, lim)
        reasons.append(why)

    # -- LiDAR ---------------------------------------------------------------
    lid = _num(lidar_near_m)
    if lid is None:
        bump(STOP, 0.0, "lidar missing")
        hard_stop = True
    elif lid < cfg.lidar_stop_m and moving:
        bump(STOP, 0.0, "obstacle %.2fm in motion direction" % lid)
        hard_stop = True

    # -- person data freshness ----------------------------------------------
    pt = _num(persons_t)
    if persons is None or pt is None:
        bump(SLOW, cfg.stale_vmax, "no person data")
        plist: list = []
    else:
        plist = list(persons)
        age = now - pt
        if age > cfg.persons_stale_s:
            bump(SLOW, cfg.stale_vmax, "person data stale (%.2fs)" % age)

    # -- per person zones + TTC ---------------------------------------------
    rob_vels = _robot_velocity((vx, vy, vyaw), odom)
    for p in plist:
        px, py = _num(_get(p, "x")), _num(_get(p, "y"))
        if px is None or py is None:
            bump(SLOW, cfg.stale_vmax, "malformed person track")
            continue
        pvx = _num(_get(p, "vx", 0.0)) or 0.0
        pvy = _num(_get(p, "vy", 0.0)) or 0.0
        d = math.hypot(px, py)
        nearest = d if nearest is None else min(nearest, d)
        f = cfg.vulnerable_factor if is_vulnerable(p, cfg) else 1.0
        tag = " (child/fast)" if f != 1.0 else ""

        d_eff = d
        if moving and d > 1e-6:
            cos_a = (px * mdir[0] + py * mdir[1]) / d
            if cos_a > 0.0:
                d_eff = d - cfg.stretch_s * v_lin * cos_a
        stop_m, slow_m, caution_m = cfg.stop_m * f, cfg.slow_m * f, cfg.caution_m * f
        if d_eff < stop_m:
            bump(STOP, 0.0, "person %.2fm (eff %.2fm)%s" % (d, d_eff, tag))
            person_stop = True
            if px <= 0.0:
                stop_persons_front = False
        elif d_eff < slow_m:
            bump(SLOW, slow_vmax_at(d_eff, stop_m, slow_m, cfg),
                 "person %.2fm (eff %.2fm)%s" % (d, d_eff, tag))
        elif d_eff < caution_m:
            bump(CAUTION, cfg.caution_vmax, "person %.2fm%s" % (d, tag))

        for rvx_r, rvy_r in rob_vels:
            ttc = time_to_collision(px, py, pvx - rvx_r, pvy - rvy_r,
                                    cfg.ttc_radius_m * f, cfg.ttc_horizon_s)
            if ttc is None:
                continue
            best_ttc = ttc if best_ttc is None else min(best_ttc, ttc)
            if ttc < cfg.ttc_stop_s:
                bump(STOP, 0.0, "TTC %.2fs" % ttc)
                person_stop = True
                if px <= 0.0:
                    stop_persons_front = False
            elif ttc < cfg.ttc_slow_s:
                bump(SLOW, cfg.ttc_slow_vmax, "TTC %.2fs" % ttc)

    raw_level, raw_vmax = level, vmax

    # -- hysteresis -----------------------------------------------------------
    relax_since: Optional[float] = None
    if prev is not None and _RANK[raw_level] < _RANK[prev.level]:
        relax_since = prev.relax_since if prev.relax_since is not None else now
        if now - relax_since < cfg.relax_hold_s:
            level = prev.level
            vmax = min(raw_vmax, prev.vmax)
            reasons.append("hysteresis hold %s" % prev.level)
        else:
            relax_since = None
    vyaw_max = {CLEAR: cfg.clear_vyaw_max, CAUTION: cfg.caution_vyaw_max,
                SLOW: cfg.slow_vyaw_max, STOP: cfg.stop_vyaw_max}[level]

    # -- sprint (only ever raises the CLEAR cap; every other path unchanged) ---
    sprinting = False
    if sprint:
        if level == CLEAR and raw_level == CLEAR:
            blockers = persons_in_corridor(plist, (vx, vy), cfg.sprint_corridor_half_w_m,
                                           cfg.sprint_corridor_len_m)
            if blockers:
                reasons.append("sprint denied: %d person(s) in corridor" % len(blockers))
            else:
                vmax = max(vmax, cfg.sprint_vmax)
                sprinting = True
                reasons.append("sprint")
        else:
            reasons.append("sprint denied: level %s" % level)

    # -- clamp (monotonic) ----------------------------------------------------
    backoff = False
    if level == STOP:
        ovx, ovy = 0.0, 0.0
        if _backoff_allowed(vx, plist, person_stop, stop_persons_front, hard_stop,
                            raw_level, rear_clear_m, cfg):
            lim = cfg.backoff_vmax if raw_level == STOP else min(cfg.backoff_vmax, raw_vmax)
            ovx = max(vx, -lim)                     # |ovx| <= |vx|
            backoff = True
            reasons.append("backing away from person in front")
        vmax_out = cfg.backoff_vmax if backoff else 0.0
    else:
        scale = 1.0 if v_lin <= vmax else (vmax / v_lin if v_lin > 0 else 0.0)
        ovx, ovy = vx * scale, vy * scale
        vmax_out = vmax
    ovyaw = max(-vyaw_max, min(vyaw_max, vyaw))

    md = _num(min_person_dist_m)
    if md is not None and (ovx != 0.0 or ovy != 0.0):
        for p in plist:
            if _get(p, "kind") == "robot":             # follow rule is for people only
                continue
            px, py = _num(_get(p, "x")), _num(_get(p, "y"))
            if px is None or py is None:
                continue
            d = math.hypot(px, py)
            if d < md and (d < 1e-6 or (ovx * px + ovy * py) / d > 1e-6):
                ovx, ovy = 0.0, 0.0
                reasons.append("follow min distance %.1fm (person %.2fm)" % (md, d))
                break

    if not reasons:
        reasons.append("clear")
    return SafetyDecision(level=level, vmax=vmax_out,
                          vyaw_max=vyaw_max, reason=reasons[0] if len(reasons) == 1 else "; ".join(reasons[:3]),
                          cmd_out=(ovx, ovy, ovyaw), raw_level=raw_level,
                          nearest_person_m=None if nearest is None else round(nearest, 3),
                          ttc_s=None if best_ttc is None else round(best_ttc, 3),
                          relax_since=relax_since, t=now, backoff=backoff, reasons=reasons,
                          sprint=sprinting)


# ---------------------------------------------------------------------------
# sprint / action / dead-man helpers (pure)

def persons_in_corridor(persons: Optional[Iterable[Any]], direction: Tuple[float, float],
                        half_w: float, length: float) -> list:
    """Persons (base frame) inside the rectangle |lateral| <= half_w,
    0 <= forward <= length along `direction` (the commanded linear motion;
    zero -> robot +x). A malformed track counts as inside (fail-safe)."""
    dx, dy = _num(direction[0]) or 0.0, _num(direction[1]) or 0.0
    n = math.hypot(dx, dy)
    ux, uy = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
    out = []
    for p in persons or ():
        px, py = _num(_get(p, "x")), _num(_get(p, "y"))
        if px is None or py is None:
            out.append(p)
            continue
        fwd = px * ux + py * uy
        lat = -px * uy + py * ux
        if 0.0 <= fwd <= length and abs(lat) <= half_w:
            out.append(p)
    return out


def battery_ok(battery_pct: Any, min_pct: float) -> Tuple[bool, str]:
    b = _num(battery_pct)
    if b is None or not 0.0 <= b <= 100.0:
        return False, "battery unknown"
    if b <= min_pct:
        return False, "battery %.0f%% <= %.0f%%" % (b, min_pct)
    return True, "battery %.0f%%" % b


def sprint_preconditions(source: str, profile: Optional[str], deadman_alive: bool,
                         battery_pct: Any, lidar_fresh: bool, persons_fresh: bool,
                         cfg: SafetyConfig,
                         sprint_sources: Tuple[str, ...] = ("mission",)) -> Tuple[bool, str]:
    """Caller-side sprint gate (the level/corridor part is in decide()).
    Returns (requested_and_allowed, reason). Not requested -> (False, "")."""
    if profile != "sprint":
        return False, ""
    if source not in sprint_sources:
        return False, "sprint denied: source %r not in %s" % (source, ",".join(sprint_sources))
    if not deadman_alive:
        return False, "sprint denied: dead-man not held"
    ok, why = battery_ok(battery_pct, cfg.sprint_battery_min_pct)
    if not ok:
        return False, "sprint denied: " + why
    if not lidar_fresh:
        return False, "sprint denied: lidar not fresh"
    if not persons_fresh:
        return False, "sprint denied: person data not fresh"
    return True, "sprint preconditions ok"


JUMP_ACTIONS = ("jump", "front_jump", "pounce", "front_pounce")
DANCE_ACTIONS = ("dance", "dance2")
GESTURE_ACTIONS = ("hello", "wave", "greet", "stretch", "sit", "stand", "stand_up", "lay_down",
                   "heart", "balance")
# flips, handstand, damp, ... are deliberately NOT here: not reachable via the guard.
ALLOWED_ACTIONS = GESTURE_ACTIONS + DANCE_ACTIONS + JUMP_ACTIONS


def action_check(name: str, deadman_alive: bool, persons: Optional[Iterable[Any]],
                 persons_fresh: bool, level: str, lidar_fresh: bool,
                 front_clear_m: Optional[float], still_for_s: float,
                 cfg: SafetyConfig) -> Tuple[bool, str]:
    """Gate a mc_motion action. Returns (allowed, reason)."""
    if name not in ALLOWED_ACTIONS:
        return False, "action %r not in allowlist" % (name,)
    if name in GESTURE_ACTIONS:
        return True, "gesture"
    plist = list(persons or ())

    def nearest() -> Optional[float]:
        best = None
        for p in plist:
            px, py = _num(_get(p, "x")), _num(_get(p, "y"))
            if px is None or py is None:
                return 0.0                       # malformed -> treat as touching
            d = math.hypot(px, py)
            best = d if best is None else min(best, d)
        return best

    if not persons_fresh:
        return False, "person data not fresh"
    near = nearest()
    if name in DANCE_ACTIONS:
        if near is not None and near < cfg.dance_person_min_m:
            return False, "person %.2fm < %.1fm" % (near, cfg.dance_person_min_m)
        return True, "dance clear"
    # jump / pounce
    if not deadman_alive:
        return False, "dead-man not held"
    if near is not None and near < cfg.action_person_min_m:
        return False, "person %.2fm < %.1fm" % (near, cfg.action_person_min_m)
    if level != CLEAR:
        return False, "level %s" % level
    if not lidar_fresh:
        return False, "lidar not fresh"
    fc = _num(front_clear_m)
    if fc is None or fc < cfg.action_front_clear_m:
        return False, "front clearance %s < %.1fm" % ("?" if fc is None else "%.2fm" % fc,
                                                      cfg.action_front_clear_m)
    if still_for_s < cfg.action_still_s:
        return False, "robot not still (%.2fs < %.1fs)" % (max(still_for_s, 0.0), cfg.action_still_s)
    return True, "jump clear"


def fleet_tracks(robots: Any, own_id: str, own_pose: Any, bubble_m: float = 1.5,
                 stop_m: float = 0.8, max_range_m: float = 10.0) -> list:
    """Other fleet robots (world frame, mc.fleet.robots) -> person-like base
    frame tracks with kind="robot". Each one is moved radially closer by
    (bubble_m - stop_m), so decide()'s person STOP zone (stop_m) fires at
    bubble_m true distance and SLOW/CAUTION/TTC scale with it. own_pose =
    (x, y, yaw) world; None -> no tracks. Malformed entries are skipped (the
    fleet feed is optional; LiDAR still sees the robot)."""
    if own_pose is None or not isinstance(robots, (list, tuple)):
        return []
    ox, oy, oyaw = (_num(v) for v in own_pose[:3])
    if ox is None or oy is None or oyaw is None:
        return []
    c, s = math.cos(oyaw), math.sin(oyaw)
    shift = max(bubble_m - stop_m, 0.0)
    out = []
    for r in robots:
        rid = str(_get(r, "id", ""))
        if not rid or rid == own_id:
            continue
        x, y = _num(_get(r, "x")), _num(_get(r, "y"))
        if x is None or y is None:
            continue
        rvx, rvy = _num(_get(r, "vx", 0.0)) or 0.0, _num(_get(r, "vy", 0.0)) or 0.0
        dx, dy = x - ox, y - oy
        bx, by = c * dx + s * dy, -s * dx + c * dy
        d = math.hypot(bx, by)
        if d > max_range_m:
            continue
        k = max(d - shift, 0.0) / d if d > 1e-6 else 0.0
        out.append({"gid": "robot:" + rid, "kind": "robot", "x": bx * k, "y": by * k, "z": 0.0,
                    "height_m": 0.0, "vx": c * rvx + s * rvy, "vy": -s * rvx + c * rvy,
                    "true_dist_m": round(d, 3)})
    return out


class DeadmanRegistry:
    """Heartbeats per client_id. Alive if any client beat within timeout_s.
    Pure: every call takes `now`. Bounded (max_clients) against abuse."""

    def __init__(self, timeout_s: float = 0.3, max_clients: int = 32) -> None:
        self.timeout_s = timeout_s
        self.max_clients = max_clients
        self.beats: dict = {}

    def beat(self, client_id: Any, now: float) -> bool:
        cid = str(client_id or "").strip()[:64]
        if not cid:
            return False
        if cid not in self.beats and len(self.beats) >= self.max_clients:
            self.prune(now)
            if len(self.beats) >= self.max_clients:
                return False
        self.beats[cid] = now
        return True

    def release(self, client_id: Any) -> None:
        self.beats.pop(str(client_id or "").strip()[:64], None)

    def prune(self, now: float, keep_s: float = 10.0) -> None:
        for cid in [c for c, t in self.beats.items() if now - t > keep_s or t > now + 1.0]:
            del self.beats[cid]

    def clients(self, now: float) -> List[str]:
        return sorted(c for c, t in self.beats.items() if 0.0 <= now - t < self.timeout_s)

    def alive(self, now: float) -> bool:
        return bool(self.clients(now))


# ---------------------------------------------------------------------------
# LiDAR helpers (pure; used by app.py)

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lidar_near_in_direction(points_base: Any, cmd: Any, cone_rad: float = 0.6,
                            z_min: float = -0.25, z_max: float = 1.5,
                            min_r: float = 0.05) -> Optional[float]:
    """Nearest horizontal range of base-frame points inside the z band and the
    cone around the commanded linear motion direction. Pure rotation / no
    linear motion -> nearest in all directions. None if points is None;
    float('inf') if no point qualifies."""
    if points_base is None:
        return None
    vx, vy, _ = sanitize_cmd(cmd)
    moving = math.hypot(vx, vy) > 1e-3
    heading = math.atan2(vy, vx)
    best = float("inf")
    for p in points_base:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if not (z_min <= z <= z_max):
            continue
        r = math.hypot(x, y)
        if r < min_r:
            continue
        if moving and abs(_wrap(math.atan2(y, x) - heading)) > cone_rad:
            continue
        best = min(best, r)
    return best


def rear_clearance(points_base: Any, cone_rad: float = 0.6, z_min: float = -0.25,
                   z_max: float = 1.5) -> Optional[float]:
    return lidar_near_in_direction(points_base, (-1.0, 0.0, 0.0), cone_rad, z_min, z_max)
