"""DWA-lite local planner (pure function, no I/O).

plan_local(pose, goal, grid, social_grid, persons_world, cfg) samples
(vx, vyaw) pairs inside a dynamic window around the current velocity,
rolls each one out for `horizon_s` (unicycle model) and scores it:

    cost = w_obstacle * obstacle proximity
         + w_social   * mean social cost along the path (0..1)
         + w_person   * predicted-person proximity
         + w_side     * passing a person on the wrong side
         + w_goal     * end distance to goal  + w_heading * end heading error
         + w_smooth   * change from the current command

Trajectories that enter a blocked cell (floor + inflation, caller-built, e.g.
astar.build_blocked_grid) or come within person_min_m of a constant-velocity
predicted person position are rejected. With nothing valid the planner
returns (0, 0). The output still goes through safety_guard /move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np


@dataclass
class LocalConfig:
    max_vx: float = 0.6
    min_vx: float = 0.0             # no reversing in the local planner
    max_vyaw: float = 1.0
    accel: float = 0.8              # m/s^2 (dynamic window)
    yaw_accel: float = 2.0          # rad/s^2
    window_dt: float = 0.5          # s of accel the window allows
    n_vx: int = 7
    n_vyaw: int = 15
    horizon_s: float = 1.5
    dt: float = 0.1
    person_min_m: float = 1.0       # reject closer predicted approach
    person_cost_m: float = 2.5      # proximity cost range
    obstacle_cost_radius_m: float = 0.3
    out_of_bounds_blocked: bool = True
    pass_side: Optional[str] = "right"   # robot passes persons on its right: person on robot's left
    pass_lateral_m: float = 1.2     # wanted lateral offset of a person being passed
    goal_tol_m: float = 0.15
    w_obstacle: float = 1.0
    w_social: float = 4.0
    w_person: float = 2.0
    w_side: float = 1.0
    w_goal: float = 1.0
    w_heading: float = 0.4
    w_smooth: float = 0.15
    cur_vx: float = 0.0             # current command (for window / smoothness)
    cur_vyaw: float = 0.0


def _get(p: Any, key: str, default: Any = None) -> Any:
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _pose3(pose: Any) -> Tuple[float, float, float]:
    if isinstance(pose, (tuple, list)):
        return float(pose[0]), float(pose[1]), float(pose[2])
    return float(_get(pose, "x")), float(_get(pose, "y")), float(_get(pose, "yaw"))


class _Grid:
    def __init__(self, grid: Any, cfg: LocalConfig) -> None:
        self.res = float(_get(grid, "resolution"))
        self.ox = float(_get(grid, "origin_x", 0.0))
        self.oy = float(_get(grid, "origin_y", 0.0))
        self.w = int(_get(grid, "width"))
        self.h = int(_get(grid, "height"))
        blocked = _get(grid, "blocked")
        if blocked is None:
            floor = np.asarray(_get(grid, "floor"), dtype=np.int16)
            blocked = floor >= 50
        self.blocked = np.asarray(blocked).astype(bool).reshape(self.h, self.w)
        self.oob = cfg.out_of_bounds_blocked

    def cells(self, xs: np.ndarray, ys: np.ndarray):
        gx = np.floor((xs - self.ox) / self.res).astype(np.int64)
        gy = np.floor((ys - self.oy) / self.res).astype(np.int64)
        inside = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)
        return gx, gy, inside

    def blocked_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        gx, gy, inside = self.cells(xs, ys)
        out = np.full(xs.shape, self.oob, dtype=bool)
        out[inside] = self.blocked[gy[inside], gx[inside]]
        return out

    def lookup(self, arr2d: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        gx, gy, inside = self.cells(xs, ys)
        out = np.zeros(xs.shape, dtype=np.float32)
        out[inside] = arr2d[gy[inside], gx[inside]]
        return out


def rollout(x: float, y: float, yaw: float, vx: np.ndarray, vyaw: np.ndarray,
            horizon_s: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unicycle rollouts for K (vx, vyaw) pairs -> xs, ys, yaws of shape (K, T)."""
    steps = int(round(horizon_s / dt))
    k = vx.shape[0]
    xs = np.empty((k, steps)); ys = np.empty((k, steps)); th = np.empty((k, steps))
    cx = np.full(k, x); cy = np.full(k, y); ct = np.full(k, yaw)
    for i in range(steps):
        ct = ct + vyaw * dt
        cx = cx + vx * np.cos(ct) * dt
        cy = cy + vx * np.sin(ct) * dt
        xs[:, i], ys[:, i], th[:, i] = cx, cy, ct
    return xs, ys, th


def plan_local(pose: Any, goal: Any, grid: Any, social_grid: Optional[Any] = None,
               persons_world: Optional[Iterable[Any]] = None,
               cfg: Optional[LocalConfig] = None) -> Tuple[float, float, dict]:
    """pose (x, y, yaw) world; goal (x, y) world; grid {resolution, origin_x,
    origin_y, width, height, blocked|floor}; social_grid uint8 width*height or
    None; persons_world dicts/objects with x, y, vx, vy (world).
    Returns (vx, vyaw, info)."""
    cfg = cfg or LocalConfig()
    x0, y0, yaw0 = _pose3(pose)
    gx_, gy_ = float(goal[0] if isinstance(goal, (tuple, list)) else _get(goal, "x")), \
        float(goal[1] if isinstance(goal, (tuple, list)) else _get(goal, "y"))
    dist0 = math.hypot(gx_ - x0, gy_ - y0)
    if dist0 <= cfg.goal_tol_m:
        return 0.0, 0.0, {"ok": True, "reached": True, "n_valid": 0}

    g = _Grid(grid, cfg)
    social = None
    if social_grid is not None:
        social = np.asarray(social_grid, dtype=np.float32).reshape(g.h, g.w) / 255.0

    # dynamic window
    dv, dw = cfg.accel * cfg.window_dt, cfg.yaw_accel * cfg.window_dt
    v_lo, v_hi = max(cfg.min_vx, cfg.cur_vx - dv), min(cfg.max_vx, cfg.cur_vx + dv)
    w_lo, w_hi = max(-cfg.max_vyaw, cfg.cur_vyaw - dw), min(cfg.max_vyaw, cfg.cur_vyaw + dw)
    vs = np.unique(np.concatenate([np.linspace(v_lo, v_hi, cfg.n_vx), [0.0] if v_lo <= 0 <= v_hi else []]))
    ws = np.unique(np.concatenate([np.linspace(w_lo, w_hi, cfg.n_vyaw), [0.0] if w_lo <= 0 <= w_hi else []]))
    VV, WW = np.meshgrid(vs, ws)
    vx, vyaw = VV.ravel(), WW.ravel()
    xs, ys, th = rollout(x0, y0, yaw0, vx, vyaw, cfg.horizon_s, cfg.dt)
    k, steps = xs.shape
    tgrid = (np.arange(steps) + 1) * cfg.dt

    # --- obstacles: reject entering a blocked cell (points that stay in the
    # start cell are ignored so a robot inside inflation can still rotate)
    gx0, gy0, _ = g.cells(np.array([x0]), np.array([y0]))
    pgx, pgy, _ = g.cells(xs, ys)
    same_cell = (pgx == gx0[0]) & (pgy == gy0[0])
    hit = g.blocked_at(xs, ys) & ~same_cell
    valid = ~hit.any(axis=1)

    # obstacle proximity: blocked samples on a small ring around the end points
    r = cfg.obstacle_cost_radius_m
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring_hits = np.zeros(k)
    for a in ang:
        ring_hits += g.blocked_at(xs + r * np.cos(a), ys + r * np.sin(a)).mean(axis=1)
    c_obst = ring_hits / len(ang)

    # --- social layer
    c_social = g.lookup(social, xs, ys).mean(axis=1) if social is not None else np.zeros(k)

    # --- persons (constant velocity prediction)
    c_person = np.zeros(k)
    c_side = np.zeros(k)
    side_sign = {"right": 1.0, "left": -1.0}.get(cfg.pass_side or "", 0.0)
    for p in persons_world or []:
        try:
            px, py = float(_get(p, "x")), float(_get(p, "y"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        pvx, pvy = float(_get(p, "vx", 0.0) or 0.0), float(_get(p, "vy", 0.0) or 0.0)
        ppx = px + pvx * tgrid                          # (T,)
        ppy = py + pvy * tgrid
        ddx, ddy = ppx[None, :] - xs, ppy[None, :] - ys
        dd = np.hypot(ddx, ddy)                         # (K, T)
        valid &= dd.min(axis=1) >= cfg.person_min_m
        c_person += np.clip(1.0 - dd.min(axis=1) / cfg.person_cost_m, 0.0, 1.0) ** 2
        if side_sign != 0.0:
            # lateral offset of the person at closest approach, robot frame
            # (+ = person on the robot's left). pass_side "right" wants the
            # person >= pass_lateral_m on the LEFT; driving straight at a
            # person ahead (lat ~ 0) or passing on the wrong side costs.
            i = dd.argmin(axis=1)
            rows = np.arange(k)
            lat = -np.sin(th[rows, i]) * ddx[rows, i] + np.cos(th[rows, i]) * ddy[rows, i]
            fwd0 = math.cos(yaw0) * (px - x0) + math.sin(yaw0) * (py - y0)
            near = dd[rows, i] < cfg.person_cost_m
            good = side_sign * lat
            pen = np.clip((cfg.pass_lateral_m - good) / cfg.pass_lateral_m, 0.0, 2.0)
            c_side += np.where(near & (fwd0 > 0.0), pen, 0.0)

    # --- goal
    ex, ey = xs[:, -1], ys[:, -1]
    # remaining distance in units of one full-speed rollout: full progress
    # over the horizon is worth 1.0 regardless of how far the goal is
    c_goal = np.hypot(gx_ - ex, gy_ - ey) / max(cfg.max_vx * cfg.horizon_s, 1e-6)
    c_head = np.abs(_wrap(np.arctan2(gy_ - ey, gx_ - ex) - th[:, -1])) / np.pi
    c_smooth = np.abs(vx - cfg.cur_vx) / max(cfg.max_vx, 1e-6) + \
        np.abs(vyaw - cfg.cur_vyaw) / max(cfg.max_vyaw, 1e-6)

    cost = (cfg.w_obstacle * c_obst + cfg.w_social * c_social + cfg.w_person * c_person
            + cfg.w_side * c_side + cfg.w_goal * c_goal + cfg.w_heading * c_head
            + cfg.w_smooth * c_smooth)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 0.0, {"ok": False, "reason": "no valid trajectory", "n_valid": 0, "n": int(k)}
    cost = np.where(valid, cost, np.inf)
    b = int(np.argmin(cost))
    info = {"ok": True, "reached": False, "n_valid": n_valid, "n": int(k), "cost": float(cost[b]),
            "terms": {"obstacle": float(c_obst[b]), "social": float(c_social[b]),
                      "person": float(c_person[b]), "side": float(c_side[b]),
                      "goal": float(c_goal[b]), "heading": float(c_head[b]),
                      "smooth": float(c_smooth[b])},
            "end": (float(ex[b]), float(ey[b]))}
    return float(vx[b]), float(vyaw[b]), info
