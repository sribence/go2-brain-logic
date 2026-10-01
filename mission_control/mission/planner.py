"""Mission planner (M2): global path, smoothing, speed profile, jumps, follower.

Pure functions only -- no I/O, no robot, no HTTP. See mission/CONTRACT.md §3.

Grid input matches mapping's schema (CONVENTIONS.md): ``grid_meta =
{resolution, origin_x, origin_y, width, height}``; ``floor`` / ``walls`` /
``social`` / ``no_go_mask`` are flat row-major arrays of ``width*height``
(index = gy*width + gx). floor: -1 unknown / 0 free / 100 occupied; walls:
0..255 height bucket; social: uint8 0..255 cost; no_go_mask: bool (hard block).

Pieces:
- ``plan_path``      weighted 8-connected A* (octile heuristic) over the
                     inflated blocked grid (same semantics as
                     navigation/astar.build_blocked_grid, numpy-built) plus an
                     additive per-cell cost from the social layer / unknown
                     cells, then cost-aware line-of-sight shortcutting and
                     ``smooth_path``.
- ``smooth_path``    centripetal Catmull-Rom, resampled every ``ds``; segments
                     whose curve hits the inflated grid fall back to the line.
- ``speed_profile``  per-point v: profile vmax, lateral-accel curvature cap,
                     person caps, forward/backward accel passes, stop at goal.
- ``plan_jumps``     FrontJump landing sequence with landing-cell checks.
- ``follow_controller`` adaptive pure pursuit; ``blend_with_dwa`` mixes in
                     navigation/nav_local.plan_local's command near people.
- ``eta``.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # navigation/astar.py is reused, not copied
    import astar as _astar  # type: ignore
except ImportError:  # pragma: no cover - depends on sys.path of the caller
    for _p in (os.environ.get("NAVIGATION_DIR", ""),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "navigation"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "navigation")):
        if _p and os.path.isfile(os.path.join(_p, "astar.py")) and _p not in sys.path:
            sys.path.insert(0, _p)
            break
    import astar as _astar  # type: ignore

world_to_grid = _astar.world_to_grid
grid_to_world = _astar.grid_to_world
nearest_free_cell = _astar.nearest_free_cell

SQRT2 = math.sqrt(2.0)


def _envf(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v > 0 else default


SPRINT_VMAX = _envf("SPRINT_VMAX", 1.5)
JUMP_LEN_M = _envf("JUMP_LEN_M", 0.6)      # FrontJump forward distance -- MEASURE on the robot

PROFILES: Dict[str, Dict[str, float]] = {
    "stealth": {"vmax": 0.25, "amax": 0.3},
    "precise": {"vmax": 0.3, "amax": 0.4},
    "normal": {"vmax": 0.6, "amax": 0.6},
    "sprint": {"vmax": SPRINT_VMAX, "amax": 1.0},
}


@dataclass
class PlannerConfig:
    # --- global path
    robot_radius_m: float = 0.25          # Chebyshev inflation (as navigation/app.py)
    stair_wall_min: Optional[int] = None  # stair band for build_blocked_grid semantics (None = off)
    stair_wall_max: Optional[int] = None
    social_weight: float = 6.0            # extra cost per metre at social==255
    unknown_cost: float = 0.3             # extra cost per metre through unknown (-1) cells
    unknown_blocked: bool = False         # treat unknown as obstacle
    heuristic_weight: float = 1.0         # >1 = weighted A* (faster, suboptimal)
    snap_radius_m: float = 1.0            # goal/start inside obstacle -> nearest free cell within this
    max_expansions: int = 2_000_000
    coarse_factor: int = 2                # big maps: try A* on a k*k-downsampled grid first
    coarse_min_cells: int = 90_000        # ... when width*height >= this (falls back to full res)
    shortcut: bool = True
    shortcut_cost_tol: float = 1e-6       # shortcut only if not costlier than the A* sub-path
    smooth: bool = True
    ds: float = 0.1
    auto_blend_m: float = 0.5             # corner radius at A* turn points (zone mode)
    # --- speed profile
    a_lat: float = 0.5                    # m/s^2 lateral acceleration limit
    v_start: float = 0.0
    v_end: float = 0.0
    v_min: float = 0.05                   # creep floor used by eta / follower
    corridor_m: float = 1.5               # +- corridor around the path (persons)
    person_slow_m: float = 3.5            # person in corridor closer than this -> stealth cap
    fine_end: bool = True                 # last point is a "fine" stop (contract 9.1 default)
    fine_approach_m: float = 0.5          # precise-profile cap this far before a fine stop
    # --- jumps
    jump_clearance_m: float = 0.35
    jump_flat_wall_max: int = 40          # walls bucket <= this = flat (~0.1 m)
    jump_stair_wall_min: int = int(os.environ.get("NAV_STAIR_WALL_MIN", "150"))
    jump_stair_wall_max: int = int(os.environ.get("NAV_STAIR_WALL_MAX", "200"))
    jump_over_wall_max: int = 60          # obstacles in flight must be lower (~0.27 m)
    # --- follower
    lookahead_min: float = 0.4
    lookahead_max: float = 1.2
    lookahead_gain_s: float = 0.8         # L = gain * v, clipped
    max_vyaw: float = 1.0
    k_yaw: float = 1.5                    # in-place rotation gain
    rotate_first_rad: float = math.radians(60.0)
    goal_tol_m: float = 0.15
    search_window: int = 60               # nearest-point search ahead of idx


@dataclass
class Path:
    points: List[List[float]] = field(default_factory=list)
    length_m: float = 0.0
    ok: bool = False
    reason: str = ""
    cells_expanded: int = 0
    ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    stops: List[int] = field(default_factory=list)      # point indices of "fine" stops (9.1)

    def to_dict(self) -> dict:
        return {"points": [[round(float(x), 3), round(float(y), 3)] for x, y in self.points],
                "length_m": round(self.length_m, 3), "ok": self.ok, "reason": self.reason,
                "cells_expanded": self.cells_expanded, "ms": round(self.ms, 2),
                "warnings": list(self.warnings), "stops": list(self.stops)}


@dataclass
class JumpPlan:
    landings: List[List[float]] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)   # takeoff yaw per jump (rad)
    ok: bool = False
    reason: str = ""
    residual_m: float = 0.0                                # last landing -> goal
    rejected: List[Tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"landings": [[round(x, 3), round(y, 3)] for x, y in self.landings],
                "headings": [round(h, 4) for h in self.headings], "ok": self.ok,
                "reason": self.reason, "residual_m": round(self.residual_m, 3),
                "rejected": [list(r) for r in self.rejected]}


class VProfile(list):
    """list[float] (contract shape) + ``warnings`` and effective ``profile``."""

    def __init__(self, v: Iterable[float] = (), warnings: Optional[List[str]] = None,
                 profile: str = "") -> None:
        super().__init__(v)
        self.warnings = list(warnings or [])
        self.profile = profile


# --------------------------------------------------------------------- grid

def _g(m: Any, key: str, default: Any = None) -> Any:
    if isinstance(m, dict):
        return m.get(key, default)
    return getattr(m, key, default)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev (square) dilation by r cells, separable shifts."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    for d in range(1, r + 1):
        out[:, d:] |= mask[:, :-d]
        out[:, :-d] |= mask[:, d:]
    tmp = out.copy()
    for d in range(1, r + 1):
        out[d:, :] |= tmp[:-d, :]
        out[:-d, :] |= tmp[d:, :]
    return out


class PlanGrid:
    """Planning view of one map: blocked (inflated) and extra-cost arrays (h, w)."""

    def __init__(self, grid_meta: Any, floor: Any = None, walls: Any = None,
                 social: Any = None, no_go_mask: Any = None,
                 cfg: Optional[PlannerConfig] = None) -> None:
        cfg = cfg or PlannerConfig()
        self.res = float(_g(grid_meta, "resolution"))
        self.ox = float(_g(grid_meta, "origin_x", 0.0))
        self.oy = float(_g(grid_meta, "origin_y", 0.0))
        self.w = int(_g(grid_meta, "width"))
        self.h = int(_g(grid_meta, "height"))
        n = self.w * self.h
        if floor is None:
            floor = _g(grid_meta, "floor")
        if walls is None:
            walls = _g(grid_meta, "walls")
        self.floor = np.asarray(floor, dtype=np.int16).reshape(self.h, self.w)
        self.walls = (np.zeros((self.h, self.w), np.int16) if walls is None or len(walls) != n
                      else np.asarray(walls, dtype=np.int16).reshape(self.h, self.w))
        obstacle = self.floor == 100
        if cfg.unknown_blocked:
            obstacle = obstacle | (self.floor < 0)
        stair = np.zeros_like(obstacle)
        if cfg.stair_wall_min is not None and cfg.stair_wall_max is not None:
            stair = (self.walls >= cfg.stair_wall_min) & (self.walls <= cfg.stair_wall_max)
        r_cells = max(0, int(round(cfg.robot_radius_m / self.res)))
        blocked = obstacle | _dilate(obstacle & ~stair, r_cells)
        blocked &= ~(stair & (self.floor != 100))     # same rule as astar.build_blocked_grid
        if no_go_mask is not None:
            blocked |= np.asarray(no_go_mask, dtype=bool).reshape(self.h, self.w)
        self.blocked = blocked
        extra = np.zeros((self.h, self.w), np.float32)
        if social is not None:
            extra += np.asarray(social, dtype=np.float32).reshape(self.h, self.w) * (cfg.social_weight / 255.0)
        if cfg.unknown_cost > 0:
            extra += (self.floor < 0).astype(np.float32) * cfg.unknown_cost
        self.extra = extra

    def as_dict(self) -> dict:
        return {"resolution": self.res, "origin_x": self.ox, "origin_y": self.oy,
                "width": self.w, "height": self.h, "blocked": self.blocked.ravel()}

    def cells(self, xs: np.ndarray, ys: np.ndarray):
        gx = np.floor((np.asarray(xs) - self.ox) / self.res).astype(np.int64)
        gy = np.floor((np.asarray(ys) - self.oy) / self.res).astype(np.int64)
        inside = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)
        return gx, gy, inside

    def blocked_at(self, xs, ys) -> np.ndarray:
        gx, gy, inside = self.cells(xs, ys)
        out = np.ones(np.shape(gx), dtype=bool)
        out[inside] = self.blocked[gy[inside], gx[inside]]
        return out

    def seg_samples(self, a, b, step: Optional[float] = None):
        step = step or self.res * 0.25
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(math.ceil(d / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, d

    def seg_free(self, a, b) -> bool:
        xs, ys, _ = self.seg_samples(a, b)
        return not bool(self.blocked_at(xs, ys).any())

    def seg_cost(self, a, b) -> float:
        xs, ys, d = self.seg_samples(a, b)
        gx, gy, inside = self.cells(xs, ys)
        ex = np.zeros(xs.shape, np.float32)
        ex[inside] = self.extra[gy[inside], gx[inside]]
        return d * (1.0 + float(ex.mean()))


def _blocked_fn(grid: Any):
    """Accept PlanGrid or dict {resolution, origin_*, width, height, blocked}."""
    if grid is None:
        return None
    if isinstance(grid, PlanGrid):
        return grid.blocked_at
    res = float(_g(grid, "resolution"))
    ox, oy = float(_g(grid, "origin_x", 0.0)), float(_g(grid, "origin_y", 0.0))
    w, h = int(_g(grid, "width")), int(_g(grid, "height"))
    b = np.asarray(_g(grid, "blocked"), dtype=bool).reshape(h, w)

    def fn(xs, ys):
        gx = np.floor((np.asarray(xs) - ox) / res).astype(np.int64)
        gy = np.floor((np.asarray(ys) - oy) / res).astype(np.int64)
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        out = np.ones(np.shape(gx), dtype=bool)
        out[inside] = b[gy[inside], gx[inside]]
        return out
    return fn


# ------------------------------------------------------------ weighted A*

def weighted_astar(blocked: np.ndarray, extra: Optional[np.ndarray],
                   start: Tuple[int, int], goal: Tuple[int, int],
                   heuristic_weight: float = 1.0, max_expansions: int = 2_000_000
                   ) -> Tuple[Optional[List[Tuple[int, int]]], int, float]:
    """8-connected A* on a (h, w) bool grid; step cost = length * (1 + extra[to]).
    Octile heuristic; no diagonal corner cutting (as navigation/astar.py).
    Returns (cells [(gx, gy)] or None, expansions, cost)."""
    h, w = blocked.shape
    sx, sy = start
    tx, ty = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= tx < w and 0 <= ty < h):
        return None, 0, 0.0
    if blocked[sy, sx] or blocked[ty, tx]:
        return None, 0, 0.0
    if start == goal:
        return [start], 0, 0.0
    W2 = w + 2
    pad = np.ones((h + 2, W2), dtype=bool)
    pad[1:-1, 1:-1] = blocked
    blk = pad.ravel().tolist()
    if extra is not None:
        ep = np.zeros((h + 2, W2), dtype=np.float64)
        ep[1:-1, 1:-1] = extra
        mul = (1.0 + ep).ravel().tolist()
    else:
        mul = None
    N = (h + 2) * W2
    s = (sy + 1) * W2 + sx + 1
    t = (ty + 1) * W2 + tx + 1
    nbrs = ((1, 1.0, 0, 0), (-1, 1.0, 0, 0), (W2, 1.0, 0, 0), (-W2, 1.0, 0, 0),
            (W2 + 1, SQRT2, 1, W2), (W2 - 1, SQRT2, -1, W2),
            (-W2 + 1, SQRT2, 1, -W2), (-W2 - 1, SQRT2, -1, -W2))
    INF = float("inf")
    gs = [INF] * N
    came = [-1] * N
    closed = bytearray(N)
    hw = heuristic_weight * (1.0 + 1e-4)          # tiny tie-break towards the goal
    dk = SQRT2 - 2.0
    gs[s] = 0.0
    heap = [(0.0, s)]
    push, pop = heapq.heappush, heapq.heappop
    exp = 0
    while heap:
        _, cur = pop(heap)
        if closed[cur]:
            continue
        if cur == t:
            break
        closed[cur] = 1
        exp += 1
        if exp > max_expansions:
            return None, exp, 0.0
        gc = gs[cur]
        for off, c, c1, c2 in nbrs:
            nb = cur + off
            if blk[nb] or closed[nb]:
                continue
            if c1 and (blk[cur + c1] or blk[cur + c2]):
                continue
            ng = gc + (c * mul[nb] if mul is not None else c)
            if ng < gs[nb]:
                gs[nb] = ng
                came[nb] = cur
                ny_, nx_ = divmod(nb, W2)
                ddx = nx_ - 1 - tx
                ddy = ny_ - 1 - ty
                if ddx < 0:
                    ddx = -ddx
                if ddy < 0:
                    ddy = -ddy
                hh = ddx + ddy + dk * (ddx if ddx < ddy else ddy)
                push(heap, (ng + hw * hh, nb))
    if gs[t] == INF:
        return None, exp, 0.0
    out = []
    cur = t
    while cur != -1:
        cy, cx = divmod(cur, W2)
        out.append((cx - 1, cy - 1))
        cur = came[cur]
    out.reverse()
    return out, exp, gs[t]


# -------------------------------------------------------------- plan_path

def _poly_len(pts) -> float:
    a = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(a) < 2:
        return 0.0
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def _turn_points(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keeps the end points and every point where the direction changes."""
    if len(pts) <= 2:
        return list(pts)
    a = np.asarray(pts, dtype=float)
    d = np.diff(a, axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot = (d[:-1] * d[1:]).sum(axis=1)
    keep = np.concatenate([[True], (np.abs(cross) > 1e-9) | (dot <= 0), [True]])
    return [pts[i] for i in np.nonzero(keep)[0]]


def _coarse_plan(pg: PlanGrid, s_free, g_free, cfg: PlannerConfig):
    """A* on a k-times downsampled grid (blocked = any fine cell blocked,
    extra = max). Returns (world points incl. fine end-cell centres, expansions)
    or (None, expansions); coarse-free cells are free at full resolution."""
    k = int(cfg.coarse_factor)
    hc, wc = -(-pg.h // k), -(-pg.w // k)
    bp = np.ones((hc * k, wc * k), dtype=bool)
    bp[:pg.h, :pg.w] = pg.blocked
    ep = np.zeros((hc * k, wc * k), dtype=np.float32)
    ep[:pg.h, :pg.w] = pg.extra
    bc = bp.reshape(hc, k, wc, k).any(axis=(1, 3))
    ec = ep.reshape(hc, k, wc, k).max(axis=(1, 3))
    fb = bc.ravel()
    cs = nearest_free_cell(fb, wc, hc, (s_free[0] // k, s_free[1] // k), max_radius=2)
    cg = nearest_free_cell(fb, wc, hc, (g_free[0] // k, g_free[1] // k), max_radius=2)
    if cs is None or cg is None:
        return None, 0
    cells, exp, _ = weighted_astar(bc, ec, tuple(cs), tuple(cg), cfg.heuristic_weight, cfg.max_expansions)
    if cells is None:
        return None, exp
    cw = [(pg.ox + (cx + 0.5) * k * pg.res, pg.oy + (cy + 0.5) * k * pg.res) for cx, cy in cells]
    a = grid_to_world(s_free[0], s_free[1], pg.ox, pg.oy, pg.res)
    b = grid_to_world(g_free[0], g_free[1], pg.ox, pg.oy, pg.res)
    pts = [a] + cw + [b]
    if not (pg.seg_free(pts[0], pts[1]) and pg.seg_free(pts[-2], pts[-1])):
        return None, exp
    return pts, exp


def _shortcut(pg: PlanGrid, pts: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    """Greedy string pulling: from each anchor jump to the farthest point that
    is visible (no blocked sample) and whose straight segment is not costlier
    than the A* sub-path (keeps social detours)."""
    pts = _turn_points(pts)
    n = len(pts)
    if n <= 2:
        return list(pts)
    step_cost = [0.0]
    for i in range(1, n):
        step_cost.append(step_cost[-1] + pg.seg_cost(pts[i - 1], pts[i]))
    out = [pts[0]]
    i = 0
    while i < n - 1:
        best = i + 1
        j = i + 2
        while j < n:
            if not pg.seg_free(pts[i], pts[j]):
                break
            if pg.seg_cost(pts[i], pts[j]) <= (step_cost[j] - step_cost[i]) * (1.0 + tol) + 1e-9:
                best = j
            j += 1
        out.append(pts[best])
        i = best
    return out


def plan_path(grid_meta: Any, floor: Any, walls: Any, start_xy: Sequence[float],
              goal_xy: Sequence[float], social: Any = None, no_go_mask: Any = None,
              cfg: Optional[PlannerConfig] = None, plan_grid: Optional[PlanGrid] = None) -> Path:
    """Global path in world coordinates. ``plan_grid`` may be passed to reuse
    a prebuilt PlanGrid (then floor/walls/social/no_go are ignored)."""
    cfg = cfg or PlannerConfig()
    t0 = time.perf_counter()
    pg = plan_grid or PlanGrid(grid_meta, floor, walls, social, no_go_mask, cfg)
    warnings: List[str] = []

    def fail(reason: str, exp: int = 0) -> Path:
        return Path(points=[], ok=False, reason=reason, cells_expanded=exp,
                    ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings)

    sx, sy = float(start_xy[0]), float(start_xy[1])
    gxw, gyw = float(goal_xy[0]), float(goal_xy[1])
    s_cell = world_to_grid(sx, sy, pg.ox, pg.oy, pg.res)
    g_cell = world_to_grid(gxw, gyw, pg.ox, pg.oy, pg.res)
    if not (0 <= s_cell[0] < pg.w and 0 <= s_cell[1] < pg.h):
        return fail("start outside map")
    if not (0 <= g_cell[0] < pg.w and 0 <= g_cell[1] < pg.h):
        return fail("goal outside map")
    snap_r = max(1, int(math.ceil(cfg.snap_radius_m / pg.res)))
    flat_blk = pg.blocked.ravel()
    s_free = nearest_free_cell(flat_blk, pg.w, pg.h, s_cell, max_radius=snap_r)
    if s_free is None:
        return fail("start blocked")
    g_free = nearest_free_cell(flat_blk, pg.w, pg.h, g_cell, max_radius=snap_r)
    if g_free is None:
        return fail("goal blocked (no free cell within %.2f m)" % cfg.snap_radius_m)
    s_snapped = tuple(s_free) != tuple(s_cell)
    g_snapped = tuple(g_free) != tuple(g_cell)
    if g_snapped:
        gxw, gyw = grid_to_world(g_free[0], g_free[1], pg.ox, pg.oy, pg.res)
        warnings.append("goal snapped to nearest free cell (%.2f, %.2f)" % (gxw, gyw))
    if s_snapped:
        warnings.append("start inside inflated obstacle, leaving via nearest free cell")

    pts = None
    exp = 0
    cost = 0.0
    if cfg.coarse_factor and cfg.coarse_factor > 1 and pg.w * pg.h >= cfg.coarse_min_cells:
        pts, exp = _coarse_plan(pg, s_free, g_free, cfg)
    if pts is None:
        cells, exp2, cost = weighted_astar(pg.blocked, pg.extra, tuple(s_free), tuple(g_free),
                                           cfg.heuristic_weight, cfg.max_expansions)
        exp += exp2
        if cells is None:
            return fail("no path" if exp2 <= cfg.max_expansions else "expansion limit", exp)
        pts = [grid_to_world(cx, cy, pg.ox, pg.oy, pg.res) for cx, cy in cells]
    # exact endpoints where they are plannable
    if not s_snapped:
        pts[0] = (sx, sy)
    if len(pts) == 1:
        pts.append((gxw, gyw))
    else:
        pts[-1] = (gxw, gyw)
    prefix = [(sx, sy)] if s_snapped else []
    core = _shortcut(pg, pts, cfg.shortcut_cost_tol) if cfg.shortcut else pts
    if cfg.smooth and len(core) >= 2:
        core = smooth_path(core, ds=cfg.ds, grid=pg)
    allp = prefix + [tuple(p) for p in core]
    # drop consecutive duplicates
    clean = [allp[0]]
    for p in allp[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
            clean.append(p)
    if len(clean) == 1:
        clean.append(clean[0])
    out = [[float(x), float(y)] for x, y in clean]
    return Path(points=out, length_m=_poly_len(out), ok=True,
                reason="ok" if not warnings else warnings[0], cells_expanded=exp,
                ms=(time.perf_counter() - t0) * 1000.0,
                warnings=warnings)


def plan_through(grid_meta: Any, floor: Any, walls: Any, waypoints: Sequence[Sequence[float]],
                 social: Any = None, no_go_mask: Any = None,
                 cfg: Optional[PlannerConfig] = None,
                 zones: Optional[Sequence[Any]] = None) -> Path:
    """Chains A* legs through waypoints (patrol / follow_path / goto chains).
    zones: per waypoint ("fine" / "z10".."z100", contract 9.1; default z30,
    last point "fine"). Waypoint corners are blended inside their zone, A*
    turn points with cfg.auto_blend_m; Path.stops = fine point indices."""
    cfg = cfg or PlannerConfig()
    t0 = time.perf_counter()
    pg = PlanGrid(grid_meta, floor, walls, social, no_go_mask, cfg)
    raw_cfg = replace(cfg, smooth=False)
    pts: List[Tuple[float, float]] = []
    zl: List[Any] = []
    wz = list(zones) if zones is not None else []
    wz += [None] * (len(waypoints) - len(wz))
    if wz and wz[0] is None:
        wz[0] = ZONE_DEFAULT
    for k in range(1, len(wz)):
        if wz[k] is None:
            wz[k] = "fine" if k == len(wz) - 1 else ZONE_DEFAULT
    exp, warns = 0, []
    if len(waypoints) < 2:
        return Path(ok=False, reason="need at least 2 waypoints")
    for k, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        p = plan_path(grid_meta, None, None, a, b, cfg=raw_cfg, plan_grid=pg)
        exp += p.cells_expanded
        warns += p.warnings
        if not p.ok:
            return Path(points=[list(q) for q in pts], length_m=_poly_len(pts) if pts else 0.0,
                        ok=False, reason="leg %d %s->%s: %s" % (k + 1, list(a), list(b), p.reason),
                        cells_expanded=exp, ms=(time.perf_counter() - t0) * 1000.0, warnings=warns)
        leg = [tuple(q) for q in p.points]
        if not pts:
            pts.append(leg[0])
            zl.append(wz[0])
        pts += leg[1:]
        zl += [None] * (len(leg) - 2) + [wz[k + 1]]
    if cfg.smooth:
        out, stops = smooth_path_ex(pts, ds=cfg.ds, grid=pg, zones=zl, auto_blend_m=cfg.auto_blend_m)
    else:
        out = [list(q) for q in pts]
        stops = [i for i, z in enumerate(zl) if i > 0 and z == "fine"]
    return Path(points=out, length_m=_poly_len(out), ok=True, reason="ok",
                cells_expanded=exp, ms=(time.perf_counter() - t0) * 1000.0,
                warnings=warns, stops=stops)


# ------------------------------------------------------------ smoothing

def _resample(pts: np.ndarray, ds: float) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    total = s[-1]
    n = max(1, int(math.ceil(total / ds - 1e-9)))
    si = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], axis=1)


def _catmull_rom(p0, p1, p2, p3, n: int, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom between p1 and p2 (Barry-Goldman form), n samples incl. ends."""
    def tj(ti, a, b):
        return ti + max(float(np.hypot(*(b - a))) ** alpha, 1e-6)
    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n)[:, None]
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


ZONE_DEFAULT = "z30"


def zone_radius(zone: Any) -> Optional[float]:
    """Contract 9.1: "fine" -> None (exact stop); "z10".."z100" -> blend
    radius in m; anything else -> the z30 default."""
    z = str(zone if zone is not None else ZONE_DEFAULT).strip().lower()
    if z == "fine":
        return None
    if z.startswith("z"):
        try:
            return max(0.0, float(z[1:]) / 100.0)
        except ValueError:
            pass
    return 0.3


def _dense_step(ds: float, grid: Any) -> float:
    step = ds / 4.0
    if grid is not None:
        gres = grid.res if isinstance(grid, PlanGrid) else float(_g(grid, "resolution"))
        step = min(step, 0.25 * gres)
    return step


def _blend_chunk(pts: np.ndarray, radii: List[float], blocked, step: float) -> np.ndarray:
    """Dense polyline through pts with every interior corner i rounded by a
    quadratic Bezier from P-r*u_in to P+r*u_out (r = radii[i], capped at half
    of each adjacent segment). A blend that touches the grid is halved until
    it is free (down to a sharp corner)."""
    n = len(pts)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    pieces = [pts[:1]]
    cur = pts[0]
    for i in range(1, n):
        P = pts[i]
        curve = None
        r = radii[i] if i < n - 1 else 0.0
        if r > 1e-6:
            u_in = (P - pts[i - 1]) / max(seg[i - 1], 1e-12)
            u_out = (pts[i + 1] - P) / max(seg[i], 1e-12)
            if float(u_in @ u_out) < 1.0 - 1e-9:
                rr = min(r, 0.5 * seg[i - 1], 0.5 * seg[i])
                while rr > step:
                    A, B = P - rr * u_in, P + rr * u_out
                    m = max(3, int(math.ceil(2 * rr / step)) + 1)
                    t = np.linspace(0.0, 1.0, m)[:, None]
                    c = (1 - t) ** 2 * A + 2 * (1 - t) * t * P + t ** 2 * B
                    if blocked is None or not blocked(c[:, 0], c[:, 1]).any():
                        curve = c
                        break
                    rr *= 0.5
        end = curve[0] if curve is not None else P
        L = float(np.hypot(*(end - cur)))
        m = max(2, int(math.ceil(L / step)) + 1)
        pieces.append((np.linspace(0.0, 1.0, m)[:, None] * (end - cur) + cur)[1:])
        if curve is not None:
            pieces.append(curve[1:])
            cur = curve[-1]
        else:
            cur = P
    return np.vstack(pieces)


def smooth_path_ex(points: Sequence[Sequence[float]], ds: float = 0.1, grid: Any = None,
                   zones: Optional[Sequence[Any]] = None, auto_blend_m: float = 0.5,
                   alpha: float = 0.5) -> Tuple[List[List[float]], List[int]]:
    """Like smooth_path, also returning the output indices of "fine" stops.
    zones: one per input point -- "fine" (exact stop, sharp corner),
    "z10".."z100" (corner blended within that radius), None = internal turn
    point (radius ``auto_blend_m``); the last point is a stop unless it is
    an explicit zXX. Fine points are kept exactly in the output."""
    if zones is None:
        return smooth_path(points, ds, grid, alpha=alpha), []
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return [], []
    zl = list(zones)[:len(pts)] + [None] * max(0, len(pts) - len(zones))
    P, Z = [pts[0]], [zl[0]]
    for i in range(1, len(pts)):                 # merge duplicates, "fine" wins
        if float(np.hypot(*(pts[i] - P[-1]))) > 1e-9:
            P.append(pts[i])
            Z.append(zl[i])
        elif str(zl[i]).lower() == "fine":
            Z[-1] = "fine"
    pts = np.asarray(P)
    if len(pts) < 2:
        return pts.tolist(), [0]
    radii = [auto_blend_m if z is None else zone_radius(z) for z in Z]
    last_stop = Z[-1] is None or radii[-1] is None
    blocked = _blocked_fn(grid)
    step = _dense_step(ds, grid)
    cut = [0] + [i for i in range(1, len(pts) - 1) if radii[i] is None] + [len(pts) - 1]
    out = [pts[:1]]
    stops: List[int] = []
    for a, b in zip(cut[:-1], cut[1:]):
        chunk = pts[a:b + 1]
        r = [0.0] + [float(x or 0.0) for x in radii[a + 1:b]] + [0.0]
        part = _resample(_blend_chunk(chunk, r, blocked, step), ds)
        part[-1] = chunk[-1]
        out.append(part[1:])
        stops.append(sum(len(o) for o in out) - 1)
    res = np.vstack(out)
    if not last_stop:
        stops = stops[:-1]
    return res.tolist(), stops


def smooth_path(points: Sequence[Sequence[float]], ds: float = 0.1, grid: Any = None,
                alpha: float = 0.5, zones: Optional[Sequence[Any]] = None) -> List[List[float]]:
    """Centripetal Catmull-Rom through ``points``, resampled every ``ds`` m.
    ``grid`` (PlanGrid or {resolution, origin_*, width, height, blocked})
    enables the collision check: a curve segment that touches a blocked cell
    (which the straight segment did not) is replaced by the straight segment.
    With ``zones`` (contract 9.1) corners are instead rounded only inside each
    waypoint's zone radius -- see smooth_path_ex."""
    if zones is not None:
        return smooth_path_ex(points, ds, grid, zones, alpha=alpha)[0]
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return []
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return pts.tolist()
    blocked = _blocked_fn(grid)
    dense_step = _dense_step(ds, grid)
    n = len(pts)
    ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    pieces = [pts[:1]]
    for i in range(n - 1):
        p1, p2 = pts[i], pts[i + 1]
        L = float(np.hypot(*(p2 - p1)))
        m = max(2, int(math.ceil(L / dense_step)) + 1)
        if n == 2:
            seg = np.linspace(0.0, 1.0, m)[:, None] * (p2 - p1) + p1
        else:
            seg = _catmull_rom(ext[i], p1, p2, ext[i + 3], m, alpha)
            if blocked is not None and blocked(seg[:, 0], seg[:, 1]).any():
                line = np.linspace(0.0, 1.0, m)[:, None] * (p2 - p1) + p1
                # fall back only if the straight segment is itself better
                if blocked(line[:, 0], line[:, 1]).sum() < blocked(seg[:, 0], seg[:, 1]).sum():
                    seg = line
        pieces.append(seg[1:])
    dense = np.vstack(pieces)
    out = _resample(dense, ds)
    out[0], out[-1] = pts[0], pts[-1]
    return out.tolist()


# ---------------------------------------------------------- speed profile

def _points_of(path: Any) -> np.ndarray:
    pts = path.points if isinstance(path, Path) else path
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def curvature(pts: np.ndarray) -> np.ndarray:
    """Menger curvature per point (0 at the ends)."""
    n = len(pts)
    k = np.zeros(n)
    if n < 3:
        return k
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab = np.hypot(*(b - a).T)
    bc = np.hypot(*(c - b).T)
    ca = np.hypot(*(a - c).T)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    den = ab * bc * ca
    k[1:-1] = np.where(den > 1e-12, 2.0 * cross / np.maximum(den, 1e-12), 0.0)
    return k


def _persons_xy(persons_world: Optional[Iterable[Any]]) -> np.ndarray:
    out = []
    for p in persons_world or []:
        try:
            if isinstance(p, (list, tuple)):
                x, y = float(p[0]), float(p[1])
            else:
                x, y = float(_g(p, "x")), float(_g(p, "y"))
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _point_seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance of points P (m,2) to segments A->B (k,2) -> (m, k)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip((AP * AB[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    return np.hypot(*(P[:, None, :] - proj).transpose(2, 0, 1))


def speed_profile_ex(path: Any, profile_name: str = "normal", cfg: Optional[PlannerConfig] = None,
                     persons_world: Optional[Iterable[Any]] = None,
                     stops: Optional[Sequence[int]] = None
                     ) -> Tuple[List[float], List[str], str]:
    """Returns (v per point, warnings, effective profile name). ``stops``
    (default Path.stops) = "fine" point indices: v = 0 there, precise-profile
    cap on the last cfg.fine_approach_m before each (and before the end when
    cfg.fine_end)."""
    cfg = cfg or PlannerConfig()
    warnings: List[str] = []
    name = profile_name if profile_name in PROFILES else "normal"
    if name != profile_name:
        warnings.append("unknown profile %r, using normal" % (profile_name,))
    pts = _points_of(path)
    n = len(pts)
    if n == 0:
        return [], warnings, name
    P = _persons_xy(persons_world)
    in_corr = np.zeros(len(P), dtype=bool)
    if len(P) and n >= 2:
        d = _point_seg_dist(P, pts[:-1], pts[1:])            # (m, n-1)
        in_corr = d.min(axis=1) <= cfg.corridor_m
    elif len(P):
        in_corr = np.hypot(*(P - pts[0]).T) <= cfg.corridor_m
    if name == "sprint" and in_corr.any():
        name = "normal"
        warnings.append("sprint downgraded to normal: person in corridor (+-%.1f m)" % cfg.corridor_m)
    prof = PROFILES[name]
    vmax, amax = prof["vmax"], prof["amax"]
    v = np.full(n, vmax)
    kap = curvature(pts)
    with np.errstate(divide="ignore"):
        v = np.minimum(v, np.where(kap > 1e-9, np.sqrt(cfg.a_lat / np.maximum(kap, 1e-9)), np.inf))
    if in_corr.any():
        Pc = P[in_corr]
        dp = np.hypot(pts[:, None, 0] - Pc[None, :, 0], pts[:, None, 1] - Pc[None, :, 1]).min(axis=1)
        near = dp <= cfg.person_slow_m
        if near.any():
            cap = PROFILES["stealth"]["vmax"]
            v[near] = np.minimum(v[near], cap)
            warnings.append("person near path: %d point(s) capped to %.2f m/s" % (int(near.sum()), cap))
    seg = np.hypot(*np.diff(pts, axis=0).T) if n >= 2 else np.zeros(0)
    if stops is None:
        stops = path.stops if isinstance(path, Path) else []
    st = sorted({int(i) for i in stops if 0 < int(i) < n})
    if cfg.fine_end:
        st.append(n - 1)
    if st and n >= 2:
        s_arc = np.concatenate([[0.0], np.cumsum(seg)])
        cap = PROFILES["precise"]["vmax"]
        for i in st:
            near = (s_arc <= s_arc[i]) & (s_arc >= s_arc[i] - cfg.fine_approach_m)
            v[near] = np.minimum(v[near], cap)
            if i < n - 1:
                v[i] = 0.0
    v[0] = min(v[0], max(cfg.v_start, 0.0))
    for i in range(n - 1):                                   # accel
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * amax * seg[i]))
    v[-1] = min(v[-1], max(cfg.v_end, 0.0))
    for i in range(n - 2, -1, -1):                           # decel (goal slow-down)
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * amax * seg[i]))
    return [float(x) for x in v], warnings, name


def speed_profile(path: Any, profile_name: str = "normal", cfg: Optional[PlannerConfig] = None,
                  persons_world: Optional[Iterable[Any]] = None,
                  stops: Optional[Sequence[int]] = None) -> VProfile:
    """v per point (list subclass; ``.warnings`` and effective ``.profile``)."""
    v, w, name = speed_profile_ex(path, profile_name, cfg, persons_world, stops)
    return VProfile(v, w, name)


def eta(path: Any, v: Any, v_min: float = 0.05) -> float:
    """Travel time in s; ``v`` scalar or per-point list (trapezoid per segment)."""
    pts = _points_of(path)
    if len(pts) < 2:
        return 0.0
    seg = np.hypot(*np.diff(pts, axis=0).T)
    if np.isscalar(v):
        return float(seg.sum() / max(float(v), v_min))
    va = np.maximum(np.asarray(v, dtype=float).reshape(-1), 0.0)
    if len(va) != len(pts):
        return float(seg.sum() / max(float(va.mean()) if len(va) else v_min, v_min))
    vm = np.maximum(0.5 * (va[:-1] + va[1:]), v_min)
    return float((seg / vm).sum())


# ------------------------------------------------------------------ jumps

def plan_jumps(grid_meta: Any, floor: Any, walls: Any, start_xy: Sequence[float],
               goal_xy: Sequence[float], jump_len_m: Optional[float] = None,
               clearance_m: Optional[float] = None, max_jumps: int = 6,
               path: Any = None, cfg: Optional[PlannerConfig] = None) -> JumpPlan:
    """Landing sequence of fixed-length FrontJumps from start towards goal,
    straight or along ``path`` (consecutive landings at chord jump_len_m).
    Each landing: known free floor within clearance_m, flat, not stair band;
    no tall obstacle under the flight line."""
    cfg = cfg or PlannerConfig()
    L = float(jump_len_m) if jump_len_m else JUMP_LEN_M
    clr = cfg.jump_clearance_m if clearance_m is None else float(clearance_m)
    res = float(_g(grid_meta, "resolution"))
    ox, oy = float(_g(grid_meta, "origin_x", 0.0)), float(_g(grid_meta, "origin_y", 0.0))
    w, h = int(_g(grid_meta, "width")), int(_g(grid_meta, "height"))
    if floor is None:
        floor = _g(grid_meta, "floor")
    if walls is None:
        walls = _g(grid_meta, "walls")
    fl = np.asarray(floor, dtype=np.int16).reshape(h, w)
    wl = (np.zeros((h, w), np.int16) if walls is None or len(walls) != w * h
          else np.asarray(walls, dtype=np.int16).reshape(h, w))
    S = np.array([float(start_xy[0]), float(start_xy[1])])
    G = np.array([float(goal_xy[0]), float(goal_xy[1])])
    poly = _points_of(path) if path is not None else np.vstack([S, G])
    if path is not None and len(poly) >= 1:
        poly = np.vstack([S, poly[1:]]) if np.hypot(*(poly[0] - S)) < 1e-6 else np.vstack([S, poly])
        G = poly[-1]
    total = float(np.hypot(*(G - S)))
    plan = JumpPlan()
    if total < 1e-6:
        plan.ok, plan.reason = True, "already at goal"
        return plan

    # landing candidates
    landings: List[np.ndarray] = []
    cur = S.copy()
    seg_i = 0
    while float(np.hypot(*(G - cur))) > 0.5 * L:
        nxt = None
        for k in range(seg_i, len(poly) - 1):     # first polyline crossing of circle(cur, L)
            A, B = poly[k], poly[k + 1]
            if np.hypot(*(B - cur)) < L - 1e-6:
                continue
            d = B - A
            f = A - cur
            a = float(d @ d)
            if a < 1e-12:
                continue
            b = 2.0 * float(f @ d)
            c = float(f @ f) - L * L
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b + math.sqrt(disc)) / (2 * a)
            if 0.0 <= t <= 1.0:
                nxt, seg_i = A + t * d, k
                break
        if nxt is None:
            break
        landings.append(nxt)
        cur = nxt
        if len(landings) > max_jumps:
            break
    if not landings:
        plan.reason = "goal closer than half a jump (%.2f m)" % total
        plan.residual_m = total
        return plan
    if len(landings) > max_jumps:
        plan.reason = "needs more than max_jumps=%d jumps" % max_jumps
        return plan

    r_c = int(math.ceil(clr / res))
    oy_, ox_ = np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]
    disk = (ox_ * res) ** 2 + (oy_ * res) ** 2 <= (clr + 0.5 * res) ** 2

    def check(p: np.ndarray) -> str:
        gx, gy = world_to_grid(float(p[0]), float(p[1]), ox, oy, res)
        if not (r_c <= gx < w - r_c and r_c <= gy < h - r_c):
            return "outside map"
        f = fl[gy - r_c:gy + r_c + 1, gx - r_c:gx + r_c + 1][disk]
        ww = wl[gy - r_c:gy + r_c + 1, gx - r_c:gx + r_c + 1][disk]
        if (f == 100).any():
            return "obstacle within %.2f m" % clr
        if (f < 0).any():
            return "unknown floor within %.2f m" % clr
        if ((ww >= cfg.jump_stair_wall_min) & (ww <= cfg.jump_stair_wall_max)).any():
            return "stair band"
        if (ww > cfg.jump_flat_wall_max).any():
            return "not flat"
        return ""

    prev = S
    for i, p in enumerate(landings):
        why = check(p)
        if not why:
            n = max(2, int(math.ceil(L / (0.5 * res))) + 1)
            t = np.linspace(0.0, 1.0, n)
            xs, ys = prev[0] + (p[0] - prev[0]) * t, prev[1] + (p[1] - prev[1]) * t
            gx = np.floor((xs - ox) / res).astype(int)
            gy = np.floor((ys - oy) / res).astype(int)
            ins = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
            if (wl[gy[ins], gx[ins]] > cfg.jump_over_wall_max).any():
                why = "tall obstacle under flight"
        if why:
            plan.rejected.append((i, why))
        plan.headings.append(float(math.atan2(p[1] - prev[1], p[0] - prev[0])))
        prev = p
    plan.landings = [[float(p[0]), float(p[1])] for p in landings]
    plan.residual_m = float(np.hypot(*(G - landings[-1])))
    if plan.rejected:
        i, why = plan.rejected[0]
        plan.reason = "landing %d rejected: %s" % (i + 1, why)
    else:
        plan.ok, plan.reason = True, "ok"
    return plan


# -------------------------------------------------------------- follower

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def follow_controller(pose: Any, path: Any, v_profile: Any, cfg: Optional[PlannerConfig] = None,
                      idx: int = 0) -> Tuple[float, float, int, bool]:
    """Adaptive pure pursuit. pose (x, y, yaw) or {x, y, yaw}; path Path or
    [[x, y]]; v_profile per-point list (or scalar). ``idx`` = previous return
    value (progress; nearest-point search runs forward from it).
    Returns (vx, vyaw, idx, done)."""
    cfg = cfg or PlannerConfig()
    if isinstance(pose, (list, tuple)):
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    else:
        x, y, yaw = float(_g(pose, "x")), float(_g(pose, "y")), float(_g(pose, "yaw"))
    pts = _points_of(path)
    n = len(pts)
    if n == 0:
        return 0.0, 0.0, 0, True
    vp = (np.full(n, float(v_profile)) if np.isscalar(v_profile)
          else np.asarray(v_profile, dtype=float).reshape(-1))
    if len(vp) != n:
        vp = np.full(n, float(vp.max()) if len(vp) else PROFILES["normal"]["vmax"])
    idx = int(min(max(idx, 0), n - 1))
    hi = min(n, idx + max(cfg.search_window, 2))
    d = np.hypot(pts[idx:hi, 0] - x, pts[idx:hi, 1] - y)
    idx = idx + int(np.argmin(d))
    gd = math.hypot(pts[-1, 0] - x, pts[-1, 1] - y)
    if gd <= cfg.goal_tol_m and (idx >= n - 2 or n == 1 or
                                 float(np.hypot(*np.diff(pts[idx:], axis=0).T).sum()) <= 2 * cfg.goal_tol_m):
        return 0.0, 0.0, n - 1, True
    # speed from the profile at the progress point (next point keeps it moving
    # from v_start=0); never faster than what stops at the end
    v_ref = max(float(vp[idx]), float(vp[min(idx + 1, n - 1)]), cfg.v_min)
    v_ref = min(v_ref, max(cfg.v_min, 1.2 * gd))
    look = min(max(cfg.lookahead_gain_s * v_ref, cfg.lookahead_min), cfg.lookahead_max)
    # target: first point at least `look` from the robot, at/after idx
    dd = np.hypot(pts[idx:, 0] - x, pts[idx:, 1] - y)
    far = np.nonzero(dd >= look)[0]
    j = idx + int(far[0]) if len(far) else n - 1
    tx, ty = pts[j]
    alpha = _wrap(math.atan2(ty - y, tx - x) - yaw)
    if abs(alpha) > cfg.rotate_first_rad:
        vyaw = math.copysign(min(cfg.k_yaw * abs(alpha), cfg.max_vyaw), alpha)
        return 0.0, vyaw, idx, False
    Ld = max(math.hypot(tx - x, ty - y), 1e-3)
    kappa = 2.0 * math.sin(alpha) / Ld
    vx = v_ref * max(math.cos(alpha), 0.3)
    vyaw = vx * kappa
    if abs(vyaw) > cfg.max_vyaw:
        vx *= cfg.max_vyaw / abs(vyaw)
        vyaw = math.copysign(cfg.max_vyaw, vyaw)
    return float(vx), float(vyaw), idx, False


def blend_with_dwa(cmd: Sequence[float], dwa_cmd: Optional[Sequence[float]], persons_near: Any,
                   w_near: float = 0.7, w_far: float = 0.0) -> Tuple[float, float]:
    """Mix the pure-pursuit command with navigation/nav_local.plan_local's
    (vx, vyaw). persons_near: bool/count -> weight w_near, else w_far.
    vx never exceeds the larger of the two inputs; a DWA stop (0, 0) with
    persons near is honoured as a stop."""
    vx, vyaw = float(cmd[0]), float(cmd[1])
    if dwa_cmd is None:
        return vx, vyaw
    dvx, dvyaw = float(dwa_cmd[0]), float(dwa_cmd[1])
    near = bool(persons_near)
    if near and dvx == 0.0 and dvyaw == 0.0:
        return 0.0, 0.0
    w = min(max(w_near if near else w_far, 0.0), 1.0)
    bvx = (1.0 - w) * vx + w * dvx
    bvyaw = (1.0 - w) * vyaw + w * dvyaw
    bvx = min(bvx, max(vx, dvx))
    if near:
        bvx = min(bvx, max(dvx, 0.0))      # near people DWA may only slow down
    return float(bvx), float(bvyaw)


def split_at_stops(path: Any, stops: Optional[Sequence[int]] = None) -> List[List[List[float]]]:
    """Splits a path at its "fine" stops into sub-paths (each ends at a stop),
    so the executor can follow, stop/settle, then continue."""
    pts = _points_of(path).tolist()
    if stops is None:
        stops = path.stops if isinstance(path, Path) else []
    cuts = [0] + sorted({int(i) for i in stops if 0 < int(i) < len(pts) - 1}) + [len(pts) - 1]
    return [pts[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a] or [pts]
