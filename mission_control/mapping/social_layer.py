"""Social cost layer (0-255) around people, for the planners.

`social_cost_grid()` rasterises one asymmetric Gaussian per person onto the
mapping grid (row-major, index = gy*width + gx, same as floor/walls):
stretched along the person's walking direction (front larger than back),
narrower to the sides. Persons are combined with max().

`SocialLayer` keeps the last known position of each person and fades its
cost out linearly over `decay_s` (~2.5 s) after it was last seen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class SocialConfig:
    sigma_side: float = 0.6        # m, lateral
    sigma_rear: float = 0.6        # m, behind the person
    sigma_front: float = 1.0       # m, ahead of a walking person (at 0 m/s)
    front_gain_s: float = 1.0      # extra front sigma per m/s of speed
    static_sigma: float = 0.8      # m, symmetric for a standing person
    moving_speed: float = 0.15     # m/s above which the person counts as walking
    cutoff_sigma: float = 3.0      # rasterise only within this many sigmas
    peak: float = 255.0
    decay_s: float = 2.5


def _get(p: Any, key: str, default: Any = None) -> Any:
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _meta(meta: Any) -> tuple:
    return (float(_get(meta, "resolution")), float(_get(meta, "origin_x", 0.0)),
            float(_get(meta, "origin_y", 0.0)), int(_get(meta, "width")), int(_get(meta, "height")))


def social_cost_grid(persons_world: Optional[Iterable[Any]], meta: Any,
                     weights: Optional[Iterable[float]] = None,
                     cfg: Optional[SocialConfig] = None) -> np.ndarray:
    """persons_world: dicts/objects with x, y (world m) and optional vx, vy.
    meta: {resolution, origin_x, origin_y, width, height}. weights: optional
    per-person scale in [0, 1] (used for decay). Returns uint8 of shape
    (width*height,), row-major (reshape(height, width) for a 2D view)."""
    cfg = cfg or SocialConfig()
    res, ox, oy, w, h = _meta(meta)
    out = np.zeros((h, w), dtype=np.float32)
    plist = list(persons_world or [])
    wl = list(weights) if weights is not None else [1.0] * len(plist)
    for p, wt in zip(plist, wl):
        wt = min(max(_f(wt, 0.0), 0.0), 1.0)
        px, py = _f(_get(p, "x"), float("nan")), _f(_get(p, "y"), float("nan"))
        if wt <= 0.0 or not (math.isfinite(px) and math.isfinite(py)):
            continue
        vx, vy = _f(_get(p, "vx", 0.0)), _f(_get(p, "vy", 0.0))
        speed = math.hypot(vx, vy)
        if speed > cfg.moving_speed:
            ux, uy = vx / speed, vy / speed
            s_front = cfg.sigma_front + cfg.front_gain_s * speed
            s_rear, s_side = cfg.sigma_rear, cfg.sigma_side
        else:
            ux, uy = 1.0, 0.0
            s_front = s_rear = s_side = cfg.static_sigma
        reach = cfg.cutoff_sigma * max(s_front, s_rear, s_side)
        gx0 = max(int(math.floor((px - reach - ox) / res)), 0)
        gx1 = min(int(math.floor((px + reach - ox) / res)) + 1, w)
        gy0 = max(int(math.floor((py - reach - oy) / res)), 0)
        gy1 = min(int(math.floor((py + reach - oy) / res)) + 1, h)
        if gx0 >= gx1 or gy0 >= gy1:
            continue
        xs = ox + (np.arange(gx0, gx1, dtype=np.float32) + 0.5) * res - px
        ys = oy + (np.arange(gy0, gy1, dtype=np.float32) + 0.5) * res - py
        dx, dy = np.meshgrid(xs, ys)                     # (rows=y, cols=x)
        along = dx * ux + dy * uy                        # + ahead of the person
        across = -dx * uy + dy * ux
        s_al = np.where(along >= 0.0, s_front, s_rear)
        g = np.exp(-0.5 * ((along / s_al) ** 2 + (across / s_side) ** 2)) * (cfg.peak * wt)
        sub = out[gy0:gy1, gx0:gx1]
        np.maximum(sub, g, out=sub)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(-1)


class SocialLayer:
    """Temporal social layer: update() with the persons seen at time t,
    grid(now) returns the decayed cost grid. A person not seen for decay_s
    seconds has no cost left and is forgotten."""

    def __init__(self, meta: Any, cfg: Optional[SocialConfig] = None) -> None:
        self.meta = meta
        self.cfg = cfg or SocialConfig()
        self._persons: Dict[Any, dict] = {}
        self._anon = 0

    def update(self, persons_world: Optional[Iterable[Any]], t: float) -> None:
        for p in persons_world or []:
            key = _get(p, "gid")
            if key is None:
                self._anon += 1
                key = ("anon", self._anon)
            self._persons[key] = {"x": _f(_get(p, "x"), float("nan")), "y": _f(_get(p, "y"), float("nan")),
                                  "vx": _f(_get(p, "vx", 0.0)), "vy": _f(_get(p, "vy", 0.0)), "t": float(t)}

    def weight(self, age_s: float) -> float:
        if age_s <= 0.0:
            return 1.0
        return max(0.0, 1.0 - age_s / self.cfg.decay_s)

    def prune(self, now: float) -> None:
        for k in [k for k, v in self._persons.items() if now - v["t"] >= self.cfg.decay_s]:
            del self._persons[k]

    def grid(self, now: float) -> np.ndarray:
        self.prune(now)
        plist = list(self._persons.values())
        return social_cost_grid(plist, self.meta, [self.weight(now - p["t"]) for p in plist], self.cfg)

    def __len__(self) -> int:
        return len(self._persons)
