"""Pure 3D geometry + track smoothing for the perception pillar.

numpy only, no camera/YOLO/FastAPI imports -- unit-testable in isolation.

Frames:
  optical  -- RealSense convention: x right, y down, z forward (metres)
  base     -- robot body (ROS REP-103): x forward, y left, z up (metres)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def deproject(self, u: float, v: float, z_m: float) -> tuple[float, float, float]:
        """Pixel (u, v) + depth (m) -> optical-frame point (x, y, z)."""
        return ((u - self.cx) * z_m / self.fx, (v - self.cy) * z_m / self.fy, z_m)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx,
                "fy": self.fy, "cx": self.cx, "cy": self.cy}


def robust_bbox_depth(
    depth_mm: np.ndarray,
    bbox: tuple[float, float, float, float],
    min_mm: int = 200,
    max_mm: int = 10000,
    percentile: float = 30.0,
    min_valid_px: int = 20,
) -> tuple[Optional[float], float, float, float]:
    """Person distance from the torso sub-window of a bbox.

    The full bbox is mostly background (between arms/legs), so only the
    middle 40% of the width and 20-60% of the height is sampled. A low
    percentile prefers the (nearer) person over any background leaking in.

    Returns (z_m or None, u, v, valid_ratio); (u, v) = sample-window centre.
    """
    h_img, w_img = depth_mm.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    wx1 = int(np.clip(x1 + 0.30 * bw, 0, w_img - 1))
    wx2 = int(np.clip(x2 - 0.30 * bw, wx1 + 1, w_img))
    wy1 = int(np.clip(y1 + 0.20 * bh, 0, h_img - 1))
    wy2 = int(np.clip(y1 + 0.60 * bh, wy1 + 1, h_img))
    u = (wx1 + wx2) / 2.0
    v = (wy1 + wy2) / 2.0

    window = depth_mm[wy1:wy2, wx1:wx2]
    valid = window[(window >= min_mm) & (window <= max_mm)]
    ratio = float(valid.size) / float(max(window.size, 1))
    if valid.size < min_valid_px:
        return None, u, v, ratio
    return float(np.percentile(valid, percentile)) / 1000.0, u, v, ratio


@dataclass
class Extrinsics:
    """Camera mount pose in the robot base frame.

    pitch_deg > 0 = camera tilted DOWN. yaw_deg > 0 = camera turned LEFT.
    Defaults are a guess for the Go2 head mount -- MEASURE before trusting
    absolute heights (same lesson as the Hesai 90 deg offset, 2026-09-04).
    """
    tx: float = 0.30
    ty: float = 0.0
    tz: float = 0.10
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    _rot: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = math.radians(self.pitch_deg)
        y = math.radians(self.yaw_deg)
        # Rotation about +y by +p tips +x toward -z (forward axis down).
        r_pitch = np.array([[math.cos(p), 0.0, math.sin(p)],
                            [0.0, 1.0, 0.0],
                            [-math.sin(p), 0.0, math.cos(p)]])
        r_yaw = np.array([[math.cos(y), -math.sin(y), 0.0],
                          [math.sin(y), math.cos(y), 0.0],
                          [0.0, 0.0, 1.0]])
        self._rot = r_yaw @ r_pitch

    def optical_to_base(self, p: tuple[float, float, float]) -> tuple[float, float, float]:
        xo, yo, zo = p
        body = np.array([zo, -xo, -yo])  # optical -> camera body (x fwd, y left, z up)
        out = self._rot @ body + np.array([self.tx, self.ty, self.tz])
        return float(out[0]), float(out[1]), float(out[2])


@dataclass
class TrackState:
    track_id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    last_t: float = 0.0
    first_t: float = 0.0
    hits: int = 1


class TrackSmoother:
    """Per-track-id EMA on base-frame position + velocity estimate.

    ByteTrack gives stable ids in 2D; this layer smooths the noisy depth
    and rejects single-frame depth jumps (bbox briefly catching a wall).
    """

    OUTLIER_JUMP_M = 1.5

    def __init__(self, alpha: float = 0.5, vel_alpha: float = 0.3, max_age_s: float = 1.0):
        self.alpha = alpha
        self.vel_alpha = vel_alpha
        self.max_age_s = max_age_s
        self._tracks: dict[int, TrackState] = {}

    def update(self, track_id: int, x: float, y: float, z: float, t: float) -> TrackState:
        s = self._tracks.get(track_id)
        if s is None:
            s = TrackState(track_id, x, y, z, last_t=t, first_t=t)
            self._tracks[track_id] = s
            return s

        if s.hits >= 3 and math.hypot(x - s.x, y - s.y) > self.OUTLIER_JUMP_M:
            return s  # keep last_t: a persistent jump ages out and re-acquires

        dt = max(t - s.last_t, 1e-3)
        a = self.alpha
        px, py = s.x, s.y
        s.x = a * x + (1 - a) * s.x
        s.y = a * y + (1 - a) * s.y
        s.z = a * z + (1 - a) * s.z
        va = self.vel_alpha
        s.vx = va * (s.x - px) / dt + (1 - va) * s.vx
        s.vy = va * (s.y - py) / dt + (1 - va) * s.vy
        s.last_t = t
        s.hits += 1
        return s

    def prune(self, t: float) -> list[int]:
        dead = [tid for tid, s in self._tracks.items() if t - s.last_t > self.max_age_s]
        for tid in dead:
            del self._tracks[tid]
        return dead

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)
