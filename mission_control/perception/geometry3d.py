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
    rejects: int = 0            # consecutive gated-out measurements
    kf_x: Optional[np.ndarray] = field(default=None, repr=False)   # [x y z vx vy vz]
    kf_P: Optional[np.ndarray] = field(default=None, repr=False)
    kf_t: float = 0.0           # time of the last predict


@dataclass
class KalmanConfig:
    accel_std: float = 0.8          # m/s^2, how hard a person can change speed
    range_std_base: float = 0.03    # m, depth noise at 0 m
    range_std_quad: float = 0.015   # m/m^2, depth noise growth (RealSense ~ z^2)
    lateral_std_base: float = 0.02  # m
    lateral_std_lin: float = 0.01   # m/m
    height_std: float = 0.08        # m, z of the torso window jumps with posture
    init_vel_std: float = 1.0       # m/s, unknown walking speed at birth
    gate_chi2: float = 16.27        # 3 dof, 99.9 %: Mahalanobis outlier gate
    reinit_after: int = 3           # consecutive rejects = real jump, restart
    min_hits_for_gate: int = 3

    @classmethod
    def from_env(cls, prefix: str = "KF_") -> "KalmanConfig":
        import os
        kw = {}
        for name, f in cls.__dataclass_fields__.items():
            raw = os.environ.get(prefix + name.upper())
            if raw is not None:
                kw[name] = type(f.default)(raw)
        return cls(**kw)


class TrackSmoother:
    """Per-track-id constant-velocity Kalman filter on base-frame position.

    ByteTrack gives stable ids in 2D; this layer filters the noisy depth.
    State [x y z vx vy vz], white-noise acceleration model. The measurement
    noise is anisotropic: along the camera ray (depth) it grows with the
    square of the range, across the ray it grows linearly. So a far person
    is trusted less in distance than in bearing, which matches how a stereo
    depth camera fails.

    A measurement outside the Mahalanobis gate (bbox briefly caught a wall)
    is dropped and the track keeps its prediction. After `reinit_after`
    consecutive rejects the jump is taken as real and the filter restarts
    at the new position. `last_t` only moves on accepted measurements, so a
    track that sees nothing but outliers still ages out.
    """

    def __init__(self, cfg: Optional[KalmanConfig] = None, max_age_s: float = 1.0):
        self.cfg = cfg or KalmanConfig()
        self.max_age_s = max_age_s
        self._tracks: dict[int, TrackState] = {}

    # -------------------------------------------------------------- filter

    def _R(self, x: float, y: float) -> np.ndarray:
        c = self.cfg
        r = math.hypot(x, y)
        sr = c.range_std_base + c.range_std_quad * r * r
        sl = c.lateral_std_base + c.lateral_std_lin * r
        th = math.atan2(y, x)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        R = np.zeros((3, 3))
        R[:2, :2] = rot @ np.diag([sr * sr, sl * sl]) @ rot.T
        R[2, 2] = c.height_std ** 2
        return R

    def _init(self, s: TrackState, x: float, y: float, z: float, t: float) -> None:
        s.kf_x = np.array([x, y, z, 0.0, 0.0, 0.0])
        P = np.zeros((6, 6))
        P[:3, :3] = self._R(x, y)
        P[3:, 3:] = np.eye(3) * self.cfg.init_vel_std ** 2
        s.kf_P = P
        s.kf_t = t
        s.rejects = 0
        self._sync(s)

    def _predict(self, s: TrackState, t: float) -> None:
        dt = t - s.kf_t
        if dt <= 0:
            return
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        g = np.array([0.5 * dt * dt, dt])
        Q1 = np.outer(g, g) * self.cfg.accel_std ** 2     # per axis [pos, vel]
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[np.ix_([i, i + 3], [i, i + 3])] = Q1
        s.kf_x = F @ s.kf_x
        s.kf_P = F @ s.kf_P @ F.T + Q
        s.kf_t = t

    @staticmethod
    def _sync(s: TrackState) -> None:
        s.x, s.y, s.z = (float(v) for v in s.kf_x[:3])
        s.vx, s.vy = float(s.kf_x[3]), float(s.kf_x[4])

    # ----------------------------------------------------------------- API

    def update(self, track_id: int, x: float, y: float, z: float, t: float) -> TrackState:
        s = self._tracks.get(track_id)
        if s is None:
            s = TrackState(track_id, x, y, z, last_t=t, first_t=t)
            self._init(s, x, y, z, t)
            self._tracks[track_id] = s
            return s

        self._predict(s, t)
        innov = np.array([x, y, z]) - s.kf_x[:3]
        S = s.kf_P[:3, :3] + self._R(x, y)
        d2 = float(innov @ np.linalg.solve(S, innov))

        if s.hits >= self.cfg.min_hits_for_gate and d2 > self.cfg.gate_chi2:
            s.rejects += 1
            if s.rejects >= self.cfg.reinit_after:
                hits, first_t = s.hits, s.first_t
                self._init(s, x, y, z, t)
                s.hits, s.first_t, s.last_t = hits + 1, first_t, t
                return s
            self._sync(s)
            return s  # keep last_t: a track of pure outliers ages out

        K = s.kf_P[:, :3] @ np.linalg.inv(S)                # H = [I 0]
        s.kf_x = s.kf_x + K @ innov
        s.kf_P = s.kf_P - K @ s.kf_P[:3, :]
        s.rejects = 0
        s.last_t = t
        s.hits += 1
        self._sync(s)
        return s

    def prune(self, t: float) -> list:
        dead = [tid for tid, s in self._tracks.items() if t - s.last_t > self.max_age_s]
        for tid in dead:
            del self._tracks[tid]
        return dead

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)
