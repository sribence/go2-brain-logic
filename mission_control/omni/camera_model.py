"""Camera projection models for the omni rig (pure numpy math, no I/O).

Conventions (CONTRACTS.md section 1):
  * camera frame = OpenCV: z forward (optical axis), x right, y down;
  * `T_base_cam` maps camera-frame points to the base frame:
    p_base = T_base_cam @ p_cam.

Models:
  * FisheyeModel -- OpenCV `cv2.fisheye` / Kannala-Brandt equidistant model
    with 4 distortion coefficients (k1..k4). Unlike cv2.fisheye we compute
    theta with atan2(r, z), so rays beyond 90 deg off-axis (>180 deg lenses)
    are handled too.
  * PinholeModel -- OpenCV standard model with (k1, k2, p1, p2, k3).

All functions are vectorised over N points.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _as_pts(pts: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    return a.reshape(-1, dim)


def _as_pts_fast(pts: np.ndarray, dim: int) -> np.ndarray:
    """Keep float32 input as float32 (fast path), everything else -> float64."""
    a = np.asarray(pts)
    if a.dtype != np.float32:
        a = a.astype(np.float64, copy=False)
    return a.reshape(-1, dim)


class CameraModel(object):
    """Base class: intrinsics K (3x3), distortion D, image size."""

    kind = "base"

    def __init__(self, K: Sequence, D: Sequence, width: int, height: int):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1)
        self.width = int(width)
        self.height = int(height)
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.skew = float(self.K[0, 1])

    # -- API -----------------------------------------------------------------
    def project(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N,3) camera-frame points -> (uv (N,2) float64, valid (N,) bool)."""
        raise NotImplementedError

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,3) unit rays in the camera frame."""
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------
    def in_image(self, uv: np.ndarray) -> np.ndarray:
        u = uv[:, 0]
        v = uv[:, 1]
        return (u >= 0.0) & (v >= 0.0) & (u <= self.width - 1.0) & (v <= self.height - 1.0)

    def _norm_to_pix(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        u = self.fx * (mx + (self.skew / self.fx) * my) + self.cx
        v = self.fy * my + self.cy
        return np.stack([u, v], axis=1)

    def _pix_to_norm(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        my = (uv[:, 1] - self.cy) / self.fy
        mx = (uv[:, 0] - self.cx - self.skew * my) / self.fx
        return mx, my

    def hfov_deg(self) -> float:
        """Approximate horizontal FOV across the image centre row."""
        uv = np.array([[0.0, self.cy], [self.width - 1.0, self.cy]])
        r = self.unproject(uv)
        a = np.arctan2(r[:, 0], r[:, 2])
        return float(np.degrees(a[1] - a[0]))

    def to_dict(self) -> dict:
        return {"type": self.kind, "K": self.K.tolist(), "D": self.D.tolist(),
                "width": self.width, "height": self.height}


class FisheyeModel(CameraModel):
    """OpenCV fisheye (Kannala-Brandt, equidistant + k1..k4 polynomial).

    theta_d = theta * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
    x' = theta_d * x / r, y' = theta_d * y / r  (r = sqrt(x^2+y^2))
    """

    kind = "fisheye"

    def __init__(self, K: Sequence, D: Sequence, width: int, height: int,
                 max_fov_deg: float = 200.0):
        super(FisheyeModel, self).__init__(K, D, width, height)
        d = np.zeros(4)
        d[:min(4, self.D.size)] = self.D[:4]
        self.D = d
        self.k1, self.k2, self.k3, self.k4 = [float(x) for x in d]
        # Rays with theta >= theta_max are treated as outside the lens.
        theta_max = np.radians(max_fov_deg) / 2.0
        # Also stop where the distortion polynomial stops being monotonic
        # (beyond that, projection would fold back).
        ths = np.linspace(1e-4, min(theta_max, np.pi - 1e-3), 2000)
        dd = self._dpoly(ths)
        bad = np.nonzero(dd <= 1e-6)[0]
        if bad.size:
            theta_max = float(ths[max(bad[0] - 1, 0)])
        self.theta_max = float(theta_max)
        self.max_fov_deg = float(np.degrees(2.0 * self.theta_max))

    def _poly(self, th: np.ndarray) -> np.ndarray:
        t2 = th * th
        return th * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def _dpoly(self, th: np.ndarray) -> np.ndarray:
        t2 = th * th
        return 1.0 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))

    def project(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """uv of points outside the lens FOV is -1 (valid False). float32 in ->
        float32 math (fast path for big point clouds), else float64."""
        p = _as_pts_fast(pts_cam, 3)
        n = p.shape[0]
        dt = p.dtype
        uv = np.full((n, 2), -1.0, dtype=dt)
        valid = np.zeros(n, dtype=bool)
        if n == 0:
            return uv, valid
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r2 = x * x + y * y
        # theta < theta_max  <=>  z > cos(theta_max) * |p|   (cheap pre-mask)
        idx = np.flatnonzero(z > dt.type(np.cos(self.theta_max)) * np.sqrt(r2 + z * z))
        if idx.size == 0:
            return uv, valid
        ri = np.sqrt(r2[idx])
        th = np.arctan2(ri, z[idx])
        t2 = th * th
        k1, k2, k3, k4 = [dt.type(k) for k in (self.k1, self.k2, self.k3, self.k4)]
        theta_d = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = theta_d / np.maximum(ri, dt.type(1e-12))
        mx = x[idx] * scale
        my = y[idx] * scale
        u = dt.type(self.fx) * mx + dt.type(self.skew) * my + dt.type(self.cx)
        v = dt.type(self.fy) * my + dt.type(self.cy)
        uv[idx, 0] = u
        uv[idx, 1] = v
        valid[idx] = (u >= 0) & (v >= 0) & (u <= self.width - 1.0) & (v <= self.height - 1.0)
        return uv, valid

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        q = _as_pts(uv, 2)
        mx, my = self._pix_to_norm(q)
        theta_d = np.hypot(mx, my)
        # Newton iteration on theta: poly(theta) = theta_d
        theta = np.clip(theta_d, 0.0, self.theta_max)
        for _ in range(20):
            f = self._poly(theta) - theta_d
            df = self._dpoly(theta)
            step = f / np.where(np.abs(df) < 1e-9, 1e-9, df)
            theta = np.clip(theta - step, 0.0, np.pi)
            if np.all(np.abs(step) < 1e-12):
                break
        s = np.sin(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ux = np.where(theta_d > 1e-12, mx / np.maximum(theta_d, 1e-12), 0.0)
            uy = np.where(theta_d > 1e-12, my / np.maximum(theta_d, 1e-12), 0.0)
        rays = np.stack([s * ux, s * uy, np.cos(theta)], axis=1)
        n = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.maximum(n, 1e-12)

    def to_dict(self) -> dict:
        d = super(FisheyeModel, self).to_dict()
        d["max_fov_deg"] = round(self.max_fov_deg, 3)
        return d


class PinholeModel(CameraModel):
    """OpenCV pinhole with radial-tangential distortion (k1, k2, p1, p2, k3)."""

    kind = "pinhole"

    def __init__(self, K: Sequence, D: Optional[Sequence], width: int, height: int):
        super(PinholeModel, self).__init__(K, D if D is not None else [], width, height)
        d = np.zeros(5)
        d[:min(5, self.D.size)] = self.D[:5]
        self.D = d
        self.k1, self.k2, self.p1, self.p2, self.k3 = [float(x) for x in d]
        # Normalised radius beyond which the distortion model is not trusted:
        # the undistorted image corners, with margin.
        corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], float)
        mx, my = self._undistort_norm(*self._pix_to_norm(corners))
        self.r_max = float(np.max(np.hypot(mx, my)) * 1.25 + 1e-6)

    def _distort(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def _undistort_norm(self, xd: np.ndarray, yd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fixed-point iteration (same scheme as cv2.undistortPoints), more steps.
        x = xd.copy()
        y = yd.copy()
        if not np.any(self.D):
            return x, y
        for _ in range(30):
            r2 = x * x + y * y
            radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
            radial = np.where(np.abs(radial) < 1e-9, 1e-9, radial)
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return x, y

    def project(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """uv of points behind the camera / far outside the FOV is -1."""
        p = _as_pts_fast(pts_cam, 3)
        n = p.shape[0]
        dt = p.dtype
        uv = np.full((n, 2), -1.0, dtype=dt)
        valid = np.zeros(n, dtype=bool)
        if n == 0:
            return uv, valid
        z = p[:, 2]
        idx = np.flatnonzero(z > 1e-6)
        zi = z[idx]
        x = p[idx, 0] / zi
        y = p[idx, 1] / zi
        keep = (x * x + y * y) <= dt.type(self.r_max * self.r_max)
        idx, x, y = idx[keep], x[keep], y[keep]
        if idx.size == 0:
            return uv, valid
        if np.any(self.D):
            xd, yd = self._distort(x, y)
        else:
            xd, yd = x, y
        u = dt.type(self.fx) * xd + dt.type(self.skew) * yd + dt.type(self.cx)
        v = dt.type(self.fy) * yd + dt.type(self.cy)
        uv[idx, 0] = u
        uv[idx, 1] = v
        valid[idx] = (u >= 0) & (v >= 0) & (u <= self.width - 1.0) & (v <= self.height - 1.0)
        return uv, valid

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        q = _as_pts(uv, 2)
        x, y = self._undistort_norm(*self._pix_to_norm(q))
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def make_model(model_type: str, K: Sequence, D: Sequence, width: int, height: int,
               **kw) -> CameraModel:
    t = (model_type or "pinhole").lower()
    if t in ("fisheye", "kb", "kannala_brandt", "equidistant"):
        return FisheyeModel(K, D, width, height, **kw)
    if t in ("pinhole", "radtan", "opencv"):
        return PinholeModel(K, D, width, height)
    raise ValueError("unknown camera model type: %r" % model_type)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N,3) points (float32 stays float32)."""
    p = _as_pts_fast(pts, 3)
    T = np.asarray(T, dtype=p.dtype)
    out = p @ T[:3, :3].T
    out += T[:3, 3]
    return out


def project_base(spec, pts_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project base-frame points into camera `spec` (a rig.CameraSpec).

    Returns (uv (N,2), valid (N,) bool, depth_cam (N,) = z in camera frame).
    valid = in front of the lens (pinhole: z > 0, fisheye: theta < fov/2)
    and inside the image.
    """
    pts_cam = transform_points(spec.T_cam_base, pts_base)
    uv, valid = spec.model.project(pts_cam)
    return uv, valid, pts_cam[:, 2]


def rays_base(spec, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels -> (origin (3,), unit ray directions (N,3)) in the base frame."""
    r = spec.model.unproject(uv)
    T = np.asarray(spec.T_base_cam, dtype=np.float64)
    return T[:3, 3].copy(), r @ T[:3, :3].T
