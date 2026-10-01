"""LiDAR point colorization from the omni camera rig (CONTRACTS.md section 4 "C").

For every RGB camera the points are projected into the image, a coarse
per-camera z-buffer (default 160x120 cells) keeps only the points that are
not occluded (depth <= zbuf_min + tolerance), and the per-point colour comes
from the camera that sees it closest to its image centre. Thermal cameras
are processed the same way into a separate temperature channel (deg C).

Also contains two small geometry helpers used before the voxel map:
``deskew`` (per-point motion compensation with a linearly interpolated 2D
pose) and ``base_to_world`` (base frame -> world/map frame from x, y, yaw).

Everything is vectorized numpy; Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

ProjectFn = Callable[[object, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _default_project_fn() -> ProjectFn:
    # Lazy import so this module (and its tests) work without camera_model.
    from omni.camera_model import project_base  # type: ignore

    return project_base


def _np_version() -> Tuple[int, int]:
    try:
        parts = np.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:  # pragma: no cover
        return 0, 0


# ufunc.at became fast (~20x) in numpy 1.25; older numpy (Jetson, py3.8)
# uses the sort-based variant instead.
_FAST_UFUNC_AT = _np_version() >= (1, 25)


def _zbuffer_visible(cell: np.ndarray, depth: np.ndarray, n_cells: int,
                     tol: np.ndarray, use_ufunc_at: Optional[bool] = None) -> np.ndarray:
    """Return a bool mask of points whose depth is within tol of the per-cell min."""
    zbuf = np.full(n_cells, np.inf, dtype=np.float32)
    if use_ufunc_at is None:
        use_ufunc_at = _FAST_UFUNC_AT
    if use_ufunc_at:
        np.minimum.at(zbuf, cell, depth)
    else:
        # Write in descending depth order: the nearest point of each cell is
        # the last write to it (sequential 1-D fancy assignment).
        order = np.argsort(-depth)
        zbuf[cell[order]] = depth[order]
    return depth <= zbuf[cell] + tol


def _cone_cos(spec) -> Optional[float]:
    """cos of the half-angle of a cone around the optical axis that contains
    the whole image (from unprojecting the image border), with 5 deg margin.
    None if the camera model cannot unproject (then no culling is done)."""
    model = getattr(spec, "model", None)
    cached = getattr(spec, "_omni_cone", None)
    if cached is not None and cached[0] is model:
        return cached[1]
    val = None
    try:
        w, h = float(spec.width), float(spec.height)
        t = np.linspace(0.0, 1.0, 9)
        border = np.concatenate([
            np.stack([t * (w - 1), np.zeros_like(t)], 1),
            np.stack([t * (w - 1), np.full_like(t, h - 1)], 1),
            np.stack([np.zeros_like(t), t * (h - 1)], 1),
            np.stack([np.full_like(t, w - 1), t * (h - 1)], 1)])
        rays = np.asarray(spec.model.unproject(border), dtype=np.float64)
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        ang = np.arccos(np.clip(rays[:, 2], -1.0, 1.0)).max() + np.radians(5.0)
        if np.isfinite(ang):
            val = float(np.cos(min(ang, np.pi)))
    except Exception:
        val = None
    try:
        spec._omni_cone = (model, val)  # cache on the spec object
    except Exception:
        pass
    return val


def colorize(points_base: np.ndarray,
             frames: Dict[str, object],
             rig,
             zbuf_size: Tuple[int, int] = (160, 120),
             depth_tol: float = 0.15,
             project_fn: Optional[ProjectFn] = None,
             depth_tol_rel: float = 0.02,
             keep_uncolored: bool = False,
             thermal_zbuf_size: Tuple[int, int] = (64, 48),
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colorize base-frame points from the latest camera frames.

    Args:
        points_base: (N, 3) float points in the base frame.
        frames: cam_id -> Frame (rgb: HxWx3 uint8 BGR, thermal: HxW float32 degC).
        rig: object with ``cameras: dict cam_id -> CameraSpec`` (modality,
            width, height, T_base_cam, model).
        zbuf_size: (w, h) of the coarse z-buffer for RGB cameras.
        depth_tol: absolute occlusion tolerance (m).
        project_fn: ``f(spec, pts_base) -> (uv (N,2), valid (N,), depth_cam (N,))``;
            defaults to ``omni.camera_model.project_base``.
        depth_tol_rel: extra tolerance proportional to depth (grazing surfaces).
        keep_uncolored: if True, return all N points (rgb = 0 where no camera).
        thermal_zbuf_size: (w, h) of the z-buffer for thermal cameras.

    Returns:
        pts (M, 3) float32, rgb (M, 3) uint8 (RGB order), temp (M,) float32
        (NaN where no thermal camera sees the point). Without keep_uncolored,
        only points with an RGB colour are returned.
    """
    pts = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    rgb = np.zeros((n, 3), dtype=np.uint8)
    temp = np.full(n, np.nan, dtype=np.float32)
    has_rgb = np.zeros(n, dtype=bool)
    if n == 0:
        return pts, rgb, temp
    if project_fn is None:
        project_fn = _default_project_fn()

    best_rgb = np.full(n, -np.inf, dtype=np.float32)
    best_th = np.full(n, -np.inf, dtype=np.float32)
    cameras = getattr(rig, "cameras", rig)

    for cam_id, frame in frames.items():
        spec = cameras.get(cam_id) if frame is not None else None
        if spec is None:
            continue
        modality = getattr(spec, "modality", getattr(frame, "modality", ""))
        if modality not in ("rgb", "thermal"):
            continue
        img = frame.image
        if img is None or img.ndim < 2:
            continue
        ih, iw = img.shape[:2]

        # Occlusion uses the Euclidean range from the optical centre (works for
        # fisheye lenses with FOV > 180 deg where depth_cam can be <= 0).
        T = np.asarray(spec.T_base_cam, dtype=np.float32)
        dp = pts - T[:3, 3]
        depth = np.sqrt(np.einsum("ij,ij->i", dp, dp))
        # Cheap cone cull before the (expensive) lens projection.
        cand = None
        cos_min = _cone_cos(spec)
        if cos_min is not None:
            cand = np.nonzero(dp @ T[:3, 2] >= cos_min * depth)[0]
            if cand.size == 0:
                continue
            uv, valid, _ = project_fn(spec, pts[cand])
        else:
            uv, valid, _ = project_fn(spec, pts)
        uv = np.asarray(uv, dtype=np.float32)
        # The frame may be delivered at a different resolution than the spec.
        sx = iw / float(spec.width)
        sy = ih / float(spec.height)
        u = uv[:, 0] * sx
        v = uv[:, 1] * sy
        m = (np.asarray(valid, dtype=bool)
             & (u >= 0.0) & (u <= iw - 1) & (v >= 0.0) & (v <= ih - 1))
        sub = np.nonzero(m)[0]
        if sub.size == 0:
            continue
        u = u[sub]
        v = v[sub]
        idx = cand[sub] if cand is not None else sub
        d = depth[idx]

        zw, zh = zbuf_size if modality == "rgb" else thermal_zbuf_size
        cx = np.minimum((u * (zw / float(iw))).astype(np.int32), zw - 1)
        cy = np.minimum((v * (zh / float(ih))).astype(np.int32), zh - 1)
        vis = _zbuffer_visible(cy * zw + cx, d, zw * zh, depth_tol + depth_tol_rel * d)
        idx = idx[vis]
        u = u[vis]
        v = v[vis]

        # Score: 1 at the image centre, ~0 at the corners (prefer central pixels,
        # least distortion / vignetting on fisheye lenses).
        du = (u - 0.5 * (iw - 1)) / (0.5 * iw)
        dv = (v - 0.5 * (ih - 1)) / (0.5 * ih)
        score = 1.0 - np.sqrt(0.5 * (du * du + dv * dv))

        ui = (u + 0.5).astype(np.intp)
        vi = (v + 0.5).astype(np.intp)
        if modality == "rgb":
            better = score > best_rgb[idx]
            sel = idx[better]
            best_rgb[sel] = score[better]
            if img.ndim == 3:
                rgb[sel] = img[vi[better], ui[better], 2::-1]  # BGR -> RGB
            else:
                g = img[vi[better], ui[better]]
                rgb[sel] = np.repeat(g[:, None], 3, axis=1)
            has_rgb[sel] = True
        else:
            tv = img[vi, ui].astype(np.float32)
            ok = np.isfinite(tv) & (score > best_th[idx])
            sel = idx[ok]
            best_th[sel] = score[ok]
            temp[sel] = tv[ok]

    if keep_uncolored:
        return pts, rgb, temp
    return pts[has_rgb], rgb[has_rgb], temp[has_rgb]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot2(pts: np.ndarray, c, s) -> Tuple[np.ndarray, np.ndarray]:
    x = pts[:, 0]
    y = pts[:, 1]
    return c * x - s * y, s * x + c * y


def base_to_world(points_base: np.ndarray, pose) -> np.ndarray:
    """Transform (N,3) base-frame points to world with a 2D pose (x, y, yaw)."""
    p = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    x0, y0, yaw = (float(pose[0]), float(pose[1]), float(pose[2]))
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(p)
    wx, wy = _rot2(p, c, s)
    out[:, 0] = wx + x0
    out[:, 1] = wy + y0
    out[:, 2] = p[:, 2]
    return out


def deskew(points: np.ndarray, point_times: np.ndarray,
           pose_at_fn: Callable[[float], tuple],
           t_ref: Optional[float] = None) -> np.ndarray:
    """Motion-compensate a LiDAR sweep into the base frame at ``t_ref``.

    Each point was measured in the base frame at its own time. The pose is
    sampled at the first and last point time only and interpolated linearly
    (x, y, unwrapped yaw) per point. ``t_ref`` defaults to the latest time.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if p.shape[0] == 0:
        return p.copy()
    t = np.asarray(point_times, dtype=np.float64).reshape(-1)
    t0 = float(t.min())
    t1 = float(t.max())
    if t_ref is None:
        t_ref = t1
    p0 = np.asarray(pose_at_fn(t0), dtype=np.float64)
    p1 = np.asarray(pose_at_fn(t1), dtype=np.float64)
    pr = np.asarray(pose_at_fn(float(t_ref)), dtype=np.float64)
    dyaw = (p1[2] - p0[2] + np.pi) % (2 * np.pi) - np.pi
    a = (t - t0) / (t1 - t0) if t1 > t0 else np.zeros_like(t)
    px = p0[0] + a * (p1[0] - p0[0])
    py = p0[1] + a * (p1[1] - p0[1])
    yaw = p0[2] + a * dyaw
    # Point -> world with its own pose.
    wx, wy = _rot2(p, np.cos(yaw), np.sin(yaw))
    wx = wx + px - pr[0]
    wy = wy + py - pr[1]
    # World -> base at t_ref.
    c, s = np.cos(-pr[2]), np.sin(-pr[2])
    out = np.empty_like(p)
    out[:, 0] = c * wx - s * wy
    out[:, 1] = s * wx + c * wy
    out[:, 2] = p[:, 2]
    return out
