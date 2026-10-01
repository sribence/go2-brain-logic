"""Cylindrical rectification LUTs (fisheye/pinhole -> level cylinder).

The cylinder axis is the base-frame z axis (gravity-aligned when the robot
stands level), centred on the camera's heading, so the horizon is a straight
horizontal line regardless of the camera's down-pitch. Columns are linear in
azimuth, rows linear in tan(elevation) -> people stay upright and unstretched,
which is what the person detector wants.

The LUT is built once (numpy); apply() is a single cv2.remap.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _dirs(u01: np.ndarray, v01: np.ndarray, hfov_deg: float, vfov_deg: float,
          center_yaw_deg: float, center_pitch_deg: float) -> np.ndarray:
    """Normalised cylinder coords (0..1, left->right / top->bottom) -> unit
    base-frame directions (N,3)."""
    az = np.radians(center_yaw_deg) + (0.5 - u01) * np.radians(hfov_deg)  # right = lower yaw
    et = (1.0 - 2.0 * v01) * np.tan(np.radians(vfov_deg) / 2.0)          # tan(elevation)
    d = np.stack([np.cos(az), np.sin(az), et], axis=-1).reshape(-1, 3)
    if center_pitch_deg:
        # tilt the cylinder down by center_pitch_deg about the heading's side axis
        p = np.radians(center_pitch_deg)
        yaw = np.radians(center_yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        cp, sp = np.cos(p), np.sin(p)
        Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        d = d @ (Rz @ Ry @ Rz.T).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cyl_dirs(hfov_deg: float, vfov_deg: float, out_w: int, out_h: int,
              center_yaw_deg: float, center_pitch_deg: float) -> np.ndarray:
    """(out_h*out_w, 3) unit directions in the base frame for each output pixel."""
    u = (np.arange(out_w, dtype=np.float64) + 0.5) / out_w
    v = (np.arange(out_h, dtype=np.float64) + 0.5) / out_h
    U, V = np.meshgrid(u, v)
    return _dirs(U.ravel(), V.ravel(), hfov_deg, vfov_deg, center_yaw_deg, center_pitch_deg)


def build_cylindrical_lut(spec, hfov_deg: float, vfov_deg: float, out_w: int, out_h: int,
                          center_yaw_deg: Optional[float] = None,
                          center_pitch_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """LUT for cv2.remap: rectified (out_h, out_w) -> source pixel coords.

    Pixels whose ray falls outside the source image get -1 (cv2.remap with
    BORDER_CONSTANT renders them black). center_yaw_deg defaults to the
    camera's base-frame heading.
    """
    yaw = spec.yaw_deg if center_yaw_deg is None else float(center_yaw_deg)
    d_base = _cyl_dirs(hfov_deg, vfov_deg, out_w, out_h, yaw, center_pitch_deg)
    R_cam_base = np.asarray(spec.T_cam_base, dtype=np.float64)[:3, :3]
    d_cam = d_base @ R_cam_base.T
    uv, valid = spec.model.project(d_cam)
    uv = np.where(valid[:, None], uv, -1.0)
    map_x = uv[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = uv[:, 1].reshape(out_h, out_w).astype(np.float32)
    return map_x, map_y


class Rectifier(object):
    """Cached cylindrical view of one camera.

    Defaults: fisheye -> 120 x 70 deg, pinhole -> its own HFOV x VFOV.
    out_h defaults to square pixels: out_w / hfov_rad * 2 tan(vfov/2).
    """

    def __init__(self, spec, hfov_deg: Optional[float] = None, vfov_deg: Optional[float] = None,
                 out_w: Optional[int] = None, out_h: Optional[int] = None,
                 center_yaw_deg: Optional[float] = None, center_pitch_deg: float = 0.0):
        self.spec = spec
        if hfov_deg is None or vfov_deg is None:
            if spec.model.kind == "fisheye":
                dh, dv = 120.0, 70.0
            else:
                dh = spec.model.hfov_deg()
                r = spec.model.unproject(np.array([[spec.model.cx, 0.0], [spec.model.cx, spec.height - 1.0]]))
                dv = float(np.degrees(np.arctan2(r[1, 1], r[1, 2]) - np.arctan2(r[0, 1], r[0, 2])))
            hfov_deg = dh if hfov_deg is None else hfov_deg
            vfov_deg = dv if vfov_deg is None else vfov_deg
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        if out_w is None:
            out_w = 640 if spec.model.kind == "fisheye" else spec.width
        if out_h is None:
            f = out_w / np.radians(self.hfov_deg)
            out_h = int(round(2.0 * f * np.tan(np.radians(self.vfov_deg) / 2.0)))
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.center_yaw_deg = spec.yaw_deg if center_yaw_deg is None else float(center_yaw_deg)
        self.center_pitch_deg = float(center_pitch_deg)
        self.map_x, self.map_y = build_cylindrical_lut(spec, self.hfov_deg, self.vfov_deg, self.out_w,
                                                       self.out_h, self.center_yaw_deg, self.center_pitch_deg)
        self.valid = self.map_x >= 0

    def apply(self, img: np.ndarray, interpolation: Optional[int] = None) -> np.ndarray:
        import cv2

        interp = cv2.INTER_LINEAR if interpolation is None else interpolation
        return cv2.remap(img, self.map_x, self.map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _lookup(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xi = np.clip(np.round(xs).astype(int), 0, self.out_w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, self.out_h - 1)
        mx = self.map_x[yi, xi]
        my = self.map_y[yi, xi]
        return mx, my, (mx >= 0) & (my >= 0)

    def to_original(self, bbox, samples: int = 9):
        """Rectified bbox (x1,y1,x2,y2) -> enclosing bbox in the original image.

        Samples the four edges (+ corners) of the box through the LUT; the
        enclosing box of the valid samples is returned, clipped to the image.
        None if no sample maps inside the original image.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        t = np.linspace(0.0, 1.0, max(int(samples), 2))
        xs = np.concatenate([x1 + (x2 - x1) * t, x1 + (x2 - x1) * t, np.full_like(t, x1), np.full_like(t, x2)])
        ys = np.concatenate([np.full_like(t, y1), np.full_like(t, y2), y1 + (y2 - y1) * t, y1 + (y2 - y1) * t])
        mx, my, ok = self._lookup(xs, ys)
        if not np.any(ok):
            return None
        mx, my = mx[ok], my[ok]
        w, h = self.spec.width, self.spec.height
        return (float(np.clip(mx.min(), 0, w - 1)), float(np.clip(my.min(), 0, h - 1)),
                float(np.clip(mx.max(), 0, w - 1)), float(np.clip(my.max(), 0, h - 1)))

    def pixel_to_ray_base(self, uv: np.ndarray) -> np.ndarray:
        """Rectified pixels (N,2) -> unit directions in base frame (analytic)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        u = (uv[:, 0] + 0.5) / self.out_w
        v = (uv[:, 1] + 0.5) / self.out_h
        return _dirs(u, v, self.hfov_deg, self.vfov_deg, self.center_yaw_deg, self.center_pitch_deg)
