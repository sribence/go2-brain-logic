"""360 degree person perception: per-camera detection -> 3D in base frame -> global tracks (Agent B).

``Perception360(rig, detector, thermal_detector=None, rectifiers=None, cfg=None)``
``.process(frames, lidar_base, t, ego_delta=None) -> list[PersonTrack]``

Range per detection, in priority order:
  (a) "lidar"        LiDAR points (base frame) projected into the camera, inside the bbox,
                     ground removed, >= min_pts -> 30th percentile of horizontal distance
  (b) "tof"          the same with the tof_rear depth frame turned into base-frame points
                     (median, 0.2-2.5 m), only if the bbox ray sees the ToF FOV
  (c) "ground_plane" bbox bottom-centre ray intersected with z = ground_z_base
  (d) "thermal_size" apparent height of a 1.7 m person (thermal / fallback)
The 3D point is the bbox-centre ray at that horizontal range from the camera.
A simple ToF blob detector also emits "depth" detections for anything close
behind the robot.

Only the contract API of the rig is used: ``rig.cameras[cam_id]`` with
``modality, width, height, model.project(pts_cam)->(uv, valid)``,
``model.unproject(uv)->rays_cam`` and ``T_base_cam`` (4x4, p_base = T @ p_cam).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .omni_types import Detection, Frame, PersonTrack
    from .thermal_detect import ThermalPersonDetector
    from .track import Det3D, MultiCamTracker
except ImportError:  # loaded as top-level modules
    from omni_types import Detection, Frame, PersonTrack  # type: ignore
    from thermal_detect import ThermalPersonDetector  # type: ignore
    from track import Det3D, MultiCamTracker  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Perception360Config:
    ground_z_base: float = -0.30        # Go2 base ~0.30 m above the floor
    ground_margin_m: float = 0.05       # points below ground + margin are ground
    max_point_z_above_ground: float = 2.3
    lidar_min_pts: int = 5
    lidar_percentile: float = 30.0
    bbox_shrink_x: float = 0.15         # ignore outer 15% left/right when sampling points
    tof_cam: str = "tof_rear"
    tof_min_m: float = 0.2
    tof_max_m: float = 2.5
    tof_min_pts: int = 5
    tof_stride: int = 2                 # pixel subsampling of the 100x100 ToF frame
    tof_blob: bool = True
    tof_blob_cell_m: float = 0.10
    tof_blob_min_h: float = 0.2         # visible height above ground of a cluster
    tof_blob_max_h: float = 2.2
    tof_blob_max_w: float = 1.2         # footprint extent (walls are rejected)
    tof_blob_min_pts: int = 15
    tof_blob_conf: float = 0.45
    person_height_m: float = 1.7
    max_range_m: float = 30.0
    min_det_conf: float = 0.3
    max_cams_per_tick: int = 0          # 0 = no limit (all RGB cams every tick)
    heading_cam: str = "rgb_front"      # always processed when budget-limited
    use_thermal: bool = True
    tracker: dict = field(default_factory=dict)  # MultiCamTracker kwargs


def _cfg(cfg) -> Perception360Config:
    if cfg is None:
        return Perception360Config()
    if isinstance(cfg, Perception360Config):
        return cfg
    keys = Perception360Config.__dataclass_fields__.keys()
    return Perception360Config(**{k: v for k, v in dict(cfg).items() if k in keys})


# ------------------------------------------------------------------ geometry helpers

def project_base(spec, pts_base: np.ndarray):
    """(uv, valid, depth_cam) of base-frame points in a camera (contract API stand-in)."""
    pts_base = np.asarray(pts_base, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(spec.T_base_cam, dtype=np.float64)
    T_inv = np.linalg.inv(T)
    pc = pts_base @ T_inv[:3, :3].T + T_inv[:3, 3]
    if len(pc) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool), np.zeros(0)
    uv, valid = spec.model.project(pc)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    valid &= np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < spec.width) & (uv[:, 1] >= 0) & (uv[:, 1] < spec.height)
    return uv, valid, pc[:, 2]


def rays_base(spec, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Camera origin (3,) and unit ray directions (N,3) in base frame for pixels uv."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(spec.model.unproject(uv), dtype=np.float64).reshape(-1, 3)
    T = np.asarray(spec.T_base_cam, dtype=np.float64)
    d = r @ T[:3, :3].T
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return T[:3, 3].copy(), d / n


def _hdist(origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])


def point_on_ray_at_range(origin: np.ndarray, d: np.ndarray, r_h: float) -> Optional[np.ndarray]:
    """Point along ray d whose horizontal distance from origin is r_h."""
    h = math.hypot(d[0], d[1])
    if h < 1e-6:
        return None
    return origin + d * (r_h / h)


# ------------------------------------------------------------------ main class

class Perception360:
    def __init__(self, rig, detector, thermal_detector=None, rectifiers=None, cfg=None, tracker=None):
        self.rig = rig
        self.detector = detector
        self.thermal_detector = thermal_detector
        self.rectifiers = dict(rectifiers or {})
        self.cfg = _cfg(cfg)
        self.tracker = tracker or MultiCamTracker(**self.cfg.tracker)
        self._rr = 0  # round-robin pointer
        self.last_detections = []  # type: List[Tuple[Detection, Det3D]]
        self.last_stats = {}  # type: dict

    @property
    def cameras(self) -> dict:
        return getattr(self.rig, "cameras", self.rig)

    # ---- scheduling
    def select_rgb_cams(self, available: Sequence[str], heading_cam: Optional[str] = None) -> List[str]:
        cams = [c for c in sorted(available)]
        n = int(self.cfg.max_cams_per_tick or 0)
        if n <= 0 or len(cams) <= n:
            return cams
        head = heading_cam or self.cfg.heading_cam
        chosen = [head] if head in cams else []
        others = [c for c in cams if c not in chosen]
        k = n - len(chosen)
        if k > 0 and others:
            start = self._rr % len(others)
            rot = others[start:] + others[:start]
            chosen += rot[:k]
            self._rr = (start + k) % len(others)
        return chosen

    # ---- main entry
    def process(self, frames: Dict[str, Frame], lidar_base: Optional[np.ndarray], t: float,
                ego_delta=None, heading_cam: Optional[str] = None) -> List[PersonTrack]:
        t0 = time.time()
        cams = self.cameras
        frames = {k: f for k, f in (frames or {}).items() if f is not None and k in cams}
        lidar = None
        if lidar_base is not None and len(lidar_base):
            lidar = self._filter_ground(np.asarray(lidar_base, dtype=np.float64)[:, :3])

        tof_pts = None
        tof_frame = frames.get(self.cfg.tof_cam)
        if tof_frame is not None:
            tof_pts = self.tof_points_base(cams[self.cfg.tof_cam], tof_frame.image)

        dets = []  # type: List[Detection]
        rgb_ids = [c for c, f in frames.items() if f.modality == "rgb"]
        sel = self.select_rgb_cams(rgb_ids, heading_cam)
        dets += self._detect_rgb(frames, sel)
        if self.thermal_detector is not None and self.cfg.use_thermal:
            for cid, f in frames.items():
                if f.modality != "thermal":
                    continue
                for bbox, conf in self.thermal_detector.detect(f.image):
                    dets.append(Detection(cid, tuple(float(v) for v in bbox), float(conf), "person", "thermal"))

        out3d = []  # type: List[Det3D]
        pairs = []
        for d in dets:
            if d.conf < self.cfg.min_det_conf:
                continue
            d3 = self.localize(d, lidar, tof_pts)
            if d3 is not None:
                out3d.append(d3)
                pairs.append((d, d3))
        if tof_pts is not None and self.cfg.tof_blob:
            for d3 in self.tof_blobs(tof_pts):
                out3d.append(d3)
                pairs.append((None, d3))
        self.last_detections = pairs
        tracks = self.tracker.update(out3d, t, ego_delta)
        self.last_stats = {"rgb_cams": sel, "n_det": len(dets), "n_det3d": len(out3d),
                           "n_tracks": len(tracks), "ms": round((time.time() - t0) * 1000.0, 2)}
        return tracks

    # ---- detection
    def _detect_rgb(self, frames: Dict[str, Frame], sel: List[str]) -> List[Detection]:
        if not sel or self.detector is None:
            return []
        imgs = []
        for cid in sel:
            img = frames[cid].image
            rect = self.rectifiers.get(cid)
            if rect is not None:
                try:
                    img = rect.apply(img)
                except Exception as e:  # keep running on the raw image
                    logger.warning("rectify %s failed: %s", cid, e)
                    rect = None
            imgs.append((cid, img, rect))
        results = self.detector.detect([im for _, im, _ in imgs])
        dets = []
        for (cid, _, rect), res in zip(imgs, results):
            for bbox, conf, cls in res:
                if cls != "person":
                    continue
                if rect is not None:
                    bbox = rect.to_original(bbox)
                spec = self.cameras[cid]
                x1, y1, x2, y2 = [float(v) for v in bbox]
                x1, x2 = max(0.0, min(x1, x2)), min(float(spec.width), max(x1, x2))
                y1, y2 = max(0.0, min(y1, y2)), min(float(spec.height), max(y1, y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                dets.append(Detection(cid, (x1, y1, x2, y2), float(conf), "person", "rgb"))
        return dets

    # ---- range + 3D
    def _filter_ground(self, pts: np.ndarray) -> np.ndarray:
        g = self.cfg.ground_z_base
        m = (pts[:, 2] > g + self.cfg.ground_margin_m) & (pts[:, 2] < g + self.cfg.max_point_z_above_ground)
        return pts[m & np.isfinite(pts).all(axis=1)]

    def _pts_in_bbox(self, spec, bbox, pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pts is None or len(pts) == 0:
            return None
        uv, valid, depth = project_base(spec, pts)
        x1, y1, x2, y2 = bbox
        sx = (x2 - x1) * self.cfg.bbox_shrink_x
        m = valid & (uv[:, 0] >= x1 + sx) & (uv[:, 0] <= x2 - sx) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        return pts[m]

    def range_lidar(self, spec, bbox, lidar_base: Optional[np.ndarray]) -> Optional[float]:
        inside = self._pts_in_bbox(spec, bbox, lidar_base)
        if inside is None or len(inside) < self.cfg.lidar_min_pts:
            return None
        origin = np.asarray(spec.T_base_cam, dtype=np.float64)[:3, 3]
        return float(np.percentile(_hdist(origin, inside), self.cfg.lidar_percentile))

    def range_tof(self, spec, bbox, tof_pts: Optional[np.ndarray]) -> Optional[float]:
        inside = self._pts_in_bbox(spec, bbox, tof_pts)
        if inside is None or len(inside) < self.cfg.tof_min_pts:
            return None
        origin = np.asarray(spec.T_base_cam, dtype=np.float64)[:3, 3]
        return float(np.median(_hdist(origin, inside)))

    def range_ground(self, spec, bbox) -> Optional[float]:
        x1, y1, x2, y2 = bbox
        o, d = rays_base(spec, np.array([[(x1 + x2) / 2.0, y2]]))
        d = d[0]
        dz = self.cfg.ground_z_base - o[2]
        if d[2] >= -1e-3 or dz >= 0:
            return None
        s = dz / d[2]
        r = math.hypot(d[0] * s, d[1] * s)
        return r if 0.1 < r <= self.cfg.max_range_m else None

    def range_size(self, spec, bbox) -> Optional[float]:
        """Range from the angle subtended by the bbox height (works for fisheye too)."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        _, d = rays_base(spec, np.array([[cx, y1], [cx, y2]]))
        ang = math.acos(float(np.clip(np.dot(d[0], d[1]), -1.0, 1.0)))
        if ang < 1e-4:
            return None
        r = self.cfg.person_height_m / (2.0 * math.tan(ang / 2.0))
        return r if r <= self.cfg.max_range_m else None

    def localize(self, det: Detection, lidar_base: Optional[np.ndarray],
                 tof_pts: Optional[np.ndarray]) -> Optional[Det3D]:
        spec = self.cameras[det.cam_id]
        r, src = self.range_lidar(spec, det.bbox, lidar_base), "lidar"
        if r is None and det.cam_id != self.cfg.tof_cam:
            r, src = self.range_tof(spec, det.bbox, tof_pts), "tof"
        if r is None and det.modality == "rgb":
            r, src = self.range_ground(spec, det.bbox), "ground_plane"
        if r is None:
            r, src = self.range_size(spec, det.bbox), "thermal_size"
        if r is None:
            r, src = self.range_ground(spec, det.bbox), "ground_plane"
        if r is None:
            return None
        x1, y1, x2, y2 = det.bbox
        o, d = rays_base(spec, np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]))
        p = point_on_ray_at_range(o, d[0], r)
        if p is None:
            return None
        height = 0.0
        if src != "thermal_size":  # size-based range assumes the height, can't measure it
            ot, dt = rays_base(spec, np.array([[(x1 + x2) / 2.0, y1]]))
            top = point_on_ray_at_range(ot, dt[0], r)
            if top is not None:
                height = max(0.0, float(top[2]) - self.cfg.ground_z_base)
        return Det3D(float(p[0]), float(p[1]), float(p[2]), float(det.conf), det.cam_id, det.modality, src,
                     height)

    # ---- ToF
    def tof_points_base(self, spec, depth: np.ndarray) -> Optional[np.ndarray]:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            return None
        s = max(1, int(self.cfg.tof_stride))
        h, w = depth.shape
        vs, us = np.mgrid[0:h:s, 0:w:s]
        z = depth[vs, us].reshape(-1)
        uv = np.stack([us.reshape(-1), vs.reshape(-1)], axis=1).astype(np.float64) + 0.5
        sx = spec.width / float(w)
        sy = spec.height / float(h)
        uv[:, 0] *= sx
        uv[:, 1] *= sy
        m = np.isfinite(z) & (z >= self.cfg.tof_min_m) & (z <= self.cfg.tof_max_m)
        if not m.any():
            return np.zeros((0, 3))
        rays = np.asarray(spec.model.unproject(uv[m]), dtype=np.float64).reshape(-1, 3)
        rz = rays[:, 2]
        ok = rz > 1e-3
        pc = rays[ok] * (z[m][ok] / rz[ok])[:, None]  # depth is z-depth (optical axis)
        T = np.asarray(spec.T_base_cam, dtype=np.float64)
        pb = pc @ T[:3, :3].T + T[:3, 3]
        return self._filter_ground(pb)

    def tof_blobs(self, tof_pts: np.ndarray) -> List[Det3D]:
        """Cluster above-ground ToF points on an xy grid -> 'depth' detections."""
        c = self.cfg
        if tof_pts is None or len(tof_pts) < c.tof_blob_min_pts:
            return []
        cell = c.tof_blob_cell_m
        ij = np.floor(tof_pts[:, :2] / cell).astype(np.int64)
        i0, j0 = ij.min(axis=0)
        ij -= np.array([i0, j0])
        H, W = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
        if H * W > 400 * 400:
            return []
        occ = np.zeros((H, W), np.uint8)
        occ[ij[:, 0], ij[:, 1]] = 255
        occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        n, labels = cv2.connectedComponents(occ, connectivity=8)
        lab = labels[ij[:, 0], ij[:, 1]]
        out = []
        tof_spec = self.cameras.get(c.tof_cam)
        for k in range(1, n):
            p = tof_pts[lab == k]
            if len(p) < c.tof_blob_min_pts:
                continue
            h = float(p[:, 2].max() - c.ground_z_base)
            ext = np.ptp(p[:, :2], axis=0)
            if h < c.tof_blob_min_h or h > c.tof_blob_max_h or float(ext.max()) > c.tof_blob_max_w:
                continue
            ctr = np.median(p, axis=0)
            out.append(Det3D(float(ctr[0]), float(ctr[1]), float(ctr[2]), c.tof_blob_conf,
                             c.tof_cam if tof_spec is not None else "tof", "depth", "tof", h))
        return out
