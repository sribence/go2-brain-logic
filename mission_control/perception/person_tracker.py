"""Person detection + tracking + 3D localisation.

YOLOv8 (class 0 = person) with ultralytics' built-in ByteTrack for stable
2D track ids; each track gets a depth from the aligned depth image, is
deprojected to the camera optical frame, transformed to the robot base
frame and smoothed per track id (`geometry3d.TrackSmoother`).

No robot I/O here -- the output is a plain dict consumed by `app.py`.
Following (turning the target into velocity commands) is a separate,
later step and must go through the armed/E-stop checks in `core`.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from geometry3d import Extrinsics, TrackSmoother, robust_bbox_depth
from rgbd_source import RGBDFrame

PERSON_CLASS = 0


class PersonTracker:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.4,
        extrinsics: Optional[Extrinsics] = None,
        tracker_cfg: str = "bytetrack.yaml",
        device: Optional[str] = None,
        imgsz: int = 640,
        max_age_s: float = 1.0,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self.device = device
        self.imgsz = imgsz
        self.extrinsics = extrinsics or Extrinsics()
        self.smoother = TrackSmoother(max_age_s=max_age_s)
        self._lock = threading.Lock()
        self._locked_target: Optional[int] = None

    # ------------------------------------------------------------------ target

    def lock_target(self, track_id: Optional[int]) -> None:
        """Lock following to one track id; None = auto (nearest person)."""
        with self._lock:
            self._locked_target = track_id

    @property
    def locked_target(self) -> Optional[int]:
        return self._locked_target

    # ------------------------------------------------------------------ core

    def process(self, frame: RGBDFrame) -> dict:
        t0 = time.time()
        kwargs = dict(persist=True, classes=[PERSON_CLASS], conf=self.conf,
                      tracker=self.tracker_cfg, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        result = self.model.track(frame.color_bgr, **kwargs)[0]
        infer_ms = (time.time() - t0) * 1000.0

        persons = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
                persons.append(self._localise(frame, int(tid), float(c), (x1, y1, x2, y2)))

        self.smoother.prune(frame.t)
        target_id = self._pick_target(persons)
        return {
            "t": frame.t,
            "infer_ms": round(infer_ms, 1),
            "image_size": [frame.intrinsics.width, frame.intrinsics.height],
            "persons": persons,
            "count": len(persons),
            "target_id": target_id,
            "target_mode": "locked" if self._locked_target is not None else "nearest",
        }

    def _localise(self, frame: RGBDFrame, tid: int, conf: float, bbox) -> dict:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        z_m, u, v, ratio = robust_bbox_depth(frame.depth_mm, (x1, y1, x2, y2))
        person = {
            "track_id": tid,
            "confidence": round(conf, 3),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "pixel": {"u": round(u, 1), "v": round(v, 1)},
            "depth_valid_ratio": round(ratio, 2),
            "depth_ok": z_m is not None,
            "position_optical": None,
            "position": None,
            "velocity": None,
            "distance_m": None,
            "bearing_deg": None,
        }
        if z_m is None:
            return person

        p_opt = frame.intrinsics.deproject(u, v, z_m)
        bx, by, bz = self.extrinsics.optical_to_base(p_opt)
        s = self.smoother.update(tid, bx, by, bz, frame.t)
        person.update({
            "position_optical": _xyz(*p_opt),
            "position": _xyz(s.x, s.y, s.z),
            "velocity": {"vx": round(s.vx, 3), "vy": round(s.vy, 3)},
            "distance_m": round(math.hypot(s.x, s.y), 3),
            "bearing_deg": round(math.degrees(math.atan2(s.y, s.x)), 1),
            "age_s": round(frame.t - s.first_t, 2),
            "hits": s.hits,
        })
        return person

    def _pick_target(self, persons: list[dict]) -> Optional[int]:
        with_3d = [p for p in persons if p["depth_ok"]]
        locked = self._locked_target
        if locked is not None:
            # Never silently switch to someone else: a lost locked target
            # yields None so the follower stops instead of chasing a stranger.
            return locked if any(p["track_id"] == locked for p in with_3d) else None
        if not with_3d:
            return None
        return min(with_3d, key=lambda p: p["distance_m"])["track_id"]


def _xyz(x: float, y: float, z: float) -> dict:
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


def annotate(frame: RGBDFrame, result: dict) -> np.ndarray:
    """Debug overlay: boxes, track ids, distance; target in magenta."""
    import cv2

    img = frame.color_bgr.copy()
    for p in result["persons"]:
        b = p["bbox"]
        is_target = p["track_id"] == result["target_id"]
        color = (255, 0, 255) if is_target else ((0, 220, 0) if p["depth_ok"] else (0, 160, 255))
        cv2.rectangle(img, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), color, 2)
        label = f"#{p['track_id']}"
        if p["depth_ok"]:
            label += f" {p['distance_m']:.2f}m {p['bearing_deg']:+.0f}deg"
        else:
            label += " no depth"
        cv2.putText(img, label, (int(b["x1"]), max(15, int(b["y1"]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.circle(img, (int(p["pixel"]["u"]), int(p["pixel"]["v"])), 4, color, -1)
    return img
