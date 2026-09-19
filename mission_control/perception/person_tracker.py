"""Person detection + tracking + 3D localisation.

YOLOv8 (class 0 = person) with ultralytics' built-in ByteTrack for stable
2D track ids; each track gets a depth from the aligned depth image, is
deprojected to the camera optical frame, transformed to the robot base
frame and filtered per track id (`geometry3d.TrackSmoother`, a Kalman filter).

No robot I/O here -- the output is a plain dict consumed by `app.py`.
Following (turning the target into velocity commands) is a separate,
later step and must go through the armed/E-stop checks in `core`.
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np

from geometry3d import Extrinsics, KalmanConfig, TrackSmoother, robust_bbox_depth
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
        self.smoother = TrackSmoother(KalmanConfig.from_env(), max_age_s=max_age_s)
        self._bbox_ema: dict = {}           # track_id -> smoothed box, display only
        self.bbox_alpha = float(os.environ.get("BBOX_SMOOTH_ALPHA", "0.4"))

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

        for tid in self.smoother.prune(frame.t):
            self._bbox_ema.pop(tid, None)
        return {
            "t": frame.t,
            "infer_ms": round(infer_ms, 1),
            "image_size": [frame.intrinsics.width, frame.intrinsics.height],
            "persons": persons,
            "count": len(persons),
        }

    def _smooth_bbox(self, tid: int, box) -> dict:
        """EMA of the YOLO box for drawing; the raw box still feeds depth."""
        new = np.array(box, dtype=float)
        old = self._bbox_ema.get(tid)
        cur = new if old is None else self.bbox_alpha * new + (1 - self.bbox_alpha) * old
        self._bbox_ema[tid] = cur
        return {k: round(float(v), 1) for k, v in zip(("x1", "y1", "x2", "y2"), cur)}

    def _localise(self, frame: RGBDFrame, tid: int, conf: float, bbox) -> dict:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        z_m, u, v, ratio = robust_bbox_depth(frame.depth_mm, (x1, y1, x2, y2))
        person = {
            "track_id": tid,
            "confidence": round(conf, 3),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "bbox_smooth": self._smooth_bbox(tid, (x1, y1, x2, y2)),
            "pixel": {"u": round(u, 1), "v": round(v, 1)},
            "depth_valid_ratio": round(ratio, 2),
            "depth_ok": z_m is not None,
            "position_optical": None,
            "position_raw": None,
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
            "position_raw": _xyz(bx, by, bz),      # unsmoothed: used for identity gating
            "position": _xyz(s.x, s.y, s.z),       # smoothed: used for control
            "velocity": {"vx": round(s.vx, 3), "vy": round(s.vy, 3)},
            "distance_m": round(math.hypot(s.x, s.y), 3),
            "bearing_deg": round(math.degrees(math.atan2(s.y, s.x)), 1),
            "age_s": round(frame.t - s.first_t, 2),
            "hits": s.hits,
        })
        return person


def _xyz(x: float, y: float, z: float) -> dict:
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


STATE_COLORS = {  # BGR
    "IDLE": (160, 160, 160), "ACQUIRING": (0, 200, 255), "TRACKING": (255, 0, 255),
    "OCCLUDED": (0, 140, 255), "LOST": (0, 0, 255),
}


def annotate(frame: RGBDFrame, result: dict) -> np.ndarray:
    """Debug overlay: boxes + ids + distance, follow HUD, top-down radar.

    Locked target: thick box in the follow-state colour. Others: green
    (with depth) / orange (no depth).
    """
    import cv2

    img = frame.color_bgr.copy()
    follow = result.get("follow") or {}
    state = follow.get("state", "IDLE")
    fcol = STATE_COLORS.get(state, (255, 255, 255))
    tid_locked = follow.get("track_id")

    for p in result["persons"]:
        b = p.get("bbox_smooth") or p["bbox"]
        is_target = p["track_id"] == tid_locked and state in ("ACQUIRING", "TRACKING")
        color = fcol if is_target else ((0, 220, 0) if p["depth_ok"] else (0, 160, 255))
        cv2.rectangle(img, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), color, 4 if is_target else 2)
        label = f"#{p['track_id']}"
        label += f" {p['distance_m']:.2f}m {p['bearing_deg']:+.0f}deg" if p["depth_ok"] else " no depth"
        cv2.putText(img, label, (int(b["x1"]), max(15, int(b["y1"]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.circle(img, (int(p["pixel"]["u"]), int(p["pixel"]["v"])), 4, color, -1)

    # --- HUD (top-left) ---
    cmd = follow.get("command") or {}
    goal = follow.get("goal")
    lines = [f"{state}  #{tid_locked}" if tid_locked is not None else state, follow.get("reason", "")]
    if cmd:
        lines.append(f"vx {cmd['vx']:+.2f} m/s  vyaw {cmd['vyaw']:+.2f} rad/s  [DRY RUN]")
    if goal:
        lines.append(f"goal {goal['distance_cm']} cm @ {goal['yaw_deg']:+.0f} deg")
    if follow.get("similarity") is not None:
        lines.append(f"appearance {follow['similarity']:.2f}")
    y = 22
    for text in lines:
        if not text:
            continue
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (6, y - h - 5), (14 + w, y + 5), (20, 20, 20), -1)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fcol, 1)
        y += 22

    _draw_radar(img, result, follow, fcol)
    return img


def _draw_radar(img: np.ndarray, result: dict, follow: dict, fcol: tuple, size: int = 170, range_m: float = 6.0) -> None:
    """Top-down inset, bottom-right: robot at bottom centre, x forward = up,
    y left = left. Shows people, the identity gate, the goal point and the
    commanded velocity arrow."""
    import cv2

    h_img, w_img = img.shape[:2]
    ox, oy = w_img - size - 8, h_img - size - 8
    roi = img[oy:oy + size, ox:ox + size]
    roi[:] = (0.35 * roi).astype(np.uint8)
    scale = (size - 20) / range_m
    rx, ry = size // 2, size - 12

    def px(x: float, y: float) -> tuple:
        # Clamp to the inset so far-away people sit on the edge, not outside.
        return (int(np.clip(rx - y * scale, 2, size - 3)), int(np.clip(ry - x * scale, 2, size - 3)))

    for r in (1, 2, 3, 4, 5, 6):
        cv2.circle(roi, (rx, ry), int(r * scale), (70, 70, 70), 1)
    for p in result["persons"]:
        if p["depth_ok"]:
            col = fcol if p["track_id"] == follow.get("track_id") else (0, 220, 0)
            cv2.circle(roi, px(p["position"]["x"], p["position"]["y"]), 5, col, -1)
    gate = follow.get("gate")
    if gate:
        cv2.circle(roi, px(gate["x"], gate["y"]), max(2, int(gate["radius_m"] * scale)), fcol, 1)
    goal = follow.get("goal")
    if goal:
        gx, gy = px(goal["x"], goal["y"])
        cv2.drawMarker(roi, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
    cmd = follow.get("command") or {}
    vx, vyaw = cmd.get("vx", 0.0), cmd.get("vyaw", 0.0)
    if abs(vx) > 1e-3 or abs(vyaw) > 1e-3:
        # Where the robot would head: length = 2 s at vx (min 0.3 m so a
        # pure turn is still visible), angle = 1 s of vyaw.
        length = max(2.0 * vx, 0.3)
        tip = px(length * math.cos(vyaw), length * math.sin(vyaw))
        cv2.arrowedLine(roi, (rx, ry), tip, (255, 255, 255), 2, tipLength=0.3)
    cv2.rectangle(roi, (rx - 6, ry - 8), (rx + 6, ry + 8), (255, 255, 255), 1)
