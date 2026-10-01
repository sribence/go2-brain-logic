"""Gesture recognition on COCO-17 pose keypoints (M6, mission CONTRACT.md section 6).

Pure numpy. Per tracked person (gid) a temporal buffer of 2D keypoints
``(17, 3) = (x, y, conf)`` in image pixels (y down) is classified into:

* ``wave``      wrist above shoulder + lateral wrist oscillation >= 2 cycles within 2 s
* ``stop_palm`` arm extended up/forward (wrist above shoulder, elbow straight), held still >= 1 s
* ``point``     arm extended sideways (roughly horizontal, straight), held still >= 0.8 s;
                direction -> image angle and an approximate robot-frame bearing

Scores pass through per-(gid, gesture) hysteresis (on/off thresholds, confirm/release
counts, cooldown) and rising edges are emitted as events
``{gid, gesture, conf, t, side, [bearing], [dir_img_deg]}`` -> Redis ``mc.omni.gesture``.

2D approximation notes: a palm pushed straight at the camera is foreshortened and not
detected as ``stop_palm``; ``point`` bearing is the person's bearing offset by
``point_offset_deg`` toward the pointed side (no depth), good enough to "look there".

Also: :class:`PoseDetector` (lazy ultralytics ``yolo11n-pose`` / ``yolov8n-pose``,
``.pt`` or TensorRT ``.engine``, batched like ``detect.UltralyticsDetector``) and
:func:`match_pose_to_tracks` (greedy IoU between pose boxes and tracked person boxes).
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12
ARMS = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}
GESTURES = ("wave", "stop_palm", "point")


@dataclass
class GestureConfig:
    kp_min_conf: float = 0.3
    buffer_s: float = 2.5
    # wave
    wave_window_s: float = 2.0
    wave_min_cycles: float = 2.0
    wave_amp: float = 0.08          # hysteresis band on lateral wrist offset (torso units)
    wave_above_frac: float = 0.7
    # stop palm
    stop_hold_s: float = 1.0
    stop_min_elev_deg: float = 40.0
    stop_max_motion: float = 0.06   # wrist std, torso units
    # point
    point_hold_s: float = 0.8
    point_max_elev_deg: float = 30.0
    point_min_out: float = 0.7      # |dx| / arm length
    point_max_motion: float = 0.08
    point_offset_deg: float = 30.0
    # shared
    min_ext: float = 0.85           # |shoulder->wrist| / (upper + forearm), 1 = straight
    hold_frac: float = 0.85
    min_frames: int = 6
    on_thr: float = 0.6
    off_thr: float = 0.35
    confirm_n: int = 2
    release_n: int = 3
    cooldown_s: float = 2.0
    max_age_s: float = 3.0


@dataclass
class _GState:
    active: bool = False
    hits: int = 0
    misses: int = 0
    last_emit_t: float = -1e9
    off_t: float = -1e9


class _Person:
    def __init__(self):
        self.buf = deque()  # type: Deque[Tuple[float, np.ndarray]]
        self.last_t = -1e9
        self.bearing_deg = None  # type: Optional[float]
        self.g = {g: _GState() for g in GESTURES}


def _as_kp(kp) -> Optional[np.ndarray]:
    a = np.asarray(kp, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 17 or a.shape[1] < 2:
        return None
    a = a[:17]
    if a.shape[1] == 2:
        a = np.concatenate([a, np.ones((17, 1))], axis=1)
    return a[:, :3]


def body_scale(kp: np.ndarray, min_conf: float) -> Optional[float]:
    """Torso length (shoulder mid -> hip mid) in px, fallbacks: shoulder width x1.6."""
    c = kp[:, 2]
    if min(c[L_SH], c[R_SH]) >= min_conf:
        sh = (kp[L_SH, :2] + kp[R_SH, :2]) / 2.0
        if min(c[L_HIP], c[R_HIP]) >= min_conf:
            s = float(np.linalg.norm(sh - (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0))
            if s > 1.0:
                return s
        w = float(np.linalg.norm(kp[L_SH, :2] - kp[R_SH, :2]))
        if w > 1.0:
            return 1.6 * w
    return None


def arm_features(kp: np.ndarray, side: str, min_conf: float) -> Optional[dict]:
    i_sh, i_el, i_wr = ARMS[side]
    if min(kp[i_sh, 2], kp[i_el, 2], kp[i_wr, 2]) < min_conf:
        return None
    sh, el, wr = kp[i_sh, :2], kp[i_el, :2], kp[i_wr, :2]
    arm_len = float(np.linalg.norm(el - sh) + np.linalg.norm(wr - el))
    if arm_len < 1.0:
        return None
    v = wr - sh
    return {
        "sh": sh, "el": el, "wr": wr,
        "conf": float(min(kp[i_sh, 2], kp[i_el, 2], kp[i_wr, 2])),
        "ext": float(np.linalg.norm(v)) / arm_len,
        "elev_deg": math.degrees(math.atan2(-v[1], abs(v[0]) + 1e-9)),   # 90 = up, 0 = horizontal
        "dir_img_deg": math.degrees(math.atan2(-v[1], v[0])),            # 0 = image right, 90 = up
        "out": float(v[0]) / arm_len,
        "above": bool(wr[1] < sh[1]),
    }


def count_half_cycles(x: Sequence[float], band: float) -> int:
    """Sign changes of a detrended signal with a hysteresis band (+-band)."""
    xs = np.asarray(x, dtype=np.float64)
    if xs.size < 3:
        return 0
    xs = xs - xs.mean()
    state, n = 0, 0
    for v in xs:
        s = 1 if v > band else (-1 if v < -band else 0)
        if s != 0 and s != state:
            if state != 0:
                n += 1
            state = s
    return n


class GestureClassifier:
    def __init__(self, cfg: Optional[GestureConfig] = None):
        self.cfg = cfg or GestureConfig()
        self._p = {}  # type: Dict[int, _Person]

    # -- public -----------------------------------------------------------
    def update(self, gid: int, keypoints, t: float, bearing_deg: Optional[float] = None) -> List[dict]:
        """Add one frame of keypoints for ``gid``; returns newly started gesture events."""
        kp = _as_kp(keypoints)
        p = self._p.get(gid)
        if p is None:
            p = self._p[gid] = _Person()
        if kp is None or t <= p.last_t:
            return []
        p.last_t = float(t)
        if bearing_deg is not None:
            p.bearing_deg = float(bearing_deg)
        p.buf.append((float(t), kp))
        while p.buf and p.buf[0][0] < t - self.cfg.buffer_s:
            p.buf.popleft()
        scores = self.scores(gid)
        return self._hysteresis(gid, p, scores, float(t))

    def update_many(self, persons: Sequence[dict], t: float) -> List[dict]:
        """``persons``: [{gid, keypoints, bearing_deg?}] (e.g. from match_pose_to_tracks)."""
        ev = []  # type: List[dict]
        for d in persons:
            ev.extend(self.update(int(d["gid"]), d["keypoints"], t, d.get("bearing_deg")))
        self.prune(t)
        return ev

    def prune(self, t: float) -> None:
        for gid in [g for g, p in self._p.items() if t - p.last_t > self.cfg.max_age_s]:
            del self._p[gid]

    def active(self, gid: int) -> List[str]:
        p = self._p.get(gid)
        return [g for g in GESTURES if p and p.g[g].active]

    def scores(self, gid: int) -> Dict[str, dict]:
        """Raw per-gesture ``{score, side, ...}`` over the current buffer (no hysteresis)."""
        p = self._p.get(gid)
        out = {g: {"score": 0.0} for g in GESTURES}
        if p is None or not p.buf:
            return out
        for side in ("left", "right"):
            for g, fn in (("wave", self._wave), ("stop_palm", self._stop), ("point", self._point)):
                r = fn(p, side)
                if r is not None and r["score"] > out[g]["score"]:
                    out[g] = r
        return out

    # -- per-gesture scores ------------------------------------------------
    def _frames(self, p: _Person, side: str, window_s: float):
        t_end = p.buf[-1][0]
        fr = []
        for t, kp in p.buf:
            if t < t_end - window_s - 1e-6:
                continue
            s = body_scale(kp, self.cfg.kp_min_conf)
            f = arm_features(kp, side, self.cfg.kp_min_conf) if s else None
            fr.append((t, f, s))
        return fr

    def _held(self, p: _Person, side: str, hold_s: float, pred, max_motion: float):
        c = self.cfg
        fr = self._frames(p, side, hold_s)
        valid = [(t, f, s) for t, f, s in fr if f is not None]
        if len(valid) < c.min_frames or valid[-1][0] - valid[0][0] < 0.9 * hold_s:
            return None
        ok = [(t, f, s) for t, f, s in valid if pred(f)]
        frac = len(ok) / float(len(fr))
        if frac < c.hold_frac:
            return None
        s_med = float(np.median([s for _, _, s in ok]))
        wr = np.array([f["wr"] for _, f, _ in ok])
        motion = float(np.sqrt(wr.var(axis=0).sum())) / s_med
        if motion > max_motion:
            return None
        conf = float(np.mean([f["conf"] for _, f, _ in ok]))
        still = 1.0 - 0.5 * motion / max_motion
        return ok, frac * still * min(1.0, conf / 0.6)

    def _stop(self, p: _Person, side: str) -> Optional[dict]:
        c = self.cfg
        r = self._held(p, side, c.stop_hold_s,
                       lambda f: f["above"] and f["elev_deg"] >= c.stop_min_elev_deg and f["ext"] >= c.min_ext,
                       c.stop_max_motion)
        if r is None:
            return None
        return {"score": r[1], "side": side}

    def _point(self, p: _Person, side: str) -> Optional[dict]:
        c = self.cfg
        r = self._held(p, side, c.point_hold_s,
                       lambda f: abs(f["elev_deg"]) <= c.point_max_elev_deg and f["ext"] >= c.min_ext
                       and abs(f["out"]) >= c.point_min_out,
                       c.point_max_motion)
        if r is None:
            return None
        ok, score = r
        # must point away from the body: same horizontal direction as shoulder->elbow and outward of shoulders
        dirs = [math.copysign(1.0, f["out"]) for _, f, _ in ok]
        if abs(sum(dirs)) < 0.8 * len(dirs):
            return None
        d_img = float(np.degrees(np.arctan2(np.mean([math.sin(math.radians(f["dir_img_deg"])) for _, f, _ in ok]),
                                            np.mean([math.cos(math.radians(f["dir_img_deg"])) for _, f, _ in ok]))))
        img_sign = 1.0 if sum(dirs) > 0 else -1.0     # +1 = toward image right
        out = {"score": score, "side": side, "dir_img_deg": round(d_img, 1),
               "img_dir": "right" if img_sign > 0 else "left"}
        if p.bearing_deg is not None:
            # image right == robot right == negative bearing (ROS: +y left)
            b = p.bearing_deg - img_sign * c.point_offset_deg
            out["bearing"] = round((b + 180.0) % 360.0 - 180.0, 1)
        return out

    def _wave(self, p: _Person, side: str) -> Optional[dict]:
        c = self.cfg
        fr = self._frames(p, side, c.wave_window_s)
        valid = [(t, f, s) for t, f, s in fr if f is not None]
        if len(valid) < c.min_frames:
            return None
        above = [(t, f, s) for t, f, s in valid if f["above"]]
        frac = len(above) / float(len(fr))
        if frac < c.wave_above_frac:
            return None
        s_med = float(np.median([s for _, _, s in above]))
        lat = [(f["wr"][0] - f["sh"][0]) / s_med for _, f, _ in above]
        cycles = count_half_cycles(lat, c.wave_amp) / 2.0
        if cycles < c.wave_min_cycles:
            return None
        conf = float(np.mean([f["conf"] for _, f, _ in above]))
        score = min(1.0, 0.7 + 0.15 * (cycles - c.wave_min_cycles)) * frac * min(1.0, conf / 0.6)
        return {"score": score, "side": side, "cycles": cycles}

    # -- hysteresis ---------------------------------------------------------
    def _hysteresis(self, gid: int, p: _Person, scores: Dict[str, dict], t: float) -> List[dict]:
        c = self.cfg
        ev = []
        for g in GESTURES:
            st, r = p.g[g], scores[g]
            sc = float(r.get("score", 0.0))
            if not st.active:
                st.hits = st.hits + 1 if sc >= c.on_thr else 0
                if st.hits >= c.confirm_n and t - st.off_t >= c.cooldown_s:
                    st.active, st.misses, st.last_emit_t = True, 0, t
                    e = {"gid": int(gid), "gesture": g, "conf": round(min(1.0, sc), 3), "t": t,
                         "side": r.get("side")}
                    for k in ("bearing", "dir_img_deg", "img_dir"):
                        if k in r:
                            e[k] = r[k]
                    ev.append(e)
            else:
                st.misses = st.misses + 1 if sc < c.off_thr else 0
                if st.misses >= c.release_n:
                    st.active, st.hits, st.off_t = False, 0, t
        return ev


# ------------------------------------------------------------------ pose detector
PoseDet = Tuple[Tuple[float, float, float, float], float, np.ndarray]  # bbox, conf, (17, 3)


def _to_np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class PoseDetector:
    """YOLO-pose (ultralytics, lazy). ``detect(images) -> list[list[(bbox, conf, kp(17,3))]]``.

    Export for the Jetson (manual): ``YOLO("yolo11n-pose.pt").export(format="engine", half=True,
    imgsz=(384, 640), batch=4, device=0)``. Static-batch engines get black-frame padding.
    """

    def __init__(self, model_path: Optional[str] = None, imgsz=(384, 640), half: bool = True,
                 conf: float = 0.35, device: Optional[str] = None, engine_batch: Optional[int] = None):
        self.model_path = model_path or os.environ.get("POSE_MODEL", "yolo11n-pose.pt")
        self.imgsz = list(imgsz) if isinstance(imgsz, (tuple, list)) else int(imgsz)
        self.half = bool(half)
        self.conf = float(conf)
        self.device = device
        self.engine_batch = engine_batch
        self._model = None

    @property
    def is_engine(self) -> bool:
        return self.model_path.endswith(".engine")

    def load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy
            if self.is_engine and not os.path.exists(self.model_path):
                raise FileNotFoundError("TensorRT engine not found: %s" % self.model_path)
            self._model = YOLO(self.model_path, task="pose")
        return self._model

    def _kwargs(self) -> dict:
        kw = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False, "classes": [0]}
        if self.device:
            kw["device"] = self.device
        if self.half and not self.is_engine:
            kw["half"] = True
        return kw

    def detect(self, images: Sequence[np.ndarray]) -> List[List[PoseDet]]:
        images = list(images)
        if not images:
            return []
        model = self.load()
        batch = images
        if self.engine_batch and len(batch) < self.engine_batch:
            batch = batch + [np.zeros_like(images[0])] * (self.engine_batch - len(batch))
        results = model.predict(batch, **self._kwargs())
        return [self._parse(r) for r in list(results)[:len(images)]]

    @staticmethod
    def _parse(res) -> List[PoseDet]:
        boxes, kps = getattr(res, "boxes", None), getattr(res, "keypoints", None)
        if boxes is None or kps is None:
            return []
        xyxy, confs = _to_np(boxes.xyxy).reshape(-1, 4), _to_np(boxes.conf).reshape(-1)
        if len(xyxy) == 0:
            return []
        data = getattr(kps, "data", None)
        if data is not None:
            k = _to_np(data)
        else:
            xy = _to_np(kps.xy)
            kc = getattr(kps, "conf", None)
            c = _to_np(kc) if kc is not None else np.ones(xy.shape[:2])
            k = np.concatenate([xy, c[..., None]], axis=-1)
        out = []  # type: List[PoseDet]
        for b, c, kp in zip(xyxy, confs, k):
            out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c),
                        np.asarray(kp, dtype=np.float32).reshape(-1, 3)[:17]))
        return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _bbox_gid(d):
    if isinstance(d, dict):
        return tuple(d["bbox"]), d["gid"], d.get("bearing_deg")
    return tuple(d[0]), d[1], (d[2] if len(d) > 2 else None)


def match_pose_to_tracks(pose_dets: Sequence[PoseDet], person_dets: Sequence, iou_min: float = 0.3) -> List[dict]:
    """Greedy max-IoU 1:1 matching (same camera image).

    ``person_dets``: ``[{bbox, gid, bearing_deg?}]`` or ``[(bbox, gid[, bearing_deg])]`` — the
    tracked person boxes of this camera with their global ids.
    Returns ``[{gid, keypoints, conf, bbox, iou, bearing_deg}]`` for ``GestureClassifier.update_many``.
    """
    persons = [_bbox_gid(d) for d in person_dets]
    pairs = []
    for i, pd in enumerate(pose_dets):
        for j, (bb, _g, _b) in enumerate(persons):
            v = iou(pd[0], bb)
            if v >= iou_min:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_i, used_j, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        bb, gid, brg = persons[j]
        out.append({"gid": gid, "keypoints": pose_dets[i][2], "conf": pose_dets[i][1],
                    "bbox": pose_dets[i][0], "iou": round(v, 3), "bearing_deg": brg})
    return out
