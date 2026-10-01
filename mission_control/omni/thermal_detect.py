"""Heat-blob person detector on float32 degC thermal images (Agent B).

Pipeline: temperature band threshold -> morphology (open/close) -> connected
components -> area/aspect/fill filters (upright humans) -> bbox + conf.
conf combines how close the blob's hot core is to skin temperature (~34 degC)
with a shape score (upright aspect ~2.5, reasonable fill ratio).

Works for both the narrow (far) and wide (near) FOV cameras: the minimum
area is given in pixels, optionally scaled by image size relative to 256x192.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

ThermalDet = Tuple[Tuple[float, float, float, float], float]

SKIN_C = 34.0


class ThermalPersonDetector:
    def __init__(self, t_min: float = 28.0, t_max: float = 40.0, min_area_px: int = 24,
                 min_height_px: int = 6, min_aspect: float = 1.3, max_aspect: float = 6.0,
                 min_fill: float = 0.25, open_k: int = 3, close_k: int = 5,
                 ref_size: Tuple[int, int] = (256, 192), scale_area: bool = True,
                 conf_min: float = 0.2, ideal_aspect: float = 2.5):
        """min_aspect/max_aspect bound bbox height/width (upright person ~2-4).

        ``ref_size`` (w, h) is the resolution the pixel thresholds were tuned for;
        with ``scale_area`` they scale with the actual image area.
        """
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.min_area_px = int(min_area_px)
        self.min_height_px = int(min_height_px)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.min_fill = float(min_fill)
        self.open_k = int(open_k)
        self.close_k = int(close_k)
        self.ref_size = ref_size
        self.scale_area = bool(scale_area)
        self.conf_min = float(conf_min)
        self.ideal_aspect = float(ideal_aspect)

    def mask(self, img_c: np.ndarray) -> np.ndarray:
        img = np.nan_to_num(np.asarray(img_c, dtype=np.float32), nan=0.0)
        m = ((img >= self.t_min) & (img <= self.t_max)).astype(np.uint8) * 255
        if self.open_k > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.open_k, self.open_k), np.uint8))
        if self.close_k > 1:
            # Vertical-ish closing joins head/torso/legs split by cooler clothing.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k + 2))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m

    def detect(self, frame_celsius: np.ndarray) -> List[ThermalDet]:
        img = np.asarray(frame_celsius, dtype=np.float32)
        if img.ndim != 2 or img.size == 0:
            return []
        h_img, w_img = img.shape
        s = 1.0
        if self.scale_area and self.ref_size:
            s = (w_img * h_img) / float(self.ref_size[0] * self.ref_size[1])
        min_area = max(4, int(round(self.min_area_px * s)))
        min_h = max(2, int(round(self.min_height_px * math.sqrt(s))))
        m = self.mask(img)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []  # type: List[ThermalDet]
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if area < min_area or h < min_h or w <= 0:
                continue
            aspect = h / float(w)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            fill = area / float(w * h)
            if fill < self.min_fill:
                continue
            blob = img[labels == i]
            blob = blob[(blob >= self.t_min) & (blob <= self.t_max)]
            core = float(np.percentile(blob, 90)) if blob.size else SKIN_C - 6.0
            conf = self._conf(core, aspect, fill)
            if conf < self.conf_min:
                continue
            out.append(((float(x), float(y), float(x + w), float(y + h)), conf))
        out.sort(key=lambda d: -d[1])
        return out

    def _conf(self, core_c: float, aspect: float, fill: float) -> float:
        t_score = math.exp(-0.5 * ((core_c - SKIN_C) / 3.0) ** 2)
        a_score = math.exp(-0.5 * (math.log(aspect / self.ideal_aspect) / 0.6) ** 2)
        f_score = min(1.0, fill / 0.6)
        return float(np.clip(0.5 * t_score + 0.35 * a_score + 0.15 * f_score, 0.0, 1.0))


def estimate_range_from_height(bbox_h_px: float, fy: float, person_height_m: float = 1.7) -> Optional[float]:
    """Pinhole range estimate from the person's apparent height (m), None if invalid."""
    if bbox_h_px is None or bbox_h_px <= 1.0 or fy <= 0:
        return None
    return float(fy * person_height_m / float(bbox_h_px))
