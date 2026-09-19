"""Lightweight appearance model for target re-identification.

HSV hue/saturation histograms of the upper body (shirt) and the lower body
(trousers). Cheap on the Jetson CPU and good enough to tell "the person we
locked" from "some other person" when clothes differ. Two people in similar
clothes are NOT separable -- the follower's spatial gate is the main guard,
appearance is the second one. Upgrade path: an OSNet ReID embedding with
the same `extract()/similarity()` interface.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

H_BINS, S_BINS = 16, 8
MIN_PIXELS = 150


def _region_hist(hsv: np.ndarray) -> Optional[np.ndarray]:
    import cv2

    # Very dark / washed-out pixels carry no reliable hue.
    mask = cv2.inRange(hsv, (0, 30, 40), (180, 255, 255))
    if cv2.countNonZero(mask) < MIN_PIXELS:
        mask = None
        if hsv.shape[0] * hsv.shape[1] < MIN_PIXELS:
            return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0:
        return None
    return (hist / total).astype(np.float32)


def extract(color_bgr: np.ndarray, bbox: dict) -> Optional[np.ndarray]:
    """Feature = stacked [upper, lower] body histograms, shape (2, H, S).

    Upper = 15-50 % of bbox height, lower = 55-90 %, both on the middle
    60 % of the width (arms and background at the sides are left out).
    """
    import cv2

    h_img, w_img = color_bgr.shape[:2]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return None
    cx1 = int(np.clip(x1 + 0.2 * bw, 0, w_img - 1))
    cx2 = int(np.clip(x2 - 0.2 * bw, cx1 + 1, w_img))
    parts = []
    for top, bottom in ((0.15, 0.50), (0.55, 0.90)):
        ry1 = int(np.clip(y1 + top * bh, 0, h_img - 1))
        ry2 = int(np.clip(y1 + bottom * bh, ry1 + 1, h_img))
        crop = color_bgr[ry1:ry2, cx1:cx2]
        parts.append(_region_hist(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)) if crop.size else None)
    if parts[0] is None:
        return None  # upper body is the minimum (legs are often cut off by the frame)
    if parts[1] is None:
        parts[1] = np.full_like(parts[0], np.nan)
    return np.stack(parts)


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """1.0 = identical colour distribution, 0.0 = disjoint. None if unknown.

    Bhattacharyya coefficient per region; regions missing in either feature
    (NaN, e.g. legs out of frame) are skipped.
    """
    if a is None or b is None:
        return None
    scores = []
    for ra, rb in zip(a, b):
        if np.isnan(ra).any() or np.isnan(rb).any():
            continue
        scores.append(float(np.sum(np.sqrt(ra * rb))))
    return float(np.mean(scores)) if scores else None


def blend(template: np.ndarray, new: np.ndarray, rate: float) -> np.ndarray:
    """Slow template update so lighting changes are followed but a single
    bad frame (occlusion by another person) does not overwrite it."""
    out = template.copy()
    for i in range(template.shape[0]):
        if np.isnan(new[i]).any():
            continue
        if np.isnan(out[i]).any():
            out[i] = new[i]
        else:
            out[i] = (1 - rate) * out[i] + rate * new[i]
    return out
