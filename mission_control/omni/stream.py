"""Image encoding helpers for the omni HTTP/WS endpoints (cv2 only)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

THERMAL_RANGE_C = (15.0, 40.0)   # fixed colormap range (degC) when not auto
DEPTH_RANGE_M = (0.2, 2.5)       # MaixSense A010 useful range


def downscale(img: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    """Resize to `max_width` (keeping aspect) if the image is wider."""
    import cv2

    if not max_width or img.shape[1] <= max_width:
        return img
    h = max(1, int(round(img.shape[0] * float(max_width) / img.shape[1])))
    return cv2.resize(img, (int(max_width), h), interpolation=cv2.INTER_AREA)


def encode_jpeg(img: np.ndarray, quality: int = 80, max_width: Optional[int] = None) -> bytes:
    import cv2

    img = downscale(img, max_width)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def encode_png(img: np.ndarray, max_width: Optional[int] = None) -> bytes:
    import cv2

    img = downscale(img, max_width)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def _normalize(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-6)
    return (np.clip((a - lo) / span, 0.0, 1.0) * 255.0).astype(np.uint8)


def thermal_range(temp_c: np.ndarray, auto: bool = False) -> Tuple[float, float]:
    if not auto:
        return THERMAL_RANGE_C
    finite = temp_c[np.isfinite(temp_c)]
    if finite.size == 0:
        return THERMAL_RANGE_C
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi - lo < 2.0:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 1.0, mid + 1.0
    return float(lo), float(hi)


def colorize_thermal(temp_c: np.ndarray, t_min: Optional[float] = None, t_max: Optional[float] = None,
                     auto: bool = False) -> np.ndarray:
    """float32 degC (H,W) -> BGR uint8 with INFERNO. Fixed range by default."""
    import cv2

    t = np.nan_to_num(np.asarray(temp_c, dtype=np.float32), nan=0.0)
    lo, hi = thermal_range(t, auto=auto)
    if t_min is not None:
        lo = float(t_min)
    if t_max is not None:
        hi = float(t_max)
    return cv2.applyColorMap(_normalize(t, lo, hi), cv2.COLORMAP_INFERNO)


def colorize_depth(depth_m: np.ndarray, d_min: float = DEPTH_RANGE_M[0], d_max: float = DEPTH_RANGE_M[1]) -> np.ndarray:
    """float32 metres (H,W), 0 = invalid -> BGR uint8 TURBO (near = red), invalid black."""
    import cv2

    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0)
    n = 255 - _normalize(d, d_min, d_max)
    img = cv2.applyColorMap(n, cv2.COLORMAP_TURBO)
    img[d <= 0] = 0
    return img


def frame_to_bgr(frame, auto: bool = False) -> np.ndarray:
    """Any omni Frame -> displayable BGR uint8 image."""
    if frame.modality == "thermal":
        return colorize_thermal(frame.image, auto=auto)
    if frame.modality == "depth":
        return colorize_depth(frame.image)
    img = frame.image
    if img.ndim == 2:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def frame_to_jpeg(frame, quality: int = 75, max_width: Optional[int] = None, auto: bool = False) -> bytes:
    return encode_jpeg(frame_to_bgr(frame, auto=auto), quality=quality, max_width=max_width)


def ws_video_message(cam_idx: int, jpeg: bytes) -> bytes:
    """/ws/video binary message: 1 byte cam index + JPEG bytes."""
    return bytes([int(cam_idx) & 0xFF]) + jpeg
