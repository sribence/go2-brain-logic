"""RGB person detectors for the omni pillar (Agent B, see CONTRACTS.md section 4/B).

API: ``Detector.detect(images: list[np.ndarray]) -> list[list[(bbox, conf, cls)]]``
with bbox = (x1, y1, x2, y2) in the pixels of the image that was passed in.

Only numpy + opencv are imported at module load; ultralytics/torch are lazy.

TensorRT engine notes (run manually on the Jetson, never at import):
    from ultralytics import YOLO
    # FP16, batch 4 (one call for the 4 RGB cams), 384x640 input:
    YOLO("yolo11n.pt").export(format="engine", half=True, imgsz=(384, 640), batch=4, device=0)
    # INT8 needs a calibration dataset yaml (500-1000 own frames):
    YOLO("yolo11n.pt").export(format="engine", int8=True, data="calib.yaml",
                              imgsz=(384, 640), batch=4, device=0)
A static-batch engine must be fed exactly ``batch`` images; UltralyticsDetector pads
the list with black frames when ``engine_batch`` is set.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DetTuple = Tuple[Tuple[float, float, float, float], float, str]


class Detector:
    """Base class: batch detection on a list of BGR images."""

    def detect(self, images: Sequence[np.ndarray]) -> List[List[DetTuple]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UltralyticsDetector(Detector):
    """YOLO detector via ultralytics; supports ``.pt`` and TensorRT ``.engine``.

    The model is loaded lazily on the first ``detect`` call (or ``load()``).
    """

    def __init__(self, model_path: str, imgsz=(384, 640), half: bool = True, conf: float = 0.35,
                 classes: Sequence[str] = ("person",), device: Optional[str] = None,
                 engine_batch: Optional[int] = None):
        self.model_path = model_path
        self.imgsz = list(imgsz) if isinstance(imgsz, (tuple, list)) else int(imgsz)
        self.half = bool(half)
        self.conf = float(conf)
        self.classes = [c.lower() for c in classes]
        self.device = device
        self.engine_batch = engine_batch
        self._model = None
        self._class_ids = None  # type: Optional[List[int]]
        self._names = {}  # type: dict

    @property
    def is_engine(self) -> bool:
        return self.model_path.endswith(".engine")

    def load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO  # lazy

        if not os.path.exists(self.model_path) and self.is_engine:
            raise FileNotFoundError("TensorRT engine not found: %s" % self.model_path)
        model = YOLO(self.model_path, task="detect")
        names = getattr(model, "names", None) or {0: "person"}
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self._names = {int(k): str(v).lower() for k, v in names.items()}
        self._class_ids = [k for k, v in self._names.items() if v in self.classes] or None
        self._model = model
        return model

    def _kwargs(self) -> dict:
        kw = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False}
        if self._class_ids is not None:
            kw["classes"] = self._class_ids
        if self.device:
            kw["device"] = self.device
        if self.half and not self.is_engine:
            kw["half"] = True
        return kw

    def detect(self, images: Sequence[np.ndarray]) -> List[List[DetTuple]]:
        images = list(images)
        if not images:
            return []
        model = self.load()
        batch = images
        if self.engine_batch and len(batch) < self.engine_batch:
            pad = np.zeros_like(images[0])
            batch = batch + [pad] * (self.engine_batch - len(batch))
        results = model.predict(batch, **self._kwargs())
        out = []  # type: List[List[DetTuple]]
        for res in list(results)[:len(images)]:
            out.append(self._parse(res))
        return out

    def _parse(self, res) -> List[DetTuple]:
        dets = []  # type: List[DetTuple]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = _to_np(boxes.xyxy)
        confs = _to_np(boxes.conf)
        clss = _to_np(boxes.cls).astype(int)
        for b, c, k in zip(xyxy, confs, clss):
            name = self._names.get(int(k), str(int(k)))
            if self.classes and name not in self.classes:
                continue
            dets.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c), name))
        return dets


def _to_np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class MockDetector(Detector):
    """Colour-blob 'person' finder for synthetic frames and tests.

    Finds connected regions whose BGR colour is within ``tol`` of ``color_bgr``
    (default pure red, i.e. what a mock source draws as a person), keeping blobs
    with area >= ``min_area`` and height/width >= ``min_aspect``.
    ``inject`` / ``set_detections`` override the output for the next calls.
    """

    def __init__(self, color_bgr=(0, 0, 255), tol: int = 60, min_area: int = 30,
                 min_aspect: float = 0.0, conf: float = 0.9):
        self.color = np.array(color_bgr, dtype=np.int16)
        self.tol = int(tol)
        self.min_area = int(min_area)
        self.min_aspect = float(min_aspect)
        self.conf = float(conf)
        self._injected = None  # type: Optional[List[List[DetTuple]]]
        self.calls = 0
        self.last_batch_size = 0

    def set_detections(self, dets: Optional[List[List[DetTuple]]]) -> None:
        """Inject per-image detections (None restores blob finding)."""
        self._injected = dets

    inject = set_detections

    def detect(self, images: Sequence[np.ndarray]) -> List[List[DetTuple]]:
        images = list(images)
        self.calls += 1
        self.last_batch_size = len(images)
        if self._injected is not None:
            inj = list(self._injected) + [[]] * max(0, len(images) - len(self._injected))
            return [list(d) for d in inj[:len(images)]]
        return [self._blobs(img) for img in images]

    def _blobs(self, img: np.ndarray) -> List[DetTuple]:
        if img is None or img.ndim != 3:
            return []
        lo = np.clip(self.color - self.tol, 0, 255).astype(np.uint8)
        hi = np.clip(self.color + self.tol, 0, 255).astype(np.uint8)
        mask = cv2.inRange(img, lo, hi)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets = []  # type: List[DetTuple]
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if area < self.min_area or w <= 0 or h / float(w) < self.min_aspect:
                continue
            dets.append(((float(x), float(y), float(x + w), float(y + h)), self.conf, "person"))
        return dets
