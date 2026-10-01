"""Shared data types for the omni pillar (see CONTRACTS.md section 3).

Owned by the core module; the other omni modules import these and never
modify them. Python 3.8 compatible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np

MODALITIES = ("rgb", "thermal", "depth")


@dataclass
class Frame:
    cam_id: str
    modality: str  # "rgb" | "thermal" | "depth"
    t: float  # capture unix timestamp
    seq: int
    image: np.ndarray  # rgb: HxWx3 uint8 BGR | thermal: HxW float32 degC | depth: HxW float32 m (0 = invalid)


@dataclass
class Detection:
    cam_id: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    conf: float
    cls: str
    modality: str


@dataclass
class PersonTrack:
    gid: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    conf: float
    cams: List[str] = field(default_factory=list)
    modality: List[str] = field(default_factory=list)
    range_src: str = "ground_plane"  # lidar | tof | ground_plane | thermal_size
    age_s: float = 0.0
    last_seen_t: float = 0.0

    @property
    def range_m(self) -> float:
        return float(np.hypot(self.x, self.y))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("x", "y", "z", "vx", "vy", "conf", "age_s"):
            d[k] = round(float(d[k]), 3)
        d["range_m"] = round(self.range_m, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PersonTrack":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})
