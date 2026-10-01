"""Camera rig description: loads `config/rig.yaml` into CameraSpec objects.

Each camera entry needs modality, model, width, height, K, D, a pose and a
source. The pose is either an explicit 4x4 `T_base_cam` (p_base = T @ p_cam)
or a `mount: {xyz: [x, y, z], yaw_deg, pitch_deg, roll_deg}` block
(pitch_deg > 0 = looking down). `T_base_cam` wins when both are given.

`Rig.ordered_ids` is the stable camera order (YAML order) -- the index into
it is the `cam_idx` byte of the /ws/video stream.
"""
from __future__ import annotations

import copy
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from omni.camera_model import CameraModel, make_model
except ImportError:  # running with omni/ itself on sys.path
    from camera_model import CameraModel, make_model  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RIG = os.path.join(HERE, "config", "rig.yaml")
MOCK_RIG = os.path.join(HERE, "config", "rig_mock.yaml")

# camera frame (x right, y down, z fwd) -> base frame (x fwd, y left, z up)
R_BASE_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def T_from_mount(xyz, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """4x4 T_base_cam for a camera at `xyz` (base frame), looking along base
    yaw `yaw_deg` (CCW, 0 = forward, 90 = left), tilted DOWN by `pitch_deg`
    and rolled about its optical axis by `roll_deg`."""
    R = _rz(np.radians(yaw_deg)) @ _ry(np.radians(pitch_deg)) @ _rx(np.radians(roll_deg)) @ R_BASE_OPTICAL
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


@dataclass
class CameraSpec:
    cam_id: str
    modality: str
    width: int
    height: int
    model: CameraModel
    T_base_cam: np.ndarray
    source: dict = field(default_factory=dict)
    index: int = 0

    def __post_init__(self) -> None:
        T = np.array(self.T_base_cam, dtype=np.float64).reshape(4, 4)
        # YAML values are rounded: snap the rotation back to orthonormal
        U, _, Vt = np.linalg.svd(T[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            raise ValueError("camera %s: T_base_cam rotation is a reflection" % self.cam_id)
        T[:3, :3] = R
        T[3] = [0.0, 0.0, 0.0, 1.0]
        self.T_base_cam = T
        Ti = np.eye(4)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ T[:3, 3]
        self.T_cam_base = Ti

    @property
    def position(self) -> np.ndarray:
        return self.T_base_cam[:3, 3].copy()

    @property
    def yaw_deg(self) -> float:
        """Base-frame heading of the optical axis."""
        z = self.T_base_cam[:3, 2]
        return float(np.degrees(np.arctan2(z[1], z[0])))

    def to_dict(self) -> dict:
        m = self.model.to_dict()
        return {
            "id": self.cam_id,
            "index": self.index,
            "modality": self.modality,
            "width": self.width,
            "height": self.height,
            "model": m["type"],
            "K": m["K"],
            "D": m["D"],
            "max_fov_deg": m.get("max_fov_deg"),
            "T_base_cam": np.round(self.T_base_cam, 6).tolist(),
            "yaw_deg": round(self.yaw_deg, 3),
            "source": {k: v for k, v in self.source.items() if k in ("type", "fps", "url", "device")},
        }


@dataclass
class Rig:
    cameras: Dict[str, CameraSpec]
    lidar: dict = field(default_factory=dict)
    name: str = "omni"
    path: Optional[str] = None
    ground_z: float = -0.32

    @property
    def ordered_ids(self) -> List[str]:
        return list(self.cameras.keys())

    def index_of(self, cam_id: str) -> int:
        return self.ordered_ids.index(cam_id)

    def by_modality(self, modality: str) -> List[CameraSpec]:
        return [c for c in self.cameras.values() if c.modality == modality]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ground_z": self.ground_z,
            "lidar": self.lidar,
            "order": self.ordered_ids,
            "cameras": [c.to_dict() for c in self.cameras.values()],
        }


def _parse_camera(cam_id: str, d: dict, index: int, force_mock: bool) -> CameraSpec:
    modality = str(d.get("modality", "rgb"))
    if modality not in ("rgb", "thermal", "depth"):
        raise ValueError("camera %s: bad modality %r" % (cam_id, modality))
    width = int(d["width"])
    height = int(d["height"])
    kw = {}
    if d.get("max_fov_deg") is not None:
        kw["max_fov_deg"] = float(d["max_fov_deg"])
    model = make_model(d.get("model", "pinhole"), d["K"], d.get("D") or [], width, height, **kw)
    if d.get("T_base_cam") is not None:
        T = np.asarray(d["T_base_cam"], dtype=np.float64)
    elif d.get("mount") is not None:
        m = d["mount"]
        T = T_from_mount(m.get("xyz", [0, 0, 0]), m.get("yaw_deg", 0.0),
                         m.get("pitch_deg", 0.0), m.get("roll_deg", 0.0))
    else:
        raise ValueError("camera %s: needs T_base_cam or mount" % cam_id)
    source = copy.deepcopy(d.get("source") or {"type": "mock"})
    if force_mock:
        source = {"type": "mock", "fps": source.get("fps", 10)}
    return CameraSpec(cam_id=cam_id, modality=modality, width=width, height=height,
                      model=model, T_base_cam=T, source=source, index=index)


def rig_from_dict(data: dict, force_mock: bool = False, path: Optional[str] = None) -> Rig:
    cams = OrderedDict()  # type: OrderedDict
    for i, (cid, cd) in enumerate((data.get("cameras") or {}).items()):
        cams[str(cid)] = _parse_camera(str(cid), cd, i, force_mock)
    lidar = copy.deepcopy(data.get("lidar") or {"source": {"type": "mock"}})
    if force_mock:
        lidar["source"] = {"type": "mock"}
    return Rig(cameras=cams, lidar=lidar, name=str(data.get("name", "omni")), path=path,
               ground_z=float(data.get("ground_z", -0.32)))


def load_rig(path: Optional[str] = None, force_mock: bool = False) -> Rig:
    """Load a rig YAML. force_mock=True replaces every source with `mock`."""
    import yaml

    path = path or DEFAULT_RIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return rig_from_dict(data, force_mock=force_mock, path=path)


def default_rig_path(mock: bool = False) -> str:
    return MOCK_RIG if mock else DEFAULT_RIG


__all__ = ["CameraSpec", "Rig", "load_rig", "rig_from_dict", "T_from_mount", "default_rig_path",
           "DEFAULT_RIG", "MOCK_RIG"]

