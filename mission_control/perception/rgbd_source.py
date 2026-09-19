"""RGB-D frame sources for the perception pillar.

All sources return `RGBDFrame`: BGR color + uint16 depth in millimetres,
ALIGNED to the color image (same HxW, same intrinsics).

  RS_SOURCE=realsense  -- pyrealsense2 directly (dev PC / x86 container with
                          USB passthrough). No ARM64 wheel on the Jetson.
  RS_SOURCE=rosbridge  -- reads `go2-hardware-bridge/realsense_bridge`
                          (ROS1 Noetic + rosbridge :9091). The Jetson path.
  RS_SOURCE=mock       -- static image + synthetic moving depth plane, so the
                          whole pipeline runs with no camera at all.
"""
from __future__ import annotations

import abc
import base64
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry3d import Intrinsics

logger = logging.getLogger("perception.rgbd_source")


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray        # HxWx3 uint8
    depth_mm: np.ndarray         # HxW uint16, 0 = invalid, aligned to color
    intrinsics: Intrinsics
    t: float                     # unix seconds


class RGBDSource(abc.ABC):
    name: str

    @abc.abstractmethod
    def read(self, timeout_s: float = 1.0) -> Optional[RGBDFrame]:
        """Latest frame, or None if nothing arrived within timeout."""

    def close(self) -> None:
        pass


class RealSenseSource(RGBDSource):
    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color)
        self._depth_scale_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intr = Intrinsics(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy)
        logger.info("realsense started %dx%d@%d, depth_scale=%.4f mm", width, height, fps, self._depth_scale_mm)

    def read(self, timeout_s: float = 1.0) -> Optional[RGBDFrame]:
        try:
            frames = self._pipe.wait_for_frames(int(timeout_s * 1000))
        except RuntimeError:
            return None
        frames = self._align.process(frames)
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            return None
        depth = np.asanyarray(d.get_data())
        if abs(self._depth_scale_mm - 1.0) > 1e-6:  # D435 default is exactly 1 mm
            depth = (depth.astype(np.float32) * self._depth_scale_mm).astype(np.uint16)
        return RGBDFrame(np.asanyarray(c.get_data()).copy(), depth.copy(), self._intr, time.time())

    def close(self) -> None:
        self._pipe.stop()


class RosbridgeSource(RGBDSource):
    """Subscribes to realsense_bridge topics over rosbridge (roslibpy).

    Needs `align_depth:=true` in the bridge launch. Depth comes from the
    image_transport `/compressed` topic: 16-bit lossless PNG, ~3x smaller
    than raw 16UC1 over JSON/base64 (~0.8 MB/frame at 640x480).
    """
    name = "rosbridge"

    COLOR = "/camera/color/image_raw/compressed"
    DEPTH = "/camera/aligned_depth_to_color/image_raw/compressed"
    INFO = "/camera/color/camera_info"

    def __init__(self, host: str, port: int = 9091, throttle_ms: int = 100):
        import roslibpy

        self._cond = threading.Condition()
        self._color: Optional[tuple[float, np.ndarray]] = None
        self._depth: Optional[tuple[float, np.ndarray]] = None
        self._intr: Optional[Intrinsics] = None
        self._last_served = 0.0

        self._ros = roslibpy.Ros(host=host, port=port)
        self._ros.run(timeout=10)
        kw = {"throttle_rate": throttle_ms, "queue_length": 1}
        roslibpy.Topic(self._ros, self.COLOR, "sensor_msgs/CompressedImage", **kw).subscribe(self._on_color)
        roslibpy.Topic(self._ros, self.DEPTH, "sensor_msgs/CompressedImage", **kw).subscribe(self._on_depth)
        roslibpy.Topic(self._ros, self.INFO, "sensor_msgs/CameraInfo", throttle_rate=2000).subscribe(self._on_info)
        logger.info("rosbridge source connected to %s:%d", host, port)

    @staticmethod
    def _stamp(msg: dict) -> float:
        s = msg.get("header", {}).get("stamp", {})
        return s.get("secs", 0) + s.get("nsecs", 0) * 1e-9

    def _on_color(self, msg: dict) -> None:
        import cv2

        img = cv2.imdecode(np.frombuffer(base64.b64decode(msg["data"]), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self._cond:
                self._color = (self._stamp(msg), img)
                self._cond.notify_all()

    def _on_depth(self, msg: dict) -> None:
        import cv2

        # compressed_image_transport: format "16UC1; png compressed", plain PNG.
        # compressedDepth transport prefixes a 12-byte header -- tolerate both.
        data = base64.b64decode(msg["data"])
        depth = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None and len(data) > 12:
            depth = cv2.imdecode(np.frombuffer(data[12:], np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            logger.warning("depth frame not decodable as 16-bit (format=%s)", msg.get("format"))
            return
        with self._cond:
            self._depth = (self._stamp(msg), depth)
            self._cond.notify_all()

    def _on_info(self, msg: dict) -> None:
        k = msg["K"]
        self._intr = Intrinsics(msg["width"], msg["height"], k[0], k[4], k[2], k[5])

    def read(self, timeout_s: float = 1.0) -> Optional[RGBDFrame]:
        deadline = time.time() + timeout_s
        with self._cond:
            while True:
                if self._color and self._depth and self._intr:
                    tc, color = self._color
                    td, depth = self._depth
                    newest = max(tc, td)
                    # Pair only frames within 100 ms and never serve the same pair twice.
                    if abs(tc - td) < 0.1 and newest > self._last_served and color.shape[:2] == depth.shape:
                        self._last_served = newest
                        return RGBDFrame(color, depth, self._intr, time.time())
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def close(self) -> None:
        self._ros.terminate()


class MockSource(RGBDSource):
    """Static photo with people + a depth plane that slowly moves 1.5-4.5 m.

    Default image is ultralytics' bundled `bus.jpg` (4 people). Not a
    tracking-accuracy test -- only proves the pipeline end to end.
    """
    name = "mock"

    def __init__(self, image_path: Optional[str] = None, fps: float = 10.0):
        import cv2

        if not image_path:
            from ultralytics.utils import ASSETS
            image_path = str(ASSETS / "bus.jpg")
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"mock image not readable: {image_path}")
        self._img = img
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(math.radians(69.0) / 2)  # D435 color HFOV ~69 deg
        self._intr = Intrinsics(w, h, f, f, w / 2, h / 2)
        self._period = 1.0 / fps
        self._t0 = time.time()
        self._next = self._t0

    def read(self, timeout_s: float = 1.0) -> Optional[RGBDFrame]:
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self._period, time.time())
        t = time.time()
        z_mm = int(3000 + 1500 * math.sin((t - self._t0) * 0.4))
        depth = np.full(self._img.shape[:2], z_mm, dtype=np.uint16)
        return RGBDFrame(self._img.copy(), depth, self._intr, t)


def make_source() -> RGBDSource:
    kind = os.environ.get("RS_SOURCE", "mock").lower()
    if kind == "realsense":
        return RealSenseSource(
            int(os.environ.get("RS_WIDTH", "640")),
            int(os.environ.get("RS_HEIGHT", "480")),
            int(os.environ.get("RS_FPS", "15")),
        )
    if kind == "rosbridge":
        return RosbridgeSource(
            os.environ.get("REALSENSE_ROSBRIDGE_HOST", "localhost"),
            int(os.environ.get("REALSENSE_ROSBRIDGE_PORT", "9091")),
        )
    if kind == "mock":
        return MockSource(os.environ.get("MOCK_IMAGE") or None)
    raise ValueError(f"unknown RS_SOURCE={kind!r} (realsense|rosbridge|mock)")
