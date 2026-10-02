"""Frame capture for the omni rig: one thread per camera, latest-frame cache.

Sources (CameraSource: open() / read() -> Frame|None / close()):
  * UvcSource  -- USB UVC camera via cv2.VideoCapture (V4L2, MJPG), or a
                  GStreamer pipeline with the Jetson HW MJPEG decoder.
  * HttpSource -- polls a bridge URL (.npy for thermal/depth, .jpg/.png rgb).
  * MockSource -- synthetic, geometrically consistent scene (MockScene):
                  ground plane + a walking "person" circling the robot.

CaptureManager runs every source in its own daemon thread; a failing camera
is reopened with backoff and never affects the others.

Evidence support (omni/recorder.py):
  * Frames may carry the compressed image: `frame.jpeg` (bytes, full sensor
    resolution), `frame.full_w/full_h`, `frame.mode`. UvcSource reads the raw
    MJPEG buffer without decoding (V4L2 CAP_PROP_CONVERT_RGB=0) and decodes
    `frame.image` lazily, at the rig (calibration) size, only when someone
    (perception, stream) touches it -- see LazyFrame.
  * Resolution modes: source.modes {normal: [w,h,fps], evidence: [w,h,fps] |
    [[w,h,fps], ...] | auto}. `request_mode(name)` / `set_mode(w,h,fps)` are
    applied by the capture thread between two reads (safe reopen, reverts to
    the previous mode when the new one fails). `frame.image` always stays at
    spec.width x spec.height so the calibration keeps working.
"""
from __future__ import annotations

import abc
import io
import logging
import re
import subprocess
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from omni.omni_types import Frame
    from omni.camera_model import transform_points
except ImportError:  # omni/ itself on sys.path
    from omni_types import Frame  # type: ignore
    from camera_model import transform_points  # type: ignore

logger = logging.getLogger("omni.capture")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class CameraSource(abc.ABC):
    """One camera. read() returns the next Frame or None on a failed read."""

    #: True if read() blocks until the device delivers (no extra pacing needed)
    self_paced = False

    def __init__(self, spec):
        self.spec = spec
        self.cam_id = spec.cam_id
        self.modality = spec.modality
        self.fps = float((spec.source or {}).get("fps", 10) or 10)
        self._seq = 0
        self.modes = parse_modes(spec)
        self.mode = "normal"
        self.cur_w, self.cur_h = int(spec.width), int(spec.height)
        self.mode_error = ""
        self._pending = None  # type: Optional[Tuple[str, list]]

    # -- resolution modes -------------------------------------------------------
    def request_mode(self, name: str) -> bool:
        """Ask for a named mode ("normal" / "evidence"); applied before the next read."""
        if name not in self.modes:
            return False
        if name == self.mode and self._pending is None:
            return True
        self._pending = (name, self.modes[name])
        return True

    def set_mode(self, width: int, height: int, fps: float) -> None:
        """Explicit resolution switch; applied by the reading thread before the next read."""
        name = "custom"
        for n, cands in self.modes.items():
            if isinstance(cands, list) and len(cands) == 1 and tuple(cands[0][:2]) == (int(width), int(height)):
                name = n
        self._pending = (name, [(int(width), int(height), float(fps))])

    @property
    def mode_pending(self) -> bool:
        return self._pending is not None

    def _take_pending(self):
        p, self._pending = self._pending, None
        return p

    def _apply_pending(self) -> None:
        """Default: synthetic/HTTP sources just adopt the first candidate size."""
        p = self._take_pending()
        if p is None:
            return
        name, cands = p
        if not isinstance(cands, list) or not cands:
            cands = self.modes.get("normal")
        w, h, fps = cands[0]
        self.cur_w, self.cur_h, self.mode = int(w), int(h), name
        if fps:
            self.fps = float(fps)

    @abc.abstractmethod
    def open(self) -> bool:
        ...

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        ...

    def close(self) -> None:
        pass

    def _frame(self, image: np.ndarray, t: Optional[float] = None) -> Frame:
        self._seq += 1
        return Frame(cam_id=self.cam_id, modality=self.modality,
                     t=time.time() if t is None else t, seq=self._seq, image=image)


def _mode_tuple(v) -> Tuple[int, int, float]:
    v = list(v)
    return (int(v[0]), int(v[1]), float(v[2]) if len(v) > 2 and v[2] else 0.0)


def parse_modes(spec) -> Dict[str, object]:
    """{name: [(w, h, fps), ...] | "auto"} from spec.source["modes"].

    normal defaults to the rig size; evidence defaults to "auto" for UVC
    (largest MJPEG sizes from v4l2-ctl), 1.5x the rig size for mock RGB, and
    the normal mode otherwise. A mode may list several candidates (tried in
    order until the camera accepts one)."""
    src = spec.source or {}
    fps = float(src.get("fps", 10) or 10)
    out = {"normal": [(int(spec.width), int(spec.height), fps)]}  # type: Dict[str, object]
    typ = str(src.get("type", "mock")).lower()
    if typ == "uvc":
        out["evidence"] = "auto"
    elif typ == "mock" and spec.modality == "rgb":
        out["evidence"] = [(int(round(spec.width * 1.5)), int(round(spec.height * 1.5)), max(fps / 2.0, 1.0))]
    else:
        out["evidence"] = list(out["normal"])
    for name, val in (src.get("modes") or {}).items():
        if isinstance(val, str):
            out[str(name)] = val.lower()
            continue
        val = list(val or [])
        if val and isinstance(val[0], (list, tuple)):
            out[str(name)] = [_mode_tuple(v) for v in val]
        elif val:
            out[str(name)] = [_mode_tuple(val)]
    return out


_V4L2_FMT = re.compile(r"\[\d+\]:\s*'(\w+)'")
_V4L2_SIZE = re.compile(r"Size:\s*\w+\s+(\d+)x(\d+)")
_V4L2_FPS = re.compile(r"\(([\d.]+)\s*fps\)")


def parse_v4l2_formats(text: str) -> Dict[str, List[Tuple[int, int, float]]]:
    """`v4l2-ctl --list-formats-ext` output -> {fourcc: [(w, h, max_fps)]}."""
    out = {}  # type: Dict[str, List[Tuple[int, int, float]]]
    fmt = None
    cur = None
    for line in text.splitlines():
        m = _V4L2_FMT.search(line)
        if m:
            fmt = m.group(1)
            out.setdefault(fmt, [])
            cur = None
            continue
        m = _V4L2_SIZE.search(line)
        if m and fmt:
            cur = [int(m.group(1)), int(m.group(2)), 0.0]
            out[fmt].append(cur)  # type: ignore[arg-type]
            continue
        m = _V4L2_FPS.search(line)
        if m and cur is not None:
            cur[2] = max(cur[2], float(m.group(1)))
    return {k: [tuple(v) for v in vs] for k, vs in out.items()}  # type: ignore[misc]


def probe_v4l2_modes(device, fourcc: str = "MJPG", min_fps: float = 2.0, max_n: int = 3,
                     timeout: float = 2.0) -> List[Tuple[int, int, float]]:
    """Largest sizes the device offers in `fourcc` (area desc). [] if v4l2-ctl is missing."""
    dev = "/dev/video%d" % device if isinstance(device, int) else str(device)
    try:
        txt = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=timeout, check=False).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []
    modes = [m for m in parse_v4l2_formats(txt).get(fourcc, []) if m[2] == 0.0 or m[2] >= min_fps]
    modes.sort(key=lambda m: (-m[0] * m[1], -m[2]))
    return [(w, h, min(f, 15.0) if f else 5.0) for w, h, f in modes[:max_n]]


def jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from the SOF marker, without decoding. None if not found."""
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        mk = data[i + 1]
        if mk == 0xFF:
            i += 1
            continue
        if mk in (0xD8, 0x01) or 0xD0 <= mk <= 0xD7:
            i += 2
            continue
        seg = (data[i + 2] << 8) | data[i + 3]
        if mk in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return ((data[i + 7] << 8) | data[i + 8], (data[i + 5] << 8) | data[i + 6])
        if mk == 0xDA:
            return None
        i += 2 + seg
    return None


def decode_jpeg(data: bytes, out_w: int, out_h: int, full_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Decode to (out_h, out_w, 3) BGR, using libjpeg DCT down-scaling (1/2, 1/4, 1/8)
    when the JPEG is much larger than wanted -- far cheaper than a full decode."""
    import cv2

    fw, fh = full_size or jpeg_size(data) or (out_w, out_h)
    ratio = min(fw / float(max(out_w, 1)), fh / float(max(out_h, 1)))
    flag = cv2.IMREAD_COLOR
    if ratio >= 8:
        flag = cv2.IMREAD_REDUCED_COLOR_8
    elif ratio >= 4:
        flag = cv2.IMREAD_REDUCED_COLOR_4
    elif ratio >= 2:
        flag = cv2.IMREAD_REDUCED_COLOR_2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if img is None:
        return None
    if img.shape[1] != out_w or img.shape[0] != out_h:
        img = cv2.resize(img, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
    return img


def attach_jpeg(fr: Frame, jpeg: Optional[bytes], full_w: int, full_h: int, mode: str = "normal") -> Frame:
    """Evidence fields on a Frame (dataclass without slots -> plain attributes)."""
    fr.jpeg = jpeg  # type: ignore[attr-defined]
    fr.full_w = int(full_w)  # type: ignore[attr-defined]
    fr.full_h = int(full_h)  # type: ignore[attr-defined]
    fr.mode = mode  # type: ignore[attr-defined]
    return fr


class LazyFrame(Frame):
    """Frame carrying the camera's compressed JPEG; `image` (rig size, BGR) is
    decoded on first access only (decode-on-demand). isinstance(.., Frame) holds."""

    def __init__(self, cam_id: str, modality: str, t: float, seq: int, jpeg: bytes,
                 full_w: int, full_h: int, out_w: int, out_h: int, mode: str = "normal"):
        self._img = None  # type: Optional[np.ndarray]
        self._out = (int(out_w), int(out_h))
        Frame.__init__(self, cam_id, modality, t, seq, None)  # type: ignore[arg-type]
        attach_jpeg(self, jpeg, full_w, full_h, mode)

    @property  # type: ignore[override]
    def image(self) -> np.ndarray:  # noqa: F811
        img = self._img
        if img is None:
            img = decode_jpeg(self.jpeg, self._out[0], self._out[1], (self.full_w, self.full_h))
            if img is None:
                img = np.zeros((self._out[1], self._out[0], 3), np.uint8)
            self._img = img
        return img

    @image.setter
    def image(self, value) -> None:
        if value is not None:
            self._img = value

    @property
    def decoded(self) -> bool:
        return self._img is not None

    def __repr__(self) -> str:
        return "LazyFrame(%s seq=%d %dx%d %s)" % (self.cam_id, self.seq, self.full_w, self.full_h, self.mode)


def gst_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Jetson pipeline: MJPEG from V4L2 decoded by NVJPG (nvv4l2decoder mjpeg=1)."""
    return ("v4l2src device={dev} io-mode=2 ! image/jpeg,width={w},height={h},framerate={f}/1 ! "
            "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
            ).format(dev=device, w=width, h=height, f=fps)


class UvcSource(CameraSource):
    """UVC camera. With MJPG over V4L2 (default) the compressed buffer is read
    as-is (CAP_PROP_CONVERT_RGB=0: OpenCV's V4L2 backend then returns the raw
    driver buffer as a 1xN uint8 Mat instead of decoding it) and wrapped in a
    LazyFrame. If the backend still returns BGR, or the raw buffer is not a
    decodable JPEG, it falls back to decode + re-encode (source.reencode_quality,
    default 92) so frame.jpeg is always available for the recorder.
    source.passthrough: false disables the raw path."""

    self_paced = True

    def __init__(self, spec):
        super(UvcSource, self).__init__(spec)
        self.cap = None
        self._warned_size = False
        src = spec.source or {}
        self.passthrough = bool(src.get("passthrough", True))
        self.reencode_quality = int(src.get("reencode_quality", 92))
        self.raw = False  # raw MJPEG buffers are being delivered
        self._raw_checked = False
        self.cur_fps = float(src.get("fps", 15) or 15)

    def _device(self):
        dev = self.spec.source.get("device", 0)
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        return dev

    def _open_with(self, w: int, h: int, fps: float) -> bool:
        import cv2

        src = self.spec.source
        dev = self._device()
        self.close()
        self.raw = False
        self._raw_checked = False
        fourcc = str(src.get("fourcc", "MJPG"))[:4]
        if src.get("gst") or src.get("pipeline"):
            pipe = src.get("pipeline") or gst_pipeline(str(dev), w, h, int(fps or 15))
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, int(fps or 15))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self.passthrough and fourcc == "MJPG":
                    try:
                        self.raw = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
                    except Exception:
                        self.raw = False
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.raw = False
            return False
        self.cap = cap
        self.cur_w, self.cur_h, self.cur_fps = int(w), int(h), float(fps or self.cur_fps)
        return True

    def open(self) -> bool:
        cands = self.modes.get(self.mode)
        if not isinstance(cands, list) or not cands:
            cands = self.modes["normal"]
            self.mode = "normal"
        w, h, fps = cands[0]
        return self._open_with(w, h, fps or self.cur_fps)

    def _candidates(self, name: str, cands) -> list:
        if cands == "auto" or cands == "max":
            probed = probe_v4l2_modes(self._device(), str(self.spec.source.get("fourcc", "MJPG"))[:4])
            if probed:
                self.modes[name] = list(probed)  # cache: probe once
            return list(probed) or list(self.modes["normal"])
        return list(cands or self.modes["normal"])

    def _apply_pending(self) -> None:
        """Reopen at the pending mode; revert to the previous one if no candidate works.
        Runs in the reading thread, so it never races with read()."""
        p = self._take_pending()
        if p is None:
            return
        import cv2

        name, cands = p
        prev = (self.cur_w, self.cur_h, self.cur_fps, self.mode)
        cands = self._candidates(name, cands)
        for i, (w, h, fps) in enumerate(cands):
            if not self._open_with(w, h, fps or prev[2]):
                continue
            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if aw in (0, w) and ah in (0, h) or i == len(cands) - 1:
                self.mode = name
                self.mode_error = "" if aw in (0, w) else "asked %dx%d got %dx%d" % (w, h, aw, ah)
                if aw and ah:
                    self.cur_w, self.cur_h = aw, ah
                    if abs(aw / float(ah) - self.spec.width / float(self.spec.height)) > 0.02:
                        logger.warning("%s mode %s %dx%d has a different aspect than the calibration "
                                       "%dx%d -- perception geometry will be off", self.cam_id, name, aw, ah,
                                       self.spec.width, self.spec.height)
                logger.info("%s -> mode %s %dx%d@%s", self.cam_id, name, self.cur_w, self.cur_h, fps)
                return
        self.mode_error = "mode %s failed, staying at %dx%d" % (name, prev[0], prev[1])
        logger.warning("%s: %s", self.cam_id, self.mode_error)
        self._open_with(prev[0], prev[1], prev[2])
        self.mode = prev[3]

    @staticmethod
    def _is_jpeg_buffer(buf) -> bool:
        return (buf is not None and buf.dtype == np.uint8 and (buf.ndim == 1 or (buf.ndim == 2 and buf.shape[0] == 1))
                and buf.size > 4 and int(buf.flat[0]) == 0xFF and int(buf.flat[1]) == 0xD8)

    def read(self) -> Optional[Frame]:
        import cv2

        if self._pending is not None:
            self._apply_pending()
        if self.cap is None:
            return None
        ok, img = self.cap.read()
        t = time.time()
        if not ok or img is None:
            return None
        W, H = self.spec.width, self.spec.height
        if self.raw:
            if self._is_jpeg_buffer(img):
                data = img.tobytes()
                size = jpeg_size(data) or (self.cur_w, self.cur_h)
                if not self._raw_checked:  # one-time sanity check: is it decodable?
                    self._raw_checked = True
                    if decode_jpeg(data, 64, 36, size) is None:
                        logger.warning("%s: raw MJPEG not decodable -- falling back to decode+re-encode", self.cam_id)
                        self.passthrough = False
                        self._pending = (self.mode, [(self.cur_w, self.cur_h, self.cur_fps)])
                        return None
                self._seq += 1
                return LazyFrame(self.cam_id, self.modality, t, self._seq, data, size[0], size[1], W, H, self.mode)
            if img.ndim == 3:
                logger.info("%s: backend ignores CONVERT_RGB=0 -- using decode+re-encode", self.cam_id)
                self.raw = False
            else:
                return None
        full_w, full_h = img.shape[1], img.shape[0]
        jpeg = None
        if self.modality == "rgb":
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.reencode_quality])
            jpeg = buf.tobytes() if ok else None
        if full_w != W or full_h != H:
            if self.mode == "normal" and not self._warned_size:
                logger.warning("%s delivers %dx%d, rig says %dx%d -- resizing (check calibration)",
                               self.cam_id, full_w, full_h, W, H)
                self._warned_size = True
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return attach_jpeg(self._frame(img, t), jpeg, full_w, full_h, self.mode)

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class HttpSource(CameraSource):
    """Polls a bridge endpoint. .npy -> float32 array, anything else -> image."""

    def __init__(self, spec):
        super(HttpSource, self).__init__(spec)
        self.url = spec.source["url"]
        self.timeout = float(spec.source.get("timeout_s", 0.5))

    def open(self) -> bool:
        return True

    def fetch(self) -> bytes:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:  # noqa: S310 (local bridge)
            return resp.read()

    def decode(self, data: bytes) -> np.ndarray:
        if self.url.split("?")[0].endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float32)
            if self.modality == "rgb" and arr.ndim == 3:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        import cv2

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("undecodable image from %s" % self.url)
        return img

    def read(self) -> Optional[Frame]:
        try:
            data = self.fetch()
            t = time.time()
            img = self.decode(data)
        except Exception as exc:
            logger.debug("%s http read failed: %s", self.cam_id, exc)
            return None
        fr = self._frame(img, t)
        if data[:2] == b"\xff\xd8":  # keep the bridge's JPEG as-is for the recorder
            attach_jpeg(fr, data, img.shape[1], img.shape[0], self.mode)
        return fr


# ---------------------------------------------------------------------------
# Synthetic scene (mock mode)
# ---------------------------------------------------------------------------

class MockScene(object):
    """Deterministic robot-relative world: ground at `ground_z`, a person
    (box 0.5 x 0.5 x 1.7 m) walking on a wobbling circle around the robot,
    and walls at `wall_r` for the LiDAR ring. Everything is in base frame."""

    PERSON_W = 0.5
    PERSON_H = 1.7

    def __init__(self, ground_z: float = -0.32, seed: int = 0, n_persons: int = 1,
                 wall_r: float = 6.0):
        self.ground_z = float(ground_z)
        self.seed = int(seed)
        self.n_persons = int(n_persons)
        self.wall_r = float(wall_r)

    def person_positions(self, t: float) -> np.ndarray:
        """(n,3) person centres at ground level (x, y, ground_z)."""
        out = []
        for i in range(self.n_persons):
            ph = i * 2.0 * np.pi / max(self.n_persons, 1) + 0.37 * self.seed
            ang = 0.25 * t + ph
            r = 3.5 + 1.5 * np.sin(0.13 * t + ph)
            out.append([r * np.cos(ang), r * np.sin(ang), self.ground_z])
        return np.asarray(out, dtype=np.float64).reshape(-1, 3)

    def person_box(self, centre: np.ndarray) -> np.ndarray:
        """8 corners of the person box (base frame)."""
        h = self.PERSON_W / 2.0
        xs = [centre[0] - h, centre[0] + h]
        ys = [centre[1] - h, centre[1] + h]
        zs = [self.ground_z, self.ground_z + self.PERSON_H]
        return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)

    def lidar_points(self, t: float, n_ring: int = 720) -> np.ndarray:
        """(N,3) float32 base-frame points: wall ring (3 heights), ground ring,
        and a dense column of points on each person."""
        rng = np.random.RandomState(self.seed + int(t * 10) % 1000)
        a = np.linspace(-np.pi, np.pi, n_ring, endpoint=False)
        pts = []
        for z in (self.ground_z + 0.3, self.ground_z + 1.0, self.ground_z + 1.8):
            pts.append(np.stack([self.wall_r * np.cos(a), self.wall_r * np.sin(a), np.full_like(a, z)], 1))
        for r in (1.5, 2.5, 4.0):
            pts.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, self.ground_z)], 1))
        for c in self.person_positions(t):
            m = 300
            ang = rng.uniform(0, 2 * np.pi, m)
            pz = rng.uniform(self.ground_z, self.ground_z + self.PERSON_H, m)
            r = self.PERSON_W / 2.0
            pts.append(np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), pz], 1))
        p = np.concatenate(pts, 0)
        p += rng.normal(0, 0.01, p.shape)
        return p.astype(np.float32)


class MockSource(CameraSource):
    """Renders the MockScene through the camera's real model + extrinsics.

    rgb: BGR uint8 with sky/checker ground, person in colour, timestamp text;
    thermal: float32 degC, ~20 C background, person 34-36 C;
    depth: float32 m (z in camera frame), 0 outside 0.2..2.5 m.
    Deterministic for a given (seed, t) -- see render(t).
    """

    def __init__(self, spec, scene: Optional[MockScene] = None, ground_z: float = -0.32):
        super(MockSource, self).__init__(spec)
        seed = int(spec.source.get("seed", 0))
        self.scene = scene or MockScene(ground_z=ground_z, seed=0)
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        self._bg = None  # type: Optional[np.ndarray]
        self._next_t = 0.0
        self.emit_jpeg = bool(spec.source.get("jpeg", True))
        self.jpeg_quality = int(spec.source.get("jpeg_quality", 85))

    # static background, computed once from per-pixel rays
    def _build_background(self) -> None:
        spec = self.spec
        W, H = spec.width, spec.height
        # large images: compute on a /f grid and upscale (keeps startup fast)
        f = 4 if W * H > 400000 else 1
        w, h = W // f, H // f
        uu, vv = np.meshgrid((np.arange(w, dtype=np.float64) + 0.5) * f - 0.5,
                             (np.arange(h, dtype=np.float64) + 0.5) * f - 0.5)
        uv = np.stack([uu.ravel(), vv.ravel()], 1)
        rays = spec.model.unproject(uv)
        R = spec.T_base_cam[:3, :3]
        o = spec.T_base_cam[:3, 3]
        d = rays @ R.T
        dz = d[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dz < -1e-6, (self.scene.ground_z - o[2]) / dz, np.inf)
        hit = np.isfinite(s) & (s < 40.0)
        s = np.where(hit, s, 0.0)
        gx = o[0] + s * d[:, 0]
        gy = o[1] + s * d[:, 1]
        z_cam = s * rays[:, 2]
        if self.modality == "rgb":
            tint = np.array([(spec.index * 53) % 255, 120, 200 - (spec.index * 31) % 120], np.float64)
            checker = ((np.floor(np.where(hit, gx, 0)) + np.floor(np.where(hit, gy, 0))) % 2).astype(np.float64)
            ground = 70 + 60 * checker
            img = np.empty((h * w, 3), np.float64)
            img[:, 0] = np.where(hit, ground * 0.6 + tint[0] * 0.2, 200 + 40 * np.clip(dz, 0, 1))
            img[:, 1] = np.where(hit, ground * 0.9, 160 + 30 * np.clip(dz, 0, 1))
            img[:, 2] = np.where(hit, ground * 0.7 + tint[2] * 0.2, 120)
            # fisheye: black outside the lens circle
            ok = spec.model.project(rays)[1] if spec.model.kind == "fisheye" else np.ones(len(rays), bool)
            img[~ok] = 0
            bg = np.clip(img, 0, 255).astype(np.uint8).reshape(h, w, 3)
            if f != 1:
                import cv2

                bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST)
            self._bg = bg
        elif self.modality == "thermal":
            dist = np.where(hit, s, 30.0)
            bg = 20.0 + 1.5 * np.exp(-dist / 3.0) + self.rng.normal(0, 0.2, h * w)
            self._bg = bg.astype(np.float32).reshape(h, w)
        else:
            dep = np.where(hit, z_cam, 0.0)
            dep = np.where((dep >= 0.2) & (dep <= 2.5), dep, 0.0)
            self._bg = dep.astype(np.float32).reshape(h, w)

    def open(self) -> bool:
        if self._bg is None:
            self._build_background()
        return True

    def _person_hulls(self, t: float) -> List[Tuple[np.ndarray, float]]:
        """[(hull int32 (k,1,2), depth_cam)] for persons visible in this camera."""
        out = []
        for c in self.scene.person_positions(t):
            corners = self.scene.person_box(c)
            pc = transform_points(self.spec.T_cam_base, corners)
            uv, valid = self.spec.model.project(pc)
            # require every corner in front of the lens (in-image not needed)
            if self.spec.model.kind == "fisheye":
                th = np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2])
                front = th < self.spec.model.theta_max
            else:
                front = pc[:, 2] > 0.05
            # every corner must be in front of the lens, at least one in the image
            if not np.all(front) or not np.any(valid):
                continue
            import cv2

            hull = cv2.convexHull(np.round(uv).astype(np.int32).reshape(-1, 1, 2))
            out.append((hull, float(np.mean(pc[:, 2]))))
        out.sort(key=lambda x: -x[1])  # far first
        return out

    def render(self, t: float) -> np.ndarray:
        import cv2

        if self._bg is None:
            self._build_background()
        img = self._bg.copy()
        for hull, depth in self._person_hulls(t):
            if self.modality == "rgb":
                cv2.fillConvexPoly(img, hull, (0, 0, 255))  # pure red = MockDetector colour
                cv2.polylines(img, [hull], True, (20, 20, 20), 2)
            elif self.modality == "thermal":
                mask = np.zeros(img.shape, np.uint8)
                cv2.fillConvexPoly(mask, hull, 1)
                temp = 35.0 + 0.8 * np.sin(t)
                img[mask > 0] = temp
            else:
                if 0.2 <= depth <= 2.5:
                    cv2.fillConvexPoly(img, hull, float(depth))
        if self.modality == "rgb":
            txt = "%s %.2f" % (self.cam_id, t)
            cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.modality == "thermal":
            noise = np.random.RandomState((self._seed * 7919 + int(t * 1000)) % (2 ** 31))
            img = img + noise.normal(0, 0.05, img.shape).astype(np.float32)
        return img.astype(np.float32) if self.modality != "rgb" else img

    def read(self) -> Optional[Frame]:
        """rgb frames also carry JPEG bytes at the current mode's size (evidence
        mode: the rig-size render upscaled, like a bigger sensor would deliver)."""
        if self._pending is not None:
            self._apply_pending()
        t = time.time()
        img = self.render(t)
        fr = self._frame(img, t)
        if self.modality == "rgb" and self.emit_jpeg:
            import cv2

            big = img
            if (self.cur_w, self.cur_h) != (img.shape[1], img.shape[0]):
                big = cv2.resize(img, (self.cur_w, self.cur_h), interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            attach_jpeg(fr, buf.tobytes() if ok else None, big.shape[1], big.shape[0], self.mode)
        else:
            attach_jpeg(fr, None, img.shape[1], img.shape[0], self.mode)
        return fr


def make_source(spec, scene: Optional[MockScene] = None, ground_z: float = -0.32) -> CameraSource:
    typ = str((spec.source or {}).get("type", "mock")).lower()
    if typ == "uvc":
        return UvcSource(spec)
    if typ == "http":
        return HttpSource(spec)
    if typ == "mock":
        return MockSource(spec, scene=scene, ground_z=ground_z)
    raise ValueError("camera %s: unknown source type %r" % (spec.cam_id, typ))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _CamState(object):
    def __init__(self, source: CameraSource):
        self.source = source
        self.frame = None  # type: Optional[Frame]
        self.available = False
        self.frames = 0
        self.drops = 0
        self.reopens = 0
        self.fps = 0.0
        self.last_t = 0.0
        self.last_error = ""
        self.thread = None  # type: Optional[threading.Thread]


class CaptureManager(object):
    """One capture thread per camera; latest frame per camera in memory."""

    MAX_CONSEC_FAIL = 15

    def __init__(self, rig, scene: Optional[MockScene] = None, sources: Optional[Dict[str, CameraSource]] = None):
        self.rig = rig
        self.scene = scene or MockScene(ground_z=getattr(rig, "ground_z", -0.32))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cams = {}  # type: Dict[str, _CamState]
        self._listeners = []  # type: list
        for cid, spec in rig.cameras.items():
            try:
                src = (sources or {}).get(cid) or make_source(spec, scene=self.scene, ground_z=self.scene.ground_z)
            except Exception as exc:
                logger.error("camera %s: %s", cid, exc)
                continue
            self._cams[cid] = _CamState(src)

    def start(self) -> "CaptureManager":
        self._stop.clear()
        for cid, st in self._cams.items():
            if st.thread is not None and st.thread.is_alive():
                continue
            st.thread = threading.Thread(target=self._run, args=(cid,), name="omni-cap-" + cid, daemon=True)
            st.thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for st in self._cams.values():
            if st.thread is not None:
                st.thread.join(timeout)
            try:
                st.source.close()
            except Exception:
                pass

    def _run(self, cid: str) -> None:
        st = self._cams[cid]
        src = st.source
        opened = False
        fails = 0
        backoff = 0.5
        while not self._stop.is_set():
            if not opened:
                try:
                    opened = bool(src.open())
                except Exception as exc:
                    st.last_error = "open: %s" % exc
                    opened = False
                if not opened:
                    st.available = False
                    st.last_error = st.last_error or "open failed"
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, 10.0)
                    continue
                backoff = 0.5
                fails = 0
            t0 = time.time()
            period = 1.0 / max(src.fps, 0.1)  # may change with the mode
            try:
                fr = src.read()
            except Exception as exc:
                fr = None
                st.last_error = "read: %s" % exc
            if fr is None:
                st.drops += 1
                fails += 1
                if fails >= 3:
                    st.available = False
                if fails >= self.MAX_CONSEC_FAIL:
                    logger.warning("camera %s: %d failed reads, reopening", cid, fails)
                    try:
                        src.close()
                    except Exception:
                        pass
                    opened = False
                    st.reopens += 1
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, 10.0)
                    continue
                self._stop.wait(min(period, 0.2))
                continue
            fails = 0
            with self._lock:
                if st.last_t > 0:
                    dt = max(fr.t - st.last_t, 1e-3)
                    st.fps = 0.8 * st.fps + 0.2 * (1.0 / dt) if st.fps > 0 else 1.0 / dt
                st.frame = fr
                st.frames += 1
                st.last_t = fr.t
                st.available = True
                st.last_error = ""
            for fn in self._listeners:  # e.g. recorder ring buffer; must be cheap
                try:
                    fn(fr)
                except Exception as exc:
                    logger.debug("frame listener failed: %s", exc)
            if not src.self_paced:
                rest = period - (time.time() - t0)
                if rest > 0:
                    self._stop.wait(rest)

    # -- API -------------------------------------------------------------------
    def latest(self, cam_id: str) -> Optional[Frame]:
        st = self._cams.get(cam_id)
        if st is None:
            return None
        with self._lock:
            return st.frame

    def latest_all(self) -> Dict[str, Frame]:
        with self._lock:
            return {cid: st.frame for cid, st in self._cams.items() if st.frame is not None}

    def wait_ready(self, timeout: float = 5.0, cams: Optional[List[str]] = None) -> bool:
        """Block until every (given) camera has a frame. For tests / startup."""
        want = cams or list(self._cams.keys())
        end = time.time() + timeout
        while time.time() < end:
            have = self.latest_all()
            if all(c in have for c in want):
                return True
            time.sleep(0.02)
        return False

    def stats(self) -> Dict[str, dict]:
        now = time.time()
        out = {}
        with self._lock:
            for cid, st in self._cams.items():
                out[cid] = {
                    "available": bool(st.available),
                    "source": str(st.source.spec.source.get("type", "?")),
                    "fps": round(st.fps, 2),
                    "frames": st.frames,
                    "drops": st.drops,
                    "reopens": st.reopens,
                    "last_t": st.last_t,
                    "age_s": round(now - st.last_t, 3) if st.last_t else None,
                    "last_error": st.last_error,
                    "mode": getattr(st.source, "mode", "normal"),
                    "size": [getattr(st.source, "cur_w", 0), getattr(st.source, "cur_h", 0)],
                }
                if isinstance(st.source, UvcSource):
                    out[cid]["passthrough"] = bool(st.source.raw)
                if getattr(st.source, "mode_error", ""):
                    out[cid]["mode_error"] = st.source.mode_error
        return out

    # -- evidence support ------------------------------------------------------
    def add_listener(self, fn) -> None:
        """fn(frame) is called in the camera's capture thread for every new frame."""
        if fn not in self._listeners:
            self._listeners = self._listeners + [fn]

    def remove_listener(self, fn) -> None:
        self._listeners = [f for f in self._listeners if f is not fn]

    def source(self, cam_id: str) -> Optional[CameraSource]:
        st = self._cams.get(cam_id)
        return st.source if st is not None else None

    def mode(self, cam_id: str) -> Optional[str]:
        src = self.source(cam_id)
        return getattr(src, "mode", None) if src is not None else None

    def set_mode(self, cam_ids, mode: str) -> Dict[str, bool]:
        """Request a named mode for cameras (None = all). Non-blocking: the
        capture thread reopens the device between two reads."""
        ids = list(self._cams) if cam_ids is None else [c for c in cam_ids if c in self._cams]
        return {c: bool(self._cams[c].source.request_mode(mode)) for c in ids}

    def wait_frames(self, cam_id: str, n: int = 1, timeout: float = 3.0, after_seq: int = -1,
                    mode: Optional[str] = None) -> List[Frame]:
        """Collect up to n new frames (seq > after_seq, optionally of `mode`)."""
        out = []  # type: List[Frame]
        last = after_seq
        end = time.time() + timeout
        while time.time() < end and len(out) < n:
            fr = self.latest(cam_id)
            if fr is not None and fr.seq > last and (mode is None or getattr(fr, "mode", "normal") == mode):
                out.append(fr)
                last = fr.seq
                continue
            time.sleep(0.01)
        return out
