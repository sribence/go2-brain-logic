"""Frame capture for the omni rig: one thread per camera, latest-frame cache.

Sources (CameraSource: open() / read() -> Frame|None / close()):
  * UvcSource  -- USB UVC camera via cv2.VideoCapture (V4L2, MJPG), or a
                  GStreamer pipeline with the Jetson HW MJPEG decoder.
  * HttpSource -- polls a bridge URL (.npy for thermal/depth, .jpg/.png rgb).
  * MockSource -- synthetic, geometrically consistent scene (MockScene):
                  ground plane + a walking "person" circling the robot.

CaptureManager runs every source in its own daemon thread; a failing camera
is reopened with backoff and never affects the others.
"""
from __future__ import annotations

import abc
import io
import logging
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


def gst_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Jetson pipeline: MJPEG from V4L2 decoded by NVJPG (nvv4l2decoder mjpeg=1)."""
    return ("v4l2src device={dev} io-mode=2 ! image/jpeg,width={w},height={h},framerate={f}/1 ! "
            "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
            ).format(dev=device, w=width, h=height, f=fps)


class UvcSource(CameraSource):
    self_paced = True

    def __init__(self, spec):
        super(UvcSource, self).__init__(spec)
        self.cap = None
        self._warned_size = False

    def open(self) -> bool:
        import cv2

        src = self.spec.source
        dev = src.get("device", 0)
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        w, h = self.spec.width, self.spec.height
        fps = int(src.get("fps", 15))
        self.close()
        if src.get("gst") or src.get("pipeline"):
            pipe = src.get("pipeline") or gst_pipeline(str(dev), w, h, fps)
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                fourcc = str(src.get("fourcc", "MJPG"))[:4]
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False
        self.cap = cap
        return True

    def read(self) -> Optional[Frame]:
        import cv2

        if self.cap is None:
            return None
        ok, img = self.cap.read()
        t = time.time()
        if not ok or img is None:
            return None
        if img.shape[1] != self.spec.width or img.shape[0] != self.spec.height:
            if not self._warned_size:
                logger.warning("%s delivers %dx%d, rig says %dx%d -- resizing (check calibration)",
                               self.cam_id, img.shape[1], img.shape[0], self.spec.width, self.spec.height)
                self._warned_size = True
            img = cv2.resize(img, (self.spec.width, self.spec.height))
        return self._frame(img, t)

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
        return self._frame(img, t)


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
        t = time.time()
        return self._frame(self.render(t), t)


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
        period = 1.0 / max(src.fps, 0.1)
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
                }
        return out
