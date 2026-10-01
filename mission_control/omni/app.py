"""omni pillar -- OmniVision 360 service (FastAPI, :9114).

Pipeline (one background thread, OMNI_RATE_HZ):
  CaptureManager.latest_all() + LiDAR points (core robot_client or mock)
  -> Perception360.process -> persons (mc.omni.persons)
  -> every 2nd tick: colorize + VoxelMap.integrate (world frame)
  -> mc.omni.health once a second.
Perception / voxel modules are optional: a missing or broken module only
disables that stage (persons=[] / no map), the service still starts.

Run: python3 app.py   (mock: OMNI_MOCK=1 python3 app.py)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MC_ROOT = os.path.dirname(HERE)
for _p in (MC_ROOT, os.path.join(MC_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from omni import stream  # noqa: E402
from omni.capture import CaptureManager  # noqa: E402
from omni.rig import default_rig_path, load_rig  # noqa: E402

PILLAR = "omni"
PORT = int(os.environ.get("OMNI_PORT", "9114"))
MOCK = os.environ.get("OMNI_MOCK", "0").lower() in ("1", "true", "yes")
RIG_PATH = os.environ.get("OMNI_RIG") or default_rig_path(MOCK)
if not os.path.isabs(RIG_PATH):
    RIG_PATH = os.path.join(HERE, RIG_PATH)
RATE_HZ = float(os.environ.get("OMNI_RATE_HZ", "10"))
STREAM_FPS = float(os.environ.get("OMNI_STREAM_FPS", "10"))
STREAM_WIDTH = int(os.environ.get("OMNI_STREAM_WIDTH", "640"))
STREAM_QUALITY = int(os.environ.get("OMNI_STREAM_QUALITY", "70"))
DETECTOR = os.environ.get("OMNI_DETECTOR", "mock" if MOCK else "ultralytics").lower()  # ultralytics|mock|none
MODEL_PATH = os.environ.get("OMNI_MODEL", "yolo11n.pt")
RECTIFY = os.environ.get("OMNI_RECTIFY", "1").lower() in ("1", "true", "yes")
VOXEL_RES = float(os.environ.get("OMNI_VOXEL_RES", "0.05"))
VOXEL_EVERY = max(1, int(os.environ.get("OMNI_VOXEL_EVERY", "2")))
FRAME_MAX_AGE_S = float(os.environ.get("OMNI_FRAME_MAX_AGE_S", "1.0"))
LOGS_DIR = os.environ.get("OMNI_LOG_DIR", os.path.join(HERE, "logs"))
LOG_PATH = os.path.join(LOGS_DIR, "events.jsonl")
try:
    os.makedirs(LOGS_DIR, exist_ok=True)
except OSError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("omni")


def log_event(level: str, msg: str, **extra) -> None:
    record = {"t": time.time(), "pillar": PILLAR, "level": level, "msg": msg}
    record.update(extra)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


class _Bus(object):
    """Optional Redis publisher; never raises."""

    def __init__(self) -> None:
        self._r = None
        host = os.environ.get("REDIS_HOST")
        if not host:
            return
        try:
            import redis

            self._r = redis.Redis(host=host, port=int(os.environ.get("REDIS_PORT", "6379")),
                                  socket_timeout=0.2, socket_connect_timeout=0.5)
            self._r.ping()
            logger.info("redis connected at %s", host)
        except Exception as exc:
            logger.warning("redis unavailable (%s) -- publishing disabled", exc)
            log_event("warn", "redis unavailable", error=str(exc))
            self._r = None

    @property
    def connected(self) -> bool:
        return self._r is not None

    def publish(self, channel: str, payload: dict) -> None:
        if self._r is None:
            return
        try:
            self._r.publish(channel, json.dumps(payload, default=float))
        except Exception as exc:
            logger.debug("redis publish %s failed: %s", channel, exc)


def _person_dict(p) -> dict:
    if isinstance(p, dict):
        return p
    try:
        return p.to_dict()
    except Exception:
        return dict(getattr(p, "__dict__", {}))


def ego_delta(prev, cur):
    """(dx, dy, dyaw) of the robot between two world poses (x, y, yaw),
    expressed in the PREVIOUS base frame."""
    dxw, dyw = cur[0] - prev[0], cur[1] - prev[1]
    c, s = math.cos(-prev[2]), math.sin(-prev[2])
    dyaw = math.atan2(math.sin(cur[2] - prev[2]), math.cos(cur[2] - prev[2]))
    return (c * dxw - s * dyw, s * dxw + c * dyw, dyaw)


def base_to_world(pts: np.ndarray, pose) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = np.empty_like(pts, dtype=np.float32)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + pose[0]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + pose[1]
    out[:, 2] = pts[:, 2]
    return out


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OmniService(object):
    def __init__(self) -> None:
        self.rig = load_rig(RIG_PATH, force_mock=MOCK)
        self.cap = CaptureManager(self.rig)
        self.bus = _Bus()
        self.lock = threading.Lock()
        self.vm_lock = threading.Lock()
        self.persons = []  # type: List[dict]
        self.persons_t = 0.0
        self.tick = 0
        self.timing = {}  # type: Dict[str, float]
        self.rate = 0.0
        self.errors = {}  # type: Dict[str, str]
        self._stop = threading.Event()
        self._thread = None  # type: Optional[threading.Thread]
        self._robot = None
        self._robot_failed = False
        self._prev_pose = None
        self._last_health = 0.0
        self._jpeg_cache = {}  # type: Dict[str, tuple]
        self._jpeg_lock = threading.Lock()
        self.perception = self._make_perception()
        self.voxels = self._make_voxels()
        self._colorize = self._import_colorize()

    # -- optional modules --------------------------------------------------
    def _make_perception(self):
        try:
            from omni.perception360 import Perception360
        except Exception as exc:
            self.errors["perception"] = "import: %s" % exc
            logger.warning("perception360 unavailable: %s", exc)
            return None
        detector = None
        try:
            from omni import detect

            if DETECTOR == "mock":
                detector = detect.MockDetector()
            elif DETECTOR == "ultralytics":
                detector = detect.UltralyticsDetector(MODEL_PATH)
        except Exception as exc:
            self.errors["detector"] = str(exc)
            logger.warning("detector %s unavailable: %s", DETECTOR, exc)
        thermal = None
        try:
            from omni.thermal_detect import ThermalPersonDetector

            thermal = ThermalPersonDetector()
        except Exception as exc:
            self.errors["thermal_detector"] = str(exc)
        rect = {}
        if RECTIFY:
            try:
                from omni.rectify import Rectifier

                for spec in self.rig.by_modality("rgb"):
                    if spec.model.kind == "fisheye":
                        rect[spec.cam_id] = Rectifier(spec)
            except Exception as exc:
                self.errors["rectify"] = str(exc)
        try:
            return Perception360(self.rig, detector, thermal_detector=thermal, rectifiers=rect)
        except Exception as exc:
            self.errors["perception"] = "init: %s" % exc
            logger.warning("perception360 init failed: %s", exc)
            return None

    def _make_voxels(self):
        try:
            from omni.voxel_map import VoxelMap

            return VoxelMap(res=VOXEL_RES)
        except Exception as exc:
            self.errors["voxel_map"] = str(exc)
            logger.warning("voxel_map unavailable: %s", exc)
            return None

    def _import_colorize(self):
        try:
            from omni.colorize import colorize

            return colorize
        except Exception as exc:
            self.errors["colorize"] = str(exc)
            return None

    # -- robot -----------------------------------------------------------
    def _robot_client(self):
        if MOCK or self._robot_failed:
            return None
        if self._robot is None:
            try:
                from robot_client import get_robot_client

                self._robot = get_robot_client()
            except Exception as exc:
                self._robot_failed = True
                self.errors["robot_client"] = str(exc)
                logger.warning("robot_client unavailable: %s", exc)
        return self._robot

    def lidar_points(self, t: float) -> Optional[np.ndarray]:
        typ = str((self.rig.lidar.get("source") or {}).get("type", "mock")).lower()
        if typ == "mock":
            return self.cap.scene.lidar_points(t)
        if typ != "robot_client":
            return None
        robot = self._robot_client()
        if robot is None:
            return None
        try:
            pts = robot.get_lidar_points()
        except Exception as exc:
            self.errors["lidar"] = str(exc)
            return None
        if pts is None or len(pts) == 0:
            return None
        return np.asarray(pts, dtype=np.float32).reshape(-1, 3)

    def pose(self):
        robot = self._robot_client()
        if robot is None:
            return (0.0, 0.0, 0.0)
        try:
            p = robot.get_pose()
            return (float(p.x), float(p.y), float(p.yaw))
        except Exception:
            return self._prev_pose or (0.0, 0.0, 0.0)

    # -- pipeline --------------------------------------------------------
    def start(self) -> None:
        self.cap.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="omni-pipeline", daemon=True)
        self._thread.start()
        log_event("info", "omni started", rig=RIG_PATH, mock=MOCK, cams=self.rig.ordered_ids,
                  perception=self.perception is not None, voxels=self.voxels is not None)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        self.cap.stop()

    def _run(self) -> None:
        period = 1.0 / max(RATE_HZ, 0.1)
        last = time.time()
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.step(t0)
            except Exception as exc:
                logger.exception("pipeline tick failed")
                self.errors["pipeline"] = str(exc)
            dt = time.time() - last
            last = time.time()
            self.rate = 0.9 * self.rate + 0.1 * (1.0 / max(dt, 1e-3)) if self.rate else 1.0 / max(dt, 1e-3)
            rest = period - (time.time() - t0)
            if rest > 0:
                self._stop.wait(rest)

    def _time(self, key: str, t0: float) -> None:
        ms = (time.time() - t0) * 1000.0
        self.timing[key] = round(0.8 * self.timing.get(key, ms) + 0.2 * ms, 2)

    def step(self, t: Optional[float] = None) -> List[dict]:
        """One pipeline tick (also callable directly from tests)."""
        t = time.time() if t is None else t
        self.tick += 1
        frames = {k: f for k, f in self.cap.latest_all().items() if t - f.t <= FRAME_MAX_AGE_S}
        t0 = time.time()
        lidar = self.lidar_points(t)
        self._time("lidar_ms", t0)
        pose = self.pose()
        ed = ego_delta(self._prev_pose, pose) if self._prev_pose is not None else None
        self._prev_pose = pose

        persons = []  # type: list
        if self.perception is not None and frames:
            t0 = time.time()
            try:
                persons = self.perception.process(frames, lidar, t, ego_delta=ed) or []
                self.errors.pop("perception_run", None)
            except Exception as exc:
                self.errors["perception_run"] = str(exc)
                logger.warning("perception failed: %s", exc)
            self._time("perception_ms", t0)
        pd = [_person_dict(p) for p in persons]
        with self.lock:
            self.persons = pd
            self.persons_t = t
        self.bus.publish("mc.omni.persons", {"t": t, "persons": pd})

        if (self.voxels is not None and self._colorize is not None and lidar is not None
                and len(lidar) and self.tick % VOXEL_EVERY == 0):
            t0 = time.time()
            try:
                pts, rgb, temp = self._colorize(lidar, frames, self.rig)
                if len(pts):
                    near = None
                    if pd:
                        pp = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in pd], np.float32)
                        d2 = ((pts[:, None, :2] - pp[None, :, :]) ** 2).sum(-1)
                        near = d2.min(1) < 0.6 ** 2
                    world = base_to_world(np.asarray(pts, np.float32), pose)
                    with self.vm_lock:
                        self.voxels.integrate(world, rgb, temp, t=t, near_person_mask=near,
                                              origin=np.array([pose[0], pose[1], 0.0], np.float32))
                self.errors.pop("voxel_run", None)
            except Exception as exc:
                self.errors["voxel_run"] = str(exc)
                logger.warning("colorize/voxel failed: %s", exc)
            self._time("voxel_ms", t0)

        if t - self._last_health >= 1.0:
            self._last_health = t
            self.bus.publish("mc.omni.health", self.health())
        return pd

    # -- views -----------------------------------------------------------
    def health(self) -> dict:
        st = self.cap.stats()
        avail = [c for c, s in st.items() if s["available"]]
        return {
            "t": time.time(),
            "ok": True,
            "pillar": PILLAR,
            "mock": MOCK,
            "rig": os.path.basename(RIG_PATH),
            "cams_total": len(st),
            "cams_available": len(avail),
            "cams_down": sorted(set(st) - set(avail)),
            "rate_hz": round(self.rate, 2),
            "persons": len(self.persons),
            "modules": {
                "perception": self.perception is not None,
                "voxel_map": self.voxels is not None,
                "colorize": self._colorize is not None,
                "redis": self.bus.connected,
            },
        }

    def stats(self) -> dict:
        out = {"t": time.time(), "tick": self.tick, "rate_hz": round(self.rate, 2), "timing_ms": self.timing,
               "cameras": self.cap.stats(), "errors": self.errors}
        if self.perception is not None:
            out["perception"] = getattr(self.perception, "last_stats", {})
        if self.voxels is not None:
            try:
                with self.vm_lock:
                    out["voxels"] = self.voxels.stats()
            except Exception as exc:
                out["voxels"] = {"error": str(exc)}
        return out

    def stream_jpeg(self, cam_id: str, width: int = STREAM_WIDTH, quality: int = STREAM_QUALITY) -> Optional[tuple]:
        """(seq, jpeg) of the latest frame, encoded once per (cam, seq, width)."""
        fr = self.cap.latest(cam_id)
        if fr is None:
            return None
        key = "%s/%d/%d" % (cam_id, width, quality)
        with self._jpeg_lock:
            hit = self._jpeg_cache.get(key)
            if hit is not None and hit[0] == fr.seq:
                return hit
        jpg = stream.frame_to_jpeg(fr, quality=quality, max_width=width)
        val = (fr.seq, jpg)
        with self._jpeg_lock:
            self._jpeg_cache[key] = val
        return val


svc = OmniService()

app = FastAPI(title="mission-control: omni (OmniVision 360)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    svc.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    svc.stop()
    log_event("info", "omni stopped")


def _spec(cam_id: str):
    spec = svc.rig.cameras.get(cam_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="unknown camera %s" % cam_id)
    return spec


def _latest(cam_id: str):
    fr = svc.cap.latest(cam_id)
    if fr is None:
        raise HTTPException(status_code=503, detail="no frame from %s yet" % cam_id)
    return fr


_NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/health")
def health():
    return svc.health()


@app.get("/rig")
def rig():
    return svc.rig.to_dict()


@app.get("/stats")
def stats():
    return svc.stats()


@app.get("/persons")
def persons():
    with svc.lock:
        return {"t": svc.persons_t, "persons": list(svc.persons)}


@app.get("/cameras/{cam_id}/frame.jpg")
def camera_frame(cam_id: str, width: Optional[int] = None, quality: int = 80, rectified: bool = False):
    spec = _spec(cam_id)
    fr = _latest(cam_id)
    img = stream.frame_to_bgr(fr)
    if rectified:
        from omni.rectify import Rectifier

        rect = None
        if svc.perception is not None:
            rect = getattr(svc.perception, "rectifiers", {}).get(cam_id)
        img = (rect or Rectifier(spec)).apply(img)
    return Response(stream.encode_jpeg(img, quality=quality, max_width=width), media_type="image/jpeg",
                    headers=_NO_CACHE)


@app.get("/thermal/{cam_id}.png")
def thermal_png(cam_id: str, t_min: Optional[float] = None, t_max: Optional[float] = None, auto: bool = False):
    spec = _spec(cam_id)
    if spec.modality != "thermal":
        raise HTTPException(status_code=400, detail="%s is not a thermal camera" % cam_id)
    fr = _latest(cam_id)
    img = stream.colorize_thermal(fr.image, t_min=t_min, t_max=t_max, auto=auto)
    return Response(stream.encode_png(img), media_type="image/png", headers=_NO_CACHE)


@app.get("/depth/{cam_id}.png")
def depth_png(cam_id: str, d_min: float = stream.DEPTH_RANGE_M[0], d_max: float = stream.DEPTH_RANGE_M[1]):
    spec = _spec(cam_id)
    if spec.modality != "depth":
        raise HTTPException(status_code=400, detail="%s is not a depth camera" % cam_id)
    fr = _latest(cam_id)
    return Response(stream.encode_png(stream.colorize_depth(fr.image, d_min, d_max)), media_type="image/png",
                    headers=_NO_CACHE)


@app.get("/map/export.ply")
def export_ply():
    if svc.voxels is None:
        raise HTTPException(status_code=503, detail="voxel map module not available")
    fd, path = tempfile.mkstemp(prefix="omni_map_", suffix=".ply")
    os.close(fd)
    with svc.vm_lock:
        n = svc.voxels.export_ply(path)
    log_event("info", "map exported", voxels=n)
    return FileResponse(path, media_type="application/octet-stream", filename="omni_map.ply")


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    """Binary: 1 byte cam_idx (/rig order) + JPEG. Query: cams=a,b fps= width="""
    await ws.accept()
    qp = ws.query_params
    ids = svc.rig.ordered_ids
    want = [c for c in (qp.get("cams") or "").split(",") if c in ids] or ids
    fps = float(qp.get("fps") or STREAM_FPS)
    width = int(qp.get("width") or STREAM_WIDTH)
    sent = {}  # type: Dict[str, int]
    try:
        while True:
            t0 = time.time()
            for cid in want:
                res = await run_in_threadpool(svc.stream_jpeg, cid, width)
                if res is None or sent.get(cid) == res[0]:
                    continue
                sent[cid] = res[0]
                await ws.send_bytes(stream.ws_video_message(ids.index(cid), res[1]))
            await asyncio.sleep(max(0.0, 1.0 / max(fps, 0.5) - (time.time() - t0)))
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/persons")
async def ws_persons(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with svc.lock:
                msg = {"t": svc.persons_t, "persons": list(svc.persons)}
            await ws.send_text(json.dumps(msg, default=float))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/voxels")
async def ws_voxels(ws: WebSocket):
    """OVX1 binary: first a full snapshot, then deltas every 0.5 s."""
    await ws.accept()
    try:
        if svc.voxels is None:
            while True:  # keep the socket open; the UI just sees no map
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
        vm = svc.voxels

        def _snap():
            with svc.vm_lock:
                return vm.version, vm.snapshot()

        def _delta(v):
            with svc.vm_lock:
                if vm.version == v:
                    return v, None
                return vm.version, vm.delta_since(v)

        version, data = await run_in_threadpool(_snap)
        await ws.send_bytes(data)
        while True:
            await asyncio.sleep(0.5)
            version, data = await run_in_threadpool(_delta, version)
            if data is not None:
                await ws.send_bytes(data)
    except (WebSocketDisconnect, RuntimeError):
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
