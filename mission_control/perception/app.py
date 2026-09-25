"""mission-control: perception pillar (port 9112).

Person detection (YOLOv8) + tracking (ByteTrack) + 3D localisation from the
Intel RealSense D435i aligned depth, plus a person follower that computes
the velocity command it WOULD send (`target_follower.py`), with operator
modes on top (`follow_modes.py`). READ-ONLY: this pillar never commands the
robot. `dry_run` is true unless PERCEPTION_ALLOW_LIVE=1, and even then it only
marks the command as allowed for a separate executor.

Endpoints:
    GET  /                    -- debug page (annotated stream + live JSON)
    GET  /status              -- source, fps, inference time, errors
    GET  /persons             -- latest result (JSON)
    GET  /persons/stream      -- same, as Server-Sent Events
    GET  /frame.jpg           -- latest annotated frame
    GET  /stream.mjpg         -- annotated MJPEG stream
    GET  /follow              -- mode, settings, state, command (flat) + follower detail
    POST /follow              -- {mode?, target_distance_m?, audio_alert?, dry_run?}
    POST /follow/lock         -- {"track_id": int}: lock ONE person to follow
    POST /follow/release      -- drop the lock, mode -> off
    POST /follow/gesture      -- {"gesture": "wave"|"stop"|"ok"}, trick mode only
    POST /follow/ego          -- {dx, dy, dyaw}: robot motion from odometry
    POST /target              -- legacy alias: {"track_id": int} = lock, null = release

Redis (optional, best effort -- the pillar runs fine without it):
    mc.perception.persons     -- every processed frame
    mc.core.proximity_alert   -- nearest person closer than PROXIMITY_ALERT_M
    mc.perception.alert       -- intruder locked / target lost (follow modes)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
import urllib.request
from collections import deque
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

import appearance
import camera_recovery
from geometry3d import Extrinsics
from model_manager import ModelManager, ModelSwitchError
from person_tracker import PersonTracker, annotate
import system_info
from rgbd_source import RGBDFrame, make_source
from follow_modes import FollowSupervisor, ModeError
from target_follower import IDLE, FollowConfig, TargetFollower

PILLAR = "perception"
PORT = int(os.environ.get("PERCEPTION_PORT", "9112"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOGS_DIR, "events.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.4"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE") or None   # "0" = first CUDA GPU
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
# Default model at start: FP16 TensorRT engine (built on first start, cached
# in the models volume). A choice made with POST /model overrides it.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n")
YOLO_FORMAT = os.environ.get("YOLO_FORMAT", "engine")
PROXIMITY_ALERT_M = float(os.environ.get("PROXIMITY_ALERT_M", "0.8"))
PROXIMITY_ALERT_COOLDOWN_S = 2.0
API_TOKEN = os.environ.get("MC_API_TOKEN", "")
ALLOW_LIVE = os.environ.get("PERCEPTION_ALLOW_LIVE", "0") == "1"
# Optional: GET/POST this URL on an alert when audio_alert is on, e.g.
# http://127.0.0.1:8000/audio/play/{kind}  ({kind} = intruder | target_lost)
AUDIO_ALERT_URL = os.environ.get("AUDIO_ALERT_URL", "")

EXTRINSICS = Extrinsics(
    tx=float(os.environ.get("CAM_TX", "0.30")),
    ty=float(os.environ.get("CAM_TY", "0.0")),
    tz=float(os.environ.get("CAM_TZ", "0.10")),
    pitch_deg=float(os.environ.get("CAM_PITCH_DEG", "0.0")),
    yaw_deg=float(os.environ.get("CAM_YAW_DEG", "0.0")),
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("perception")
app = FastAPI(title="mission-control: perception")
# The operator UIs (go2-console, web_dashboard) are served from other ports,
# so fetch()/EventSource need CORS. Comma-separated origins, "*" = any.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-MC-Token"],
)


def log_event(level: str, msg: str, **extra) -> None:
    record = {"t": time.time(), "pillar": PILLAR, "level": level, "msg": msg, **extra}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def require_token(x_mc_token: str = Header(default="")) -> None:
    # /target changes no robot state (perception only), so an unconfigured
    # token is allowed here -- unlike core's arm/move, which fail closed.
    if API_TOKEN and not secrets.compare_digest(x_mc_token, API_TOKEN):
        raise HTTPException(status_code=401, detail="invalid token")


# ---------------------------------------------------------------------------
# Redis (optional)
# ---------------------------------------------------------------------------

class _Bus:
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
            self._r = None

    def publish(self, channel: str, payload: dict) -> None:
        if self._r is None:
            return
        try:
            self._r.publish(channel, json.dumps(payload))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Processing loop (one background thread)
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.result: Optional[dict] = None
        self.frame: Optional[RGBDFrame] = None
        self.seq = 0
        self.fps = 0.0
        self.source_name = os.environ.get("RS_SOURCE", "mock")
        self.last_error: Optional[str] = None
        self.tracker: Optional[PersonTracker] = None
        self.follow_cfg = FollowConfig.from_env()
        self.follower = TargetFollower(self.follow_cfg)
        self.supervisor = FollowSupervisor(self.follower, allow_live=ALLOW_LIVE, on_alert=self._on_alert)
        self.follow_lock = threading.Lock()   # loop thread vs HTTP handlers
        self.bus = _Bus()
        self._last_alert = 0.0
        self._stop = threading.Event()
        self.latency = deque(maxlen=50)      # capture -> result, seconds
        self.done_at = deque(maxlen=30)      # result times, for the real loop rate
        self.source: Optional[object] = None
        self.camera = camera_recovery.CameraSupervisor()
        self.loop_beat = time.time()

    def _deadman(self) -> None:
        """If the loop thread hangs (a blocked librealsense call), the API would
        stay "Up" with stale data. Exit so Docker restarts a clean process."""
        limit = float(os.environ.get("PERCEPTION_DEADMAN_S", "120"))
        while not self._stop.wait(5.0):
            if time.time() - self.loop_beat > limit:
                logger.error("perception loop stuck for %.0f s, exiting for restart", time.time() - self.loop_beat)
                log_event("error", "loop stuck, exiting for restart")
                os._exit(4)

    def _apply_model(self, model, spec: dict) -> None:
        half = spec["format"] == "pt" and bool(YOLO_DEVICE)
        self.tracker.set_model(model, spec["imgsz"], half=half)
        log_event("info", "model switched", **spec)

    def latency_stats(self) -> dict:
        v = sorted(self.latency)
        if not v:
            return {"p50_s": None, "p90_s": None, "window": 0}
        return {"p50_s": round(v[len(v) // 2], 3), "p90_s": round(v[int(len(v) * 0.9)], 3), "window": len(v)}

    def rate_hz(self) -> float:
        d = self.done_at
        return round((len(d) - 1) / (d[-1] - d[0]), 1) if len(d) > 2 and d[-1] > d[0] else 0.0

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="perception-loop").start()

    def _run(self) -> None:
        self.models = ModelManager(YOLO_DEVICE, apply=self._apply_model)
        wanted = self.models.saved_choice({"id": YOLO_MODEL, "format": YOLO_FORMAT, "imgsz": YOLO_IMGSZ})
        try:
            model, loaded = self.models.load_now(wanted)
        except Exception as exc:
            logger.exception("model load failed, falling back to %s", YOLO_WEIGHTS)
            self._fail(f"model load failed: {exc}")
            loaded = {"id": YOLO_WEIGHTS.replace(".pt", ""), "format": "pt", "imgsz": YOLO_IMGSZ}
            model = None
        half = loaded["format"] == "pt" and bool(YOLO_DEVICE)
        self.tracker = PersonTracker(YOLO_WEIGHTS, YOLO_CONF, EXTRINSICS, device=YOLO_DEVICE,
                                     imgsz=loaded["imgsz"], model=model, half=half)
        self.models.current = dict(loaded, half=half or loaded["format"] == "engine")
        if loaded != wanted:   # engine not cached yet: build it while the .pt model runs
            try:
                self.models.request(wanted["id"], wanted["format"], wanted["imgsz"])
            except ModelSwitchError as exc:
                logger.warning("engine auto-build not started: %s", exc)
        source = None
        misses = 0
        self.loop_beat = time.time()
        threading.Thread(target=self._deadman, daemon=True, name="perception-deadman").start()
        while not self._stop.is_set():
            self.loop_beat = time.time()
            if source is None:
                # Not on the USB bus at all: nothing to open and no restart helps.
                # Watch sysfs (cheap) and open the moment the camera is back.
                absent = camera_recovery.usb_bus_readable() and camera_recovery.find_realsense() is None
                self.camera.set_absent(absent)
                if absent:
                    self._fail("camera is not on the USB bus (hung firmware or unplugged) - replug the USB cable")
                    self._stop.wait(2.0)
                    continue
                action = self.camera.before_open()
                if action != "retry":
                    log_event("warn", "camera recovery step", action=action,
                              failures=self.camera.failures, blind=self.camera.blind)
                if action == "exit":
                    # librealsense can be blind inside a long-running process. A
                    # clean process (Docker restarts the container) sees the camera.
                    logger.error("camera failed %d times in a row, exiting for a clean restart",
                                 self.camera.failures)
                    log_event("error", "camera not usable, exiting for restart", failures=self.camera.failures)
                    os._exit(3)
                try:
                    source = make_source(hw_reset=self.camera.want_hw_reset)
                    self.source = source
                    self.source_name = source.name
                    self.last_error = None
                    log_event("info", "rgbd source opened", source=source.name, action=action)
                except Exception as exc:
                    self._fail(f"source open failed: {exc}")
                    blind = "No device connected" in str(exc) and camera_recovery.find_realsense() is not None
                    self.camera.note_failure(blind=blind)
                    self._stop.wait(camera_recovery.backoff_s(self.camera.failures))
                    continue
            frame = source.read(timeout_s=2.0)
            if frame is None:
                misses += 1
                self._fail("no frame within 2 s")
                if misses >= 3:
                    # The camera sometimes stalls (first start after a robot
                    # boot, USB hiccup). Reopen it; each further failure in a
                    # row climbs the recovery ladder in camera_recovery.
                    log_event("warn", "rgbd source stalled, reopening", source=source.name)
                    try:
                        source.close()
                    except Exception:
                        pass
                    source, misses = None, 0
                    self.camera.note_failure()
                continue
            misses = 0
            self.camera.note_frame()
            t0 = time.time()
            try:
                result = self.tracker.process(frame)
            except Exception as exc:
                self._fail(f"tracker error: {exc}")
                logger.exception("tracker error")
                continue
            dt = time.time() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(dt, 1e-3)) if self.fps else 1.0 / max(dt, 1e-3)
            result["source"] = source.name
            features = {p["track_id"]: appearance.extract(frame.color_bgr, p["bbox"])
                        for p in result["persons"] if p["depth_ok"]}
            with self.follow_lock:
                prev_state = self.follower.state
                self.supervisor.before_update(result["persons"], frame.t)
                follow = self.follower.update(result["persons"], features, frame.t, now=time.time())
                follow = self.supervisor.after_update(follow, frame.t)
            if follow["state"] != prev_state:
                log_event("info", "follow state change", frm=prev_state, to=follow["state"],
                          reason=follow["reason"], track_id=follow["track_id"])
            result["follow"] = follow
            result["target_id"] = follow["track_id"] if follow["state"] in ("ACQUIRING", "TRACKING") else None
            result["target_mode"] = "none" if follow["state"] == IDLE else "locked"
            self.last_error = None
            done = time.time()
            self.latency.append(done - frame.t)
            self.done_at.append(done)
            with self.cond:
                self.seq += 1
                result["seq"] = self.seq
                self.result, self.frame = result, frame
                self.cond.notify_all()
            self.bus.publish("mc.perception.persons", result)
            self._maybe_alert(result)

    def _fail(self, msg: str) -> None:
        if msg != self.last_error:
            logger.warning(msg)
            log_event("warn", msg)
        self.last_error = msg

    def _on_alert(self, alert: dict) -> None:
        # Called from the loop thread under follow_lock: keep it non-blocking.
        log_event("warn" if alert["kind"] == "intruder" else "info", "follow alert", **alert)
        self.bus.publish("mc.perception.alert", alert)
        if alert["audio"] and AUDIO_ALERT_URL:
            url = AUDIO_ALERT_URL.format(kind=alert["kind"])
            threading.Thread(target=_fire_and_forget, args=(url,), daemon=True).start()

    def _maybe_alert(self, result: dict) -> None:
        near = [p for p in result["persons"] if p["depth_ok"] and p["distance_m"] < PROXIMITY_ALERT_M]
        if not near or time.time() - self._last_alert < PROXIMITY_ALERT_COOLDOWN_S:
            return
        self._last_alert = time.time()
        p = min(near, key=lambda q: q["distance_m"])
        payload = {"source": PILLAR, "track_id": p["track_id"], "distance_m": p["distance_m"],
                   "bearing_deg": p["bearing_deg"], "t": result["t"]}
        self.bus.publish("mc.core.proximity_alert", payload)
        log_event("info", "proximity alert", **payload)

    def snapshot(self) -> tuple[Optional[dict], Optional[RGBDFrame], int]:
        with self.cond:
            return self.result, self.frame, self.seq

    def wait_next(self, seq: int, timeout: float = 2.0) -> int:
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.seq


def _fire_and_forget(url: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=b"", method="POST"), timeout=3).read()
    except Exception as exc:
        logger.warning("audio alert %s failed: %s", url, exc)


pipeline = Pipeline()


@app.on_event("startup")
def _startup() -> None:
    pipeline.start()
    log_event("info", "perception started", weights=YOLO_WEIGHTS, device=YOLO_DEVICE, source=pipeline.source_name)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    result, _, seq = pipeline.snapshot()
    return {
        "pillar": PILLAR,
        "source": pipeline.source_name,
        "model_loaded": pipeline.tracker is not None,
        "weights": YOLO_WEIGHTS,
        "model": pipeline.models.current if hasattr(pipeline, "models") else None,
        "device": YOLO_DEVICE or "auto",
        "seq": seq,
        "loop_fps": round(pipeline.fps, 1),
        "rate_hz": pipeline.rate_hz(),
        "latency": pipeline.latency_stats(),
        "align_ms": round(getattr(pipeline.source, "align_ms", 0.0), 1),
        "infer_ms": result["infer_ms"] if result else None,
        "result_age_s": round(time.time() - result["t"], 2) if result else None,
        "last_error": pipeline.last_error,
        "camera": pipeline.camera.status(),
        "follow_state": result["follow"]["state"] if result else None,
        "extrinsics": {"tx": EXTRINSICS.tx, "ty": EXTRINSICS.ty, "tz": EXTRINSICS.tz,
                       "pitch_deg": EXTRINSICS.pitch_deg, "yaw_deg": EXTRINSICS.yaw_deg},
    }


class ModelChoice(BaseModel):
    id: str
    format: str = "engine"
    imgsz: int = 640


@app.get("/models")
def models():
    if not hasattr(pipeline, "models"):
        raise HTTPException(503, "model manager not started yet")
    return pipeline.models.summary()


@app.post("/model", status_code=202, dependencies=[Depends(require_token)])
def switch_model(choice: ModelChoice):
    if not hasattr(pipeline, "models"):
        raise HTTPException(503, "model manager not started yet")
    try:
        return {"switch": pipeline.models.request(choice.id, choice.format, choice.imgsz)}
    except ModelSwitchError as exc:
        raise HTTPException(exc.status, str(exc))


@app.get("/system/power")
def system_power():
    return system_info.power()


@app.get("/persons")
def persons():
    result, _, _ = pipeline.snapshot()
    if result is None:
        raise HTTPException(status_code=503, detail=pipeline.last_error or "no frame processed yet")
    return JSONResponse({**result, "age_s": round(time.time() - result["t"], 3)})


async def _in_thread(fn, *args):
    # asyncio.to_thread is 3.9+; the Jetson (JetPack 5) image ships Python 3.8.
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


@app.get("/persons/stream")
async def persons_stream():
    async def gen():
        seq = -1
        while True:
            seq = await _in_thread(pipeline.wait_next, seq)
            result, _, _ = pipeline.snapshot()
            if result is not None:
                yield f"data: {json.dumps(result)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def _annotated_jpeg() -> Optional[bytes]:
    result, frame, _ = pipeline.snapshot()
    if result is None or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", annotate(frame, result), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else None


@app.get("/frame.jpg")
def frame_jpg():
    jpeg = _annotated_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(jpeg, media_type="image/jpeg")


@app.get("/stream.mjpg")
async def stream_mjpg():
    async def gen():
        seq = -1
        while True:
            seq = await _in_thread(pipeline.wait_next, seq)
            jpeg = await _in_thread(_annotated_jpeg)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


DEPTH_MAX_MM = 6000


def _depth_jpeg() -> Optional[bytes]:
    """Depth as a JET heat map (near = red, far = blue, no data = black)."""
    _, frame, _ = pipeline.snapshot()
    if frame is None:
        return None
    d = frame.depth_mm
    norm = np.clip(255 - d.astype(np.float32) * (255.0 / DEPTH_MAX_MM), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat[d == 0] = 0
    ok, buf = cv2.imencode(".jpg", heat, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else None


@app.get("/depth.jpg")
def depth_jpg():
    jpeg = _depth_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(jpeg, media_type="image/jpeg")


@app.get("/depth.mjpg")
async def depth_mjpg():
    async def gen():
        seq = -1
        while True:
            seq = await _in_thread(pipeline.wait_next, seq)
            jpeg = await _in_thread(_depth_jpeg)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/pointcloud.json")
def pointcloud_json(step: int = 8, max_m: float = 6.0):
    """Coloured point cloud in the camera optical frame (x right, y down, z
    forward, metres). `step` = pixel stride (8 -> ~4800 points)."""
    _, frame, _ = pipeline.snapshot()
    if frame is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    step = max(2, min(step, 32))
    i = frame.intrinsics
    d = frame.depth_mm[::step, ::step].astype(np.float32) / 1000.0
    v, u = np.mgrid[0:frame.depth_mm.shape[0]:step, 0:frame.depth_mm.shape[1]:step]
    ok = (d > 0.1) & (d < max_m)
    z = d[ok]
    x = (u[ok] - i.cx) * z / i.fx
    y = (v[ok] - i.cy) * z / i.fy
    rgb = frame.color_bgr[::step, ::step][ok][:, ::-1]
    pts = np.column_stack([x, y, z, rgb]).round(3)
    return {"t": frame.t, "frame": "camera_optical", "fields": ["x", "y", "z", "r", "g", "b"],
            "points": pts.tolist()}


class TargetCmd(BaseModel):
    track_id: Optional[int] = None


def _lock(track_id: int) -> dict:
    result, _, _ = pipeline.snapshot()
    if result is None:
        raise HTTPException(status_code=503, detail="no frame processed yet")
    try:
        with pipeline.follow_lock:
            pipeline.supervisor.lock(track_id, result["persons"], result["t"])
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    log_event("info", "follow lock", track_id=track_id)
    return {"state": pipeline.follower.state, "track_id": track_id, "mode": pipeline.supervisor.mode}


def _release() -> dict:
    with pipeline.follow_lock:
        pipeline.supervisor.release(time.time())
    log_event("info", "follow release")
    return {"state": pipeline.follower.state, "track_id": None, "mode": "off"}


def _follow_payload() -> dict:
    result, _, _ = pipeline.snapshot()
    follow = result.get("follow") if result else None
    with pipeline.follow_lock:
        flat = pipeline.supervisor.summary(follow)
    return {
        **flat,
        "follow": follow,
        "age_s": round(time.time() - result["t"], 3) if result else None,
        "config": asdict(pipeline.follow_cfg),
    }


class FollowSettings(BaseModel):
    mode: Optional[str] = None
    target_distance_m: Optional[float] = None
    audio_alert: Optional[bool] = None
    dry_run: Optional[bool] = None


class GestureCmd(BaseModel):
    gesture: str


@app.get("/follow")
def follow_state():
    return _follow_payload()


@app.post("/follow", dependencies=[Depends(require_token)])
def follow_settings(cmd: FollowSettings):
    sup = pipeline.supervisor
    try:
        with pipeline.follow_lock:
            # validate everything before changing anything
            if cmd.dry_run is False and not sup.allow_live:
                sup.set_dry_run(False)                       # raises 403
            if cmd.target_distance_m is not None:
                sup.set_distance(cmd.target_distance_m)
            if cmd.mode is not None:
                sup.set_mode(cmd.mode, time.time())
            if cmd.audio_alert is not None:
                sup.audio_alert = cmd.audio_alert
            if cmd.dry_run is not None:
                sup.set_dry_run(cmd.dry_run)
    except ModeError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    log_event("info", "follow settings", **cmd.dict(exclude_none=True))
    return _follow_payload()


class EgoCmd(BaseModel):
    dx: float
    dy: float
    dyaw: float


@app.post("/follow/ego", dependencies=[Depends(require_token)])
def follow_ego(cmd: EgoCmd):
    """Robot motion since the last call, in the previous base frame (from
    odometry). Sent by the follow executor so the gate stays on the person
    while the robot turns or walks."""
    with pipeline.follow_lock:
        pipeline.follower.apply_ego_motion(cmd.dx, cmd.dy, cmd.dyaw)
        if pipeline.tracker is not None:
            pipeline.tracker.smoother.apply_ego_motion(cmd.dx, cmd.dy, cmd.dyaw)
    return {"ok": True}


@app.post("/follow/gesture", dependencies=[Depends(require_token)])
def follow_gesture(cmd: GestureCmd):
    try:
        with pipeline.follow_lock:
            g = pipeline.supervisor.gesture(cmd.gesture, time.time())
    except ModeError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    log_event("info", "follow gesture", **g)
    return {"ok": True, **g}


@app.post("/follow/lock", dependencies=[Depends(require_token)])
def follow_lock(cmd: TargetCmd):
    if cmd.track_id is None:
        raise HTTPException(status_code=422, detail="track_id required (use /follow/release to unlock)")
    return _lock(cmd.track_id)


@app.post("/follow/release", dependencies=[Depends(require_token)])
def follow_release():
    return _release()


@app.post("/target", dependencies=[Depends(require_token)])
def set_target(cmd: TargetCmd):
    """Legacy alias kept for early UI code."""
    return _release() if cmd.track_id is None else _lock(cmd.track_id)


INDEX_HTML = """<!doctype html><html lang="hu"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perception</title>
<style>
body{margin:0;background:#101218;color:#dde;font:14px system-ui,sans-serif}
main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
img{max-width:100%;border:1px solid #333;border-radius:6px}
pre{flex:1;min-width:280px;margin:0;background:#181b24;padding:12px;border-radius:6px;overflow:auto;max-height:80vh}
button{background:#2a2f3d;color:#dde;border:1px solid #444;border-radius:4px;padding:4px 10px;margin:2px;cursor:pointer}
</style>
<main><div><img src="stream.mjpg" alt="annotated stream"><div id="btns"></div></div><pre id="out">...</pre></main>
<script>
const out=document.getElementById('out'),btns=document.getElementById('btns');
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.ok||r.json().then(j=>alert(j.detail)))}
function lock(id){post('follow/lock',{track_id:id})}
new EventSource('persons/stream').onmessage=e=>{
 const r=JSON.parse(e.data);out.textContent=JSON.stringify(r,null,1);
 const f=r.follow||{};
 btns.innerHTML=`<b>${f.state||''}</b> ${f.reason||''}<br><button onclick="post('follow/release')">elengedes</button>`+
  r.persons.filter(p=>p.depth_ok).map(p=>`<button onclick="lock(${p.track_id})">kovetes #${p.track_id}${p.track_id===r.target_id?' *':''}</button>`).join('');
};
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
