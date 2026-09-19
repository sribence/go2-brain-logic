"""mission-control: perception pillar (port 9112).

Person detection (YOLOv8) + tracking (ByteTrack) + 3D localisation from the
Intel RealSense D435i aligned depth. READ-ONLY: never commands the robot.
The follow behaviour will consume `/persons` or `mc.perception.persons`.

Endpoints:
    GET  /                    -- debug page (annotated stream + live JSON)
    GET  /status              -- source, fps, inference time, errors
    GET  /persons             -- latest result (JSON)
    GET  /persons/stream      -- same, as Server-Sent Events
    GET  /frame.jpg           -- latest annotated frame
    GET  /stream.mjpg         -- annotated MJPEG stream
    POST /target              -- {"track_id": int|null}: lock / release target

Redis (optional, best effort -- the pillar runs fine without it):
    mc.perception.persons     -- every processed frame
    mc.core.proximity_alert   -- nearest person closer than PROXIMITY_ALERT_M
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from typing import Optional

import cv2
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from geometry3d import Extrinsics
from person_tracker import PersonTracker, annotate
from rgbd_source import RGBDFrame, make_source

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
PROXIMITY_ALERT_M = float(os.environ.get("PROXIMITY_ALERT_M", "0.8"))
PROXIMITY_ALERT_COOLDOWN_S = 2.0
API_TOKEN = os.environ.get("MC_API_TOKEN", "")

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
        self.bus = _Bus()
        self._last_alert = 0.0
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="perception-loop").start()

    def _run(self) -> None:
        self.tracker = PersonTracker(YOLO_WEIGHTS, YOLO_CONF, EXTRINSICS, device=YOLO_DEVICE, imgsz=YOLO_IMGSZ)
        source = None
        while not self._stop.is_set():
            if source is None:
                try:
                    source = make_source()
                    self.source_name = source.name
                    self.last_error = None
                    log_event("info", "rgbd source opened", source=source.name)
                except Exception as exc:
                    self._fail(f"source open failed: {exc}")
                    self._stop.wait(3.0)
                    continue
            frame = source.read(timeout_s=2.0)
            if frame is None:
                self._fail("no frame within 2 s")
                continue
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
            self.last_error = None
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
        "device": YOLO_DEVICE or "auto",
        "seq": seq,
        "loop_fps": round(pipeline.fps, 1),
        "infer_ms": result["infer_ms"] if result else None,
        "result_age_s": round(time.time() - result["t"], 2) if result else None,
        "last_error": pipeline.last_error,
        "target_lock": pipeline.tracker.locked_target if pipeline.tracker else None,
        "extrinsics": {"tx": EXTRINSICS.tx, "ty": EXTRINSICS.ty, "tz": EXTRINSICS.tz,
                       "pitch_deg": EXTRINSICS.pitch_deg, "yaw_deg": EXTRINSICS.yaw_deg},
    }


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


class TargetCmd(BaseModel):
    track_id: Optional[int] = None


@app.post("/target", dependencies=[Depends(require_token)])
def set_target(cmd: TargetCmd):
    if pipeline.tracker is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    pipeline.tracker.lock_target(cmd.track_id)
    log_event("info", "target lock changed", track_id=cmd.track_id)
    return {"target_lock": cmd.track_id}


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
function lock(id){fetch('target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({track_id:id})})}
new EventSource('persons/stream').onmessage=e=>{
 const r=JSON.parse(e.data);out.textContent=JSON.stringify(r,null,1);
 btns.innerHTML='<button onclick="lock(null)">auto (legkozelebbi)</button>'+
  r.persons.map(p=>`<button onclick="lock(${p.track_id})">#${p.track_id}${p.track_id===r.target_id?' *':''}</button>`).join('');
};
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
