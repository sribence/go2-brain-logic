# perception pillar (port 9112)

Detects people with YOLOv8, tracks them with ByteTrack (stable `track_id`),
and computes the 3D position of each person from the aligned depth image of
the Intel RealSense D435i. The pillar runs **on the robot** (Jetson Orin NX,
CUDA) in its own Docker container.

The pillar is **read-only**: it never sends a command to the robot. The
follow behaviour will be a separate step that consumes `/persons`, and it
must go through `core` (armed check, watchdog, E-stop).

## Data flow

```
RealSense D435i ──USB──> realsense_bridge (ROS1 Noetic, :9091 rosbridge)
    /camera/color/image_raw/compressed                  JPEG
    /camera/aligned_depth_to_color/image_raw/compressed 16-bit PNG
    /camera/color/camera_info                           intrinsics
                 │ roslibpy (localhost)
                 ▼
perception (:9112)  YOLOv8 person + ByteTrack ─> torso depth ─> deproject
                    ─> camera->base transform ─> per-track EMA + velocity
```

The depth is aligned to the color image, so each color pixel has the matching
depth pixel. The distance comes from the "torso window" of the bbox (middle
40 % of the width, 20–60 % of the height), using the 30th percentile. This
way the background between the legs and arms does not pull the distance
back. A single-frame jump larger than 1.5 m (for example, the bbox catches a
wall) is rejected by `TrackSmoother`.

## Coordinate frames

- `position_optical`: camera optical frame (x right, y down, z forward).
- `position`: **robot base frame** (x forward, y left, z up), in metres.
  `bearing_deg` > 0 means the person is to the left.

The camera mount (`CAM_TX/TY/TZ/PITCH_DEG/YAW_DEG`) is **not measured yet**.
The defaults (0.30 / 0 / 0.10 m, 0°) are a guess. The distance and bearing are
correct regardless, because they come from the camera. The absolute `z`
(height) needs a measured `CAM_PITCH_DEG`.

## Output (`GET /persons`)

```json
{
  "t": 1789813100.1, "seq": 812, "source": "rosbridge", "infer_ms": 14.2,
  "count": 1, "target_id": 3, "target_mode": "nearest",
  "persons": [{
    "track_id": 3, "confidence": 0.87,
    "bbox": {"x1": 210, "y1": 40, "x2": 380, "y2": 470},
    "depth_ok": true, "depth_valid_ratio": 0.93,
    "position": {"x": 2.14, "y": 0.31, "z": 0.12},
    "velocity": {"vx": -0.05, "vy": 0.02},
    "distance_m": 2.162, "bearing_deg": 8.2, "age_s": 4.1, "hits": 38
  }]
}
```

Target selection: `target_mode=nearest` picks the nearest person that has
valid depth. `POST /target {"track_id": 3}` locks one person. If a locked
person is lost, `target_id` becomes `null`. The pillar **never** switches to
another person by itself. `{"track_id": null}` switches back to automatic
mode.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RS_SOURCE` | `mock` | `rosbridge` (robot), `realsense` (pyrealsense2 on a PC), `mock` |
| `REALSENSE_ROSBRIDGE_HOST` / `_PORT` | `localhost` / `9091` | realsense_bridge address |
| `YOLO_WEIGHTS` | `yolov8n.pt` | a `.engine` file (TensorRT) also works |
| `YOLO_DEVICE` | auto | `0` = Jetson GPU |
| `YOLO_CONF` / `YOLO_IMGSZ` | `0.4` / `640` | |
| `CAM_TX`, `CAM_TY`, `CAM_TZ`, `CAM_PITCH_DEG`, `CAM_YAW_DEG` | 0.30, 0, 0.10, 0, 0 | camera mount in the base frame |
| `PROXIMITY_ALERT_M` | `0.8` | below this distance, publishes `mc.core.proximity_alert` |
| `REDIS_HOST` | – | if not set, there is no Redis publish |
| `CORS_ORIGINS` | `*` | comma-separated origins allowed to call the API from a browser |
| `MC_API_TOKEN` | – | if set, `POST /target` requires the `X-MC-Token` header |

## Running on the robot

The build context is `mission_control/` (see `CONVENTIONS.md`):

```bash
docker build -f perception/Dockerfile -t nero_go2/perception .
docker run -d --name nero_go2_perception --runtime nvidia --network host \
  --restart unless-stopped -e RS_SOURCE=rosbridge -e YOLO_DEVICE=0 \
  nero_go2/perception
```

The container needs `realsense_bridge` with `align_depth:=true` and PNG depth
(see `go2-hardware-bridge/realsense_bridge/Dockerfile`).

Debug page: `http://192.168.123.18:9112/`, which shows the annotated stream,
the live JSON and target-lock buttons.

## Tests

```bash
python -m pytest tests/test_perception_geometry.py -q        # pure math, 12 tests
RS_SOURCE=mock python perception/app.py                      # full pipeline on a photo
```
