# omni pillar — OmniVision 360 (port 9114)

360° person perception and a coloured 3D map for the Go2. Reads a 7-camera
rig (4 USB fisheye RGB, 2 front thermal, 1 rear MaixSense ToF) plus the
LiDAR, finds and tracks people in the base frame, colours LiDAR points from
the cameras into a voxel map, and streams everything to the `omni.html` UI.

Read-only: it never moves the robot. Pursuit commands (`pursuit.py`) go out
only through `safety_guard` (:9115). Interfaces: [CONTRACTS.md](CONTRACTS.md).
Plan: `NERO_GO2/docs/21-omnivision-360-terv.md`.

## Modules

| File | What |
|---|---|
| `config/rig.yaml` | real rig: intrinsics (uncalibrated estimates), `T_base_cam`, sources. Camera order = `/ws/video` `cam_idx` |
| `config/rig_mock.yaml` | same geometry, every source `mock` |
| `rig.py` | `load_rig(path, force_mock=False) -> Rig` (`cameras`, `ordered_ids`, `to_dict()`), `T_from_mount()` |
| `camera_model.py` | `FisheyeModel` (OpenCV KB, k1..k4, supports >180°), `PinholeModel` (k1,k2,p1,p2,k3), `project_base`, `rays_base` |
| `capture.py` | `UvcSource` (V4L2 MJPG or Jetson GStreamer `nvv4l2decoder mjpeg=1`), `HttpSource` (bridge `.npy`/`.jpg`), `MockSource` + `MockScene`, `CaptureManager` |
| `rectify.py` | `build_cylindrical_lut`, `Rectifier.apply / to_original / pixel_to_ray_base` (level cylinder, horizon straight) |
| `stream.py` | JPEG/PNG encoding, thermal (INFERNO) / depth (TURBO) colourising |
| `app.py` | FastAPI service + pipeline thread |
| `detect.py`, `thermal_detect.py`, `track.py`, `perception360.py` | person detection + tracking (optional at runtime) |
| `colorize.py`, `voxel_map.py` | LiDAR colouring, OVX1 voxel map (optional at runtime) |

## Environment

| Var | Default | Meaning |
|---|---|---|
| `OMNI_PORT` | `9114` | HTTP port |
| `OMNI_MOCK` | `0` | `1` = `rig_mock.yaml`, all sources synthetic, mock LiDAR, mock detector |
| `OMNI_RIG` | `config/rig.yaml` | rig file (relative to `omni/`) |
| `OMNI_RATE_HZ` | `10` | pipeline rate |
| `OMNI_STREAM_FPS` / `OMNI_STREAM_WIDTH` / `OMNI_STREAM_QUALITY` | `10` / `640` / `70` | `/ws/video` defaults |
| `OMNI_DETECTOR` | `ultralytics` (`mock` in mock mode) | `ultralytics` \| `mock` \| `none` |
| `OMNI_MODEL` | `yolo11n.pt` | `.pt` or TensorRT `.engine` |
| `OMNI_RECTIFY` | `1` | cylindrical rectification of fisheye cams before detection |
| `OMNI_VOXEL_RES` / `OMNI_VOXEL_EVERY` | `0.05` / `2` | voxel size (m) / integrate every N ticks |
| `OMNI_FRAME_MAX_AGE_S` | `1.0` | older frames are ignored by the pipeline |
| `OMNI_LOG_DIR` | `omni/logs` | JSONL events (`events.jsonl`) |
| `REDIS_HOST` / `REDIS_PORT` | unset / `6379` | optional; publishing failures only log |
| `ROBOT_CLIENT_MODE`, `ROBOT_BACKEND` | core defaults | LiDAR + pose via `core/robot_client` |

Redis channels: `mc.omni.persons` `{"t", "persons": [PersonTrack.to_dict()]}` every tick,
`mc.omni.health` once a second.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, cameras up/down, module availability |
| GET | `/rig` | `{order, cameras: [{id, index, modality, width, height, model, K, D, T_base_cam, ...}]}` |
| GET | `/stats` | per-camera fps/drops/errors, stage timings, voxel stats |
| GET | `/persons` | `{"t", "persons": [...]}` |
| GET | `/cameras/{cam_id}/frame.jpg?width=&quality=&rectified=1` | latest frame (thermal/depth colourised) |
| GET | `/thermal/{cam_id}.png?t_min=&t_max=&auto=1` | INFERNO thermal image |
| GET | `/depth/{cam_id}.png` | TURBO depth image |
| GET | `/map/export.ply` | binary PLY of the voxel map |
| WS | `/ws/video?cams=a,b&fps=&width=` | binary: `uint8 cam_idx` + JPEG (rgb, thermal and depth colourised) |
| WS | `/ws/persons` | JSON `{"t", "persons"}` at 10 Hz |
| WS | `/ws/voxels` | OVX1 binary: snapshot first, then deltas every 0.5 s |

## Run

```bash
cd mission_control/omni
pip install -r requirements.txt numpy opencv-python-headless
OMNI_MOCK=1 python3 app.py            # no hardware needed

curl -s localhost:9114/health | jq
curl -s localhost:9114/rig | jq '.cameras[] | {id, index, model}'
curl -s localhost:9114/persons | jq
curl -so front.jpg localhost:9114/cameras/rgb_front/frame.jpg
curl -so th.png    localhost:9114/thermal/th_front_wide.png
curl -so map.ply   localhost:9114/map/export.ply
```

Docker: see the header of `Dockerfile` (dev: `python:3.8-slim`, robot:
`--build-arg BASE=ultralytics/ultralytics:latest-jetson-jetpack5 --build-arg EXTRA_REQS=""`).
On the Jetson set `gst: true` on the UVC sources in `rig.yaml` to decode MJPEG
in hardware, and point `device:` at the stable `/dev/v4l/by-id/...` paths.

## Tests

```bash
cd mission_control && python3 -m pytest tests/test_omni_core.py -q
```
