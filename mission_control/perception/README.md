# perception pillar (port 9112)

Detects people with YOLOv8, tracks them with ByteTrack (stable `track_id`),
and computes the 3D position of each person from the aligned depth image of
the Intel RealSense D435i. The pillar runs **on the robot** (Jetson Orin NX,
CUDA) in its own Docker container.

The pillar is **read-only**: it never sends a command to the robot. It
contains a person follower that computes the `Move(vx, vy, vyaw)` command it
*would* send, but every command is marked `dry_run: true` and only shown on
the HUD, the radar and the JSON output. A live executor is a later step, and
it must go through `core` (armed check, watchdog, E-stop).

## Status (2026-09-19)

Deployed and running on the Go2 Jetson (`192.168.123.18`):

| Item | Value |
|---|---|
| Container | `nero_go2_perception`, `--runtime nvidia --network host --restart unless-stopped` |
| Base image | `ultralytics/ultralytics:latest-jetson-jetpack5` (13.7 GB, Python 3.8, torch 2.1 with CUDA) |
| Source | `RS_SOURCE=rosbridge`, the `realsense_bridge` container on `localhost:9091` |
| Model | `yolov8n.pt` on GPU (`YOLO_DEVICE=0`) |
| Measured | loop about 17 fps, inference about 78 ms per frame |
| Camera | color 640x480 at 15 Hz, aligned depth about 10 Hz (16-bit PNG) |
| Follower | dry run only, no motion command leaves the container |

Open items:

- The camera mount extrinsics (`CAM_*`) are not measured yet. The defaults
  (`tx=0.30`, `tz=0.10`, no pitch or yaw) can put positions a few cm off.
- TensorRT export (`yolov8n.engine`) for faster inference.
- Live executor through `core`, which uses `apply_ego_motion()` with odometry
  and honours `command.valid_until`. It needs explicit approval before any
  test on the real robot.

## Data flow

```
RealSense D435i ──USB──> realsense_bridge (ROS1 Noetic, :9091 rosbridge)
    /camera/color/image_raw/compressed                  JPEG
    /camera/aligned_depth_to_color/image_raw/compressed 16-bit PNG
    /camera/color/camera_info                           intrinsics
                 │ roslibpy (localhost)
                 ▼
perception (:9112)  YOLOv8 person + ByteTrack ─> torso depth ─> deproject
                    ─> camera->base transform ─> per-track Kalman filter (position + velocity)
```

The depth is aligned to the color image, so each color pixel has the matching
depth pixel. The distance comes from the "torso window" of the bbox (middle
40 % of the width, 20–60 % of the height), using the 30th percentile. This
way the background between the legs and arms does not pull the distance
back. A single-frame jump larger than 1.5 m (for example, the bbox catches a
wall) is rejected by `TrackSmoother`.

## Filtering

Each track has a constant-velocity Kalman filter in the base frame
(`geometry3d.TrackSmoother`, state `[x y z vx vy vz]`). The measurement noise
follows the depth camera: along the camera ray it grows with the square of
the distance, across the ray it grows linearly. A measurement outside the
Mahalanobis gate (chi-square 16.27, 3 dof) is dropped as an outlier; three
outliers in a row restart the filter at the new position. `position` and
`velocity` are the filtered values, `position_raw` is the unfiltered one
that the follower uses for jump detection. `bbox_smooth` is an EMA of the
YOLO box, used only for drawing.

Live measurement on 2026-09-19, a seated person at 2.04 m: position noise
is about 1 cm before and after filtering. The filter matters for walking
people and depth outliers, not for a still person.

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
  "count": 1, "target_id": 3, "target_mode": "locked",
  "persons": [{
    "track_id": 3, "confidence": 0.87,
    "bbox": {"x1": 210, "y1": 40, "x2": 380, "y2": 470},
    "depth_ok": true, "depth_valid_ratio": 0.93,
    "position": {"x": 2.14, "y": 0.31, "z": 0.12},
    "velocity": {"vx": -0.05, "vy": 0.02},
    "distance_m": 2.162, "bearing_deg": 8.2, "age_s": 4.1, "hits": 38
  }],
  "follow": {
    "state": "TRACKING", "reason": "tracking", "track_id": 3,
    "similarity": 0.91, "last_seen_age_s": 0.0,
    "gate": {"x": 2.15, "y": 0.31, "radius_m": 0.7},
    "target": {"track_id": 3, "position": {"x": 2.14, "y": 0.31, "z": 0.12}, "...": "..."},
    "goal": {"x": 0.95, "y": 0.14, "yaw_deg": 8.2, "distance_m": 0.96, "distance_cm": 96},
    "command": {"vx": 0.42, "vy": 0.0, "vyaw": 0.21, "vx_desired": 0.5,
                "vyaw_desired": 0.21, "valid_until": 1789813100.6, "dry_run": true}
  }
}
```

## Person following (dry run)

`target_follower.py` follows **one locked person**. It computes the command
it *would* send and puts it in every result under `follow`. It sends
nothing to the robot: `command.dry_run` is always `true`.

Identity is protected in two ways:

1. **Spatial gate.** The target must be near its predicted position
   (last position + velocity). The gate radius is `gate_base_m` plus
   `max_person_speed` × time since last seen, up to `gate_max_m`. A move
   that is faster than a person can walk is treated as a different person.
2. **Appearance.** The first `acquire_frames` frames after the lock learn
   HSV colour histograms of the upper and lower body (`appearance.py`).
   The locked track must keep matching this template.

| State | Meaning | Command |
|---|---|---|
| `IDLE` | no lock | 0 |
| `ACQUIRING` | locked, learning the appearance | 0 |
| `TRACKING` | target confirmed in this frame | computed |
| `OCCLUDED` | target missing, jumped, or looks different | 0 |
| `LOST` | occluded longer than `lost_timeout_s` | 0, needs a manual re-lock |

In `OCCLUDED`, the follower takes back only a person who is inside the gate
**and** matches the appearance, and only if exactly one person matches. This
covers the case where ByteTrack gives the same person a new id. If two
people match, it waits. It **never** selects someone else by itself.

Limit: two people in similar clothes cannot be told apart by appearance.
Then the gate is the only guard. The upgrade path is an OSNet ReID
embedding behind the same `appearance.extract()/similarity()` interface.

Command (`follow.command`) uses the units of the Go2 `Move(vx, vy, vyaw)`:
m/s and rad/s, with `vyaw` > 0 = turn left. The command is a P controller
on the distance error and the bearing, with these rules:

- The speed ramps up (`max_accel`), but stopping is immediate.
- The robot turns first: `vx` goes to 0 when the bearing reaches
  `turn_first_deg`.
- The robot never drives forward inside `min_safe_distance_m`.
- Reverse is off by default.

`follow.goal` is the point `follow_distance_m` in front of the person, in
the base frame, with `distance_cm`.

**Before live motion:** the executor must call `apply_ego_motion()` with the
robot odometry. Without it, the gate stays where the person was and drifts
off them. The executor must also drop any command after `valid_until`.

Every `FollowConfig` field can be set as `FOLLOW_<FIELD>`, for example
`FOLLOW_MAX_VX=0.3`. `GET /follow` returns the active config.

The annotated stream shows the follow state, the reason, the command and
the goal. A top-down radar in the bottom-right corner shows the people, the
gate, the goal and the heading arrow.

## Follow modes (`follow_modes.py`)

The go2-console drives the follower through modes. `FollowSupervisor` wraps
`TargetFollower` and holds the operator settings.

| Mode | Behaviour |
|---|---|
| `off` | follower released, command zero |
| `user_follow` | the operator locks one person with `/follow/lock`; never an automatic lock |
| `intruder` | locks the nearest person with depth within 6 m automatically and raises an `intruder` alert; after `LOST` it may lock the next person only after 3 s |
| `trick` | like `user_follow`, plus manual gestures |

Settings (`POST /follow`, any subset):

- `target_distance_m`: 1.0 to 4.0 m, sets `FollowConfig.follow_distance_m`.
- `audio_alert`: alerts (`intruder`, `target_lost`) are always logged and
  published on `mc.perception.alert`; with audio on, `AUDIO_ALERT_URL` is
  also called when it is set.
- `dry_run`: always `true` at start. `false` is refused with 403 unless the
  container runs with `PERCEPTION_ALLOW_LIVE=1`. Even then the pillar sends
  nothing to the robot: the flag only tells a separate executor that the
  command may go to `mc_motion`.

Gestures (`POST /follow/gesture`, trick mode only, 409 otherwise):
`wave` plans `hello`, `stop` plans `sit` and holds the command at zero while
tracking continues, `ok` resumes. Gesture recognition and action execution
are not implemented yet: the response has `executed: false`.

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
| `MC_API_TOKEN` | – | if set, `POST /follow/*` and `/target` require the `X-MC-Token` header |
| `FOLLOW_*` | see `FollowConfig` | follower tuning |
| `PERCEPTION_ALLOW_LIVE` | `0` | `1` allows `dry_run: false` |
| `AUDIO_ALERT_URL` | empty | POSTed on alerts, `{kind}` is replaced |
| `KF_*` | see `KalmanConfig` | Kalman tuning, e.g. `KF_ACCEL_STD=0.8`, `KF_RANGE_STD_QUAD=0.015` |
| `BBOX_SMOOTH_ALPHA` | `0.4` | EMA of the drawn box (display only) |

## Running on the robot

The build context is `mission_control/` (see `CONVENTIONS.md`):

```bash
docker build -f perception/Dockerfile -t nero_go2/perception .
docker run -d --name nero_go2_perception --runtime nvidia --network host \
  --restart unless-stopped -e RS_SOURCE=rosbridge -e YOLO_DEVICE=0 \
  nero_go2/perception
```

Or use `deploy.sh`: copy this folder (with `CONVENTIONS.md`'s build context
layout) to `/home/unitree/nero_go2_dev/perception/ctx/` on the robot, then:

```bash
bash /home/unitree/nero_go2_dev/perception/deploy.sh
```

From a Windows PC on the robot network (SSH with `plink`, `scp` is blocked):

```bash
tar cf - -C mission_control perception | plink -ssh -batch unitree@192.168.123.18 "mkdir -p ~/nero_go2_dev/perception/ctx && tar xf - -C ~/nero_go2_dev/perception/ctx"
```

Logs of follower state changes go to `/home/unitree/nero_go2_dev/perception/logs`.

The container needs `realsense_bridge` with `align_depth:=true` and PNG depth
(see `go2-hardware-bridge/realsense_bridge/Dockerfile`).

Debug page: `http://192.168.123.18:9112/`, which shows the annotated stream,
the live JSON, and lock/release buttons.

## Live motion test, 2026-09-19

Set-up: `mc_motion` (remote override, `/odom`), `follow_executor` with
`EXEC_MAX_VX=0` (turn only), perception with `PERCEPTION_ALLOW_LIVE=1`.

Results:

- Remote override works. A stick push (`ly=-0.81`, later `lx=0.28`) while
  armed disarmed `mc_motion` at once and latched the executor off.
- The console E-stop did NOT reach `mc_motion`. The `go2_console` container
  on the robot runs without `MOTION_URL`, so `/api/estop` posts nowhere.
  Until that is fixed, stop with the remote or `POST :9102/estop`.
- The robot turned toward the locked person (bearing 14.5 to 5.7 degrees),
  but far too slowly: about 5 s after the person moved, then overshoot and
  small back-and-forth steps.

Cause, measured: camera to perception transport latency is 0.5 to 0.9 s at
only 4 to 6 Hz (rosbridge sends 640x480 JPEG plus 16-bit PNG as base64 JSON).
With the executor's 0.4 s staleness limit the robot kept stopping (14 stops).
The Kalman filter also treated the robot's own turn as person motion.

Fixes in this commit:

- Frame time is the camera capture stamp, not the arrival time, so `age_s`
  shows the real latency.
- `POST /follow/ego` also moves the Kalman tracks into the new robot frame.
- The follower predicts the bearing over the latency
  (`bearing -= current_vyaw * latency`, `FOLLOW_LATENCY_COMP`), yaw deadband
  6 degrees, `max_yaw_accel` 3.0.
- A 424x240 camera mode was tried to cut the data: the D435i depth stream
  fails to start in that mode ("Depth stream start failure"), so the bridge
  is back at 640x480.

Open: the transport latency itself. Options: read the camera directly in the
perception container (pyrealsense2, `RS_SOURCE=realsense`) instead of through
rosbridge, or a lower color resolution with a supported depth mode.

## Live dry-run test

1. Open `http://192.168.123.18:9112/`.
2. Let one person stand in front of the camera and wait for the `#N` box.
3. Press "kovetes #N". The state goes `ACQUIRING`, then `TRACKING` after
   `acquire_frames` frames.
4. Watch the HUD (state, reason, `vx`/`vyaw` with `[DRY RUN]`, goal in cm) and
   the radar (gate circle, goal cross, heading arrow).
5. Check these cases:
   - the person jumps fast: `OCCLUDED` with "jump", command zero;
   - the person is hidden: `OCCLUDED`, a search inside the gate;
   - another person steps in: no switch to the other person;
   - the person is gone for `lost_timeout` (2 s): `LOST`.
6. Press "elengedes" to return to `IDLE`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /stream.mjpg`, `GET /frame.jpg` | annotated image with HUD and radar |
| `GET /persons`, `GET /persons/stream` (SSE) | people + `follow` block, every frame |
| `GET /follow` | flat UI block (`mode`, `target_distance_m`, `audio_alert`, `dry_run`, `live_allowed`, `hold`, `state`, `reason`, `target_id`, `command`, `target_dist_cm`, `goal`, `last_gesture`, `alerts`) + `follow` detail + `config` |
| `POST /follow` | `{mode?, target_distance_m?, audio_alert?, dry_run?}`; 422 bad value, 403 live not allowed |
| `POST /follow/gesture` `{"gesture": "wave"}` | trick mode only; planned action, not executed |
| `POST /follow/lock` `{"track_id": 3}` | lock; returns 409 if the id has no valid depth |
| `POST /follow/release` | back to `IDLE`, mode `off` |
| `POST /target` | legacy alias: an id = lock, `null` = release |
| `GET /status` | health, `follow_state` |

## Tests

```bash
python -m pytest tests/test_perception_geometry.py tests/test_target_follower.py tests/test_follow_modes.py -q
RS_SOURCE=mock python perception/app.py                      # full pipeline on a photo
```
