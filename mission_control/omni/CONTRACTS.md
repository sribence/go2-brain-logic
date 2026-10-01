# OmniVision 360 — interfész-szerződés (KÖTELEZŐ minden agentnek)

Terv: `NERO_GO2/docs/21-omnivision-360-terv.md`. Branch: `feature/omnivision-360` (mindhárom repóban).
Ez a fájl a mérvadó a modulhatárokra. Ha eltérsz, itt jelöld (`> ELTÉRÉS:`).

## 0. Hardver-rig (valós, USB alapú)

| cam_id | modality | optika | forrás | megjegyzés |
|---|---|---|---|---|
| `rgb_front` | rgb | halszem (fisheye) | UVC `/dev/v4l/by-id/...` MJPEG 1280×720 | elöl |
| `rgb_left` | rgb | halszem / széles | UVC | bal oldal |
| `rgb_right` | rgb | halszem / széles | UVC | jobb oldal |
| `rgb_rear` | rgb | halszem / széles | UVC | hátul |
| `th_front_narrow` | thermal | pinhole, szűk FOV (~25–35°) | `thermal_bridge` HTTP | elöl, messzire |
| `th_front_wide` | thermal | pinhole, széles FOV (~50–90°) | `thermal_bridge` HTTP | elöl, közel |
| `tof_rear` | depth | pinhole ~70°×60°, 100×100 | `maixsense_bridge` HTTP | MaixSense A010 ToF, hátul, 0.2–2.5 m |

Cél-alkalmazás: biztonsági járőr, éjszakai felügyelet, ember keresése és **követése biztonságos távolságból** (shadow/pursuit mód). A robot **soha nem kerül 2.5 m-nél közelebb** üldözött/követett emberhez, a `safety_guard` zónái nem írhatók felül semmilyen módból.

## 1. Koordináta-rendszerek
- `base`: robot test, ROS-konvenció: x előre, y balra, z fel, méter.
- `cam`: OpenCV: z előre (optikai tengely), x jobbra, y le.
- `world`/`map`: a `mapping` pillér grid-frame-je (x, y, yaw a `robot.get_pose()`-ból).
- `T_base_cam`: 4×4 homogén, `p_base = T_base_cam @ p_cam`.
- Időbélyeg: `time.time()` unix float másodperc, a **rögzítés** ideje.

## 2. Konfiguráció — `omni/config/rig.yaml` (Agent A írja, mindenki olvassa)
```yaml
cameras:
  rgb_front:
    modality: rgb            # rgb | thermal | depth
    model: fisheye           # fisheye (OpenCV KB, D=4) | pinhole (D=5)
    width: 1280
    height: 720
    K: [[fx,0,cx],[0,fy,cy],[0,0,1]]
    D: [k1,k2,k3,k4]
    T_base_cam: [[...4x4...]]
    source: {type: uvc, device: /dev/video0, fourcc: MJPG, fps: 15}
    # type: uvc | http | mock ; http: {type: http, url: http://127.0.0.1:9120/cams/th_front_narrow/frame.npy}
lidar:
  source: {type: robot_client}   # robot_client | mock
```
A defaultok becsült értékek (kalibráció előtt), YAML-ben kommentelve.

## 3. Közös adattípusok — `omni/omni_types.py` (Agent A írja ELSŐKÉNT; a többiek importálják, nem módosítják)
```python
@dataclass
class Frame:
    cam_id: str
    modality: str        # "rgb" | "thermal" | "depth"
    t: float             # capture unix ts
    seq: int
    image: np.ndarray    # rgb: HxWx3 uint8 BGR | thermal: HxW float32 °C | depth: HxW float32 m (0 = invalid)

@dataclass
class Detection:         # egy kamera, egy képkocka
    cam_id: str
    bbox: tuple          # (x1, y1, x2, y2) az EREDETI kamerakép pixeleiben
    conf: float
    cls: str             # "person"
    modality: str

@dataclass
class PersonTrack:       # globális, base frame
    gid: int
    x: float; y: float; z: float      # base frame, m
    vx: float; vy: float              # base frame, m/s (ego-motion kompenzált)
    conf: float
    cams: list           # cam_id-k, amik most látják
    modality: list       # pl. ["rgb","thermal"]
    range_src: str       # "lidar" | "tof" | "ground_plane" | "thermal_size"
    age_s: float
    last_seen_t: float
    def to_dict(self) -> dict
```

## 4. Modulok és tulajdonosok (egy fájlt CSAK a tulajdonosa ír)

| Fájl (mission_control/ alatt) | Agent | API |
|---|---|---|
| `omni/omni_types.py`, `omni/rig.py`, `omni/camera_model.py`, `omni/capture.py`, `omni/rectify.py`, `omni/stream.py`, `omni/app.py`, `omni/config/rig.yaml`, `omni/Dockerfile`, `omni/requirements.txt`, `omni/README.md` | **A (core)** | lent |
| `omni/detect.py`, `omni/thermal_detect.py`, `omni/track.py`, `omni/perception360.py` | **B (perception)** | lent |
| `omni/colorize.py`, `omni/voxel_map.py` | **C (3D)** | lent |
| `safety_guard/*`, `navigation/nav_local.py`, `mapping/social_layer.py`, `omni/pursuit.py` | **D (safety+nav)** | lent |
| `digital-twin/static/omni.html`, `digital-twin/static/omni.js`, `digital-twin/static/omni/*`, `digital-twin/server.py` (csak új route-ok) | **E (UI)** | lent |
| `go2-hardware-bridge/thermal_bridge/*`, `go2-hardware-bridge/maixsense_bridge/*` | **F (hw)** | lent |
| tesztek: `tests/test_omni_<modul>.py`, `tests/test_safety_guard.py` stb. | a modul tulajdonosa | |

### A — core
- `rig.load_rig(path) -> Rig`; `Rig.cameras: dict[str, CameraSpec]`; `CameraSpec(cam_id, modality, width, height, model: CameraModel, T_base_cam: np.ndarray, source: dict)`.
- `camera_model.CameraModel`: `project(pts_cam: (N,3)) -> (uv: (N,2) float, valid: (N,) bool)`; `unproject(uv: (N,2)) -> rays_cam: (N,3) unit`. Alosztályok: `FisheyeModel`, `PinholeModel`. Vektorizált numpy.
- `camera_model.project_base(spec, pts_base) -> (uv, valid, depth_cam)` kényelmi függvény.
- `capture.CameraSource` (`open()`, `read() -> Frame|None`, `close()`), `UvcSource`, `HttpSource`, `MockSource`; `CaptureManager(rig).start()`, `.latest(cam_id) -> Frame|None`, `.latest_all() -> dict[str, Frame]`, `.stats() -> dict`.
- `rectify.build_cylindrical_lut(spec, hfov_deg, vfov_deg, out_w, out_h) -> (map_x, map_y)` (float32, `cv2.remap`-hez); `rectify.Rectifier(spec, ...)`: `.apply(img)`, `.to_original(bbox) -> bbox` (rektifikált → eredeti pixel).
- `app.py`: FastAPI `:9114` (env `OMNI_PORT`), CORS engedélyezve. Pipeline-szál: capture → `Perception360.process` → `VoxelMap.integrate`; az importok try/except mögött, hogy hiányzó modul esetén is induljon.

### B — perception
- `detect.Detector` (`detect(images: list[np.ndarray]) -> list[list[Detection-szerű tuple (bbox, conf, cls)]]`), `UltralyticsDetector(model_path, imgsz, half)` (TensorRT `.engine` vagy `.pt`, lazy import), `MockDetector`.
- `thermal_detect.ThermalPersonDetector(t_min=28.0, t_max=40.0, min_area_px=..)`: `detect(frame_celsius) -> list[(bbox, conf)]`; hőfolt-alapú, opcionálisan modellel.
- `track.MultiCamTracker`: kameránkénti asszociáció + globális fúzió base frame-ben, Kalman (x, y, vx, vy), `update(dets_3d, t, ego_delta=None) -> list[PersonTrack]`.
- `perception360.Perception360(rig, detector, thermal_detector=None)`: `process(frames: dict[str, Frame], lidar_base: np.ndarray|None, t: float, ego_delta=None) -> list[PersonTrack]`. Távolság sorrendben: LiDAR-pontok a bboxban → ToF (`tof_rear`) → talajsík (bbox alja) → hő-méret alapján.
- Redis: `mc.omni.persons` → `{"t": float, "persons": [PersonTrack.to_dict()]}`.

### C — 3D
- `colorize.colorize(points_base: (N,3), frames: dict[str, Frame], rig, zbuf_size=(160,120)) -> (pts (M,3), rgb (M,3) uint8, temp (M,) float32 NaN ha nincs)`; z-buffer takarásszűréssel, a legjobb kamera választásával.
- `voxel_map.VoxelMap(res=0.05, max_voxels=2_000_000)`: `integrate(pts, rgb, temp=None, t=None)`, `version: int`, `delta_since(version) -> bytes`, `snapshot() -> bytes`, `export_ply(path)`, `stats()`.
- **Bináris formátum (OVX1)**, little-endian: header `b"OVX1"` + `uint32 version` + `uint32 count` + `float32 res`, majd count × 16 bájt: `float32 x, y, z` + `uint8 r, g, b` + `uint8 flags` (bit0 = van hőadat és > 28 °C, bit1 = ember közelében). Az E agent ezt dekódolja JS-ben.

### D — safety + nav
- `safety_guard/` pillér, `:9115` (env `SAFETY_PORT`). `decide(persons, lidar_near_m, odom, cmd, cfg) -> SafetyDecision(level, vmax, vyaw_max, reason, cmd_out)` tiszta függvény.
- HTTP: `POST /move {vx, vy, vyaw, source}` → szűrt parancs továbbítása az `mc_motion /move`-ra (env `MOTION_URL`); `POST /stop`; `GET /state`. Redis: `mc.safety.state` → `{"t", "level": "CLEAR|CAUTION|SLOW|STOP", "vmax", "nearest_person_m", "reason"}`, 20 Hz.
- Zónák: STOP < 0.8 m, SLOW < 2.0 m, CAUTION < 3.5 m; a TTC < 1 s → STOP, < 2 s → SLOW; ha 0.3 s-nál régebbi a person-adat → SLOW. Gyerek / gyors ember → ×1.5.
- `omni/pursuit.py`: `PursuitController(cfg).step(target: PersonTrack, odom) -> (vx, vy, vyaw)`; tartott távolság ≥ 3.0 m (min 2.5 m), max 1.0 m/s, csak a `safety_guard`-on át megy ki.
- `navigation/nav_local.py`: DWA-lite tiszta függvény; `mapping/social_layer.py`: `social_cost_grid(persons_world, grid_meta) -> np.ndarray uint8`.

### E — UI
- `digital-twin/static/omni.html` + `omni.js` (+ `static/omni/*.js` modulok). Vanilla JS + three.js cdnjs-ről, nincs build. Az `omni` (`:9114`) és a `safety_guard` (`:9115`) közvetlenül elérhető; a host a `?omni=` és `?safety=` query paraméterrel felülírható (default: `location.hostname`).
- Végpontok, amiket fogyaszt (az A agent biztosítja): `GET /rig`, `WS /ws/video` (bináris: `uint8 cam_idx` + JPEG; a cam_idx a `/rig` sorrendjéből), `WS /ws/persons` (JSON, mint `mc.omni.persons`), `WS /ws/voxels` (OVX1 bináris; az első üzenet snapshot, utána delták), `GET /thermal/{cam_id}.png`, `GET /health`.

### F — hardware bridge (`go2-hardware-bridge` repó)
- `thermal_bridge/`, `:9120`: UVC hőkamerák (InfiRay P2 Pro / TOPDON TC001 típusú, 256×192, 2×-es magasságú YUYV keret, alsó fele nyers uint16 hőmérséklet: `°C = raw/64 − 273.15`; konfigurálható decoder). `GET /cams` · `GET /cams/{id}/frame.npy` (float32 °C, `np.save` formátum) · `GET /cams/{id}/frame.png` (színezett) · `GET /health`.
- `maixsense_bridge/`, `:9121`: MaixSense A010 ToF USB-serialon. `GET /frame.npy` (100×100 float32 méter, 0 = érvénytelen) · `GET /frame.png` · `GET /points.npy` (N×3 cam frame) · `GET /health`.
- Mindkettőben `--mock` mód (szintetikus képek), hogy hardver nélkül is fusson.

## 5. Közös szabályok
- Python 3.8-kompatibilis (Jetson JetPack 5): nincs `match`, nincs `X | Y` típus-szintaxis runtime-ban (`from __future__ import annotations` OK).
- Függőségek: numpy, opencv(-headless), fastapi, uvicorn, redis, pyyaml. Torch / ultralytics / TensorRT csak lazy import, opcionálisan.
- Minden új logika tiszta, egységtesztelhető függvényben; tesztek a `mission_control/tests/` alatt, a meglévő 105 teszt maradjon zöld.
- Mozgásparancs CSAK a `safety_guard`-on át. Armolni semmi nem armol.
- Nem commitolsz, nem pusholsz: a fájlokat a koordinátor commitolja.

## 6. Eltérések / kiegészítések (integráció után, 2026-10-01)
> ELTÉRÉS: `PersonTrack.height_m` (álló magasság a talajtól, 0 = ismeretlen). A `z` a test **középpontja**, nem magasság. A `safety_guard` gyerek-szabálya (`< 1.3 m` → zónák ×1.5) csak a `height_m`-et használja (`child_from_z=False`).
> ELTÉRÉS: `/rig` válasz: `{order, cameras:[{id, index, modality, width, height, model, K, D, max_fov_deg, T_base_cam, yaw_deg}]}`. `/ws/video` paraméterek: `?cams=a,b&fps=&width=`.
> ELTÉRÉS: `project_base` érvénytelen pontra `uv = -1`; mindig a `valid` maszkkal szűrj.
> ELTÉRÉS: `PursuitController.step(target, odom, now) -> (vx, vy, vyaw, state)`.
> ELTÉRÉS: `social_cost_grid` lapos, row-major uint8 tömb (`width*height`).
> ELTÉRÉS: OVX1-ben az evictált voxelek nem jelennek meg deltában; a kliens a következő snapshotnál tisztul.
> KIEGÉSZÍTÉS: `safety_guard` a `pursuit`/`follow`/`follow_executor` forrásnál 2.5 m-en belül nullázza a közelítő lineáris parancsot.
> INTEGRÁLVA: `navigation`, `mapping` (`_safe_move`/`_safe_stop`, `SAFETY_URL`) és `follow_executor` (`MOVE_URL`) a guardon át mozgat; compose: `omni`, `safety_guard` szolgáltatás.
> ISMERT: `mc_motion` default `MOTION_PORT=9102` ütközik a `mapping` portjával — külön hoston fut, compose-ban `MOTION_URL`-lel kell megadni.
