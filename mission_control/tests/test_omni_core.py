"""omni core (Agent A): camera models, rig, rectify, capture, stream, app smoke.

No hardware, network or redis needed.
"""
import importlib
import os
import sys
import time

import numpy as np
import pytest

MC = os.path.join(os.path.dirname(__file__), "..")
if MC not in sys.path:
    sys.path.insert(0, MC)

cv2 = pytest.importorskip("cv2")

from omni.camera_model import FisheyeModel, PinholeModel, project_base, rays_base  # noqa: E402
from omni.capture import CaptureManager, HttpSource, MockScene, MockSource, make_source  # noqa: E402
from omni.rectify import Rectifier, build_cylindrical_lut  # noqa: E402
from omni.rig import DEFAULT_RIG, MOCK_RIG, T_from_mount, load_rig  # noqa: E402
from omni import stream  # noqa: E402

K_FISH = np.array([[400.0, 0, 640.0], [0, 402.0, 361.0], [0, 0, 1]])
D_FISH = np.array([-0.02, 0.005, -0.001, 0.0003])
K_PIN = np.array([[446.0, 0, 128.0], [0, 446.0, 96.0], [0, 0, 1]])
D_PIN = np.array([-0.12, 0.03, 0.001, -0.0015, 0.0])


def _rand_dirs(n, max_theta_deg, seed=0):
    rng = np.random.RandomState(seed)
    th = np.radians(rng.uniform(0, max_theta_deg, n))
    ph = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(0.5, 10.0, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1) * r[:, None]


# ---------------------------------------------------------------- camera models

def test_fisheye_matches_cv2_fisheye():
    m = FisheyeModel(K_FISH, D_FISH, 1280, 720)
    pts = _rand_dirs(500, 85.0)  # cv2.fisheye only handles z > 0
    uv, _ = m.project(pts)
    ref, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_FISH, D_FISH)
    assert np.abs(uv - ref.reshape(-1, 2)).max() < 1e-6


def test_fisheye_unproject_matches_cv2_and_roundtrips():
    m = FisheyeModel(K_FISH, D_FISH, 1280, 720)
    rng = np.random.RandomState(1)
    uv = np.stack([rng.uniform(0, 1279, 400), rng.uniform(0, 719, 400)], 1)
    rays = m.unproject(uv)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    uv2, valid = m.project(rays * 3.0)
    # image corners lie beyond the 200 deg lens circle -> invalid there only
    th = np.arccos(np.clip(rays[:, 2], -1, 1))
    assert np.array_equal(valid, th < m.theta_max)
    assert valid.mean() > 0.9
    assert np.abs(uv2[valid] - uv[valid]).max() < 1e-6
    # cv2 undistort gives tan(theta) coords -> compare directions for z > 0
    und = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), K_FISH, D_FISH).reshape(-1, 2)
    front = rays[:, 2] > 0.2
    ref = np.concatenate([und, np.ones((len(und), 1))], 1)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.abs(ref[front] - rays[front]).max() < 1e-5


def test_fisheye_wide_angle_beyond_90deg():
    K = np.array([[300.0, 0, 640.0], [0, 300.0, 360.0], [0, 0, 1]])
    m = FisheyeModel(K, [0, 0, 0, 0], 1280, 720, max_fov_deg=200)
    p = np.array([[1.0, 0.0, -0.1]])  # 95.7 deg off-axis, to the right
    uv, valid = m.project(p)
    assert valid[0] and abs(uv[0, 0] - (640 + 300 * np.arctan2(1.0, -0.1))) < 1e-6
    back = m.unproject(uv)
    assert np.allclose(back[0], p[0] / np.linalg.norm(p[0]), atol=1e-6)
    _, valid = m.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    assert not valid.any()


def test_pinhole_matches_cv2_and_roundtrips():
    m = PinholeModel(K_PIN, D_PIN, 256, 192)
    pts = _rand_dirs(300, 14.0, seed=2)
    uv, valid = m.project(pts)
    ref, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K_PIN, D_PIN)
    assert np.abs(uv - ref.reshape(-1, 2)).max() < 1e-6
    rays = m.unproject(uv[valid])
    uv2, v2 = m.project(rays * 5.0)
    assert v2.all() and np.abs(uv2 - uv[valid]).max() < 1e-4


def test_pinhole_validity():
    m = PinholeModel(K_PIN, None, 256, 192)
    uv, valid = m.project(np.array([[0, 0, 2.0], [0, 0, -2.0], [5.0, 0, 1.0], [0.1, 0.05, 1.0]]))
    assert valid.tolist() == [True, False, False, True]
    assert uv[0].tolist() == [128.0, 96.0]


def test_float32_fast_path_close_to_float64():
    m = FisheyeModel(K_FISH, D_FISH, 1280, 720)
    pts = _rand_dirs(1000, 99.0, seed=3)
    uv64, v64 = m.project(pts)
    uv32, v32 = m.project(pts.astype(np.float32))
    assert uv32.dtype == np.float32
    both = v64 & v32
    assert both.sum() > 0.95 * v64.sum()
    assert np.abs(uv32[both] - uv64[both]).max() < 0.05


# ---------------------------------------------------------------- rig

def test_rig_loads_seven_cameras_in_stable_order():
    for path in (DEFAULT_RIG, MOCK_RIG):
        rig = load_rig(path)
        assert rig.ordered_ids == ["rgb_front", "rgb_left", "rgb_right", "rgb_rear",
                                   "th_front_narrow", "th_front_wide", "tof_rear"]
        assert [rig.cameras[c].index for c in rig.ordered_ids] == list(range(7))
    rig = load_rig(DEFAULT_RIG)
    assert rig.cameras["rgb_front"].source["type"] == "uvc"
    assert rig.cameras["th_front_narrow"].source["url"].startswith("http://127.0.0.1:9120/")
    assert rig.cameras["tof_rear"].modality == "depth"
    assert (rig.cameras["tof_rear"].width, rig.cameras["tof_rear"].height) == (100, 100)
    mock = load_rig(MOCK_RIG)
    assert all(c.source["type"] == "mock" for c in mock.cameras.values())
    forced = load_rig(DEFAULT_RIG, force_mock=True)
    assert all(c.source["type"] == "mock" for c in forced.cameras.values())


def test_rig_headings_and_fovs():
    rig = load_rig(DEFAULT_RIG)
    yaws = {c: round(rig.cameras[c].yaw_deg) for c in rig.ordered_ids}
    assert yaws["rgb_front"] == 0 and yaws["rgb_left"] == 90
    assert yaws["rgb_right"] == -90 and abs(yaws["rgb_rear"]) == 180
    assert 28 < rig.cameras["th_front_narrow"].model.hfov_deg() < 36
    assert 50 < rig.cameras["th_front_wide"].model.hfov_deg() < 62
    assert rig.cameras["rgb_front"].model.hfov_deg() > 170
    d = rig.to_dict()
    assert d["order"] == rig.ordered_ids and len(d["cameras"]) == 7
    c0 = d["cameras"][0]
    for k in ("id", "index", "modality", "width", "height", "model", "K", "D", "T_base_cam"):
        assert k in c0
    assert np.asarray(c0["T_base_cam"]).shape == (4, 4)


def test_T_from_mount_axes():
    T = T_from_mount([0.2, 0, 0.1], yaw_deg=90, pitch_deg=0)
    # camera z (optical axis) -> base +y (left); camera y (down) -> base -z
    assert np.allclose(T[:3, 2], [0, 1, 0], atol=1e-9)
    assert np.allclose(T[:3, 1], [0, 0, -1], atol=1e-9)
    T = T_from_mount([0, 0, 0], pitch_deg=12)
    assert T[2, 2] < 0  # looks down


def test_project_base_front_and_masks():
    rig = load_rig(DEFAULT_RIG)
    front = rig.cameras["rgb_front"]
    rear = rig.cameras["rgb_rear"]
    pts = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    uv, valid, depth = project_base(front, pts)
    assert valid.tolist() == [True, False]
    assert abs(uv[0, 0] - 640) < 1.0 and depth[0] > 4.5
    uv, valid, _ = project_base(rear, pts)
    assert valid.tolist() == [False, True]
    th = rig.cameras["th_front_narrow"]
    _, valid, _ = project_base(th, np.array([[5.0, 0.0, 0.1], [5.0, 4.0, 0.1], [-5.0, 0, 0]]))
    assert valid.tolist() == [True, False, False]
    # rays_base: ray through the projected pixel passes through the point
    uv, valid, _ = project_base(front, np.array([[3.0, 1.0, 0.5]]))
    o, d = rays_base(front, uv)
    p = np.array([3.0, 1.0, 0.5])
    s = np.dot(p - o, d[0])
    assert np.linalg.norm(o + s * d[0] - p) < 1e-6
    # empty input
    uv, valid, depth = project_base(front, np.zeros((0, 3)))
    assert uv.shape == (0, 2) and valid.shape == (0,)


def test_project_base_timing_30k_points():
    rig = load_rig(DEFAULT_RIG)
    pts = np.random.RandomState(0).uniform(-8, 8, (30000, 3)).astype(np.float32)
    out = {}
    for cid, spec in rig.cameras.items():
        project_base(spec, pts)
        t0 = time.perf_counter()
        for _ in range(5):
            project_base(spec, pts)
        out[cid] = (time.perf_counter() - t0) / 5 * 1000
    print("project_base 30k pts ms:", {k: round(v, 2) for k, v in out.items()})
    assert max(out.values()) < 50.0  # generous: shared CI boxes


# ---------------------------------------------------------------- rectify

def test_cylindrical_lut_sanity_and_to_original():
    rig = load_rig(DEFAULT_RIG)
    spec = rig.cameras["rgb_front"]
    mx, my = build_cylindrical_lut(spec, 120.0, 70.0, 320, 200)
    assert mx.shape == (200, 320) and mx.dtype == np.float32
    assert (mx >= 0).mean() > 0.95
    # centre column looks straight ahead -> near image centre column
    assert abs(mx[100, 160] - 640) < 5
    # left side of the cylinder -> left side of the image (yaw left = image left)
    assert mx[100, 10] < 640 < mx[100, 310]
    # level horizon: row-middle maps above the image centre (camera pitched down 12 deg)
    assert my[100, 160] < 360

    rect = Rectifier(spec)
    assert rect.out_w == 640 and rect.out_h > 200
    img = np.zeros((720, 1280, 3), np.uint8)
    img[300:420, 600:680] = 255
    out = rect.apply(img)
    assert out.shape == (rect.out_h, rect.out_w, 3)
    box = rect.to_original((300, 150, 340, 250))
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 1279 and 0 <= y1 < y2 <= 719
    # the original-image box must contain the LUT image of the rect box centre
    cx, cy = rect.map_x[200, 320], rect.map_y[200, 320]
    assert x1 <= cx <= x2 and y1 <= cy <= y2
    # a point projected into the original lies in the to_original box of its rect pixel
    d = rect.pixel_to_ray_base(np.array([[320.0, 200.0]]))
    uv, valid, _ = project_base(spec, spec.position + 4.0 * d)
    assert valid[0] and abs(uv[0, 0] - cx) < 1.0 and abs(uv[0, 1] - cy) < 1.0


def test_rectifier_pinhole_defaults():
    rig = load_rig(DEFAULT_RIG)
    rect = Rectifier(rig.cameras["th_front_wide"])
    t = np.full((192, 256), 20.0, np.float32)
    out = rect.apply(t)
    assert out.dtype == np.float32 and out.shape == (rect.out_h, rect.out_w)
    box = rect.to_original((10, 10, 50, 50))
    assert box is not None and box[0] < box[2] and box[1] < box[3]


# ---------------------------------------------------------------- capture

def test_mock_source_frames():
    rig = load_rig(MOCK_RIG)
    shapes = {}
    for spec in rig.cameras.values():
        src = MockSource(spec)
        assert src.open()
        fr = src.read()
        shapes[spec.cam_id] = (fr.image.shape, fr.image.dtype, fr.modality)
        assert fr.seq == 1 and fr.t > 0
        # deterministic for a given t
        a, b = src.render(12.5), src.render(12.5)
        assert np.array_equal(a, b)
    assert shapes["rgb_front"] == ((720, 1280, 3), np.uint8, "rgb")
    assert shapes["th_front_wide"] == ((192, 256), np.float32, "thermal")
    assert shapes["tof_rear"] == ((100, 100), np.float32, "depth")


def test_mock_scene_is_geometrically_consistent():
    rig = load_rig(MOCK_RIG)
    scene = MockScene(ground_z=rig.ground_z)
    # find a t where the person is in front of the robot
    for t in np.arange(0, 60, 0.5):
        p = scene.person_positions(t)[0]
        if p[0] > 2.0 and abs(p[1]) < 0.5:
            break
    spec = rig.cameras["rgb_front"]
    img = MockSource(spec, scene=scene).render(t)
    red = (img[:, :, 2] == 255) & (img[:, :, 0] == 0)
    assert red.sum() > 200
    ys, xs = np.nonzero(red)
    mid = p.copy()
    mid[2] += 0.85
    uv, valid, _ = project_base(spec, mid[None])
    assert valid[0]
    assert xs.min() <= uv[0, 0] <= xs.max() and ys.min() <= uv[0, 1] <= ys.max()
    th = MockSource(rig.cameras["th_front_wide"], scene=scene).render(t)
    assert th.max() > 33.0 and np.median(th) < 23.0
    lid = scene.lidar_points(t)
    assert lid.dtype == np.float32 and lid.shape[1] == 3 and len(lid) > 1000


class _FailingSource(MockSource):
    def open(self):
        raise RuntimeError("no device")


def test_capture_manager_mock_rig_and_failure_isolation():
    rig = load_rig(MOCK_RIG)
    bad = _FailingSource(rig.cameras["rgb_rear"])
    cm = CaptureManager(rig, sources={"rgb_rear": bad}).start()
    try:
        others = [c for c in rig.ordered_ids if c != "rgb_rear"]
        assert cm.wait_ready(15.0, cams=others)
        time.sleep(0.3)
        st = cm.stats()
        assert set(st) == set(rig.ordered_ids)
        assert st["rgb_rear"]["available"] is False and "no device" in st["rgb_rear"]["last_error"]
        assert st["rgb_front"]["available"] and st["rgb_front"]["frames"] >= 1
        allf = cm.latest_all()
        assert "rgb_rear" not in allf and set(others) <= set(allf)
        assert cm.latest("rgb_rear") is None and cm.latest("nope") is None
    finally:
        cm.stop()


def test_http_source_decodes_npy(monkeypatch):
    import io

    rig = load_rig(DEFAULT_RIG)
    spec = rig.cameras["th_front_narrow"]
    src = make_source(spec)
    assert isinstance(src, HttpSource)
    buf = io.BytesIO()
    np.save(buf, np.full((192, 256), 30.0, np.float32))
    monkeypatch.setattr(src, "fetch", lambda: buf.getvalue())
    fr = src.read()
    assert fr.image.dtype == np.float32 and fr.image.shape == (192, 256) and fr.modality == "thermal"

    def boom():
        raise OSError("connection refused")

    monkeypatch.setattr(src, "fetch", boom)
    assert src.read() is None


# ---------------------------------------------------------------- stream

def test_stream_helpers():
    t = np.linspace(15, 40, 192 * 256, dtype=np.float32).reshape(192, 256)
    c = stream.colorize_thermal(t)
    assert c.shape == (192, 256, 3) and c.dtype == np.uint8
    d = np.zeros((100, 100), np.float32)
    d[50:, :] = 1.0
    cd = stream.colorize_depth(d)
    assert cd[0, 0].tolist() == [0, 0, 0] and cd[99, 0].any()
    jpg = stream.encode_jpeg(np.zeros((720, 1280, 3), np.uint8), quality=70, max_width=640)
    assert jpg[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == (360, 640, 3)
    assert stream.encode_png(c)[:4] == b"\x89PNG"
    msg = stream.ws_video_message(3, jpg)
    assert msg[0] == 3 and msg[1:] == jpg


# ---------------------------------------------------------------- app smoke

@pytest.fixture(scope="module")
def omni_client(tmp_path_factory):
    os.environ["OMNI_MOCK"] = "1"
    os.environ.pop("REDIS_HOST", None)
    os.environ["OMNI_LOG_DIR"] = str(tmp_path_factory.mktemp("omni_logs"))
    from fastapi.testclient import TestClient

    mod = importlib.import_module("omni.app")
    mod = importlib.reload(mod)
    with TestClient(mod.app) as client:
        assert mod.svc.cap.wait_ready(20.0)
        yield client, mod


def test_app_health_rig_frames(omni_client):
    client, mod = omni_client
    h = client.get("/health").json()
    assert h["ok"] and h["mock"] and h["cams_total"] == 7 and h["modules"]["redis"] is False
    rig = client.get("/rig").json()
    assert [c["id"] for c in rig["cameras"]] == rig["order"]
    assert rig["cameras"][0]["model"] == "fisheye" and len(rig["cameras"][0]["D"]) == 4
    r = client.get("/cameras/rgb_front/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    r = client.get("/cameras/th_front_wide/frame.jpg?width=128")
    assert r.status_code == 200
    assert client.get("/cameras/nope/frame.jpg").status_code == 404
    r = client.get("/thermal/th_front_narrow.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get("/thermal/rgb_front.png").status_code == 400
    assert client.get("/depth/tof_rear.png").status_code == 200
    s = client.get("/stats").json()
    assert "cameras" in s and s["cameras"]["rgb_front"]["available"]


def test_app_persons_and_ws(omni_client):
    client, mod = omni_client
    mod.svc.step()  # one tick, independent of the background thread timing
    p = client.get("/persons").json()
    assert "t" in p and isinstance(p["persons"], list)
    with client.websocket_connect("/ws/video?cams=rgb_left,th_front_wide") as ws:
        got = set()
        for _ in range(2):
            m = ws.receive_bytes()
            got.add(m[0])
            assert m[1:3] == b"\xff\xd8"
        assert got == {1, 5}
    with client.websocket_connect("/ws/persons") as ws:
        msg = ws.receive_json()
        assert "persons" in msg and "t" in msg
    with client.websocket_connect("/ws/voxels") as ws:
        if mod.svc.voxels is not None:
            assert ws.receive_bytes()[:4] == b"OVX1"


def test_ego_delta():
    from omni.app import ego_delta

    dx, dy, dyaw = ego_delta((1.0, 1.0, np.pi / 2), (1.0, 2.0, np.pi / 2 + 0.1))
    assert dx == pytest.approx(1.0) and dy == pytest.approx(0.0, abs=1e-9) and dyaw == pytest.approx(0.1)
