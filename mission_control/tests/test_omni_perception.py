"""omni perception (Agent B): detectors, tracker, Perception360 -- no GPU/network/redis.

Uses tiny local pinhole / equidistant-fisheye stand-ins so it does not depend on
omni/camera_model.py; one optional test runs against the real models if present.
"""
import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni.detect import MockDetector, UltralyticsDetector  # noqa: E402
from omni.omni_types import Frame  # noqa: E402
from omni.perception360 import Perception360, Perception360Config, project_base  # noqa: E402
from omni.thermal_detect import ThermalPersonDetector, estimate_range_from_height  # noqa: E402
from omni.track import Det3D, MultiCamTracker, _hungarian, linear_assignment  # noqa: E402

GROUND = -0.30


# ------------------------------------------------------------------ stand-in rig

class Pinhole:
    def __init__(self, w, h, hfov_deg):
        self.w, self.h = w, h
        self.f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx, self.cy = w / 2.0, h / 2.0

    def project(self, pc):
        pc = np.asarray(pc, float).reshape(-1, 3)
        z = pc[:, 2]
        zs = np.where(z > 1e-6, z, 1.0)
        uv = np.stack([self.f * pc[:, 0] / zs + self.cx, self.f * pc[:, 1] / zs + self.cy], axis=1)
        return uv, z > 1e-6

    def unproject(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        r = np.stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))], axis=1)
        return r / np.linalg.norm(r, axis=1, keepdims=True)


class Equidistant:
    """Ideal fisheye r = f * theta."""

    def __init__(self, w, h, fov_deg):
        self.w, self.h = w, h
        self.half = math.radians(fov_deg) / 2.0
        self.f = (min(w, h) / 2.0) / self.half
        self.cx, self.cy = w / 2.0, h / 2.0

    def project(self, pc):
        pc = np.asarray(pc, float).reshape(-1, 3)
        r = np.hypot(pc[:, 0], pc[:, 1])
        th = np.arctan2(r, pc[:, 2])
        rs = np.where(r > 1e-9, r, 1.0)
        uv = np.stack([self.cx + self.f * th * pc[:, 0] / rs, self.cy + self.f * th * pc[:, 1] / rs], axis=1)
        return uv, th < self.half

    def unproject(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        dx, dy = uv[:, 0] - self.cx, uv[:, 1] - self.cy
        rd = np.hypot(dx, dy)
        th = rd / self.f
        rs = np.where(rd > 1e-9, rd, 1.0)
        s = np.sin(th)
        return np.stack([s * dx / rs, s * dy / rs, np.cos(th)], axis=1)


def T_cam(xyz, yaw_deg):
    # cam z -> base x, cam x -> base -y, cam y -> base -z, then yaw about base z
    R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ R0
    T[:3, 3] = xyz
    return T


def spec(cam_id, modality, model, xyz, yaw):
    return SimpleNamespace(cam_id=cam_id, modality=modality, width=model.w, height=model.h,
                           model=model, T_base_cam=T_cam(xyz, yaw))


def make_rig(fisheye=False):
    def rgb_model():
        return Equidistant(640, 480, 180) if fisheye else Pinhole(640, 480, 100)

    cams = {
        "rgb_front": spec("rgb_front", "rgb", rgb_model(), (0.30, 0, 0.12), 0),
        "rgb_left": spec("rgb_left", "rgb", rgb_model(), (0, 0.10, 0.12), 90),
        "rgb_right": spec("rgb_right", "rgb", rgb_model(), (0, -0.10, 0.12), -90),
        "rgb_rear": spec("rgb_rear", "rgb", rgb_model(), (-0.30, 0, 0.12), 180),
        "th_front_wide": spec("th_front_wide", "thermal", Pinhole(256, 192, 60), (0.30, 0.03, 0.12), 0),
        "tof_rear": spec("tof_rear", "depth", Pinhole(100, 100, 70), (-0.32, 0, 0.0), 180),
    }
    return SimpleNamespace(cameras=cams)


def person_box(px, py, w=0.5, h=1.7):
    xs = [px - w / 2, px + w / 2]
    ys = [py - w / 2, py + w / 2]
    zs = [GROUND, GROUND + h]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def bbox_of(sp, px, py):
    uv, valid, _ = _proj_all(sp, person_box(px, py))
    if not valid.all():
        return None
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    if x2 < 0 or y2 < 0 or x1 >= sp.width or y1 >= sp.height:
        return None
    return (max(0.0, x1), max(0.0, y1), min(sp.width - 1.0, x2), min(sp.height - 1.0, y2))


def _proj_all(sp, pts):
    Ti = np.linalg.inv(sp.T_base_cam)
    pc = pts @ Ti[:3, :3].T + Ti[:3, 3]
    uv, valid = sp.model.project(pc)
    return uv, valid, pc[:, 2]


def lidar_person(px, py, n=200, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.uniform(0, 2 * math.pi, n)
    z = rng.uniform(GROUND + 0.1, GROUND + 1.7, n)
    pts = np.stack([px + 0.2 * np.cos(a), py + 0.2 * np.sin(a), z], axis=1)
    # Keep only the side facing the robot (what a LiDAR would see).
    return pts[np.hypot(pts[:, 0], pts[:, 1]) <= math.hypot(px, py)]


def lidar_ground(n=500, seed=1):
    rng = np.random.RandomState(seed)
    xy = rng.uniform(-8, 8, (n, 2))
    return np.concatenate([xy, np.full((n, 1), GROUND + 0.01)], axis=1)


def render_rgb(sp, persons):
    img = np.full((sp.height, sp.width, 3), 90, np.uint8)
    for px, py in persons:
        bb = bbox_of(sp, px, py)
        if bb is not None:
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), -1)
    return img


def render_tof(sp, persons, w=0.5, h=1.7):
    """z-depth image of vertical person slabs + floor."""
    vs, us = np.mgrid[0:sp.height, 0:sp.width]
    uv = np.stack([us.ravel() + 0.5, vs.ravel() + 0.5], axis=1)
    rc = sp.model.unproject(uv)
    T = sp.T_base_cam
    d = rc @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (GROUND - o[2]) / d[:, 2]
        best = np.where((s > 0), s, best)
        for px, py in persons:
            # slab facing the robot: plane x = px +- (w/2) toward origin, approximate with x = px
            s2 = (px - o[0]) / d[:, 0]
            hit = o + d * s2[:, None]
            ok = (s2 > 0) & (np.abs(hit[:, 1] - py) <= w / 2) & (hit[:, 2] >= GROUND) & (hit[:, 2] <= GROUND + h)
            best = np.where(ok & (s2 < best), s2, best)
    depth = best * rc[:, 2]
    depth[~np.isfinite(depth) | (depth > 2.5) | (depth < 0.2)] = 0.0
    return depth.reshape(sp.height, sp.width).astype(np.float32)


# ------------------------------------------------------------------ detectors

def test_mock_detector_finds_red_blob_and_injection():
    img = np.zeros((120, 160, 3), np.uint8)
    img[20:80, 50:70] = (0, 0, 255)
    det = MockDetector()
    res = det.detect([img, np.zeros_like(img)])
    assert len(res) == 2 and res[1] == []
    (bb, conf, cls), = res[0]
    assert cls == "person" and bb == (50.0, 20.0, 70.0, 80.0)
    det.set_detections([[((1, 2, 3, 4), 0.5, "person")]])
    assert det.detect([img, img]) == [[((1, 2, 3, 4), 0.5, "person")], []]


def test_ultralytics_detector_is_lazy():
    d = UltralyticsDetector("missing_model.engine")  # no import of ultralytics here
    assert d._model is None and d.is_engine
    assert d.detect([]) == []


def _thermal_scene(w=256, h=192):
    img = np.full((h, w), 21.0, np.float32)
    img += np.random.RandomState(0).normal(0, 0.3, img.shape).astype(np.float32)
    return img


def test_thermal_person_detected_cup_rejected():
    img = _thermal_scene()
    img[40:140, 100:130] = 31.0          # clothed body
    img[40:58, 106:124] = 34.5           # face/head
    img[150:160, 200:210] = 36.0         # small hot cup (square, small)
    img[20:30, 20:30] = 65.0             # kettle: too hot
    dets = ThermalPersonDetector().detect(img)
    assert len(dets) == 1
    (x1, y1, x2, y2), conf = dets[0]
    assert 98 <= x1 <= 101 and 39 <= y1 <= 41 and 129 <= x2 <= 132 and 139 <= y2 <= 142
    assert conf > 0.6


def test_thermal_rejects_wide_warm_blob_and_handles_small_far_person():
    img = _thermal_scene()
    img[150:170, 20:220] = 30.0          # warm radiator: wide, flat
    img[60:80, 180:187] = 33.5           # far person in narrow FOV (7x20 px)
    dets = ThermalPersonDetector().detect(img)
    assert len(dets) == 1
    assert dets[0][0][0] >= 178


def test_thermal_range_from_height():
    assert estimate_range_from_height(100, 500.0) == pytest.approx(8.5)
    assert estimate_range_from_height(0, 500.0) is None


# ------------------------------------------------------------------ assignment / tracker

def test_hungarian_matches_bruteforce():
    import itertools
    rng = np.random.RandomState(3)
    for shape in [(3, 3), (2, 4), (4, 2), (5, 5)]:
        c = rng.rand(*shape)
        pairs = _hungarian(c)
        got = sum(c[r, k] for r, k in pairs)
        n = min(shape)
        best = min(sum(c[r, k] for r, k in zip(rows, cols))
                   for rows in itertools.combinations(range(shape[0]), n)
                   for cols in itertools.permutations(range(shape[1]), n))
        assert got == pytest.approx(best)
    assert linear_assignment(np.array([[5.0, 0.2]]), max_cost=1.0) == [(0, 1)]
    assert linear_assignment(np.array([[5.0]]), max_cost=1.0) == []


def test_tracker_birth_needs_two_hits_and_death_after_max_age():
    tr = MultiCamTracker()
    assert tr.update([Det3D(3, 0, 0.5, 0.9, "rgb_front")], 0.0) == []
    out = tr.update([Det3D(3.02, 0, 0.5, 0.9, "rgb_front")], 0.1)
    assert len(out) == 1 and out[0].gid == 1 and out[0].cams == ["rgb_front"]
    out = tr.update([], 1.0)                      # coasting, still alive
    assert len(out) == 1 and out[0].cams == []
    assert tr.update([], 1.7) == []               # > 1.5 s since last seen
    # single spurious hit never becomes a track
    tr.update([Det3D(-2, 2, 0.5, 0.9, "rgb_left")], 2.0)
    assert tr.update([], 2.6) == []


def test_tracker_gid_stable_when_crossing_front_to_left_view():
    tr = MultiCamTracker()
    gids = set()
    rng = np.random.RandomState(1)
    # person walks on a circle of radius 3 m from bearing 0 deg to 120 deg at 0.5 rad/s
    t = 0.0
    while t < 4.2:
        ang = 0.5 * t
        x, y = 3 * math.cos(ang), 3 * math.sin(ang)
        bearing = math.degrees(ang)
        dets = []
        if bearing < 60:
            dets.append(Det3D(x + rng.normal(0, .05), y + rng.normal(0, .05), 0.5, 0.8, "rgb_front"))
        if bearing > 35:
            dets.append(Det3D(x + rng.normal(0, .05), y + rng.normal(0, .05), 0.5, 0.8, "rgb_left"))
        out = tr.update(dets, t)
        assert len(out) <= 1
        if out:
            gids.add(out[0].gid)
            if 40 < bearing < 58:
                assert out[0].cams == ["rgb_front", "rgb_left"]
        t += 0.1
    assert gids == {1}
    assert math.hypot(out[0].x - x, out[0].y - y) < 0.3


def test_tracker_velocity_estimate():
    tr = MultiCamTracker()
    for i in range(30):
        t = i * 0.1
        out = tr.update([Det3D(2.0 + 1.0 * t, 1.0 - 0.5 * t, 0.5, 0.9, "rgb_front", "rgb", "lidar")], t)
    assert out[0].vx == pytest.approx(1.0, abs=0.1)
    assert out[0].vy == pytest.approx(-0.5, abs=0.1)
    assert out[0].range_src == "lidar"


def test_tracker_ego_motion_compensation_translation_and_rotation():
    tr = MultiCamTracker()
    # static person at world (4, 0); robot drives forward 0.1 m per tick
    rx = 0.0
    for i in range(20):
        t = i * 0.1
        ego = (0.1, 0.0, 0.0) if i else None
        if i:
            rx += 0.1
        out = tr.update([Det3D(4.0 - rx, 0.0, 0.5, 0.9, "rgb_front")], t, ego_delta=ego)
    p = out[0]
    assert p.x == pytest.approx(4.0 - rx, abs=0.05)
    assert abs(p.vx) < 0.1 and abs(p.vy) < 0.1     # world velocity ~ 0
    # robot turns +90 deg in place: person must appear on the right (-y) without detections
    out = tr.update([], 2.0, ego_delta=(0.0, 0.0, math.pi / 2))
    assert out[0].gid == p.gid
    assert out[0].x == pytest.approx(0.0, abs=0.1)
    assert out[0].y == pytest.approx(-(4.0 - rx), abs=0.1)
    out = tr.update([Det3D(0.0, -(4.0 - rx), 0.5, 0.9, "rgb_right")], 2.1)
    assert len(out) == 1 and out[0].gid == p.gid


def test_tracker_fuses_rgb_and_thermal_into_one_track():
    tr = MultiCamTracker()
    for i in range(3):
        out = tr.update([Det3D(5.0, 0.3, 0.5, 0.7, "rgb_front", "rgb", "ground_plane"),
                         Det3D(5.3, 0.25, 0.5, 0.6, "th_front_wide", "thermal", "thermal_size")], i * 0.1)
    assert len(out) == 1
    assert out[0].modality == ["rgb", "thermal"]
    assert out[0].cams == ["rgb_front", "th_front_wide"]
    assert out[0].conf > 0.8
    assert out[0].range_src == "ground_plane"


def test_tracker_two_people_same_camera_not_fused():
    tr = MultiCamTracker()
    for i in range(3):
        out = tr.update([Det3D(3.0, 0.0, 0.5, 0.9, "rgb_front"), Det3D(3.0, 0.5, 0.5, 0.9, "rgb_front")], i * 0.1)
    assert len(out) == 2 and {p.gid for p in out} == {1, 2}
    d = out[0].to_dict()
    assert "range_m" in d and d["gid"] in (1, 2)


# ------------------------------------------------------------------ range priority

def _det(cam, bbox, mod="rgb"):
    from omni.omni_types import Detection
    return Detection(cam, bbox, 0.9, "person", mod)


def test_range_priority_lidar_then_ground_front():
    rig = make_rig()
    p = Perception360(rig, MockDetector())
    sp = rig.cameras["rgb_front"]
    bb = bbox_of(sp, 4.0, 0.0)
    lidar = p._filter_ground(np.concatenate([lidar_person(4.0, 0.0), lidar_ground()]))
    d3 = p.localize(_det("rgb_front", bb), lidar, None)
    assert d3.range_src == "lidar"
    assert d3.x == pytest.approx(3.8, abs=0.15) and abs(d3.y) < 0.1
    # ground points only -> filtered away -> ground plane
    d3 = p.localize(_det("rgb_front", bb), p._filter_ground(lidar_ground()), None)
    assert d3.range_src == "ground_plane"
    assert d3.x == pytest.approx(3.75, abs=0.3)


def test_range_priority_tof_rear_beats_ground_and_lidar_beats_tof():
    rig = make_rig()
    p = Perception360(rig, MockDetector())
    sp = rig.cameras["rgb_rear"]
    bb = bbox_of(sp, -1.6, 0.0)
    tof = p.tof_points_base(rig.cameras["tof_rear"], render_tof(rig.cameras["tof_rear"], [(-1.6, 0.0)]))
    assert tof is not None and len(tof) > 50
    d3 = p.localize(_det("rgb_rear", bb), None, tof)
    assert d3.range_src == "tof"
    assert d3.x == pytest.approx(-1.6, abs=0.12) and abs(d3.y) < 0.1
    d3 = p.localize(_det("rgb_rear", bb), p._filter_ground(lidar_person(-1.6, 0.0)), tof)
    assert d3.range_src == "lidar"
    d3 = p.localize(_det("rgb_rear", bb), None, None)
    assert d3.range_src == "ground_plane" and d3.x == pytest.approx(-1.6, abs=0.3)


def test_thermal_range_uses_size_without_lidar():
    rig = make_rig()
    p = Perception360(rig, MockDetector())
    sp = rig.cameras["th_front_wide"]
    bb = bbox_of(sp, 6.0, 0.03)
    d3 = p.localize(_det("th_front_wide", bb, "thermal"), None, None)
    assert d3.range_src == "thermal_size"
    assert d3.x == pytest.approx(6.0, rel=0.1)


def test_tof_blob_detector_finds_person_behind():
    rig = make_rig()
    p = Perception360(rig, MockDetector())
    tof = p.tof_points_base(rig.cameras["tof_rear"], render_tof(rig.cameras["tof_rear"], [(-1.2, 0.2)]))
    blobs = p.tof_blobs(tof)
    assert len(blobs) == 1
    assert blobs[0].modality == "depth" and blobs[0].range_src == "tof"
    assert blobs[0].x == pytest.approx(-1.2, abs=0.1) and blobs[0].y == pytest.approx(0.2, abs=0.15)
    empty = p.tof_points_base(rig.cameras["tof_rear"], render_tof(rig.cameras["tof_rear"], []))
    assert p.tof_blobs(empty) == []


# ------------------------------------------------------------------ end-to-end

def _frames(rig, persons, t, tof=True):
    fr = {}
    for cid, sp in rig.cameras.items():
        if sp.modality == "rgb":
            fr[cid] = Frame(cid, "rgb", t, 0, render_rgb(sp, persons))
        elif sp.modality == "depth" and tof:
            fr[cid] = Frame(cid, "depth", t, 0, render_tof(sp, persons))
        elif sp.modality == "thermal":
            img = np.full((sp.height, sp.width), 20.0, np.float32)
            for px, py in persons:
                bb = bbox_of(sp, px, py)
                if bb is not None:
                    img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = 33.5
            fr[cid] = Frame(cid, "thermal", t, 0, img)
    return fr


@pytest.mark.parametrize("fisheye", [False, True])
def test_perception360_end_to_end(fisheye):
    rig = make_rig(fisheye)
    det = MockDetector()
    p = Perception360(rig, det, ThermalPersonDetector())
    persons = [(4.0, 0.5), (-1.5, 0.0), (0.5, 3.0)]
    lidar = np.concatenate([lidar_person(4.0, 0.5), lidar_ground()])
    out = []
    for i in range(3):
        out = p.process(_frames(rig, persons, i * 0.1), lidar, i * 0.1)
    assert det.last_batch_size == 4 and det.calls == 3       # one batch call per tick
    assert len(out) == 3, [x.to_dict() for x in out]
    by_pos = {}
    for tr in out:
        k = min(range(3), key=lambda j: math.hypot(tr.x - persons[j][0], tr.y - persons[j][1]))
        by_pos[k] = tr
        assert math.hypot(tr.x - persons[k][0], tr.y - persons[k][1]) < 0.5
    assert by_pos[0].range_src == "lidar" and "thermal" in by_pos[0].modality
    assert by_pos[1].range_src == "tof" and "depth" in by_pos[1].modality
    assert by_pos[2].range_src == "ground_plane" and by_pos[2].cams == ["rgb_left"]


def test_perception360_round_robin_keeps_heading_cam():
    rig = make_rig()
    det = MockDetector()
    p = Perception360(rig, det, cfg={"max_cams_per_tick": 2, "heading_cam": "rgb_front"})
    seen = []
    for i in range(3):
        p.process(_frames(rig, [(3.0, 0.0)], i * 0.1, tof=False), None, i * 0.1)
        seen.append(p.last_stats["rgb_cams"])
        assert "rgb_front" in seen[-1] and len(seen[-1]) == 2
    assert {s[1] for s in seen} == {"rgb_left", "rgb_rear", "rgb_right"}
    assert isinstance(p.cfg, Perception360Config)


def test_perception360_uses_rectifier_to_original():
    rig = make_rig()

    class HalfRect:
        """Fake rectifier: 2x downscale; to_original scales back."""

        def __init__(self):
            self.n = 0

        def apply(self, img):
            self.n += 1
            return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

        def to_original(self, bbox):
            return tuple(2.0 * v for v in bbox)

    r = HalfRect()
    p = Perception360(rig, MockDetector(min_area=5), rectifiers={"rgb_front": r})
    for i in range(2):
        out = p.process(_frames(rig, [(4.0, 0.0)], i * 0.1, tof=False), None, i * 0.1)
    assert r.n == 2 and len(out) == 1
    assert out[0].x == pytest.approx(3.75, abs=0.3)  # ground plane hits the near face


def test_perception360_with_real_camera_model_if_available():
    cm = pytest.importorskip("omni.camera_model")
    rig = make_rig()
    K = [[200.0, 0, 320], [0, 200.0, 240], [0, 0, 1]]
    model = cm.FisheyeModel(K, [0, 0, 0, 0], 640, 480)
    model.w, model.h = 640, 480
    rig.cameras["rgb_front"].model = model
    p = Perception360(rig, MockDetector())
    for i in range(2):
        out = p.process(_frames(rig, [(3.5, -1.0)], i * 0.1, tof=False),
                        np.concatenate([lidar_person(3.5, -1.0), lidar_ground()]), i * 0.1)
    assert len(out) == 1 and out[0].range_src == "lidar"
    assert math.hypot(out[0].x - 3.5, out[0].y + 1.0) < 0.4
