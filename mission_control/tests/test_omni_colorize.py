"""Tests for omni.colorize (occlusion z-buffer, best camera, thermal, perf, helpers)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from omni.colorize import base_to_world, colorize, deskew  # noqa: E402
from omni.omni_types import Frame  # noqa: E402

# base->cam rotation for a camera looking along base +x (cam z = base x,
# cam x = -base y, cam y = -base z).
R_FWD = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]).T  # rows before .T = cam x,y,z axes in base


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Spec(object):
    """Stand-in pinhole CameraSpec (duck-typed like omni.rig.CameraSpec)."""

    def __init__(self, cam_id, modality, yaw=0.0, w=320, h=240, hfov_deg=90.0, pos=(0, 0, 0)):
        self.cam_id = cam_id
        self.modality = modality
        self.width = w
        self.height = h
        f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.K = np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = _yaw(yaw) @ R_FWD
        T[:3, 3] = pos
        self.T_base_cam = T
        self.model = None


def pinhole_project(spec, pts_base):
    T = np.linalg.inv(spec.T_base_cam)
    pc = pts_base @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = spec.K[0, 0] * pc[:, 0] / zs + spec.K[0, 2]
    v = spec.K[1, 1] * pc[:, 1] / zs + spec.K[1, 2]
    return np.stack([u, v], axis=1), z > 1e-6, z


class Rig(object):
    def __init__(self, specs):
        self.cameras = {s.cam_id: s for s in specs}


def _img(spec, bgr):
    im = np.zeros((spec.height, spec.width, 3), np.uint8)
    im[:] = bgr
    return im


def test_basic_color_and_bgr_to_rgb():
    s = Spec("rgb_front", "rgb")
    rig = Rig([s])
    frames = {"rgb_front": Frame("rgb_front", "rgb", 0.0, 0, _img(s, (255, 0, 10)))}
    pts = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])  # second is behind the camera
    p, rgb, temp = colorize(pts, frames, rig, project_fn=pinhole_project)
    assert p.shape == (1, 3)
    assert tuple(rgb[0]) == (10, 0, 255)
    assert np.isnan(temp[0])
    p, rgb, temp = colorize(pts, frames, rig, project_fn=pinhole_project, keep_uncolored=True)
    assert p.shape == (2, 3) and tuple(rgb[1]) == (0, 0, 0)


def test_occluded_point_does_not_get_wall_color():
    # Front camera sees a red wall at x=2; a far point at x=6 is behind it.
    # The left-rear camera (at y=+1.5 looking at the far point side-on) sees it in green.
    front = Spec("rgb_front", "rgb")
    side = Spec("rgb_left", "rgb", yaw=-np.pi / 2, pos=(6.0, 3.0, 0.0))
    rig = Rig([front, side])
    wy, wz = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
    wall = np.stack([np.full(wy.size, 2.0), wy.ravel(), wz.ravel()], axis=1)
    far = np.array([[6.0, 0.0, 0.0]])
    pts = np.vstack([wall, far])

    # Front camera image: red where the wall is (everything).
    frames = {"rgb_front": Frame("rgb_front", "rgb", 0, 0, _img(front, (0, 0, 255)))}
    p, rgb, _ = colorize(pts, frames, rig, project_fn=pinhole_project, keep_uncolored=True)
    assert tuple(rgb[-1]) == (0, 0, 0), "occluded point must not get the wall colour"
    assert (rgb[:-1, 0] == 255).mean() > 0.95

    # With the side camera too, the far point is coloured by it (green).
    frames["rgb_left"] = Frame("rgb_left", "rgb", 0, 0, _img(side, (0, 255, 0)))
    p, rgb, _ = colorize(pts, frames, rig, project_fn=pinhole_project, keep_uncolored=True)
    assert tuple(rgb[-1]) == (0, 255, 0)


def test_best_camera_is_most_central():
    # Two cameras at the origin yawed +-30 deg; point at yaw +25 deg is central
    # for the +30 camera and near the edge of the -30 camera.
    a = Spec("rgb_a", "rgb", yaw=np.radians(30), hfov_deg=130)
    b = Spec("rgb_b", "rgb", yaw=np.radians(-30), hfov_deg=130)
    rig = Rig([a, b])
    frames = {"rgb_a": Frame("rgb_a", "rgb", 0, 0, _img(a, (0, 0, 200))),
              "rgb_b": Frame("rgb_b", "rgb", 0, 0, _img(b, (200, 0, 0)))}
    ang = np.radians([25.0, -25.0])
    pts = np.stack([4 * np.cos(ang), 4 * np.sin(ang), np.zeros(2)], axis=1)
    _, rgb, _ = colorize(pts, frames, rig, project_fn=pinhole_project)
    assert tuple(rgb[0]) == (200, 0, 0)  # from camera a (R=200)
    assert tuple(rgb[1]) == (0, 0, 200)  # from camera b (B=200)


def test_thermal_sampling_low_res_and_nan():
    s = Spec("rgb_front", "rgb")
    th = Spec("th_front_wide", "thermal", w=256, h=192, hfov_deg=60)
    rig = Rig([s, th])
    timg = np.full((96, 128), 20.0, np.float32)  # delivered at half resolution
    timg[:, 64:] = 36.5  # right image half (base -y side) is hot
    timg[0:10, 0:10] = np.nan
    frames = {"rgb_front": Frame("rgb_front", "rgb", 0, 0, _img(s, (50, 50, 50))),
              "th_front_wide": Frame("th_front_wide", "thermal", 0, 0, timg)}
    pts = np.array([[4.0, 1.0, 0.0], [4.0, -1.0, 0.0], [4.0, 3.9, 0.0]])
    _, rgb, temp = colorize(pts, frames, rig, project_fn=pinhole_project)
    assert rgb.shape == (3, 3)
    assert abs(temp[0] - 20.0) < 1e-6
    assert abs(temp[1] - 36.5) < 1e-6
    assert np.isnan(temp[2])  # outside the thermal FOV


def test_perf_30k_points_4_cams():
    specs = [Spec("rgb_%d" % i, "rgb", yaw=i * np.pi / 2, w=1280, h=720, hfov_deg=120)
             for i in range(4)]
    rig = Rig(specs)
    rng = np.random.default_rng(0)
    frames = {s.cam_id: Frame(s.cam_id, "rgb", 0, 0,
                              rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
              for s in specs}
    ang = rng.uniform(-np.pi, np.pi, 30000)
    r = rng.uniform(1, 15, 30000)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang), rng.uniform(-1, 2, 30000)], axis=1)
    colorize(pts, frames, rig, project_fn=pinhole_project)  # warm-up
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        p, rgb, _ = colorize(pts, frames, rig, project_fn=pinhole_project)
        best = min(best, time.perf_counter() - t0)
    print("\ncolorize 30k pts x 4 cams (incl. stand-in projection): %.2f ms, %d coloured"
          % (best * 1e3, p.shape[0]))
    assert p.shape[0] > 15000
    assert best < 0.2


def test_base_to_world_and_deskew():
    pts = np.array([[1.0, 0.0, 0.5]])
    w = base_to_world(pts, (2.0, 1.0, np.pi / 2))
    assert np.allclose(w, [[2.0, 2.0, 0.5]], atol=1e-6)

    # Robot moves +x at 1 m/s; a static world point at x=5 measured at t=0 is at
    # base x=5, at t=1 at base x=4. Deskewed to t_ref=1 both must read 4.
    def pose_at(t):
        return (t, 0.0, 0.0)

    meas = np.array([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    out = deskew(meas, np.array([0.0, 1.0]), pose_at)
    assert np.allclose(out[:, 0], 4.0, atol=1e-5)

    # Pure rotation (yaw across +-pi wrap): world point on +x axis.
    def pose_rot(t):
        return (0.0, 0.0, np.pi - 0.1 + 0.2 * t)  # crosses pi

    yaw0, yaw1 = np.pi - 0.1, np.pi + 0.1
    world = np.array([10.0, 0.0])
    meas = []
    for yaw in (yaw0, yaw1):
        c, s = np.cos(-yaw), np.sin(-yaw)
        meas.append([c * world[0] - s * world[1], s * world[0] + c * world[1], 0.0])
    out = deskew(np.array(meas), np.array([0.0, 1.0]), pose_rot)
    assert np.allclose(out[0], out[1], atol=1e-4)


def test_zbuffer_sort_fallback_matches_ufunc_at():
    from omni.colorize import _zbuffer_visible

    rng = np.random.default_rng(3)
    cell = rng.integers(0, 50, 5000).astype(np.int32)
    d = rng.uniform(1, 10, 5000).astype(np.float32)
    tol = np.full(5000, 0.1, np.float32)
    a = _zbuffer_visible(cell, d, 50, tol, use_ufunc_at=True)
    b = _zbuffer_visible(cell, d, 50, tol, use_ufunc_at=False)
    assert np.array_equal(a, b) and 0 < a.sum() < 5000
