"""perception/geometry3d.py -- pure-math regression tests (no camera, no YOLO)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))

from geometry3d import Extrinsics, Intrinsics, TrackSmoother, robust_bbox_depth  # noqa: E402

INTR = Intrinsics(640, 480, 600.0, 600.0, 320.0, 240.0)


def test_deproject_centre_is_on_axis():
    assert INTR.deproject(320, 240, 2.0) == (0.0, 0.0, 2.0)


def test_deproject_right_pixel_is_positive_x():
    x, y, z = INTR.deproject(620, 240, 2.0)
    assert x == pytest.approx(1.0) and y == 0.0


def test_person_straight_ahead_maps_to_base_forward():
    ext = Extrinsics(tx=0.3, ty=0.0, tz=0.1)
    x, y, z = ext.optical_to_base((0.0, 0.0, 2.0))
    assert (x, y, z) == pytest.approx((2.3, 0.0, 0.1))


def test_optical_right_is_base_right():
    # optical +x = right of image; base +y = left, so y must be negative
    _, y, _ = Extrinsics(tx=0, tz=0).optical_to_base((1.0, 0.0, 2.0))
    assert y == pytest.approx(-1.0)


def test_positive_pitch_tilts_down():
    x, _, z = Extrinsics(tx=0, tz=0, pitch_deg=30).optical_to_base((0.0, 0.0, 2.0))
    assert z == pytest.approx(-2.0 * math.sin(math.radians(30)))
    assert x == pytest.approx(2.0 * math.cos(math.radians(30)))


def test_positive_yaw_turns_left():
    _, y, _ = Extrinsics(tx=0, tz=0, yaw_deg=90).optical_to_base((0.0, 0.0, 2.0))
    assert y == pytest.approx(2.0)


def test_robust_depth_prefers_person_over_background():
    depth = np.full((480, 640), 6000, np.uint16)          # wall at 6 m
    depth[100:400, 250:390] = 2000                         # person torso at 2 m
    z, u, v, ratio = robust_bbox_depth(depth, (200, 50, 440, 470))
    assert z == pytest.approx(2.0)
    assert 250 <= u <= 390 and ratio > 0.9


def test_robust_depth_all_invalid_returns_none():
    z, *_ = robust_bbox_depth(np.zeros((480, 640), np.uint16), (100, 100, 300, 400))
    assert z is None


def test_robust_depth_bbox_at_image_edge_does_not_crash():
    depth = np.full((480, 640), 1500, np.uint16)
    z, *_ = robust_bbox_depth(depth, (600, 400, 700, 600))
    assert z == pytest.approx(1.5)


def test_smoother_rejects_single_frame_jump():
    s = TrackSmoother(alpha=1.0)
    for i in range(3):
        s.update(1, 2.0, 0.0, 0.0, t=i * 0.1)
    st = s.update(1, 6.0, 0.0, 0.0, t=0.4)                # bbox caught the wall
    assert st.x == pytest.approx(2.0)


def test_smoother_velocity_sign():
    s = TrackSmoother(alpha=1.0, vel_alpha=1.0)
    s.update(1, 2.0, 0.0, 0.0, t=0.0)
    st = s.update(1, 2.1, 0.0, 0.0, t=0.1)                 # walking away 1 m/s
    assert st.vx == pytest.approx(1.0)


def test_smoother_prunes_stale_tracks():
    s = TrackSmoother(max_age_s=1.0)
    s.update(1, 1.0, 0.0, 0.0, t=0.0)
    s.update(2, 1.0, 0.0, 0.0, t=1.5)
    assert s.prune(t=2.0) == [1]
    assert s.get(2) is not None
