"""navigation/nav_local.py: DWA-lite goal seeking, obstacle + person avoidance."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "navigation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mapping"))

from nav_local import LocalConfig, plan_local, rollout  # noqa: E402
from social_layer import social_cost_grid  # noqa: E402

RES, W, H = 0.05, 200, 200
META = {"resolution": RES, "origin_x": -5.0, "origin_y": -5.0, "width": W, "height": H}


def grid(blocks=()):
    b = np.zeros((H, W), dtype=bool)
    for (x0, y0, x1, y1) in blocks:
        gx0, gx1 = int((x0 + 5) / RES), int((x1 + 5) / RES)
        gy0, gy1 = int((y0 + 5) / RES), int((y1 + 5) / RES)
        b[gy0:gy1, gx0:gx1] = True
    return dict(META, blocked=b.ravel())


def path_points(vx, vyaw, cfg=None):
    cfg = cfg or LocalConfig()
    xs, ys, _ = rollout(0.0, 0.0, 0.0, np.array([vx]), np.array([vyaw]), cfg.horizon_s, cfg.dt)
    return xs[0], ys[0]


def test_free_space_drives_toward_goal():
    vx, vyaw, info = plan_local((0, 0, 0), (3.0, 0.0), grid(), cfg=LocalConfig(cur_vx=0.4))
    assert info["ok"] and vx > 0.3 and abs(vyaw) < 0.2


def test_turns_toward_goal_on_the_left():
    _, vyaw, info = plan_local((0, 0, 0), (0.0, 3.0), grid())
    assert info["ok"] and vyaw > 0.3


def test_goal_reached():
    vx, vyaw, info = plan_local((1.0, 1.0, 0.0), (1.05, 1.0), grid())
    assert (vx, vyaw) == (0.0, 0.0) and info["reached"]


def test_avoids_obstacle_ahead_closed_loop():
    g = grid([(0.8, -0.3, 1.2, 0.3)])                       # box on the straight line
    blocked = g["blocked"].reshape(H, W)
    x = y = th = vx = vyaw = 0.0
    for _ in range(150):
        vx, vyaw, info = plan_local((x, y, th), (3.0, 0.0), g, cfg=LocalConfig(cur_vx=vx, cur_vyaw=vyaw))
        th += vyaw * 0.1
        x += vx * math.cos(th) * 0.1
        y += vx * math.sin(th) * 0.1
        assert not blocked[int((y + 5) / RES), int((x + 5) / RES)]
        if info.get("reached"):
            break
    assert math.hypot(3.0 - x, y) <= 0.2


def test_boxed_in_returns_zero():
    g = grid([(0.2, -5, 0.6, 5), (-0.6, -5, -0.2, 5), (-5, 0.2, 5, 0.6), (-5, -0.6, 5, -0.2)])
    vx, vyaw, info = plan_local((0, 0, 0), (3.0, 0.0), g, cfg=LocalConfig(min_vx=0.3, cur_vx=0.3))
    assert (vx, vyaw) == (0.0, 0.0) and not info["ok"]


def test_rotation_allowed_inside_inflated_start_cell():
    g = grid([(-0.1, -0.1, 0.1, 0.1)])                      # robot sits in inflation
    _, _, info = plan_local((0, 0, 0), (0.0, 3.0), g)
    assert info["ok"]


def test_keeps_1m_from_predicted_person_positions():
    persons = [{"x": 1.3, "y": 0.0, "vx": -0.5, "vy": 0.0}]  # walking toward us
    cfg = LocalConfig(cur_vx=0.4)
    vx, vyaw, info = plan_local((0, 0, 0), (4.0, 0.0), grid(), persons_world=persons, cfg=cfg)
    xs, ys = path_points(vx, vyaw, cfg)
    t = (np.arange(len(xs)) + 1) * cfg.dt
    d = np.hypot(1.3 - 0.5 * t - xs, -ys)
    assert d.min() >= 1.0 or (vx, vyaw) == (0.0, 0.0)


def test_rejects_everything_if_person_too_close():
    persons = [{"x": 0.5, "y": 0.0}]
    vx, vyaw, info = plan_local((0, 0, 0), (4.0, 0.0), grid(), persons_world=persons)
    assert (vx, vyaw) == (0.0, 0.0) and not info["ok"]


@pytest.mark.parametrize("side,sign", [("right", -1.0), ("left", 1.0)])
def test_passes_person_on_configured_side(side, sign):
    # person standing 2.5 m ahead on the path; goal beyond. "right": robot
    # swerves to its right (vyaw < 0) so the person stays on its left.
    persons = [{"x": 2.5, "y": 0.0}]
    cfg = LocalConfig(cur_vx=0.4, pass_side=side)
    vx, vyaw, info = plan_local((0, 0, 0), (6.0, 0.0), grid(), persons_world=persons, cfg=cfg)
    assert info["ok"] and math.copysign(1.0, vyaw) == sign and abs(vyaw) > 0.05


def test_social_grid_pushes_path_away():
    persons = [{"x": 1.5, "y": 0.4}]
    sg = social_cost_grid(persons, META)
    cfg = LocalConfig(cur_vx=0.4, pass_side=None, w_person=0.0)
    _, vyaw_social, _ = plan_local((0, 0, 0), (4.0, 0.0), grid(), social_grid=sg, cfg=cfg)
    _, vyaw_plain, _ = plan_local((0, 0, 0), (4.0, 0.0), grid(), cfg=cfg)
    assert vyaw_social < vyaw_plain + 1e-9 and vyaw_social < 0.0   # veers right, away from person at +y


def test_dynamic_window_limits_speed_change():
    vx, _, _ = plan_local((0, 0, 0), (5.0, 0.0), grid(), cfg=LocalConfig(cur_vx=0.0, accel=0.4, window_dt=0.5))
    assert vx <= 0.2 + 1e-9
