"""mission/planner.py: A* + social/no-go costs, smoothing, speed profile, jumps, follower."""
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "navigation"))
sys.path.insert(0, os.path.join(HERE, "..", "mapping"))
sys.path.insert(0, os.path.join(HERE, "..", "mission"))

import astar  # noqa: E402
import planner as P  # noqa: E402
from social_layer import social_cost_grid  # noqa: E402

RES = 0.05


def meta(w=100, h=100, ox=0.0, oy=0.0):
    return {"resolution": RES, "origin_x": ox, "origin_y": oy, "width": w, "height": h}


def free(w=100, h=100):
    return np.zeros((h, w), dtype=np.int16)


def box(f, x0, y0, x1, y1, val=100):
    """Fill world rect [x0,x1)x[y0,y1) (origin 0) with val."""
    f[int(round(y0 / RES)):int(round(y1 / RES)), int(round(x0 / RES)):int(round(x1 / RES))] = val
    return f


def path_free(pg, pts):
    for a, b in zip(pts[:-1], pts[1:]):
        if not pg.seg_free(a, b):
            return False
    return True


def min_dist(pts, q):
    a = np.asarray(pts)
    return float(np.hypot(a[:, 0] - q[0], a[:, 1] - q[1]).min())


# ------------------------------------------------------------------ grid

def test_blocked_grid_matches_astar_build_blocked_grid():
    rng = np.random.default_rng(1)
    w, h = 60, 50
    floor = rng.choice([-1, 0, 100], size=(h, w), p=[0.1, 0.87, 0.03]).astype(int)
    walls = rng.integers(0, 256, size=(h, w))
    for smin, smax in ((None, None), (150, 200)):
        cfg = P.PlannerConfig(robot_radius_m=0.15, stair_wall_min=smin, stair_wall_max=smax)
        pg = P.PlanGrid(meta(w, h), floor.ravel(), walls.ravel(), cfg=cfg)
        ref = astar.build_blocked_grid(floor.ravel().tolist(), walls.ravel().tolist(), w, h, 3, smin, smax)
        assert pg.blocked.ravel().tolist() == ref


def test_weighted_astar_matches_reference_cost_without_weights():
    rng = np.random.default_rng(2)
    w, h = 40, 40
    blk = rng.random((h, w)) < 0.2
    blk[0, 0] = blk[-1, -1] = False
    cells, _, cost = P.weighted_astar(blk, None, (0, 0), (w - 1, h - 1))
    ref = astar.astar(w, h, blk.ravel().tolist(), (0, 0), (w - 1, h - 1))
    assert (cells is None) == (ref is None)
    if ref is not None:
        ref_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ref[:-1], ref[1:]))
        assert cost == pytest.approx(ref_len, rel=1e-6)


# ------------------------------------------------------------- plan_path

def test_straight_open_map():
    p = P.plan_path(meta(), free().ravel(), None, (0.5, 0.5), (4.5, 0.5))
    assert p.ok and p.reason == "ok"
    assert p.points[0] == pytest.approx([0.5, 0.5]) and p.points[-1] == pytest.approx([4.5, 0.5])
    assert p.length_m == pytest.approx(4.0, abs=0.02)
    assert max(abs(y - 0.5) for _, y in p.points) < 1e-6
    d = np.hypot(*np.diff(np.asarray(p.points), axis=0).T)
    assert d.max() <= 0.1 + 1e-6                              # resampled at ds


def test_corner_wall_path_is_collision_free():
    f = box(free(), 2.0, 0.0, 2.2, 3.5)                        # wall with a gap at the top
    p = P.plan_path(meta(), f.ravel(), None, (1.0, 1.0), (3.5, 1.0))
    assert p.ok
    pg = P.PlanGrid(meta(), f.ravel(), None)
    assert path_free(pg, p.points)
    assert max(y for _, y in p.points) > 3.5                   # went over the wall end
    assert p.length_m > 2.5


def test_u_shaped_trap():
    f = free()
    box(f, 1.5, 1.5, 3.5, 1.7)                                 # bottom
    box(f, 1.5, 1.5, 1.7, 3.5)                                 # left
    box(f, 3.3, 1.5, 3.5, 3.5)                                 # right  (open at the top)
    p = P.plan_path(meta(), f.ravel(), None, (2.5, 2.5), (2.5, 0.5))
    assert p.ok
    pg = P.PlanGrid(meta(), f.ravel(), None)
    assert path_free(pg, p.points)
    assert max(y for _, y in p.points) > 3.5                   # had to leave through the top
    assert p.length_m > 4.0


def test_goal_inside_obstacle_is_snapped():
    f = box(free(), 2.0, 2.0, 2.4, 2.4)
    p = P.plan_path(meta(), f.ravel(), None, (0.5, 0.5), (2.2, 2.2))
    assert p.ok and any("snapped" in w for w in p.warnings)
    pg = P.PlanGrid(meta(), f.ravel(), None)
    gx, gy = astar.world_to_grid(p.points[-1][0], p.points[-1][1], 0, 0, RES)
    assert not pg.blocked[gy, gx]
    assert math.hypot(p.points[-1][0] - 2.2, p.points[-1][1] - 2.2) < 0.6


def test_goal_deep_inside_big_obstacle_fails():
    f = box(free(), 1.0, 1.0, 4.5, 4.5)
    p = P.plan_path(meta(), f.ravel(), None, (0.3, 0.3), (2.75, 2.75), cfg=P.PlannerConfig(snap_radius_m=0.5))
    assert not p.ok and "goal blocked" in p.reason


def test_no_path():
    f = box(free(), 2.4, 0.0, 2.6, 5.0)                        # wall across the whole map
    p = P.plan_path(meta(), f.ravel(), None, (1.0, 2.5), (4.0, 2.5))
    assert not p.ok and p.reason == "no path" and p.points == []
    assert P.plan_path(meta(), f.ravel(), None, (1, 1), (9, 1)).reason == "goal outside map"


def test_social_cost_bends_path_away_from_person():
    m = meta()
    person = {"x": 2.5, "y": 2.5}
    soc = social_cost_grid([person], m)
    plain = P.plan_path(m, free().ravel(), None, (0.3, 2.5), (4.7, 2.5))
    social = P.plan_path(m, free().ravel(), None, (0.3, 2.5), (4.7, 2.5), social=soc)
    assert plain.ok and social.ok
    assert min_dist(plain.points, (2.5, 2.5)) < 0.05
    assert min_dist(social.points, (2.5, 2.5)) > 1.0
    assert social.length_m > plain.length_m


def test_no_go_mask_is_hard_block():
    m = meta()
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:80, 45:55] = True                                   # band x 2.25..2.75, gap above y=4
    p = P.plan_path(m, free().ravel(), None, (1.0, 1.0), (4.0, 1.0), no_go_mask=mask.ravel())
    assert p.ok
    a = np.asarray(p.points)
    gx = np.floor(a[:, 0] / RES).astype(int)
    gy = np.floor(a[:, 1] / RES).astype(int)
    assert not mask[gy, gx].any()
    assert a[:, 1].max() >= 4.0
    full = np.zeros((100, 100), dtype=bool)
    full[:, 45:55] = True
    assert not P.plan_path(m, free().ravel(), None, (1, 1), (4, 1), no_go_mask=full.ravel()).ok


def test_start_inside_inflation_still_plans():
    f = box(free(), 1.0, 0.0, 1.1, 5.0)
    p = P.plan_path(meta(), f.ravel(), None, (1.2, 2.5), (4.0, 2.5))
    assert p.ok and p.points[0] == pytest.approx([1.2, 2.5])
    assert any("start" in w for w in p.warnings)


def test_coarse_and_full_resolution_agree_and_narrow_gap_falls_back():
    w = h = 320                                                 # >= coarse_min_cells
    f = np.zeros((h, w), dtype=np.int16)
    f[:, 150:160] = 100
    f[200:211, 150:160] = 0                                     # 11-cell door: fits fine (r=5), not coarse
    m = meta(w, h)
    p = P.plan_path(m, f.ravel(), None, (2.0, 2.0), (14.0, 2.0))
    q = P.plan_path(m, f.ravel(), None, (2.0, 2.0), (14.0, 2.0), cfg=P.PlannerConfig(coarse_factor=1))
    assert p.ok and q.ok
    pg = P.PlanGrid(m, f.ravel(), None)
    assert path_free(pg, p.points)
    assert p.length_m == pytest.approx(q.length_m, rel=0.05)


def test_smoothing_never_collides_random_maps():
    rng = np.random.default_rng(7)
    m = meta()
    n_ok = 0
    for _ in range(12):
        f = free()
        for _ in range(6):
            x, y = rng.uniform(0.5, 4.0, 2)
            box(f, x, y, x + rng.uniform(0.1, 1.0), y + rng.uniform(0.1, 1.0))
        p = P.plan_path(m, f.ravel(), None, (0.2, 0.2), (4.8, 4.8))
        if p.ok:
            n_ok += 1
            assert path_free(P.PlanGrid(m, f.ravel(), None), p.points)
    assert n_ok >= 6


def test_smooth_path_collision_fallback_and_resampling():
    f = box(free(), 2.0, 2.0, 3.0, 3.0)
    pg = P.PlanGrid(meta(), f.ravel(), None, cfg=P.PlannerConfig(robot_radius_m=0.0))
    pts = [[1.0, 1.0], [1.95, 1.95], [1.95, 4.0], [4.0, 4.0]]   # hugs the box corner
    safe = P.smooth_path(pts, ds=0.1, grid=pg)
    assert path_free(pg, safe)
    d = np.hypot(*np.diff(np.asarray(safe), axis=0).T)
    assert d.max() <= 0.1 + 1e-6 and d.min() > 0.05
    assert safe[0] == pytest.approx(pts[0]) and safe[-1] == pytest.approx(pts[-1])


# --------------------------------------------------------- speed profile

def _accel_ok(pts, v, amax):
    s = np.hypot(*np.diff(np.asarray(pts), axis=0).T)
    v = np.asarray(v)
    dv2 = np.abs(v[1:] ** 2 - v[:-1] ** 2)
    return bool((dv2 <= 2 * amax * s + 1e-6).all())


def test_speed_profile_straight_limits_and_goal_decel():
    pts = [[x, 0.0] for x in np.arange(0, 10.01, 0.1)]
    v = P.speed_profile(pts, "normal", cfg=P.PlannerConfig(fine_end=False))
    assert isinstance(v, list) and len(v) == len(pts)
    assert max(v) == pytest.approx(0.6) and v[0] == 0.0 and v[-1] == 0.0
    assert _accel_ok(pts, v, 0.6)
    assert v[len(v) // 2] == pytest.approx(0.6)
    # decel near the goal: v <= sqrt(2 a d)
    for (x, _), vi in zip(pts, v):
        assert vi <= math.sqrt(2 * 0.6 * (10.0 - x)) + 1e-6


def test_speed_profile_curvature_cap():
    r = 1.0
    th = np.linspace(0, 3 * math.pi, 400)
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1).tolist()
    v = P.speed_profile(pts, "sprint", cfg=P.PlannerConfig(a_lat=0.5))
    vlim = math.sqrt(0.5 / (1.0 / r))
    assert max(v) <= vlim + 1e-3
    assert max(v) > 0.9 * vlim
    assert _accel_ok(pts, v, 1.0)


def test_fine_end_uses_precise_last_half_metre():
    pts = [[x, 0.0] for x in np.arange(0, 10.01, 0.1)]
    v = P.speed_profile(pts, "sprint")
    for (x, _), vi in zip(pts, v):
        if x >= 9.5:
            assert vi <= P.PROFILES["precise"]["vmax"] + 1e-9


def test_sprint_downgraded_with_person_in_corridor():
    pts = [[x, 0.0] for x in np.arange(0, 20.01, 0.1)]
    clear = P.speed_profile(pts, "sprint", persons_world=[{"x": 10.0, "y": 5.0}])
    assert clear.profile == "sprint" and max(clear) > 1.4 and not clear.warnings
    v = P.speed_profile(pts, "sprint", persons_world=[{"x": 15.0, "y": 1.0}])
    assert v.profile == "normal" and any("downgraded" in w for w in v.warnings)
    assert max(v) <= 0.6 + 1e-9
    # within 3.5 m of the person -> stealth cap; far before it -> normal
    for (x, y), vi in zip(pts, v):
        if math.hypot(x - 15.0, y - 1.0) <= 3.5:
            assert vi <= 0.25 + 1e-9
    assert v[50] == pytest.approx(0.6)
    assert _accel_ok(pts, v, 0.6)


def test_unknown_profile_falls_back_to_normal():
    v = P.speed_profile([[0, 0], [1, 0]], "warp")
    assert v.profile == "normal" and v.warnings


def test_eta():
    pts = [[0, 0], [1, 0], [2, 0]]
    assert P.eta(pts, 0.5) == pytest.approx(4.0)
    assert P.eta(pts, [0.5, 0.5, 0.5]) == pytest.approx(4.0)
    v = P.speed_profile([[x, 0.0] for x in np.arange(0, 5.01, 0.1)], "normal")
    t = P.eta([[x, 0.0] for x in np.arange(0, 5.01, 0.1)], v)
    assert 5.0 / 0.6 < t < 30.0


def test_profiles_contract():
    assert P.PROFILES["stealth"] == {"vmax": 0.25, "amax": 0.3}
    assert P.PROFILES["precise"] == {"vmax": 0.3, "amax": 0.4}
    assert P.PROFILES["normal"] == {"vmax": 0.6, "amax": 0.6}
    assert P.PROFILES["sprint"]["vmax"] == pytest.approx(float(os.environ.get("SPRINT_VMAX", "1.5")))
    assert P.JUMP_LEN_M == pytest.approx(float(os.environ.get("JUMP_LEN_M", "0.6")))


# ------------------------------------------------------------------ jumps

def test_jumps_open_floor():
    f = free()
    j = P.plan_jumps(meta(), f.ravel(), None, (1.0, 1.0), (3.4, 1.0), jump_len_m=0.6)
    assert j.ok and len(j.landings) == 4
    assert j.landings[-1] == pytest.approx([3.4, 1.0], abs=1e-6)
    assert all(h == pytest.approx(0.0) for h in j.headings)
    prev = (1.0, 1.0)
    for x, y in j.landings:
        assert math.hypot(x - prev[0], y - prev[1]) == pytest.approx(0.6)
        prev = (x, y)


@pytest.mark.parametrize("val,walls,why", [(-1, 0, "unknown"), (100, 0, "obstacle"),
                                           (0, 170, "stair"), (0, 90, "not flat")])
def test_jump_landing_rejected(val, walls, why):
    f = free()
    wl = np.zeros_like(f)
    # cell right at the 2nd landing (x = 2.2)
    f[20, 44] = val
    wl[20, 44] = walls
    j = P.plan_jumps(meta(), f.ravel(), wl.ravel(), (1.0, 1.0), (3.4, 1.0), jump_len_m=0.6)
    assert not j.ok and j.rejected and j.rejected[0][0] == 1
    assert why in j.rejected[0][1] and "landing 2" in j.reason


def test_jump_clearance_and_max_jumps():
    f = free()
    f[20, 44 + 6] = 100                                        # 0.3 m from landing 2 -> inside 0.35
    j = P.plan_jumps(meta(), f.ravel(), None, (1.0, 1.0), (3.4, 1.0), jump_len_m=0.6)
    assert not j.ok and "obstacle" in j.reason
    j2 = P.plan_jumps(meta(), free().ravel(), None, (0.5, 0.5), (4.5, 4.5), jump_len_m=0.6, max_jumps=3)
    assert not j2.ok and "max_jumps" in j2.reason


def test_jumps_along_path_headings():
    path = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]]
    j = P.plan_jumps(meta(), free().ravel(), None, (1.0, 1.0), (3.0, 3.0), jump_len_m=0.6, path=path, max_jumps=10)
    assert j.ok and len(j.landings) == len(j.headings)
    assert j.headings[0] == pytest.approx(0.0)
    assert j.headings[-1] == pytest.approx(math.pi / 2, abs=0.2)
    assert j.residual_m <= 0.3


def test_jump_tall_obstacle_under_flight():
    f = free()
    wl = np.zeros_like(f)
    wl[20, 24] = 200                                           # tall thing between start and landing 1
    j = P.plan_jumps(meta(), f.ravel(), wl.ravel(), (1.0, 1.0), (1.6, 1.0), jump_len_m=0.6)
    assert not j.ok and "flight" in j.reason


# --------------------------------------------------------------- follower

def _sim(path, v, pose, steps=2000, dt=0.05, cfg=None):
    idx = 0
    x, y, th = pose
    for k in range(steps):
        vx, vyaw, idx, done = P.follow_controller((x, y, th), path, v, cfg, idx=idx)
        if done:
            return (x, y, th), k, True
        th += vyaw * dt
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
    return (x, y, th), steps, False


def test_follow_controller_converges_on_planned_path():
    f = box(free(), 2.0, 0.0, 2.2, 3.5)
    path = P.plan_path(meta(), f.ravel(), None, (1.0, 1.0), (3.5, 1.0))
    v = P.speed_profile(path, "normal")
    (x, y, _), k, done = _sim(path, v, (1.0, 1.0, 0.0))
    assert done and math.hypot(x - 3.5, y - 1.0) <= 0.15
    assert k * 0.05 < 3 * P.eta(path, v) + 5


def test_follow_controller_heading_first_and_speed_bounds():
    path = [[x, 0.0] for x in np.arange(0, 3.01, 0.1)]
    v = P.speed_profile(path, "stealth")
    vx, vyaw, idx, done = P.follow_controller((0.0, 0.0, math.pi), path, v)
    assert vx == 0.0 and abs(vyaw) > 0 and not done
    vx, vyaw, idx, done = P.follow_controller({"x": 1.5, "y": 0.05, "yaw": 0.0}, path, v, idx=0)
    assert 0 < vx <= 0.25 + 1e-9 and idx > 5
    assert P.follow_controller((3.0, 0.0, 0.0), path, v, idx=29)[3] is True


def test_blend_with_dwa():
    assert P.blend_with_dwa((0.5, 0.1), None, True) == (0.5, 0.1)
    assert P.blend_with_dwa((0.5, 0.1), (0.2, 0.5), False) == pytest.approx((0.5, 0.1))
    vx, vyaw = P.blend_with_dwa((0.5, 0.0), (0.2, 0.5), True)
    assert vx <= 0.2 + 1e-9 and vyaw == pytest.approx(0.35)
    assert P.blend_with_dwa((0.5, 0.0), (0.0, 0.0), 2) == (0.0, 0.0)


# ------------------------------------------------------- zones (9.1)

def _corner_profile(zone):
    pts = [[0.5, 0.5], [3.0, 0.5], [3.0, 4.0]]
    sm, stops = P.smooth_path_ex(pts, ds=0.05, zones=["z30", zone, "fine"])
    v = P.speed_profile(sm, "sprint", stops=stops)
    return sm, stops, v


def test_zone_fine_point_stops():
    pts = [[0.5, 0.5], [2.0, 0.5], [3.5, 0.5], [3.5, 3.0]]
    sm, stops = P.smooth_path_ex(pts, ds=0.1, zones=["z30", "fine", "z60", "fine"])
    assert len(stops) == 2 and stops[-1] == len(sm) - 1
    assert sm[stops[0]] == pytest.approx([2.0, 0.5])
    v = P.speed_profile(sm, "normal", stops=stops)
    assert v[stops[0]] == 0.0 and v[-1] == 0.0
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(np.asarray(sm), axis=0).T))])
    near = (s <= s[stops[0]]) & (s >= s[stops[0]] - 0.5)
    assert max(np.asarray(v)[near]) <= 0.3 + 1e-9
    parts = P.split_at_stops(sm, stops)
    assert len(parts) == 2 and parts[0][-1] == pytest.approx([2.0, 0.5])


def test_zone_z60_smoother_and_faster_than_z10():
    sm10, _, v10 = _corner_profile("z10")
    sm60, _, v60 = _corner_profile("z60")
    k10 = P.curvature(np.asarray(sm10)).max()
    k60 = P.curvature(np.asarray(sm60)).max()
    assert k60 < k10
    i10 = int(np.argmin(np.hypot(*(np.asarray(sm10) - [3.0, 0.5]).T)))
    i60 = int(np.argmin(np.hypot(*(np.asarray(sm60) - [3.0, 0.5]).T)))
    assert v60[i60] > v10[i10] > 0.0                          # fly-by: no stop at the corner
    assert P.eta(sm60, v60) < P.eta(sm10, v10)
    # rounding stays inside the zone: corner point within ~radius of the waypoint
    assert min_dist(sm10, (3.0, 0.5)) <= 0.1
    assert 0.1 < min_dist(sm60, (3.0, 0.5)) <= 0.6


def test_zone_blend_is_collision_checked():
    f = box(free(), 2.2, 0.65, 2.55, 1.0)                      # block inside the corner
    pg = P.PlanGrid(meta(), f.ravel(), None, cfg=P.PlannerConfig(robot_radius_m=0.0))
    pts = [[0.5, 0.5], [2.6, 0.5], [2.6, 4.0]]
    sm = P.smooth_path(pts, ds=0.05, grid=pg, zones=["z30", "z100", "fine"])
    assert path_free(pg, sm)
    unsafe = P.smooth_path(pts, ds=0.05, zones=["z30", "z100", "fine"])
    assert not path_free(pg, unsafe)                           # the unchecked blend would cut it


def test_plan_through_zones():
    f = box(free(), 2.0, 0.0, 2.2, 3.0)
    wps = [[1.0, 1.0], [3.5, 1.0], [3.5, 4.0], [1.0, 4.0]]
    p = P.plan_through(meta(), f.ravel(), None, wps, zones=[None, "fine", "z60", None])
    assert p.ok and len(p.stops) == 2 and p.stops[-1] == len(p.points) - 1
    assert p.points[p.stops[0]] == pytest.approx([3.5, 1.0])
    assert path_free(P.PlanGrid(meta(), f.ravel(), None), p.points)
    v = P.speed_profile(p, "normal")
    assert v[p.stops[0]] == 0.0
    assert p.to_dict()["stops"] == p.stops


# ----------------------------------------------------------- performance

def test_performance_400x400():
    w = h = 400
    m = meta(w, h)
    f = np.zeros((h, w), dtype=np.int16)
    f[100:300, 100:102] = 100                                  # U trap
    f[100:300, 298:300] = 100
    f[298:300, 100:300] = 100
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y = rng.integers(0, 380, 2)
        f[y:y + 8, x:x + 8] = 100
    f[5:15, 5:15] = 0
    f[385:395, 385:395] = 0
    cases = {"open diagonal": (free(w, h), (0.5, 0.5), (19.5, 19.5)),
             "U trap escape": (f, (10.0, 12.0), (10.0, 18.0)),
             "cluttered diag": (f, (0.5, 0.5), (19.5, 19.5))}
    for name, (fl, s, g) in cases.items():
        for cf in (2, 1):
            t = time.perf_counter()
            p = P.plan_path(m, fl.ravel(), None, s, g, cfg=P.PlannerConfig(coarse_factor=cf))
            ms = (time.perf_counter() - t) * 1000
            print("\n[planner perf] 400x400 %-15s coarse=%d ok=%s expanded=%6d %.1f ms len=%.2f m"
                  % (name, cf, p.ok, p.cells_expanded, ms, p.length_m))
            assert p.ok
            assert ms < 5000
