"""omni/pursuit.py: shadow distance, never approach inside 2.5 m, search, lost."""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni.omni_types import PersonTrack  # noqa: E402
from omni.pursuit import (LOST, SEARCH, SHADOW, PursuitConfig,  # noqa: E402
                          PursuitController, select_target)


def tgt(x, y, t, gid=1, conf=0.9):
    return PersonTrack(gid=gid, x=x, y=y, z=1.7, vx=0.0, vy=0.0, conf=conf, last_seen_t=t)


def test_far_target_drives_forward_up_to_max_with_accel_limit():
    pc = PursuitController()
    t, vxs = 0.0, []
    for _ in range(40):
        vx, vy, vyaw, st = pc.step(tgt(8.0, 0.0, t), now=t)
        vxs.append(vx)
        t += 0.1
    assert st == SHADOW and vy == 0.0
    assert vxs[0] <= 0.05 + 1e-9                     # 0.5 m/s^2 * 0.1 s
    assert all(b - a <= 0.05 + 1e-9 for a, b in zip(vxs, vxs[1:]))
    assert max(vxs) <= 1.0 and vxs[-1] == pytest.approx(1.0)


def test_holds_at_follow_distance():
    pc = PursuitController()
    vx, _, vyaw, st = pc.step(tgt(3.0, 0.0, 0.0), now=0.0)
    assert st == SHADOW and vx == pytest.approx(0.0) and vyaw == pytest.approx(0.0)


@pytest.mark.parametrize("d", [0.5, 1.5, 2.0, 2.4, 2.49])
def test_inside_min_distance_backs_off_or_stops(d):
    pc = PursuitController()
    pc.vx = 1.0                                      # even when coming in fast
    vx, _, _, st = pc.step(tgt(d, 0.0, 0.0), now=0.0)
    assert st == SHADOW and vx <= 0.0


def test_never_commands_approach_inside_min_dist_property():
    rng = random.Random(7)
    cfg = PursuitConfig()
    pc = PursuitController(cfg)
    t = 0.0
    for _ in range(5000):
        t += rng.uniform(0.02, 0.3)
        x, y = rng.uniform(-6, 6), rng.uniform(-6, 6)
        if rng.random() < 0.1:
            pc.vx = rng.uniform(-0.3, 1.0)           # arbitrary internal state
        vx, vy, vyaw, st = pc.step(tgt(x, y, t), now=t)
        d, b = math.hypot(x, y), math.atan2(y, x)
        approach = vx * math.cos(b)                  # rate of range decrease
        assert abs(vx) <= cfg.max_v + 1e-9 and abs(vyaw) <= cfg.max_vyaw + 1e-9
        if d <= cfg.min_dist:
            assert approach <= 1e-9
        else:
            # cannot reach min_dist within the lookahead
            assert approach * cfg.approach_lookahead_s <= d - cfg.min_dist + 1e-9


def test_turns_before_driving_when_target_beside():
    pc = PursuitController()
    vx, _, vyaw, _ = pc.step(tgt(0.0, 6.0, 0.0), now=0.0)
    assert vx == 0.0 and vyaw > 0.0


def test_search_rotates_toward_last_bearing_then_lost():
    pc = PursuitController()
    pc.step(tgt(2.0, -4.0, 0.0), now=0.0)            # target to the right
    vx, vy, vyaw, st = pc.step(None, now=1.0)
    assert st == SEARCH and vx == 0.0 and vy == 0.0 and vyaw < 0.0
    for t in (2.0, 5.0, 9.9):
        vx, _, vyaw, st = pc.step(None, now=t)
        assert st == SEARCH and vx == 0.0 and vyaw < 0.0
    assert pc.step(None, now=10.2) == (0.0, 0.0, 0.0, LOST)


def test_stale_target_counts_as_lost():
    pc = PursuitController()
    pc.step(tgt(5.0, 1.0, 100.0), now=100.0)
    _, _, _, st = pc.step(tgt(5.0, 1.0, 100.0), now=102.0)   # same old sighting
    assert st == SEARCH


def test_search_uses_odom_to_find_last_world_bearing():
    pc = PursuitController()
    pc.step(tgt(0.0, 5.0, 0.0), odom={"yaw": 0.0}, now=0.0)   # left, world bearing +90 deg
    # robot has since turned to yaw=+180 deg: target is now to its right
    _, _, vyaw, st = pc.step(None, odom={"yaw": math.pi}, now=0.5)
    assert st == SEARCH and vyaw < 0.0


def test_initial_without_target_is_lost():
    assert PursuitController().step(None, now=0.0) == (0.0, 0.0, 0.0, LOST)


def test_nan_target_is_ignored():
    pc = PursuitController()
    assert pc.step(tgt(float("nan"), 0.0, 0.0), now=0.0)[3] == LOST


def test_select_target():
    ps = [tgt(5.0, 0.0, 0, gid=1), tgt(2.0, 1.0, 0, gid=2), tgt(1.0, 0.0, 0, gid=3, conf=0.1)]
    assert select_target(ps, "nearest").gid == 3
    assert select_target(ps, "nearest", min_conf=0.3).gid == 2
    assert select_target(ps, "gid", gid=1).gid == 1
    assert select_target(ps, "gid", gid=99) is None          # never switches person
    assert select_target(ps, "gid") is None
    assert select_target([], "nearest") is None
    assert select_target([p.to_dict() for p in ps], "gid", gid=2)["gid"] == 2
    with pytest.raises(ValueError):
        select_target(ps, "random")
