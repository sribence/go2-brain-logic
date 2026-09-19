"""perception/target_follower.py -- state machine + dry-run command tests.

Synthetic persons and appearance features only (no camera, no YOLO).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))

import appearance  # noqa: E402
from target_follower import (  # noqa: E402
    ACQUIRING, IDLE, LOST, OCCLUDED, TRACKING, FollowConfig, TargetFollower,
)

DT = 0.1


def person(tid, x, y, vx=0.0, vy=0.0):
    pos = {"x": x, "y": y, "z": 0.0}
    return {"track_id": tid, "depth_ok": True, "position": pos, "position_raw": pos,
            "velocity": {"vx": vx, "vy": vy}, "distance_m": math.hypot(x, y),
            "bearing_deg": math.degrees(math.atan2(y, x))}


def feat(hue_bin):
    """Appearance feature concentrated in one hue bin (= one shirt colour)."""
    f = np.zeros((2, appearance.H_BINS, appearance.S_BINS), np.float32)
    f[:, hue_bin, 4] = 1.0
    return f


RED, BLUE = feat(0), feat(10)


def locked_follower(cfg=None, x=2.0, y=0.0):
    """Follower locked on #1 (red shirt) and past ACQUIRING."""
    f = TargetFollower(cfg or FollowConfig())
    t = 0.0
    f.lock(1, [person(1, x, y)], t)
    for _ in range(f.cfg.acquire_frames):
        t += DT
        out = f.update([person(1, x, y)], {1: RED}, t)
    assert out["state"] == TRACKING
    return f, t


def test_idle_outputs_zero_command():
    out = TargetFollower().update([person(1, 2.0, 0.0)], {1: RED}, 0.0)
    assert out["state"] == IDLE
    assert out["command"]["vx"] == 0 and out["command"]["vyaw"] == 0
    assert out["command"]["dry_run"] is True


def test_lock_requires_visible_person_with_depth():
    with pytest.raises(ValueError):
        TargetFollower().lock(7, [person(1, 2.0, 0.0)], 0.0)


def test_acquiring_then_tracking():
    f = TargetFollower()
    f.lock(1, [person(1, 2.0, 0.0)], 0.0)
    out = f.update([person(1, 2.0, 0.0)], {1: RED}, DT)
    assert out["state"] == ACQUIRING
    assert out["command"]["vx"] == 0  # no motion while learning
    locked_follower()


def test_forward_command_when_far_and_ramped():
    f, t = locked_follower(x=3.0)
    out = f.update([person(1, 3.0, 0.0)], {1: RED}, t + DT)
    cmd = out["command"]
    assert cmd["vx_desired"] == pytest.approx(0.5)     # clamped to max_vx
    assert 0 < cmd["vx"] <= cmd["vx_desired"]            # ramped, not a step
    assert out["goal"]["distance_cm"] == 180             # 3.0 - 1.2 m


def test_no_forward_inside_min_safe_distance():
    f, t = locked_follower(x=0.5)
    out = f.update([person(1, 0.5, 0.0)], {1: RED}, t + DT)
    assert out["command"]["vx"] <= 0


def test_turns_toward_person_on_left_and_turns_first():
    f, t = locked_follower(x=1.0, y=1.0)                 # 45 deg left
    out = f.update([person(1, 1.0, 1.0)], {1: RED}, t + DT)
    assert out["command"]["vyaw_desired"] > 0
    assert out["command"]["vx_desired"] == 0             # bearing >= turn_first_deg


def test_never_switches_to_other_person_when_target_disappears():
    f, t = locked_follower()
    for _ in range(40):                                  # 4 s, stranger nearby
        t += DT
        out = f.update([person(2, 1.5, 0.3)], {2: BLUE}, t)
        assert out["track_id"] == 1
        assert out["command"]["vx"] == 0 and out["command"]["vyaw"] == 0
    assert out["state"] == LOST


def test_same_colour_stranger_outside_gate_is_not_reacquired():
    f, t = locked_follower(x=2.0, y=0.0)
    t += DT
    f.update([], {}, t)                                  # target vanishes
    t += DT
    out = f.update([person(5, 4.5, -2.0)], {5: RED}, t)  # far away, same shirt
    assert out["state"] == OCCLUDED and out["track_id"] == 1


def test_reacquire_after_id_switch_inside_gate_with_same_appearance():
    f, t = locked_follower(x=2.0, y=0.0)
    t += DT
    f.update([], {}, t)
    t += DT
    out = f.update([person(9, 2.1, 0.05)], {9: RED}, t)  # ByteTrack gave a new id
    assert out["state"] == TRACKING and out["track_id"] == 9


def test_different_appearance_inside_gate_is_rejected():
    f, t = locked_follower()
    t += DT
    f.update([], {}, t)
    t += DT
    out = f.update([person(9, 2.0, 0.0)], {9: BLUE}, t)
    assert out["state"] == OCCLUDED


def test_ambiguous_two_matches_waits():
    f, t = locked_follower()
    t += DT
    f.update([], {}, t)
    t += DT
    out = f.update([person(8, 2.0, 0.2), person(9, 2.0, -0.2)], {8: RED, 9: RED}, t)
    assert out["state"] == OCCLUDED and "ambiguous" in out["reason"]


def test_position_jump_on_same_id_stops():
    f, t = locked_follower(x=2.0, y=0.0)
    out = f.update([person(1, 4.0, 1.5)], {1: RED}, t + DT)   # 2.5 m in 0.1 s
    assert out["state"] == OCCLUDED and "jump" in out["reason"]
    assert out["command"]["vx"] == 0


def test_appearance_swap_on_same_id_stops_after_k_frames():
    f, t = locked_follower()
    for i in range(f.cfg.mismatch_frames):
        t += DT
        out = f.update([person(1, 2.0, 0.0)], {1: BLUE}, t)
    assert out["state"] == OCCLUDED and "appearance" in out["reason"]


def test_stop_is_immediate_when_target_lost():
    f, t = locked_follower(x=3.0)
    for _ in range(20):
        t += DT
        f.update([person(1, 3.0, 0.0)], {1: RED}, t)
    t += DT
    out = f.update([], {}, t)
    assert out["command"]["vx"] == 0


def test_release_returns_to_idle():
    f, t = locked_follower()
    f.release(t)
    out = f.update([person(1, 2.0, 0.0)], {1: RED}, t + DT)
    assert out["state"] == IDLE and out["track_id"] is None


def test_ego_motion_keeps_gate_on_person():
    f, t = locked_follower(x=2.0, y=0.0)
    f.apply_ego_motion(0.5, 0.0, 0.0)                    # robot walked 0.5 m forward
    out = f.update([person(1, 1.5, 0.0)], {1: RED}, t + DT)
    assert out["state"] == TRACKING


def test_similarity_identical_and_disjoint():
    assert appearance.similarity(RED, RED) == pytest.approx(1.0)
    assert appearance.similarity(RED, BLUE) == pytest.approx(0.0)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("FOLLOW_MAX_VX", "0.2")
    monkeypatch.setenv("FOLLOW_ACQUIRE_FRAMES", "3")
    cfg = FollowConfig.from_env()
    assert cfg.max_vx == 0.2 and cfg.acquire_frames == 3
