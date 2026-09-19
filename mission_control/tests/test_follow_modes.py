"""perception/follow_modes.py -- operator modes on top of the follower."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))

from follow_modes import FollowSupervisor, ModeError  # noqa: E402
from target_follower import ACQUIRING, IDLE, LOST, TRACKING, TargetFollower  # noqa: E402
from test_target_follower import DT, RED, BLUE, person  # noqa: E402


def sup(**kw):
    return FollowSupervisor(TargetFollower(), **kw)


def run(s, persons, feats, t):
    s.before_update(persons, t)
    out = s.follower.update(persons, feats, t)
    return s.after_update(out, t)


def test_defaults_are_safe():
    s = sup()
    out = s.summary(None)
    assert out["mode"] == "off" and out["dry_run"] is True and out["live_allowed"] is False


def test_dry_run_cannot_be_disabled_without_live_permission():
    s = sup()
    with pytest.raises(ModeError) as e:
        s.set_dry_run(False)
    assert e.value.status == 403 and s.dry_run is True


def test_dry_run_flag_reaches_command_when_live_allowed():
    s = sup(allow_live=True)
    s.set_dry_run(False)
    out = run(s, [person(1, 2.0, 0.0)], {1: RED}, 0.0)
    assert out["command"]["dry_run"] is False


def test_distance_slider_changes_follow_distance_and_is_bounded():
    s = sup()
    s.set_distance(2.5)
    assert s.follower.cfg.follow_distance_m == 2.5
    for bad in (0.5, 4.5):
        with pytest.raises(ModeError):
            s.set_distance(bad)


def test_unknown_mode_rejected():
    with pytest.raises(ModeError):
        sup().set_mode("attack", 0.0)


def test_lock_from_off_switches_to_user_follow():
    s = sup()
    s.lock(1, [person(1, 2.0, 0.0)], 0.0)
    assert s.mode == "user_follow" and s.follower.state == ACQUIRING


def test_user_follow_never_locks_automatically():
    s = sup()
    s.set_mode("user_follow", 0.0)
    out = run(s, [person(1, 2.0, 0.0)], {1: RED}, DT)
    assert out["state"] == IDLE


def test_off_releases_lock():
    s = sup()
    s.lock(1, [person(1, 2.0, 0.0)], 0.0)
    s.set_mode("off", DT)
    assert s.follower.state == IDLE


def test_intruder_locks_nearest_and_alerts():
    alerts = []
    s = sup(on_alert=alerts.append)
    s.set_mode("intruder", 0.0)
    out = run(s, [person(1, 3.0, 0.0), person(2, 1.8, 0.5)], {1: RED, 2: BLUE}, DT)
    assert out["track_id"] == 2 and out["state"] == ACQUIRING
    assert alerts[0]["kind"] == "intruder" and alerts[0]["track_id"] == 2


def test_intruder_ignores_people_out_of_range():
    s = sup()
    s.set_mode("intruder", 0.0)
    out = run(s, [person(1, 8.0, 0.0)], {1: RED}, DT)
    assert out["state"] == IDLE


def test_intruder_rearms_only_after_lost_and_cooldown():
    alerts = []
    s = sup(on_alert=alerts.append)
    s.set_mode("intruder", 0.0)
    t = 0.0
    for _ in range(6):
        t += DT
        run(s, [person(1, 2.0, 0.0)], {1: RED}, t)
    # target gone, stranger present: must not switch while OCCLUDED
    while s.follower.state != LOST:
        t += DT
        out = run(s, [person(2, 2.5, 1.0)], {2: BLUE}, t)
        assert out["track_id"] == 1
    assert [a["kind"] for a in alerts] == ["intruder", "target_lost"]
    t_lost = t
    while t - t_lost < s.cfg.intruder_rearm_s - DT:
        t += DT
        out = run(s, [person(2, 2.5, 1.0)], {2: BLUE}, t)
        assert out["track_id"] == 1
    for _ in range(3):
        t += DT
        out = run(s, [person(2, 2.5, 1.0)], {2: BLUE}, t)
    assert out["track_id"] == 2


def test_leaving_intruder_mode_drops_automatic_lock():
    s = sup()
    s.set_mode("intruder", 0.0)
    run(s, [person(1, 2.0, 0.0)], {1: RED}, DT)
    s.set_mode("user_follow", 2 * DT)
    assert s.follower.state == IDLE


def test_gesture_only_in_trick_mode():
    s = sup()
    with pytest.raises(ModeError):
        s.gesture("wave", 0.0)
    s.set_mode("trick", 0.0)
    with pytest.raises(ModeError):
        s.gesture("dance", 0.0)
    g = s.gesture("wave", 0.0)
    assert g["action"] == "hello" and g["executed"] is False


def test_stop_gesture_holds_command_and_ok_resumes():
    s = sup()
    s.set_mode("trick", 0.0)
    t = 0.0
    s.lock(1, [person(1, 3.0, 0.0)], t)
    for _ in range(8):
        t += DT
        out = run(s, [person(1, 3.0, 0.0)], {1: RED}, t)
    assert out["state"] == TRACKING and out["command"]["vx"] > 0
    s.gesture("stop", t)
    t += DT
    out = run(s, [person(1, 3.0, 0.0)], {1: RED}, t)
    assert out["state"] == TRACKING and out["command"]["vx"] == 0 and out["hold"] is True
    s.gesture("ok", t)
    t += DT
    out = run(s, [person(1, 3.0, 0.0)], {1: RED}, t)
    assert out["command"]["vx"] > 0


def test_summary_flat_fields_for_console():
    s = sup()
    s.lock(1, [person(1, 2.0, 0.0)], 0.0)
    t = 0.0
    for _ in range(6):
        t += DT
        out = run(s, [person(1, 2.0, 0.0)], {1: RED}, t)
    flat = s.summary(out)
    assert flat["state"] == TRACKING and flat["target_id"] == 1
    assert flat["target_dist_cm"] == 200
    assert {"vx", "vyaw", "dry_run"} <= set(flat["command"])
