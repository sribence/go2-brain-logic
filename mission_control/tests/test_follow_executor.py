"""follow_executor/executor.py -- the gates between follower and mc_motion."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "follow_executor"))

from executor import decide, ego_delta  # noqa: E402

ARMED = {"armed": True}


def follow(**kw):
    base = {"state": "TRACKING", "hold": False, "age_s": 0.1,
            "command": {"vx": 0.3, "vyaw": 0.5, "dry_run": False}}
    base.update(kw)
    return base


def test_sends_when_every_gate_passes_and_clamps_to_stage_limits():
    send, vx, vyaw, _ = decide(follow(), ARMED, True, max_vx=0.2, max_vyaw=0.4)
    assert send and vx == 0.2 and vyaw == 0.4


@pytest.mark.parametrize("f,h,enabled,why", [
    (follow(), ARMED, False, "disabled"),
    (follow(), None, True, "unreachable"),
    (follow(), {"armed": False}, True, "not armed"),
    (None, ARMED, True, "perception unreachable"),
    (follow(age_s=1.0), ARMED, True, "stale"),
    (follow(age_s=None), ARMED, True, "stale"),
    (follow(command={"vx": 0.3, "vyaw": 0.1, "dry_run": True}), ARMED, True, "dry_run"),
    (follow(command={"vx": 0.3, "vyaw": 0.1}), ARMED, True, "dry_run"),
    (follow(state="OCCLUDED"), ARMED, True, "state"),
    (follow(state="ACQUIRING"), ARMED, True, "state"),
    (follow(hold=True), ARMED, True, "hold"),
])
def test_blocks(f, h, enabled, why):
    send, vx, vyaw, reason = decide(f, h, enabled)
    assert not send and vx == 0 and vyaw == 0 and why in reason


def test_never_reverses_and_rejects_nan():
    send, vx, vyaw, _ = decide(follow(command={"vx": -0.3, "vyaw": float("nan"), "dry_run": False}),
                               ARMED, True, max_vx=0.5, max_vyaw=0.5)
    assert send and vx == 0.0 and vyaw == 0.0


def test_ego_delta_forward_in_rotated_frame():
    dx, dy, dyaw = ego_delta((1.0, 1.0, math.pi / 2), (1.0, 1.5, math.pi / 2))
    assert dx == pytest.approx(0.5) and dy == pytest.approx(0.0) and dyaw == pytest.approx(0.0)


def test_ego_delta_yaw_wraps():
    _, _, dyaw = ego_delta((0, 0, math.pi - 0.1), (0, 0, -math.pi + 0.1))
    assert dyaw == pytest.approx(0.2)
