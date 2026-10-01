"""safety_guard: decide() zones, fail-safes, monotonic clamp, hysteresis, HTTP API."""
import importlib.util
import math
import os
import random
import sys

import pytest

HERE = os.path.dirname(__file__)
SG = os.path.join(HERE, "..", "safety_guard")
sys.path.insert(0, os.path.join(HERE, ".."))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("safety_guard_core", os.path.join(SG, "core.py"))
from omni.omni_types import PersonTrack  # noqa: E402

decide, SafetyConfig = core.decide, core.SafetyConfig
CLEAR, CAUTION, SLOW, STOP = core.CLEAR, core.CAUTION, core.SLOW, core.STOP
NOW = 1000.0
CFG = SafetyConfig(child_from_z=False)   # z-height child rule tested separately


def person(x, y, vx=0.0, vy=0.0, z=0.0, **kw):
    return PersonTrack(gid=kw.pop("gid", 1), x=x, y=y, z=z, vx=vx, vy=vy, conf=0.9,
                       last_seen_t=NOW, **kw)


def run(persons=(), cmd=(0.0, 0.0, 0.0), lidar=10.0, persons_t=NOW, now=NOW, cfg=CFG, **kw):
    return decide(list(persons) if persons is not None else None, lidar, None, cmd, now,
                  persons_t, cfg, **kw)


# -- zones ----------------------------------------------------------------------

@pytest.mark.parametrize("d,level", [
    (0.5, STOP), (0.79, STOP), (0.81, SLOW), (1.5, SLOW), (1.99, SLOW),
    (2.01, CAUTION), (3.0, CAUTION), (3.49, CAUTION), (3.51, CLEAR), (8.0, CLEAR),
])
def test_zone_levels_stationary_robot(d, level):
    # person to the side so no stretch; robot not moving
    assert run([person(0.0, d)]).level == level


def test_no_person_is_clear_and_passes_command_unchanged():
    dec = run([], cmd=(0.5, 0.1, 0.3))
    assert dec.level == CLEAR and dec.cmd_out == (0.5, 0.1, 0.3)


def test_slow_vmax_linear_between_01_and_04():
    lo = run([person(0.0, 0.8001)]).vmax
    mid = run([person(0.0, 1.4)]).vmax
    hi = run([person(0.0, 1.9999)]).vmax
    assert lo == pytest.approx(0.1, abs=1e-3)
    assert mid == pytest.approx(0.25, abs=1e-3)
    assert hi == pytest.approx(0.4, abs=1e-3)


def test_caution_caps_at_06():
    dec = run([person(0.0, -3.0)], cmd=(1.0, 0.0, 0.0))
    assert dec.level == CAUTION and dec.cmd_out[0] == pytest.approx(0.6)


def test_zone_stretched_ahead_of_motion_not_behind():
    # 3.8 m ahead, moving forward 0.5 m/s -> effective 3.3 m -> CAUTION
    assert run([person(3.8, 0.0)], cmd=(0.5, 0.0, 0.0)).level == CAUTION
    # same person behind the robot -> no stretch -> CLEAR
    assert run([person(-3.8, 0.0)], cmd=(0.5, 0.0, 0.0)).level == CLEAR
    # stationary robot -> CLEAR
    assert run([person(3.8, 0.0)]).level == CLEAR


def test_moving_toward_person_at_1m_is_stop():
    # 1.0 m ahead with 0.3 m/s forward -> effective 0.7 m -> STOP
    dec = run([person(1.0, 0.0)], cmd=(0.3, 0.0, 0.0))
    assert dec.level == STOP and dec.cmd_out[:2] == (0.0, 0.0)


# -- TTC ------------------------------------------------------------------------

def test_ttc_below_1s_is_stop():
    # person 3.0 m ahead running at us at 3 m/s: TTC ~0.73 s
    dec = run([person(3.0, 0.0, vx=-3.0)], cmd=(0.0, 0.0, 0.0))
    assert dec.level == STOP and "TTC" in dec.reason


def test_ttc_between_1_and_2s_is_slow():
    # 4.0 m away (CLEAR zone), closing at 2 m/s -> TTC = (4-0.8)/2 = 1.6 s
    dec = run([person(4.0, 0.0, vx=-2.0)])
    assert dec.level == SLOW and dec.ttc_s == pytest.approx(1.6, abs=0.01)


def test_ttc_uses_robot_command():
    # stationary person 4.5 m ahead, robot commands 2 m/s forward -> TTC 1.85 s
    dec = run([person(4.5, 0.0)], cmd=(2.0, 0.0, 0.0))
    assert dec.level in (SLOW, STOP) and dec.cmd_out[0] <= 0.4


def test_person_walking_past_has_no_ttc():
    dec = run([person(4.0, 3.0, vx=1.0)])
    assert dec.ttc_s is None and dec.level == CLEAR


def test_time_to_collision_helper():
    assert core.time_to_collision(2.0, 0.0, -1.0, 0.0, 1.0, 5.0) == pytest.approx(1.0)
    assert core.time_to_collision(2.0, 0.0, 1.0, 0.0, 1.0, 5.0) is None
    assert core.time_to_collision(0.5, 0.0, 0.0, 0.0, 1.0, 5.0) == 0.0


# -- child / fast ----------------------------------------------------------------

def test_fast_person_multiplies_zones():
    # 2.5 m to the side: adult -> CAUTION; fast (2.5 m/s tangential) -> SLOW (2.0*1.5=3.0)
    assert run([person(0.0, 2.5, vx=0.5)]).level == CAUTION
    assert run([person(0.0, 2.5, vx=2.5)]).level == SLOW


def test_child_by_height_multiplies_zones():
    p = {"gid": 1, "x": 0.0, "y": 1.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "height_m": 1.1}
    assert run([p]).level == STOP          # 1.0 < 0.8*1.5
    p["height_m"] = 1.75
    assert run([p]).level == SLOW


def test_child_from_z_when_enabled():
    cfg = SafetyConfig(child_from_z=True)
    assert run([person(0.0, 1.0, z=1.0)], cfg=cfg).level == STOP
    assert run([person(0.0, 1.0, z=1.7)], cfg=cfg).level == SLOW
    assert run([person(0.0, 1.0, z=0.0)], cfg=cfg).level == SLOW   # z unknown


# -- fail-safes --------------------------------------------------------------------

def test_stale_person_data_is_slow():
    dec = run([], persons_t=NOW - 0.31, cmd=(1.0, 0.0, 0.0))
    assert dec.level == SLOW and "stale" in dec.reason
    assert math.hypot(*dec.cmd_out[:2]) <= CFG.stale_vmax + 1e-9
    assert run([], persons_t=NOW - 0.29).level == CLEAR


def test_no_person_data_ever_is_slow():
    dec = run(None, persons_t=None, cmd=(1.0, 0.0, 0.0))
    assert dec.level == SLOW and "no person data" in dec.reason


def test_lidar_missing_is_stop_rotation_only():
    dec = run([], lidar=None, cmd=(0.5, 0.2, 1.0))
    assert dec.level == STOP and dec.cmd_out == (0.0, 0.0, CFG.stop_vyaw_max)


def test_lidar_near_in_motion_direction_is_stop():
    assert run([], lidar=0.4, cmd=(0.3, 0.0, 0.0)).level == STOP
    assert run([], lidar=0.6, cmd=(0.3, 0.0, 0.0)).level == CLEAR
    # not moving linearly -> nearby obstacle does not block rotation
    dec = run([], lidar=0.4, cmd=(0.0, 0.0, 0.4))
    assert dec.level == CLEAR and dec.cmd_out == (0.0, 0.0, 0.4)


def test_nan_command_is_zeroed():
    dec = run([], cmd=(float("nan"), float("inf"), "x"))
    assert dec.cmd_out == (0.0, 0.0, 0.0)


def test_malformed_person_is_slow():
    assert run([{"x": None, "y": 1.0}]).level == SLOW


# -- STOP behaviour ----------------------------------------------------------------

def test_stop_allows_only_capped_rotation():
    dec = run([person(0.5, 0.0)], cmd=(0.5, 0.3, -1.5))
    assert dec.level == STOP
    assert dec.cmd_out == (0.0, 0.0, -CFG.stop_vyaw_max)


def test_stop_allows_backing_away_when_rear_clear():
    dec = run([person(0.6, 0.0)], cmd=(-0.5, 0.2, 0.0), rear_clear_m=2.0)
    assert dec.level == STOP and dec.backoff
    assert dec.cmd_out == (-CFG.backoff_vmax, 0.0, 0.0)


@pytest.mark.parametrize("kw,persons", [
    ({"rear_clear_m": 0.3}, [person(0.6, 0.0)]),                  # rear blocked
    ({"rear_clear_m": None}, [person(0.6, 0.0)]),                 # rear unknown
    ({"rear_clear_m": 2.0, "lidar": None}, [person(0.6, 0.0)]),   # lidar missing
    ({"rear_clear_m": 2.0}, [person(-0.6, 0.0)]),                 # person behind
    ({"rear_clear_m": 2.0}, [person(0.6, 0.0), person(-1.5, 0.0, gid=2)]),
])
def test_stop_backoff_refused(kw, persons):
    dec = run(persons, cmd=(-0.3, 0.0, 0.0), **kw)
    assert dec.level == STOP and dec.cmd_out[:2] == (0.0, 0.0) and not dec.backoff


# -- monotonic clamp property ----------------------------------------------------------

def test_random_commands_are_never_amplified():
    rng = random.Random(42)
    prev = None
    for i in range(3000):
        cmd = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3))
        persons = [person(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3),
                          rng.uniform(-3, 3), z=rng.uniform(0, 2), gid=j)
                   for j in range(rng.randint(0, 3))]
        lidar = rng.choice([None, rng.uniform(0, 3), float("inf")])
        pt = rng.choice([None, NOW + i * 0.05, NOW + i * 0.05 - 1.0])
        dec = decide(persons, lidar, None, cmd, NOW + i * 0.05, pt, SafetyConfig(),
                     rear_clear_m=rng.choice([None, 0.2, 3.0]), prev=prev)
        prev = dec
        for o, c in zip(dec.cmd_out, cmd):
            assert abs(o) <= abs(c) + 1e-9
            assert o == 0.0 or (o > 0) == (c > 0)        # never reverses a component
        if dec.level == STOP:
            assert dec.cmd_out[1] == 0.0 and abs(dec.cmd_out[2]) <= 0.5 + 1e-9
            assert dec.cmd_out[0] <= 0.0
        assert math.hypot(dec.cmd_out[0], dec.cmd_out[1]) <= max(dec.vmax, 0.0) + 1e-9


# -- hysteresis ------------------------------------------------------------------------

def test_hysteresis_relaxes_only_after_hold():
    d0 = run([person(0.0, 0.5)], now=NOW)
    assert d0.level == STOP
    d1 = run([person(0.0, 5.0)], now=NOW + 0.1, persons_t=NOW + 0.1, prev=d0)
    assert d1.level == STOP and d1.raw_level == CLEAR
    d2 = run([person(0.0, 5.0)], now=NOW + 0.4, persons_t=NOW + 0.4, prev=d1)
    assert d2.level == STOP
    d3 = run([person(0.0, 5.0)], now=NOW + 0.61, persons_t=NOW + 0.61, prev=d2)
    assert d3.level == CLEAR


def test_hysteresis_restarts_if_raw_level_flickers_back():
    d0 = run([person(0.0, 0.5)], now=NOW)
    d1 = run([person(0.0, 5.0)], now=NOW + 0.3, persons_t=NOW + 0.3, prev=d0)
    d2 = run([person(0.0, 0.5)], now=NOW + 0.4, persons_t=NOW + 0.4, prev=d1)
    assert d2.level == STOP and d2.relax_since is None
    d3 = run([person(0.0, 5.0)], now=NOW + 0.7, persons_t=NOW + 0.7, prev=d2)
    assert d3.level == STOP


def test_escalation_is_immediate():
    d0 = run([person(0.0, 5.0)], now=NOW)
    d1 = run([person(0.0, 0.5)], now=NOW + 0.05, persons_t=NOW + 0.05, prev=d0)
    assert d1.level == STOP


# -- LiDAR helpers -------------------------------------------------------------------------

def test_lidar_near_in_direction():
    pts = [(1.0, 0.0, 0.3), (-0.4, 0.0, 0.3), (0.0, 2.0, 0.3), (0.3, 0.0, 3.0)]
    assert core.lidar_near_in_direction(pts, (0.5, 0, 0)) == pytest.approx(1.0)
    assert core.lidar_near_in_direction(pts, (-0.5, 0, 0)) == pytest.approx(0.4)
    assert core.lidar_near_in_direction(pts, (0, 0, 0.5)) == pytest.approx(0.4)
    assert core.lidar_near_in_direction(None, (0.5, 0, 0)) is None
    assert core.rear_clearance(pts) == pytest.approx(0.4)


# -- HTTP API (motion forwarding mocked) ------------------------------------------------------

@pytest.fixture()
def api(monkeypatch):
    from fastapi.testclient import TestClient
    app_mod = _load("safety_guard_app", os.path.join(SG, "app.py"))
    calls = []

    def fake_forward(path, body=None):
        calls.append((path, body))
        return 200, {"ok": True}

    monkeypatch.setattr(app_mod, "forwarder", fake_forward)
    g = app_mod.Guard(SafetyConfig(child_from_z=False))
    monkeypatch.setattr(app_mod, "guard", g)
    return app_mod, TestClient(app_mod.app), calls, g


def _feed(app_mod, g, persons, lidar_pts=((5.0, 0.0, 0.3),)):
    import time
    now = time.time()
    g.persons.ingest({"t": now, "persons": [p.to_dict() for p in persons]}, recv_t=now)
    g.set_lidar(list(lidar_pts), now)


def test_api_move_clear_forwards_unchanged(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [])
    r = client.post("/move", json={"vx": 0.3, "vy": 0.0, "vyaw": 0.2, "source": "nav"})
    assert r.status_code == 200
    assert calls == [("/move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.2})]
    assert r.json()["decision"]["level"] == CLEAR


def test_api_move_clamped_in_caution(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [person(0.0, 3.0)])
    client.post("/move", json={"vx": 1.0, "vy": 0.0, "vyaw": 0.0})
    assert calls[-1][0] == "/move" and calls[-1][1]["vx"] == pytest.approx(0.6)


def test_api_stop_level_sends_stop(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [person(0.5, 0.0)])
    r = client.post("/move", json={"vx": 0.5, "vy": 0.0, "vyaw": 0.0})
    assert r.json()["forwarded"] == "stop" and calls[-1] == ("/stop", None)


def test_api_stop_level_still_forwards_rotation(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [person(0.5, 0.0)])
    client.post("/move", json={"vx": 0.5, "vy": 0.0, "vyaw": 0.9})
    assert calls[-1] == ("/move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.5})


def test_api_no_lidar_is_stop(api):
    app_mod, client, calls, g = api
    import time
    g.persons.ingest({"t": time.time(), "persons": []})
    client.post("/move", json={"vx": 0.3})
    assert calls[-1] == ("/stop", None)


def test_api_stop_endpoint_and_state(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [person(0.0, 1.5)])
    client.post("/move", json={"vx": 0.3})
    assert client.post("/stop", json={"source": "ui"}).status_code == 200
    assert calls[-1] == ("/stop", None)
    s = client.get("/state").json()
    assert s["level"] == SLOW and s["nearest_person_m"] == pytest.approx(1.5)
    bus = g.bus_state()
    assert set(["t", "level", "vmax", "nearest_person_m", "reason"]) <= set(bus)


def test_api_motion_refusal_is_passed_through(api, monkeypatch):
    app_mod, client, calls, g = api
    monkeypatch.setattr(app_mod, "forwarder", lambda p, b=None: (409, {"error": "not armed"}))
    _feed(app_mod, g, [])
    assert client.post("/move", json={"vx": 0.2}).status_code == 409
    monkeypatch.setattr(app_mod, "forwarder", lambda p, b=None: (0, {"error": "down"}))
    assert client.post("/move", json={"vx": 0.2}).status_code == 502


def test_api_never_arms(api):
    app_mod, client, calls, g = api
    assert client.post("/arm", json={"armed": True}).status_code in (404, 405)
    _feed(app_mod, g, [])
    client.post("/move", json={"vx": 0.2})
    client.post("/stop")
    assert all("arm" not in path for path, _ in calls)


def test_tick_resends_smaller_command_on_escalation_only(api):
    app_mod, client, calls, g = api
    import time
    _feed(app_mod, g, [])
    client.post("/move", json={"vx": 0.5})
    n = len(calls)
    g.tick(time.time())                         # nothing changed -> no send
    assert len(calls) == n
    _feed(app_mod, g, [person(0.5, 0.0)])       # person steps in front
    g.tick(time.time())
    assert calls[-1] == ("/stop", None)
    n = len(calls)
    _feed(app_mod, g, [])                       # gone again: never re-accelerates
    g.tick(time.time() + 1.0)
    assert len(calls) == n


def test_tick_without_active_caller_sends_nothing(api):
    app_mod, client, calls, g = api
    import time
    _feed(app_mod, g, [person(0.5, 0.0)])
    g.tick(time.time())
    assert calls == []


# -- follow / pursuit minimum distance (CONTRACTS.md section 0) ----------------------------

def test_follow_min_distance_blocks_approach_only():
    p = [person(2.3, 0.0)]                     # SLOW zone, inside 2.5 m
    assert run(p, cmd=(0.3, 0.0, 0.0)).cmd_out[0] > 0.0          # plain source: allowed
    dec = run(p, cmd=(0.3, 0.0, 0.2), min_person_dist_m=2.5)
    assert dec.cmd_out == (0.0, 0.0, 0.2) and "follow min distance" in dec.reason
    back = run(p, cmd=(-0.2, 0.0, 0.0), min_person_dist_m=2.5)    # retreat allowed
    assert back.cmd_out[0] == pytest.approx(-0.2)
    far = run([person(2.7, 0.0)], cmd=(0.3, 0.0, 0.0), min_person_dist_m=2.5)
    assert far.cmd_out[0] > 0.0


def test_api_pursuit_source_gets_follow_min_distance(api):
    app_mod, client, calls, g = api
    _feed(app_mod, g, [person(2.3, 0.0)])
    client.post("/move", json={"vx": 0.3, "source": "pursuit"})
    assert calls[-1] == ("/move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
    client.post("/move", json={"vx": 0.3, "source": "navigation"})
    assert calls[-1][1]["vx"] > 0.0
