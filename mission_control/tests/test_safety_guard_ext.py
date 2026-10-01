"""safety_guard extensions (mission/CONTRACT.md section 4 + 9.4/9.6): dead-man,
sprint gate, action gating, dead-man drop during sprint, fleet robots."""
import importlib.util
import math
import os
import random
import sys
import time

import pytest

HERE = os.path.dirname(__file__)
SG = os.path.join(HERE, "..", "safety_guard")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("safety_guard_core", os.path.join(SG, "core.py"))
decide, SafetyConfig = core.decide, core.SafetyConfig
CLEAR, CAUTION, SLOW, STOP = core.CLEAR, core.CAUTION, core.SLOW, core.STOP
CFG = SafetyConfig(child_from_z=False)


def P(x, y, **kw):
    return dict({"gid": 1, "x": x, "y": y, "z": 0.0, "vx": 0.0, "vy": 0.0}, **kw)


# -- dead-man registry ---------------------------------------------------------------------

def test_deadman_timing_and_multiple_clients():
    d = core.DeadmanRegistry(0.3)
    assert not d.alive(100.0)
    assert d.beat("ui", 100.0)
    assert d.alive(100.29) and not d.alive(100.31)
    d.beat("gamepad", 100.2)
    assert d.clients(100.35) == ["gamepad"] and d.alive(100.49) and not d.alive(100.51)
    d.beat("ui", 100.4)
    assert d.clients(100.45) == ["gamepad", "ui"]
    d.release("ui")
    assert d.clients(100.45) == ["gamepad"]


def test_deadman_rejects_empty_and_bounds_clients():
    d = core.DeadmanRegistry(0.3, max_clients=2)
    assert not d.beat("", 1.0) and not d.beat(None, 1.0)
    assert d.beat("a", 1.0) and d.beat("b", 1.0) and not d.beat("c", 1.0)
    assert d.beat("c", 20.0)               # old ones pruned
    assert not d.alive(1.0 + 0.3)
    # a heartbeat stamped in the future must not count as alive
    d2 = core.DeadmanRegistry(0.3)
    d2.beat("x", 50.0)
    assert not d2.alive(49.0)


# -- pure sprint pieces ----------------------------------------------------------------

def test_persons_in_corridor():
    ps = [P(5.0, 1.4), P(5.0, 1.6), P(8.5, 0.0), P(-1.0, 0.0), {"x": None, "y": 1.0}]
    inside = core.persons_in_corridor(ps, (1.0, 0.0), 1.5, 8.0)
    assert [p.get("x") for p in inside] == [5.0, None]           # malformed counts as inside
    # direction follows the command (sideways)
    assert len(core.persons_in_corridor([P(1.0, 5.0)], (0.0, 0.5), 1.5, 8.0)) == 1


@pytest.mark.parametrize("kw,ok", [
    ({}, True),
    ({"profile": "normal"}, False), ({"profile": None}, False),
    ({"source": "navigation"}, False), ({"source": ""}, False), ({"source": "rule"}, False),
    ({"deadman": False}, False),
    ({"battery": 30.0}, False), ({"battery": 29.0}, False), ({"battery": None}, False),
    ({"battery": float("nan")}, False), ({"battery": 120.0}, False), ({"battery": 31.0}, True),
    ({"lidar": False}, False), ({"persons": False}, False),
])
def test_sprint_preconditions_table(kw, ok):
    a = dict(source="mission", profile="sprint", deadman=True, battery=80.0, lidar=True, persons=True)
    a.update(kw)
    got, why = core.sprint_preconditions(a["source"], a["profile"], a["deadman"], a["battery"],
                                         a["lidar"], a["persons"], CFG)
    assert got is ok
    if a["profile"] == "sprint" and not ok:
        assert why.startswith("sprint denied")


def test_decide_sprint_only_raises_clear_cap():
    cmd = (2.0, 0.0, 0.0)
    assert decide([], 10.0, None, cmd, 0.0, 0.0, CFG, sprint=True).cmd_out[0] == pytest.approx(1.5)
    assert decide([], 10.0, None, cmd, 0.0, 0.0, CFG, sprint=False).cmd_out[0] == pytest.approx(1.0)
    # CAUTION person: sprint flag changes nothing
    a = decide([P(0.0, 3.0)], 10.0, None, cmd, 0.0, 0.0, CFG, sprint=True)
    assert a.level == CAUTION and a.cmd_out[0] == pytest.approx(0.6) and not a.sprint
    # person far ahead in the corridor (CLEAR by zones): denied
    b = decide([P(7.5, 0.5)], 10.0, None, (1.5, 0.0, 0.0), 0.0, 0.0, CFG, sprint=True)
    assert not b.sprint and b.cmd_out[0] <= 1.0 and "corridor" in b.reason
    # person behind: allowed
    c = decide([P(-6.0, 0.0)], 10.0, None, (1.5, 0.0, 0.0), 0.0, 0.0, CFG, sprint=True)
    assert c.sprint and c.cmd_out[0] == pytest.approx(1.5)
    # stale persons / no lidar: never
    assert not decide([], 10.0, None, cmd, 1.0, 0.0, CFG, sprint=True).sprint
    assert not decide([], None, None, cmd, 0.0, 0.0, CFG, sprint=True).sprint


def test_monotonic_and_non_sprint_unchanged_random():
    rnd = random.Random(7)
    for _ in range(400):
        persons = [P(rnd.uniform(-6, 6), rnd.uniform(-6, 6), vx=rnd.uniform(-2, 2), vy=rnd.uniform(-2, 2))
                   for _ in range(rnd.randint(0, 3))]
        cmd = (rnd.uniform(-3, 3), rnd.uniform(-1, 1), rnd.uniform(-2, 2))
        lidar = rnd.choice([None, 0.3, 2.0, 10.0])
        base = decide(persons, lidar, None, cmd, 0.0, 0.0, CFG, rear_clear_m=2.0)
        same = decide(persons, lidar, None, cmd, 0.0, 0.0, CFG, rear_clear_m=2.0, sprint=False)
        assert base == same
        for dec in (base, decide(persons, lidar, None, cmd, 0.0, 0.0, CFG, rear_clear_m=2.0, sprint=True)):
            for o, i in zip(dec.cmd_out, cmd):
                assert abs(o) <= abs(i) + 1e-9 and (o == 0.0 or math.copysign(1, o) == math.copysign(1, i))
        if not base.sprint:
            assert math.hypot(*base.cmd_out[:2]) <= CFG.clear_vmax + 1e-9


# -- action gating (pure) ------------------------------------------------------------------

def _act(name, **kw):
    a = dict(deadman_alive=True, persons=[], persons_fresh=True, level=CLEAR, lidar_fresh=True,
             front_clear_m=5.0, still_for_s=2.0)
    a.update(kw)
    return core.action_check(name, cfg=CFG, **a)


@pytest.mark.parametrize("name,kw,ok", [
    ("jump", {}, True), ("front_jump", {}, True), ("pounce", {}, True), ("front_pounce", {}, True),
    ("jump", {"deadman_alive": False}, False),
    ("jump", {"persons": [P(-3.0, 1.0)]}, False),            # 3.16 m, behind: still too close
    ("jump", {"persons": [P(3.6, 0.0)]}, True),
    ("jump", {"persons": [{"x": "?", "y": 1}]}, False),
    ("jump", {"persons_fresh": False}, False),
    ("jump", {"level": CAUTION}, False), ("jump", {"level": STOP}, False),
    ("jump", {"lidar_fresh": False}, False),
    ("jump", {"front_clear_m": 0.8}, False), ("jump", {"front_clear_m": None}, False),
    ("jump", {"still_for_s": 0.3}, False), ("jump", {"still_for_s": -0.2}, False),
    ("hello", {"deadman_alive": False, "persons": [P(1.0, 0.0)], "level": STOP}, True),
    ("sit", {"deadman_alive": False}, True), ("stand", {}, True), ("lay_down", {}, True),
    ("dance", {"persons": [P(1.5, 0.0)]}, False), ("dance", {"persons": [P(2.5, 0.0)]}, True),
    ("dance", {"persons_fresh": False}, False),
    ("back_flip", {}, False), ("front_flip", {}, False), ("damp", {}, False), ("handstand", {}, False),
])
def test_action_gating_table(name, kw, ok):
    assert _act(name, **kw)[0] is ok


# -- fleet tracks (pure) ---------------------------------------------------------------------

def test_fleet_tracks_transform_and_bubble():
    robots = [{"id": "go2", "x": 0.0, "y": 0.0}, {"id": "b", "x": 1.0, "y": 2.4, "vx": 0.0, "vy": -1.0},
              {"id": "far", "x": 50.0, "y": 0.0}, {"id": "bad", "x": None, "y": 1.0}]
    t = core.fleet_tracks(robots, "go2", (1.0, 1.0, math.pi / 2), 1.5, 0.8, 10.0)
    assert [r["gid"] for r in t] == ["robot:b"]
    r = t[0]
    # world (0, +1.4) relative, robot facing +y -> base (1.4, 0); shifted by 0.7
    assert r["x"] == pytest.approx(0.7, abs=1e-9) and r["y"] == pytest.approx(0.0, abs=1e-9)
    assert r["vx"] == pytest.approx(-1.0) and r["vy"] == pytest.approx(0.0, abs=1e-9)
    assert core.fleet_tracks(robots, "go2", None) == []
    assert not core.is_vulnerable(dict(r, height_m=1.0), SafetyConfig(child_from_z=True))


def test_fleet_robot_bubble_levels():
    for true_d, level in ((1.4, STOP), (2.0, SLOW), (2.6, SLOW), (3.0, CAUTION), (4.0, CAUTION), (4.5, CLEAR)):
        t = core.fleet_tracks([{"id": "b", "x": 0.0, "y": true_d}], "go2", (0.0, 0.0, 0.0))
        assert decide(t, 10.0, None, (0, 0, 0), 0.0, 0.0, CFG).level == level, true_d


def test_fleet_robot_not_subject_to_follow_rule():
    t = core.fleet_tracks([{"id": "b", "x": 3.0, "y": 0.0}], "go2", (0.0, 0.0, 0.0))   # eff 2.3 m
    dec = decide(t, 10.0, None, (0.3, 0.0, 0.0), 0.0, 0.0, CFG, min_person_dist_m=2.5)
    assert dec.cmd_out[0] > 0.0
    person = decide([P(2.3, 0.0)], 10.0, None, (0.3, 0.0, 0.0), 0.0, 0.0, CFG, min_person_dist_m=2.5)
    assert person.cmd_out[0] == 0.0


# -- HTTP API -------------------------------------------------------------------------------

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
    monkeypatch.setattr(app_mod, "SPRINT_SOURCES", ("mission", "gamepad"))
    monkeypatch.setattr(app_mod, "FLEET_AWARE", True)
    return app_mod, TestClient(app_mod.app), calls, g


def ready(g, persons=(), battery=80.0, deadman=True, lidar_pts=((6.0, 0.0, 0.3),), now=None):
    now = time.time() if now is None else now
    g.persons.ingest({"t": now, "persons": list(persons)}, recv_t=now)
    g.set_lidar(list(lidar_pts), now)
    g.set_battery(battery, now)
    if deadman:
        g.deadman_beat("ui", now)


SPRINT = {"vx": 2.0, "vy": 0.0, "vyaw": 0.0, "source": "mission", "profile": "sprint"}


def test_api_deadman_and_state(api):
    app_mod, client, calls, g = api
    r = client.post("/deadman", json={"client_id": "ui"}).json()
    assert r["deadman_alive"] and r["deadman_clients"] == ["ui"]
    client.post("/deadman", json={"client_id": "gamepad"})
    s = client.get("/state").json()
    assert s["deadman_alive"] and s["deadman_clients"] == ["gamepad", "ui"]
    assert client.post("/deadman", json={"client_id": ""}).status_code == 422
    client.post("/deadman", json={"client_id": "ui", "release": True})
    client.post("/deadman", json={"client_id": "gamepad", "release": True})
    assert client.get("/state").json()["deadman_alive"] is False
    assert calls == []                      # dead-man never commands motion by itself


def test_api_sprint_permitted(api):
    app_mod, client, calls, g = api
    ready(g)
    r = client.post("/move", json=SPRINT)
    assert r.status_code == 200 and r.json()["decision"]["sprint"] is True
    assert calls[-1] == ("/move", {"vx": 1.5, "vy": 0.0, "vyaw": 0.0, "sprint": True})
    assert client.get("/state").json()["sprint_active"] is True


@pytest.mark.parametrize("change", [
    {"deadman": False}, {"battery": 25.0}, {"battery": None}, {"persons": [P(6.0, 1.0)]},
    {"source": "navigation"}, {"source": "pursuit"}, {"profile": "normal"}, {"profile": None},
    {"stale_battery": True}, {"stale_persons": True},
])
def test_api_sprint_denied_falls_back_to_normal(api, change):
    app_mod, client, calls, g = api
    now = time.time()
    ready(g, persons=change.get("persons", ()), battery=change.get("battery", 80.0),
          deadman=change.get("deadman", True))
    if change.get("stale_battery"):
        g.set_battery(80.0, now - 10.0)
    if change.get("stale_persons"):
        g.persons.ingest({"t": now - 1.0, "persons": []}, recv_t=now)
    body = dict(SPRINT, **{k: v for k, v in change.items() if k in ("source", "profile")})
    client.post("/move", json=body)
    path, sent = calls[-1]
    assert path == "/move" and "sprint" not in sent
    assert sent["vx"] <= g.cfg.clear_vmax + 1e-9


def test_api_gamepad_sprint_needs_all_conditions(api):
    app_mod, client, calls, g = api
    ready(g)
    client.post("/move", json={"vx": 2.0, "source": "gamepad"})            # no profile: normal cap
    assert calls[-1] == ("/move", {"vx": 1.0, "vy": 0.0, "vyaw": 0.0})
    client.post("/move", json=dict(SPRINT, source="gamepad"))
    assert calls[-1][1].get("sprint") is True
    g.deadman.release("ui")
    client.post("/move", json={"vx": 2.0, "source": "gamepad", "profile": "sprint"})
    assert calls[-1] == ("/stop", None)                                     # dropped mid-sprint


def test_api_non_sprint_body_unchanged(api):
    app_mod, client, calls, g = api
    ready(g)
    client.post("/move", json={"vx": 0.3, "vyaw": 0.2, "source": "nav", "profile": "normal"})
    assert calls[-1] == ("/move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.2})


def test_deadman_drop_during_sprint_tick_sends_one_stop(api):
    app_mod, client, calls, g = api
    ready(g)
    client.post("/move", json=SPRINT)
    assert calls[-1][1]["sprint"] is True
    n = len(calls)
    g.tick(time.time())                          # still held: nothing
    assert len(calls) == n
    g.deadman.release("ui")
    g.tick(time.time())
    assert calls[n:] == [("/stop", None)]
    g.tick(time.time())                          # only one
    assert len(calls) == n + 1
    s = client.get("/state").json()
    assert s["sprint_active"] is False and s["sprint_latch"] is True and s["deadman_drops"] == 1
    # caller keeps sending sprint: latched -> /stop, never moves
    client.post("/move", json=SPRINT)
    assert calls[-1] == ("/stop", None)
    # operator holds the dead-man again: sprint allowed again
    g.deadman_beat("ui")
    client.post("/move", json=SPRINT)
    assert calls[-1][1].get("sprint") is True


def test_deadman_drop_detected_on_move_and_cleared_by_normal_profile(api):
    app_mod, client, calls, g = api
    ready(g)
    client.post("/move", json=SPRINT)
    g.deadman.beats["ui"] = time.time() - 1.0          # expired heartbeat
    r = client.post("/move", json=SPRINT)
    assert calls[-1] == ("/stop", None) and r.json()["forwarded"] == "stop"
    client.post("/move", json=dict(SPRINT, profile="normal"))
    assert calls[-1] == ("/move", {"vx": 1.0, "vy": 0.0, "vyaw": 0.0})
    assert g.sprint_latch is False


def test_sprint_profile_without_deadman_ever_is_normal_move(api):
    app_mod, client, calls, g = api
    ready(g, deadman=False)
    client.post("/move", json=SPRINT)
    assert calls[-1] == ("/move", {"vx": 1.0, "vy": 0.0, "vyaw": 0.0})


def test_api_action_jump_allowed_and_forwarded(api):
    app_mod, client, calls, g = api
    ready(g)
    r = client.post("/action/jump", json={"source": "mission"})
    assert r.status_code == 200 and r.json()["decision"] == "allow"
    assert calls[-1] == ("/action/jump", None)


@pytest.mark.parametrize("setup,reason", [
    (lambda g: g.deadman.release("ui"), "dead-man"),
    (lambda g: g.persons.ingest({"t": time.time(), "persons": [P(0.0, 3.0)]}), "person"),
    (lambda g: g.set_lidar([(0.6, 0.0, 0.3)], time.time()), "front clearance"),
    (lambda g: g.set_lidar([(6.0, 0.0, 0.3)], time.time() - 5.0), "level"),
])
def test_api_action_jump_denied_409(api, setup, reason):
    app_mod, client, calls, g = api
    ready(g)
    setup(g)
    r = client.post("/action/front_jump")
    assert r.status_code == 409 and r.json()["decision"] == "deny" and reason in r.json()["reason"]
    assert all(not p.startswith("/action") for p, _ in calls)


def test_api_action_jump_denied_while_moving(api):
    app_mod, client, calls, g = api
    ready(g)
    client.post("/move", json={"vx": 0.3, "source": "nav"})
    r = client.post("/action/jump")
    assert r.status_code == 409 and "still" in r.json()["reason"]
    # watchdog window + 0.5 s after the last command: allowed
    g.last_caller_t -= 1.1
    g.last_motion_t -= 1.1
    assert client.post("/action/jump").status_code == 200


def test_api_action_not_allowlisted_and_gesture(api):
    app_mod, client, calls, g = api
    ready(g, deadman=False)
    assert client.post("/action/back_flip").status_code == 409
    assert client.post("/action/hello").status_code == 200
    assert calls == [("/action/hello", None)]


def test_api_action_motion_refusal_passed_through(api, monkeypatch):
    app_mod, client, calls, g = api
    ready(g)
    monkeypatch.setattr(app_mod, "forwarder", lambda p, b=None: (409, {"error": "not armed"}))
    assert client.post("/action/sit").status_code == 409
    monkeypatch.setattr(app_mod, "forwarder", lambda p, b=None: (0, {"error": "down"}))
    assert client.post("/action/sit").status_code == 502


# -- fleet in the guard ---------------------------------------------------------------------------

def test_guard_fleet_robot_stops_and_ages_out(api, monkeypatch):
    app_mod, client, calls, g = api
    now = time.time()
    ready(g, now=now)
    g.set_pose(10.0, 10.0, 0.0, now)
    g.ingest_fleet({"t": now, "robots": [{"id": "go2", "x": 10.0, "y": 10.0},
                                         {"id": "b", "x": 11.2, "y": 10.0}]}, recv_t=now)
    client.post("/move", json={"vx": 0.3, "source": "nav"})
    assert calls[-1] == ("/stop", None)
    assert client.get("/state").json()["fleet_robots_n"] == 1
    # stale fleet message -> no-op (prev reset: skip the 0.5 s STOP hysteresis hold)
    g.ingest_fleet({"t": now - 5.0, "robots": [{"id": "b", "x": 11.2, "y": 10.0}]}, recv_t=now)
    g.prev = None
    client.post("/move", json={"vx": 0.3, "source": "nav"})
    assert calls[-1] == ("/move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0})
    # fleet-aware off -> no-op
    g.ingest_fleet({"t": time.time(), "robots": [{"id": "b", "x": 11.2, "y": 10.0}]})
    monkeypatch.setattr(app_mod, "FLEET_AWARE", False)
    g.prev = None
    client.post("/move", json={"vx": 0.3, "source": "nav"})
    assert calls[-1][0] == "/move"


def test_guard_fleet_needs_fresh_own_pose(api):
    app_mod, client, calls, g = api
    now = time.time()
    ready(g, now=now)
    g.set_pose(0.0, 0.0, 0.0, now - 2.0)
    g.ingest_fleet({"t": now, "robots": [{"id": "b", "x": 1.0, "y": 0.0}]}, recv_t=now)
    assert g._fleet(now) == []


def test_parse_core_state():
    app_mod = _load("safety_guard_app", os.path.join(SG, "app.py"))
    ok = {"battery": {"percent": 55.0}, "pose": {"x": 1, "y": 2, "yaw": 0.5},
          "link": {"tracked": True, "healthy": True}}
    assert app_mod.parse_core_battery(ok) == 55.0 and app_mod.parse_core_pose(ok) == (1.0, 2.0, 0.5)
    bad_link = dict(ok, link={"tracked": True, "healthy": False})
    assert app_mod.parse_core_battery(bad_link) is None and app_mod.parse_core_pose(bad_link) is None
    assert app_mod.parse_core_battery({"battery": {"percent": "x"}}) is None
    assert app_mod.parse_core_battery({"battery": {"percent": 101}}) is None
    assert app_mod.parse_core_pose({"pose": None}) is None
    assert app_mod.parse_core_battery(None) is None
