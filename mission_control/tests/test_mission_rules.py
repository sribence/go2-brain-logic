"""mission/rules.py: triggers, dwell, cooldown, dedup, modality, templates, safety."""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mission import rules as R  # noqa: E402
from mission import world as W  # noqa: E402

NIGHT = dt.datetime(2026, 10, 1, 23, 0).timestamp()
NOON = dt.datetime(2026, 10, 1, 12, 0).timestamp()
YARD = [[4, -2], [8, -2], [8, 2], [4, 2]]


def person(gid, x, y, mod=("rgb", "thermal"), conf=0.9):
    return {"gid": gid, "x": x, "y": y, "z": 0.9, "vx": 0, "vy": 0, "conf": conf,
            "modality": list(mod), "height_m": 1.7}


@pytest.fixture
def world(tmp_path):
    w = W.WorldStore(str(tmp_path / "w.json"))
    w.upsert_zone({"id": "yard", "name": "yard", "kind": "watch", "polygon": YARD})
    return w


def make(world, rules, **kw):
    sub, alerts = [], []
    eng = R.RuleEngine(world, lambda m: sub.append(m) or "m%d" % len(sub), clock=lambda: NIGHT,
                       alert_fn=alerts.append, rules=rules, validate_schema=False, **kw)
    return eng, sub, alerts


IN_ZONE = {"id": "r1", "when": {"event": "person_in_zone", "zone": "yard", "dwell_s": 2.0},
           "then": [{"op": "say", "text": "Person {gid} in {zone}"},
                    {"op": "watch", "gid": "{gid}", "timeout_s": 30}],
           "cooldown_s": 10}


def test_dwell_then_fire_with_templates(world):
    eng, sub, alerts = make(world, [IN_ZONE])
    assert eng.on_persons([person(7, 6, 0)], (0, 0, 0), NIGHT) == []
    assert eng.on_persons([person(7, 6, 0)], (0, 0, 0), NIGHT + 1.0) == []
    out = eng.on_persons([person(7, 6, 0)], (0, 0, 0), NIGHT + 2.1)
    assert len(out) == 1 and out[0]["gid"] == 7 and out[0]["zone"] == "yard"
    assert out[0]["mission_id"] == "m1" and alerts == out
    m = sub[0]
    assert m["source"] == "rule"
    assert m["steps"][0]["text"] == "Person 7 in yard"
    assert m["steps"][1]["gid"] == 7 and isinstance(m["steps"][1]["gid"], int)
    assert set(out[0]) >= {"t", "rule_id", "kind", "zone", "gid", "x", "y", "modality", "message"}


def test_dedup_per_gid_and_cooldown(world):
    eng, sub, _ = make(world, [dict(IN_ZONE, when=dict(IN_ZONE["when"], dwell_s=0))])
    assert len(eng.on_persons([person(1, 6, 0)], None, NIGHT)) == 1
    assert eng.on_persons([person(1, 6, 0)], None, NIGHT + 20) == []          # same gid, still inside
    assert eng.on_persons([person(1, 6, 0), person(2, 5, 1)], None, NIGHT + 21) != []  # new gid, cooldown over
    assert eng.on_persons([person(3, 5, 1)], None, NIGHT + 22) == []          # cooldown (10 s)
    assert len(eng.on_persons([person(3, 5, 1)], None, NIGHT + 32)) == 1      # fires after cooldown
    # gid 1 leaves (beyond grace) and comes back -> re-armed
    eng.on_persons([], None, NIGHT + 40)
    assert len(eng.on_persons([person(1, 6, 0)], None, NIGHT + 50)) == 1
    assert len(sub) == 4


def test_dwell_survives_short_dropout_but_resets_on_leave(world):
    eng, _, _ = make(world, [IN_ZONE])
    eng.on_persons([person(1, 6, 0)], None, NIGHT)
    eng.on_persons([], None, NIGHT + 0.5)                                      # flicker < grace
    assert len(eng.on_persons([person(1, 6, 0)], None, NIGHT + 2.0)) == 1
    eng2, _, _ = make(world, [IN_ZONE])
    eng2.on_persons([person(1, 6, 0)], None, NIGHT)
    eng2.on_persons([person(1, 0, 0)], None, NIGHT + 1.5)                     # left zone
    eng2.on_persons([person(1, 0, 0)], None, NIGHT + 3.0)
    assert eng2.on_persons([person(1, 6, 0)], None, NIGHT + 3.5) == []         # dwell restarted


def test_pose_transform_used(world):
    eng, _, _ = make(world, [dict(IN_ZONE, when=dict(IN_ZONE["when"], dwell_s=0))])
    # robot at (6, -6) facing +y; person 6 m ahead in base -> world (6, 0) = yard
    out = eng.on_persons([person(4, 6, 0)], {"x": 6, "y": -6, "yaw": 1.5707963}, NIGHT)
    assert len(out) == 1 and abs(out[0]["x"] - 6) < 1e-3 and abs(out[0]["y"]) < 1e-3
    assert eng.on_persons([person(5, 6, 0)], {"x": 0, "y": 6, "yaw": 0}, NIGHT + 60) == []


def test_thermal_only_modality_filter_and_hours(world):
    rule = {"id": "th", "when": {"event": "person_in_zone", "zone": "yard", "modality": "thermal",
                                 "modality_only": True, "hours": "22:00-06:00", "min_conf": 0.5},
            "then": [], "alert": True, "cooldown_s": 0}
    eng, _, alerts = make(world, [rule])
    assert eng.on_persons([person(1, 6, 0, mod=("rgb",))], None, NIGHT) == []
    assert eng.on_persons([person(2, 6, 0, mod=("rgb", "thermal"))], None, NIGHT) == []   # not thermal-only
    assert eng.on_persons([person(3, 6, 0, mod=("thermal",), conf=0.3)], None, NIGHT) == []
    out = eng.on_persons([person(4, 6, 0, mod=("thermal",))], None, NIGHT)
    assert len(out) == 1 and out[0]["modality"] == "thermal" and out[0]["mission_id"] is None
    assert eng.on_persons([person(5, 6, 0, mod=("thermal",))], None, NOON) == []          # outside hours
    any_th = dict(rule, id="th2", when=dict(rule["when"], modality_only=False))
    eng2, _, _ = make(world, [any_th])
    assert len(eng2.on_persons([person(2, 6, 0, mod=("rgb", "thermal"))], None, NIGHT)) == 1


def test_zone_active_hours_respected(world):
    world.upsert_zone({"id": "day", "name": "day", "kind": "watch", "polygon": YARD, "active_hours": "08:00-18:00"})
    rule = {"id": "d", "when": {"event": "person_in_zone", "zone": "day"}, "then": [], "cooldown_s": 0}
    eng, _, _ = make(world, [rule])
    assert eng.on_persons([person(1, 6, 0)], None, NIGHT) == []
    assert len(eng.on_persons([person(2, 6, 0)], None, NOON)) == 1


def test_sprint_and_jump_rejected():
    assert R.check_rule_steps([{"op": "goto", "x": 1, "y": 1, "speed": "sprint"}])
    assert R.check_rule_steps([{"op": "jump_to", "x": 1, "y": 1}])
    assert R.check_rule_steps([{"op": "action", "name": "pounce"}])
    assert R.check_rule_steps([{"op": "goto", "x": 1, "y": 1, "speed": "stealth"}, {"op": "say", "text": "hi"}]) == []
    bad = {"id": "b", "when": {"event": "person_in_zone", "zone": "yard"},
           "then": [{"op": "goto", "x": "{x}", "y": "{y}", "speed": "sprint"}]}
    assert any("sprint" in e for e in R.validate_rule(bad))


def test_sprint_rule_never_submitted(world):
    bad = {"id": "b", "when": {"event": "person_in_zone", "zone": "yard"},
           "then": [{"op": "goto", "x": "{x}", "y": "{y}", "speed": "sprint"}]}
    eng, sub, _ = make(world, [bad])
    assert "b" in eng.rule_errors and eng.rules() == []
    assert eng.on_persons([person(1, 6, 0)], None, NIGHT) == [] and sub == []
    # a template that would produce sprint at runtime is also blocked
    sneaky = {"id": "s", "when": {"event": "gesture", "gesture": "wave"},
              "then": [{"op": "goto", "x": "{x}", "y": "{y}", "speed": "{gesture}"}]}
    eng2, sub2, _ = make(world, [sneaky])
    eng2._seen[1] = dict(person(1, 6, 0), _t=NIGHT)
    out = eng2.on_gesture({"gid": 1, "gesture": "wave", "t": NIGHT})
    assert sub2 and sub2[0]["steps"][0]["speed"] == "wave"  # not sprint -> fine (schema decides)
    m, errs = eng2.build_mission(sneaky, dict(eng2._ctx(sneaky, NIGHT, eng2._seen[1], None), gesture="sprint"))
    assert any("sprint" in e for e in errs)
    assert out[0]["error"] is None


def test_gesture_wave_to_goto_approach(world):
    rule = {"id": "wave", "when": {"event": "gesture", "gesture": "wave"}, "approach_m": 3.0,
            "then": [{"op": "goto", "x": "{approach_x}", "y": "{approach_y}", "speed": "precise"},
                     {"op": "look_at", "x": "{x}", "y": "{y}"}], "cooldown_s": 5}
    eng, sub, _ = make(world, [rule])
    eng.on_persons([person(9, 10, 0)], (0, 0, 0), NIGHT)
    out = eng.on_gesture({"gid": 9, "gesture": "wave", "t": NIGHT + 0.1})
    assert len(out) == 1 and out[0]["kind"] == "gesture"
    goto = sub[0]["steps"][0]
    assert abs(goto["x"] - 7.0) < 1e-6 and abs(goto["y"]) < 1e-6           # 3 m short of the person
    assert sub[0]["steps"][1] == {"op": "look_at", "x": 10.0, "y": 0.0}
    assert eng.on_gesture({"gid": 9, "gesture": "wave", "t": NIGHT + 1}) == []   # cooldown
    assert eng.on_gesture({"gid": 9, "gesture": "point", "t": NIGHT + 10}) == []
    # unknown person -> unresolved coordinates -> no mission, error reported
    out = eng.on_gesture({"gid": 77, "gesture": "wave", "t": NIGHT + 20})
    assert out[0]["error"] and len(sub) == 1


def test_approach_never_closer_than_min(world):
    eng, _, _ = make(world, [])
    eng._pose = (0.0, 0.0, 0.0)
    ax, ay = eng._approach(10.0, 0.0, 1.0)
    assert abs(ax - 7.5) < 1e-9
    ax, ay = eng._approach(2.0, 0.0, 3.0)  # already closer: stay
    assert (ax, ay) == (0.0, 0.0)


def test_person_count_over(world):
    rule = {"id": "crowd", "when": {"event": "person_count_over", "count": 2, "zone": "yard"},
            "then": [{"op": "say", "text": "{count} people"}], "cooldown_s": 0,
            "message": "{count} in {zone}"}
    eng, sub, _ = make(world, [rule])
    two = [person(1, 5, 0), person(2, 6, 0), person(3, 0, 0)]
    assert eng.on_persons(two, None, NIGHT) == []
    three = [person(1, 5, 0), person(2, 6, 0), person(3, 7, 0)]
    out = eng.on_persons(three, None, NIGHT + 1)
    assert len(out) == 1 and out[0]["message"] == "3 in yard" and abs(out[0]["x"] - 6) < 1e-9
    assert sub[0]["steps"][0]["text"] == "3 people"
    assert eng.on_persons(three, None, NIGHT + 2) == []          # latched
    eng.on_persons(two, None, NIGHT + 5)                          # re-arm
    assert len(eng.on_persons(three, None, NIGHT + 6)) == 1


def test_schedule_rule_on_tick(world):
    rule = {"id": "patrol", "when": {"event": "schedule", "cron": "*/30 22-06"},
            "then": [{"op": "patrol", "zone": "{zone}", "loops": 1, "speed": "stealth"}],
            "cooldown_s": 0}
    rule["when"]["zone"] = "yard"
    eng, sub, _ = make(world, [rule])
    t = dt.datetime(2026, 10, 1, 23, 30, 2).timestamp()
    assert len(eng.on_tick(t)) == 1 and sub[0]["steps"][0]["zone"] == "yard"
    assert eng.on_tick(t + 10) == []
    assert eng.on_tick(dt.datetime(2026, 10, 1, 12, 0).timestamp()) == []
    assert len(eng.on_tick(dt.datetime(2026, 10, 2, 0, 0).timestamp())) == 1


def test_lost_target(world):
    rule = {"id": "lost", "when": {"event": "lost_target", "gid": "target", "timeout_s": 3},
            "then": [{"op": "goto", "x": "{x}", "y": "{y}", "speed": "normal"}], "cooldown_s": 0}
    eng, sub, _ = make(world, [rule])
    eng.target_gid = 5
    eng.on_persons([person(5, 6, 1), person(6, 2, 2)], None, NIGHT)
    assert eng.on_tick(NIGHT + 2) == []
    out = eng.on_tick(NIGHT + 3.5)
    assert len(out) == 1 and out[0]["gid"] == 5 and sub[0]["steps"][0]["x"] == 6.0
    assert eng.on_tick(NIGHT + 5) == []                       # once per loss
    eng.on_persons([person(5, 6, 1)], None, NIGHT + 6)         # seen again -> re-armed
    assert len(eng.on_tick(NIGHT + 10)) == 1


def test_rules_from_world_reload_and_payload_form(world):
    eng = R.RuleEngine(world, None, clock=lambda: NIGHT, validate_schema=False)
    assert eng.rules() == []
    world.upsert_rule({"id": "x", "when": {"event": "person_in_zone", "zone": "yard"}, "then": []})
    out = eng.on_persons({"t": NIGHT, "persons": [person(1, 6, 0)]})
    assert len(out) == 1 and out[0]["rule_id"] == "x"
    world.upsert_rule({"id": "x", "enabled": False, "when": {"event": "person_in_zone", "zone": "yard"}, "then": []})
    assert eng.on_persons([person(2, 6, 0)], None, NIGHT + 100) == []


def test_submit_failure_reported(world):
    def boom(m):
        raise RuntimeError("busy")
    eng = R.RuleEngine(world, boom, rules=[dict(IN_ZONE, when=dict(IN_ZONE["when"], dwell_s=0))],
                       validate_schema=False)
    out = eng.on_persons([person(1, 6, 0)], None, NIGHT)
    assert "busy" in out[0]["error"]


def test_render_steps_types():
    steps = R.render_steps([{"op": "goto", "x": "{x}", "note": "at {x},{y} {unknown}", "pts": [["{x}", 1]]}],
                           {"x": 1.5, "y": 2.0})
    assert steps[0]["x"] == 1.5 and steps[0]["note"] == "at 1.50,2.00 {unknown}" and steps[0]["pts"] == [[1.5, 1]]


def test_evidence_prepends_record_and_starts_recording_immediately(world):
    rec = []
    rule = dict(IN_ZONE, id="ev", evidence=True, when=dict(IN_ZONE["when"], dwell_s=0))
    eng, sub, _ = make(world, [rule], start_recording=rec.append)
    out = eng.on_persons([person(3, 6, 0, mod=("thermal",))], None, NIGHT)
    assert len(rec) == 1 and "ev" in rec[0] and out[0]["evidence"] is True
    assert sub[0]["steps"][0] == {"op": "record", "on": True, "mode": "evidence"}
    assert [s["op"] for s in sub[0]["steps"]] == ["record", "say", "watch"]
    # evidence-only rule (no then) still records, submits nothing
    eng2, sub2, _ = make(world, [{"id": "e2", "evidence": True, "alert": False,
                                  "when": {"event": "person_in_zone", "zone": "yard"}}],
                         start_recording=rec.append)
    assert R.validate_rule(eng2.rules()[0]) == []
    eng2.on_persons([person(4, 6, 0)], None, NIGHT)
    assert len(rec) == 2 and sub2 == []


def test_recorder_failure_does_not_block_mission(world):
    def boom(reason):
        raise IOError("omni down")
    rule = dict(IN_ZONE, id="ev", evidence=True, when=dict(IN_ZONE["when"], dwell_s=0))
    eng, sub, _ = make(world, [rule], start_recording=boom)
    out = eng.on_persons([person(3, 6, 0)], None, NIGHT)
    assert "omni down" in out[0]["error"] and len(sub) == 1


def test_new_ops_allowed_in_rules():
    steps = [{"op": "record", "on": True}, {"op": "capture", "cams": "all", "mode": "max"},
             {"op": "request_passage", "text": "Please open the door"},
             {"op": "wait_for", "event": "path_clear", "timeout_s": 60}]
    assert R.check_rule_steps(steps) == []


def test_example_rules_valid(world):
    for r in R.EXAMPLE_RULES:
        assert R.validate_rule(r) == [], r["id"]
    ex = dict(R.EXAMPLE_RULES[0], when=dict(R.EXAMPLE_RULES[0]["when"], zone="yard"))
    rec = []
    eng, sub, _ = make(world, [ex], start_recording=rec.append)
    eng.on_persons([person(1, 6, 0, mod=("rgb",))], None, NIGHT)
    assert eng.on_persons([person(1, 6, 0, mod=("rgb",))], None, NIGHT + 5) == []   # rgb only: ignored
    eng.on_persons([person(2, 6, 0, mod=("thermal",))], None, NIGHT)
    out = eng.on_persons([person(2, 6, 0, mod=("thermal",))], None, NIGHT + 3.5)
    assert len(out) == 1 and rec and sub[0]["steps"][-1]["gid"] == 2
