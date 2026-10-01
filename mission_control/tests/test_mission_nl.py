"""mission/nl.py: NL -> mission via (fake) Claude tool use; no network."""
import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mission import nl  # noqa: E402
from mission.nl import NLError, NLPlanner  # noqa: E402


def tool_use(inp, id_="tu_1"):
    return NS(type="tool_use", name=nl.TOOL_NAME, id=id_, input=inp)


def resp(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=1, output_tokens=1))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClient:
    def __init__(self, *responses):
        self.messages = FakeMessages(responses)
        self.beta = NS(messages=self.messages)


CTX = {
    "labels": [{"name": "kapu", "x": 12.0, "y": 3.0}, {"name": "garázs", "x": -4.0, "y": 8.0}],
    "zones": [{"id": "z1", "name": "kert", "kind": "patrol"}],
    "home": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    "pose": {"x": 1.0, "y": 2.0, "yaw": 0.0},
    "persons": [{"gid": 12, "x": 6.0, "y": 1.0}, {"gid": 7, "range_m": 15.0, "bearing_deg": 90}],
}


def plan(text, *responses, **kw):
    client = FakeClient(*responses)
    p = NLPlanner(client=client, **kw)
    return p.to_mission(text, CTX), client


# ----------------------------------------------------------------- Hungarian examples
def test_hu_run_to_gate_max_speed_is_goto_label_sprint():
    (m, expl, warns), client = plan(
        "fuss a kapuhoz max sebességgel",
        resp(tool_use({"name": "Kapu", "steps": [{"op": "goto_label", "label": "kapu", "speed": "sprint"}],
                       "explanation": "A kapuhoz futok sprint profillal."})))
    assert m["source"] == "nl"
    assert m["steps"] == [{"op": "goto_label", "label": "kapu", "speed": "sprint"}]
    assert m["on_fail"] == "stop"
    assert expl.startswith("A kapuhoz")
    assert warns == []
    req = client.messages.calls[0]
    assert req["model"] == nl.DEFAULT_MODEL
    assert req["messages"][0]["content"] == "fuss a kapuhoz max sebességgel"


def test_hu_jump_3m_forward_is_jump_to():
    (m, _, warns), _c = plan(
        "ugrálj ide 3 métert előre",
        resp(tool_use({"steps": [{"op": "jump_to", "x": 4.0, "y": 2.0}], "explanation": "3 m előre ugrálok."})))
    assert m["steps"][0]["op"] == "jump_to"
    assert (m["steps"][0]["x"], m["steps"][0]["y"]) == (4.0, 2.0)
    assert not any("jump" in w for w in warns)


def test_hu_follow_person_12_is_shadow_with_safe_distance():
    (m, _, warns), _c = plan(
        "kövesd a 12-es embert",
        resp(tool_use({"steps": [{"op": "shadow", "gid": 12, "dist_m": 3.0}], "explanation": "Követem."})))
    st = m["steps"][0]
    assert st["op"] == "shadow" and st["gid"] == 12 and st["dist_m"] >= 2.5
    assert warns == []


# ----------------------------------------------------------------- request shape
def test_request_has_tool_schema_cache_and_context():
    (_m, _, _w), client = plan("nézz körül", resp(tool_use({"steps": [{"op": "scan"}], "explanation": "ok"})))
    req = client.messages.calls[0]
    assert req["betas"] == [nl.FALLBACK_BETA] and req["fallbacks"] == "default"
    tools = req["tools"]
    assert len(tools) == 1 and tools[0]["name"] == "submit_mission"
    items = tools[0]["input_schema"]["properties"]["steps"]["items"]
    assert len(items["oneOf"]) == len(nl.STEP_SPECS)
    for s in items["oneOf"]:
        assert s["additionalProperties"] is False
        assert not {"vx", "vy", "vyaw", "v"} & set(s["properties"])
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "2.5 m" in req["system"][0]["text"]
    dyn = req["system"][1]["text"]
    assert "kapu" in dyn and '"gid": 12' in dyn and "forward_unit" in dyn
    assert "cache_control" not in req["system"][1]
    # Opus 5.5 rejects forced tool_choice -> auto with a single tool
    assert req["tool_choice"]["type"] == "auto"


def test_forced_tool_choice_on_older_model_and_no_beta_without_fallbacks():
    (_m, _, _w), client = plan("állj meg", resp(tool_use({"steps": [{"op": "wait", "s": 1}], "explanation": "ok"})),
                               model="claude-opus-4-8", fallbacks=False)
    req = client.messages.calls[0]
    assert req["tool_choice"] == {"type": "tool", "name": "submit_mission"}
    assert "betas" not in req and req["model"] == "claude-opus-4-8"


def test_env_model_override(monkeypatch):
    monkeypatch.setenv("NL_MODEL", "claude-sonnet-5-5")
    assert NLPlanner(client=object()).model == "claude-sonnet-5-5"


def test_person_range_bearing_from_xy():
    n = nl.normalize_context(CTX)
    p12 = [p for p in n["persons"] if p["gid"] == 12][0]
    assert p12["range_m"] == pytest.approx(6.1, abs=0.05)
    assert p12["bearing_deg"] == pytest.approx(9, abs=1)
    assert n["persons"][0]["gid"] == 12  # sorted by range


# ----------------------------------------------------------------- safety post-processing
def test_sprint_without_explicit_request_downgraded():
    (m, _, warns), _c = plan("menj a kapuhoz",
                             resp(tool_use({"steps": [{"op": "goto_label", "label": "kapu", "speed": "sprint"}],
                                            "explanation": "x"})))
    assert m["steps"][0]["speed"] == "normal"
    assert any("sprint" in w for w in warns)


def test_jump_without_explicit_request_becomes_goto():
    (m, _, warns), _c = plan("menj 3 métert előre",
                             resp(tool_use({"steps": [{"op": "jump_to", "x": 4.0, "y": 2.0}], "explanation": "x"})))
    assert m["steps"][0]["op"] == "goto"
    assert any("jump" in w for w in warns)


def test_goal_near_person_and_unknown_label_warn():
    (m, _, warns), _c = plan("go next to the man", resp(tool_use({
        "steps": [{"op": "goto", "x": 7.0, "y": 3.0}, {"op": "goto_label", "label": "tető"}],
        "explanation": "x"})))
    assert any("person 12" in w for w in warns)
    assert any("unknown label" in w for w in warns)


# ----------------------------------------------------------------- validation
def test_raw_velocity_rejected_then_repaired():
    bad = tool_use({"steps": [{"op": "goto", "x": 1, "y": 1, "vx": 1.5}], "explanation": "x"}, "tu_bad")
    good = tool_use({"steps": [{"op": "goto", "x": 1, "y": 1}], "explanation": "x"}, "tu_good")
    (m, _, _w), client = plan("menj ide", resp(bad), resp(good))
    assert m["steps"] == [{"op": "goto", "x": 1, "y": 1}]
    second = client.messages.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert second[-1]["content"][0]["tool_use_id"] == "tu_bad"
    assert second[-1]["content"][0]["is_error"] is True
    assert "vx" in second[-1]["content"][0]["content"]


@pytest.mark.parametrize("step", [
    {"op": "goto", "x": 1, "y": 1, "speed": 1.2},          # numeric speed
    {"op": "fly", "x": 1},                                  # unknown op
    {"op": "shadow"},                                       # missing gid
    {"op": "shadow", "gid": 3, "dist_m": 1.0},              # < 2.5 (schema min)
    {"op": "patrol"},                                       # neither points nor area
    {"op": "goto", "x": 1, "y": 1, "zone": "z25"},          # unknown zone value
    {"op": "capture", "cams": "front"},                     # cams neither "all" nor list
    {"op": "wait_for", "event": "sunrise"},
    {"op": "wait", "s": "five"},
])
def test_invalid_steps_raise_422(step):
    client = FakeClient(resp(tool_use({"steps": [step], "explanation": "x"})))
    with pytest.raises(NLError) as ei:
        NLPlanner(client=client, max_repairs=0).to_mission("bármi", CTX)
    assert ei.value.status == 422 and ei.value.errors


def test_postprocess_clamps_shadow_distance():
    steps, warns = nl.postprocess([{"op": "escort", "gid": 12, "dist_m": 1.0}], "kísérd", nl.normalize_context(CTX))
    assert steps[0]["dist_m"] == 2.5 and warns


def test_check_schema_accepts_all_example_steps():
    ok = [{"op": "follow_path", "points": [[0, 0], [1, 1]], "speed": "precise", "smooth": True},
          {"op": "look_at", "x": 1, "y": 2}, {"op": "action", "name": "hello"}, {"op": "say", "text": "Állj!"},
          {"op": "watch", "gid": 1, "timeout_s": 30}, {"op": "patrol", "zone": "kert", "loops": 2, "pass_zone": "z60"},
          {"op": "explore"}, {"op": "return_home", "speed": "stealth"}]
    assert nl.validate_mission({"name": "x", "source": "nl", "steps": ok, "on_fail": "return_home"}) == []


# ----------------------------------------------------------------- errors
def test_text_reply_is_clarification():
    client = FakeClient(resp(NS(type="text", text="Melyik kapura gondolsz?"), stop="end_turn"))
    with pytest.raises(NLError) as ei:
        NLPlanner(client=client).to_mission("menj a kapuhoz", CTX)
    assert ei.value.status == 422 and "Melyik kapura" in ei.value.message


def test_refusal():
    client = FakeClient(resp(stop="refusal"))
    with pytest.raises(NLError) as ei:
        NLPlanner(client=client).to_mission("x", CTX)
    assert ei.value.status == 422


def test_api_exception_mapped():
    class Boom(Exception):
        status_code = 529
    client = FakeClient(Boom("overloaded"))
    with pytest.raises(NLError) as ei:
        NLPlanner(client=client).to_mission("x", CTX)
    assert ei.value.status == 503 and "Claude API" in ei.value.message


def test_no_key_is_503(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    p = NLPlanner()
    assert p.available()[0] is False
    with pytest.raises(NLError) as ei:
        p.to_mission("menj haza", CTX)
    assert ei.value.status == 503 and "ANTHROPIC_API_KEY" in ei.value.message


def test_empty_text():
    with pytest.raises(NLError):
        NLPlanner(client=FakeClient()).to_mission("  ", CTX)


def test_world_provider_merged():
    client = FakeClient(resp(tool_use({"steps": [{"op": "goto_label", "label": "pajta"}], "explanation": "x"})))
    p = NLPlanner(client=client, world_provider=lambda: {"labels": [{"name": "pajta", "x": 1, "y": 1}]})
    m, _, warns = p.to_mission("menj a pajtához", {})
    assert "pajta" in client.messages.calls[0]["system"][1]["text"]
    assert warns == []


# ----------------------------------------------------------------- CONTRACT 9: zone + new ops
CTX9 = dict(CTX, labels=CTX["labels"] + [{"name": "ajtó", "x": 5.0, "y": -2.0}],
            zones=CTX["zones"] + [{"id": "z2", "name": "parkoló", "kind": "patrol"}])


def plan9(text, inp):
    client = FakeClient(resp(tool_use(inp)))
    return NLPlanner(client=client).to_mission(text, CTX9), client


def test_hu_door_request_passage():
    (m, _, warns), _c = plan9("menj az ajtóhoz, és kérd meg, hogy engedjenek ki", {
        "steps": [{"op": "goto_label", "label": "ajtó", "zone": "fine"},
                  {"op": "request_passage", "text": "Kérem, engedjenek ki!", "on_timeout": "alert"}],
        "explanation": "Az ajtóhoz megyek és átengedést kérek."})
    assert m["steps"][0] == {"op": "goto_label", "label": "ajtó", "zone": "fine"}
    assert m["steps"][1]["op"] == "request_passage" and m["steps"][1]["text"]
    assert warns == []


def test_hu_photo_of_gate_capture_fine():
    (m, _, warns), _c = plan9("készíts egy pontos fotót a kapuról", {
        "steps": [{"op": "capture", "x": 10.0, "y": 3.0, "yaw": 0.0, "label": "kapu",
                   "cams": ["rgb_front", "th_front_narrow"], "mode": "max"}],
        "explanation": "A kapu elé állok és fotózok."})
    st = m["steps"][0]
    assert st["op"] == "capture" and st["label"] == "kapu" and st["mode"] == "max"
    assert (st["x"], st["y"]) == (10.0, 3.0)
    assert warns == []


def test_hu_patrol_parking_z30():
    (m, _, warns), _c = plan9("fuss körbe a parkolón 30 centis zónával", {
        "steps": [{"op": "patrol", "zone": "parkoló", "pass_zone": "z30", "speed": "sprint"}],
        "explanation": "Körbejárom a parkolót."})
    st = m["steps"][0]
    assert st["op"] == "patrol" and st["pass_zone"] == "z30" and st["zone"] == "parkoló"
    assert st["speed"] == "sprint"  # "fuss" = explicit run
    assert warns == []


def test_zone_in_tool_schema_for_moving_steps():
    schemas = {s["properties"]["op"]["const"]: s for s in
               nl.tool_input_schema()["properties"]["steps"]["items"]["oneOf"]}
    assert schemas["patrol"]["properties"]["pass_zone"]["enum"] == nl.ZONES
    for op in ("goto", "follow_path", "goto_label", "return_home"):
        assert schemas[op]["properties"]["zone"]["enum"] == ["fine", "z10", "z30", "z60", "z100"]
    for op in ("capture", "request_passage", "wait_for", "record"):
        assert op in schemas


def test_record_and_wait_for_validate():
    assert nl.validate_mission({"source": "nl", "steps": [
        {"op": "record", "on": True, "mode": "evidence"}, {"op": "wait_for", "event": "person_gone", "timeout_s": 60},
        {"op": "follow_path", "points": [[0, 0], [2, 0]], "zones": ["z10", "fine"]}]}) == []


def test_capture_unknown_label_warns():
    (m, _, warns), _c = plan9("fotózd le a tetőt", {
        "steps": [{"op": "capture", "label": "tető", "cams": "all"}], "explanation": "x"})
    assert any("unknown label" in w for w in warns)
