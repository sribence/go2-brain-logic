"""mc_motion (go2-brain-logic/mc_motion/motion.py): sprint flag, /capabilities,
/gait, /speed_level, and "flags off = unchanged behaviour".

motion.py imports flask at module level; when flask is not installed a tiny
stub (route registry + request/jsonify) is injected only for the import.
The SportClient is a fake object; the SDK import fails and is ignored by
motion.py's own init thread."""
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MOTION_PY = os.path.join(HERE, "..", "..", "mc_motion", "motion.py")

pytestmark = pytest.mark.skipif(not os.path.exists(MOTION_PY), reason="mc_motion/motion.py not found")


class _Req:
    method = "GET"
    _json = None

    def get_json(self, silent=False):
        return self._json


def _stub_flask():
    mod = types.ModuleType("flask")
    mod.request = _Req()

    class Flask:
        def __init__(self, name):
            self.routes = {}

        def route(self, rule, methods=None):
            def deco(fn):
                for m in (methods or ["GET"]):
                    self.routes[(m, rule)] = fn
                return fn
            return deco

    def jsonify(*args, **kw):
        return dict(args[0]) if args else dict(kw)

    mod.Flask, mod.jsonify = Flask, jsonify
    return mod


def _load_motion():
    name = "mc_motion_under_test"
    if name in sys.modules:
        return sys.modules[name]
    stubbed = False
    try:
        import flask  # noqa: F401
    except ImportError:
        sys.modules["flask"] = _stub_flask()
        stubbed = True
    try:
        spec = importlib.util.spec_from_file_location(name, MOTION_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if stubbed:
            del sys.modules["flask"]      # do not leak the stub to other tests
    mod._stubbed_flask = stubbed
    return mod


class FakeSportNew:
    """Method set of unitree_sdk2py >= 026978e (2025-07)."""

    def __init__(self):
        self.calls = []

    def Move(self, vx, vy, vyaw):
        self.calls.append(("Move", vx, vy, vyaw))
        return 0

    def SpeedLevel(self, level):
        self.calls.append(("SpeedLevel", level))
        return 0

    def FreeWalk(self):
        self.calls.append(("FreeWalk",))
        return 0

    def ClassicWalk(self, flag):
        self.calls.append(("ClassicWalk", flag))
        return 0

    def TrotRun(self):
        self.calls.append(("TrotRun",))
        return 0

    def FrontJump(self):
        return 0


class FakeSportOld(FakeSportNew):
    """<= 3d497a6: SwitchGait(int), EconomicGait(flag), FreeWalk(flag)."""

    def SwitchGait(self, t):
        self.calls.append(("SwitchGait", t))
        return 0

    def FreeWalk(self, flag):
        self.calls.append(("FreeWalk", flag))
        return 0

    TrotRun = None
    ClassicWalk = None


@pytest.fixture()
def mm(monkeypatch):
    m = _load_motion()
    sport = FakeSportNew()
    monkeypatch.setattr(m, "_sport", sport)
    monkeypatch.setattr(m, "_armed", True)
    monkeypatch.setattr(m, "_last_cmd_t", 0.0)
    monkeypatch.setattr(m, "_last_cmd", (0.0, 0.0, 0.0))
    for k in ("MOTION_ALLOW_SPRINT", "MOTION_ALLOW_GAIT"):
        monkeypatch.setattr(m, k, False)
    monkeypatch.setattr(m, "MAX_VX", 0.6)
    monkeypatch.setattr(m, "MAX_VX_SPRINT", 0.6)

    def call(method, path, json=None):
        if not m._stubbed_flask:
            c = m.app.test_client()
            r = c.open(path, method=method, json=json)
            return r.status_code, r.get_json()
        fn = m.app.routes[(method, path)]
        m.request.method, m.request._json = method, json
        res = fn()
        if isinstance(res, tuple):
            return res[1], res[0]
        return 200, res

    m._call = call
    m._fake = sport
    return m


def _old_move_clamp(m, body):
    """The pre-change /move clamp, verbatim semantics."""
    return {"vx": m._clamp(body.get("vx"), m.MAX_VX), "vy": m._clamp(body.get("vy"), m.MAX_VY),
            "vyaw": m._clamp(body.get("vyaw"), m.MAX_VYAW)}


# -- flags off: unchanged ------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {"vx": 1.0}, {"vx": 1.0, "sprint": True}, {"vx": -2.0, "vy": 0.9, "vyaw": 3.0, "sprint": True},
    {"vx": "nan"}, {"vx": 0.3, "sprint": "1"}, {},
])
def test_flags_off_move_is_unchanged(mm, body):
    code, data = mm._call("POST", "/move", body)
    assert code == 200
    assert data == {"ok": True, "applied": _old_move_clamp(mm, body)}


def test_flags_off_gait_and_speed_level_are_403(mm):
    assert mm._call("POST", "/gait", {"mode": "free_walk"})[0] == 403
    assert mm._call("POST", "/speed_level", {"level": 1})[0] == 403
    assert not any(c[0] in ("FreeWalk", "SpeedLevel") for c in mm._fake.calls)


def test_flags_off_sprint_ignored_even_if_sprint_limit_configured(mm, monkeypatch):
    monkeypatch.setattr(mm, "MAX_VX_SPRINT", 1.5)
    code, data = mm._call("POST", "/move", {"vx": 1.4, "sprint": True})
    assert data["applied"]["vx"] == pytest.approx(0.6)


def test_max_vx_sprint_defaults_to_max_vx(mm):
    src = open(MOTION_PY).read()
    assert 'MAX_VX_SPRINT = float(os.environ.get("MAX_VX_SPRINT", str(MAX_VX)))' in src
    assert 'os.environ.get("MOTION_ALLOW_SPRINT", "0") == "1"' in src
    assert 'os.environ.get("MOTION_ALLOW_GAIT", "0") == "1"' in src


# -- sprint flag on ------------------------------------------------------------------

def test_sprint_on_raises_vx_only_with_boolean_flag(mm, monkeypatch):
    monkeypatch.setattr(mm, "MOTION_ALLOW_SPRINT", True)
    monkeypatch.setattr(mm, "MAX_VX_SPRINT", 1.5)
    assert mm._call("POST", "/move", {"vx": 2.0, "sprint": True})[1]["applied"]["vx"] == pytest.approx(1.5)
    assert mm._call("POST", "/move", {"vx": 2.0, "sprint": "true"})[1]["applied"]["vx"] == pytest.approx(0.6)
    assert mm._call("POST", "/move", {"vx": 2.0})[1]["applied"]["vx"] == pytest.approx(0.6)
    # vy / vyaw limits are never widened by sprint
    d = mm._call("POST", "/move", {"vx": 0.0, "vy": 5.0, "vyaw": 5.0, "sprint": True})[1]["applied"]
    assert d["vy"] == pytest.approx(mm.MAX_VY) and d["vyaw"] == pytest.approx(mm.MAX_VYAW)


def test_sprint_still_needs_arm(mm, monkeypatch):
    monkeypatch.setattr(mm, "MOTION_ALLOW_SPRINT", True)
    monkeypatch.setattr(mm, "_armed", False)
    assert mm._call("POST", "/move", {"vx": 1.0, "sprint": True})[0] == 409


# -- capabilities ----------------------------------------------------------------------

def test_capabilities_introspects_methods(mm):
    code, d = mm._call("GET", "/capabilities")
    assert code == 200
    assert d["methods"]["SpeedLevel"]["exists"] and d["methods"]["FrontJump"]["exists"]
    assert not d["methods"]["SwitchGait"]["exists"] and not d["methods"]["Dance1"]["exists"]
    assert d["methods"]["ClassicWalk"]["arity"] == 1 and d["methods"]["FreeWalk"]["arity"] == 0
    assert d["flags"] == {"allow_sprint": False, "allow_gait": False}
    assert d["limits"]["max_vx_sprint"] == pytest.approx(0.6)
    assert d["gait_modes"]["free_walk"] and not d["gait_modes"]["switch_gait"]


def test_capabilities_without_sdk(mm, monkeypatch):
    monkeypatch.setattr(mm, "_sport", None)
    code, d = mm._call("GET", "/capabilities")
    assert code == 200 and d["sdk_ready"] is False
    assert not any(v["exists"] for v in d["methods"].values())


# -- gait / speed level (flag on) ---------------------------------------------------------

@pytest.fixture()
def gait_on(mm, monkeypatch):
    monkeypatch.setattr(mm, "MOTION_ALLOW_GAIT", True)
    return mm


def test_gait_calls_by_runtime_arity(gait_on):
    m = gait_on
    assert m._call("POST", "/gait", {"mode": "free_walk"})[0] == 200
    assert m._call("POST", "/gait", {"mode": "classic_walk", "enable": False})[0] == 200
    assert m._fake.calls[-2:] == [("FreeWalk",), ("ClassicWalk", False)]
    assert m._call("POST", "/gait", {"mode": "free_walk", "enable": False})[0] == 422
    assert m._call("POST", "/gait", {"mode": "economic"})[0] == 501       # not in this SDK
    assert m._call("POST", "/gait", {"mode": "handstand"})[0] == 422      # not reachable
    assert m._call("POST", "/gait", {"mode": "classic_walk", "enable": "yes"})[0] == 422


def test_gait_old_sdk_switch_gait(gait_on, monkeypatch):
    m = gait_on
    old = FakeSportOld()
    monkeypatch.setattr(m, "_sport", old)
    assert m._call("POST", "/gait", {"mode": "switch_gait", "value": 1})[0] == 200
    assert m._call("POST", "/gait", {"mode": "switch_gait", "value": 9})[0] == 422
    assert m._call("POST", "/gait", {"mode": "free_walk"})[0] == 200
    assert old.calls == [("SwitchGait", 1), ("FreeWalk", True)]
    assert m._call("POST", "/gait", {"mode": "trot_run"})[0] == 501


def test_gait_requires_arm_sdk_and_idle(gait_on, monkeypatch):
    m = gait_on
    monkeypatch.setattr(m, "_armed", False)
    assert m._call("POST", "/gait", {"mode": "free_walk"})[0] == 409
    monkeypatch.setattr(m, "_armed", True)
    import time
    monkeypatch.setattr(m, "_last_cmd_t", time.time())
    monkeypatch.setattr(m, "_last_cmd", (0.3, 0.0, 0.0))
    assert m._call("POST", "/gait", {"mode": "free_walk"})[0] == 409    # moving
    assert m._call("POST", "/speed_level", {"level": 0})[0] == 409
    monkeypatch.setattr(m, "_sport", None)
    assert m._call("POST", "/gait", {"mode": "free_walk"})[0] == 503


@pytest.mark.parametrize("level,code", [(-1, 200), (0, 200), (1, 200), (2, 422), (True, 422),
                                        ("1", 422), (None, 422)])
def test_speed_level_validation(gait_on, level, code):
    assert gait_on._call("POST", "/speed_level", {"level": level})[0] == code
