"""Natural-language -> mission (M6, CONTRACT.md section 6).

``NLPlanner(client=None, model=..., world_provider=None).to_mission(text, context)``
returns ``(mission_dict, explanation, warnings)`` or raises :class:`NLError`
(``.status`` is an HTTP-ish code: 503 = no key / SDK / API unavailable,
502 = upstream API error, 422 = refusal, clarification or invalid mission).

The Claude API is called through the Anthropic Python SDK (lazy import,
``ANTHROPIC_API_KEY``) with a single client tool ``submit_mission`` whose
``input_schema`` is the mission step schema (``oneOf`` by ``op``; raw velocities
are impossible: ``additionalProperties: false`` and speed is a profile name).
The result is validated (``mission.schema.validate`` when importable, plus the
local schema check), safety-postprocessed, and always gets ``source="nl"`` so the
mission service puts it into ``pending_approval``.

Env: ``NL_MODEL`` (default ``claude-opus-5-5``), ``NL_EFFORT`` (default
``medium``), ``NL_FALLBACKS`` (``1`` = server-side refusal fallbacks, default on),
``NL_MAX_REPAIRS`` (validation-repair round trips, default 1).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_MODEL = "claude-opus-5-5"
TOOL_NAME = "submit_mission"
SPEEDS = ["stealth", "precise", "normal", "sprint"]
ACTIONS = ["hello", "wave", "greet", "stretch", "sit", "stand", "stand_up", "lay_down", "balance",
           "heart", "dance", "dance2", "shake"]  # = mission.schema.ACTIONS (no jumps/flips)
MIN_PERSON_DIST_M = 2.5
# Models that reject forced tool_choice ("any"/"tool") with a 400 -> auto + instruction.
NO_FORCED_TOOL_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1",
                         "claude-mythos-5-1")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_NUM = {"type": "number"}
_INT = {"type": "integer"}
_SPEED = {"type": "string", "enum": SPEEDS,
          "description": "Speed profile. 'sprint' ONLY if the operator explicitly asked for max/full speed or running."}
_POINTS = {"type": "array", "minItems": 1, "maxItems": 200,
           "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM},
           "description": "World-frame [[x, y], ...] in metres."}
_GID = {"type": "integer", "minimum": 0, "description": "Global person track id (gid)."}
_PDIST = {"type": "number", "minimum": MIN_PERSON_DIST_M, "maximum": 15.0,
          "description": "Kept distance to the person in metres, never below 2.5."}
_TIMEOUT = {"type": "number", "minimum": 1, "maximum": 3600}
ZONES = ["fine", "z10", "z30", "z60", "z100"]
_ZONE = {"type": "string", "enum": ZONES,
         "description": ("Arrival precision (industrial-robot style, CONTRACT 9.1): 'fine' = stop exactly "
                         "(<=5 cm, <=3 deg) and settle - use before photos, measurements, doors; "
                         "'z10'/'z30'/'z60'/'z100' = fly-by within 10/30/60/100 cm, no slow-down, "
                         "corners blended. Omit for the default (z30, last point fine).")}
CAMS = ["rgb_front", "rgb_left", "rgb_right", "rgb_rear", "th_front_narrow", "th_front_wide"]

# op -> (required props, optional props, description)
STEP_SPECS = {
    "goto": ({"x": _NUM, "y": _NUM},
             {"yaw": {"type": "number", "description": "Final heading, radians (world)."},
              "speed": _SPEED, "tol_m": {"type": "number", "minimum": 0.05, "maximum": 3.0},
              "zone": _ZONE},
             "Walk to a world point (path planned around obstacles)."),
    "follow_path": ({"points": _POINTS}, {"speed": _SPEED, "smooth": {"type": "boolean"}, "zone": _ZONE,
                                          "zones": {"type": "array", "items": _ZONE, "maxItems": 200,
                                                    "description": "Optional per-point zone, same length as points."}},
                    "Follow a polyline of world points."),
    "jump_to": ({"x": _NUM, "y": _NUM}, {"max_jumps": {"type": "integer", "minimum": 1, "maximum": 12}},
                "Reach a world point by a series of forward jumps (~0.6 m each). ONLY on explicit request."),
    "look_at": ({"x": _NUM, "y": _NUM}, {"z": _NUM}, "Turn in place to face a world point."),
    "scan": ({}, {"deg": {"type": "number", "minimum": 10, "maximum": 1080},
                  "speed_dps": {"type": "number", "minimum": 5, "maximum": 90}},
             "Rotate in place to look around."),
    "action": ({"name": {"type": "string", "enum": ACTIONS}}, {}, "Built-in motion (hello = wave paw)."),
    "wait": ({"s": {"type": "number", "minimum": 0, "maximum": 3600}}, {}, "Wait in place."),
    "say": ({"text": {"type": "string", "minLength": 1, "maxLength": 300}}, {},
            "Speak through the robot's speaker (use the operator's language)."),
    "shadow": ({"gid": _GID}, {"dist_m": _PDIST, "timeout_s": _TIMEOUT},
               "Follow a person from a safe distance (>= 2.5 m)."),
    "watch": ({"gid": _GID}, {"timeout_s": _TIMEOUT}, "Stay in place and keep facing a person."),
    "escort": ({"gid": _GID}, {"side": {"type": "string", "enum": ["left", "right"]},
                               "dist_m": _PDIST, "timeout_s": _TIMEOUT},
               "Walk beside a person at a safe distance."),
    "patrol": ({}, {"points": _POINTS,
                    "zone": {"type": "string", "minLength": 1,
                             "description": "Named patrol AREA (from context zones) - not a precision value."},
                    "pass_zone": _ZONE,
                    "loops": {"type": "integer", "minimum": 1, "maximum": 100},
                    "shuffle": {"type": "boolean"}, "speed": _SPEED},
               "Patrol either a list of points or a named area `zone` (exactly one); precision via pass_zone."),
    "explore": ({}, {"zone": {"type": "string"}}, "Autonomous exploration/mapping."),
    "goto_label": ({"label": {"type": "string", "minLength": 1}}, {"speed": _SPEED, "zone": _ZONE},
                   "Walk to a named semantic label (preferred over raw coordinates)."),
    "return_home": ({}, {"speed": _SPEED, "zone": _ZONE}, "Return to the home pose."),
    "capture": ({}, {"x": _NUM, "y": _NUM, "yaw": {"type": "number", "description": "Heading to shoot from, rad."},
                     "cams": {"oneOf": [{"type": "string", "enum": ["all"]},
                                        {"type": "array", "minItems": 1, "items": {"type": "string", "enum": CAMS}}]},
                     "mode": {"type": "string", "enum": ["max", "normal"]},
                     "label": {"type": "string", "minLength": 1, "description": "What is photographed (label name)."}},
                "Take max-resolution photos (+thermal). With x/y the robot first walks there with zone fine. "
                "To photograph a label, stand ~2-3 m from it (not on it) facing it."),
    "request_passage": ({"text": {"type": "string", "minLength": 1, "maxLength": 300}},
                        {"repeat_s": {"type": "number", "minimum": 5, "maximum": 600},
                         "zone_polygon": {"type": "array", "minItems": 3, "maxItems": 50,
                                          "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM}},
                         "ahead_m": {"type": "number", "minimum": 0.5, "maximum": 5.0},
                         "clear_for_s": {"type": "number", "minimum": 0.5, "maximum": 30},
                         "timeout_s": {"type": "number", "minimum": 5, "maximum": 7200},
                         "on_timeout": {"type": "string", "enum": ["alert", "return", "abort"]}},
                        "Stop, repeatedly say `text` (asking people to open a door / let it through) and wait "
                        "until the way ahead is clear, the operator continues, or someone waves."),
    "wait_for": ({"event": {"type": "string", "enum": ["path_clear", "operator", "gesture", "person_gone"]}},
                 {"timeout_s": {"type": "number", "minimum": 1, "maximum": 7200}, "gid": _GID,
                  "ahead_m": {"type": "number", "minimum": 0.5, "maximum": 5.0},
                  "clear_for_s": {"type": "number", "minimum": 0.5, "maximum": 60},
                  "zone_polygon": {"type": "array", "minItems": 3, "maxItems": 50,
                                   "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": _NUM}}},
                 "Wait for an event (person_gone: optional gid)."),
    "record": ({"on": {"type": "boolean"}}, {"mode": {"type": "string", "enum": ["evidence"]}},
               "Switch evidence recording on/off."),
}


class NLError(Exception):
    def __init__(self, message: str, status: int = 502, errors: Optional[List[str]] = None):
        Exception.__init__(self, message)
        self.message = message
        self.status = int(status)
        self.errors = list(errors or [])

    def to_dict(self) -> dict:
        return {"error": self.message, "status": self.status, "errors": self.errors}


# ------------------------------------------------------------------ schema
def step_schema(op: str) -> dict:
    req, opt, desc = STEP_SPECS[op]
    props = {"op": {"type": "string", "const": op}}
    props.update(req)
    props.update(opt)
    return {"type": "object", "description": desc, "properties": props,
            "required": ["op"] + list(req.keys()), "additionalProperties": False}


def tool_input_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 80, "description": "Short mission name."},
            "steps": {"type": "array", "minItems": 1, "maxItems": 50,
                      "items": {"oneOf": [step_schema(op) for op in STEP_SPECS]}},
            "on_fail": {"type": "string", "enum": ["stop", "skip", "return_home"]},
            "explanation": {"type": "string", "maxLength": 600,
                            "description": "1-3 sentences for the operator, in the operator's language."},
            "warnings": {"type": "array", "items": {"type": "string"},
                         "description": "Assumptions, ambiguities, safety notes."},
        },
        "required": ["steps", "explanation"],
        "additionalProperties": False,
    }


def tool_definition() -> dict:
    return {"name": TOOL_NAME,
            "description": ("Submit the mission plan for the robot dog. It is shown to the human "
                            "operator for approval before anything moves."),
            "input_schema": tool_input_schema()}


_JS_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def check_schema(value: Any, schema: dict, path: str = "$") -> List[str]:
    """Minimal JSON-schema subset validator (type/enum/const/min/max/items/props/oneOf-by-op)."""
    errs = []  # type: List[str]
    if "oneOf" in schema:
        opts = schema["oneOf"]
        if isinstance(value, dict) and "op" in value:
            for s in opts:
                if s.get("properties", {}).get("op", {}).get("const") == value["op"]:
                    return check_schema(value, s, path)
            return ["%s: unknown op %r" % (path, value.get("op"))]
        ok = [s for s in opts if not check_schema(value, s, path)]
        return [] if len(ok) == 1 else ["%s: matches %d of oneOf" % (path, len(ok))]
    t = schema.get("type")
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ["%s: expected %s" % (path, t)]
        if t == "integer" and not float(value).is_integer():
            return ["%s: expected integer" % path]
        if not math.isfinite(float(value)):
            return ["%s: not finite" % path]
        if "minimum" in schema and value < schema["minimum"]:
            errs.append("%s: %s < minimum %s" % (path, value, schema["minimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errs.append("%s: %s > maximum %s" % (path, value, schema["maximum"]))
    elif t in _JS_TYPES:
        if not isinstance(value, _JS_TYPES[t]):
            return ["%s: expected %s" % (path, t)]
    if "const" in schema and value != schema["const"]:
        errs.append("%s: must be %r" % (path, schema["const"]))
    if "enum" in schema and value not in schema["enum"]:
        errs.append("%s: %r not in %s" % (path, value, schema["enum"]))
    if t == "string":
        if len(value) < schema.get("minLength", 0):
            errs.append("%s: too short" % path)
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append("%s: too long" % path)
    if t == "array":
        if len(value) < schema.get("minItems", 0):
            errs.append("%s: needs >= %d items" % (path, schema["minItems"]))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append("%s: needs <= %d items" % (path, schema["maxItems"]))
        if "items" in schema:
            for i, v in enumerate(value):
                errs.extend(check_schema(v, schema["items"], "%s[%d]" % (path, i)))
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append("%s: missing %r" % (path, k))
        for k, v in value.items():
            if k in props:
                errs.extend(check_schema(v, props[k], "%s.%s" % (path, k)))
            elif schema.get("additionalProperties") is False:
                errs.append("%s: field %r not allowed" % (path, k))
    return errs


def _schema_module():
    try:
        from mission import schema as m  # type: ignore
        return m
    except Exception:
        pass
    try:
        from . import schema as m  # type: ignore
        return m
    except Exception:
        return None


def validate_mission(mission: dict) -> List[str]:
    """Local schema check + ``mission.schema.validate`` (M1) when importable."""
    errs = check_schema({k: v for k, v in mission.items() if k in ("name", "steps", "on_fail")},
                        {"type": "object", "properties": {k: v for k, v in tool_input_schema()["properties"].items()
                                                          if k in ("name", "steps", "on_fail")},
                         "required": ["steps"], "additionalProperties": False})
    for i, st in enumerate(mission.get("steps") or []):
        if isinstance(st, dict) and st.get("op") == "patrol" and bool(st.get("points")) == bool(st.get("zone")):
            errs.append("$.steps[%d]: patrol needs exactly one of points / zone (area)" % i)
    m = _schema_module()
    if m is not None and hasattr(m, "validate"):
        try:
            res = m.validate(mission)
            ext = res[1] if isinstance(res, tuple) and len(res) == 2 else []
            errs.extend(str(e) for e in (ext or []) if str(e) not in errs)
        except Exception as e:  # schema bug must not crash NL
            errs.append("schema.validate failed: %s" % e)
    return errs


# ------------------------------------------------------------------ prompt
SYSTEM_STATIC = """You are the mission planner of NERO, a Unitree Go2 quadruped security robot dog.
A human operator (usually Hungarian, sometimes English) gives a command in natural language.
You translate it into a mission by calling the `submit_mission` tool exactly once. You never move
the robot directly: the operator reviews and approves every mission before it runs.

Robot and world:
- Coordinates are the 2D world/map frame in metres; yaw in radians, counter-clockwise, 0 = +x.
- "forward" means along the robot's current yaw; "left" is +90 deg from yaw. Relative requests
  ("3 metres forward", "2 méterrel balra") must be converted to world x, y using the robot pose
  (the context gives the forward and left unit vectors).
- Speed is never given in m/s; only the profile names stealth | precise | normal | sprint.
- Person bearings are degrees in the robot frame (0 = straight ahead, +90 = left).

Safety rules (hard):
1. Never plan to get closer than 2.5 m to any person. shadow/escort dist_m >= 2.5 (default 3.0).
   Never send the robot to a coordinate within 2.5 m of a visible person.
2. Use speed "sprint" ONLY if the operator explicitly asks for running / maximum / full speed
   (e.g. "fuss", "rohanj", "max sebességgel", "teljes sebességgel", "run", "full speed").
3. Use jump_to ONLY if the operator explicitly asks to jump ("ugorj", "ugrálj", "jump", "hop").
4. Never enter no_go zones. Prefer named labels (goto_label) and zones over raw coordinates.
5. Refer to people only by gid from the visible-persons list. "the nearest person" = smallest
   range. If the target person or place is unknown, still submit the best safe mission and say
   so in warnings (e.g. a scan step), unless the request is unsafe or unrelated to the robot.
6. Default speed is "normal"; "óvatosan/lassan/quietly" -> stealth or precise.
7. Moving steps take an optional `zone`: "fine" to stop exactly and settle (before a photo, at a
   door, "pontosan"), "z10".."z100" to pass through without slowing ("30 centis zóna" -> "z30").
   Omit it when the operator says nothing about precision. For patrol the precision field is
   `pass_zone` (patrol `zone` is the named area).
8. To get through a closed door / ask people to let the robot pass: goto (zone fine) in front of
   it, then request_passage with a polite `text` in the operator's language.
9. Photos ("fotó", "fénykép", "kép"): capture with mode "max"; to photograph a label, put x/y
   2-3 m in front of it with yaw facing it, and set label to its name.

Hungarian hints: kövesd = follow (shadow), figyeld = watch, kísérd = escort, menj = go,
fuss = run, ugorj/ugrálj = jump, járőrözz = patrol, nézz körül = scan, gyere vissza/haza = return_home,
kérd meg, hogy engedjenek ki/be = request_passage, fotó/fénykép = capture,
pontos = fine, rögzíts/vedd fel = record, várj amíg = wait_for, parkoló = parking lot,
kapu = gate, ajtó = door, előre = forward, hátra = back, balra = left, jobbra = right, méter = metre.

Output: call `submit_mission` with steps, on_fail ("stop" unless asked otherwise), a short
name, an `explanation` (1-3 sentences in the operator's language) and `warnings` (assumptions,
ambiguities). Do not answer in plain text unless the request cannot be expressed as a mission;
in that case reply with one short question in the operator's language."""


def _f(v, nd=2):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def normalize_context(ctx: Optional[dict]) -> dict:
    ctx = dict(ctx or {})
    out = {}  # type: Dict[str, Any]
    labels = ctx.get("labels") or []
    if isinstance(labels, dict):
        labels = [dict(v, name=k) if isinstance(v, dict) else {"name": k} for k, v in labels.items()]
    out["labels"] = [{"name": str(l.get("name")), "x": _f(l.get("x")), "y": _f(l.get("y"))}
                     if isinstance(l, dict) else {"name": str(l)} for l in labels]
    zones = ctx.get("zones") or []
    out["zones"] = [{"name": str(z.get("name") or z.get("id")), "kind": z.get("kind", "?"),
                     "id": z.get("id")} if isinstance(z, dict) else {"name": str(z)} for z in zones]
    home = ctx.get("home")
    out["home"] = {"x": _f(home.get("x")), "y": _f(home.get("y")), "yaw": _f(home.get("yaw"))} \
        if isinstance(home, dict) else None
    pose = ctx.get("pose") or ctx.get("robot_pose")
    if isinstance(pose, dict):
        yaw = float(pose.get("yaw", 0.0) or 0.0)
        out["pose"] = {"x": _f(pose.get("x", 0.0)), "y": _f(pose.get("y", 0.0)), "yaw": _f(yaw, 3),
                       "forward_unit": [_f(math.cos(yaw), 3), _f(math.sin(yaw), 3)],
                       "left_unit": [_f(-math.sin(yaw), 3), _f(math.cos(yaw), 3)]}
    else:
        out["pose"] = None
    persons = []
    for p in ctx.get("persons") or []:
        if not isinstance(p, dict) or p.get("gid") is None:
            continue
        rng, brg = p.get("range_m"), p.get("bearing_deg")
        if (rng is None or brg is None) and p.get("x") is not None and p.get("y") is not None:
            rng = math.hypot(float(p["x"]), float(p["y"]))
            brg = math.degrees(math.atan2(float(p["y"]), float(p["x"])))
        e = {"gid": int(p["gid"]), "range_m": _f(rng, 1), "bearing_deg": _f(brg, 0)}
        for k in ("modality", "zone"):
            if p.get(k):
                e[k] = p[k]
        persons.append(e)
    persons.sort(key=lambda e: e["range_m"] if e["range_m"] is not None else 1e9)
    out["persons"] = persons
    for k in ("safety_level", "battery_pct", "deadman_alive"):
        if k in ctx:
            out[k] = ctx[k]
    return out


def render_context(nctx: dict) -> str:
    return "Current context (live, may change):\n" + json.dumps(nctx, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ post-processing
_SPRINT_RE = re.compile(r"sprint|fuss|fut[ns]|rohan|max\w*\s*seb|teljes\s*seb|leggyorsabb|"
                        r"\brun\b|\brunning\b|full\s*speed|max\w*\s*speed|as fast as", re.I)
_JUMP_RE = re.compile(r"ugr|ugor|szökk|jump|hop\b|leap", re.I)


def explicit_sprint(text: str) -> bool:
    return bool(_SPRINT_RE.search(text or ""))


def explicit_jump(text: str) -> bool:
    return bool(_JUMP_RE.search(text or ""))


def postprocess(steps: List[dict], text: str, nctx: dict) -> Tuple[List[dict], List[str]]:
    """Deterministic safety layer on top of the model output (never trusts the LLM alone)."""
    warns = []  # type: List[str]
    out = []
    sprint_ok, jump_ok = explicit_sprint(text), explicit_jump(text)
    labels = {l.get("name") for l in nctx.get("labels") or []}
    zones = {z.get("name") for z in nctx.get("zones") or []} | {z.get("id") for z in nctx.get("zones") or []}
    gids = {p["gid"] for p in nctx.get("persons") or []}
    for i, st in enumerate(steps):
        st = dict(st)
        op = st.get("op")
        if st.get("speed") == "sprint" and not sprint_ok:
            st["speed"] = "normal"
            warns.append("step %d: sprint not explicitly requested -> normal" % i)
        if op == "jump_to" and not jump_ok:
            st = {"op": "goto", "x": st.get("x"), "y": st.get("y"), "speed": "normal"}
            op = "goto"
            warns.append("step %d: jump not explicitly requested -> goto" % i)
        if op in ("shadow", "escort"):
            d = st.get("dist_m")
            if isinstance(d, (int, float)) and not isinstance(d, bool) and d < MIN_PERSON_DIST_M:
                st["dist_m"] = MIN_PERSON_DIST_M
                warns.append("step %d: dist_m raised to %.1f m" % (i, MIN_PERSON_DIST_M))
        if op in ("shadow", "watch", "escort") and gids and st.get("gid") not in gids:
            warns.append("step %d: person gid %s is not visible now" % (i, st.get("gid")))
        if op == "goto_label" and labels and st.get("label") not in labels:
            warns.append("step %d: unknown label %r" % (i, st.get("label")))
        area = st.get("zone") if op in ("patrol", "explore") else None
        if isinstance(area, str) and zones and area not in zones:
            warns.append("step %d: unknown area %r" % (i, area))
        if op == "capture" and st.get("label") and labels and st["label"] not in labels:
            warns.append("step %d: unknown label %r" % (i, st["label"]))
        if op == "capture" and (st.get("x") is None) != (st.get("y") is None):
            warns.append("step %d: capture needs both x and y (pose ignored)" % i)
            st.pop("x", None)
            st.pop("y", None)
        if op in ("goto", "jump_to", "capture") and isinstance(st.get("x"), (int, float)) and isinstance(st.get("y"), (int, float)):
            pose = nctx.get("pose")
            if pose and nctx.get("persons"):
                c, s = math.cos(pose["yaw"] or 0.0), math.sin(pose["yaw"] or 0.0)
                for p in nctx["persons"]:
                    if p.get("range_m") is None or p.get("bearing_deg") is None:
                        continue
                    b = math.radians(p["bearing_deg"])
                    bx, by = p["range_m"] * math.cos(b), p["range_m"] * math.sin(b)
                    wx, wy = pose["x"] + c * bx - s * by, pose["y"] + s * bx + c * by
                    if math.hypot(st["x"] - wx, st["y"] - wy) < MIN_PERSON_DIST_M:
                        warns.append("step %d: goal is within %.1f m of person %d" % (i, MIN_PERSON_DIST_M, p["gid"]))
        out.append(st)
    return out, warns


# ------------------------------------------------------------------ planner
def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class NLPlanner:
    def __init__(self, client=None, model: Optional[str] = None,
                 world_provider: Optional[Callable[[], dict]] = None,
                 max_tokens: int = 16000, effort: Optional[str] = None,
                 fallbacks: Optional[bool] = None, max_repairs: Optional[int] = None,
                 timeout_s: float = 60.0):
        self.client = client
        self.model = model or os.environ.get("NL_MODEL") or DEFAULT_MODEL
        self.world_provider = world_provider
        self.max_tokens = int(max_tokens)
        self.effort = effort or os.environ.get("NL_EFFORT", "medium")
        self.fallbacks = (os.environ.get("NL_FALLBACKS", "1") != "0") if fallbacks is None else bool(fallbacks)
        self.max_repairs = int(os.environ.get("NL_MAX_REPAIRS", "1")) if max_repairs is None else int(max_repairs)
        self.timeout_s = float(timeout_s)
        self.last_usage = None

    # -- client -----------------------------------------------------------
    def available(self) -> Tuple[bool, str]:
        if self.client is not None:
            return True, "ok"
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "ANTHROPIC_API_KEY is not set; natural-language commands are disabled"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic SDK not installed (pip install anthropic)"
        return True, "ok"

    def _client(self):
        if self.client is not None:
            return self.client
        ok, why = self.available()
        if not ok:
            raise NLError(why, 503)
        import anthropic
        self.client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=2)
        return self.client

    def _tool_choice(self) -> dict:
        if os.environ.get("NL_FORCE_TOOL") == "1" or not self.model.startswith(NO_FORCED_TOOL_MODELS):
            return {"type": "tool", "name": TOOL_NAME}
        # Current Opus/Sonnet/Fable reject forced tool_choice: auto + one tool + prompt instruction.
        return {"type": "auto", "disable_parallel_tool_use": True}

    def build_request(self, text: str, nctx: dict) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": SYSTEM_STATIC, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": render_context(nctx)},
            ],
            "tools": [tool_definition()],
            "tool_choice": self._tool_choice(),
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": text}],
        }

    def _call(self, client, req: dict):
        try:
            if self.fallbacks and getattr(client, "beta", None) is not None:
                return client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **req)
            return client.messages.create(**req)
        except NLError:
            raise
        except Exception as e:
            raise _map_api_error(e)

    # -- main -------------------------------------------------------------
    def context(self, context: Optional[dict]) -> dict:
        base = {}  # type: Dict[str, Any]
        if self.world_provider is not None:
            try:
                base.update(self.world_provider() or {})
            except Exception:
                pass
        base.update(context or {})
        return normalize_context(base)

    def to_mission(self, text: str, context: Optional[dict] = None) -> Tuple[dict, str, List[str]]:
        text = (text or "").strip()
        if not text:
            raise NLError("empty command", 422)
        if len(text) > 2000:
            raise NLError("command too long (max 2000 chars)", 422)
        nctx = self.context(context)
        client = self._client()
        req = self.build_request(text, nctx)
        last_errs = []  # type: List[str]
        for attempt in range(self.max_repairs + 1):
            resp = self._call(client, req)
            self.last_usage = _get(resp, "usage")
            stop = _get(resp, "stop_reason")
            if stop == "refusal":
                raise NLError("the model declined this request", 422)
            content = list(_get(resp, "content") or [])
            tu = next((b for b in content if _get(b, "type") == "tool_use" and _get(b, "name") == TOOL_NAME), None)
            if tu is None:
                msg = " ".join(str(_get(b, "text", "")) for b in content if _get(b, "type") == "text").strip()
                if stop == "max_tokens":
                    raise NLError("model output truncated (max_tokens)", 502)
                raise NLError("clarification needed: " + (msg or "no mission produced"), 422)
            inp = _get(tu, "input") or {}
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except ValueError:
                    inp = {}
            mission, explanation, warns, errs = self._build(inp, text, nctx)
            if not errs:
                return mission, explanation, warns
            last_errs = errs
            if attempt >= self.max_repairs:
                break
            req = dict(req)
            req["messages"] = list(req["messages"]) + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": _get(tu, "id"), "is_error": True,
                    "content": "Mission rejected by validator:\n- " + "\n- ".join(errs[:20]) +
                               "\nCall submit_mission again with a corrected mission."}]},
            ]
        raise NLError("invalid mission from model", 422, last_errs)

    def _build(self, inp: dict, text: str, nctx: dict):
        if not isinstance(inp, dict):
            return None, "", [], ["tool input is not an object"]
        steps = inp.get("steps")
        errs = check_schema(inp, tool_input_schema())
        if errs or not isinstance(steps, list):
            return None, "", [], errs or ["steps missing"]
        steps, pp_warns = postprocess(steps, text, nctx)
        mission = {"name": str(inp.get("name") or text[:60]), "source": "nl",
                   "steps": steps, "on_fail": inp.get("on_fail") or "stop"}
        errs = validate_mission(mission)
        if errs:
            return None, "", [], errs
        warns = [str(w) for w in (inp.get("warnings") or [])] + pp_warns
        return mission, str(inp.get("explanation") or ""), warns, []


def _map_api_error(e: Exception) -> NLError:
    try:
        import anthropic  # type: ignore
    except ImportError:
        anthropic = None
    if anthropic is not None:
        if isinstance(e, anthropic.AuthenticationError):
            return NLError("Claude API authentication failed (check ANTHROPIC_API_KEY)", 503)
        if isinstance(e, anthropic.PermissionDeniedError):
            return NLError("Claude API key lacks permission for model", 503)
        if isinstance(e, anthropic.NotFoundError):
            return NLError("Claude model not found (check NL_MODEL)", 502)
        if isinstance(e, anthropic.RateLimitError):
            return NLError("Claude API rate limited, try again shortly", 503)
        if isinstance(e, anthropic.BadRequestError):
            return NLError("Claude API rejected the request: %s" % getattr(e, "message", e), 502)
        if isinstance(e, anthropic.APIStatusError):
            code = getattr(e, "status_code", 500)
            return NLError("Claude API error %s: %s" % (code, getattr(e, "message", e)),
                           503 if code >= 500 else 502)
        if isinstance(e, anthropic.APIConnectionError):
            return NLError("cannot reach Claude API (network)", 503)
    code = getattr(e, "status_code", None)
    return NLError("Claude API call failed: %s%s" % (type(e).__name__, (" %s" % code) if code else ""),
                   503 if code is None or code >= 500 or code == 429 else 502)


_default = None  # type: Optional[NLPlanner]


def to_mission(text: str, context: Optional[dict] = None):
    """Module-level convenience with a lazily created default planner."""
    global _default
    if _default is None:
        _default = NLPlanner()
    return _default.to_mission(text, context)
