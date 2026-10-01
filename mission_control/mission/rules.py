"""Event -> mission rule engine (M5, CONTRACT section 5).

Rule format (stored in world.json ``rules``)::

    {"id": "r1", "name": "night intruder", "enabled": true,
     "when": {"event": "person_in_zone", "zone": "yard",          # zone id or name
              "modality": "thermal",        # optional: str|list, person must have any of them
              "modality_only": false,       # optional: person modalities must be a subset
              "min_conf": 0.4, "dwell_s": 3.0, "hours": "22:00-06:00"},
     "then": [{"op": "say", "text": "Person {gid} in {zone}"},
              {"op": "watch", "gid": "{gid}", "timeout_s": 60}],
     "cooldown_s": 60, "alert": true, "on_fail": "stop", "message": "optional {zone} text",
     "evidence": true}

``evidence: true`` prepends ``{"op": "record", "on": true, "mode": "evidence"}`` to the
mission AND calls the injected ``start_recording(reason)`` immediately on fire (the
recorder pre-buffer must not wait for the mission to be scheduled). It also works on
rules without ``then`` (record-only alert). See ``EXAMPLE_RULES`` for the default night
thermal-intruder rule.

Events (``when.event``):

* ``person_in_zone``     zone, modality?, modality_only?, min_conf?, dwell_s?
* ``person_count_over``  count (fires when > count), zone? (None = everywhere), min_conf?, dwell_s?
* ``gesture``            gesture (str|list, e.g. "wave"), zone?, min_conf?  (from ``mc.omni.gesture``)
* ``schedule``           cron (see mission/scheduler.py), e.g. "*/30 22-06"
* ``lost_target``        gid? (int or "target" = engine.target_gid; omitted = any), zone?, timeout_s=5

Common: ``hours`` (or rule ``active_hours``) "HH:MM-HH:MM" (crosses midnight), ``cooldown_s``
(per rule, default 30), ``dwell_s`` (continuous presence; ``grace_s`` tolerates short
detection drop-outs, default 1.0). Per-person dedup: a gid fires a rule once per presence
(it must leave the zone / condition to re-arm).

Template placeholders in ``then`` step values: ``{gid} {x} {y} {zone} {rule} {count}
{gesture} {modality} {approach_x} {approach_y}``. A value that is exactly one placeholder
keeps its type (``"{gid}"`` -> int, ``"{x}"`` -> float). ``approach_x/y`` is a point
``approach_m`` (rule-level, default 3.0, never < 2.5) from the person toward the robot.

Safety: rule missions may not contain ``jump_to``, ``speed/profile == "sprint"`` or the
``jump``/``pounce`` actions (dead-man is never held for an automatic mission). They are
submitted as ``{"name", "source": "rule", "steps", "on_fail"}`` and validated with
``mission.schema.validate`` when that module is importable.
"""
from __future__ import annotations

import copy
import math
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:  # package import (mission.rules) or flat import inside the container
    from . import world as W
    from .scheduler import CronExpr, Scheduler
except ImportError:  # pragma: no cover
    import world as W  # type: ignore
    from scheduler import CronExpr, Scheduler  # type: ignore

EVENTS = ("person_in_zone", "person_count_over", "gesture", "schedule", "lost_target")
FORBIDDEN_OPS = ("jump_to",)
FORBIDDEN_ACTIONS = ("jump", "pounce", "front_jump", "frontjump")
MIN_APPROACH_M = 2.5
_PH_RE = re.compile(r"^\{(\w+)\}$")


# Default example (not auto-installed): night, thermal-only person lingering in a
# watch zone -> evidence recording, warning, watch from where we stand.
EXAMPLE_RULES = [
    {"id": "night_thermal_intruder", "name": "Night thermal intruder",
     "when": {"event": "person_in_zone", "zone": "<watch zone id>", "modality": "thermal",
              "min_conf": 0.4, "dwell_s": 3.0, "hours": "22:00-06:00"},
     "then": [{"op": "say", "text": "Attention, this area is under surveillance."},
              {"op": "watch", "gid": "{gid}", "timeout_s": 120}],
     "evidence": True, "cooldown_s": 120, "alert": True,
     "message": "Thermal person {gid} in {zone}"},
    {"id": "wave_come", "name": "Wave -> approach to 3 m",
     "when": {"event": "gesture", "gesture": "wave"}, "approach_m": 3.0,
     "then": [{"op": "goto", "x": "{approach_x}", "y": "{approach_y}", "speed": "precise", "zone": "fine"},
              {"op": "look_at", "x": "{x}", "y": "{y}"}],
     "cooldown_s": 20, "alert": False},
]


# ------------------------------------------------------------------ validation

def check_rule_steps(steps: Any) -> List[str]:
    """Safety check for rule-originated steps (no sprint / jumps)."""
    errs = []
    if not isinstance(steps, list):
        return ["then must be a list of steps"]
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not s.get("op"):
            errs.append("step %d: needs op" % i)
            continue
        if s["op"] in FORBIDDEN_OPS:
            errs.append("step %d: op %s forbidden in rule missions" % (i, s["op"]))
        for k in ("speed", "profile"):
            if str(s.get(k, "")).lower() == "sprint":
                errs.append("step %d: sprint forbidden in rule missions" % i)
        if s["op"] == "action" and str(s.get("name", "")).lower() in FORBIDDEN_ACTIONS:
            errs.append("step %d: action %s forbidden in rule missions" % (i, s.get("name")))
    return errs


def validate_rule(rule: Dict[str, Any], world: Optional[W.WorldStore] = None) -> List[str]:
    if not isinstance(rule, dict):
        return ["rule must be an object"]
    errs = []
    when = rule.get("when")
    if not isinstance(when, dict):
        return ["rule.when must be an object"]
    ev = when.get("event")
    if ev not in EVENTS:
        errs.append("when.event must be one of %s" % (EVENTS,))
    if ev == "person_in_zone" and not when.get("zone"):
        errs.append("person_in_zone needs when.zone")
    if ev == "person_count_over":
        try:
            int(when.get("count"))
        except (TypeError, ValueError):
            errs.append("person_count_over needs integer when.count")
    if ev == "gesture" and not when.get("gesture"):
        errs.append("gesture needs when.gesture")
    if ev == "schedule":
        try:
            CronExpr(when.get("cron", ""))
        except ValueError as e:
            errs.append("bad cron: %s" % e)
    zone = when.get("zone")
    if zone and world is not None and world.get_zone(zone) is None:
        errs.append("unknown zone %r" % (zone,))
    for hk in (when.get("hours"), rule.get("active_hours")):
        if hk:
            try:
                W.parse_hours(hk)
            except ValueError as e:
                errs.append(str(e))
    errs += check_rule_steps(rule.get("then", []))
    if not rule.get("then") and not rule.get("alert", True) and not rule.get("evidence"):
        errs.append("rule does nothing (no then, alert false)")
    return errs


# ------------------------------------------------------------------ templating

class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_steps(steps: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deep-substitute placeholders. Whole-string placeholders keep the ctx type."""
    def sub(v: Any) -> Any:
        if isinstance(v, str):
            m = _PH_RE.match(v)
            if m and m.group(1) in ctx:
                return ctx[m.group(1)]
            try:
                return v.format_map(_SafeDict({k: ("" if val is None else _fmt(val)) for k, val in ctx.items()}))
            except (ValueError, IndexError):
                return v
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v
    return [sub(copy.deepcopy(s)) for s in steps]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "%.2f" % v
    return str(v)


def _unresolved(steps: Any) -> List[str]:
    """Placeholders left (unknown or None values) -> list of messages."""
    out = []

    def walk(v: Any, path: str) -> None:
        if v is None and path.endswith((".x", ".y", ".gid")):
            out.append("%s unresolved" % path)
        elif isinstance(v, str) and re.search(r"\{\w+\}", v):
            out.append("%s has unresolved placeholder %r" % (path, v))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, "%s[%d]" % (path, i))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, "%s.%s" % (path, k))
    walk(steps, "then")
    return out


# ------------------------------------------------------------------ helpers

def _modalities(p: Dict[str, Any]) -> List[str]:
    m = p.get("modality") or []
    if isinstance(m, str):
        m = [m]
    return [str(x).lower() for x in m]


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).lower() for x in v]
    return [str(v).lower()]


class RuleEngine:
    """Feed it with ``on_persons(persons_base, pose, t)``, ``on_gesture(evt)`` and
    ``on_tick(t)``; each returns the list of fired alert dicts::

        {t, rule_id, kind, zone, gid, x, y, modality, message, alert, mission_id, error}

    ``submit_mission_fn(mission_dict)`` -> mission id (or any value / raise).
    ``alert_fn(alert)`` is called for rules with ``alert`` true (default).
    ``clock()`` is used when no ``t`` is given. Rules come from ``world.rules()``
    (re-read automatically when the store version changes) unless ``rules`` is given.
    """

    def __init__(self, world: Optional[W.WorldStore], submit_mission_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 clock: Optional[Callable[[], float]] = None, alert_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 rules: Optional[List[Dict[str, Any]]] = None, grace_s: float = 1.0, validate_schema: bool = True,
                 start_recording: Optional[Callable[[str], Any]] = None):
        self.world = world
        self.submit = submit_mission_fn
        self.clock = clock or time.time
        self.alert_fn = alert_fn
        self.start_recording = start_recording
        self.grace_s = float(grace_s)
        self.validate_schema = validate_schema
        self.target_gid = None  # type: Optional[int]
        self.history = []  # type: List[Dict[str, Any]]
        self._lock = threading.RLock()
        self._static_rules = rules
        self._rules_version = None  # type: Any
        self._rules = []  # type: List[Dict[str, Any]]
        self.rule_errors = {}  # type: Dict[str, List[str]]
        self._sched = Scheduler([])
        self._last_fire = {}  # type: Dict[str, float]
        self._presence = {}  # type: Dict[Tuple[str, Any], Dict[str, Any]]
        self._seen = {}  # type: Dict[int, Dict[str, Any]]   gid -> last world person + t
        self._lost_fired = set()  # type: set
        self._pose = (0.0, 0.0, 0.0)
        self._refresh(force=True)

    # --- rules
    def set_rules(self, rules: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._static_rules = list(rules)
            self._refresh(force=True)

    def rules(self) -> List[Dict[str, Any]]:
        self._refresh()
        return copy.deepcopy(self._rules)

    def _refresh(self, force: bool = False) -> None:
        ver = None if self._static_rules is not None else getattr(self.world, "version", None)
        if not force and ver == self._rules_version:
            return
        src = self._static_rules if self._static_rules is not None else (self.world.rules() if self.world else [])
        good, self.rule_errors = [], {}
        for r in src:
            rid = str(r.get("id") or "rule_%d" % len(good))
            errs = validate_rule(r, None)
            if errs:
                self.rule_errors[rid] = errs
                continue
            r = dict(r, id=rid)
            good.append(r)
        self._rules = good
        self._rules_version = ver
        self._sched.set_entries([{"id": r["id"], "cron": r["when"].get("cron", ""), "enabled": r.get("enabled", True)}
                                 for r in good if r["when"].get("event") == "schedule"])

    # --- gates
    def _hours_ok(self, r: Dict[str, Any], t: float) -> bool:
        w = r["when"]
        return W.hours_active(w.get("hours"), t) and W.hours_active(r.get("active_hours"), t)

    def _cooldown_ok(self, r: Dict[str, Any], t: float) -> bool:
        last = self._last_fire.get(r["id"])
        return last is None or t - last >= float(r.get("cooldown_s", 30.0))

    def _zone(self, key: Any) -> Optional[Dict[str, Any]]:
        if not key or self.world is None:
            return None
        return self.world.get_zone(key)

    def _person_ok(self, w: Dict[str, Any], p: Dict[str, Any]) -> bool:
        if float(p.get("conf", 1.0) or 0.0) < float(w.get("min_conf", 0.0) or 0.0):
            return False
        want = _as_list(w.get("modality"))
        if want:
            have = _modalities(p)
            if not any(m in have for m in want):
                return False
            if w.get("modality_only") and not set(have) <= set(want):
                return False
        return True

    def _in_zone(self, zone: Optional[Dict[str, Any]], p: Dict[str, Any], t: float) -> bool:
        if zone is None:
            return True
        if not W.hours_active(zone.get("active_hours"), t):
            return False
        return bool(W.point_in_polygon(zone["polygon"], [p["x"], p["y"]]))

    def _dwell(self, key: Tuple[str, Any], t: float, dwell: float) -> bool:
        """Update presence; True when dwell met and not yet fired for this key."""
        st = self._presence.get(key)
        if st is None:
            st = self._presence[key] = {"enter_t": t, "last_t": t, "fired": False}
        st["last_t"] = t
        return (not st["fired"]) and (t - st["enter_t"] >= dwell)

    def _prune_presence(self, t: float, active: set) -> None:
        for key in list(self._presence):
            if key not in active and t - self._presence[key]["last_t"] > self.grace_s:
                del self._presence[key]

    # --- feeds
    def on_persons(self, persons_base: Any, pose: Any = None, t: Optional[float] = None) -> List[Dict[str, Any]]:
        """persons_base: list of PersonTrack dicts (base frame) or an
        ``mc.omni.persons`` payload ``{"t", "persons": [...]}``."""
        if isinstance(persons_base, dict):
            t = persons_base.get("t", t) if t is None else t
            persons_base = persons_base.get("persons", [])
        t = float(self.clock() if t is None else t)
        with self._lock:
            self._refresh()
            if pose is not None:
                self._pose = W._pose(pose)
            persons = W.persons_to_world(persons_base or [], self._pose)
            for p in persons:
                if p.get("gid") is not None:
                    self._seen[int(p["gid"])] = dict(p, _t=t)
                    self._lost_fired = {k for k in self._lost_fired if k[1] != int(p["gid"])}
            fired, active = [], set()
            for r in self._rules:
                if not r.get("enabled", True):
                    continue
                ev = r["when"]["event"]
                if ev == "person_in_zone":
                    fired += self._eval_in_zone(r, persons, t, active)
                elif ev == "person_count_over":
                    fired += self._eval_count(r, persons, t, active)
            self._prune_presence(t, active)
            fired += self._eval_lost(t)
            return fired

    def on_gesture(self, evt: Dict[str, Any], t: Optional[float] = None) -> List[Dict[str, Any]]:
        """evt: ``{gid, gesture, t}`` from ``mc.omni.gesture``."""
        if t is None:
            t = evt.get("t")
        t = float(self.clock() if t is None else t)
        g = str(evt.get("gesture", "")).lower()
        gid = evt.get("gid")
        with self._lock:
            self._refresh()
            p = self._seen.get(int(gid)) if gid is not None else None
            fired = []
            for r in self._rules:
                w = r["when"]
                if not r.get("enabled", True) or w["event"] != "gesture" or g not in _as_list(w.get("gesture")):
                    continue
                if not self._hours_ok(r, t) or not self._cooldown_ok(r, t):
                    continue
                zone = self._zone(w.get("zone"))
                if w.get("zone"):
                    if zone is None or p is None or not self._in_zone(zone, p, t):
                        continue
                if p is not None and not self._person_ok(w, p):
                    continue
                ctx = self._ctx(r, t, p, zone, gid=gid, gesture=g)
                fired.append(self._fire(r, t, ctx))
            return fired

    def on_tick(self, t: Optional[float] = None) -> List[Dict[str, Any]]:
        t = float(self.clock() if t is None else t)
        with self._lock:
            self._refresh()
            fired = []
            due = {e["id"] for e in self._sched.due(t)}
            for r in self._rules:
                if r["id"] in due and r.get("enabled", True) and self._hours_ok(r, t) and self._cooldown_ok(r, t):
                    zone = self._zone(r["when"].get("zone"))
                    fired.append(self._fire(r, t, self._ctx(r, t, None, zone)))
            fired += self._eval_lost(t)
            return fired

    # --- evaluators
    def _eval_in_zone(self, r, persons, t, active) -> List[Dict[str, Any]]:
        w = r["when"]
        zone = self._zone(w.get("zone"))
        if zone is None or not self._hours_ok(r, t):
            return []
        out = []
        dwell = float(w.get("dwell_s", 0.0) or 0.0)
        for p in persons:
            if p.get("gid") is None or not self._person_ok(w, p) or not self._in_zone(zone, p, t):
                continue
            key = (r["id"], int(p["gid"]))
            active.add(key)
            if self._dwell(key, t, dwell) and self._cooldown_ok(r, t):
                self._presence[key]["fired"] = True
                out.append(self._fire(r, t, self._ctx(r, t, p, zone)))
        return out

    def _eval_count(self, r, persons, t, active) -> List[Dict[str, Any]]:
        w = r["when"]
        zone = self._zone(w.get("zone"))
        if (w.get("zone") and zone is None) or not self._hours_ok(r, t):
            return []
        sel = [p for p in persons if self._person_ok(w, p) and self._in_zone(zone, p, t)]
        if len(sel) <= int(w["count"]):
            return []
        key = (r["id"], "count")
        active.add(key)
        if not (self._dwell(key, t, float(w.get("dwell_s", 0.0) or 0.0)) and self._cooldown_ok(r, t)):
            return []
        self._presence[key]["fired"] = True
        cx = sum(p["x"] for p in sel) / len(sel)
        cy = sum(p["y"] for p in sel) / len(sel)
        first = dict(sel[0], x=cx, y=cy)
        return [self._fire(r, t, self._ctx(r, t, first, zone, count=len(sel)))]

    def _eval_lost(self, t: float) -> List[Dict[str, Any]]:
        out = []
        rules = [r for r in self._rules if r.get("enabled", True) and r["when"]["event"] == "lost_target"]
        for r in rules:
            w = r["when"]
            timeout = float(w.get("timeout_s", 5.0))
            want = w.get("gid")
            if want == "target":
                want = self.target_gid
                if want is None:
                    continue
            zone = self._zone(w.get("zone"))
            for gid, p in list(self._seen.items()):
                if want is not None and int(want) != gid:
                    continue
                if t - p["_t"] < timeout or (r["id"], gid) in self._lost_fired:
                    continue
                if w.get("zone") and (zone is None or not self._in_zone(zone, p, t)):
                    continue
                if not self._hours_ok(r, t) or not self._cooldown_ok(r, t):
                    continue
                self._lost_fired.add((r["id"], gid))
                out.append(self._fire(r, t, self._ctx(r, t, p, zone)))
        # forget long-gone persons
        for gid in [g for g, p in self._seen.items() if t - p["_t"] > 600.0]:
            del self._seen[gid]
        return out

    # --- firing
    def _ctx(self, r, t, p, zone, **extra) -> Dict[str, Any]:
        ctx = {"rule": r["id"], "zone": (zone or {}).get("name") or (zone or {}).get("id") or "",
               "gid": None, "x": None, "y": None, "modality": "", "count": 0, "gesture": "",
               "approach_x": None, "approach_y": None, "t": t}
        if p is not None:
            ctx["gid"] = int(p["gid"]) if p.get("gid") is not None else None
            ctx["x"], ctx["y"] = round(float(p["x"]), 3), round(float(p["y"]), 3)
            ctx["modality"] = ",".join(_modalities(p))
            ax, ay = self._approach(float(p["x"]), float(p["y"]), float(r.get("approach_m", 3.0)))
            ctx["approach_x"], ctx["approach_y"] = round(ax, 3), round(ay, 3)
        for k, v in extra.items():
            if v is not None:
                ctx[k] = int(v) if k == "gid" else v
        return ctx

    def _approach(self, px: float, py: float, dist: float) -> Tuple[float, float]:
        dist = max(dist, MIN_APPROACH_M)
        rx, ry, _ = self._pose
        dx, dy = rx - px, ry - py
        n = math.hypot(dx, dy)
        if n <= dist:  # already closer than dist: stay where we are
            return rx, ry
        return px + dx / n * dist, py + dy / n * dist

    def _fire(self, r: Dict[str, Any], t: float, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._last_fire[r["id"]] = t
        msg_tpl = r.get("message") or "{rule}: %s{zone_sfx}" % r["when"]["event"]
        msg_ctx = dict(ctx, zone_sfx=(" in " + ctx["zone"]) if ctx["zone"] else "")
        try:
            message = msg_tpl.format_map(_SafeDict({k: ("" if v is None else _fmt(v)) for k, v in msg_ctx.items()}))
        except (ValueError, IndexError):
            message = msg_tpl
        alert = {"t": t, "rule_id": r["id"], "kind": r["when"]["event"], "zone": ctx["zone"] or None,
                 "gid": ctx["gid"], "x": ctx["x"], "y": ctx["y"], "modality": ctx["modality"] or None,
                 "message": message, "alert": bool(r.get("alert", True)), "mission_id": None, "error": None,
                 "evidence": bool(r.get("evidence"))}
        if r.get("evidence") and self.start_recording is not None:
            try:
                self.start_recording("rule %s: %s" % (r["id"], message))
            except Exception as e:  # noqa: BLE001 - recorder down must not block the rule
                alert["error"] = "start_recording failed: %s" % e
        steps = r.get("then") or []
        if steps:
            mission, errs = self.build_mission(r, ctx)
            if errs:
                alert["error"] = "; ".join(([alert["error"]] if alert["error"] else []) + errs)
            elif self.submit is not None:
                try:
                    res = self.submit(mission)
                    if isinstance(res, dict):
                        res = res.get("mission_id", res)
                    alert["mission_id"] = res
                except Exception as e:  # noqa: BLE001 - submit failure must not kill the feed
                    alert["error"] = ((alert["error"] + "; ") if alert["error"] else "") + "submit failed: %s" % e
        self.history.append(alert)
        del self.history[:-200]
        if alert["alert"] and self.alert_fn is not None:
            try:
                self.alert_fn(alert)
            except Exception:  # noqa: BLE001
                pass
        return alert

    def build_mission(self, r: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        steps = render_steps(r.get("then") or [], ctx)
        if r.get("evidence") and not (steps and steps[0].get("op") == "record"):
            steps.insert(0, {"op": "record", "on": True, "mode": "evidence"})
        errs = check_rule_steps(steps) + _unresolved(steps)
        mission = {"name": r.get("name") or "rule:%s" % r["id"], "source": "rule",
                   "steps": steps, "on_fail": r.get("on_fail", "stop")}
        if not errs and self.validate_schema:
            errs += _schema_errors(mission)
        return mission, errs


def _schema_errors(mission: Dict[str, Any]) -> List[str]:
    """Validate with M1's schema when available (lazy import, optional)."""
    try:
        try:
            from . import schema  # type: ignore
        except ImportError:
            import schema  # type: ignore
    except Exception:  # noqa: BLE001 - schema module absent / broken -> skip
        return []
    try:
        _m, errs = schema.validate(mission)
        return list(errs or [])
    except Exception as e:  # noqa: BLE001
        return ["schema validation crashed: %s" % e]
