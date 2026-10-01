"""Tiny cron-like scheduler for timed patrols (M5).

Expression forms (local time):

* 2 fields  ``"MIN HOUR"``                 e.g. ``"*/30 22-06"`` (every 30 min, 22:00..06:59)
* 5 fields  ``"MIN HOUR DOM MON DOW"``     e.g. ``"0 */2 * * *"``

Per field: ``*``, ``*/n``, ``a``, ``a-b``, ``a-b/n``, comma lists. A range with
a > b wraps (``22-06`` = 22..23,0..6; ``5-1`` for DOW = Fri..Mon). DOW: 0 or 7 =
Sunday. When both DOM and DOW are restricted, a day matches if either does
(classic cron semantics).

``Scheduler(entries).due(now)`` returns the entries whose expression matches the
minute of ``now``; each entry fires at most once per minute even if polled often.
Entries are dicts with at least ``{"id", "cron"}`` (anything else is passed back).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Set

_FIELDS = (("minute", 0, 59), ("hour", 0, 23), ("dom", 1, 31), ("month", 1, 12), ("dow", 0, 7))


def _to_dt(now: Any) -> _dt.datetime:
    if now is None:
        return _dt.datetime.now()
    if isinstance(now, _dt.datetime):
        return now
    return _dt.datetime.fromtimestamp(float(now))


def _parse_field(spec: str, lo: int, hi: int, name: str) -> Set[int]:
    out = set()  # type: Set[int]
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty %s field" % name)
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
            if step <= 0:
                raise ValueError("bad step in %s" % name)
        if part == "*":
            a, b = lo, (6 if name == "dow" else hi)
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
        else:
            a = b = int(part)
            if step != 1:
                b = hi  # "5/15" == "5-59/15"
        if not (lo <= a <= hi and lo <= b <= hi):
            raise ValueError("%s value out of range %d-%d: %r" % (name, lo, hi, spec))
        if a <= b:
            vals = list(range(a, b + 1))
        else:  # wrap-around range, e.g. 22-06
            vals = list(range(a, hi + 1)) + list(range(lo, b + 1))
        out.update(v % 7 if name == "dow" else v for v in vals[::step])
    return out


class CronExpr:
    """Parsed expression. ``matches(dt)`` / ``next_fire(now)``."""

    def __init__(self, expr: str):
        parts = str(expr or "").split()
        if len(parts) == 2:
            parts += ["*", "*", "*"]
        if len(parts) != 5:
            raise ValueError("cron needs 2 or 5 fields, got %r" % (expr,))
        self.expr = expr
        sets = [_parse_field(p, lo, hi, n) for p, (n, lo, hi) in zip(parts, _FIELDS)]
        self.minutes, self.hours, self.doms, self.months, self.dows = sets
        self._dom_any = parts[2] == "*"
        self._dow_any = parts[4] == "*"

    def _day_ok(self, d: _dt.date) -> bool:
        if d.month not in self.months:
            return False
        dom_ok = d.day in self.doms
        dow_ok = ((d.weekday() + 1) % 7) in self.dows
        if self._dom_any and self._dow_any:
            return True
        if self._dom_any:
            return dow_ok
        if self._dow_any:
            return dom_ok
        return dom_ok or dow_ok

    def matches(self, now: Any) -> bool:
        d = _to_dt(now)
        return d.minute in self.minutes and d.hour in self.hours and self._day_ok(d.date())

    def next_fire(self, now: Any = None, max_days: int = 400) -> Optional[_dt.datetime]:
        """First matching minute strictly after ``now`` (None if none within max_days)."""
        start = _to_dt(now).replace(second=0, microsecond=0) + _dt.timedelta(minutes=1)
        hours = sorted(self.hours)
        mins = sorted(self.minutes)
        day = start.date()
        for i in range(max_days):
            if self._day_ok(day):
                first = i == 0
                for h in hours:
                    if first and h < start.hour:
                        continue
                    for m in mins:
                        if first and h == start.hour and m < start.minute:
                            continue
                        return _dt.datetime(day.year, day.month, day.day, h, m, tzinfo=start.tzinfo)
            day = day + _dt.timedelta(days=1)
        return None


def parse(expr: str) -> CronExpr:
    return CronExpr(expr)


def next_fire(expr: str, now: Any = None) -> Optional[_dt.datetime]:
    return CronExpr(expr).next_fire(now)


class Scheduler:
    """Holds entries ``{"id", "cron", "enabled"?: bool, ...}``; invalid crons are
    skipped and reported in ``errors``."""

    def __init__(self, entries: Optional[Iterable[Dict[str, Any]]] = None):
        self.errors = {}  # type: Dict[str, str]
        self._entries = []  # type: List[tuple]
        self._last = {}  # type: Dict[str, tuple]
        self.set_entries(entries or [])

    def set_entries(self, entries: Iterable[Dict[str, Any]]) -> None:
        self.errors = {}
        ents = []
        for i, e in enumerate(entries):
            eid = str(e.get("id") or "sched_%d" % i)
            try:
                ents.append((eid, CronExpr(e.get("cron", "")), e))
            except ValueError as ex:
                self.errors[eid] = str(ex)
        self._entries = ents

    def due(self, now: Any = None) -> List[Dict[str, Any]]:
        d = _to_dt(now)
        key = (d.year, d.month, d.day, d.hour, d.minute)
        out = []
        for eid, cx, e in self._entries:
            if e.get("enabled", True) is False:
                continue
            if cx.matches(d) and self._last.get(eid) != key:
                self._last[eid] = key
                out.append(e)
        return out

    def next_fires(self, now: Any = None) -> Dict[str, Optional[str]]:
        return {eid: (lambda n: n.isoformat() if n else None)(cx.next_fire(now))
                for eid, cx, _e in self._entries}
