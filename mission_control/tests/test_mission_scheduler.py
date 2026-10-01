"""mission/scheduler.py: cron subset parsing, next_fire across midnight, due()."""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mission.scheduler import CronExpr, Scheduler, next_fire  # noqa: E402

D = dt.datetime


def test_two_field_wraps_midnight():
    c = CronExpr("*/30 22-06")
    assert c.hours == {22, 23, 0, 1, 2, 3, 4, 5, 6}
    assert c.minutes == {0, 30}
    assert c.matches(D(2026, 10, 1, 23, 30)) and c.matches(D(2026, 10, 2, 6, 30))
    assert not c.matches(D(2026, 10, 1, 7, 0)) and not c.matches(D(2026, 10, 1, 23, 15))


def test_next_fire_across_midnight():
    c = CronExpr("*/30 22-06")
    assert c.next_fire(D(2026, 10, 1, 23, 45)) == D(2026, 10, 2, 0, 0)
    assert c.next_fire(D(2026, 10, 1, 6, 30)) == D(2026, 10, 1, 22, 0)
    assert c.next_fire(D(2026, 10, 1, 12, 0)) == D(2026, 10, 1, 22, 0)
    assert c.next_fire(D(2026, 12, 31, 23, 59, 30)) == D(2027, 1, 1, 0, 0)
    assert next_fire("*/30 22-06", D(2026, 10, 1, 22, 0)) == D(2026, 10, 1, 22, 30)  # strictly after


def test_five_field():
    c = CronExpr("0 */2 * * *")
    assert c.hours == set(range(0, 24, 2)) and c.minutes == {0}
    assert c.next_fire(D(2026, 10, 1, 13, 5)) == D(2026, 10, 1, 14, 0)
    assert c.next_fire(D(2026, 10, 1, 22, 0)) == D(2026, 10, 2, 0, 0)
    # 2026-10-01 is a Thursday (cron dow 4); weekends only:
    we = CronExpr("15 3 * * 6,0")
    assert we.next_fire(D(2026, 10, 1, 12, 0)) == D(2026, 10, 3, 3, 15)
    assert CronExpr("0 0 * * 7").next_fire(D(2026, 10, 1)) == D(2026, 10, 4, 0, 0)
    assert CronExpr("0 0 1 1 *").next_fire(D(2026, 10, 1)) == D(2027, 1, 1)
    assert CronExpr("5,10-12 1 * * *").minutes == {5, 10, 11, 12}
    assert CronExpr("0-30/10 1").minutes == {0, 10, 20, 30}


@pytest.mark.parametrize("bad", ["", "* * *", "61 *", "* 24", "*/0 *", "a b", "0 0 0 * *", "0 0 * 13 *"])
def test_bad_expressions(bad):
    with pytest.raises(ValueError):
        CronExpr(bad)


def test_scheduler_due_once_per_minute():
    s = Scheduler([{"id": "p", "cron": "*/30 22-06", "zone": "yard"},
                   {"id": "off", "cron": "* *", "enabled": False},
                   {"id": "bad", "cron": "nope"}])
    assert "bad" in s.errors
    t = D(2026, 10, 1, 23, 30, 5)
    assert [e["id"] for e in s.due(t)] == ["p"]
    assert s.due(t + dt.timedelta(seconds=20)) == []
    assert s.due(D(2026, 10, 1, 23, 31)) == []
    assert [e["zone"] for e in s.due(D(2026, 10, 2, 0, 0).timestamp())] == ["yard"]
    assert s.next_fires(D(2026, 10, 2, 0, 0))["p"] == "2026-10-02T00:30:00"
