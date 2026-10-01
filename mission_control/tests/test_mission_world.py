"""mission/world.py: store CRUD, persistence, polygon geometry, no-go raster."""
import datetime as dt
import json
import math
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mission import world as W  # noqa: E402

SQ = [[0, 0], [2, 0], [2, 2], [0, 2]]


@pytest.fixture
def store(tmp_path):
    return W.WorldStore(str(tmp_path / "world.json"))


def test_zone_crud_and_persistence(store, tmp_path):
    z = store.upsert_zone({"name": "yard", "kind": "watch", "polygon": SQ, "active_hours": "22:00-06:00"})
    assert z["id"] and z["kind"] == "watch"
    assert store.get_zone("yard")["id"] == z["id"]
    z2 = store.upsert_zone(dict(z, kind="no_go"))
    assert z2["id"] == z["id"] and len(store.zones()) == 1
    again = W.WorldStore(str(tmp_path / "world.json"))
    assert again.zones()[0]["kind"] == "no_go"
    assert again.zones()[0]["active_hours"] == "22:00-06:00"
    assert store.delete_zone(z["id"]) and not store.delete_zone(z["id"])
    assert W.WorldStore(str(tmp_path / "world.json")).zones() == []


def test_label_unique_and_home(store, tmp_path):
    store.upsert_label({"name": "gate", "x": 1, "y": 2})
    store.upsert_label({"name": "gate", "x": 3, "y": 4, "yaw": 1.5})
    assert store.labels() == [{"name": "gate", "x": 3.0, "y": 4.0, "yaw": 1.5}]
    assert store.resolve_label("GATE") == (3.0, 4.0, 1.5)
    with pytest.raises(W.WorldError):
        store.upsert_label({"name": "", "x": "a", "y": 1})
    store.set_home(1, 2, 0.5)
    assert W.WorldStore(str(tmp_path / "world.json")).get_home() == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    assert store.delete_label("gate") and store.labels() == []


def test_returned_copies_do_not_mutate_store(store):
    store.upsert_zone({"id": "a", "kind": "watch", "polygon": SQ})
    store.zones()[0]["polygon"].append([9, 9])
    assert len(store.get_zone("a")["polygon"]) == 4


def test_atomic_write_leaves_no_tmp_and_valid_json(store, tmp_path):
    def worker(i):
        store.upsert_label({"name": "l%d" % (i % 5), "x": i, "y": i})
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    files = os.listdir(str(tmp_path))
    assert files == ["world.json"]
    data = json.load(open(str(tmp_path / "world.json")))
    assert len(data["labels"]) == 5


def test_failed_write_keeps_old_file(store, tmp_path, monkeypatch):
    store.upsert_label({"name": "a", "x": 0, "y": 0})
    before = open(str(tmp_path / "world.json")).read()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(W.os, "replace", boom)
    with pytest.raises(OSError):
        store.upsert_label({"name": "b", "x": 0, "y": 0})
    assert open(str(tmp_path / "world.json")).read() == before
    assert os.listdir(str(tmp_path)) == ["world.json"]


def test_corrupt_file_loads_empty(tmp_path):
    p = tmp_path / "world.json"
    p.write_text("{nope")
    s = W.WorldStore(str(p))
    assert s.zones() == [] and s.get_home() is None


@pytest.mark.parametrize("poly,ok", [
    (SQ, True),
    ([[0, 0], [1, 0]], False),                         # < 3 points
    ([[0, 0], [2, 2], [2, 0], [0, 2]], False),         # bow-tie
    ([[0, 0], [0.2, 0], [0.2, 0.2]], False),           # area 0.02 m^2
    ([[0, 0], [1, 0], [2, 0]], False),                 # collinear, zero area
    ([[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]], True),  # concave
    ([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], True),  # explicitly closed
])
def test_validate_polygon(poly, ok):
    assert (W.validate_polygon(poly) == []) is ok


def test_invalid_zone_rejected(store):
    with pytest.raises(W.WorldError):
        store.upsert_zone({"kind": "lava", "polygon": SQ})
    with pytest.raises(W.WorldError):
        store.upsert_zone({"kind": "watch", "polygon": SQ, "active_hours": "25:00-01:00"})
    with pytest.raises(W.WorldError):
        store.upsert_zone({"kind": "watch", "polygon": [[0, 0], [2, 2], [2, 0], [0, 2]]})


def test_point_in_polygon_vectorized_and_edges():
    concave = [[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]]
    pts = np.array([[1, 0.5], [2, 3], [3.8, 3], [-1, 1], [5, 1], [2, 0.99]])
    assert W.point_in_polygon(concave, pts).tolist() == [True, False, True, False, False, True]
    assert W.point_in_polygon(SQ, [1, 1]) is True
    assert W.point_in_polygon(SQ, [2.0001, 1]) is False
    # vertex-height ray must not double count
    tri = [[0, 0], [2, 1], [0, 2]]
    assert W.point_in_polygon(tri, [1, 1]) is True
    assert W.point_in_polygon(tri, [-1, 1]) is False
    assert W.point_in_polygon(tri, [3, 1]) is False
    big = np.random.RandomState(0).uniform(-1, 3, (5000, 2))
    inside = W.point_in_polygon(SQ, big)
    expect = (big[:, 0] > 0) & (big[:, 0] < 2) & (big[:, 1] > 0) & (big[:, 1] < 2)
    assert (inside == expect).mean() > 0.999


def test_perimeter_points():
    pts = W.polygon_perimeter_points(SQ, spacing=0.5)
    assert len(pts) == 16 and pts[0] == [0.0, 0.0]
    for x, y in pts:
        assert min(abs(x), abs(x - 2), abs(y), abs(y - 2)) < 1e-6
    ins = W.polygon_perimeter_points(SQ, spacing=0.5, inset=0.3)
    assert all(0 < x < 2 and 0 < y < 2 for x, y in ins)


def test_no_go_mask_rasterization():
    meta = {"resolution": 0.5, "origin_x": -1.0, "origin_y": -1.0, "width": 8, "height": 6}
    poly = [[0.1, 0.1], [1.9, 0.1], [1.9, 0.9], [0.1, 0.9]]  # cols 2..5, rows 2..3
    m = W.rasterize_polygons([poly], meta)
    assert m.dtype == bool and m.shape == (48,)
    g = m.reshape(6, 8)
    exp = np.zeros((6, 8), bool)
    exp[2:4, 2:6] = True
    assert (g == exp).all()
    # outside grid / partially outside -> clipped, no crash
    m2 = W.rasterize_polygons([[[-5, -5], [-4, -5], [-4, -4]], [[2, 1], [9, 1], [9, 9]]], meta).reshape(6, 8)
    assert not m2[:, :6].any() and m2[:, 6:].any()


def test_thin_zone_not_lost():
    meta = {"resolution": 1.0, "origin_x": 0.0, "origin_y": 0.0, "width": 5, "height": 5}
    thin = [[0.1, 2.1], [4.9, 2.1], [4.9, 2.4], [0.1, 2.4]]  # no cell centre inside
    g = W.rasterize_polygons([thin], meta).reshape(5, 5)
    assert g[2, :].all() and g.sum() == 5


def test_store_no_go_mask_and_active_hours(store):
    meta = {"resolution": 1.0, "origin_x": 0.0, "origin_y": 0.0, "width": 4, "height": 4}
    store.upsert_zone({"id": "ng", "kind": "no_go", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
                       "active_hours": "22:00-06:00"})
    store.upsert_zone({"id": "w", "kind": "watch", "polygon": [[2, 2], [4, 2], [4, 4], [2, 4]]})
    assert store.no_go_mask(meta).sum() >= 4
    noon = dt.datetime(2026, 10, 1, 12, 0)
    night = dt.datetime(2026, 10, 1, 23, 0)
    assert store.no_go_mask(meta, now=noon).sum() == 0
    assert store.no_go_mask(meta, now=night).reshape(4, 4)[0:2, 0:2].all()
    assert store.zone_at(1, 1)["id"] == "ng"
    assert store.zone_at(1, 1, now=noon) is None
    assert [z["id"] for z in store.zones_active(noon)] == ["w"]


@pytest.mark.parametrize("spec,hhmm,exp", [
    ("22:00-06:00", (23, 30), True), ("22:00-06:00", (0, 0), True), ("22:00-06:00", (5, 59), True),
    ("22:00-06:00", (6, 0), False), ("22:00-06:00", (12, 0), False), ("22:00-06:00", (22, 0), True),
    ("08:00-17:30", (17, 29), True), ("08:00-17:30", (17, 30), False), ("00:00-24:00", (13, 0), True),
    (None, (3, 0), True),
])
def test_hours_active(spec, hhmm, exp):
    now = dt.datetime(2026, 10, 1, *hhmm)
    assert W.hours_active(spec, now) is exp
    assert W.hours_active(spec, now.timestamp()) is exp


def test_base_to_world_transform():
    pose = {"x": 1.0, "y": 2.0, "yaw": math.pi / 2}
    wx, wy = W.base_to_world(pose, 3.0, 0.0)  # 3 m ahead while facing +y
    assert abs(wx - 1.0) < 1e-9 and abs(wy - 5.0) < 1e-9
    ps = W.persons_to_world([{"gid": 1, "x": 3.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "modality": ["rgb"]}], pose)
    assert abs(ps[0]["y"] - 5.0) < 1e-9 and abs(ps[0]["vy"] - 1.0) < 1e-9
    assert ps[0]["x_base"] == 3.0 and ps[0]["modality"] == ["rgb"]


def test_seed_world_json_loads():
    p = os.path.join(os.path.dirname(W.__file__), "data", "world.json")
    s = W.WorldStore(p)
    assert s.zones() == [] and s.labels() == []


def test_version_updated_at_export_replace_all(store, tmp_path):
    assert store.revision == 0
    store.upsert_zone({"id": "a", "kind": "watch", "polygon": SQ})
    store.upsert_label({"name": "gate", "x": 1, "y": 1})
    store.upsert_rule({"id": "r", "when": {"event": "schedule", "cron": "0 *"}, "then": []})
    ex = store.export()
    assert ex["version"] == 3 and ex["updated_at"] > 0
    v0 = store.version
    out = store.replace_all({"zones": [{"id": "b", "kind": "no_go", "polygon": SQ}],
                             "labels": [{"name": "door", "x": 2, "y": 3, "yaw": 0}], "home": {"x": 1, "y": 1}})
    assert store.version != v0 and out["version"] == 4
    assert [z["id"] for z in store.zones()] == ["b"] and store.labels()[0]["name"] == "door"
    assert store.rules()[0]["id"] == "r"                                    # absent section kept
    assert store.get_home() == {"x": 1.0, "y": 1.0, "yaw": 0.0}
    again = W.WorldStore(str(tmp_path / "world.json"))
    assert again.revision == 4 and again.zones()[0]["kind"] == "no_go"


def test_replace_all_is_all_or_nothing(store):
    store.upsert_zone({"id": "a", "kind": "watch", "polygon": SQ})
    with pytest.raises(W.WorldError) as ei:
        store.replace_all({"zones": [{"id": "ok", "kind": "watch", "polygon": SQ},
                                     {"id": "bad", "kind": "watch", "polygon": [[0, 0], [1, 1]]}],
                           "labels": [{"name": "x", "x": 0, "y": 0}, {"name": "x", "x": 1, "y": 1}]})
    assert len(ei.value.errors) == 2
    assert [z["id"] for z in store.zones()] == ["a"] and store.labels() == [] and store.revision == 1
