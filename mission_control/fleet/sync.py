"""Shared fleet world (zones / labels / home) + per-robot sync (CONTRACT.md 9.6).

Stored in map frame in ``fleet/data/world.json``; pushed to every robot's
mission API (``/zones``, ``/labels``, ``/home``) transformed by
T_map_robot^-1. Push is idempotent: a robot whose last push matches
(world version, alignment) is skipped, and only differing items are written.
Fleet-owned zone ids start with ``fz_``; fleet-owned label names are tracked
per robot so a deleted shared label is deleted on the robot too.
"""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from .registry import Align, Robot, _list_of, atomic_write_json, ok, poly_transform
except ImportError:  # run as a script (PYTHONPATH=/app/fleet)
    from registry import Align, Robot, _list_of, atomic_write_json, ok, poly_transform  # type: ignore

ZONE_KINDS = ("no_go", "watch", "patrol")
FLEET_ZONE_PREFIX = "fz_"


def _empty() -> Dict[str, Any]:
    return {"schema": 1, "version": 0, "updated_at": 0.0, "map_frame": "map",
            "zones": [], "labels": [], "home": None, "sync": {}}


def _finite(*vals: Any) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def clean_zone(z: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(z, dict):
        raise ValueError("zone must be an object")
    if z.get("kind") not in ZONE_KINDS:
        raise ValueError("zone.kind must be one of %s" % (ZONE_KINDS,))
    poly = z.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3 or not all(
            isinstance(p, (list, tuple)) and len(p) >= 2 and _finite(p[0], p[1]) for p in poly):
        raise ValueError("zone.polygon needs >= 3 finite [x,y] points")
    zid = str(z.get("id") or "")
    if not zid.startswith(FLEET_ZONE_PREFIX):
        zid = FLEET_ZONE_PREFIX + (zid or uuid.uuid4().hex[:8])
    return {"id": zid, "name": str(z.get("name") or ""), "kind": z["kind"],
            "polygon": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in poly],
            "active_hours": z.get("active_hours") or None}


def clean_label(lb: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(lb, dict) or not str(lb.get("name") or "").strip():
        raise ValueError("label.name required")
    if not _finite(lb.get("x"), lb.get("y")):
        raise ValueError("label.x/y must be finite numbers")
    out = {"name": str(lb["name"]).strip(), "x": round(float(lb["x"]), 4), "y": round(float(lb["y"]), 4)}
    if lb.get("yaw") is not None:
        if not _finite(lb["yaw"]):
            raise ValueError("label.yaw must be a number")
        out["yaw"] = round(float(lb["yaw"]), 5)
    return out


# --- frame helpers (map <-> robot) ------------------------------------------
def zone_frame(z: Dict[str, Any], align: Align, to_robot: bool = True) -> Dict[str, Any]:
    z = dict(z)
    z["polygon"] = poly_transform(z["polygon"], align.to_robot if to_robot else align.to_map)
    return z


def label_frame(lb: Dict[str, Any], align: Align, to_robot: bool = True) -> Dict[str, Any]:
    lb = dict(lb)
    fn = align.to_robot if to_robot else align.to_map
    lb["x"], lb["y"] = [round(v, 4) for v in fn(float(lb["x"]), float(lb["y"]))]
    if lb.get("yaw") is not None:
        lb["yaw"] = round((align.yaw_to_robot if to_robot else align.yaw_to_map)(float(lb["yaw"])), 5)
    return lb


def _same(a: Any, b: Any, tol: float = 2e-3) -> bool:
    """Structural equality with float tolerance (robot may re-round)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(k for k in a if a[k] is not None) | set(k for k in b if b[k] is not None)
        return all(_same(a.get(k), b.get(k), tol) for k in keys)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= tol
    return a == b


class FleetWorld:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        base = _empty()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                for k in base:
                    if d.get(k) is not None:
                        base[k] = d[k]
        except (OSError, ValueError):
            pass
        return base

    def _save(self, bump: bool = True) -> None:
        if bump:
            self.data["version"] = int(self.data.get("version", 0)) + 1
            self.data["updated_at"] = round(time.time(), 3)
        atomic_write_json(self.path, self.data)

    @property
    def version(self) -> int:
        return int(self.data.get("version", 0))

    def export(self) -> Dict[str, Any]:
        with self.lock:
            d = copy.deepcopy(self.data)
        d.pop("sync", None)
        return d

    # --- mutations (each bumps version only when something changed)
    def upsert_zone(self, z: Dict[str, Any]) -> Dict[str, Any]:
        cz = clean_zone(z)
        with self.lock:
            zs = self.data["zones"]
            for i, old in enumerate(zs):
                if old["id"] == cz["id"]:
                    if _same(old, cz):
                        return old
                    zs[i] = cz
                    break
            else:
                zs.append(cz)
            self._save()
            return cz

    def delete_zone(self, zid: str) -> bool:
        with self.lock:
            n = len(self.data["zones"])
            self.data["zones"] = [z for z in self.data["zones"] if z["id"] != zid and z.get("name") != zid]
            if len(self.data["zones"]) == n:
                return False
            self._save()
            return True

    def upsert_label(self, lb: Dict[str, Any]) -> Dict[str, Any]:
        cl = clean_label(lb)
        with self.lock:
            ls = self.data["labels"]
            for i, old in enumerate(ls):
                if old["name"] == cl["name"]:
                    if _same(old, cl):
                        return old
                    ls[i] = cl
                    break
            else:
                ls.append(cl)
            self._save()
            return cl

    def delete_label(self, name: str) -> bool:
        with self.lock:
            n = len(self.data["labels"])
            self.data["labels"] = [x for x in self.data["labels"] if x["name"] != name]
            if len(self.data["labels"]) == n:
                return False
            self._save()
            return True

    def set_home(self, h: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        new = None
        if h is not None:
            if not _finite(h.get("x"), h.get("y"), h.get("yaw", 0.0)):
                raise ValueError("home.x/y/yaw must be finite")
            new = {"x": round(float(h["x"]), 4), "y": round(float(h["y"]), 4),
                   "yaw": round(float(h.get("yaw", 0.0) or 0.0), 5)}
        with self.lock:
            if _same(self.data.get("home"), new):
                return new
            self.data["home"] = new
            self._save()
            return new

    # --- per-robot sync bookkeeping
    def sync_state(self, rid: str) -> Dict[str, Any]:
        with self.lock:
            return dict(self.data["sync"].get(rid) or {})

    def set_sync_state(self, rid: str, st: Dict[str, Any]) -> None:
        with self.lock:
            self.data["sync"][rid] = st
            self._save(bump=False)

    def forget_robot(self, rid: str) -> None:
        with self.lock:
            if self.data["sync"].pop(rid, None) is not None:
                self._save(bump=False)


def push(world: FleetWorld, robot: Robot, transport: Any, force: bool = False) -> Dict[str, Any]:
    """Write the shared world into one robot's mission API (robot frame).
    Returns ``{robot_id, status: up_to_date|synced|partial|no_mission|error, writes, deletes, errors}``."""
    rid = robot.id
    url = robot.spec.urls.get("mission")
    res = {"robot_id": rid, "status": "no_mission", "writes": 0, "deletes": 0, "errors": []}
    if not url:
        return res
    align = robot.spec.align
    snap = world.export()
    prev = world.sync_state(rid)
    key = {"version": snap["version"], "align": align.key()}
    if not force and prev.get("version") == key["version"] and prev.get("align") == key["align"]:
        res["status"] = "up_to_date"
        return res
    c1, dz = transport.get(url + "/zones")
    c2, dl = transport.get(url + "/labels")
    if not (ok(c1) and ok(c2)):
        res["status"] = "error"
        res["errors"].append("mission unreachable (%s/%s)" % (c1, c2))
        return res
    remote_z = {str(z.get("id")): z for z in _list_of(dz, "zones") if isinstance(z, dict)}
    remote_l = {str(x.get("name")): x for x in _list_of(dl, "labels") if isinstance(x, dict)}

    def w(code: int, what: str) -> None:
        if ok(code):
            res["writes" if not what.startswith("DEL") else "deletes"] += 1
        else:
            res["errors"].append("%s -> %s" % (what, code))

    want_z = {z["id"]: zone_frame(z, align) for z in snap["zones"]}
    for zid, z in want_z.items():
        cur = remote_z.get(zid)
        if cur is None or not _same({k: cur.get(k) for k in z}, z):
            w(transport.post(url + "/zones", z)[0], "zone " + zid)
    for zid in remote_z:
        if zid.startswith(FLEET_ZONE_PREFIX) and zid not in want_z:
            w(transport.delete(url + "/zones/" + zid)[0], "DEL zone " + zid)
    want_l = {lb["name"]: label_frame(lb, align) for lb in snap["labels"]}
    for name, lb in want_l.items():
        cur = remote_l.get(name)
        if cur is None or not _same({k: cur.get(k) for k in lb}, lb):
            w(transport.post(url + "/labels", lb)[0], "label " + name)
    for name in prev.get("labels") or []:
        if name not in want_l and name in remote_l:
            w(transport.delete(url + "/labels/" + name)[0], "DEL label " + name)
    home = robot.spec.home
    if home is None and snap.get("home"):
        h = snap["home"]
        hx, hy = align.to_robot(h["x"], h["y"])
        home = {"x": round(hx, 4), "y": round(hy, 4), "yaw": round(align.yaw_to_robot(h.get("yaw", 0.0)), 5)}
    if home is not None and not _same(prev.get("home"), home):
        w(transport.post(url + "/home", home)[0], "home")
    if res["errors"]:
        res["status"] = "partial"
        return res
    world.set_sync_state(rid, {"version": key["version"], "align": key["align"], "t": round(time.time(), 3),
                               "labels": sorted(want_l), "home": home})
    res["status"] = "synced"
    return res


def pull_merge(world: FleetWorld, robot: Robot, transport: Any) -> Dict[str, Any]:
    """Import a robot's own (non-fleet) zones / labels into the shared world
    (map frame). Existing fleet items win; nothing is deleted."""
    url = robot.spec.urls.get("mission")
    res = {"robot_id": robot.id, "zones": 0, "labels": 0, "errors": []}
    if not url:
        return res
    align = robot.spec.align
    c1, dz = transport.get(url + "/zones")
    c2, dl = transport.get(url + "/labels")
    if not ok(c1) or not ok(c2):
        res["errors"].append("mission unreachable")
        return res
    have_z = {z["id"] for z in world.export()["zones"]}
    have_l = {x["name"] for x in world.export()["labels"]}
    for z in _list_of(dz, "zones"):
        if not isinstance(z, dict) or str(z.get("id", "")).startswith(FLEET_ZONE_PREFIX):
            continue
        zid = FLEET_ZONE_PREFIX + "%s_%s" % (robot.id, z.get("id") or uuid.uuid4().hex[:6])
        if zid in have_z:
            continue
        try:
            world.upsert_zone(dict(zone_frame(z, align, to_robot=False), id=zid))
            res["zones"] += 1
        except (ValueError, KeyError) as e:
            res["errors"].append("zone %s: %s" % (z.get("id"), e))
    for lb in _list_of(dl, "labels"):
        if not isinstance(lb, dict) or str(lb.get("name")) in have_l:
            continue
        try:
            world.upsert_label(label_frame(lb, align, to_robot=False))
            res["labels"] += 1
        except (ValueError, KeyError) as e:
            res["errors"].append("label %s: %s" % (lb.get("name"), e))
    return res


def push_all(world: FleetWorld, robots: List[Robot], transport: Any, force: bool = False,
             only_online: bool = True) -> List[Dict[str, Any]]:
    out = []
    for r in robots:
        if only_online and not r.online():
            out.append({"robot_id": r.id, "status": "offline", "writes": 0, "deletes": 0, "errors": []})
            continue
        out.append(push(world, r, transport, force))
    return out
