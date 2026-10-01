"""World model for the mission pillar (M5): zones, semantic labels, home pose, rules.

Storage: one JSON file (default ``mission/data/world.json``, env ``MISSION_WORLD``),
written atomically (tmp file in the same dir + fsync + os.replace).

File layout::

    {
      "schema": 1,
      "version": 12,              # revision, +1 on every write (fleet sync / cache key)
      "updated_at": 1790000000.0, # unix ts of the last write
      "zones":  [{"id": "z1", "name": "gate", "kind": "no_go|watch|patrol",
                  "polygon": [[x, y], ...], "active_hours": "22:00-06:00"}],
      "labels": [{"name": "gate", "x": 1.0, "y": 2.0, "yaw": 0.0}],
      "home":   {"x": 0.0, "y": 0.0, "yaw": 0.0} | null,
      "rules":  [ <rule dict, see mission/rules.py> ]
    }

All coordinates are in the world (mapping grid) frame, metres. ``active_hours`` is
optional ("HH:MM-HH:MM", may cross midnight; missing = always active).

Grid convention (same as ``mapping/social_layer.py``): ``grid_meta =
{resolution, origin_x, origin_y, width, height}``; cell (row r, col c) has its centre
at ``(origin_x + (c + 0.5) * res, origin_y + (r + 0.5) * res)``; masks are flat
row-major arrays of length ``width * height``.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ZONE_KINDS = ("no_go", "watch", "patrol")
MIN_AREA_M2 = 0.05
DEFAULT_PATH = os.environ.get(
    "MISSION_WORLD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "world.json"))

_HOURS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


class WorldError(ValueError):
    """Validation error; ``errors`` holds the individual messages."""

    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


# --------------------------------------------------------------------------- time

def parse_hours(spec: str) -> Tuple[int, int]:
    """"HH:MM-HH:MM" -> (start_min, end_min) minutes of day. Raises ValueError."""
    m = _HOURS_RE.match(str(spec or ""))
    if not m:
        raise ValueError("active_hours must be 'HH:MM-HH:MM', got %r" % (spec,))
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    s, e = h1 * 60 + m1, h2 * 60 + m2
    if m1 >= 60 or m2 >= 60 or s > 1440 or e > 1440:
        raise ValueError("active_hours out of range: %r" % (spec,))
    return s % 1440, e % 1440  # "24:00" == "00:00"


def to_datetime(now: Any = None) -> _dt.datetime:
    """Accept None (now), unix float or datetime; returns a naive local datetime."""
    if now is None:
        return _dt.datetime.now()
    if isinstance(now, _dt.datetime):
        return now
    return _dt.datetime.fromtimestamp(float(now))


def hours_active(spec: Optional[str], now: Any = None) -> bool:
    """True if ``now`` (local time) falls inside ``spec``. Empty spec = always.
    Start inclusive, end exclusive; "22:00-06:00" crosses midnight; equal
    start/end means the whole day."""
    if not spec:
        return True
    start, end = parse_hours(spec)
    d = to_datetime(now)
    cur = d.hour * 60 + d.minute
    if start == end:
        return True
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end


# ------------------------------------------------------------------- geometry

def _poly_array(poly: Any) -> np.ndarray:
    a = np.asarray(poly, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("polygon must be a list of [x, y]")
    a = a[:, :2]
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]  # tolerate explicitly closed rings
    return a


def polygon_area(poly: Any) -> float:
    """Unsigned shoelace area (m^2)."""
    a = _poly_array(poly)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _orient(p, q, r) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_seg(p, q, r, eps=1e-12) -> bool:
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)


def segments_intersect(p1, p2, p3, p4, eps: float = 1e-12) -> bool:
    """Closed-segment intersection test (touching counts)."""
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_seg(p3, p1, p4):
        return True
    if abs(d2) <= eps and _on_seg(p3, p2, p4):
        return True
    if abs(d3) <= eps and _on_seg(p1, p3, p2):
        return True
    if abs(d4) <= eps and _on_seg(p1, p4, p2):
        return True
    return False


def is_simple_polygon(poly: Any) -> bool:
    """No two non-adjacent edges intersect (O(n^2), fine for UI-drawn zones)."""
    a = _poly_array(poly)
    n = len(a)
    if n < 3:
        return False
    for i in range(n):
        p1, p2 = a[i], a[(i + 1) % n]
        if np.allclose(p1, p2):
            return False  # duplicate consecutive vertex
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # adjacent edges share a vertex
            if segments_intersect(p1, p2, a[j], a[(j + 1) % n]):
                return False
    return True


def validate_polygon(poly: Any, min_area: float = MIN_AREA_M2) -> List[str]:
    """Return a list of error strings (empty = valid)."""
    try:
        a = _poly_array(poly)
    except (ValueError, TypeError) as e:
        return [str(e)]
    if not np.all(np.isfinite(a)):
        return ["polygon has non-finite coordinates"]
    if len(a) < 3:
        return ["polygon needs >= 3 points"]
    errs = []
    if not is_simple_polygon(a):
        errs.append("polygon is self-intersecting")
    area = polygon_area(a)
    if area <= min_area:
        errs.append("polygon area %.3f m^2 <= %.2f m^2" % (area, min_area))
    return errs


def point_in_polygon(poly: Any, pts: Any) -> np.ndarray:
    """Vectorised even-odd ray casting. ``pts``: (N,2) or a single [x, y].
    Returns bool (N,) (or a 0-d bool for a single point). Points exactly on an
    edge follow the half-open convention (left/bottom edges in, right/top out)."""
    a = _poly_array(poly)
    p = np.asarray(pts, dtype=np.float64)
    single = p.ndim == 1
    p = p.reshape(-1, 2)
    if len(a) < 3:
        out = np.zeros(len(p), dtype=bool)
        return out[0] if single else out
    x, y = p[:, 0:1], p[:, 1:2]                   # (N,1)
    x1, y1 = a[:, 0][None, :], a[:, 1][None, :]   # (1,M)
    x2, y2 = np.roll(a[:, 0], -1)[None, :], np.roll(a[:, 1], -1)[None, :]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hit = crosses & (x < xint)
    out = (np.count_nonzero(hit, axis=1) % 2) == 1
    return bool(out[0]) if single else out


def polygon_perimeter_points(poly: Any, spacing: float = 1.0, inset: float = 0.0) -> List[List[float]]:
    """Waypoints evenly spaced (~``spacing`` m) along the closed ring, starting at
    vertex 0 (not repeated at the end). ``inset`` > 0 shrinks the ring toward the
    centroid by that many metres (crude, fine for convex-ish patrol zones)."""
    a = _poly_array(poly)
    if len(a) < 2:
        return a.tolist()
    if inset > 0:
        c = a.mean(axis=0)
        d = a - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n < 1e-9] = 1e-9
        a = c + d * np.clip((n - inset) / n, 0.0, 1.0)
    ring = np.vstack([a, a[:1]])
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return [a[0].tolist()]
    k = max(int(round(total / max(float(spacing), 1e-3))), 3)
    s = np.linspace(0.0, total, k, endpoint=False)
    xs = np.interp(s, cum, ring[:, 0])
    ys = np.interp(s, cum, ring[:, 1])
    return [[round(float(px), 3), round(float(py), 3)] for px, py in zip(xs, ys)]


def _meta(meta: Any) -> Tuple[float, float, float, int, int]:
    g = meta.get if isinstance(meta, dict) else (lambda k, d=None: getattr(meta, k, d))
    return (float(g("resolution")), float(g("origin_x", 0.0) or 0.0), float(g("origin_y", 0.0) or 0.0),
            int(g("width")), int(g("height")))


def rasterize_polygons(polys: Iterable[Any], grid_meta: Any) -> np.ndarray:
    """Bool flat row-major (h*w) mask: a cell is set when its centre is inside a
    polygon OR a polygon edge passes through it (conservative: thin zones are
    never lost). Pure numpy, bbox-limited per polygon."""
    res, ox, oy, w, h = _meta(grid_meta)
    mask = np.zeros((h, w), dtype=bool)
    for poly in polys:
        try:
            a = _poly_array(poly)
        except (ValueError, TypeError):
            continue
        if len(a) < 3:
            continue
        c0 = max(int(math.floor((a[:, 0].min() - ox) / res)), 0)
        c1 = min(int(math.floor((a[:, 0].max() - ox) / res)) + 1, w)
        r0 = max(int(math.floor((a[:, 1].min() - oy) / res)), 0)
        r1 = min(int(math.floor((a[:, 1].max() - oy) / res)) + 1, h)
        if c0 >= c1 or r0 >= r1:
            continue
        xs = ox + (np.arange(c0, c1) + 0.5) * res
        ys = oy + (np.arange(r0, r1) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)
        inside = point_in_polygon(a, np.stack([gx.ravel(), gy.ravel()], axis=1))
        mask[r0:r1, c0:c1] |= inside.reshape(gy.shape)
        # edges: sample at res/2 and mark the containing cells
        ring = np.vstack([a, a[:1]])
        for p, q in zip(ring[:-1], ring[1:]):
            n = max(int(math.ceil(np.linalg.norm(q - p) / (res * 0.5))), 1)
            t = np.linspace(0.0, 1.0, n + 1)
            sx = p[0] + (q[0] - p[0]) * t
            sy = p[1] + (q[1] - p[1]) * t
            cc = np.floor((sx - ox) / res).astype(int)
            rr = np.floor((sy - oy) / res).astype(int)
            ok = (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
            mask[rr[ok], cc[ok]] = True
    return mask.reshape(-1)


# ------------------------------------------------------------- frames

def base_to_world(pose: Any, x: float, y: float) -> Tuple[float, float]:
    """Transform a base-frame point to world given robot pose (x, y, yaw)."""
    px, py, yaw = _pose(pose)
    c, s = math.cos(yaw), math.sin(yaw)
    return px + c * x - s * y, py + s * x + c * y


def _pose(pose: Any) -> Tuple[float, float, float]:
    if pose is None:
        return 0.0, 0.0, 0.0
    if isinstance(pose, dict):
        return float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("yaw", 0.0) or 0.0)
    if isinstance(pose, (list, tuple)):
        return float(pose[0]), float(pose[1]), float(pose[2]) if len(pose) > 2 else 0.0
    return float(getattr(pose, "x", 0.0)), float(getattr(pose, "y", 0.0)), float(getattr(pose, "yaw", 0.0))


def persons_to_world(persons: Iterable[Any], pose: Any) -> List[Dict[str, Any]]:
    """PersonTrack dicts/objects (base frame) -> new dicts with world x, y, vx, vy
    (rotated) plus ``x_base``/``y_base``; other fields copied unchanged."""
    _, _, yaw = _pose(pose)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for p in persons or []:
        d = dict(p) if isinstance(p, dict) else (p.to_dict() if hasattr(p, "to_dict") else dict(vars(p)))
        bx, by = float(d.get("x", 0.0)), float(d.get("y", 0.0))
        wx, wy = base_to_world(pose, bx, by)
        vx, vy = float(d.get("vx", 0.0) or 0.0), float(d.get("vy", 0.0) or 0.0)
        d.update(x=wx, y=wy, vx=c * vx - s * vy, vy=s * vx + c * vy, x_base=bx, y_base=by)
        out.append(d)
    return out


# --------------------------------------------------------------- validation

def validate_zone(z: Dict[str, Any]) -> List[str]:
    errs = []
    if not isinstance(z, dict):
        return ["zone must be an object"]
    if z.get("kind") not in ZONE_KINDS:
        errs.append("zone.kind must be one of %s" % (ZONE_KINDS,))
    errs += ["zone.polygon: " + e for e in validate_polygon(z.get("polygon"))]
    if z.get("active_hours"):
        try:
            parse_hours(z["active_hours"])
        except ValueError as e:
            errs.append(str(e))
    return errs


def validate_label(lb: Dict[str, Any]) -> List[str]:
    if not isinstance(lb, dict):
        return ["label must be an object"]
    errs = []
    if not str(lb.get("name") or "").strip():
        errs.append("label.name required")
    for k in ("x", "y"):
        try:
            if not math.isfinite(float(lb.get(k))):
                raise ValueError
        except (TypeError, ValueError):
            errs.append("label.%s must be a finite number" % k)
    if lb.get("yaw") is not None:
        try:
            float(lb["yaw"])
        except (TypeError, ValueError):
            errs.append("label.yaw must be a number")
    return errs


def _empty() -> Dict[str, Any]:
    return {"schema": 1, "version": 0, "updated_at": 0.0, "zones": [], "labels": [], "home": None, "rules": []}


def _clean_zone(zone: Dict[str, Any]) -> Dict[str, Any]:
    errs = validate_zone(zone)
    if errs:
        raise WorldError(errs)
    a = _poly_array(zone["polygon"])
    z = {"id": str(zone.get("id") or "z_" + uuid.uuid4().hex[:8]),
         "name": str(zone.get("name") or ""),
         "kind": zone["kind"],
         "polygon": [[float(x), float(y)] for x, y in a],
         "active_hours": zone.get("active_hours") or None}
    if not z["name"]:
        z["name"] = z["id"]
    return z


def _clean_label(label: Dict[str, Any]) -> Dict[str, Any]:
    errs = validate_label(label)
    if errs:
        raise WorldError(errs)
    lb = {"name": str(label["name"]).strip(), "x": float(label["x"]), "y": float(label["y"])}
    if label.get("yaw") is not None:
        lb["yaw"] = float(label["yaw"])
    return lb


# ------------------------------------------------------------------- store

class WorldStore:
    """Thread-safe JSON-backed store. All getters return deep copies."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_PATH
        self._lock = threading.RLock()
        self._data = self._load()
        self._mem_rev = 0  # in-process counter, also bumped on reload()

    @property
    def version(self) -> Any:
        """Changes on every mutation / reload (cache key for masks and rules)."""
        return (int(self._data.get("version", 0) or 0), self._mem_rev)

    @property
    def revision(self) -> int:
        """Persisted revision counter (``version`` field of the file)."""
        return int(self._data.get("version", 0) or 0)

    # --- io
    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return _empty()
        base = _empty()
        if isinstance(d, dict):
            for k in base:
                if k in d and d[k] is not None:
                    base[k] = d[k]
        return base

    def _save(self) -> None:
        self._data["version"] = int(self._data.get("version", 0) or 0) + 1
        self._data["updated_at"] = round(time.time(), 3)
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".world.", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._mem_rev += 1

    def reload(self) -> None:
        with self._lock:
            self._data = self._load()
            self._mem_rev += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def export(self) -> Dict[str, Any]:
        """Full world dict for fleet sync (``version``, ``updated_at`` included)."""
        return self.snapshot()

    def replace_all(self, world: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the sections present in ``world`` (zones / labels / home /
        rules) in one atomic write; absent sections are kept. Everything is
        validated first: on any error nothing changes (WorldError lists all).
        Returns the new export()."""
        if not isinstance(world, dict):
            raise WorldError(["world must be an object"])
        new, errs = {}, []
        if "zones" in world:
            zs, ids = [], set()
            for i, z in enumerate(world.get("zones") or []):
                try:
                    cz = _clean_zone(z)
                except WorldError as e:
                    errs += ["zones[%d]: %s" % (i, m) for m in e.errors]
                    continue
                if cz["id"] in ids:
                    errs.append("zones[%d]: duplicate id %s" % (i, cz["id"]))
                ids.add(cz["id"])
                zs.append(cz)
            new["zones"] = zs
        if "labels" in world:
            ls, names = [], set()
            for i, lb in enumerate(world.get("labels") or []):
                try:
                    cl = _clean_label(lb)
                except WorldError as e:
                    errs += ["labels[%d]: %s" % (i, m) for m in e.errors]
                    continue
                if cl["name"] in names:
                    errs.append("labels[%d]: duplicate name %s" % (i, cl["name"]))
                names.add(cl["name"])
                ls.append(cl)
            new["labels"] = ls
        if "home" in world:
            h = world.get("home")
            if h:
                try:
                    new["home"] = {"x": float(h["x"]), "y": float(h["y"]), "yaw": float(h.get("yaw", 0.0) or 0.0)}
                except (KeyError, TypeError, ValueError):
                    errs.append("home must be {x, y, yaw?}")
            else:
                new["home"] = None
        if "rules" in world:
            if not isinstance(world.get("rules") or [], list):
                errs.append("rules must be a list")
            else:
                new["rules"] = [dict(r, id=str(r.get("id") or "r_" + uuid.uuid4().hex[:8]))
                                for r in (world.get("rules") or [])]
        if errs:
            raise WorldError(errs)
        with self._lock:
            old = self._data
            self._data = dict(copy.deepcopy(old), **new)
            try:
                self._save()
            except BaseException:
                self._data = old
                raise
            return copy.deepcopy(self._data)

    # --- zones
    def zones(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(z) for z in self._data["zones"] if kind is None or z.get("kind") == kind]

    def get_zone(self, key: str) -> Optional[Dict[str, Any]]:
        """Lookup by id, then by name."""
        with self._lock:
            for z in self._data["zones"]:
                if z.get("id") == key:
                    return copy.deepcopy(z)
            for z in self._data["zones"]:
                if z.get("name") == key:
                    return copy.deepcopy(z)
        return None

    def upsert_zone(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        """Create (no/unknown id) or replace (existing id). Raises WorldError."""
        z = _clean_zone(zone)
        with self._lock:
            zs = self._data["zones"]
            for i, old in enumerate(zs):
                if old.get("id") == z["id"]:
                    zs[i] = z
                    break
            else:
                zs.append(z)
            self._save()
        return copy.deepcopy(z)

    def delete_zone(self, key: str) -> bool:
        with self._lock:
            zs = self._data["zones"]
            keep = [z for z in zs if z.get("id") != key]
            if len(keep) == len(zs):
                keep = [z for z in zs if z.get("name") != key]
            if len(keep) == len(zs):
                return False
            self._data["zones"] = keep
            self._save()
            return True

    # --- labels
    def labels(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data["labels"])

    def get_label(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for lb in self._data["labels"]:
                if lb.get("name") == name:
                    return copy.deepcopy(lb)
            low = str(name).strip().lower()
            for lb in self._data["labels"]:
                if str(lb.get("name", "")).lower() == low:
                    return copy.deepcopy(lb)
        return None

    def upsert_label(self, label: Dict[str, Any]) -> Dict[str, Any]:
        """Names are unique: same name replaces the existing label."""
        lb = _clean_label(label)
        with self._lock:
            ls = self._data["labels"]
            for i, old in enumerate(ls):
                if old.get("name") == lb["name"]:
                    ls[i] = lb
                    break
            else:
                ls.append(lb)
            self._save()
        return dict(lb)

    def delete_label(self, name: str) -> bool:
        with self._lock:
            ls = self._data["labels"]
            keep = [lb for lb in ls if lb.get("name") != name]
            if len(keep) == len(ls):
                return False
            self._data["labels"] = keep
            self._save()
            return True

    # --- home
    def get_home(self) -> Optional[Dict[str, float]]:
        with self._lock:
            h = self._data.get("home")
            return dict(h) if h else None

    def set_home(self, x: float, y: float, yaw: float = 0.0) -> Dict[str, float]:
        h = {"x": float(x), "y": float(y), "yaw": float(yaw or 0.0)}
        if not all(math.isfinite(v) for v in h.values()):
            raise WorldError(["home must be finite"])
        with self._lock:
            self._data["home"] = h
            self._save()
        return dict(h)

    # --- rules (raw storage; validation lives in mission.rules.validate_rule)
    def rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data["rules"])

    def upsert_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        r = copy.deepcopy(rule)
        r["id"] = str(r.get("id") or "r_" + uuid.uuid4().hex[:8])
        with self._lock:
            rs = self._data["rules"]
            for i, old in enumerate(rs):
                if old.get("id") == r["id"]:
                    rs[i] = r
                    break
            else:
                rs.append(r)
            self._save()
        return copy.deepcopy(r)

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            rs = self._data["rules"]
            keep = [r for r in rs if r.get("id") != rule_id]
            if len(keep) == len(rs):
                return False
            self._data["rules"] = keep
            self._save()
            return True

    # --- queries
    def zones_active(self, now: Any = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        return [z for z in self.zones(kind) if hours_active(z.get("active_hours"), now)]

    def zone_at(self, x: float, y: float, now: Any = None, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """First zone containing (x, y); ``now`` given -> only active zones."""
        zs = self.zones(kind) if now is None else self.zones_active(now, kind)
        for z in zs:
            if point_in_polygon(z["polygon"], [x, y]):
                return z
        return None

    def zones_at(self, x: float, y: float, now: Any = None) -> List[Dict[str, Any]]:
        zs = self.zones() if now is None else self.zones_active(now)
        return [z for z in zs if point_in_polygon(z["polygon"], [x, y])]

    def no_go_mask(self, grid_meta: Any, now: Any = None) -> np.ndarray:
        """Bool flat row-major mask of no_go zones for the planner. ``now`` None ->
        every no_go zone regardless of active_hours (conservative)."""
        zs = self.zones("no_go") if now is None else self.zones_active(now, "no_go")
        return rasterize_polygons([z["polygon"] for z in zs], grid_meta)

    def resolve_label(self, name: str) -> Optional[Tuple[float, float, Optional[float]]]:
        lb = self.get_label(name)
        if lb is None:
            return None
        return lb["x"], lb["y"], lb.get("yaw")

    def patrol_points(self, zone_key: str, spacing: float = 1.0, inset: float = 0.3) -> Optional[List[List[float]]]:
        z = self.get_zone(zone_key)
        if z is None:
            return None
        return polygon_perimeter_points(z["polygon"], spacing, inset)


def no_go_mask(zones: Iterable[Dict[str, Any]], grid_meta: Any, now: Any = None) -> np.ndarray:
    """Module-level variant over a plain zone list (kind != no_go ignored)."""
    polys = [z["polygon"] for z in zones if z.get("kind") == "no_go"
             and (now is None or hours_active(z.get("active_hours"), now))]
    return rasterize_polygons(polys, grid_meta)
