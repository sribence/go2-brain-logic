"""Global multi-camera person tracker in the robot base frame (Agent B).

Per tick:
  1. ego-motion compensation of all tracks (robot moved by ego_delta since last update),
  2. constant-velocity Kalman predict (state x, y, vx, vy),
  3. fusion of same-tick detections from different cameras / modalities into
     clusters (one person seen by rgb_front + rgb_left + thermal -> one measurement),
  4. Hungarian (scipy if present, else built-in) association cluster <-> track,
     Euclidean gate (default 1.0 m),
  5. birth after ``min_hits`` hits (stable incremental gid), death after ``max_age_s``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .omni_types import PersonTrack
except ImportError:  # loaded as a top-level module
    from omni_types import PersonTrack  # type: ignore

RANGE_PRIORITY = ("lidar", "tof", "ground_plane", "thermal_size")
# Measurement std-dev (m) at 1 m range, grows linearly with range for the weak sources.
RANGE_SIGMA = {"lidar": (0.10, 0.01), "tof": (0.08, 0.02), "ground_plane": (0.15, 0.06),
               "thermal_size": (0.25, 0.10)}


@dataclass
class Det3D:
    x: float
    y: float
    z: float
    conf: float
    cam_id: str
    modality: str = "rgb"
    range_src: str = "ground_plane"
    height_m: float = 0.0  # 0 = unknown


def _as_det(d) -> Det3D:
    if isinstance(d, Det3D):
        return d
    if isinstance(d, dict):
        return Det3D(**{k: d[k] for k in Det3D.__dataclass_fields__ if k in d})
    return Det3D(*d)


def meas_sigma(d: Det3D) -> float:
    a, b = RANGE_SIGMA.get(d.range_src, (0.3, 0.1))
    return a + b * math.hypot(d.x, d.y)


def _src_rank(src: str) -> int:
    return RANGE_PRIORITY.index(src) if src in RANGE_PRIORITY else len(RANGE_PRIORITY)


# ------------------------------------------------------------------ assignment

def linear_assignment(cost: np.ndarray, max_cost: float = float("inf")) -> List[Tuple[int, int]]:
    """Minimum-cost assignment; pairs with cost > max_cost are dropped."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    big = 1e6
    c = np.where(np.isfinite(cost) & (cost <= max_cost), cost, big)
    try:
        from scipy.optimize import linear_sum_assignment  # optional
        rows, cols = linear_sum_assignment(c)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs = _hungarian(c)
    return [(r, k) for r, k in pairs if c[r, k] < big and cost[r, k] <= max_cost]


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """O(n^3) Hungarian algorithm (Jonker-Volgenant style potentials), rectangular OK."""
    n_r, n_c = cost.shape
    transposed = n_r > n_c
    a = cost.T if transposed else cost
    n, m = a.shape  # n <= m
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, m + 1):
        if p[j]:
            pairs.append((j - 1, p[j] - 1) if transposed else (p[j] - 1, j - 1))
    pairs.sort()
    return pairs


# ------------------------------------------------------------------ Kalman track

class _Track:
    def __init__(self, tid: int, d: "_Cluster", t: float, accel_std: float):
        self.tid = tid
        self.gid = None  # type: Optional[int]
        self.x = np.array([d.x, d.y, 0.0, 0.0])
        s2 = d.sigma ** 2
        self.P = np.diag([s2, s2, 1.0, 1.0])
        self.z = d.z
        self.height_m = d.height_m
        self.accel_std = accel_std
        self.hits = 1
        self.birth_t = t
        self.last_t = t
        self.last_seen_t = t
        self.conf = d.conf
        self.cams = list(d.cams)
        self.range_src = d.range_src
        self.mod_seen = {m: t for m in d.modalities}  # type: Dict[str, float]

    def ego(self, dx: float, dy: float, dyaw: float) -> None:
        c, s = math.cos(-dyaw), math.sin(-dyaw)
        R = np.array([[c, -s], [s, c]])
        pos = R @ (self.x[:2] - np.array([dx, dy]))
        vel = R @ self.x[2:]
        self.x = np.concatenate([pos, vel])
        R4 = np.zeros((4, 4))
        R4[:2, :2] = R
        R4[2:, 2:] = R
        self.P = R4 @ self.P @ R4.T

    def predict(self, t: float) -> None:
        dt = max(0.0, t - self.last_t)
        self.last_t = t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel_std ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q1 = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, d: "_Cluster", t: float) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        Rm = np.eye(2) * d.sigma ** 2
        y = np.array([d.x, d.y]) - H @ self.x
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.z = 0.7 * self.z + 0.3 * d.z
        if d.height_m > 0:
            self.height_m = d.height_m if self.height_m <= 0 else 0.7 * self.height_m + 0.3 * d.height_m
        self.hits += 1
        self.last_seen_t = t
        self.conf = 0.6 * self.conf + 0.4 * d.conf
        self.cams = list(d.cams)
        self.range_src = d.range_src
        for m in d.modalities:
            self.mod_seen[m] = t

    def to_person(self, t: float, mod_hold_s: float) -> PersonTrack:
        mods = sorted(m for m, ts in self.mod_seen.items() if t - ts <= mod_hold_s)
        seen_now = abs(t - self.last_seen_t) < 1e-6
        return PersonTrack(
            gid=int(self.gid if self.gid is not None else -1),
            x=float(self.x[0]), y=float(self.x[1]), z=float(self.z),
            vx=float(self.x[2]), vy=float(self.x[3]),
            conf=float(self.conf if seen_now else self.conf * 0.8),
            cams=list(self.cams) if seen_now else [],
            modality=mods, range_src=self.range_src,
            age_s=float(t - self.birth_t), last_seen_t=float(self.last_seen_t),
            height_m=float(self.height_m))


class _Cluster:
    """Fused same-tick measurement (inverse-variance weighted)."""

    def __init__(self, d: Det3D):
        self.dets = [d]

    def can_add(self, d: Det3D, fuse_dist: float) -> bool:
        if any(o.cam_id == d.cam_id for o in self.dets):
            return False
        return math.hypot(d.x - self.x, d.y - self.y) <= fuse_dist

    def add(self, d: Det3D) -> None:
        self.dets.append(d)

    def _w(self) -> np.ndarray:
        return np.array([1.0 / meas_sigma(d) ** 2 for d in self.dets])

    @property
    def x(self) -> float:
        w = self._w()
        return float(np.dot(w, [d.x for d in self.dets]) / w.sum())

    @property
    def y(self) -> float:
        w = self._w()
        return float(np.dot(w, [d.y for d in self.dets]) / w.sum())

    @property
    def z(self) -> float:
        w = self._w()
        return float(np.dot(w, [d.z for d in self.dets]) / w.sum())

    @property
    def height_m(self) -> float:
        hs = [d.height_m for d in self.dets if d.height_m > 0]
        return float(max(hs)) if hs else 0.0

    @property
    def sigma(self) -> float:
        return float(1.0 / math.sqrt(self._w().sum()))

    @property
    def conf(self) -> float:
        # Independent evidence: 1 - prod(1 - c)
        return float(1.0 - np.prod([1.0 - min(0.999, max(0.0, d.conf)) for d in self.dets]))

    @property
    def cams(self) -> List[str]:
        return sorted({d.cam_id for d in self.dets})

    @property
    def modalities(self) -> List[str]:
        return sorted({d.modality for d in self.dets})

    @property
    def range_src(self) -> str:
        return min((d.range_src for d in self.dets), key=_src_rank)


def fuse_detections(dets: Sequence[Det3D], fuse_dist: float = 0.6) -> List[_Cluster]:
    """Greedy fusion, strongest range source / confidence first."""
    order = sorted(dets, key=lambda d: (_src_rank(d.range_src), -d.conf))
    clusters = []  # type: List[_Cluster]
    for d in order:
        best, best_dist = None, None
        for c in clusters:
            if c.can_add(d, fuse_dist):
                dist = math.hypot(d.x - c.x, d.y - c.y)
                if best is None or dist < best_dist:
                    best, best_dist = c, dist
        if best is None:
            clusters.append(_Cluster(d))
        else:
            best.add(d)
    return clusters


# ------------------------------------------------------------------ tracker

class MultiCamTracker:
    def __init__(self, gate_m: float = 1.0, fuse_dist_m: float = 0.6, min_hits: int = 2,
                 max_age_s: float = 1.5, tentative_max_age_s: float = 0.5, accel_std: float = 2.0,
                 modality_hold_s: float = 1.0, min_conf: float = 0.0):
        self.gate_m = float(gate_m)
        self.fuse_dist_m = float(fuse_dist_m)
        self.min_hits = int(min_hits)
        self.max_age_s = float(max_age_s)
        self.tentative_max_age_s = float(tentative_max_age_s)
        self.accel_std = float(accel_std)
        self.modality_hold_s = float(modality_hold_s)
        self.min_conf = float(min_conf)
        self._tracks = []  # type: List[_Track]
        self._next_tid = 1
        self._next_gid = 1
        self.last_t = None  # type: Optional[float]

    def reset(self) -> None:
        self._tracks = []
        self.last_t = None

    def update(self, dets_3d, t: float, ego_delta=None) -> List[PersonTrack]:
        dets = [_as_det(d) for d in (dets_3d or []) if _as_det(d).conf >= self.min_conf]
        if ego_delta is not None:
            dx, dy, dyaw = [float(v) for v in ego_delta]
            if dx or dy or dyaw:
                for tr in self._tracks:
                    tr.ego(dx, dy, dyaw)
        for tr in self._tracks:
            tr.predict(t)

        clusters = fuse_detections(dets, self.fuse_dist_m)
        # Confirmed tracks get first pick, then tentative ones.
        unmatched_c = list(range(len(clusters)))
        for confirmed in (True, False):
            idx = [i for i, tr in enumerate(self._tracks) if (tr.gid is not None) == confirmed]
            if not idx or not unmatched_c:
                continue
            cost = np.zeros((len(idx), len(unmatched_c)))
            for a, ti in enumerate(idx):
                tp = self._tracks[ti].x[:2]
                for b, ci in enumerate(unmatched_c):
                    c = clusters[ci]
                    cost[a, b] = math.hypot(c.x - tp[0], c.y - tp[1])
            used = set()
            for a, b in linear_assignment(cost, self.gate_m):
                self._tracks[idx[a]].update(clusters[unmatched_c[b]], t)
                used.add(unmatched_c[b])
            unmatched_c = [ci for ci in unmatched_c if ci not in used]

        for ci in unmatched_c:
            self._tracks.append(_Track(self._next_tid, clusters[ci], t, self.accel_std))
            self._next_tid += 1

        alive = []
        for tr in self._tracks:
            if tr.gid is None and tr.hits >= self.min_hits:
                tr.gid = self._next_gid
                self._next_gid += 1
            lim = self.max_age_s if tr.gid is not None else self.tentative_max_age_s
            if t - tr.last_seen_t <= lim:
                alive.append(tr)
        self._tracks = alive
        self.last_t = t
        out = [tr.to_person(t, self.modality_hold_s) for tr in self._tracks if tr.gid is not None]
        out.sort(key=lambda p: p.gid)
        return out
