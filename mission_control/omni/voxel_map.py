"""Coloured voxel map (world frame) with OVX1 delta export (CONTRACTS.md 4 "C").

Storage
-------
Voxel keys pack (ix, iy, iz) = floor(p / res) into one int64 (21 bits per
axis, offset 2**20 -> +-52 km at 5 cm). A sorted key array + ``np.searchsorted``
maps keys to slots in flat numpy attribute arrays (no Python dict, no
per-point loops). Each ``integrate`` call reduces its points per voxel with
``np.unique`` + ``np.bincount`` and updates the touched slots in one shot.

Per voxel: mean xyz, weighted running mean rgb, running mean temperature
(NaN-aware), near-person flag, n_obs, last_seen t, last_changed version.
Running means use a capped effective weight (``max_weight``) so that old
observations fade and the map follows changes in the scene.

A voxel's ``last_changed`` version is only bumped when what a client sees
changes (new voxel, colour moved by >= ``color_eps``, flags changed, mean
position moved by > ``pos_eps``), so deltas stay small for static scenes.

OVX1 binary format (little-endian)
----------------------------------
    offset 0   4 bytes  magic b"OVX1"
    offset 4   uint32   version  (map version at encode time)
    offset 8   uint32   count
    offset 12  float32  res      (voxel edge, m)
    offset 16  count x 16 bytes:
               float32 x, float32 y, float32 z   (voxel mean position, world, m)
               uint8 r, uint8 g, uint8 b
               uint8 flags  bit0 = has temperature and temp > 28 degC
                            bit1 = near a person
Evicted voxels are not signalled in deltas (clients keep them until the next
snapshot).
"""
from __future__ import annotations

import struct
import time
from typing import Optional

import numpy as np

OVX1_MAGIC = b"OVX1"
OVX1_HEADER = struct.Struct("<4sIIf")
OVX1_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1"), ("flags", "u1")])
assert OVX1_DTYPE.itemsize == 16

FLAG_HOT = 1
FLAG_PERSON = 2
HOT_TEMP_C = 28.0

_KBITS = 21
_KOFF = 1 << (_KBITS - 1)
_KMASK = (1 << _KBITS) - 1


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    """(N,3) int voxel indices -> (N,) int64 keys."""
    a = ijk.astype(np.int64) + _KOFF
    return (a[:, 0] << (2 * _KBITS)) | (a[:, 1] << _KBITS) | a[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64)
    out = np.empty((k.shape[0], 3), dtype=np.int64)
    out[:, 0] = ((k >> (2 * _KBITS)) & _KMASK) - _KOFF
    out[:, 1] = ((k >> _KBITS) & _KMASK) - _KOFF
    out[:, 2] = (k & _KMASK) - _KOFF
    return out


def encode_ovx1(version: int, res: float, xyz: np.ndarray, rgb: np.ndarray,
                flags: np.ndarray) -> bytes:
    n = int(xyz.shape[0])
    rec = np.empty(n, dtype=OVX1_DTYPE)
    rec["x"] = xyz[:, 0]
    rec["y"] = xyz[:, 1]
    rec["z"] = xyz[:, 2]
    rec["r"] = rgb[:, 0]
    rec["g"] = rgb[:, 1]
    rec["b"] = rgb[:, 2]
    rec["flags"] = flags
    return OVX1_HEADER.pack(OVX1_MAGIC, int(version) & 0xFFFFFFFF, n, float(res)) + rec.tobytes()


def decode_ovx1(buf: bytes) -> dict:
    """Decode an OVX1 message -> dict(version, count, res, xyz, rgb, flags)."""
    if len(buf) < OVX1_HEADER.size:
        raise ValueError("OVX1: short buffer")
    magic, version, count, res = OVX1_HEADER.unpack_from(buf, 0)
    if magic != OVX1_MAGIC:
        raise ValueError("OVX1: bad magic %r" % (magic,))
    if len(buf) != OVX1_HEADER.size + count * OVX1_DTYPE.itemsize:
        raise ValueError("OVX1: size mismatch")
    rec = np.frombuffer(buf, dtype=OVX1_DTYPE, count=count, offset=OVX1_HEADER.size)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], axis=1)
    return {"version": version, "count": count, "res": res,
            "xyz": xyz, "rgb": rgb, "flags": rec["flags"].copy()}


class VoxelMap:
    """Sparse coloured voxel map in the world frame."""

    def __init__(self, res: float = 0.05, max_voxels: int = 2_000_000,
                 max_weight: float = 50.0, color_eps: int = 4,
                 pos_eps: Optional[float] = None, evict_frac: float = 0.9,
                 initial_capacity: int = 65536) -> None:
        self.res = float(res)
        self.max_voxels = int(max_voxels)
        self.max_weight = float(max_weight)
        self.color_eps = int(color_eps)
        self.pos_eps = float(pos_eps) if pos_eps is not None else 0.25 * self.res
        self.evict_frac = float(evict_frac)
        self.version = 0
        self._n_integrations = 0
        self._n_evicted = 0
        self._last_ms = 0.0
        # Sorted key index.
        self._keys = np.empty(0, dtype=np.int64)
        self._kslot = np.empty(0, dtype=np.int64)
        # Slot storage.
        self._cap = 0
        self._top = 0  # slots [0, _top) were ever allocated
        self._free = np.empty(0, dtype=np.int64)
        self._alloc(max(16, int(initial_capacity)))

    # -- storage ------------------------------------------------------------
    def _alloc(self, cap: int) -> None:
        def grow(a, shape, dtype, fill):
            # np.zeros is calloc-backed (lazy pages): much cheaper than np.full.
            assert fill == 0
            b = np.zeros(shape, dtype=dtype)
            if a is not None:
                b[:a.shape[0]] = a
            return b

        g = lambda name: getattr(self, name, None)  # noqa: E731
        self.alive = grow(g("alive"), (cap,), bool, False)
        self.key = grow(g("key"), (cap,), np.int64, 0)
        self.xyz = grow(g("xyz"), (cap, 3), np.float32, 0.0)
        self.n_obs = grow(g("n_obs"), (cap,), np.uint32, 0)
        self.rgb = grow(g("rgb"), (cap, 3), np.float32, 0.0)
        self.rgb_w = grow(g("rgb_w"), (cap,), np.float32, 0.0)
        self.temp = grow(g("temp"), (cap,), np.float32, 0.0)
        self.temp_n = grow(g("temp_n"), (cap,), np.float32, 0.0)
        self.person = grow(g("person"), (cap,), bool, False)
        self.last_seen = grow(g("last_seen"), (cap,), np.float64, 0.0)
        self.changed = grow(g("changed"), (cap,), np.uint32, 0)
        # Last values emitted to clients (for change detection).
        self.em_rgb = grow(g("em_rgb"), (cap, 3), np.int16, 0)
        self.em_flags = grow(g("em_flags"), (cap,), np.uint8, 0)
        self.em_xyz = grow(g("em_xyz"), (cap, 3), np.float32, 0.0)
        self._cap = cap

    def _new_slots(self, k: int) -> np.ndarray:
        nf = min(k, self._free.shape[0])
        slots = self._free[:nf]
        self._free = self._free[nf:]
        rest = k - nf
        if rest:
            if self._top + rest > self._cap:
                cap = self._cap
                while cap < self._top + rest:
                    cap *= 2
                self._alloc(cap)
            slots = np.concatenate([slots, np.arange(self._top, self._top + rest, dtype=np.int64)])
            self._top += rest
        return slots

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    # -- integration --------------------------------------------------------
    def integrate(self, pts_world: np.ndarray, rgb: np.ndarray,
                  temp: Optional[np.ndarray] = None, t: Optional[float] = None,
                  weights: Optional[np.ndarray] = None,
                  near_person_mask: Optional[np.ndarray] = None,
                  origin: Optional[np.ndarray] = None) -> int:
        """Fuse coloured world points. Returns the new map version.

        weights: per-point colour weight; if None and ``origin`` (robot
        position, world) is given, w = 1 / (1 + range / 2 m), else 1.
        """
        t0 = time.perf_counter()
        if t is None:
            t = time.time()
        p = np.asarray(pts_world, dtype=np.float32).reshape(-1, 3)
        self.version += 1
        self._n_integrations += 1
        n = p.shape[0]
        if n == 0:
            return self.version
        c = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
        if weights is None:
            if origin is not None:
                r = np.linalg.norm(p - np.asarray(origin, dtype=np.float32).reshape(1, 3), axis=1)
                w = 1.0 / (1.0 + 0.5 * r)
            else:
                w = np.ones(n, dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)

        keys = pack_keys(np.floor(p / self.res))
        ukeys, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        m = ukeys.shape[0]
        cnt = np.bincount(inv, minlength=m).astype(np.float32)
        bsum = np.stack([np.bincount(inv, p[:, i], m) for i in range(3)], axis=1)
        bw = np.bincount(inv, w, m)
        bc = np.stack([np.bincount(inv, c[:, i] * w, m) for i in range(3)], axis=1)
        if temp is not None:
            tp = np.asarray(temp, dtype=np.float32).reshape(-1)
            tv = np.isfinite(tp)
            bt = np.bincount(inv, np.where(tv, tp, 0.0), m)
            btn = np.bincount(inv, tv.astype(np.float32), m)
        if near_person_mask is not None:
            bp = np.bincount(inv, np.asarray(near_person_mask, dtype=np.float32).reshape(-1), m) > 0
        else:
            bp = None

        # Key -> slot lookup / insert.
        pos = np.searchsorted(self._keys, ukeys)
        if self._keys.shape[0]:
            found = self._keys[np.minimum(pos, self._keys.shape[0] - 1)] == ukeys
        else:
            found = np.zeros(m, dtype=bool)
        slots = np.empty(m, dtype=np.int64)
        slots[found] = self._kslot[pos[found]]
        new = ~found
        n_new = int(new.sum())
        if n_new:
            ns = self._new_slots(n_new)
            slots[new] = ns
            self._keys = np.insert(self._keys, pos[new], ukeys[new])
            self._kslot = np.insert(self._kslot, pos[new], ns)
            self.alive[ns] = True
            self.key[ns] = ukeys[new]
            self.xyz[ns] = 0.0
            self.n_obs[ns] = 0
            self.rgb[ns] = 0.0
            self.rgb_w[ns] = 0.0
            self.temp[ns] = 0.0
            self.temp_n[ns] = 0.0
            self.person[ns] = False

        cap = self.max_weight
        # Position: running mean with capped effective count.
        n_old = np.minimum(self.n_obs[slots].astype(np.float32), cap)
        self.xyz[slots] = (self.xyz[slots] * n_old[:, None] + bsum) / (n_old + cnt)[:, None]
        self.n_obs[slots] += cnt.astype(np.uint32)
        # Colour: weighted running mean.
        w_old = np.minimum(self.rgb_w[slots], cap)
        w_new = w_old + bw
        okw = w_new > 0
        sw = slots[okw]
        self.rgb[sw] = (self.rgb[sw] * w_old[okw, None] + bc[okw]) / w_new[okw, None]
        self.rgb_w[slots] = w_new
        # Temperature: NaN-aware running mean.
        if temp is not None:
            st = slots[btn > 0]
            tn_old = np.minimum(self.temp_n[st], cap)
            self.temp[st] = (self.temp[st] * tn_old + bt[btn > 0]) / (tn_old + btn[btn > 0])
            self.temp_n[st] = tn_old + btn[btn > 0]
        if bp is not None:
            self.person[slots] = bp
        self.last_seen[slots] = t

        # Change detection against the last emitted state.
        rgb8 = self._rgb8(slots).astype(np.int16)
        fl = self._flags(slots)
        ch = (new
              | (np.abs(rgb8 - self.em_rgb[slots]).max(axis=1) >= self.color_eps)
              | (fl != self.em_flags[slots])
              | (np.abs(self.xyz[slots] - self.em_xyz[slots]).max(axis=1) > self.pos_eps))
        sc = slots[ch]
        self.changed[sc] = self.version
        self.em_rgb[sc] = rgb8[ch]
        self.em_flags[sc] = fl[ch]
        self.em_xyz[sc] = self.xyz[sc]

        if len(self) > self.max_voxels:
            self._evict(int(self.max_voxels * self.evict_frac))
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        return self.version

    def _evict(self, target: int) -> None:
        """Drop least-recently-seen voxels until ``target`` remain."""
        target = max(0, min(target, self.max_voxels))
        k = len(self) - target
        if k <= 0:
            return
        live = self._kslot
        ls = self.last_seen[live]
        drop_i = np.argpartition(ls, k - 1)[:k] if k < live.shape[0] else np.arange(live.shape[0])
        drop_slots = live[drop_i]
        keep = np.ones(live.shape[0], dtype=bool)
        keep[drop_i] = False
        self._keys = self._keys[keep]
        self._kslot = self._kslot[keep]
        self.alive[drop_slots] = False
        self._free = np.concatenate([self._free, drop_slots])
        self._n_evicted += k

    # -- views --------------------------------------------------------------
    def _rgb8(self, slots: np.ndarray) -> np.ndarray:
        x = self.rgb[slots]
        x += 0.5
        np.clip(x, 0, 255, out=x)
        return x.astype(np.uint8)

    def _temp(self, slots: np.ndarray) -> np.ndarray:
        tn = self.temp_n[slots]
        return np.where(tn > 0, self.temp[slots], np.nan).astype(np.float32)

    def _flags(self, slots: np.ndarray) -> np.ndarray:
        hot = (self.temp_n[slots] > 0) & (self.temp[slots] > HOT_TEMP_C)
        return (hot.astype(np.uint8) * FLAG_HOT) | (self.person[slots].astype(np.uint8) * FLAG_PERSON)

    def _encode(self, slots: np.ndarray) -> bytes:
        return encode_ovx1(self.version, self.res, self.xyz[slots], self._rgb8(slots),
                           self._flags(slots))

    def delta_since(self, version: int) -> bytes:
        """OVX1 of voxels whose visible state changed after ``version``."""
        top = self._top
        slots = np.nonzero(self.alive[:top] & (self.changed[:top] > int(version)))[0]
        return self._encode(slots)

    def snapshot(self) -> bytes:
        """OVX1 of every voxel in the map."""
        return self._encode(np.sort(self._kslot))

    def arrays(self) -> dict:
        """Current voxels as numpy arrays (xyz, rgb uint8, temp NaN, flags, n_obs, last_seen)."""
        s = np.sort(self._kslot)
        return {"xyz": self.xyz[s].copy(), "rgb": self._rgb8(s), "temp": self._temp(s),
                "flags": self._flags(s), "n_obs": self.n_obs[s].copy(),
                "last_seen": self.last_seen[s].copy()}

    def export_ply(self, path: str, with_temp: bool = True) -> int:
        """Binary little-endian PLY (x y z red green blue [temperature]). Returns count."""
        a = self.arrays()
        n = a["xyz"].shape[0]
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                  ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        if with_temp:
            fields.append(("temperature", "<f4"))
        rec = np.empty(n, dtype=np.dtype(fields))
        rec["x"], rec["y"], rec["z"] = a["xyz"][:, 0], a["xyz"][:, 1], a["xyz"][:, 2]
        rec["red"], rec["green"], rec["blue"] = a["rgb"][:, 0], a["rgb"][:, 1], a["rgb"][:, 2]
        props = ["property float x", "property float y", "property float z",
                 "property uchar red", "property uchar green", "property uchar blue"]
        if with_temp:
            rec["temperature"] = a["temp"]
            props.append("property float temperature")
        header = "\n".join(["ply", "format binary_little_endian 1.0",
                            "comment omni voxel_map res=%g" % self.res,
                            "element vertex %d" % n] + props + ["end_header"]) + "\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(rec.tobytes())
        return n

    def stats(self) -> dict:
        return {
            "voxels": len(self),
            "capacity": self._cap,
            "max_voxels": self.max_voxels,
            "version": self.version,
            "res": self.res,
            "integrations": self._n_integrations,
            "evicted": self._n_evicted,
            "last_integrate_ms": round(self._last_ms, 3),
            "snapshot_bytes": OVX1_HEADER.size + 16 * len(self),
        }
