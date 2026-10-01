"""Tests for omni.voxel_map (OVX1, deltas, averaging, temperature, eviction, PLY, perf)."""
from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from omni.voxel_map import (FLAG_HOT, FLAG_PERSON, VoxelMap, decode_ovx1,  # noqa: E402
                            pack_keys, unpack_keys)


def _grid(n, res=0.05, z=0.0, x0=0.0):
    """n voxel centres along x."""
    x = x0 + (np.arange(n) + 0.5) * res
    return np.stack([x, np.zeros(n), np.full(n, z + 0.5 * res)], axis=1)


def test_key_pack_roundtrip():
    ijk = np.array([[0, 0, 0], [-1, 2, -3], [100000, -100000, 5], [-(1 << 20), (1 << 20) - 1, 0]])
    assert np.array_equal(unpack_keys(pack_keys(ijk)), ijk)


def test_ovx1_roundtrip_header_and_values():
    vm = VoxelMap(res=0.05)
    pts = _grid(10)
    rgb = np.tile(np.array([[10, 200, 30]], np.uint8), (10, 1))
    temp = np.full(10, np.nan, np.float32)
    temp[3] = 36.0
    near = np.zeros(10, bool)
    near[5] = True
    v = vm.integrate(pts, rgb, temp=temp, t=1.0, near_person_mask=near)
    buf = vm.snapshot()
    assert buf[:4] == b"OVX1"
    magic, ver, count, res = struct.unpack_from("<4sIIf", buf, 0)
    assert ver == v == 1 and count == 10 and abs(res - 0.05) < 1e-7
    assert len(buf) == 16 + 16 * 10
    d = decode_ovx1(buf)
    order = np.argsort(d["xyz"][:, 0])
    assert np.allclose(d["xyz"][order], pts, atol=1e-6)
    assert (d["rgb"] == [10, 200, 30]).all()
    fl = d["flags"][order]
    assert fl[3] == FLAG_HOT and fl[5] == FLAG_PERSON
    assert (np.delete(fl, [3, 5]) == 0).all()
    # Raw record layout: x,y,z float32 then r,g,b,flags uint8.
    x, y, z, r, g, b, f = struct.unpack_from("<fffBBBB", buf, 16)
    assert (r, g, b) == (10, 200, 30)


def test_delta_since_only_changed():
    vm = VoxelMap(res=0.05)
    pts = _grid(20)
    rgb = np.full((20, 3), 100, np.uint8)
    v1 = vm.integrate(pts, rgb, t=1.0)
    assert decode_ovx1(vm.delta_since(0))["count"] == 20
    assert decode_ovx1(vm.delta_since(v1))["count"] == 0
    # Re-observe the same voxels with the same colour: nothing visible changed.
    v2 = vm.integrate(pts, rgb, t=2.0)
    assert decode_ovx1(vm.delta_since(v1))["count"] == 0
    # Change colour of 3 voxels strongly + add 2 new voxels.
    rgb2 = rgb.copy()
    rgb2[:3] = 255
    pts_new = _grid(2, x0=5.0)
    v3 = vm.integrate(np.vstack([pts, pts_new]),
                      np.vstack([rgb2, np.full((2, 3), 7, np.uint8)]), t=3.0)
    d = decode_ovx1(vm.delta_since(v2))
    assert d["count"] == 5 and d["version"] == v3
    assert decode_ovx1(vm.delta_since(v3))["count"] == 0
    assert decode_ovx1(vm.snapshot())["count"] == 22


def test_color_weighted_average():
    vm = VoxelMap(res=0.1)
    p = np.array([[0.05, 0.05, 0.05]] * 3)
    rgb = np.array([[0, 0, 0], [90, 90, 90], [255, 0, 0]], np.uint8)
    vm.integrate(p, rgb, weights=np.array([1.0, 2.0, 0.0]), t=0.0)
    a = vm.arrays()
    assert a["xyz"].shape == (1, 3)
    assert tuple(a["rgb"][0]) == (60, 60, 60)
    assert a["n_obs"][0] == 3
    # Second batch, unit weight: (60*3 + 150*1) / 4 = 82.5 -> 83 (rounded)
    vm.integrate(p[:1], np.array([[150, 150, 150]], np.uint8), t=1.0)
    assert tuple(vm.arrays()["rgb"][0]) == (83, 83, 83)
    # Range weights from origin: near point dominates.
    vm2 = VoxelMap(res=0.5)
    q = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    vm2.integrate(q, np.array([[200, 0, 0], [0, 0, 200]], np.uint8), origin=(0.0, 0.0, 0.0))
    c = vm2.arrays()["rgb"][0]
    assert c[0] > c[2]


def test_temperature_nan_handling():
    vm = VoxelMap(res=0.1)
    p = np.array([[0.05, 0.05, 0.05]] * 4 + [[1.05, 0.05, 0.05]])
    rgb = np.zeros((5, 3), np.uint8)
    temp = np.array([30.0, np.nan, 34.0, np.nan, np.nan], np.float32)
    vm.integrate(p, rgb, temp=temp, t=0.0)
    a = vm.arrays()
    o = np.argsort(a["xyz"][:, 0])
    assert abs(a["temp"][o[0]] - 32.0) < 1e-5
    assert np.isnan(a["temp"][o[1]])
    assert a["flags"][o[0]] & FLAG_HOT and not a["flags"][o[1]] & FLAG_HOT
    # Integrating without temp keeps the old value; a cold reading lowers it.
    vm.integrate(p[:1], rgb[:1], t=1.0)
    assert abs(vm.arrays()["temp"][o[0]] - 32.0) < 1e-5
    vm.integrate(p[:1], rgb[:1], temp=np.array([20.0]), t=2.0)
    assert abs(vm.arrays()["temp"][o[0]] - (32.0 * 2 + 20.0) / 3) < 1e-4


def test_eviction_drops_least_recently_seen():
    vm = VoxelMap(res=0.05, max_voxels=100, evict_frac=1.0, initial_capacity=16)
    old = _grid(60, x0=0.0)
    mid = _grid(30, x0=10.0)
    new = _grid(30, x0=20.0)
    rgb = np.zeros((60, 3), np.uint8)
    vm.integrate(old, rgb, t=1.0)
    vm.integrate(mid, rgb[:30], t=2.0)
    vm.integrate(old[:20], rgb[:20], t=3.0)  # refresh first 20 old voxels
    vm.integrate(new, rgb[:30], t=4.0)
    assert len(vm) == 100
    a = vm.arrays()
    xs, ls = a["xyz"][:, 0], a["last_seen"]
    # 120 voxels -> 20 evicted, all from the 40 stale ones seen only at t=1.
    assert np.sum(ls == 1.0) == 20 and np.sum(ls == 3.0) == 20
    assert np.sum(ls == 2.0) == 30 and np.sum(ls == 4.0) == 30 and np.sum(xs > 19) == 30
    assert vm.stats()["evicted"] == 20
    # Slots get reused and lookups stay consistent.
    v = vm.integrate(_grid(5, x0=30.0), rgb[:5], t=5.0)
    assert len(vm) == 100
    d = decode_ovx1(vm.delta_since(v - 1))
    assert d["count"] == 5 and (d["xyz"][:, 0] > 29).all()
    assert decode_ovx1(vm.snapshot())["count"] == 100


def test_export_ply(tmp_path):
    vm = VoxelMap(res=0.05)
    vm.integrate(_grid(7), np.full((7, 3), 9, np.uint8), temp=np.full(7, 25.0), t=0.0)
    path = str(tmp_path / "map.ply")
    assert vm.export_ply(path) == 7
    with open(path, "rb") as f:
        data = f.read()
    hdr_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:hdr_end].decode("ascii").splitlines()
    assert header[0] == "ply" and header[1] == "format binary_little_endian 1.0"
    assert "element vertex 7" in header
    assert "property float temperature" in header
    body = data[hdr_end:]
    assert len(body) == 7 * (12 + 3 + 4)
    x, y, z, r, g, b, tc = struct.unpack_from("<fffBBBf", body, 0)
    assert (r, g, b) == (9, 9, 9) and abs(tc - 25.0) < 1e-5
    # Without temperature.
    vm.export_ply(path, with_temp=False)
    with open(path, "rb") as f:
        data = f.read()
    assert b"temperature" not in data[:data.index(b"end_header")]
    assert len(data) - data.index(b"end_header\n") - 11 == 7 * 15


def test_perf_integrate_30k():
    rng = np.random.default_rng(1)
    vm = VoxelMap(res=0.05)
    times = []
    for i in range(20):
        pts = rng.uniform(-10, 10, (30000, 3)).astype(np.float32) * np.array([1, 1, 0.1], np.float32)
        rgb = rng.integers(0, 255, (30000, 3), dtype=np.uint8)
        temp = np.where(rng.random(30000) < 0.1, 30.0, np.nan).astype(np.float32)
        t0 = time.perf_counter()
        vm.integrate(pts, rgb, temp=temp, t=float(i))
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    d = vm.delta_since(vm.version - 1)
    t_delta = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = vm.snapshot()
    t_snap = time.perf_counter() - t0
    med = float(np.median(times))
    print("\nVoxelMap.integrate 30k pts: median %.2f ms (map %d voxels); delta %.2f ms (%d B); "
          "snapshot %.2f ms (%d B)" % (med * 1e3, len(vm), t_delta * 1e3, len(d), t_snap * 1e3, len(s)))
    assert med < 0.2
