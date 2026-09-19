"""Read-only Jetson power / load info for `GET /system/power`.

Reads sysfs (visible in the container) and, when mounted read-only, the
host's nvpmodel files:
  -v /var/lib/nvpmodel:/host/nvpmodel:ro -v /etc/nvpmodel.conf:/host/nvpmodel.conf:ro

Switching the power mode changes the native robot system (nvpmodel, sudo)
and is deliberately not offered here: `switch_supported` stays false.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Optional

NVP_STATUS = os.environ.get("NVP_STATUS", "/host/nvpmodel/status")
NVP_CONF = os.environ.get("NVP_CONF", "/host/nvpmodel.conf")
GPU_LOAD_GLOBS = ("/sys/devices/platform/*.gpu/load", "/sys/devices/platform/gpu.0/load",
                  "/sys/devices/gpu.0/load", "/sys/devices/platform/bus@0/*.gpu/load")


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _count_cpus(spec: Optional[str]) -> int:
    """'0-3,6' -> 5"""
    n = 0
    for part in (spec or "").split(","):
        if "-" in part:
            a, b = part.split("-")
            n += int(b) - int(a) + 1
        elif part.strip():
            n += 1
    return n


def _mode() -> Optional[str]:
    status = _read(NVP_STATUS)            # e.g. "pmode:0002 fmode:..."
    conf = _read(NVP_CONF)
    if not status or not conf:
        return None
    m = re.search(r"pmode:(\d+)", status)
    if not m:
        return None
    names = dict(re.findall(r"POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", conf))
    return names.get(str(int(m.group(1))))


def _gpu_load_pct() -> Optional[float]:
    for pattern in GPU_LOAD_GLOBS:
        for path in glob.glob(pattern):
            v = _read(path)
            if v and v.isdigit():
                return round(int(v) / 10.0, 1)   # sysfs reports per-mille
    return None


def power() -> dict:
    freqs = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"),
                       key=lambda p: int(re.search(r"cpu(\d+)", p).group(1))):
        v = _read(path)
        freqs.append(int(v) // 1000 if v and v.isdigit() else None)
    return {
        "mode": _mode(),
        "cpu_online": _count_cpus(_read("/sys/devices/system/cpu/online")),
        "cpu_total": _count_cpus(_read("/sys/devices/system/cpu/present")),
        "cpu_freq_mhz": freqs,
        "gpu_load_pct": _gpu_load_pct(),
        "switch_supported": False,
    }
