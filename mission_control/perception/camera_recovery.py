"""Self-healing for the RealSense USB camera.

Failure modes seen on the Go2 Jetson (2026-09-25):
  * "blind": the kernel lists the camera in sysfs, but this long-running
    process' librealsense reports "No device connected". Only a fresh process
    sees the camera again.
  * stall / I/O error (VIDIOC_S_FMT errno 5): librealsense hardware_reset()
    brings the camera back.
  * dead on the bus: the kernel logs "Device not responding to setup address"
    (error -71) and the camera is gone from sysfs. The firmware hung. No
    software step revives it (root hub re-authorize, port power cycle and
    USB reset were all tried); the cable must be replugged once. USBDEVFS_RESET
    on a streaming camera even caused this state, so it is NOT used here.

So the ladder only uses the safe steps:
  * a fresh process resets the camera once before its first open;
  * consecutive failures: retry, retry, hw_reset, hw_reset, then the process
    exits and Docker (restart: unless-stopped) starts a clean one;
  * a blind camera skips straight to the clean restart;
  * a camera that is not on the USB bus is only watched (cheap sysfs poll, no
    restarts) and is opened the moment it is plugged back in.

Needs the container to run `--privileged -v /dev/bus/usb:/dev/bus/usb`.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("perception.camera_recovery")

INTEL_VID = "8086"

HW_RESET_AT = 2
EXIT_AT = int(os.environ.get("RS_EXIT_AFTER_FAILURES", "4"))
STABLE_STREAM_S = 10.0           # streaming this long clears the failure count

STATE_OPENING = "opening"
STATE_STREAMING = "streaming"
STATE_ABSENT = "absent"          # not on the USB bus: replug the cable


def action_for(failures: int) -> str:
    if failures >= EXIT_AT:
        return "exit"
    if failures >= HW_RESET_AT:
        return "hw_reset"
    return "retry"


def backoff_s(failures: int) -> float:
    return min(2.0 + failures, 8.0)


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def usb_bus_readable() -> bool:
    """False if sysfs is not visible (then "absent" cannot be judged)."""
    return bool(glob.glob("/sys/bus/usb/devices/usb*"))


def find_realsense() -> Optional[dict]:
    """The camera's USB device, from sysfs (works even when librealsense
    cannot see it). None if the kernel does not list it at all."""
    for path in sorted(glob.glob("/sys/bus/usb/devices/*")):
        if _read(path + "/idVendor") != INTEL_VID:
            continue
        product = _read(path + "/product") or ""
        if "realsense" not in product.lower():
            continue
        return {"sys": path, "product": product}
    return None


def keep_awake() -> str:
    """Turn USB autosuspend off for the camera. Changes the host's sysfs (it
    resets at reboot), so it only runs with RS_USB_KEEP_AWAKE=1."""
    d = find_realsense()
    if d is None:
        return "keep_awake: camera not in sysfs"
    try:
        with open(d["sys"] + "/power/control", "w") as f:
            f.write("on")
        return "keep_awake: ok (%s)" % d["sys"]
    except OSError as exc:
        return "keep_awake: failed (%s)" % exc


class CameraSupervisor:
    """Counts consecutive failures, picks the recovery step and reports state
    for /status. Not thread safe: the loop thread owns it."""

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self.failures = 0
        self.total_failures = 0
        self.total_restarts_requested = 0
        self.blind = False
        self.opens = 0
        self.state = STATE_OPENING
        self.last_action: Optional[str] = None
        self.last_action_t: Optional[float] = None
        self.streaming_since: Optional[float] = None
        self.last_frame_t: Optional[float] = None
        self.absent_since: Optional[float] = None
        self.want_hw_reset = False

    def set_absent(self, absent: bool) -> None:
        if absent:
            if self.absent_since is None:
                self.absent_since = self._clock()
            self.state = STATE_ABSENT
            self.streaming_since = None
        else:
            if self.absent_since is not None:
                logger.info("camera is back on the USB bus after %.0f s", self._clock() - self.absent_since)
            self.absent_since = None
            if self.state == STATE_ABSENT:
                self.state = STATE_OPENING

    def before_open(self) -> str:
        """Decide the step for the next open. Returns retry | hw_reset | exit."""
        action = "exit" if self.blind else action_for(self.failures)
        self.want_hw_reset = action == "hw_reset"
        if self.opens == 0 and action == "retry" and os.environ.get("RS_RESET_ON_START", "1") != "0":
            action, self.want_hw_reset = "hw_reset", True      # fresh process: start from a known state
        self.opens += 1
        if action != "retry":
            self.last_action, self.last_action_t = action, self._clock()
            if action == "exit":
                self.total_restarts_requested += 1
            logger.warning("camera recovery: %s (failures=%d blind=%s)", action, self.failures, self.blind)
        return action

    def note_failure(self, blind: bool = False) -> None:
        """blind: the camera is in sysfs but librealsense cannot see it."""
        self.failures += 1
        self.total_failures += 1
        self.blind = blind
        self.streaming_since = None
        self.state = STATE_OPENING

    def note_frame(self) -> None:
        now = self._clock()
        self.last_frame_t = now
        self.state = STATE_STREAMING
        if self.streaming_since is None:
            self.streaming_since = now
        elif now - self.streaming_since >= STABLE_STREAM_S and self.failures:
            logger.info("camera stable, failure count cleared (was %d)", self.failures)
            self.failures = 0
            self.blind = False

    def status(self) -> dict:
        now = self._clock()
        absent = self.state == STATE_ABSENT
        return {
            "state": self.state,
            "needs_replug": absent,
            "hint": "camera is not on the USB bus (hung firmware or unplugged): replug the USB cable" if absent else None,
            "absent_for_s": round(now - self.absent_since, 0) if self.absent_since else None,
            "consecutive_failures": self.failures,
            "blind": self.blind,
            "total_failures": self.total_failures,
            "restarts_requested": self.total_restarts_requested,
            "last_recovery": self.last_action,
            "last_recovery_age_s": round(now - self.last_action_t, 1) if self.last_action_t else None,
            "last_frame_age_s": round(now - self.last_frame_t, 2) if self.last_frame_t else None,
            "streaming": self.streaming_since is not None,
            "usb_device": (find_realsense() or {}).get("sys"),
        }
