"""Operator-facing follow modes on top of `TargetFollower`.

The go2-console sends a mode, a follow distance, an audio switch, a dry-run
switch and manual gestures. This module turns those into follower actions,
so the rules live in one testable place instead of in HTTP handlers.

Modes:
    off          -- follower released, command zero.
    user_follow  -- the operator locks ONE person (`lock()`); the follower
                    never switches to anyone else (see target_follower.py).
    intruder     -- no operator lock: the nearest person with valid depth
                    inside `intruder_range_m` is locked automatically and an
                    alert is raised. After LOST, the next person may be
                    locked once `intruder_rearm_s` has passed. This is the
                    only mode that locks without an operator.
    trick        -- like user_follow, plus gestures (`gesture()`):
                    wave -> hello, stop -> hold (command zero) + sit,
                    ok -> resume following.

dry_run defaults to True and can only be switched off when the process was
started with live motion allowed (env PERCEPTION_ALLOW_LIVE=1). This pillar
still never sends a motion command itself: `dry_run: false` only marks the
command as allowed for a separate executor that goes through mc_motion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from target_follower import IDLE, LOST, TRACKING, TargetFollower

MODES = ("off", "user_follow", "intruder", "trick")
GESTURES = {
    # gesture: (planned robot action, effect on following)
    "wave": ("hello", None),
    "stop": ("sit", "hold"),
    "ok": (None, "resume"),
}
MIN_DISTANCE_M, MAX_DISTANCE_M = 1.0, 4.0


@dataclass
class ModeConfig:
    intruder_range_m: float = 6.0
    intruder_rearm_s: float = 3.0
    max_alerts: int = 20


class ModeError(Exception):
    """Request not allowed in the current state; `status` is the HTTP code."""

    def __init__(self, msg: str, status: int = 409):
        super().__init__(msg)
        self.status = status


class FollowSupervisor:
    def __init__(self, follower: TargetFollower, allow_live: bool = False,
                 cfg: Optional[ModeConfig] = None,
                 on_alert: Optional[Callable[[dict], None]] = None):
        self.follower = follower
        self.allow_live = allow_live
        self.cfg = cfg or ModeConfig()
        self.on_alert = on_alert
        self.mode = "off"
        self.audio_alert = True
        self.dry_run = True
        self.last_gesture: Optional[dict] = None
        self.alerts: list[dict] = []
        self._lost_since: Optional[float] = None
        self._prev_state = IDLE

    # ------------------------------------------------------------ settings

    @property
    def target_distance_m(self) -> float:
        return self.follower.cfg.follow_distance_m

    def set_mode(self, mode: str, t: float) -> None:
        if mode not in MODES:
            raise ModeError(f"unknown mode {mode!r}, expected one of {MODES}", 422)
        if mode == self.mode:
            return
        self.follower.hold = False
        if mode in ("off", "intruder") or self.mode == "intruder":
            # intruder locks are automatic; never carry them into an operator mode
            self.follower.release(t)
        self.mode = mode
        self._lost_since = None

    def set_distance(self, meters: float) -> None:
        if not MIN_DISTANCE_M <= meters <= MAX_DISTANCE_M:
            raise ModeError(f"target_distance_m must be {MIN_DISTANCE_M}-{MAX_DISTANCE_M} m", 422)
        self.follower.cfg.follow_distance_m = float(meters)

    def set_dry_run(self, value: bool) -> None:
        if not value and not self.allow_live:
            raise ModeError("live motion is disabled on this pillar (start with PERCEPTION_ALLOW_LIVE=1)", 403)
        self.dry_run = bool(value)

    # ------------------------------------------------------ operator input

    def lock(self, track_id: int, persons: list[dict], t: float) -> None:
        if self.mode == "off":
            self.mode = "user_follow"
        self.follower.hold = False
        self.follower.lock(track_id, persons, t)

    def release(self, t: float) -> None:
        self.follower.hold = False
        self.follower.release(t)
        self.mode = "off"

    def gesture(self, name: str, t: float) -> dict:
        if name not in GESTURES:
            raise ModeError(f"unknown gesture {name!r}, expected one of {tuple(GESTURES)}", 422)
        if self.mode != "trick":
            raise ModeError("gestures are only accepted in trick mode")
        action, effect = GESTURES[name]
        if effect == "hold":
            self.follower.hold = True
        elif effect == "resume":
            self.follower.hold = False
        self.last_gesture = {
            "gesture": name, "t": t, "action": action, "effect": effect,
            # no executor yet: the action is planned, not performed
            "executed": False, "dry_run": self.dry_run,
        }
        return self.last_gesture

    # ------------------------------------------------------- every frame

    def before_update(self, persons: list[dict], t: float) -> None:
        """Automatic intruder lock. Call before follower.update()."""
        if self.mode != "intruder":
            return
        f = self.follower
        if f.state == LOST:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since < self.cfg.intruder_rearm_s:
                return
            f.release(t)
        if f.state != IDLE:
            return
        cands = [p for p in persons if p["depth_ok"] and p["distance_m"] is not None
                 and p["distance_m"] <= self.cfg.intruder_range_m]
        if not cands:
            return
        p = min(cands, key=lambda q: q["distance_m"])
        f.lock(p["track_id"], persons, t)
        self._lost_since = None
        self._alert("intruder", t, track_id=p["track_id"], distance_m=p["distance_m"],
                    bearing_deg=p["bearing_deg"])

    def after_update(self, follow: dict, t: float) -> dict:
        """Apply mode, hold and dry-run to the follower output."""
        state = follow["state"]
        if state == LOST and self._prev_state != LOST and self.mode != "off":
            self._alert("target_lost", t, track_id=follow["track_id"])
        self._prev_state = state
        follow["command"]["dry_run"] = self.dry_run
        follow["mode"] = self.mode
        follow["hold"] = self.follower.hold
        if self.follower.hold and state == TRACKING:
            follow["reason"] = "hold (gesture stop) -- tracking, not moving"
        return follow

    def _alert(self, kind: str, t: float, **extra) -> None:
        alert = {"kind": kind, "t": t, "mode": self.mode, "audio": self.audio_alert, **extra}
        self.alerts.append(alert)
        del self.alerts[:-self.cfg.max_alerts]
        if self.on_alert:
            self.on_alert(alert)

    # ------------------------------------------------------------- output

    def summary(self, follow: Optional[dict], now: Optional[float] = None) -> dict:
        """Flat block for the UI; `follow` is the latest follower output."""
        follow = follow or {}
        target = follow.get("target") or {}
        cmd = follow.get("command") or {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "dry_run": self.dry_run}
        dist = target.get("distance_m")
        return {
            "mode": self.mode,
            "target_distance_m": round(self.target_distance_m, 2),
            "audio_alert": self.audio_alert,
            "dry_run": self.dry_run,
            "live_allowed": self.allow_live,
            "hold": self.follower.hold,
            "state": follow.get("state", self.follower.state),
            "reason": follow.get("reason", self.follower.reason),
            "target_id": follow.get("track_id"),
            "command": cmd,
            "target_dist_cm": None if dist is None else int(round(dist * 100)),
            "goal": follow.get("goal"),
            "last_gesture": self.last_gesture,
            "alerts": self.alerts[-5:],
            "t": now if now is not None else time.time(),
        }
