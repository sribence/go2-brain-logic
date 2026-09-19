"""Follow ONE locked person -- state machine + dry-run velocity command.

Pure logic: no camera, no YOLO, no robot I/O. `app.py` feeds it one
perception result per frame and publishes what it WOULD command. Nothing
here sends a motion command; a later executor must go through `core`
(armed check, watchdog, E-stop) and must honour `command.valid_until`.

States
    IDLE       no lock
    ACQUIRING  locked, learning the appearance template (first N frames)
    TRACKING   target confirmed this frame -> command computed
    OCCLUDED   target not confirmed: missing, jumped too fast, or looks
               different. Command = 0. Re-acquire ONLY a person inside the
               spatial gate whose appearance matches the template, and only
               if exactly one such person exists.
    LOST       occluded longer than `lost_timeout_s`. Command = 0. Needs a
               manual re-lock -- the follower never picks someone new.

Frames: base frame, x forward, y left (metres); vyaw > 0 = turn left.
Command units match the Go2 SportClient `Move(vx, vy, vyaw)` (m/s, rad/s).
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

import appearance

IDLE, ACQUIRING, TRACKING, OCCLUDED, LOST = "IDLE", "ACQUIRING", "TRACKING", "OCCLUDED", "LOST"


@dataclass
class FollowConfig:
    # --- controller ---
    follow_distance_m: float = 1.2     # keep this far from the person
    distance_deadband_m: float = 0.15  # |error| below this -> vx = 0
    min_safe_distance_m: float = 0.7   # closer than this -> never drive forward
    k_distance: float = 0.8            # vx = k * distance error
    k_yaw: float = 1.5                 # vyaw = k * bearing (rad)
    yaw_deadband_deg: float = 5.0
    turn_first_deg: float = 45.0       # vx scaled to 0 at this bearing: turn before driving
    max_vx: float = 0.5                # m/s  (Go2 can do much more -- start slow)
    max_reverse_vx: float = 0.0        # m/s  backing away disabled by default
    max_vyaw: float = 0.8              # rad/s
    max_accel: float = 0.5             # m/s^2, ramp-up limit (stopping is immediate)
    max_yaw_accel: float = 1.5         # rad/s^2
    command_ttl_s: float = 0.5         # executor must drop a command older than this
    # --- target identity ---
    acquire_frames: int = 5            # frames to learn the appearance template
    max_person_speed: float = 2.0      # m/s, gate growth while unseen (fast walk)
    gate_base_m: float = 0.5           # gate radius around the predicted position
    gate_max_m: float = 1.5
    reid_min_similarity: float = 0.55  # appearance match threshold (0..1)
    mismatch_frames: int = 3           # consecutive mismatches -> OCCLUDED
    template_update_min: float = 0.75  # only confident matches update the template
    template_update_rate: float = 0.05
    lost_timeout_s: float = 2.0

    @classmethod
    def from_env(cls) -> "FollowConfig":
        cfg = cls()
        for k, v in asdict(cfg).items():
            env = os.environ.get("FOLLOW_" + k.upper())
            if env is not None:
                setattr(cfg, k, type(v)(float(env)) if isinstance(v, int) else float(env))
        return cfg


class TargetFollower:
    def __init__(self, cfg: Optional[FollowConfig] = None):
        self.cfg = cfg or FollowConfig()
        self.hold = False    # operator/gesture pause: keep tracking, command zero
        self._reset(IDLE, "no target locked")

    # ------------------------------------------------------------------ API

    def lock(self, track_id: int, persons: list[dict], t: float) -> None:
        p = next((q for q in persons if q["track_id"] == track_id and q["depth_ok"]), None)
        if p is None:
            raise ValueError(f"track #{track_id} is not visible with valid depth -- cannot lock")
        self._reset(ACQUIRING, "learning appearance")
        self.track_id = track_id
        self.locked_at = t
        self.state_since = t
        self._accept(p, t)

    def release(self, t: float) -> None:
        self._reset(IDLE, "released by operator")
        self.state_since = t

    def apply_ego_motion(self, dx: float, dy: float, dyaw: float) -> None:
        """Robot moved by (dx, dy, dyaw) in its previous base frame since the
        last update: re-express the remembered target position in the new
        frame. Not needed while the robot stands still (dry-run); REQUIRED
        once it moves, otherwise the gate drifts away from the person."""
        if self.last_pos is None:
            return
        c, s = math.cos(-dyaw), math.sin(-dyaw)
        px, py = self.last_pos[0] - dx, self.last_pos[1] - dy
        self.last_pos = np.array([c * px - s * py, s * px + c * py])
        vx, vy = self.last_vel
        self.last_vel = np.array([c * vx - s * vy, s * vx + c * vy])

    def update(self, persons: list[dict], features: dict[int, Optional[np.ndarray]], t: float) -> dict:
        visible = {p["track_id"]: p for p in persons if p["depth_ok"]}
        target: Optional[dict] = None

        if self.state in (ACQUIRING, TRACKING):
            target = self._check_locked_track(visible, features, t)

        if self.state == OCCLUDED:
            if t - self.last_seen_t > self.cfg.lost_timeout_s:
                self._set_state(LOST, f"not re-identified within {self.cfg.lost_timeout_s:.1f} s -- re-lock manually", t)
            else:
                target = self._try_reacquire(visible, features, t)

        return self._output(target, t)

    # ------------------------------------------------------------ internals

    def _reset(self, state: str, reason: str) -> None:
        self.state = state
        self.reason = reason
        self.track_id: Optional[int] = None
        self.locked_at: Optional[float] = None
        self.state_since = 0.0
        self.template: Optional[np.ndarray] = None
        self._samples: list[np.ndarray] = []
        self.last_pos: Optional[np.ndarray] = None
        self.last_vel = np.zeros(2)
        self.last_seen_t = 0.0
        self.last_similarity: Optional[float] = None
        self._mismatches = 0
        self._occlusion_cause = ""
        self._cmd = np.zeros(2)          # current (ramped) vx, vyaw
        self._cmd_t: Optional[float] = None

    def _set_state(self, state: str, reason: str, t: float) -> None:
        if state != self.state:
            self.state_since = t
            if state == OCCLUDED:
                self._occlusion_cause = reason
        self.state = state
        self.reason = reason

    def _predicted(self, t: float) -> tuple[np.ndarray, float]:
        dt = max(0.0, t - self.last_seen_t)
        pred = self.last_pos + self.last_vel * min(dt, 1.0)
        gate = min(self.cfg.gate_base_m + self.cfg.max_person_speed * dt, self.cfg.gate_max_m)
        return pred, gate

    @staticmethod
    def _raw_xy(p: dict) -> np.ndarray:
        pos = p.get("position_raw") or p["position"]
        return np.array([pos["x"], pos["y"]])

    def _accept(self, p: dict, t: float) -> None:
        self.last_pos = self._raw_xy(p)
        v = p.get("velocity") or {"vx": 0.0, "vy": 0.0}
        self.last_vel = np.array([v["vx"], v["vy"]])
        self.last_seen_t = t

    def _check_locked_track(self, visible: dict, features: dict, t: float) -> Optional[dict]:
        p = visible.get(self.track_id)
        if p is None:
            self._set_state(OCCLUDED, "target not visible", t)
            return None

        pred, gate = self._predicted(t)
        jump = float(np.linalg.norm(self._raw_xy(p) - pred))
        if jump > gate:
            # Same ByteTrack id but physically impossible move: an id switch
            # onto someone else, or a depth glitch. Either way, not our person.
            self._set_state(OCCLUDED, f"position jump {jump:.2f} m > gate {gate:.2f} m", t)
            return None

        feat = features.get(self.track_id)
        if self.state == ACQUIRING:
            if feat is not None:
                self._samples.append(feat)
            if len(self._samples) >= self.cfg.acquire_frames:
                self.template = np.nanmean(np.stack(self._samples), axis=0)
                self._set_state(TRACKING, "appearance learned", t)
            else:
                self.reason = f"learning appearance {len(self._samples)}/{self.cfg.acquire_frames}"
            self._accept(p, t)
            return p

        sim = appearance.similarity(self.template, feat)
        self.last_similarity = sim
        if sim is not None and sim < self.cfg.reid_min_similarity:
            self._mismatches += 1
            if self._mismatches >= self.cfg.mismatch_frames:
                self._set_state(OCCLUDED, f"appearance mismatch ({sim:.2f} < {self.cfg.reid_min_similarity:.2f})", t)
                return None
        else:
            self._mismatches = 0
            if sim is not None and sim >= self.cfg.template_update_min:
                self.template = appearance.blend(self.template, feat, self.cfg.template_update_rate)
        self._set_state(TRACKING, "tracking", t)
        self._accept(p, t)
        return p

    def _try_reacquire(self, visible: dict, features: dict, t: float) -> Optional[dict]:
        pred, gate = self._predicted(t)
        candidates = []
        for tid, p in visible.items():
            if np.linalg.norm(self._raw_xy(p) - pred) > gate:
                continue
            if self.template is None:
                # Still acquiring: no appearance yet, accept only the same id.
                if tid == self.track_id:
                    candidates.append((tid, p, None))
                continue
            sim = appearance.similarity(self.template, features.get(tid))
            if sim is not None and sim >= self.cfg.reid_min_similarity:
                candidates.append((tid, p, sim))

        if len(candidates) > 1:
            self.reason = f"{self._occlusion_cause}; ambiguous: {len(candidates)} matching people in gate -- waiting"
            return None
        if not candidates:
            self.reason = f"{self._occlusion_cause}; searching (gate {gate:.2f} m, {t - self.last_seen_t:.1f} s)"
            return None

        tid, p, sim = candidates[0]
        switched = tid != self.track_id
        self.track_id = tid
        self.last_similarity = sim
        self._mismatches = 0
        self._set_state(TRACKING if self.template is not None else ACQUIRING,
                        f"re-identified as #{tid}" if switched else "re-acquired", t)
        self._accept(p, t)
        return p

    def _compute_command(self, target: Optional[dict], t: float) -> tuple[np.ndarray, Optional[dict]]:
        cfg = self.cfg
        goal = None
        desired = np.zeros(2)
        if self.state == TRACKING and target is not None:
            x, y = target["position"]["x"], target["position"]["y"]
            dist = math.hypot(x, y)
            bearing = math.atan2(y, x)
            err = dist - cfg.follow_distance_m

            vx = 0.0 if abs(err) < cfg.distance_deadband_m else cfg.k_distance * err
            vx = float(np.clip(vx, -cfg.max_reverse_vx, cfg.max_vx))
            if dist < cfg.min_safe_distance_m:
                vx = min(vx, 0.0)
            if vx > 0:
                vx *= float(np.clip(1.0 - abs(math.degrees(bearing)) / cfg.turn_first_deg, 0.0, 1.0))

            vyaw = 0.0 if abs(math.degrees(bearing)) < cfg.yaw_deadband_deg else cfg.k_yaw * bearing
            vyaw = float(np.clip(vyaw, -cfg.max_vyaw, cfg.max_vyaw))
            desired = np.zeros(2) if self.hold else np.array([vx, vyaw])

            step = max(dist - cfg.follow_distance_m, 0.0)
            goal = {
                "x": round(x - cfg.follow_distance_m * math.cos(bearing), 3),
                "y": round(y - cfg.follow_distance_m * math.sin(bearing), 3),
                "yaw_deg": round(math.degrees(bearing), 1),
                "distance_m": round(step, 3),
                "distance_cm": int(round(step * 100)),
            }

        # Ramp up gradually, stop immediately.
        dt = 0.1 if self._cmd_t is None else max(1e-3, min(t - self._cmd_t, 0.5))
        limits = np.array([cfg.max_accel, cfg.max_yaw_accel]) * dt
        for i in range(2):
            cur, des = float(self._cmd[i]), float(desired[i])
            if cur * des < 0:
                cur = 0.0                      # direction change: stop first
            if abs(des) <= abs(cur):
                cur = des                      # slowing down: immediate
            else:
                cur += float(np.clip(des - cur, -limits[i], limits[i]))
            self._cmd[i] = cur
        self._cmd_t = t
        return desired, goal

    def _output(self, target: Optional[dict], t: float) -> dict:
        desired, goal = self._compute_command(target, t)
        pred_gate = None
        if self.last_pos is not None and self.state in (ACQUIRING, TRACKING, OCCLUDED):
            pred, gate = self._predicted(t)
            pred_gate = {"x": round(float(pred[0]), 3), "y": round(float(pred[1]), 3), "radius_m": round(gate, 2)}

        tgt = None
        if target is not None:
            tgt = {k: target[k] for k in ("track_id", "position", "velocity", "distance_m", "bearing_deg")}
        vx, vyaw = (round(float(v), 3) for v in self._cmd)
        return {
            "state": self.state,
            "reason": self.reason,
            "track_id": self.track_id,
            "locked_at": self.locked_at,
            "state_since": self.state_since,
            "last_seen_age_s": round(t - self.last_seen_t, 2) if self.last_seen_t else None,
            "similarity": None if self.last_similarity is None else round(self.last_similarity, 3),
            "gate": pred_gate,
            "target": tgt,
            "goal": goal,
            "command": {
                "vx": vx, "vy": 0.0, "vyaw": vyaw,
                "vx_desired": round(float(desired[0]), 3), "vyaw_desired": round(float(desired[1]), 3),
                "valid_until": t + self.cfg.command_ttl_s,
                "dry_run": True,
            },
        }
