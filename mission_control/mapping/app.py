"""mission-control: mapping pillar (port 9102).

Autonomous "wander and map" engine. Builds a dual-layer (floor + walls)
occupancy grid by frontier exploration: repeatedly find the nearest
reachable unexplored boundary cell, walk there while continuously
integrating LiDAR into the grid, repeat until no frontier remains.

See mapping_core.py for the actual grid/frontier/log-odds logic (pure,
testable) -- this file is just the FastAPI + background-thread + Redis
wiring around it, per CONVENTIONS.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from fastapi import FastAPI, HTTPException
import uvicorn

from robot_client import get_robot_client

from mapping_core import (
    MapStore,
    PointMap,
    StairDetector,
    integrate_scan,
    next_frontier_target,
    STAIR_WALL_MIN,
    STAIR_WALL_MAX,
)

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

PILLAR = "mapping"
PORT = 9102

GRID_WIDTH = int(os.environ.get("MAP_WIDTH", "200"))
GRID_HEIGHT = int(os.environ.get("MAP_HEIGHT", "200"))
GRID_RESOLUTION = float(os.environ.get("MAP_RESOLUTION", "0.05"))
GRID_ORIGIN_X = float(os.environ.get("MAP_ORIGIN_X", str(-GRID_WIDTH * GRID_RESOLUTION / 2)))
GRID_ORIGIN_Y = float(os.environ.get("MAP_ORIGIN_Y", str(-GRID_HEIGHT * GRID_RESOLUTION / 2)))

FORWARD_SPEED = float(os.environ.get("EXPLORE_FORWARD_SPEED", "0.35"))
SLOW_SPEED = float(os.environ.get("EXPLORE_SLOW_SPEED", "0.12"))
YAW_KP = float(os.environ.get("EXPLORE_YAW_KP", "1.4"))
WAYPOINT_TOL_M = float(os.environ.get("EXPLORE_WAYPOINT_TOL_M", "0.12"))
WAYPOINT_TIMEOUT_S = float(os.environ.get("EXPLORE_WAYPOINT_TIMEOUT_S", "8.0"))
TICK_S = float(os.environ.get("EXPLORE_TICK_S", "0.1"))
MAP_PUBLISH_INTERVAL_S = 1.0

# -- live object overlay (perception -> world-frame markers) -------------
PERCEPTION_URL = os.environ.get("PERCEPTION_URL", "http://127.0.0.1:9112")
OBJECTS_POLL_HZ = float(os.environ.get("OBJECTS_POLL_HZ", "5"))
OBJECTS_TTL_S = float(os.environ.get("OBJECTS_TTL_S", "2.0"))
OBJECTS_HTTP_TIMEOUT_S = 0.5

# -- continuous LiDAR ingest into the 2D grid + persistent 3D point map --
# Runs always (armed or not, exploring or teleop-driven), unlike the 2D
# grid's own integrate_scan calls inside _explore_loop which only fire
# during autonomous frontier exploration -- this is what makes points
# "stick" while the robot is just being walked around by hand.
LIDAR_INGEST_HZ = float(os.environ.get("LIDAR_INGEST_HZ", "5"))
POINTS3D_DEFAULT_LIMIT = int(os.environ.get("POINTS3D_DEFAULT_LIMIT", "200000"))

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)
_log_lock = threading.Lock()


def log_event(level: str, msg: str, **extra) -> None:
    rec = {"t": time.time(), "pillar": PILLAR, "level": level, "msg": msg, **extra}
    line = json.dumps(rec)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def base_to_world(robot_x: float, robot_y: float, robot_yaw: float,
                   fwd: float, left: float) -> tuple[float, float]:
    """Rotates a perception base-frame point (x=forward, y=left) into the
    world frame using the robot's pose, same convention as
    follow_executor.ego_delta (mission_control/follow_executor/executor.py)."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return robot_x + c * fwd - s * left, robot_y + s * fwd + c * left


class ObjectOverlay:
    """Polls perception's /persons and keeps a TTL'd world-frame marker per
    track_id -- independent of exploration/arming, so it runs even when the
    robot is just sitting still and being driven manually. Not folded into
    the log-odds grid: these are transient, moving detections, not static
    occupancy.
    """

    def __init__(self, robot):
        self._robot = robot
        self._lock = threading.Lock()
        self._objects: dict[int, dict] = {}
        self._last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="object_overlay")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / OBJECTS_POLL_HZ
        while True:
            t0 = time.time()
            try:
                self._poll_once()
            except Exception as exc:  # pragma: no cover - defensive, keep the service alive
                self._last_error = str(exc)
            self._prune()
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _poll_once(self) -> None:
        try:
            with urllib.request.urlopen(f"{PERCEPTION_URL}/persons", timeout=OBJECTS_HTTP_TIMEOUT_S) as r:
                result = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            self._last_error = str(exc)
            return
        self._last_error = None

        pose = self._robot.get_pose()
        now = time.time()
        with self._lock:
            for p in result.get("persons", []):
                pos = p.get("position")
                if not pos:
                    continue
                wx, wy = base_to_world(pose.x, pose.y, pose.yaw, pos["x"], pos["y"])
                self._objects[p["track_id"]] = {
                    "id": p["track_id"],
                    "cls": "person",
                    "x": round(wx, 3),
                    "y": round(wy, 3),
                    "confidence": p.get("confidence"),
                    "last_seen": now,
                }

    def _prune(self) -> None:
        now = time.time()
        with self._lock:
            stale = [tid for tid, o in self._objects.items() if now - o["last_seen"] > OBJECTS_TTL_S]
            for tid in stale:
                del self._objects[tid]

    def list(self) -> dict:
        now = time.time()
        with self._lock:
            objects = [
                {**o, "age_s": round(now - o["last_seen"], 2)}
                for o in self._objects.values()
            ]
        return {"objects": objects, "count": len(objects), "perception_error": self._last_error, "t": now}


class MappingService:
    def __init__(self):
        self.robot = get_robot_client()
        self.store = MapStore(resolution=GRID_RESOLUTION, width=GRID_WIDTH, height=GRID_HEIGHT,
                               origin_x=GRID_ORIGIN_X, origin_y=GRID_ORIGIN_Y)
        self.stair_detector = StairDetector()

        self._redis = None
        if redis is not None:
            try:
                self._redis = redis.Redis(
                    host=os.environ.get("REDIS_HOST", "redis"), port=6379, decode_responses=True,
                    socket_connect_timeout=1.0,
                )
                self._redis.ping()
            except Exception as exc:
                log_event("warn", "redis unavailable, map_update publishing disabled", error=str(exc))
                self._redis = None

        self._explore_lock = threading.Lock()
        self._explore_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state = "idle"          # idle | exploring | error
        self._last_error: Optional[str] = None
        self._last_publish_t = 0.0

        self.objects = ObjectOverlay(self.robot)
        self.objects.start()

        self.points = PointMap()
        self._ingest_thread = threading.Thread(target=self._lidar_ingest_loop, daemon=True,
                                                 name="lidar_ingest")
        self._ingest_thread.start()

    def _lidar_ingest_loop(self) -> None:
        period = 1.0 / LIDAR_INGEST_HZ
        while True:
            t0 = time.time()
            try:
                pose = self.robot.get_pose()
                pts = self.robot.get_lidar_points()
                if pts:
                    integrate_scan(self.store.grid, pose.x, pose.y, pts)
                    self.points.add(pts)
                    self._maybe_publish()
            except Exception as exc:  # pragma: no cover - defensive, keep the service alive
                log_event("warn", "lidar ingest failed", error=str(exc))
            time.sleep(max(0.0, period - (time.time() - t0)))

    # -- map access -------------------------------------------------
    def map_dict(self, level_id: Optional[str] = None) -> Optional[dict]:
        grid = self.store.get(level_id)
        return grid.to_schema_dict() if grid else None

    def _maybe_publish(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_publish_t) < MAP_PUBLISH_INTERVAL_S:
            return
        self._last_publish_t = now
        if self._redis is None:
            return
        try:
            self._redis.publish("mc.mapping.map_update", json.dumps(self.store.grid.to_schema_dict()))
        except Exception as exc:
            log_event("warn", "map_update publish failed", error=str(exc))

    def _publish_anomaly(self, msg: str, **extra) -> None:
        if self._redis is None:
            return
        try:
            self._redis.publish("mc.core.anomaly", json.dumps({
                "t": time.time(), "pillar": PILLAR, "msg": msg, **extra,
            }))
        except Exception as exc:
            log_event("warn", "anomaly publish failed", error=str(exc))

    # -- status -------------------------------------------------------
    def status(self) -> dict:
        grid = self.store.grid
        explored, coverage = grid.coverage_stats()
        return {
            "state": self._state,
            "level_id": grid.level_id,
            "levels": list(self.store.levels.keys()),
            "cells_explored": explored,
            "coverage_percent": round(coverage, 2),
            "total_cells": grid.width * grid.height,
            "last_error": self._last_error,
        }

    # -- explore control -----------------------------------------------
    def start_explore(self) -> None:
        with self._explore_lock:
            if self._explore_thread is not None and self._explore_thread.is_alive():
                return
            if not self.robot.is_armed():
                raise PermissionError("robot not armed")
            self._stop_event.clear()
            self._last_error = None
            self._state = "exploring"
            self._explore_thread = threading.Thread(target=self._explore_loop, daemon=True)
            self._explore_thread.start()
            log_event("info", "exploration started")

    def stop_explore(self) -> None:
        self._stop_event.set()
        t = self._explore_thread
        if t is not None:
            t.join(timeout=5.0)
        self.robot.stop()
        self._state = "idle"
        log_event("info", "exploration stopped")

    # -- the actual wander-and-map loop ---------------------------------
    def _explore_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self.robot.is_armed():
                    self._last_error = "robot disarmed mid-exploration"
                    log_event("warn", self._last_error)
                    break

                pose = self.robot.get_pose()
                imu = self.robot.get_imu()

                if self.stair_detector.update(imu.pitch):
                    new_level = self.store.new_level()
                    log_event("info", "level change detected, switched active grid",
                              new_level_id=new_level, pitch=imu.pitch)

                grid = self.store.grid
                integrate_scan(grid, pose.x, pose.y, self.robot.get_lidar_points())
                self._maybe_publish()

                sgx, sgy = grid.clamp_cell(*grid.world_to_grid(pose.x, pose.y))
                result = next_frontier_target(grid, sgx, sgy)
                if result is None:
                    log_event("info", "exploration complete, no reachable frontier left",
                              level_id=grid.level_id)
                    break

                target_cell, path_cells = result
                progressed = self._drive_path(grid, path_cells)
                if not progressed:
                    break
        except Exception as exc:  # pragma: no cover - defensive, keep the service alive
            self._last_error = str(exc)
            log_event("error", "exploration loop crashed", error=str(exc))
        finally:
            self.robot.stop()
            self._state = "idle"
            self._maybe_publish(force=True)

    def _drive_path(self, grid, path_cells: list[tuple[int, int]]) -> bool:
        """Drives the robot along path_cells (grid cells, start..target
        inclusive), re-scanning the LiDAR into `grid` continuously.
        Returns False if the caller should stop exploring altogether
        (disarmed / stop requested), True to keep exploring (even if this
        particular leg timed out -- the next outer-loop iteration will
        just replan against the freshly-updated grid)."""
        waypoints = [grid.grid_to_world(gx, gy) for gx, gy in path_cells[1:]]
        for wx, wy in waypoints:
            t_start = time.time()
            while True:
                if self._stop_event.is_set():
                    return False
                if not self.robot.is_armed():
                    self._last_error = "robot disarmed mid-exploration"
                    log_event("warn", self._last_error)
                    return False

                pose = self.robot.get_pose()
                dx, dy = wx - pose.x, wy - pose.y
                dist = math.hypot(dx, dy)
                if dist < WAYPOINT_TOL_M:
                    break
                if time.time() - t_start > WAYPOINT_TIMEOUT_S:
                    log_event("warn", "waypoint timeout, replanning", x=wx, y=wy)
                    self.robot.stop()
                    return True

                target_yaw = math.atan2(dy, dx)
                yaw_err = _wrap_angle(target_yaw - pose.yaw)

                gx, gy = grid.clamp_cell(*grid.world_to_grid(pose.x, pose.y))
                wall_val = grid.walls[grid.idx(gx, gy)]
                on_stairs = STAIR_WALL_MIN <= wall_val <= STAIR_WALL_MAX
                speed = SLOW_SPEED if on_stairs else FORWARD_SPEED

                vyaw = max(-1.0, min(1.0, YAW_KP * yaw_err))
                vx = speed * max(0.15, 1.0 - abs(yaw_err) / math.pi)
                try:
                    self.robot.move(vx, 0.0, vyaw)
                except Exception as exc:
                    self._last_error = f"move command refused: {exc}"
                    log_event("error", self._last_error)
                    return False

                integrate_scan(grid, pose.x, pose.y, self.robot.get_lidar_points())
                self._maybe_publish()

                time.sleep(TICK_S)
        self.robot.stop()
        return True


svc = MappingService()
app = FastAPI(title="mission-control: mapping")


@app.get("/map")
def get_map(level_id: Optional[str] = None):
    m = svc.map_dict(level_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"unknown level_id {level_id!r}")
    return m


@app.get("/objects")
def get_objects():
    return svc.objects.list()


@app.get("/points3d")
def get_points3d(xmin: Optional[float] = None, xmax: Optional[float] = None,
                  ymin: Optional[float] = None, ymax: Optional[float] = None,
                  limit: int = POINTS3D_DEFAULT_LIMIT):
    """Persistent 3D point map (mapping_core.PointMap) -- every voxel the
    robot has ever seen, world-frame, points stick permanently. Pass
    xmin/xmax/ymin/ymax to window around the robot instead of pulling the
    whole (potentially building-sized) map every poll."""
    bounded = xmin is not None and xmax is not None and ymin is not None and ymax is not None
    pts = svc.points.query(xmin, xmax, ymin, ymax, limit=limit) if bounded \
        else svc.points.query(limit=limit)
    total = svc.points.count()
    return {
        "points": [{"x": x, "y": y, "z": z} for x, y, z in pts],
        "count": len(pts),
        "total": total,
        "truncated": len(pts) < total if not bounded else None,
        "voxel_size": svc.points.voxel_size,
    }


@app.post("/explore/start", status_code=202)
def explore_start():
    try:
        svc.start_explore()
    except PermissionError:
        raise HTTPException(status_code=409, detail="robot is not armed")
    return svc.status()


@app.post("/explore/stop")
def explore_stop():
    svc.stop_explore()
    return svc.status()


@app.get("/explore/status")
def explore_status():
    return svc.status()


@app.get("/health")
def health():
    return {"ok": True, "pillar": PILLAR, "state": svc._state}


if os.environ.get("MC_ENABLE_DEBUG_ENDPOINTS") == "1":
    # Only mounted when explicitly enabled -- these bypass core's shared
    # arm state (AUDIT-2026-09-10.md, P0-7).

    @app.post("/_debug/arm")
    def debug_arm():
        svc.robot.set_armed(True)
        return {"armed": True}

    @app.post("/_debug/disarm")
    def debug_disarm():
        svc.robot.set_armed(False)
        return {"armed": False}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
