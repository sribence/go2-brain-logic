"""Fleet pillar — multi-robot mission control backend (port 9111, CONTRACT.md 9.6).

- Robot registry (``robots.yaml`` + ``POST/DELETE /robots``) polled at 1 Hz
  (core /state, safety /state, mission /missions, omni /health /persons
  /record/status); everything expressed in a common ``map`` frame via the
  per-robot ``T_map_robot`` (``POST /robots/{id}/align``).
- Shared zones / labels / home -> each robot's mission API (``sync.py``).
- Allocation (``allocator.py``): ``POST /fleet/missions`` (``robot_id`` or
  ``"auto"``), ``POST /fleet/alert``.
- ``GET /fleet/state``, ``WS /ws/fleet`` (2 Hz), ``GET /fleet/incidents``,
  Redis ``mc.fleet.robots`` (safety_guard keeps robots apart).

Legacy routes (``/robots``, ``/robots/{id}/state``, ``/health``) keep their
shape; robots unknown to the registry fall back to the old mock state.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

try:
    from . import allocator, sync
    from .registry import Align, Registry, Transport, ok, transform_mission
except ImportError:  # run as a script (PYTHONPATH=/app/fleet)
    import allocator  # type: ignore
    import sync  # type: ignore
    from registry import Align, Registry, Transport, ok, transform_mission  # type: ignore

PILLAR = "fleet"
PORT = int(os.environ.get("FLEET_PORT", "9111"))
HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS_YAML = os.environ.get("FLEET_ROBOTS_YAML", os.path.join(HERE, "robots.yaml"))
DATA_DIR = os.environ.get("FLEET_DATA_DIR", os.path.join(HERE, "data"))
POLL_S = float(os.environ.get("FLEET_POLL_S", "1.0"))
PUB_HZ = float(os.environ.get("FLEET_PUB_HZ", "2.0"))
WS_HZ = float(os.environ.get("FLEET_WS_HZ", "2.0"))
USE_PLAN = os.environ.get("FLEET_USE_PLAN", "1") == "1"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
API_TOKEN = os.environ.get("MC_API_TOKEN", "")
BUS_CHANNEL = "mc.fleet.robots"
TERMINAL = ("done", "completed", "succeeded", "failed", "aborted", "cancelled", "rejected", "error")
MAX_HISTORY = 200

LEGACY_IDS = ["go2", "xavier"]
_start_t = time.time()


def _mock_state(robot_id: str) -> dict:
    t = time.time() - _start_t
    phase = 0.0 if robot_id == "go2" else math.pi
    x = 2.0 * math.sin(t * 0.1 + phase)
    y = 2.0 * math.cos(t * 0.1 + phase)
    yaw = (t * 0.05 + phase) % (2 * math.pi)
    battery_pct = max(0.0, 100.0 - (t * 0.01) % 100.0)
    return {
        "robot_id": robot_id, "connected": True, "armed": False,
        "pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0, "yaw": round(yaw, 3),
                 "level_id": "ground", "t": time.time()},
        "battery": {"voltage": round(22.0 + 0.06 * battery_pct, 2), "current": round(0.4 + 0.3 * random.random(), 2),
                    "percent": round(battery_pct, 1), "t": time.time()},
        "status": "idle", "current_task_id": None, "t": time.time(),
    }


def log_event(level: str, msg: str, **extra: Any) -> None:
    rec = dict({"t": time.time(), "pillar": PILLAR, "level": level, "msg": msg}, **extra)
    line = json.dumps(rec, default=str)
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        with open(os.path.join(HERE, "logs", "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _default_redis(url: Optional[str] = None) -> Any:
    try:
        import redis  # noqa: WPS433
    except ImportError:
        return None
    try:
        if url:
            return redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        return redis.Redis(host=REDIS_HOST, port=6379, socket_timeout=0.5, socket_connect_timeout=0.5)
    except Exception:  # noqa: BLE001
        return None


class Fleet:
    """All fleet logic, HTTP-free (the FastAPI layer is thin)."""

    def __init__(self, registry: Registry, world: "sync.FleetWorld", transport: Any,
                 redis_factory: Optional[Callable[[Optional[str]], Any]] = None, use_plan: bool = USE_PLAN):
        self.reg = registry
        self.world = world
        self.tx = transport
        self.redis_factory = redis_factory or _default_redis
        self.use_plan = use_plan
        self.lock = threading.RLock()
        self.missions: List[Dict[str, Any]] = []     # fleet assignments, newest last
        self.alerts: List[Dict[str, Any]] = []
        self._redis: Dict[str, Any] = {}
        self._sync_retry: Dict[str, float] = {}
        self.reg.on_poll.append(self._after_poll)

    # --- state ---------------------------------------------------------------
    def _assignment_for(self, rid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            for a in reversed(self.missions):
                if a["robot_id"] == rid and a["state"] not in TERMINAL:
                    return a
        return None

    def robots(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        out = self.reg.summaries(now)
        for r in out:
            a = self._assignment_for(r["id"])
            r["priority"] = a["priority"] if a else 0
            r["target"] = a["target"] if a else None
            r["target_gid"] = a.get("gid") if a else None
            r["fleet_mission_id"] = a["fleet_mission_id"] if a else None
            r["sync"] = self.world.sync_state(r["id"]).get("version")
        return out

    def state(self) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            missions = [dict(m) for m in self.missions[-50:]]
            alerts = [dict(a) for a in self.alerts[-50:]]
        return {"t": round(now, 3), "map_frame": "map", "world_version": self.world.version,
                "robots": self.robots(now), "missions": missions, "alerts": alerts}

    def _after_poll(self, reg: Registry) -> None:
        now = time.time()
        # assignment states from each robot's mission list
        with self.lock:
            for a in self.missions:
                if a["state"] in TERMINAL or not a.get("robot_mission_id"):
                    continue
                r = reg.get(a["robot_id"])
                if r is None:
                    continue
                s = r.st.mission_states.get(str(a["robot_mission_id"]))
                if s:
                    a["state"] = s
                    a["updated_at"] = round(now, 3)
        # keep the shared world on every online robot (idempotent, cheap when up to date)
        for r in reg.all():
            if not r.online(now) or "mission" not in r.spec.urls:
                continue
            ss = self.world.sync_state(r.id)
            if ss.get("version") == self.world.version and ss.get("align") == r.spec.align.key():
                continue
            if now < self._sync_retry.get(r.id, 0.0):
                continue
            res = sync.push(self.world, r, self.tx)
            if res["status"] not in ("synced", "up_to_date"):
                self._sync_retry[r.id] = now + 10.0
                log_event("warn", "world sync failed", robot_id=r.id, errors=res["errors"][:5])

    # --- redis ---------------------------------------------------------------
    def _r(self, key: str, url: Optional[str]) -> Any:
        if key not in self._redis:
            self._redis[key] = self.redis_factory(url)
        return self._redis[key]

    def publish(self, now: Optional[float] = None) -> int:
        """mc.fleet.robots: map frame on the fleet's redis; each robot with a
        ``redis`` URL also gets the list in its own frame (its safety_guard
        compares against its own core pose)."""
        n = 0
        msg = self.reg.bus_robots(now)
        targets = [("fleet", None, msg)]
        for rob in self.reg.all():
            if rob.spec.redis:
                targets.append(("robot:" + rob.id, rob.spec.redis, self.reg.bus_robots(now, frame=rob.id)))
        for key, url, payload in targets:
            r = self._r(key, url)
            if r is None:
                continue
            try:
                r.publish(BUS_CHANNEL, json.dumps(payload))
                n += 1
            except Exception:  # noqa: BLE001
                self._redis.pop(key, None)
        return n

    # --- missions ------------------------------------------------------------
    def _plan_fn(self) -> Optional[Callable]:
        if not self.use_plan:
            return None

        def plan(robot: Dict[str, Any], target) -> Optional[float]:
            rob = self.reg.get(robot["id"])
            if rob is None or "mission" not in rob.spec.urls:
                return None
            x, y = rob.spec.align.to_robot(target[0], target[1])
            code, d = self.tx.post(rob.spec.urls["mission"] + "/plan",
                                   {"op": "goto", "x": round(x, 3), "y": round(y, 3)}, 1.5)
            if not ok(code) or not isinstance(d, dict):
                return None
            for src in (d, d.get("plan") or {}, d.get("path") or {}):
                if isinstance(src, dict) and src.get("length_m") is not None:
                    if src.get("ok") is False:
                        return None
                    return float(src["length_m"])
            return None
        return plan

    def _dispatch(self, rid: str, mission: Dict[str, Any], priority: int, target, kind: str,
                  gid: Any = None, execute: bool = True, preempt: bool = False,
                  robot_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rob = self.reg.get(rid)
        a = {"fleet_mission_id": "fm_" + uuid.uuid4().hex[:10], "robot_id": rid, "kind": kind,
             "priority": int(priority), "target": [round(target[0], 3), round(target[1], 3)] if target else None,
             "gid": gid, "name": mission.get("name") or kind, "state": "error", "robot_mission_id": None,
             "error": None, "created_at": round(time.time(), 3), "updated_at": round(time.time(), 3)}
        if rob is None or "mission" not in rob.spec.urls:
            a["error"] = "robot has no mission url"
        else:
            url = rob.spec.urls["mission"]
            if preempt and robot_summary and robot_summary.get("mission"):
                cur = robot_summary["mission"].get("mission_id")
                if cur:
                    self.tx.post("%s/missions/%s/abort" % (url, cur), {}, 1.5)
                    with self.lock:
                        for old in self.missions:
                            if old["robot_id"] == rid and old["state"] not in TERMINAL:
                                old["state"], old["error"] = "aborted", "preempted"
            m = transform_mission(mission, rob.spec.align, to_robot=True)
            m.setdefault("source", "api")
            m["fleet"] = {"fleet_mission_id": a["fleet_mission_id"], "priority": int(priority)}
            code, d = self.tx.post(url + "/missions" + ("" if execute else "?execute=0"), m, 5.0)
            if ok(code) and isinstance(d, dict):
                a["robot_mission_id"] = d.get("mission_id")
                a["state"] = d.get("state") or "queued"
                if isinstance(d.get("plan"), dict):
                    a["plan_eta_s"] = d["plan"].get("eta_s")
            else:
                a["error"] = "mission POST -> %s %s" % (code, (d or {}).get("detail", "") if isinstance(d, dict) else "")
        with self.lock:
            self.missions.append(a)
            del self.missions[:-MAX_HISTORY]
        log_event("info" if a["error"] is None else "warn", "dispatch", **{k: a[k] for k in ("fleet_mission_id", "robot_id", "kind", "state", "error")})
        return a

    def submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        mission = body.get("mission")
        if not isinstance(mission, dict) or not isinstance(mission.get("steps"), list) or not mission["steps"]:
            raise HTTPException(400, "mission.steps required")
        rid = body.get("robot_id") or "auto"
        priority = int(body.get("priority") or 0)
        swarm = bool(body.get("swarm"))
        target = allocator.mission_target(mission)
        robots = self.robots()
        by_id = {r["id"]: r for r in robots}
        if rid == "auto":
            if target is None:
                raise HTTPException(400, "auto allocation needs a spatial step (goto/patrol/...) or robot_id")
            ch = allocator.choose(robots, target, priority, self._plan_fn(), swarm=swarm,
                                  count=int(body.get("count") or 1), gid=body.get("gid"))
            if not ch["robot_ids"]:
                return {"ok": False, "reason": ch["reason"], "already": ch["already"], "ranking": ch["ranking"],
                        "assignments": []}
            ids, ranking = ch["robot_ids"], ch["ranking"]
        else:
            if rid not in by_id:
                raise HTTPException(404, "unknown robot_id")
            okk, why = allocator.eligibility(by_id[rid], priority)
            if not okk and why in ("offline", "safety_stop") and not body.get("force"):
                return {"ok": False, "reason": why, "ranking": [], "assignments": []}
            ids, ranking = [rid], []
        out = [self._dispatch(i, mission, priority, target, "mission", body.get("gid"),
                              execute=body.get("execute", True) is not False,
                              preempt=allocator.is_busy(by_id[i]), robot_summary=by_id[i]) for i in ids]
        return {"ok": all(a["error"] is None for a in out), "assignments": out, "ranking": ranking,
                "reason": "ok"}

    def alert(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            x, y = float(body["x"]), float(body["y"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "x, y required")
        frame = body.get("frame") or "map"
        if frame != "map":
            rob = self.reg.get(frame)
            if rob is None:
                raise HTTPException(400, "unknown frame")
            x, y = rob.spec.align.to_map(x, y)
        al = {"alert_id": "al_" + uuid.uuid4().hex[:10], "t": round(time.time(), 3), "x": round(x, 3),
              "y": round(y, 3), "kind": str(body.get("kind") or "alert"), "gid": body.get("gid"),
              "robot_id": body.get("robot_id"), "text": body.get("text"), "assignments": [], "reason": None}
        priority = int(body.get("priority") if body.get("priority") is not None else 5)
        robots = self.robots()
        ch = allocator.choose(robots, (x, y), priority, self._plan_fn(), swarm=bool(body.get("swarm")),
                              count=int(body.get("count") or 1), gid=body.get("gid"))
        al["reason"] = ch["reason"]
        al["already"] = ch["already"]
        if body.get("dispatch", True) is not False:
            by_id = {r["id"]: r for r in robots}
            for rid in ch["robot_ids"]:
                m = allocator.alert_mission(dict(body, x=x, y=y), by_id[rid])
                a = self._dispatch(rid, m, priority, (x, y), "alert", body.get("gid"),
                                   preempt=allocator.is_busy(by_id[rid]), robot_summary=by_id[rid])
                al["assignments"].append(a["fleet_mission_id"])
                al.setdefault("robot_ids", []).append(rid)
        with self.lock:
            self.alerts.append(al)
            del self.alerts[:-MAX_HISTORY]
        return dict(al, ranking=ch["ranking"])

    def mission_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(m) for m in reversed(self.missions)]

    def mission_verb(self, fmid: str, verb: str) -> Dict[str, Any]:
        with self.lock:
            a = next((m for m in self.missions if m["fleet_mission_id"] == fmid), None)
        if a is None:
            raise HTTPException(404, "unknown fleet mission")
        rob = self.reg.get(a["robot_id"])
        if rob is None or not a.get("robot_mission_id") or "mission" not in rob.spec.urls:
            raise HTTPException(409, "not dispatched")
        code, d = self.tx.post("%s/missions/%s/%s" % (rob.spec.urls["mission"], a["robot_mission_id"], verb), {}, 2.0)
        if ok(code) and isinstance(d, dict) and d.get("state"):
            with self.lock:
                a["state"] = d["state"]
        return {"ok": ok(code), "status": code, "robot": d, "assignment": a}

    def incidents(self) -> Dict[str, Any]:
        robots = [r for r in self.reg.all() if "omni" in r.spec.urls and r.online()]

        def one(rob):
            code, d = self.tx.get(rob.spec.urls["omni"] + "/incidents", 1.5)
            items = []
            if ok(code):
                lst = d if isinstance(d, list) else (d or {}).get("incidents") or []
                for it in lst:
                    if isinstance(it, dict):
                        items.append(dict(it, robot_id=rob.id, robot_name=rob.spec.name,
                                          omni_url=rob.spec.urls["omni"]))
            return rob.id, ok(code), items

        out: List[Dict[str, Any]] = []
        status = {}
        if robots:
            with ThreadPoolExecutor(max_workers=min(8, len(robots))) as ex:
                for rid, okk, items in ex.map(one, robots):
                    status[rid] = okk
                    out += items

        def ts(it):
            for k in ("t", "t_start", "started_at", "created_at", "ts"):
                try:
                    return float(it.get(k))
                except (TypeError, ValueError):
                    continue
            return 0.0
        out.sort(key=ts, reverse=True)
        return {"t": round(time.time(), 3), "incidents": out, "robots": status}


def create_app(fleet: Optional[Fleet] = None, background: bool = True) -> FastAPI:
    if fleet is None:
        tx = Transport(token=API_TOKEN)
        reg = Registry(ROBOTS_YAML, os.path.join(DATA_DIR, "registry.json"), tx)
        fleet = Fleet(reg, sync.FleetWorld(os.path.join(DATA_DIR, "world.json")), tx)
    app = FastAPI(title="mission-control: fleet")
    app.state.fleet = fleet
    F = fleet

    if background:
        @app.on_event("startup")
        def _start() -> None:
            F.reg.start(POLL_S)

            def pub_loop() -> None:
                while True:
                    t0 = time.time()
                    try:
                        F.publish()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(max(0.05, 1.0 / max(PUB_HZ, 0.2) - (time.time() - t0)))
            threading.Thread(target=pub_loop, daemon=True, name="fleet-pub").start()
            log_event("info", "fleet started", port=PORT, robots=F.reg.ids())

    # --- legacy routes (shape kept) ------------------------------------------
    def legacy_state(robot_id: str) -> dict:
        rob = F.reg.get(robot_id)
        if rob is None:
            if robot_id in LEGACY_IDS:
                return _mock_state(robot_id)
            raise HTTPException(status_code=404, detail="unknown robot_id")
        s = [r for r in F.robots() if r["id"] == robot_id][0]
        p = s["pose"] or {}
        m = s["mission"] or {}
        return {"robot_id": robot_id, "connected": s["online"], "armed": bool(s["armed"]),
                "pose": {"x": p.get("x"), "y": p.get("y"), "z": 0.0, "yaw": p.get("yaw"), "level_id": "ground",
                         "t": time.time()} if p else None,
                "battery": {"percent": s["battery"], "t": time.time()},
                "status": m.get("state") or ("idle" if s["online"] else "offline"),
                "current_task_id": m.get("mission_id"), "t": time.time(), "fleet": s}

    @app.get("/robots/go2/state")
    def go2_state():
        return legacy_state("go2")

    @app.get("/robots/xavier/state")
    def xavier_state():
        return legacy_state("xavier")

    @app.get("/robots")
    def list_robots():
        ids = F.reg.ids()
        return {"robots": ids or list(LEGACY_IDS), "details": [r.spec.to_dict() for r in F.reg.all()]}

    @app.get("/robots/{robot_id}/state")
    def robot_state(robot_id: str):
        return legacy_state(robot_id)

    @app.get("/health")
    def health():
        rs = F.robots()
        return {"ok": True, "pillar": PILLAR, "robots": [r["id"] for r in rs] or list(LEGACY_IDS),
                "online": [r["id"] for r in rs if r["online"]], "world_version": F.world.version}

    # --- registry --------------------------------------------------------------
    @app.post("/robots")
    def add_robot(body: Dict[str, Any] = Body(...)):
        try:
            r = F.reg.add(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        F.world.forget_robot(r.id)
        return {"ok": True, "robot": r.spec.to_dict()}

    @app.delete("/robots/{robot_id}")
    def del_robot(robot_id: str):
        if not F.reg.remove(robot_id):
            raise HTTPException(404, "unknown robot_id")
        F.world.forget_robot(robot_id)
        return {"ok": True}

    @app.post("/robots/{robot_id}/align")
    def align_robot(robot_id: str, body: Dict[str, Any] = Body(...)):
        if F.reg.get(robot_id) is None:
            raise HTTPException(404, "unknown robot_id")
        try:
            a = Align.from_dict(body)
            if not all(math.isfinite(v) for v in (a.dx, a.dy, a.dyaw)):
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(400, "dx, dy, dyaw must be finite numbers")
        return {"ok": True, "align": F.reg.set_align(robot_id, a.to_dict()).to_dict()}

    # --- fleet -----------------------------------------------------------------
    @app.get("/fleet/state")
    def fleet_state():
        return F.state()

    @app.get("/fleet/robots")
    def fleet_robots():
        return F.reg.bus_robots()

    @app.post("/fleet/missions")
    def fleet_missions_post(body: Dict[str, Any] = Body(...)):
        res = F.submit(body)
        return JSONResponse(status_code=200 if res["ok"] else 409, content=res)

    @app.get("/fleet/missions")
    def fleet_missions_get():
        return {"missions": F.mission_list()}

    @app.post("/fleet/missions/{fmid}/{verb}")
    def fleet_mission_verb(fmid: str, verb: str):
        if verb not in ("approve", "pause", "resume", "abort", "continue"):
            raise HTTPException(400, "bad verb")
        return F.mission_verb(fmid, verb)

    @app.post("/fleet/alert")
    def fleet_alert(body: Dict[str, Any] = Body(...)):
        return F.alert(body)

    @app.get("/fleet/alerts")
    def fleet_alerts():
        with F.lock:
            return {"alerts": list(reversed(F.alerts))}

    @app.get("/fleet/incidents")
    def fleet_incidents():
        return F.incidents()

    # --- shared world ------------------------------------------------------------
    @app.get("/fleet/world")
    def world_get():
        return F.world.export()

    @app.post("/fleet/zones")
    def zone_post(body: Dict[str, Any] = Body(...)):
        try:
            return {"ok": True, "zone": F.world.upsert_zone(body), "version": F.world.version}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/fleet/zones/{zid}")
    def zone_del(zid: str):
        if not F.world.delete_zone(zid):
            raise HTTPException(404, "unknown zone")
        return {"ok": True, "version": F.world.version}

    @app.post("/fleet/labels")
    def label_post(body: Dict[str, Any] = Body(...)):
        try:
            return {"ok": True, "label": F.world.upsert_label(body), "version": F.world.version}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/fleet/labels/{name}")
    def label_del(name: str):
        if not F.world.delete_label(name):
            raise HTTPException(404, "unknown label")
        return {"ok": True, "version": F.world.version}

    @app.get("/fleet/home")
    def home_get():
        return {"home": F.world.export().get("home")}

    @app.post("/fleet/home")
    def home_post(body: Optional[Dict[str, Any]] = Body(None)):
        try:
            return {"ok": True, "home": F.world.set_home(body), "version": F.world.version}
        except (ValueError, AttributeError) as e:
            raise HTTPException(400, str(e))

    @app.post("/fleet/sync")
    def world_sync(pull: bool = False, force: bool = False):
        pulled = [sync.pull_merge(F.world, r, F.tx) for r in F.reg.all() if r.online()] if pull else []
        pushed = sync.push_all(F.world, F.reg.all(), F.tx, force=force)
        return {"version": F.world.version, "pulled": pulled, "pushed": pushed}

    @app.websocket("/ws/fleet")
    async def ws_fleet(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_text(json.dumps(F.state(), default=str))
                await asyncio.sleep(1.0 / max(WS_HZ, 0.2))
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log_event("warn", "ws_fleet loop error", error=str(exc))

    return app


app = create_app() if os.environ.get("FLEET_NO_AUTOAPP") != "1" else None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
