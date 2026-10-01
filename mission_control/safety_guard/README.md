# safety_guard — független sebesség-korlátozó (port 9115)

Az EGYETLEN folyamat, ami `mc_motion /move`-ot hív. Minden mozgásforrás
(navigation, mapping explore, follow_executor, omni pursuit) ide küld; a
`core.decide()` a friss ember-trackek (`mc.omni.persons`) és a LiDAR alapján
szűr, csak a kimenet megy tovább. **Nem armol, nem indít mozgást.**

| Fájl | Tartalom |
|---|---|
| `core.py` | tiszta `decide()` + LiDAR-segédek (I/O nélkül) |
| `app.py` | FastAPI, redis/HTTP bemenetek, forwarder, 20 Hz state |
| tesztek | `tests/test_safety_guard.py` |

## Döntés (`core.decide`)

| Feltétel | Szint | Hatás |
|---|---|---|
| ember < 0.8 m (eff.) / TTC < 1 s | STOP | vx=vy=0, csak forgás \|vyaw\|≤0.5; hátrálás ≤0.2 m/s ha minden STOP-ember elöl van és hátul ≥0.8 m szabad |
| ember < 2.0 m / TTC < 2 s | SLOW | vmax 0.1→0.4 lineárisan (TTC: 0.25) |
| ember < 3.5 m | CAUTION | vmax 0.6 |
| person-adat > 0.3 s / soha | SLOW | vmax 0.3 |
| LiDAR hiányzik/elavult (>0.5 s) | STOP | hátrálás sem |
| LiDAR < 0.5 m a mozgásirányban | STOP | hátrálás sem |
| gyerek (`height_m` vagy `z` < 1.3 m) / \|v\| > 2 m/s | — | zónák ×1.5 |
| forrás `pursuit`/`follow`/`follow_executor` | — | 2.5 m-en belüli emberhez közelítő lineáris parancs = 0 |

- Megnyújtás: mozgásirányban lévő ember effektív távolsága `d − 1.0 s·|v|·cos(α)`.
- TTC: konstans sebesség, relatív sebesség = ember − robot (parancs és, ha van, odom vx/vy), 0.8 m (×1.5) sugár.
- Monoton: `cmd_out` komponensenként soha nem nagyobb, előjelet nem vált.
- Hiszterézis: szigorítás azonnal, lazítás csak 0.5 s folyamatos enyhébb állapot után.
- A STOP + nulla kimenet → `mc_motion /stop`; STOP + forgás/hátrálás → `/move`.
- Háttér-tick (20 Hz): ha egy hívó 0.3 s-on belül küldött és azóta szigorodott a szint, a kisebb parancsot (vagy `/stop`-ot) újraküldi; nagyobbat soha. Hívó nélkül nem küld semmit (a `mc_motion` 0.5 s watchdogja állít meg).

## API

| Végpont | Leírás |
|---|---|
| `POST /move {vx,vy,vyaw,source}` | szűrés + továbbítás; mc_motion hibakódja átmegy (409 not armed, 502 unreachable) |
| `POST /stop {source}` | mindig továbbít `/stop` |
| `POST /estop` | átadja `mc_motion /estop`-nak |
| `GET /state` | utolsó döntés + bemenet-életkorok, események |
| `GET /health` | |

Redis: `mc.safety.state` 20 Hz → `{"t","level","vmax","nearest_person_m","reason","vyaw_max","ttc_s"}`.

## Env

| Változó | Default | |
|---|---|---|
| `SAFETY_PORT` | 9115 | |
| `MOTION_URL` | `http://127.0.0.1:9102` | mc_motion |
| `OMNI_URL` | `http://127.0.0.1:9114` | `/persons` HTTP fallback, ha nincs redis |
| `REDIS_HOST` | `redis` | |
| `SAFETY_LIDAR_SOURCE` | `robot_client` | `robot_client` (world→base a `get_pose()`-zal) \| `http` (`SAFETY_LIDAR_URL`, base frame pontok) \| `mock` \| `none` (=mindig STOP) |
| `SAFETY_LIDAR_MAX_AGE_S` | 0.5 | |
| `SAFETY_LIDAR_CONE_RAD`, `SAFETY_LIDAR_Z_MIN/MAX` | 0.6, −0.25, 1.5 | base frame |
| `SAFETY_FOLLOW_SOURCES` | `pursuit,follow,follow_executor` | |
| `SAFETY_<MEZŐ>` | `SafetyConfig` | pl. `SAFETY_STOP_M=1.0`, `SAFETY_CHILD_FROM_Z=0`, `SAFETY_Z_OFFSET_M=0.3` |

## Futtatás / teszt

```bash
cd mission_control/safety_guard
MOTION_URL=http://127.0.0.1:9102 python3 app.py --mock     # mock LiDAR-gyűrű 3 m-en
curl -s -XPOST localhost:9115/move -H 'content-type: application/json' -d '{"vx":0.3,"source":"curl"}'
curl -s localhost:9115/state
cd .. && python3 -m pytest tests/test_safety_guard.py -q
```

## Integrációs patch (a koordinátor alkalmazza — D agent nem módosíthatja)

Cél (SAF-4): minden mozgásforrás a `safety_guard /move`-ra küld, csak a guard beszél `mc_motion /move`-val.

**1. `follow_executor/executor.py`** — a health/odom marad `MOTION`-on, a move/stop a guardra megy:
```python
# a MOTION = ... sor után
MOVE_URL = os.environ.get("SAFETY_URL", MOTION)   # http://127.0.0.1:9115 élesben
# tick():
-            if _post(f"{MOTION}/move", {"vx": vx, "vy": 0.0, "vyaw": vyaw}) is None:
+            if _post(f"{MOVE_URL}/move", {"vx": vx, "vy": 0.0, "vyaw": vyaw, "source": "follow_executor"}) is None:
-            _post(f"{MOTION}/stop")          # mindkét helyen (tick + run hibaág)
+            _post(f"{MOVE_URL}/stop")
```
Deploy: `SAFETY_URL=http://127.0.0.1:9115` a follow_executor env-jébe.

**2. `navigation/app.py`** és **3. `mapping/app.py`** (explore) — most `self.robot.move(...)`-ot hívnak (core robot_client → élesben közvetlen SportClient, megkerüli az mc_motion-t). Helper mindkettőbe:
```python
SAFETY_URL = os.environ.get("SAFETY_URL", "")   # üres = régi viselkedés (robot.move)

def _safe_move(robot, vx, vy, vyaw, source):
    if not SAFETY_URL:
        robot.move(vx, vy, vyaw); return
    r = requests.post(f"{SAFETY_URL}/move", json={"vx": vx, "vy": vy, "vyaw": vyaw, "source": source}, timeout=0.3)
    r.raise_for_status()      # 409 not armed / 502 -> a meglévő hibakezelés ágára fusson

def _safe_stop(robot):
    if SAFETY_URL:
        try: requests.post(f"{SAFETY_URL}/stop", json={"source": PILLAR}, timeout=0.3)
        except Exception: pass
    robot.stop()
```
- `navigation/app.py:307`: `self.robot.move(vx, 0.0, vyaw)` → `_safe_move(self.robot, vx, 0.0, vyaw, "navigation")`
- `mapping/app.py:369`: `self.robot.move(vx, 0.0, vyaw)` → `_safe_move(self.robot, vx, 0.0, vyaw, "mapping")`
- a `robot.stop()` hívások → `_safe_stop(self.robot)`.
- A `robot.is_armed()` előfeltétel marad (a guard nem armol; az mc_motion saját armed-állapota a mérvadó, 409-cel jön vissza).

**4. `docker-compose.yml`** — új szolgáltatás:
```yaml
  safety_guard:
    build: {context: ., dockerfile: safety_guard/Dockerfile}
    environment:
      <<: *common-env
      ROBOT_CLIENT_MODE: remote
      CORE_URL: http://core:9101
      MOTION_URL: ${MOTION_URL:-http://host.docker.internal:9102}
      OMNI_URL: http://omni:9114
    ports: ["9115:9115"]
    depends_on: [redis, core]
```
+ `navigation`, `mapping`: `SAFETY_URL: http://safety_guard:9115`.

**5. `omni/pursuit.py` kimenete** → `POST SAFETY_URL/move {..., "source": "pursuit"}` (a "pursuit" forrás kapja a 2.5 m-es közelítés-tiltást).

> Megjegyzés: `mc_motion` default `MOTION_PORT=9102` ütközik a `mapping` 9102-es portjával (CONVENTIONS) — külön hoston fut, de compose-ban figyelni kell.
