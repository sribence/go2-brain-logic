# safety_guard — független sebesség-korlátozó (port 9115)

Az EGYETLEN folyamat, ami `mc_motion /move`-ot hív. Minden mozgásforrás
(navigation, mapping explore, follow_executor, omni pursuit) ide küld; a
`core.decide()` a friss ember-trackek (`mc.omni.persons`) és a LiDAR alapján
szűr, csak a kimenet megy tovább. **Nem armol, nem indít mozgást.**

| Fájl | Tartalom |
|---|---|
| `core.py` | tiszta `decide()` + LiDAR-segédek (I/O nélkül) |
| `app.py` | FastAPI, redis/HTTP bemenetek, forwarder, 20 Hz state |
| tesztek | `tests/test_safety_guard.py`, `tests/test_safety_guard_ext.py` (dead-man/sprint/akció/flotta), `tests/test_mc_motion_caps.py` (mc_motion új route-ok) |

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

## Dead-man, sprint, akciók, flotta (mission/CONTRACT.md 4, 9.4, 9.6)

**Dead-man** — `POST /deadman {client_id, release?}` ≥ 5 Hz (UI 10 Hz, gamepad-bridge `client_id:"gamepad"`). Élő, ha bármely kliens utolsó jele < `DEADMAN_TIMEOUT_S` (0.3 s). Max 32 kliens. A dead-man önmagában soha nem küld mozgást.

**Sprint** — `/move` opcionális `profile` (`stealth|precise|normal|sprint`; csak a `sprint` változtat bármit). Engedve (`vmax` `clear_vmax` 1.0 → `sprint_vmax` 1.5), CSAK ha mind igaz:

| Feltétel | |
|---|---|
| `source` ∈ `SAFETY_SPRINT_SOURCES` | default `mission,gamepad` |
| `profile == "sprint"` | |
| dead-man él | |
| akku ismert és > 30 % | core `/state` `battery.percent`, ≤ 5 s régi; link unhealthy / hiányzó = ismeretlen = tilt |
| LiDAR friss, person-adat friss | |
| szint `CLEAR` (végső ÉS nyers, hiszterézis nélkül) | a mozgásirányú nyújtás és TTC a sprint-sebességgel számol |
| folyosó üres | ember/robot a parancs irányában 0..8 m előre, ±1.5 m oldalt; hibás track = bent |

Különben a döntés bitre azonos a sprint nélkülivel. Sprint-engedéskor a továbbított body `{"vx","vy","vyaw","sprint":true}`; minden más body változatlanul `{"vx","vy","vyaw"}`.

**mc_motion oldal**: `MAX_VX=0.6` vágna. Sprinthez a robot-hoston: `MOTION_ALLOW_SPRINT=1` + `MAX_VX_SPRINT=1.5` (default = `MAX_VX`, azaz nincs változás). Az mc_motion a `sprint: true` (szigorúan bool) jelzőt csak `MOTION_ALLOW_SPRINT=1` mellett veszi figyelembe, és csak a `vx` korlátot emeli (`vy`, `vyaw` marad).

**Dead-man elengedés sprint közben** — a 20 Hz-es tick (vagy a következő sprint-`/move`) azonnal egy `/stop`-ot küld, és **latch**: további `profile:"sprint"` mozgás → `/stop`, amíg a dead-man újra nem él, vagy a forrás nem-sprint profilt küld (akkor normál plafon). Ha a sprint még el sem indult, a dead-man nélküli sprint-kérés egyszerűen normál plafont kap. Ugrás közbeni elengedés nem szakít meg (levegőben leállítani veszélyesebb); a következő ugrás tiltott.

**Akciók** — `POST /action/{name} {source?}` → engedély → `MOTION_URL/action/{name}`. Válasz `{decision: allow|deny, reason, ...}`; tiltás = **409**, mc_motion hibakód átmegy (409/502).

| Akció | Kapu |
|---|---|
| `hello, wave, greet, stretch, sit, stand, stand_up, lay_down, heart, balance` | nincs (helyben) |
| `dance, dance2` | friss person-adat, senki 2.0 m-en belül |
| `jump, front_jump, pounce, front_pounce` | dead-man + friss person-adat + senki (ember/robot) 3.5 m-en belül + szint `CLEAR` + friss LiDAR + előre ≥ 1.0 m szabad + robot áll (\|cmd\| < 0.05 legalább 0.5 s; nem-nulla utolsó parancs után az mc_motion 0.5 s watchdog-ablaka is beszámít) |
| minden más (flipek, `damp`, handstand, …) | tiltva (allowlist) |

**Flotta** (`SAFETY_FLEET_AWARE=1` default) — `mc.fleet.robots {t, robots:[{id,x,y,vx,vy}]}` (világ-frame) + saját póz a core `/state`-ből (10 Hz). A többi robot (≠ `ROBOT_ID`) person-szerű trackként megy a zóna-/TTC-logikába, sugárirányban `bubble − stop_m` = 0.7 m-rel közelebb téve → a STOP 1.5 m valós távolságnál (SLOW 2.7, CAUTION 4.2). Gyerek-/gyors-szorzó és a 2.5 m-es pursuit-szabály NEM vonatkozik rájuk; a sprint-folyosó és az ugrás-kapu igen. Elavult flotta-üzenet (> 1 s) vagy saját póz (> 0.5 s) → no-op.

## API

| Végpont | Leírás |
|---|---|
| `POST /move {vx,vy,vyaw,source,profile?}` | szűrés + továbbítás; mc_motion hibakódja átmegy (409 not armed, 502 unreachable) |
| `POST /deadman {client_id, release?}` | dead-man heartbeat |
| `POST /action/{name} {source?}` | akció-kapu → mc_motion; 409 tiltáskor |
| `POST /stop {source}` | mindig továbbít `/stop` |
| `POST /estop` | átadja `mc_motion /estop`-nak |
| `GET /state` | utolsó döntés + bemenet-életkorok, események, `deadman_alive`, `deadman_clients`, `battery_pct`, `sprint_active`, `sprint_latch`, `sprint_reason`, `still_for_s`, `fleet_robots_n`, `actions` |
| `GET /health` | |

Redis: `mc.safety.state` 20 Hz → `{"t","level","vmax","nearest_person_m","reason","vyaw_max","ttc_s","deadman_alive","sprint"}`.

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
| `SAFETY_<MEZŐ>` | `SafetyConfig` | pl. `SAFETY_STOP_M=1.0`, `SAFETY_CHILD_FROM_Z=0`, `SAFETY_Z_OFFSET_M=0.3`, `SAFETY_SPRINT_BATTERY_MIN_PCT=30`, `SAFETY_ACTION_PERSON_MIN_M=3.5` |
| `SPRINT_VMAX` (= `SAFETY_SPRINT_VMAX`) | 1.5 | |
| `DEADMAN_TIMEOUT_S` | 0.3 | |
| `SAFETY_SPRINT_SOURCES` | `mission,gamepad` | |
| `CORE_URL` | `http://127.0.0.1:9101` | akku + világ-póz (`/state`) |
| `SAFETY_BATTERY_MAX_AGE_S`, `SAFETY_CORE_POLL_HZ` | 5.0, 10 | |
| `SAFETY_MOTION_WATCHDOG_S` | 0.5 | = mc_motion `COMMAND_TIMEOUT_S` (álló-állapot számításhoz) |
| `SAFETY_FLEET_AWARE`, `ROBOT_ID`, `SAFETY_FLEET_BUBBLE_M` | 1, `go2`, 1.5 | |
| `SAFETY_FLEET_MAX_AGE_S`, `SAFETY_POSE_MAX_AGE_S` | 1.0, 0.5 | |

## mc_motion bővítés (`go2-brain-logic/mc_motion/motion.py`, csak új route + flag)

| Flag / route | Default | |
|---|---|---|
| `MOTION_ALLOW_SPRINT` | 0 | 1 = `/move` `sprint:true` → `MAX_VX_SPRINT` |
| `MAX_VX_SPRINT` | = `MAX_VX` | |
| `MOTION_ALLOW_GAIT` | 0 | 0 → `/gait`, `/speed_level` = 403 |
| `GET /capabilities` | — | SportClient metódusok (`hasattr` + arity), limitek, flagek, `gait_modes` |
| `POST /gait {mode, enable?, value?}` | 403 | `free_walk, classic_walk, trot_run, static_walk, economic, switch_gait(value 0-4)`; armed + álló robot kell (409), hiányzó metódus 501 |
| `POST /speed_level {level}` | 403 | `-1|0|1`; armed + álló |

SDK-metódusok verziónként (`unitree_sdk2py/go2/sport/sport_client.py`, github.com/unitreerobotics/unitree_sdk2_python):
- ≤ `3d497a6` (2025-01): `SwitchGait(int)`, `EconomicGait(bool)`, `FreeWalk(bool)`, `SpeedLevel(int)`, `GetSpeedLevel`, `WiggleHips`, `FrontJump()`, `FrontPounce()`, `Dance1/2()` …
- ≥ `026978e` (2025-07): `SwitchGait`/`EconomicGait`/`WiggleHips` törölve; új `FreeWalk()` (arg nélkül!), `ClassicWalk(bool)`, `TrotRun()`, `StaticWalk()`, `FreeBound/FreeJump/FreeAvoid/WalkUpright/CrossStep/HandStand(bool)`, `AutoRecoverySet/Get`, `SwitchAvoidMode`.
- Ezért a `/gait` a futó kliens szignatúráját nézi (`inspect`): 0 arg → `fn()`, 1 arg → `fn(enable)`. A `SpeedLevel` szintek (-1/0/1) a Unitree doksiból — ezen a firmware-en NEM ellenőrzött.

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
