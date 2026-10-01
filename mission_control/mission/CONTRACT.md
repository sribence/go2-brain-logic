# Mission — 3D parancsolás interfész-szerződés (KÖTELEZŐ minden agentnek)

Ötletek és sorrend: `NERO_GO2/docs/22-3d-parancsolas-otletek.md`. Alap: `omni/CONTRACTS.md`.
Branch: `feature/omnivision-360`. Eltérés esetén ide írd: `> ELTÉRÉS:`.

## 0. Elvek
- **Minden mozgás**: `POST {SAFETY_URL}/move {vx,vy,vyaw,source}` (a `safety_guard` :9115). Akció (ugrás, hello): `POST {SAFETY_URL}/action/{name}` (új, M3 írja).
- Világ-frame: a `mapping` grid-frame (x, y, yaw a `core` `/state` pózából). A UI világ-koordinátában küld.
- Sebességet a misszió soha nem ad nyers m/s-ban, csak profil-nevet: `stealth | precise | normal | sprint`.
- `sprint` és `jump_to` csak élő **dead-man** mellett futhat (a `safety_guard` ellenőrzi; a UI tartja).
- LLM-generált misszió **mindig** `pending_approval` állapotba kerül.

## 1. Új pillér: `mission` :9116 (env `MISSION_PORT`)
| Végpont | Leírás |
|---|---|
| `POST /missions` | misszió-JSON → validálás → `{mission_id, state, plan}` (`plan` = előnézet: globális pálya, sebességprofil, ugrás-landolások, ETA, figyelmeztetések). `?execute=0` csak tervez. |
| `POST /missions/{id}/approve` | `pending_approval` → `queued` |
| `POST /missions/{id}/pause` / `resume` / `abort` | állapotgép |
| `GET /missions/{id}`, `GET /missions` | állapot |
| `POST /plan` | egyetlen `goto`/`follow_path`/`jump_to` lépés tervezése, előnézethez (nem hajt végre) |
| `GET/POST/DELETE /zones`, `GET/POST/DELETE /labels`, `GET/POST/DELETE /rules`, `GET/POST /home` | M5 |
| `POST /nl` `{text}` | M6: természetes nyelv → misszió `pending_approval`-ban |
| `GET /health` | |
| `WS /ws/mission` | állapot-push (mint `mc.mission.state`) |

Redis: `mc.mission.state` `{t, mission_id, state, step_index, step_op, progress, eta_s, error}`; `mc.mission.event` `{t, mission_id, kind, ...}`; `mc.mission.alert` (rules → UI / push).

## 2. Misszió-séma (`mission/schema.py`, M1 írja — mindenki ezt importálja)
```json
{"mission_id": "opcionális", "name": "...", "source": "ui|gamepad|nl|rule|api",
 "steps": [ <step>, ... ], "on_fail": "stop|skip|return_home"}
```
Lépések (`op`):
| op | mezők | megjegyzés |
|---|---|---|
| `goto` | `x, y, yaw?, speed="normal", tol_m=0.3` | globális terv + DWA-követés |
| `follow_path` | `points:[[x,y],...], speed, smooth=true` | szabadkézi rajz → spline |
| `jump_to` | `x, y, max_jumps=6` | `FrontJump`-sorozat; landolási pontok ellenőrizve |
| `look_at` | `x, y, z=0` | helyben fordul |
| `scan` | `deg=360, speed_dps=30` | helyben forgás |
| `action` | `name` (`hello`, `stretch`, `sit`, `stand`, `dance`, …) | `mc_motion` akció a guardon át |
| `wait` | `s` | |
| `say` | `text` | `AUDIO_URL /audio/speak` |
| `shadow` | `gid, dist_m=3.0 (min 2.5), timeout_s` | `omni/pursuit.py` |
| `watch` | `gid, timeout_s` | helyben, felé fordulva |
| `escort` | `gid, side="right", dist_m=2.5, timeout_s` | mellette halad |
| `patrol` | `points` vagy `zone`, `loops=1`, `shuffle=false`, `speed` | |
| `explore` | `zone?` | `mapping /explore` |
| `goto_label` | `label`, `speed` | M5 szemantikus címke → `goto` |
| `return_home` | `speed` | M5 home |
`schema.validate(mission_dict) -> (Mission, list[str] errors)`; `Mission.to_dict()`.

## 3. Tervező (`mission/planner.py`, M2 írja; tiszta függvények)
- `plan_path(grid_meta, floor, walls, start_xy, goal_xy, social=None, no_go_mask=None, cfg) -> Path(points world [[x,y]], length_m, ok, reason)` — a `navigation/astar.py` újrahasznosítva (import), social költséggel, simítással.
- `smooth_path(points, ds=0.1) -> points` (Catmull-Rom/B-spline + ütközés-ellenőrzés).
- `speed_profile(path, profile_name, cfg, persons_world=None) -> list[float] v per point` (görbület- és gyorsulás-korlát, a pálya folyosójában lévő ember → lassít; `sprint` csak ha a folyosó ±1.5 m üres).
- `PROFILES = {stealth:{vmax:0.25,amax:0.3}, precise:{0.3,0.4}, normal:{0.6,0.6}, sprint:{vmax: SPRINT_VMAX (default 1.5), amax:1.0}}`.
- `plan_jumps(grid..., start, goal, jump_len_m=JUMP_LEN_M (default 0.6, MÉRENDŐ), clearance) -> JumpPlan(landings [[x,y]], ok, reason)`.
- `eta(path, v) -> s`. `follow_controller(pose, path, v_profile, cfg) -> (vx, vyaw, idx, done)` pure-pursuit, DWA-val kombinálható (`navigation/nav_local.py`).

## 4. Safety-bővítés (M3 írja: `safety_guard/*`, `mc_motion/motion.py` csak új route-ok)
- `POST /deadman {client_id}` heartbeat (≥ 5 Hz); `deadman_alive` ha < 0.3 s. `GET /state` tartalmazza.
- `sprint`: a guard `cmd.source=="mission"` és `profile=="sprint"` (új opcionális mező a `/move`-ban) esetén `sprint_vmax`-ig engedi, CSAK ha dead-man él, a szint `CLEAR`, az akku > 30% és a folyosó üres; különben normál plafon.
- `POST /action/{name}` → engedélyezés (`jump`/`pounce`: dead-man + nincs ember 3.5 m-en belül + szint `CLEAR`) → `MOTION_URL/action/{name}`.
- `mc_motion`: `GET /capabilities` (melyik SportClient-metódus létezik: `SpeedLevel`, `SwitchGait`, `FrontJump`, …), `POST /gait {mode}` és `POST /speed_level {level}` — feature-flag mögött (`MOTION_ALLOW_GAIT=0` default), a meglévő viselkedést nem változtatja.

## 5. Szemantika, zónák, szabályok (M5: `mission/world.py`, `mission/rules.py`, `mission/scheduler.py`)
- Tárolás: JSON fájl (`mission/data/world.json`), atomikus írás.
- `Zone{id, name, kind: no_go|watch|patrol, polygon:[[x,y],...], active_hours?:"22:00-06:00"}`; `Label{name, x, y, yaw?}`; `home{x,y,yaw}`.
- `no_go_mask(grid_meta) -> bool[h*w]` a tervezőnek.
- Szabály: `{id, when:{event:"person_in_zone", zone, modality?:"thermal", hours?}, then:[steps...] , cooldown_s, alert:true}` → misszió `source:"rule"` (automatikusan fut, mert nem LLM; de `shadow`/`watch`/`say`/`goto`/`goto_label` lépések engedettek, `sprint`/`jump_to` TILTOTT szabályból).
- `scheduler`: időzített járőr (`cron`-szerű `"*/30 22-06"`).

## 6. NL + gesztus (M6: `mission/nl.py`, `omni/gesture.py`)
- `nl.py`: Claude API (Anthropic SDK, lazy import, `ANTHROPIC_API_KEY` env), tool-use a misszió-sémával; a promptban a címkék/zónák listája; kimenet validálva; mindig `pending_approval`. Kulcs nélkül: 503 érthető hibával.
- `gesture.py`: tiszta osztályozó YOLO-pose kulcspontokon (`wave`, `stop_palm`, `point`), időbeli simítással; `mc.omni.gesture` esemény `{gid, gesture, t}`; a `rules` erre is tud triggerelni (`wave` → `goto` 3 m-re a személyhez).

## 7. UI (M4: `digital-twin/static/omni/mission*.js`, `omni.js` és `omni.html` minimális bővítés, `server.py` új route-ok)
- Cél kattintás + húzás = yaw; `Shift`/`Alt`/`Ctrl` profilok; `POST /plan` → útvonal-vonal sebesség-színezéssel + szellem-kutya előnézet + ETA; GO / ABORT.
- Szabadkézi rajz (`D` mód) → `follow_path`; waypoint-lánc szerkesztő szakaszonkénti akcióval; ugrás-mód (`J`) landolási gyűrűkkel.
- Ember jobb klikk → árnyék / figyeld / kísérd. Zóna-rajzolás, címke-elhelyezés, home beállítása.
- **DEAD-MAN gomb** (nyomva tartva vagy `Shift`+egér, gamepad RB) → `POST /deadman` 10 Hz. Felengedés = azonnal leáll.
- Gamepad (Gamepad API), NL parancssor (szöveg → `/nl` → előnézet → jóváhagyás).
- Demó mód (`?demo=1`) maradjon működő; a mission backend hiánya ne dobjon hibát.

## 8. Fájl-tulajdon (egy fájlt CSAK a tulajdonos ír)
| Agent | Fájlok |
|---|---|
| M1 executor | `mission/schema.py`, `mission/executor.py`, `mission/app.py`, `mission/Dockerfile`, `mission/requirements.txt`, `mission/README.md`, `tests/test_mission_schema.py`, `tests/test_mission_executor.py` |
| M2 planner | `mission/planner.py`, `tests/test_mission_planner.py` |
| M3 safety | `safety_guard/*`, `mc_motion/motion.py` (csak új route-ok + feature-flag), `tests/test_safety_guard*.py` |
| M4 UI | `digital-twin/static/omni/mission*.js`, `digital-twin/static/omni/gamepad.js`, `omni.js`, `omni.html`, `digital-twin/server.py` (új route-ok), `tests/test_digital_twin_omni*.py` |
| M5 world | `mission/world.py`, `mission/rules.py`, `mission/scheduler.py`, `mission/data/`, `tests/test_mission_world.py`, `tests/test_mission_rules.py` |
| M6 NL+gesture | `mission/nl.py`, `omni/gesture.py`, `tests/test_mission_nl.py`, `tests/test_omni_gesture.py` |
| M7 mobil UI | `digital-twin/static/omni_mobile.html`, `digital-twin/static/omni/mobile/*`, `digital-twin/static/manifest.webmanifest`, `digital-twin/static/sw.js`, `server.py`: `/m` route |
| R1 recorder | `omni/recorder.py`, `omni/capture.py` (MJPEG passthrough + felbontás-váltás, A agent API-ját megtartva), `omni/app.py` (csak /record és /incidents route-ok + hook), `tests/test_omni_recorder.py` |
| G1 gamepad | `go2-hardware-bridge/gamepad_bridge/*` |
| F1 flotta | `fleet/*`, `digital-twin/static/fleet.html`, `digital-twin/static/fleet/*`, `server.py`: `/fleet` route, `tests/test_fleet*.py` |
Koordinátor: `docker-compose.yml`, `CONVENTIONS.md`, ez a fájl, commit/push. **Senki nem commitol.**
Közös: Python 3.8 szintaxis (`from __future__ import annotations`), numpy/fastapi/requests, tesztek hálózat/redis/hardver nélkül, a teljes `python3 -m pytest tests -q` zöld marad.

## 9. Kiegészítések (2026-10-01, 2. kör — kötelező)

### 9.1 Zóna-érték (ipari robot `fine` / `zXX` mintára)
Minden mozgó lépés (`goto`, `follow_path` pontjai, `patrol`, `goto_label`, `return_home`) kap egy `zone` mezőt:
- `"fine"` = pontos megállás: pozíció ≤ 0.05 m, yaw ≤ 3°, a robot **megáll és beáll** (lassú végső közelítés, `precise` profil az utolsó 0.5 m-en). Fotó, mérés, ajtó előtt.
- `"z10"`, `"z30"`, `"z60"`, `"z100"` = átrepülés: a waypointot ennyi cm-en belül „elérettnek” tekinti, nem lassít le, a pályát a zónán belül lekerekíti (blend). Default: `"z30"`; a misszió utolsó pontja default `"fine"`.
- Tervező (M2): `smooth_path(..., zones=[...])` a sarkokat a zóna sugarán belül kerekíti; `speed_profile` a `fine` pontoknál nullára lassít.
- Executor (M1): a `fine` pontnál megáll, megvárja a stabilizálódást (0.5 s), és csak utána lép tovább.

### 9.2 Új lépések
| op | mezők | jelentés |
|---|---|---|
| `capture` | `x?, y?, yaw?, cams:["rgb_front",...]\|"all", mode:"max"\|"normal", label?` | (ha van póz: `goto` `zone:"fine"`-nal) → max felbontású fotó a megadott kamerákból + hőkép → `recorder` (9.3); a misszió eredményébe a fájl-URL-ek |
| `request_passage` | `text`, `repeat_s=30`, `zone_polygon?` vagy `ahead_m=1.5`, `clear_for_s=2.0`, `timeout_s=600`, `on_timeout:"alert"\|"return"\|"abort"` | megáll, `say(text)` ismételve; vár, amíg az előtte lévő folyosó (vagy az ajtó-poligon) a LiDAR szerint **szabad** `clear_for_s` ideig (= kinyitották az ajtót), VAGY az operátor `POST /missions/{id}/continue`-t küld, VAGY gesztus `wave`; utána megy tovább. Közben `mc.mission.alert` „várakozik: …” |
| `wait_for` | `event: "path_clear"\|"operator"\|"gesture"\|"person_gone"`, `timeout_s` | általános várakozás |
| `record` | `on:true/false`, `mode:"evidence"` | a bizonyíték-rögzítés kézi ki/be kapcsolása |
`POST /missions/{id}/continue` — új végpont (operátor „továbbmehetsz”).

### 9.3 Bizonyíték-rögzítő (`omni/recorder.py`, R1)
- Normál működés: **nincs tartós tárolás**, csak RAM-gyűrű: kameránként az utolsó `PREBUFFER_S=10` s **nyers MJPEG-bájtjai** (újrakódolás nélkül, olcsó) + hőképek + person-trackek.
- Trigger: `watch`-zóna szabály, gyanús ember (éjjel, hőkamera), operátor gomb, `record` lépés → **EVIDENCE mód**: a pre-buffer mentése + a kamerák átkapcsolása max felbontásra (UVC-nél pl. 1920×1080 vagy amit a kamera tud, MJPEG passthrough) + `best_shot`: emberenként a legélesebb, legnagyobb kivágás (Laplace-élesség × bbox-méret) teljes felbontásban, + hő-kivágás. `post_s=30` s az utolsó észlelés után, aztán vissza normálra.
- Incidens-mappa: `incidents/<ts>_<robot>/` → `frames/<cam>/<ts>.jpg`, `best/<gid>_<cam>.jpg`, `thermal/*.npy+png`, `meta.jsonl` (póz, trackek, gps?), `manifest.json` SHA-256 hash-lánccal (bizonyítási integritás). Tárhely-kvóta + rotáció (`MAX_INCIDENT_GB`).
- API (omni :9114): `POST /record/start {reason}`, `POST /record/stop`, `GET /record/status`, `GET /incidents`, `GET /incidents/{id}` (+ fájlok). Redis `mc.omni.record` `{state, incident_id, reason}`. A `blackbox` pillér `mc.core.anomaly`-ként is kapja.

### 9.4 Gamepad
- **Böngésző** (M4/M7): Gamepad API — a telefonhoz/laptophoz párosított bármilyen BT-pad (Xbox, PS4/5, 8BitDo, generic), `standard` mapping + kézi újratérképezés.
- **Robot-oldali** (G1, `go2-hardware-bridge/gamepad_bridge/`): a Jetsonhoz közvetlenül párosított BT-pad (evdev), telefon nélkül is. A bal kar → `safety_guard /move` `source:"gamepad"` (normál plafon), RB = dead-man (`/deadman`), B = stop, Y = stand/sit, A = akció, Start = misszió pause/resume. Ha a pad kapcsolata megszakad → `/stop`. Profilok: Xbox, DualShock/DualSense, 8BitDo, generic (konfig YAML-ban).

### 9.5 Mobil UI (M7)
Külön, érintés-first oldal: `digital-twin/static/omni_mobile.html` + `static/omni/mobile/*.js`, route `GET /m`. Telefon álló és fekvő módban, PWA (manifest + service worker a statikus fájlokra), nagy gombok, egy ujjas célkijelölés, két ujjas forgatás, hosszú nyomás = kontextus-menü (ember: árnyék / figyeld; padló: menj ide / fotó itt / ajtón átkérés), alsó lap (bottom sheet) a misszió-lépésekhez, haptikus visszajelzés (`navigator.vibrate`), dead-man = nyomva tartott nagy gomb, E-STOP mindig látható. A meglévő modulokat (`net.js`, `frames.js`, `persons.js`, `pointcloud.js`, `bowl.js`) használja, könnyített renderrel (alacsony pixel ratio, kevesebb pont).

### 9.6 Flotta (több robot) — `fleet` pillér (9111) bővítése (F1)
- Robot-regiszter: `fleet/robots.yaml` + `POST /robots` (id, név, típus `go2|xavier|...`, URL-ek: core, omni, safety, mission, digital-twin; szín). Heartbeat / állapot-aggregálás (póz, akku, safety-szint, misszió-állapot, kamerák).
- Közös világ: közös zónák/címkék (a fleet tárolja, robotonként szinkronizálja a `mission /zones /labels`-be), közös térkép-referencia (`map_frame` + robotonkénti `T_map_robot` offset, kézi illesztés).
- Feladat-kiosztás: `POST /fleet/missions {mission, robot_id?|"auto"}` → auto = legközelebbi szabad robot (útvonal-hossz a robot tervezőjével, akku-súlyozva); `POST /fleet/alert` → a legközelebbi robot `watch`/`shadow`-ot kap.
- Ütközés-elkerülés robotok közt: minden robot a többit „mozgó akadályként” kapja (`mc.fleet.robots` → `safety_guard` opcionálisan személyként kezeli, 1.5 m-es buborékkal).
- UI: `GET /fleet` oldal (digital-twin route, F1 írja: `static/fleet.html` + `static/fleet/*.js`) — felülnézeti közös térkép az összes robottal, misszió-kiosztás drag&drop-pal, robotonkénti kamera-csempék, riasztás-lista, robotra kattintva megnyílik az adott robot OmniView-ja.
