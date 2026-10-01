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
Koordinátor: `docker-compose.yml`, `CONVENTIONS.md`, ez a fájl, commit/push. **Senki nem commitol.**
Közös: Python 3.8 szintaxis (`from __future__ import annotations`), numpy/fastapi/requests, tesztek hálózat/redis/hardver nélkül, a teljes `python3 -m pytest tests -q` zöld marad.
