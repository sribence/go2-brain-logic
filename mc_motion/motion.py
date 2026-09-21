"""mc_motion -- the one service on the dock that may command the robot.

Deliberately separate from mc_sensor_hub: the hub stays incapable of movement
no matter what happens here, and this container can be removed with a single
`docker rm` to return the dock to observe-only.

Every call pattern below is copied from docker/web_dashboard/app.py, which is
the only code in this project proven against the live robot:
    ChannelFactoryInitialize(0, iface) -> SportClient().Init()
    sport_client.Move(vx, vy, vyaw)
    RecoveryStand / StandDown / Sit / Hello / Heart

Safety, all enforced here rather than in the caller:
  - disarmed at startup; arming needs an explicit request
  - velocity clamped; NaN rejected
  - a watchdog stops the robot if commands stop arriving (a dropped WiFi link
    or a closed browser tab must not leave the last velocity latched)
  - /estop is unauthenticated and always answers: it can only make the robot
    safer, so nothing may stand between an operator and stopping it
  - the Unitree remote wins: any stick deflection or button press on
    rt/wirelesscontroller while armed disarms and stops (REMOTE_OVERRIDE=1).
    Whether the firmware itself lets the remote beat SDK Move() is not
    verified, so this makes it true for every caller of this service
  - /odom republishes rt/sportmodestate (x, y, yaw) for ego-motion
    compensation in the person follower
"""
import math
import os
import threading
import time
from typing import Optional

from flask import Flask, jsonify, request

IFACE = os.environ.get("DDS_NETWORK_INTERFACE", "eth10")
PORT = int(os.environ.get("MOTION_PORT", "9102"))

MAX_VX = float(os.environ.get("MAX_VX", "0.6"))
MAX_VY = float(os.environ.get("MAX_VY", "0.3"))
MAX_VYAW = float(os.environ.get("MAX_VYAW", "0.8"))
COMMAND_TIMEOUT_S = float(os.environ.get("COMMAND_TIMEOUT_S", "0.5"))
ARM_TIMEOUT_S = float(os.environ.get("ARM_TIMEOUT_S", "300"))
REMOTE_OVERRIDE = os.environ.get("REMOTE_OVERRIDE", "1") == "1"
REMOTE_DEADZONE = float(os.environ.get("REMOTE_DEADZONE", "0.15"))
# 1 = refuse /arm unless the remote was heard within REMOTE_MAX_AGE_S. Only
# useful if the remote publishes while idle; check /health "remote" first.
REQUIRE_REMOTE = os.environ.get("REQUIRE_REMOTE", "0") == "1"
REMOTE_MAX_AGE_S = float(os.environ.get("REMOTE_MAX_AGE_S", "1.0"))

app = Flask(__name__)

_lock = threading.Lock()
_sport = None
_sdk_error = "nem indult el"
_vui = None
_vui_error = "nem indult el"
_led_switch = None       # 0/1, last known (synced from hardware at init)
_led_brightness = None   # 0-10, last known (synced from hardware at init)
_armed = False
_armed_t = 0.0
_last_cmd_t = 0.0
_last_cmd = (0.0, 0.0, 0.0)
_watchdog_trips = 0
_events = []
_odom = None            # (x, y, yaw, t) from rt/sportmodestate
_remote_t = 0.0         # last rt/wirelesscontroller message
_remote_msgs = 0
_remote_overrides = 0
_subs = []              # keep DDS subscribers alive


def _log(level, msg, **extra):
    rec = {"t": time.time(), "level": level, "msg": msg, **extra}
    _events.append(rec)
    del _events[:-200]
    print(f"[mc_motion] {level}: {msg}", flush=True)


def _init_sdk():
    global _sport, _sdk_error, _vui, _vui_error, _led_switch, _led_brightness
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, IFACE)
    except Exception as exc:
        with _lock:
            _sdk_error = str(exc)[:200]
            _vui_error = str(exc)[:200]
        _log("error", f"ChannelFactory nem indult: {exc}")
        return

    try:
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        c = SportClient()
        c.SetTimeout(3.0)
        c.Init()
        with _lock:
            _sport = c
            _sdk_error = None
        _log("info", f"SportClient kesz ({IFACE})")
    except Exception as exc:
        with _lock:
            _sdk_error = str(exc)[:200]
        _log("error", f"SportClient nem indult: {exc}")

    try:
        # go2/vui, NOT a2/audio -- the a2 AudioClient's "voice" DDS service
        # does not answer on this unit (RPC_ERR_CLIENT_SEND every call,
        # confirmed on GetVolume() too, not LED-specific). go2/vui's "vui"
        # service does answer: it is the robot's real status-LED control,
        # on/off + brightness only, no RGB (vui_api.py has no color call).
        from unitree_sdk2py.go2.vui.vui_client import VuiClient

        vc = VuiClient()
        vc.SetTimeout(3.0)
        vc.Init()
        code_sw, sw = vc.GetSwitch()
        code_br, br = vc.GetBrightness()
        with _lock:
            _vui = vc
            _vui_error = None
            _led_switch = sw if code_sw == 0 else None
            _led_brightness = br if code_br == 0 else None
        _log("info", f"VuiClient (LED) kesz ({IFACE}), switch={sw} brightness={br}")
    except Exception as exc:
        with _lock:
            _vui_error = str(exc)[:200]
        _log("error", f"VuiClient nem indult: {exc}")

    _init_subscribers()


def _init_subscribers():
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_

        a = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        a.Init(_on_sport_state, 10)
        b = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        b.Init(_on_remote, 10)
        _subs.extend([a, b])
        _log("info", "DDS feliratkozas: sportmodestate + wirelesscontroller")
    except Exception as exc:
        _log("error", f"DDS feliratkozas sikertelen: {exc}")


def _on_sport_state(msg):
    global _odom
    _odom = (float(msg.position[0]), float(msg.position[1]), float(msg.imu_state.rpy[2]), time.time())


def _on_remote(msg):
    """Runs on the DDS thread. Any operator input on the remote while armed
    disarms: the person holding the remote must always win."""
    global _remote_t, _remote_msgs, _armed, _last_cmd_t, _remote_overrides
    _remote_t = time.time()
    _remote_msgs += 1
    if not REMOTE_OVERRIDE:
        return
    axes = (msg.lx, msg.ly, msg.rx, msg.ry)
    active = msg.keys != 0 or any(abs(a) > REMOTE_DEADZONE for a in axes)
    if not active:
        return
    with _lock:
        was_armed = _armed
        _armed = False
        _last_cmd_t = 0.0
        if was_armed:
            _remote_overrides += 1
    if was_armed:
        _stop_now("remote override")
        _log("error", f"TAVIRANYITO felulirta: lezarva (keys={msg.keys}, "
                      f"axes={[round(a, 2) for a in axes]})")


def _remote_age():
    return None if not _remote_t else round(time.time() - _remote_t, 2)


def _clamp(v, limit):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(-limit, min(limit, v))


def _stop_now(reason):
    """Best effort zero-velocity. Never raises: this runs on the failure path."""
    try:
        if _sport is not None:
            _sport.Move(0.0, 0.0, 0.0)
    except Exception as exc:
        _log("error", f"stop sikertelen ({reason}): {exc}")


def _watchdog():
    """The robot keeps walking on the last velocity until told otherwise, so a
    caller that stops sending (crash, closed tab, lost WiFi) must time out."""
    global _last_cmd_t, _armed, _watchdog_trips
    while True:
        time.sleep(0.1)
        now = time.time()
        with _lock:
            last, armed, armed_t = _last_cmd_t, _armed, _armed_t
        if last and (now - last) > COMMAND_TIMEOUT_S:
            with _lock:
                if _last_cmd_t != last:
                    continue
                _last_cmd_t = 0.0
                _watchdog_trips += 1
            _stop_now("watchdog")
            _log("warn", f"watchdog: {COMMAND_TIMEOUT_S}s-en belul nem jott parancs, megallitva")
        # An arm left on by a forgotten browser tab expires on its own.
        if armed and ARM_TIMEOUT_S and (now - armed_t) > ARM_TIMEOUT_S:
            with _lock:
                _armed = False
            _stop_now("arm timeout")
            _log("warn", f"elesites lejart ({ARM_TIMEOUT_S}s tetlenseg), lezarva")


threading.Thread(target=_init_sdk, daemon=True, name="motion-sdk").start()
threading.Thread(target=_watchdog, daemon=True, name="motion-watchdog").start()


# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    with _lock:
        return jsonify({"ok": True, "pillar": "mc_motion", "sdk_ready": _sport is not None,
                        "sdk_error": _sdk_error,
                        "led_ready": _vui is not None, "led_error": _vui_error,
                        "led": {"switch": _led_switch, "brightness": _led_brightness},
                        "armed": _armed,
                        "limits": {"max_vx": MAX_VX, "max_vy": MAX_VY, "max_vyaw": MAX_VYAW},
                        "command_timeout_s": COMMAND_TIMEOUT_S,
                        # The console polls /health, so the arm timeout has to
                        # be visible here: an operator meeting the expiry as a
                        # bare 409 mid-drive learns nothing from it.
                        "arm_timeout_s": ARM_TIMEOUT_S,
                        "armed_for_s": round(time.time() - _armed_t, 1) if _armed else None,
                        "watchdog_trips": _watchdog_trips,
                        "remote": {"override": REMOTE_OVERRIDE, "required": REQUIRE_REMOTE,
                                   "msgs": _remote_msgs, "age_s": _remote_age(),
                                   "overrides": _remote_overrides},
                        "odom_age_s": None if _odom is None else round(time.time() - _odom[3], 2)})


@app.route("/odom")
def odom():
    o = _odom
    if o is None:
        return jsonify({"error": "nincs sportmodestate"}), 503
    return jsonify({"x": o[0], "y": o[1], "yaw": o[2], "t": o[3], "age_s": round(time.time() - o[3], 3)})


@app.route("/status")
def status():
    with _lock:
        return jsonify({"armed": _armed, "sdk_ready": _sport is not None,
                        "sdk_error": _sdk_error, "last_cmd": _last_cmd,
                        "armed_for_s": round(time.time() - _armed_t, 1) if _armed else None,
                        "watchdog_trips": _watchdog_trips,
                        "events": _events[-20:][::-1]})


@app.route("/arm", methods=["POST"])
def arm():
    global _armed, _armed_t
    body = request.get_json(silent=True) or {}
    want = bool(body.get("armed", True))
    with _lock:
        if want and _sport is None:
            return jsonify({"error": f"SportClient nem elerheto: {_sdk_error}"}), 503
        if want and REQUIRE_REMOTE and (not _remote_t or time.time() - _remote_t > REMOTE_MAX_AGE_S):
            return jsonify({"error": "a taviranyito nem kuld adatot -- kapcsold be"}), 409
        _armed = want
        _armed_t = time.time()
    if not want:
        _stop_now("disarm")
    _log("warn" if want else "info", "ELESITVE" if want else "lezarva")
    return jsonify({"armed": want})


@app.route("/move", methods=["POST"])
def move():
    global _last_cmd_t, _last_cmd
    body = request.get_json(silent=True) or {}
    with _lock:
        if _sport is None:
            return jsonify({"error": f"SportClient nem elerheto: {_sdk_error}"}), 503
        if not _armed:
            return jsonify({"error": "a robot nincs elesitve"}), 409
        vx = _clamp(body.get("vx"), MAX_VX)
        vy = _clamp(body.get("vy"), MAX_VY)
        vyaw = _clamp(body.get("vyaw"), MAX_VYAW)
        sport = _sport
    try:
        sport.Move(vx, vy, vyaw)
    except Exception as exc:
        _log("error", f"Move hiba: {exc}")
        return jsonify({"error": str(exc)}), 500
    with _lock:
        _last_cmd_t = time.time()
        _last_cmd = (vx, vy, vyaw)
        _armed_t = time.time()
    return jsonify({"ok": True, "applied": {"vx": vx, "vy": vy, "vyaw": vyaw}})


@app.route("/stop", methods=["POST"])
def stop():
    global _last_cmd_t
    with _lock:
        _last_cmd_t = 0.0
    _stop_now("stop")
    return jsonify({"ok": True})


@app.route("/estop", methods=["POST"])
def estop():
    """Unauthenticated on purpose: it can only ever stop the robot."""
    global _armed, _last_cmd_t
    with _lock:
        _armed = False
        _last_cmd_t = 0.0
    _stop_now("estop")
    _log("error", "VESZLEALLITAS")
    return jsonify({"estop": True, "armed": False})


ACTIONS = ("stand_up", "stand", "lay_down", "laydown", "sit", "damp", "balance", "wave", "heart",
           "front_flip", "back_flip", "left_flip", "stretch", "dance",
           "front_jump", "front_pounce")

_obstacle_avoid_enabled = True


@app.route("/obstacle_avoid", methods=["GET", "POST"])
def obstacle_avoid():
    global _obstacle_avoid_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        enable = bool(data.get("enable", True))
        with _lock:
            # Try forwarding to webrtc_bridge :5001 if available
            try:
                requests.post("http://127.0.0.1:5001/obstacle_avoid", json={"enable": enable}, timeout=1.0)
            except Exception:
                pass
            _obstacle_avoid_enabled = enable
        _log("info", f"Akadalykerules atallitva: {enable}")
        return jsonify({"ok": True, "obstacle_avoid": _obstacle_avoid_enabled})
    return jsonify({"obstacle_avoid": _obstacle_avoid_enabled})


# go2/vui's real, live service: on/off + brightness only. No RGB/color
# call exists in vui_api.py -- confirmed against the live robot, see
# _init_sdk's comment. These presets are switch+brightness combos, not colors.
LED_PRESETS = {
    "off": {"switch": 0},
    "on": {"switch": 1, "brightness": 10},
    "dim": {"switch": 1, "brightness": 3},
    "bright": {"switch": 1, "brightness": 10},
}


def _set_led(switch: Optional[int] = None, brightness: Optional[int] = None):
    """Real hardware call: go2.vui.VuiClient.SetSwitch / SetBrightness (the
    robot's status LED). Never gated on arming -- a light is not a
    safety-relevant motion. Only the fields passed are changed."""
    global _led_switch, _led_brightness
    with _lock:
        if _vui is None:
            return f"VuiClient nem elerheto: {_vui_error}"
        vui = _vui

    if switch is not None:
        code = vui.SetSwitch(switch)
        if code != 0:
            _log("error", f"LED switch beallitas sikertelen: kod={code}")
            return f"SetSwitch hibakod: {code}"
        with _lock:
            _led_switch = switch

    if brightness is not None:
        code = vui.SetBrightness(brightness)
        if code != 0:
            _log("error", f"LED fenyero beallitas sikertelen: kod={code}")
            return f"SetBrightness hibakod: {code}"
        with _lock:
            _led_brightness = brightness

    _log("info", f"LED beallitva: switch={_led_switch} brightness={_led_brightness}")
    return None


@app.route("/led", methods=["GET", "POST"])
def led():
    """POST body: {"switch": 0|1} and/or {"brightness": 0-10}. Both optional,
    only the fields present are changed. GET returns the last known state
    (read from hardware at startup, then tracked locally on every set)."""
    if request.method == "GET":
        return jsonify({"switch": _led_switch, "brightness": _led_brightness, "vui_error": _vui_error})

    data = request.get_json(silent=True) or {}
    if "switch" not in data and "brightness" not in data:
        return jsonify({"error": "legalabb 'switch' (0|1) vagy 'brightness' (0-10) kotelezo"}), 422

    switch = None
    if "switch" in data:
        try:
            switch = int(data["switch"])
        except (TypeError, ValueError):
            return jsonify({"error": "switch: 0 vagy 1"}), 422
        if switch not in (0, 1):
            return jsonify({"error": "switch: 0 vagy 1"}), 422

    brightness = None
    if "brightness" in data:
        try:
            brightness = int(data["brightness"])
        except (TypeError, ValueError):
            return jsonify({"error": "brightness: 0-10 egesz"}), 422
        if not 0 <= brightness <= 10:
            return jsonify({"error": "brightness tartomany 0-10"}), 422

    err = _set_led(switch, brightness)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"ok": True, "switch": _led_switch, "brightness": _led_brightness})


@app.route("/led/presets", methods=["GET"])
def led_presets():
    return jsonify(LED_PRESETS)


@app.route("/led/preset/<name>", methods=["POST"])
def led_preset(name):
    preset = LED_PRESETS.get(name)
    if preset is None:
        return jsonify({"error": f"ismeretlen preset {name!r}", "known": sorted(LED_PRESETS)}), 404
    err = _set_led(preset.get("switch"), preset.get("brightness"))
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"ok": True, "preset": name, "switch": _led_switch, "brightness": _led_brightness})


@app.route("/action/<name>", methods=["POST"])
def action(name):
    """Posture and gesture commands."""
    if name in ("search_light", "light"):
        target = 0 if _led_switch else 1
        err = _set_led(switch=target, brightness=10 if target else None)
        if err:
            return jsonify({"error": err}), 503
        return jsonify({"ok": True, "action": name, "light": bool(target),
                         "switch": _led_switch, "brightness": _led_brightness})

    with _lock:
        if _sport is None:
            return jsonify({"error": f"SportClient nem elerheto: {_sdk_error}"}), 503
        if not _armed and name not in ("damp",):
            return jsonify({"error": "a robot nincs elesitve"}), 409
        sport = _sport

    fns = {
        "stand": getattr(sport, "RecoveryStand", getattr(sport, "StandUp", None)),
        "stand_up": getattr(sport, "RecoveryStand", getattr(sport, "StandUp", None)),
        "standup": getattr(sport, "RecoveryStand", getattr(sport, "StandUp", None)),
        "lay_down": getattr(sport, "StandDown", None),
        "laydown": getattr(sport, "StandDown", None),
        "sit": getattr(sport, "Sit", None),
        "damp": getattr(sport, "Damp", getattr(sport, "StopMove", None)),
        "balance": getattr(sport, "BalanceStand", getattr(sport, "RecoveryStand", None)),
        "wave": getattr(sport, "Hello", None),
        "greet": getattr(sport, "Hello", None),
        "hello": getattr(sport, "Hello", None),
        "heart": getattr(sport, "FingerHeart", getattr(sport, "Heart", None)),
        "love": getattr(sport, "FingerHeart", getattr(sport, "Heart", None)),
        "shake_hand": getattr(sport, "WiggleHips", getattr(sport, "Scrape", None)),
        "shake": getattr(sport, "WiggleHips", getattr(sport, "Scrape", None)),
        "stretch": getattr(sport, "Stretch", None),
        "pounce": getattr(sport, "FrontPounce", None),
        "front_pounce": getattr(sport, "FrontPounce", None),
        "jump": getattr(sport, "FrontJump", None),
        "front_jump": getattr(sport, "FrontJump", None),
        "dance": getattr(sport, "Dance1", None),
        "dance2": getattr(sport, "Dance2", None),
        "front_flip": getattr(sport, "FrontFlip", None),
        "back_flip": getattr(sport, "BackFlip", None),
        "left_flip": getattr(sport, "LeftFlip", None),
        "right_flip": getattr(sport, "RightFlip", None),
    }
    fn = fns.get(name)
    if fn is None:
        return jsonify({"error": f"ismeretlen parancs: {name}"}), 404
    try:
        code = fn()
    except Exception as exc:
        _log("error", f"{name} hiba: {exc}")
        return jsonify({"error": str(exc)}), 500
    with _lock:
        _armed_t = time.time()
    _log("info", f"parancs: {name} (kod={code})")
    return jsonify({"ok": True, "action": name, "code": code})


if __name__ == "__main__":
    print(f"[mc_motion] port {PORT}, iface {IFACE}, "
          f"limits vx={MAX_VX} vy={MAX_VY} vyaw={MAX_VYAW}, "
          f"watchdog {COMMAND_TIMEOUT_S}s", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
