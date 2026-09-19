"""YOLO model selection and TensorRT engine cache for the perception pillar.

`GET /models` lists the models, `POST /model` switches at runtime without a
container restart. A switch runs in a background thread: build the TensorRT
engine if it is not cached yet (3-10 min on the Orin NX), load it, warm it
up, and only then hand it to the tracker. The old model keeps running until
then.

A switch resets the ByteTrack ids, so a locked follow target is lost and the
follower stops (LOST). That is the safe outcome; lock the person again.

Engines are cached in MODELS_DIR (a volume on the robot) as
`<id>_<imgsz>_fp16.engine`. The last choice is saved in `selected.json` and
restored at start. At start the default is the FP16 engine; if it is not
built yet, the .pt model runs while the engine builds.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("perception.models")

MODELS = {            # id -> parameters, millions (ultralytics model cards)
    "yolov8n": 3.2,
    "yolov8s": 11.2,
    "yolo11n": 2.6,
    "yolo11s": 9.4,
}
FORMATS = ("pt", "engine")
IMGSZ_OPTIONS = (320, 416, 480, 640)
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))


class ModelSwitchError(Exception):
    def __init__(self, msg: str, status: int = 422):
        super().__init__(msg)
        self.status = status


def engine_path(model_id: str, imgsz: int) -> str:
    return os.path.join(MODELS_DIR, f"{model_id}_{imgsz}_fp16.engine")


def validate(model_id: str, fmt: str, imgsz: int) -> None:
    if model_id not in MODELS:
        raise ModelSwitchError(f"unknown model {model_id!r}, choose from {sorted(MODELS)}")
    if fmt not in FORMATS:
        raise ModelSwitchError(f"unknown format {fmt!r}, choose from {list(FORMATS)}")
    if imgsz not in IMGSZ_OPTIONS:
        raise ModelSwitchError(f"imgsz {imgsz} not allowed, choose from {list(IMGSZ_OPTIONS)}")


class ModelManager:
    def __init__(self, device: Optional[str], apply: Callable[[object, dict], None]):
        """`apply(model, spec)` hands a loaded, warmed-up model to the tracker."""
        self.device = device
        self._apply = apply
        self._lock = threading.Lock()
        self.current: Optional[dict] = None
        self.switch = {"state": "idle", "target": None, "error": None,
                       "started_at": None, "progress_note": ""}
        os.makedirs(MODELS_DIR, exist_ok=True)

    # ---------------------------------------------------------------- state

    def saved_choice(self, default: dict) -> dict:
        try:
            with open(os.path.join(MODELS_DIR, "selected.json")) as f:
                spec = json.load(f)
            validate(spec["id"], spec["format"], int(spec["imgsz"]))
            return {"id": spec["id"], "format": spec["format"], "imgsz": int(spec["imgsz"])}
        except Exception:
            return default

    def _save_choice(self, spec: dict) -> None:
        try:
            with open(os.path.join(MODELS_DIR, "selected.json"), "w") as f:
                json.dump({k: spec[k] for k in ("id", "format", "imgsz")}, f)
        except OSError as exc:
            logger.warning("cannot save model choice: %s", exc)

    def summary(self) -> dict:
        return {
            "current": self.current,
            "available": [{"id": m, "params_m": p,
                           "engine_ready": {str(s): os.path.exists(engine_path(m, s)) for s in IMGSZ_OPTIONS}}
                          for m, p in MODELS.items()],
            "formats": list(FORMATS),
            "imgsz_options": list(IMGSZ_OPTIONS),
            "switch": dict(self.switch),
        }

    # ---------------------------------------------------------------- loading

    def load_now(self, spec: dict):
        """Blocking load, used at start. Falls back to .pt if no engine is cached.

        Returns (model, spec actually loaded)."""
        if spec["format"] == "engine" and not os.path.exists(engine_path(spec["id"], spec["imgsz"])):
            fallback = dict(spec, format="pt")
            model = self._load(fallback)
            return model, fallback
        return self._load(spec), spec

    def _load(self, spec: dict):
        from ultralytics import YOLO

        if spec["format"] == "engine":
            model = YOLO(engine_path(spec["id"], spec["imgsz"]), task="detect")
        else:
            model = YOLO(f"{spec['id']}.pt")
        # Warm-up: the first inference allocates buffers and can take seconds.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        kw = {"imgsz": spec["imgsz"], "verbose": False}
        if self.device:
            kw["device"] = self.device
        if spec["format"] == "pt" and self.device:
            kw["half"] = True
        model.predict(dummy, **kw)
        return model

    def _build_engine(self, spec: dict) -> str:
        from ultralytics import YOLO

        out = engine_path(spec["id"], spec["imgsz"])
        self.switch["progress_note"] = (f"TensorRT FP16 engine build, {spec['id']} imgsz {spec['imgsz']} "
                                        "(3-10 min, the current model keeps running)")
        logger.info(self.switch["progress_note"])
        built = YOLO(f"{spec['id']}.pt").export(format="engine", half=True, imgsz=spec["imgsz"],
                                                device=self.device or 0, verbose=False)
        shutil.move(str(built), out)
        return out

    # ---------------------------------------------------------------- switch

    def request(self, model_id: str, fmt: str, imgsz: int) -> dict:
        validate(model_id, fmt, imgsz)
        spec = {"id": model_id, "format": fmt, "imgsz": imgsz}
        with self._lock:
            if self.switch["state"] in ("building", "loading"):
                raise ModelSwitchError(f"a switch is already running ({self.switch['state']})", 409)
            self.switch = {"state": "loading", "target": spec, "error": None,
                           "started_at": time.time(), "progress_note": "starting"}
        threading.Thread(target=self._run_switch, args=(spec,), daemon=True, name="model-switch").start()
        return dict(self.switch)

    def _run_switch(self, spec: dict) -> None:
        try:
            if spec["format"] == "engine" and not os.path.exists(engine_path(spec["id"], spec["imgsz"])):
                self.switch["state"] = "building"
                self._build_engine(spec)
            self.switch["state"] = "loading"
            self.switch["progress_note"] = "loading and warming up"
            model = self._load(spec)
            self._apply(model, spec)
            self.current = dict(spec, half=spec["format"] == "engine" or bool(self.device))
            self._save_choice(spec)
            self.switch.update(state="idle", progress_note="done", error=None)
            logger.info("model switched to %s", spec)
        except Exception as exc:          # keep the old model on any failure
            logger.exception("model switch failed")
            self.switch.update(state="error", error=str(exc)[:300], progress_note="failed, old model kept")
