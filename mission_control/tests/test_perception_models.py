"""perception model_manager / system_info -- pure parts, no GPU needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))

import model_manager as mm  # noqa: E402
import system_info  # noqa: E402


def test_validate_rejects_unknown_values():
    mm.validate("yolov8n", "engine", 640)
    for args in (("yolov9x", "pt", 640), ("yolov8n", "onnx", 640), ("yolov8n", "pt", 500)):
        with pytest.raises(mm.ModelSwitchError) as e:
            mm.validate(*args)
        assert e.value.status == 422


def test_saved_choice_falls_back_on_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    m = mm.ModelManager(None, apply=lambda *_: None)
    default = {"id": "yolov8n", "format": "engine", "imgsz": 640}
    assert m.saved_choice(default) == default
    (tmp_path / "selected.json").write_text('{"id": "nope", "format": "pt", "imgsz": 640}')
    assert m.saved_choice(default) == default
    m._save_choice({"id": "yolo11s", "format": "pt", "imgsz": 480})
    assert m.saved_choice(default) == {"id": "yolo11s", "format": "pt", "imgsz": 480}


def test_second_switch_while_busy_is_409(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    m = mm.ModelManager(None, apply=lambda *_: None)
    m.switch["state"] = "building"
    with pytest.raises(mm.ModelSwitchError) as e:
        m.request("yolov8s", "engine", 640)
    assert e.value.status == 409


def test_summary_lists_engine_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", str(tmp_path))
    (tmp_path / "yolov8n_480_fp16.engine").write_bytes(b"x")
    s = mm.ModelManager(None, apply=lambda *_: None).summary()
    v8n = next(a for a in s["available"] if a["id"] == "yolov8n")
    assert v8n["engine_ready"]["480"] and not v8n["engine_ready"]["640"]


def test_cpu_count_spec():
    assert system_info._count_cpus("0-3") == 4
    assert system_info._count_cpus("0-3,6") == 5
    assert system_info._count_cpus("") == 0


def test_power_mode_name(tmp_path, monkeypatch):
    (tmp_path / "status").write_text("pmode:0002 fmode:fanNull")
    (tmp_path / "conf").write_text("< POWER_MODEL ID=0 NAME=MAXN >\n< POWER_MODEL ID=2 NAME=15W >\n")
    monkeypatch.setattr(system_info, "NVP_STATUS", str(tmp_path / "status"))
    monkeypatch.setattr(system_info, "NVP_CONF", str(tmp_path / "conf"))
    assert system_info._mode() == "15W"
    assert system_info.power()["switch_supported"] is False
