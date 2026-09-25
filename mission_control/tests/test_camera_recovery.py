"""camera_recovery -- ladder, supervisor and sysfs lookup. No camera needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))

import camera_recovery as cr  # noqa: E402


def test_ladder_escalates_and_ends_in_exit():
    seq = [cr.action_for(n) for n in range(cr.EXIT_AT + 1)]
    assert seq[:cr.HW_RESET_AT] == ["retry"] * cr.HW_RESET_AT
    assert seq[cr.HW_RESET_AT:cr.EXIT_AT] == ["hw_reset"] * (cr.EXIT_AT - cr.HW_RESET_AT)
    assert seq[cr.EXIT_AT] == "exit"


def test_ladder_never_uses_usb_reset():
    assert not hasattr(cr, "usb_reset") and not hasattr(cr, "replug")   # they killed a streaming camera


def test_backoff_is_capped():
    assert cr.backoff_s(0) == 2.0
    assert cr.backoff_s(100) == 8.0


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_failures_clear_only_after_stable_streaming():
    clock = FakeClock()
    sup = cr.CameraSupervisor(clock=clock)
    sup.note_failure()
    sup.note_failure()
    sup.note_frame()
    clock.t += 3
    sup.note_frame()
    assert sup.failures == 2          # a short burst of frames does not count as healthy
    clock.t += cr.STABLE_STREAM_S
    sup.note_frame()
    assert sup.failures == 0
    assert sup.total_failures == 2    # the lifetime counter stays


def test_a_failure_restarts_the_stable_timer():
    clock = FakeClock()
    sup = cr.CameraSupervisor(clock=clock)
    sup.note_failure()
    sup.note_frame()
    clock.t += cr.STABLE_STREAM_S - 1
    sup.note_failure()                # the camera dies just before it counted as stable
    sup.note_frame()
    clock.t += 2
    sup.note_frame()
    assert sup.failures == 2


def test_before_open_records_the_step(monkeypatch):
    monkeypatch.setenv("RS_RESET_ON_START", "0")
    sup = cr.CameraSupervisor(clock=FakeClock())
    assert sup.before_open() == "retry" and not sup.want_hw_reset
    sup.failures = cr.HW_RESET_AT
    assert sup.before_open() == "hw_reset" and sup.want_hw_reset
    assert sup.status()["last_recovery"] == "hw_reset"
    sup.failures = cr.EXIT_AT
    assert sup.before_open() == "exit"
    assert sup.status()["restarts_requested"] == 1


def test_fresh_process_resets_the_camera_once_before_first_open(monkeypatch):
    monkeypatch.delenv("RS_RESET_ON_START", raising=False)
    sup = cr.CameraSupervisor(clock=FakeClock())
    assert sup.before_open() == "hw_reset" and sup.want_hw_reset
    assert sup.before_open() == "retry" and not sup.want_hw_reset   # only the first open


def test_blind_camera_goes_straight_to_a_clean_restart(monkeypatch):
    monkeypatch.setenv("RS_RESET_ON_START", "0")
    sup = cr.CameraSupervisor(clock=FakeClock())
    sup.note_failure(blind=True)
    assert sup.before_open() == "exit"


def test_a_non_blind_failure_leaves_the_shortcut(monkeypatch):
    monkeypatch.setenv("RS_RESET_ON_START", "0")
    sup = cr.CameraSupervisor(clock=FakeClock())
    sup.note_failure(blind=True)
    sup.note_failure(blind=False)     # the camera is visible again but the stream fails
    assert not sup.blind and sup.before_open() == "hw_reset"


def test_absent_camera_is_reported_and_clears(monkeypatch):
    clock = FakeClock()
    sup = cr.CameraSupervisor(clock=clock)
    monkeypatch.setattr(cr, "find_realsense", lambda: None)
    sup.set_absent(True)
    clock.t += 30
    st = sup.status()
    assert st["needs_replug"] and st["state"] == "absent" and st["absent_for_s"] == 30
    sup.set_absent(False)
    st = sup.status()
    assert not st["needs_replug"] and st["state"] == "opening" and st["absent_for_s"] is None


def _fake_usb(tmp_path, name, vid, product):
    d = tmp_path / name
    d.mkdir()
    (d / "idVendor").write_text(vid + "\n")
    (d / "product").write_text(product + "\n")


def test_find_realsense_reads_sysfs(tmp_path, monkeypatch):
    _fake_usb(tmp_path, "1-3", "1a86", "USB Serial")
    _fake_usb(tmp_path, "2-1", "8086", "Intel(R) RealSense(TM) Depth Camera 435i")
    (tmp_path / "2-1_if0").mkdir()                    # an interface entry has no idVendor
    monkeypatch.setattr(cr.glob, "glob", lambda pat: sorted(str(p) for p in tmp_path.iterdir()))
    assert cr.find_realsense()["sys"].endswith("2-1")


def test_find_realsense_none_when_absent(tmp_path, monkeypatch):
    _fake_usb(tmp_path, "1-3", "1a86", "USB Serial")
    monkeypatch.setattr(cr.glob, "glob", lambda pat: sorted(str(p) for p in tmp_path.iterdir()))
    assert cr.find_realsense() is None
    assert "not in sysfs" in cr.keep_awake()
