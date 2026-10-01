"""omni/gesture.py: synthetic COCO-17 sequences for wave / stop_palm / point + negatives."""
import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni import gesture as G  # noqa: E402
from omni.gesture import GestureClassifier, PoseDetector, match_pose_to_tracks  # noqa: E402

FPS = 15.0
UPPER, FORE = 55.0, 55.0


def _dir(a_deg):
    """Image-plane unit vector, angle CCW from image-right with y up (image y is down)."""
    a = math.radians(a_deg)
    return np.array([math.cos(a), -math.sin(a)])


def skeleton(cx=320.0, cy=300.0, left=(-90.0, -90.0), right=(-90.0, -90.0), conf=0.9):
    """Person facing the camera; torso 100 px. COCO 'left' arm appears on image right.

    ``left``/``right`` = (upper-arm angle, forearm angle) in image degrees.
    """
    kp = np.zeros((17, 3))
    kp[:, 2] = conf
    kp[0, :2] = (cx, cy - 140)
    kp[1, :2], kp[2, :2] = (cx + 8, cy - 148), (cx - 8, cy - 148)
    kp[3, :2], kp[4, :2] = (cx + 15, cy - 144), (cx - 15, cy - 144)
    kp[G.L_SH, :2], kp[G.R_SH, :2] = (cx + 40, cy - 100), (cx - 40, cy - 100)
    kp[G.L_HIP, :2], kp[G.R_HIP, :2] = (cx + 25, cy), (cx - 25, cy)
    kp[13, :2], kp[14, :2] = (cx + 25, cy + 60), (cx - 25, cy + 60)
    kp[15, :2], kp[16, :2] = (cx + 25, cy + 120), (cx - 25, cy + 120)
    for (sh, el, wr), (a_up, a_fo) in ((( G.L_SH, G.L_EL, G.L_WR), left), ((G.R_SH, G.R_EL, G.R_WR), right)):
        kp[el, :2] = kp[sh, :2] + UPPER * _dir(a_up)
        kp[wr, :2] = kp[el, :2] + FORE * _dir(a_fo)
    return kp


def run(seq, clf=None, gid=1, bearing=None):
    clf = clf or GestureClassifier()
    ev = []
    for t, kp in seq:
        ev.extend(clf.update(gid, kp, t, bearing_deg=bearing))
    return ev, clf


def gen(fn, dur_s, t0=0.0, noise_px=1.0, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    n = int(dur_s * FPS)
    for i in range(n):
        t = t0 + i / FPS
        kp = fn(t).copy()
        kp[:, :2] += rng.normal(0, noise_px, size=(17, 2))
        out.append((t, kp))
    return out


def wave_seq(freq=1.5, dur=2.5, amp_deg=35.0, **kw):
    return gen(lambda t: skeleton(left=(55.0, 90.0 + amp_deg * math.sin(2 * math.pi * freq * t))), dur, **kw)


def stop_seq(dur=1.6, **kw):
    return gen(lambda t: skeleton(right=(100.0, 95.0)), dur, **kw)


def point_seq(dur=1.4, **kw):  # left arm (image right) horizontal
    return gen(lambda t: skeleton(left=(5.0, 3.0)), dur, **kw)


def walking_seq(dur=4.0, freq=1.0, side_view=True, **kw):
    """Arms swing below the shoulders; side view -> large lateral wrist motion; body translates."""
    def f(t):
        sw = 30.0 * math.sin(2 * math.pi * freq * t)
        return skeleton(cx=200.0 + 60.0 * t, left=(-90.0 + sw, -90.0 + 1.3 * sw),
                        right=(-90.0 - sw, -90.0 - 1.3 * sw))
    return gen(f, dur, **kw)


def names(ev):
    return [e["gesture"] for e in ev]


# ----------------------------------------------------------------- positives
def test_wave_detected():
    ev, clf = run(wave_seq())
    assert names(ev) == ["wave"]
    e = ev[0]
    assert e["gid"] == 1 and e["side"] == "left" and 0.6 <= e["conf"] <= 1.0
    assert e["t"] <= 2.5
    assert clf.active(1) == ["wave"]


def test_stop_palm_detected_after_hold():
    ev, _ = run(stop_seq())
    assert names(ev) == ["stop_palm"]
    assert ev[0]["side"] == "right"
    assert ev[0]["t"] >= 0.9


def test_point_detected_with_bearing():
    ev, _ = run(point_seq(), bearing=20.0)
    assert names(ev) == ["point"]
    e = ev[0]
    assert e["img_dir"] == "right" and abs(e["dir_img_deg"]) < 15
    assert e["bearing"] == pytest.approx(20.0 - 30.0)  # image right == robot right == smaller bearing


def test_point_left_without_bearing():
    ev, _ = run(gen(lambda t: skeleton(right=(178.0, 180.0)), 1.4))
    assert names(ev) == ["point"] and ev[0]["img_dir"] == "left" and "bearing" not in ev[0]


# ----------------------------------------------------------------- negatives
@pytest.mark.parametrize("side_view", [True, False])
def test_walking_arm_swing_is_not_wave(side_view):
    ev, _ = run(walking_seq(side_view=side_view))
    assert ev == []


def test_fast_walk_swing_not_wave():
    ev, _ = run(walking_seq(freq=2.0))
    assert ev == []


def test_arms_down_still_nothing():
    ev, _ = run(gen(lambda t: skeleton(), 3.0))
    assert ev == []


def test_single_wave_cycle_not_enough():
    ev, _ = run(wave_seq(freq=0.5, dur=2.0))
    assert "wave" not in names(ev)


def test_brief_raise_not_stop():
    ev, _ = run(stop_seq(dur=0.6))
    assert ev == []


def test_waving_is_not_stop_palm():
    ev, _ = run(wave_seq(dur=3.0))
    assert "stop_palm" not in names(ev)


def test_raised_arm_bent_elbow_not_stop():
    ev, _ = run(gen(lambda t: skeleton(right=(160.0, 70.0)), 1.6))
    assert "stop_palm" not in names(ev)


def test_low_confidence_keypoints_ignored():
    ev, _ = run([(t, np.concatenate([kp[:, :2], np.full((17, 1), 0.1)], axis=1)) for t, kp in stop_seq()])
    assert ev == []


# ----------------------------------------------------------------- temporal logic
def test_hysteresis_single_event_while_held_and_rearm_after_cooldown():
    clf = GestureClassifier()
    ev1, _ = run(stop_seq(dur=4.0), clf)
    assert names(ev1) == ["stop_palm"]
    ev2, _ = run(gen(lambda t: skeleton(), 1.0, t0=4.0), clf)  # arm down
    assert ev2 == [] and clf.active(1) == []
    ev3, _ = run(stop_seq(dur=1.6, t0=7.5), clf)  # after cooldown
    assert names(ev3) == ["stop_palm"]


def test_reraise_within_cooldown_suppressed():
    clf = GestureClassifier()
    run(stop_seq(dur=1.6), clf)
    run(gen(lambda t: skeleton(), 0.5, t0=1.6), clf)
    ev, _ = run(stop_seq(dur=1.2, t0=2.1), clf)
    assert ev == []


def test_per_gid_independence_and_prune():
    clf = GestureClassifier()
    ev = []
    for (t, a), (_, b) in zip(stop_seq(), gen(lambda t: skeleton(), 1.6)):
        ev += clf.update_many([{"gid": 3, "keypoints": a}, {"gid": 4, "keypoints": b}], t)
    assert [(e["gid"], e["gesture"]) for e in ev] == [(3, "stop_palm")]
    clf.prune(100.0)
    assert clf.active(3) == []


def test_out_of_order_and_bad_input_ignored():
    clf = GestureClassifier()
    assert clf.update(1, skeleton(), 1.0) == []
    assert clf.update(1, skeleton(), 0.5) == []
    assert clf.update(1, np.zeros((5, 3)), 2.0) == []
    assert clf.update(1, skeleton()[:, :2], 2.1) == []  # (17,2) accepted, conf=1


def test_count_half_cycles_band():
    x = [0.2 * math.sin(2 * math.pi * i / 10.0) for i in range(30)]
    assert G.count_half_cycles(x, 0.08) == 5
    assert G.count_half_cycles([0.01 * v for v in x], 0.08) == 0


# ----------------------------------------------------------------- pose detector + matching
def test_match_pose_to_tracks_iou_greedy():
    k1, k2 = skeleton(cx=100), skeleton(cx=400)
    poses = [((50, 150, 150, 430), 0.9, k1), ((350, 150, 450, 430), 0.8, k2), ((800, 0, 900, 100), 0.7, k2)]
    persons = [{"bbox": (355, 155, 452, 425), "gid": 7, "bearing_deg": -10.0}, ((48, 152, 148, 428), 3)]
    m = match_pose_to_tracks(poses, persons)
    got = {d["gid"]: d for d in m}
    assert set(got) == {3, 7}
    assert got[7]["keypoints"] is k2 and got[7]["bearing_deg"] == -10.0
    assert got[3]["keypoints"] is k1 and got[3]["iou"] > 0.9
    assert match_pose_to_tracks(poses, [((0, 0, 10, 10), 1)]) == []


class _FakeModel:
    def __init__(self):
        self.batches = []

    def predict(self, batch, **kw):
        self.batches.append((len(batch), kw))
        kp = skeleton()[None].astype(np.float32)
        return [NS(boxes=NS(xyxy=np.array([[1, 2, 3, 4.0]]), conf=np.array([0.8])),
                   keypoints=NS(data=kp)) for _ in batch]


def test_pose_detector_batch_and_parse():
    pd = PoseDetector("yolo11n-pose.engine", engine_batch=4)
    pd._model = _FakeModel()
    imgs = [np.zeros((48, 64, 3), np.uint8)] * 2
    out = pd.detect(imgs)
    assert len(out) == 2 and pd._model.batches[0][0] == 4
    assert "half" not in pd._model.batches[0][1]
    bbox, conf, kp = out[0][0]
    assert bbox == (1.0, 2.0, 3.0, 4.0) and conf == pytest.approx(0.8) and kp.shape == (17, 3)
    assert pd.detect([]) == []


def test_pose_parse_xy_conf_fallback_and_empty():
    xy = skeleton()[None, :, :2]
    res = NS(boxes=NS(xyxy=np.array([[0, 0, 1, 1.0]]), conf=np.array([0.5])),
             keypoints=NS(xy=xy, conf=np.full((1, 17), 0.7)))
    (b, c, kp), = PoseDetector._parse(res)
    assert kp[:, 2].min() == pytest.approx(0.7)
    assert PoseDetector._parse(NS(boxes=None, keypoints=None)) == []


def test_pipeline_pose_to_events():
    clf = GestureClassifier()
    ev = []
    for t, kp in stop_seq():
        m = match_pose_to_tracks([((270, 150, 370, 430), 0.9, kp)], [((272, 150, 368, 432), 12, 5.0)])
        ev += clf.update_many(m, t)
    assert [(e["gid"], e["gesture"]) for e in ev] == [(12, "stop_palm")]
