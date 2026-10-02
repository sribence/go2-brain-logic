"""Evidence recorder for the omni pillar (CONTRACT.md 9.3).

Normal operation stores nothing on disk: every camera frame goes into a RAM
ring buffer (last PREBUFFER_S seconds, byte-bounded) as the camera's own
compressed JPEG (rgb, no re-encode) or float16 .npy bytes (thermal).

On a trigger the Recorder goes IDLE -> EVIDENCE:
  * the pre-buffer is dumped to the incident folder,
  * cameras switch to their "evidence" mode (max resolution, MJPEG passthrough),
  * frames are written at evidence_fps per camera by a background writer
    thread (bounded queue, drop-oldest, never blocks capture or pipeline),
  * best_shot per (gid, cam): score = Laplacian variance x bbox area x conf,
    the winner is cropped from the full-resolution frame (20 % margin) on stop,
  * meta.jsonl gets (t, pose, persons) lines.
EVIDENCE -> POST when no person is seen, POST -> EVIDENCE when one reappears,
POST -> IDLE post_s after the last person (unless hold=True: manual stop).

Incident folder `<incident_dir>/<YYYYmmddTHHMMSS>_<robot>/`:
  frames/<cam>/<t>.jpg, thermal/<cam>/<t>.npy + .png, best/<gid>_<cam>.jpg|.npy|.png,
  captures/<label>_<cam>_<t>.jpg, meta.jsonl, manifest.json.
manifest.json: every file with SHA-256, chained: entry.hash = sha256(canonical
entry incl. prev hash), head = last hash -> verify_incident() detects edits,
removals and reordering. While recording, entries stream to chain.jsonl.

Quota: MAX_INCIDENT_GB over all incidents; the oldest finished incidents are
deleted first, the active one never (when it alone exceeds the quota, new
frames are dropped and counted).
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("omni.recorder")

PREBUFFER_S = 10.0
RING_MAX_BYTES = 64 * 1024 * 1024
POST_S = 30.0
EVIDENCE_FPS = 5.0
MAX_INCIDENT_GB = 20.0
BEST_MARGIN = 0.20
MANIFEST_VERSION = 1

IDLE, EVIDENCE, POST = "IDLE", "EVIDENCE", "POST"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_hash(entry: dict) -> str:
    """Hash of a chain entry (all fields except 'hash', canonical JSON)."""
    body = {k: entry[k] for k in sorted(entry) if k != "hash"}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def npy_bytes(arr: np.ndarray, dtype=np.float16) -> bytes:
    bio = io.BytesIO()
    np.save(bio, np.asarray(arr, dtype=dtype), allow_pickle=False)
    return bio.getvalue()


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    import cv2

    if gray is None or gray.size < 9:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def thermal_to_u8(temp: np.ndarray, lo: float = 15.0, hi: float = 40.0) -> np.ndarray:
    t = np.nan_to_num(np.asarray(temp, np.float32), nan=lo)
    return (np.clip((t - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def to_gray(img: np.ndarray, modality: str = "rgb") -> np.ndarray:
    import cv2

    if modality != "rgb":
        return thermal_to_u8(img)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def shot_score(img: np.ndarray, bbox, conf: float, modality: str = "rgb") -> float:
    """best_shot score = Laplacian variance (crop) x bbox area x conf."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = to_gray(img[y1:y2, x1:x2], modality)
    return sharpness(crop) * float((x2 - x1) * (y2 - y1)) * float(conf)


def expand_bbox(bbox, w: int, h: int, margin: float = BEST_MARGIN, scale_x: float = 1.0,
                scale_y: float = 1.0) -> Tuple[int, int, int, int]:
    """Scale a bbox to another resolution and grow it by `margin` per side, clipped."""
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 * scale_x, x2 * scale_x
    y1, y2 = y1 * scale_y, y2 * scale_y
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my))))


def parse_night_hours(spec: str) -> Tuple[int, int]:
    """'20:00-06:00' -> (start_min, end_min)."""
    a, b = spec.split("-")

    def _m(x):
        hh, mm = (x.strip().split(":") + ["0"])[:2]
        return int(hh) * 60 + int(mm)

    return _m(a), _m(b)


def is_night(t: float, spec: str = "20:00-06:00") -> bool:
    s, e = parse_night_hours(spec)
    lt = time.localtime(t)
    m = lt.tm_hour * 60 + lt.tm_min
    return (s <= m < e) if s <= e else (m >= s or m < e)


def safe_join(root: str, rel: str) -> Optional[str]:
    """root/rel if it stays inside root (no '..', no absolute, no symlink escape)."""
    if not rel or "\x00" in rel or rel.startswith(("/", "\\")) or ".." in re.split(r"[\\/]+", rel):
        return None
    root_r = os.path.realpath(root)
    p = os.path.realpath(os.path.join(root_r, rel))
    if os.path.commonpath([root_r, p]) != root_r or p == root_r:
        return None
    return p


def dir_size(path: str) -> int:
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def verify_incident(path: str) -> dict:
    """Recompute every file hash and the chain of manifest.json."""
    errors = []  # type: List[str]
    try:
        with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": ["manifest: %s" % exc]}
    prev = "0" * 64
    for i, e in enumerate(man.get("entries", [])):
        if e.get("i") != i:
            errors.append("entry %d: index %r" % (i, e.get("i")))
        if e.get("prev") != prev:
            errors.append("entry %d: broken chain" % i)
        if entry_hash(e) != e.get("hash"):
            errors.append("entry %d: entry hash mismatch" % i)
        fp = safe_join(path, e.get("path", ""))
        if fp is None or not os.path.isfile(fp):
            errors.append("missing %s" % e.get("path"))
        elif sha256_file(fp) != e.get("sha256"):
            errors.append("modified %s" % e.get("path"))
        prev = e.get("hash", "")
    if man.get("head") != prev:
        errors.append("head mismatch")
    return {"ok": not errors, "errors": errors, "files": len(man.get("entries", []))}


def enforce_quota(root: str, max_bytes: int, protect=(), sizes: Optional[Dict[str, int]] = None) -> List[str]:
    """Delete the oldest incident folders (name order = time order) until the
    total is <= max_bytes. Folders in `protect` are never deleted."""
    if not os.path.isdir(root):
        return []
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    sz = {}
    for n in names:
        if sizes is not None and n in sizes and n not in protect:
            sz[n] = sizes[n]
        else:
            sz[n] = dir_size(os.path.join(root, n))
            if sizes is not None and n not in protect:
                sizes[n] = sz[n]
    total = sum(sz.values())
    deleted = []
    for n in names:
        if total <= max_bytes:
            break
        if n in protect:
            continue
        shutil.rmtree(os.path.join(root, n), ignore_errors=True)
        total -= sz[n]
        deleted.append(n)
        if sizes is not None:
            sizes.pop(n, None)
    return deleted


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class RingBuffer(object):
    """Time- and byte-bounded FIFO of (t, data, w, h, kind); kind 'jpg' | 'npy'.
    push() is O(1) amortised and only stores references (no copies)."""

    def __init__(self, max_s: float = PREBUFFER_S, max_bytes: int = RING_MAX_BYTES):
        self.max_s = float(max_s)
        self.max_bytes = int(max_bytes)
        self._q = collections.deque()  # type: collections.deque
        self.bytes = 0
        self.evicted = 0
        self._lock = threading.Lock()

    def push(self, t: float, data: bytes, w: int, h: int, kind: str = "jpg") -> None:
        n = len(data)
        with self._lock:
            self._q.append((t, data, w, h, kind))
            self.bytes += n
            q = self._q
            while q and (self.bytes > self.max_bytes or t - q[0][0] > self.max_s):
                old = q.popleft()
                self.bytes -= len(old[1])
                self.evicted += 1

    def snapshot(self, since: Optional[float] = None) -> list:
        with self._lock:
            return [e for e in self._q if since is None or e[0] >= since]

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
            self.bytes = 0

    def __len__(self) -> int:
        return len(self._q)

    def span_s(self) -> float:
        with self._lock:
            return (self._q[-1][0] - self._q[0][0]) if len(self._q) > 1 else 0.0

    def stats(self) -> dict:
        return {"frames": len(self._q), "bytes": self.bytes, "span_s": round(self.span_s(), 2),
                "evicted": self.evicted}


# ---------------------------------------------------------------------------
# Incident (one folder + hash chain)
# ---------------------------------------------------------------------------

class Incident(object):
    def __init__(self, root: str, incident_id: str, reason: str, robot: str, t: float, kind: str = "evidence"):
        self.id = incident_id
        self.dir = os.path.join(root, incident_id)
        self.reason = reason
        self.robot = robot
        self.kind = kind
        self.started = t
        self.ended = None  # type: Optional[float]
        self.entries = []  # type: List[dict]
        self.prev = "0" * 64
        self.bytes = 0
        self.counts = collections.Counter()  # type: collections.Counter
        self.best = {}  # type: Dict[Tuple[int, str], dict]
        self.finalized = False
        self.lock = threading.Lock()
        self._used = set()  # type: set
        os.makedirs(self.dir, exist_ok=True)
        self._chain = open(os.path.join(self.dir, "chain.jsonl"), "a", encoding="utf-8")
        self._meta = open(os.path.join(self.dir, "meta.jsonl"), "a", encoding="utf-8")

    def unique(self, rel: str) -> str:
        base, ext = os.path.splitext(rel)
        k = 1
        while rel in self._used or os.path.exists(os.path.join(self.dir, rel)):
            rel = "%s_%d%s" % (base, k, ext)
            k += 1
        self._used.add(rel)
        return rel

    def _add_entry(self, rel: str, digest: str, size: int, t: float) -> dict:
        e = {"i": len(self.entries), "path": rel, "sha256": digest, "size": int(size), "t": round(float(t), 3),
             "prev": self.prev}
        e["hash"] = entry_hash(e)
        self.prev = e["hash"]
        self.entries.append(e)
        try:
            self._chain.write(json.dumps(e, sort_keys=True) + "\n")
            self._chain.flush()
        except (OSError, ValueError):
            pass
        return e

    def write(self, rel: str, data: bytes, t: float, kind: str = "file") -> str:
        """Write a file and append it to the hash chain. Returns the final relative path."""
        with self.lock:
            rel = self.unique(rel)
            p = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            self._add_entry(rel, sha256_bytes(data), len(data), t)
            self.bytes += len(data)
            self.counts[kind] += 1
            return rel

    def meta(self, record: dict) -> None:
        with self.lock:
            if self.finalized:
                return
            line = json.dumps(record, default=float, separators=(",", ":")) + "\n"
            self._meta.write(line)
            self.bytes += len(line)
            self.counts["meta"] += 1

    def finalize(self, t: float, extra: Optional[dict] = None) -> dict:
        with self.lock:
            if self.finalized:
                return self.manifest_head()
            self.ended = t
            try:
                self._meta.close()
            except OSError:
                pass
            mp = os.path.join(self.dir, "meta.jsonl")
            if os.path.exists(mp):
                self._add_entry("meta.jsonl", sha256_file(mp), os.path.getsize(mp), t)
            man = {"version": MANIFEST_VERSION, "incident_id": self.id, "kind": self.kind, "robot": self.robot,
                   "reason": self.reason, "started": self.started, "ended": t,
                   "counts": dict(self.counts), "bytes": self.bytes, "hash_alg": "sha256",
                   "chain": "entry.hash = sha256(canonical JSON of entry without 'hash'); "
                            "entry.prev = previous entry hash (first: 64 x '0'); head = last hash",
                   "entries": self.entries, "head": self.prev}
            man.update(extra or {})
            tmp = os.path.join(self.dir, "manifest.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(man, f, indent=1, default=float)
            os.replace(tmp, os.path.join(self.dir, "manifest.json"))
            try:
                self._chain.close()
                os.remove(os.path.join(self.dir, "chain.jsonl"))
            except OSError:
                pass
            self.finalized = True
            return self.manifest_head()

    def manifest_head(self) -> dict:
        return {"incident_id": self.id, "files": len(self.entries), "head": self.prev, "bytes": self.bytes}


# ---------------------------------------------------------------------------
# Writer thread (bounded, drop-oldest for frames, never blocks the producer)
# ---------------------------------------------------------------------------

class _Writer(object):
    def __init__(self, max_frames: int = 256):
        self.max_frames = int(max_frames)
        self._q = collections.deque()  # type: collections.deque
        self._n_frames = 0
        self._cv = threading.Condition()
        self.dropped = 0
        self.written = 0
        self.errors = 0
        self.last_error = ""
        self.busy = False
        self._thread = None  # type: Optional[threading.Thread]
        self._stop = False

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="omni-rec-writer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def put(self, fn: Callable[[], None], droppable: bool = True) -> bool:
        """Enqueue a job. Droppable (frame) jobs: when the queue holds max_frames
        of them, the OLDEST droppable job is discarded (counted). Never blocks."""
        dropped = False
        with self._cv:
            if droppable:
                if self._n_frames >= self.max_frames:
                    for i, (d, _) in enumerate(self._q):
                        if d:
                            del self._q[i]
                            self._n_frames -= 1
                            self.dropped += 1
                            dropped = True
                            break
                self._n_frames += 1
            self._q.append((droppable, fn))
            self._cv.notify()
        return not dropped

    def pending(self) -> int:
        return len(self._q)

    def drain(self, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if not self._q and not self.busy:
                return True
            time.sleep(0.01)
        return False

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(0.5)
                if not self._q and self._stop:
                    return
                d, fn = self._q.popleft()
                if d:
                    self._n_frames -= 1
                self.busy = True
            try:
                fn()
                self.written += 1
            except Exception as exc:  # disk full etc.: keep going
                self.errors += 1
                self.last_error = str(exc)
                logger.warning("recorder write failed: %s", exc)
            finally:
                self.busy = False


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder(object):
    def __init__(self, cap=None, rig=None, incident_dir: str = "incidents", robot_id: str = "go2",
                 prebuffer_s: float = PREBUFFER_S, ring_bytes: int = RING_MAX_BYTES, post_s: float = POST_S,
                 evidence_fps=EVIDENCE_FPS, max_incident_gb: float = MAX_INCIDENT_GB, queue_frames: int = 256,
                 meta_hz: float = 2.0, publish: Optional[Callable[[str, dict], None]] = None,
                 auto: str = "0", night_hours: str = "20:00-06:00", switch_modes: bool = True,
                 clock: Callable[[], float] = time.time):
        self.cap = cap
        self.rig = rig
        self.root = os.path.abspath(incident_dir)
        self.robot = re.sub(r"[^A-Za-z0-9_-]", "", str(robot_id)) or "robot"
        self.prebuffer_s = float(prebuffer_s)
        self.post_s = float(post_s)
        self.evidence_fps = evidence_fps  # float or {cam_id: fps}
        self.max_bytes = int(float(max_incident_gb) * 1e9)
        self.meta_hz = float(meta_hz)
        self.publish = publish
        self.auto = str(auto or "0").lower()
        self.night_hours = night_hours
        self.switch_modes = bool(switch_modes)
        self.clock = clock
        cams = self._cam_ids()
        n = max(1, len(cams))
        self.rings = {c: RingBuffer(prebuffer_s, max(1, int(ring_bytes) // n)) for c in cams}  # type: Dict[str, RingBuffer]
        self._ring_bytes = int(ring_bytes)
        self.writer = _Writer(queue_frames)
        self.writer.start()
        self.state = IDLE
        self.incident = None  # type: Optional[Incident]
        self.last_incident = None  # type: Optional[dict]
        self.reason = ""
        self.hold = False
        self.last_person_t = 0.0
        self._last_write = {}  # type: Dict[str, float]
        self._last_meta = 0.0
        self._lock = threading.RLock()
        self._sizes = {}  # type: Dict[str, int]
        self.quota_drops = 0
        self.quota_deleted = []  # type: List[str]
        self._writes_since_quota = 0
        self.on_frame_s = 0.0  # cumulative listener time (overhead metric)
        self.on_frame_n = 0
        if cap is not None and hasattr(cap, "add_listener"):
            cap.add_listener(self.on_frame)

    # -- config helpers ------------------------------------------------------
    def _cam_ids(self) -> List[str]:
        if self.rig is None:
            return []
        return [c for c, s in self.rig.cameras.items() if s.modality in ("rgb", "thermal")]

    def _modality(self, cam_id: str) -> str:
        spec = self.rig.cameras.get(cam_id) if self.rig is not None else None
        return spec.modality if spec is not None else "rgb"

    def _fps(self, cam_id: str) -> float:
        f = self.evidence_fps
        if isinstance(f, dict):
            f = f.get(cam_id, f.get("default", EVIDENCE_FPS))
        return float(f)

    def _ring(self, cam_id: str) -> RingBuffer:
        r = self.rings.get(cam_id)
        if r is None:
            r = self.rings[cam_id] = RingBuffer(self.prebuffer_s, max(1, self._ring_bytes // max(1, len(self.rings) + 1)))
        return r

    # -- capture-thread listener (hot path: keep cheap) ----------------------------
    @staticmethod
    def frame_payload(fr) -> Optional[Tuple[bytes, int, int, str]]:
        if fr.modality == "rgb":
            jpg = getattr(fr, "jpeg", None)
            if jpg is None:
                return None
            return jpg, int(getattr(fr, "full_w", 0)), int(getattr(fr, "full_h", 0)), "jpg"
        if fr.modality == "thermal":
            img = fr.image
            return npy_bytes(img), int(img.shape[1]), int(img.shape[0]), "npy"
        return None

    def on_frame(self, fr) -> None:
        if fr.modality not in ("rgb", "thermal"):
            return
        t0 = time.perf_counter()
        pl = self.frame_payload(fr)
        if pl is None:
            return
        data, w, h, kind = pl
        self._ring(fr.cam_id).push(fr.t, data, w, h, kind)
        inc = self.incident
        if inc is not None and self.state != IDLE and not inc.finalized:
            last = self._last_write.get(fr.cam_id, 0.0)
            if fr.t - last >= 1.0 / max(self._fps(fr.cam_id), 0.01) - 1e-3:
                self._last_write[fr.cam_id] = fr.t
                self.writer.put(self._job_frame(inc, fr.cam_id, fr.t, data, w, h, kind))
        self.on_frame_s += time.perf_counter() - t0
        self.on_frame_n += 1

    # -- writer jobs -----------------------------------------------------------
    def _write_entry(self, inc: Incident, cam: str, t: float, data: bytes, w: int, h: int, kind: str) -> None:
        if self.max_bytes and inc.bytes + len(data) > self.max_bytes:
            self.quota_drops += 1
            return
        ts = "%.3f" % t
        if kind == "jpg":
            inc.write("frames/%s/%s.jpg" % (cam, ts), data, t, "frame")
        else:
            inc.write("thermal/%s/%s.npy" % (cam, ts), data, t, "thermal")
            try:
                from omni import stream
            except ImportError:  # pragma: no cover
                import stream  # type: ignore
            arr = np.load(io.BytesIO(data), allow_pickle=False).astype(np.float32)
            inc.write("thermal/%s/%s.png" % (cam, ts), stream.encode_png(stream.colorize_thermal(arr)), t, "thermal_png")
        self._writes_since_quota += 1
        if self._writes_since_quota >= 50:
            self._writes_since_quota = 0
            self._quota()

    def _job_frame(self, inc, cam, t, data, w, h, kind):
        return lambda: self._write_entry(inc, cam, t, data, w, h, kind)

    def _quota(self) -> None:
        if not self.max_bytes:
            return
        inc = self.incident
        protect = {inc.id} if inc is not None else set()
        deleted = enforce_quota(self.root, self.max_bytes, protect=protect, sizes=self._sizes)
        if deleted:
            self.quota_deleted = (self.quota_deleted + deleted)[-20:]
            logger.info("quota: deleted incidents %s", deleted)

    # -- state machine -----------------------------------------------------------
    def _new_id(self, t: float, suffix: str = "") -> str:
        base = time.strftime("%Y%m%dT%H%M%S", time.localtime(t)) + "_" + self.robot + suffix
        iid, k = base, 1
        while os.path.exists(os.path.join(self.root, iid)):
            k += 1
            iid = "%s-%d" % (base, k)
        return iid

    def _publish(self, extra: Optional[dict] = None) -> None:
        if self.publish is None:
            return
        msg = {"t": self.clock(), "state": self.state, "incident_id": self.incident.id if self.incident else None,
               "reason": self.reason}
        msg.update(extra or {})
        try:
            self.publish("mc.omni.record", msg)
        except Exception:
            pass

    def _rgb_switchable(self) -> List[str]:
        if self.rig is None:
            return []
        return [c for c, s in self.rig.cameras.items() if s.modality == "rgb"]

    def start(self, reason: str = "manual", persons: Optional[list] = None, hold: bool = False,
              post_s: Optional[float] = None, source: str = "api") -> dict:
        now = self.clock()
        with self._lock:
            if self.state != IDLE and self.incident is not None:
                self.hold = self.hold or bool(hold)
                if persons:
                    self.last_person_t = now
                return self.status()
            os.makedirs(self.root, exist_ok=True)
            self._quota()
            inc = Incident(self.root, self._new_id(now), str(reason)[:200], self.robot, now)
            self.incident = inc
            self.reason = str(reason)[:200]
            self.hold = bool(hold)
            if post_s is not None:
                self.post_s = float(post_s)
            self.last_person_t = now
            self.state = EVIDENCE
            self._last_write = {}
            self._last_meta = 0.0
            inc.meta({"t": now, "event": "start", "reason": self.reason, "source": source, "persons": persons or []})
            # pre-buffer dump: one non-droppable job per camera
            for cam, ring in list(self.rings.items()):
                snap = ring.snapshot(since=now - self.prebuffer_s)
                if snap:
                    self.writer.put(self._job_prebuffer(inc, cam, snap), droppable=False)
                    self._last_write[cam] = snap[-1][0]
        if self.switch_modes and self.cap is not None:
            self.cap.set_mode(self._rgb_switchable(), "evidence")
        self._publish({"source": source})
        if self.publish is not None:
            try:
                self.publish("mc.core.anomaly", {"t": now, "source": "omni", "reason": "omni_evidence",
                                                 "detail": self.reason, "incident_id": inc.id,
                                                 "persons": len(persons or [])})
            except Exception:
                pass
        logger.info("evidence recording started: %s (%s)", inc.id, self.reason)
        return self.status()

    def _job_prebuffer(self, inc, cam, snap):
        def job():
            fps = self._fps(cam)
            last = -1e9
            for (t, data, w, h, kind) in snap:  # thin out to evidence_fps as well
                if t - last < 1.0 / max(fps, 0.01) - 1e-3:
                    continue
                last = t
                self._write_entry(inc, cam, t, data, w, h, kind)
        return job

    def stop(self, reason: str = "api") -> dict:
        with self._lock:
            inc = self.incident
            if inc is None or self.state == IDLE:
                return self.status()
            now = self.clock()
            inc.meta({"t": now, "event": "stop", "reason": reason})
            best = dict(inc.best)
            self.state = IDLE
            self.incident = None
            self.hold = False
            self.writer.put(self._job_finalize(inc, best, now, reason), droppable=False)
            self.last_incident = {"incident_id": inc.id, "started": inc.started, "ended": now, "stop_reason": reason,
                                  "finalized": False}
        if self.switch_modes and self.cap is not None:
            self.cap.set_mode(self._rgb_switchable(), "normal")
        self._publish({"incident_id": inc.id, "stop_reason": reason})
        logger.info("evidence recording stopped: %s (%s)", inc.id, reason)
        return self.status()

    def _job_finalize(self, inc: Incident, best: dict, now: float, reason: str):
        def job():
            shots = []
            for (gid, cam), c in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                try:
                    shots += self._write_best(inc, gid, cam, c)
                except Exception as exc:
                    logger.warning("best shot %s/%s failed: %s", gid, cam, exc)
            head = inc.finalize(now, {"stop_reason": reason, "best_shots": shots,
                                      "dropped_frames": self.writer.dropped, "quota_drops": self.quota_drops})
            self._sizes[inc.id] = inc.bytes
            if self.last_incident and self.last_incident.get("incident_id") == inc.id:
                self.last_incident.update({"finalized": True, "files": head["files"], "head": head["head"]})
            self._publish({"event": "finalized", "incident_id": inc.id, "head": head["head"]})
        return job

    def _write_best(self, inc: Incident, gid: int, cam: str, c: dict) -> List[str]:
        import cv2

        out = []
        if c["kind"] == "jpg":
            img = cv2.imdecode(np.frombuffer(c["data"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return out
            H, W = img.shape[:2]
            x1, y1, x2, y2 = expand_bbox(c["bbox"], W, H, BEST_MARGIN, W / float(c["spec_w"]), H / float(c["spec_h"]))
            ok, buf = cv2.imencode(".jpg", img[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                out.append(inc.write("best/%d_%s.jpg" % (gid, cam), buf.tobytes(), c["t"], "best"))
        else:
            arr = c["data"]
            H, W = arr.shape[:2]
            x1, y1, x2, y2 = expand_bbox(c["bbox"], W, H, BEST_MARGIN)
            crop = arr[y1:y2, x1:x2]
            out.append(inc.write("best/%d_%s.npy" % (gid, cam), npy_bytes(crop), c["t"], "best"))
            try:
                from omni import stream
            except ImportError:  # pragma: no cover
                import stream  # type: ignore
            png = stream.encode_png(stream.colorize_thermal(crop))
            out.append(inc.write("best/%d_%s.png" % (gid, cam), png, c["t"], "best"))
        return out

    # -- pipeline hook ------------------------------------------------------------
    def auto_trigger(self, t: float, persons: list) -> Optional[str]:
        """Reason string if the auto rule fires, else None."""
        if not persons or self.auto in ("0", "", "off", "false", "no"):
            return None
        if self.auto == "always":
            return "auto:person"
        if is_night(t, self.night_hours):
            for p in persons:
                if "thermal" in (p.get("modality") or []):
                    return "auto:thermal_person_night"
        return None

    def on_tick(self, t: float, persons: list, frames: Optional[dict] = None, pose=None,
                detections: Optional[list] = None) -> None:
        """Called once per pipeline tick; cheap when IDLE (no persons, no work)."""
        if self.state == IDLE:
            why = self.auto_trigger(t, persons)
            if why is None:
                return
            self.start(why, persons, source="auto")
        inc = self.incident
        if inc is None:
            return
        with self._lock:
            if persons:
                self.last_person_t = t
                if self.state == POST:
                    self.state = EVIDENCE
                    self._publish()
            elif self.state == EVIDENCE:
                self.state = POST
                self._publish()
            if self.state == POST and not self.hold and t - self.last_person_t > self.post_s:
                self.stop("post_timeout")
                return
        if self.meta_hz > 0 and t - self._last_meta >= 1.0 / self.meta_hz:
            self._last_meta = t
            inc.meta({"t": round(t, 3), "pose": list(pose) if pose is not None else None, "persons": persons,
                      "state": self.state})
        if detections and frames:
            self.update_best(inc, persons, frames, detections)

    @staticmethod
    def match_gid(d3, persons: list, max_d: float = 1.0) -> Optional[int]:
        best, bd = None, max_d * max_d
        x, y = float(getattr(d3, "x", 0.0)), float(getattr(d3, "y", 0.0))
        for p in persons or []:
            d = (p.get("x", 0.0) - x) ** 2 + (p.get("y", 0.0) - y) ** 2
            if d <= bd:
                best, bd = p.get("gid"), d
        return best

    def update_best(self, inc: Incident, persons: list, frames: dict, detections: list) -> int:
        """Score each detection on the perception-size image; keep the top-1 per
        (gid, cam) as a reference to the full-res JPEG (rgb) / array (thermal)."""
        n = 0
        for pair in detections:
            det, d3 = pair if isinstance(pair, (tuple, list)) else (pair, pair)
            if det is None or d3 is None:
                continue
            fr = frames.get(det.cam_id)
            if fr is None or fr.modality not in ("rgb", "thermal"):
                continue
            gid = self.match_gid(d3, persons)
            if gid is None:
                continue
            img = fr.image
            s = shot_score(img, det.bbox, det.conf, fr.modality)
            key = (int(gid), det.cam_id)
            cur = inc.best.get(key)
            if s <= 0 or (cur is not None and cur["score"] >= s):
                continue
            if fr.modality == "rgb":
                jpg = getattr(fr, "jpeg", None)
                if jpg is None:
                    continue
                c = {"kind": "jpg", "data": jpg}
            else:
                c = {"kind": "npy", "data": img}
            c.update({"score": s, "t": fr.t, "bbox": tuple(float(v) for v in det.bbox), "conf": float(det.conf),
                      "spec_w": img.shape[1], "spec_h": img.shape[0]})
            inc.best[key] = c
            n += 1
        return n

    # -- single shots (mission `capture` op) -------------------------------------------
    def capture(self, cams="all", mode: str = "max", label: str = "", n: int = 3, timeout: float = 4.0) -> dict:
        """Immediate full-resolution single shots: switch to evidence mode, grab
        n frames, keep the sharpest, switch back. Written into the active incident
        (captures/) or a new finalized 'capture' incident. Blocking (seconds)."""
        if self.cap is None or self.rig is None:
            raise RuntimeError("no capture manager")
        ids = [c for c, s in self.rig.cameras.items() if s.modality in ("rgb", "thermal")]
        if cams not in (None, "all", ["all"]):
            want = [cams] if isinstance(cams, str) else list(cams)
            unknown = [c for c in want if c not in self.rig.cameras]
            if unknown:
                raise ValueError("unknown cameras: %s" % ", ".join(unknown))
            ids = [c for c in want if c in ids]
        rgb = [c for c in ids if self._modality(c) == "rgb"]
        want_mode = "evidence" if str(mode).lower() in ("max", "evidence") else "normal"
        switched = []
        if rgb and want_mode == "evidence":
            switched = [c for c in rgb if self.cap.mode(c) != "evidence"]
            if switched:
                self.cap.set_mode(switched, "evidence")
        try:
            shots = {}
            for c in ids:
                if self._modality(c) == "rgb":
                    m = want_mode if self.cap.mode(c) == want_mode or c in switched else None
                    frs = self.cap.wait_frames(c, n=max(1, int(n)), timeout=timeout, mode=m)
                    if not frs:
                        fr = self.cap.latest(c)
                        frs = [fr] if fr is not None else []
                    frs = [f for f in frs if getattr(f, "jpeg", None) is not None]
                    if frs:
                        scored = [(sharpness(to_gray(f.image)), f) for f in frs]
                        scored.sort(key=lambda x: -x[0])
                        shots[c] = scored[0]
                else:
                    fr = self.cap.latest(c)
                    if fr is not None:
                        shots[c] = (0.0, fr)
        finally:
            if switched and self.state == IDLE:
                self.cap.set_mode(switched, "normal")
        now = self.clock()
        lab = _LABEL_RE.sub("_", str(label or "capture"))[:40].strip("_") or "capture"
        with self._lock:
            inc = self.incident
            own = inc is None
            if own:
                os.makedirs(self.root, exist_ok=True)
                self._quota()
                inc = Incident(self.root, self._new_id(now, "_capture"), "capture:" + lab, self.robot, now,
                               kind="capture")
        files = []
        for c, (sh, fr) in shots.items():
            ts = "%.3f" % fr.t
            if fr.modality == "rgb":
                rel = inc.write("captures/%s_%s_%s.jpg" % (lab, c, ts), fr.jpeg, fr.t, "capture")
                files.append({"cam": c, "path": rel, "w": getattr(fr, "full_w", 0), "h": getattr(fr, "full_h", 0),
                              "sharpness": round(sh, 2), "mode": getattr(fr, "mode", "normal"), "t": fr.t})
            else:
                rel = inc.write("captures/%s_%s_%s.npy" % (lab, c, ts), npy_bytes(fr.image), fr.t, "capture")
                try:
                    from omni import stream
                except ImportError:  # pragma: no cover
                    import stream  # type: ignore
                rel2 = inc.write("captures/%s_%s_%s.png" % (lab, c, ts),
                                 stream.encode_png(stream.colorize_thermal(fr.image)), fr.t, "capture")
                files.append({"cam": c, "path": rel, "png": rel2, "w": fr.image.shape[1], "h": fr.image.shape[0],
                              "t": fr.t})
        inc.meta({"t": now, "event": "capture", "label": lab, "files": [f["path"] for f in files]})
        if own:
            inc.finalize(now, {"label": lab})
            self._sizes[inc.id] = inc.bytes
        return {"incident_id": inc.id, "label": lab, "files": files}

    # -- views -----------------------------------------------------------------
    def status(self) -> dict:
        inc = self.incident
        ring_bytes = sum(r.bytes for r in self.rings.values())
        out = {"state": self.state, "incident_id": inc.id if inc else None, "reason": self.reason if inc else None,
               "hold": self.hold, "post_s": self.post_s, "last_person_t": self.last_person_t,
               "started": inc.started if inc else None,
               "files": len(inc.entries) if inc else 0, "bytes": inc.bytes if inc else 0,
               "best_shots": len(inc.best) if inc else 0,
               "queue": self.writer.pending(), "dropped": self.writer.dropped, "write_errors": self.writer.errors,
               "quota_drops": self.quota_drops, "quota_deleted": self.quota_deleted[-5:],
               "ring": {"bytes": ring_bytes, "frames": sum(len(r) for r in self.rings.values()),
                        "cams": {c: r.stats() for c, r in self.rings.items()}},
               "overhead_us_per_frame": round(1e6 * self.on_frame_s / self.on_frame_n, 1) if self.on_frame_n else 0.0,
               "last_incident": self.last_incident, "auto": self.auto}
        return out

    def brief(self) -> dict:
        inc = self.incident
        return {"state": self.state, "incident_id": inc.id if inc else None, "reason": self.reason if inc else None}

    def list_incidents(self) -> List[dict]:
        if not os.path.isdir(self.root):
            return []
        out = []
        active = self.incident.id if self.incident else None
        for n in sorted(os.listdir(self.root), reverse=True):
            p = os.path.join(self.root, n)
            if not os.path.isdir(p) or not _ID_RE.match(n):
                continue
            item = {"id": n, "active": n == active, "finalized": os.path.exists(os.path.join(p, "manifest.json"))}
            if item["finalized"]:
                try:
                    with open(os.path.join(p, "manifest.json"), "r", encoding="utf-8") as f:
                        m = json.load(f)
                    item.update({k: m.get(k) for k in ("kind", "reason", "started", "ended", "bytes", "head")})
                    item["files"] = len(m.get("entries", []))
                except (OSError, ValueError):
                    item["finalized"] = False
            out.append(item)
        return out

    def incident_path(self, incident_id: str) -> Optional[str]:
        if not _ID_RE.match(incident_id or "") or ".." in incident_id:
            return None
        p = safe_join(self.root, incident_id)
        return p if p and os.path.isdir(p) else None

    def manifest(self, incident_id: str) -> Optional[dict]:
        p = self.incident_path(incident_id)
        if p is None:
            return None
        mp = os.path.join(p, "manifest.json")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as f:
                return json.load(f)
        inc = self.incident
        if inc is not None and inc.id == incident_id:  # still recording
            with inc.lock:
                return {"incident_id": inc.id, "active": True, "reason": inc.reason, "started": inc.started,
                        "entries": list(inc.entries), "head": inc.prev}
        return {"incident_id": incident_id, "active": False, "finalized": False}

    def file_path(self, incident_id: str, rel: str) -> Optional[str]:
        p = self.incident_path(incident_id)
        if p is None:
            return None
        fp = safe_join(p, rel)
        return fp if fp and os.path.isfile(fp) else None

    def close(self) -> None:
        if self.state != IDLE:
            self.stop("shutdown")
        self.writer.drain(10.0)
        self.writer.stop()
        if self.cap is not None and hasattr(self.cap, "remove_listener"):
            self.cap.remove_listener(self.on_frame)


def from_env(cap, rig, publish=None, base_dir: Optional[str] = None) -> Recorder:
    here = base_dir or os.path.dirname(os.path.abspath(__file__))
    d = os.environ.get("OMNI_INCIDENT_DIR") or os.path.join(here, "incidents")
    if not os.path.isabs(d):
        d = os.path.join(here, d)
    fps_env = os.environ.get("OMNI_EVIDENCE_FPS", str(EVIDENCE_FPS))
    fps = float(EVIDENCE_FPS)  # type: object
    try:
        if ":" in fps_env:  # "default:5,rgb_front:10"
            fps = {k.strip(): float(v) for k, v in (kv.split(":") for kv in fps_env.split(",") if kv.strip())}
        else:
            fps = float(fps_env)
    except ValueError:
        pass
    return Recorder(cap, rig, incident_dir=d,
                    robot_id=os.environ.get("OMNI_ROBOT_ID") or os.environ.get("ROBOT_ID") or "go2",
                    prebuffer_s=float(os.environ.get("OMNI_PREBUFFER_S", PREBUFFER_S)),
                    ring_bytes=int(float(os.environ.get("OMNI_RING_MB", "64")) * 1024 * 1024),
                    post_s=float(os.environ.get("OMNI_POST_S", POST_S)),
                    evidence_fps=fps,
                    max_incident_gb=float(os.environ.get("MAX_INCIDENT_GB", MAX_INCIDENT_GB)),
                    publish=publish,
                    auto=os.environ.get("OMNI_AUTO_EVIDENCE", "0"),
                    night_hours=os.environ.get("OMNI_NIGHT_HOURS", "20:00-06:00"))
