// OmniView — coordinate frames + camera models (single source of truth).
//
// base  (ROS): x fwd, y left, z up            (CONTRACTS.md §1)
// cam (OpenCV): z fwd (optical axis), x right, y down
// world (map) : x, y, yaw from robot pose; z kept as in base (body-centric)
// three.js    : y up.   ROS (x, y, z)  ->  three (x, z, -y)
//
// Any THREE.Group with rotation.x = -PI/2 performs exactly that mapping, so
// geometry built in ROS coords can be dropped into a "ROS group" unchanged.

export const BASE_HEIGHT = 0.33;   // body (base_link) above floor, m (Go2 standing)
export const ZONES = { STOP: 0.8, SLOW: 2.0, CAUTION: 3.5 };

export function b2t(x, y, z, out) {
  return (out || new THREE.Vector3()).set(x, z, -y);
}
export function t2b(v) { return { x: v.x, y: -v.z, z: v.y }; }

export function makeRosGroup() {
  const g = new THREE.Group();
  g.rotation.x = -Math.PI / 2;
  return g;
}

// base-frame point -> world-frame point for pose {x, y, yaw}
export function baseToWorld(pose, x, y) {
  const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  return { x: pose.x + c * x - s * y, y: pose.y + s * x + c * y };
}
export function worldToBase(pose, x, y) {
  const dx = x - pose.x, dy = y - pose.y;
  const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  return { x: c * dx + s * dy, y: -s * dx + c * dy };
}
export function rotVec(yaw, vx, vy) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return { x: c * vx - s * vy, y: s * vx + c * vy };
}

export function zoneOf(d) {
  if (d < ZONES.STOP) return "red";
  if (d < ZONES.SLOW) return "orange";
  if (d < ZONES.CAUTION) return "yellow";
  return "green";
}
export const ZONE_COLOR = { red: 0xff3344, orange: 0xff8a1e, yellow: 0xffd23a, green: 0x35f28a };
export const ZONE_CSS = { red: "#ff3344", orange: "#ff8a1e", yellow: "#ffd23a", green: "#35f28a" };

// 4x4 nested rows (row-major, as in rig.yaml) -> THREE.Matrix4
export function mat4FromRows(rows) {
  const m = new THREE.Matrix4();
  if (!rows || rows.length < 3) return m;
  const r = (i, j) => (rows[i] && rows[i][j] != null ? +rows[i][j] : (i === j ? 1 : 0));
  m.set(r(0, 0), r(0, 1), r(0, 2), r(0, 3),
        r(1, 0), r(1, 1), r(1, 2), r(1, 3),
        r(2, 0), r(2, 1), r(2, 2), r(2, 3),
        0, 0, 0, 1);
  return m;
}

// T_base_cam for a camera at (tx,ty,tz) looking toward base yaw, pitched down.
export function camPoseRows(yawDeg, pitchDownDeg, tx, ty, tz) {
  const y = yawDeg * Math.PI / 180, p = pitchDownDeg * Math.PI / 180;
  const zc = [Math.cos(y) * Math.cos(p), Math.sin(y) * Math.cos(p), -Math.sin(p)];
  const xc = [Math.sin(y), -Math.cos(y), 0];
  const yc = [zc[1] * xc[2] - zc[2] * xc[1], zc[2] * xc[0] - zc[0] * xc[2], zc[0] * xc[1] - zc[1] * xc[0]];
  return [
    [xc[0], yc[0], zc[0], tx],
    [xc[1], yc[1], zc[1], ty],
    [xc[2], yc[2], zc[2], tz],
    [0, 0, 0, 1],
  ];
}

// Normalise one /rig camera entry. Accepts list or dict forms.
export function normalizeRig(rig) {
  let list = [];
  const src = rig && (rig.cameras || rig);
  if (Array.isArray(src)) list = src.map((c, i) => ({ ...c, id: c.id || c.cam_id || `cam${i}` }));
  else if (src && typeof src === "object") list = Object.keys(src).map((k) => ({ ...src[k], id: src[k].id || src[k].cam_id || k }));
  list.forEach((c, i) => { c._ord = c.order != null ? +c.order : c.index != null ? +c.index : c.idx != null ? +c.idx : i; });
  list.sort((a, b) => a._ord - b._ord);
  return list.map((c, i) => {
    const K = c.K || [[c.width / 2, 0, c.width / 2], [0, c.width / 2, c.height / 2], [0, 0, 1]];
    const T = mat4FromRows(c.T_base_cam);
    return {
      id: c.id, idx: i, modality: c.modality || "rgb",
      model: (c.model && c.model.type) || c.model || "pinhole",
      width: +c.width || 640, height: +c.height || 480,
      fx: +K[0][0], fy: +K[1][1], cx: +K[0][2], cy: +K[1][2],
      D: (c.D || []).map(Number),
      T, Tinv: T.clone().invert(),
      Trows: c.T_base_cam,
    };
  });
}

// ---- camera models in JS (used by demo synth + FPV) ----
// KB fisheye: theta_d = theta(1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
export function projectCam(cam, x, y, z) {
  if (cam.model === "fisheye") {
    const r = Math.hypot(x, y);
    const th = Math.atan2(r, z);
    if (th > maxThetaOf(cam)) return null;
    const t2 = th * th, d = cam.D;
    const thd = th * (1 + (d[0] || 0) * t2 + (d[1] || 0) * t2 * t2 + (d[2] || 0) * t2 * t2 * t2 + (d[3] || 0) * t2 * t2 * t2 * t2);
    const sx = r > 1e-9 ? x / r : 0, sy = r > 1e-9 ? y / r : 0;
    return [cam.fx * thd * sx + cam.cx, cam.fy * thd * sy + cam.cy];
  }
  if (z <= 0.05) return null;
  return [cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy];
}

export function unprojectCam(cam, u, v) {
  const mx = (u - cam.cx) / cam.fx, my = (v - cam.cy) / cam.fy;
  if (cam.model !== "fisheye") {
    const n = Math.hypot(mx, my, 1);
    return [mx / n, my / n, 1 / n];
  }
  const thd = Math.hypot(mx, my);
  let th = thd;
  const d = cam.D;
  for (let i = 0; i < 6; i++) {           // Newton on f(th) = th*(1+...) - thd
    const t2 = th * th;
    const f = th * (1 + (d[0] || 0) * t2 + (d[1] || 0) * t2 * t2 + (d[2] || 0) * t2 ** 3 + (d[3] || 0) * t2 ** 4) - thd;
    const fp = 1 + 3 * (d[0] || 0) * t2 + 5 * (d[1] || 0) * t2 * t2 + 7 * (d[2] || 0) * t2 ** 3 + 9 * (d[3] || 0) * t2 ** 4;
    th -= f / fp;
  }
  const s = Math.sin(th), sx = thd > 1e-9 ? mx / thd : 0, sy = thd > 1e-9 ? my / thd : 0;
  return [s * sx, s * sy, Math.cos(th)];
}

export function maxThetaOf(cam) {
  if (cam.model === "fisheye") return cam.maxTheta || (97 * Math.PI / 180);
  return Math.atan(Math.hypot(cam.width / 2 / cam.fx, cam.height / 2 / cam.fy)) * 1.02;
}

// base point -> cam coords using cam.Tinv (THREE.Matrix4)
const _v = new THREE.Vector3();
export function baseToCam(cam, x, y, z) {
  _v.set(x, y, z).applyMatrix4(cam.Tinv);
  return [_v.x, _v.y, _v.z];
}
