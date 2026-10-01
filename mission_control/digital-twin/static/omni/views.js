// OmniView — camera rig: orbit / chase / top-down / FPV / point-cloud modes
// with smooth transitions. Hand-rolled controls (no OrbitControls dependency).

import { b2t } from "./frames.js";

export const VIEW_NAMES = { 1: "ORBIT", 2: "CHASE", 3: "TOP-DOWN", 4: "FPV", 5: "POINT CLOUD" };

export class CameraRig {
  constructor(camera, dom) {
    this.camera = camera;
    this.mode = 1;
    this.az = 0.9; this.polar = 1.05; this.radius = 7.5;     // orbit params
    this.pcAz = 0.6; this.pcPolar = 0.85; this.pcRadius = 16;
    this.topH = 22;
    this.fpvCam = 0; this.fpvFov = 90;
    this.pos = camera.position.clone();
    this.look = new THREE.Vector3();
    this.up = new THREE.Vector3(0, 1, 0);
    this.fov = camera.fov;
    this._wantPos = new THREE.Vector3(); this._wantLook = new THREE.Vector3(); this._wantUp = new THREE.Vector3(0, 1, 0);
    this.dragMoved = 0;
    this._bind(dom);
  }

  _bind(dom) {
    let drag = false, lx = 0, ly = 0;
    const pts = new Map();
    let pinch0 = 0;
    dom.addEventListener("pointerdown", (e) => {
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      drag = true; lx = e.clientX; ly = e.clientY; this.dragMoved = 0;
      try { dom.setPointerCapture(e.pointerId); } catch (_) {}
    });
    dom.addEventListener("pointermove", (e) => {
      if (!drag) return;
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 2) {
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch0) this.zoom(pinch0 / d);
        pinch0 = d; this.dragMoved += 10;
        return;
      }
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      this.dragMoved += Math.abs(dx) + Math.abs(dy);
      this.rotate(dx, dy);
    });
    const end = (e) => { pts.delete(e.pointerId); if (pts.size < 2) pinch0 = 0; if (!pts.size) drag = false; };
    dom.addEventListener("pointerup", end);
    dom.addEventListener("pointercancel", end);
    dom.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
  }

  rotate(dx, dy) {
    if (this.mode === 5) { this.pcAz -= dx * 0.006; this.pcPolar = clamp(this.pcPolar - dy * 0.005, 0.1, 1.5); }
    else if (this.mode === 1 || this.mode === 2) { this.az -= dx * 0.006; this.polar = clamp(this.polar - dy * 0.005, 0.15, 1.52); if (this.mode === 2) this.mode = 1; }
  }
  zoom(f) {
    if (this.mode === 5) this.pcRadius = clamp(this.pcRadius * f, 2, 80);
    else if (this.mode === 3) this.topH = clamp(this.topH * f, 4, 80);
    else if (this.mode === 4) this.fpvFov = clamp(this.fpvFov * f, 25, 130);
    else this.radius = clamp(this.radius * f, 1.2, 40);
  }

  setMode(m) { this.mode = m; }
  setFpvCam(idx, cam) {
    this.fpvCam = idx;
    this.fpvFov = !cam ? 90 : cam.model === "fisheye" ? 105 : clamp(2 * Math.atan(cam.height / 2 / cam.fy) * 180 / Math.PI, 20, 120);
    this.mode = 4;
  }

  // robot: {pos: Vector3 (three, floor level), yaw}, cams: normalized rig
  update(dt, robot, cams) {
    const k = 1 - Math.exp(-dt * 5);
    const c = robot.pos;
    let fov = 55;
    this._wantUp.set(0, 1, 0);
    switch (this.mode) {
      case 2: {   // chase: behind the dog, follows yaw
        const back = new THREE.Vector3(-Math.cos(robot.yaw), 0, Math.sin(robot.yaw));
        this._wantPos.copy(c).addScaledVector(back, 3.4).add(new THREE.Vector3(0, 1.7, 0));
        this._wantLook.copy(c).addScaledVector(back, -2.5).add(new THREE.Vector3(0, 0.4, 0));
        break;
      }
      case 3: {   // top-down, forward = screen up
        this._wantPos.set(c.x, this.topH, c.z + 0.001);
        this._wantLook.copy(c);
        this._wantUp.set(Math.cos(robot.yaw), 0, -Math.sin(robot.yaw));
        fov = 50;
        break;
      }
      case 4: {   // FPV through selected camera
        const cam = cams && cams[this.fpvCam];
        if (cam) {
          const e = cam.T.elements;  // column-major
          const pb = [e[12], e[13], e[14]];
          const zc = [e[8], e[9], e[10]], yc = [e[4], e[5], e[6]];
          const toW = (x, y, z) => {
            const cs = Math.cos(robot.yaw), sn = Math.sin(robot.yaw);
            return [cs * x - sn * y, sn * x + cs * y, z];
          };
          const pw = toW(...pb), zw = toW(...zc), yw = toW(...yc);
          const p = b2t(pw[0], pw[1], pw[2]).add(new THREE.Vector3(c.x, robot.baseH, c.z));
          this._wantPos.copy(p);
          this._wantLook.copy(p).add(b2t(zw[0], zw[1], zw[2]));
          this._wantUp.copy(b2t(-yw[0], -yw[1], -yw[2]));
          fov = this.fpvFov;
        }
        break;
      }
      case 5: {
        this._wantPos.set(
          c.x + this.pcRadius * Math.sin(this.pcPolar) * Math.cos(this.pcAz),
          this.pcRadius * Math.cos(this.pcPolar),
          c.z + this.pcRadius * Math.sin(this.pcPolar) * Math.sin(this.pcAz));
        this._wantLook.copy(c);
        break;
      }
      default: {
        this._wantPos.set(
          c.x + this.radius * Math.sin(this.polar) * Math.cos(this.az),
          0.4 + this.radius * Math.cos(this.polar),
          c.z + this.radius * Math.sin(this.polar) * Math.sin(this.az));
        this._wantLook.copy(c).add(new THREE.Vector3(0, 0.4, 0));
      }
    }
    this.fov = fov;
    const kk = this.mode === 4 ? Math.max(k, 1 - Math.exp(-dt * 9)) : k;
    this.pos.lerp(this._wantPos, kk);
    this.look.lerp(this._wantLook, kk);
    this.up.lerp(this._wantUp, kk).normalize();
    this.camera.position.copy(this.pos);
    this.camera.up.copy(this.up);
    this.camera.lookAt(this.look);
    const nf = this.camera.fov + (this.fov - this.camera.fov) * kk;
    if (Math.abs(nf - this.camera.fov) > 0.01) { this.camera.fov = nf; this.camera.updateProjectionMatrix(); }
  }
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
