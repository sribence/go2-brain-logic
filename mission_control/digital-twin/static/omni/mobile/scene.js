// NERO OmniVision mobile — lightweight 3D scene (bowl + dog + persons + cloud
// + plan ribbon + goal marker), touch orbit camera. Reuses ../frames.js,
// ../bowl.js, ../robot.js, ../pointcloud.js, ../persons.js unchanged.

import * as F from "../frames.js";
import { Bowl } from "../bowl.js";
import { RobotModel } from "../robot.js";
import { VoxelCloud } from "../pointcloud.js";
import { PersonLayer } from "../persons.js";

export const QUALITY = {
  eco: { pr: 0.75, maxPts: 80_000, bowl: "low" },
  std: { pr: 1.0, maxPts: 150_000, bowl: "low" },
  hd: { pr: 1.5, maxPts: 300_000, bowl: "med" },
};

export function speedColor(v) {
  if (v < 0.28) return 0x4aa8ff;     // stealth / precise
  if (v < 0.62) return 0x35f28a;     // normal
  if (v < 1.0) return 0xffb020;
  return 0xff3b6b;                   // sprint
}

export class MobileScene {
  constructor(container, labelRoot, quality = "std") {
    const Q = QUALITY[quality] || QUALITY.std;
    this.quality = quality;
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance", alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Q.pr));
    this.renderer.setClearColor(0x02050a, 1);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x02050a, 0.02);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.05, 400);

    this.scene.add(new THREE.HemisphereLight(0x8fbaff, 0x0a0f18, 0.9));
    const sun = new THREE.DirectionalLight(0xcfe2ff, 0.7);
    sun.position.set(-4, 10, 3);
    this.scene.add(sun);

    // grid floor (cheap shader, fades out)
    this.gridMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uColor: { value: new THREE.Color(0x00d0ff) }, uCenter: { value: new THREE.Vector2() } },
      vertexShader: "varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.0); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }",
      fragmentShader: `uniform vec3 uColor; uniform vec2 uCenter; varying vec3 vW;
        float gl(vec2 p, float s, float w){ vec2 g=abs(fract(p/s-0.5)-0.5)*s; return 1.0-smoothstep(0.0,w,min(g.x,g.y)); }
        void main(){ float d=length(vW.xz-uCenter); float a = gl(vW.xz,1.0,0.03)*0.22 + gl(vW.xz,5.0,0.06)*0.45;
          gl_FragColor = vec4(uColor, (a + 0.03) * (1.0 - smoothstep(12.0, 50.0, d))); }`,
    });
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), this.gridMat);
    grid.rotation.x = -Math.PI / 2;
    grid.renderOrder = -2;
    this.scene.add(grid);

    this.robotYaw = new THREE.Group();
    this.scene.add(this.robotYaw);
    this.baseRos = F.makeRosGroup();
    this.baseRos.position.y = F.BASE_HEIGHT;
    this.robotYaw.add(this.baseRos);
    this.worldRos = F.makeRosGroup();
    this.worldRos.position.y = F.BASE_HEIGHT;
    this.scene.add(this.worldRos);

    this.bowl = new Bowl(this.baseRos, Q.bowl);
    this.robot = new RobotModel(this.baseRos);
    this.cloud = new VoxelCloud(this.worldRos, Q.maxPts);
    this.cloud.setMaxPx(6);
    this.persons = new PersonLayer(this.scene, labelRoot, { onSelect: () => {} });

    // safety rings (STOP / SLOW / CAUTION)
    this.rings = [[F.ZONES.STOP, 0xff3344], [F.ZONES.SLOW, 0xff8a1e], [F.ZONES.CAUTION, 0xffd23a]].map(([r, c]) => {
      const m = new THREE.Mesh(new THREE.RingGeometry(r - 0.03, r + 0.03, 72),
        new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false }));
      m.rotation.x = -Math.PI / 2; m.position.y = 0.02;
      this.robotYaw.add(m);
      return m;
    });

    // plan ribbon (vertex-coloured triangle strip, drawn on top of the bowl)
    this.ribbonGeo = new THREE.BufferGeometry();
    this.ribbon = new THREE.Mesh(this.ribbonGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.92, depthTest: false, side: THREE.DoubleSide }));
    this.ribbon.renderOrder = 10;
    this.ribbon.frustumCulled = false;
    this.ribbon.visible = false;
    this.scene.add(this.ribbon);
    this.landings = new THREE.Group();
    this.scene.add(this.landings);

    // goal marker: ring + yaw arrow
    this.goal = new THREE.Group();
    const gm = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthTest: false, side: THREE.DoubleSide });
    this.goalMat = gm;
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.36, 40), gm);
    ring.rotation.x = -Math.PI / 2;
    this.goal.add(ring);
    const arrowShape = new THREE.Shape();
    arrowShape.moveTo(0.75, 0); arrowShape.lineTo(0.42, 0.16); arrowShape.lineTo(0.42, -0.16); arrowShape.lineTo(0.75, 0);
    const arrow = new THREE.Mesh(new THREE.ShapeGeometry(arrowShape), gm);
    arrow.rotation.x = -Math.PI / 2;
    this.goalArrow = new THREE.Group();
    this.goalArrow.add(arrow);
    this.goal.add(this.goalArrow);
    this.goal.position.y = 0.04;
    this.goal.renderOrder = 11;
    this.goal.traverse((o) => { o.renderOrder = 11; });
    this.goal.visible = false;
    this.scene.add(this.goal);

    // gamepad cursor
    this.cursor = new THREE.Mesh(new THREE.RingGeometry(0.18, 0.24, 4),
      new THREE.MeshBasicMaterial({ color: 0xff2bd6, transparent: true, opacity: 0.9, depthTest: false, side: THREE.DoubleSide }));
    this.cursor.rotation.x = -Math.PI / 2;
    this.cursor.renderOrder = 12;
    this.cursor.visible = false;
    this.scene.add(this.cursor);

    // orbit camera state (around the robot)
    this.orbit = { az: Math.PI * 0.75, el: 0.62, dist: 7.5, follow: true, target: new THREE.Vector3() };
    this.ray = new THREE.Raycaster();
    this.floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.pose = { x: 0, y: 0, yaw: 0 };
    this._robotPos = new THREE.Vector3();
    this.clock = new THREE.Clock();
    this.paused = false;
    this.fps = 0; this._frames = 0; this._fpsT = 0;
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth || innerWidth, h = this.container.clientHeight || innerHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
    this.cloud.setScale(h * this.renderer.getPixelRatio(), this.camera.fov);
    this.w = w; this.h = h;
  }

  setQuality(q) {
    const Q = QUALITY[q] || QUALITY.std;
    this.quality = q;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Q.pr));
    this.cloud.maxPoints = Q.maxPts;
    this.resize();
  }

  setPose(p) { if (p && Number.isFinite(+p.x)) this.pose = { x: +p.x, y: +p.y, yaw: +p.yaw || 0 }; }

  // screen (client px) -> {ndc}
  _ndc(cx, cy) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: ((cx - r.left) / r.width) * 2 - 1, y: -((cy - r.top) / r.height) * 2 + 1 };
  }
  // -> world {x, y} on the floor plane or null
  floorAt(cx, cy) {
    this.ray.setFromCamera(this._ndc(cx, cy), this.camera);
    const p = new THREE.Vector3();
    if (!this.ray.ray.intersectPlane(this.floor, p)) return null;
    const w = F.t2b(p);
    if (Math.hypot(w.x - this.pose.x, w.y - this.pose.y) > 80) return null;
    return { x: w.x, y: w.y };
  }
  personAt(cx, cy) {
    this.ray.setFromCamera(this._ndc(cx, cy), this.camera);
    const gid = this.persons.pick(this.ray);
    if (gid != null) return gid;
    // fat-finger: nearest person within 36 px of the touch
    let best = null, bd = 36;
    const v = new THREE.Vector3();
    const r = this.renderer.domElement.getBoundingClientRect();
    for (const it of this.persons.items.values()) {
      if (!it.pos) continue;
      for (const hgt of [0.2, 0.9, 1.6]) {
        v.set(it.pos.x, hgt, it.pos.z).project(this.camera);
        if (v.z > 1) continue;
        const d = Math.hypot(r.left + (v.x + 1) / 2 * r.width - cx, r.top + (1 - v.y) / 2 * r.height - cy);
        if (d < bd) { bd = d; best = it.gid; }
      }
    }
    return best;
  }
  personWorld(gid) { const p = this.persons.list.find((q) => q.gid === gid); return p ? { x: p.wx, y: p.wy, range: p.range } : null; }

  setGoal(g) {
    if (!g) { this.goal.visible = false; return; }
    F.b2t(g.x, g.y, 0.04, this.goal.position);
    this.goalArrow.visible = g.yaw != null;
    if (g.yaw != null) this.goalArrow.rotation.y = g.yaw;
    this.goalMat.color.setHex(g.color || 0xffffff);
    this.goal.visible = true;
  }
  setCursor(c) {
    if (!c) { this.cursor.visible = false; return; }
    F.b2t(c.x, c.y, 0.05, this.cursor.position);
    this.cursor.visible = true;
  }

  // plan: {path:[[x,y]], v:[...], landings:[[x,y]]}
  setPlan(plan) {
    for (const c of [...this.landings.children]) { this.landings.remove(c); c.geometry.dispose(); }
    const pts = plan && plan.path ? plan.path : [];
    if (pts.length < 2) { this.ribbon.visible = false; }
    else {
      const n = pts.length, hw = 0.07;
      const pos = new Float32Array(n * 2 * 3), col = new Float32Array(n * 2 * 3), idx = [];
      const c = new THREE.Color();
      for (let i = 0; i < n; i++) {
        const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
        let tx = b[0] - a[0], ty = b[1] - a[1];
        const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
        const nx = -ty * hw, ny = tx * hw;
        const [x, y] = pts[i];
        // world (x,y) -> three (x, h, -y)
        pos.set([x + nx, 0.06, -(y + ny), x - nx, 0.06, -(y - ny)], i * 6);
        c.setHex(speedColor(plan.v ? plan.v[i] || 0 : 0.6));
        col.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
        if (i < n - 1) { const k = i * 2; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
      }
      this.ribbonGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      this.ribbonGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      this.ribbonGeo.setIndex(idx);
      this.ribbonGeo.computeBoundingSphere();
      this.ribbon.visible = true;
    }
    for (const l of (plan && plan.landings) || []) {
      const m = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.22, 24), new THREE.MeshBasicMaterial({ color: 0xff2bd6, depthTest: false, side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2; F.b2t(l[0], l[1], 0.07, m.position); m.renderOrder = 11;
      this.landings.add(m);
    }
  }

  // two-finger gesture deltas
  orbitBy(dAz, dEl, zoomF) {
    const o = this.orbit;
    o.az += dAz;
    o.el = Math.max(0.12, Math.min(1.45, o.el + dEl));
    o.dist = Math.max(1.8, Math.min(45, o.dist * zoomF));
  }
  panBy(dxPx, dyPx) {   // move the orbit target in the camera's ground plane
    const o = this.orbit;
    const k = o.dist / Math.max(this.h, 1) * 1.2;
    const fx = -Math.sin(o.az), fz = -Math.cos(o.az);   // forward (into screen) on ground
    const rx = Math.cos(o.az), rz = -Math.sin(o.az);
    o.target.x += (-dxPx * rx + dyPx * fx) * k;
    o.target.z += (-dxPx * rz + dyPx * fz) * k;
    o.follow = false;
  }
  recenter() { this.orbit.follow = true; }
  // camera heading in world frame (for gamepad cursor: stick up = away from camera)
  get viewYaw() { const o = this.orbit; return Math.atan2(Math.cos(o.az), -Math.sin(o.az)); }

  setSafetyLevel(lvl) { this.level = lvl; }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const t = this.clock.elapsedTime;
    F.b2t(this.pose.x, this.pose.y, 0, this._robotPos);
    this.robotYaw.position.lerp(this._robotPos, 1 - Math.exp(-dt * 10));
    let dy = this.pose.yaw - this.robotYaw.rotation.y;
    dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    this.robotYaw.rotation.y += dy * (1 - Math.exp(-dt * 10));
    this.gridMat.uniforms.uCenter.value.set(this.robotYaw.position.x, this.robotYaw.position.z);
    this.robot.setJoints(null, t);
    const lvl = this.level;
    const pulse = 0.5 + 0.5 * Math.sin(t * 8);
    this.rings.forEach((m, i) => {
      const act = (lvl === "STOP" && i === 0) || (lvl === "SLOW" && i === 1) || (lvl === "CAUTION" && i === 2);
      m.material.opacity = act ? 0.5 + 0.5 * pulse : 0.28;
    });
    if (this.goal.visible) this.goal.children[0].scale.setScalar(1 + 0.1 * Math.sin(t * 5));
    if (this.cursor.visible) this.cursor.rotation.z = t * 2;
    this.bowl.tick(t);
    this.cloud.tick(t);
    this.persons.tick(dt, performance.now() / 1000, this.camera, this.w, this.h);

    const o = this.orbit;
    if (o.follow) o.target.lerp(this.robotYaw.position, 1 - Math.exp(-dt * 6));
    const ce = Math.cos(o.el);
    this.camera.position.set(o.target.x + o.dist * ce * Math.sin(o.az), o.target.y + o.dist * Math.sin(o.el), o.target.z + o.dist * ce * Math.cos(o.az));
    this.camera.lookAt(o.target.x, o.target.y + 0.3, o.target.z);
    if (!this.paused) this.renderer.render(this.scene, this.camera);
    this._frames++;
    if (t - this._fpsT >= 1) { this.fps = this._frames / (t - this._fpsT); this._frames = 0; this._fpsT = t; }
    return dt;
  }
}
