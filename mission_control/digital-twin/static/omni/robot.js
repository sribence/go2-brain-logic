// OmniView — Go2 model. Minimal URDF parser + ColladaLoader meshes, built
// in ROS coordinates (put it in a ROS group). Fallback: procedural box-dog
// (always present until the real meshes are loaded).

const URDF_URL = "/static/go2_description/go2_description.urdf";
const MESH_BASE = "/static/go2_description/meshes/";
const LEG_NAMES = ["FR", "FL", "RR", "RL"];
// Standing pose (hip, thigh, calf) — base ~0.33 m above floor.
const STAND = [0.0, 0.8, -1.5];

function parseXYZ(s, def = [0, 0, 0]) {
  if (!s) return def.slice();
  const a = s.trim().split(/\s+/).map(Number);
  return [a[0] || 0, a[1] || 0, a[2] || 0];
}

function applyOrigin(obj, originEl) {
  if (!originEl) return;
  const xyz = parseXYZ(originEl.getAttribute("xyz"));
  const rpy = parseXYZ(originEl.getAttribute("rpy"));
  obj.position.set(xyz[0], xyz[1], xyz[2]);
  obj.rotation.set(rpy[0], rpy[1], rpy[2], "ZYX");   // URDF: R = Rz(y) Ry(p) Rx(r)
}

export class RobotModel {
  constructor(parentRosGroup) {
    this.root = new THREE.Group();
    parentRosGroup.add(this.root);
    this.joints = {};          // name -> {pivot, axis:Vector3}
    this.loaded = false;
    this.material = new THREE.MeshStandardMaterial({
      color: 0x39424f, metalness: 0.55, roughness: 0.38, emissive: 0x04121c,
    });
    this.accent = new THREE.MeshStandardMaterial({
      color: 0x10161e, metalness: 0.3, roughness: 0.6, emissive: 0x00b4ff, emissiveIntensity: 0.35,
    });
    this._buildFallback();
  }

  _buildFallback() {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.19, 0.11), this.material);
    g.add(body);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.12, 0.08), this.accent);
    head.position.set(0.23, 0, 0.02);
    g.add(head);
    const legGeo = new THREE.BoxGeometry(0.04, 0.04, 0.32);
    for (const [x, y] of [[0.19, 0.12], [0.19, -0.12], [-0.19, 0.12], [-0.19, -0.12]]) {
      const leg = new THREE.Mesh(legGeo, this.material);
      leg.position.set(x, y, -0.17);
      g.add(leg);
    }
    this.fallback = g;
    this.root.add(g);
  }

  async load() {
    if (typeof THREE.ColladaLoader !== "function") throw new Error("ColladaLoader not available");
    const res = await fetch(URDF_URL);
    if (!res.ok) throw new Error(`URDF HTTP ${res.status}`);
    const xml = new DOMParser().parseFromString(await res.text(), "application/xml");
    const robotEl = xml.querySelector("robot");
    if (!robotEl) throw new Error("bad URDF");
    const links = {}, jointsByChild = {}, children = {};
    for (const l of robotEl.children) if (l.tagName === "link") links[l.getAttribute("name")] = l;
    for (const j of robotEl.children) {
      if (j.tagName !== "joint") continue;
      const parent = j.querySelector("parent").getAttribute("link");
      const child = j.querySelector("child").getAttribute("link");
      jointsByChild[child] = j;
      (children[parent] = children[parent] || []).push(child);
    }
    const rootName = Object.keys(links).find((n) => !jointsByChild[n]) || "base";

    const loader = new THREE.ColladaLoader();
    const cache = {};
    const loadDae = (file) => {
      if (!cache[file]) {
        cache[file] = new Promise((resolve, reject) => {
          loader.load(MESH_BASE + file, (c) => {
            const s = c.scene;
            s.quaternion.identity();     // undo Z_UP->Y_UP: URDF meshes are already in link (Z-up) frame
            s.updateMatrix();
            resolve(s);
          }, undefined, reject);
        });
      }
      return cache[file];
    };

    const meshJobs = [];
    const buildLink = (name) => {
      const g = new THREE.Group();
      g.name = name;
      const linkEl = links[name];
      if (linkEl) {
        for (const vis of linkEl.querySelectorAll(":scope > visual")) {
          const meshEl = vis.querySelector("geometry > mesh");
          if (!meshEl) continue;
          const file = (meshEl.getAttribute("filename") || "").split("/").pop();
          if (!file) continue;
          const holder = new THREE.Group();
          applyOrigin(holder, vis.querySelector("origin"));
          g.add(holder);
          meshJobs.push(loadDae(file).then((tpl) => {
            const inst = tpl.clone(true);
            const mat = /calf|foot/.test(file) ? this.accent : this.material;
            inst.traverse((o) => { if (o.isMesh) { o.material = mat; o.castShadow = false; } });
            holder.add(inst);
          }));
        }
      }
      for (const ch of children[name] || []) {
        const j = jointsByChild[ch];
        const frame = new THREE.Group();
        applyOrigin(frame, j.querySelector("origin"));
        const pivot = new THREE.Group();
        frame.add(pivot);
        const axisEl = j.querySelector("axis");
        const ax = parseXYZ(axisEl && axisEl.getAttribute("xyz"), [1, 0, 0]);
        const axis = new THREE.Vector3(ax[0], ax[1], ax[2]);
        if (axis.lengthSq() > 0) axis.normalize();
        const type = j.getAttribute("type");
        if (type === "revolute" || type === "continuous") this.joints[j.getAttribute("name")] = { pivot, axis };
        pivot.add(buildLink(ch));
        g.add(frame);
      }
      return g;
    };
    const urdfRoot = buildLink(rootName);
    await Promise.all(meshJobs);
    this.root.add(urdfRoot);
    this.fallback.visible = false;
    this.loaded = true;
    this.setJoints(null);
    return true;
  }

  // q: 12 values in FR,FL,RR,RL × (hip, thigh, calf) order, or null for idle stand.
  setJoints(q, t = 0) {
    for (let i = 0; i < 4; i++) {
      const leg = LEG_NAMES[i];
      const parts = ["hip", "thigh", "calf"];
      for (let k = 0; k < 3; k++) {
        const j = this.joints[`${leg}_${parts[k]}_joint`];
        if (!j) continue;
        let a;
        if (q && q.length >= 12 && Number.isFinite(q[i * 3 + k])) a = q[i * 3 + k];
        else {
          // idle "breathing" animation
          const br = Math.sin(t * 1.6 + (i % 2) * 0.4) * 0.03;
          a = STAND[k] + (k === 1 ? br : k === 2 ? -br * 1.6 : 0);
        }
        j.pivot.quaternion.setFromAxisAngle(j.axis, a);
      }
    }
  }
}
