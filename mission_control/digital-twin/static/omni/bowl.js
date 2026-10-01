// OmniView — bowl surround view. GPU-side 360° stitching of up to 4 RGB
// cameras (fisheye Kannala-Brandt or pinhole) + optional thermal overlay
// (2 pinhole thermal cams, additive inferno tint). The mesh is built in ROS
// base-frame coordinates and must be placed inside the base ROS group.

import { BASE_HEIGHT, maxThetaOf } from "./frames.js";

const MAX_RGB = 4, MAX_TH = 2;

const VERT = /* glsl */`
varying vec3 vBase;
void main() {
  vBase = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex0, uTex1, uTex2, uTex3;
uniform mat4 uM[4];      // cam <- base (inverse of T_base_cam)
uniform vec4 uK[4];      // fx fy cx cy
uniform vec4 uD[4];      // k1..k4 (fisheye) / k1 k2 p1 p2 (pinhole, radial only used)
uniform vec4 uP[4];      // width height model(0=fisheye,1=pinhole) maxTheta
uniform float uGain[4];
uniform float uOn[4];
uniform sampler2D uTh0, uTh1;
uniform mat4 uThM[2];
uniform vec4 uThK[2];
uniform vec4 uThP[2];
uniform float uThOn[2];
uniform float uThermal, uNight, uOpacity, uTime, uFloorZ, uFeather, uGrid;
varying vec3 vBase;

vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// returns normalized uv in xy, weight in z (0 = not visible)
vec3 projectPt(vec3 pc, vec4 K, vec4 D, vec4 P) {
  vec2 uv;
  float w;
  if (P.z < 0.5) {
    float r = length(pc.xy);
    float th = atan(r, pc.z);
    if (th > P.w) return vec3(0.0);
    float t2 = th * th;
    float thd = th * (1.0 + D.x * t2 + D.y * t2 * t2 + D.z * t2 * t2 * t2 + D.w * t2 * t2 * t2 * t2);
    vec2 dir = r > 1e-6 ? pc.xy / r : vec2(0.0);
    uv = vec2(K.x * thd * dir.x + K.z, K.y * thd * dir.y + K.w);
    w = (1.0 - smoothstep(P.w * (1.0 - uFeather * 2.5), P.w, th)) * mix(1.0, 0.3, th / P.w);
  } else {
    if (pc.z < 0.05) return vec3(0.0);
    vec2 m = pc.xy / pc.z;
    float r2 = dot(m, m);
    m *= 1.0 + D.x * r2 + D.y * r2 * r2;
    uv = vec2(K.x * m.x + K.z, K.y * m.y + K.w);
    w = 1.0;
  }
  vec2 n = uv / P.xy;
  float e = min(min(n.x, 1.0 - n.x), min(n.y, 1.0 - n.y));
  if (e <= 0.0) return vec3(0.0);
  w *= smoothstep(0.0, uFeather, e);
  return vec3(n, w);
}

vec4 camSample(sampler2D tex, mat4 M, vec4 K, vec4 D, vec4 P, float gain, float on) {
  if (on < 0.5) return vec4(0.0);
  vec3 pc = (M * vec4(vBase, 1.0)).xyz;
  vec3 q = projectPt(pc, K, D, P);
  if (q.z <= 0.0) return vec4(0.0);
  vec3 c = texture2D(tex, q.xy).rgb * gain;
  return vec4(c * q.z, q.z);
}

float heatSample(sampler2D tex, mat4 M, vec4 K, vec4 P, float on) {
  if (on < 0.5) return 0.0;
  vec3 pc = (M * vec4(vBase, 1.0)).xyz;
  vec3 q = projectPt(pc, K, vec4(0.0), P);
  if (q.z <= 0.0) return 0.0;
  vec3 c = texture2D(tex, q.xy).rgb;
  float lum = dot(c, vec3(0.299, 0.587, 0.114));
  return smoothstep(0.38, 0.95, lum) * q.z;
}

void main() {
  vec4 acc = camSample(uTex0, uM[0], uK[0], uD[0], uP[0], uGain[0], uOn[0])
           + camSample(uTex1, uM[1], uK[1], uD[1], uP[1], uGain[1], uOn[1])
           + camSample(uTex2, uM[2], uK[2], uD[2], uP[2], uGain[2], uOn[2])
           + camSample(uTex3, uM[3], uK[3], uD[3], uP[3], uGain[3], uOn[3]);
  vec3 col = acc.a > 1e-4 ? acc.rgb / acc.a : vec3(0.0);
  float cov = clamp(acc.a * 5.0, 0.0, 1.0);

  if (uNight > 0.5) {
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    col = vec3(0.30, 1.0, 0.45) * pow(l, 0.8) * 1.35;
  }
  if (uThermal > 0.0) {
    float h = max(heatSample(uTh0, uThM[0], uThK[0], uThP[0], uThOn[0]),
                  heatSample(uTh1, uThM[1], uThK[1], uThP[1], uThOn[1]));
    col = mix(col, inferno(0.35 + 0.65 * h), h * uThermal * 0.85);
    cov = max(cov, h * uThermal);
  }
  // floor grid + scan sweep (sci-fi), only on the flat part
  float onFloor = 1.0 - smoothstep(uFloorZ + 0.02, uFloorZ + 0.25, vBase.z);
  vec2 g = abs(fract(vBase.xy) - 0.5);
  float line = 1.0 - smoothstep(0.47, 0.495, max(g.x, g.y));
  float r = length(vBase.xy);
  float sweep = smoothstep(0.6, 0.0, abs(fract(r * 0.12 - uTime * 0.25) - 0.5) * 6.0);
  vec3 gridCol = uNight > 0.5 ? vec3(0.9, 0.65, 0.15) : vec3(0.0, 0.85, 1.0);
  col += gridCol * line * onFloor * 0.18 * uGrid;
  col += gridCol * sweep * 0.06 * uGrid;
  float alpha = uOpacity * cov + line * onFloor * 0.25 * uGrid * (1.0 - cov);
  if (alpha < 0.04) discard;      // uncovered areas: no depth write, grid shows through
  gl_FragColor = vec4(col, alpha);
}`;

function buildBowlGeometry(R0 = 3.0, R = 12.0, Hrim = 5.0, nr = 64, ns = 160) {
  const floorZ = -BASE_HEIGHT + 0.012;
  const k = Hrim / ((R - R0) * (R - R0));
  const pos = new Float32Array((nr + 1) * (ns + 1) * 3);
  let p = 0;
  for (let i = 0; i <= nr; i++) {
    const t = i / nr;
    const r = R * Math.pow(t, 1.35);
    const z = floorZ + (r > R0 ? k * (r - R0) * (r - R0) : 0);
    for (let j = 0; j <= ns; j++) {
      const a = (j / ns) * Math.PI * 2;
      pos[p++] = r * Math.cos(a); pos[p++] = r * Math.sin(a); pos[p++] = z;
    }
  }
  const idx = [];
  for (let i = 0; i < nr; i++) {
    for (let j = 0; j < ns; j++) {
      const a = i * (ns + 1) + j, b = a + ns + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeBoundingSphere();
  return { geo, floorZ };
}

function makeTex() {
  const t = new THREE.Texture();
  t.flipY = false;                 // image row 0 (top) == v = 0, matches OpenCV pixel coords
  t.generateMipmaps = false;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

export class Bowl {
  constructor(parentRosGroup, quality = "med") {
    const segs = quality === "low" ? [40, 96] : quality === "high" ? [96, 240] : [64, 160];
    const { geo, floorZ } = buildBowlGeometry(3.0, 12.0, 5.0, segs[0], segs[1]);
    this.tex = [makeTex(), makeTex(), makeTex(), makeTex()];
    this.thTex = [makeTex(), makeTex()];
    const m4 = () => new THREE.Matrix4();
    const v4 = () => new THREE.Vector4(1, 1, 0, 0);
    this.uniforms = {
      uTex0: { value: this.tex[0] }, uTex1: { value: this.tex[1] },
      uTex2: { value: this.tex[2] }, uTex3: { value: this.tex[3] },
      uM: { value: [m4(), m4(), m4(), m4()] },
      uK: { value: [v4(), v4(), v4(), v4()] },
      uD: { value: [new THREE.Vector4(), new THREE.Vector4(), new THREE.Vector4(), new THREE.Vector4()] },
      uP: { value: [v4(), v4(), v4(), v4()] },
      uGain: { value: [1, 1, 1, 1] },
      uOn: { value: [0, 0, 0, 0] },
      uTh0: { value: this.thTex[0] }, uTh1: { value: this.thTex[1] },
      uThM: { value: [m4(), m4()] },
      uThK: { value: [v4(), v4()] },
      uThP: { value: [v4(), v4()] },
      uThOn: { value: [0, 0] },
      uThermal: { value: 0 }, uNight: { value: 0 }, uOpacity: { value: 0.96 },
      uTime: { value: 0 }, uFloorZ: { value: floorZ }, uFeather: { value: 0.08 }, uGrid: { value: 1 },
    };
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
      transparent: true, depthWrite: true, side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.renderOrder = -1;
    this.mesh.frustumCulled = false;
    parentRosGroup.add(this.mesh);
    this.slotOfCam = new Map();     // cam idx -> {kind:'rgb'|'th', slot}
    this.texReady = [false, false, false, false];
    this.thReady = [false, false];
  }

  setRig(cams) {
    this.slotOfCam.clear();
    const U = this.uniforms;
    let ri = 0, ti = 0;
    for (const c of cams) {
      if (c.modality === "rgb" && ri < MAX_RGB) {
        U.uM.value[ri].copy(c.Tinv);
        U.uK.value[ri].set(c.fx, c.fy, c.cx, c.cy);
        const d = c.D;
        U.uD.value[ri].set(d[0] || 0, d[1] || 0, c.model === "fisheye" ? (d[2] || 0) : 0, c.model === "fisheye" ? (d[3] || 0) : 0);
        U.uP.value[ri].set(c.width, c.height, c.model === "fisheye" ? 0 : 1, maxThetaOf(c));
        this.slotOfCam.set(c.idx, { kind: "rgb", slot: ri });
        ri++;
      } else if (c.modality === "thermal" && ti < MAX_TH) {
        U.uThM.value[ti].copy(c.Tinv);
        U.uThK.value[ti].set(c.fx, c.fy, c.cx, c.cy);
        U.uThP.value[ti].set(c.width, c.height, 1, maxThetaOf(c));
        this.slotOfCam.set(c.idx, { kind: "th", slot: ti });
        ti++;
      }
    }
    for (let i = 0; i < MAX_RGB; i++) U.uOn.value[i] = 0;
    for (let i = 0; i < MAX_TH; i++) U.uThOn.value[i] = 0;
    this.texReady = [false, false, false, false];
    this.thReady = [false, false];
  }

  // source: ImageBitmap | HTMLCanvasElement | HTMLImageElement
  setFrame(camIdx, source) {
    const s = this.slotOfCam.get(camIdx);
    if (!s) return false;
    const tex = s.kind === "rgb" ? this.tex[s.slot] : this.thTex[s.slot];
    tex.image = source;
    tex.needsUpdate = true;
    if (s.kind === "rgb") this.uniforms.uOn.value[s.slot] = 1;
    else this.uniforms.uThOn.value[s.slot] = 1;
    return true;
  }

  setGain(slot, g) { this.uniforms.uGain.value[slot] = g; }
  set thermal(v) { this.uniforms.uThermal.value = v; }
  set night(v) { this.uniforms.uNight.value = v ? 1 : 0; }
  set visible(v) { this.mesh.visible = v; }
  get visible() { return this.mesh.visible; }
  tick(t) { this.uniforms.uTime.value = t; }
}
