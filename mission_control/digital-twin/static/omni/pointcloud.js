// OmniView — colored voxel point cloud (OVX1 decoder + upsert buffers).
//
// OVX1 (little-endian): "OVX1" | u32 version | u32 count | f32 res |
//   count × { f32 x, f32 y, f32 z, u8 r, u8 g, u8 b, u8 flags }   (16 B)
// flags bit0 = thermal hot (>28 °C), bit1 = near a person. WORLD frame.

export function decodeOVX1(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < 16) return null;
  const dv = new DataView(buf);
  if (dv.getUint8(0) !== 0x4f || dv.getUint8(1) !== 0x56 || dv.getUint8(2) !== 0x58 || dv.getUint8(3) !== 0x31) return null;
  const version = dv.getUint32(4, true);
  let count = dv.getUint32(8, true);
  const res = dv.getFloat32(12, true);
  count = Math.min(count, Math.floor((buf.byteLength - 16) / 16));
  return { version, count, res, dv };
}

export function encodeOVX1(version, res, pts) {   // pts: [[x,y,z,r,g,b,flags],...] (demo/tests)
  const buf = new ArrayBuffer(16 + 16 * pts.length);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x4f); dv.setUint8(1, 0x56); dv.setUint8(2, 0x58); dv.setUint8(3, 0x31);
  dv.setUint32(4, version, true); dv.setUint32(8, pts.length, true); dv.setFloat32(12, res, true);
  let o = 16;
  for (const p of pts) {
    dv.setFloat32(o, p[0], true); dv.setFloat32(o + 4, p[1], true); dv.setFloat32(o + 8, p[2], true);
    dv.setUint8(o + 12, p[3]); dv.setUint8(o + 13, p[4]); dv.setUint8(o + 14, p[5]); dv.setUint8(o + 15, p[6] || 0);
    o += 16;
  }
  return buf;
}

const VERT = /* glsl */`
attribute vec3 acolor;
attribute float aflags;
uniform float uSize, uScale, uMode, uTime, uZmin, uZmax, uMaxPx;
varying vec3 vCol;
varying float vHot;
vec3 turbo(float t) {
  t = clamp(t, 0.0, 1.0);
  return clamp(vec3(
    0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.5666 - t * 58.1375)))),
    0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 + t * 2.7747)))),
    0.1067 + t * (12.5925 - t * (60.1097 - t * (109.0745 - t * (88.5066 - t * 26.8183))))), 0.0, 1.0);
}
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float hot = mod(aflags, 2.0);
  float nearP = mod(floor(aflags / 2.0), 2.0);
  vec3 c = acolor;
  if (uMode > 1.5) {
    c = turbo((position.z - uZmin) / max(uZmax - uZmin, 0.01));
  } else if (uMode > 0.5) {
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = hot > 0.5 ? mix(vec3(1.0, 0.35, 0.05), vec3(1.0, 0.95, 0.6), 0.5 + 0.5 * sin(uTime * 5.0)) : vec3(0.12, 0.16, 0.24) * (0.5 + l);
  } else if (nearP > 0.5) {
    c = mix(c, vec3(1.0, 0.2, 0.8), 0.35);
  }
  vCol = c;
  vHot = hot;
  gl_PointSize = clamp(uSize * uScale / max(-mv.z, 0.05), 1.0, uMaxPx) * (uMode > 0.5 && uMode < 1.5 && hot > 0.5 ? 1.6 : 1.0);
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */`
varying vec3 vCol;
varying float vHot;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  gl_FragColor = vec4(vCol * (1.0 - r2 * 1.2), 1.0);
}`;

export class VoxelCloud {
  constructor(parentRosGroup, maxPoints = 1_000_000) {
    this.parent = parentRosGroup;
    this.maxPoints = maxPoints;
    this.count = 0;
    this.cap = 0;
    this.index = new Map();       // quantized key -> buffer index
    this.res = 0.05;
    this.version = 0;
    this.dropped = 0;
    this.zMin = -0.4; this.zMax = 2.5;
    this.uniforms = {
      uSize: { value: 0.045 }, uScale: { value: 600 }, uMode: { value: 0 }, uTime: { value: 0 },
      uZmin: { value: -0.35 }, uZmax: { value: 2.5 }, uMaxPx: { value: 22 },
    };
    this.material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms, transparent: true });
    this.points = null;
    this._alloc(65536);
  }

  _alloc(cap) {
    const pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), fl = new Uint8Array(cap);
    if (this.points) {
      pos.set(this.pos.subarray(0, this.count * 3));
      col.set(this.col.subarray(0, this.count * 3));
      fl.set(this.fl.subarray(0, this.count));
      this.parent.remove(this.points);
      this.points.geometry.dispose();
    }
    this.pos = pos; this.col = col; this.fl = fl; this.cap = cap;
    const geo = new THREE.BufferGeometry();
    this.aPos = new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage);
    this.aCol = new THREE.BufferAttribute(col, 3, true).setUsage(THREE.DynamicDrawUsage);
    this.aFl = new THREE.BufferAttribute(fl, 1, false).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", this.aPos);
    geo.setAttribute("acolor", this.aCol);
    geo.setAttribute("aflags", this.aFl);
    geo.setDrawRange(0, this.count);
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 1;   // after the (transparent, depth-writing) bowl -> occluded by it
    this.parent.add(this.points);
  }

  _key(x, y, z) {
    const r = this.res;
    const ix = Math.round(x / r) + 524288, iy = Math.round(y / r) + 524288, iz = Math.round(z / r) + 2048;
    return (ix * 1048576 + iy) * 4096 + iz;   // < 2^52, exact in float64
  }

  clear() {
    this.count = 0; this.index.clear(); this.dropped = 0;
    this.points.geometry.setDrawRange(0, 0);
  }

  // Returns number of points touched. isSnapshot => clear first.
  ingest(buf, isSnapshot = false) {
    const h = decodeOVX1(buf);
    if (!h) return 0;
    if (isSnapshot) this.clear();
    if (h.res > 0) this.res = h.res;
    this.version = h.version;
    const dv = h.dv;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0, o = 16; i < h.count; i++, o += 16) {
      const x = dv.getFloat32(o, true), y = dv.getFloat32(o + 4, true), z = dv.getFloat32(o + 8, true);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const k = this._key(x, y, z);
      let idx = this.index.get(k);
      if (idx === undefined) {
        if (this.count >= this.maxPoints) { this.dropped++; continue; }
        if (this.count >= this.cap) this._alloc(Math.min(this.cap * 2, Math.max(this.maxPoints, 65536)));
        idx = this.count++;
        this.index.set(k, idx);
      }
      const p = idx * 3;
      this.pos[p] = x; this.pos[p + 1] = y; this.pos[p + 2] = z;
      this.col[p] = dv.getUint8(o + 12); this.col[p + 1] = dv.getUint8(o + 13); this.col[p + 2] = dv.getUint8(o + 14);
      this.fl[idx] = dv.getUint8(o + 15);
      if (z < lo) lo = z; if (z > hi) hi = z;
    }
    if (lo < this.zMin) this.zMin = lo;
    if (hi > this.zMax) this.zMax = Math.min(hi, 6);
    this.uniforms.uZmin.value = this.zMin; this.uniforms.uZmax.value = this.zMax;
    this.aPos.needsUpdate = true; this.aCol.needsUpdate = true; this.aFl.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.count);
    return h.count;
  }

  // big sprites poke through the bowl at grazing angles -> keep them small while the bowl is shown
  setMaxPx(px) { this.uniforms.uMaxPx.value = px; }
  setMode(m) { this.uniforms.uMode.value = m; }
  setScale(viewportH, fovDeg) { this.uniforms.uScale.value = viewportH / (2 * Math.tan((fovDeg * Math.PI) / 360)); }
  tick(t) { this.uniforms.uTime.value = t; }
  set visible(v) { if (this.points) this.points.visible = v; }
}
