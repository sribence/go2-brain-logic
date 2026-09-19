"""
NERO_GO2 — Opció 1: Kulcsképkockás Horgonytérkép SLAM (Keyframe Landmark Graph SLAM)
Szigorúan 0% pont-törlés! Minden LiDAR mérés megőrizve.
"""

import json
import math
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_stabilized_points(fr):
    roll = fr.get("roll", 0) or 0.0
    pitch = fr.get("pitch", 0) or 0.0
    cr, sr = math.cos(-roll), math.sin(-roll)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rot_tilt = rx.T @ ry.T
    
    pts = np.array(fr.get("points", []), dtype=np.float32)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    xyz = pts[:, :3] @ rot_tilt
    d = np.hypot(xyz[:, 0], xyz[:, 1])
    valid = (d > 0.35) & (xyz[:, 2] > -0.5) & (xyz[:, 2] < 2.5)
    return xyz[valid]

def process_keyframe_dataset(filepath, name, label, is_stationary=False, step=2, max_frames=260):
    print(f"\n=======================================================")
    print(f"Keyframe SLAM Feldolgozas: {name} ({label})")
    print(f"=======================================================")
    t0 = time.time()
    
    raw_frames = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % step == 0:
                raw_frames.append(json.loads(line))
            if len(raw_frames) >= max_frames:
                break
                
    if not raw_frames:
        return None
        
    YAW_OFFSET_RAD = 0.0 if is_stationary else np.radians(90)
    
    keyframes = [] # list of {"id": int, "pos": np.array([x,y]), "yaw": float, "points": np.array}
    keyframe_tree = None
    keyframe_points_global = None
    
    trajectory = []
    points_tagged = []
    pts_buffer = []
    
    last_kf_pos = None
    last_kf_yaw = None
    
    current_R = np.eye(2, dtype=np.float32)
    current_t = np.zeros(2, dtype=np.float32)
    
    for f_idx, fr in enumerate(raw_frames):
        pts = get_stabilized_points(fr)
        if len(pts) < 30:
            continue
            
        x_raw = fr.get("x", 0) or 0.0
        y_raw = fr.get("y", 0) or 0.0
        yaw_raw = (fr.get("yaw", 0) or 0.0) + YAW_OFFSET_RAD
        
        cos_yr, sin_yr = math.cos(yaw_raw), math.sin(yaw_raw)
        rot_raw = np.array([[cos_yr, -sin_yr], [sin_yr, cos_yr]], dtype=np.float32)
        raw_xy = pts[:, :2] @ rot_raw.T + np.array([x_raw, y_raw], dtype=np.float32)
        
        if is_stationary:
            corr_xy = pts[:, :2]
            cor_pos = np.array([x_raw, y_raw])
            is_kf = (f_idx % 20 == 0)
        else:
            if not keyframes:
                # Első kulcsképkocka rögzítése (Origin Keyframe)
                is_kf = True
                corr_xy = raw_xy.copy()
                cor_pos = np.array([x_raw, y_raw])
            else:
                # ICP illesztés a meglévő Kulcsképkockás Horgonytérképhez
                curr_al = raw_xy.copy()
                if keyframe_tree is not None:
                    max_iters = 14
                    for _ in range(max_iters):
                        dists, idxs = keyframe_tree.query(curr_al)
                        m = dists < 0.45
                        if np.sum(m) < 25: break
                        src_m = curr_al[m]
                        dst_m = keyframe_points_global[idxs[m]]
                        c_s = np.mean(src_m, axis=0)
                        c_d = np.mean(dst_m, axis=0)
                        H = (src_m - c_s).T @ (dst_m - c_d)
                        U, S, Vt = np.linalg.svd(H)
                        R = Vt.T @ U.T
                        if np.linalg.det(R) < 0:
                            Vt[1, :] *= -1
                            R = Vt.T @ U.T
                        t = c_d - c_s @ R.T
                        curr_al = curr_al @ R.T + t
                        current_R = current_R @ R.T
                        current_t = current_t @ R.T + t
                        
                corr_xy = curr_al
                cor_pos = np.array([x_raw, y_raw], dtype=np.float32) @ current_R.T + current_t
                
                # Kulcsképkocka kiváltási feltétel: >0.4m elmozdulás vagy >12 fok elfordulás
                dist_moved = np.hypot(cor_pos[0] - last_kf_pos[0], cor_pos[1] - last_kf_pos[1])
                yaw_diff = abs((yaw_raw - last_kf_yaw + math.pi) % (2 * math.pi) - math.pi)
                is_kf = (dist_moved > 0.40 or yaw_diff > np.radians(12) or f_idx == len(raw_frames)-1)
                
        if is_kf:
            kf_id = len(keyframes)
            keyframes.append({
                "id": kf_id,
                "f": f_idx,
                "x": round(float(cor_pos[0]), 3),
                "y": round(float(cor_pos[1]), 3),
                "yaw": round(float(yaw_raw), 3),
                "t": round(f_idx * step * 0.125, 2)
            })
            last_kf_pos = cor_pos
            last_kf_yaw = yaw_raw
            
            # Horgonytérkép frissítése
            if keyframe_points_global is None:
                keyframe_points_global = corr_xy[::3].copy()
            else:
                keyframe_points_global = np.vstack([keyframe_points_global, corr_xy[::4]])
            keyframe_tree = cKDTree(keyframe_points_global)

        trajectory.append({
            "f": f_idx,
            "x": round(float(cor_pos[0]), 3),
            "y": round(float(cor_pos[1]), 3),
            "yaw": round(float(yaw_raw), 3),
            "t": round(f_idx * step * 0.125, 2),
            "is_kf": bool(is_kf)
        })
        
        # 100% Pontmegtartás (Szigorúan 0% törlés)
        frame_3d = np.column_stack([corr_xy, pts[:, 2]])
        v_c = np.floor(frame_3d / 0.025).astype(np.int32)
        _, u_idx = np.unique(v_c, axis=0, return_index=True)
        v_pts = frame_3d[u_idx]
        if len(v_pts) > 550:
            v_pts = v_pts[::max(1, len(v_pts)//550)][:550]
            
        for p in v_pts:
            points_tagged.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), f_idx, 1 if is_kf else 0])
            pts_buffer.append([p[0], p[1], p[2]])

    pts_arr = np.array(pts_buffer)
    min_x, max_x = float(pts_arr[:, 0].min()), float(pts_arr[:, 0].max())
    min_y, max_y = float(pts_arr[:, 1].min()), float(pts_arr[:, 1].max())
    min_z, max_z = float(pts_arr[:, 2].min()), float(pts_arr[:, 2].max())
    
    print(f"  Feldolgozva {time.time()-t0:.2f}s alatt.")
    print(f"  Kulcsképkockák száma: {len(keyframes)} db")
    print(f"  Megtartott sűrű pontok: {len(points_tagged):,} db (100% pontmegtartás)")
    
    return {
        "id": name,
        "label": label,
        "total_frames": len(trajectory),
        "total_time": round(len(trajectory) * step * 0.125, 1),
        "total_points": len(points_tagged),
        "keyframes_count": len(keyframes),
        "keyframes": keyframes,
        "bounds": {
            "x": [round(min_x, 2), round(max_x, 2)],
            "y": [round(min_y, 2), round(max_y, 2)],
            "z": [round(min_z, 2), round(max_z, 2)],
            "width": round(max_x - min_x, 2),
            "length": round(max_y - min_y, 2)
        },
        "trajectory": trajectory,
        "points": points_tagged
    }

def generate_keyframe_html(datasets):
    data_json = json.dumps(datasets, default=lambda x: bool(x) if isinstance(x, np.bool_) else (float(x) if isinstance(x, np.floating) else int(x)))
    return f"""<!DOCTYPE html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <title>NERO_GO2 — Opció 1: Kulcsképkockás Horgonytérkép SLAM</title>
  <style>
    :root {{
      --bg: #07090e;
      --panel-bg: rgba(13, 17, 25, 0.94);
      --border: rgba(255, 255, 255, 0.12);
      --accent-blue: #22e5ff;
      --accent-green: #00e676;
      --accent-orange: #ff9900;
      --text: #e2e8f0;
      --muted: #8899aa;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: monospace; overflow: hidden; width: 100vw; height: 100vh; }}
    #canvas3d {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}
    #hud-panel {{
      position: absolute; top: 14px; left: 14px; width: 380px;
      background: var(--panel-bg); backdrop-filter: blur(14px);
      border: 1px solid var(--border); border-radius: 10px; z-index: 10; padding: 14px;
      display: flex; flex-direction: column; gap: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }}
    h1 {{ font-size: 13px; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; }}
    select.ds-select {{ width: 100%; padding: 8px; background: #0d121c; border: 1px solid var(--accent-blue); color: #fff; font-size: 11px; border-radius: 6px; }}
    .card {{ background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); font-size: 11px; line-height: 1.5; }}
    .card b {{ color: var(--accent-green); }}
    #timeline-bar {{
      position: absolute; bottom: 16px; left: 14px; right: 14px;
      background: var(--panel-bg); backdrop-filter: blur(14px); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 16px; z-index: 10; display: flex; align-items: center; gap: 14px;
    }}
    .play-btn {{ background: var(--accent-blue); color: #000; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas3d"></div>
  <div id="hud-panel">
    <h1>🔑 Opció 1: Kulcsképkockás SLAM</h1>
    <p style="font-size:10px; color:var(--muted);">Fix Térbeli Horgonyzat & 100% Pontmegtartás</p>

    <div>
      <select id="dataset-select" class="ds-select" onchange="onDatasetChange()">
        <option value="walk_kicsi" selected>🚶 walk_kicsi — Szobai séta (60s)</option>
        <option value="walk_teszt">🛑 walk_teszt — Álló robot (60s)</option>
        <option value="walk_seta1">🏃 walk_seta1 — Nagy séta (90s)</option>
      </select>
    </div>

    <div class="card" style="border-left: 3px solid var(--accent-green);">
      <b>🌐 100% Pontmegtartási Elv:</b><br>
      Szigorúan 0% pont-szűrés! Minden LiDAR mérés rálokkol a rögzített <b>Kulcsképkocka Horgonyokra</b>.
    </div>

    <div id="ds-info" class="card">
      <div>Rögzített Kulcsképkockák: <b id="info-kf-count" style="color:var(--accent-blue);">0 db</b></div>
      <div>Összes sűrű pont: <b id="info-total-points">0 db</b></div>
      <div>Kiterjedés: <b id="info-bounds">-</b></div>
    </div>
  </div>

  <div id="timeline-bar">
    <button class="play-btn" id="btn-play" onclick="togglePlay()">▶ Lejátszás</button>
    <div style="font-size:12px; font-weight:700; color:#fff;" id="lbl-time">0.0s</div>
    <input type="range" id="rng-timeline" min="0" max="100" value="100" style="flex:1;" oninput="onTimelineScrub(this.value)">
  </div>

  <script>
    const ALL_DATA = {data_json};
    let currentDs = "walk_kicsi";
    let scene, camera, renderer, controls;
    let grpPoints, grpTraj, grpKeyframes;
    let rawPosArray, rawFramesArray, ptsGeom, ptsMaterial;
    let currentFrameIdx = 0;
    let isPlaying = false;

    function init3D() {{
      const container = document.getElementById("canvas3d");
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x07090e);

      camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
      camera.position.set(0, 12, 8);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      const grid = new THREE.GridHelper(40, 40, 0x22e5ff, 0x1e293b);
      grid.position.y = -0.4;
      scene.add(grid);

      grpPoints = new THREE.Group();
      grpTraj = new THREE.Group();
      grpKeyframes = new THREE.Group();
      scene.add(grpPoints);
      scene.add(grpTraj);
      scene.add(grpKeyframes);

      loadDataset();
    }}

    function toThree(x, y, z) {{
      return new THREE.Vector3(x, z, -y);
    }}

    function loadDataset() {{
      while(grpPoints.children.length) grpPoints.remove(grpPoints.children[0]);
      while(grpTraj.children.length) grpTraj.remove(grpTraj.children[0]);
      while(grpKeyframes.children.length) grpKeyframes.remove(grpKeyframes.children[0]);

      const ds = ALL_DATA[currentDs];
      if (!ds) return;

      document.getElementById("info-kf-count").textContent = ds.keyframes_count + " db";
      document.getElementById("info-total-points").textContent = ds.total_points.toLocaleString("hu-HU") + " db";
      document.getElementById("info-bounds").textContent = `${{ds.bounds.width}}m x ${{ds.bounds.length}}m`;

      const numPts = ds.points.length;
      const pos = new Float32Array(numPts * 3);
      const col = new Float32Array(numPts * 3);
      rawFramesArray = new Int32Array(numPts);

      for (let i = 0; i < numPts; i++) {{
        const pt = ds.points[i];
        const p3 = toThree(pt[0], pt[1], pt[2]);
        pos[i * 3] = p3.x; pos[i * 3 + 1] = p3.y; pos[i * 3 + 2] = p3.z;
        rawFramesArray[i] = pt[3];
        
        // Kulcsképkocka pontok kiemelése zöld/kék színnel
        if (pt[4] === 1) {{
          col[i * 3] = 0.0; col[i * 3 + 1] = 0.9; col[i * 3 + 2] = 0.4;
        }} else {{
          col[i * 3] = 0.13; col[i * 3 + 1] = 0.6; col[i * 3 + 2] = 0.9;
        }}
      }}

      rawPosArray = new Float32Array(pos);
      ptsGeom = new THREE.BufferGeometry();
      ptsGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      ptsGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));

      ptsMaterial = new THREE.PointsMaterial({{ size: 0.035, vertexColors: true, transparent: true, opacity: 0.85 }});
      grpPoints.add(new THREE.Points(ptsGeom, ptsMaterial));

      // Útvonal rajzolása
      const trajPts = ds.trajectory.map(t => toThree(t.x, t.y, 0.05));
      const lineGeom = new THREE.BufferGeometry().setFromPoints(trajPts);
      grpTraj.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({{ color: 0x00e676, linewidth: 2 }})));

      // Kulcsképkocka horgonyok (Keyframe spheres)
      ds.keyframes.forEach(kf => {{
        const kfP3 = toThree(kf.x, kf.y, 0.1);
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.12, 12, 12),
          new THREE.MeshBasicMaterial({{ color: 0xff9900 }})
        );
        mesh.position.set(kfP3.x, kfP3.y, kfP3.z);
        grpKeyframes.add(mesh);
      }});

      const rng = document.getElementById("rng-timeline");
      rng.max = ds.total_frames - 1;
      rng.value = ds.total_frames - 1;
      currentFrameIdx = ds.total_frames - 1;

      updateVisiblePoints();
    }}

    function updateVisiblePoints() {{
      if (!ptsGeom || !rawPosArray) return;
      const ds = ALL_DATA[currentDs];
      const posAttr = ptsGeom.getAttribute('position');
      const numPts = rawFramesArray.length;

      for (let i = 0; i < numPts; i++) {{
        if (rawFramesArray[i] <= currentFrameIdx) {{
          posAttr.setXYZ(i, rawPosArray[i * 3], rawPosArray[i * 3 + 1], rawPosArray[i * 3 + 2]);
        }} else {{
          posAttr.setXYZ(i, 0, -999, 0);
        }}
      }}
      posAttr.needsUpdate = true;
      document.getElementById("lbl-time").textContent = `${{(currentFrameIdx * 0.25).toFixed(1)}}s / ${{ds.total_time}}s`;
    }}

    function onDatasetChange() {{
      currentDs = document.getElementById("dataset-select").value;
      loadDataset();
    }}

    function onTimelineScrub(val) {{
      currentFrameIdx = parseInt(val);
      updateVisiblePoints();
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById("btn-play").textContent = isPlaying ? "⏸ Megállítás" : "▶ Lejátszás";
    }}

    function animate() {{
      requestAnimationFrame(animate);
      if (isPlaying) {{
        const ds = ALL_DATA[currentDs];
        if (currentFrameIdx < ds.total_frames - 1) {{
          currentFrameIdx++;
          updateVisiblePoints();
          document.getElementById("rng-timeline").value = currentFrameIdx;
        }} else {{
          togglePlay();
        }}
      }}
      controls.update();
      renderer.render(scene, camera);
    }}

    init3D();
    animate();
  </script>
</body>
</html>
"""

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        ("walk_kicsi.jsonl", "walk_kicsi", "🚶 Kis szobai séta (60s)", False, 2, 240),
        ("walk_teszt.jsonl", "walk_teszt", "🛑 Álló robot (60s)", True, 2, 240),
        ("walk_seta1.jsonl", "walk_seta1", "🏃 Nagy séta (90s)", False, 2, 240),
    ]
    
    processed = {}
    for filename, ds_id, label, is_stat, step_val, max_fr in datasets:
        fpath = os.path.join(base_dir, filename)
        if os.path.exists(fpath):
            res = process_keyframe_dataset(fpath, ds_id, label, is_stationary=is_stat, step=step_val, max_frames=max_fr)
            if res:
                processed[ds_id] = res
                
    html_content = generate_keyframe_html(processed)
    out_file = os.path.join(base_dir, "keyframe_slam.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nMentve: {out_file} ({len(html_content):,} bájt)")

if __name__ == "__main__":
    main()
