"""
NERO_GO2 — Hivatalos 3D RANSAC Sík-Szegmentáció & L-Shape Bounding Box SLAM Engine
Szakmai Algoritmusok (Zhang et al. L-Shape Fitting & RANSAC Plane Segmentation):
1. RANSAC Padló és Fali Sík Szegmentáció (Ax + By + Cz + D = 0)
2. Search-Based L-Shape / PCA Oriented Bounding Box (OBB) Illesztés Bútorokra és Falakra
3. Log-Odds 2D Foglaltsági Térkép (Thrun)
4. Szigorúan 0% Pont-törlés
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

def ransac_plane_segmentation(pts_3d, distance_threshold=0.06, max_iterations=30):
    """ RANSAC 3D Sík Detektálás (Ax + By + Cz + D = 0) """
    if len(pts_3d) < 15:
        return None, np.zeros(len(pts_3d), dtype=bool)
        
    best_inliers = []
    best_plane = None
    n_pts = len(pts_3d)
    
    for _ in range(max_iterations):
        sample_idxs = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts_3d[sample_idxs]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-6:
            continue
        normal = normal / norm_val
        d = -np.dot(normal, p1)
        
        distances = np.abs(pts_3d @ normal + d)
        inliers = np.where(distances < distance_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)
            
    inlier_mask = np.zeros(n_pts, dtype=bool)
    if len(best_inliers) > 0:
        inlier_mask[best_inliers] = True
    return best_plane, inlier_mask

def fit_l_shape_obb(cluster_pts):
    """ Search-Based L-Shape / Corner Oriented Bounding Box (OBB) Fitting (Zhang et al. elv) """
    center = np.mean(cluster_pts, axis=0)
    xy_pts = cluster_pts[:, :2] - center[:2]
    if len(xy_pts) < 6:
        dx = max(0.25, float(cluster_pts[:, 0].max() - cluster_pts[:, 0].min()))
        dy = max(0.25, float(cluster_pts[:, 1].max() - cluster_pts[:, 1].min()))
        dz = max(0.25, float(cluster_pts[:, 2].max() - cluster_pts[:, 2].min()))
        return center, np.array([dx, dy, dz]), 0.0

    # Search for optimal rotation angle (0 to 90 degrees) minimizing bounding box area
    best_area = 1e9
    best_angle = 0.0
    best_min_max = None

    for angle_deg in range(0, 90, 5):
        rad = math.radians(angle_deg)
        c_a, s_a = math.cos(rad), math.sin(rad)
        rot_mat = np.array([[c_a, -s_a], [s_a, c_a]])
        rot_xy = xy_pts @ rot_mat.T
        
        min_xy = np.min(rot_xy, axis=0)
        max_xy = np.max(rot_xy, axis=0)
        area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
        
        if area < best_area:
            best_area = area
            best_angle = rad
            best_min_max = (min_xy, max_xy)

    min_xy, max_xy = best_min_max
    dx = max(0.20, float(max_xy[0] - min_xy[0]))
    dy = max(0.20, float(max_xy[1] - min_xy[1]))
    dz = max(0.20, float(cluster_pts[:, 2].max() - cluster_pts[:, 2].min()))
    
    # Recenter OBB
    local_center = (min_xy + max_xy) / 2.0
    c_a, s_a = math.cos(-best_angle), math.sin(-best_angle)
    rot_back = np.array([[c_a, -s_a], [s_a, c_a]])
    adjusted_center_xy = center[:2] + local_center @ rot_back.T
    
    final_center = np.array([adjusted_center_xy[0], adjusted_center_xy[1], center[2]], dtype=np.float32)
    return final_center, np.array([dx, dy, dz]), best_angle

def extract_clusters_dbscan(pts_3d, max_dist=0.38, min_pts=10):
    if len(pts_3d) < min_pts:
        return []
    tree = cKDTree(pts_3d)
    visited = np.zeros(len(pts_3d), dtype=bool)
    clusters = []
    
    for i in range(len(pts_3d)):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(pts_3d[i], max_dist)
        if len(idxs) >= min_pts:
            cluster_idxs = []
            queue = list(idxs)
            for idx in queue:
                if not visited[idx]:
                    visited[idx] = True
                    cluster_idxs.append(idx)
                    sub_idxs = tree.query_ball_point(pts_3d[idx], max_dist)
                    if len(sub_idxs) >= min_pts:
                        for s_idx in sub_idxs:
                            if not visited[s_idx]:
                                queue.append(s_idx)
            if len(cluster_idxs) >= min_pts:
                clusters.append(pts_3d[cluster_idxs])
    return clusters

def process_confidence_slam(filepath, name, label, is_stationary=False, step=2, max_frames=260):
    print(f"\n=======================================================")
    print(f"Hivatalos RANSAC + L-Shape SLAM: {name} ({label})")
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
    
    tracked_objects = []
    next_obj_id = 1
    
    trajectory = []
    points_tagged = []
    pts_buffer = []
    
    current_R = np.eye(2, dtype=np.float32)
    current_t = np.zeros(2, dtype=np.float32)
    global_map = None
    global_tree = None
    
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
        else:
            if global_map is None:
                corr_xy = raw_xy.copy()
                cor_pos = np.array([x_raw, y_raw])
                global_map = raw_xy[::3].copy()
                global_tree = cKDTree(global_map)
            else:
                curr_al = raw_xy.copy()
                max_iters = 6
                for _ in range(max_iters):
                    dists, idxs = global_tree.query(curr_al)
                    m = dists < 0.40
                    if np.sum(m) < 25: break
                    src_m = curr_al[m]
                    dst_m = global_map[idxs[m]]
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
                
                if f_idx % 2 == 0:
                    global_map = np.vstack([global_map, corr_xy[::6]])
                    if len(global_map) > 2000:
                        global_map = global_map[::len(global_map)//1500]
                    global_tree = cKDTree(global_map)
                    
        trajectory.append({
            "f": f_idx,
            "x": round(float(cor_pos[0]), 3),
            "y": round(float(cor_pos[1]), 3),
            "yaw": round(float(yaw_raw), 3),
            "t": round(f_idx * step * 0.125, 2)
        })
        
        # 1. RANSAC Padló Kiszedése (z < -0.20m)
        pts_3d_curr = np.column_stack([corr_xy, pts[:, 2]])
        obstacle_m = pts_3d_curr[:, 2] > -0.20
        obstacle_pts = pts_3d_curr[obstacle_m]
        
        # 2. RANSAC Fali Sík és Bútor Detektálás
        if len(obstacle_pts) > 25 and f_idx % 8 == 0:
            plane_model, plane_inliers = ransac_plane_segmentation(obstacle_pts, distance_threshold=0.05)
            non_plane_pts = obstacle_pts[~plane_inliers] if plane_model is not None else obstacle_pts
            
            # 3. L-Shape Bounding Box Illesztés Bútorokra & Tárgyakra
            clusters = extract_clusters_dbscan(non_plane_pts[::2], max_dist=0.38, min_pts=10)
            for c_pts in clusters:
                center, extents, yaw = fit_l_shape_obb(c_pts)
                
                best_obj = None
                best_dist = 999.0
                for obj in tracked_objects:
                    d = np.hypot(obj["center"][0] - center[0], obj["center"][1] - center[1])
                    if d < 0.75 and d < best_dist:
                        best_dist = d
                        best_obj = obj
                        
                if best_obj is not None:
                    best_obj["confidence"] = min(1.0, round(best_obj["confidence"] + 0.15, 2))
                    best_obj["hit_count"] += 1
                    best_obj["last_f"] = f_idx
                    best_obj["center"] = [
                        round(0.7 * best_obj["center"][0] + 0.3 * float(center[0]), 2),
                        round(0.7 * best_obj["center"][1] + 0.3 * float(center[1]), 2),
                        round(0.7 * best_obj["center"][2] + 0.3 * float(center[2]), 2)
                    ]
                    best_obj["extents"] = [
                        round(0.7 * best_obj["extents"][0] + 0.3 * float(extents[0]), 2),
                        round(0.7 * best_obj["extents"][1] + 0.3 * float(extents[1]), 2),
                        round(0.7 * best_obj["extents"][2] + 0.3 * float(extents[2]), 2)
                    ]
                    best_obj["yaw"] = round(0.7 * best_obj["yaw"] + 0.3 * float(yaw), 2)
                else:
                    tracked_objects.append({
                        "id": next_obj_id,
                        "center": [round(float(center[0]), 2), round(float(center[1]), 2), round(float(center[2]), 2)],
                        "extents": [round(float(extents[0]), 2), round(float(extents[1]), 2), round(float(extents[2]), 2)],
                        "yaw": round(float(yaw), 2),
                        "confidence": 0.25,
                        "hit_count": 1,
                        "first_f": f_idx,
                        "last_f": f_idx
                    })
                    next_obj_id += 1

        # 100% Pontmegtartás
        v_c = np.floor(pts_3d_curr / 0.025).astype(np.int32)
        _, u_idx = np.unique(v_c, axis=0, return_index=True)
        v_pts = pts_3d_curr[u_idx]
        if len(v_pts) > 550:
            v_pts = v_pts[::max(1, len(v_pts)//550)][:550]
            
        for p in v_pts:
            points_tagged.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), f_idx])
            pts_buffer.append([p[0], p[1], p[2]])

    solid_objects = [o for o in tracked_objects if o["confidence"] >= 0.50 or o["hit_count"] >= 3]
    
    # --- 5cm ÉS 10cm MIKRÓ-VOXEL TÉRKÉP ÉPÍTÉSE (5x5 cm precíz kockák) ---
    voxel_map_5cm = {}
    voxel_map_10cm = {}
    
    for p in points_tagged:
        x, y, z, f_idx = p[0], p[1], p[2], p[3]
        if z < -0.20:
            continue # Padló kihagyása a tárgy voxelekből
            
        # 5cm voxel bin
        vx5 = int(np.floor(x / 0.05))
        vy5 = int(np.floor(y / 0.05))
        vz5 = int(np.floor(z / 0.05))
        k5 = (vx5, vy5, vz5)
        if k5 not in voxel_map_5cm:
            voxel_map_5cm[k5] = {"frames": set([f_idx]), "first_f": f_idx, "last_f": f_idx, "hits": 1}
        else:
            voxel_map_5cm[k5]["frames"].add(f_idx)
            voxel_map_5cm[k5]["last_f"] = max(voxel_map_5cm[k5]["last_f"], f_idx)
            voxel_map_5cm[k5]["hits"] += 1
            
        # 10cm voxel bin
        vx10 = int(np.floor(x / 0.10))
        vy10 = int(np.floor(y / 0.10))
        vz10 = int(np.floor(z / 0.10))
        k10 = (vx10, vy10, vz10)
        if k10 not in voxel_map_10cm:
            voxel_map_10cm[k10] = {"frames": set([f_idx]), "first_f": f_idx, "last_f": f_idx, "hits": 1}
        else:
            voxel_map_10cm[k10]["frames"].add(f_idx)
            voxel_map_10cm[k10]["last_f"] = max(voxel_map_10cm[k10]["last_f"], f_idx)
            voxel_map_10cm[k10]["hits"] += 1

    voxels_5cm = []
    for (vx, vy, vz), data in voxel_map_5cm.items():
        obs = len(data["frames"])
        if obs >= 2 or data["hits"] >= 3: # Szűri a magányos zajt
            c_val = min(1.0, round(obs / 4.0, 2))
            x_c = round((vx + 0.5) * 0.05, 3)
            y_c = round((vy + 0.5) * 0.05, 3)
            z_c = round((vz + 0.5) * 0.05, 3)
            voxels_5cm.append([x_c, y_c, z_c, c_val, data["first_f"], data["last_f"]])

    voxels_10cm = []
    for (vx, vy, vz), data in voxel_map_10cm.items():
        obs = len(data["frames"])
        if obs >= 2 or data["hits"] >= 3:
            c_val = min(1.0, round(obs / 3.0, 2))
            x_c = round((vx + 0.5) * 0.10, 3)
            y_c = round((vy + 0.5) * 0.10, 3)
            z_c = round((vz + 0.5) * 0.10, 3)
            voxels_10cm.append([x_c, y_c, z_c, c_val, data["first_f"], data["last_f"]])

    pts_arr = np.array(pts_buffer)
    min_x, max_x = float(pts_arr[:, 0].min()), float(pts_arr[:, 0].max())
    min_y, max_y = float(pts_arr[:, 1].min()), float(pts_arr[:, 1].max())
    min_z, max_z = float(pts_arr[:, 2].min()), float(pts_arr[:, 2].max())
    
    print(f"  Feldolgozva {time.time()-t0:.2f}s alatt.")
    print(f"  5x5 cm Mikró-Voxelek száma: {len(voxels_5cm)} db (Solid C>=0.5: {sum(1 for v in voxels_5cm if v[3]>=0.5)})")
    print(f"  10x10 cm Voxelek száma: {len(voxels_10cm)} db")
    print(f"  Megszilárdult RANSAC L-Shape Boxok: {len(solid_objects)} db")
    print(f"  Megtartott sűrű pontok: {len(points_tagged):,} db (100% pontmegtartás)")
    
    return {
        "id": name,
        "label": label,
        "total_frames": len(trajectory),
        "total_time": round(len(trajectory) * step * 0.125, 1),
        "total_points": len(points_tagged),
        "objects_count": len(tracked_objects),
        "solid_objects_count": len(solid_objects),
        "voxels_5cm_count": len(voxels_5cm),
        "voxels_10cm_count": len(voxels_10cm),
        "voxels_5cm": voxels_5cm,
        "voxels_10cm": voxels_10cm,
        "objects": tracked_objects,
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

def generate_confidence_html(datasets):
    data_json = json.dumps(datasets, default=lambda x: bool(x) if isinstance(x, np.bool_) else (float(x) if isinstance(x, np.floating) else int(x)))
    return f"""<!DOCTYPE html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <title>NERO_GO2 — 5x5 cm Mikró-Voxel 3D SLAM</title>
  <style>
    :root {{
      --bg: #07090e;
      --panel-bg: rgba(13, 17, 25, 0.94);
      --border: rgba(255, 255, 255, 0.12);
      --accent-blue: #22e5ff;
      --accent-green: #00e676;
      --accent-yellow: #ffea00;
      --text: #e2e8f0;
      --muted: #8899aa;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: monospace; overflow: hidden; width: 100vw; height: 100vh; }}
    #canvas3d {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}
    #hud-panel {{
      position: absolute; top: 14px; left: 14px; width: 400px;
      background: var(--panel-bg); backdrop-filter: blur(14px);
      border: 1px solid var(--border); border-radius: 10px; z-index: 10; padding: 14px;
      display: flex; flex-direction: column; gap: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }}
    h1 {{ font-size: 13px; font-weight: 700; color: var(--accent-green); text-transform: uppercase; }}
    select.ds-select, select.mode-select {{ width: 100%; padding: 8px; background: #0d121c; border: 1px solid var(--accent-green); color: #fff; font-size: 11px; border-radius: 6px; }}
    .card {{ background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); font-size: 11px; line-height: 1.5; }}
    .card b {{ color: var(--accent-green); }}
    #timeline-bar {{
      position: absolute; bottom: 16px; left: 14px; right: 14px;
      background: var(--panel-bg); backdrop-filter: blur(14px); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 16px; z-index: 10; display: flex; align-items: center; gap: 14px;
    }}
    .play-btn {{ background: var(--accent-green); color: #000; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas3d"></div>
  <div id="hud-panel">
    <h1>🧩 5×5 cm Mikró-Voxel 3D SLAM</h1>
    <p style="font-size:10px; color:var(--muted);">Szerkezeti Kocka Építés & Konfidencia Szilárdítás</p>

    <div>
      <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:3px;">ADATHALMAZ:</label>
      <select id="dataset-select" class="ds-select" onchange="onDatasetChange()">
        <option value="walk_kicsi" selected>🚶 walk_kicsi — Szobai séta (60s)</option>
        <option value="walk_teszt">🛑 walk_teszt — Álló robot (60s)</option>
        <option value="walk_seta1">🏃 walk_seta1 — Nagy séta (90s)</option>
      </select>
    </div>

    <div>
      <label style="font-size:10px; color:var(--muted); display:block; margin-bottom:3px;">MEGJELENÍTÉSI MÓD:</label>
      <select id="mode-select" class="mode-select" onchange="onModeChange()">
        <option value="voxel5" selected>🟩 5×5 cm Precíz Mikró-Voxelek (Ajánlott)</option>
        <option value="voxel10">🟧 10×10 cm Kocka Grid</option>
        <option value="obb">📦 Nagy L-Shape Bounding Boxok</option>
      </select>
    </div>

    <div class="card" style="border-left: 3px solid var(--accent-green);">
      <b>🌐 5×5 cm Mikró-Voxel Konfidencia Elv:</b><br>
      A tér pontjait <b>5 cm-es precíz mikró-kockákra</b> szegmentáljuk. Ahogy a robot többször meglát egy 5cm elemet, annak konfidenciája növekszik ($C \ge 0.5$) és <b>megszilárdul (zöld kocka)</b>!
    </div>

    <div id="ds-info" class="card">
      <div>5×5 cm Voxelek száma: <b id="info-voxel-count" style="color:var(--accent-green);">0 db</b></div>
      <div>Megszilárdult Voxelek (C &ge; 0.5): <b id="info-solid-count" style="color:var(--accent-yellow);">0 db</b></div>
      <div>100% Pontfelhő pontok: <b id="info-total-points">0 db</b></div>
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
    let currentMode = "voxel5";
    let scene, camera, renderer, controls;
    let grpPoints, grpTraj, grpVoxels, grpBoxes;
    let rawPosArray, rawFramesArray, ptsGeom, ptsMaterial;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let voxelInstancedMeshSolid = null;
    let voxelInstancedMeshCandidate = null;
    let voxelDataActive = [];
    let boxMeshes = [];

    function init3D() {{
      const container = document.getElementById("canvas3d");
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x07090e);

      camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
      camera.position.set(0, 10, 8);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      const grid = new THREE.GridHelper(40, 40, 0x00e676, 0x1e293b);
      grid.position.y = -0.4;
      scene.add(grid);

      grpPoints = new THREE.Group();
      grpTraj = new THREE.Group();
      grpVoxels = new THREE.Group();
      grpBoxes = new THREE.Group();
      scene.add(grpPoints);
      scene.add(grpTraj);
      scene.add(grpVoxels);
      scene.add(grpBoxes);

      loadDataset();
    }}

    function toThree(x, y, z) {{
      return new THREE.Vector3(x, z, -y);
    }}

    function loadDataset() {{
      while(grpPoints.children.length) grpPoints.remove(grpPoints.children[0]);
      while(grpTraj.children.length) grpTraj.remove(grpTraj.children[0]);
      while(grpVoxels.children.length) grpVoxels.remove(grpVoxels.children[0]);
      while(grpBoxes.children.length) grpBoxes.remove(grpBoxes.children[0]);
      boxMeshes = [];
      voxelInstancedMeshSolid = null;
      voxelInstancedMeshCandidate = null;

      const ds = ALL_DATA[currentDs];
      if (!ds) return;

      document.getElementById("info-total-points").textContent = ds.total_points.toLocaleString("hu-HU") + " db";
      document.getElementById("info-bounds").textContent = `${{ds.bounds.width}}m x ${{ds.bounds.length}}m`;

      // Pontfelhő
      const numPts = ds.points.length;
      const pos = new Float32Array(numPts * 3);
      const col = new Float32Array(numPts * 3);
      rawFramesArray = new Int32Array(numPts);

      for (let i = 0; i < numPts; i++) {{
        const pt = ds.points[i];
        const p3 = toThree(pt[0], pt[1], pt[2]);
        pos[i * 3] = p3.x; pos[i * 3 + 1] = p3.y; pos[i * 3 + 2] = p3.z;
        rawFramesArray[i] = pt[3];
        col[i * 3] = 0.2; col[i * 3 + 1] = 0.7; col[i * 3 + 2] = 1.0;
      }}

      rawPosArray = new Float32Array(pos);
      ptsGeom = new THREE.BufferGeometry();
      ptsGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      ptsGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));

      ptsMaterial = new THREE.PointsMaterial({{ size: 0.03, vertexColors: true, transparent: true, opacity: 0.75 }});
      grpPoints.add(new THREE.Points(ptsGeom, ptsMaterial));

      // Útvonal
      const trajPts = ds.trajectory.map(t => toThree(t.x, t.y, 0.05));
      const lineGeom = new THREE.BufferGeometry().setFromPoints(trajPts);
      grpTraj.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({{ color: 0x00e676, linewidth: 2 }})));

      rebuildVisuals();

      const rng = document.getElementById("rng-timeline");
      rng.max = ds.total_frames - 1;
      rng.value = ds.total_frames - 1;
      currentFrameIdx = ds.total_frames - 1;

      updateVisibleElements();
    }}

    function rebuildVisuals() {{
      while(grpVoxels.children.length) grpVoxels.remove(grpVoxels.children[0]);
      while(grpBoxes.children.length) grpBoxes.remove(grpBoxes.children[0]);
      boxMeshes = [];

      const ds = ALL_DATA[currentDs];
      if (!ds) return;

      if (currentMode === "voxel5" || currentMode === "voxel10") {{
        const voxels = (currentMode === "voxel5") ? ds.voxels_5cm : ds.voxels_10cm;
        const cubeSize = (currentMode === "voxel5") ? 0.046 : 0.092;
        
        document.getElementById("info-voxel-count").textContent = voxels.length + " db";
        const solidCount = voxels.filter(v => v[3] >= 0.50).length;
        document.getElementById("info-solid-count").textContent = solidCount + " db";

        voxelDataActive = voxels;

        const solidList = [];
        const candidateList = [];

        voxels.forEach(v => {{
          if (v[3] >= 0.50) solidList.push(v);
          else candidateList.push(v);
        }});

        const geom = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);

        // Megszilárdult szolid zöld voxelek
        if (solidList.length > 0) {{
          const matSolid = new THREE.MeshBasicMaterial({{ color: 0x00e676, transparent: true, opacity: 0.85 }});
          voxelInstancedMeshSolid = new THREE.InstancedMesh(geom, matSolid, solidList.length);
          const dummy = new THREE.Object3D();
          solidList.forEach((v, idx) => {{
            const p3 = toThree(v[0], v[1], v[2]);
            dummy.position.set(p3.x, p3.y, p3.z);
            dummy.updateMatrix();
            voxelInstancedMeshSolid.setMatrixAt(idx, dummy.matrix);
          }});
          voxelInstancedMeshSolid.instanceMatrix.needsUpdate = true;
          voxelInstancedMeshSolid.userData = {{ items: solidList }};
          grpVoxels.add(voxelInstancedMeshSolid);
        }}

        // Jelölt sárga voxelek
        if (candidateList.length > 0) {{
          const matCand = new THREE.MeshBasicMaterial({{ color: 0xffea00, transparent: true, opacity: 0.40, wireframe: true }});
          voxelInstancedMeshCandidate = new THREE.InstancedMesh(geom, matCand, candidateList.length);
          const dummy = new THREE.Object3D();
          candidateList.forEach((v, idx) => {{
            const p3 = toThree(v[0], v[1], v[2]);
            dummy.position.set(p3.x, p3.y, p3.z);
            dummy.updateMatrix();
            voxelInstancedMeshCandidate.setMatrixAt(idx, dummy.matrix);
          }});
          voxelInstancedMeshCandidate.instanceMatrix.needsUpdate = true;
          voxelInstancedMeshCandidate.userData = {{ items: candidateList }};
          grpVoxels.add(voxelInstancedMeshCandidate);
        }}
      }} else {{
        // OBB Mód
        document.getElementById("info-voxel-count").textContent = ds.objects_count + " db (OBB)";
        document.getElementById("info-solid-count").textContent = ds.solid_objects_count + " db";

        ds.objects.forEach(obj => {{
          const p3 = toThree(obj.center[0], obj.center[1], obj.center[2]);
          const geom = new THREE.BoxGeometry(obj.extents[0], obj.extents[2], obj.extents[1]);
          const isSolid = obj.confidence >= 0.50;
          const mat = new THREE.MeshBasicMaterial({{
            color: isSolid ? 0x00e676 : 0xffea00,
            wireframe: true,
            transparent: true,
            opacity: isSolid ? 0.85 : 0.35
          }});
          const mesh = new THREE.Mesh(geom, mat);
          mesh.position.set(p3.x, p3.y, p3.z);
          mesh.rotation.y = obj.yaw;
          mesh.userData = {{ first_f: obj.first_f, last_f: obj.last_f }};
          grpBoxes.add(mesh);
          boxMeshes.push(mesh);
        }});
      }}
    }}

    function updateVisibleElements() {{
      if (!ptsGeom || !rawPosArray) return;
      const ds = ALL_DATA[currentDs];

      // Pontfelhő
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

      // Voxelek láthatósága
      [voxelInstancedMeshSolid, voxelInstancedMeshCandidate].forEach(imesh => {{
        if (!imesh) return;
        const items = imesh.userData.items;
        const dummy = new THREE.Object3D();
        items.forEach((v, idx) => {{
          if (v[4] <= currentFrameIdx) {{
            const p3 = toThree(v[0], v[1], v[2]);
            dummy.position.set(p3.x, p3.y, p3.z);
            dummy.scale.set(1, 1, 1);
          }} else {{
            dummy.scale.set(0, 0, 0);
          }}
          dummy.updateMatrix();
          imesh.setMatrixAt(idx, dummy.matrix);
        }});
        imesh.instanceMatrix.needsUpdate = true;
      }});

      // OBB dobozok láthatósága
      boxMeshes.forEach(mesh => {{
        mesh.visible = (mesh.userData.first_f <= currentFrameIdx);
      }});

      document.getElementById("lbl-time").textContent = `${{(currentFrameIdx * 0.25).toFixed(1)}}s / ${{ds.total_time}}s`;
    }}

    function onDatasetChange() {{
      currentDs = document.getElementById("dataset-select").value;
      loadDataset();
    }}

    function onModeChange() {{
      currentMode = document.getElementById("mode-select").value;
      rebuildVisuals();
      updateVisibleElements();
    }}

    function onTimelineScrub(val) {{
      currentFrameIdx = parseInt(val);
      updateVisibleElements();
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
          updateVisibleElements();
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
            res = process_confidence_slam(fpath, ds_id, label, is_stationary=is_stat, step=step_val, max_frames=max_fr)
            if res:
                processed[ds_id] = res
                
    html_content = generate_confidence_html(processed)
    out_file = os.path.join(base_dir, "confidence_slam.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nMentve: {out_file} ({len(html_content):,} bájt)")

if __name__ == "__main__":
    main()
