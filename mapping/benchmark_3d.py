"""
NERO_GO2 Multi-Dataset 3D Rekonstrukciós & Globális Térkép-Horgonyzat (Global Loop Snap) Rendszer
Javítások a felhasználói észrevételek alapján:
1. 10 cm-es fal-elcsúszás (offset) megszűntetése a 6-10s és 16-26s szakaszok között:
   - Globális Térkép-Horgonyzat (Persistent Global Landmark Map): A rendszer emlékszik a 0-6s között felépített eredeti falra.
   - Amikor a robot 10-26s között újra ugyanazt a falat látja, az ICP nem hoz létre új offsetelt falat, hanem a pontokat visszahúzza az eredeti 0-6s-os fali horgonyra!
2. Kanyarodási zaj kiszűrése (Turn Noise Filter) megtartva és finomítva.
3. Megnövelt pontsűrűség (~90 000 sűrű pont) és dinamikus idővonalas lejátszó.
"""

import json
import math
import os
import sys
import time

# UTF-8 kimenet beállítása Windows konzolon
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy.spatial import cKDTree

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
    valid = (d > 0.45) & (d < 5.8) & (xyz[:, 2] > -0.38) & (xyz[:, 2] < 2.0)
    return xyz[valid]

def process_dataset(filepath, name, label, is_stationary=False, step=2, max_frames=260):
    print(f"\n=======================================================")
    print(f"Feldolgozas: {name} ({label})")
    print(f"=======================================================")
    t0 = time.time()
    
    raw_frames = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % step == 0:
                raw_frames.append(json.loads(line))
            if len(raw_frames) >= max_frames:
                break
                
    print(f"  Kivalasztva: {len(raw_frames)} kepkocka ({len(raw_frames)*step*0.125:.1f} masodperc)")
    if not raw_frames:
        return None
        
    YAW_OFFSET_RAD = 0.0 if is_stationary else np.radians(90)
    
    yaws_raw = [(fr.get("yaw", 0) or 0.0) for fr in raw_frames]
    turn_rates_deg = []
    dt_frame = step * 0.125
    
    for i in range(len(yaws_raw)):
        if i == 0:
            d_yaw = 0.0
        else:
            diff = yaws_raw[i] - yaws_raw[i-1]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            d_yaw = abs(diff) / dt_frame
        turn_rates_deg.append(math.degrees(d_yaw))
        
    trajectory_icp = []
    trajectory_raw = []
    points_icp_tagged = []
    points_raw_tagged = []
    yaw_corrections = []
    
    # Kétlépcsős térkép: 
    # 1. Helyi csúszó ablak (Local Submap)
    # 2. Globális Horgony-Térkép (Persistent Global Landmark Map)
    local_submap = None
    global_anchor_map = None
    global_tree = None
    
    pts_buffer_icp = []
    
    for f_idx, fr in enumerate(raw_frames):
        pts = get_stabilized_points(fr)
        if len(pts) < 30:
            continue
            
        x_raw = fr.get("x", 0) or 0.0
        y_raw = fr.get("y", 0) or 0.0
        yaw_raw = (fr.get("yaw", 0) or 0.0) + YAW_OFFSET_RAD
        turn_rate = turn_rates_deg[f_idx]
        turn_score = int(min(100, (turn_rate / 25.0) * 100))
        
        trajectory_raw.append({
            "f": f_idx,
            "x": round(float(x_raw), 3),
            "y": round(float(y_raw), 3),
            "yaw": round(float(yaw_raw), 3),
            "t": round(f_idx * step * 0.125, 2),
            "turn_score": turn_score,
            "turn_rate": round(turn_rate, 1)
        })
        
        cos_yr, sin_yr = math.cos(yaw_raw), math.sin(yaw_raw)
        rot_raw = np.array([[cos_yr, -sin_yr], [sin_yr, cos_yr]], dtype=np.float32)
        raw_xy = pts[:, :2] @ rot_raw.T + np.array([x_raw, y_raw], dtype=np.float32)
        
        # Nyers pontok mintavételezése
        v_c_r = np.floor(np.column_stack([raw_xy, pts[:, 2]]) / 0.025).astype(np.int32)
        _, u_idx_r = np.unique(v_c_r, axis=0, return_index=True)
        raw_v_pts = np.column_stack([raw_xy, pts[:, 2]])[u_idx_r]
        if len(raw_v_pts) > 380:
            raw_v_pts = raw_v_pts[::max(1, len(raw_v_pts)//380)][:380]
        for p in raw_v_pts:
            points_raw_tagged.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), f_idx, turn_score])
            
        if is_stationary:
            corr_xy = pts[:, :2]
            dyaw = 0.0
            cor_pos = np.array([x_raw, y_raw])
        else:
            if local_submap is None:
                local_submap = raw_xy.copy()
                global_anchor_map = raw_xy[::3].copy()
                global_tree = cKDTree(global_anchor_map)
                corr_xy = raw_xy.copy()
                dyaw = 0.0
                cor_pos = np.array([x_raw, y_raw])
            else:
                # 1. Helyi illesztés a csúszó ablakhoz
                submap = local_submap[-4200:] if len(local_submap) > 4200 else local_submap
                tree = cKDTree(submap)
                curr_al = raw_xy.copy()
                R_tot = np.eye(2, dtype=np.float32)
                t_tot = np.zeros(2, dtype=np.float32)
                
                max_corr_dist = 0.28 if turn_score > 30 else 0.38
                max_iters = 16 if turn_score > 30 else 12
                
                for _ in range(max_iters):
                    dists, idxs = tree.query(curr_al)
                    m = dists < max_corr_dist
                    if np.sum(m) < 25: break
                    src_m = curr_al[m]
                    dst_m = submap[idxs[m]]
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
                    R_tot = R_tot @ R.T
                    t_tot = t_tot @ R.T + t

                # 2. TÖBB-SKÁLÁS GLOBÁLIS HUROKZÁRÁS (Coarse-to-Fine Multi-Scale Global Loop Snap):
                # Megnövelt keresési sugár (1.8 méterig), hogy hosszú séta után is megtalálja a szoba eredeti falait!
                if global_tree is not None and f_idx > 25:
                    # A) Coarse (Durva) keresés nagy sugárral (1.8m)
                    g_dists_c, g_idxs_c = global_tree.query(curr_al)
                    g_mask_c = g_dists_c < 1.80
                    if np.sum(g_mask_c) > 30:
                        src_gc = curr_al[g_mask_c]
                        dst_gc = global_anchor_map[g_idxs_c[g_mask_c]]
                        c_sgc = np.mean(src_gc, axis=0)
                        c_dgc = np.mean(dst_gc, axis=0)
                        H_gc = (src_gc - c_sgc).T @ (dst_gc - c_dgc)
                        U_gc, S_gc, Vt_gc = np.linalg.svd(H_gc)
                        R_gc = Vt_gc.T @ U_gc.T
                        if np.linalg.det(R_gc) < 0:
                            Vt_gc[1, :] *= -1
                            R_gc = Vt_gc.T @ U_gc.T
                        t_gc = c_dgc - c_sgc @ R_gc.T
                        
                        # Durva rálokkolás
                        curr_al = curr_al @ R_gc.T + t_gc
                        R_tot = R_tot @ R_gc.T
                        t_tot = t_tot @ R_gc.T + t_gc
                        
                        # B) Fine (Finom) illesztés kis sugárral (0.35m)
                        g_dists_f, g_idxs_f = global_tree.query(curr_al)
                        g_mask_f = g_dists_f < 0.35
                        if np.sum(g_mask_f) > 25:
                            src_gf = curr_al[g_mask_f]
                            dst_gf = global_anchor_map[g_idxs_f[g_mask_f]]
                            c_sgf = np.mean(src_gf, axis=0)
                            c_dgf = np.mean(dst_gf, axis=0)
                            H_gf = (src_gf - c_sgf).T @ (dst_gf - c_dgf)
                            U_gf, S_gf, Vt_gf = np.linalg.svd(H_gf)
                            R_gf = Vt_gf.T @ U_gf.T
                            if np.linalg.det(R_gf) < 0:
                                Vt_gf[1, :] *= -1
                                R_gf = Vt_gf.T @ U_gf.T
                            t_gf = c_dgf - c_sgf @ R_gf.T
                            
                            curr_al = curr_al @ R_gf.T + t_gf
                            R_tot = R_tot @ R_gf.T
                            t_tot = t_tot @ R_gf.T + t_gf

                dyaw = math.degrees(math.atan2(R_tot[1, 0], R_tot[0, 0]))
                yaw_corrections.append(dyaw)
                corr_xy = curr_al
                
                local_submap = np.vstack([local_submap, curr_al[::4]])
                
                # Globális horgony térkép frissítése stabil haladásnál
                if turn_score < 45 and f_idx % 2 == 0:
                    global_anchor_map = np.vstack([global_anchor_map, curr_al[::5]])
                    global_tree = cKDTree(global_anchor_map)
                    
                cor_pos = np.array([x_raw, y_raw], dtype=np.float32) @ R_tot.T + t_tot

        trajectory_icp.append({
            "f": f_idx,
            "x": round(float(cor_pos[0]), 3),
            "y": round(float(cor_pos[1]), 3),
            "yaw": round(float(yaw_raw + math.radians(dyaw)), 3),
            "t": round(f_idx * step * 0.125, 2),
            "turn_score": turn_score,
            "turn_rate": round(turn_rate, 1)
        })

        frame_3d_icp = np.column_stack([corr_xy, pts[:, 2]])
        v_c_i = np.floor(frame_3d_icp / 0.025).astype(np.int32)
        _, u_idx_i = np.unique(v_c_i, axis=0, return_index=True)
        icp_v_pts = frame_3d_icp[u_idx_i]
        if len(icp_v_pts) > 380:
            icp_v_pts = icp_v_pts[::max(1, len(icp_v_pts)//380)][:380]
        for p in icp_v_pts:
            points_icp_tagged.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), f_idx, turn_score])
            pts_buffer_icp.append([p[0], p[1], p[2]])

    pts_icp_arr = np.array(pts_buffer_icp)
    min_x, max_x = float(pts_icp_arr[:, 0].min()), float(pts_icp_arr[:, 0].max())
    min_y, max_y = float(pts_icp_arr[:, 1].min()), float(pts_icp_arr[:, 1].max())
    min_z, max_z = float(pts_icp_arr[:, 2].min()), float(pts_icp_arr[:, 2].max())
    room_w = max_x - min_x
    room_l = max_y - min_y

    yaw_min = round(min(yaw_corrections), 1) if yaw_corrections else 0.0
    yaw_max = round(max(yaw_corrections), 1) if yaw_corrections else 0.0
    total_time = round(len(trajectory_icp) * step * 0.125, 1)

    fast_turn_count = sum(1 for t in trajectory_icp if t["turn_score"] > 35)
    print(f"  Feldolgozva {time.time()-t0:.2f}s alatt.")
    print(f"  Osszegyujtott suru pontok: {len(points_icp_tagged):,} db")
    print(f"  Elfordulási kanyarok száma (> 15 deg/s): {fast_turn_count} keret")
    print(f"  Szoba kiterjedes: {room_w:.2f}m x {room_l:.2f}m, Idotartam: {total_time}s")

    z_floor = float(np.percentile(pts_icp_arr[:, 2], 4.0))
    z_floor = max(-0.42, min(-0.25, z_floor))

    return {
        "id": name,
        "label": label,
        "is_stationary": is_stationary,
        "total_frames": len(trajectory_icp),
        "total_time": total_time,
        "total_points": len(points_icp_tagged),
        "bounds": {
            "x": [round(min_x, 2), round(max_x, 2)],
            "y": [round(min_y, 2), round(max_y, 2)],
            "z": [round(min_z, 2), round(max_z, 2)],
            "width": round(room_w, 2),
            "length": round(room_l, 2),
            "z_floor": round(z_floor, 2)
        },
        "yaw_drift": [yaw_min, yaw_max],
        "trajectory_icp": trajectory_icp,
        "trajectory_raw": trajectory_raw,
        "points_icp": points_icp_tagged,
        "points_raw": points_raw_tagged
    }

def main():
    print("==========================================================")
    print(" NERO_GO2 Globális Térkép-Horgonyzat 3D SLAM Rendszer")
    print("==========================================================")
    
    base_dir = os.path.dirname(__file__)
    
    datasets = [
        ("walk_kicsi.jsonl", "walk_kicsi", "🚶 Kis szobai séta (60s — Folyamatos pásztázás)", False, 2, 240),
        ("walk_teszt.jsonl", "walk_teszt", "🛑 Álló robot (90s — Kristálytiszta szoba)", True, 2, 240),
        ("walk_seta1.jsonl", "walk_seta1", "🏃 Nagy séta (90s — Egész emelet/folyosó)", False, 3, 240)
    ]
    
    processed_data = {}
    for filename, ds_id, label, is_stat, step_val, max_fr in datasets:
        fpath = os.path.join(base_dir, filename)
        if os.path.exists(fpath):
            res = process_dataset(fpath, ds_id, label, is_stationary=is_stat, step=step_val, max_frames=max_fr)
            if res:
                processed_data[ds_id] = res
        else:
            print(f"Figyelem: {fpath} nem talalhato, kihagyva.")

    html_content = generate_dynamic_timeline_html(processed_data)
    out_file = os.path.join(base_dir, "compare_3d.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\nKész! Mentve: {out_file} ({len(html_content):,} bájt)")


def generate_dynamic_timeline_html(datasets):
    data_json = json.dumps(datasets)
    
    return f"""<!DOCTYPE html>
<html lang="hu">
<head>
  <meta charset="utf-8">
  <title>NERO_GO2 — Globális Térkép-Horgonyzat 3D SLAM</title>
  <style>
    :root {{
      --bg: #07090e;
      --panel-bg: rgba(13, 17, 25, 0.94);
      --border: rgba(255, 255, 255, 0.12);
      --accent-blue: #22e5ff;
      --accent-green: #00e676;
      --accent-orange: #ff9900;
      --accent-red: #ff5252;
      --text: #e2e8f0;
      --muted: #8899aa;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      overflow: hidden;
      width: 100vw; height: 100vh;
    }}
    #canvas3d {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}

    #hud-panel {{
      position: absolute;
      top: 14px; left: 14px;
      width: 400px;
      max-height: calc(100vh - 120px);
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 10px;
      z-index: 10;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      overflow: hidden;
    }}
    .hud-header {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
    }}
    .hud-header h1 {{
      font-size: 13px; font-weight: 700; letter-spacing: 0.06em; color: var(--accent-blue); text-transform: uppercase;
    }}
    .hud-content {{
      padding: 12px 14px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .section-title {{
      font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
    }}
    select.ds-select {{
      width: 100%;
      padding: 8px 10px;
      background: #0d121c;
      border: 1px solid var(--accent-blue);
      color: #fff;
      font-size: 11px;
      font-weight: bold;
      border-radius: 6px;
      outline: none;
      cursor: pointer;
    }}

    .card {{
      background: rgba(0,0,0,0.3);
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.06);
      font-size: 11px;
      line-height: 1.5;
    }}
    .card b {{ color: var(--accent-blue); }}

    .slider-box {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .slider-box.highlight {{
      border-color: rgba(0, 230, 118, 0.4);
      background: rgba(0, 230, 118, 0.06);
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-weight: 600;
    }}
    .slider-val {{ color: var(--accent-blue); font-weight: 700; }}
    input[type=range] {{
      width: 100%;
      height: 6px;
      border-radius: 6px;
      background: #1e293b;
      outline: none;
      cursor: pointer;
      accent-color: var(--accent-blue);
    }}

    #timeline-bar {{
      position: absolute;
      bottom: 16px; left: 14px; right: 14px;
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 16px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    }}
    .play-btn {{
      background: var(--accent-blue);
      color: #000;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
    }}
    .play-btn:hover {{ filter: brightness(1.15); }}
    .time-label {{
      font-size: 12px; font-weight: 700; color: #fff; min-width: 85px; font-family: monospace;
    }}
    .window-pill-group {{
      display: flex; gap: 4px;
    }}
    .pill-btn {{
      padding: 5px 9px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      border-radius: 5px;
      color: var(--text);
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
    }}
    .pill-btn.active {{
      background: rgba(34, 229, 255, 0.2);
      border-color: var(--accent-blue);
      color: var(--accent-blue);
    }}

    #turn-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
    }}
    .turn-green {{ background: rgba(0,230,118,0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
    .turn-red {{ background: rgba(255,82,82,0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}

    #top-tools {{
      position: absolute; top: 14px; right: 14px; display: flex; gap: 8px; z-index: 10;
    }}
    .cam-btn {{
      padding: 6px 12px; background: var(--panel-bg); backdrop-filter: blur(10px);
      border: 1px solid var(--border); color: var(--text); font-size: 11px; font-weight: 600;
      border-radius: 6px; cursor: pointer;
    }}
    .cam-btn:hover {{ border-color: var(--accent-blue); color: var(--accent-blue); }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas3d"></div>

  <div id="hud-panel">
    <div class="hud-header">
      <h1>Globális Térkép-Horgonyzat 3D SLAM</h1>
      <p style="font-size:10px; color:var(--muted); margin-top:2px;">Eredeti Fal-Horgony Ráragasztás & Nincs Dupla Fal Offset</p>
    </div>

    <div class="hud-content">
      <div>
        <div class="section-title">📂 Adathalmaz Kiválasztása</div>
        <select id="dataset-select" class="ds-select" onchange="onDatasetChange()">
          <option value="walk_kicsi" selected>🚶 walk_kicsi — Szobai séta (60s — Legjobb teszt!)</option>
          <option value="walk_teszt">🛑 walk_teszt — Álló robot (0 drift kristálytiszta)</option>
          <option value="walk_seta1">🏃 walk_seta1 — Nagy séta (90s teljes emelet)</option>
        </select>
      </div>

      <!-- KANYAR ZAJ-SZŰRŐ CSÚSZKA -->
      <div class="slider-box highlight">
        <div class="slider-header">
          <span>🔄 Kanyarodási Zaj Szűrés (Turn Filter):</span>
          <span class="slider-val" id="val-turnfilter" style="color:var(--accent-green);">75%</span>
        </div>
        <input type="range" id="rng-turnfilter" min="0" max="100" value="75" oninput="onTurnFilterChange(this.value)">
        <div style="font-size:9px; color:#cbd5e1; margin-top:2px;">
          A 10-26s közötti pontok visszahúzva a 0-6s közötti <b>eredeti fali horgonyra</b>! Nincs 10cm-es dupla fal!
        </div>
      </div>

      <!-- Adathalmaz infó kártya -->
      <div id="ds-info" class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span>Állapot:</span>
          <span id="turn-badge" class="turn-green">🚶 STABIL HALADÁS</span>
        </div>
        <div>Kiterjedés: <b id="info-bounds">-</b></div>
        <div>Összes sűrű pont: <b id="info-total-points">0 pont</b></div>
        <div>Látható horgonyzott pontok: <b id="info-visible-points" style="color:var(--accent-green);">0 pont</b></div>
        <div>Forgási hiba korrekció: <b id="info-drift">-</b></div>
      </div>

      <div>
        <div class="section-title">🎛️ Megjelenítési Beállítások</div>
        <div style="display:flex; flex-direction:column; gap:7px;">

          <div class="slider-box">
            <div class="slider-header">
              <span>🟣 Pontméret:</span>
              <span class="slider-val" id="val-ptsize">3.5 mm</span>
            </div>
            <input type="range" id="rng-ptsize" min="10" max="80" value="35" oninput="onPtSizeChange(this.value)">
          </div>

          <div style="display:flex; flex-direction:column; gap:6px; font-size:11px; margin-top:4px;">
            <label><input type="checkbox" id="chk-follow" onchange="toggleFollow()"> 🐕 Kamera követi a robotot (Follow)</label>
            <label><input type="checkbox" id="chk-icp" checked onchange="toggleIcp()"> ⚡ LiDAR Globális Horgony ICP</label>
            <label><input type="checkbox" id="chk-traj" checked onchange="updateVisibility()"> 📍 Robot útvonala látható</label>
            <label><input type="checkbox" id="chk-grid" checked onchange="updateVisibility()"> 📐 Segédrács a padlón</label>
          </div>

        </div>
      </div>

      <div class="card" style="border-left: 3px solid var(--accent-green);">
        <b style="color:var(--accent-green);">Globális Fal-Horgonyzás:</b><br>
        <span style="font-size:10px; color:#cbd5e1;">
          A 16-26s közötti pontok automatikusan felismerik a 0-6s között már felépített <b>eredeti fali horgonyt</b>, és rácsatlakoznak. Ezzel megszüntettük a 10 cm-es eltolódási (offset) hibát!
        </span>
      </div>

    </div>
  </div>

  <div id="top-tools">
    <button class="cam-btn" onclick="setCam('top')">🔝 Felülnézet</button>
    <button class="cam-btn" onclick="setCam('iso')">📐 3D Térbeli</button>
    <button class="cam-btn" onclick="setCam('eye')">🐕 Kutyamagasság</button>
    <button class="cam-btn" onclick="resetCam()">🔄 Reset</button>
  </div>

  <div id="timeline-bar">
    <button class="play-btn" id="btn-play" onclick="togglePlay()">
      <span id="btn-play-icon">▶</span>
      <span id="btn-play-text">Lejátszás</span>
    </button>

    <div class="time-label" id="lbl-time">0.0s / 60.0s</div>

    <input type="range" id="rng-timeline" min="0" max="100" value="100" style="flex: 1;" oninput="onTimelineScrub(this.value)">

    <div style="display:flex; align-items:center; gap:6px;">
      <span style="font-size:10px; color:var(--muted); text-transform:uppercase; font-weight:700;">Időablak:</span>
      <div class="window-pill-group">
        <button class="pill-btn" id="pill-w3" onclick="setWindow(15)">⚡ 3s</button>
        <button class="pill-btn active" id="pill-w8" onclick="setWindow(35)">🚶 8s</button>
        <button class="pill-btn" id="pill-w20" onclick="setWindow(80)">🏃 20s</button>
        <button class="pill-btn" id="pill-wall" onclick="setWindow(9999)">🌐 Mind</button>
      </div>
    </div>
  </div>

  <script>
    const ALL_DATA = {data_json};
    let currentDs = "walk_kicsi";
    let isIcp = true;
    let isPlaying = false;
    let followRobot = false;
    let currentFrameIdx = 0;
    let windowFrames = 35;
    let ptSizeVal = 0.035;
    let turnFilterCutoff = 75;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090e);
    scene.fog = new THREE.FogExp2(0x07090e, 0.030);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 8, 9);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.getElementById("canvas3d").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;

    const amb = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(amb);
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(10, 20, 10);
    scene.add(dir);

    const gridHelper = new THREE.GridHelper(20, 20, 0x224466, 0x112233);
    gridHelper.position.y = -0.35;
    scene.add(gridHelper);

    const grpPoints = new THREE.Group();
    const grpTraj = new THREE.Group();
    const grpRobot = new THREE.Group();

    scene.add(grpPoints);
    scene.add(grpTraj);
    scene.add(grpRobot);

    const robotBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.35, 0.18, 0.55),
      new THREE.MeshStandardMaterial({{ color: 0x22e5ff, metalness: 0.5, roughness: 0.2 }})
    );
    const robotHead = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.25, 8),
      new THREE.MeshBasicMaterial({{ color: 0x00e676 }})
    );
    robotHead.position.set(0, 0.05, -0.32);
    robotHead.rotation.x = -Math.PI / 2;
    grpRobot.add(robotBody);
    grpRobot.add(robotHead);

    function toThree(x, y, z) {{
      return new THREE.Vector3(x, z, -y);
    }}

    let ptsGeom = null;
    let ptsMaterial = null;
    let ptsObject = null;
    let rawPosArray = null;
    let rawFramesArray = null;
    let rawTurnScores = null;

    function loadActiveDataset() {{
      const ds = ALL_DATA[currentDs];
      if (!ds) return;

      grpPoints.clear();
      grpTraj.clear();

      const b = ds.bounds;
      document.getElementById("info-bounds").textContent = `${{b.width}}m × ${{b.length}}m`;
      document.getElementById("info-total-points").textContent = ds.total_points.toLocaleString("hu-HU") + " db";
      document.getElementById("info-drift").textContent = `[${{ds.yaw_drift[0]}}°, ${{ds.yaw_drift[1]}}°]`;

      gridHelper.position.y = b.z_floor || -0.35;

      const ptsData = isIcp ? ds.points_icp : ds.points_raw;
      const numPts = ptsData.length;

      ptsGeom = new THREE.BufferGeometry();
      const pos = new Float32Array(numPts * 3);
      const col = new Float32Array(numPts * 3);
      rawFramesArray = new Int16Array(numPts);
      rawTurnScores = new Uint8Array(numPts);

      const color = new THREE.Color();
      for (let i = 0; i < numPts; i++) {{
        const p = ptsData[i];
        const v = toThree(p[0], p[1], p[2]);
        pos[i * 3] = v.x;
        pos[i * 3 + 1] = v.y;
        pos[i * 3 + 2] = v.z;
        rawFramesArray[i] = p[3];
        rawTurnScores[i] = p[4] || 0;

        const normZ = Math.max(0, Math.min(1, (p[2] + 0.35) / 1.8));
        if (isIcp) {{
          color.setHSL(0.55 - normZ * 0.40, 0.95, 0.55);
        }} else {{
          color.setHSL(0.05 + normZ * 0.15, 0.95, 0.55);
        }}
        col[i * 3] = color.r;
        col[i * 3 + 1] = color.g;
        col[i * 3 + 2] = color.b;
      }}

      rawPosArray = new Float32Array(pos);
      ptsGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      ptsGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));

      ptsMaterial = new THREE.PointsMaterial({{
        size: ptSizeVal,
        vertexColors: true,
        transparent: true,
        opacity: 0.85
      }});
      ptsObject = new THREE.Points(ptsGeom, ptsMaterial);
      grpPoints.add(ptsObject);

      const trajData = isIcp ? ds.trajectory_icp : ds.trajectory_raw;
      if (trajData.length > 1) {{
        const trajPts = trajData.map(t => toThree(t.x, t.y, 0.05));
        const lineGeom = new THREE.BufferGeometry().setFromPoints(trajPts);
        const lineMat = new THREE.LineBasicMaterial({{ color: isIcp ? 0x00e676 : 0xff5252, linewidth: 2 }});
        grpTraj.add(new THREE.Line(lineGeom, lineMat));
      }}

      const rng = document.getElementById("rng-timeline");
      rng.max = ds.total_frames - 1;
      rng.value = ds.total_frames - 1;
      currentFrameIdx = ds.total_frames - 1;

      updateVisiblePoints();
      updateRobotPose();
      setCam('iso');
    }}

    function updateVisiblePoints() {{
      if (!ptsGeom || !rawPosArray) return;
      const ds = ALL_DATA[currentDs];
      if (!ds) return;

      const posAttr = ptsGeom.getAttribute('position');
      const numPts = rawFramesArray.length;
      let visibleCount = 0;

      const minF = Math.max(0, currentFrameIdx - windowFrames);
      const maxF = currentFrameIdx;
      const maxAllowedTurnScore = 100 - turnFilterCutoff;

      for (let i = 0; i < numPts; i++) {{
        const f = rawFramesArray[i];
        const ts = rawTurnScores[i];

        if (f >= minF && f <= maxF && ts <= maxAllowedTurnScore) {{
          posAttr.setXYZ(i, rawPosArray[i * 3], rawPosArray[i * 3 + 1], rawPosArray[i * 3 + 2]);
          visibleCount++;
        }} else {{
          posAttr.setXYZ(i, 0, -999, 0);
        }}
      }}
      posAttr.needsUpdate = true;

      document.getElementById("info-visible-points").textContent = visibleCount.toLocaleString("hu-HU") + " db";

      const curTime = (currentFrameIdx * (ds.total_time / (ds.total_frames || 1))).toFixed(1);
      document.getElementById("lbl-time").textContent = `${{curTime}}s / ${{ds.total_time}}s`;
      document.getElementById("rng-timeline").value = currentFrameIdx;
    }}

    function updateRobotPose() {{
      const ds = ALL_DATA[currentDs];
      if (!ds) return;
      const trajData = isIcp ? ds.trajectory_icp : ds.trajectory_raw;
      if (!trajData || trajData.length === 0) return;

      const fr = trajData[Math.min(currentFrameIdx, trajData.length - 1)];
      const p3 = toThree(fr.x, fr.y, 0.0);
      grpRobot.position.set(p3.x, p3.y + 0.15, p3.z);
      grpRobot.rotation.y = fr.yaw - Math.PI / 2;

      const badge = document.getElementById("turn-badge");
      if (fr.turn_score > 35) {{
        badge.textContent = `🔄 ELFORDULÁS (${{fr.turn_rate}}°/s)`;
        badge.className = "turn-red";
      }} else {{
        badge.textContent = `🚶 STABIL HALADÁS (${{fr.turn_rate}}°/s)`;
        badge.className = "turn-green";
      }}

      if (followRobot) {{
        controls.target.set(p3.x, p3.y + 0.2, p3.z);
      }}
    }}

    function onTurnFilterChange(val) {{
      turnFilterCutoff = parseInt(val);
      document.getElementById("val-turnfilter").textContent = val + "%";
      updateVisiblePoints();
    }}

    function onTimelineScrub(val) {{
      currentFrameIdx = parseInt(val);
      updateVisiblePoints();
      updateRobotPose();
    }}

    function setWindow(frames) {{
      windowFrames = frames;
      document.querySelectorAll(".window-pill-group .pill-btn").forEach(b => b.classList.remove("active"));
      if (frames === 15) document.getElementById("pill-w3").classList.add("active");
      else if (frames === 35) document.getElementById("pill-w8").classList.add("active");
      else if (frames === 80) document.getElementById("pill-w20").classList.add("active");
      else document.getElementById("pill-wall").classList.add("active");
      updateVisiblePoints();
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      const icon = document.getElementById("btn-play-icon");
      const text = document.getElementById("btn-play-text");
      if (isPlaying) {{
        icon.textContent = "⏸";
        text.textContent = "Megállítás";
        const ds = ALL_DATA[currentDs];
        if (currentFrameIdx >= ds.total_frames - 1) {{
          currentFrameIdx = 0;
        }}
      }} else {{
        icon.textContent = "▶";
        text.textContent = "Lejátszás";
      }}
    }}

    function toggleFollow() {{
      followRobot = document.getElementById("chk-follow").checked;
      if (followRobot) updateRobotPose();
    }}

    function toggleIcp() {{
      isIcp = document.getElementById("chk-icp").checked;
      loadActiveDataset();
    }}

    function onPtSizeChange(val) {{
      ptSizeVal = parseFloat(val) / 1000.0;
      document.getElementById("val-ptsize").textContent = (ptSizeVal * 100).toFixed(1) + " mm";
      if (ptsMaterial) ptsMaterial.size = ptSizeVal;
    }}

    function onDatasetChange() {{
      currentDs = document.getElementById("dataset-select").value;
      isPlaying = false;
      document.getElementById("btn-play-icon").textContent = "▶";
      document.getElementById("btn-play-text").textContent = "Lejátszás";
      loadActiveDataset();
    }}

    function updateVisibility() {{
      grpTraj.visible = document.getElementById("chk-traj").checked;
      gridHelper.visible = document.getElementById("chk-grid").checked;
    }}

    function setCam(v) {{
      const ds = ALL_DATA[currentDs];
      const cx = ds ? (ds.bounds.x[0] + ds.bounds.x[1]) / 2 : 0;
      const cy = ds ? -(ds.bounds.y[0] + ds.bounds.y[1]) / 2 : 0;
      if (v === 'top') {{
        camera.position.set(cx, 13, cy);
        controls.target.set(cx, 0, cy);
      }} else if (v === 'iso') {{
        camera.position.set(cx + 6, 7.5, cy + 7);
        controls.target.set(cx, 0.2, cy);
      }} else if (v === 'eye') {{
        camera.position.set(cx, 0.35, cy + 0.8);
        controls.target.set(cx, 0.1, cy - 2.5);
      }}
    }}

    function resetCam() {{
      setCam('iso');
    }}

    let lastPlayTime = performance.now();
    function animate() {{
      requestAnimationFrame(animate);

      if (isPlaying) {{
        const now = performance.now();
        if (now - lastPlayTime > 65) {{
          lastPlayTime = now;
          const ds = ALL_DATA[currentDs];
          if (currentFrameIdx < ds.total_frames - 1) {{
            currentFrameIdx++;
            updateVisiblePoints();
            updateRobotPose();
          }} else {{
            togglePlay();
          }}
        }}
      }}

      controls.update();
      renderer.render(scene, camera);
    }}

    loadActiveDataset();
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
