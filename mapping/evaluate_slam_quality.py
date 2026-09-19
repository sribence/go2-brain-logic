"""
NERO_GO2 — Automatikus SLAM Minőségértékelő & Szemléltető Rendszer
"""

import json
import math
import os
import sys
import time
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

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

def evaluate_slam_config(filepath, voxel_res=0.05, max_icp_dist=0.35, max_trans_corr=0.15, max_rot_corr=0.10, yaw_scale=1.0):
    t0 = time.time()
    frames = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                frames.append(json.loads(line))
            if len(frames) >= 240:
                break

    YAW_OFFSET = np.radians(90)
    
    voxel_grid = {}
    etalon_centroids = []
    etalon_tree = None
    
    last_slam_pose = None
    last_raw_pose = None
    
    raw_traj = []
    slam_traj = []
    all_non_floor_pts = []
    
    first_half_voxels = set()
    second_half_voxels = set()
    mid_frame = len(frames) // 2
    
    for f_idx, fr in enumerate(frames):
        pts = get_stabilized_points(fr)
        if len(pts) < 30:
            continue
            
        x_raw = float(fr.get("x", 0) or 0.0)
        y_raw = float(fr.get("y", 0) or 0.0)
        yaw_raw = float(wrap_angle(((fr.get("yaw", 0) or 0.0) * yaw_scale) + YAW_OFFSET))
        raw_traj.append((x_raw, y_raw, yaw_raw))
        
        if last_slam_pose is None:
            slam_x, slam_y, slam_yaw = x_raw, y_raw, yaw_raw
        else:
            dx = x_raw - last_raw_pose[0]
            dy = y_raw - last_raw_pose[1]
            dyaw = wrap_angle(yaw_raw - last_raw_pose[2])
            
            slam_x = last_slam_pose[0] + dx
            slam_y = last_slam_pose[1] + dy
            slam_yaw = wrap_angle(last_slam_pose[2] + dyaw)

        last_raw_pose = (x_raw, y_raw, yaw_raw)
        
        cy, sy = math.cos(slam_yaw), math.sin(slam_yaw)
        rot_world = np.array([[cy, -sy], [sy, cy]], dtype=np.float32)
        
        pts_world_xy = pts[:, :2] @ rot_world.T + np.array([slam_x, slam_y], dtype=np.float32)
        pts_world_3d = np.column_stack([pts_world_xy, pts[:, 2]])
        
        # Etalon Scan-to-Submap ICP Korrekció
        if etalon_tree is not None and len(etalon_centroids) > 40:
            non_floor_m = pts_world_3d[:, 2] > -0.20
            non_floor_xy = pts_world_xy[non_floor_m]
            
            if len(non_floor_xy) > 25:
                curr_xy = non_floor_xy.copy()
                d_R = np.eye(2, dtype=np.float32)
                d_t = np.zeros(2, dtype=np.float32)
                etalon_arr = np.array(etalon_centroids, dtype=np.float32)
                
                for _ in range(6):
                    dists, idxs = etalon_tree.query(curr_xy)
                    valid_m = dists < max_icp_dist
                    if np.sum(valid_m) < 20:
                        break
                    src = curr_xy[valid_m]
                    dst = etalon_arr[idxs[valid_m]]
                    
                    c_src = np.mean(src, axis=0)
                    c_dst = np.mean(dst, axis=0)
                    
                    H = (src - c_src).T @ (dst - c_dst)
                    U, S, Vt = np.linalg.svd(H)
                    R_icp = Vt.T @ U.T
                    if np.linalg.det(R_icp) < 0:
                        Vt[1, :] *= -1
                        R_icp = Vt.T @ U.T
                    t_icp = c_dst - c_src @ R_icp.T
                    
                    curr_xy = curr_xy @ R_icp.T + t_icp
                    d_R = d_R @ R_icp.T
                    d_t = d_t @ R_icp.T + t_icp

                corr_dyaw = wrap_angle(math.atan2(d_R[1, 0], d_R[0, 0]))
                corr_dx = float(d_t[0])
                corr_dy = float(d_t[1])
                
                if abs(corr_dx) < max_trans_corr and abs(corr_dy) < max_trans_corr and abs(corr_dyaw) < max_rot_corr:
                    slam_x += corr_dx
                    slam_y += corr_dy
                    slam_yaw = wrap_angle(slam_yaw + corr_dyaw)
                    
                    cy, sy = math.cos(slam_yaw), math.sin(slam_yaw)
                    rot_world = np.array([[cy, -sy], [sy, cy]], dtype=np.float32)
                    pts_world_xy = pts[:, :2] @ rot_world.T + np.array([slam_x, slam_y], dtype=np.float32)
                    pts_world_3d = np.column_stack([pts_world_xy, pts[:, 2]])

        last_slam_pose = (slam_x, slam_y, slam_yaw)
        slam_traj.append((slam_x, slam_y, slam_yaw))
        
        non_floor_m = pts_world_3d[:, 2] > -0.20
        non_floor_pts = pts_world_3d[non_floor_m]
        
        if len(non_floor_pts) > 0:
            all_non_floor_pts.append(non_floor_pts[::3])
            voxels_raw = np.floor(non_floor_pts / voxel_res).astype(np.int32)
            unique_vk, counts = np.unique(voxels_raw, axis=0, return_counts=True)
            new_etalon = False
            for vk_arr, cnt in zip(unique_vk, counts):
                vk = (int(vk_arr[0]), int(vk_arr[1]), int(vk_arr[2]))
                if f_idx < mid_frame:
                    first_half_voxels.add(vk)
                else:
                    second_half_voxels.add(vk)
                    
                if vk not in voxel_grid:
                    voxel_grid[vk] = {"hits": int(cnt), "is_etalon": False}
                else:
                    voxel_grid[vk]["hits"] += int(cnt)
                    if not voxel_grid[vk]["is_etalon"] and voxel_grid[vk]["hits"] >= 3:
                        voxel_grid[vk]["is_etalon"] = True
                        c_x = (vk[0] + 0.5) * voxel_res
                        c_y = (vk[1] + 0.5) * voxel_res
                        etalon_centroids.append([c_x, c_y])
                        new_etalon = True
                        
            if new_etalon and len(etalon_centroids) > 20:
                etalon_arr = np.array(etalon_centroids, dtype=np.float32)
                if len(etalon_arr) > 2500:
                    etalon_arr = etalon_arr[::len(etalon_arr)//2000]
                etalon_tree = cKDTree(etalon_arr)

    total_unique_voxels = len(voxel_grid)
    new_voxels_in_2nd_half = len(second_half_voxels - first_half_voxels)
    duplication_ratio = round(new_voxels_in_2nd_half / max(1, len(first_half_voxels)), 3)
    
    all_pts_np = np.vstack(all_non_floor_pts)[:, :2]
    if len(etalon_centroids) > 20:
        etalon_arr = np.array(etalon_centroids, dtype=np.float32)
        tree = cKDTree(etalon_arr)
        dists, _ = tree.query(all_pts_np)
        wall_thickness_cm = round(float(np.mean(dists)) * 100.0, 2)
    else:
        wall_thickness_cm = 999.0
        
    elapsed = round(time.time() - t0, 2)
    
    return {
        "elapsed_sec": elapsed,
        "total_voxels": total_unique_voxels,
        "first_half_voxels": len(first_half_voxels),
        "new_voxels_2nd_half": new_voxels_in_2nd_half,
        "duplication_ratio": duplication_ratio,
        "wall_thickness_cm": wall_thickness_cm,
        "etalon_count": len(etalon_centroids),
        "slam_traj": np.array(slam_traj),
        "raw_traj": np.array(raw_traj),
        "all_pts": all_pts_np
    }

def render_slam_map_png(res_dict, output_img_path, title_extra=""):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    ax.set_facecolor("#07090e")
    fig.patch.set_facecolor('#07090e')
    
    pts = res_dict["all_pts"]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="#38bdf8", alpha=0.3, label="LiDAR Pontok")
    
    raw = res_dict["raw_traj"]
    slam = res_dict["slam_traj"]
    ax.plot(raw[:, 0], raw[:, 1], color="#ff5252", linestyle="--", linewidth=1.5, label="Nyers Odometria")
    ax.plot(slam[:, 0], slam[:, 1], color="#00e676", linewidth=2.5, label="Etalon SLAM Útvonal")
    
    ax.set_title(f"NERO_GO2 {title_extra}\nFal Vastagság: {res_dict['wall_thickness_cm']}cm | Duplikáció: {res_dict['duplication_ratio']}", color="#38bdf8", fontsize=11)
    ax.legend(facecolor="#1e293b", labelcolor="#f8fafc", loc="upper right")
    ax.grid(True, color="#1e293b", linestyle=":")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()

def main():
    fpath = "docker/mapping/walk_kicsi.jsonl"
    print("=======================================================")
    print("GYORS PARAMÉTER OPTIMIZÁCIÓÉS TESZTELÉS")
    print("=======================================================")
    
    configs = [
        ("01_szoros_icp", 0.05, 0.20, 0.08, np.radians(1.5), 1.0),
        ("02_tágabb_icp", 0.05, 0.35, 0.15, np.radians(5.0), 1.0),
        ("03_szabad_forgas", 0.05, 0.40, 0.20, np.radians(12.0), 1.0),
        ("04_yaw_korrekcio", 0.05, 0.35, 0.15, np.radians(6.0), 0.98), # 2% yaw skála finomítás
    ]
    
    best_config = None
    best_score = 999.0
    
    for cfg_name, v_res, icp_d, max_t, max_r, y_scale in configs:
        res = evaluate_slam_config(fpath, voxel_res=v_res, max_icp_dist=icp_d, max_trans_corr=max_t, max_rot_corr=max_r, yaw_scale=y_scale)
        # Pontszám: Duplikáció * 50 + Fal vastagság
        score = res['duplication_ratio'] * 50.0 + res['wall_thickness_cm']
        print(f"Konfiguráció [{cfg_name}]: Score={score:.2f} | Duplikáció={res['duplication_ratio']} | FalVastagság={res['wall_thickness_cm']}cm ({res['elapsed_sec']}s)")
        
        img_out = f"docker/mapping/eval_{cfg_name}.png"
        render_slam_map_png(res, img_out, f"[{cfg_name}]")
        
        if score < best_score:
            best_score = score
            best_config = (cfg_name, res)

    print(f"\n🏆 LEGJOBB PARAMÉTER KONFIGURÁCIÓ: {best_config[0]} (Score: {best_score:.2f})")

if __name__ == "__main__":
    main()
