import argparse
import os
import time
import numpy as np
import torch
import mujoco as mj
import mujoco.viewer as mjv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from general_motion_retargeting import ROBOT_XML_DICT, RobotMotionViewer, load_robot_motion

def fit_parabola_and_sample(y0, y1, N, g=9.81):
    T = np.sqrt(2.0 * (y0 - y1) / g)
    t_all = np.linspace(0.0, T, N + 2)
    y_all = y0 - 0.5 * g * t_all**2
    t_mid = t_all[1:-1]
    y_mid = y_all[1:-1]
    return t_all, y_all, t_mid, y_mid

def robust_shape_preserving_smooth(qpos, kernel_size=3):
    T, D = qpos.shape
    t_original = np.arange(T)
    qpos_f = np.zeros_like(qpos)

    for d in range(D):
        x = qpos[:, d]
        x_denoised = medfilt(x, kernel_size=kernel_size)
        pchip = PchipInterpolator(t_original, x_denoised)
        qpos_f[:, d] = pchip(t_original)
        
    return qpos_f

def find_local_maxima_indices(z, resolution_factor=100):
    if len(z) < 3:
        raise ValueError("Input array must have at least 3 elements")
    
    t = len(z)
    x_original = np.arange(t)
    cs = CubicSpline(x_original, z)
    
    def cs_derivative(x):
        return cs(x, 1)
    
    def cs_second_derivative(x):
        return cs(x, 2)
    
    potential_maxima = []
    for i in range(t - 1):
        x_fine = np.linspace(i, i + 1, resolution_factor)
        derivative_values = cs_derivative(x_fine)
        
        for j in range(len(x_fine) - 1):
            if derivative_values[j] * derivative_values[j + 1] <= 0:
                if derivative_values[j] == 0:
                    candidate = x_fine[j]
                elif derivative_values[j + 1] == 0:
                    candidate = x_fine[j + 1]
                else:
                    left = x_fine[j]
                    right = x_fine[j + 1]
                    for _ in range(10):
                        mid = (left + right) / 2
                        if cs_derivative(left) * cs_derivative(mid) <= 0:
                            right = mid
                        else:
                            left = mid
                    candidate = (left + right) / 2
                
                if cs_second_derivative(candidate) < 0:
                    potential_maxima.append(candidate)
    
    if potential_maxima:
        potential_maxima = np.array(potential_maxima)
        unique_maxima = []
        threshold = 0.5
        
        sorted_indices = np.argsort(potential_maxima)
        for idx in sorted_indices:
            x = potential_maxima[idx]
            if not unique_maxima or abs(x - unique_maxima[-1]) > threshold:
                unique_maxima.append(x)
        
        maxima_indices = []
        for x_max in unique_maxima:
            nearest_idx = np.argmin(np.abs(x_original - x_max))
            if nearest_idx == 0:
                if z[nearest_idx] > z[nearest_idx + 1]:
                    maxima_indices.append(nearest_idx)
            elif nearest_idx == t - 1:
                if z[nearest_idx] > z[nearest_idx - 1]:
                    maxima_indices.append(nearest_idx)
            else:
                if (z[nearest_idx] > z[nearest_idx - 1] and 
                    z[nearest_idx] > z[nearest_idx + 1]):
                    maxima_indices.append(nearest_idx)
        
        maxima_indices = sorted(set(maxima_indices))
        return maxima_indices
    else:
        return []

def find_local_minima_indices(z, resolution_factor=100):
    if len(z) < 3:
        raise ValueError("Input array must have at least 3 elements")
    
    t = len(z)
    x_original = np.arange(t)
    cs = CubicSpline(x_original, z)
    
    def cs_derivative(x):
        return cs(x, 1)
    
    def cs_second_derivative(x):
        return cs(x, 2)
    
    potential_minima = []
    for i in range(t - 1):
        x_fine = np.linspace(i, i + 1, resolution_factor)
        derivative_values = cs_derivative(x_fine)
        
        for j in range(len(x_fine) - 1):
            if derivative_values[j] * derivative_values[j + 1] <= 0:
                if derivative_values[j] == 0:
                    candidate = x_fine[j]
                elif derivative_values[j + 1] == 0:
                    candidate = x_fine[j + 1]
                else:
                    left = x_fine[j]
                    right = x_fine[j + 1]
                    for _ in range(10):
                        mid = (left + right) / 2
                        if cs_derivative(left) * cs_derivative(mid) <= 0:
                            right = mid
                        else:
                            left = mid
                    candidate = (left + right) / 2
                
                if cs_second_derivative(candidate) > 0:
                    potential_minima.append(candidate)
    
    if potential_minima:
        potential_minima = np.array(potential_minima)
        unique_minima = []
        threshold = 0.5
        
        sorted_indices = np.argsort(potential_minima)
        for idx in sorted_indices:
            x = potential_minima[idx]
            if not unique_minima or abs(x - unique_minima[-1]) > threshold:
                unique_minima.append(x)
        
        minima_indices = []
        for x_min in unique_minima:
            nearest_idx = np.argmin(np.abs(x_original - x_min))
            if nearest_idx == 0:
                if z[nearest_idx] < z[nearest_idx + 1]:
                    minima_indices.append(nearest_idx)
            elif nearest_idx == t - 1:
                if z[nearest_idx] < z[nearest_idx - 1]:
                    minima_indices.append(nearest_idx)
            else:
                if (z[nearest_idx] < z[nearest_idx - 1] and 
                    z[nearest_idx] < z[nearest_idx + 1]):
                    minima_indices.append(nearest_idx)
        
        minima_indices = sorted(set(minima_indices))
        return minima_indices
    else:
        return []

def get_min_body_z_from_qpos(qpos):
    robot_data.qpos[:] = qpos
    mj.mj_forward(mj_model, robot_data)

    z_vals = robot_data.xpos[:, 2]
    robot_body_ids = np.arange(1, mj_model.nbody)
    robot_z_vals = z_vals[1:]

    min_idx = int(np.argmin(robot_z_vals))
    min_z = float(robot_z_vals[min_idx])
    body_id = robot_body_ids[min_idx]
    body_name = mj_model.body(body_id).name

    return min_z, body_id, body_name

def apply_gravity_until_contact(
    qpos_seq,
    dt,
    contact_z_thresh=0.02,
    g=9.81,
):
    qpos_seq = qpos_seq.copy()
    T, N = qpos_seq.shape

    for t in range(T):
        min_z, body_id, body_name = get_min_body_z_from_qpos(qpos_seq[t])
        has_contacted = min_z <= contact_z_thresh

        if not has_contacted:
            qpos_seq[t, 2] -= 0.5 * g * dt * dt

    return qpos_seq

def adjust_root_z(qpos, window=1, vel_thresh=5e-4):
    qpos = qpos.copy()
    T, N = qpos.shape

    root_z = qpos[:, 2]
    root_vel = np.zeros_like(root_z)
    root_vel[:-1] = root_z[1:] - root_z[:-1]

    rebuild_root_z = np.zeros_like(root_z)
    minidxes = find_local_minima_indices(qpos[:,2])
    maxidxes = find_local_maxima_indices(qpos[:,2])
    rebuild_root_z[0] = root_z[0] - get_min_body_z_from_qpos(qpos[0])[0]
    
    skip = [minidxes[i] for i in [-3]]

    for t in range(1,T):
        if t in minidxes and t not in skip:
            new_z = root_z[t] - get_min_body_z_from_qpos(qpos[t])[0]
            new_qpos_t = qpos[t]
            new_qpos_t[2] = new_z
            if get_min_body_z_from_qpos(new_qpos_t)[0] >= 0:
                rebuild_root_z[t] = new_z
                continue
        rebuild_root_z[t] = rebuild_root_z[t-1] + root_vel[t-1]

    for i in maxidxes:
        for j in minidxes:
            if j in skip:
                break
            if j > i and rebuild_root_z[i] > rebuild_root_z[j]:
                rebuild_root_z[i+1:j] = fit_parabola_and_sample(rebuild_root_z[i],rebuild_root_z[j], j - i - 1)[-1]
                break
    
    qpos[:,2] = rebuild_root_z

    for t in range(T):
        minz = get_min_body_z_from_qpos(qpos[t])[0]
        if minz < 0:
            rebuild_root_z[t] = root_z[t] - get_min_body_z_from_qpos(qpos[t])[0]

    qpos[:,2] = rebuild_root_z
    return qpos, minidxes, maxidxes

def piecewise_savgol(
    qpos,
    window=7,
    poly=3,
    deriv_thresh=4.0,
):
    qpos_f = qpos.copy()
    T, D = qpos.shape

    for d in range(D):
        x = qpos[:, d]
        dx = np.diff(x, prepend=x[0])
        ddx = np.abs(np.diff(dx, prepend=dx[0]))

        turning_points = np.where(ddx > deriv_thresh * np.std(ddx))[0]
        segments = np.split(np.arange(T), turning_points)

        for seg in segments:
            if len(seg) < window:
                continue
            qpos_f[seg, d] = savgol_filter(
                x[seg], window_length=window, polyorder=poly
            )

    return qpos_f

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument(
        "--robot",
        choices=["unitree_g1","unitree_g1_with_hands", "unitree_g1_with_wrist_roll", "unitree_g1_sim2sim",  "unitree_g1_23", 
                 "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung",
                "mini_pi"
            ],
        default="unitree_g1",
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the root z comparison figure (default: False)"
    )
    
    args = parser.parse_args()
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_root_pos = np.load(args.robot_motion_path)
    qpos_mtx = motion_root_pos['qpos']

    robot_xml_path = ROBOT_XML_DICT[args.robot]
    mj_model = mj.MjModel.from_xml_path(str(robot_xml_path))
    robot_data = mj.MjData(mj_model)

    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.plot([i * motion_root_pos['fps'] for i in range(qpos_mtx.shape[0])], qpos_mtx[:,2], label='before', alpha=0.7)
    
    qpos_mtx, minidxes, maxidxes = adjust_root_z(qpos_mtx)
    qpos_mtx = piecewise_savgol(qpos_mtx, window=24, poly=3)

    if args.plot:
        plt.plot([i * motion_root_pos['fps'] for i in range(qpos_mtx.shape[0])], qpos_mtx[:,2], label='after', alpha=0.7)
        plt.scatter([i * motion_root_pos['fps'] for i in minidxes], qpos_mtx[:,2][minidxes], label='local min', color='red', marker='*')
        plt.scatter([i * motion_root_pos['fps'] for i in maxidxes], qpos_mtx[:,2][maxidxes], label='local max', color='green', marker='*')
        plt.xlabel('Time (s)')
        plt.ylabel('Root Z Position (m)')
        plt.title('Root Z Position Before and After Adjustment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=float(motion_root_pos['fps']),
        camera_follow=False,
        record_video=args.record_video, 
        video_path=args.video_path
    )
    
    frame_idx = 0
    while True:
        qpos = qpos_mtx[frame_idx]
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=True
        )
        frame_idx += 1
        if frame_idx >= len(qpos_mtx):
            break
        if not args.record_video:
            time.sleep(1/motion_root_pos['fps'])

    robot_motion_viewer.close()

    if args.save_path is not None:
        folder_path = os.path.dirname(args.save_path)
        os.makedirs(folder_path, exist_ok=True)
        np.savez(
            args.save_path,
            fps=motion_root_pos['fps'],
            qpos=qpos_mtx
        )
        print(f"Saved to {args.save_path}")