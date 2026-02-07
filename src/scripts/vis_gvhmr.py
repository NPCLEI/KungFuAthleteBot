import os
import argparse
import pathlib
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
import torch
from tqdm import tqdm
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_file", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()

def render_smpl_pose(smpl, motion_parms) -> list:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["PYRENDER_PLATFORM"] = "egl"

    import pyrender
    import trimesh
    
    smpl_joint_map = {name: i for i, name in enumerate(JOINT_NAMES)}
    smpl_body_names = [
        'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle',
        'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist'
    ]
    smpl_keypoint_indices = [smpl_joint_map[name] for name in smpl_body_names]
    pelvis_joint_idx = smpl_joint_map["pelvis"]

    viewport_width = 1280
    viewport_height = 960
    renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)

    smpl_vertices_all_frame = []
    smpl_joints_all_frame = []

    transl = motion_parms["transl"]
    num_frames = transl.shape[0]

    for i in range(num_frames):
        motion_parms_frame = {k: v[None, i] for k, v in motion_parms.items()}
        output = smpl(**motion_parms_frame)
        
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        
        smpl_vertices_all_frame.append(vertices)
        smpl_joints_all_frame.append(joints)

    avg_width = 0.5 * (min(transl[:, 0]) + max(transl[:, 0]))
    avg_depth = 0.5 * (min(transl[:, 2]) + max(transl[:, 2]))
    avg_height = 0.5 * (min(transl[:, 1]) + max(transl[:, 1]))

    frames = []
    SMPL_COLOR = [0.98, 0.855, 0.369, 1.0]
    FLOOR_COLOR = [0.95, 0.95, 0.9, 1.0]
    BG_COLOR = [0.98, 0.98, 0.98, 1.0]

    for vertices, joints in tqdm(zip(smpl_vertices_all_frame, smpl_joints_all_frame), total=num_frames):
        vertices = vertices - [avg_width, avg_height, avg_depth]
        joints = joints[smpl_keypoint_indices] - [avg_width, avg_height, avg_depth]

        current_frame_joints = smpl_joints_all_frame[len(frames)]
        current_frame_joints_normalized = current_frame_joints - [avg_width, avg_height, avg_depth]
        pelvis_x, pelvis_y, pelvis_z = current_frame_joints_normalized[pelvis_joint_idx]

        vertex_colors = np.ones([vertices.shape[0], 4]) * SMPL_COLOR
        tri_mesh = trimesh.Trimesh(vertices, smpl.faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        
        scene = pyrender.Scene(bg_color=BG_COLOR)
        scene.ambient_light = [0.1, 0.1, 0.1]
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5)
        cam_pose = np.eye(4)
        cam_pose[0, 3] = pelvis_x
        cam_pose[1, 3] = pelvis_y
        cam_pose[2, 3] = pelvis_z + 2.0
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4)
        light_pose[0, 3] = pelvis_x
        light_pose[1, 3] = pelvis_y + 2.0
        light_pose[2, 3] = pelvis_z + 4.0
        
        tilt_angle = np.radians(30)
        tilt_rotation = trimesh.transformations.rotation_matrix(tilt_angle, [1, 0, 0])
        light_pose = np.dot(light_pose, tilt_rotation) 

        light_node = scene.add(light, pose=light_pose)
        light_node.shadow_map_size = (4096, 4096)
        light_node.shadow_near = 0.0001
        light_node.shadow_far = 50.0
        light_node.shadow_bias = 0.0005

        plane_thickness = 0.001
        plane_width = 50
        plane_depth = 50

        plane_box = trimesh.primitives.Box(
            extents=[plane_width, plane_depth, plane_thickness],
            transform=np.eye(4)
        )
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        plane_box.apply_transform(R)
        plane_box.apply_translation([0, plane_thickness / -2.0 - avg_height, 0])
        
        plane_color = np.ones((plane_box.vertices.shape[0], 4), dtype=np.float32) * FLOOR_COLOR
        plane_box.visual.vertex_colors = plane_color
        plane_mesh = pyrender.Mesh.from_trimesh(plane_box, smooth=False)
        scene.add(plane_mesh)

        color, _ = renderer.render(scene)
        frames.append(color)

    renderer.delete()
    return frames

def write_video(file: str, frames: list, fps: int = 30):
    if not frames:
        raise ValueError("No frames to write!")
        
    os.makedirs(os.path.dirname(file), exist_ok=True)
    writer = imageio.get_writer(file, fps=fps)
    
    for frame in tqdm(frames, desc="writing video"):
        writer.append_data(frame)
    
    writer.close()
    print(f"video written to: {file}")

def main(args):
    if not os.path.exists(args.pose_file):
        raise FileNotFoundError(f"Pose file not found: {args.pose_file}")
    if not args.pose_file.endswith('.pt'):
        raise ValueError(f"Invalid pose file format: {args.pose_file} (must be .pt)")
    
    pred = torch.load(args.pose_file)
    smplx_data = pred["smpl_params_global"]
    
    HERE = pathlib.Path(__file__).parent
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    body_model = smplx.create(
        SMPLX_FOLDER, 
        model_type="smplx", 
        gender="neutral", 
        num_pca_comps=45
    )

    render_motion_parms = {
        'transl': smplx_data['transl'].clone(),
        'global_orient': smplx_data['global_orient'].clone(),
        'body_pose': smplx_data['body_pose'].clone(),
    }

    if not args.save_path.endswith('.mp4'):
        raise ValueError(f"Invalid save path format: {args.save_path} (must be .mp4)")
    frames = render_smpl_pose(body_model, render_motion_parms)
    write_video(args.save_path, frames, fps=args.fps)

if __name__ == '__main__':
    args = parse_args()
    main(args)