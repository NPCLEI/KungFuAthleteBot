from general_motion_retargeting import RobotMotionViewer, load_robot_motion
from tqdm import tqdm

import argparse
import os,time
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_root_pos = np.load(args.robot_motion_path)

    robot_motion_viewer = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=float(motion_root_pos['fps']),
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    qpos_mtx = motion_root_pos['qpos']
    frame_idx = 0
    while True:
        qpos = qpos_mtx[frame_idx]
        # visualize
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
        time.sleep(1/30)
    robot_motion_viewer.close()