# KungfuAthlete

<img src="./docs/cover_ub.png" controls></img>


[![Project Page](https://img.shields.io/badge/Project-Homepage-green?style=for-the-badge&logo=googlechrome)](https://kungfuathletebot.github.io/)
&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2602.13656)
&nbsp;
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/8v2pAaRQcPw)
&nbsp;
[![Bilibili](https://img.shields.io/badge/Bilibili-Video-orange?style=for-the-badge&logo=bilibili)](https://www.bilibili.com/video/BV1xJZCBrE4d/)
&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-Data-blue?style=for-the-badge&logo=dataverse)](https://drive.google.com/drive/folders/1ZntW9jPA-BXxttvCWlKQsSbmXt91fSsh?usp=sharing)
&nbsp;
[![Hugging Face](https://img.shields.io/badge/HuggingFace-KungfuAthleteBot-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/LuluCao/KungfuAthleteBot)

## Dataset Overview

The dataset originates from athletes’ **daily martial arts training videos**, totaling **197 video clips**.
Each clip may consist of multiple merged segments. We apply **automatic temporal segmentation**, resulting in **1,726 sub-clips**, ensuring that most segments avoid abrupt transitions that could introduce excessive motion discontinuities.

All sub-clips are processed using **GVHMR** for motion capture, followed by **GMR-based reorientation**.
After filtering and post-processing, the final dataset contains **848 motion samples**, primarily reflecting routine training activities.

Due to substantial noise in the source videos, the dataset has undergone multiple rounds of manual screening and meticulous per-sample refinement. While minor imperfections may still remain in certain samples, we ensure that the vast majority satisfy the requirements for reliable motion tracking.

> **Project Status: Active Development with Ready Ground Subset**
> Model training is currently under active development.
> The **Ground** subset of the dataset is largely complete and ready for training.
> The **Jump** subset still has minor imperfections due to video source limitations. Most samples have been carefully screened, though training performance may vary.
> Your feedback and suggestions are greatly appreciated.

## 🤩 What’s New: Mjlab-Based Fall Recovery Code (Open Source)

This project is built upon the open-source **[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)** framework from Unitree Robotics. We have extended it with implementations of fall recovery and LKE sampling modules, and hereby publicly release the code repository related to 1307 motion training.

See the [Training with Unitree RL Mjlab](#code) section for detailed usage.

## Open-Source Status

Todo checklist:

- [x] Dataset
- [x] Height-Adjusted Code
- [x] Training Code
- [x] 1307(Tai Chi) Fall Recovery Checkpoint
- [x] Real Deployment
- [ ] More Data

## Dataset Category Distribution

| Category       | Count  | Example Subcategories                                       |
| -------------- | ------ | ----------------------------------------------------------- |
| Daily Training | 624    | Training(624)                                               |
| Saber / Sword  | 111    | Saber(50), Sword(37), Southern Saber(15), Tai Chi Sword(9)  |
| Fist           | 98     | Long Fist(69), Tai Chi Fist(21), Southern Fist(8)           |
| Staff          | 90     | Staff Technique(90)                                         |
| Skills         | 69     | Side Flip(35), Air Spin(18), Lotus Swing(13), Backflip(3)   |


| Category | Total | Subcategory          | Count |
| -------- | ----- | --------------------| ----- |
| Ground   | 822   | Daily Training       | 550   |
|          |       | Fist                | 93    |
|          |       | Saber / Sword        | 99    |
|          |       | Staff               | 80    |
| Jump     | 170   | Daily Training       | 74    |
|          |       | Fist                | 5     |
|          |       | Saber / Sword        | 12    |
|          |       | Skills              | 69    |
|          |       | - Side Flip          | 35    |
|          |       | - Air Spin           | 18    |
|          |       | - Lotus Swing        | 13    |
|          |       | - Backflip           | 3     |
|          |       | - Other Skills       | 0     |
|          |       | Staff               | 10    |

### Notes

* The dataset consists of **992 motion samples** in total, divided into **Ground (822)** and **Jump (170)** categories.
* **Daily Training** accounts for the largest proportion, with **624 samples** across both ground and jump motions, mainly reflecting basic routine movements.
* Ground motions include **Fist (93), Saber / Sword (99)**, and **Staff (80)**; jump motions include a small number of **Fist (5), Saber / Sword (12)**, and **Staff (10)** techniques.
* **Skill-based movements** only exist in jump motions, totaling **69 samples**, focusing on high-difficulty acrobatics such as **Side Flip (35), Air Spin (18), Lotus Swing (13)**, and **Backflip (3)**.
* Weapon-related motions only contain overall movement sequences and do not include detailed weapon or hand operations (to be updated in future versions).
---

## Motion Statistics Comparison

All metrics are averaged over the entire dataset.

| Dataset                    | Body Lin Vel(w) | Body Ang Vel(w) | Frames Range |
| -------------------------- | --------------- | --------------- | ------------ |
| AMASS                      | 0.188±0.165     | 0.510±0.309     | [2,8788]     |
| PHUMA                      | 0.193±0.130     | 0.585±0.368     | [49,222]     |
| LAFAN1                     | 0.790±0.251     | 1.962±0.553     | [3065,9171]  |
| **KungFuAthlete (Ground)** | 0.496±0.324     | 1.537±0.959     | [24,15792]   |
| **KungFuAthlete (Jump)**   | 0.948±0.473     | 2.832±1.431     | [25,2842]    |

---

## Ground vs. Jump Subsets

We divide the dataset based on the presence of jumping motions:

* **KungFuAthlete (Ground)**
  Contains non-jumping actions, emphasizing:

  * Continuous ground-based power generation
  * Rapid body rotations
  * Weapon manipulation and stance transitions

* **KungFuAthlete (Jump)**
  Includes high-dynamic aerial motions such as:

  * Somersaults
  * Cartwheels
  * Other acrobatic jumps

### Key Observations

* The **Jump subset** exhibits the **highest joint velocity, body linear velocity, and angular velocity** among all compared datasets.
* The **Ground subset**, while excluding jumps, still shows significantly higher dynamics than natural motion datasets (e.g., LAFAN1, AMASS).
* Compared to PHUMA and AMASS, which focus on daily activities and walking motions, **KungFuAthlete demonstrates stronger non-stationarity, larger motion amplitudes, and more challenging transient dynamics**, even at comparable or higher frame rates.

### ⚠️ Safety Warning

> The *Jump* subset is intended for studying the upper limits of humanoid motion. Direct training or deployment on real robots without strict safety constraints may lead to serious hardware damage.

## Demo

<table>
<tr>
<td align="center" width="50%">

![](./docs/example_gvhmr_278.gif)  
**278 GVMR SMPL**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_278.gif)  
**278 GMR, Height-Adjusted**

</td>
</tr>
</table>

## Pipeline

KungfuAthlete's data pipeline: 

```
[Video] → Cut by scene → GVHMR → [GVHMR-Pred (smplh)] → GMR → [Robot Motion (qpos)] → Artificial Selection → Height-Adjusted → [KungfuAthlete]
```

### Supported Robots

| Robot | ID | DOF |
|-------|-----|-----|
| Unitree G1 | `unitree_g1` | 29 |

See [GMR README](https://github.com/YanjieZe/GMR) for other list


## Data Format

### GVHMR pred

```python
# gvhmr_pred.pt
{
  "smpl_params_global":
  {
    "body_pose": torch.Tensor,    # (N, 63)
    "betas": torch.Tensor,    # (N, 10)
    "global_orient": torch.Tensor,    # (N, 3)
    "transl": torch.Tensor,    # (N, 3)
  }
  "smpl_params_incam":
  ...
}
```

If you intend to perform motion retargeting to other robotic platforms, or to employ alternative retargeting methodologies, please use the SMPL-H files provided in the  `collection_gvhmr` directory.

### GMR qpos (`org_smoothed/`)

```python
# robot_qpos.npz
{
  "fps": array(30),
  "qpos": np.ndarray,    # (N, 36) 36 = 3(position xyz) + 4(quaternion wxyz) + 29(DOF)
}
```


### For BeyondMimic (`org_smoothed_mj/`)

All motion files under `org_smoothed_mj/` are stored in NumPy `.npz` format and contain time-sequential kinematic data for Unitree G1 training. Each file represents a single motion clip sampled at `fps` Hz and includes joint-level states and rigid-body states expressed in the **world coordinate frame** (quaternion format: xyzw; units: meters, radians, m/s, rad/s).

```python
motion_clip = {
    "fps": np.array([50]),               # Frame rate (Hz)
    "joint_pos": np.ndarray,             # Shape (T, 36), joint DoF positions (rad)
    "joint_vel": np.ndarray,             # Shape (T, 35), joint angular velocities (rad/s)
    "body_pos_w": np.ndarray,            # Shape (T, 53, 3), rigid body positions in world frame (m)
    "body_quat_w": np.ndarray,           # Shape (T, 53, 4), rigid body orientations in world frame (xyzw)
    "body_lin_vel_w": np.ndarray,        # Shape (T, 53, 3), rigid body linear velocities (m/s)
    "body_ang_vel_w": np.ndarray,        # Shape (T, 53, 3), rigid body angular velocities (rad/s)
    "joint_names": np.ndarray,           # Shape (29,), controllable joint names (ordered)
    "body_names": np.ndarray,            # Shape (53,), rigid body names (ordered, includes head_link)
}
```

Here, `T` denotes the number of frames in the motion clip.

## Download Dataset

You can obtain the KungfuAthlete dataset through **[this link](https://drive.google.com/drive/folders/1ZntW9jPA-BXxttvCWlKQsSbmXt91fSsh?usp=sharing)** and use it directly for your robot training. We provide GVHMR pred data and pre-cleaned **g1** robot qpos data. 

The KungfuAthlete dataset is constructed from publicly available high-dynamic videos on the , which undergo GVHMR action extraction, GMR retargeting, and data cleaning. The KungfuAthlete dataset is divided into two types: **Ground** and **Jump**. Ground indicates that there will always be one foot on the ground during the entire motion, while Jump indicates that both feet are off the ground during motion. 

The following content includes visualizations of GVHMR and GMR data, as well as examples of how we use height adjustment algorithms to process the qpos data. If you wish to apply this dataset to other robots, you can refer to our processing pipeline.

## Project Structure

```
docs/                                      # Document files
unitree_rl_mjlab/                          # Unitree RL Mjlab Training Code
retarget/                                  # Data Adjustment code
├── demo/                                  # Data demo files 
│   ├── gvhmr/                             # Pose data (gvhmr-pred .pt)
│   │   ├── ground/                        # One foot always on the ground data
│   │   └── jump/                          # Data containing jumping actions
│   └── g1/                                # g1 data (robot qpos .npz)
│       ├── ground/                        # One foot always on the ground data
│       └── jump/                          # Data containing jumping actions
│
├── scripts/                               # KungfuAthlete scripts
│   ├── vis_gvhmr.py                       # Vis gvhmr data
│   ├── adjust_robot_height_by_gravity.py  # Newly added GMR script
│   ├── vis_robot_qpos.py                  # Newly added GMR script
│   └── gvhmr_to_qpos.py                   # Newly added GMR script
│
└── third_party/                           # External dependencies (submodules)
    └── GMR/                               # Motion retargeting
```

## Training with Unitree RL Mjlab (BeyondMimic-style) <a id="code"></a>

### 1. Installation

Use the following command to create a virtual environment:

```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
cd unitree_rl_mjlab
pip install -e .
```

If you encounter any issues during installation, please refer to [the installation documentation for Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/doc/setup_en.md).

### 2. Prepare Motion Files

We provide a **ready-to-use** 1307 motion file for direct training.

To use other motions, run the commands below to convert qpos files in the `org_smoothed/` directory to NPZ training files.

```bash
python scripts/qpos_to_npz.py \
--input-file org_smooth/ground_smoothed/1307/1307.npz \
--output-name 1307.npz \
```

### 3. Training

After confirming the NPZ file are prepared, you can launch training. For detailed training parameters, please refer to [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/README.md#22-training).

Our training is divided into **three stages**:
- Stage 1: Enable the policy to roughly track motions and acquire basic fall recovery capabilities.
  ```bash
  python scripts/train.py Unitree-G1-1307-Stage-I --motion_file=src/assets/motions/g1/1307.npz --env.scene.num-envs=8192 --env.commands.motion.sampling-mode=adaptive
  ```
- Stage 2: Improve the precision of motion tracking for the policy.
  ```bash
  python scripts/train.py Unitree-G1-1307-Stage-II --motion_file=src/assets/motions/g1/1307.npz --env.scene.num-envs=8192 --env.commands.motion.sampling-mode=adaptive --agent.resume=True
  ```
- Stage 3: Enhance the robustness of the policy to reduce the likelihood of falling.
  ```bash
  python scripts/train.py Unitree-G1-1307-Stage-III --motion_file=src/assets/motions/g1/1307.npz --env.scene.num-envs=8192 --env.commands.motion.sampling-mode=adaptive --agent.resume=True
  ```

### Released Models

Model Path: `unitree_rl_mjlab\models\`

* **1307** — Tai Chi sequence (~5 minutes continuous motion)

#### ⚠️ Safety Notice

> ⚠️ **Deployment to physical hardware involves inherent risks.**
> Although these policies demonstrate fall-resilient behavior in simulation, real-world execution may lead to unexpected dynamics, instability, or hardware stress.
> Users are strongly advised to conduct careful validation in simulation before deploying to real robots. We assume no responsibility for any potential hardware damage resulting from improper use or deployment.

### Inference

To visualize policy behavior in MuJoCo:

```bash
python scripts/play.py Unitree-G1-1307-Checkpoint --motion_file=src/assets/motions/g1/1307.npz --checkpoint_file=models/1307/1307.pt
```

<table>
<tr>
<td align="center" width="50%">

![](./docs/1307_simulation_mujoco.gif)  
**1307 Motion Inference Example**

</td>
<td align="center" width="50%">

![](./docs/1307_real_deployment.gif)  
**1307 Motion Real Deployment**

</td>
</tr>
</table>

### Real and Simulation Deployment (Same to Unitree RL Mjlab)

You can use the motion file 1307.npz, together with the policy files policy.onnx and policy.onnx.data, for both real and simulation deployment.

For detailed deployment instructions and code, please refer to [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/README.md#4-real-deployment).

## For Data Adjustment

We have included a video (.mp4) for each action dataset in the download link. If you wish to utilize the root node adjustment feature or visualize the data yourself, please navigate to the `retarget/` directory and install the third-party packages listed below the repository:

### 1. GMR Environment (Robot Retargeting)

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
cd third_party/GMR
pip install -e .
cd ./..
```

### 2. Vis-GVHMR Environment (Pose Visualization)

```bash
conda create -n vis-gvhmr python=3.9 -y
conda activate vis-gvhmr
pip install -r requirements.txt
```

### 3. Add new GMR scripts

To use GMR-retargeted data for training, we have added scripts to GMR that adapt the data to training program required qpos format.

```bash
cp ./scripts/gvhmr_to_qpos.py ./third_party/GMR/scripts/
cp ./scripts/vis_robot_qpos.py ./third_party/GMR/scripts/
cp ./scripts/adjust_robot_height_by_gravity.py ./third_party/GMR/scripts/
```
## Usage

> **Note**: Our **height adjustment algorithm** only applies to **qpos data** that is retargeted to **jump** type data, and **ground** type data does not require height adjustment after retargeting.

> **Note**: The GVHMR visualization script and the GMR project rely on different environments. Please ensure that you are in the correct file directory and conda environment (`gmr` or `vis-gvhmr`) when executing different tasks.

### Ground data

```bash
# Visualize GVHMR data (conda env: vis-gvhmr, directory: KungfuAthlete/)
conda activate vis-gvhmr
python scripts/vis_gvhmr.py --pose_file ./KungfuAthlete/gvhmr/ground/3/3.pt --save_path ./KungfuAthlete/gvhmr/ground/3/3.mp4

# Retarget to robot motion (conda env: gmr, directory: KungfuAthlete/third_party/GMR/)
conda activate gmr
cd third_party/GMR/
python scripts/gvhmr_to_qpos.py --gvhmr_pred_file=././KungfuAthlete/gvhmr/ground/3/3.pt --save_path=././KungfuAthlete/g1/ground/3/3.npz --record_video --video_path=././KungfuAthlete/g1/ground/3/3.mp4

# Visualize GMR data (conda env: gmr, directory: KungfuAthlete/third_party/GMR/)
conda activate gmr
python scripts/vis_robot_qpos.py --robot_motion_path=././KungfuAthlete/g1/ground/3/3.npz --record_video --video_path=././KungfuAthlete/g1/ground/3/3.mp4
```

### Jump data

```bash
# Visualize GVHMR data (conda env: vis-gvhmr, directory: KungfuAthlete/)
conda activate vis-gvhmr
python scripts/vis_gvhmr.py --pose_file ./KungfuAthlete/gvhmr/jump/278/278.pt --save_path ./KungfuAthlete/gvhmr/jump/278/278.mp4

# Retarget to robot motion (conda env: gmr, directory: KungfuAthlete/third_party/GMR/)
conda activate gmr
cd third_party/GMR/
python scripts/gvhmr_to_qpos.py --gvhmr_pred_file=././KungfuAthlete/gvhmr/jump/278/278.pt --save_path=././KungfuAthlete/g1/jump/278/278_before.npz --record_video --video_path=././KungfuAthlete/g1/jump/278/278_before.mp4

# Adjust height (conda env: gmr, directory: KungfuAthlete/third_party/GMR/)
conda activate gmr
python scripts/adjust_robot_height_by_gravity.py --robot_motion_path=././KungfuAthlete/g1/jump/278/278_before.npz --save_path=././KungfuAthlete/g1/jump/278/278_after.npz --record_video --video_path=././KungfuAthlete/g1/jump/278/278_after.mp4

# Visualize GMR data (conda env: gmr, directory: KungfuAthlete/third_party/GMR/)
conda activate gmr
python scripts/vis_robot_qpos.py --robot_motion_path=././KungfuAthlete/g1/jump/278/278_after.npz --record_video --video_path=././KungfuAthlete/g1/jump/278/278_vis.mp4
```

## The Height-Adjusted Examples
When predicting jump-type actions or data with excessive leg movement variations, the root node of GVHMR often exhibits height drift over time. We propose an extreme point correction method to adjust the root node height.
<table>
<tr>
<td align="center" width="50%">

![](./docs/example_g1_before_78.gif)  
**78 before**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_78.gif)  
**78 after**

</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">

![](./docs/example_g1_before_117.gif)  
**117 before**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_117.gif)  
**117 after**

</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">

![](./docs/example_g1_before_213.gif)  
**213 before**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_213.gif)  
**213 after**

</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">

![](./docs/example_g1_before_278.gif)  
**278 before**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_278.gif)  
**278 after**

</td>
</tr>
</table>

## Video Source and Acknowledgement

<table>
<tr>
<td width="180" valign="top">

<img src="./docs/xyh_1.jpg" alt="Xie Yuanhang" width="160"/>

</td>
<td valign="top">

<p>
<strong>Xie Yuanhang</strong> is an athlete of the <strong>Guangxi Wushu Team</strong>,
a <strong>National-Level Elite Athlete of China</strong>, and holds the rank of
<strong>Chinese Wushu 6th Duan</strong>. He achieved <strong>third place in the Wushu Taolu event
at the 10th National Games of the People’s Republic of China</strong>.
His video content systematically covers a wide range of
<strong>International Wushu Competition Taolu</strong>, including Changquan, Nanquan,
weapon routines, and Taijiquan (including Taijijian).
</p>

<p>
We would like to express our <strong>special and sincere gratitude to Xie Yuanhang</strong>
for granting permission to use his video materials for
<strong>research and academic purposes</strong>.
</p>

<p>
🔗 <strong>Personal Homepage (Bilibili):</strong><br>
<a href="https://space.bilibili.com/1475395086">
https://space.bilibili.com/1475395086
</a>
</p>

<p>
本项目所使用的视频素材主要来源于谢远航教练/运动员在其个人平台公开发布的系列武术训练与竞赛示范视频。谢远航系广西武术队运动员，国家级运动健将，中国武术六段，并曾获得中华人民共和国第十届运动会武术套路项目第三名。其视频内容系统覆盖国际武术竞赛套路中的长拳、南拳、器械及太极拳（剑）等多个项目，动作规范、节奏清晰，具有较高的专业性与示范价值。
</p>

<p>
在此，我们特别鸣谢谢远航先生对本项目的大力支持与授权，允许我们基于其公开视频素材进行整理、处理与研究使用。本数据集仅用于科研与学术目的。
</p>

</td>
</tr>
</table>

## Acknowledgement

This project builds upon the following excellent open-source projects:

* [GVHMR](https://github.com/zju3dv/GVHMR): 3D human mesh recovery from video
* [GMR](https://github.com/YanjieZe/GMR): general motion retargeting framework
* [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab): A lightweight, modular framework for RL robotics research and sim-to-real deployment.

We gratefully acknowledge these projects, upon which this dataset and training pipeline are built. GVHMR recovers 3D human motion directly in a gravity-aligned reference frame, enabling physically consistent motion reconstruction from raw training videos. GMR is used for motion reorientation and normalization. Unitree RL Mjlab is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots. Without these open-source projects, large-scale processing of in-the-wild martial arts videos and the open release of this dataset would not have been feasible.  


## License

This project depends on third-party library with its own licenses:


Please review these licenses before use.

## Citation

If you use this project in your research, please consider citing:

```bibtex
@article{lei2026kungfuathletebot,
  author  = {Zhongxiang Lei and Lulu Cao and Xuyang Wang and Tianyi Qian and Jinyan Liu and Xuesong Li},
  title   = {A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking},
  year    = {2026},
  eprint  = {2602.13656},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
