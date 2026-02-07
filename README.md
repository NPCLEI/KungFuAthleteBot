# KungfuAthlete

<img src="./docs/cover.png" controls></img>

## Dataset Overview

The dataset originates from athletes’ **daily martial arts training videos**, totaling **197 video clips**.
Each clip may consist of multiple merged segments. We apply **automatic temporal segmentation**, resulting in **1,726 sub-clips**, ensuring that most segments avoid abrupt transitions that could introduce excessive motion discontinuities.

All sub-clips are processed using **GVHMR** for motion capture, followed by **GMR-based reorientation**.
After filtering and post-processing, the final dataset contains **848 motion samples**, primarily reflecting routine training activities.

## Category Distribution

| Category       | Count  | Example Subcategories                                |
| -------------- | ------ | ---------------------------------------------------- |
| Daily Training | 715    | –                                                    |
| Fist           | 53     | Long Fist (33), Tai Chi Fist (14), Southern Fist (6) |
| Staff          | 30     | Staff Technique (30)                                 |
| Skills         | 28     | Backflip (12), Lotus Swing (9)                       |
| Saber / Sword  | 15 / 7 | Southern Saber (15), Tai Chi Sword (7)               |

### Notes

* **Daily Training** dominates the dataset (**715 samples, ~84%**), mainly representing standard practice routines rather than explicit technique demonstrations.
* **Boxing techniques** form the largest specialized category, with **Changquan (Long Fist)** being the most prevalent.
* **Skill-based movements** concentrate on high-difficulty acrobatics such as somersaults and lotus swings.
* Weapon-based motions are limited but structured, focusing on standardized staff, saber, and Tai Chi sword forms.

---

## Motion Statistics Comparison

All metrics are averaged over the entire dataset.

| Dataset                    | FPS  | Joint Vel. | Body Lin. Vel. | Body Ang. Vel. | Average Frames   |
| -------------------------- | ---- | ---------- | -------------- | -------------- | -------- |
| LAFAN1                     | 50.0 | 0.00142    | 0.00021        | 0.01147        | 10749.23 |
| PHUMA                      | 50.0 | 0.00120    | 0.00440        | -0.00131       | 169.59   |
| AMASS                      | 30.0 | 0.00048    | -0.00568       | 0.00903        | 370.65   |
| **KungFuAthlete (Ground)** | 50.0 | -0.00199   | 0.01057        | 0.04034        | 577.68   |
| **KungFuAthlete (Jump)**   | 50.0 | 0.02384    | 0.05297        | 0.18017        | 397.21   |

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





## Demo
<table>
<tr>
<td align="center" width="50%">
<video src="./docs/example_gvhmr_278.mp4" width="400" controls></video>
<br><b>GVHMR</b>
</td>
<td align="center" width="50%">
<video src="./docs/example_g1_after_278.mp4" width="400" controls></video>
<br><b>GMR (after root height adjusted)</b>
</td>
</tr>
</table>

## Pipeline

KungfuAthlete's data pipeline: Kungfu Video → Human Pose Extraction → Robot Motion Conversion → Data Cleaning

```
[Video] → GVHMR → [GVHMR-Pred] → GMR → [Robot Motion] → Cleaning → [KungfuAthlete]
```

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

### GMR qpos

```python
# robot_qpos.npz
{
  "fps": array(30),
  "qpos": np.ndarray,    # (N, 36) 36 = 3(position xyz) + 4(quaternion wxyz) + 29(DOF)
}
```

## Download

You can obtain the KungfuAthlete dataset through **[this link](https://drive.google.com/drive/folders/1ZntW9jPA-BXxttvCWlKQsSbmXt91fSsh?usp=sharing)** and use it directly for your robot training. We provide GVHMR pred data and pre-cleaned **g1** robot qpos data. 

The KungfuAthlete dataset is constructed from publicly available high-dynamic videos on the , which undergo GVHMR action extraction, GMR retargeting, and data cleaning. The KungfuAthlete dataset is divided into two types: **Ground** and **Jump**. Ground indicates that there will always be one foot on the ground during the entire motion, while Jump indicates that both feet are off the ground during motion. 

The following content includes visualizations of GVHMR and GMR data, as well as examples of how we use height adjustment algorithms to process the qpos data. If you wish to apply this dataset to other robots, you can refer to our processing pipeline.

## Project Structure

```
src/
├── demo/                         # Data demo files 
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
├── ./docs/                                  # Document files
│
└── third_party/                           # External dependencies (submodules)
    └── GMR/                               # Motion retargeting
```

## Installation

We have included a video (.mp4) for each action dataset in the download link. If you wish to utilize the root node adjustment feature or visualize the data yourself, please install the third-party packages listed below the repository:


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
cp retarget/scripts/gvhmr_to_qpos.py ./third_party/GMR/scripts/
cp retarget/scripts/vis_robot_qpos.py ./third_party/GMR/scripts/
cp retarget/scripts/adjust_robot_height_by_gravity.py ./third_party/GMR/scripts/
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
<table>
<tr>
<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_before_78.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
<br><b>78 before</b>
</td>

<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_after_78.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
<br><b>78 after</b>
</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_before_117.mp4" type="video/mp4">
</video>
<br><b>117 before</b>
</td>

<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_after_117.mp4" type="video/mp4">
</video>
<br><b>117 after</b>
</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_before_213.mp4" type="video/mp4">
</video>
<br><b>213 before</b>
</td>

<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_after_213.mp4" type="video/mp4">
</video>
<br><b>213 after</b>
</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_before_278.mp4" type="video/mp4">
</video>
<br><b>278 before</b>
</td>

<td align="center" width="50%">
<video width="400" controls>
  <source src="./docs/example_g1_after_278.mp4" type="video/mp4">
</video>
<br><b>278 after</b>
</td>
</tr>
</table>


## Supported Robots

| Robot | ID | DOF |
|-------|-----|-----|
| Unitree G1 | `unitree_g1` | 29 |

See [GMR README](https://github.com/YanjieZe/GMR) for other list


## Video Source and Acknowledgement

<img src="./docs/xieyuanhang.png" alt="Xie Yuanhang" width="160"/>

The video materials used in this project are primarily sourced from a series of publicly released martial arts training and competition demonstration videos by **Xie Yuanhang**.

**Xie Yuanhang** is an athlete of the **Guangxi Wushu Team**, a **National-Level Elite Athlete of China**, and holds the rank of **Chinese Wushu 6th Duan**. He achieved **third place in the Wushu Taolu event at the 10th National Games of the People’s Republic of China**. His video content systematically covers a wide range of **International Wushu Competition Taolu**, including Changquan, Nanquan, weapon routines, and Taijiquan (including Taijijian). The demonstrations are technically precise, rhythmically clear, and of high professional and instructional value.

We would like to express our **special and sincere gratitude to Xie Yuanhang** for his strong support and for **granting permission to use his video materials** in this project. Under this authorization, the dataset is **constructed and processed based on his publicly available videos**, and is intended **solely for research and academic purposes**. His generous support has been instrumental in ensuring the high quality and reliability of this dataset.

🔗 **Personal Homepage (Bilibili):**  
https://space.bilibili.com/1475395086

本项目所使用的视频素材主要来源于 谢远航 教练/运动员在其个人平台公开发布的系列武术训练与竞赛示范视频。谢远航系广西武术队运动员，国家级运动健将，中国武术六段，并曾获得中华人民共和国第十届运动会武术套路项目第三名。其视频内容系统覆盖国际武术竞赛套路中的长拳、南拳、器械及太极拳（剑）等多个项目，动作规范、节奏清晰，具有较高的专业性与示范价值。

在此，我们特别鸣谢谢远航先生对本项目的大力支持与授权，允许我们基于其公开视频素材进行整理、处理与研究使用。本数据集即在上述授权前提下，基于其公开视频内容构建与制作，相关使用仅用于科研与学术目的。谢远航先生的无私支持为本数据集的高质量构建提供了重要保障，在此谨致以诚挚感谢。

## Acknowledgements

This project builds upon the following excellent open source projects:

- [GVHMR](https://github.com/zju3dv/GVHMR): 3D human mesh recovery from video
- [GMR](https://github.com/YanjieZe/GMR): general motion retargeting framework

## License

This project depends on third-party library with its own licenses:


Please review this licenses before use.
