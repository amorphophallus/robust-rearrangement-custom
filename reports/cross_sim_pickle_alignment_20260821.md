# 跨仿真器 pickle 与 LMDB 对齐审查

审查日期：2026-08-21（Asia/Shanghai）

审查的仓库：

- robust-rearrangement：`f62eff3`
- [hmnkapa/ManiSkill](https://github.com/hmnkapa/ManiSkill)：`d1ead54`
- [hmnkapa/IsaacLab](https://github.com/hmnkapa/IsaacLab)：`72e9db9`

## 总体结论

共享数据约定、真实三仿真器验证集以及 robust-rearrangement 训练路径均已实现，并在每个仿真器各一个任务上完成验证。这是一个已经闭环的最小集成结果，不代表所有已暴露的 ManiSkill 任务或全部 100 个 AutoMate 装配体都已得到验证。

- FurnitureBench：3 条真实成功的 `one_leg` 轨迹已迁移为标准原始 pickle。全部 969 个 observation 均通过共享严格验证，随后完成 224x224 LMDB 转换和完整 LMDB 验证。第二轮几何审查发现，旧数据中的点、位姿、零件和相机均处于仿真器局部坐标系；迁移逻辑和后续写入现在会把它们一起变换到机械臂 base 坐标系。修正后，全部 968 组可比较的 oracle 3D 到 2D 投影误差均不超过 0.70 px。
- ManiSkill：action、state、图像、`env`、标注、writer 和 validator 的实现已经对齐。候选补丁应用在 `d1ead54` 的干净 checkout 上，为生成器支持的所有任务覆盖 front camera 位姿/FOV，同时把标量夹爪状态修正为 `(1,) float32`。r218 驱动重启并创建干净环境后，3 条真实成功的 `PickCube-v1` 轨迹同时通过 ManiSkill 原生 writer validator 和 RR 的共享严格 validator。
- AutoMate：明确接受 EULA 后，使用随机采样的 SRSA 策略为装配体 `00211` 采集了 3 条真实成功轨迹。两个相机均以 320x240 渲染后中心裁剪到 224；保存的 front camera 标定为 `fx=fy=308.092`、平移 `[1.2,0,0.235]`，前向轴与 FB 对齐。每个 observation 都包含 `skill="insert"`、robot-base 坐标系中的 fixed-asset-tip guidance point，以及由标定投影得到且位于图像内的 front pixel。3 个文件均通过原生 writer validator 和 RR 独立严格 validator。
- 最终真实 LMDB 每个仿真器恰好包含 3 个 episode：共 9 episodes、1,219 transitions，保存 224x224 RGB-D。完整 LMDB 重扫、包含三来源的 lazy batch、双 rank 精确 `50/35/15` 采样、短程 CPU/Gloo DDP 优化，以及两条 depth checkpoint 恢复路径均通过。最终 front depth 统计有效，但 AutoMate 的开放场景包含很远的背景，因此标准差较大；这是已记录的数据设计问题，不是 schema 错误。

### 9 条成功 rollout 视频

以下视频均由对应真实成功 pickle 直接生成。画面左侧为 wrist、右侧为 front；红色标记表示存在的 `guidance_point_2d`。视频使用 H.264、10 fps，并在首尾各停留 1 秒，便于检查初态和成功终态。

| 来源 | 轨迹 | 原始 observation 数 | 视频 |
|---|---:|---:|---|
| FurnitureBench `one_leg` | 1/3 | 323 | [01_furniturebench_one_leg_01.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/01_furniturebench_one_leg_01.mp4) |
| FurnitureBench `one_leg` | 2/3 | 323 | [02_furniturebench_one_leg_02.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/02_furniturebench_one_leg_02.mp4) |
| FurnitureBench `one_leg` | 3/3 | 323 | [03_furniturebench_one_leg_03.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/03_furniturebench_one_leg_03.mp4) |
| ManiSkill `PickCube-v1` | 1/3 | 75 | [04_maniskill_pickcube-v1_01.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/04_maniskill_pickcube-v1_01.mp4) |
| ManiSkill `PickCube-v1` | 2/3 | 75 | [05_maniskill_pickcube-v1_02.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/05_maniskill_pickcube-v1_02.mp4) |
| ManiSkill `PickCube-v1` | 3/3 | 51 | [06_maniskill_pickcube-v1_03.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/06_maniskill_pickcube-v1_03.mp4) |
| AutoMate `00211` | 1/3 | 11 | [07_automate_00211_01.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/07_automate_00211_01.mp4) |
| AutoMate `00211` | 2/3 | 10 | [08_automate_00211_02.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/08_automate_00211_02.mp4) |
| AutoMate `00211` | 3/3 | 37 | [09_automate_00211_03.mp4](../logs/cross-sim-alignment-20260821/visual/rollout_videos/09_automate_00211_03.mp4) |

## 需求矩阵

| 需求 | FurnitureBench | ManiSkill | AutoMate |
|---|---|---|---|
| Front camera 与 FB 对齐 | 裁剪并将外参从 simulator-local 迁移到 base 后通过 | 真实渲染轨迹通过：224x224，`fx=307.717`，base 平移 `[1.2,0,0.235]` | 真实渲染轨迹通过：224x224，`fx=308.092`，base 平移 `[1.2,0,0.235]`，前向轴与 FB 对齐 |
| Action `(T,8)` = delta xyz + delta quat xyzw + absolute gripper | 新写入通过；旧样本数值为 delta，但错误标为 `action_type="pos"` | 通过，包括原生夹爪符号到 RR 语义的转换 | 通过；已观测/已实现的 collector 只覆盖固定 `+1` 闭合 |
| Gripper `-1=open`、`+1=closed`，范围 `[-1,1]` | 通过 | 通过 | 格式通过；尚无打开夹爪的行为覆盖 |
| 机器人、零件和 guidance 几何均在 robot-base 坐标系 | parts、3D guidance、pose、camera 和 state alias 联合迁移后通过 | 3 条真实轨迹严格验证通过；标量字段均为精确 `(1,) float32` | 3 条真实轨迹严格验证通过；58 个 observation 全部包含插入点和标定像素 |
| RGB/depth 为 224x224 | 新写入和 `--image-size 224` 转换通过；审查的旧原始文件为 240x320 | 通过 | 通过 |
| 顶层 `env` 区分来源 | `FurnitureBench` | `ManiSkill` | `AutoMate` |
| 提供 point/skill guidance | 通过 | 已为支持的脚本规划器/FSM 实现 | 通过：每个真实 observation 都有几何导出的 `insert` 点和标定 front pixel |
| 一个任务、3 条真实 rollout 的最小验证 | 通过：`one_leg`，3 次成功 | 通过：`PickCube-v1`，3 次成功 | 通过：装配体 `00211`，5 次尝试中 3 次成功 |
| 合并真实三来源 LMDB | 通过 | 通过 | 通过：精确 env 计数 `3/3/3` |

## 标准原始 pickle 约定

共享顶层对象如下：

| Key | 类型/形状 | 含义 |
|---|---|---|
| `env` | string | 严格为 `FurnitureBench`、`ManiSkill` 或 `AutoMate` |
| `task` | string | 任务标识符 |
| `success` | bool | episode 结果 |
| `action_type` | string | `delta` |
| `observations` | 长度为 `T+1` 的 list | `T` 个 transition 之前、之间和之后的状态 |
| `actions` | float array/list，`(T,8)` | robot base 下的 `[dx,dy,dz,dqx,dqy,dqz,dqw,gripper]` |
| `rewards` | float array/list，`(T,)` | 每个 action 一个值 |
| `camera_info.front_camera` | mapping | 最终图像对应的标定 |

Action 四元数为 `xyzw` 顺序的单位四元数。相对旋转定义为 `inverse(current_ee_quat) * target_ee_quat`。夹爪项是 `[-1,1]` 范围内的绝对命令，不是增量。

每个 observation 包含：

- `robot_state`
- `color_image1`、`color_image2`：wrist/front `uint8`，`(224,224,3)`
- `depth_image1`、`depth_image2`：wrist/front `float32`，`(224,224)`，单位为米；非负，0 视为无效深度
- `parts_poses`：展开的 base-frame 零件位姿，`7*N`，四元数为 `xyzw`
- `point_cloud`
- `skill`
- `guidance_point`、`guidance_point_clean`
- `guidance_pose`、`guidance_pose_clean`
- `guidance_gripper_width`
- `guidance_point_2d`、`grasp_annotation_2d`

`parts_poses`、`guidance_point{,_clean}` 和 `guidance_pose{,_clean}` 与 robot state、action 使用同一 robot-base 坐标系。这是语义约定，而不只是形状约定。对于存在可比较 oracle front pixel 的记录，严格 validator 会用保存的标定投影 3D 点，误差超过 1 px 即拒绝。VLM 预测像素会与独立 oracle 像素比较，而不会被强制要求等于 3D oracle target。

共享 robot-state key 为：

- `ee_pos (3)`、`ee_quat (4, xyzw)`
- 兼容 alias `ee_pos_sim (3)`、`ee_quat_sim (4, xyzw)`；现在必须等于 base-frame EE pose
- `ee_pos_vel (3)`、`ee_ori_vel (3)`
- `gripper_width (1)`
- `joint_positions (7)`、`joint_velocities (7)`、`joint_torques (9)`
- `gripper_finger_1_pos (1)`、`gripper_finger_2_pos (1)`

`camera_info.front_camera` 包含最终的 `image_size=[224,224]`、3x3 intrinsic matrix、`camera_to_sim_local` 及其逆矩阵 `sim_local_to_camera`；在此约定中，`sim_local` 表示 robot base。

FurnitureBench 还可以携带额外 VLM/oracle 标注字段。消费者必须容忍这些扩展字段，不能因为其他来源缺少它们就判断 schema 不一致。

## 相机审查

### FurnitureBench 参考

审查的 3 个真实 pickle 保存了 320x240 front frame，近似参数为：

```text
fx=307.611, fy=308.092, cx=160, cy=120
```

中心裁剪为 224x224 后，`fx` 和 `fy` 不变，`cx=cy=112`，得到的水平 FOV 约为 40.0 度。新 writer 还会根据裁剪偏移同步平移 2D guidance/grasp 标注和主点。旧标定把相机位姿保存在 simulator-local 坐标系中；迁移逻辑和后续写入现在从配对的 simulator-local/base EE pose 推导 robot-base 变换，并以 base frame 保存相机位姿（所审查采集批次为 `[1.2, 0, 0.235]`）。同一恢复出的变换也应用于展开的零件位姿和 3D guidance point/pose。只变换相机是错误的：即使每个数组单独都能通过形状检查，也会让 3D label 留在另一个坐标系。

### ManiSkill

PickCube 配置中已经安装与 FB 对齐的 224x224、约 40 度 front camera。registry 暴露 11 个任务：

`LiftPegUpright`、`PegInsertionSide`、`PickCube`、`PlaceSphere`、`PlugCharger`、`PokeCube`、`PullCube`、`PullCubeTool`、`PushCube`、`StackCube` 和 `StackPyramid`。

被审查 commit 中，其他任务定义仍提供各自的相机位姿、分辨率和 FOV。只修改 sensor 宽高并不能对齐位姿或 FOV。应用的补丁新增可复用的 `rr_aligned_sensor_overrides`，在公共生成器中为 `base_camera` 指定专用位姿、40 度 FOV、near/far 范围和 224 分辨率。同时把 `gripper_width` 和两个 finger position 从 Python scalar 改为 `(1,) float32` 数组。全部 105 个 trajectory 和 skill annotation 测试通过（另有 13 个慢测试未选择）。3 条真实渲染 PickCube rollout 确认最终 front 标定为 `fx=fy=307.717`、`cx=cy=112`，robot base 下的相机平移为 `[1.2, approximately 0, 0.235]`。虽然 schema registry 中包含 PokeCube，生成器目前仍没有对应 motion planner。

### AutoMate

Front pose 以 robot base 为参照，位置为 `pos=(1.2,0,0.235)`，朝向与 FurnitureBench 相似，因此外参意图正确。但 IsaacLab 使用以下公式计算焦距：

```text
fx = width * focal_length / horizontal_aperture
```

当 width 为 640、focal length 为 `1.848554759492`、horizontal aperture 为 `1.92` 时，`fx` 约为 616.18 px。随后 state adapter 不做缩放，直接把 640x480 frame 裁剪为 224x224，因此最终水平 FOV 约为 20.6 度，而 FurnitureBench 约为 40.0 度。

候选补丁采用最直接的精确修正：wrist/front 均使用相同 focal-length/aperture 设置以 320x240 渲染，再中心裁剪到 224。它还把 AutoMate 的干净 `fixed_pos_obs_frame`（fixed asset tip/insertion frame）转换为 robot-base `guidance_point`，把阶段标为 `insert`，并使用保存的标定把该点投影到 front image。严格 validator 会检查保存像素能否复现该投影。直接数值回归测试把最终 FOV 锁定在约 40 度，全部 15 个纯数据采集测试通过。独立 AST/数值审查又把实际配置的位置和 OpenGL quaternion 与一份真实标准 FB 标定比较：4x4 base-frame camera matrix 的误差不超过 `3e-7`；AutoMate 配置焦距为 `308.092460` px，FB 为 `fx/fy=307.611053/308.092438`，最终水平 FOV 为 `39.9552` 度。3 条真实渲染的 assembly-`00211` 成功轨迹进一步确认了运行时保存的标定。每个 observation 的 target 都投影在图像内，像素范围为 `x=98.05..113.95`、`y=24.48..47.85`，严格 3D 到 2D 重投影验证通过。contact sheet 也显示相同的 robot-base-front 观察方向；AutoMate 场景本身更开放，小型插入目标会被机器人部分遮挡。

## 干净的 IsaacLab 与 Isaac Sim 环境

没有激活或使用任何已有 Isaac/robot 环境。r218 环境来自全新 source checkout 和新建 Conda 环境，遵循该 fork 的 IsaacLab 2.3.2 pip 安装流程：

```text
source: /data/hy/IsaacLab
commit: 72e9db91b792cf1abdd280485d6ed5c8829489bf
environment: /home/hy/anaconda3/envs/rr-isaaclab
Python: 3.11.15
PyTorch/torchvision: 2.7.0+cu128 / 0.22.0+cu128
Isaac Sim: 5.1.0.0
IsaacLab/IsaacLab-RL: 0.54.4 / 0.5.2 (editable source installs)
rl_games: 1.6.1, official python3.11 branch head 6b3534f29568158e9e29ec8bf83cc88fce5f0cae
rl_games source archive SHA-256: b25edd13ddcd6b6a7e0b0258589d2401e9c2bbe8dbde526c07bef17e6efaa350
```

在 `rl_games` 步骤中，GitHub 多次中止官方脚本的 filtered Git clone。随后下载等价的官方分支 archive，通过 ZIP 完整性检查后安装到同一新环境。该分支未受约束的传递依赖原本会选择比 Isaac Sim 精确锁定版本更新的包。最终解析结果保持在所有声明范围内，同时保留 NVIDIA 的版本锁：Ray 2.45.0、IPython 9.8.0、ONNX 1.18.0、W&B 0.22.0、packaging 23.0、psutil 5.9.8、typing-extensions 4.12.2、click 8.1.7。`pip check` 除上游已知的 FastAPI 0.115.7 与 IsaacLab 固定 Starlette 0.49.1 冲突警告外均干净。Ray 在 `ray/thirdparty_files` 下单独 vendor 了 psutil 7.0.0；在导入 Ray 前由环境解析的仍是要求的 psutil 5.9.8。

该干净环境通过 r218 RTX 3060 上的真实 CUDA tensor 运算、全部 15 个纯 AutoMate 数据采集测试、Isaac Sim 5.1 RTX 启动以及 3 条真实 rollout 采集。设置 `PYTHONNOUSERSITE=1` 的路径审查确认 Python、Torch、NumPy、IsaacLab 和 `rl_games` 均解析到 `/home/hy/anaconda3/envs/rr-isaaclab`，没有来自其他环境或 user site。用户已明确接受 NVIDIA Omniverse EULA；通过每个进程的 `OMNI_KIT_ACCEPT_EULA=YES` 传入，未做全局持久化。

另在 `zju_4090_232` 准备了一套独立的干净 fallback。它只使用 NAS Conda，代码和 rollout staging 位于服务器本地数据盘：

```text
environment: /mnt/nas/share/home/hy/miniconda3/envs/rr-isaaclab-2.3.2
source: /data/hy/rr-cross-sim-20260821/IsaacLab
commit: 72e9db91b792cf1abdd280485d6ed5c8829489bf
Python: 3.11.15
PyTorch: 2.7.0+cu128
Isaac Sim: 5.1.0.0
IsaacLab/IsaacLab-RL: 0.54.4 / 0.5.2 (editable source installs)
rl_games/Ray: 1.6.1 / 2.45.0
NumPy/packaging: 1.26.0 / 23.0
pip check: only the same Isaac Sim FastAPI / IsaacLab Starlette metadata conflict
CUDA smoke: RTX 4090, pass
AutoMate pure data-collection tests: 15 passed
```

服务器命令强制设置 `PYTHONNOUSERSITE=1`、`PIP_USER=0` 和 `--no-user`，因为 pip 的可写性探测会错误地把实际可写的 NAS 环境当作只读。在识别出该行为之前，首次 Torch 安装尝试曾把包写入服务器本地 `~/.local`。所有报告中的检查均未激活或使用这些 user-site 包；考虑到该位置可能包含用户此前无关的状态，没有删除它们。后续安装和全部服务器验证都强制隔离 user site，并确认 Torch 解析到 NAS 环境。6 个 IsaacLab source package 和官方 `rl_games` 分支 archive 已安装到这个新的 NAS 环境。本地环境和服务器环境从未互相复用。

AutoMate policy 输入也已固定，并独立于运行时完成验证。官方 NVlabs SRSA specialist bundle 通过其发布的 LFS size/hash 和 archive 检查：

```text
bundle size: 52,214,269 bytes
bundle SHA-256: d942604dde688c7b8ba29a67a429969477ce96fc5f54be4c6a4994a34fe29e66
selected assembly ID: 00211
checkpoint: logs/cross-sim-alignment-20260821/assets/checkpoints/00211.pth
checkpoint size: 11,311,464 bytes
checkpoint SHA-256: c2bc63bffe9175fa5cbd73da311196aceb79aec608ab8accb8c18b0e4b87b5e4
```

仅使用 CPU 的静态加载得到 epoch 1437、frame 5,885,952、mean reward 536.5768、24 维 observation statistics、6 个 policy output、一个 256-128-64 MLP 和两层宽度 256 的 LSTM。这些维度与当前 AutoMate PPO collector 匹配。完全相同的 checkpoint 和 hash 也镜像到了服务器本地 staging 目录；首次下载得到的损坏部分文件已隔离，不会被使用。

## FurnitureBench 真实数据验证

输入数据：

```text
task: one_leg
成功轨迹数: 3
observation 长度: 每条 323
action 长度: 每条 322
对齐前旧 reward 长度: 337, 352, 350
```

审查的 action 是有限 8D 数值，包含单位四元数，最大平移增量约 3 cm，gripper 值位于 `[-1,1]`。其实际数值是 delta action，但这些旧文件错误地记录了 `action_type="pos"`。

审查发现，旧数据的 `ee_pos_sim`、camera、parts 和 3D guidance annotation 位于 simulator-local 坐标系，与 robot-base 值之间相差固定 simulator base transform。面向模型的标准 state 和 geometry 现在全部位于 base frame；后续写入也会让 compatibility alias 和所有保存几何都使用 base frame。

3 个源文件也已迁移为标准原始 pickle。共享 validator 使用 FB 参考内外参检查了每个 observation：

```text
标准原始 pickle: 3
每个 pickle 的 transition: 322
每个 pickle 检查的 observation: 323
front fx: 307.611
robot base 下的 front translation: [1.2, 0.0, 0.235]
严格原始 validator: pass
检查的 oracle 3D-to-front-2D 配对: 968
最大重投影残差: 0.70 px
```

3 个真实 episode 使用 `--image-size 224` 转换为一个 LMDB：

```text
标准 pickle 目录:
  logs/cross-sim-alignment-20260821/rollouts/furniturebench/canonical/
LMDB:
  logs/cross-sim-alignment-20260821/lmdb/furniturebench-one_leg-real.lmdb
episode 数: 3
frame/transition 数: 966
env 计数: FurnitureBench=3
保存的 RGB/depth 形状: 224x224
完整 LMDB validator: pass
LMDB 磁盘大小: 650 MiB
```

LMDB 中保存的非零有限 depth moment 为：

| 相机 | Count | Mean (m) | Std (m) |
|---|---:|---:|---:|
| wrist | 48,470,016 | 0.09923281 | 0.04528964 |
| front | 48,470,016 | 0.99165719 | 0.47229034 |

这些是验证子集统计，并非建议使用的最终训练统计。

## ManiSkill 真实数据验证

使用干净 source checkout 和新建 Conda 环境；没有任何已有 ManiSkill/robot 环境提供包：

```text
source: /data/hy/ManiSkill
commit: d1ead54fdb8b5b22ea831fadb9a571fb3c52a22f
environment: /home/hy/anaconda3/envs/rr-maniskill
Python: 3.11.15
PyTorch: 2.13.0+cu130
SAPIEN: 3.0.3
NumPy/OpenCV: 1.26.4 / 4.11.0
```

README 中未设上界的依赖最初解析到 NumPy 2.4.6 和 OpenCV 5.0.0.93。SAPIEN 渲染可以工作，但 fork 固定的 `mplib==0.1.1` 在 `ArticulatedModel` 内部 segfault。把同一新环境中的 NumPy/OpenCV 固定到兼容的 1.26/4.x 版本线后消除了二进制 ABI 故障，单轨迹探测和正式运行随后均成功完成。这些兼容范围也写入 ManiSkill source patch，避免未来干净安装时静默重现不兼容解析。

正式运行采用 `PickCube-v1`、CPU PhysX、GPU Vulkan RGB-D 渲染、seed 0--2 及项目附带的 Panda motion planner。任务在 tmux session `rr_ms_pickcube_3_v2` 中运行并以 exit code 0 结束。3 个文件包含：

```text
transition 数: 74, 74, 50
observation 数: 75, 75, 51
success: 3 条均为 true
action_type/env: delta / ManiSkill
action 宽度: 8
观测到的 gripper 值: {-1, +1}
最大绝对 xyz delta: 0.0399 m
RGB/depth: wrist 和 front，224x224
depth dtype/单位: float32 米
front fx/fy/cx/cy: 307.717 / 307.717 / 112 / 112
parts_poses 宽度: 14（cube 加 goal-site pose）
观测到的 skill: null, pick, place
```

3 条轨迹中分别有 70/75、71/75 和 46/51 个 observation 带有 guidance point/pose/width；null frame 是没有定义 active point 的阶段边界。原生写入会在 atomic rename 前验证整条 trajectory。随后 RR 的独立共享 validator 检查 3 个文件的每个 observation，包括 action/quaternion 语义、base-frame geometry、image/depth 约定、camera inverse 和 3D-to-2D guidance 重投影。全部通过，front focal length 与 FB 参考值的差异小于 0.04%。

产物和完整运行日志位于：

```text
logs/cross-sim-alignment-20260821/rollouts/maniskill/
logs/cross-sim-alignment-20260821/runtime/maniskill_pickcube3_retry.log
logs/cross-sim-alignment-20260821/visual/fb_ms_real_contact_sheet.jpg
```

Contact sheet 展示一条真实 FB episode 和一条真实 ManiSkill episode 的首帧、中间帧、末帧 wrist/front RGB。两个 front stream 都从相同 robot-base-front 方向观察并保持操作工作区可见；场景外观和遮挡自然不同，不视为几何未对齐。

## AutoMate 真实数据验证

最终运行使用官方 NVlabs SRSA assembly-`00211` specialist、随机 policy sampling、单环境以及本地干净 IsaacLab 环境。启动方式遵循文档流程（先 `conda activate`，再 `./isaaclab.sh -p`），并按进程传入 EULA 接受。5 次尝试得到 3 次成功；失败的第 2、4 次被丢弃且从未序列化：

```text
tmux session: rr_local_automate_00211_formal3_cpu_20260821
exit code: 0
成功 attempt ID: 1, 3, 5
transition 数: 10, 9, 36
observation 数: 11, 10, 37
action_type/env: delta / AutoMate
action 宽度: 8
观测到的 gripper 值: {+1}
每个 episode 的最大绝对 xyz delta: 0.0784, 0.0790, 0.0953 m
parts_poses 宽度: 14（held 和 fixed asset）
观测到的 skill: insert
guidance point 覆盖: 58/58 observations
```

最初的 GPU-PhysX camera collector 暴露了两个不同的 reset 问题。每个内部 IK/gripper physics substep 都渲染会使 held object 和 robot 数值爆炸；而在 GPU PhysX 上完全禁止 reset 渲染（或周期渲染）又可能触发 PhysX/RTX illegal-memory access。稳定的最小验证配置使用 CPU PhysX 运行单环境、RTX 3060 负责 RGB-D 渲染，reset helper 遵循正常 `DirectRLEnv.step()` 每 8 个 physics substep 渲染一次的节奏。这个小规模验证也把 SRSA policy 放在同一个 CPU device。这不会改变任务参数或记录约定，policy 也成功闭环，但生产规模 GPU 采集应把底层 PhysX/RTX 交互作为独立工程问题处理。

3 次原生写入都在 atomic rename 前通过验证。RR 严格 validator 独立检查全部 58 个 observation，包括单位 delta quaternion、base-frame state/parts/guidance、224x224 RGB-D、互逆 camera matrix、FB focal/pose/forward 参考，以及 3D-to-front-2D 重投影。保存的标定为：

```text
front fx/fy/cx/cy: 308.092 / 308.092 / 112 / 112
robot base 下的 front translation: [1.2, 0.0, 0.235]
front forward axis: [-0.9834533, approximately 0, -0.1811624]
guidance pixel 范围: x=98.05..113.95, y=24.48..47.85（全部在图像内）
```

Wrist depth 范围较集中（`0.059..0.453 m`）。AutoMate 开放场景使 front camera 背景深度最高达到 `46.765 m`：在计入 LMDB 的 AutoMate pixel 中，22.85% 超过 2 m，17.43% 超过 5 m。这些是有限的 renderer 测量值，符合当前约定，但会显著扩大共享 front normalizer。未来训练数据决策应对比原值保留、明确记录的 far-depth mask 或 clip；验证代码目前不会静默修改它们。

产物：

```text
pickles:
  logs/cross-sim-alignment-20260821/rollouts/automate/00211/formal_srsa_resetfix_20260821/success/
runtime log:
  logs/cross-sim-alignment-20260821/runtime/automate_00211_formal3_cpu_20260821.log
three-source contact sheet:
  logs/cross-sim-alignment-20260821/visual/three_source_real_contact_sheet.jpg
```

## 真实两来源 staging 验证

在 AutoMate EULA 门槛解除之前，已有的 6 条真实 FB/ManiSkill 轨迹被转换为各自独立的持久 LMDB，并合并成一个真实两来源 LMDB：

```text
仅 FB:          3 episodes / 966 transitions / 650 MiB
仅 ManiSkill:   3 episodes / 198 transitions / 134 MiB
两来源:         6 episodes / 1,164 transitions / 783 MiB
两来源 env 计数: FurnitureBench=3, ManiSkill=3
两来源路径:
  logs/cross-sim-alignment-20260821/lmdb/two-source-real.lmdb
完整 LMDB stats validator: pass
可变 parts_poses 宽度: 14 和 42，pass
```

合并后的真实 depth moment 为：

| 相机 | Count | Mean (m) | Std (m) |
|---|---:|---:|---:|
| wrist | 58,404,864 | 0.10798684 | 0.05532296 |
| front | 55,066,815 | 0.95558838 | 0.46112306 |

训练侧 lazy `RGBDDataset` 加载全部 6 个 episode，并从两个来源各选一个样本放入同一 batch。得到的形状为 RGB `(2,2,3,224,224)`、depth `(2,2,1,224,224)`、robot state `(2,2,16)`、skill `(2,2,5)`、action `(2,4,10)`；两个 depth stream 均为有限值。这证明了真实 FB/ManiSkill 数据可以通过合并 dataset 和 lazy image store，结果作为中间审查产物保留。

## 最终真实三来源 LMDB 与 DDP 验证

一个精确输入目录中，每个仿真器各有 3 个 hard-link source pickle，它们被转换为一个持久 LMDB。首次误建的 15-episode 版本暴露了重复的旧输入链接；该版本已隔离，从未作为结果验证，并在精确 9 条重建前删除。没有任何 source pickle 被删除或覆盖。

```text
输入:
  logs/cross-sim-alignment-20260821/merged-input-exact/
LMDB:
  logs/cross-sim-alignment-20260821/lmdb/three-source-real.lmdb
磁盘大小: 820 MiB
episode/transition 数: 9 / 1,219
env 计数: FurnitureBench=3, AutoMate=3, ManiSkill=3
各来源 transition 数: 966 / 55 / 198
RGB/depth 形状: 224x224
parts_poses 宽度: 14 和 42
完整 LMDB validator: pass
```

保存的非零有限 depth moment 与独立重算结果一致：

| 相机 | Count | Mean (m) | Std (m) |
|---|---:|---:|---:|
| wrist | 61,164,544 | 0.11660362 | 0.07163558 |
| front | 57,280,786 | 1.03884094 | 1.35608447 |

Actor 级 checkpoint 审查加载这些 LMDB 统计，把它们分别安装到 wrist/front RGB-D encoder，序列化 model buffer 和可读的 `depth_normalizer_stats` 记录，再恢复一个新 actor；每个 count/mean/std 都精确一致。

最终双 rank CPU/Gloo DDP smoke 使用 lazy LMDB read、100 个全局样本、每个 rank batch size 10，以及精确来源权重。每个 rank 执行 5 个 optimizer step，并完成保存/重载：

```text
checkpoint:
  logs/cross-sim-alignment-20260821/checkpoints/three-source-real-ddp-smoke.pt
全局来源计数: FurnitureBench=50, AutoMate=35, ManiSkill=15
world size / 每个 rank 的 step: 2 / 5
混合 RGB 形状:   (3,2,3,224,224)
混合 depth 形状: (3,2,1,224,224)
混合 state/skill/action: (3,2,16) / (3,2,5) / (3,4,10)
checkpoint depth-stat round trip: pass
```

首次在受限 workspace 内调用 DDP 时无法打开 localhost TCPStore，因而被终止；完全相同的命令在 network sandbox 外立即通过。这是执行权限差异，不是 sampler 或 distributed training 故障。

## 混合来源约定 smoke test

3 个完成几何修正的真实 FurnitureBench pickle，加上 1 个 schema-valid 合成 ManiSkill pickle 和 1 个 schema-valid 合成 AutoMate pickle，被转换到同一 224x224 LMDB。AutoMate fixture 包含新的几何导出 `insert` point/pixel；ManiSkill fixture 由 NumPy 2 写入。5 个 front focal length 与 FB 参考值的差异均在 1% 以内。验证和 lazy dataset 加载结果如下：

```text
episode/transition 数: 5 / 968
env 计数: FurnitureBench=3, ManiSkill=1, AutoMate=1
batched RGB:   (3,1,3,224,224)
batched depth: (3,1,1,224,224)
RR quaternion-to-6D 预处理后的 batched action 宽度: 10
接受可变 parts_poses 宽度: 14 和 42
完整 LMDB depth/stat validator: pass
AutoMate insert skill 经 LMDB 预处理后仍保留: pass
AutoMate 彩色 guidance point 在标定像素 [112,112] 渲染: pass
```

ManiSkill fixture 由 NumPy 2 序列化，再由 RR 的 NumPy 1 环境加载。这专门覆盖了否则会导致致命错误的 `numpy._core` 与 `numpy.core` pickle namespace 变化。

后处理 action 宽度有意设为 10，因为 RR 会把原始 4D quaternion rotation 转换为模型使用的 6D rotation representation。原始 pickle 约定仍严格为 8D。

来源加权 sampler 在 1,000 次 draw 中得到精确 quota：

```text
FurnitureBench=500, AutoMate=350, ManiSkill=150
```

现在单 GPU `bc.py` 与 episode-sharded 多 rank `bc_ddp.py` 使用完全相同的精确 source-quota 语义。固定 10 step、batch size 10 的单 GPU DataLoader smoke 精确得到 50 个 FB、35 个 AutoMate 和 15 个 ManiSkill 样本。DDP 根据各 rank 本地来源可用性分配全局 quota；分层 train/validation split 会保证两个 split 都包含每个正权重来源，否则明确失败。来源加权不能与旧的 `minority_class_power` 重采样同时启用。

重启前的合成混合来源 fixture 有意保存在 `/tmp`，重启后已被清除。上文记录了其验证结果，但不会复用。3 条真实 FB 轨迹现已重新标准化到持久任务日志，FB-only LMDB 也已重建并完整重扫。当前持久路径为：

```text
几何修正后的真实 FB pickle:
  logs/cross-sim-alignment-20260821/rollouts/furniturebench/canonical/
几何修正后的真实 FB-only LMDB:
  logs/cross-sim-alignment-20260821/lmdb/furniturebench-one_leg-real.lmdb
```

已清除的纯合成路径为：

```text
5 episode 混合 smoke LMDB:
  /tmp/rr-cross-env-review-20260821/three-source-contract-v3.lmdb
AutoMate point-rendering smoke LMDB:
  /tmp/rr-cross-env-review-20260821/automate-guidance-point-v3.lmdb
```

混合 smoke LMDB 的 depth moment 为：wrist `count=48,570,368, mean=0.10047407, std=0.05456215`，front `count=48,570,368, mean=0.99146782, std=0.47184243`。它们包含合成外部 frame，因此只作为验证证据，不作为训练 normalizer 数值。

## Depth 归一化与 checkpoint

训练路径满足要求的设计：

1. Pickle-to-LMDB 对所有保存的有限、非零 depth pixel 计算 streaming moment。
2. Wrist（`depth_image1`）和 front（`depth_image2`）分别累计。
3. 多 LMDB 训练精确合并 moment；无法复用整个 shard metadata 时，会扫描选定子集。
4. Dataset statistics 初始化两个 RGB-D encoder。
5. Count/mean/std 是持久 model buffer；checkpoint 还保存可读的顶层 `depth_normalizer_stats` 记录。
6. 推理通过 `Actor.load_state_dict` 恢复 buffer；只缺少部分相机统计会被拒绝。仅对旧 FurnitureBench checkpoint 提供有文档说明的 legacy fallback。

持久两来源 staging LMDB 和最终真实 3/3/3 LMDB 均完成端到端验证。对最终 dataset，精确 wrist/front moment 被加载到双相机 RGB-D actor，使用 `torch.save` 序列化，再恢复到新 actor 并逐字段比较。独立双 rank DDP smoke 把相同统计保存为可读 checkpoint metadata，也能精确恢复。两个结果都不使用合成 smoke statistics。

## robust-rearrangement 代码修改

- 新增纯函数标准 pickle helper module，负责同步 RGB-D 裁剪、正值米制 depth 转换、2D annotation/intrinsic 调整、camera/parts/guidance geometry 从 simulator-local 到 robot-base 的转换、base-frame state alias，以及严格 `T+1/T` action/reward 验证。
- 后续 FurnitureBench pickle 写入改为 224x224 和 `action_type=delta`。
- Pickle-to-Zarr 与 pickle-to-LMDB 处理新增 `--image-size 224`，用于确定性转换旧数据。
- Actor 和 front-camera augmentation 可同时消费标准 224x224 与旧 240x320 输入，不再对标准 frame 做 padding。
- 修正 quaternion magnitude clipping，使其按 timestep 计算。旧的 batch-wide norm 会让 rotation action 随 trajectory 变长而缩小。
- 公共数据读取路径新增兼容 NumPy 1/2 的 pickle loader。
- 新增稳定的严格 raw-pickle validator 核心模块，支持可选 FB focal、camera position、forward axis 检查，以及经过标定的 oracle 3D-to-2D guidance 重投影检查。
- 新增显式旧 FB migration 核心模块；除非调用方确认对应采集批次实际已保存 delta，否则拒绝把 `action_type='pos'` 重新解释为 delta。一次性 CLI 包装脚本经审查后未保留。
- 新增聚焦 image/state/migration/validator 的测试。
- 移除 LMDB converter 和 lazy dataset loader 对 visualization、FurnitureBench、Zarr、Torchvision 和 debugger module 的 import-time 依赖。Zarr 与 image resizing 仍是显式可选路径，调用时缺少依赖会给出聚焦错误；在新仿真环境中，224 LMDB 路径现在只需额外安装项目固定的 `lmdb` 包即可运行。
- 单 GPU `bc.py` 和 episode-sharded `bc_ddp.py` 都新增精确 env-weighted sampling，包括确定性 epoch shuffle、分层 split、小型辅助来源 replacement，以及同时启用 `minority_class_power` 重采样时的冲突拒绝。

修改后的验证结果：

```text
pickle image/action/depth 约定: 8 passed
pickle state/action 约定:     5 passed
raw pickle writer/validator/migration: 10 passed
depth-statistics 路径:        7 passed
checkpoint RGB-D normalizer:  4 passed
source-weighted sampling:     8 passed
LMDB validator helper:        3 passed
总计:                        45 passed
compileall:                  pass
git diff --check:            pass
```

ManiSkill trajectory 与 skill-annotation 测试结果为 `105 passed, 13 deselected`；AutoMate 纯数据采集测试结果为 `15 passed`。

尚未 push 到两个 GitHub 仓库的外部补丁保存在 `logs/cross-sim-alignment-20260821/patches/`。它们对 fresh clone 通过 `git apply --check --whitespace=error`，并已应用到 `/data/hy/ManiSkill` 和 `/data/hy/IsaacLab` 的本地任务分支；上文测试与 rollout 结果来自这些 working-tree 修改。

```text
maniskill_alignment.patch
maniskill_camera_contract_new_file.patch
maniskill_environment_compat.patch
isaaclab_alignment.patch
isaaclab_collection_skip_dense_reward.patch
isaaclab_collection_runtime_alignment.patch
```

## 最终运行状态与适用范围内的注意点

- 当前执行主机是 `r218`（OS hostname `lht-3060-12G`，用户 `hy`，Tailscale IP `100.97.142.107`）。主机表只是资产清单，不能据此推断 Codex 运行在 `control-win`。
- 2026-08-21 的重启解决了 NVIDIA 版本不匹配。当前 kernel 为 `6.8.0-136`；已加载 module、`modinfo`、NVML 和 userspace 均报告 `580.173.02`。RTX 3060 上的 `nvidia-smi`、PyTorch CUDA、SAPIEN RGB-D 和 NVIDIA Vulkan 全部通过。
- 干净 ManiSkill 环境和 3 条真实 rollout 已完成。初始故障被独立定位为 `mplib 0.1.1`/NumPy 2 ABI，不是已修复的 GPU driver。后续工作未使用已有 robot 环境。
- 本地干净 Isaac Lab 环境按照 fork 自身的 2.3.2 文档安装：Python 3.11、PyTorch 2.7.0/cu128、Isaac Sim 5.1.0、editable source package，以及官方 `rl_games` python3.11 分支。CUDA 和全部纯 AutoMate 数据测试通过。Assembly `00211` 及其官方 specialist checkpoint 均已固定并完成 hash 验证。EULA 接受、真实渲染运行、3 次成功、严格验证及最终 LMDB/DDP 集成均已完成。
- 可用 4090 集群可承担 GPU 密集任务，但它与本机是独立环境。服务器环境必须位于 NAS Conda base `/mnt/nas/share/home/hy/miniconda3` 下，rollout 数据应保存在所选服务器本地 SSD/NVMe。审查期间，`zju_4090_232` 的 GPU 1/2 已被现有 RR 训练占用，未进行干扰。该节点的干净 NAS fallback 环境现已包含 editable IsaacLab source package、Isaac Sim 5.1、官方 `rl_games` 和精确兼容 pin。其 package-path 审查、RTX 4090 CUDA smoke 和 15 个纯 AutoMate 测试均在不使用 user-site package 的情况下通过。`pip check` 只报告上文所述不可避免的官方 FastAPI 0.115.7 / Starlette 0.49.1 metadata 冲突。最终 runtime collection 在 r218 完成：232 的可用 GPU 后来与其他图形进程共享；243 虽然 GPU 空闲，但 Kit 启动期间反复收到外部 SIGTERM（exit 143）。仅停止了属于本任务的精确 tmux session，没有触碰无关进程。

## 完成验收

在“每个仿真器选择一个任务”的范围内，最小数据里程碑已经完成：

1. 对所有选定 ManiSkill 任务应用共享 camera override：通过。
2. 应用 AutoMate camera/guidance 候选方案，并在真实渲染 assembly-`00211` 轨迹中验证几何导出的 fixed-asset-tip target：通过。
3. 每个仿真器各从一个任务采集 3 条成功 rollout：通过。
4. 运行各仿真器严格 raw-pickle validator，并进行数值和可视化检查：通过。
5. 构建真实合并 LMDB，`env` 计数为 `3/3/3`、所有 frame 为 224x224，且完整 validator 确认重算与保存的 depth moment 相等：通过。
6. 加载包含全部三来源的 batch，以 `0.50/0.35/0.15` 运行短程 DDP smoke，保存并重载 checkpoint，精确验证两个相机的 depth statistics：通过。

把同样验证扩展到更多 ManiSkill 任务和 AutoMate assembly 是未来的广度证据；已完成的最小验收并不隐含这些更大范围已经得到验证。

## 论文叙事与实验设计

清晰的中心论点不是“我们新增了两个仿真器”，而是：

> Guidance point 是一种与任务和仿真器无关的空间接口，把“与什么、在哪里交互”和“如何执行局部行为”分离开来；多样化交互数据可以提升它的 grounding 和 tracking，同时不会在原有长时序家具任务上造成负迁移。

把 3 个 FurnitureBench 任务作为受控、接触丰富的主 benchmark，把 ManiSkill/AutoMate 作为广度和规模扩展。代码目前暴露 11 个 ManiSkill 任务和原始 100 个 AutoMate assembly ID；用户已明确从生产集合排除 AutoMate `00755`，所以当前正式范围是 3 + 11 + 99 = 113 个任务。原始 114 只表示代码可暴露的潜在范围，不是本轮数据配额。

建议结果组织方式：

1. **核心方法：** 在相同的 3 个 FurnitureBench 任务、seed、demonstration 和 compute 下做受控比较。
2. **规模迁移：** 比较 FB-only、FB+ManiSkill、FB+AutoMate 和全来源训练；始终在相同 held-out FB 条件上评估。
3. **任务数量曲线：** 在固定 optimization step 和 FB 数据 exposure 的前提下，加入约 3/10/30/100 个辅助任务，从而区分多样性收益和只是看到更多 FB sample 的收益。
4. **广度：** 评估支持的外部 task family，同时报告 aggregate 和 family-level success，避免简单插入任务掩盖负迁移。
5. **接口消融：** 比较具有有效 3D/2D guidance supervision 的辅助数据与只有 RGB-D/action 的辅助数据。AutoMate 现在包含几何导出的 insertion point，生产共享相机也已更新为用户批准的 v2；但在正式 99 个 assembly ID 逐一完成真实 rollout 的数值 gate 前，不能把抽查描述成 point interface 的 99-task 验证。
6. **数据混合：** 至少比较 uniform sampling 和建议的 `50% FB / 35% AutoMate / 15% ManiSkill`。除非证据足以把 source-aware curation 声明为贡献，否则应把该配比视为工程选择。

应预先根据任务属性（例如 horizon、skill transition 和 contact structure）说明为什么这 3 个 FB 任务构成核心集合，而不是根据事后成功率解释。外部 suite 用于回答任务广度质疑，但不应被包装成与长时序 FB benchmark 等价。
