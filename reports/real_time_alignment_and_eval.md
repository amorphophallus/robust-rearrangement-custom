# 真机数据时间对齐与 Deoxys policy eval

状态：第一版已实现；默认都不会改旧数据或驱动机器人。

## 每个训练 step 的定义

统一以动作发送时间 `t_k` 为主时钟：

- 旧 v3 pickle 没有动作发送时间，使用该 action 对应 observation 的
  `control_wall_time_ns` 作为明确标注的 legacy proxy。
- 新 v4 pickle 使用 `action_timestamps_ns[k]`，即 Deoxys arm command 的本机
  wall-time。
- robot state 使用 action 前紧邻采样的本体状态；新数据同时保存 NUC source
  time、PC receive time，后续可以通过标定延迟插值。
- front/wrist 图像分别用 RealSense sensor/source timestamp 匹配 `t_k`，不是
  Python 完成取帧或 PromptDA 完成时间。
- PromptDA submit/start/ready time 单独保存。PromptDA 延迟是逐帧动态量，不用
  300 ms 常数替代。

旧数据转换采用单调、一对一的相机帧匹配，最大残差 75 ms；两个相机都匹配成功
后，按缺帧或 action gap 大于 150 ms 切段，只保留至少 8 步的连续段。源 pickle
不修改。实际 40 条数据 dry-run 结果为：

- 输入 30,360 actions；
- 保留 25,185 actions（82.95%）；
- 生成 1,013 个连续段。

此前 83.12% 是用 Python camera wall-time 近似得到的。当前少 50 步是因为改用
更准确的 RealSense exposure/source time。

## 旧数据转换

先 dry-run：

```bash
python -m src.real.align_pickles \
  --input-dir data/raw/osc/real/one_leg/teleop/low/success/annotated \
  --output-dir data/aligned/osc/real/one_leg/teleop/low/success \
  --skip-annotations
```

确认保留率后写新 pickle；写入时默认重跑 skill/guidance：

```bash
python -m src.real.align_pickles \
  --input-dir data/raw/osc/real/one_leg/teleop/low/success/annotated \
  --output-dir data/aligned/osc/real/one_leg/teleop/low/success \
  --write
```

之后显式把 aligned 目录交给现有 LMDB 处理器：

```bash
python -m src.data_processing.process_pickles_to_lmdb \
  --controller osc --domain real --task one_leg --source teleop \
  --randomness low --demo-outcome success \
  --input-dir data/aligned/osc/real/one_leg/teleop/low/success \
  --output-dir data/processed/osc/real/one_leg/teleop/low/success/aligned.lmdb \
  --frame-compression none
```

本机当前环境没有 `zstandard`，已实际生成上述未压缩 LMDB：1,013 episodes、
25,185 timesteps。安装并验证 `zstandard` 后可以另建 zstd shard 以节省磁盘；不要
原地覆盖本次产物。

### Real40 全时间轴恢复（2026-09-02）

旧 recorder 在同一个物理 rollout 中按相机/PromptDA 可用性切段，因而不能把切段后的
pickle 当成独立 episode。也不能无条件把旧 aligned 段直接首尾相接：被删时间段内机械臂
可能已经移动，这会产生最大 `0.17267 m / 1.6132 rad` 的 absolute-target 边界跳变。

最终方案从每条原始 `deoxys_furniturebench_raw_v2` pickle 的机器时间恢复一条完整 10 Hz
时间轴，而不是拼接 aligned 子段：

- 每个 source action 映射到唯一、严格递增的最近 100 ms grid slot；超过 75 ms residual
  立即拒绝；
- 缺 action 的 grid slot 写 identity/no-op pose action，并保持最近 gripper command；
- 有完整 front+wrist RGB-D 匹配的 slot 才是 `obs_valid=true` 的训练 observation anchor；
- 缺图像的 slot 保留插值 robot state、stateful skill 和 action，但写零图像占位，绝不把
  占位图作为模型输入；
- skill annotator 按所有 source observations 的原顺序运行，因此其 FSM 不会因缺图像帧
  或切段而重置；
- `pred_horizon=32` 的 action sequence 可以跨过缺 observation 的 slot；episode 尾部不足
  32 个 action 时保留样本，但 loss mask 只计算真实存在的 action，padding 不参与 loss；
- colored guidance point 只渲染到 LMDB 的 front RGB 副本；wrist RGB、两路 depth 和源
  pickle 均不修改。

40 条审计总计 `30,360` 个 source actions，恢复为 `32,921` 个 10 Hz slots，其中
`2,561` 个 synthetic no-op、`28,577` 个有效视觉 anchor；量化 residual p95/max 为
`47.52/72.91 ms`。按 `pred_horizon=32` 检查，`30,359/30,360` 个历史 source actions
至少出现在一个有效 observation 的预测窗口中；唯一未覆盖项没有足够邻近的有效视觉
anchor，不伪造 observation 来强行训练。

生成 RGB-D + colored guidance point 的新 LMDB 使用全新输出路径：

```bash
/home/huyue/miniconda3/envs/rr/bin/python \
  -m src.data_processing.process_pickles_to_lmdb \
  --controller osc --domain real --task one_leg --source teleop \
  --randomness low --demo-outcome success \
  --input-dir data/raw/osc/real/one_leg/teleop/low/success/annotated \
  --timeline-mode legacy-real-10hz --timeline-frequency-hz 10 \
  --max-timeline-residual-ms 75 --max-camera-residual-ms 75 \
  --annotation-source scripted \
  --image-annotation-mode guidance-point-colored \
  --frame-compression zstd --frame-compression-level 1 \
  --n-cpus 1 --batch-size 1 \
  --output-dir \
    data/processed/osc/real/one_leg/teleop/low/success/real40-timeline10hz-rgbd-skill-point-colored-zstd.lmdb
```

这是旧 real40 唯一允许使用的显式 legacy salvage。转换时按同帧几何重新运行 scripted
annotator，并在 LMDB metadata 同时保留历史 source provenance 与本次重算 provenance；
不得反向修改或冒充源 pickle。普通 `--timeline-mode pickle` 会要求源 pickle 顶层本来就是
`annotation_source=scripted`，且绝不重采样时间轴。

同训 sim400 也必须从批准的 400 条原始 pickle 重建 colored-GP LMDB，不能继续混用
RGB-D-only condition：

```bash
/home/huyue/miniconda3/envs/rr/bin/python \
  -m src.data_processing.process_pickles_to_lmdb \
  --controller diffik --domain sim --task one_leg --source rollout \
  --randomness med --demo-outcome success \
  --input-dir data/raw/diffik/sim/one_leg/rollout/med/rgbd-only-skill/success \
  --timeline-mode pickle --annotation-source scripted \
  --image-annotation-mode guidance-point-colored \
  --frame-compression zstd --frame-compression-level 1 \
  --output-dir \
    data/processed/diffik/sim/one_leg/rollout/med/success/rgbd-skill-point-colored-zstd.lmdb
```

这次不改变 RR legacy action preprocessing；xyz clip 与 episode rotation scaling
仍按 `reports/rr_legacy_action_preprocessing.md` 暂时保留。由于一个源 episode 会被
切成多个 pickle，每个输出段都保存从“未切分源 episode”计算的
`legacy_rotation_episode_scale`；LMDB 处理器复用该 scale，避免分段意外改变这个
legacy bug 的数值效果。

## 新数据采集

下一版 Deoxys recorder 的正式合同是：

- 固定 `--record-fps`（默认 10 Hz），所有已执行 no-op 都进入动作 buffer；
- 相机 30 Hz 原始 RGB-D、robot state、gripper state 和 action command 分别完整缓存；
- 控制期不运行 PromptDA，不用最新完成的慢结果替换某个 observation；
- 按 `e` 后才以 action target grid 排序，相机做单调一对一匹配，robot pose/joint 做
  线性 + SLERP 插值，gripper 做最近邻；
- 最终对齐状态重新计算 delta 和 absolute target，再逐帧运行 PromptDA 和
  `mode=offline` scripted/geometry annotation；
- 任一 action 缺格、过期、相机 buffer 溢出、时间 residual/skew 超阈值或几何标注
  失败，整条 episode 拒绝保存；
- 在 episode finalize/生成 pickle 时形成权威 10 Hz 时序；pickle schema 目标为
  `deoxys_furniturebench_raw_v6_offline_buffered`，新数据直接是 `N observation / N action`，
  无需再由 `align_pickles` 或 pickle-to-LMDB 二次切段/重采样。

PromptDA 只替换最终 observation 的 canonical `depth_image1/2`，原始 RealSense depth
保留为 `depth_image1/2_realsense`，RGB 从不写 marker。colored guidance point 仍只在
pickle-to-LMDB 阶段渲染。保存 metadata 必须包含完整 alignment residual 统计，并把顶层
`annotation_source` 固定为 `scripted`、实现记录为 `real_skill_annotation_util`。在该
recorder 合同完成实现和真实数据 audit 前，不得把新采集数据加入 production campaign。

## Deoxys policy eval

入口为 `python -m src.real.evaluate_policy`，与旧 Polymetis `minimal.py` 解耦。
当前第一版只接受 `control.control_mode=pos` 的 RGBD checkpoint，在线运行双相机
PromptDA，并使用 Deoxys `OSC_POSE` 的原生 absolute goal：

- position 直接发送；
- RR rotation-6D 转为绝对 rotation matrix，再转 axis-angle；
- controller `is_delta=false`，translation/rotation scale 都是 1；
- `LINEAR_POSE` 默认用 `time_fraction=2.0` 覆盖一个 10 Hz step；
- action chunk 默认每 4 步重新 query；旧 chunk 的未来槽位由新 chunk 覆盖；
- 已过期 prefix 丢弃，整块过期或 future coverage 不足时进入 hold，不复用旧动作；
- robot/gripper 使用独立 command latency，gripper 只在 sign 改变时发命令。

所有 action 在运行时检查 finite、6D rotation、显式 workspace、最低 EE 高度、单步
平移/旋转以及速度。异常 action 被拒绝并记录，绝不静默 clip。

dry-run 仍会连接相机和机器人并完成推理，但不会发送动作：

```bash
python -m src.real.evaluate_policy \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --max-steps 100
```

真机执行必须显式提供 `--execute`、延迟 profile 和 workspace：

```bash
python -m src.real.evaluate_policy \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --latency-profile /absolute/path/to/latency.json \
  --workspace-min 0.30 -0.35 0.03 \
  --workspace-max 0.75 0.35 0.60 \
  --min-ee-z 0.04 \
  --execute
```

profile 必须满足 `src/real/latency_profile.schema.json`。其中 camera observation
latency 可由 sensor-to-receive 直接统计；robot/gripper observation latency 及两种
action latency 必须通过独立的轻量往返/轨迹实验标定，不能从旧 40 条 pickle
可靠反推。每次换相机模式、网络路径、控制频率或 gripper 配置都应重测。

## 与 UMI 保持一致及有意不同之处

借鉴：source-time anchor、状态插值（xyz/q/dq 线性，orientation SLERP）、动作目标
时间、stale-prefix 删除、future overwrite、独立 observation/action latency。

有意更严格：300 ms 只是 observation-age watchdog；不会给仿真数据整体加 300 ms；
不会静默重复缺失相机帧；不会在整块 action 过期时执行 chunk 最后一个 action；
不会在安全边界外 clip policy label/output。
