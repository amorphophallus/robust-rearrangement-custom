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

这次不改变 RR legacy action preprocessing；xyz clip 与 episode rotation scaling
仍按 `reports/rr_legacy_action_preprocessing.md` 暂时保留。由于一个源 episode 会被
切成多个 pickle，每个输出段都保存从“未切分源 episode”计算的
`legacy_rotation_episode_scale`；LMDB 处理器复用该 scale，避免分段意外改变这个
legacy bug 的数值效果。

## 新数据采集

配套 Deoxys recorder 已改成：

- 固定 `--record-fps`（默认 10 Hz），不再按 action norm 丢掉 no-op；
- 固定网格递增，慢循环会明确跳过错过的 grid，不用 `now + period` 累积漂移；
- 分别保存相机 source/receive、PromptDA submit/start/ready、robot/gripper
  source/receive、robot/gripper command send time；
- 同时保存原始 delta action 与由当时状态算出的 absolute target，便于闭环校验；
- pickle schema 为 `deoxys_furniturebench_raw_v4_timestamped`。

新数据仍建议经过同一个 `align_pickles`，它会优先使用
`action_timestamps_ns`，从而使旧数据和新数据的最终 step 定义完全一致。

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
