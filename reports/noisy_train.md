# FurnitureBench noisy train 数据登记

## Campaign

- ID：`rr_furniturebench_noisy_train_scripted_100per_task_v1`
- 状态：`本地 rollout + 3-LMDB postprocess + 全量矩阵审计完成；3 份 LMDB 已发布 NAS 并通过全量 verification；raw 未删除`
- 任务：`one_leg`、`round_table`、`lamp`，每任务 100 条通过严格审计的成功轨迹
- 标注来源：`scripted`
- raw 图像：未绘制标记，`image_annotation_mode=none`
- 本地工作目录：`logs/noisy-train-production-20260921/`
- 目标 NAS：`/mnt/nas/datasets_tmp/rr_furniturebench_noisy_train_scripted_100per_task_v1/`

## 固定噪声合同

- 每个 scripted FSM phase 采样一个逐轴裁剪到 `[-2,2]` 的标准正态向量，并在 phase 内固定。
- n2：`p_gt + 0.006*z`；n4：`p_gt + 0.024*z`，二者共享 `z`。
- base noise seed 为 0；任务 offset 分别为 `one_leg=0`、`round_table=100000`、`lamp=200000`。
- noisy 2D point 由同帧 noisy 3D point 和保存的 front-camera calibration 投影。
- 若投影落在图像外，3D 点与噪声不变；2D marker 固定为图像域内距离该投影最近的像素。
- grasp-part 的 pick/place 标注只按同一个 3D 位移平移 GT pose 中心，保留 GT 朝向和 gripper width。

## Smoke gate

- Raw：1 条真实 one_leg 成功轨迹、299 transitions；clean/n2/n4 全帧完整。
- 最大 n4 比例误差：`5.960464477539063e-08 m`。
- 最大重投影误差：`0 px`。
- 2×4 LMDB matrix：8/8 PASS；episode/frame index 与 guidance/camera lowdim 一致，非 front-RGB 像素一致，8 路 front-RGB 内容均不同。
- 回执：`logs/noisy-train-production-20260921/smoke-raw-audit.json`、`logs/noisy-train-production-20260921/smoke-matrix-audit.json`。

## 正式执行记录

- rollout tmux：`rr_fb_noisy_train_rollout_v1`
- postprocess tmux：`rr_fb_noisy_train_postprocess_v2`（`2026-09-22T16:39:38+08:00` 正常完成）
- simulator base seed：`22921000`
- 环境：`/home/hy/anaconda3/envs/rr/bin/python`
- 正式 rollout 于 `2026-09-21T21:42:01+08:00` 启动。
- raw audit：PASS；300 episodes，132,257 frames。
- LMDB matrix audit：PASS；`max_n4_ratio_error_m=1.1920928955078125e-07`，`max_reprojection_error_px=1.0`。
- 安全 ready 标记：`logs/noisy-train-production-20260921/publish/POSTPROCESS_READY_FOR_APPROVED_PUBLISH`。
- NAS `receipts/nas-verification.json` 已通过：`all_checks_pass=true`、9 files、121,159,818,323 bytes；3 份 `data.mdb` 与本地审计 SHA 完全一致。未执行 raw pickle 清理，`.all-complete` 也未写入。

## 最终产物

本地审计完成的 3 份 LMDB：

| Noise | Condition | Episodes / frames | `data.mdb` bytes | SHA-256 |
|---|---|---:|---:|---|
| clean | `rgbd-skill` | 300 / 132,257 | 40,378,638,336 | `a892681b4733b6daf80185218c110feb0767fc486d2ad5e27b1a4d7a73538205` |
| n2 | `rgbd-colored-gp` | 300 / 132,257 | 40,385,064,960 | `301b80357022468300ce44bdf3abe5657fd86a0516183d3e422710fe3449ab12` |
| n4 | `rgbd-colored-gp` | 300 / 132,257 | 40,385,576,960 | `c1ab9bc606ea5e8c58f658dfc280dbfa10de443f6c4dfba12038f17379e3205e` |

NAS 根目录：`/mnt/nas/datasets_tmp/rr_furniturebench_noisy_train_scripted_100per_task_v1/`。verification receipt 已登记；raw cleanup receipt 与 `.all-complete` 不存在，raw 保留。

## Noisy colored-GP 正式训练矩阵（2026-09-22）

- Condition：`rgbd_colored_gp`；noise level 为 n2、n4，每个 level 训练 seeds `2026092201`、`2026092202`、`2026092203`，共 6 个正式 run。
- 优先级：每个 level 的前两个 seed（共 4 run）为最高优先级；seed `2026092203`（共 2 run）为高优先级接力任务。
- 训练合同沿用当前 Main supplemental colored-GP：`+experiment=rgbd/dit`、ResNet18 RGB-D、global batch 512、2 GPU DDP、3000 epochs × 100 steps、save every 500 epochs、randomness=low、scripted provenance、colored guidance point、无 skill one-hot；每个 run 独立 W&B ID 和 checkpoint 目录。
- 数据：NAS `processed/lmdb/n2/rgbd-colored-gp/furniturebench.lmdb` 与 `n4/...`；必须先复制到训练节点本地 SSD/NVMe并核验 NAS verification receipt、bytes、SHA-256、300 episodes、132,257 frames、`guidance_noise_level` 与 positive-meters depth，不允许直接从 NAS 训练。
- Eval：one_leg、round_table、lamp；每 checkpoint/task 先 12-rollout validation，再 36-rollout formal，seed=0、n_envs=12、randomness=low、physics reward success、scripted annotation verification。Simulator 原始 wrist/front RGB-D 为 240×320，必须同步中心裁剪 `[8:232,48:272]` 到 224×224；内参同步执行 `cx-=48, cy-=8`，RGB、depth、2D point 使用同一裁剪；depth 为 finite positive float32 metres。

### 训练启动登记（2026-09-22）

| Level | Seed | Host / GPU | Formal first step | W&B ID | Local LMDB SHA | 状态 / 初始速度 |
|---|---:|---|---|---|---|---|
| n2 | 2026092201 | 230 / 0,2 | 19:13 | `a614d748` | `301b8035…ab12` | running；约 37–40 s/epoch |
| n2 | 2026092202 | 230 / 3,6 | 19:13 | `ab28ad2b` | `301b8035…ab12` | running；约 46–48 s/epoch |
| n4 | 2026092201 | 236 / 0,7 | 19:02 | `72416022` | `c1ab9bc6…205e` | running；约 45–47 s/epoch |
| n4 | 2026092202 | 236 / 2,3 | 19:02 | `47d84938` | `c1ab9bc6…205e` | running；约 45–47 s/epoch |
| n2 | 2026092203 | first completed n2 lane | queued | — | shares n2 local LMDB | high priority successor |
| n4 | 2026092203 | first completed n4 lane | queued | — | shares n4 local LMDB | high priority successor |

- 四条最高优先级 run 均已通过两 rank、global batch 512 的 one-step smoke，并进入 formal finite-loss optimizer steps；训练输入均为服务器本地 SSD/NVMe，不从 NAS 直接读取。
- 236 首次 formal 初始化因旧的无认证 proxy URL 卡在 W&B；未完成首个 optimizer step。保留 smoke/config/W&B ID 后，改用各机 `bashrc` 的 SSH authenticated `vpn` 通道恢复，同一 W&B ID 已开始上传曲线。
- 被抢占的 predecessor 均为可恢复暂停：RGBExt grasp-part-colored seed1802 为 epoch1539/step154000、seed1803 为 epoch1699/step170000；Joint90 rgbd-skill seed1802 为 epoch1909/step191000、seed1803 为 epoch2399/step240000。四份 `actor_chkpt_last.pt` 均直接加载并验证 optimizer/scheduler state 与 SHA 后才放行 noisy run。
- 按初始实测吞吐，前四条预计在 2026-09-24 02:30–10:30 间陆续完成；两条 seed2026092203 随各 level 首个完成 lane 接力，全部 6 条初步预计在 2026-09-25 10:00–22:00 完成。该 ETA 将以更长窗口和 checkpoint 实测继续收敛。

### Eval lineage gate

- 本数据的训练 LMDB 已存储 224×224，但其 lineage 是原始 wrist/front 240×320 经过共同的中心裁剪 `[8:232,48:272]`，无 resize/interpolation；因此 simulator eval 必须对两相机执行同一 center-crop-224，不能依据 checkpoint 缺省字段改用 legacy resize。
- 相机内参必须同步 `cx-=48, cy-=8`；RGB、positive-float32-metre depth 与 2D guidance point 使用同一裁剪。condition flags 从 checkpoint resolved config 解析，必须为 `guidance-point-colored`、scripted provenance、无 skill one-hot。
- 每个 checkpoint/task 先 12-rollout validation，合同全过后再 36-rollout formal；固定 `n_envs=12`、seed0、randomness low、physics_reward success。总量为 validation 216 + formal 648 rollouts。
