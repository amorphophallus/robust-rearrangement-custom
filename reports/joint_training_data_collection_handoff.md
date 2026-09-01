# 联合训练数据采集执行交接

更新时间：2026-09-01（Asia/Shanghai）

这份文档是新 Codex session 启动正式数据采集时的唯一配置入口。实验依据、逐任务诊断和容量推导见 `reports/joint_training_full_experiment_plan_0821.md`；跨仿真 schema 依据见 `reports/cross_sim_pickle_alignment_20260821.md`。若旧日志、旧命令或历史报告与本文冲突，以本文为准。

## 1. 不可变合同

| 项目 | 正式值 |
|---|---|
| Annotation source | 所有来源固定为 `scripted` / geometry GT；禁止 VLM，不再逐次询问 |
| Raw pickle 图像 | `image_annotation_mode=none`；采集时绝不把点或文字烧入 RGB |
| Raw pickle metadata | 保存 skill、3D guidance、front `guidance_point_2d` 和相机标定 |
| 图像标注阶段 | 仅在 pickle→LMDB/检查视频时离线绘制 |
| 正式点样式 | 统一 annotation util：红色 2 px、50% alpha；高对比 review marker 不得进入训练数据 |
| 数据复用 | 本轮全部重新采集；旧 pickle、失败样本和 diagnostic 文件不得进入 production manifest |
| 成功口径 | task success 后仍须通过 schema、source/mode、active 3D/2D、画内和重投影 strict gate |
| FurnitureBench randomness | `low`；task/lane 使用互不重叠的 reset seed |
| ManiSkill randomness | task-native randomized reset；manifest 写 `randomness_semantics=task_native` 和精确 simulator seed |
| AutoMate rollout | `hardest` + deterministic specialist policy；显式传 `--deterministic`，禁止 `--enable-sbc` |
| AutoMate 排除项 | `00755` 永久排除；collector 已 fail-closed，不能用于采集、相机、配额或 manifest |

正式物理数据集为 113 个 task、6,650 条 success、约 526,386 transitions：FurnitureBench 3×200=600，ManiSkill 11×100=1,100，AutoMate 99×50=4,950。ManiSkill 验证阶段的“最多 10 条/task”不是正式 quota；AutoMate 的 speed gate 固定 attempted-rollout 数，不以成功率作为并行速度指标。

训练 source sampling 固定起始配置为 FurnitureBench / AutoMate / ManiSkill = 50% / 35% / 15%。这是训练 sampler 权重，不改变物理采集条数。

## 2. 主机、环境和源码

任何本地或远端动作前先运行 `hostname; whoami; tailscale ip -4; tailscale status`。主机标签只是 inventory；当前仓库通常在 r218（OS hostname `lht-3060-12G`、用户 `hy`），不要从 r218 SSH 回 r218。

| 用途 | r218 | 236（4×4090） |
|---|---|---|
| robust-rearrangement | `/data/hy/robust-rearrangement` | 按正式 launch 固定 source snapshot，并记录 commit/hash |
| FurnitureBench env | `/home/hy/anaconda3/envs/rr` | 现有 NAS `rr`，启动前重新定位并记录，不新建本地盘 env |
| ManiSkill source | `/data/hy/ManiSkill` | `/home/hy/rr_joint_0821_20260822/ManiSkill` |
| ManiSkill env | `/home/hy/anaconda3/envs/rr-maniskill` | `/mnt/nas/share/home/hy/miniconda3/envs/rr-maniskill-gpu-0821-20260822` |
| IsaacLab source | `/data/hy/IsaacLab` | `/home/hy/rr_joint_0821_20260822/IsaacLab` |
| IsaacLab env | `/home/hy/anaconda3/envs/rr-isaaclab` | `/mnt/nas/share/home/hy/miniconda3/envs/rr-isaaclab-gpu-0821-20260822` |

本地与服务器是两套独立环境。服务器只使用 NAS Conda；不要再创建 camera/task 临时环境。Isaac Sim 必须设置 `OMNI_KIT_ACCEPT_EULA=YES`，并设置 `PYTHONNOUSERSITE=1`。

生产源码 pin：[`amorphophallus/ManiSkill`](https://github.com/amorphophallus/ManiSkill) `main` commit `b751bb1`；[`amorphophallus/IsaacLab`](https://github.com/amorphophallus/IsaacLab) `main` commit `3a9018f5`；FurnitureBench 使用主仓库 gitlink 固定的 [`amorphophallus/FurnitureBench-custom`](https://github.com/amorphophallus/FurnitureBench-custom) commit `dc7f435`。本地 ManiSkill/IsaacLab 均以用户 fork 为 `origin`、`hmnkapa` 仓库为只读 `upstream`，后续从 fork 的 `main` 继续。正式 launch 仍须记录完整 40 位 commit 和实际文件 SHA；236 的同步 snapshot 已通过本地/远端逐文件 SHA 对账。

## 3. 相机合同

### FurnitureBench

沿用现有 FurnitureBench front/wrist 配置；正式 collector 只负责 raw RGB-D 和 metadata，不传任何 `*-on-image` 或 `grasp-part-annotate` 参数。

### ManiSkill

唯一配置文件为 `/data/hy/ManiSkill/mani_skill/trajectory/pickle/camera_contract.py`，236 使用同步后的同路径源码。224×224、FOV 40°。默认 front 为 eye `(0.585, 0, 0.235)`、target `(-1.315, 0, -0.115)`；task override 为：

| Task | eye | target |
|---|---|---|
| `PegInsertionSide-v1` | `(1.2144098281860352, 0, 0.350943922996521)` | `(-0.6855901917679821, 0, 0.0009440313183419202)` |
| `PokeCube-v1` | `(0.9980502128601074, 0, 0.3110882043838501)` | `(-0.9019498070939098, 0, -0.03891168729432898)` |
| `PullCubeTool-v1` | 与 `PokeCube-v1` 相同 | 与 `PokeCube-v1` 相同 |

LiftPegUpright 保留默认相机。它的旧画外结论来自 reset/FSM 与 target 漂移 bug，不是相机；修复后 pick/lift 阶段 target 在阶段进入时冻结。

### AutoMate

用户批准的共享 v2 front camera 已写入 `/data/hy/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_env_cfg.py`，并同步到 236：

```text
pos=(1.05, 0.0, 0.315)
rot=(0.5434064844747748, 0.4524482209388897,
     0.45244822093888976, 0.5434064844747747)
convention="opengl"
```

它相对原相机向任务靠近 15 cm、向上 8 cm，orientation 与 intrinsics 不变。本地和 236 配置文件 SHA256 均为 `52b3204b8f07346ca04aa2e3566200271e03228db703d7011f6b6f80b405124f`。无 CLI camera override 的 `00410` 回归保存 44 帧，44/44 active point 在画内：`logs/joint-training-full-0821/automate_front_camera_shared_applied_20260901_v1/00410-shared-v2-applied.mp4`。

## 4. Task、expert 与正式入口

### FurnitureBench：3 个 task

`one_leg round_table lamp` 均使用 gpu-snatcher `auto_data_preparation` 已跑通的 specialist policy 参数，但唯一维护入口为：

```bash
/home/hy/anaconda3/envs/rr/bin/python scripts/data_collection/collect_furniturebench.py \
  --tasks one_leg round_table lamp \
  --target-successes 200 \
  --annotation-source scripted \
  --output-suffix <new-production-raw-suffix> \
  --gpu 0 --n-envs 4 --randomness low
```

必须使用全新 suffix；禁止 `--allow-existing-output`，除非是在同一 manifest 证明过的断点续跑。

### ManiSkill：11 个 task

六个 PPO task：`LiftPegUpright-v1 PickCube-v1 PokeCube-v1 PullCube-v1 PushCube-v1 StackCube-v1`。checkpoint 为 236 上 `/home/hy/rr_joint_0821_20260822/assets/maniskill/<TASK>/ppo_pd_ee_delta_pose_ckpt.pt`，入口为：

```bash
<rr-maniskill-python> scripts/data_generation/generate_ppo_pickle.py \
  --env-id <TASK> \
  --checkpoint <CHECKPOINT> \
  --annotation-source scripted \
  --num-traj 100 --num-eval-steps 400 \
  --record-dir <fresh-local-staging-root> \
  --start-seed <fresh-seed> --max-attempts <task-cap> \
  --shader minimal --compress
```

五个 MP task：`PegInsertionSide-v1 PlaceSphere-v1 PlugCharger-v1 PullCubeTool-v1 StackPyramid-v1`，使用当前源码中修正后的 bundled Panda solver：

```bash
<rr-maniskill-python> scripts/data_generation/generate_pickle.py \
  --env-id <TASK> --annotation-source scripted \
  --num-traj 100 --record-dir <fresh-local-staging-root> \
  --start-seed <fresh-seed> --sim-backend physx_cuda \
  --max-attempts <task-cap> --shader minimal --compress
```

起始安全 cap：PPO 常规 task 500、`StackCube-v1` 1,000；Peg 300、Place 200、Plug 3,000、PullCubeTool 500、StackPyramid 500。达到 cap 但不足 100 时只记录 shard 状态并从新 seed 续跑，不能降低 quota。Peg 若再次出现 MPLib 卡死，保持每 seed 独立进程和 600 秒边界。

### AutoMate：99 个 task

正式 ID 是下列集合；任何 manifest 必须恰好匹配，且不得含 `00755`：

```text
00004 00007 00014 00015 00016 00021 00028 00030 00032 00042 00062 00074
00077 00078 00081 00083 00103 00110 00117 00133 00138 00141 00143 00163
00175 00186 00187 00190 00192 00210 00211 00213 00255 00256 00271 00293
00296 00301 00308 00318 00319 00320 00329 00340 00345 00346 00360 00388
00410 00417 00422 00426 00437 00444 00446 00470 00471 00480 00486 00499
00506 00514 00537 00553 00559 00581 00597 00614 00615 00638 00648 00649
00652 00659 00681 00686 00700 00703 00726 00731 00741 00768 00783 00831
00855 00860 00863 01026 01029 01036 01041 01053 01079 01092 01102 01125
01129 01132 01136
```

每个 ID 使用：

- checkpoint `/mnt/nas/share2/home/lq/logs/rl_games/Assembly/automate_assembly_<ID>_2x_noise/nn/Assembly.pth`；
- disassembly `/mnt/nas/share/home/lq/IsaacLab/AutoMate/<ID>/disassemble_traj.json`；
- `RR_ISAAC_ASSET_ROOT=<verified-local-Isaac-root>`，它必须包含 ground、table、Franka、共享 AutoMate 文件和该 ID 的 10 个官方 USD/OBJ/config 文件；
- v2 shared camera；`hardest`、deterministic policy、`--skip-dense-reward`、raw `none`。

下面是维护版 multi-env collector 完成并通过 gate 后的单 task 模板；当前 process-local diagnostic override 不得代替它启动生产：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export RR_ISAAC_ASSET_ROOT=<verified-local-Isaac-root>
./isaaclab.sh -p scripts/automate/generate_pickle.py \
  --headless --device cuda:0 \
  --checkpoint /mnt/nas/share2/home/lq/logs/rl_games/Assembly/automate_assembly_<ID>_2x_noise/nn/Assembly.pth \
  --assembly-id <ID> --annotation-source scripted \
  --disassembly-path /mnt/nas/share/home/lq/IsaacLab/AutoMate/<ID>/disassemble_traj.json \
  --output-dir <fresh-local-staging-root>/<ID> \
  --num-envs 16 --num-successes 50 --max-attempts 10000 \
  --deterministic --compress --skip-dense-reward --seed <fresh-seed> \
  env.camera.gpu_collision_stack_size=134217728
```

不传 `--enable-sbc`。10,000 是低产率 task 的安全 cap，不是成功 quota；collector 必须精确 selected 50 条后退出，同步 batch 多出的 success 记录为 `excluded`。默认 16 env；只有预计至少需要约 256 attempts 的 task 才改为 32 env。

生产实现必须支持 regex parent cameras、per-env recorder、精确 50-save cap、bounded background validation/xz writer 和 `num_envs=1` 向后兼容。实现后先重跑 multi-env schema、`scripted` provenance、same-frame camera geometry、payload/initial-state uniqueness 和 selected/excluded 审计，未通过前不得开正式 campaign。

## 5. 共享 4090 与落盘规则

236 GPU0–3 是公共资源。正式 collector 不能裸启动，必须沿用 `logs/joint-training-full-0821/tools/run_automate_review_chain_then_reserve_236.sh` 和 `run_maniskill_mp_chain_then_reserve_236.sh` 已验证的交接协议，并为正式 campaign 改成独立 production root/50-success target：

1. 启动 2 GiB handoff 并确认 ready；
2. 只释放精确旧 reservation tmux/PID；
3. handoff 贯穿 collector 或串行 task chain；
4. `EXIT/HUP/INT/TERM` 先结束精确 collector 进程组，再启动 full reservation；
5. 看到 `reserved_bytes=` 后才释放 handoff；full reservation 失败则保留 handoff；
6. collector 暂停时必须占卡或无空窗串行下一 task。

不要按用户名或模糊进程名杀进程。每个 run 记录 host、物理 GPU、Git commit、环境路径、checkpoint 路径+SHA、source/schema hash、process seed、env index、global attempt index、result、selected/excluded、wall time 和输出 manifest；不同 lane 不得复用 base seed。

AutoMate 每张 4090 只运行一个 Isaac 进程，默认 16 env。长 horizon `00110` 实测 1/2/4/8/16/32 env 为 166.7/304.7/537.7/754.0/993.2/1049.9 attempted rollouts/hour；32 env 比 16 env 总吞吐只高 5.7%，仅用于预计至少约 256 attempts 的任务。四卡任务队列按 predicted attempts 从大到小调度，低产率 task 起始分散到四卡；lane 结束后领取剩余最长 task，避免单卡长尾。

每个 task 用 `attempts_i=ceil(50/p_i)` 估 attempt 数，但运行速度始终报告 attempted rollouts/hour。四卡均按 16 env 时，aggregate `p=0.8/0.5/0.3/0.1` 对应纯 rollout 约 1.56/2.49/4.15/12.46 小时；加入 strict/xz 和 99 次 app 启停后，AutoMate 中心 ETA 为 6–9 小时，极低产率情形 15–20 小时。每关闭一个 task 就根据实际 attempts/time 更新滚动 ETA。

NAS campaign 根固定为 `/mnt/nas/datasets_tmp/rr_joint_training_0821_scripted_prod_<launch-date>_v1/`，结构为 `raw/{furniturebench,maniskill,automate}`、`processed/lmdb/{furniturebench,maniskill,automate}`、`manifests`、`logs`、`staging`。236 上该路径是 volume2 NFSv4.1、约 19 TiB 可用；r218 当前未正确挂载该路径，恢复前禁止直接写。collector 使用 16–20 GiB aggregate 有界本地 staging，strict/atomic validate 后由 uploader 上传；正式开跑前必须从两台实际机器对精确 volume2 路径完成 durable write/read/hash gate。

## 6. pickle→LMDB

FurnitureBench 使用专用 wrapper：

```bash
/home/hy/anaconda3/envs/rr/bin/python scripts/data_collection/process_furniturebench_pickles_to_lmdb.py \
  --tasks one_leg round_table lamp \
  --input-suffix <production-raw-suffix> \
  --output-suffix <production-lmdb-suffix> \
  --image-annotation-mode guidance-point \
  --episodes-per-task 200
```

该 wrapper 当前调用通用 converter 的 zstd/1 默认值；正式转换前先运行 `--dry-run`，确认展开命令对应最新 RR 默认 `frame_compression=zstd`、level 1。若后续给 wrapper 增加显式 pass-through，再在 launch metadata 中记录完整 flag；不能向当前 wrapper 传它尚不支持的参数。

ManiSkill/AutoMate 使用通用入口；`--task` 必须列出该 source manifest 中的精确 task，不得用占位或超集：

```bash
/home/hy/anaconda3/envs/rr/bin/python -m src.data_processing.process_pickles_to_lmdb \
  --controller diffik --domain sim \
  --task <exact-task-list-in-this-shard> \
  --source scripted --randomness low --demo-outcome success \
  --image-size 224 \
  --image-annotation-mode guidance-point \
  --require-source-image-annotation-mode none \
  --frame-compression zstd --frame-compression-level 1 \
  --input-dir <strict-selected-raw-shard> \
  --output-dir <fresh-lmdb-shard> \
  --provenance-json <source-manifest-summary.json> \
  --n-cpus <bounded-workers> --batch-size <bounded-batch> \
  --map-size-gb <shard-map-size>
```

通用 converter 的 `--randomness low` 当前只是 legacy 路径/枚举字段，不得覆盖真实仿真语义；`provenance-json` 必须对 ManiSkill 写 `task_native`、对 AutoMate 写 `hardest_init`，并保存精确 seed/initial-state 清单。若 converter 在正式实现前支持新的 canonical enum，应改用新 enum 并保留向后兼容映射。

源 pickle 不原地修改。默认合成 FurnitureBench、ManiSkill、AutoMate 各一个 LMDB，预计分别 90–140、28–45、28–45 GiB，总计 145–230 GiB。若 representative shard 显示 FurnitureBench 会超过约 120–140 GiB，或本地构建无法保留 40 GiB 安全余量，才按 task 或 30–40 GiB shard 拆分；ManiSkill/AutoMate 默认不拆。

输出必须从 fresh `.building.lmdb` 开始，禁止 overwrite。每个 LMDB 记录 source manifest/hash、精确 task/episode/frame 数、`source_image_annotation_mode=none`、`image_annotation_mode=guidance-point`、zstd level 1、source commit 和 teacher hash；完成 full validator、loader smoke、`data.mdb` bytes/SHA-256 后才能原子改名。

PPU96 `/root` 当前约 660 GiB 可用，145–230 GiB 最终集放入后预计余 430–515 GiB，无需预先迁走现有数据。先完整阅读 `reports/claude/ppu96_single_card_4way_zstd_runbook.md`，在远端用 clean worktree/clone 更新到审计后的最新 RR，不覆盖其 dirty worktree。每个 LMDB 直接 rsync 到 `/root/rr-local-data/processed/joint_training_0821/<source>.lmdb.incoming/`，核对 bytes/SHA、validator 和 loader 后原子改名并写 `.transfer-complete`。

## 7. 启动 gate 与验证边界

用户要求的检查范围已经完成：FurnitureBench 3/3、ManiSkill 11/11 均有成功轨迹、strict 2D 与视频；AutoMate 对历史 final 指标最低的 20 个模型做了 fresh 抽查，保留的 19 个 hardest task 可产生 strict success，`00755` 被排除；`00211` 另完成四卡 pipeline smoke。AutoMate v2 shared camera 的 `00410` 无 override 回归为 44/44 点画内。

这不等于 AutoMate 99/99 已逐 task fresh rollout。当前仍是 planning-only；用户批准本文档前禁止创建 production NAS 目录、造数据、转换 LMDB 或上传。批准后按下列顺序执行：

1. 实现并提交维护版 AutoMate multi-env collector，完成 1/16/32 env 严格回归；
2. 补齐并 SHA 验证正式 99 task 的官方 Isaac 5.1 本地资产 bundle；现有 review bundle只覆盖历史低指标样本；
3. r218 恢复 `/mnt/nas/datasets_tmp` 正确挂载，并与 236 一起通过 volume2 durable write/read/hash gate；
4. 每个保留 ID 的首个新 strict success 同时作为 per-task camera/schema gate；失败 task 隔离后继续其他 task，不擅自改相机，attempted-rollout 吞吐和剩余 ETA持续写入 manifest；
5. 三个仓库的生产源码均 commit/push，并把 commit/hash写入 launch metadata；PPU96 使用 clean worktree/clone，不覆盖远端修改。

只有 task success 且 raw contract/active point/3D→2D/内容哈希全部通过的文件才能进入 production manifest。任何 diagnostic、SBC、failure、旧 pickle、camera candidate 和 validation 超采文件都必须被 manifest 排除。
