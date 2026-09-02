# 联合训练完整实验规划 0821

状态：数据配额已由用户于 2026-08-23 通过；2026-08-31 用户将本 campaign 及后续 rollout 的 annotation source 固定为 `scripted/geometry GT`，不使用 VLM，后续 launch 不再逐次确认。FurnitureBench 三任务已确认通过。ManiSkill 11 个 task 均已有成功轨迹与 strict 2D 验证；最新四个 MP 难项各选择 10 条新成功轨迹，最终 40/40 审计通过，随后修复版 Peg/Plug 又各用 10 条全新 strict success 完成生产吞吐复测。**这里的 10 条只代表当前 task 检查门槛；正式数据集仍为 ManiSkill 100 条/task、合计 1,100 条。** AutoMate 已对训练 final success 最低的 20 个模型做 fresh 抽查：19 个 hardest-init 可产成功数据；`00755` hardest-init 不能成功，已由用户明确从正式任务集合中排除，AutoMate 正式任务数固定为 99。2026-09-01 用户将 AutoMate 正式配额下调为 **50 条/task**，并要求以 attempted rollouts/hour 为并行速度指标。用户随后批准生产并要求持续监督；2026-09-02 又把最终数据固定为 **一份 raw pickle、五个训练 condition、默认 15 个物理 LMDB**，其中两个 condition 放 PPU96 根盘，三个放 EPFS。

新 Codex session 不应从历史日志反推生产参数；唯一执行交接为 `reports/joint_training_data_collection_handoff.md`。本文保留任务证据、资源规划和历史诊断，若执行配置与交接文档冲突，以交接文档为准。

## 1. 数据集由什么组成

### 1.1 总量与口径

本轮只生成新数据，不复用或补标旧 pickle。排除 AutoMate `00755` 后，物理数据集更新为 **113 个 task、6,650 条成功轨迹、约 526,386 个训练 transition**：

| Environment | Task 数 | 每 task 成功轨迹 | 成功轨迹合计 | 预计 transition | 数量来源 |
|---|---:|---:|---:|---:|---|
| FurnitureBench | 3 | 200 | 600 | 约 328,000 | 历史正式 campaign 就是 200/task；600 条曾产生 328,386 transitions |
| ManiSkill | 11 | 100 | 1,100 | 约 99,000 | 100/task 已由用户确认；step 按当前成功 pilot 或 task horizon 外推 |
| AutoMate | 99 | **50** | **4,950** | 约 99,000 | 明确排除 `00755`；按 20 transitions/trajectory 规划 |
| 合计 | 113 | — | **6,650** | **约 526,386** | transition 指落盘 action step，不是 PhysX substep |

物理条数和训练时的 source sampling 是两件事。历史讨论中的主训练权重为 `FurnitureBench / AutoMate / ManiSkill = 50% / 35% / 15%`；精确来源是 `/home/hy/.codex/sessions/2026/08/21/rollout-2026-08-21T11-55-44-01a02275-dcb7-7191-83f8-48bfc936542f.jsonl` ordinal 10。该比例已有 sampler/DDP smoke，但仍是工程配置，不是已证明最优的比例。

randomness 合同按仿真器原生语义记录：FurnitureBench=`low`；ManiSkill 保留每个 task 的 native randomized reset，并写 `randomness_semantics=task_native`；AutoMate=`hardest_init`、deterministic specialist policy、no SBC，变化只来自不重复的 reset seed。manifest 必须记录 process seed、env index、global attempt index 和初始状态 hash，不能把 ManiSkill/AutoMate 都伪装成 FurnitureBench 的 `low/med/high`。

物理配额下降不改变训练时的 source sampler 权重。容量按五个 condition 管理：canonical raw 仍只有一份，约 **350–450 GiB**；每个 condition 的三来源 zstd level-1 LMDB 当前中心估计约 **141–158 GiB**，五套约 **705–790 GiB**。旧的 145–230 GiB/condition 区间继续作为保守上界检查。加上单个 `.building.lmdb`、NAS `.incoming`、本地 staging、manifest、reject 和安全余量，NAS campaign 预留从 1.2 TiB 提高到 **2.0 TiB**。旧数据只作为步数、大小和吞吐参考，不进入本轮 manifest。

### 1.2 FurnitureBench：3 个 task

历史 600 条正式数据共有 328,386 transitions，suite 均值约 547 transitions/trajectory。旧报告只保留 aggregate，没有可靠的逐 task transition 总数，因此下表先把 aggregate 均分为容量预算；20/task 新 pilot 后替换成每 task 的真实均值和 p95。

| Task | 新成功轨迹 | 规划 transition | 成功轨迹验证 | scripted 2D guidance 验证 | GPU 仿真 | 当前结论 |
|---|---:|---:|---|---|---|---|
| `one_leg` | 200 | 约 109,000 | v2 保存 1 条 success，296 transitions | 全帧 strict + 重投影通过；raw mode=`none` | r218 GPU PhysX 已验证 | 用户视频目检通过 |
| `round_table` | 200 | 约 109,000 | v2 保存 1 条 success，541 transitions | 全帧 strict + 重投影通过；raw mode=`none` | r218 GPU PhysX 已验证 | 用户视频目检通过 |
| `lamp` | 200 | 约 109,000 | v2 保存 1 条 success，608 transitions | 全帧 strict + 重投影通过；raw mode=`none` | r218 GPU PhysX 已验证 | 用户视频目检通过 |

首轮按用户指定复用了 `/data/hy/gpu-snatcher/auto_data_preparation.sh` 和 r218 现有 `rr` 环境。标准流程使用 4 env、`action_type=pos`、task-specific horizon/padding、low randomness，5 分 14 秒内产出 11 条成功轨迹。原始动作在 FurnitureSim 内会饱和到 `[-1,1]`；writer 现在把实际执行的 gripper 命令写入 canonical pickle。11/11 文件的 base-frame delta action、时序、GT 2D metadata 和重投影均通过，但 `--guidance-point-on-image` 在采集期把红点烧入了 front raw，因此整批 4,845-transition/3.18-GiB gate 只能作为错误证据，不能进入正式 manifest 或 LMDB。

修正后的正式合同是两阶段：`scripts/data_collection/collect_furniturebench.py` 只保存无标记 wrist/front RGB-D 和 skill/guidance metadata，并在 pickle 顶层写 `image_annotation_mode=none`；`scripts/data_collection/process_furniturebench_pickles_to_lmdb.py` 要求源 pickle 明确为 `none`，再按显式 `--image-annotation-mode` 离线绘制并把 mode 写入 LMDB metadata。首轮旧文件缺少 `none` provenance，已验证会被新 conversion runner 拒绝。

#### 固定执行入口（context 切换后仍以此为准）

FurnitureBench 后续不再直接调用 `gpu-snatcher/auto_data_preparation.sh`；它只作为参数来源参考。唯一维护入口和完整说明是 `scripts/data_collection/README.md`，执行顺序固定为：

1. annotation source 固定为 `scripted/geometry GT`；collector 必须显式传 `--annotation-source scripted`，不再逐次询问。
2. 用 `collect_furniturebench.py` 写全新 suffix。采集命令不得含任何 `*-on-image`/`grasp-part-annotate` flag；pickle 必须为 `image_annotation_mode=none`。
3. 对每个新 pickle 做 schema、GT 3D→2D 重投影和 raw-image provenance 审计。
4. 只有上述检查通过，才用 `process_furniturebench_pickles_to_lmdb.py` 离线绘图；conversion 固定要求 `--require-source-image-annotation-mode none`，并将所选 `image_annotation_mode` 写入 LMDB metadata。
5. 原始 pickle 永不原地修改；不同 raw/LMDB campaign 使用不同 suffix。首轮污染数据 `rgbd-skill-point-fb-gate-20260823-scripted` 永久排除。

本轮三任务 v2 验证的精确采集命令是：

```bash
/home/hy/anaconda3/envs/rr/bin/python scripts/data_collection/collect_furniturebench.py \
  --tasks one_leg round_table lamp \
  --target-successes 1 \
  --annotation-source scripted \
  --output-suffix rgbd-skill-guidance-metadata-fb-gate-20260823-v2 \
  --gpu 0 --n-envs 4 --randomness low
```

目检通过后的正式 200/task campaign 只更换为 `--target-successes 200` 和新的 production suffix。下面是已验证过的 `rgbd+gp+skill` 转换示例；正式流水线对 3.4 节列出的五个 mode 分别调用同一维护入口：

```bash
/home/hy/anaconda3/envs/rr/bin/python scripts/data_collection/process_furniturebench_pickles_to_lmdb.py \
  --tasks one_leg round_table lamp \
  --input-suffix <clean-raw-suffix> \
  --output-suffix <annotated-lmdb-suffix> \
  --image-annotation-mode guidance-point \
  --episodes-per-task 200
```

v2 实测结果：三任务各保存 1 条成功轨迹，共 1,445 transitions；3/3 pickle 全帧 strict audit 通过，均为 `annotation_source=scripted`、`image_annotation_mode=none`。样本 LMDB 含 3 episodes/1,445 timesteps，全量 stats/schema 校验通过，metadata 明确记录 `source_image_annotation_mode=none`、`image_annotation_mode=guidance-point`。三个 LMDB 首帧与同一 raw pickle 的离线 annotation 结果逐像素一致，源 pickle未修改；三个替换视频已由用户于 2026-08-23 目检通过。审计证据位于 `logs/joint-training-full-0821/furniturebench_metadata_v2_gate_r218_20260823/runtime/`。

### 1.3 ManiSkill：11 个 task

`预计 transition` 是每 task 100 条数据的规划值。带 `*` 的数值只是 task horizon 上界外推，因为当前没有成功轨迹，不能视为可靠容量或 ETA 证据。

| Task | Rollout policy | 20-success pilot / 原始 task success | 问题归因 | 标注结论 | 生产状态 |
|---|---|---|---|---|---|
| `LiftPegUpright-v1` | 官方 state PPO，`pd_ee_delta_pose` | reset guard 修复后 strict 20/20；目标点缓存修复后的 fresh success 1/1 | **已修复两类 annotation bug**：reset 首帧 FSM 泄漏；pick/lift target 每帧依赖当前 TCP/peg 重算而漂移 | 新轨迹 pick 5 帧与 lift/place 7 帧各自的 3D/2D target span 均为 0；pick 点距 peg 中心 0.10 m，小于 0.12 m 半长；全点在画内 | 通过；保留原始 40° front 相机，使用阶段进入时冻结的 geometry-GT target |
| `PickCube-v1` | 官方 state PPO，`pd_ee_delta_pose` | 修复 reset 泄漏后 21 attempts 得到 20 task success，strict 20/20 | **已修复 annotation FSM reset**；仅 1 条 policy fail，不是相机或投影问题 | 首 skill 20/20 为 `pick`；304/304 active 点在画内，重投影零分歧 | 通过新 20-success 复核；保留原始 40° 相机 |
| `PokeCube-v1` | 官方 state PPO，`pd_ee_delta_pose` | 相机后退 0.42 m 后，新 cohort 28 attempts 得到 20 task success，strict 20/20 | reset 泄漏与相机下缘覆盖均已解决；8 条 policy fail 与视觉无关 | 新 505/505 active 点在画内；旧 exact seeds 重跑 19/20 task success 且 19/19 strict，未复现 seed 的旧记录在新相机下重投影 15/15 可见；均零分歧 | 通过 fresh 20-success；使用 task-specific 40° 相机 |
| `PullCube-v1` | 官方 state PPO，`pd_ee_delta_pose` | strict 20/20 | 无当前阻塞 | 20 条全帧 strict pass | 通过 pilot |
| `PushCube-v1` | 官方 state PPO，`pd_ee_delta_pose` | strict 20/20 | 无当前阻塞 | 20 条全帧 strict pass | 通过 pilot |
| `StackCube-v1` | 官方 state PPO，`pd_ee_delta_pose` | strict 20/84；原始 task success 23/84（27.4%） | **主要是 policy 成功率**：61 条 task fail；另有 3 条相机拒绝 | 20 条全帧 strict pass | 达到 20 条，但采集产率仅 23.8% |
| `PegInsertionSide-v1` | 修正后的 bundled Panda MP solver | 旧实现合计 10 strict/83 attempts；3 mm 补偿修复后 fresh strict 10/11，314 秒 | 除 GPU stale OBB 外，原 solver 虽规划/抓取均成功，peg head 仍因约 3 mm 孔间隙与下垂残差撞在孔外；hole-frame z `+3 mm` 的 fresh10 配对由 1/10 提升到 6/10 | 修复版 10 条全帧通过，0 camera/annotation reject | 通过 10-success 数据与生产吞吐 gate；100 条单卡纯采集约 0.87 小时 |
| `PlaceSphere-v1` | 修正后的 bundled Panda MP solver | 新批次 20/20 strict；按用户上限选择前 10 条 | **已确认实现 bug**：OBB 中心为 builder 原点而不是 live sphere；另有单段对角搬运与释放等待不足 | 20/20 全帧通过，选择 10、排除多采的 10 | 通过；修正后当前样本成功率 100% |
| `PlugCharger-v1` | 修正后的 bundled Panda MP solver，`pd_joint_pos` | 旧实现 10 strict/202 attempts（4.95%）；修复版 fresh strict 10/97，1,215 秒 | 已补全 motion 失败传播和 grasp 验证；socket-frame z `+0.5 mm` 在 60 个配对 seed 中保留全部 7 个 baseline success 并新增 4 个。5/10 段插入均 0/4，已拒绝 | 修复版 10 条全帧通过，0 camera/annotation reject | 通过 10-success 数据与生产吞吐 gate；100 条单卡纯采集约 3.38 小时 |
| `PullCubeTool-v1` | ManiSkill bundled Panda MP solver | 0.32 m 相机下旧/新分别 19/20、18/20；改为总后退 0.42 m 后只复核三个旧失败 seed | 相机问题已关闭：`831408` 与 `832116` 实际重跑 strict pass；`832115` 本次 `screw plan failed`，但其上一轮成功记录在新标定下 259/259 重投影可见 | `831408` 为 436/436、最小边缘余量 9.31 px；`832116` 为 245/245、9.95 px；`832115` 重投影 259/259、10.72 px；均零分歧 | 通过相机/annotation gate；生产时 MP solver 偶发失败只影响吞吐，不写入数据 |
| `StackPyramid-v1` | 修正后的 bundled Panda MP solver | 新批次产出 12 条 strict；按用户上限选择前 10 条 | **已确认多处实现 bug**：stale OBB、错误绝对 push target、夹爪轴、失败后继续执行、抓取候选不足 | 12/12 全帧通过，选择 10、排除多采的 2 | 通过 10-success 数据 gate；仍有少量 base push / grasp 失败 |

最新四任务清单为 `logs/joint-training-full-0821/maniskill_remaining4_gt_20260831_v1/final_select_first_audit_20260831.json`，SHA256 为 `7ab44098de1f4a765dace2f4e62b4634ac22ccdf48f64afeac09210a4bda7902`。最终选择 **40 条、7,127 transitions、457.759 MiB**：Plug 1,643 steps，Peg 1,354，Place 1,381，Stack 2,749。40/40 均为 `annotation_source=scripted`、`image_annotation_mode=none`，无 invalid、无重复内容哈希；各 task 的最坏 front 边缘余量分别为 20.22、12.55、56.89、4.11 px，最大 3D→2D 重投影误差均小于 `4.30e-5 px`。Place 多出的 10 条和 Stack 多出的 2 条只列为 excluded，原文件未删除或移动。

对 bundled MP solver 的共同实现审计确认：GPU backend 下 `get_actor_obb()` 使用底层 PhysX component 的 builder pose，而不是 reset 后 `Actor.pose` 的 live transform。固定 seed 的 legacy OBB 与按 live pose 重建的 OBB 中心偏差为 Peg 20.5 cm、PlaceSphere 11.7 cm、PullCubeTool 22.0 cm、Stack 三块 cube 20.2/96.0/112.0 cm，旋转矩阵也明显不一致；同一检查在 `physx_cpu` 下所有中心和旋转误差均为 0。这是原版 Peg/Place/Stack 在 GPU 仿真中近乎全失败的共同上游 bug；PullCubeTool 虽绕过错误 center，仍从 stale OBB 读取 closing axis。另一个共同缺陷是示例 solver 普遍不检查所有 motion 返回值、不验证抓取稳定性，也没有单次规划超时；因此这些脚本只能作为待验证的 expert 起点，不能因“ManiSkill 自带”直接视为 production-ready。

公共 `get_actor_obb` 现已改为始终从 ManiSkill `Actor.pose` 的 live tensor transform 构建 world OBB，不再使用 GPU 下未同步的 component pose；修复后同一 GPU 对照的 OBB center 误差降到数值零。PlaceSphere 对旋转对称球体不再使用任意 OBB axis，而采用正交化后的当前 TCP closing axis；Place 固定 seeds `110–114` 为 5/5。Peg 新相机成功集合 5/5、Stack 3/3、PullCubeTool 1/2（另 1 次明确 planner failure）、Plug 正式修复后代表 success seeds 5/5。本地定向单测 14/14、236 NAS 环境同样 14/14。236 GPU0 的 fresh strict 回归 Plug/Place 各 1/1，独立审计均为 `scripted`/`none`、无 invalid/duplicate/runtime error；证据为 `logs/joint-training-full-0821/maniskill_common_bugfix_20260831_v1/common_bugfix_gpu0_v1_audit.json`。GPU0 运行时由 handoff 无空窗交接，结束后 full reservation 恢复到约 18.3 GB。

Peg 的新增阶段诊断进一步把低产率定位到物理插入，而不是 planner、相机或标注：四个失败 seed 的 reach/grasp/pre-insert/insert 均返回成功并保持抓取，但插入后 peg head 仍停在孔外 8.0–9.8 cm；插入前 z 残差约 2.4–4.1 mm，已接近或超过 3 mm 单边间隙。hole-frame z 补偿的固定 7-seed 对照中，2 mm 为 5/7、3 mm 为 6/7；独立 fresh10 配对从 0 mm 的 1/10 提升到 3 mm 的 6/10。将 3 mm 写入 solver 后，正式 RGB-D/压缩/strict collector 在 seeds `843101–843111` 中得到 10/11 success、314 秒，10/10 pickle 审计通过。对应 10 条为 1,363 transitions、78,254,944 bytes，最小 front 边缘余量 43.95 px，最大重投影误差 `2.39e-5 px`。该补偿只作用于 Panda/PhysX expert 的执行目标；skill annotation 仍使用任务的名义孔中心 geometry GT，没有把 solver 调参常数变成新的 annotation source。

Plug 修复版的正式 strict collector 在 seeds `840101–840197` 中得到 10/97 success、1,215 秒，10/10 pickle 审计通过；共 1,600 transitions、99,819,468 bytes，最小 front 边缘余量 38.90 px，最大重投影误差 `2.50e-5 px`。两项均为 `annotation_source=scripted`、`image_annotation_mode=none`，无 invalid、重复哈希、runtime error/warning 或 camera reject。完整 A/B、strict manifest 与 job results 固化在 `logs/joint-training-full-0821/maniskill_solver_bugfix_20260831_v2/`；Peg/Plug audit 文件 SHA256 分别为 `33587500697e3919f4d9a60513686f2a4a2b78b52d8ea0bfd64077731112182f`、`13ffbbdcd066a4cf38f9ff0f56c54116f0cc30d9e11780d2fc607394e7c20688`。

2026-08-23 的首批足量 pilot 共保存 **132 条新 success、8,042 transitions**。所有保存文件均为 `annotation_source=scripted`、`image_annotation_mode=none`，全帧 active point 在 224×224 内；独立 3D→2D 最大重投影误差为 `4.613e-5 px`。但它发生在 reset guard 修复前，`PickCube/PokeCube/PullCubeTool` 的跨 episode skill 状态可能错误；该批只保留为历史诊断，不进入本轮生产 manifest，也不能再用“凑够 20 个可见样本”证明这三项语义正确。2026-08-23 历史 manifest 为 `logs/joint-training-full-0821/maniskill_20success_pilot_20260823/runtime/audit_manifest.json`。

问题样本检查视频位于 `logs/joint-training-full-0821/videos/maniskill_problem_review_20260823/mp4/`，三栏固定为 wrist raw｜front raw｜pickle→video 时由 offline annotation util 绘制的 front guidance point。PPO 短轨迹提供 5 fps 慢放版；`PlugCharger` 和 `PullCubeTool` 使用 15 fps：

- `PickCube-v1-success-slow.mp4`、`PokeCube-v1-success-slow.mp4`、`StackCube-v1-success-slow.mp4`。
- `PlugCharger-v1-success.mp4`、`PullCubeTool-v1-success.mp4`。
- `LiftPegUpright-v1-success-slow.mp4` 仅是 2026-08-22 的历史 smoke。2026-08-29 新诊断视频为 `logs/joint-training-full-0821/videos/lift_peg_upright_camera_20260829_v1/mp4/LiftPegUpright-v1-seed824102-diagnostic.mp4`；它是 `diagnostic_only=true`，不进入数据 manifest。

这些视频展示的是被 strict gate 接受的成功例子。三个 0-success solver task 与相机拒绝样本没有保存 production pickle。对应的失败视频流程已固化为 `logs/joint-training-full-0821/tools/collect_maniskill_diagnostic_pickle_236.py`、`render_maniskill_diagnostic_video.py` 和 `run_maniskill_problem_video_diagnostics_then_reserve_gpu2_236.sh`：只写独立 diagnostics 根并标记 `diagnostic_only=true`，绝不进入数据 manifest；中栏保持 raw front，右栏先使用标准 annotation util，对画外 GT 另加明确标为 `DIAG offscreen` 的边界箭头。

Lift seed `824102` 已于 2026-08-29 用 scripted/geometry GT 单独补录：11 steps task success，12 个 observation 的 front point 全部有限且在 224×224 内。`--num-eval-steps` 与诊断 `--max-steps` 默认值已按用户要求从 50 统一改为 400，并由 236 单测 `2 passed`；但任务注册的 `max_episode_steps=50` 仍会 truncation，因此这项改动不是 Lift 的修复依据。其余待补录案例为 PlugCharger `823601`、PegInsertionSide `825105`、PlaceSphere `825201`、StackPyramid `825501`。

2026-09-01 的视频目检暴露了 Lift 的第二类语义问题：旧 pick target 在每次 annotation query 时用当前 TCP 姿态重新求 grasp frame，导致进入抓取前最后一帧横跳约 12.6 cm；lift target 又由当前 peg/TCP 每帧重算，看起来随夹爪移动。修复后 pick、lift、rotate、lower 均在阶段进入时缓存 geometry-GT target，并在 reset 或 grasp loss 时失效；pick 的初始 intended near-end grasp 语义没有改变。fresh seed `850201` 为 1/1 success，pick 5 帧和 place 7 帧的 3D/2D span 均为 0，最小 front 边缘余量 26.10 px；pick 点距 peg 中心 0.10 m，位于 0.12 m 半长内。本地完整 annotation 测试为 57 passed，source/video audit 均通过。证据与替换视频位于 `logs/joint-training-full-0821/videos/lift_guidance_semantics_fix_20260901_v1/`；旧统一目录 Lift 视频已被该 fresh 视频取代。

2026-08-31 用正式 PPO generator 新增的 pre-validation 证据复现并定位了旧结论：修复前在原始 40° 相机下取 20 条 task success，只有 1 条 strict pass；但 19 条失败轨迹的 observation 0 已错误进入 `place`，使用高度约 0.34–0.39 m 的目标并投影到图像上边界外，只有第一条从 `pick` 开始。记录的 2D point 与独立 3D→2D 重投影完全一致，因此不是投影器把可见点误判为画外。根因是前一 episode 的 grasp/contact 状态在 reset 后首次 annotation query 仍可能为真，使已重置为 `PICK` 的 FSM 立即跳到 `LIFT/PLACE`。FSM 现在仅对刚 reset 的 env 抑制一次 `PICK→LIFT` 转移；同一 guard 也加入 `PickCube/PokeCube/PullCubeTool`。本地完整 annotation 测试为 56 passed；236 上 annotation 定向测试为 7 passed、PPO generator 为 3 passed、MP generator 为 6 passed。Lift 修复后以新 seeds `831101–831120` 重跑：20/20 task success 全部 strict pass，首个 active skill 为 `pick` 的计数是 20/20，248/248 active 点几何可见且被记录，独立重投影零分歧，首个点距图像边缘的最小余量为 13.63 px。两轮清单分别位于 `logs/joint-training-full-0821/lift_visibility_recheck_20260831/v1/audit_manifest.json` 与 `v2/audit_manifest.json`；两轮均为隔离的 `diagnostic_only` 输出，不进入 production manifest。

同日三任务 fresh recheck 固定使用 `annotation_source=scripted` 和原始 40° front 相机，并按“恰好 20 条原始 task success”保留 strict gate 前证据：Pick 为 20/20，Poke 为 13/20，PullCubeTool 为 12/20；三项首个 active skill 都是 20/20 `pick`，记录与几何重投影分歧均为 0。20-seed 覆盖计算表明，若只沿原光轴扩大方形 FOV，Poke 覆盖当前样本至少需 48.51°（留 8 px 为 51.77°），PullCubeTool 至少需 65.57°（留 8 px 为 69.50°）；这些只是相机候选，不会未经目检写回。审计清单和逐 seed JSON 位于 `logs/joint-training-full-0821/three_task_visibility_recheck_20260831_v1/`。相机拒绝代表视频位于 `logs/joint-training-full-0821/videos/poke_pull_camera_review_20260831_v1/mp4/`：Poke seed `831301` 在 `push` 阶段从下缘出界，PullCubeTool seed `831402` 在 `push` 阶段从左侧出界；两者均为 `diagnostic_only=true`。

用户目检后只沿原光轴后退相机，朝向与 40° FOV 均不变；task-specific override 保证其他任务继续使用原相机。Poke 总后退 0.42 m：新 20 task success 全部 strict pass；旧 20 exact success seeds 中 19 个成功复现且 19/19 strict，seed `831323` 本次 policy task fail，但其旧记录的 15 个 active 3D 点用新相机标定重投影后 15/15 可见。PullCubeTool 先总后退 0.32 m，旧 exact 20 中 19/20 strict、新 20 中 18/20 strict，剩余三条后续 `push` 点最坏 `u=-3.93 px`；随后按用户要求改为与 Poke 相同的总后退 0.42 m，并只复核这三个失败 seed。`831408` 与 `832116` 实际重跑分别 436/436、245/245 active 点可见且 strict pass，最小边缘余量为 9.31、9.95 px；`832115` 本次 MP `screw plan failed`，未产生 task success，但其上一轮成功记录的 259/259 active 3D 点在 0.42 m 新标定下重投影全部可见，最小余量 10.72 px。由此 Pull 相机/annotation 项关闭，`832115` 只保留为 solver 吞吐证据。完整 old/new 证据位于 `logs/joint-training-full-0821/poke_pull_retreat_old_new20_20260831_v1/`，定向三 seed 证据位于 `logs/joint-training-full-0821/pull_retreat042_failed3_recheck_20260831_v1/`；全部标注均为 `scripted` 且 recorded-vs-geometry 分歧为 0。

这次 old/new 复核入口已固化为 `logs/joint-training-full-0821/tools/run_poke_pull_retreat_old_new20_then_reserve_gpu2_236.sh`：旧 cohort 使用上一轮清单中的 exact success seeds 并逐 seed `max-attempts=1`，新 cohort 使用不重叠 seed 连续收集 20 个原始 task success；四组分目录、分别审计且不读取旧 pickle。0.42 m Pull 三 seed 定向复核入口为 `run_pull_retreat042_failed3_then_reserve_gpu2_236.sh`。两个 runner 的第三参数均强制为 `scripted`，使用 handoff 占卡、精确结束上一 GPU2 session，结束后恢复全显存 reservation。所有输出均在独立 diagnostics 根，不进入 production manifest。

这套复核流程已固化：正式 PPO/MP generator 分别支持 `--diagnostic-task-successes` 和 `--attempt-diagnostics-dir`，因此 strict reject 也会留下 3D point、相机标定、记录 2D 与独立重投影证据；`run_three_task_visibility_recheck_then_reserve_gpu2_236.sh` 串行复核并恢复占卡；`analyze_maniskill_camera_coverage.py` 汇总多 seed FOV 边界；`run_three_task_problem_videos_then_reserve_gpu2_236.sh` 与 `render_maniskill_diagnostic_video.py` 生成 wrist raw｜front raw｜离线 annotation 三栏视频。上述 runner 都要求第三个参数显式为 `scripted`，输出根存在时 fail-closed，并保持 diagnostic 与 production manifest 隔离。

PlugCharger 的 2026-08-31 首轮本地阶段诊断使用既有 `rr-maniskill`、3060 GPU 和 4 个固定 seed，不写 pickle、不记录图像。4/4 均完成 reach、grasp、pre-insert 且保持抓取；失败样本最终停在目标前约 17–18 mm，姿态偏转约 0.14–0.25 rad，正好对应双插脚在窄孔入口碰撞。该小组对额外保持、插入降速、固定偏置、overshoot、抓取倾角和闭环重算没有给出稳定正结果，因此这些候选当时均未直接写入。随后扩大到三个互不重叠的 20-seed 配对组，socket-frame z `+0.5 mm` 合计把 7/60 提升到 11/60 且保留全部 baseline success，才据此进入正式 solver；5/10 段插入在反例中为 0/4，保持拒绝。阶段证据在 `logs/joint-training-full-0821/plug_solver_stage_diagnosis_20260831_v1/`，扩大 A/B 与 strict 吞吐证据在 `maniskill_solver_bugfix_20260831_v2/`。

交互相机工具已固化为 `scripts/data_collection/debug_maniskill_camera.py`，并按用户要求固定在 r218 本地桌面运行，不占用或释放 236 的四卡 reservation。后续本地 ManiSkill 调试固定复用已有 `/home/hy/anaconda3/envs/rr-maniskill`，其当前可用合同为 Python 3.11.15、ManiSkill 3.0.1 editable source `/data/hy/ManiSkill`、SAPIEN 3.0.3、Gymnasium 1.3.0、NumPy 1.26.4 和 OpenCV 4.11.0.86，`pip check` 无冲突；不再为单次相机或诊断任务重复新建环境。工具循环重放 seed `824102` 的官方 state PPO 成功 pickle，强制要求顶层 `annotation_source=scripted`；为避免本地依赖/动力学差异把相机检查变成失败的重新 rollout，最终模式逐帧写回 pickle 中保存的 Panda qpos 和 peg pose，并把保存的 geometry-GT guidance point 从 base frame 还原到 world frame。写回检查的 qpos 误差为 0、peg pose 矩阵最大误差为 `5.96e-8`；本地 GUI 日志明确为 `source=scripted backend=physx_cpu replay=recorded_state frames=12 checkpoint=provenance_only`，每轮固定打印 11 transitions、`success=True` 后自动重播。checkpoint 只核对并记录 policy provenance，不在相机工具中重新推理。`P` 暂停、`R` 重播、`C` 保存 viewer proposal；显式 `--apply-on-confirm` 时才写回 `camera_contract.py`。viewer 在 scene attach 后强制恢复与数据相同的 40° FOV 和 800×800 方形 viewport，避免 SAPIEN 默认 90° viewer 造成视觉合同不一致。3060 只承担 Vulkan 显示渲染；不可清空 `CUDA_VISIBLE_DEVICES`，否则 SAPIEN 看不到渲染设备。滚轮默认 `--scroll-speed 0.02`（每格 2 cm），按住 Shift 时为每格 2 mm。单条循环回放只用于语义和构图目检，不能替代多 seed pre-validation 覆盖率审计；本轮已恢复并保留原始 FB 对齐的 40° 相机，不采用手工调整的 proposal。

其余诊断采用 handoff→精确结束旧 reservation→并行或串行诊断→恢复全显存 reservation。annotation source 已固定为 scripted/geometry GT，不再逐次确认；仍须根据当前 reservation session 更新精确命令，以下旧批量命令仅留作流程参考，不应直接复制运行：

```bash
tmux new-session -d -s rr_joint0821_maniskill_problem_videos_gpu2 \
  "bash /home/hy/rr_joint_0821_20260822/run_maniskill_problem_video_diagnostics_then_reserve_gpu2_236.sh \
  2 rr_joint0821_maniskill_20success_gpu2 scripted"
```

#### 固定 pilot 入口与当前执行边界

236 上唯一使用的干净环境为 `/mnt/nas/share/home/hy/miniconda3/envs/rr-maniskill-gpu-0821-20260822`，source copy 为 `/home/hy/rr_joint_0821_20260822/ManiSkill`。6 个 PPO checkpoint 位于 `/home/hy/rr_joint_0821_20260822/assets/maniskill/<task>/ppo_pd_ee_delta_pose_ckpt.pt`；5 个 MP task 使用该 source copy 内置的 Panda solver。PPO 与 MP 的生成入口分别固定为 `scripts/data_generation/generate_ppo_pickle.py` 和 `scripts/data_generation/generate_pickle.py`。

所有 pilot 都不读取旧 pickle。2026-08-31 用户把当前验证上限改为 **每 task 最多选择 10 条新的 strict-pass 成功轨迹**；已启动而多采的 Place/Stack 文件保留并在清单中显式 excluded，不删除、不计入 selected。生产数据目标仍是本节表中的 100/task；Peg/Plug 已用修正后的 expert 各自重测 10 条 strict success，不能把验证上限误写成正式 quota。

四卡通用 runner 为 `logs/joint-training-full-0821/tools/run_maniskill_mp_chain_then_reserve_236.sh`，服务器副本位于 `/home/hy/rr_joint_0821_20260822/`。它既支持精确旧 tmux session，也支持经命令行核验为对应 GPU `occupy_one_gpu_236.py` 的 `pid:PID` 接管；两种路径都必须先建立 2 GiB handoff，任务结束或收到信号时先确认 full reservation ready 才释放 handoff。固定行为如下：

1. annotation 参数只接受 `scripted`，generator 同样要求显式 `--annotation-source scripted`。
2. 每个 run tag 使用独立 fresh shard；存在同名输出就 fail-closed，不拼接旧 pickle。
3. raw pickle 只保存原始 RGB-D 与 skill/guidance metadata，顶层强制 `image_annotation_mode=none`；marker 只在 pickle→LMDB 阶段绘制。
4. Peg 每个 seed 使用独立进程和 600 秒边界，隔离 MPLib 偶发卡死；其他任务可 batch。
5. 每张共享 GPU 在 collector 前后由 2 GiB handoff 与 full reservation 无空窗交接，`EXIT/HUP/INT/TERM` 都恢复占卡；只结束精确旧 session/PID。
6. `audit_maniskill_mp_shards.py --select-first` 审计所有生成文件，按稳定路径顺序选择每 task 前 10 个 valid 文件，并保留 excluded 和 runtime warning。

### 1.4 AutoMate：99 个 assembly task（排除 `00755`）

每个保留 ID 的生产目标相同：**50 条新成功轨迹、暂按 1,000 transitions/task 规划**。原始 100 个 ID 都有两个 checkpoint 文件、抓取参数和 `disassemble_traj.json`；排除 `00755` 后，正式 99-task 子集的这些输入也是 99/99 完整。按训练 TensorBoard 的 `successes/iter ≥ 0.8` 门槛，正式子集中有 69 个 final-ready，18 个训练末尾回落，12 个从未达阈值。这里的“回落/低成功率”来源是各 task `summaries/events.out.tfevents.*` 中的训练标量，不是本轮重新 rollout 的实测成功率。训练时 `if_sbc=true`，而正式采集固定关闭 SBC、使用 hardest-init 和 deterministic specialist policy；训练标量只用于估算获得 50 条成功轨迹所需的 attempts，不作为并行速度指标。

低指标 20 task 的历史 checkpoint 已进一步逐一检查：训练根及 NAS 镜像中每个 task 只保留 `Assembly.pth` 和 `last_Assembly_ep_100_*.pth`；两者都记录 epoch 100，且 20/20 的模型 tensor 完全一致（最大绝对差为 0）。`Assembly.pth` 是 rl_games 按 rolling mean reward 覆盖的 best 文件，不是按 `successes/iter` 峰值保存；因此即使 TensorBoard 显示更早 epoch 的 success 更高，也没有对应权重可直接换用。本轮按用户决定保留这些模型，接受较低采集产率，以真实 hardest-distribution rollout 决定 attempts/成功率。

本轮历史抽查选择 final `successes/iter` 最低的 20 个模型做 fresh rollout。19/20 在关闭 SBC 的 hardest-init 下取得成功；代表性的低产率项为 `00110` 第 66 次成功、`00192` 第 34 次、`00863` 第 26 次、`00855` 第 17 次，其余 15 个均在 13 次以内成功。唯一例外 `00755`：deterministic hardest 110/110 失败，两个独立 stochastic hardest cohort 合计 200/200 失败，因此不能把它写成“成功率略低”。同一 checkpoint 在明确开启 SBC 后第 5 次取得成功，只证明 checkpoint 加载、资产、GPU 相机、GT guidance 与 pickle writer 可工作；该文件和一条完整 75-step hardest failure 都固定为 `diagnostic_only`。用户现已决定排除 `00755`，它不再进入正式 task list、配额、相机调试样本、ETA 或 manifest。完整历史证据仍保留在 `logs/joint-training-full-0821/automate_training_checkpoint_inventory_20260831.json`、`logs/joint-training-full-0821/automate_review_checkpoint_pair_audit_20260831.json`、`logs/joint-training-full-0821/videos/automate_low_success_review_20260831_v1/selection_provenance.json` 与 `logs/joint-training-full-0821/videos/automate_00755_full_failure_20260901_v1/`。

| 类别 | ID（已覆盖正式 99 task） | 每 ID 目标/step | 成功与 GPU/2D 结论 |
|---|---|---|---|
| final-ready，69 个 | `00004 00007 00014 00015 00021 00028 00030 00042 00074 00077 00078 00083 00103 00133 00138 00143 00163 00175 00186 00187 00190 00211 00213 00255 00256 00293 00296 00308 00319 00320 00329 00340 00345 00346 00417 00422 00426 00437 00470 00471 00480 00499 00514 00553 00559 00581 00597 00615 00649 00652 00659 00681 00686 00700 00726 00731 00768 00783 00831 00860 01036 01041 01053 01079 01092 01102 01125 01129 01136` | 50 / 约 1,000 | `00211` 四卡各 1/1 仅为 pipeline smoke；69 个 task 均需进入正式 attempt manifest |
| 曾达标但 final 回落，18 个 | `00016 00032 00062 00141 00210 00318 00360 00444 00446 00486 00506 00537 00614 00638 00648 01026 01029 01132` | 50 / 约 1,000 | NAS 未保留 success 峰值 epoch；先用现存 `Assembly.pth` 实测，只有无法产出时才进入续训/补 checkpoint 分支 |
| 从未达阈值，12 个 | `00081 00110 00117 00192 00271 00301 00388 00410 00703 00741 00855 00863` | 50 / 约 1,000 | 不再以 0.8 阈值阻塞；低产率只通过 `ceil(50/p_i)` 增加 attempt 预算 |

AutoMate 的 annotation 对正式 99 个 task 使用同一个几何合同：skill=`insert`，3D target 为 fixed asset insertion tip，2D 为保存标定下的同帧 front 投影。NAS 中正式子集 99/99 的 checkpoint、抓取参数和 disassembly JSON 完整，但通用 AutoMate USD/OBJ 原先依赖官方 Omniverse S3，236 无法直连；本轮由 r218 从同一官方 URL 下载低指标 review 20 task 的 200 个资产文件，并合入隔离的本地 Isaac 5.1 bundle。整个 bundle 共 227 个文件、约 97 MiB，runner 在接管 GPU 前执行 SHA-256 全量校验，不修改既有 `00211` 最小包。

AutoMate 的共享 front camera 调试流程已固化为 `scripts/data_collection/debug_automate_camera.py`，固定在 r218 本地使用已有 `/home/hy/anaconda3/envs/rr-isaaclab` 和 3060，不新建环境。相机调试样本固定为保留任务 `00410` 的 hardest-init 成功轨迹（43 transitions / 44 frames），不得使用已排除的 `00755`。用户于 2026-09-01 批准 v2：`AssemblyCameraCfg.front.offset.pos=(1.05,0,0.315)`，OpenGL quaternion 为 `(0.5434064844747748,0.4524482209388897,0.45244822093888976,0.5434064844747747)`，即相对原配置向任务靠近 15 cm、向上 8 cm，intrinsics 不变。配置已写入本地 IsaacLab 并精确同步到 236；两端文件 SHA256 均为 `52b3204b8f07346ca04aa2e3566200271e03228db703d7011f6b6f80b405124f`。不传 camera override 的回归视频 `logs/joint-training-full-0821/automate_front_camera_shared_applied_20260901_v1/00410-shared-v2-applied.mp4` 为 44/44 frame 点在画内、`scripted`、raw mode=`none`。正式 99-task 小批 gate 仍需验证 task 分布覆盖，但不得再把 v2 描述为未批准 candidate。

历史低指标 20 个视频位于 `logs/joint-training-full-0821/videos/automate_low_success_review_20260831_v1/mp4/`，固定三栏为 wrist raw｜front raw｜pickle→video 阶段离线绘制的 front GT point。保留的 19 个 hardest-init fresh pickle 均为 `annotation_source=scripted`、`image_annotation_mode=none`、success=true，所有 observation 的 front point 在 224×224 内，内容哈希唯一；MP4 19/19 可解码。`00755.mp4` 的 2 帧 SBC diagnostic 和完整 75-transition hardest failure 只保留为排除依据，均不得进入正式 manifest。

#### 多 env attempted-rollout 吞吐与生产选择

速度基准固定统计 **attempted rollouts/hour**；成功率不参与“是否加速”的判断，只在后续用 `attempts_i=ceil(50/p_i)` 把每 task 的 50 条成功配额换算成 attempt 数。长 horizon 任务 `00110` 的 attempted rollout 通常跑满 75 steps；collection wall time 包含 reset、policy/render、record、episode finalization，以及碰巧保存成功轨迹时的同步 xz 写入，不含 Isaac app 启动和进程退出。

| envs/GPU | Attempts | Collection wall time | Attempted rollouts/hour | 去掉首 batch 后 | 峰值显存 | 相对 1 env |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 172.722 s | 166.7 | — | 约 10.2 GiB | 1.00× |
| 2 | 8 | 94.505 s | 304.7 | — | 约 10.3 GiB | 1.83× |
| 4 | 32 | 214.243 s | 537.7 | 543.8/h | 10,442 MiB | 3.22× |
| 8 | 32 | 152.783 s | 754.0 | 765.7/h | 10,856 MiB | 4.52× |
| 16 | 32 | 115.989 s | **993.2** | **1009.5/h** | 11,646 MiB | **5.96×** |
| 32 | 64 | 219.439 s | 1049.9 | 1150.8/h | 13,656 MiB | 6.30× |

easy task `00211` 的固定 16-attempt 对照也得到 1/2/4/8 env 分别 472.5/632.1/902.3/1057.8 attempts/hour。64/64 保存文件通过独立 RR strict audit，payload hash 64/64 唯一；每个 env 设置内部 initial-state hash 16/16 唯一；3D→2D 误差为 0，camera geometry 未混入 env origin。由此生产默认值定为 **每张 4090 一个 Isaac 进程、每进程 16 env**。32 env 相对 16 env 的总增益仅 5.7%，只用于预计至少需要约 256 attempts 的长任务；不要在同一张 4090 上起多个 Isaac 进程。

当前多 env collector 是 diagnostic process-local override，**不能直接进入生产**。正式开跑前必须在 IsaacLab 的维护入口实现 `--num-envs`、regex parent cameras、per-env recorder、精确 50-save cap、bounded background validation/xz writer，并保持 `num_envs=1` 向后兼容；随后重跑 schema/provenance/camera/重复性 gate。同步 batch 多出的 success 只能记为 `excluded`，正式 selected 必须恰好 50。

本轮 236 runner 固化为 `logs/joint-training-full-0821/tools/run_automate_review_chain_then_reserve_236.sh`。CLI 显式区分 `POLICY_MODE=deterministic|stochastic` 与 `INIT_MODE=hardest|sbc`；正式数据固定 `deterministic + hardest`。collector 在独立进程组中运行，收到 `EXIT/HUP/INT/TERM` 时先终止精确 collector 进程组，再确认 full reservation ready 后释放 2 GiB handoff，避免 Isaac 子进程延迟退出造成显存占用随后跌落。

## 2. 生产前最小验证与需用户讨论的问题

每个 task 只做与造数据直接相关的最小 gate：

1. 每个 task 先生成至少 1 条本轮新 success；失败 task 记录 attempts、wall time 和明确错误后停止。FurnitureBench 使用 `gpu-snatcher/auto_data_preparation.sh` 的标准 4-env batch，因此首批实际得到 4/4/3 条。
2. 每条都验证 `annotation_source=scripted`、active frame 有有限且在 224×224 内的 2D point，并由同帧 GT 3D point/相机标定重投影一致。
3. 日志证明使用 GPU physics/render，没有 CPU fallback；保存 attempted rollouts、attempted rollouts/hour、transitions、字节数和 selected/excluded 数。成功率只用于配额所需 attempt 数，不作为并行速度指标。
4. 每个 environment 选样本生成三栏视频（wrist raw｜front raw｜annotation util front point）供用户目检。FurnitureBench v2 与 ManiSkill 11/11 task 已确认；AutoMate 历史低指标 20 模型完成抽查，19 个保留 task 有 hardest success，`00755` 失败并排除。该抽查完成了用户要求的验证范围，但不等价于正式 99 个 ID 逐 task fresh rollout。
5. ManiSkill 当前检查阶段每 task 最多选择 10 条新 strict success，正式 quota 仍为 100/task；AutoMate 正式 quota 已改为 50/task，其并行 gate 使用固定 attempted-rollout 数，不能为了测速度先造一批未批准的 production 数据。历史 pilot 只作为既有证据，不进入新 manifest。

LiftPegUpright 的相机项已经关闭：原始 40° 相机能完整覆盖修复后的 GT target；旧 1/20 是 FSM reset 状态泄漏，随后发现的横向跳点/随夹爪漂移则来自 target 每帧重算。两者均已在 annotation 实现中修复，不需要用户继续手调相机。最新 pickle 的 pick 帧 0–4 中，3D point 固定为 `[0.76677674, 0.06642199, 0.02500001]`，front 2D 固定为 `[156.04062, 196.89694]`；用户目检中“pick 没有点”来自标准 annotation util 的 2 px、50% alpha 标记与蓝色 peg 对比不足，而不是点缺失。另生成白色外环加红色十字的 review-only 视频用于目检，既不修改 raw pickle，也不改变正式 pickle→LMDB 标注定义。

当前需要与用户讨论、不会擅自改视觉定义的事项：

| 问题 | 已知事实 | 需要的选择 |
|---|---|---|
| r218→NAS durable write 卡住 | 512 MiB payload 已写入，但 `fdatasync` 超过 400 秒仍处于 NFS `D` 状态 | NAS 恢复后再过小批 durable-write gate；未恢复前不能正式开全量采集 |

## 3. 可用资源、NAS 和总耗时

### 3.1 当前资源

| 节点 | GPU | 当前状态 | 环境/用途 |
|---|---|---|---|
| r218 | RTX 3060 12 GB ×1 | FurnitureBench 4 env、`low` randomness；也承担本地检查 | 使用既有 `/home/hy/anaconda3/envs/rr`，本地 NVMe `/tmp` 约 202 GiB 可用 |
| 236 | RTX 4090 24 GB ×4（GPU0–3） | AutoMate 每卡一个 16-env Isaac 进程；释放的卡再跑 ManiSkill | AutoMate/ManiSkill 使用既有 NAS Conda；本地 SSD 仅约 68 GiB 可用 |
| NAS | `/mnt/nas/datasets_tmp` | 236 实测为 `10.71.106.245:/volume2/datasets_tmp` NFSv4.1，约 19 TiB 可用 | 作为 canonical raw、五套 LMDB、manifest 和日志存储；预留 2.0 TiB |

用户批准后，本轮使用的 campaign 根为：

```text
/mnt/nas/datasets_tmp/rr_joint_training_0821_scripted_prod_<launch-date>_v1/
  raw/{furniturebench,maniskill,automate}/
  processed/lmdb/<condition>/{furniturebench,maniskill,automate}.lmdb/
  manifests/
  logs/
  staging/
```

2026-09-02 复核时，r218 的 `/mnt/nas/datasets_tmp` 已是 `10.71.106.246:/volume2/datasets_tmp` NFSv4.1，但挂载为只读；236 的同一路径是 `10.71.106.245:/volume2/datasets_tmp` 且可写。因此 r218 直接读取和校验 canonical raw，在本地 NVMe 串行构建一个 LMDB，再通过 236 写入 NAS `.incoming`；r218 从只读挂载复核 SHA/schema/loader，最后由 236 原子改名并写 completion marker。禁止把只读挂载误判成可写路径，也不为此修改系统 mount。

GPU0–3 是共享资源。collector 暂停、退出或 task 切换时，必须先恢复占卡脚本并确认显存，或用一个串行 runner 无空窗衔接下一 task；不得让卡在两个进程之间空出来。只停止精确 tmux session/run ID，不按用户名杀进程。

### 3.2 NAS 瓶颈和落盘方式

此前卡住的是 volume1 `/mnt/nas/share` 的 durable flush，不能直接外推到本次 volume2 `/mnt/nas/datasets_tmp`。正式启动 gate 是从 r218（挂载恢复后）和 236 分别对 **精确 volume2 路径**做 disposable write→`fdatasync`→read→SHA-256 探针，并记录 durable MiB/s；探针不通过就不启动生产。

生产落盘使用有界流水线：

- 每个 collector 写本地 fresh staging，完成 strict audit 后由独立 uploader 传 NAS；不在 NAS 上做四路无界随机写。
- 236 四卡 aggregate staging 上限约 **16–20 GiB**，uploader backlog 接近上限时暂停对应 lane，并先恢复精确 GPU reservation。
- r218 优先使用约 202 GiB 空闲的 NVMe `/tmp`，不用约 179 GiB 空闲的旋转 `/data` 盘构建大 LMDB。
- 每个 closed shard 记录源文件列表、字节数和 manifest hash；上传后按 `du -sh`、manifest/SHA 对账，再释放 staging。
- volume2 的实际 durable throughput 记为 `B` MiB/s 后，传输 ETA 用 `bytes/B` 更新；在探针前不沿用旧 volume1 的 3–20 MB/s 区间。

### 3.3 采集 ETA

AutoMate 第 `i` 个 task 的计划公式为：

```text
attempts_i = ceil(50 / p_i)
total_attempts = sum(attempts_i)
pure_rollout_hours = total_attempts / sum(active GPU attempted-rollout throughput)
```

四张 4090 均按 16 env、每卡实测 993.2 attempts/hour 估算。下表是对 4,950 总配额的一次性 aggregate scenario 换算；正式执行仍按 task 分别取 ceiling，因此最终 attempts 可能略高：

| 用于换算配额的 aggregate success rate | 4,950 success 所需 attempts | 纯 4-GPU rollout 时间 |
|---:|---:|---:|
| 80% | 6,188 | 1.56 h |
| 50% | 9,900 | 2.49 h |
| 30% | 16,500 | 4.15 h |
| 10% | 49,500 | 12.46 h |

另加约 0.7–1.7 小时同步 strict validation/xz 写入，以及 99 个 task 的 app 启停/NFS tail 约 0.5–2 小时。当前 AutoMate 中心窗口为 **6–9 小时**；若 aggregate attempts 接近 10% 场景，则为 **15–20 小时**。这里始终以 attempted rollouts/hour 报告并行速度；`p_i` 只改变 attempts 总量。

| 工作 | 当前计算窗口 | 说明 |
|---|---:|---|
| FurnitureBench 600 success / r218 3060 | 6–12 h | 历史同规模 9 h 35 min；low randomness、4 env |
| AutoMate 4,950 success / 4×4090 | 6–9 h 中心；15–20 h 低产率情形 | 默认每卡 16 env；最长 predicted work first |
| ManiSkill 1,100 success | 3–8 h | AutoMate lane 提前结束后移 1–2 张 4090；Peg/Plug 单卡外推 0.87/3.38 h |
| 全 campaign 纯计算 | **约 8–16 h** | 3060 与四张 4090 重叠；极低 AutoMate 产率另计 |

调度采用 longest-processing-time-first：四个低产率 AutoMate task 先分散到四卡，每个 lane 完成后从剩余队列领取 predicted attempts 最大的 task；AutoMate 长尾收缩后，把 1–2 张卡切给 ManiSkill 难项。每个 lane 使用具名 tmux，记录 host、物理 GPU、Git commit、环境绝对路径、checkpoint hash、seed、env index、global attempt index、selected/excluded 和滚动 ETA。运行中每关闭一个 task 就用实际 attempts/time 更新剩余 ETA；不等整批结束才汇报。

### 3.4 五个 condition、15 个 LMDB 的构建与大小

最新 RR writer 使用 per-frame lossless zstd：`--frame-compression zstd --frame-compression-level 1`。三个来源的 source pickle 始终只有一份，且保持 `image_annotation_mode=none`；五个 condition 从同一份 raw 离线渲染，不复制或原地修改 pickle。condition 与物理图像 mode 固定如下：

| 训练 condition | 物理目录名 | `image_annotation_mode` | 默认物理 LMDB 数 | PPU96 层级 |
|---|---|---|---:|---|
| `rgbd+skill` | `rgbd-skill` | `none` | 3 | `/` 根盘 |
| `rgbd+gp+skill` | `rgbd-gp-skill` | `guidance-point` | 3 | `/` 根盘 |
| `rgbd+colored gp` | `rgbd-colored-gp` | `guidance-point-colored` | 3 | EPFS（当前挂载点 `/mnt/cpfs`） |
| `rgbd+grasp-part` | `rgbd-grasp-part` | `grasp-part` | 3 | EPFS（当前挂载点 `/mnt/cpfs`） |
| `rgbd+grasp-part colored` | `rgbd-grasp-part-colored` | `grasp-part-colored` | 3 | EPFS（当前挂载点 `/mnt/cpfs`） |

每个 condition 默认包含 FurnitureBench、ManiSkill、AutoMate 各一个 LMDB，即 **5×3=15 个物理 LMDB**。训练开关仍按 condition 配置：普通 GP 同时启用 point 与 skill；colored GP 用颜色编码 skill；两个 grasp-part condition 分别使用普通/colored grasp-part 输入。condition、物理 mode 和 provenance 必须一一对应，不能只改目录名。

2026-09-02 对本轮 production raw 做了 read-only 五模式 smoke：FurnitureBench lamp 348 帧（含 178 个 pick/place）、ManiSkill Peg 119 帧（含 50 个 pick）和 AutoMate 9 帧均能渲染全部五种 mode，`missing_grasp_geometry=0`，且 `none` 模式逐像素保持原图。另用一条 AutoMate production pickle 实际写出五个临时 zstd LMDB；五套 full-stats、condition/provenance/hash reader 均通过且 `data.mdb` hash 各不相同，临时 LMDB 随后删除。构建器会在每个 source 闭合后重跑 raw render gate；最终逐帧转换本身仍是全量 fail-closed 检查。

| 范围 | Episodes | Transitions | 当前 zstd 中心估计 |
|---|---:|---:|---:|
| 一个 condition：FurnitureBench | 600 | 约 328k | 约 85–96 GiB |
| 一个 condition：ManiSkill | 1,100 | 约 99k | 约 28–31 GiB |
| 一个 condition：AutoMate | 4,950 | 约 99k | 约 28–31 GiB |
| 一个 condition 合计 | 6,650 | 约 526k | **约 141–158 GiB** |
| 五个 condition 合计 | 33,250 个 condition-episode | 约 2.63M 个 condition-transition | **约 705–790 GiB** |

旧规划的 145–230 GiB/condition 作为保守上界，不直接乘成已发生占用；第一套完成后按实际 compressed bytes/frame 更新其余四套 ETA 和空间。NAS 同时保留 raw 350–450 GiB 与最终 LMDB 705–790 GiB，中心总量约 1.03–1.21 TiB；再考虑一个本地 `.building.lmdb`、一个 NAS `.incoming` 和安全余量，campaign 预留 **2.0 TiB**。

构建按 source-major 顺序进行：每个 source 闭合后只做一次 raw manifest/hash/provenance gate，再串行渲染五个 condition。r218 `/tmp` 一次只保留一个 `.building.lmdb`；该 LMDB 经 236 写入 NAS、通过 hash/schema/loader gate并生成 completion marker 后，才删除本地派生副本并构建下一个。默认不拆分；若 FurnitureBench 单 LMDB 的实测大小使 r218 不能保留至少 40 GiB 安全余量，则只对该 source 按 task 或 30–40 GiB balanced shard 拆分，并同步增加五个 condition 的 shard 数。

每个输出从 fresh `.building.lmdb` 开始，禁止 overwrite；依次验证精确 episode/task/count、source manifest/hash、`scripted`、source mode `none`、condition 对应的 output mode、zstd metadata、全量 stats、action/state/depth/order 等价和 loader/training-reader smoke。每个 condition/source 都单独记录 `data.mdb` bytes、SHA-256、本机/NAS/PPU audit 与 completion marker。

### 3.5 PPU96 容量与交付

正式执行前完整阅读 `reports/claude/ppu96_single_card_4way_zstd_runbook.md`。规划时 RR `main` 为 `bfc2ca8`，已包含该 runbook 及 native zstd writer/reader；上传前在 PPU96 用 clean worktree/clone materialize 审计后的最新 commit，不能覆盖远端 RR 现有 tracked/untracked 修改。

PPU96 `/root` 实测约 **660 GiB 可用**；EPFS 在当前实例上对应 `/mnt/cpfs`（与 `/mnt/workspace` 同一共享文件系统），实测约 **52.7 TiB 可用**。根盘只接收前两个 condition，中心占用约 **282–316 GiB**，完成后预计仍余 **344–378 GiB**，高于 runbook 的 100 GiB 安全线；因此按当前估计不需要用户先迁走已有根盘数据。上传器在每个 LMDB 传输前仍按“payload bytes + 100 GiB 根盘余量”动态 fail-closed，不能靠估计硬塞。后三个 condition 中心占用约 **423–474 GiB**，放 EPFS 并保留至少 1 TiB 动态余量。

目标布局固定为：

```text
/root/rr-local-data/processed/joint_training_0821/
  {rgbd-skill,rgbd-gp-skill}/{furniturebench,maniskill,automate}.lmdb/

/mnt/cpfs/users/hy/rr-local-data/processed/joint_training_0821/
  {rgbd-colored-gp,rgbd-grasp-part,rgbd-grasp-part-colored}/
    {furniturebench,maniskill,automate}.lmdb/
```

每个闭合 LMDB 由能直接读取 canonical NAS 的 r218 传到对应目标的 `.lmdb.incoming/`，不经过控制工作站中转，使用 resumable `rsync -aP --partial --append-verify`。目标端核对 bytes、`data.mdb` SHA-256、condition/mode/provenance、metadata validator 和 loader smoke 后原子改名并写 `.transfer-complete`。NAS 是 canonical copy；PPU96 根盘和 EPFS 是训练副本。

r218→PPU96 已测单流约 0.81 MiB/s，四流 aggregate 约 0.91 MiB/s，没有值得承担复杂度的并行增益，因此使用单流顺序断点续传。按 705–790 GiB 估计，纯传输约 **9.2–11.6 天**；含重连、逐 LMDB SHA/loader 校验与发布，操作窗口按 **10–13 天**。第一套实际 bytes 出来后立即更新该窗口；若落到旧保守上界，窗口会相应延长。

### 3.6 当前执行授权与 gate

用户已经明确批准开始数采并要求持续监督，因此 collector 和滚动审计继续运行；本次五-condition 修改不重启或扰动现有 raw 采集。LMDB 构建必须等待对应 source 的精确配额、全量 scripted/raw/projection audit、NAS manifest/hash 和上传 marker 全部闭合。上传必须等待每个 condition/source 的 NAS LMDB completion marker，随后逐个执行断点传输、目标空间 gate、SHA、metadata 与 loader 验收。最终完成条件从原来的三个 LMDB 改为 **15 个 LMDB 全部在 NAS 闭合，根盘两套/EPFS 三套全部发布并有 15 份 transfer receipt**。

## 4. 附录：此前实现细节的验证结论

### A. 环境与来源

- 当前主机按事实识别为 r218：hostname `lht-3060-12G`、用户 `hy`。主机标签不用于推断执行位置。
- AutoMate/Isaac Lab 与 ManiSkill 都按各自 README/文档从头建立了干净的 NAS Conda 环境；服务器不复用本地环境。FurnitureBench 是用户明确例外，两台机器分别使用各自现有 `rr`。
- NVIDIA Omniverse EULA 已由用户同意；Isaac Sim runner 显式设置 `OMNI_KIT_ACCEPT_EULA=YES`。
- 本轮及后续 annotation 固定为 scripted/geometry GT，不使用 VLM且不再逐次复核；仍须在命令和 pickle 顶层显式记录 provenance。

### B. checkpoint/solver 到 pickle 的流水线

- FurnitureBench：生产入口改为 `scripts/data_collection/collect_furniturebench.py` 与 `process_furniturebench_pickles_to_lmdb.py`。前者基于 gpu-snatcher 的标准 collect 参数，但强制显式 annotation source、全新 suffix 和 raw-image-only；后者才离线绘制图像标注。
- ManiSkill：6 个官方 PPO checkpoint 已下载并可加载；PPO online rollout→RR pickle 和 5 个 bundled MP solver→`RecordPickle` 均已接通。
- AutoMate：`generate_pickle.py` 已接通 specialist checkpoint、plug grasp、disassembly trajectory、GPU PhysX/RTX camera 和 RR pickle；正式 99/99 task 的 checkpoint/抓取参数/disassembly JSON 齐全，`00755` 明确排除。USD/OBJ 仍来自 NVIDIA 官方 Isaac 5.1 资产；历史低指标 review 20 task 已固化为独立本地校验 bundle，正式 99-task campaign 前按同一下载+SHA 流程补齐保留任务的全量资产。
- 三来源都写 8D delta action、224×224 wrist/front RGB-D、base-frame state/geometry、skill 和 guidance。生产合同要求所有 raw pickle 顶层均为 `image_annotation_mode=none`，图像 marker 只在 pickle→LMDB 时绘制；三来源 validator 均已 fail-closed。三个正式 collector 也要求显式 `--annotation-source scripted`；AutoMate 另在启动 Isaac Sim 前 fail-closed 排除 `00755`。

### C. GPU 与 2D point 已验证的范围

- AutoMate `00211` 在四张物理 4090 上分别 1/1 success，transitions 15/18/13/19，无 CUDA illegal access；4 条均为 scripted 且 strict pass。后续固定 attempted-rollout benchmark 覆盖 1/2/4/8 env，并以 `00110` 覆盖 1/2/4/8/16/32 env；64 条 `00211` multi-env 输出全量 strict/unique/camera audit 通过。该结果证明 multi-env 方向可行，但 process-local override 仍不能代表维护版 production collector 已通过。
- ManiSkill 11/11 task 的 SAPIEN RGB-D/schema/action/active-point render 与成功轨迹路径均已跑通。最新四个 MP 难项各选择 10 条，40/40 strict audit 通过；随后 Peg 3 mm 与 Plug 0.5 mm 插入补偿的生产吞吐 gate 分别得到 fresh strict 10/11 和 10/97，均 10/10 文件审计通过。按当前实测纯采集线性外推，Peg/Plug 100 条分别约 0.87/3.38 小时；这仍不包含 NAS durable upload。
- ManiSkill 2026-08-23 的 132 条、8,042 transitions 全量投影审计通过，但其中 Pick/Poke/PullCubeTool 发生在 reset guard 前，只保留为历史诊断。2026-08-31 reset 修复后 Lift、Pick 均 strict 20/20；Poke 在 task-specific 相机总后退 0.42 m 后新 20/20；PullCubeTool 最终采用相同的 0.42 m，相机定向复核覆盖上一轮全部三个失败 seed，其中两个实际 strict pass、另一个旧成功记录 259/259 重投影可见而本次仅 solver 失败。所有 recorded-vs-geometry projection 分歧均为 0。
- AutoMate 历史低指标 20 模型中，19 个保留 task 的 hardest-init fresh success 已通过 raw/GT/2D/video 审计；`00755` 的 SBC diagnostic 与 hardest failure 只保留为排除依据。历史 20 个成功/诊断视频位于 `logs/joint-training-full-0821/videos/automate_low_success_review_20260831_v1/mp4/`，完整 `00755` failure 位于 `logs/joint-training-full-0821/videos/automate_00755_full_failure_20260901_v1/mp4/`，两处的 `00755` 文件均不得进入正式数据。ManiSkill 11/11 task 各一条的统一目录为 `logs/joint-training-full-0821/videos/maniskill_all_tasks_20260831_v1/mp4/`；其中 Lift 已由 `lift_guidance_semantics_fix_20260901_v1/mp4/` 的 fresh 修复版取代。

### D. 历史错误 2D point 对训练的处理

`logs/vlm_dataset_null_audit_20260820/audit.json` 在 300 rollout、130,910 帧中发现 8,966 个 null 2D point（6.85%），主要集中在 trailing `screw` frame。结论不变：主 DiT 数据删除/修复错误 point、越界/null active frame 和伪装成 active skill 的尾帧，但保留 2D point 输入模态；无 point 只能作为单独受控消融，不能把旧坏标注当 baseline。

### E. 主要来源

- 配比原始讨论：`/home/hy/.codex/sessions/2026/08/21/rollout-2026-08-21T11-55-44-01a02275-dcb7-7191-83f8-48bfc936542f.jsonl` ordinal 10。
- 三来源 schema、真实小样本和 50/35/15 DDP smoke：`reports/cross_sim_pickle_alignment_20260821.md`。
- FurnitureBench 600 条、328,386 transitions、9 小时 35 分采集、容量和历史传输：`logs/data_assets_med_0801.md`。
- AutoMate 权威训练根：`/mnt/nas/share2/home/lq/logs/rl_games/Assembly/`；原始 100 task 的 TensorBoard final/max 来自各自 `summaries/events.out.tfevents.*`，正式数据只使用排除 `00755` 后的 99-task 子集。
- AutoMate 资产根：`/mnt/nas/share/home/lq/IsaacLab/AutoMate/<ID>/disassemble_traj.json`。
- ManiSkill 当前 MP 独立审计：`logs/joint-training-full-0821/maniskill_mp_gate_audit_manifest.json`。
- FurnitureBench 标准 gate runner/audit：`logs/joint-training-full-0821/furniturebench_standard_gate_r218_20260823/runtime/runner.log` 与 `all.audit.jsonl`。
- FurnitureBench metadata-only v2：`logs/joint-training-full-0821/furniturebench_metadata_v2_gate_r218_20260823/runtime/runner.log`、`all.audit.jsonl` 与 `lmdb_annotation.audit.log`。
- AutoMate multi-env 独立审计：`logs/automate-parallel-benchmark-20260901/audit.json`；长 horizon 原始 metrics 位于 `hard00110-v1/`、`v2-runs/hard00110_warm_gpu1_v2/` 和 `v2-runs/hard00110_env32_gpu1_v2/`。env4/8/16/32 metrics SHA-256 依次为 `1342c30d8e5ecc5aa2af079962b352d98d525c7407c7ed3bf6fb64e9c74fb352`、`4512eb235befdf0165007d7e8fd421bc1ee97ea4407abe612701698f72cc21c5`、`f371208579c599b1710b0eb6cc68238e5f993b8f66904bdeeff2252f880818bb`、`e20048392d3ae655ae8563128a5ee636f0ba3edda34df7fb605266935ff7e88a`。
- PPU96 压缩、容量与上传合同：`reports/claude/ppu96_single_card_4way_zstd_runbook.md`。
