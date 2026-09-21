# FurnitureBench checkpoint eval protocol：按训练 lineage 区分

更新时间：2026-09-21（Asia/Shanghai）

本文可整体复制给训练/评测调度 agent。目的不是把所有 checkpoint 强行套进同一个
图像合同，而是根据训练数据 lineage 选择正确的相机预处理。以下命令均指
FurnitureBench simulation checkpoint eval；real-sim checkpoint 的真机测试另走
`src.real.evaluate_policy`，不在本文命令范围内。

## 可直接复制给调度 agent 的执行说明

```text
请严格按 checkpoint 的训练 lineage 选择 eval protocol，不要只看 condition 名称或
checkpoint 内的 data.image_spatial_transform 字段。

共同规则：
1. 正确入口只能使用：python -m src.eval.evaluate_model。
2. FurnitureBench 任务为 one_leg / round_table / lamp；real-sim 混训只测 one_leg。
3. 标准预算为每个 checkpoint/task 36 rollouts；新协议先跑每格 12 rollouts validation，
   全部合同检查通过后才跑 36 rollouts formal。
4. 固定 seed=0、n_envs=12、max_rollout_steps=1000、action_type=pos。
5. Main 与 Joint 使用 randomness=low；real-sim full-frame 的 sim 测试使用 randomness=med。
6. 必须传 --annotate-skill --enable-annotation-verify --annotation-source scripted。
7. endpoint success 只能采用 physics reward/assembly completion；禁止重新叠加 scripted FSM
   gate。结果 JSON 必须有 success_criterion=physics_reward。
8. RGB-D depth 必须是 finite positive metres；结果 JSON 必须有
   eval_depth_contract=positive_meters。RGB 应为 not_applicable。
9. 禁止传 --guidance-point-on-image、--grasp-annotation-on-image、
   --grasp-part-annotate、--skill-on-image。不要保存带 marker 的 pickle。
10. policy condition flags 由 checkpoint config 自动解析；不要凭 condition 名手工拼 marker
    flags。每个结果必须检查 eval_annotation_config 与 checkpoint training_config 一致。
11. 固定 --tracking-metric-type position、--if-exists error、
    --eepose-frame robot-base，并为每格写唯一 --task-summary-out。
12. 运行前检查 GPU compute PID 和显式 --gpu 绑定进程；一次只允许一个 FurnitureBench
    evaluator 占用目标 GPU。使用具名 tmux，并记录 host/GPU/code/checkpoint SHA/完整命令。

四类 lineage：

A. Main 3-seed 登记实验
- 它不是完全统一的三 seed 数据 lineage。
- 2026090701/2026090702 supplemental checkpoint：训练 LMDB 是 300 episodes / 125307
  frames，原始 240x320 两路 RGB-D 同步中心裁剪 [8:232,48:272] 到 224x224，无插值，
  depth 为正米。eval 必须显式 --wrist-image-transform center-crop-224，front 在 policy
  eval transform 中同样 center crop 到 224。
- 2026091701 新 RGB/RGB-D checkpoint 也属于 canonical center-crop lineage；旧的
  legacy-resize diagnostic 结果无效，重测必须用 center-crop-224。
- Main 表中沿用的 0610 historical seed 属于旧协议，必须按 D 节 legacy 命令运行；不能
  对它使用 center-crop-224。
- 旧 true-firefly-8 RGB 和 clear-water-12 RGB-D 的 loader 只登记 96 episodes / 49083
  samples，已判定为错误 lineage，从 Main 正式统计排除，只能做历史诊断。

B. Joint train / Joint90
- 数据源固定为 ModelScope huyue233/furniture-bench-joint-train-data-original，revision
  723b63b459a1e9fb804989d0c37cdfb1b0f9dbca。
- FurnitureBench base 为 600 episodes / 251980 frames；240x320 两路 RGB-D 同步中心裁剪
  [8:232,48:272] 到 224x224，无插值，depth 为正米。
- Joint checkpoint 内可能保存 data.image_spatial_transform=legacy-224。这是因为 LMDB
  已经是 224x224，训练时 legacy resize 实际为 identity；它不能用于 raw simulator eval。
  eval 必须强制 --wrist-image-transform center-crop-224。
- sim front camera 使用 original preset。

C. Real-sim full-frame cotrain
- real40 与 sim400 都保存完整 240x320 RGB-D，data.image_spatial_transform=none；禁止
  resize/crop 到 224。
- eval 必须传 --wrist-image-transform checkpoint，并在结果中确认 resolved transform=none、
  wrist input/policy size 都是 [240,320]。若 checkpoint 没有保存 none，立即 fail closed。
- sim400 使用 locked_20260910 front camera，eval 必须传
  --sim-front-camera-preset locked_20260910，并确认运行代码实际实现该 preset。
- 代码门禁：FurnitureBench 至少包含 7d570476（dc7f435 的 robot-base 改动与
  f0d9cde 的 camera preset 合流），同时必须包含 positive-depth patch。只支持
  dc7f435+positive-depth、但会静默忽略 camera preset 的 Joint/base bundle 禁止用于
  real-sim formal eval。
- 该命令是 sim one_leg/med 测试，不是真机 rollout。

D. 最早 0610 Main experiment 单 seed
- 这是历史 raw/legacy 图像合同：rollout 先得到 240x320，wrist 使用 legacy resize 到
  224x224；front 使用旧 eval center crop 到 224x224。
- 必须显式 --wrist-image-transform legacy-resize，不能改成 center-crop-224。
- RGB-D 仍必须用 positive-meters depth；旧 signed-depth eval 不能复用。
- original front camera preset，randomness=low。
- 这类结果只能作为 historical/provenance 或旧 condition seed；不能与 canonical
  supplemental/Joint/full-frame checkpoint 假装成同一输入合同。

任何结果只有同时通过以下检查才能入表：checkpoint path/SHA 正确；n_envs=12；rollout
数正确；seed=0；randomness 正确；scripted provenance；physics_reward success；transform
requested/resolved 与本说明一致；RGB-D positive metres；tracking_error.complete=true；
完整 eval_command 已登记；无上述四种 forbidden marker flag。
```

## 统一运行环境模板

在 r218 使用已配置环境：

```bash
export RR_ROOT=/data/hy/robust-rearrangement
export RR_PYTHON=/home/hy/anaconda3/envs/rr/bin/python
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/hy/anaconda3/envs/rr/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RR_ROOT:$RR_ROOT/furniture-bench"
cd "$RR_ROOT"
```

在 `base` 使用隔离的 Joint/Main evaluator bundle：

```bash
export RR_ROOT=/home/huyue/tmp/rr-joint90-corrected-eval-0921/code
export RR_ASSET_ROOT=/home/huyue/projects/robust-rearrangement-custom
export RR_PYTHON=/home/huyue/miniconda3/envs/rr/bin/python
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/huyue/miniconda3/envs/rr/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RR_ROOT:$RR_ROOT/furniture-bench:$RR_ASSET_ROOT/isaacgym/python"
cd "$RR_ROOT"
```

上述 base bundle 适合 original-camera 的 Main/Joint eval，不满足 real-sim
`locked_20260910` 门禁。Real-sim 必须另建包含 `7d570476 + positive-depth patch` 的冻结
bundle，不能在命令层假设 preset 已生效。

## A. Main 3-seed 实验

### A.1 两个合格 supplemental seed

checkpoint 模板：

```text
/data/hy/robust-rearrangement/checkpoints/main_supplement_0906/seed2026090701/<condition>/actor_chkpt_last.pt
/data/hy/robust-rearrangement/checkpoints/main_supplement_0906/seed2026090702/<condition>/actor_chkpt_last.pt
```

`<condition>` 为：`rgb`、`rgbd`、`rgbd_colored_gp`、`rgbd_gp_skill`、
`rgbd_skill`、`rgbd_grasp_part`、`rgbd_grasp_part_colored`。这批 supplemental 没有
同-lineage 的纯 `rgbd_gp`。

单格 12-rollout validation：

```bash
CHECKPOINT=/data/hy/robust-rearrangement/checkpoints/main_supplement_0906/seed2026090701/rgbd/actor_chkpt_last.pt
TASK=one_leg
SUMMARY=/data/hy/robust-rearrangement/logs/main-eval-corrected/validation12/seed2026090701/rgbd/${TASK}.json
mkdir -p "$(dirname "$SUMMARY")"

"$RR_PYTHON" -m src.eval.evaluate_model \
  --wt-path "$CHECKPOINT" \
  --gpu 0 \
  --task "$TASK" \
  --n-envs 12 \
  --n-rollouts 12 \
  --seed 0 \
  --randomness low \
  --max-rollout-steps 1000 \
  --action-type pos \
  --observation-space image \
  --annotate-skill \
  --enable-annotation-verify \
  --annotation-source scripted \
  --sim-front-camera-preset original \
  --wrist-image-transform center-crop-224 \
  --eepose-frame robot-base \
  --tracking-metric-type position \
  --if-exists error \
  --task-summary-out "$SUMMARY"
```

formal 使用完全相同命令，只改：

```bash
--n-rollouts 36
```

并把输出路径从 `validation12` 改为 `formal36`。三个任务分别运行
`one_leg round_table lamp`。

### A.2 2026091701 新 RGB/RGB-D checkpoint

```text
/data/hy/robust-rearrangement/checkpoints/main_retrain_0917/seed2026091701/rgb/actor_chkpt_last.pt
/data/hy/robust-rearrangement/checkpoints/main_retrain_0917/seed2026091701/rgbd/actor_chkpt_last.pt
```

命令与 A.1 完全一致，仍然必须是 `--wrist-image-transform center-crop-224`。禁止复用
曾产生 RGB `42/108`、RGB-D `0/108` 的 legacy-resize diagnostic 结果。

### A.3 “3-seed”统计的边界

Main 报告中 conditional 方法的第三个 seed 是 0610 historical checkpoint，按 D 节命令
运行。它和两个 supplemental seed 的数据版本、padding 行为及图像合同不同。因此：

- 不能写一个循环对三者统一传 `center-crop-224`；
- 不能把三者的差异解释成纯训练 seed 方差；
- RGB/RGB-D 的旧 0610 seed 已从正式表排除；
- `rgbd_gp` 的三个 seed 都是历史登记，没有 supplemental 同-lineage 对照。

## B. Joint train / Joint90

当前七个完成 checkpoint 的模板：

```text
/data/hy/robust-rearrangement/checkpoints/joint90_0917/seed2026091801/<condition>/actor_chkpt_last.pt
```

`<condition>` 为：`rgb`、`rgbd`、`rgbd_colored_gp`、`rgbd_gp`、
`rgbd_gp_skill`、`rgbd_grasp_part_colored`、`rgbd_skill`。

单格命令：

```bash
CHECKPOINT=/data/hy/robust-rearrangement/checkpoints/joint90_0917/seed2026091801/rgbd/actor_chkpt_last.pt
TASK=one_leg
SUMMARY=/data/hy/robust-rearrangement/logs/joint90-fb-corrected-eval-0921/r218/validation12/results/rgbd/${TASK}.json
mkdir -p "$(dirname "$SUMMARY")"

"$RR_PYTHON" -m src.eval.evaluate_model \
  --wt-path "$CHECKPOINT" \
  --gpu 0 \
  --task "$TASK" \
  --n-envs 12 \
  --n-rollouts 12 \
  --seed 0 \
  --randomness low \
  --max-rollout-steps 1000 \
  --action-type pos \
  --observation-space image \
  --annotate-skill \
  --enable-annotation-verify \
  --annotation-source scripted \
  --sim-front-camera-preset original \
  --wrist-image-transform center-crop-224 \
  --eepose-frame robot-base \
  --tracking-metric-type position \
  --if-exists error \
  --task-summary-out "$SUMMARY"
```

formal 改为 `--n-rollouts 36` 和 `formal36` 输出目录。先完成全部
`7 conditions × 3 tasks × 12`，全部通过合同 gate 后，再完成
`7 × 3 × 36`。

当前已验证的 r218 完整矩阵入口为：

```bash
tmux new-session -d -s rr_joint90_corrected_eval_r218_0921 \
  "cd /data/hy/robust-rearrangement && bash logs/joint90-fb-corrected-eval-0921/tools/run_r218_eval_matrix.sh"
```

它会先按 `CHECKPOINTS.sha256` 校验七个 checkpoint，逐格验证 validation12 JSON，并仅在
21/21 全部通过后进入 formal36。base 迁移曾因 Tailscale DERP 链路过慢而取消，partial 文件
保留但不能视为完整 staging；若以后改在 base 执行，必须先续传、核对七个 checkpoint SHA、
同步冻结代码和资产，再使用对应的 `run_base_eval_matrix.sh`，禁止直接复制上面的 r218 路径。

## C. Real-sim full-frame cotrain

正式示例 checkpoint（real40+sim400）：

```text
/mnt/nas/datasets_tmp/rr_real_sim_fullframe_cotrain_0907/training/checkpoints/real40_sim400/rr_fullframe0907_real40_sim400_b256_seed2026090711/rr_fullframe0907_real40_sim400_b256_seed2026090711/actor_chkpt_last.pt
```

在满足 `7d570476 + positive-depth patch` 的冻结代码 bundle 中运行：

```bash
CHECKPOINT=/mnt/nas/datasets_tmp/rr_real_sim_fullframe_cotrain_0907/training/checkpoints/real40_sim400/rr_fullframe0907_real40_sim400_b256_seed2026090711/rr_fullframe0907_real40_sim400_b256_seed2026090711/actor_chkpt_last.pt
TASK=one_leg
SUMMARY=/data/hy/robust-rearrangement/logs/real-sim-corrected-eval/validation12/real40_sim400/${TASK}.json
mkdir -p "$(dirname "$SUMMARY")"

"$RR_PYTHON" -m src.eval.evaluate_model \
  --wt-path "$CHECKPOINT" \
  --gpu 0 \
  --task "$TASK" \
  --n-envs 12 \
  --n-rollouts 12 \
  --seed 0 \
  --randomness med \
  --max-rollout-steps 1000 \
  --action-type pos \
  --observation-space image \
  --annotate-skill \
  --enable-annotation-verify \
  --annotation-source scripted \
  --sim-front-camera-preset locked_20260910 \
  --wrist-image-transform checkpoint \
  --eepose-frame robot-base \
  --tracking-metric-type position \
  --if-exists error \
  --task-summary-out "$SUMMARY"
```

formal 改为 `--n-rollouts 36`。接受结果前必须额外检查：

```text
wrist_image_transform_requested == checkpoint
wrist_image_transform == none
wrist_image_input_size == [240, 320]
wrist_image_policy_size == [240, 320]
eval_depth_contract == positive_meters
```

同时在 run metadata 记录 frozen root commit、FurnitureBench commit、positive-depth patch
hash 和实际 front-camera position/target/FOV。仅仅在命令中出现
`--sim-front-camera-preset locked_20260910` 不足以证明 preset 生效。

## D. 最早的 Main experiment 单 seed checkpoint

### D.1 本地历史 RGB-D checkpoint

本地已登记并校验的例子：

```text
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_seed1363100997.pt
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_colored_gp_seed4073560603.pt
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_gp_skill_seed3903998646.pt
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_skill_seed962224487.pt
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_grasp_part_seed1574954007.pt
/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_grasp_part_colored_seed3651007064.pt
```

0610 RGB/RGB-D 原始 run 的 NAS 例子：

```text
/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-07-00.318495/models/clear-water-12_2026-06-13_13-07-27.238477/actor_chkpt_latest_3000.pt
/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-24-32.560941/models/true-firefly-8_2026-06-13_13-26-01.422600/actor_chkpt_latest_3000.pt
```

### D.2 正确历史单-seed复测命令

```bash
CHECKPOINT=/data/hy/robust-rearrangement/checkpoints/main_depthfix_old_0906/rgbd_seed1363100997.pt
TASK=one_leg
SUMMARY=/data/hy/robust-rearrangement/logs/main0610-historical-corrected-eval/validation12/rgbd/${TASK}.json
mkdir -p "$(dirname "$SUMMARY")"

"$RR_PYTHON" -m src.eval.evaluate_model \
  --wt-path "$CHECKPOINT" \
  --gpu 0 \
  --task "$TASK" \
  --n-envs 12 \
  --n-rollouts 12 \
  --seed 0 \
  --randomness low \
  --max-rollout-steps 1000 \
  --action-type pos \
  --observation-space image \
  --annotate-skill \
  --enable-annotation-verify \
  --annotation-source scripted \
  --sim-front-camera-preset original \
  --wrist-image-transform legacy-resize \
  --eepose-frame robot-base \
  --tracking-metric-type position \
  --if-exists error \
  --task-summary-out "$SUMMARY"
```

formal 改为 `--n-rollouts 36`。若 checkpoint 是 RGB，结果的 depth contract 应为
`not_applicable`；若是 RGB-D，则必须为 `positive_meters`。

## 四类协议速查

| Lineage | 训练/策略输入 | Wrist eval 参数 | Front camera | Eval randomness | 可否与 canonical Main/Joint 直接合并 |
|---|---|---|---|---|---|
| Main supplemental `2026090701/02`；Main retrain `2026091701` | 224×224 center crop，positive depth | `center-crop-224` | `original`，policy center crop | `low` | 同合同内可以；不同数据版本仍需登记 |
| Joint90 | 224×224 center crop，positive depth | 强制 `center-crop-224`，忽略误导性的 checkpoint legacy 默认 | `original`，policy center crop | `low` | 只在 Joint 内比较；不能冒充 Main |
| Real-sim full-frame | 240×320，`spatial_transform=none`，positive depth | `checkpoint`，且必须解析为 `none` | `locked_20260910` | `med` | 不可以；相机、分辨率、任务和数据域都不同 |
| Main 0610 historical single seed | 旧 240×320→wrist resize 224；front crop 224 | `legacy-resize` | `original` | `low` | 不可以；仅历史/混合-lineage登记 |

## 结果进入报告前的最小审计

每个 summary JSON 至少检查：

```text
checkpoint_path              == 本次已校验 SHA 的 checkpoint
task                         == 目标 task
n_envs                       == 12
n_rollouts                   == 12 或 36
simulator_seed               == 0
annotation_source            == scripted
success_criterion            == physics_reward
tracking_error.complete      == true
wrist_image_transform        == 当前 lineage 要求
eval_depth_contract          == positive_meters 或 RGB 的 not_applicable
eval_command                 == 本次完整命令
```

此外应从 cell log 检查 `Policy pre-transform wrist observation`。Canonical Main/Joint 应为
240×320 后进入 224 crop；real-sim policy 应保持 240×320。任何合同字段缺失、preset 未证实
生效、checkpoint SHA 不符或出现第二个 evaluator，都必须停止，不得把结果写入正式报告。
