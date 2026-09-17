# AutoMate 多任务对照与 Joint-condition 扩展评测

更新时间：2026-09-17（Asia/Shanghai）

## 跨环境主对比：AutoMate 扩展能力与 FurnitureBench 保持率

| Condition | AutoMate ID SR（99 train tasks） | AutoMate OOD SR（`00755`） | FB overall（joint formal） | FB Δ vs main |
|---|---:|---:|---:|---:|
| `rgb` | 739/1188 = 62.2% | 3/12 = 25.0% | 50/108 = 46.3% | −3.70 pp |
| `rgbd` | 761/1188 = 64.1% | 9/12 = 75.0% | 67/108 = 62.0% | **+10.65 pp** |
| `rgbd_colored_gp` | **773/1188 = 65.1%** | **10/12 = 83.3%** | 58/108 = 53.7% | −3.70 pp |
| `rgbd_gp` | 754/1188 = 63.5% | **10/12 = 83.3%** | 50/108 = 46.3% | −6.17 pp |
| `rgbd_gp_skill` | 703/1188 = 59.2% | 8/12 = 66.7% | 57/108 = 52.8% | −10.49 pp |
| `rgbd_skill` | 750/1188 = 63.1% | 8/12 = 66.7% | **65/108 = 60.2%** | **+6.48 pp** |
| `rgbd_grasp_part` | 748/1188 = 63.0% | 9/12 = 75.0% | 47/108 = 43.5% | −2.78 pp |
| `rgbd_grasp_part_colored` | 755/1188 = 63.6% | 8/12 = 66.7% | 44/108 = 40.7% | −6.79 pp |

`FB Δ = joint formal FB overall − updated main-experiment overall`。Main 的 RGB/RGB-D 已丢弃错误的原始 checkpoint，暂按两个有效 supplemental train seed 计算：RGB `50.00±1.31%`，RGB-D `51.39±0.65%`。因此旧表中 RGB 的正增益消失为 `−3.70 pp`，RGB-D 的增益收敛为 `+10.65 pp`。

### Motivation

AutoMate 的核心问题是 assembly skill 能否随任务数量扩展，而 FurnitureBench 检验在加入大规模 AutoMate 数据后，原有长时程家具装配能力能否保留。我们因此同时报告 AutoMate 的 99-task ID、单个 held-out assembly 的 OOD 成功率，以及相同 condition 在 FurnitureBench 上相对 main experiment 的变化，而不只报告某一个 simulator 的绝对成功率。

### Experimental setting

我们按照 AutoMate 的 specialist 训练方法分别训练 99 个 assembly specialist，并用其成功轨迹构建 joint visual imitation-learning data。单个 joint policy 同时学习 FurnitureBench 与 AutoMate 数据，输入为视觉观测及对应 condition。AutoMate 使用 default ×1、fixed-hardest、SBC-off，每 task 12 rollout；ID 汇总 99 个训练 assembly，OOD 为训练中未见的 `00755`。FurnitureBench 使用 formal joint checkpoint，每 task 36 rollout，并与更新后的 main baseline 比较。

AutoMate 100-task 面板主要使用 paired checkpoint，而 FurnitureBench 36-rollout 面板使用 formal checkpoint；`rgbd_skill` 的 AutoMate 行也是 formal。所以上表是 condition-level 的跨环境证据，不是逐行同一 checkpoint 的严格 head-to-head。要得到完全 matched 的跨环境表，需要补跑其余 formal checkpoint 的 AutoMate 100-task 面板。

#### SR 口径决策与已知限制

当前所有 AutoMate 主表、逐任务矩阵和后续 90/10 评测继续使用 AutoMate/IsaacLab 官方 success predicate，以保持与 AutoMate 原文及已有结果的同口径可比性；本报告暂不修改任何已登记 SR。该判据要求 peg root 位于 socket origin 与 `socket origin + disassembly_dist` 之间，且平均 keypoint distance 小于全局 `15 mm` 阈值；它不检查 peg 是否被孔腔包含、接触关系、实际插入深度或成功是否持续稳定。

`00320` 的 rollout 视频暴露了该限制：其 `disassembly_dist=15 mm`，socket mesh 顶面为 `13.746 mm`，因此 peg 底端仍可在 socket 顶面上方约 `1.25 mm` 时满足高度条件。该任务的目标孔径约 `9.33 mm`、peg 直径约 `8.36 mm`、单侧径向间隙仅约 `0.49 mm`，而 12 个 `rgbd_colored_gp` 重放的最小 keypoint distance 均只到 `14.14–15.00 mm`。这些 rollout 在官方口径下仍记为 `12/12`，但不应额外解读为经视觉确认的 12 次稳定物理插入。后续如需研究物理完成质量，另行增加 post-step 观测、孔腔包含/插入深度与连续多步稳定性判据，并将其作为独立的 strict physical-insertion SR，不回溯覆盖官方 SR。

### Analysis

Joint policy 在 99 个 AutoMate 训练任务上达到 `59.2–65.1%`，说明视觉 BC 在 100-task 规模仍保持广泛能力；`rgbd_colored_gp` 的 ID 成功率最高，为 `65.1%`。在 held-out `00755` 上，除 RGB 外的七个 RGB-D condition 达到 `66.7–83.3%`，显示出未在 joint data 中训练该 assembly 时的零样本 OOD 行为，但该结论目前只有一个 held-out task、每个 condition 12 rollout。

FurnitureBench 的影响具有明显 condition dependence：RGB-D 和 skill 分别提高 `+10.65` 与 `+6.48 pp`，其余六个 condition 下降 `2.78–10.49 pp`；八个 condition 的平均变化为 `−2.06 pp`。因此增加 AutoMate 数据没有带来一致的 FB 提升，也没有使 FB 能力整体失效。更准确的表述是：joint training 获得了大规模 AutoMate 能力，同时以较小的平均 FB 损失保留原有任务能力，但不同接口受到的正迁移或干扰不同。

### Conclusion

当前方案在 FurnitureBench 与 AutoMate 的联合训练中实现了更大的任务规模：FB 能力总体得以保留，并在 99 个 AutoMate ID tasks 上获得约六成以上成功率；TAGPoint (`rgbd_colored_gp`) 在 AutoMate ID 与单个 OOD task 上均取得本组最高值。与 AutoMate 原文的 20-task 结果比较时，应突出任务规模、视觉输入和训练资源差异；不能把当前单 checkpoint、12 rollout/task 的结果与原文 5-seed best-checkpoint、5×1000 tests/task 的 `80.42±20.93%` 当作同协议排名。

## 与 AutoMate 原文的主结果对照

下表把**论文原始登记值**和本工作已完成的 default ×1 面板放到同一张主表中。数值并不构成 head-to-head：论文使用 geometry generalist、BC/DAgger/RL/SBC、5 个 seed 中挑选最佳 checkpoint、每 assembly 5 × 1000 tests；本工作是单个 visual joint-BC checkpoint、scripted annotation、fixed-maximum/SBC-off、12 tests/task。`Paper-20` 列仅是与论文 20 个 assembly ID 的交集；`100-task` 是本工作的扩展面板。

| Source / method | Reset and evaluation | Paper-20 SR | 100-task SR |
|---|---|---:|---:|
| AutoMate paper: BC | paper protocol | 28.84% ± 15.23% | — |
| AutoMate paper: BC + DAgger | paper protocol | 31.06% ± 15.06% | — |
| AutoMate paper: BC + DAgger + RL | paper protocol | 52.85% ± 14.01% | — |
| AutoMate paper: BC + DAgger + RL + SBC | paper protocol | **80.42% ± 20.93%** | — |
| This work: `paired_rgb` | default ×1, fixed-hardest, SBC-off | 132/240 = 55.0% | 742/1200 = 61.8% |
| This work: `paired_rgbd` | default ×1, fixed-hardest, SBC-off | 132/240 = 55.0% | 770/1200 = 64.2% |
| This work: `paired_rgbd_colored_gp` | default ×1, fixed-hardest, SBC-off | **141/240 = 58.8%** | **783/1200 = 65.2%** |
| This work: `paired_rgbd_gp` | default ×1, fixed-hardest, SBC-off | 132/240 = 55.0% | 764/1200 = 63.7% |
| This work: `paired_rgbd_gp_skill` | default ×1, fixed-hardest, SBC-off | 124/240 = 51.7% | 711/1200 = 59.2% |
| This work: `paired_rgbd_grasp_part` | default ×1, fixed-hardest, SBC-off | 128/240 = 53.3% | 757/1200 = 63.1% |
| This work: `paired_rgbd_grasp_part_colored` | default ×1, fixed-hardest, SBC-off | 132/240 = 55.0% | 763/1200 = 63.6% |
| This work: `formal_rgbd_skill` | default ×1, fixed-hardest, SBC-off | 122/240 = 50.8% | 758/1200 = 63.2% |

论文数值来自 [AutoMate paper](https://bingjietang718.github.io/pdfs/rss2024.pdf)。原始可机读数据见 [TSV](../logs/automate-paper-hardest-eval-0914/paper_default1_all_conditions/condition_task_success_rates.tsv)。该 x1 面板审计结果为 `all_checks_pass=true`。

## 100-task success-rate matrix

以下矩阵直接内嵌于本主报告；每格为 `successes/12 (SR)`。 `Specialist final train SR` 来自各 specialist 的 TensorBoard `successes/iter` 最终值。

Protocol: 12 rollouts/task; `automate_paper_default1_hardest_curriculum`; fixed-hardest curriculum; SBC disabled; scripted annotation; AutoMate/IsaacLab official success predicate. 表中 SR 不是额外的 strict physical-insertion SR，判据限制见上文。

| Assembly | Group | Paper-20 | Specialist final train SR† | `paired_rgb` | `paired_rgbd` | `paired_rgbd_colored_gp` | `paired_rgbd_gp` | `paired_rgbd_gp_skill` | `paired_rgbd_grasp_part` | `paired_rgbd_grasp_part_colored` | `formal_rgbd_skill` |
|---|---|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00004` | `train/final_at_least_0.8` | no | 86.4% | 9/12 (75.0%) | 5/12 (41.7%) | 6/12 (50.0%) | 5/12 (41.7%) | 4/12 (33.3%) | 6/12 (50.0%) | 6/12 (50.0%) | 4/12 (33.3%) |
| `00007` | `train/final_at_least_0.8` | no | 99.7% | 11/12 (91.7%) | 11/12 (91.7%) | 9/12 (75.0%) | 8/12 (66.7%) | 7/12 (58.3%) | 8/12 (66.7%) | 10/12 (83.3%) | 11/12 (91.7%) |
| `00014` | `train/final_at_least_0.8` | no | 98.8% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00015` | `train/final_at_least_0.8` | yes | 91.9% | 7/12 (58.3%) | 7/12 (58.3%) | 5/12 (41.7%) | 6/12 (50.0%) | 4/12 (33.3%) | 6/12 (50.0%) | 9/12 (75.0%) | 4/12 (33.3%) |
| `00016` | `train/declined_after_reaching_0.8` | no | 76.1% | 8/12 (66.7%) | 9/12 (75.0%) | 7/12 (58.3%) | 9/12 (75.0%) | 8/12 (66.7%) | 8/12 (66.7%) | 12/12 (100.0%) | 9/12 (75.0%) |
| `00021` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00028` | `train/final_at_least_0.8` | yes | 80.6% | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 3/12 (25.0%) |
| `00030` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00032` | `train/declined_after_reaching_0.8` | no | 72.2% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00042` | `train/final_at_least_0.8` | no | 97.0% | 11/12 (91.7%) | 12/12 (100.0%) | 10/12 (83.3%) | 11/12 (91.7%) | 10/12 (83.3%) | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `00062` | `train/declined_after_reaching_0.8` | no | 79.5% | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 2/12 (16.7%) | 0/12 (0.0%) | 3/12 (25.0%) | 1/12 (8.3%) | 2/12 (16.7%) |
| `00074` | `train/final_at_least_0.8` | no | 89.6% | 7/12 (58.3%) | 12/12 (100.0%) | 7/12 (58.3%) | 7/12 (58.3%) | 8/12 (66.7%) | 10/12 (83.3%) | 8/12 (66.7%) | 9/12 (75.0%) |
| `00077` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00078` | `train/final_at_least_0.8` | no | 91.4% | 6/12 (50.0%) | 5/12 (41.7%) | 7/12 (58.3%) | 8/12 (66.7%) | 7/12 (58.3%) | 7/12 (58.3%) | 7/12 (58.3%) | 8/12 (66.7%) |
| `00081` | `train/never_reached_0.8` | yes | 63.6% | 3/12 (25.0%) | 3/12 (25.0%) | 5/12 (41.7%) | 4/12 (33.3%) | 4/12 (33.3%) | 1/12 (8.3%) | 2/12 (16.7%) | 1/12 (8.3%) |
| `00083` | `train/final_at_least_0.8` | no | 99.6% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00103` | `train/final_at_least_0.8` | no | 96.5% | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00110` | `train/never_reached_0.8` | yes | 73.3% | 1/12 (8.3%) | 1/12 (8.3%) | 2/12 (16.7%) | 1/12 (8.3%) | 0/12 (0.0%) | 3/12 (25.0%) | 1/12 (8.3%) | 0/12 (0.0%) |
| `00117` | `train/never_reached_0.8` | no | 66.9% | 4/12 (33.3%) | 3/12 (25.0%) | 4/12 (33.3%) | 3/12 (25.0%) | 2/12 (16.7%) | 3/12 (25.0%) | 5/12 (41.7%) | 3/12 (25.0%) |
| `00133` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 9/12 (75.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `00138` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00141` | `train/declined_after_reaching_0.8` | no | 75.2% | 5/12 (41.7%) | 7/12 (58.3%) | 6/12 (50.0%) | 7/12 (58.3%) | 7/12 (58.3%) | 8/12 (66.7%) | 7/12 (58.3%) | 6/12 (50.0%) |
| `00143` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00163` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) |
| `00175` | `train/final_at_least_0.8` | no | 99.3% | 6/12 (50.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 11/12 (91.7%) | 10/12 (83.3%) | 10/12 (83.3%) | 10/12 (83.3%) | 12/12 (100.0%) |
| `00186` | `train/final_at_least_0.8` | no | 96.5% | 6/12 (50.0%) | 6/12 (50.0%) | 9/12 (75.0%) | 4/12 (33.3%) | 6/12 (50.0%) | 5/12 (41.7%) | 5/12 (41.7%) | 5/12 (41.7%) |
| `00187` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `00190` | `train/final_at_least_0.8` | no | 98.9% | 3/12 (25.0%) | 4/12 (33.3%) | 7/12 (58.3%) | 9/12 (75.0%) | 3/12 (25.0%) | 5/12 (41.7%) | 8/12 (66.7%) | 11/12 (91.7%) |
| `00192` | `train/never_reached_0.8` | no | 70.8% | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| `00210` | `train/declined_after_reaching_0.8` | no | 77.0% | 3/12 (25.0%) | 1/12 (8.3%) | 3/12 (25.0%) | 0/12 (0.0%) | 3/12 (25.0%) | 2/12 (16.7%) | 5/12 (41.7%) | 1/12 (8.3%) |
| `00211` | `train/final_at_least_0.8` | no | 95.0% | 10/12 (83.3%) | 9/12 (75.0%) | 7/12 (58.3%) | 9/12 (75.0%) | 6/12 (50.0%) | 9/12 (75.0%) | 8/12 (66.7%) | 4/12 (33.3%) |
| `00213` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00255` | `train/final_at_least_0.8` | no | 96.9% | 6/12 (50.0%) | 7/12 (58.3%) | 5/12 (41.7%) | 5/12 (41.7%) | 5/12 (41.7%) | 8/12 (66.7%) | 6/12 (50.0%) | 5/12 (41.7%) |
| `00256` | `train/final_at_least_0.8` | no | 99.3% | 9/12 (75.0%) | 10/12 (83.3%) | 11/12 (91.7%) | 11/12 (91.7%) | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 12/12 (100.0%) |
| `00271` | `train/never_reached_0.8` | yes | 76.7% | 3/12 (25.0%) | 3/12 (25.0%) | 5/12 (41.7%) | 3/12 (25.0%) | 2/12 (16.7%) | 3/12 (25.0%) | 5/12 (41.7%) | 1/12 (8.3%) |
| `00293` | `train/final_at_least_0.8` | no | 96.8% | 10/12 (83.3%) | 10/12 (83.3%) | 10/12 (83.3%) | 10/12 (83.3%) | 9/12 (75.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 11/12 (91.7%) |
| `00296` | `train/final_at_least_0.8` | yes | 83.2% | 12/12 (100.0%) | 11/12 (91.7%) | 10/12 (83.3%) | 11/12 (91.7%) | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00301` | `train/never_reached_0.8` | no | 71.2% | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 0/12 (0.0%) | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) |
| `00308` | `train/final_at_least_0.8` | no | 90.7% | 4/12 (33.3%) | 4/12 (33.3%) | 6/12 (50.0%) | 7/12 (58.3%) | 5/12 (41.7%) | 5/12 (41.7%) | 5/12 (41.7%) | 1/12 (8.3%) |
| `00318` | `train/declined_after_reaching_0.8` | no | 79.0% | 2/12 (16.7%) | 4/12 (33.3%) | 6/12 (50.0%) | 4/12 (33.3%) | 1/12 (8.3%) | 4/12 (33.3%) | 3/12 (25.0%) | 2/12 (16.7%) |
| `00319` | `train/final_at_least_0.8` | no | 80.3% | 7/12 (58.3%) | 5/12 (41.7%) | 6/12 (50.0%) | 4/12 (33.3%) | 7/12 (58.3%) | 8/12 (66.7%) | 4/12 (33.3%) | 5/12 (41.7%) |
| `00320` | `train/final_at_least_0.8` | yes | 99.8% | 9/12 (75.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 10/12 (83.3%) | 11/12 (91.7%) | 11/12 (91.7%) | 6/12 (50.0%) |
| `00329` | `train/final_at_least_0.8` | no | 98.7% | 4/12 (33.3%) | 9/12 (75.0%) | 7/12 (58.3%) | 5/12 (41.7%) | 6/12 (50.0%) | 5/12 (41.7%) | 4/12 (33.3%) | 8/12 (66.7%) |
| `00340` | `train/final_at_least_0.8` | yes | 80.2% | 6/12 (50.0%) | 5/12 (41.7%) | 7/12 (58.3%) | 3/12 (25.0%) | 1/12 (8.3%) | 3/12 (25.0%) | 3/12 (25.0%) | 5/12 (41.7%) |
| `00345` | `train/final_at_least_0.8` | no | 87.7% | 6/12 (50.0%) | 8/12 (66.7%) | 6/12 (50.0%) | 8/12 (66.7%) | 2/12 (16.7%) | 5/12 (41.7%) | 5/12 (41.7%) | 8/12 (66.7%) |
| `00346` | `train/final_at_least_0.8` | yes | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00360` | `train/declined_after_reaching_0.8` | no | 77.7% | 2/12 (16.7%) | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 0/12 (0.0%) | 1/12 (8.3%) | 3/12 (25.0%) |
| `00388` | `train/never_reached_0.8` | yes | 61.2% | 3/12 (25.0%) | 2/12 (16.7%) | 4/12 (33.3%) | 4/12 (33.3%) | 6/12 (50.0%) | 3/12 (25.0%) | 0/12 (0.0%) | 2/12 (16.7%) |
| `00410` | `train/never_reached_0.8` | no | 61.8% | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) |
| `00417` | `train/final_at_least_0.8` | yes | 99.1% | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `00422` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 10/12 (83.3%) | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `00426` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) |
| `00437` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 11/12 (91.7%) |
| `00444` | `train/declined_after_reaching_0.8` | no | 79.3% | 2/12 (16.7%) | 4/12 (33.3%) | 6/12 (50.0%) | 5/12 (41.7%) | 3/12 (25.0%) | 3/12 (25.0%) | 6/12 (50.0%) | 4/12 (33.3%) |
| `00446` | `train/declined_after_reaching_0.8` | yes | 74.7% | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) | 1/12 (8.3%) | 2/12 (16.7%) | 1/12 (8.3%) | 0/12 (0.0%) | 2/12 (16.7%) |
| `00470` | `train/final_at_least_0.8` | no | 86.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00471` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00480` | `train/final_at_least_0.8` | no | 100.0% | 10/12 (83.3%) | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00486` | `train/declined_after_reaching_0.8` | no | 78.7% | 0/12 (0.0%) | 2/12 (16.7%) | 2/12 (16.7%) | 3/12 (25.0%) | 1/12 (8.3%) | 3/12 (25.0%) | 3/12 (25.0%) | 3/12 (25.0%) |
| `00499` | `train/final_at_least_0.8` | no | 98.4% | 6/12 (50.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 6/12 (50.0%) | 6/12 (50.0%) | 9/12 (75.0%) | 9/12 (75.0%) |
| `00506` | `train/declined_after_reaching_0.8` | no | 74.8% | 1/12 (8.3%) | 0/12 (0.0%) | 2/12 (16.7%) | 1/12 (8.3%) | 0/12 (0.0%) | 4/12 (33.3%) | 2/12 (16.7%) | 7/12 (58.3%) |
| `00514` | `train/final_at_least_0.8` | no | 80.1% | 10/12 (83.3%) | 9/12 (75.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 10/12 (83.3%) | 8/12 (66.7%) | 8/12 (66.7%) |
| `00537` | `train/declined_after_reaching_0.8` | no | 70.1% | 7/12 (58.3%) | 5/12 (41.7%) | 9/12 (75.0%) | 8/12 (66.7%) | 5/12 (41.7%) | 10/12 (83.3%) | 10/12 (83.3%) | 6/12 (50.0%) |
| `00553` | `train/final_at_least_0.8` | no | 97.7% | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 8/12 (66.7%) | 9/12 (75.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) |
| `00559` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00581` | `train/final_at_least_0.8` | no | 99.6% | 4/12 (33.3%) | 5/12 (41.7%) | 9/12 (75.0%) | 5/12 (41.7%) | 3/12 (25.0%) | 3/12 (25.0%) | 6/12 (50.0%) | 9/12 (75.0%) |
| `00597` | `train/final_at_least_0.8` | no | 92.4% | 11/12 (91.7%) | 8/12 (66.7%) | 7/12 (58.3%) | 9/12 (75.0%) | 10/12 (83.3%) | 8/12 (66.7%) | 8/12 (66.7%) | 6/12 (50.0%) |
| `00614` | `train/declined_after_reaching_0.8` | no | 72.1% | 0/12 (0.0%) | 2/12 (16.7%) | 1/12 (8.3%) | 1/12 (8.3%) | 2/12 (16.7%) | 0/12 (0.0%) | 2/12 (16.7%) | 1/12 (8.3%) |
| `00615` | `train/final_at_least_0.8` | no | 96.2% | 10/12 (83.3%) | 7/12 (58.3%) | 6/12 (50.0%) | 7/12 (58.3%) | 4/12 (33.3%) | 3/12 (25.0%) | 4/12 (33.3%) | 2/12 (16.7%) |
| `00638` | `train/declined_after_reaching_0.8` | no | 78.9% | 4/12 (33.3%) | 3/12 (25.0%) | 5/12 (41.7%) | 4/12 (33.3%) | 3/12 (25.0%) | 6/12 (50.0%) | 4/12 (33.3%) | 6/12 (50.0%) |
| `00648` | `train/declined_after_reaching_0.8` | no | 75.0% | 1/12 (8.3%) | 1/12 (8.3%) | 2/12 (16.7%) | 1/12 (8.3%) | 2/12 (16.7%) | 2/12 (16.7%) | 1/12 (8.3%) | 1/12 (8.3%) |
| `00649` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00652` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 10/12 (83.3%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00659` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00681` | `train/final_at_least_0.8` | yes | 98.3% | 1/12 (8.3%) | 2/12 (16.7%) | 3/12 (25.0%) | 4/12 (33.3%) | 2/12 (16.7%) | 3/12 (25.0%) | 3/12 (25.0%) | 2/12 (16.7%) |
| `00686` | `train/final_at_least_0.8` | no | 99.8% | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00700` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00703` | `train/never_reached_0.8` | no | 77.5% | 1/12 (8.3%) | 2/12 (16.7%) | 0/12 (0.0%) | 1/12 (8.3%) | 0/12 (0.0%) | 1/12 (8.3%) | 0/12 (0.0%) | 1/12 (8.3%) |
| `00726` | `train/final_at_least_0.8` | no | 96.6% | 0/12 (0.0%) | 4/12 (33.3%) | 7/12 (58.3%) | 5/12 (41.7%) | 6/12 (50.0%) | 6/12 (50.0%) | 5/12 (41.7%) | 5/12 (41.7%) |
| `00731` | `train/final_at_least_0.8` | yes | 100.0% | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00741` | `train/never_reached_0.8` | no | 77.1% | 3/12 (25.0%) | 3/12 (25.0%) | 2/12 (16.7%) | 1/12 (8.3%) | 2/12 (16.7%) | 4/12 (33.3%) | 4/12 (33.3%) | 3/12 (25.0%) |
| `00768` | `train/final_at_least_0.8` | yes | 97.3% | 4/12 (33.3%) | 2/12 (16.7%) | 4/12 (33.3%) | 0/12 (0.0%) | 1/12 (8.3%) | 1/12 (8.3%) | 3/12 (25.0%) | 0/12 (0.0%) |
| `00783` | `train/final_at_least_0.8` | no | 100.0% | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00831` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00855` | `train/never_reached_0.8` | no | 69.8% | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| `00860` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00863` | `train/never_reached_0.8` | yes | 59.7% | 0/12 (0.0%) | 1/12 (8.3%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| `01026` | `train/declined_after_reaching_0.8` | no | 77.4% | 5/12 (41.7%) | 3/12 (25.0%) | 1/12 (8.3%) | 2/12 (16.7%) | 0/12 (0.0%) | 2/12 (16.7%) | 0/12 (0.0%) | 1/12 (8.3%) |
| `01029` | `train/declined_after_reaching_0.8` | no | 75.9% | 2/12 (16.7%) | 6/12 (50.0%) | 2/12 (16.7%) | 3/12 (25.0%) | 3/12 (25.0%) | 4/12 (33.3%) | 4/12 (33.3%) | 3/12 (25.0%) |
| `01036` | `train/final_at_least_0.8` | yes | 98.9% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01041` | `train/final_at_least_0.8` | yes | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01053` | `train/final_at_least_0.8` | no | 98.9% | 7/12 (58.3%) | 8/12 (66.7%) | 4/12 (33.3%) | 4/12 (33.3%) | 3/12 (25.0%) | 1/12 (8.3%) | 5/12 (41.7%) | 6/12 (50.0%) |
| `01079` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01092` | `train/final_at_least_0.8` | no | 100.0% | 10/12 (83.3%) | 12/12 (100.0%) | 12/12 (100.0%) | 11/12 (91.7%) | 11/12 (91.7%) | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) |
| `01102` | `train/final_at_least_0.8` | no | 99.9% | 12/12 (100.0%) | 11/12 (91.7%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01125` | `train/final_at_least_0.8` | no | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01129` | `train/final_at_least_0.8` | yes | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `01132` | `train/declined_after_reaching_0.8` | no | 78.3% | 4/12 (33.3%) | 3/12 (25.0%) | 3/12 (25.0%) | 3/12 (25.0%) | 4/12 (33.3%) | 1/12 (8.3%) | 1/12 (8.3%) | 3/12 (25.0%) |
| `01136` | `train/final_at_least_0.8` | yes | 100.0% | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `00755` | `unseen/never_reached_0.8` | no | 46.3% | 3/12 (25.0%) | 9/12 (75.0%) | 10/12 (83.3%) | 10/12 (83.3%) | 8/12 (66.7%) | 9/12 (75.0%) | 8/12 (66.7%) | 8/12 (66.7%) |

### Specialist 难度是否传递到 generalist？

† Specialist 列是 RL specialist 训练末端的 TensorBoard `successes/iter`，generalist 列是 default ×1、fixed-hardest 的 12-rollout task SR。两者不是同协议估计量，因此绝对差值只能作为诊断；任务排序相关性更适合回答“specialist 难的任务是否也让 generalist 困难”。以下统计只使用 99 个 joint-train ID tasks，排除 OOD `00755`。

- Specialist final train SR 与 8 个 generalist SR 的逐任务平均值具有较强相关性：Pearson `r=0.773`，Spearman `ρ=0.818`。
- 对八个 generalist condition 分别计算，Pearson `r` 均在 `0.713–0.783`，Spearman `ρ` 均在 `0.786–0.836`。该关系不是由某一个 condition 单独造成。
- 任务组进一步支持难度传递：specialist 最终仍 ≥0.8 的 69 个任务，generalist 平均 SR 为 `80.0%`；达到过 0.8、但训练末期回落的 18 个任务为 `31.0%`；从未达到 0.8 的 12 个任务仅为 `12.8%`。
- “降低幅度相似”不成立。Specialist 到 generalist 平均 SR 的 mean gap 为 `26.7 pp`，task-level sample std 为 `27.7 pp`，范围为 `−27.8` 到 `+81.6 pp`。而且 gap 随 specialist 难度增加：上述三组平均 gap 分别为 `16.7`、`45.2` 和 `56.3 pp`。

因此，specialist difficulty 是 generalist difficulty 的强预测信号，但 generalist 并不是给每个 task 施加近似恒定的成功率折损。难 specialist 往往在 joint policy 中进一步放大为低成功率任务。`00755` 是一个值得单列的例外：它未进入 joint training，specialist final train SR 为 `46.3%`，而八个 generalist 的零样本平均为 `67.7%`；这个单任务结果提示跨任务共享可能带来正迁移，但需要更多 held-out assemblies 才能概括。

机器可读分析见 [`specialist_generalist_task_analysis.tsv`](../logs/main-joint-report-update-0917/specialist_generalist_task_analysis.tsv) 与 [`specialist_generalist_summary.json`](../logs/main-joint-report-update-0917/specialist_generalist_summary.json)。

## 结论先行

`rgbd_gp` 的 x2 结果为 **713/1200 = 59.4%**（99 train：708/1188 = 59.6%；unseen `00755`：5/12）。旧目录中有部分结果以符号链接保存，常规 `find -type f` 曾错误漏计；回填 runner 已跟随链接逐条完成协议验证，确认 100/100 task 有有效结果。最终仍以统一 8-condition 审计后的 TSV 为唯一总表证据。

AutoMate 原文的最强 generalist 是 `BC + DAgger + RL + SBC`：在其固定 20-task benchmark 上报 **80.42% ± 20.93%**。当前模型在相同 20 个 assembly ID 的子集上是 **130/240 = 54.2%**，但这只能作为同-task 的诊断，**不能作为论文级直接比较或“低于/高于原文”的结论**：模型、训练方法、观测、trial 数和 seed-selection 均未对齐；且当前扩展面板使用 2× default reset，严于原文的 default reset。

## 当前评测身份与完整数据

| 项目 | 当前设定 |
|---|---|
| checkpoint | paired `rgbd_gp`，seed `2026090601`，SHA-256 `c3c64a393421a1ec9a0ae2877fa03cee50ad693f28f1da636e94f6978aeeeefd` |
| policy / training scope | Joint visual BC policy；99-task joint training，不是 AutoMate 的 geometry-only generalist 或其 RL-distillation pipeline |
| reset / curriculum | **当前扩展评测**使用 2× default 基础扰动、每个 asset 固定 curriculum 上界、SBC 关闭（`training_2x_hardest_curriculum`）。fixed-upper / SBC-off 对齐原文 maximum-bound 测试语义，但幅度是原文 default 的 2 倍，故它是更难的鲁棒性诊断，而非原文 reset 分布 |
| rollout | 12/task，12 env，固定 seed；100 task（99 train + unseen `00755`） |
| provenance | `annotation_source=scripted`；每 task 初始 task/inserted 成功率为 0；无 rollout error |
| raw per-task data | [100-task results table](../logs/automate-paper-hardest-eval-0914/extension100/results_table.md)，[per-task JSON](../logs/automate-paper-hardest-eval-0914/extension100/results/) |

| Group | Success | SR |
|---|---:|---:|
| 99 train: final specialist ≥ 0.8 | 638/828 | 77.1% |
| 99 train: later declined after ≥ 0.8 | 52/216 | 24.1% |
| 99 train: never reached 0.8 | 18/144 | 12.5% |
| **99 train total** | **708/1188** | **59.6%** |
| unseen `00755` | 5/12 | 41.7% |
| **99 train + 1 unseen** | **713/1200** | **59.4%** |

## 与 AutoMate 原文的正确比较方式

AutoMate 的 generalist 从 specialist 的成功轨迹开始，按 BC、DAgger、RL fine-tuning 和 SBC 的阶段训练；原文训练比较针对固定 20 assemblies，使用 5 个随机 seed 中的最佳 checkpoint，并对每个 assembly 做 5 × 1000 次测试。其 reported mean success rate 为 BC 28.84% ± 15.23%，BC+DAgger 31.06% ± 15.06%，BC+DAgger+RL 52.85% ± 14.01%，以及 BC+DAgger+RL+SBC 80.42% ± 20.93%；RL-from-scratch+SBC 为 48.43% ± 15.28%。[AutoMate paper](https://bingjietang718.github.io/pdfs/rss2024.pdf)

| Result | Task set | Protocol | Result | 是否可直接比较 |
|---|---|---|---:|---|
| AutoMate BC+DAgger+RL+SBC (paper) | 固定 20 tasks | geometry generalist；SBC；5×1000 tests/task；multi-seed selection | 80.42% ± 20.93% | 原文指标 |
| Joint `rgbd_gp` (this run) | 相同 20 IDs 的交集 | visual joint BC；2× default、fixed maximum curriculum、12 tests/task、单 checkpoint | 130/240 = 54.2% | 仅同-task diagnostic；reset 比原文更难 |
| Joint `rgbd_gp` (this run) | 99 train tasks | 同上 | 708/1188 = 59.6% | 任务规模不同，不能与原文 20-task 数字比较 |

因此可以写的事实是：在当前更大规模的 99-train-task joint BC 设置中，`rgbd_gp` 的 hardest-curriculum 成功率是 59.6%；在 AutoMate 的 20 个 ID 上，同一评测脚本得到 54.2%。不应写成“复现/超过/落后 AutoMate 80.42%”。要形成可发表的 head-to-head，需要至少锁定原文的 20 个 task、其最大 reset/observation-noise 语义与 SBC、等效表示与训练配方，并以足够的 trial 和多 seed 报告置信区间。原文的扩展实验也显示其 generalist 从 20 task 扩展到 30、40 和超过 40 task 时平均成功率明显下降，所以 99-task 设定本身不是原论文主表所覆盖的同一规模。[AutoMate paper](https://bingjietang718.github.io/pdfs/rss2024.pdf)

### Reset 条件的边界

本次 8-condition 队列不是 `training_2x_sbc` 的训练时采样面板。实际 client 为所有 task 固定 2× default 基础噪声（fixed asset position `[0.1, 0.1, 0.1]`、orientation `20°`、held asset position `[0.02, 0.02, 0.02]`、fixed-asset observation position `[0.002, 0.002, 0.002]`），并显式校验；使用 `--curriculum-mode hardest`，故每个 asset 的 `curr_max_disp_m` 固定为 upper bound（当前为 `0.028 m`），且 `if_sbc=false`。论文 generalist 的**测试**同样固定 maximum bounds、而 SBC 只用于 RL 训练中逐阶段提高初始高度下界；但其 default 数值为 `[0.05, 0.05, 0.05]`、`10°`、`[0.01, 0.01, 0.01]` 和 `[0.001, 0.001, 0.001]`，恰为当前值的一半。因此本表是“**2×default、fixed-hardest、无 SBC**”的更难鲁棒性评测，不是原文 default × 1 测试分布。严格的原文-reset 面板应保持 fixed maximum / SBC-off，但改用 default × 1；此外仍需区分其 `BC+DAgger+RL+SBC` 方法及 5 seeds 选最佳、5×1000 rollout/task 的统计协议。

### 执行顺序（2026-09-14 修订）

按实验优先级，先运行 **default ×1、fixed-maximum、SBC-off** 的 100-task × 8-condition 面板，输出根为 `logs/automate-paper-hardest-eval-0914/paper_default1_all_conditions/`；它对齐原文的 reset 分布，但仍是本项目的 12-rollout 扩展统计。完成后再恢复 **2×default、fixed-maximum、SBC-off** 的压力测试矩阵，继续使用原 `all_conditions/` 根。

已完成的 2×结果不会重跑或覆盖：`paired_rgb` 为 100/100 task（695/1200 = 57.9%）；`paired_rgbd` 从已落盘结果续跑；`paired_rgbd_gp` 为 100/100（713/1200 = 59.4%，含以链接保存的已验证 JSON）。恢复时 runner 会复用并审计所有完整 `result.json`，仅补缺失 task。

default ×1 面板的 `paired_rgb` 已完成 100/100 task：742/1200 = **61.8%**（99 train：739/1188 = **62.2%**；原文 paper-20 ID 子集：132/240 = **55.0%**）。全部 task 逐条通过 `default1` 参数（`0.05 m` / `10°` / `0.01 m` / `0.001 m`）、`hardest` upper bound、SBC-off、scripted provenance、12 rollout 与零 rollout-error 审计。其原始 JSON 位于 `paper_default1_all_conditions/paired_rgb/results/`；其余 7 个 x1 condition 仍在同一队列中。

default ×1 的 `paired_rgbd` 也已完成 100/100 task：770/1200 = **64.2%**。其 100 个 task 均通过相同的 default1、fixed-maximum、SBC-off、scripted、12-rollout 与零错误审计；原始 JSON 位于 `paper_default1_all_conditions/paired_rgbd/results/`。

### default ×1、原文 reset 分布：已完整完成

该面板已完成全部 **8 condition × 100 task × 12 rollout = 9,600 rollout**。汇总器生成的 [condition × task TSV](../logs/automate-paper-hardest-eval-0914/paper_default1_all_conditions/condition_task_success_rates.tsv)、[结果表](../logs/automate-paper-hardest-eval-0914/paper_default1_all_conditions/results_table.md) 与 [审计](../logs/automate-paper-hardest-eval-0914/paper_default1_all_conditions/audit.json) 均已落盘；总审计为 `true`。

| Condition | 99 train tasks | Paper-20 子集 | 99 train + 1 unseen |
|---|---:|---:|---:|
| `paired_rgb` | 739/1188 (62.2%) | 132/240 (55.0%) | 742/1200 (61.8%) |
| `paired_rgbd` | 761/1188 (64.1%) | 132/240 (55.0%) | 770/1200 (64.2%) |
| `paired_rgbd_colored_gp` | 773/1188 (65.1%) | 141/240 (58.8%) | 783/1200 (65.2%) |
| `paired_rgbd_gp` | 754/1188 (63.5%) | 132/240 (55.0%) | 764/1200 (63.7%) |
| `paired_rgbd_gp_skill` | 703/1188 (59.2%) | 124/240 (51.7%) | 711/1200 (59.2%) |
| `paired_rgbd_grasp_part` | 748/1188 (63.0%) | 128/240 (53.3%) | 757/1200 (63.1%) |
| `paired_rgbd_grasp_part_colored` | 755/1188 (63.6%) | 132/240 (55.0%) | 763/1200 (63.6%) |
| `formal_rgbd_skill` | 750/1188 (63.1%) | 122/240 (50.8%) | 758/1200 (63.2%) |

所有行均为 `automate_paper_default1_hardest_curriculum`：default ×1 参数、各 task fixed maximum curriculum、SBC-off、scripted annotation、12 rollout/task。`00755` 不计入 99-train 分母，并正确纳入最后一列的 100-task 总和。

### default ×2、训练扰动强度压力测试：已完整完成

该面板同样完成全部 **8 condition × 100 task × 12 rollout = 9,600 rollout**，使用 `training_2x_hardest_curriculum`（2× default、fixed maximum curriculum、SBC-off、scripted annotation）。8 个 checkpoint 均有 100/100 完整 JSON，最终审计为 `all_checks_pass=true`。这是与上表不同的、更难 reset 分布；两张表不得混合求平均。

| Condition | 99 train tasks | Paper-20 子集 | 99 train + 1 unseen |
|---|---:|---:|---:|
| `paired_rgb` | 693/1188 (58.3%) | 121/240 (50.4%) | 695/1200 (57.9%) |
| `paired_rgbd` | 722/1188 (60.8%) | 128/240 (53.3%) | 726/1200 (60.5%) |
| `paired_rgbd_colored_gp` | 724/1188 (60.9%) | 132/240 (55.0%) | 729/1200 (60.8%) |
| `paired_rgbd_gp` | 708/1188 (59.6%) | 130/240 (54.2%) | 713/1200 (59.4%) |
| `paired_rgbd_gp_skill` | 698/1188 (58.8%) | 118/240 (49.2%) | 699/1200 (58.2%) |
| `paired_rgbd_grasp_part` | 705/1188 (59.3%) | 130/240 (54.2%) | 707/1200 (58.9%) |
| `paired_rgbd_grasp_part_colored` | 720/1188 (60.6%) | 126/240 (52.5%) | 725/1200 (60.4%) |
| `formal_rgbd_skill` | 719/1188 (60.5%) | 128/240 (53.3%) | 726/1200 (60.5%) |

最终工件为 [x2 summary](../logs/automate-paper-hardest-eval-0914/all_conditions/results_table.md)、[condition × task TSV](../logs/automate-paper-hardest-eval-0914/all_conditions/condition_task_success_rates.tsv) 和 [audit](../logs/automate-paper-hardest-eval-0914/all_conditions/audit.json)。

## 其余 condition 的完整扩展矩阵（执行历史；最终数值见上表）

以下为队列执行时的状态记录；所有 8 个 condition 后续均已完成，最终数值以上一节及其链接的最终审计为准：

| Family | Condition | Status |
|---|---|---|
| paired seed `2026090601` | `rgb` | complete; 695/1200 = 57.9%; 99-train 689/1188 = 58.0%; paper-20 subset 127/240 = 52.9%; 100/100 protocol audit passes |
| paired seed `2026090601` | `rgbd` | queued/running |
| paired seed `2026090601` | `rgbd_colored_gp` | queued/running |
| paired seed `2026090601` | `rgbd_gp` | complete; 713/1200 = 59.4%; 100/100 protocol validation passes (including reused symlink results) |
| paired seed `2026090601` | `rgbd_gp_skill` | queued/running |
| paired seed `2026090601` | `rgbd_grasp_part` | queued/running |
| paired seed `2026090601` | `rgbd_grasp_part_colored` | queued/running |
| formal batch-512 online | `rgbd_skill` | queued/running; family differs and will not be treated as paired comparison |

每个 condition 的 checkpoint 路径与 SHA-256 固定在 `logs/automate-paper-hardest-eval-0914/other_conditions.tsv`。完成后会自动生成 [condition × task TSV](../logs/automate-paper-hardest-eval-0914/all_conditions/condition_task_success_rates.tsv)、[condition summary](../logs/automate-paper-hardest-eval-0914/all_conditions/results_table.md) 和 [audit](../logs/automate-paper-hardest-eval-0914/all_conditions/audit.json)。

已完成的 `rgb` 条件独立复核为 100 个 task × 12 rollout；所有结果均为 `annotation_source=scripted`、`training_2x_hardest_curriculum`、SBC 关闭，并且每个 asset 均使用 curriculum upper bound。其 aggregate 成功率为 695/1200 = 57.9%（99 train：689/1188 = 58.0%；AutoMate paper-20 ID 子集：127/240 = 52.9%）。其逐 task JSON 位于 `all_conditions/paired_rgb/results/`；最终总表仍以全部八个 condition 的统一 `audit.json` 为准。

队列执行期间，已落盘 task 的累计 SR 和协议状态每分钟更新在 [live progress table](../logs/automate-paper-hardest-eval-0914/all_conditions/live_results_table.md)；它明确标记为未完成进度，不能替代最终 `audit.json`。
