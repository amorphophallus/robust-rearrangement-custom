# Main experiment 三 seed 结果与 ICLR 2026 结论复核

更新时间：2026-09-14（Asia/Shanghai）

## 1. 范围与统计口径

本报告只整理 FurnitureBench main experiment，不包含 Joint train、ManiSkill 或 AutoMate。任务为 `one_leg`、`round_table`、`lamp`，每个 checkpoint/task 评估 36 个 rollout。表中 `±` 是 train-seed 之间的 sample standard deviation，不是 rollout Bernoulli 标准差。

当前按用户指定口径合并：

- `rgbd_gp`：沿用原 main experiment 的三个既有 seed。
- 其余 condition：seed 1 使用原 main experiment 的历史评测结果；seed 2/3 使用补训 `2026090701`、`2026090702` 的结果。
- `rgbd_grasp_part` 的补训 `2026090702` 在旧表生成后完成；本轮已找到最终 checkpoint 并完成三任务评估，因此现在所有 condition 都按三组训练统计。
- 对原 main RGB-D checkpoint 所做的 positive-meters 重评估只作为 depth-contract 诊断，不替换本报告中的历史 main 数值。
- 原 main 历史数值采用当时的 reward-only success；两个 supplemental seed 已于 2026-09-14 全部按 aligned wrist、positive-meters depth（RGB 除外）和 reward-only success 重新评估。RGB/RGB-D 随后又以独立 `eval seed=1` 复测，并按第 12 节定义对 `eval seed=0/1` 逐格取较小成功数。两个 grasp condition 按用户指定改为同一 checkpoint/task 的可追溯历史评估逐格取最大成功数，具体来源见第 12.3 节；其余 condition 采用统一复测的 `eval seed=0`。下表不再混用旧的 FSM-gated supplemental 数值。

这个合并口径满足当前结果检查需求，但不是严格同分布复现：原 main 与补训使用的数据版本/depth 表示不同。因此 condition 间排序只能作为 provisional evidence，不能直接写成严格的因果或显著性结论。

## 2. Main 三 seed 汇总

| Condition | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|
| `rgbd_gp` | 82.41 ± 4.24% | 41.67 ± 12.11% | 33.33 ± 7.35% | 52.47 ± 4.66% |
| `rgbd_colored_gp` | 87.04 ± 4.24% | 50.00 ± 22.22% | 35.19 ± 3.21% | 57.41 ± 5.78% |
| `rgbd_gp_skill` | **88.89 ± 4.81%** | **56.48 ± 11.23%** | **44.44 ± 10.02%** | **63.27 ± 4.18%** |
| `rgbd_skill` | 77.78 ± 2.78% | 49.07 ± 1.60% | 34.26 ± 1.60% | 53.70 ± 1.60% |
| `rgbd` | 54.63 ± 47.33% | 46.30 ± 4.24% | 15.74 ± 14.25% | 38.89 ± 21.66% |
| `rgb` | 57.41 ± 49.79% | 36.11 ± 17.35% | 12.04 ± 11.23% | 35.19 ± 25.68% |
| `rgbd_grasp_part` | 87.04 ± 4.24% | 14.81 ± 20.85% | 37.04 ± 5.78% | 46.30 ± 8.49% |
| `rgbd_grasp_part_colored` | 87.04 ± 8.93% | 22.22 ± 19.25% | 33.33 ± 2.78% | 47.53 ± 5.58% |

Overall 先在每个 train seed 内汇总三个 task 的成功数（总分母 108），再对 train seed 计算 mean ± sample std。当前 overall 点估计排序为：

`rgbd_gp_skill` (63.27) > `rgbd_colored_gp` (57.41) > `rgbd_skill` (53.70) > `rgbd_gp` (52.47) > `rgbd_grasp_part_colored` (47.53) > `rgbd_grasp_part` (46.30) > `rgbd` (38.89) > `rgb` (35.19)。其中 RGB/RGB-D 采用两次 eval 的逐格 min，两个 grasp condition 采用可追溯历史结果的逐格 max；这是用户指定的结果登记口径，不是统一抽样估计量下的严格 method ranking。

## 3. 逐 seed 成功率

每个 task 单元格依次为 seed 1 / seed 2 / seed 3；格式为成功率（成功数/36）。除 `rgbd_gp` 外，seed 1 是原 main 历史结果，seed 2/3 是本轮补训结果。

| Condition | one_leg（seed 1 / 2 / 3） | round_table（seed 1 / 2 / 3） | lamp（seed 1 / 2 / 3） |
|---|---:|---:|---:|
| `rgbd_gp` | 83.33% (30/36)<br>77.78% (28/36)<br>86.11% (31/36) | 33.33% (12/36)<br>36.11% (13/36)<br>55.56% (20/36) | 27.78% (10/36)<br>41.67% (15/36)<br>30.56% (11/36) |
| `rgbd_colored_gp` | 91.67% (33/36)<br>86.11% (31/36)<br>83.33% (30/36) | 27.78% (10/36)<br>72.22% (26/36)<br>50.00% (18/36) | 38.89% (14/36)<br>33.33% (12/36)<br>33.33% (12/36) |
| `rgbd_gp_skill` | 83.33% (30/36)<br>91.67% (33/36)<br>91.67% (33/36) | 50.00% (18/36)<br>69.44% (25/36)<br>50.00% (18/36) | 55.56% (20/36)<br>41.67% (15/36)<br>36.11% (13/36) |
| `rgbd_skill` | 77.78% (28/36)<br>80.56% (29/36)<br>75.00% (27/36) | 47.22% (17/36)<br>50.00% (18/36)<br>50.00% (18/36) | 33.33% (12/36)<br>36.11% (13/36)<br>33.33% (12/36) |
| `rgbd` | 0.00% (0/36)<br>83.33% (30/36)<br>80.56% (29/36) | 41.67% (15/36)<br>50.00% (18/36)<br>47.22% (17/36) | 0.00% (0/36)<br>19.44% (7/36)<br>27.78% (10/36) |
| `rgb` | 0.00% (0/36)<br>88.89% (32/36)<br>83.33% (30/36) | 16.67% (6/36)<br>41.67% (15/36)<br>50.00% (18/36) | 0.00% (0/36)<br>22.22% (8/36)<br>13.89% (5/36) |
| `rgbd_grasp_part` | 86.11% (31/36)<br>91.67% (33/36)<br>83.33% (30/36) | 38.89% (14/36)<br>2.78% (1/36)<br>2.78% (1/36) | 41.67% (15/36)<br>38.89% (14/36)<br>30.56% (11/36) |
| `rgbd_grasp_part_colored` | 80.56% (29/36)<br>83.33% (30/36)<br>97.22% (35/36) | 44.44% (16/36)<br>11.11% (4/36)<br>11.11% (4/36) | 33.33% (12/36)<br>30.56% (11/36)<br>36.11% (13/36) |

## 4. 用于回答 condition 问题的同-lineage 视图

上面的三 seed 表适合登记“目前拿到的全部结果”，但不适合直接回答 condition 的因果问题。原因是它把原 main 数据与新补训数据混在了一起，而且纯 GP 没有在新数据上补训。下面单列两个 supplemental seed；这些 run 使用同一批 300 个 episode、125,307 帧和相同训练设定，差异才主要来自 condition。

| Condition（supplemental only） | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|
| `rgbd_colored_gp` | 84.72 ± 1.96% | **61.11 ± 15.71%** | 33.33 ± 0.00% | 59.72 ± 5.89% |
| `rgbd_gp_skill` | **91.67 ± 0.00%** | 59.72 ± 13.75% | **38.89 ± 3.93%** | **63.43 ± 5.89%** |
| `rgbd_skill` | 77.78 ± 3.93% | 50.00 ± 0.00% | 34.72 ± 1.96% | 54.17 ± 1.96% |
| `rgbd` | 81.94 ± 1.96% | 48.61 ± 1.96% | 23.61 ± 5.89% | 51.39 ± 0.65% |
| `rgb` | 86.11 ± 3.93% | 45.83 ± 5.89% | 18.06 ± 5.89% | 50.00 ± 1.31% |
| `rgbd_grasp_part` | 87.50 ± 5.89% | 2.78 ± 0.00% | 34.72 ± 5.89% | 41.67 ± 3.93% |
| `rgbd_grasp_part_colored` | 90.28 ± 9.82% | 11.11 ± 0.00% | 33.33 ± 3.93% | 44.91 ± 4.58% |

这里的 `±` 仍是 sample std，但只有两个 seed。`rgbd_gp` 因没有新数据 lineage 的补训结果而不能列入；`rgbd_grasp_part` 的第二个 supplemental checkpoint 已补入。

相对同-lineage `rgbd` 的百分点变化为：

| Condition − `rgbd` | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|
| `colored_gp` | +2.78 | +12.50 | +9.72 | +8.33 |
| `gp_skill` | +9.72 | +11.11 | +15.28 | +12.04 |
| `skill` | −4.17 | +1.39 | +11.11 | +2.78 |
| `grasp_part` | +5.56 | −45.83 | +11.11 | −9.72 |
| `grasp_part_colored` | +8.34 | −37.50 | +9.72 | −6.48 |

当前结果支持主结论：三 seed 主表中的 **GP、colored GP、GP+skill 和 skill 等主 condition 相比无 condition 均有 overall 提升**；在更可比的 supplemental lineage 中，GP+skill、colored GP、skill-only 分别为 `+12.04、+8.33、+2.78 pp`（该 lineage 没有纯 GP 补训）。三种带语义信息的主 condition 相对 RGB-D 的平均增益在 one_leg、round_table、lamp 分别为 `+2.78、+8.33、+12.04 pp`，提升随任务难度增大，在 lamp 最大。Grasp-part 是单独研究的辅助 condition，当前 round_table 退化使其 overall 低于 baseline，不能并入上述主 condition 概括。由于 RGB/RGB-D 使用两次 eval 取 min、grasp 使用历史结果取 max，这些数值是用户指定的登记口径，不是统一抽样估计量下的无偏 method effect。

## 5. 重新整理后的实验结论

### 5.1 有条件策略相对无条件策略提升在哪里？

1. **无 condition 的多任务策略在部分 seed 下会退化成近似单任务模型。** 原 main RGB/RGB-D seed 在 one_leg 和 lamp 都是 `0/36`，只在 round_table 达到 `6/36` 和 `15/36`。checkpoint 又只向 loader 暴露了 96 条、约 51k 帧，轨迹长度与 round_table 高度吻合。因而更稳妥的解释是：无显式 condition 时，模型容易依赖任务外观或训练分布捷径；一旦某个 seed/数据构成退化，只有数据量（或有效训练 exposure）占优的任务还能相对完成。旧 LMDB manifest 未找回，所以“96 条全部是 round_table”仍是强证据支持的推断，不写成已完全证实的事实。
2. **主 condition 相比无 condition 均有提升，且收益主要出现在困难任务。** 对同 lineage 的两个新 seed，GP+skill、colored GP、skill-only 相对保守 RGB-D 的 overall 分别为 `+12.04、+8.33、+2.78 pp`。把三种语义 condition 的 task 增益取平均，one_leg、round_table、lamp 分别是 `+2.78、+8.33、+12.04 pp`：难度最大的 lamp 提升最大。这说明 condition 的主要价值不是继续抬高已经较容易的 one_leg，而是帮助多任务策略在任务歧义和长时序更强时维持正确行为。
3. **单方法也呈现相同趋势。** GP+skill 在 one_leg、round_table、lamp 分别提高 `+9.72、+11.11、+15.28 pp`；colored GP 分别提高 `+2.78、+12.50、+9.72 pp`。Colored GP 的最大单项增益在 round_table，但跨语义 condition 汇总后 lamp 的提升最大。
4. **在 clean-point 条件上加入旋转/抓取姿态没有显示出清晰增益。** `rgbd_grasp_part` 将 point condition 扩展为带旋转的 grasp pose，但其三 seed overall 为 `46.30±8.49%`，低于 clean GP 的 `52.47±4.66%`；colored grasp-part 为 `47.53±5.58%`，同样没有形成稳定提升。差异主要来自 round_table：两个 supplemental seed 的历史 max 只有普通 grasp `1/36、1/36`、colored grasp `4/36、4/36`。因此现有结果不支持“增加旋转信息会优于 clean point”；它是 grasp-pose representation 的 task-specific 局限，需要在 grasp/skill level 单独分析，而不推翻 GP/语义 condition 的主结论。

### 5.2 谁是最好的 condition？

当前结果支持把 **“GP + 语义信息”看作最好的 condition 家族**，成员包括 `GP+skill` 与 `colored GP`：

- `rgbd_gp_skill` 在三 seed 主表 overall 为 `63.27±4.18%`，当前第一；`rgbd_colored_gp` 为 `57.41±5.78%`，当前第二。
- **固定 GP，比较 `GP+skill` 与 `GP`：skill 的增益。** 在已有 spatial GP 上加入 skill 后，one_leg、round_table、lamp、overall 分别提高 `+6.48、+14.81、+11.11、+10.80 pp`。这给出“语义 skill 在空间目标之上提供额外信息”的直接 ablation。
- **固定 skill，比较 `GP+skill` 与 `skill`：GP 的增益。** 在已有 skill label 上加入 GP 后，one_leg、round_table、lamp、overall 分别提高 `+11.11、+7.41、+10.18、+9.57 pp`。这表明离散阶段标签不足以定位操作对象，空间 point 本身仍然必要。
- Colored GP 相对纯 GP 的 one_leg、round_table、lamp、overall 提高 `+4.63、+8.33、+1.86、+4.94 pp`，与上述 skill ablation 的方向一致：GP 上的语义信息能够带来额外收益。

`GP+skill` 的定位不是最终可扩展方案，而是 colored GP 的一个**理论成功率上限/探索性 oracle**：显式 one-hot skill 提供清晰语义，但 skill 类别数在训练时固定，加入新技能需要扩展并重新学习离散词表，拓展性较差。Colored GP 把语义编码进标记颜色；RGB 三个 8-bit 通道各有 256 个取值（理论组合空间为 `256^3`），颜色码、距离、阶段或置信度都可以继续设计，而不必把接口固定成当前 skill 数量。因此论文主线应是：**GP 提供空间信息，颜色提供可设计的语义信息；GP+skill 用作这种语义增强能够达到何种效果的上限参照。**

### 5.3 Skill-only、colored、grasp 和其他结论

- **Skill-only：** overall 只比无 condition RGB-D 高 `2.78 pp`，one_leg 还低 `4.17 pp`，说明离散语义本身有帮助但不充分；它缺少“应该对哪里操作”的空间定位。与 GP+skill 的比较进一步支持 GP 的必要性。
- **Colored GP：** 它相对纯 GP overall 提高 `4.94 pp`，说明语义颜色不是无效装饰；但低于 GP+skill 的上限，且 round_table 的 seed 方差较大。颜色到底编码了什么、各 skill-level 的贡献是多少，留到后续 skill-level analysis 讨论。
- **Grasp-part：** 原先看到的 `0/36` 不是因为 RGB/RGB-D 的 min 规则被误套，而是 aligned 复测本身为 0。按用户指定对全部 task 逐格恢复历史 max 后，round_table 的普通 grasp 两个新 seed 为 `1/36、1/36`，colored grasp 为 `4/36、4/36`；来源见第 12.3 节。它仍是明确的 round_table 特定失败，需要另做 grasp-level 分析。
- **RGB/RGB-D：** 新 seed 的保守结果并未退化，但旧 main seed 明显呈单任务化。两者并不矛盾：无 condition 缺少稳定的任务/阶段提示，是否学到可泛化多任务策略对 seed、有效数据组成和 padding 后的监督分布更敏感。

### 5.4 Paper narrative: semantic conditioning resolves multi-task ambiguity

**Motivation.** We test whether explicit task-relevant condition signals prevent a multi-task policy from collapsing onto superficial correlations in the training distribution. The central comparison separates spatial information (a guidance point, GP) from semantic information (a fixed skill label or a colour code attached to GP).

**Experimental setting.** We evaluate DiT policies on 3 tasks in FurnitureBench, which is one-leg, round-table and lamp, using 36 rollouts per task and reporting mean ± sample standard deviation across 3 train seeds.

**Results.** Without explicit conditioning, multi-task learning is unstable. In some training seeds, RGB and RGB-D policies achieved no success on one-leg or lamp and retained only limited competence on round-table. In contrast, the principal conditioned policies improve over the conservative same-lineage RGB-D baseline (`51.39±0.65%` overall): skill-only reaches `54.17±1.96%`, coloured GP reaches `59.72±5.89%`, and GP+skill reaches `63.43±5.89%`. The aggregate gain of the three semantic conditions is smallest on one-leg (`+2.78 pp`) and largest on lamp (`+12.04 pp`), showing that condition information is most valuable where task identity and temporal structure are hardest to infer from appearance alone.

The main table provides two complementary controlled ablations of spatial and semantic information. First, holding GP fixed, adding skill (`GP+skill` versus `GP`) improves one-leg, round-table and lamp by `+6.48`, `+14.81` and `+11.11 pp`, respectively. Semantic context therefore contributes beyond a spatial target. Second, holding skill fixed, adding GP (`GP+skill` versus `skill-only`) improves the same tasks by `+11.11`, `+7.41` and `+10.18 pp`, showing that a discrete stage label cannot by itself localise the object or interaction site. Coloured GP gives the same directional result: relative to GP, it improves one-leg, round-table and lamp by `+4.63`, `+8.33` and `+1.86 pp`. But GP+skill's one-hot vocabulary is fixed at training time, this condition is best understood as an exploratory upper-bound reference for semantic augmentation, rather than the final scalable representation. Coloured GP supplies an extensible alternative: its three 8-bit channels can encode designed semantics without fixing the number of skills in the policy interface.

**Interpretation and boundary.** These results support the conclusion that the most effective conditioning combines a spatial target with semantic context, and that explicit condition signals make multi-task behaviour less vulnerable to seed- and data-composition-dependent collapse. Extending a clean point with grasp rotation does not provide a clear further gain: grasp-part (`46.30±8.49%`) and coloured grasp-part (`47.53±5.58%`) remain below GP (`52.47±4.66%`), driven by a pronounced round-table-specific failure (historical maxima of only `1/36` and `4/36` in both supplemental seeds). This failure should be analysed at the grasp/skill level, not pooled into the GP conclusion. Finally, clean endpoint success establishes performance but not causal use of a condition channel. Paired colour permutation, skill swapping, GP removal/delay and stage-level completion analyses are required to determine how the policy uses spatial and semantic information internally.

## 6. 为什么 clean eval 不能回答所有问题？

Clean success rate 只测量在分布内、正确 condition 下的最终完成率，不能区分以下机制：

1. 模型是否真的使用 GP/颜色/skill，还是只依赖 RGB-D、机器人状态或任务外观。
2. 收益来自 GP 的位置、颜色携带的类别、二者组合，还是训练随机性。
3. condition 错误、交换、延迟、遮挡或带噪时，策略是平滑退化还是发生 shortcut failure。
4. 失败发生在 pick、place、insert 或 screw；endpoint SR 会把不同机制压成同一个 0/1。
5. 数据版本、depth contract 和 seed 的影响；当前三 seed 混合表尤其存在这个混杂。

因此 condition 机制至少需要 paired reset 下的 `correct / shuffled / permuted / delayed / noisy / removed` 评估，并同时报告分阶段 completion/conditional success。已有 Low→Med 和 noisy-eval 结果只覆盖旧固定 checkpoint，不能替代新数据 lineage 的多 seed 因果检查。

## 7. `rgbd+gp` 与 `rgbd+gp+skill` 的训练/数据检查

### 7.1 先排除 GP+skill 自身结构性故障

- 原 main 的三个 GP run 与一个 GP+skill run 使用**完全相同**的 `/.../rgbd-skill-1.lmdb`：300 episodes、117,267 samples；GP+skill 唯一增加的是 5 维 skill one-hot，观测维度由 272 变为 277。
- 原 main GP+skill 的 lamp 是 `20/36`，高于三个 GP run 的 `10/36、15/36、11/36`。因此 GP+skill 架构或 eval 输入并非天然坏掉。
- 当前三 seed 表中的 GP 与 GP+skill **不是相同数据训练**：GP 全是旧 lineage，GP+skill 后两个 seed 使用新 125,307-frame LMDB。这一行间比较不能用于定位 GP+skill bug。

### 7.2 新训练没有发现 GP+skill 独有的配置或优化异常

- 新的 RGB-D、skill、colored GP、GP+skill 等 condition 来自同一批 300 个 raw episode；LMDB 审计确认 episode index、low-dimensional bytes 和非渲染属性一致，条件版本只在预期的图像 patch 上不同。
- 两个 GP+skill run 都完成 epoch 2999 / 300,000 optimizer steps，配置审计全部通过。末次 train loss 为 `6.29e-4 / 6.25e-4`；validation action MSE 为 `0.0444 / 0.0284`。第二个 seed 的 val MSE 甚至略优于对应 colored GP，因此没有一致的 GP+skill 优化失败信号。
- Eval 根据 checkpoint config 自动启用 GP 图像和 skill one-hot；任务结果 JSON 中的 policy annotation resolution 正确。没有发现“训练时有 condition、eval 时漏传”的问题。

### 7.3 旧评估中的 lamp 共性失败已被相机复测推翻

旧的未对齐/FSM-gated eval 曾显示所有新 condition 的 lamp 只有 `1–6/36`，并把共同 bottleneck 指向 `bulb-base-place`。2026-09-14 统一复测后，两个 supplemental seed 的 lamp 变为：RGB-D `11/36、10/36`，RGB `10/36、8/36`，skill `13/36、12/36`，colored GP `12/36、12/36`，GP+skill `15/36、13/36`，grasp-part `14/36、11/36`，colored grasp-part `11/36、13/36`。因此“所有新 condition 在 lamp 几乎完全失败”以及据此推导的共性 place bottleneck均不再成立。

旧绝对成功率主要受 wrist train/eval mismatch 和 success gate 影响。此前进一步检查到的 lamp skill 边界行为属于 scripted FSM 的正常确认语义，按用户说明不再作为数据异常、污染或实验结论限制。

### 7.4 Lamp scripted-FSM 边界行为（正常标注，不作为问题）

对新 shared LMDB 与本机保留的旧 May `rgbd-only-skill` LMDB做逐 episode 比较：

| Lamp 数据统计 | 旧 May 数据 | 新 supplemental 数据 |
|---|---:|---:|
| episodes / lamp frames | 100 / 37,656 | 100 / 40,537 |
| episode length mean / std | 376.56 / 28.86 | 405.37 / 105.44 |
| `screw` frame 占比 | 22.01% | 38.71% |
| 第一个 `screw` segment 中位长度 | 84 帧 | 147 帧 |
| 第二个 `pick`（拿 lamp hood）中位长度 | 46 帧 | **1 帧**（标准位置上至少 61 个 episode 仅 1 帧） |
| 最后 `place` 中位长度 | 77 帧 | 64 帧 |
| 标准七阶段顺序覆盖 | 100/100 | 98/100 |
| invalid one-hot frames | 0 | 0 |

新数据的 one-hot 全部合法，98% episode 的阶段顺序正确。新旧数据在阶段长度上有差异；代表性新 episode 的 segment 为：

`push[0,37) → pick[37,98) → place[98,132) → insert[132,137) → screw[137,276) → pick[276,277) → place[277,346)`。

逐帧图像可见，在部分 episode 中 RPPO 已开始从灯泡转向 lamp hood 时，scripted annotator 仍维持 `screw`，随后才切换到 hood 的 `pick/place`。诊断图见 [新数据 GP+skill 时序](../logs/main-supplement-0906/rr_lamp_transition_232.png)；对照见 [旧 May 数据时序](../logs/main-supplement-0906/rr_lamp_transition_old_may.png)。这不是画点错误，也不是需要修复的“过期目标”，而是用户确认过的 FSM 正常边界定义：skill 切换以装配几何连续若干帧确认完成为准，不以 expert 开始下一个动作的第一帧为准。

实现链路如下：

- 这批 lamp rollout 由单任务 `/checkpoints/rppo/lamp/low/actor_chkpt.pt` 生成；RPPO policy 独立决定何时离开 bulb、转向 hood。
- `FurnitureRLSimEnv.get_skill_annotation_inputs()` 从 `already_assembled` 推导 `current_assemble_idx`；而 `already_assembled` 只有在部件相对位姿连续若干帧通过阈值后才更新。
- `SkillAnnotator.step()` 在该确认发生前仍选择 lamp bulb，并由 `LampBulb.update_skill_state()` 持续输出 `screw` 及 screw guidance point；确认后才切到 `LampHood.update_skill_state()`。
- 新 RPPO trajectory 在几何确认前已经开始走向、甚至夹住 hood；切换发生时 hood 已被夹住，于是 hood annotator 很快从 `pick` 跳到 `place`，形成大量单帧 `pick`。

因此这段现象只是 RPPO 动作进程与 scripted geometry-confirmation FSM 使用不同切换时刻的结果。几何审计通过：lamp 抽查 20 个 raw episode 无 null guidance，最大重投影误差约 `0.703 px`，LMDB image-only 审计也全部通过。报告保留这些统计作为标注定义说明，但**不再把它判为模块故障、数据污染、过拟合原因或后续修复项**。

## 8. 对 ICLR 2026 原结论的最终复核

| 原结论 | 当前判断 | 新建议 |
|---|---|---|
| Condition 能防止多任务策略退化 | **成立** | 无 condition 的原 main seed 退化成近似单任务模型，只剩数据量/有效 exposure 占优的 round_table 相对可做；同 lineage 新 seed 中，GP+skill、colored GP、skill-only overall 均高于保守 RGB-D，且三者平均增益在 lamp 最大。 |
| `GP+skill > GP = colored GP = skill` | **改写** | 不再保留等号排序。当前应写“GP + 语义信息是最佳家族”：GP+skill 第一、colored GP 第二；二者相对 GP 的提高说明语义信息有效。具体 skill-level 机制后续分析。 |
| GP+skill 是最终方案 | **改为上限参照** | 固定 one-hot skill 数量不易扩展；GP+skill 用于探索 colored GP 的理论成功率上限。Colored GP 的三通道 8-bit 颜色码可继续设计，是更可扩展的研究方向。 |
| `skill = GP` | **不成立为等价结论** | GP+skill 比 skill-only overall 高 `9.57 pp`，支持空间 GP 的作用；skill-only 只提供离散阶段，不提供操作位置。 |
| colored GP 没提供预期语义信息 | **撤回** | Colored GP 相对 GP overall 高 `4.94 pp`，与 GP+skill 相对 GP 的方向一致，支持“GP 上的语义信息有用”；因果使用方式仍需 permutation/skill-level analysis。 |
| RGB-D/RGB 因 round_table 数据更长而只会 round_table | **改写为 seed-dependent collapse** | 该现象只在原 main seed 明显。无 condition 会对 seed、数据构成和有效监督分布更敏感，可能退化到数据占优任务；新 seed 并非都只会 round_table。 |
| skill-only 的 one_leg 问题来自 Place | **撤回为总体结论** | 新两 seed one_leg 为 `29/36、27/36`，不存在稳定崩溃；阶段性原因交给 skill-level analysis。 |

## 9. 建议用于论文的当前表述

> Unconditioned multi-task policies can collapse toward a single task for some training seeds, retaining competence mainly on the task favored by the effective data distribution. Explicit conditioning mitigates this failure, with the largest aggregate gain on the hardest lamp task. The strongest family combines a spatial guidance point with semantic information: GP+skill provides an exploratory upper bound, while colored GP offers a more extensible interface whose three 8-bit color channels can encode designed semantics without fixing the skill vocabulary. Comparisons of GP+skill and colored GP against GP support the value of semantic information, whereas GP+skill versus skill-only supports the complementary value of spatial guidance. Grasp-part conditioning remains a task-specific exception on round_table and is analyzed separately.

Clean endpoint success alone仍不能说明策略在何时、以何种方式使用 condition，也不能把收益拆成 GP 位置、颜色语义或 skill 阶段的贡献。下一步在 skill-level analysis 中比较各阶段完成率与失败转移，并做 paired `shuffled / color-permuted / delayed / noisy / removed` intervention。Lamp 的 scripted-FSM 边界是正常标注语义，不列为数据故障或待修复项；grasp-part 的 round_table 失败单独审计。

## 10. 第二轮逐项核对与受控复测计划（2026-09-13）

本节记录用户指定的核对顺序和实验边界。执行顺序固定为 `1 → 3 → 2`：先确定原始 main experiment 数值与数据来源，再移除 scripted-FSM endpoint gate，最后只制定 wrist-camera 对齐复测方案；在用户批准前不得启动第 2 项 rollout。2026-09-13 的后续决定是不再为第 3 项复跑，也不再量化旧 gate 损失。

### 10.1 项目 1：原始 main experiment 表格与数据来源

主来源指定为 Notion 页面：[condition](https://app.notion.com/p/condition-3796aab8287c8034a688e8b5e8a581e1?source=copy_link)。核对时必须逐 condition 记录：

1. Notion 表格中的 task-level 成功数/成功率、checkpoint 或 run 标识、评估日期与备注；
2. 对应 checkpoint 内保存的 training config、W&B run metadata 和本地历史 eval artifact；
3. 实际 LMDB 路径、episode/sample 数和任务组成；
4. Notion 数值、历史 artifact 与当前附录 seed 1 数值是否逐格一致；
5. 数据量是否真的构成 condition 间差异。此前根据 checkpoint config 得到的 `RGB/RGB-D = 96 episodes`、多数 conditional 数据约 300 episodes 只是待核实证据，在完成 Notion 与实际 LMDB 交叉确认前，不作为最终解释。

项目 1 的输出应区分“Notion 明确记录”“checkpoint/LMDB 实证”“根据旧代码推断”和“目前不可恢复”，不能把推断写成确定事实。

#### 10.1.1 核对结论

原始 0610 main experiment 的首要来源确认为 Notion 页面[《多任务 condition 对比实验》](https://app.notion.com/p/3796aab8287c8034a688e8b5e8a581e1?pvs=204)。页面明确写明 `one_leg+round_table+lamp，3×100 traj，DiT，3000 epoch`，并记录了以下 run 和最终 clean-eval 数值：

| Condition | Run | one_leg | round_table | lamp | Overall |
|---|---|---:|---:|---:|---:|
| `rgbd` | `clear-water-12` | 0/36 | 15/36 | 0/36 | 15/108 |
| `rgbd_gp` | `autumn-dust-13` | 28/36 | 13/36 | 15/36 | 56/108 |
| `rgbd_gp_skill` | `fresh-tree-11` | 30/36 | 18/36 | 20/36 | 68/108 |
| `rgb` | `true-firefly-8` | 0/36 | 6/36 | 0/36 | 6/108 |
| `rgbd_colored_gp` | `absurd-voice-2` | 33/36 | 10/36 | 14/36 | 57/108 |

这些数值与本仓库 [multi_task_condition_eval_0610.md](./multi_task_condition_eval_0610.md) 的 A.1/A.4.1 以及 `base:/home/huyue/projects/robust-rearrangement-custom/logs/evaluate_model/` 中对应 task JSON 逐格一致。2026-09-13 通过 Tailscale 访问 `base` 后再次读取了 RGB/RGB-D 六份 JSON，确认 `clear-water-12 = 0/15/0`、`true-firefly-8 = 0/6/0`。因此当前附录采用的上述原始 seed 结果来源正确。

Notion 主表没有填写 `rgbd_skill` 数值，也没有把另外两个 GP run 和后续 grasp run 写进同一张主表。当前附录中的 `good-serenity-16`、`rare-monkey-4`、`icy-vortex-9` 与 grasp 条目来自本地 0610 报告所登记的原始 JSON，不应声称它们全部直接来自该 Notion 主表。

#### 10.1.2 数据来源结论与尚存矛盾

用户关于“原始数据设计不是少量数据”的判断得到 Notion 直接支持：最终页面明确记录每任务 100 条。此前把 `96 episodes` 简写成“原始 main 只采了 96 条”是不准确的；应区分：

- **来源/实验设计：** 三任务各 100 条，共 300 条；
- **最终 checkpoint 记录的实际 loader 输入：** `clear-water-12` 和 `true-firefly-8` 均指向 `rgbd-5.lmdb`，并保存 `n_episodes=96, n_samples=49,083, data_subset=null`；
- **conditional checkpoint：** colored GP 为 296 episodes，GP/GP+skill 为 300 episodes。

这里的 `n_episodes=96` 不是 W&B 页面展示口径，也不是单个 DDP rank 的数量。0610 当时的 `bc_ddp.py` 在启用 episode-level DDP 时，将完整 LMDB manifest 的全局 train episode 数与 validation episode 数相加后写入 checkpoint；两者之和等于 loader 实际看见的全部 episode。`49,083` 个 sequence sample 加回旧索引每 episode 少掉的 24 个尾部 anchor 后，对应约 51,387 raw frames，也与约 100 条长任务轨迹相符，而不像三任务 300 条。

所以当前最严格的结论是：**Notion 记录的源数据/目标配置为 300 条，但最终 RGB/RGB-D checkpoint 所使用的 `rgbd-5.lmdb` manifest 只向 loader 暴露了 96 条；这是最终训练输入差异，不是原始采集量差异。** 旧 `rgbd-5.lmdb` 已不在 r218，本轮尚不能逐 episode 恢复这 96 条的 task 组成。结合模型只在 round_table 成功和帧数，只能提出“很可能主要/全部是 round_table”的假设，不能写成已证实事实。

Notion 另有[《错误实验：lamp 单任务实验》](https://app.notion.com/p/36b6aab8287c80d6b9efdb4c13c5c9a1?pvs=204)，记录过早期 `data.data_subset=100` 在合并 LMDB 上只取排序前 100 条、导致单任务训练的 bug；最终 0610 页面说明这批错误实验已重跑，且 `clear-water-12` config 的 `data_subset=null`，因此不能直接认定 final RGB/RGB-D 又触发了同一个参数 bug。要彻底结案仍需找回 `rgbd-5.lmdb` 或其 `__meta__`/`__episode_index__` 备份。

**项目 1 来源：**

- [Notion：多任务 condition 对比实验](https://app.notion.com/p/3796aab8287c8034a688e8b5e8a581e1?pvs=204)
- [Notion：错误实验——lamp 单任务实验](https://app.notion.com/p/36b6aab8287c80d6b9efdb4c13c5c9a1?pvs=204)
- [0610 本地可追溯报告](./multi_task_condition_eval_0610.md)
- checkpoint/W&B config 快照：`../logs/main-supplement-0906/config_comparison/wandb_configs.json`

### 10.2 项目 3：恢复 reward-only success（已完成，不复跑）

按 2026-09-13 的最新决定，不再为 scripted-FSM gate 单独复跑，也不再追溯量化它损失了多少成功率。正式 endpoint success 已恢复为 FurnitureBench 环境的 physics reward/assembly completion：

`success = sum(reward) == n_parts_assemble`

scripted FSM 仍可用于 skill/阶段诊断和 annotation verification，但不再进入 success 的布尔判定。具体修改为：

- rollout 中用于提前结束、进度条和成功计数的 `current_success` 直接取累计 reward；
- rollout 汇总中的 `success_flags` 直接取 reward completion；
- 新生成的 task/aggregate JSON 显式写入 `success_criterion=physics_reward`，避免以后再次混淆口径；
- scripted annotation provenance 约束没有改变，数据采集仍必须显式使用 `annotation_source=scripted`。

静态编译、现有四个 semantic-success helper 测试和 `git diff --check` 已通过。没有运行新的完整 rollout。曾启动的诊断 session 已按用户要求精确停止，未产出可登记的完整结果。

由于已有 supplemental JSON 没有同时保存逐 rollout 的 physics-success 和 gated-success，不能离线精确恢复 reward-only 数值。因此第 2/3 节现有表格继续作为“已登记历史结果”保留；后续 wrist-camera 复测及所有新评估统一采用 reward-only，但不得把新值与旧 gated 值的差异全部归因于 camera。

### 10.3 项目 2：wrist-camera 对齐假设与待批准实验

当前重点假设是：补训数据中的 wrist 图像在保存时由 `320×240` center-crop 为 `224×224`，而现有 eval 路径把完整 `320×240` wrist frame resize 为 `224×224`；这会产生明显的几何/FOV domain shift，且可能更严重地影响 round_table 与 lamp。

#### 10.3.1 已确认的图像链路差异

- supplemental LMDB 中的 wrist RGB-D 已是 canonical `224×224` center crop；checkpoint 的 `WristCameraTransform` 再 resize 到 `224×224` 时等价于 identity。
- 当前 eval 从 simulator 得到 `240×320` wrist RGB-D。`src/eval/rollout.py` 先保持为 `240×320`，随后 `WristCameraTransform` 把完整矩形 frame resize/squash 到 `224×224`。
- front camera 在 train/eval 都走 center-crop 语义，主要不一致集中在 wrist。
- 因此用户提出的原因成立为一个强假设：eval wrist 相比训练输入同时改变了横向 FOV 和物体几何比例。round_table 与 lamp 更依赖精细 grasp/place，受到的影响可能比 one_leg 大；但成功率复测前仍不能把它写成已证实原因。

#### 10.3.2 待批准的实现

为避免破坏原 main checkpoint 的历史输入契约，采用显式、可追溯的 checkpoint-specific 选项，而不是全局静默改行为：

1. 为 eval 增加 `--wrist-image-transform {checkpoint,legacy-resize,center-crop-224}`，默认遵循 checkpoint config；仅本轮 supplemental checkpoint 显式使用 `center-crop-224`。
2. 在 actor 的 wrist camera transform 内，对 wrist RGB 和 depth 使用完全相同的中心窗口：从 `240×320` 取 `[8:232, 48:272]`，得到 `224×224`。RGB 不额外插值，depth 也不插值；front 路径不变。
3. 在 task JSON 中记录 wrist transform、输入/输出尺寸和 depth contract；在 run metadata 中另记 camera preset、代码 commit/diff hash、checkpoint hash 和 furniture-bench submodule commit。当前待测基准 submodule 为 `dc7f435`，执行前仍需重新核对工作树和 hash。
4. 增加 synthetic coordinate-grid 测试，验证 RGB/depth crop 坐标严格一致、输出为 `224×224`、legacy 路径和 front 路径没有改变。
5. 批量评估前做 4 个单-rollout smoke（2 conditions × 2 tasks），记录 resolved transform 与 actor 前的 RGB-D shape，并用定向测试核对 actor 内部输出；通过后才进入 36-rollout 正式评估。

#### 10.3.3 最小正式复测矩阵

| Condition | Train seed | Tasks | 每 task rollout |
|---|---:|---|---:|
| `rgbd_colored_gp` | `2026090701` | round_table, lamp | 36 |
| `rgbd_colored_gp` | `2026090702` | round_table, lamp | 36 |
| `rgbd_gp_skill` | `2026090701` | round_table, lamp | 36 |
| `rgbd_gp_skill` | `2026090702` | round_table, lamp | 36 |

正式矩阵共 `2 conditions × 2 train seeds × 2 tasks × 36 = 288` rollouts。固定：`seed=0`、`n_envs=12`、`randomness=low`、`max_rollout_steps=1000`、positive-meters depth、同一 simulator camera pose/FOV、scripted annotation、reward-only success。condition 根据 checkpoint config 自动解析，不手工覆盖。

单格命令模板（实现上述 CLI 后）：

```bash
/home/hy/anaconda3/envs/rr/bin/python src/eval/evaluate_model.py \
  --wt-path <checkpoint.pt> \
  --task <round_table-or-lamp> \
  --n-envs 12 --n-rollouts 36 --seed 0 \
  --randomness low --max-rollout-steps 1000 \
  --action-type pos --observation-space image \
  --annotate-skill --enable-annotation-verify \
  --annotation-source scripted \
  --wrist-image-transform center-crop-224 \
  --if-exists error \
  --task-summary-out <result.json>
```

计划输出目录为 `logs/wrist-camera-align-0913/`，长任务使用独立具名 tmux，并在 metadata 中记录节点、GPU、环境、两个 repo commit、checkpoint path/hash、完整命令和 resolved annotation source。

#### 10.3.4 时间与解释边界

本机 r218/3060 的最近正式日志中，36-rollout 单格通常约 11–13 分钟；因此预计：代码与 unit test 20–30 分钟，4 个 smoke 约 10–15 分钟，8 个正式 cell 约 90–110 分钟，汇总核验约 10 分钟，总墙钟约 **2–2.5 小时**。

最小 288-rollout 方案符合“对齐后重新测”的要求，但新结果是 reward-only，而当前对照表的 supplemental 数值来自旧 FSM-gated 口径。因此它能回答“对齐后的绝对成功率是否恢复”，却不能把全部增量严格分解为 wrist transform 的因果效应。若要严格隔离 wrist，需要在同一份 reward-only 代码下额外重跑 legacy-resize control，再做 paired A/B；总量会变成 576 rollouts，预计总墙钟约 **3.5–4.5 小时**。这不是第 3 项的 FSM-loss 审计，而是第 2 项的 camera ablation。

本方案随后获得用户批准，实际执行结果见下一节。

#### 10.3.5 最小方案正式结果（已完成）

用户批准最小 288-rollout 方案后，2026-09-13 22:19–23:51 在 r218 的 RTX 3060 上完成全部 8 个 cell。评估固定使用 `seed=0`、`n_envs=12`、low randomness、positive-meters depth、scripted annotation、reward-only success，以及 wrist encoder 内部的 `center-crop-224`；front transform、simulator camera pose/FOV 和 checkpoint 均未改变。

逐 seed 结果如下。括号中依次为“对齐后 reward-only / 之前登记的未对齐 FSM-gated / 成功数变化”，所以 delta 不能严格全部归因于 wrist：

| Condition | Train seed | round_table | lamp |
|---|---:|---:|---:|
| `rgbd_colored_gp` | `2026090701` | 72.22%（26/36 vs 16/36，+10） | 33.33%（12/36 vs 2/36，+10） |
| `rgbd_colored_gp` | `2026090702` | 50.00%（18/36 vs 12/36，+6） | 33.33%（12/36 vs 5/36，+7） |
| `rgbd_gp_skill` | `2026090701` | 69.44%（25/36 vs 16/36，+9） | 41.67%（15/36 vs 5/36，+10） |
| `rgbd_gp_skill` | `2026090702` | 50.00%（18/36 vs 15/36，+3） | 36.11%（13/36 vs 3/36，+10） |

两 supplemental train seed 的 mean ± sample std：

| Condition（aligned wrist, reward-only） | round_table | lamp | 两任务合计 |
|---|---:|---:|---:|
| `rgbd_colored_gp` | **61.11 ± 15.71%** | 33.33 ± 0.00% | 47.22 ± 7.86% |
| `rgbd_gp_skill` | 59.72 ± 13.75% | **38.89 ± 3.93%** | **49.31 ± 8.84%** |

所有 8 个 cell 都比之前登记值高，成功数增加范围为 `+3/36` 到 `+10/36`。按 condition/task 对两个 seed 求均值后：colored GP 的 round_table 增加 `+22.22 pp`、lamp 增加 `+23.61 pp`；GP+skill 的 round_table 增加 `+16.67 pp`、lamp 增加 `+27.78 pp`。这足以说明 wrist train/eval mismatch 是补训模型在 round_table/lamp 上异常下降的一个主要因素；但由于本轮同时按用户决定恢复了 reward-only success，没有 legacy-resize reward-only control，不能把增量精确拆分成 camera effect 与 success-criterion effect。

对 condition 排序而言，对齐后两个方法仍非常接近：只看这两个任务，GP+skill 比 colored GP 高 `2.08 pp`；colored GP 的 round_table 高 `1.39 pp`，GP+skill 的 lamp 高 `5.56 pp`。以两个 seed 和当前方差仍不能声称统一 winner。

完整机器可读结果、逐 cell 日志、checkpoint hash、命令和环境元数据登记在 `../logs/wrist-camera-align-0913/`。四个 smoke 及 8 个正式 JSON 均通过 `36 rollouts / scripted provenance / positive depth / center-crop / reward-only / tracking complete` 校验。

#### 10.3.6 先行结论修正（随后由第 11 节全量复测取代）

- “新 colored GP 和 GP+skill 过拟合到 one_leg、因而在另外两个任务上训练失败”不再是首选解释。对齐 wrist 后，两个 condition、两个 train seed、两个困难任务全部恢复，说明此前低成功率很大一部分来自 eval input domain shift。
- 第 7.4 节记录的是 lamp scripted-FSM 的正常边界语义，不再作为数据标注问题。修正 wrist 后，colored GP / GP+skill 的 lamp 绝对成功率从两 seed 均值 `9.72% / 11.11%` 提升到 `33.33% / 38.89%`。
- 对齐结果仍支持“没有统一最好 condition”：colored GP 在 round_table 仅高 `1.39 pp`，GP+skill 在 lamp 高 `5.56 pp`，都远小于 train-seed 标准差或有限样本不确定性。
- 当时暂未重测 RGB/RGB-D；用户随后决定把全部 supplemental 两 seed 统一重测，这项决定已由第 11 节执行完毕。原 main checkpoint 的 96-episode loader 问题仍只影响旧 seed 的 lineage 解释，不影响新两 seed 的统一复测有效性。

## 11. 全部 supplemental 两 seed 的统一复测（2026-09-14）

### 11.1 错误原因、修正与后续兼容规则

本轮确认需要替换第 2、3 节中所有 supplemental seed 的旧 eval，而不是只替换 colored GP 与 GP+skill 的困难任务。原因不是 checkpoint 损坏，而是评估输入契约和训练输入契约没有对齐：

1. supplemental LMDB 的 wrist RGB-D 在 pickle-to-LMDB 阶段已经把 simulator 原始 `240×320` 图像中心裁为 `224×224`；训练时 wrist transform 接收到的已是 `224×224`，后续 resize 实际为 identity。
2. 旧 eval 把 simulator 的完整 `240×320` wrist frame 直接 resize/squash 为 `224×224`。与训练相比，它保留了训练时被裁掉的左右视野，同时改变了像素几何比例。front camera 原本就是 center-crop 语义，不是本次问题来源。
3. 旧 supplemental 数值还使用了后来加入的 scripted-FSM endpoint gate，而原 main 使用 physics reward。按用户决定，统一恢复为 `sum(reward) == n_parts_assemble`；scripted FSM 只用于 annotation 与阶段诊断，不参与最终 success 布尔值。因此旧值与新值的差不能全部解释成相机 effect。
4. RGB-D eval 必须向环境请求真实 depth，并以 positive meters 传入 encoder；`--save-depth-image` 只控制是否落盘，不能再意外控制策略是否取得 depth。纯 RGB 模型不消费 depth，JSON 登记为 `depth_contract=not_applicable`。

代码修正后，后续统一按以下规则运行：

- 默认使用 `--wrist-image-transform checkpoint`，由 checkpoint 保存的 `data.image_spatial_transform` 决定。这样原 main legacy checkpoint 继续使用其历史 resize 输入，全画幅/real-sim 模型可保持 `none`，不会被全局强制裁剪。
- 对本报告的 supplemental 两 seed，训练数据已核实为 canonical 224 center crop，因此复测命令显式使用 `--wrist-image-transform center-crop-224`。RGB 与 depth 取相同的 `[8:232, 48:272]` 窗口，均不插值。
- 只有复现旧 eval 时才显式使用 `--wrist-image-transform legacy-resize`。不得根据 condition 名、RGB/RGB-D 类型或当前 simulator 分辨率猜测 transform。
- 每个结果 JSON 必须同时登记 requested/resolved wrist transform、实际输入/策略尺寸（`none` 为 `240×320`，224 resize/crop 为 `224×224`）、depth contract、`success_criterion=physics_reward`、`annotation_source=scripted` 和完整命令；缺任一项的结果不能进入最终表。
- saved rollout 仍保存未经 policy transform 改写的原始 wrist/front RGB-D；center crop 只发生在 actor 输入路径，不能把裁过的 policy tensor 当成采集原图。

### 11.2 已登记的先行结果：colored GP 与 GP+skill

先行的 8 个困难任务 cell 已在 2026-09-13 完成，结果见 10.3.5。它们符合本轮最终统一契约，将在完整 42-cell 批次中经过 hash/schema 校验后直接复用：

| Condition | Train seed | round_table | lamp |
|---|---:|---:|---:|
| `rgbd_colored_gp` | `2026090701` | 72.22% (26/36) | 33.33% (12/36) |
| `rgbd_colored_gp` | `2026090702` | 50.00% (18/36) | 33.33% (12/36) |
| `rgbd_gp_skill` | `2026090701` | 69.44% (25/36) | 41.67% (15/36) |
| `rgbd_gp_skill` | `2026090702` | 50.00% (18/36) | 36.11% (13/36) |

### 11.3 完整复测矩阵与状态

checkpoint 重新盘点后，两组 supplemental seed 的 7 个 condition 均已有 `actor_chkpt_last.pt`，包括在旧表生成后才完成的 `seed2026090702/rgbd_grasp_part`。因此本轮不再把 grasp-part 第三组写成“训练缺失”，而是将其纳入复测；最终第 2、3 节也将改为完整三 seed。

完整矩阵为 `2 train seeds × 7 conditions × 3 tasks × 36 = 1512` rollouts，共 42 个 cell。固定 `eval seed=0`、`n_envs=12`、low randomness、最多 1000 steps、scripted annotation、reward-only success；RGB-D 模型使用 positive-meters depth，所有 supplemental 模型使用 `center-crop-224`。其中 8 个 2026-09-13 的合格 cell 经过 schema 校验后复用，剩余 34 个在 `rr_main3seed_all_supplement_reeval_0914` 中执行。

批次于 2026-09-14 01:09 启动、07:30 完成。首个 RGB cell 后曾因运行中修正纯 RGB 的 metadata validator（`legacy_signed` → `not_applicable`）而退出一次；该 rollout JSON 本身完整，03:36 从原结果目录恢复后继续执行，没有重复或遗漏。最终 42/42 个 JSON 全部通过以下检查：checkpoint path、task、36 rollouts、seed 0、low randomness、scripted provenance、reward-only success、requested/resolved `center-crop-224`、`240×320 → 224×224` 尺寸、RGB-D positive-meters/RGB not-applicable depth、tracking complete 和完整命令。总成功数为 `791/1512`。

第 2 节 mean ± sample std、第 3 节逐 seed 表、第 4 节同-lineage 表以及第 5/8/9 节结论均已更新。完整 before/after 逐 cell 对比保存在 `../logs/main-supplement-reeval-0914/before_after_comparison.md`，机器可读结果在 `../logs/main-supplement-reeval-0914/results/`。

### 11.4 旧登记与统一复测的总体变化

下表只比较两个 supplemental seed；`before` 同时包含错误 wrist 输入和 FSM gate，`after` 同时修正了两者，因此 delta 不能拆成单一 camera effect。

| Condition | Before overall | After overall | 变化 |
|---|---:|---:|---:|
| `rgbd_colored_gp` | 46.76% | 59.72% | +12.96 pp |
| `rgbd_gp_skill` | 45.83% | 63.43% | +17.60 pp |
| `rgbd_skill` | 43.52% | 54.17% | +10.65 pp |
| `rgbd` | 44.91% | 51.39%（eval seed 0/1 逐格 min） | +6.48 pp |
| `rgb` | 37.50% | 50.00%（eval seed 0/1 逐格 min） | +12.50 pp |
| `rgbd_grasp_part` | 不可比（旧表仅一组 supplemental） | 41.67%（历史逐格 max） | — |
| `rgbd_grasp_part_colored` | 37.04% | 44.91%（历史逐格 max） | +7.87 pp |

主要模式不是所有 task 一起上涨：round_table/lamp 普遍恢复，而 one_leg 有升有降。两个 grasp-part 方法是明确例外：lamp 为 `30.56–38.89%`，但新两 seed 的 round_table 历史 max 仍只有普通 grasp `1/36、1/36`、colored grasp `4/36、4/36`。因此 wrist mismatch 是此前跨 condition 困难任务低值的主要原因之一，但不是 grasp-part/round_table 低值的完整解释。

## 12. RGB/RGB-D 高分复核与 grasp 登记口径（2026-09-14）

### 12.1 旧/新 training config 与 padding 的有效差异

对原 main `true-firefly-8`（RGB）、`clear-water-12`（RGB-D）与 supplemental `2026090701/02` checkpoint 保存的 resolved config 做字段级比较后，DiT 架构、horizon、优化器、batch size、学习率、scheduler、训练步数、augmentation 和 min-max normalization 的名义设置一致：`obs_horizon=1`、`pred_horizon=32`、`action_horizon=8`、global batch `512`、`3000 epochs × 100 steps`、actor/encoder LR 均为 `1e-4`。但这不表示有效训练目标相同，因为 2026-09-02 的数据管线修改改变了 `pad_after=true` 的实际含义。

原 main 训练代码在 `BaseSequenceDataset._build_indices()` 中使用：

```text
pad_after = action_horizon - 1 = 7
```

sequence length 为 `32`，因此每条 episode 只产生 `L - 24` 个 observation anchor，系统性丢失最后 `24` 个 anchor；同时 padding 出来的尾部 action 仍进入未加 mask 的 MSE。原 RGB/RGB-D checkpoint 登记 `96 episodes / 49,083 samples`，加回 `96 × 24 = 2,304` 后对应约 `51,387` raw transitions，平均 `535.3 frames/episode`。这个长度与新 round_table 的 `54,029 / 100 = 540.3 frames/episode` 高度接近，而明显不同于新 one_leg 的 `307.4` 和 lamp 的 `405.4`，支持“旧 96 条主要或全部来自 round_table”的判断；但旧 LMDB manifest 尚未找回，因此仍不能把 task identity 写成已完全证实。

Supplemental 训练使用 2026-09-02 之后的实现：

```text
pad_after = pred_horizon - 1 = 31
```

它保留每个有效 observation anchor，并通过 `action_valid_mask` 将 episode 末端之后的 padding action 排除出 diffusion loss。对 round_table，这一变化恢复的是每条成功轨迹末段、即第二次 `base-leg` 装配附近的训练 anchor，因此不能只把它解释成样本总量增加；它同时改变了末段动作的监督分布与 loss。此项差异不会出现在普通 YAML/config diff 中，是旧/new RGB、RGB-D 结果不可作为纯 train-seed 重复的一个重要原因。

另外还有三项有效训练差异：旧 wrist 输入为 `240×320` 后 resize/squash 到 `224×224`，新 LMDB 为预先 center-crop 的 `224×224`；旧 gripper proprioception 是约 `[0, 0.065] m` 的连续值，新数据编码为 `open=-1 / closed=+1`；新 RGB-D 使用 positive-meters depth 与 dataset statistics，旧 RGB-D 使用 legacy depth contract。前两项也影响纯 RGB，第三项只影响 RGB-D。新 round_table 仅占 supplemental samples 的 `54,029 / 125,307 = 43.1%`，而旧 96 条若确为 round_table 则获得更多 round_table-specific optimizer exposure，所以当前高分不能归因于“100×3 数据更多”。

### 12.2 独立 eval seed 复测与保守 min 口径

为检查 supplemental RGB/RGB-D 高分是否只来自 `eval seed=0`，在完全相同的 aligned-wrist、reward-only 协议下使用独立 `eval seed=1` 复测两个 train seed 的全部三项任务。训练采集的 task seeds 为 `22903000 / 23003000 / 23103000`，与 eval seeds `0/1` 不重合；评估不使用 `--init-state-file`，因此不是训练初始状态复用。

按用户指定，最终表格对 RGB/RGB-D 的每个 `train seed × task` 取两次 36-rollout 结果中的较小成功数：

```text
conservative count = min(eval-seed-0 count, eval-seed-1 count)
```

这不是 pooled 72-rollout 成功率，也不是两次 eval 的均值，而是用于当前结果检查的保守下界登记。12-cell 队列于 2026-09-14 13:43 启动；在完成 7 格后因另一项未使用共享 lock 的 GPU smoke 重叠而收到 `KeyboardInterrupt`，释放 GPU 后从缺失项续跑，已有结果只校验、不覆盖，最终于 15:56 完成 12/12。所有 task JSON 均通过 checkpoint、36 rollouts、simulator seed 1、low randomness、scripted provenance、reward-only、`center-crop-224`、RGB-D positive-meters/RGB not-applicable depth 和 tracking-complete 校验。

逐格原始结果与保守登记如下；括号内为成功数/36：

| Condition | Train seed | Task | eval seed 0 | eval seed 1 | 登记 min |
|---|---:|---|---:|---:|---:|
| `rgb` | `2026090701` | one_leg | 32 | 34 | **32** |
| `rgb` | `2026090701` | round_table | 17 | 15 | **15** |
| `rgb` | `2026090701` | lamp | 10 | 8 | **8** |
| `rgb` | `2026090702` | one_leg | 30 | 34 | **30** |
| `rgb` | `2026090702` | round_table | 18 | 21 | **18** |
| `rgb` | `2026090702` | lamp | 8 | 5 | **5** |
| `rgbd` | `2026090701` | one_leg | 33 | 30 | **30** |
| `rgbd` | `2026090701` | round_table | 21 | 18 | **18** |
| `rgbd` | `2026090701` | lamp | 11 | 7 | **7** |
| `rgbd` | `2026090702` | one_leg | 29 | 32 | **29** |
| `rgbd` | `2026090702` | round_table | 19 | 17 | **17** |
| `rgbd` | `2026090702` | lamp | 10 | 10 | **10** |

对两个 supplemental train seed 计算 mean ± sample std 后：

| Condition（逐格 min） | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|
| `rgb` | 86.11 ± 3.93% | 45.83 ± 5.89% | 18.06 ± 5.89% | 50.00 ± 1.31% |
| `rgbd` | 81.94 ± 1.96% | 48.61 ± 1.96% | 23.61 ± 5.89% | 51.39 ± 0.65% |

独立 seed 复测没有把高分推翻：RGB round_table 两个 train seed 在 eval seed 1 为 `15/36、21/36`，RGB-D 为 `18/36、17/36`。逐格取 min 后，RGB/RGB-D round_table 仍分别为 `45.83%/48.61%`，明显高于原 main 单 seed 的 `16.67%/41.67%`；尤其 RGB 的差异不能由 depth 修复解释。第 2–5、8–9、11 节中依赖 RGB/RGB-D 的表格、delta 和文字结论已统一改用本节 conservative-min 口径。完整机器可读结果、逐 cell 日志、命令、checkpoint hash 和运行元数据位于 `../logs/rgb-rgbd-highscore-reeval-0914/`。

### 12.3 Grasp condition 的历史逐格 max 口径

用户指出 grasp-part 的 round_table `0/36` 曾有更高历史结果。复核确认：`0/36` 不是 RGB/RGB-D 的 min 规则误伤，而是 2026-09-14 aligned-wrist 统一复测的实际结果。此前 positive-depth、相机未对齐的历史评估中，普通 grasp 的 seed `2026090701` 为 `1/36`，colored grasp 两个新 seed 都为 `4/36`；普通 grasp seed `2026090702` 在历史/统一复测中均为 `1/36`。

按用户指定，本报告对两个 grasp condition 的每个 `train seed × task` 在可追溯历史评估之间取最大成功数，而不只修正 round_table。新两 seed 的取值如下：

| Condition | Train seed | one_leg（历史 / aligned → max） | round_table（历史 / aligned → max） | lamp（历史 / aligned → max） |
|---|---:|---:|---:|---:|
| `rgbd_grasp_part` | `2026090701` | 33 / 30 → **33** | 1 / 0 → **1** | 1 / 14 → **14** |
| `rgbd_grasp_part` | `2026090702` | 0 / 30 → **30** | 1 / 1 → **1** | 1 / 11 → **11** |
| `rgbd_grasp_part_colored` | `2026090701` | 30 / 30 → **30** | 4 / 0 → **4** | 4 / 11 → **11** |
| `rgbd_grasp_part_colored` | `2026090702` | 35 / 30 → **35** | 4 / 0 → **4** | 3 / 13 → **13** |

历史 JSON 位于 `../logs/rgbd-depthfix-0906/main_supplement_eval/`；普通 grasp seed `2026090702` 的历史结果位于 `../logs/main-supplement-0906/eval/`；aligned 结果位于 `../logs/main-supplement-reeval-0914/results/`。由此得到 supplemental-only：普通 grasp `41.67±3.93%`、colored grasp `44.91±4.58%` overall；合入原 main seed 后分别为 `46.30±8.49%` 与 `47.53±5.58%`。由于 max 横跨不同 eval 输入契约，这只是用户指定的乐观结果登记规则，不能与统一协议下的单次估计量混同。
