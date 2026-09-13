# ICLR 2026 实验结果与论文叙事整理

> 整理日期：2026-09-12（VLM 覆盖范围噪声实验更新）<br>
> 代码基线：`main@19ab7cc3c4eaa841c6c0d4751dba9dc40c3e1889`（整理时与 `origin/main` 一致）<br>
> VLM 覆盖范围噪声 campaign 基线：`main@f6877ff75418b4f4af835f3b4a9dfecea0ffc9ab`<br>
> 原始文档：`D:\ZJU\研一春夏\ICLR2026\实验结果整理\实验结果整理.md`<br>
> 说明：§2.1–2.2 保留两张源表；§2.3 已用三个评测 replicate、108 rollout/cell 的 VLM 覆盖范围噪声实验替换早期 seed-0 结论；§2.4 补充 skill-level 分析及其表图。

## 1. 论文想讲的故事

长时程家具拼装要求机器人在不同装配任务和操作阶段之间切换，同时准确定位当前需要交互的零件及其目标位置。因此，一个共享动作策略必须同时回答两个相互关联的问题：当前应该执行哪一项操作，以及应该在何处执行这一操作。仅依赖 RGB 或 RGB-D 观测时，策略需要从高维视觉输入中隐式恢复任务语义和空间目标，容易在多任务训练中偏向某个数据占优的任务。one-hot skill condition 可以直接提供任务或阶段语义，却不包含目标位置；二维 guidance point（GP）则将任务相关的空间目标显式投影到图像中，为低层连续控制提供一个紧凑、直观且可诊断的接口。

本文以多任务家具拼装为研究场景，系统比较 DiT action expert 的多种条件形式，包括 GP、colored GP、one-hot skill condition 以及 6D grasp-part annotation，并进一步考察这些条件对空间分布变化和标注误差的响应。现有结果并不支持“某一种条件在所有设置下都占优”这一简单结论，而是揭示了一条更具体的规律：显式条件使共享策略获得覆盖多个任务的能力，其中 `rgbd+GP+skill` 在 low randomness 主实验中取得最高总体成功率，而 GP 类条件，尤其是 colored GP，在连续数值噪声下表现出更稳定的成功率。由此，本文的核心主张不是 GP 已经解决了空间泛化，而是 GP 为高层语义定位与低层动作生成之间提供了一个有效且可分析的空间接口，其优势和失效边界可以通过受控扰动实验被直接测量。

实验论证围绕这一主张逐步展开。首先，在不额外改变零件空间分布的多任务评估中，RGB 和 RGB-D 基线只在个别任务上获得成功，而加入 GP 或 one-hot skill condition 后，同一个策略能够覆盖全部三个任务。这一结果与“显式条件缓解了多任务策略中的任务混淆”一致。随后，单任务与多任务对比表明，多任务训练并未系统性提高 round_table 的单任务成功率，因此主实验更适合被解释为“显式条件支持共享策略覆盖多个任务”，而不是“数据合并本身提高了单任务性能”。

为了检验这种能力能否延伸到训练分布之外，我们进一步从零件初始位姿变化和 guidance 噪声两个维度进行评估。在 Low→Med 实验中，用于随机化零件初始位姿的最大外力由 0.2 提高到 0.5，最大扰动力矩由 0.007 提高到 0.01。所有方法的成功率均明显下降，说明当前策略仍然受限于零件初始位姿的分布外变化。带显式条件的多任务策略保留了少量成功案例，而且部分失败 rollout 中仍可观察到目标跟踪行为；然而，每个任务仅有 12 个 rollout，现有结果只能说明空间条件可能提供局部帮助，尚不足以可靠比较不同条件的空间泛化能力。

最后，clean-train → noisy-eval 实验将上游定位误差显式注入空间条件，以检验这一接口在实际部署中的容错性。按各自 VLM-equivalent σ 的真实工作区间比较，colored GP 在 point family 的 round_table/lamp 上保持高于 plain GP 的 task success，`GP+skill` 则提供了显式 low-dimensional semantic information 的参考上限；grasp-part 保留 clean advantage，但同时承受位置与旋转的联合误差。n0→n7 的 pooled 变化介于 `-6.2` 与 `+4.0 pp`，Wilson 区间重叠，说明噪声幅度增大或 Shuffle 并没有带来反常成功率提升。更重要的是，n7 的 position tracking 在 15/15 个配对单元均高于 Shuffle，平均差 `11.0 cm`：大数值噪声下模型仍尝试当前 subtask，Shuffle 下则跟随被置换的 semantic guidance。Point VLM formal guidance 的成功率为 `181/324=55.9%`，Grasp VLM 为 `123/216=56.9%`；结合 n7 已覆盖典型 VLM σ 的结果，这条证据链支持双系统在本 benchmark 内完成端到端闭合，而不是只报告接口通过。

这些结果共同指向目标系统 `VLM → 2D guidance point → DiT action expert`：上游 VLM 负责把语义目标转化为空间点，下游 action expert 负责连续控制。后续端到端评估应沿用同一条证据链，分别测量 VLM 的打点误差、action expert 对不同误差类型和幅度的响应，以及完整系统的任务成功率，从而区分定位瓶颈与控制瓶颈。

## 2. 实验结果

> [!NOTE]
> §2.1–2.2 整理历史源表；§2.3 汇总已完成并通过严格审计的 VLM 覆盖范围噪声实验；§2.4 为 skill-level 分析，包含来源、统计口径及解释局限。

### 2.1 多任务 condition 主实验

来源：[`multi_task_condition_eval_0610.md`](./multi_task_condition_eval_0610.md#1-总览)

设置：`one_leg + round_table + lamp` 多任务，DiT diffusion policy，每个任务 100 条训练轨迹；评估使用 `N_ENVS=3, N_ROLLOUTS=36`。

#### 表 1：不同 condition 的多任务成功率

| # | 实验 | RUN_ID | one_leg | round_table | lamp | Overall |
|---|---------|--------|---------|-------------|------|---------|
| 1 | rgbd+only skill | good-serenity-16 | 77.78% (28/36) | 47.22% (17/36) | 33.33% (12/36) | 52.78% (57/108) |
| 2 | rgbd | clear-water-12 | 0.00% (0/36) | 41.67% (15/36) | 0.00% (0/36) | 13.89% (15/108) |
| 3 | rgbd+colored GP | absurd-voice-2 | **91.67% (33/36)** | 27.78% (10/36) | 38.89% (14/36) | 52.78% (57/108) |
| 4a | rgbd+GP | rare-monkey-4 | 83.33% (30/36) | 33.33% (12/36) | 27.78% (10/36) | 48.15% (52/108) |
| 4b | rgbd+GP | autumn-dust-13 | 77.78% (28/36) | 36.11% (13/36) | 41.67% (15/36) | 51.85% (56/108) |
| 4c | rgbd+GP | icy-vortex-9 | 86.11% (31/36) | **55.56% (20/36)** | 30.56% (11/36) | 57.41% (62/108) |
| 5 | rgbd+GP+skill | fresh-tree-11 | 83.33% (30/36) | 50.00% (18/36) | **55.56% (20/36)** | **62.96% (68/108)** |
| 6 | rgb | true-firefly-8 | 0.00% (0/36) | 16.67% (6/36) | 0.00% (0/36) | 5.56% (6/108) |
| 7 | rgbd+grasp-part | morning-glitter-1 | **86.11% (31/36)** | 38.89% (14/36) | 41.67% (15/36) | 55.56% (60/108) |
| 8 | rgbd+grasp-part-colored | eternal-cosmos-2 | 80.56% (29/36) | **44.44% (16/36)** | 33.33% (12/36) | 52.78% (57/108) |

#### 原报告结论（逐字保留）

1. gp, skill one-hot 两种引导信息都能让多任务模型有区分不同任务的能力。
2. 多任务泛化性 rgbd+skill+gp > rgbd+gp = rgbd+colored gp = rgbd+only skill。skill 级别的信息就能够做到辅助拟合，当前实验设定并不能体现出 gp 的优势。
3. 目前 skill = gp，想让 gp > skill 的两个方向：
    1. 提升任务泛化难度，med rand 和 permutation 肯定会有帮助。
    2. gp+grasp，提供 rot 信息。
4. colored 并没有如预期地提供类似 one-hot skill 的信息。可能的原因：
    1. 信号大小：2px 点占图像的 0.008%。ResNet 的 32× 下采样后，这个点变成了 ~0.06 feature-map-pixel。颜色差异（红 vs 黄）在 deep features 里几乎不可区分。
    2. 信息量 vs 通路带宽：skill one-hot 贡献 5 个独占维度（531 维 conditioning 中的 5 维），而 colored GP 的 1 bit 信息和整个场景（物体、机械臂、桌面）"挤"在同一个 512-dim visual feature 空间里竞争。
    - 改成红色和蓝色的点来区分可能更好，单独放两个通道。但是大小无法解决
5. 为什么 rgbd 和 rgb 过拟合到 round_table 而不是最简单的 one_leg？轨迹更长，数据更多。
6. rgbd+skill 为什么做不好 one_leg？瓶颈在 place，失去点的位置引导之后，policy 的插入精度大幅下降。但是其他两个任务因为数据更多而受到的影响更小。

#### 整理意见（不属于原报告结论）

> 表 1 能直接支持显式条件帮助共享策略区分并覆盖多个任务。若后续将 condition 之间的排序写入论文主结论，建议为除纯 GP 外的条件补充多 seed 结果或置信区间；colored GP 的信号尺寸与通路带宽解释目前属于机制假设。

### 2.2 Low-train → Med-eval 空间泛化

来源：[`low2med_generalization.md`](./low2med_generalization.md#results)

**Eval completed: 2026-06-27**. All models trained on low randomness, evaluated on med randomness, 12 rollouts per task.
> Additional med eval appended on 2026-07-04: `good-serenity-16` (checkpoint config is `rgbd-only-skill`).
> Additional med eval appended on 2026-07-07: `morning-glitter-1` (checkpoint config is `rgbd-skill-grasp-part`, `annotate_grasp_part=True`, `annotate_skill_one_hot=False`, `annotate_guidance_point=False`).
> Additional med eval appended on 2026-07-08: `eternal-cosmos-2` (checkpoint config is `rgbd-skill-grasp-part-colored`, `annotate_grasp_part=True`, `annotate_guidance_point_colored=True`, `annotate_grasp_colored=True`, `annotate_skill_one_hot=False`).

#### 表 2：Low → Med Randomness Generalization

| Type | Condition | RUN_ID | one_leg | round_table | lamp | **Overall** |
|------|---------|--------|:---:|:---:|:---:|:---:|
| mt-bc | rgbd+gp | autumn-dust-13 | 25.00% (3/12) | 0.00% (0/12) | 0.00% (0/12) | **8.33% (3/36)** |
| mt-bc | rgbd+gp+skill | fresh-tree-11 | 8.33% (1/12) | 0.00% (0/12) | 0.00% (0/12) | **2.78% (1/36)** |
| mt-bc | rgbd+colored gp | absurd-voice-2 | 16.67% (2/12) | 0.00% (0/12) | 0.00% (0/12) | **5.56% (2/36)** |
| mt-bc | rgbd+only skill | good-serenity-16 | 8.33% (1/12) | 0.00% (0/12) | 0.00% (0/12) | **2.78% (1/36)** |
| mt-bc | rgbd | clear-water-12 | 0.00% (0/12) | 0.00% (0/12) | 0.00% (0/12) | **0.00% (0/36)** |
| mt-bc | rgb | true-firefly-8 | 0.00% (0/12) | 0.00% (0/12) | 0.00% (0/12) | **0.00% (0/36)** |
| mt-bc | rgbd+grasp-part | morning-glitter-1 | 25.00% (3/12) | 0.00% (0/12) | 0.00% (0/12) | **8.33% (3/36)** |
| mt-bc | rgbd+grasp-part-colored | eternal-cosmos-2 | 16.67% (2/12) | 0.00% (0/12) | 8.33% (1/12) | **8.33% (3/36)** |
| rppo | one_leg | — | 25.00% (3/12) | — | — | **25.00% (3/12)** |
| rppo | round_table | — | — | 16.67% (2/12) | — | **16.67% (2/12)** |
| rppo | lamp | — | — | — | 8.33% (1/12) | **8.33% (1/12)** |
| st-bc | rgbd+gp | dauntless-breeze-2 | — | 8.33% (1/12) | — | **8.33% (1/12)** |
| st-bc | rgbd+only skill | misunderstood-firebrand-6 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgbd+gp+skill | vocal-bush-11 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgbd+colored gp | gentle-fog-7 | — | 8.33% (1/12) | — | **8.33% (1/12)** |
| st-bc | rgbd | breezy-rain-3 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgb | fiery-snowball-4 | — | 0.00% (0/12) | — | **0.00% (0/12)** |

#### 原报告结论（逐字保留）

1. 从成功率看起来 guidance point conditioned 更能做空间上的泛化
2. 然后看 failure case 的话，单任务 colored guidance point 会比单色 guidance point 的点跟随更好
3. colored gp 和 gp 主要的 failure case 是 grasp 失败或者 grasp pose OOD 导致 place 失败。
4. 光是空间上随机性增加一点成功率已经掉完了，拼装上 zero-shot 新任务泛化肯定是做不了。

#### 整理意见（不属于原报告结论）

> 每个 task 只有 12 个 rollout，单次成败对应 8.33 个百分点。因此，原报告关于 GP 空间泛化和 failure case 的观察可以保留，但若用于论文中的方法排序，建议增加 rollout 数、配对 reset seed 和多训练 seed。

### 2.3 Clean-train → noisy-eval

正式来源：[`annotation_noise_vlm_cover_108.md`](./annotation_noise_vlm_cover_108.md)；早期 seed-0 对照见 [`annotation_noise_clean_train_fresh36.md`](./annotation_noise_clean_train_fresh36.md#1-结果图)。

> [!IMPORTANT]
> **VLM 覆盖范围补充实验已完成并通过严格审计。** 新 manifest 包含预期的 111/111 个 invocation，全部成功、无重复 key、无 summary/path collision。合并只读 legacy seed-0 后，n0–n7、Shuffle 的每个 `condition × level × task` 均为 108 rollout；两个 grasp condition 的 r180 每 task 也为 108 rollout。

> [!NOTE]
> **成功率与 tracking 的样本量不同。** 所有 success cell 都汇总三个评测 replicate、每 task `n=108`。n0–n4 与 Shuffle 的正式 tracking/diagnostics 只使用新 seed-1/2，故每 task `tracking_n=72`；n5–n7 与 r180 使用三个 replicate，`tracking_n=108`。旧 saved-8 tracking 未混入。三个 replicate 是同一 checkpoint 的评测 seed，不是独立 training seed。

> Numeric-noise tracking 以 policy 实际接收到的 displayed noisy target 为参照，VLM formal rollout 则以 clean-GT target 为参照；因此 n7 的大 tracking error 表示末端执行器与错误目标之间的距离，不是直接的 clean-GT 误差。

#### 图 1：task-level success rate at the real noise scale

![VLM-cover task-level success](figures/vlm_cover_108/vlm_cover_108_success_numeric.png)

图 1 的三个分面分别对应 `one_leg`、`round_table` 和 `lamp`；横轴使用真实 position σ/axis，右侧仅保留 `n7→Shuffle` 的 categorical endpoint。图中每个 task 只标出 Point VLM 与 Grasp VLM 两条 position-equivalent σ 线；Grasp 的 orientation-equivalent scale 只在文字和附录说明，不在图上增加第三条纵线。曲线来自已校验表，不展示 replicate 散点。

#### 图 2：pooled tracking response

![VLM-cover pooled tracking](figures/vlm_cover_108/vlm_cover_108_tracking_error.png)

图 2 按 fresh36 指标显示 position、orientation 和 total tracking error。它保留 `n7→Shuffle` 对照，用于区分“同一 subtask 的数值目标偏移”和“被置换 subtask 的 guidance”。Numeric-noise tracking 以 policy 实际接收的 displayed target 为参照，因此 n7 的大 tracking error 不是到 clean-GT 的距离。

三 task pooled success 图和 pooled position tracking 图移至附录，不在正文重复；它们只用于给出跨任务的总体摘要。n0–n7 进入 ordinal 描述性趋势，Shuffle 与 r180 不进入连续拟合；grasp n0–n7 同时改变位置和旋转，只能解释为联合扰动。

#### 108 实验的主要结论（对应补充报告 §1.4）

##### Motivation

完全 clean evaluation 对检验策略能否使用一个 condition 是必要的，但它把 guidance point 设为正确且无偏的 oracle target，无法区分不同 condition 在真实部署中的优势。部署时 VLM 可能预测错误的 skill 或 semantic type，即使语义正确，二维目标也可能偏离真正的交互位置。因此，本实验将上游 guidance 不完美转化为可测的 test-time perturbation，比较各 condition 在各自真实 VLM 误差尺度附近能否继续完成长时程装配。

##### Experimental setting and evaluation protocol

我们在 `one_leg`、`round_table` 和 `lamp` 上评估 `rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`、`rgbd+grasp-part` 和 `rgbd+grasp-part-colored`。所有 condition 使用相同的多任务 demonstrations、DiT action-expert architecture 和优化流程；每个 `condition × noise × task` 汇总三个 evaluation replicate，每格 `108` 条 rollout，三 task pooled cell 为 `324` 条 rollout。n0–n7 的 position σ/axis 为 `0、3、6、12、24、48、96、192 mm`；grasp 同时绑定 `0、2.5、5、10、20、40、60、90°` 的 orientation schedule。Shuffle 替换 semantic state 的 guidance，r180 只改变 rotation。Task success 为完整装配完成，skill success 为 `completed/entered`，tracking 取每个 semantic segment 的最终有效状态。所有图严格遵循 `JSON → tables → figures`，图只读取通过 `table_validation.csv` 的结果表，本轮 34 项校验全部为 `ok`。

##### Analysis

Point family 的真实工作区间由 Point-VLM σ 给出：one_leg 为 `36.15`、round_table 为 `72.86`、lamp 为 `68.34 mm/axis`，分别落在 n4–n5、n5–n6 和 n5–n6。colored GP 在这些窗口中于 round_table/lamp 高于 plain GP，one_leg 接近持平；对应 position tracking 差异不超过 `0.35 cm`，因此 colored GP 的主要收益体现在 task completion，而不是单纯缩短末端到 target 的距离。skill-level 的优势集中在 pick 与 screw，push、place 和 insert 则方向混合。`rgbd+GP+skill` 在 n6 的跨 task place 为 `88.0%`，高于 colored GP 的 `82.7%` 和 GP 的 `82.9%`，说明显式 low-dimensional skill 不带来减益；它应被视为 colored GP 的 semantic-information reference，而不是统计意义上的性能上界。

Grasp family 的 VLM σ 更大，位于 n6–n7。`grasp-part` 相对 GP 的 clean n0 advantage 为 `+8.9 pp`，在 n6/n7 仍为 `+4.6/+3.1 pp`，但高噪声下的正向差距主要来自 lamp，one_leg 与 round_table 混合或接近持平。Grasp 同时传递位置和旋转；n6→n7 的 pooled position/orientation/total tracking 从 `14.86→25.12 cm`、`98.4→124.8°` 和 `34.5→50.1` 增长。orientation-equivalent scale 只作为文字与附录诊断，不在图中展示。

n0→n7 的 pooled success 变化范围为 `-6.2` 到 `+4.0 pp`，Wilson 区间重叠，因而没有证据表明噪声幅度增加或 Shuffle 造成反常的成功率提升。相反，n7 的 position tracking 在 `15/15` 个配对单元均高于 Shuffle，平均差 `11.0 cm`。大数值噪声保留当前 semantic subtask、只把同一目标推远；Shuffle 改变点所代表的 subtask，tracking 反而更低，这与模型跟随被置换 guidance 并尝试另一动作一致。该解释是行为层面的推断，不等同于已证明内部 gating 机制。

Formal VLM guidance evaluation 中，Point family 完成 `181/324=55.9%` 个 rollout，Grasp family 完成 `123/216=56.9%`；其中 colored GP 为 `64/108=59.3%`，grasp-part 为 `68/108=63.0%`。结合 n7 的 `192 mm/axis` 已覆盖六条 task-level VLM position-equivalent σ，这说明上游 VLM 的典型点误差可以穿过下游 action expert 并形成完整任务。这个系统级结论仍限于 RMS-equivalent 的典型尺度，不延伸到 skill-specific p95 长尾或 failure/OOD 时序。

##### Conclusion

在真实 VLM 条件下，colored GP 是 point family 中最有说服力的稳健接口，但优势依赖 task 和 skill；`GP+skill` 提供了显式语义信息的参考上限。Grasp 在 clean 条件下带来额外收益，尤其体现在 place/screw，但需要同时承受位置与旋转误差，因此强噪声下相对优势收窄。总体上，二维 guidance point 为 VLM visual-semantic grounding 与连续 action generation 提供了可测的外部空间参考：当点的语义仍指向当前 subtask 时，action expert 能在显著空间偏移下继续完成任务，并在 Shuffle 下对被置换 guidance 作出响应。该结论的边界是当前 benchmark 覆盖的 RMS-equivalent 典型误差；skill-specific p95 长尾、failure/OOD 状态和更严格的 paired-reset 机制仍需单独验证。

**整理意见（不属于原报告结论）**

> 下列历史条目解释早期单 seed 曲线，不再作为论文的正式噪声结论。正式数字、图和数据口径均以 `annotation_noise_vlm_cover_108.md` 及 `reports/data/vlm_cover_108/` 为准。

### 2.4 skill-level 成功率分析

来源与范围：由[主实验报告 §3.7](./multi_task_condition_eval_0610.md)整理，更新于 2026-09-04。主实验 condition 对比使用原主实验批次；grasp 单独采用八月 clean n0、每 task 36-rollout 的重评，不与主实验 GP 三次运行 pooled 混用。§2.1 原样摘录的七月 grasp 历史行不作为本节证据；此前有覆盖或口径问题的 tracking 不用于本节结论。

长时程家具拼装的任务级成功率同时受多个操作阶段影响，因而无法揭示 condition 的收益是否普遍分布于各个步骤。为解释主实验中整体性能提升的来源，我们首先比较 RGB-D+GP 与 RGB-D、RGB-D+skill 与 RGB-D 的 skill-level 条件完成率，考察额外 condition 是否使不同步骤普遍改善，还是主要改变了某些阶段的表现。随后比较 RGB-D+GP+skill 与 RGB-D+skill，进一步判断在已有阶段信息时，空间 point 还能帮助哪些操作。分析同时覆盖跨任务汇总和单任务拆解，以检验总体趋势是否适用于不同家具，并定位单任务中特别困难或差异更突出的步骤。

实验基于 FurnitureBench 的 one_leg、round_table 和 lamp 三个家具拼装任务，从随机化的初始零件分布出发执行完整任务，并依据评测记录的完整语义标签计算条件完成率 C/R。其中 R 为到达该标签的 rollout 数，C 为按既定进度规则完成该标签的 rollout 数；同一 rollout 内同一标签至多计一次。主分析比较 RGB、RGB-D、RGB-D+skill、RGB-D+GP、RGB-D+colored GP 和 RGB-D+GP+skill，在 task 内或跨 task 分别对同类步骤求 ΣC/ΣR，而不平均不同步骤的百分比。每个 checkpoint/task 评测 36 条 rollout；plain GP 汇总三个 checkpoint，其余主实验 condition 各一个。`skill` 专指显式低维阶段信息。主实验的原始来源为 base 上八条 run 的 24 份 task JSON，路径与远程核验记录见[主报告 A.4–A.5](./multi_task_condition_eval_0610.md#a4-查证路径)。Grasp 的协议、审计和待补分析单列于本节正文之后，不用于支撑下述主结论。

两类 condition 的收益并不是各步骤的普遍、均匀提升，而是在跨任务汇总中最集中于 Push 和 Pick。相对 RGB-D，RGB-D+GP 的 Push 从 52/79（65.82%）提高到 322/324（99.38%），Pick 从 63/109（57.80%）提高到 504/525（96.00%），分别增加 33.56 和 38.20 个百分点；RGB-D+skill 对应为 105/108（97.22%）和 163/171（95.32%），分别增加 31.40 和 37.52 个百分点。相比之下，Place 的增幅分别为 10.62 和 3.95 pp，Screw 则未呈现一致提升（图 3）。因此，“主要改善前段操作”是这一比较得到的结果，而不是预先限定的实验问题。

分任务结果进一步说明，整体趋势并不意味着每个家具都以相同方式受益（附录表 B.1）。one_leg 的前置 `top-leg-pick` 在 RGB-D 下仅为 7/36，加入 skill 或 GP 后均达到 100%；lamp 的 Push 从 16/36 提高到两种 condition 下的 100%。round_table 的 RGB-D Push 已为 36/36，因此没有同样的提升空间；而 lamp 的 Place 在两种 condition 下也有较大的变化，不能将所有收益都归于前段。一个可能的解释是，GP 和 skill 都有助于 policy 区分当前任务或操作阶段，但完成前序步骤并不意味着后续操作已经获得了标准化的入口状态。

例如，Pick 达到成功判据，只说明零件已被抓起，并不保证每次抓取后零件相对于夹爪的位置、角度都相同。前序 Push 或 Pick 即使成功，也可能留下不同的零件位姿与抓取偏差，使后续步骤需要适应额外的状态变异。这里的“随机性”指 policy 执行过程产生的状态差异，而不是中途新增的环境随机化。阶段标签能够说明接下来做什么，却不能单独说明在当前抓取状态下如何调整动作；空间 point 是否能进一步帮助处理这些差异，需要在已经提供 skill 的条件下检验。

![各 skill 相对 RGB-D 的提升与 GP 增量](./figures/skill_level/skill_level_condition_contrasts.png)

图 3 | Condition 收益的阶段分布。每个 skill 的三个最终差值统一以 RGB-D 为减数，分别比较 RGB-D+GP、RGB-D+skill、RGB-D+GP+skill。第三根柱的浅绿色层表示 skill−RGB-D，深绿色层表示 GP+skill−skill；两者按带符号差值相加，菱形和加粗数字标出相对 RGB-D 的最终差值。Insert、Screw 的两层方向相反，深色窄层与箭头保留抵消关系，不把绝对值相加。分层只是三个观测比例的代数分解，不是独立因果贡献估计。每个 checkpoint/task 为 36 rollout，GP 合并三个 checkpoint，其余各一个；先分别求 ΣC/ΣR，再作差，单位为百分点（pp），不作轨迹配对或 seed 均值解释。沿用[主报告 §3.4](./multi_task_condition_eval_0610.md#34-cross-task-skill-type-success-rates)的 pooled 分组，包括原记录中的 hood 计数；Insert 仅为完整呈现保留。不显示训练 seed 误差条或显著性标记；原始 C/R 见[主报告第 3 节](./multi_task_condition_eval_0610.md#3-分步成功率分析)，跨任务原始值见表 3，分任务结果见附录表 B.1。

表 3 | 跨三个 task 汇总的 skill-level 条件完成率（主报告 §3.4）。单元格为 100×ΣC/ΣR % (ΣC/ΣR)，不是 task success，也不是各 task 百分比的均值。保留原表全部 8 行、40 个 skill 单元格及批次标记。

| Condition | Source batch | Push | Pick | Place | Insert | Screw |
|---|---|---:|---:|---:|---:|---:|
| rgb | 0610 main | 55.00% (44/80) | 50.51% (50/99) | 80.95% (34/42) | 100.00% (34/34) | 73.53% (25/34) |
| rgbd | 0610 main | 65.82% (52/79) | 57.80% (63/109) | 73.21% (41/56) | 100.00% (41/41) | 87.80% (36/41) |
| rgbd+only skill | 0610 main | 97.22% (105/108) | 95.32% (163/171) | 77.17% (98/127) | 97.73% (86/88) | 89.53% (77/86) |
| rgbd+colored GP | 0610 main | 100.00% (108/108) | 98.29% (172/175) | 80.88% (110/136) | 99.00% (99/100) | 78.79% (78/99) |
| rgbd+GP | 0610 main, 3 runs pooled | 99.38% (322/324) | 96.00% (504/525) | 83.84% (332/396) | 99.67% (304/305) | 78.29% (238/304) |
| rgbd+GP+skill | 0610 main | 99.07% (107/108) | 96.72% (177/183) | **91.49% (129/141)** | 100.00% (112/112) | 81.25% (91/112) |
| rgbd+grasp-part † | fresh36 clean n0 | 92.59% (100/108) | 93.87% (153/163) | 82.91% (97/117) | 96.63% (86/89) | 86.05% (74/86) |
| rgbd+grasp-part-colored † | fresh36 clean n0 | 97.22% (105/108) | 95.93% (165/172) | 79.07% (102/129) | 100.00% (90/90) | 88.89% (80/90) |

来源：[主报告 §3.4](./multi_task_condition_eval_0610.md#34-cross-task-skill-type-success-rates)。plain GP 合并三个主实验 checkpoint，其余各一个；每 checkpoint/task 评测 36 rollout。† grasp 两行来自八月 fresh36 clean n0，不与主实验 GP pooled 作跨批次差值；与 point 的同协议比较仍单列于后文的 grasp 待分析部分。分任务表移至[附录表 B.1](#附录表-b1分任务-condition--skill-成功率)，不在正文重复。

在已经提供 skill 信息后，GP 的主要额外收益集中在 Place。pooled Place 从 98/127（77.17%）提高到 129/141（91.49%），增加 14.32 个百分点，而 Push/Pick 的剩余变化较小。Insert 的入口已经满足 Place 对位置和朝向的严格要求，到位后较易完成，因此本节不将其高 C/R 解释为独立技能能力，也不用于支持 condition 的优势。按单任务合并同类型步骤后，one_leg Place 从 29/33（87.88%）提高到 32/34（94.12%），round_table 从 41/49（83.67%）提高到 53/54（98.15%），lamp 从 28/45（62.22%）提高到 44/53（83.02%）。三个任务方向一致，lamp 的增幅最大。再拆到完整 Place 步骤，可以定位这种收益具体发生在哪里（图 4）。

图 4 仅展示 one_leg 的 leg Place、round_table 的 leg/base Place 和 lamp 的 bulb Place；hood 因标签覆盖不足，不作为独立论证行或绘图类别，原始记录仍保留在[主报告 §3.3 及审计附录](./multi_task_condition_eval_0610.md)中。

lamp 的 bulb Place 是最突出的局部例子。skill-only 的 C/R 为 51.43%，低于同一 policy 的 one_leg Place（87.88%）与 round_table 两个 Place（80.00%、89.47%）；加入 GP 后提高到 75.00%，+23.57 pp 是四个主要可比较 Place 步骤中最大的观测差值。这将 task-level 的改善定位到需要更精确空间调整的放置过程，而不是所有操作阶段的均匀提升。

这组结果进一步引出一个问题：为什么 Place 比已得到改善的 Push/Pick 更难，且在已有 skill 信息后，从 point 获得的增益最大？关键在于，Place 不仅需要选对当前动作，还需要把“前一步成功但状态并不标准”的被抓零件送到满足装配约束的位置与朝向。即使目标装配关系相同，只要零件在夹爪中的位置或角度不同，所需的末端运动就会改变。因而，前段成功率升高并不会自动消除放置困难：后段仍需补偿前序操作留下的空间偏差。

Point 对 Place 的帮助可能正来自这一状态依赖性。Skill 标签只给出“现在应执行 Place”，而本实验中的 point 随目标零件位姿及当前零件—夹爪相对关系调整，给出当前状态下末端应移向的位置。因此，不同抓取状态不必对应同一个固定运动目标；policy 获得了一个可以随状态变化的空间参照。这为“已有阶段信息后，GP 的额外收益在 Place 上最大”提供了可能解释。当前实现中的目标换算支持这种解释（[实现依据](./multi_task_condition_eval_0610.md#统一-clean-n0-对照的精确来源)），但单个位置点并不包含完整旋转目标，也不能仅凭 C/R 确认 policy 已通过这一机制消除了抓取偏差。

Lamp 的灯泡放置提供了具体例子。根据任务执行观察，灯泡在抓取后可能形成不同的 grasp pose；即使都满足 Pick 成功判据，随后要把灯泡对准底座时，需要补偿的偏移与朝向仍不相同。这与 bulb Place 在 skill-only 下较低、加入 GP 后增幅较大的结果相容。当前尚未量化抓取后姿态分布，因而这里解释的是一种可能的困难来源，不把“姿态多样 → 放置困难 → point 缓解”的完整链条写成已验证机制。

![四个 Place 步骤的 condition 对比与 GP 增幅](./figures/skill_level/skill_level_place_comparison.png)

图 4 | 不同家具放置步骤对空间 point 的响应。a 四种 condition 的完整 Place 步骤 C/R，b 在 RGB-D+skill 基础上加入 GP 的差值（pp）。所有 policy 均使用 RGB-D，图例仅标记额外输入；GP 合并三条主实验 run，其余各一条，每条 run/task 为 36 rollout。柱长表示记录中的条件完成率，不是跨 seed 均值；无训练 seed 误差条或显著性标记，分子与分母见[主报告表 3.8](./multi_task_condition_eval_0610.md#37-skill-level-分析的动机对照与暂定结论工作稿待确认)，正文不再重复该表。仅展示四个标签覆盖可用于本轮比较的 Place 步骤，hood 的独立行按上述规则排除；grasp 留待联合 tracking 分析，不纳入本图。

colored GP 可视为同时包含空间位置和一定对象/阶段区分线索的 condition，但其收益具有任务依赖性。相对 skill-only，单任务 Place 在 one_leg 从 87.88% 提高到 94.44%，在 lamp 从 62.22% 提高到 73.33%，在 round_table 却从 83.67% 降到 78.18%；round_table Pick 则从 87.50% 提高到 96.49%。因此，颜色提供的区分线索与显式 skill 并不是可直接互换的输入：它可能帮助选择操作对象，但并未在所有任务的 Place 上复现显式 skill 与 GP 组合的收益。

同一 policy 内部的比较还揭示了改善之后剩余的瓶颈。以 RGB-D+GP+skill 为例，round_table Place 已达 53/54（98.15%），Screw 则为 42/53（79.25%）；lamp 的 Place 和 Screw 分别为 44/53（83.02%）和 20/27（74.07%）。回到完整步骤，round_table 的 leg Place 为 32/33，紧随其后的 leg Screw 为 24/32；lamp 的 bulb Place 为 27/36，Screw 为 20/27。总体上，condition 缓解前段 Push/Pick 困难后，放置与旋拧仍是更值得进一步分析的阶段，而它们的重要程度随家具任务改变。

#### 2.4.1 附录

**Grasp：前序状态变异与 tracking error 的待分析问题**

前置步骤留下的状态变异，不仅可能要求调整末端位置，也可能要求调整朝向。由此需要进一步考察：当成功的前置操作产生不同入口状态时，额外的旋转提示是否能帮助后续操作适应这些差异，以及 policy 是否实际跟随了这些提示。灯泡是一个具体例子：不同 grasp pose 都可能满足 Pick 成功，但对准底座时所需的位置与朝向补偿不同。这个问题不限定于灯泡，也不能仅通过某一行 Place 成功率回答。

本节 grasp 数据仅采用已核验的八月 fresh36 clean n0 的 point/grasp 对照：三个任务、low randomness、每 task 36 rollout、3 个并行环境、每条最多 1000 步，每个 condition 一个 checkpoint。八月数据中的 policy 输入配置可核验；七月批次存在视觉标记是否进入 policy 图像的历史兼容性疑问，不用于本节表格或结论，仅在[主报告 A.8](./multi_task_condition_eval_0610.md#a8-grasp-统一分析口径1836-的含义与合并敏感性2026-09-04-补充)留档；此前覆盖不足或口径不一致的 tracking 也不作为本节证据。下表保留同协议 C/R 作为待解释的数据，不据此得出“旋转信息冗余”“grasp 没有帮助”或“point 已足以解决姿态变化”的结论。

表 4 | 八月 clean n0 的 point/grasp 同协议对照（待解释）。五种 condition 均为每 task 完整 36 rollout、三个 task 共 108 rollout；以下 R 是这些轨迹到达相应步骤的次数之和，不是额外 rollout。GP 在此仅使用 icy-vortex-9 的 n0 重评，不使用主实验的三次运行 pooled 值。精确来源与哈希见[六份 grasp task JSON](./multi_task_condition_eval_0610.md#a43-grasp-condition-的-verified-fresh36-clean-n0-来源)和[五种 condition 的 n0 aggregate JSON](./multi_task_condition_eval_0610.md#统一-clean-n0-对照的精确来源)；数值来自完整评测的 C/R 字段，不读取 tracking 字段。

| Condition | Pooled Pick | Pooled Place | Pooled Screw | lamp / bulb-base-place |
|---|---:|---:|---:|---:|
| rgbd+GP | 96.99% (161/166) | 83.20% (104/125) | 75.53% (71/94) | 71.43% (25/35) |
| rgbd+colored GP | 93.82% (167/178) | 80.62% (104/129) | 85.87% (79/92) | 62.86% (22/35) |
| rgbd+GP+skill | 95.95% (166/173) | 82.03% (105/128) | 82.65% (81/98) | 61.76% (21/34) |
| rgbd+grasp-part | 93.87% (153/163) | 82.91% (97/117) | 86.05% (74/86) | 64.71% (22/34) |
| rgbd+grasp-part-colored | 95.93% (165/172) | 79.07% (102/129) | 88.89% (80/90) | 50.00% (17/34) |

现有 C/R 不能区分：额外旋转提示没有提供所需信息、policy 没有准确跟随提示，或两种 policy 在进入 Place 前已经产生不同的抓取状态。下一步需在这组完整 36-rollout 评测的可用轨迹上，逐条连接入口状态、位置/朝向 tracking error 与成功/失败；若原始记录不足，则另行补齐统一协议数据，不能用此前的 saved-8 子集代替完整覆盖。比较前还需统一坐标系、参考目标、目标时刻、片段选择规则和有效样本范围；缺失朝向指标不补零。本轮不使用旧 tracking 数值解释 C/R，也不将 grasp 纳入正文主结论。

Grasp-part 仅在 Pick/Place 绘制 grasp，其余阶段仍绘制 point，因此 Screw 的 C/R 差异不能直接归因于该阶段收到额外旋转提示。当前可确认的是所选评测的样本规模、配置与计数来源；这不等同于所有语义标签或逐轨迹物理成功判定已经验证，已知标签覆盖风险仍见下述局限性。

**上述结论的局限性**

上述结论基于各 policy 从任务起点执行时记录的 C/R，仍用于比较 condition 的阶段性表现，但不等同于固定入口状态下的独立 skill 能力。即使环境只在初始零件分布上随机化，上游 policy 产生的抓取姿态、零件位姿和机器人状态仍可能结构性地影响后续表现；同一 policy 的不同阶段、不同噪声设置也有各自的入口状态分布。增加 rollout 可以减少固定 checkpoint 估计的抽样波动，却不能消除这种结构性差异。同一 policy 的低 C/R 可用于定位当前执行流程的瓶颈，但仅凭这一排序不能确定修复某个 skill 对最终 task success 的收益最大。

跨 task 或同 task 内合并同类型 skill 都会改变各完整步骤的权重，且同一轨迹可以贡献多个 Pick/Place；ΣR 不是独立轨迹数。特别是 lamp 的 hood 标签存在覆盖不足，不能将其较高 C/R 解释为该阶段已经可靠解决。[主报告的轨迹与代码审计](./multi_task_condition_eval_0610.md)保留了历史轨迹证据，并在当前代码中复现了 `done` 回退旧标签的风险。因此表中的加总已核对，但标签完整性尚未全面通过验证；相关 grasp、lamp 结论仍需复核，不能因 C≤R 或汇总一致就升级为物理成功判定已验证。

当前训练重复不均衡，多数 condition 只有一个 checkpoint；plain GP 的 pooled 三次运行不是统一的 mean ± std 比较。以上“最大增幅”“主要收益”等指本批记录中的差值，不是统计显著性或机制因果检验。此外，grasp 与 point 对照同时改变了视觉标记形式，且来自不同 checkpoint，并不是同一 policy 仅开关旋转通道的严格消融，相关解释须待 tracking error 与入口状态分析补足；本轮不保留旋转冗余或计算效率结论。RGB-D 在 round_table 的较高 Place 数值也保留为当前观察，因此总体结论限定为阶段和任务依赖的收益，而不是“所有 GP condition 在所有 Place 步骤都优于无 GP”。

**多 training seed 复验计划与暂不展开的 RGB-D Place 高值**

现阶段不以 RGB-D 的较高 Place 值作为讨论主线，先按上述结果形成大致结论；所有表格仍保留原始计数，不删样本、不改数值，也不预设它是统计离群点。后续应对各 condition 采用多个独立 training seed、统一评测预算和初始状态采样规则，在每个 seed 内分别计算跨任务及单任务的 ΣC/ΣR，再报告这些 seed-level 比率的 mean ± sample std，同时列出训练 seed 数和各 seed 的原始 C/R。某个 seed 的 R=0 时该比率未定义，应标注有效 seed 数，不能补成 0%。不能把不同任务、rollout 或同一 checkpoint 的重复评测当作训练 seed，也不能将先合并全部 seed 的计数所得比例标为 mean ± std。

多 seed 的目的是检验结论是否稳定，而不是让 RGB-D 的数值朝预期方向变化；目前没有数据保证其均值会下降。新增评测应预先确定预算与停止规则，保留所有 seed 和评测结果。若后续需要区分 GP 的直接局部作用与前置状态质量影响，则另行设置共同中间状态的 skill 实验；这与从起点增加 rollout 回答不同问题。本轮只记录计划，不启动训练或评测。

RGB-D Place 的原始组成分解继续保留在[主报告 §3.7 的复核记录](./multi_task_condition_eval_0610.md)，不在此重复展开；本节表 3 与附录表 B.1 保留其现有汇总值，完整 Place 步骤计数见主报告表 3.8。


<a id="附录表-b1分任务-condition--skill-成功率"></a>

**分任务 condition × skill 成功率**

本表拟放论文附录，不占用 §2.4 正文篇幅。采用更新后的[主报告 §3.5](./multi_task_condition_eval_0610.md#35-分任务的-condition--skill-type-成功率工作稿待确认)，与主报告表 3.7 的清晰版一致，用于检验跨任务结论在单任务上的适配性，并定位不同家具的操作难点。

| Task | Condition | Push | Pick | Place | Insert | Screw |
|---|---|---:|---:|---:|---:|---:|
| one_leg | RGB | 0.00% (0/8) | 22.22% (8/36) | —（0 次到达） | —（0 次到达） | —（0 次到达） |
| one_leg | RGB-D | 0.00% (0/7) | 19.44% (7/36) | —（0 次到达） | —（0 次到达） | —（0 次到达） |
| one_leg | RGB-D+skill | 91.67% (33/36) | 100.00% (69/69) | 87.88% (29/33) | 96.55% (28/29) | 100.00% (28/28) |
| one_leg | RGB-D+colored GP | 100.00% (36/36) | 100.00% (72/72) | 94.44% (34/36) | 100.00% (34/34) | 97.06% (33/34) |
| one_leg | RGB-D+GP（3 runs pooled） | 98.15% (106/108) | 100.00% (213/213) | 89.52% (94/105) | 100.00% (94/94) | 93.62% (88/94) |
| one_leg | RGB-D+GP+skill | 97.22% (35/36) | 100.00% (70/70) | 94.12% (32/34) | 100.00% (32/32) | 90.62% (29/32) |
| round_table | RGB | 97.22% (35/36) | 74.07% (40/54) | 85.00% (34/40) | 100.00% (34/34) | 73.53% (25/34) |
| round_table | RGB-D | 100.00% (36/36) | 84.21% (48/57) | 85.42% (41/48) | 100.00% (41/41) | 87.80% (36/41) |
| round_table | RGB-D+skill | 100.00% (36/36) | 87.50% (49/56) | 83.67% (41/49) | 97.56% (40/41) | 92.50% (37/40) |
| round_table | RGB-D+colored GP | 100.00% (36/36) | 96.49% (55/57) | 78.18% (43/55) | 97.67% (42/43) | 73.81% (31/42) |
| round_table | RGB-D+GP（3 runs pooled） | 100.00% (108/108) | 88.70% (157/177) | 92.36% (145/157) | 99.31% (144/145) | 79.17% (114/144) |
| round_table | RGB-D+GP+skill | 100.00% (36/36) | 90.00% (54/60) | 98.15% (53/54) | 100.00% (53/53) | 79.25% (42/53) |
| lamp | RGB | 25.00% (9/36) | 22.22% (2/9) | 0.00% (0/2) | —（0 次到达） | —（0 次到达） |
| lamp | RGB-D | 44.44% (16/36) | 50.00% (8/16) | 0.00% (0/8) | —（0 次到达） | —（0 次到达） |
| lamp | RGB-D+skill | 100.00% (36/36) | 97.83% (45/46) | 62.22% (28/45) | 100.00% (18/18) | 66.67% (12/18) |
| lamp | RGB-D+colored GP | 100.00% (36/36) | 97.83% (45/46) | 73.33% (33/45) | 100.00% (23/23) | 60.87% (14/23) |
| lamp | RGB-D+GP（3 runs pooled） | 100.00% (108/108) | 99.26% (134/135) | 69.40% (93/134) | 100.00% (66/66) | 54.55% (36/66) |
| lamp | RGB-D+GP+skill | 100.00% (36/36) | 100.00% (53/53) | 83.02% (44/53) | 100.00% (27/27) | 74.07% (20/27) |

统计口径：在同一 task、同一 condition 内，将相同 skill 类型的完整语义步骤合并为 ΣC/ΣR；单元格给出百分比及完成数/到达数。18 行均来自主实验批次，GP 为三个 checkpoint pooled，其余各一个，每 checkpoint/task 为 36 rollout。— 表示 R=0，而不是 0%；一条轨迹可以贡献多个 Pick/Place，因此分母可超过 36。Grasp 不纳入本表，其八月 clean n0 数据见正文表 3 的标记行及表 4 的同协议待分析对照。

来源与核对：base 上八条 run 的 24 份 task JSON，精确路径与节点核验记录见[主报告 A.4–A.5](./multi_task_condition_eval_0610.md#a4-查证路径)。本表 90 个单元格与更新后的 §3.5 完全一致；按 task 加总 C、R 后，对应正文表 3 中六种非 grasp 主实验 condition 的 30 项。保留原有 hood 聚合计数；其标签覆盖与 Insert 解释边界沿用 §2.4 的局限性说明。


### 2.5 VLM 引导下的端到端评测

来源：[VLM + DiT guidance point 评测报告 §2.2](./vlm_dit_guidance_eval.md#22-真实-vlm-引导误差及其下游影响)。

完全无噪声的 oracle 评测将上游定位误差排除在实验之外，因而无法回答真实部署中的核心问题：当 VLM 根据视觉观察提出的目标点或抓取姿态并不完全准确时，下游 DiT policy 是否仍能把这份引导转化为可执行的长时程行为。为此，我们同时测量完整任务成功率与 VLM 目标点误差，并利用 §2.3 的受控噪声评测确定真实 VLM 误差相对于下游容忍范围的位置。

#### 实验设计与评测协议

我们比较 point-based 与 grasp-based 两类引导接口，并在 `one_leg`、`round_table` 和 `lamp` 三个任务上采用一致的轨迹评测协议。VLM 每 8 个环境步更新一次引导，其间沿用缓存结果。每个 condition/task 包含 36 条完整 rollout，成功仅在完整家具装配完成时计入。

**跨任务双头微调（cross-task dual-head fine-tuning）。** 我们从三个任务的 scripted rollouts 构建统一标注集，并分别微调两个 VLM。Ver1 输出 skill 与二维目标点，对应 point guidance；Ver2 进一步输出 `target_rotation_6d`，对应 grasp guidance。Rotation6D 由 scripted guidance pose 的旋转矩阵前两行构成，并通过逐行 Gram--Schmidt 正交化解码为合法的 `SO(3)` 姿态。

**VLM 目标点误差建模（VLM point-error modelling）。** 对正式评测轨迹中的每个有效控制步，我们记录脚本生成的几何真值点投影 `p_gt` 与实际送入 policy 的 VLM point `p_vlm`，并定义二维误差向量

`e_vlm = p_vlm - p_gt`。

我们以 `F_vlm` 表示这些误差向量的经验分布。均值反映系统偏置，协方差特征值比反映方向性，径向误差 `||e_vlm||₂` 反映幅度，P90/P95 则刻画长尾。由于不同下游 condition 共享同一个 VLM，模型层面的误差在任务和 skill 层面合并计算。

**相机匹配的投影尺度对齐（camera-matched projection alignment）。** VLM 输出位于图像平面，而 §2.3 的噪声实验在三维目标上施加扰动。固定的 pixel-to-mm 比例无法同时反映深度和透视，因此我们使用同一相机模型将三维噪声投影到图像平面，再与 `F_vlm` 比较。对每个有效控制步及每个噪声尺度 `σ_n`，从裁剪至 `[-2,2]` 的标准三维高斯中采样 200 个扰动，并计算

`e_nj = π(P_gt + σ_n z_j) - π(P_gt)`。

在相邻噪声等级之间对 projected RMSE 线性插值，即可得到 VLM 目标点误差的等效三维扰动尺度。该尺度表示在相同相机、深度和投影模型下产生相同二维 RMSE 的 `mm/axis` 噪声，用于连接 VLM 评测与 §2.3 的噪声响应曲线。

#### 端到端结果

表 5 | 真实 VLM 引导下的完整任务成功率。每个 condition/task 包含 36 条 rollout；Overall 汇总三个任务。Point 与 grasp condition 在同一张表中比较。

| Condition | one_leg | round_table | lamp | Overall |
| --- | ---: | ---: | ---: | ---: |
| RGB-D+GP | 86.1% (31/36) | 44.4% (16/36) | 44.4% (16/36) | 58.3% (63/108) |
| RGB-D+colored GP | 86.1% (31/36) | 41.7% (15/36) | 50.0% (18/36) | 59.3% (64/108) |
| RGB-D+GP+skill | 83.3% (30/36) | 27.8% (10/36) | 38.9% (14/36) | 50.0% (54/108) |
| RGB-D+grasp-part | 88.9% (32/36) | 38.9% (14/36) | 61.1% (22/36) | **63.0% (68/108)** |
| RGB-D+grasp-part-colored | 72.2% (26/36) | 38.9% (14/36) | 41.7% (15/36) | 50.9% (55/108) |

**不完美的 VLM 引导仍能形成端到端行为。** 三个 point condition 在 324 条 rollout 中完成 181 次完整任务（55.9%），其整体成功率为 50.0–59.3%；两个 grasp condition 分别达到 63.0% 和 50.9%。其中，colored GP 是 point interface 中表现最好的 condition，grasp-part 则取得五个 condition 中最高的整体成功率。结合附录 A.2 中 Point VLM 的 3-task 等效位置噪声 `63.44 mm/axis`，这些结果表明下游 action expert 能够在真实 VLM 目标点误差下保持完整任务执行。

**显式姿态通道为 grasp guidance 提供了额外表达能力。** `RGB-D+grasp-part` 的整体成功率为 63.0%，在 lamp 上达到 61.1%，相对三个 point condition 提高 11.1–22.2 个百分点。这一任务依赖的优势与 Ver2 增加旋转输出的设计一致：当接触方向和末端姿态更关键时，grasp guidance 可以提供仅有位置点时缺失的约束。`grasp-part-colored` 的整体成功率为 50.9%，说明旋转信息的收益并不在所有接口组合与任务上稳定出现；同时，旋转预测也增加了上游模型需要拟合的误差维度。

**VLM 误差呈现结构化分布。** Point 与 Grasp VLM 的 task-level 等效 `σ` 分别跨越 `36.15–72.86` 和 `96.26–170.09 mm/axis`，而 n4 在三个任务上固定为 `24 mm/axis`。跨任务差异与非零 bias 表明误差具有任务相关的系统偏置；大于 1 的 anisotropy 显示方向性，相对 n4 非零的 centered SWD 和 radial W1 则显示非高斯形状（表 6）。因此，等效 `σ` 只对齐误差幅度，不代表两者具有相同分布。

表 6 | Point VLM、Grasp VLM 与 n4 受控噪声的分布摘要。Equivalent `σ` 为三个任务的范围，`Δ` 为跨任务跨度；bias norm 与 anisotropy 分别衡量平均偏移和方向性；centered SWD 与 radial W1 分别比较中心化后的二维形状和误差半径。VLM 数值为同一 family 内各 condition 的范围，分布距离均以 n4 为参照。

| Error source | Task-level equivalent σ (mm/axis) | Bias norm (px) | Anisotropy | n4 centered SWD (px) | n4 radial W1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Point VLM | 36.15–72.86 (Δ36.71) | 7.27–8.78 | 3.91–5.73 | 11.00–11.85 | 16.80–17.03 |
| Grasp VLM | 96.26–170.09 (Δ73.83) | 26.41–26.50 | 2.27–2.30 | 24.64–26.03 | 44.15–46.41 |
| Controlled noise (n4) | 24.00 (Δ0) | 零均值设定 | 三维各向同性设定 | 0（自参照） | 0（自参照） |

综合表 5、表 6、§2.3 的受控噪声响应和附录 A.2 的等效尺度，下游策略覆盖了 Point VLM 的典型目标点误差范围，并在真实 VLM 引导下完成了相当比例的长时程任务。由此得到的系统级结论是，二维目标点可以作为视觉语义理解与连续动作生成之间的有效接口；等效 `σ` 仅用于对齐误差幅度，而不意味着 VLM 误差与受控噪声具有相同分布。结论范围限于当前三个 FurnitureBench 任务及其观测到的 VLM 误差分布。


## 3. 仍在进行或计划中的实验

1. 真机实验：形成与表 1 对齐的 condition 对比表，并提供代表性 demo。
2. VLM 误差分层分析：按目标点误差区间统计 action-expert 成功率，并进一步分解 skill-specific p95 长尾与 failure/OOD 状态。
3. Med-train → high-eval：high randomness 加入零件初始位姿排列组合，预期成功率较低；需要多 seed 和足够 rollout 才能比较 condition。
4. 任务扩展：加入 ManiSkill3 与 Isaac Lab / Isaac Gym 的新任务和数据，验证 guidance point 作为跨任务接口的可扩展性。

## 附录 A：结果来源

- 多任务 condition 主实验：[`multi_task_condition_eval_0610.md`](./multi_task_condition_eval_0610.md#1-总览)
- Low→Med 空间泛化：[`low2med_generalization.md`](./low2med_generalization.md#results)
- Clean-train → noisy-eval 正式结果：[`annotation_noise_vlm_cover_108.md`](./annotation_noise_vlm_cover_108.md)；早期 seed-0 对照：[`annotation_noise_clean_train_fresh36.md`](./annotation_noise_clean_train_fresh36.md#1-结果图)。
- 主文图 1：`figures/vlm_cover_108/vlm_cover_108_success_numeric.png`（task-level SR；仅 Point/Grasp 两条 position-equivalent σ 线）。
- 主文图 2：`figures/vlm_cover_108/vlm_cover_108_tracking_error.png`（pooled position/orientation/total tracking；不再使用单独 endpoint 图）。
- VLM 端到端评测：[`vlm_dit_guidance_eval.md`](./vlm_dit_guidance_eval.md#22-真实-vlm-引导误差及其下游影响)。

### 附录 A.1：108 实验补充图

三 task pooled success（仅保留 Point/Grasp 两条 position-equivalent σ 线）：

![Three-task pooled success](figures/vlm_cover_108/vlm_cover_108_success_3task_pooled.png)

三 task pooled position tracking：

![Three-task pooled position tracking](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.png)

Skill-level success、position/orientation/total tracking：

Skill-level 图中的第三条参考线是 Grasp VLM 的 point-error p95 stress line，不是 orientation-equivalent scale；orientation 对齐不在任何图中绘制。

![Skill-level success rate](figures/vlm_cover_108/vlm_cover_108_skill_success_rate.png)

![Skill-level position tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_position.png)

![Skill-level orientation tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_orientation.png)

![Skill-level total tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_total.png)

### 附录 A.2：VLM position-equivalent σ anchor table

完整 anchor table 放在附录，不占用正文版面。σ 是将有效 VLM–GT 点误差映射到共同的 `mm/axis` 参考后的等效位置尺度；Point 与 Grasp 分开、每个 task 独立计算。

| VLM family | Task | Source trajectories | Valid control-step pairs | Equivalent position σ (mm/axis) |
| --- | --- | ---: | ---: | ---: |
| Point VLM | one_leg | 108 | 45,555 | 36.15 |
| Point VLM | round_table | 108 | 93,621 | 72.86 |
| Point VLM | lamp | 108 | 73,900 | 68.34 |
| Grasp VLM | one_leg | 72 | 12,168 | 170.09 |
| Grasp VLM | round_table | 72 | 58,626 | 100.28 |
| Grasp VLM | lamp | 72 | 39,492 | 96.26 |
| Point VLM | 3-task pooled | 324 | 213,076 | 63.44 |
| Grasp VLM | 3-task pooled | 216 | 110,286 | 106.54 |

Grasp 的 orientation-equivalent scale 不在图中展示。它由 n0（未注入 orientation noise）的 clean-GT orientation tracking residual 与 `0/2.5/5/10/20/40/60/90°` schedule 匹配，再映射回绑定的 position schedule；该量表示下游策略的行为等效覆盖位置，而不是 raw VLM orientation σ。

| Task | n0 orientation tracking error (deg) | Matched level | Position-axis coordinate (mm) | Tracking states |
| --- | ---: | --- | ---: | ---: |
| one_leg | 14.74 | n3–n4 | 17.69 | 817 |
| round_table | 30.14 | n4–n5 | 36.17 | 996 |
| lamp | 44.45 | n5–n6 | 58.68 | 700 |
| 3-task pooled | 29.12 | n4–n5 | 34.95 | 2,513 |

### 附录 A.3：完整数值表与数据链

完整 condition × noise × task 的 success、Wilson CI、replicate、skill-level 结果、tracking、实际几何扰动和 VLM tail 统计均保存在正式报告及其数据产品中：[`annotation_noise_vlm_cover_108.md`](./annotation_noise_vlm_cover_108.md)。其中成功率表约含 135 个以上结果单元，正文不重复展开；图和表均遵循 `JSON → tables → figures`，`table_validation.csv` 的 34 项检查全部通过。
