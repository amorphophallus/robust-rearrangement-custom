# ICLR 2026 实验结果与论文叙事整理

> 整理日期：2026-09-17（main baseline 与 joint delta 更新）<br>
> 代码基线：`main@19ab7cc3c4eaa841c6c0d4751dba9dc40c3e1889`（整理时与 `origin/main` 一致）<br>
> VLM 覆盖范围噪声 campaign 基线：`main@f6877ff75418b4f4af835f3b4a9dfecea0ffc9ab`<br>
> 原始文档：`D:\ZJU\研一春夏\ICLR2026\实验结果整理\实验结果整理.md`<br>
> 说明：§2.1–2.2 保留两张源表；§2.3 已用三个评测 replicate、108 rollout/cell 的 VLM 覆盖范围噪声实验替换早期 seed-0 结论；§2.4 补充 skill-level 分析及其表图。

## 1. 论文想讲的故事

Long-horizon furniture assembly requires robots to switch across tasks and stages, localize the part and target, and execute contact-rich skills such as insertion and screwing; failure at any stage can invalidate the assembly. We use a vision-language model (VLM) for instruction understanding, task decomposition, and visual grounding, allowing a Diffusion Transformer (DiT) action expert to generate continuous actions from a compact conditioning interface rather than infer task stage and interaction target solely from high-dimensional observations. We ask which interface best improves downstream action generation while remaining robust to inevitable VLM prediction errors. Focusing on image-space targets, we compare 5 conditioning interfaces: a spatial-only guidance point (GP), a semantic-only low-dimensional skill condition, their explicit combination (GP+skill), a Target–Action Guidance Point (TAGPoint) that encodes low-dimensional information in point color, and a 6D grasp annotation that augments positional guidance with end-effector orientation. We evaluate these interfaces through FurnitureBench assembly, skill-level performance, controlled perturbations, end-to-end execution with real VLM predictions, and a larger set of single-step assembly tasks. Explicit 2D targets generally improve multitask performance, with the clearest gains in placement; skill conditioning alone provides limited benefits but becomes complementary when paired with a spatial target. We therefore propose TAGPoint, which encodes skill group in point color to combine spatial and semantic guidance in a compact, image-aligned representation. TAGPoint performs best among point-based interfaces under real VLM guidance, remains stable across target-point perturbation levels, and remains effective as task scale increases. Controlled perturbations make the role and failure boundaries of the intermediate interface measurable, providing guidance for representation design in VLM–action expert systems.

长时程家具拼装要求机器人跨任务与操作阶段切换，定位交互零件及其目标，并执行插入、旋拧等接触密集动作；任一环节失败都可能导致整体装配失败。我们由视觉语言模型（VLM）负责指令理解、任务分解与视觉定位，使 Diffusion Transformer（DiT）action expert 能够结合紧凑的条件接口生成连续动作，而无需仅凭高维观测推断任务阶段与交互目标。本文研究的核心问题是：何种条件接口既能改善下游动作表现，又能容忍真实 VLM 的预测误差？
围绕图像空间目标，我们比较仅包含空间位置的 guidance point（GP）、仅包含阶段语义的低维 skill condition、二者的显式组合 GP+skill、以点颜色编码低维信息的 Target–Action Guidance Point（TAGPoint），以及增加末端旋转信息的 6D grasp annotation。我们在 FurnitureBench 长时程装配、分阶段分析、受控引导扰动、真实 VLM 端到端评测和更大规模的单步装配任务上评估这些接口。实验表明，显式二维目标总体上改善了多任务表现，且提升最集中于放置阶段；skill condition 单独使用时收益有限，但与空间目标结合后表现出互补作用。基于这一观察，我们提出 TAGPoint，利用目标点颜色编码粗粒度夹爪动作模式，将空间与语义引导整合为紧凑、图像对齐的表示。TAGPoint 在真实 VLM 引导下取得 point-based interfaces 中最好的表现，在不同级别的目标点扰动下保持稳定，并在任务规模扩大后继续有效。受控扰动还使中间接口的作用与失效边界能够被直接测量，从而为 VLM–action expert 系统的中间表示设计提供依据。

## 2. 实验结果

> [!NOTE]
> §2.1–2.2 整理历史源表；§2.3 汇总已完成并通过严格审计的 VLM 覆盖范围噪声实验；§2.4 为 skill-level 分析，包含来源、统计口径及解释局限。

### 2.1 多任务 condition 主实验（Main multi-seed）

详细数据来源、checkpoint 对应关系、逐 seed 成功数、评估输入契约和排除规则见 [main experiment review](./main_3seed_experiment_review_0913.md)。

设置：DiT policy 在 FurnitureBench 的 `one_leg`、`round_table`、`lamp` 三个任务上训练；每个 checkpoint/task 评估 36 rollout。Overall 先在每个训练 seed 内对 108 个 rollout 汇总，再在同一 condition 的训练 seed 间计算 mean ± sample standard deviation。原始 RGB/RGB-D checkpoint 因错误 lineage 被排除，这两行暂为两个有效 supplemental seed；其余 condition 为三个 train seed。

#### 表 1：不同 condition 的多任务成功率（mean ± std）

| Condition | n_train | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|---:|
| RGB-D + GP | 3 | 82.41 ± 4.24% | 41.67 ± 12.11% | 33.33 ± 7.35% | 52.47 ± 4.66% |
| RGB-D + colored GP | 3 | 87.04 ± 4.24% | 50.00 ± 22.22% | 35.19 ± 3.21% | 57.41 ± 5.78% |
| RGB-D + GP + skill | 3 | **88.89 ± 4.81%** | **56.48 ± 11.23%** | **44.44 ± 10.02%** | **63.27 ± 4.18%** |
| RGB-D + skill | 3 | 77.78 ± 2.78% | 49.07 ± 1.60% | 34.26 ± 1.60% | 53.70 ± 1.60% |
| RGB-D | 2 | 81.94 ± 1.96% | 48.61 ± 1.96% | 23.61 ± 5.89% | 51.39 ± 0.65% |
| RGB | 2 | 86.11 ± 3.93% | 45.83 ± 5.89% | 18.06 ± 5.89% | 50.00 ± 1.31% |
| RGB-D + grasp-part | 3 | 87.04 ± 4.24% | 14.81 ± 20.85% | 37.04 ± 5.78% | 46.30 ± 8.49% |
| RGB-D + colored grasp-part | 3 | 87.04 ± 8.93% | 22.22 ± 19.25% | 33.33 ± 2.78% | 47.53 ± 5.58% |

#### Paper narrative: spatial and semantic conditioning improve multi-task assembly

**Motivation.** We test whether explicit task-relevant conditions help a shared policy infer both what interaction to execute and where to execute it. The comparison separates spatial information (a guidance point, GP) from semantic information (a fixed skill label or a colour code attached to GP).

**Experimental setting.** We evaluate DiT policies on one-leg, round-table and lamp with 36 rollouts per checkpoint and task. Most conditions use three independent training runs. The original RGB and RGB-D checkpoints are excluded because they belong to an erroneous data/checkpoint lineage; these two baselines currently use two valid supplemental runs.

**Results.** In the controlled two-run lineage, RGB-D reaches `51.39±0.65%` overall success. Skill-only reaches `54.17±1.96%`, coloured GP reaches `59.72±5.89%`, and GP+skill reaches `63.43±5.89%`. The average gain of these three semantic conditions is `+2.78 pp` on one-leg, `+8.33 pp` on round-table and `+12.04 pp` on lamp. In the broader registered table, GP+skill has the highest point estimate (`63.27±4.18%`) and coloured GP the second highest (`57.41±5.78%`), with unequal `n_train` and mixed lineages limiting strict ranking claims.

Holding GP fixed, adding skill improves one-leg, round-table and lamp by `+6.48`, `+14.81` and `+11.11 pp`. Holding skill fixed, adding GP improves the same tasks by `+11.11`, `+7.41` and `+10.18 pp`. Coloured GP improves over GP by `+4.63`, `+8.33` and `+1.86 pp`. These ablations support complementary spatial and semantic information; GP+skill is an information-rich reference, while coloured GP is the scalable visual interface.

**Interpretation and boundary.** The evidence supports combining a spatial target with semantic context, with the largest same-lineage gain on lamp. It does not support a claim that RGB or RGB-D collapses to a single task, because the checkpoint that motivated that interpretation has been removed. Grasp rotation does not provide a clear additional gain and remains a round-table-specific failure mode. Clean endpoint success cannot establish causal condition use; paired interventions and stage-level analysis remain necessary.

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

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_success_numeric.pdf)

图 1 的三个分面分别对应 `one_leg`、`round_table` 和 `lamp`；横轴使用真实 position σ/axis，右侧仅保留 `n7→Shuffle` 的 categorical endpoint。图中每个 task 只标出 Point VLM 与 Grasp VLM 两条 position-equivalent σ 线；Grasp 的 orientation-equivalent scale 只在文字和附录说明，不在图上增加第三条纵线。曲线来自已校验表，不展示 replicate 散点。

#### 图 2：pooled tracking response

![VLM-cover pooled position tracking](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.pdf)

图 2 将三个任务的 position tracking 按有效 final skill-state 数加权后合并为一组 condition 曲线，并保留 `n7→Shuffle` 对照，用于区分“同一 subtask 的数值目标偏移”和“被置换 subtask 的 guidance”。Numeric-noise tracking 以 policy 实际接收的 displayed target 为参照，因此 n7 的大 tracking error 不是到 clean-GT 的距离。Position 是所有五种接口共享且直接对应 2D guidance 的主要 tracking 指标；3 task × 3 metric 的密集网格移至附录，用于补充 grasp 的 orientation/total 诊断。

三 task pooled success 图移至附录，不在正文重复。n0–n7 进入 ordinal 描述性趋势，Shuffle 与 r180 不进入连续拟合；grasp n0–n7 同时改变位置和旋转，只能解释为联合扰动。

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

### 2.4 Multi-seed skill-level 成功率分析

完整分析、逐 checkpoint/逐 task 表、数据来源与解释边界见新增文档：[`skill_level_analysis_multiseed_0917.md`](./skill_level_analysis_multiseed_0917.md)。旧版单/少 seed 分析仍保留在 [`multi_task_condition_eval_0610.md`](./multi_task_condition_eval_0610.md)，但不再作为论文当前 skill-level 数值来源。

#### 数据范围与统计口径

本轮使用 66 份 task JSON。RGB-D+skill、colored GP（TAGPoint）、GP+skill、grasp 与 colored grasp 各合并 3 个训练 checkpoint；plain GP 沿用 3 个原始 checkpoint。RGB/RGB-D 排除 original checkpoint，只使用 `2026090701`、`2026090702` 两组新 checkpoint。每个 checkpoint/task 为 36 条 rollout。统计继续采用完整语义标签的 pooled 条件完成率 `100×ΣC/ΣR`，与旧图逻辑一致；逐 seed 比例和 sample std 另存于机器可读数据，但本轮图中不画误差条。

#### 更新后的核心结果

| Condition | n_train | Push | Pick | Place | Insert | Screw |
|---|---:|---:|---:|---:|---:|---:|
| RGB-D | 2 | 97.69% (211/216) | 98.31% (348/354) | 77.45% (213/275) | 98.98% (195/197) | 90.77% (177/195) |
| RGB-D+skill | 3 | 96.91% (314/324) | 96.32% (498/517) | 77.12% (300/389) | 96.74% (267/276) | 91.39% (244/267) |
| RGB-D+GP | 3 | 99.38% (322/324) | 96.00% (504/525) | 83.84% (332/396) | 99.67% (304/305) | 78.29% (238/304) |
| RGB-D+colored GP (TAGPoint) | 3 | 96.91% (314/324) | 98.10% (517/527) | 80.15% (327/408) | 98.66% (295/299) | 88.81% (262/295) |
| RGB-D+GP+skill | 3 | 97.53% (316/324) | 96.86% (525/542) | **86.99% (361/415)** | 99.38% (320/322) | 88.09% (281/319) |

新版 RGB-D checkpoint 已使 Push/Pick 接近饱和，因此旧版“condition 的收益主要集中于 Push/Pick”不再成立。最清楚的正向差异转移到 Place：相对 RGB-D，plain GP 为 `+6.38 pp`，TAGPoint 为 `+2.69 pp`，GP+skill 为 `+9.53 pp`；skill-only 与 RGB-D 基本持平（`-0.33 pp`）。在已有 skill condition 时加入 GP，Place 从 `77.12% (300/389)` 提高到 `86.99% (361/415)`，增加 `9.87 pp`。

![各 skill 相对新 RGB-D 的 multi-seed 差值](./figures/skill_level_multiseed/skill_level_condition_contrasts_multiseed.png)

[矢量 PDF](./figures/skill_level_multiseed/skill_level_condition_contrasts_multiseed.pdf)

图 3 | Multi-seed condition 收益的阶段分布。RGB-D 使用两个新 checkpoint；GP、skill 与 GP+skill 各使用三个训练 checkpoint。柱高为 pooled `ΣC/ΣR` 的差值，不是 seed mean，也不是 paired trajectory 的因果效应。

![四个 Place 步骤的 multi-seed condition 对比](./figures/skill_level_multiseed/skill_level_place_comparison_multiseed.png)

[矢量 PDF](./figures/skill_level_multiseed/skill_level_place_comparison_multiseed.pdf)

图 4 | Multi-seed Place 步骤分析。四种 condition 均合并三个训练 checkpoint。GP+skill 相对 skill-only 在四个 Place 标签上均为正，增量依次为 `+5.07`、`+4.99`、`+12.07` 和 `+18.10 pp`。hood placement 因标签覆盖不完整继续排除。

该分析仍受 stage-entry distribution 影响：各 policy 从任务起点执行，后续阶段的零件位姿、抓取状态与机器人状态由前序行为共同决定。因此 pooled C/R 可用于定位当前流程差异，但不等同于固定入口状态下的独立 skill 能力；original 与 supplemental checkpoint 的数据/评测 lineage 也不完全一致。


### 2.5 VLM 引导下的端到端评测

来源：[VLM + DiT guidance point 评测报告 §2.2](./vlm_dit_guidance_eval.md#22-真实-vlm-引导误差及其下游影响)；无 VLM 的 RGB-D 对照来自[主实验三 seed 复核报告](./main_3seed_experiment_review_0913.md)。

完全无噪声的 oracle 评测将上游定位误差排除在实验之外，因而无法回答真实部署中的核心问题：当 VLM 根据视觉观察提出的目标点或抓取姿态并不完全准确时，下游 DiT policy 是否仍能把这份引导转化为可执行的长时程行为。为此，我们同时测量完整任务成功率与 VLM 目标点误差，并利用 §2.3 的受控噪声评测确定真实 VLM 误差相对于下游容忍范围的位置。

#### 实验设计与评测协议

我们比较无 VLM 的 RGB-D 基线以及 point-based 与 grasp-based 两类 VLM 引导接口，并在 `one_leg`、`round_table` 和 `lamp` 三个任务上采用一致的完整任务成功判据。RGB-D 基线合并两个有效训练 run，每个任务共 72 条 rollout；每个 VLM condition/task 包含 36 条 rollout。VLM 每 8 个环境步更新一次引导，其间沿用缓存结果。成功仅在完整家具装配完成时计入。

**跨任务双头微调（cross-task dual-head fine-tuning）。** 我们从三个任务的 scripted rollouts 构建统一标注集，并分别微调两个 VLM。Ver1 输出 skill 与二维目标点，对应 point guidance；Ver2 进一步输出 `target_rotation_6d`，对应 grasp guidance。Rotation6D 由 scripted guidance pose 的旋转矩阵前两行构成，并通过逐行 Gram--Schmidt 正交化解码为合法的 `SO(3)` 姿态。

**VLM 目标点误差建模（VLM point-error modelling）。** 对正式评测轨迹中的每个有效控制步，我们记录脚本生成的几何真值点投影 `p_gt` 与实际送入 policy 的 VLM point `p_vlm`，并定义二维误差向量

`e_vlm = p_vlm - p_gt`。

我们以 `F_vlm` 表示这些误差向量的经验分布。均值反映系统偏置，协方差特征值比反映方向性，径向误差 `||e_vlm||₂` 反映幅度，P90/P95 则刻画长尾。由于不同下游 condition 共享同一个 VLM，模型层面的误差在任务和 skill 层面合并计算。

**相机匹配的投影尺度对齐（camera-matched projection alignment）。** VLM 输出位于图像平面，而 §2.3 的噪声实验在三维目标上施加扰动。固定的 pixel-to-mm 比例无法同时反映深度和透视，因此我们使用同一相机模型将三维噪声投影到图像平面，再与 `F_vlm` 比较。对每个有效控制步及每个噪声尺度 `σ_n`，从裁剪至 `[-2,2]` 的标准三维高斯中采样 200 个扰动，并计算

`e_nj = π(P_gt + σ_n z_j) - π(P_gt)`。

在相邻噪声等级之间对 projected RMSE 线性插值，即可得到 VLM 目标点误差的等效三维扰动尺度。该尺度表示在相同相机、深度和投影模型下产生相同二维 RMSE 的 `mm/axis` 噪声，用于连接 VLM 评测与 §2.3 的噪声响应曲线。

#### 端到端结果

表 5 | 无 VLM RGB-D 基线与真实 VLM 引导系统的完整任务成功率。RGB-D 基线合并两个有效训练 run（每个任务 72 条 rollout，Overall 216 条）；每个 VLM condition/task 包含 36 条 rollout（Overall 108 条）。不同策略独立训练且样本量不同，因此无 VLM/有 VLM 以及 point/grasp 间的差异均为系统级描述性比较，而非单变量因果消融。

| Condition | one_leg | round_table | lamp | Overall |
| --- | ---: | ---: | ---: | ---: |
| RGB-D（no VLM） | 81.9% (59/72) | 48.6% (35/72) | 23.6% (17/72) | 51.4% (111/216) |
| RGB-D+GP | 86.1% (31/36) | 44.4% (16/36) | 44.4% (16/36) | 58.3% (63/108) |
| RGB-D+colored GP | 86.1% (31/36) | 41.7% (15/36) | 50.0% (18/36) | 59.3% (64/108) |
| RGB-D+GP+skill | 83.3% (30/36) | 27.8% (10/36) | 38.9% (14/36) | 50.0% (54/108) |
| RGB-D+grasp-part | 88.9% (32/36) | 38.9% (14/36) | 61.1% (22/36) | **63.0% (68/108)** |
| RGB-D+grasp-part-colored | 72.2% (26/36) | 38.9% (14/36) | 41.7% (15/36) | 50.9% (55/108) |

**不完美的 VLM 引导仍能形成端到端行为。** 无 VLM 的 RGB-D 基线在两个有效训练 run 上完成 `111/216=51.4%` 个 rollout。三个 point condition 在真实 VLM 引导下完成 `181/324=55.9%` 个 rollout，其整体成功率为 50.0–59.3%；其中 colored GP（TAGPoint）达到 `64/108=59.3%`，相对无 VLM RGB-D 基线高 `7.9 pp`。两个 grasp condition 分别达到 63.0% 和 50.9%，grasp-part 取得表中最高的整体成功率。这个对照说明带 VLM guidance 的完整系统可以超过无 guidance 的 RGB-D 系统，但由于策略分别训练、基线合并两个 run 且没有 paired reset，它不能单独识别“加入 VLM”这一变量的因果贡献。结合附录 A.2 中 Point VLM 的 3-task 等效位置噪声 `63.44 mm/axis`，结果表明下游 action expert 能够在真实 VLM 目标点误差下保持完整任务执行。

**显式姿态通道为 grasp guidance 提供了额外表达能力。** `RGB-D+grasp-part` 的整体成功率为 63.0%，在 lamp 上达到 61.1%，相对三个 point condition 提高 11.1–22.2 个百分点。这一任务依赖的优势与 Ver2 增加旋转输出的设计一致：当接触方向和末端姿态更关键时，grasp guidance 可以提供仅有位置点时缺失的约束。`grasp-part-colored` 的整体成功率为 50.9%，说明旋转信息的收益并不在所有接口组合与任务上稳定出现；同时，旋转预测也增加了上游模型需要拟合的误差维度。

**VLM 误差呈现结构化分布。** Point 与 Grasp VLM 的 task-level 等效 `σ` 分别跨越 `36.15–72.86` 和 `96.26–170.09 mm/axis`，而 n4 在三个任务上固定为 `24 mm/axis`。跨任务差异与非零 bias 表明误差具有任务相关的系统偏置；大于 1 的 anisotropy 显示方向性，相对 n4 非零的 centered SWD 和 radial W1 则显示非高斯形状（表 6）。因此，等效 `σ` 只对齐误差幅度，不代表两者具有相同分布。

表 6 | Point VLM、Grasp VLM 与 n4 受控噪声的分布摘要。Equivalent `σ` 为三个任务的范围，`Δ` 为跨任务跨度；bias norm 与 anisotropy 分别衡量平均偏移和方向性；centered SWD 与 radial W1 分别比较中心化后的二维形状和误差半径。VLM 数值为同一 family 内各 condition 的范围，分布距离均以 n4 为参照。

| Error source | Task-level equivalent σ (mm/axis) | Bias norm (px) | Anisotropy | n4 centered SWD (px) | n4 radial W1 (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Point VLM | 36.15–72.86 (Δ36.71) | 7.27–8.78 | 3.91–5.73 | 11.00–11.85 | 16.80–17.03 |
| Grasp VLM | 96.26–170.09 (Δ73.83) | 26.41–26.50 | 2.27–2.30 | 24.64–26.03 | 44.15–46.41 |
| Controlled noise (n4) | 24.00 (Δ0) | 零均值设定 | 三维各向同性设定 | 0（自参照） | 0（自参照） |

综合表 5、表 6、§2.3 的受控噪声响应和附录 A.2 的等效尺度，下游策略覆盖了 Point VLM 的典型目标点误差范围，并在真实 VLM 引导下完成了相当比例的长时程任务。由此得到的系统级结论是，二维目标点可以作为视觉语义理解与连续动作生成之间的有效接口；等效 `σ` 仅用于对齐误差幅度，而不意味着 VLM 误差与受控噪声具有相同分布。结论范围限于当前三个 FurnitureBench 任务及其观测到的 VLM 误差分布。

### 2.6 FurnitureBench–AutoMate 联合训练的任务规模扩展

#### Motivation 与实验设置

前三个 FurnitureBench 任务用于控制变量地比较 condition，但不能单独支持任务规模扩展的结论。为此，我们将相同的视觉 condition 接口用于 FurnitureBench 与 AutoMate 的联合行为克隆：AutoMate ID 面板覆盖 99 个训练 assembly，每个任务评测 12 条 rollout；OOD 面板使用训练中未见的 assembly `00755`，每个 condition 评测 12 条 rollout；FurnitureBench 则使用 joint formal checkpoint，在三个任务上各评测 36 条 rollout。AutoMate 使用 default ×1、fixed-hardest、SBC-off 协议。

表 7 | FurnitureBench–AutoMate 联合训练的跨环境成功率。`FB Δ` 为 joint formal FurnitureBench overall 减去对应 condition 的更新后 main-experiment overall。AutoMate 100-task 面板主要使用 paired checkpoint，FurnitureBench 面板使用 formal checkpoint，`RGB-D+skill` 的 AutoMate 行也是 formal；因此该表提供 condition-level 的跨环境证据，而不是每一行完全 matched checkpoint 的严格比较。

| Condition | AutoMate ID SR（99 train tasks） | AutoMate OOD SR（`00755`） | FB overall（joint formal） | FB Δ vs main |
| --- | ---: | ---: | ---: | ---: |
| RGB | 739/1188 = 62.2% | 3/12 = 25.0% | 50/108 = 46.3% | −3.70 pp |
| RGB-D | 761/1188 = 64.1% | 9/12 = 75.0% | **67/108 = 62.0%** | **+10.65 pp** |
| RGB-D+skill | 750/1188 = 63.1% | 8/12 = 66.7% | 65/108 = 60.2% | +6.48 pp |
| RGB-D+GP | 754/1188 = 63.5% | **10/12 = 83.3%** | 50/108 = 46.3% | −6.17 pp |
| RGB-D+TAGPoint | **773/1188 = 65.1%** | **10/12 = 83.3%** | 58/108 = 53.7% | −3.70 pp |
| RGB-D+GP+skill | 703/1188 = 59.2% | 8/12 = 66.7% | 57/108 = 52.8% | −10.49 pp |
| RGB-D+grasp | 748/1188 = 63.0% | 9/12 = 75.0% | 47/108 = 43.5% | −2.78 pp |
| RGB-D+colored grasp | 755/1188 = 63.6% | 8/12 = 66.7% | 44/108 = 40.7% | −6.79 pp |

#### 结果与结论边界

联合策略在 99 个 AutoMate 训练 assembly 上达到 `59.2–65.1%`，表明同一个视觉行为克隆策略在约百任务规模仍保留广泛的 assembly 能力；TAGPoint 的 ID 成功率最高，为 `65.1%`。在单个 held-out `00755` 上，GP 与 TAGPoint 均达到 `83.3%`，但这一 OOD 观察目前只有一个任务、每个 condition 12 条 rollout，不能外推为稳定的跨几何泛化排名。

联合训练对 FurnitureBench 的影响具有明显的 condition dependence：RGB-D 与 skill 分别相对 main experiment 提高 `+10.65` 与 `+6.48 pp`，其余六种接口下降 `2.78–10.49 pp`，八种 condition 的平均变化为 `−2.06 pp`。因此，大规模 AutoMate 数据没有一致提升所有 FurnitureBench condition，也没有使原有长时程任务能力整体失效。更准确的结论是：joint policy 获得了 99-task AutoMate 能力，同时以较小的平均损失保留 FurnitureBench 能力，但不同接口受到的正迁移或干扰不同。由于当前 AutoMate annotation 只有 Insert，该实验验证的是 assembly identity 与任务家族的规模扩展，而不是五种 skill 的跨环境迁移。


## 3. 仍在进行或计划中的实验

1. 真机实验：形成与表 1 对齐的 condition 对比表，并提供代表性 demo。
2. VLM 误差分层分析：按目标点误差区间统计 action-expert 成功率，并进一步分解 skill-specific p95 长尾与 failure/OOD 状态。
3. Med-train → high-eval：high randomness 加入零件初始位姿排列组合，预期成功率较低；需要多 seed 和足够 rollout 才能比较 condition。
4. 任务扩展：加入 ManiSkill3 与 Isaac Lab / Isaac Gym 的新任务和数据，验证 guidance point 作为跨任务接口的可扩展性。

## 附录 A：结果来源

- 多任务 condition 主实验：[`multi_task_condition_eval_0610.md`](./multi_task_condition_eval_0610.md#1-总览)
- Main multi-seed 复核：[`main_3seed_experiment_review_0913.md`](./main_3seed_experiment_review_0913.md)
- Multi-seed skill-level 分析：[`skill_level_analysis_multiseed_0917.md`](./skill_level_analysis_multiseed_0917.md)
- Low→Med 空间泛化：[`low2med_generalization.md`](./low2med_generalization.md#results)
- Clean-train → noisy-eval 正式结果：[`annotation_noise_vlm_cover_108.md`](./annotation_noise_vlm_cover_108.md)；早期 seed-0 对照：[`annotation_noise_clean_train_fresh36.md`](./annotation_noise_clean_train_fresh36.md#1-结果图)。
- 主文图 1：`figures/vlm_cover_108/vlm_cover_108_success_numeric.pdf`（task-level SR；仅 Point/Grasp 两条 position-equivalent σ 线）。
- 主文图 2：`figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.pdf`（三任务合并的 position tracking；保留 `n7→Shuffle` endpoint）。
- 附录 tracking grid：`figures/vlm_cover_108/vlm_cover_108_tracking_error.pdf`（3 task × position/orientation/total，仅用于细粒度诊断）。
- VLM 端到端评测：[`vlm_dit_guidance_eval.md`](./vlm_dit_guidance_eval.md#22-真实-vlm-引导误差及其下游影响)。
- FurnitureBench–AutoMate 联合训练：[`automate_multitask_generalist_comparison_0914.md`](./automate_multitask_generalist_comparison_0914.md#跨环境主对比automate-扩展能力与-furniturebench-保持率)。

### 附录 A.1：108 实验补充图

三 task pooled success（仅保留 Point/Grasp 两条 position-equivalent σ 线）：

![Three-task pooled success](figures/vlm_cover_108/vlm_cover_108_success_3task_pooled.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_success_3task_pooled.pdf)

三 task pooled position tracking：

![Three-task pooled position tracking](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.pdf)

Task × metric tracking breakdown：

![Task-by-metric tracking breakdown](figures/vlm_cover_108/vlm_cover_108_tracking_error.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_tracking_error.pdf)

Skill-level success、position/orientation/total tracking：

Skill-level 图中的第三条参考线是 Grasp VLM 的 point-error p95 stress line，不是 orientation-equivalent scale；orientation 对齐不在任何图中绘制。

![Skill-level success rate](figures/vlm_cover_108/vlm_cover_108_skill_success_rate.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_success_rate.pdf)

![Skill-level position tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_position.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_position.pdf)

![Skill-level orientation tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_orientation.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_orientation.pdf)

![Skill-level total tracking](figures/vlm_cover_108/vlm_cover_108_skill_tracking_total.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_total.pdf)

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
