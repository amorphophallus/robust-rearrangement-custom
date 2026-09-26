# JUICER 对比结果的论文整合方案

## 1. 结论与推荐决策

建议在正文中保留 JUICER 的定量结果，但不要把它作为现有 VLM 结果表中的普通可比行，也不要用 JUICER 与本文结果之间的差值归因于 backbone、RGB-D 或 VLM 中的任何单一因素。

推荐将当前 VLM 结果表改成两个清楚分隔的 panel：

- **Panel A：Current study — system-level comparison**。保留当前 Table 3 的全部内部结果，用于回答“在本文系统和评估协议内，实际 VLM guidance 相对于 no-VLM RGB-D baseline 有什么增量”。
- **Panel B：Published FurnitureBench context — not protocol matched**。加入 JUICER 的 TA+CI 和 Multi-task 结果，只用于回答“本文系统的绝对性能是否处于已有工作的合理或有竞争力的范围”。该 panel 不参加加粗、下划线、显著性检验或跨 panel 排名。

如果正文版面不足，次优方案是在 Table 3 后用一句正文给出 JUICER TA+CI 的三个任务结果，并把包含 TA+CI、Multi-task 以及协议差异的完整表放到 appendix。完全不报告 JUICER 数值并非方法学错误，因为本文核心是受控 interface ablation；但 JUICER 使用同一 FurnitureBench 任务且已在 Related Work 中出现，完全省略外部定量锚点容易引起“只与自己比较”的审稿质疑。

本文可以讲的核心故事不是“VLM 击败 JUICER”，而是：

1. 本文的 shared RGB-D DiT action expert 在没有 VLM 时已经是一个较强的内部 baseline；
2. 在这个较强 baseline 上，实际 VLM--TAGPoint 系统仍观察到总体增量，但增量具有明显的 task dependence；
3. JUICER 提供外部性能背景，说明本文完整系统具有竞争力，但不能说明差值来自 DiT backbone、深度输入或 VLM；
4. 对 VLM/interface 作用的证据来自本文内部对比，对外部竞争力的证据来自单独标注为 non-protocol-matched 的 JUICER panel。

## 2. 当前正文中的放置位置

当前最新稿件位于：

`logs/table2-display-20260921/main.tex`

相关位置：

- Table 2（clean scripted guidance）：约第 245--301 行。
- Table 3（actual VLM guidance）：约第 479--553 行。
- JUICER Related Work：约第 680--696 行。
- Discussion 中对 descriptive comparisons 的限制：约第 772--783 行。

### 推荐位置：扩展当前 Table 3，而不是 Table 2

JUICER 不应进入 Table 2。Table 2 的研究问题是：在相同 DiT action expert、训练目标和本文数据条件下，不同 clean conditioning interface 如何影响性能。JUICER 同时改变了训练数据构成、policy scope、视觉输入、policy architecture 和评估统计量，把它放进 Table 2 会破坏该表的受控 ablation 含义。

JUICER 与 Table 3 的关联更自然，因为 Table 3 已经是完整系统层面的结果，并且正文把它定义为 system-level comparison。不过 JUICER 仍不能成为与 RGB-D、GP、TAGPoint 和 grasp 并列排名的普通行。因此，推荐在同一个 table float 中新增独立的 Panel B，并在视觉上和 caption 中明确切断跨 panel 排名。

如果排版允许单独增加一张表，则可把 Panel B 独立成紧随 Table 3 的小表，题为 **Published FurnitureBench context on overlapping tasks**。独立表在方法学上最清楚，但会增加一个 main-text display，并重复部分本文数字。因此，两 panel 的 Table 3 是更节省版面的首选。

## 3. JUICER 的数据来源与应报告的结果

JUICER 的正式来源为：

- Lars Ankile, Anthony Simeonov, Idan Shenfeld, and Pulkit Agrawal. *JUICER: Data-Efficient Imitation Learning for Robotic Assembly*. IROS 2024. arXiv:2404.03729v3.
- 官方页面：<https://arxiv.org/abs/2404.03729>

与本文三个任务重叠的 published average success rates 为：

| JUICER condition | One-leg | Round table | Lamp | 三任务算术平均 |
|---|---:|---:|---:|---:|
| TA+CI | 74.0 | 32.0 | 28.0 | 44.7 |
| Multi-task | 58.0 | 25.0 | 12.0 | 31.7 |

其中：

- TA+CI 是 JUICER 主要结果中最强且最适合作为保守外部锚点的设置。它是按任务训练的模型，不是与本文完全一致的 shared three-task policy。
- Multi-task 是 policy scope 上更接近本文的 published row，但其训练 mix 是 round-table、lamp 和 square-table，共 150 条 demonstrations；不是本文的 one-leg、round-table、lamp 三任务组合。One-leg 是 square-table 的子任务，因此这行不能被称为相同任务集上的 matched multitask baseline。
- JUICER 对每个 task/dataset condition 训练 5 个不同初始化的模型，每个模型评估 100 次 rollouts，并报告 mean 和 max。这里采用论文的 mean，而不是 max。
- JUICER 直接使用 RGB image observations；完整 pipeline 还包含 human teleoperation、trajectory augmentation、collect-and-infer rollout expansion 等数据处理。

只报告较弱的 JUICER Multi-task 行会产生 cherry-picking 风险。因此，若加入 Multi-task，必须同时报告更强的 TA+CI published result。正文叙事可以把 TA+CI 称为“strongest overlapping published reference”，把 Multi-task 称为“closer in policy scope but trained on a different task mixture”。

## 4. 推荐的 Table 3 结构与数据

### 4.1 推荐展示

| Condition | One-leg (%) | Round table (%) | Lamp (%) | Three-task summary |
|---|---:|---:|---:|---:|
| **Panel A: Current study — system-level comparison** |||||
| RGB-D (no VLM) | 81.9 | 48.6 | 23.6 | 51.4% (111/216) |
| RGB-D + GP | 86.1 | 44.4 | 44.4 | 58.3% (63/108) |
| RGB-D + TAGPoint | 86.1 | 41.7 | 50.0 | 59.3% (64/108) |
| RGB-D + GP + skill | 83.3 | 27.8 | 38.9 | 50.0% (54/108) |
| RGB-D + grasp | 88.9 | 38.9 | 61.1 | 63.0% (68/108) |
| RGB-D + colored grasp | 72.2 | 38.9 | 41.7 | 50.9% (55/108) |
| **Panel B: Published context — different protocol** |||||
| JUICER TA+CI (task-specific, RGB) | 74.0 | 32.0 | 28.0 | 44.7%† |
| JUICER Multi-task (RGB)‡ | 58.0 | 25.0 | 12.0 | 31.7%† |

† JUICER 的 three-task summary 是本文根据三个 published task averages 计算的算术平均，不是 JUICER 原文报告的 pooled trial estimator。

‡ JUICER Multi-task 使用 round-table、lamp 和 square-table 训练 mix，而不是本文的 three-task mix。

### 4.2 排名和格式规则

- Panel A 可以保留当前列内的 bold/underline，但 caption 必须写清这些标记只在 Panel A 内计算。
- Panel B 不加粗、不下划线，也不参与 Panel A 的 best/second-best 排名。
- 不对 Panel A 与 Panel B 做显著性检验。
- 不把 JUICER 的 44.7%/31.7% 与本文的 rollout counts 放在同一统计单位下解释。JUICER 数字是 task-level seed averages，本文 Table 3 的 Overall 是等量任务 rollouts 的 pooled result；由于本文每个任务试验数相等，两者数值上都可得到三任务平均，但 estimand 和不确定性结构并不相同。
- 表中可以保留 JUICER 的 three-task summary 方便阅读，但必须用脚注明确它是 derived descriptive statistic。最保守的版本可以把 JUICER 的 summary 写成“--”，只比较三个 task columns。

### 4.3 推荐英文 caption

> **System-level results and published FurnitureBench context.** Panel A compares the current systems with and without actual VLM guidance. The no-VLM RGB-D baseline pools two valid training runs (72 rollouts per task; 216 overall), whereas each VLM-conditioned policy is evaluated on 36 rollouts per task (108 overall). These policies are trained separately and the evaluation starts are not paired, so differences within Panel A are descriptive system comparisons rather than single-variable VLM effects. Panel B reports published JUICER averages on the overlapping FurnitureBench tasks for external context only. JUICER differs in observation modality, policy scope, training-data construction, model training, and evaluation aggregation; its rows are excluded from cross-panel ranking and significance claims. JUICER three-task summaries are arithmetic means computed from the published task averages.

如 caption 过长，可把 task-mix 差异移到表下注释，但必须保留“external context only”“not protocol matched”“excluded from cross-panel ranking”三层限定。

## 5. 数值对比应该如何解释

### 5.1 本文内部：VLM 的增量

以 no-VLM RGB-D 为内部 baseline，VLM--TAGPoint 的观察差值为：

| Comparison | One-leg | Round table | Lamp | Overall |
|---|---:|---:|---:|---:|
| TAGPoint minus no-VLM RGB-D | +4.2 pp | -6.9 pp | +26.4 pp | +7.9 pp |

这组结果支持以下结论：

- 实际 VLM--TAGPoint 系统在总体上高于强 RGB-D baseline；
- 增量不是 uniform improvement，而是明显由 lamp 的提升驱动；
- round-table 上存在退化，因此不能写“VLM improves all tasks”或“VLM consistently improves performance”；
- TAGPoint 只比 GP 多 1/108 个成功 rollout（59.3% 对 58.3%），不能把这一差值写成统计显著优势；
- grasp 达到 63.0%，高于 TAGPoint，因此 TAGPoint 只能称为“best observed point-based interface under actual VLM guidance”，不能称为最佳 VLM interface 或最佳完整系统。

### 5.2 外部背景：本文与 JUICER 的数值位置

相对 JUICER TA+CI 的 published task averages：

| Current system minus JUICER TA+CI | One-leg | Round table | Lamp | 三任务平均差值 |
|---|---:|---:|---:|---:|
| no-VLM RGB-D | +7.9 pp | +16.6 pp | -4.4 pp | +6.7 pp |
| VLM--TAGPoint | +12.1 pp | +9.7 pp | +22.0 pp | +14.6 pp |
| VLM--grasp | +14.9 pp | +6.9 pp | +33.1 pp | +18.3 pp |

这些差值只能描述 numerical context：本文系统的数字在三个重叠任务上总体处于有竞争力的范围，VLM--TAGPoint 和 VLM--grasp 的 published-task averages 数值上高于 JUICER TA+CI。它们不能被解释为 matched performance gains，也不能被用于分解 backbone、depth、data 或 VLM 的贡献。

no-VLM RGB-D 已经高于 JUICER 三任务均值并不是删除 JUICER 的理由。更合适的解释是：本文对 VLM 的内部增量是在一个较强的 action expert baseline 上观察到的，因此 VLM 结果不是通过选择一个明显较弱的内部 baseline 获得的。但这种说法仍应建立在 Panel A 的内部差值上，而不是将 JUICER 当作 VLM ablation。

## 6. 推荐的论文故事线

### 6.1 四步叙事

**第一步：先定义两种证据回答不同问题。**

- Published JUICER context 回答：本文系统与既有 FurnitureBench imitation-learning system 相比，绝对性能处于什么位置？
- 本文 no-VLM 与 actual-VLM rows 回答：在本文 action-expert family 和系统设计内，VLM-generated guidance 是否带来额外的 task-level capability？

**第二步：强调 strong-baseline setting。**

本文 no-VLM shared RGB-D action expert 的 three-task result 已经达到 51.4%。因此 VLM 并非补救一个完全失败的 low-level policy，而是在较强 action expert 上提供额外的 task decomposition 和 visual grounding signal。

**第三步：把 VLM 增量讲成 task-dependent，而不是普遍提升。**

TAGPoint 的总体结果提高 7.9 pp，主要来自 lamp 的 26.4 pp 增量；round-table 下降 6.9 pp。更准确的结论是 VLM guidance 在部分需要更强 stage/target disambiguation 的任务上提供明显价值，但也可能在已有 baseline 较强或 grounding/control mismatch 更严重的任务上带来退化。

**第四步：将贡献收束到 interface 设计。**

外部 JUICER 结果说明完整系统具有竞争力，但本文的可识别贡献仍来自固定 action-expert family 下对 GP、skill、GP+skill、TAGPoint 和 grasp interfaces 的比较。论文不需要转向“新 backbone 优于 JUICER”的主线；那会偏离题目 *What Should a VLM Tell an Action Expert?*，并触发审稿人要求 matched backbone ablation。

### 6.2 推荐英文正文段落

可放在 Table 3 后、当前结果解释段之后：

> The no-VLM RGB-D policy is already a strong system baseline, completing 111/216 rollouts (51.4%). Actual VLM--TAGPoint guidance raises the observed three-task result to 64/108 (59.3%), a 7.9-point system-level difference. This gain is task dependent: TAGPoint improves one-leg and lamp by 4.2 and 26.4 points, respectively, but is 6.9 points lower on round-table. The result therefore supports an added capability from explicit visual grounding on some tasks rather than a uniform benefit from introducing a VLM.

随后加入外部背景：

> For published context, JUICER reports average success rates of 74%, 32%, and 28% on one-leg, round-table, and lamp with its task-specific TA+CI pipeline, and 58%, 25%, and 12% with its multitask setting. Our VLM--TAGPoint system is numerically higher on the three overlapping tasks, while the no-VLM RGB-D system is higher on one-leg and round-table but lower on lamp. These values are not protocol matched: JUICER uses different observation inputs, task mixtures, data expansion, policy training, and evaluation aggregation. We therefore use JUICER only to contextualize absolute performance, not to attribute the observed differences to the VLM, depth input, or action backbone.

### 6.3 Abstract、Introduction 和 Conclusion 中的边界

正文加入 JUICER 后，不建议在 abstract 中写“outperforms JUICER”，因为 abstract 很难同时容纳足够的 protocol caveat。Abstract 继续使用当前的内部表述即可：TAGPoint 在 actual VLM guidance 下是 strongest observed point-based interface。

Introduction/Contributions 可以增加一句有限的 external-positioning 表述：

> On the overlapping simulated FurnitureBench tasks, the complete systems achieve success rates in a competitive range relative to published JUICER results, while our controlled comparisons focus on the information passed from the VLM to a fixed action-expert family.

Conclusion 中可以写：

> Published JUICER results place the absolute system performance in context, but the protocol differences preclude attributing cross-paper gaps to the VLM or action backbone.

避免在 title、abstract 或 contributions 中把 non-matched numerical comparison 升级为 SOTA claim。

## 7. 可以说、需要限定、不能说的结论

### 可以说

- “Our VLM--TAGPoint system numerically exceeds the published JUICER TA+CI averages on the three overlapping tasks.”
- “The JUICER comparison provides external performance context rather than a protocol-matched baseline.”
- “Within our system-level evaluation, TAGPoint is 7.9 percentage points above the no-VLM RGB-D baseline overall.”
- “The observed VLM gain is task dependent and is largest on lamp.”
- “The no-VLM RGB-D result provides a strong internal baseline.”
- “TAGPoint is the strongest observed point-based interface under actual VLM guidance.”

### 必须带限定

- “Competitive with published FurnitureBench results”必须同时说明 overlapping tasks 和 different protocol。
- “VLM improves overall performance”必须说明是 descriptive system comparison、separately trained policies 和 unpaired starts。
- “Shared multitask advantage”必须说明 JUICER Multi-task 的 task mix 与本文不同。
- “Three-task mean”必须说明 JUICER 数值是从 published task averages 派生的算术平均。

### 不能说

- “The VLM outperforms JUICER.”
- “Our method establishes a new state of the art on FurnitureBench.”
- “The DiT/backbone accounts for most of the improvement.”
- “RGB-D is responsible for the cross-paper gain.”
- “VLM guidance consistently improves every task.”
- “TAGPoint is the best VLM interface”或“TAGPoint is the best overall system”，因为 grasp 的总体结果更高。
- “JUICER is a weaker baseline”，因为其训练目标、数据预算和估计量不同。

## 8. 需要在论文中明确写出的局限性

### 8.1 外部比较的主要未对齐项

| 维度 | JUICER | 本文 | 对结论的影响 |
|---|---|---|---|
| Policy scope | TA+CI 主要为 task-specific；另有不同 task mix 的 multitask row | one shared policy for one-leg, round-table, lamp | 不能把差值解释为 multitask 或 architecture 的单独收益 |
| Observation | RGB image observations + proprioception | front/wrist RGB-D + robot state；部分行还有 rendered guidance | 深度和 guidance 同时改变，不能归因 |
| Action model | Diffusion Policy pipeline | DiT action denoiser | 没有 matched backbone ablation |
| Training data | 50 human demos/task，加 rollout expansion 和 corrective augmentation，具体数量按任务变化 | 本文数据 lineage 和 per-task demonstration counts 尚待最终审计并写入 appendix | 数据量与覆盖范围未对齐 |
| Guidance | 无 VLM intermediate guidance | actual VLM or scripted interface | JUICER 不是 VLM/no-VLM 单变量 baseline |
| Multitask mix | round-table、lamp、square-table | one-leg、round-table、lamp | JUICER Multi-task row 不是相同任务集合 |
| Seeds/evaluation | 5 models/condition，100 rollouts/model，报告 mean/max | RGB-D pools two runs；VLM rows 36 rollouts/task；Table 3 使用 pooled counts | 不确定性和 estimand 不一致 |
| Simulator/protocol | JUICER 发表时的 FurnitureBench implementation 和 horizon | 当前 custom fork、当前 evaluation settings | 可能存在版本、reset 和终止条件差异 |
| Paired starts | 跨论文不可能 paired | 本文内部也未 paired | 不能对小差值做强因果或显著性结论 |

### 8.2 推荐英文 limitations 段落

可加入 Discussion 的现有 descriptive-comparisons 段落：

> The JUICER rows provide published context rather than a protocol-matched baseline. JUICER evaluates RGB policies trained with task-specific or differently composed multitask data, including human demonstrations, corrective trajectory augmentation, and successful-policy rollouts; our systems use a shared RGB-D DiT action expert and, for the conditioned rows, an additional VLM-generated interface. The studies also differ in simulator lineage, task mixture, training-run counts, rollout aggregation, and potentially episode and reset protocols. Consequently, the cross-paper numerical gaps cannot identify the contribution of the VLM, depth input, action backbone, or dataset construction. A matched comparison would require retraining the JUICER policy parameterization and our DiT on the same audited dataset, observation inputs, three-task mixture, seeds, and evaluation starts.

随后保留当前内部限制：

> Even within our protocol, the VLM and no-VLM rows use separately trained policies and unpaired evaluation starts, and TAGPoint exceeds GP by only one success in 108 rollouts. The reported differences should therefore be interpreted as descriptive system-level outcomes rather than isolated causal effects or statistically resolved rankings.

## 9. 若要支持 backbone 结论，需要补什么实验

当前 JUICER 对比不能支持 backbone attribution。如果后续确实希望在论文中提出“DiT backbone 带来主要提升”，至少需要下面的 matched ablation：

| 固定项 | 要求 |
|---|---|
| Dataset | 完全相同的 audited demonstrations 和 episode selection |
| Task scope | 同一 one-leg/round-table/lamp shared-policy training mix |
| Observation | 同为 RGB，或同为 RGB-D；不能一边 RGB、一边 RGB-D |
| Vision encoder | 同一 encoder、pretraining、resolution 和 augmentation |
| Action horizon | 相同 prediction/action horizons 和 replanning frequency |
| Optimization | 相同 batch、updates/epochs、scheduler 和 early stopping |
| Seeds | 至少相同数量的多个 training seeds |
| Evaluation | 相同 simulator commit、randomness、timeouts、reset seeds 和 paired starts |
| 唯一变量 | JUICER/DP action backbone 与本文 DiT action backbone |

在没有这一实验前，论文应继续把 action-expert family 视为固定研究平台，把贡献聚焦在 conditioning interface，而不是声称 backbone superiority。

如果只能追加一项实验，优先级取决于希望强化的 claim：

1. 若强化论文当前的 VLM/interface 主线：优先做相同训练 lineage、相同数量 seeds 和 matched reset states 的 no-VLM 与 VLM/interface 比较。
2. 若新增 backbone 主线：再做 DP 与 DiT 的严格 matched ablation。这个实验成本更高，并会扩大论文 scope。

## 10. 改稿检查清单

- [ ] Table 3 使用两个 panel，JUICER 不作为普通可比行。
- [ ] 同时报告 JUICER TA+CI 与 Multi-task，避免只选择较弱结果。
- [ ] 明确 Multi-task 的训练 task mix 不同。
- [ ] JUICER 不参与 bold/underline 和显著性比较。
- [ ] caption 写明 external context only、not protocol matched、excluded from ranking。
- [ ] 正文分别叙述 external competitiveness 与 internal VLM increment。
- [ ] 报告 TAGPoint 相对 no-VLM 的 task-wise 差值，包括 round-table 的退化。
- [ ] 不把跨论文差值归因于 backbone、depth、VLM 或 data。
- [ ] 不写 SOTA、outperforms JUICER 或 uniformly improves。
- [ ] Discussion 加入完整 protocol mismatch limitations。
- [ ] 最终 dataset audit 后补齐本文 per-task demonstration counts 和 lineage。
- [ ] 检查 Table 3、Related Work、Discussion 和 Conclusion 的措辞一致。

## 11. 最终推荐的一句话定位

> JUICER is included as a transparent external anchor for absolute FurnitureBench performance, while the contribution of VLM-generated guidance is evaluated only through the paper's internal system comparisons; neither comparison identifies a backbone effect.
