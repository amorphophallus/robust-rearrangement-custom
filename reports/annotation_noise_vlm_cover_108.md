# VLM 覆盖范围噪声补充实验（108 rollout/cell）

> [!IMPORTANT]
> **所有成功率都使用每 task 108 rollout。** 第一张关键结果表中的 Overall 是三个 task 合计，因此每格为 324 rollout；成功率表中不存在 72-rollout 结果。

> [!NOTE]
> `72` 只表示 tracking/diagnostics 的覆盖量：n0–n4 与 Shuffle 排除旧 seed-0 tracking 后，只汇总新 seed-1/2，即每 task `tracking_n=72`；n5–n7 与 r180 使用三个 replicate，为 `tracking_n=108`。旧 saved-8 tracking 未混入。

> [!NOTE]
> 数据索引固定为 `JSON → tables → figures`：JSON 只负责生成表，`table_validation.csv` 负责确认表内 eval 结果的分母、算术、pooled/replicate 一致性、skill-level 覆盖、三 task 合并和 VLM σ 覆盖；校验通过后，图只读取已校验的结果表。

> grasp n0–n7 同时改变位置和旋转，只能解释为联合扰动。r180 是 orientation-only stress endpoint；VLM 角度指标是未处理 gripper/object symmetry 的 raw rotation error，r180 不等价于完整 VLM 错误分布。Shuffle 与 r180 均不进入连续趋势拟合。

## 1. 结果图与主要结论

> 定量图同时导出 PNG 预览与可无限缩放的矢量 PDF；论文排版与细节检查统一使用 PDF。

### 1.1 task-level success（真实噪声尺度）

![Pooled success on the actual noise scale](figures/vlm_cover_108/vlm_cover_108_success_numeric.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_success_numeric.pdf)

横轴使用真实 position σ/axis（mm；n0–n7 分别为 0、3、6、12、24、48、96、192 mm），因此 n0–n4 会集中在低噪声段，不再等距排列。每个 task 右侧的灰色窄轴按 fresh36 的方式显示 `n7→Shuffle` categorical endpoint，Shuffle 不被当作连续 mm 噪声点。小 marker 与细连线只读取已校验的 `success_tracking_pooled.csv`，不直接 query JSON；本图不展示 replicate 离散度。每个 task 只画两条上游 position-equivalent VLM σ 线：蓝色 Point VLM、红色 Grasp VLM。Grasp 的 orientation-equivalent 对齐只在正文和附录中说明，不在图上增加第三条纵线。n7=192 mm/axis 覆盖最大的 task-level position σ；ordinal trend 导出见 `ordinal_trends.csv`。

VLM σ 表示 VLM 点误差相当于多少 `mm/axis` 的 3D 位置噪声。统计单位是一个 control step 的有效 VLM–GT 点对；同一 task 内，三个 point condition 的 VLM 点合并计算一条 Point VLM position-equivalent σ 线，两个 grasp condition 的 VLM 点合并计算一条 Grasp VLM position-equivalent σ 线。Point 使用 `3×36=108` 条 source trajectory，Grasp 使用 `2×36=72` 条 source trajectory；每条线使用这些 trajectory 的全部有效 control-step pairs。Grasp 的 orientation-equivalent 对齐不作为原始 VLM orientation σ，也不在图中绘制；它只在正文文字和附录中报告。

对每个有效点对，先定义二维残差 `e_i = p_i^VLM − p_i^GT`，并计算 VLM 投影误差 `R_VLM = sqrt[(1/N) Σ_i ||e_i||₂²]`。参考噪声在同一帧的 GT 3D 点和相机标定上生成：`P_{i,n,j} = P_i^GT + σ_n z_{i,j}`，其中 `z_{i,j} ~ N(0, I₃)`，逐分量截断到 `[-2, 2]`，`σ_n ∈ {0, 3, 6, 12, 24} mm`；每个有效点对生成 `M=200` 个 Monte Carlo 样本，并在各噪声档之间复用同一批标准样本。

将扰动点投影到前视相机，得到 `r_{i,n,j} = π(P_{i,n,j}) − π(P_i^GT)`，再计算参考投影误差 `R_n = sqrt[(1/(N·M)) Σ_i Σ_j ||r_{i,n,j}||₂²]`。将参考点按 `R_n` 排序；若相邻两点满足 `R_n ≤ R_VLM ≤ R_{n+1}`，则 `σ_eq = σ_n + [(R_VLM − R_n)/(R_{n+1} − R_n)]·(σ_{n+1} − σ_n)`；超过最高参考档时，使用最高两档线性外推。

`pooled` 表示把同一 task、同一 VLM family 的 condition rows 合并，并按有效点对数加权：`σ_{f,t} = [Σ_c N_{c,t} σ_{c,t}] / [Σ_c N_{c,t}]`，其中 `N_{c,t}` 是有效 control-step pair 数。它不是不同 step 的 min–max，也不是把 Point 和 Grasp 混在一起；三个 task 分别计算，因此每个 task 的图上只有两条独立的 VLM σ 线。由于本轮 108-run 表没有保存原始 VLM residual vector，这里的 family-level σ 是对 formal diagnostic 的 condition×task Equivalent σ summary 做的 pair-weighted aggregation，而不是重新从 raw residual 逐点拟合。

图中六条 task-specific position-equivalent 纵线的统计量、有效 control-step pair 数和三 task pooled 汇总放在附录。它们是把真实 VLM 误差与数值噪声轴对齐的参考，不是额外成功率观测。

### 1.2 三 task 合并的 position tracking response

![Three-task pooled position tracking](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_tracking_position_3task_pooled.pdf)

这张正文图把 `one_leg`、`round_table` 和 `lamp` 合并到同一条 condition 曲线。Position tracking 按各 task 的有效 final skill-state 数加权，而不是简单平均三个 task 的均值。主轴显示真实 n0–n7 position σ/axis，右侧窄轴显示 `n7→Shuffle`；图不展示 replicate 离散度。两条三-task position-equivalent VLM 纵线分别对应 Point 与 Grasp。Grasp 的 orientation/total tracking 仍在补充网格图与正文数值中报告，不与主要 position tracking 结论挤在同一张正文图里。

Tracking 的参照点是该 condition 实际提供给 policy 的 displayed target：numeric-noise 条件使用加噪后的 target，VLM formal rollout 使用 clean-GT target。因此 n7 的 tracking error 是“末端执行器到错误目标点的距离”，不是末端执行器到 clean GT 的距离；它适合描述点跟随行为，不应直接当作任务几何误差。

### 1.3 补充 pooled summary 与 task/metric breakdown

![Three-task pooled success](figures/vlm_cover_108/vlm_cover_108_success_3task_pooled.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_success_3task_pooled.pdf)

![Task-by-metric tracking breakdown](figures/vlm_cover_108/vlm_cover_108_tracking_error.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_tracking_error.pdf)

Pooled success 按三个 task 的 rollout 数直接合并，因此每个 condition/noise cell 为 `3×108=324` 个 rollout。补充 tracking 网格按 fresh36 指标拆分 3 个 task × 3 个指标（position、orientation、total），其中 `total = pos_m / 0.01 + ori_deg / 5`；point condition 没有 pose-aware orientation/total 定义，因此后两行只显示 grasp condition。该网格用于诊断任务和指标差异，不作为正文 tracking 主图。r180 保留在表格中，不单独画 endpoint 图。

n0–n7 的 position 与 orientation 扰动是绑定的，而不是两个独立实验轴：

| Level | Position σ/axis (mm) | Orientation σ (deg) | Schedule |
| --- | --- | --- | --- |
| n0 | 0.0 | 0.0 | bound position+orientation perturbation |
| n1 | 3.0 | 2.5 | bound position+orientation perturbation |
| n2 | 6.0 | 5.0 | bound position+orientation perturbation |
| n3 | 12.0 | 10.0 | bound position+orientation perturbation |
| n4 | 24.0 | 20.0 | bound position+orientation perturbation |
| n5 | 48.0 | 40.0 | bound position+orientation perturbation |
| n6 | 96.0 | 60.0 | bound position+orientation perturbation |
| n7 | 192.0 | 90.0 | bound position+orientation perturbation |
| r180 | 0.0 | 180.0 | orientation-only endpoint |

图下方的 orientation 对齐只用文字说明。我们汇总 Grasp 在 n0（未注入 orientation noise）下、以 clean-GT 为参照的 orientation tracking residual，并按 `tracking_state_count` 加权；随后将该 residual 与固定的 `0/2.5/5/10/20/40/60/90°` orientation schedule 线性匹配，再用绑定的 position schedule 得到行为等效位置。这个量表示下游策略在姿态误差下的行为等效覆盖位置，而不是 raw VLM orientation σ；完整数值放在附录。

### 1.4 主要结论

#### Motivation

完全 clean evaluation 对检验策略能否使用一个条件是必要的，但它把 guidance point 设为正确且无偏的 oracle target，因而不能区分不同条件在真实部署中的优势。`main.tex` 所指出的关键缺口是：部署时 VLM 可能预测错误的 skill 或 semantic type，即使语义正确，二维目标也可能偏离真正的交互位置。一个只在 clean oracle 上比较的结果因此无法回答“哪种 condition 能在上游 guidance 不完美时继续支持长时程装配”。本实验把这个缺口转化为可测的问题：在共享 DiT action expert、相同训练数据和相同模型容量下，Point、colored GP、GP+skill 以及 grasp 条件能否在各自真实 VLM 误差尺度附近保持任务完成，并且是否仍然跟随被扰动的空间目标。

#### Experimental setting

我们在 `one_leg`、`round_table` 和 `lamp` 三个长时程家具装配任务上评估五种 condition：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`、`rgbd+grasp-part` 和 `rgbd+grasp-part-colored`。所有 condition 使用相同的多任务 demonstrations、DiT action-expert architecture 和优化流程；每个 `condition × noise × task` 汇总三个 evaluation replicate，每格为 `108` 条 rollout，condition-level pooled cell 为 `324` 条 rollout。Point family 的上游参照来自 Point VLM，Grasp family 的上游参照来自 Grasp VLM，因此两类表示在各自的误差尺度上分别判断，而不把两种 VLM 混成一条 σ。

#### Evaluation protocol

噪声只在 test time 注入 guidance，不改变训练数据。n0–n7 的 position σ/axis 为 `0、3、6、12、24、48、96、192 mm`；grasp 的 n0–n7 同时绑定 orientation schedule `0、2.5、5、10、20、40、60、90°`。Shuffle 把当前 semantic state 的点替换为另一 state 的 guidance，作为 categorical endpoint；grasp 的 r180 只改变 rotation，不进入 n0–n7 的连续趋势。Task success 是完整装配是否完成；skill success 是 `completed/entered`；tracking 取每个 semantic segment 的最终有效状态。Position tracking 是末端执行器到 policy 实际接收的 displayed target 的欧氏距离，因而 numeric-noise 的 n7 tracking 不是到 clean GT 的距离；grasp 的 `total` 定义为 `pos_m / 0.01 + ori_deg / 5`。

VLM σ 由有效 control-step 的 VLM–GT 点对计算，并通过同帧相机标定下的 3-D Monte Carlo 投影误差映射到共同的 `mm/axis` 参考。Point 与 Grasp 分开、task 内独立估计；`pooled` 仅表示按有效点对数加权的汇总。数据索引固定为 `JSON → tables → figures`，图只读取已通过 `table_validation.csv` 的结果表；本轮 34 项校验全部为 `ok`。成功率报告 pooled Wilson 区间，tracking/diagnostic 的分母单独标记（n0–n4 与 Shuffle 为每 task 72，n5–n7 为每 task 108），不把两者混作同一个样本量。

#### Analysis

真实 VLM 工作区间的比较首先给出 Point family 的答案。Point-VLM σ 为 one_leg `36.15`、round_table `72.86`、lamp `68.34 mm/axis`，分别落在 n4–n5、n5–n6、n5–n6。colored GP 相对 GP 的 task success 在这些窗口中分别为：one_leg 的 n5 `88.9% vs 86.1%`，round_table 的 n5/n6 `42.6%/38.9% vs 40.7%/33.3%`，lamp 的 n5/n6 `42.6%/38.9% vs 34.3%/33.3%`。相应 position tracking 差异不超过 `0.35 cm`，说明 colored GP 的主要收益体现在 task completion，而非单纯降低末端到 target 的距离；其优势主要集中在 round_table 和 lamp，one_leg 更接近持平。Skill-level 结果与此一致但不均匀：在 n5–n6，colored GP 相对 GP 在 pick 上高 `1.0–1.4 pp`，在 screw 上高 `3.3–4.8 pp`，push、place 和 insert 则相当或方向混合。

`rgbd+GP+skill` 提供了一个有用的 representation reference。n6 的跨 task place success 为 `88.0%`，高于 colored GP 的 `82.7%` 和 GP 的 `82.9%`，而 push、pick、insert 仍为 `97.5%/96.4%/97.8%`。这表明显式 low-dimensional skill 不会带来减益；由于 colored GP 在图像中提供了相近的语义类别信息，GP+skill 可以被视为 colored GP 的 semantic-information upper-bound/reference，但不能被表述为统计意义上的性能上界。

Grasp family 的真实工作区间位于 n6–n7，因为 Grasp-VLM σ 为 one_leg `170.09`、round_table `100.28`、lamp `96.26 mm/axis`。grasp-part 在 clean n0 相对 GP 高 `8.9 pp`，在 n6/n7 仍高 `4.6/3.1 pp`；但 task-level 的正向差距主要来自 lamp，one_leg 与 round_table 在高噪声区间混合或接近持平。colored grasp-part 相对 colored GP 的差距由 `+6.8 pp` 收窄到 n6 的 `+2.5 pp` 和 n7 的 `0.0 pp`。在 skill level，clean advantage 主要集中于 place 和 screw；在 n7，grasp-part 相对 GP 的优势分别为 `+3.5` 和 `+1.7 pp`。Grasp 同时传递位置和旋转，n6→n7 的 pooled position/orientation/total tracking 从 `14.86→25.12 cm`、`98.4→124.8°` 和 `34.5→50.1` 增长；因此它的 clean advantage 伴随着更大的联合误差压力，VLM 的旋转拟合是额外代价，但不是本文的中心变量。

噪声幅度增加或 Shuffle 没有造成反常的成功率提升。n0→n7 的 pooled 变化范围为 `-6.2` 到 `+4.0 pp`，同一 condition 的 Wilson 区间重叠，局部回升没有跨 replicate 一致出现。相反，tracking 给出了更稳定的行为信号：n7 的 position tracking 在 `15/15` 个配对单元均高于 Shuffle，平均高 `11.0 cm`；grasp 的 orientation 和 total 也分别高 `62.5°` 和 `23.5` 个 total 单位。大数值噪声保留当前 semantic subtask、只把同一目标推远，模型仍尝试原 subtask；Shuffle 改变点所代表的 subtask，tracking 反而更低，这与模型跟随被置换 guidance 并尝试另一动作一致。该解释是行为层面的推断，不等同于已经证明了内部 gating 机制。

最后，VLM 实验本身显示这个接口能够产出完整任务。Point family 的 formal guidance evaluation 完成 `181/324=55.9%` 个 rollout，Grasp family 完成 `123/216=56.9%`；其中 colored GP 为 `64/108=59.3%`，grasp-part 为 `68/108=63.0%`。按 task，Point family 的 one_leg/round_table/lamp 为 `85.2%/38.0%/44.4%`，Grasp family 为 `80.6%/38.9%/51.4%`。这些成功率与 n7=`192 mm/axis` 覆盖六条 task-level VLM σ 的结果合在一起，说明上游 VLM 的典型点误差可以被下游 action expert 转化为仍可执行的装配序列；双系统的成功因此体现在“误差经过接口后仍能完成任务”，而不只是接口输出格式正确。

#### Conclusion

在本实验的真实 VLM 条件下，colored GP 是 point family 中最有说服力的稳健接口，但它的优势是 task- and skill-dependent，而不是整条噪声曲线上的普遍排序。它在 round_table/lamp 的 VLM-equivalent 工作区间保留了相对 plain GP 的 task-completion 优势；`GP+skill` 则提供了显式语义信息的参考上限。Grasp 信息在 clean 条件下带来额外收益，尤其体现在下游 place/screw，但它必须同时承受位置和旋转误差，因此其优势在强噪声下收窄。

这组结果支持一个更窄、也更可检验的系统结论：二维 guidance point 是连接 VLM visual-semantic grounding 与连续 action generation 的有效外部空间参考；当点的语义仍指向当前 subtask 时，action expert 可以在显著的空间偏移下继续完成任务，并在 Shuffle 下表现出对被置换 guidance 的响应。该结论的边界是 RMS-equivalent 的典型 VLM 误差范围；skill-specific p95 长尾、failure/OOD 时序、旋转对称性以及更严格的 paired-reset 机制检验仍需单独验证。

### 1.5 fresh36 结论在 108 实验中的逐条复核

| fresh36 结论 | 108 复核 | 108 证据 |
| --- | --- | --- |
| colored GP 在数值噪声下最稳定 | 在真实 Point-VLM 尺度附近有条件支持 | Point-VLM 对应的 n4–n6 区间内，colored GP 在 round_table/lamp 的 task success 高于 GP，skill-level 优势集中在 pick 与 screw；但整条 n0–n7 的 pooled range 为 7.7 pp，并不是所有 task/skill 都最小。 |
| GP 与 colored GP 可能使用不同 tracking 机制 | 仍不支持机制区分 | 108 表中 position tracking 与数值噪声的 pooled-cell Pearson r 为 GP 0.976、colored GP 0.972；两者都随噪声近似单调增大，但这只能说明共享的行为响应，不能识别内部机制差异。 |
| GP round_table 的提升集中在第一段 screw | 复现并更清楚 | round_table task success 从 n0 的 28.7% 升至 n7 的 44.4%；skill-level screw 在 n0→n4 为 72.9%→84.0%，而其余 skill 为 push `100.0%→100.0%`；pick `86.5%→89.5%`；place `87.2%→83.1%`；insert `95.9%→98.4%`。 |
| GP+skill 对连续数值噪声不稳定，并可能拒绝错误 guidance | 成功率不支持 gating，但支持其作为 semantic-information reference | GP+skill 的 n0–n4 pooled range 仅 3.4 pp，n0→n7 为 +3.7 pp，n7→Shuffle 为 -0.6 pp；n6 的跨 task place 为 88.0%，高于 colored GP 的 82.7%，说明显式 low-dimensional skill 没有带来减益，可作为 colored GP 的参考上限。 |
| grasp 能容忍数值噪声，但 task variation 更大 | 支持 clean advantage，强噪声下相对优势收窄 | Grasp VLM σ 位于 n6–n7；grasp-part 相对 GP 的 task-level gap 为 n0 `+8.9 pp`、n6 `+4.6 pp`、n7 `+3.1 pp`，而 colored grasp-part 相对 colored GP 为 `+6.8→+2.5→0.0 pp`。skill-level clean advantage 主要集中在 place/screw，旋转与位置联合误差使 Grasp 的 σ 更大。 |
| Shuffle 的成功率下降说明正确 semantic guidance 仍有用 | 成功率证据变弱，tracking 证据变强 | n7→Shuffle 的 overall success 变化为 rgbd+GP -4.9 pp；rgbd+colored GP +0.3 pp；rgbd+GP+skill -0.6 pp；rgbd+grasp-part -2.5 pp；rgbd+grasp-part-colored +4.3 pp；但 n7 的 position tracking 在 15/15 个配对单元均高于 Shuffle，平均差 11.0 cm。更合适的解释是 Shuffle 改变了 semantic subtask，模型仍会跟随被置换的点，而不是成功率必然下降。 |

这里的“支持”表示 108 表中的方向和任务分解与 fresh36 一致；“部分保留”表示结论只在某一噪声区间成立；“机制解释仍待验证”表示现有汇总表能够显示行为差异，但没有 paired reset、donor-target 距离或 replicate-level tracking uncertainty 来完成因果区分。

### 1.6 新的行为解释与边界

**Shuffle 与大数值噪声检验的是两种不同的扰动。** 大数值噪声保留当前 semantic subtask，只把目标点沿同一指导语义推离；在 108 中，成功率在 n7 仍保持，而 tracking error 随噪声增大。Shuffle 则把点替换为另一 semantic state 的 guidance；此时 tracking error 反而较低，且 task success 没有系统性下降。一个与这些数据一致的解释是：模型确实使用点来组织动作。当点仍属于当前 subtask 但位置被大幅扰动时，模型尝试完成原 subtask，却必须承受错误目标；当点来自另一个 subtask 时，模型会跟随该点尝试另一个动作，因此终点更接近被置换后的 guidance。这个解释是行为层面的推断；要把它提升为机制证据，还需要记录 donor state、目标间距离，以及 paired reset 下的 episode-level trajectory。

**上游 VLM 的误差在下游被转化为可完成的任务。** Formal VLM guidance evaluation 中，Point family 的成功率为 `181/324=55.9%`，其中 colored GP 为 `64/108=59.3%`；Grasp family 的成功率为 `123/216=56.9%`，其中 grasp-part 为 `68/108=63.0%`。这些结果表明，VLM 生成的 guidance 已被 action expert 用来完成完整任务，而不是只在接口层面产生了可解析输出。对应的 task-level σ 为 Point `36.15–72.86 mm/axis`、Grasp `96.26–170.09 mm/axis`，n7 的 `192 mm/axis` 覆盖了这些典型尺度；因此在本 benchmark 内，双系统的误差接口是闭合的。这个结论仍限于 RMS-equivalent 的典型误差，不等价于所有 skill-specific p95 长尾、旋转对称性和 OOD 时序都已被覆盖。

**skill-level 结果显示鲁棒性并不均匀。** 五类 skill 合并所有 task 与 condition 后，push 的 success rate 为 98.0%→97.7%，pick 为 95.7%→95.0%，place 为 83.1%→81.6%，insert 为 98.5%→98.1%，screw 为 84.2%→87.3%。place 是整体较弱且在高噪声下略降的 skill；push、pick 和 insert 更稳定；screw 的 pooled success 可上升，但其变化受 task progression 与进入后续 skill 的选择效应影响，不能直接解释为噪声带来的能力提升。

上述 tracking 差异是跨 task、跨 condition 的一致性描述，不是 replicate-level 显著性检验；当前图按要求不展示 replicate 离散度。因此文中使用“支持”“一致于”“提示”，不使用未经检验的“证明机制”。

### 1.7 skill-level success rate 与 tracking error（5 skills × 3 tasks）

Success rate

![Skill-level success rate](figures/vlm_cover_108/vlm_cover_108_skill_success_rate.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_success_rate.pdf)

Position error

![Skill-level position tracking error](figures/vlm_cover_108/vlm_cover_108_skill_tracking_position.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_position.pdf)

Orientation error

![Skill-level orientation tracking error](figures/vlm_cover_108/vlm_cover_108_skill_tracking_orientation.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_orientation.pdf)

Total error

![Skill-level total tracking error](figures/vlm_cover_108/vlm_cover_108_skill_tracking_total.png)

[矢量 PDF](figures/vlm_cover_108/vlm_cover_108_skill_tracking_total.pdf)

四张图沿用 fresh36 的 cascading skill 定义：每个子图是一种 skill type（`push/pick/place/insert/screw`）和一个 task，曲线表示不同 condition；success rate 为 `completed/entered`，不是 task success。tracking 统计每个 skill state 的最终有效段，并按同一 skill type 汇总；n0–n4 与 Shuffle 排除旧 seed-0 tracking 后使用 72 条 tracking rollout，n5–n7 使用 108 条。主轴使用真实 n0–n7 position σ/axis，右侧窄轴按 `n7→Shuffle` 显示 categorical endpoint；不展示 replicate 离散度。Point 条件只有 position tracking，Grasp 条件同时显示 position、orientation 和 `total = pos_m / 0.01 + ori_deg / 5`。每个主轴还叠加三条 VLM 参考线：task-level Point VLM RMS-equivalent σ、task-level Grasp VLM RMS-equivalent σ，以及该 skill 的 Grasp VLM p95-equivalent；若 p95 超出 n7，红色点线在右边界截断并标记 `p95>n7`。这些参考线只读取已校验的 `vlm_sigma_by_task.csv` 与 `vlm_skill_error_reference.csv`。

### 1.8 关键 Overall 数据（每格 3 task × 108 = 324 rollout）

| Condition | n0 | n5 | n6 | n7 | Shuffle | r180 |
| --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | 157/324 (48.5%) | 174/324 (53.7%) | 168/324 (51.9%) | 170/324 (52.5%) | 154/324 (47.5%) | — |
| rgbd+colored GP | 178/324 (54.9%) | 188/324 (58.0%) | 175/324 (54.0%) | 180/324 (55.6%) | 181/324 (55.9%) | — |
| rgbd+GP+skill | 172/324 (53.1%) | 185/324 (57.1%) | 193/324 (59.6%) | 184/324 (56.8%) | 182/324 (56.2%) | — |
| rgbd+grasp-part | 186/324 (57.4%) | 175/324 (54.0%) | 183/324 (56.5%) | 180/324 (55.6%) | 172/324 (53.1%) | 176/324 (54.3%) |
| rgbd+grasp-part-colored | 200/324 (61.7%) | 176/324 (54.3%) | 183/324 (56.5%) | 180/324 (55.6%) | 194/324 (59.9%) | 191/324 (59.0%) |

## 2. 完整 pooled success 与 95% Wilson CI

| Condition | Noise | Success | 95% Wilson CI |
| --- | --- | --- | --- |
| rgbd+GP | n0 | 157/324 (48.5%) | [43.1, 53.9]% |
| rgbd+GP | n1 | 175/324 (54.0%) | [48.6, 59.4]% |
| rgbd+GP | n2 | 155/324 (47.8%) | [42.5, 53.3]% |
| rgbd+GP | n3 | 163/324 (50.3%) | [44.9, 55.7]% |
| rgbd+GP | n4 | 168/324 (51.9%) | [46.4, 57.2]% |
| rgbd+GP | n5 | 174/324 (53.7%) | [48.3, 59.1]% |
| rgbd+GP | n6 | 168/324 (51.9%) | [46.4, 57.2]% |
| rgbd+GP | n7 | 170/324 (52.5%) | [47.0, 57.8]% |
| rgbd+GP | shuffle | 154/324 (47.5%) | [42.2, 53.0]% |
| rgbd+colored GP | n0 | 178/324 (54.9%) | [49.5, 60.3]% |
| rgbd+colored GP | n1 | 168/324 (51.9%) | [46.4, 57.2]% |
| rgbd+colored GP | n2 | 182/324 (56.2%) | [50.7, 61.5]% |
| rgbd+colored GP | n3 | 163/324 (50.3%) | [44.9, 55.7]% |
| rgbd+colored GP | n4 | 179/324 (55.2%) | [49.8, 60.6]% |
| rgbd+colored GP | n5 | 188/324 (58.0%) | [52.6, 63.3]% |
| rgbd+colored GP | n6 | 175/324 (54.0%) | [48.6, 59.4]% |
| rgbd+colored GP | n7 | 180/324 (55.6%) | [50.1, 60.9]% |
| rgbd+colored GP | shuffle | 181/324 (55.9%) | [50.4, 61.2]% |
| rgbd+GP+skill | n0 | 172/324 (53.1%) | [47.6, 58.5]% |
| rgbd+GP+skill | n1 | 183/324 (56.5%) | [51.0, 61.8]% |
| rgbd+GP+skill | n2 | 180/324 (55.6%) | [50.1, 60.9]% |
| rgbd+GP+skill | n3 | 181/324 (55.9%) | [50.4, 61.2]% |
| rgbd+GP+skill | n4 | 177/324 (54.6%) | [49.2, 60.0]% |
| rgbd+GP+skill | n5 | 185/324 (57.1%) | [51.7, 62.4]% |
| rgbd+GP+skill | n6 | 193/324 (59.6%) | [54.1, 64.8]% |
| rgbd+GP+skill | n7 | 184/324 (56.8%) | [51.3, 62.1]% |
| rgbd+GP+skill | shuffle | 182/324 (56.2%) | [50.7, 61.5]% |
| rgbd+grasp-part | n0 | 186/324 (57.4%) | [52.0, 62.7]% |
| rgbd+grasp-part | n1 | 179/324 (55.2%) | [49.8, 60.6]% |
| rgbd+grasp-part | n2 | 191/324 (59.0%) | [53.5, 64.2]% |
| rgbd+grasp-part | n3 | 182/324 (56.2%) | [50.7, 61.5]% |
| rgbd+grasp-part | n4 | 188/324 (58.0%) | [52.6, 63.3]% |
| rgbd+grasp-part | n5 | 175/324 (54.0%) | [48.6, 59.4]% |
| rgbd+grasp-part | n6 | 183/324 (56.5%) | [51.0, 61.8]% |
| rgbd+grasp-part | n7 | 180/324 (55.6%) | [50.1, 60.9]% |
| rgbd+grasp-part | shuffle | 172/324 (53.1%) | [47.6, 58.5]% |
| rgbd+grasp-part | r180 | 176/324 (54.3%) | [48.9, 59.7]% |
| rgbd+grasp-part-colored | n0 | 200/324 (61.7%) | [56.3, 66.9]% |
| rgbd+grasp-part-colored | n1 | 201/324 (62.0%) | [56.6, 67.2]% |
| rgbd+grasp-part-colored | n2 | 189/324 (58.3%) | [52.9, 63.6]% |
| rgbd+grasp-part-colored | n3 | 172/324 (53.1%) | [47.6, 58.5]% |
| rgbd+grasp-part-colored | n4 | 180/324 (55.6%) | [50.1, 60.9]% |
| rgbd+grasp-part-colored | n5 | 176/324 (54.3%) | [48.9, 59.7]% |
| rgbd+grasp-part-colored | n6 | 183/324 (56.5%) | [51.0, 61.8]% |
| rgbd+grasp-part-colored | n7 | 180/324 (55.6%) | [50.1, 60.9]% |
| rgbd+grasp-part-colored | shuffle | 194/324 (59.9%) | [54.5, 65.1]% |
| rgbd+grasp-part-colored | r180 | 191/324 (59.0%) | [53.5, 64.2]% |

## 3. VLM skill mismatch 与 OOD 长尾诊断

前面的 rollout 可视化揭示了一个需要单独处理的问题：某些 `round_table/place` 帧里的 VLM 点误差只接近 n1–n2，但 formal Grasp VLM 的整体误差仍然更大，而且不同 skill 之间并不在同一误差水平上。这不是矛盾，而是两个分布被混在了一起。第一，VLM 的点预测本质上是按 skill 变化的目标回归：`pick`、`push`、`place`、`insert` 和 `screw` 对可见部位、遮挡关系和目标几何的要求不同；把它们 pooled 成一条 σ 会隐藏这种 skill mismatch。第二，rollout failure 会把下游状态带到 VLM 训练分布之外，例如姿态偏离、部分装配、遮挡和相机视角变化。OOD 状态会制造长尾误差；但长尾不只出现在失败 rollout 中，也会出现在最终成功的 rollout 中，因此它描述的是上游观测风险，而不是简单的 failure label。

### 3.1 极端误差样例（formal Grasp VLM）

![Largest VLM point errors in formal rollouts](../logs/vlm_extreme_error_20260910/generated/vlm_largest_errors_global.png)

图 3.1 | Formal Grasp VLM rollout 中最大的 2-D 点误差。每个小图取一个 rollout×skill 的最大有效残差；绿色圆圈是 scripted target，红色叉号是保存下来的 VLM point，标题中的 `success/failure` 是该 rollout 的最终状态。图中最大的误差为 `one_leg/pick` 的 185.8 px，其次为 `one_leg/screw` 的 184.4 px、`lamp/screw` 的 179.8 px 和 `one_leg/place` 的 173.6 px。该 montage 用来展示长尾的形态和 OOD 候选状态，不作为新的成功率或 σ 估计，也不替代 JSON → tables → figures 的定量链路。

这张图也解释了为什么单独查看一张低误差的 `round_table/place` 帧会低估 VLM 的总体风险：局部帧可以接近 n1–n2，但在不同 skill、不同 rollout state 和 failure/OOD 状态下，VLM 点会出现远离 scripted target 的长尾偏移。

### 3.2 Skill-level 长尾统计与 p95 选择

正式 Grasp VLM 诊断按有效 VLM–GT control-step pair 汇总。这里的 `p50` 表示典型误差，`RMS` 对较大误差更敏感，`p95` 表示最坏的 5% 尾部；`>40/>70/>100 px` 是直接报告大误差占比。对应的 `mm` 数值把同一个像素误差映射到共同的 projected-Gaussian RMS reference，便于与 n0–n7 的位置噪声轴比较。

本轮正式 diagnostic summary 只保存了 Grasp VLM 的 task×skill residual，因此表格中的 skill-specific 长尾来自 Grasp VLM；Point VLM 仍只以 task-level RMS-equivalent σ 进入图中的蓝色参考线。不能把蓝色线解释成五类 skill 各自的 Point VLM 误差。

设每个有效点对的二维像素残差幅度为 `e_i = ||p_i^VLM − p_i^GT||₂`。`p95` 定义为满足 `P(e_i ≤ q) ≥ 0.95` 的最小 `q`；在有限样本中，它是按 `e_i` 从小到大排序后位于 95% 分位的位置。阈值尾部占比定义为 `tail_τ = N(e_i > τ) / N`。因此 p95 给出尾部边界，tail fraction 给出尾部质量，两者应同时报告。

| Skill | Valid pairs | p50 (px / mm) | RMS (px / mm) | p95 (px / mm) | >40 px | >70 px | >100 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| push | 9,344 | 68.3 px / 107.1 mm | 77.3 px / 120.6 mm | 117.4 px / 181.8 mm | 72.1% | 46.8% | 24.8% |
| pick | 62,459 | 57.5 px / 89.0 mm | 73.8 px / 115.3 mm | 124.8 px / 195.8 mm | 63.5% | 41.5% | 22.6% |
| place | 12,833 | 44.9 px / 70.0 mm | 56.2 px / 87.1 mm | 95.1 px / 146.6 mm | 53.0% | 23.6% | 8.7% |
| insert | 1,383 | 42.6 px / 68.7 mm | 56.5 px / 91.3 mm | 87.6 px / 141.6 mm | 49.0% | 32.5% | 7.4% |
| screw | 24,267 | 42.0 px / 64.9 mm | 54.9 px / 83.8 mm | 91.0 px / 137.4 mm | 47.7% | 29.6% | 4.7% |

这张表说明大误差并不是均匀分布的：`push` 的 >40 px、>70 px 和 >100 px 占比分别为 72.1%、46.8% 和 24.8%；`pick` 的对应比例为 63.5%、41.5% 和 22.6%。因此“某一张图看起来像 n1–n2”不能代表所有 skill 的 VLM 误差等级。

p95 比均值更能表达 OOD 风险，因为它不会被大量小误差稀释；但 p95 不能单独作为噪声 σ。它只保留尾部位置，忽略了尾部以下的误差质量，而且部分 skill 的 p95-equivalent 已超过 n7=192 mm/axis，例如 place 的 one_leg p95-equivalent 为 270.6 mm/axis。因此本报告采用 `p50 + RMS + p95 + tail fraction` 的四件套：RMS-equivalent 作为主要 VLM σ 线，p95-equivalent 作为 skill-specific stress line，大误差占比用来说明尾部质量。

在所有 skill-level 图中，主轴现在标出三条纵向参考线：蓝色虚线是 task-level Point VLM RMS-equivalent σ，红色虚线是 task-level Grasp VLM RMS-equivalent σ，红色点线是该 skill 的 Grasp VLM p95-equivalent。p95 超过 n7 时，点线在图的右边界截断，并标注 `p95>n7`；这表示尾部已经超出当前噪声设计，而不是把超出的值伪装成 n7。

该诊断也限定了结论边界：108 实验可以证明下游策略在已测 n0–n7 范围内能够承受上游 VLM 的典型误差，并显示哪些 skill 的 VLM 尾部更危险；它不能证明 rollout failure/OOD 状态下的 VLM 误差已经被完整覆盖。下一步应优先按 skill 和 rollout state 分层报告 failure/OOD 比例，并保留每个 control step 的 VLM–GT residual，而不是只保留一个 pooled σ。

### 3.3 VLM guidance 的系统级结论

VLM 误差的关键结果不在于接口是否报错，而在于上游预测已经能够穿过下游控制接口并产生完整任务。Point family 的 formal guidance evaluation 完成 `181/324=55.9%` 个 rollout，Grasp family 完成 `123/216=56.9%`；在各自 family 内，colored GP 为 `64/108=59.3%`，grasp-part 为 `68/108=63.0%`。这些成功率与 108 噪声实验在对应 σ 区间的稳定 task success 共同说明：下游 action expert 可以把典型 VLM 点误差转化为仍可执行的动作序列，双系统在本 benchmark 的误差范围内实现了端到端闭合。该结论不延伸到 skill-specific p95 长尾或 failure/OOD 状态下的完整覆盖，后者仍需按 state 分层的 VLM residual 与 paired rollout 验证。

## 4. 三个 replicate 的独立结果

### 4.1 Overall（每 replicate 合并三个 task，n=108）

| Condition | Noise | Replicate | Success | 95% Wilson CI |
| --- | --- | --- | --- | --- |
| rgbd+GP | n0 | 0 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP | n0 | 1 | 48/108 (44.4%) | [35.4, 53.8]% |
| rgbd+GP | n0 | 2 | 52/108 (48.1%) | [39.0, 57.5]% |
| rgbd+GP | n1 | 0 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+GP | n1 | 1 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+GP | n1 | 2 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP | n2 | 0 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+GP | n2 | 1 | 45/108 (41.7%) | [32.8, 51.1]% |
| rgbd+GP | n2 | 2 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+GP | n3 | 0 | 56/108 (51.9%) | [42.5, 61.0]% |
| rgbd+GP | n3 | 1 | 46/108 (42.6%) | [33.7, 52.0]% |
| rgbd+GP | n3 | 2 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP | n4 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+GP | n4 | 1 | 48/108 (44.4%) | [35.4, 53.8]% |
| rgbd+GP | n4 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+GP | n5 | 0 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+GP | n5 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP | n5 | 2 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+GP | n6 | 0 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+GP | n6 | 1 | 53/108 (49.1%) | [39.8, 58.4]% |
| rgbd+GP | n6 | 2 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP | n7 | 0 | 53/108 (49.1%) | [39.8, 58.4]% |
| rgbd+GP | n7 | 1 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+GP | n7 | 2 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+GP | shuffle | 0 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+GP | shuffle | 1 | 46/108 (42.6%) | [33.7, 52.0]% |
| rgbd+GP | shuffle | 2 | 53/108 (49.1%) | [39.8, 58.4]% |
| rgbd+colored GP | n0 | 0 | 56/108 (51.9%) | [42.5, 61.0]% |
| rgbd+colored GP | n0 | 1 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+colored GP | n0 | 2 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+colored GP | n1 | 0 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+colored GP | n1 | 1 | 49/108 (45.4%) | [36.3, 54.8]% |
| rgbd+colored GP | n1 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n2 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n2 | 1 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+colored GP | n2 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n3 | 0 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+colored GP | n3 | 1 | 51/108 (47.2%) | [38.1, 56.6]% |
| rgbd+colored GP | n3 | 2 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+colored GP | n4 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n4 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+colored GP | n4 | 2 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+colored GP | n5 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n5 | 1 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+colored GP | n5 | 2 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+colored GP | n6 | 0 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+colored GP | n6 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+colored GP | n6 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+colored GP | n7 | 0 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+colored GP | n7 | 1 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+colored GP | n7 | 2 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+colored GP | shuffle | 0 | 56/108 (51.9%) | [42.5, 61.0]% |
| rgbd+colored GP | shuffle | 1 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+colored GP | shuffle | 2 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP+skill | n0 | 0 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+GP+skill | n0 | 1 | 53/108 (49.1%) | [39.8, 58.4]% |
| rgbd+GP+skill | n0 | 2 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP+skill | n1 | 0 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+GP+skill | n1 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP+skill | n1 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+GP+skill | n2 | 0 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+GP+skill | n2 | 1 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+GP+skill | n2 | 2 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+GP+skill | n3 | 0 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+GP+skill | n3 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+GP+skill | n3 | 2 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+GP+skill | n4 | 0 | 51/108 (47.2%) | [38.1, 56.6]% |
| rgbd+GP+skill | n4 | 1 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+GP+skill | n4 | 2 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+GP+skill | n5 | 0 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+GP+skill | n5 | 1 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP+skill | n5 | 2 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+GP+skill | n6 | 0 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+GP+skill | n6 | 1 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP+skill | n6 | 2 | 72/108 (66.7%) | [57.3, 74.8]% |
| rgbd+GP+skill | n7 | 0 | 68/108 (63.0%) | [53.6, 71.5]% |
| rgbd+GP+skill | n7 | 1 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+GP+skill | n7 | 2 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+GP+skill | shuffle | 0 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+GP+skill | shuffle | 1 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+GP+skill | shuffle | 2 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part | n0 | 0 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+grasp-part | n0 | 1 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+grasp-part | n0 | 2 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part | n1 | 0 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part | n1 | 1 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+grasp-part | n1 | 2 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+grasp-part | n2 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part | n2 | 1 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+grasp-part | n2 | 2 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part | n3 | 0 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+grasp-part | n3 | 1 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part | n3 | 2 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+grasp-part | n4 | 0 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part | n4 | 1 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part | n4 | 2 | 67/108 (62.0%) | [52.6, 70.6]% |
| rgbd+grasp-part | n5 | 0 | 56/108 (51.9%) | [42.5, 61.0]% |
| rgbd+grasp-part | n5 | 1 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+grasp-part | n5 | 2 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+grasp-part | n6 | 0 | 70/108 (64.8%) | [55.4, 73.2]% |
| rgbd+grasp-part | n6 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part | n6 | 2 | 52/108 (48.1%) | [39.0, 57.5]% |
| rgbd+grasp-part | n7 | 0 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+grasp-part | n7 | 1 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part | n7 | 2 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part | shuffle | 0 | 53/108 (49.1%) | [39.8, 58.4]% |
| rgbd+grasp-part | shuffle | 1 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+grasp-part | shuffle | 2 | 57/108 (52.8%) | [43.4, 61.9]% |
| rgbd+grasp-part | r180 | 0 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part | r180 | 1 | 55/108 (50.9%) | [41.6, 60.2]% |
| rgbd+grasp-part | r180 | 2 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part-colored | n0 | 0 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part-colored | n0 | 1 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part-colored | n0 | 2 | 73/108 (67.6%) | [58.3, 75.7]% |
| rgbd+grasp-part-colored | n1 | 0 | 68/108 (63.0%) | [53.6, 71.5]% |
| rgbd+grasp-part-colored | n1 | 1 | 69/108 (63.9%) | [54.5, 72.3]% |
| rgbd+grasp-part-colored | n1 | 2 | 64/108 (59.3%) | [49.8, 68.1]% |
| rgbd+grasp-part-colored | n2 | 0 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+grasp-part-colored | n2 | 1 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+grasp-part-colored | n2 | 2 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part-colored | n3 | 0 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+grasp-part-colored | n3 | 1 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+grasp-part-colored | n3 | 2 | 59/108 (54.6%) | [45.2, 63.7]% |
| rgbd+grasp-part-colored | n4 | 0 | 68/108 (63.0%) | [53.6, 71.5]% |
| rgbd+grasp-part-colored | n4 | 1 | 54/108 (50.0%) | [40.7, 59.3]% |
| rgbd+grasp-part-colored | n4 | 2 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+grasp-part-colored | n5 | 0 | 58/108 (53.7%) | [44.3, 62.8]% |
| rgbd+grasp-part-colored | n5 | 1 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part-colored | n5 | 2 | 52/108 (48.1%) | [39.0, 57.5]% |
| rgbd+grasp-part-colored | n6 | 0 | 61/108 (56.5%) | [47.1, 65.4]% |
| rgbd+grasp-part-colored | n6 | 1 | 60/108 (55.6%) | [46.2, 64.6]% |
| rgbd+grasp-part-colored | n6 | 2 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+grasp-part-colored | n7 | 0 | 67/108 (62.0%) | [52.6, 70.6]% |
| rgbd+grasp-part-colored | n7 | 1 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+grasp-part-colored | n7 | 2 | 50/108 (46.3%) | [37.2, 55.7]% |
| rgbd+grasp-part-colored | shuffle | 0 | 63/108 (58.3%) | [48.9, 67.2]% |
| rgbd+grasp-part-colored | shuffle | 1 | 65/108 (60.2%) | [50.8, 68.9]% |
| rgbd+grasp-part-colored | shuffle | 2 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part-colored | r180 | 0 | 66/108 (61.1%) | [51.7, 69.8]% |
| rgbd+grasp-part-colored | r180 | 1 | 62/108 (57.4%) | [48.0, 66.3]% |
| rgbd+grasp-part-colored | r180 | 2 | 63/108 (58.3%) | [48.9, 67.2]% |

### 4.2 By task（每格 n=36）

| Condition | Noise | Task | Replicate | Success | 95% Wilson CI |
| --- | --- | --- | --- | --- | --- |
| rgbd+GP | n0 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP | n0 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+GP | n0 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP | n0 | round_table | 0 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n0 | round_table | 1 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+GP | n0 | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n0 | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP | n0 | lamp | 1 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n0 | lamp | 2 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+GP | n1 | one_leg | 0 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+GP | n1 | one_leg | 1 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+GP | n1 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP | n1 | round_table | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n1 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP | n1 | round_table | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | n1 | lamp | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n1 | lamp | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n1 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n2 | one_leg | 0 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+GP | n2 | one_leg | 1 | 26/36 (72.2%) | [56.0, 84.2]% |
| rgbd+GP | n2 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP | n2 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | n2 | round_table | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n2 | round_table | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP | n2 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n2 | lamp | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n2 | lamp | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n3 | one_leg | 0 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+GP | n3 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+GP | n3 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP | n3 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP | n3 | round_table | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n3 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP | n3 | lamp | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP | n3 | lamp | 1 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+GP | n3 | lamp | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n4 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP | n4 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP | n4 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP | n4 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP | n4 | round_table | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n4 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP | n4 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n4 | lamp | 1 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n4 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n5 | one_leg | 0 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+GP | n5 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP | n5 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP | n5 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | n5 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP | n5 | round_table | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP | n5 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n5 | lamp | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | n5 | lamp | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n6 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP | n6 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP | n6 | one_leg | 2 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+GP | n6 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP | n6 | round_table | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n6 | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | n6 | lamp | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n6 | lamp | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | n6 | lamp | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | n7 | one_leg | 0 | 27/36 (75.0%) | [58.9, 86.2]% |
| rgbd+GP | n7 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP | n7 | one_leg | 2 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+GP | n7 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP | n7 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP | n7 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP | n7 | lamp | 0 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP | n7 | lamp | 1 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP | n7 | lamp | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP | shuffle | one_leg | 0 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+GP | shuffle | one_leg | 1 | 27/36 (75.0%) | [58.9, 86.2]% |
| rgbd+GP | shuffle | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP | shuffle | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP | shuffle | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP | shuffle | round_table | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP | shuffle | lamp | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP | shuffle | lamp | 1 | 5/36 (13.9%) | [6.1, 28.7]% |
| rgbd+GP | shuffle | lamp | 2 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+colored GP | n0 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n0 | one_leg | 1 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+colored GP | n0 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+colored GP | n0 | round_table | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n0 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n0 | round_table | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n0 | lamp | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n0 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+colored GP | n0 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+colored GP | n1 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n1 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+colored GP | n1 | one_leg | 2 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+colored GP | n1 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n1 | round_table | 1 | 6/36 (16.7%) | [7.9, 31.9]% |
| rgbd+colored GP | n1 | round_table | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n1 | lamp | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n1 | lamp | 1 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+colored GP | n1 | lamp | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n2 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+colored GP | n2 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+colored GP | n2 | one_leg | 2 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+colored GP | n2 | round_table | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | n2 | round_table | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | n2 | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | n2 | lamp | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+colored GP | n2 | lamp | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+colored GP | n2 | lamp | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | n3 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+colored GP | n3 | one_leg | 1 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+colored GP | n3 | one_leg | 2 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+colored GP | n3 | round_table | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | n3 | round_table | 1 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+colored GP | n3 | round_table | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n3 | lamp | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | n3 | lamp | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | n3 | lamp | 2 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+colored GP | n4 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n4 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n4 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n4 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n4 | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n4 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+colored GP | n4 | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n4 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+colored GP | n4 | lamp | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n5 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n5 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+colored GP | n5 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+colored GP | n5 | round_table | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+colored GP | n5 | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n5 | round_table | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n5 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | n5 | lamp | 1 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+colored GP | n5 | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | n6 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n6 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+colored GP | n6 | one_leg | 2 | 27/36 (75.0%) | [58.9, 86.2]% |
| rgbd+colored GP | n6 | round_table | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | n6 | round_table | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | n6 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+colored GP | n6 | lamp | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n6 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+colored GP | n6 | lamp | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n7 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+colored GP | n7 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+colored GP | n7 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+colored GP | n7 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | n7 | round_table | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n7 | round_table | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+colored GP | n7 | lamp | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+colored GP | n7 | lamp | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+colored GP | n7 | lamp | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | shuffle | one_leg | 0 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+colored GP | shuffle | one_leg | 1 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+colored GP | shuffle | one_leg | 2 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+colored GP | shuffle | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | shuffle | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+colored GP | shuffle | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+colored GP | shuffle | lamp | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+colored GP | shuffle | lamp | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+colored GP | shuffle | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP+skill | n0 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n0 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP+skill | n0 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n0 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n0 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n0 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n0 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP+skill | n0 | lamp | 1 | 6/36 (16.7%) | [7.9, 31.9]% |
| rgbd+GP+skill | n0 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP+skill | n1 | one_leg | 0 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n1 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP+skill | n1 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP+skill | n1 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n1 | round_table | 1 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n1 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n1 | lamp | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP+skill | n1 | lamp | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP+skill | n1 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP+skill | n2 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP+skill | n2 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n2 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n2 | round_table | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | n2 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n2 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n2 | lamp | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | n2 | lamp | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP+skill | n2 | lamp | 2 | 7/36 (19.4%) | [9.8, 35.0]% |
| rgbd+GP+skill | n3 | one_leg | 0 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n3 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n3 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n3 | round_table | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+GP+skill | n3 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | n3 | round_table | 2 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+GP+skill | n3 | lamp | 0 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP+skill | n3 | lamp | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP+skill | n3 | lamp | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP+skill | n4 | one_leg | 0 | 26/36 (72.2%) | [56.0, 84.2]% |
| rgbd+GP+skill | n4 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP+skill | n4 | one_leg | 2 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+GP+skill | n4 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n4 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | n4 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n4 | lamp | 0 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+GP+skill | n4 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n4 | lamp | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP+skill | n5 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+GP+skill | n5 | one_leg | 1 | 26/36 (72.2%) | [56.0, 84.2]% |
| rgbd+GP+skill | n5 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP+skill | n5 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP+skill | n5 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP+skill | n5 | round_table | 2 | 23/36 (63.9%) | [47.6, 77.5]% |
| rgbd+GP+skill | n5 | lamp | 0 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n5 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+GP+skill | n5 | lamp | 2 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP+skill | n6 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n6 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | n6 | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n6 | round_table | 0 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+GP+skill | n6 | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+GP+skill | n6 | round_table | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+GP+skill | n6 | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP+skill | n6 | lamp | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+GP+skill | n6 | lamp | 2 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+GP+skill | n7 | one_leg | 0 | 35/36 (97.2%) | [85.8, 99.5]% |
| rgbd+GP+skill | n7 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | n7 | one_leg | 2 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+GP+skill | n7 | round_table | 0 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n7 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | n7 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+GP+skill | n7 | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP+skill | n7 | lamp | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP+skill | n7 | lamp | 2 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+GP+skill | shuffle | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+GP+skill | shuffle | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+GP+skill | shuffle | one_leg | 2 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+GP+skill | shuffle | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+GP+skill | shuffle | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+GP+skill | shuffle | round_table | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+GP+skill | shuffle | lamp | 0 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+GP+skill | shuffle | lamp | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+GP+skill | shuffle | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part | n0 | one_leg | 0 | 27/36 (75.0%) | [58.9, 86.2]% |
| rgbd+grasp-part | n0 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part | n0 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n0 | round_table | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part | n0 | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part | n0 | round_table | 2 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part | n0 | lamp | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part | n0 | lamp | 1 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part | n0 | lamp | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n1 | one_leg | 0 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+grasp-part | n1 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | n1 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n1 | round_table | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n1 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n1 | round_table | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part | n1 | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n1 | lamp | 1 | 10/36 (27.8%) | [15.8, 44.0]% |
| rgbd+grasp-part | n1 | lamp | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part | n2 | one_leg | 0 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | n2 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part | n2 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part | n2 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n2 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n2 | round_table | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n2 | lamp | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n2 | lamp | 1 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part | n2 | lamp | 2 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part | n3 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part | n3 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | n3 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n3 | round_table | 0 | 21/36 (58.3%) | [42.2, 72.9]% |
| rgbd+grasp-part | n3 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n3 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n3 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n3 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n3 | lamp | 2 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part | n4 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n4 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | n4 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part | n4 | round_table | 0 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part | n4 | round_table | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part | n4 | round_table | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part | n4 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n4 | lamp | 1 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part | n4 | lamp | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n5 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part | n5 | one_leg | 1 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part | n5 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n5 | round_table | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n5 | round_table | 1 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part | n5 | round_table | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | n5 | lamp | 0 | 9/36 (25.0%) | [13.8, 41.1]% |
| rgbd+grasp-part | n5 | lamp | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part | n5 | lamp | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part | n6 | one_leg | 0 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+grasp-part | n6 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n6 | one_leg | 2 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part | n6 | round_table | 0 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part | n6 | round_table | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part | n6 | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n6 | lamp | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part | n6 | lamp | 1 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part | n6 | lamp | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n7 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part | n7 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part | n7 | one_leg | 2 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | n7 | round_table | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part | n7 | round_table | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part | n7 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n7 | lamp | 0 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | n7 | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | n7 | lamp | 2 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | shuffle | one_leg | 0 | 27/36 (75.0%) | [58.9, 86.2]% |
| rgbd+grasp-part | shuffle | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+grasp-part | shuffle | one_leg | 2 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part | shuffle | round_table | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part | shuffle | round_table | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part | shuffle | round_table | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part | shuffle | lamp | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part | shuffle | lamp | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | shuffle | lamp | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | r180 | one_leg | 0 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part | r180 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part | r180 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part | r180 | round_table | 0 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part | r180 | round_table | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part | r180 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part | r180 | lamp | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part | r180 | lamp | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part | r180 | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n0 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part-colored | n0 | one_leg | 1 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+grasp-part-colored | n0 | one_leg | 2 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part-colored | n0 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | n0 | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | n0 | round_table | 2 | 21/36 (58.3%) | [42.2, 72.9]% |
| rgbd+grasp-part-colored | n0 | lamp | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n0 | lamp | 1 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part-colored | n0 | lamp | 2 | 22/36 (61.1%) | [44.9, 75.2]% |
| rgbd+grasp-part-colored | n1 | one_leg | 0 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part-colored | n1 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part-colored | n1 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | n1 | round_table | 0 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n1 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n1 | round_table | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n1 | lamp | 0 | 21/36 (58.3%) | [42.2, 72.9]% |
| rgbd+grasp-part-colored | n1 | lamp | 1 | 22/36 (61.1%) | [44.9, 75.2]% |
| rgbd+grasp-part-colored | n1 | lamp | 2 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part-colored | n2 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | n2 | one_leg | 1 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part-colored | n2 | one_leg | 2 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part-colored | n2 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | n2 | round_table | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part-colored | n2 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part-colored | n2 | lamp | 0 | 19/36 (52.8%) | [37.0, 68.0]% |
| rgbd+grasp-part-colored | n2 | lamp | 1 | 22/36 (61.1%) | [44.9, 75.2]% |
| rgbd+grasp-part-colored | n2 | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n3 | one_leg | 0 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part-colored | n3 | one_leg | 1 | 26/36 (72.2%) | [56.0, 84.2]% |
| rgbd+grasp-part-colored | n3 | one_leg | 2 | 23/36 (63.9%) | [47.6, 77.5]% |
| rgbd+grasp-part-colored | n3 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n3 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part-colored | n3 | round_table | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part-colored | n3 | lamp | 0 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part-colored | n3 | lamp | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n3 | lamp | 2 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part-colored | n4 | one_leg | 0 | 34/36 (94.4%) | [81.9, 98.5]% |
| rgbd+grasp-part-colored | n4 | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part-colored | n4 | one_leg | 2 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part-colored | n4 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n4 | round_table | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part-colored | n4 | round_table | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n4 | lamp | 0 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part-colored | n4 | lamp | 1 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part-colored | n4 | lamp | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part-colored | n5 | one_leg | 0 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | n5 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part-colored | n5 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | n5 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n5 | round_table | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n5 | round_table | 2 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part-colored | n5 | lamp | 0 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part-colored | n5 | lamp | 1 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n5 | lamp | 2 | 8/36 (22.2%) | [11.7, 38.1]% |
| rgbd+grasp-part-colored | n6 | one_leg | 0 | 24/36 (66.7%) | [50.3, 79.8]% |
| rgbd+grasp-part-colored | n6 | one_leg | 1 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part-colored | n6 | one_leg | 2 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | n6 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n6 | round_table | 1 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part-colored | n6 | round_table | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n6 | lamp | 0 | 23/36 (63.9%) | [47.6, 77.5]% |
| rgbd+grasp-part-colored | n6 | lamp | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | n6 | lamp | 2 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n7 | one_leg | 0 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part-colored | n7 | one_leg | 1 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part-colored | n7 | one_leg | 2 | 21/36 (58.3%) | [42.2, 72.9]% |
| rgbd+grasp-part-colored | n7 | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | n7 | round_table | 1 | 11/36 (30.6%) | [18.0, 46.9]% |
| rgbd+grasp-part-colored | n7 | round_table | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | n7 | lamp | 0 | 21/36 (58.3%) | [42.2, 72.9]% |
| rgbd+grasp-part-colored | n7 | lamp | 1 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part-colored | n7 | lamp | 2 | 12/36 (33.3%) | [20.2, 49.7]% |
| rgbd+grasp-part-colored | shuffle | one_leg | 0 | 33/36 (91.7%) | [78.2, 97.1]% |
| rgbd+grasp-part-colored | shuffle | one_leg | 1 | 30/36 (83.3%) | [68.1, 92.1]% |
| rgbd+grasp-part-colored | shuffle | one_leg | 2 | 29/36 (80.6%) | [65.0, 90.2]% |
| rgbd+grasp-part-colored | shuffle | round_table | 0 | 14/36 (38.9%) | [24.8, 55.1]% |
| rgbd+grasp-part-colored | shuffle | round_table | 1 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | shuffle | round_table | 2 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part-colored | shuffle | lamp | 0 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part-colored | shuffle | lamp | 1 | 20/36 (55.6%) | [39.6, 70.5]% |
| rgbd+grasp-part-colored | shuffle | lamp | 2 | 17/36 (47.2%) | [32.0, 63.0]% |
| rgbd+grasp-part-colored | r180 | one_leg | 0 | 28/36 (77.8%) | [61.9, 88.3]% |
| rgbd+grasp-part-colored | r180 | one_leg | 1 | 31/36 (86.1%) | [71.3, 93.9]% |
| rgbd+grasp-part-colored | r180 | one_leg | 2 | 32/36 (88.9%) | [74.7, 95.6]% |
| rgbd+grasp-part-colored | r180 | round_table | 0 | 15/36 (41.7%) | [27.1, 57.8]% |
| rgbd+grasp-part-colored | r180 | round_table | 1 | 13/36 (36.1%) | [22.5, 52.4]% |
| rgbd+grasp-part-colored | r180 | round_table | 2 | 16/36 (44.4%) | [29.5, 60.4]% |
| rgbd+grasp-part-colored | r180 | lamp | 0 | 23/36 (63.9%) | [47.6, 77.5]% |
| rgbd+grasp-part-colored | r180 | lamp | 1 | 18/36 (50.0%) | [34.5, 65.5]% |
| rgbd+grasp-part-colored | r180 | lamp | 2 | 15/36 (41.7%) | [27.1, 57.8]% |

## 5. 实际扰动、tracking 覆盖与可见性

下表的 `diagnostics_n` 不是成功率分母。n0–n4/Shuffle 的 `72` 来自新 seed-1/2；n5–n7/r180 的 `108` 来自三个 replicate。所有 success cell 仍为 108 rollout，且不因 target 越界而删除。可见率下降时，应将数值偏移与 annotation 消失/出界分别解释。

| Condition | Noise | Task | diagnostics_n | Position norm RMS (mm) | Position norm P90 (mm) | Rotation geodesic RMS (deg) | Rotation geodesic P90 (deg) | Workspace valid | Front visible | Invalid/non-finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | n0 | one_leg | 72 | 0.0 | 0.0 | N/A | N/A | 96.3% | 96.3% | 3.7% |
| rgbd+GP | n0 | round_table | 72 | 0.0 | 0.0 | N/A | N/A | 74.2% | 74.3% | 25.7% |
| rgbd+GP | n0 | lamp | 72 | 0.0 | 0.0 | N/A | N/A | 87.6% | 87.6% | 12.4% |
| rgbd+GP | n1 | one_leg | 72 | 5.0 | 6.9 | N/A | N/A | 97.7% | 97.7% | 2.3% |
| rgbd+GP | n1 | round_table | 72 | 5.0 | 6.9 | N/A | N/A | 67.6% | 70.1% | 29.9% |
| rgbd+GP | n1 | lamp | 72 | 5.1 | 7.0 | N/A | N/A | 86.9% | 86.9% | 13.1% |
| rgbd+GP | n2 | one_leg | 72 | 10.1 | 13.8 | N/A | N/A | 96.1% | 96.1% | 3.9% |
| rgbd+GP | n2 | round_table | 72 | 10.0 | 13.9 | N/A | N/A | 66.0% | 69.8% | 30.2% |
| rgbd+GP | n2 | lamp | 72 | 10.1 | 13.9 | N/A | N/A | 91.4% | 91.4% | 8.6% |
| rgbd+GP | n3 | one_leg | 72 | 20.0 | 27.6 | N/A | N/A | 91.5% | 95.9% | 4.1% |
| rgbd+GP | n3 | round_table | 72 | 20.1 | 27.9 | N/A | N/A | 70.1% | 74.5% | 25.5% |
| rgbd+GP | n3 | lamp | 72 | 20.2 | 27.7 | N/A | N/A | 88.6% | 90.3% | 9.7% |
| rgbd+GP | n4 | one_leg | 72 | 39.9 | 55.2 | N/A | N/A | 84.6% | 96.7% | 3.3% |
| rgbd+GP | n4 | round_table | 72 | 40.0 | 55.7 | N/A | N/A | 69.5% | 76.0% | 24.0% |
| rgbd+GP | n4 | lamp | 72 | 40.0 | 55.3 | N/A | N/A | 79.0% | 86.5% | 13.5% |
| rgbd+GP | n5 | one_leg | 108 | 80.4 | 109.8 | N/A | N/A | 74.9% | 97.5% | 2.5% |
| rgbd+GP | n5 | round_table | 108 | 79.7 | 110.4 | N/A | N/A | 59.7% | 69.4% | 30.6% |
| rgbd+GP | n5 | lamp | 108 | 81.0 | 110.5 | N/A | N/A | 75.1% | 87.5% | 12.5% |
| rgbd+GP | n6 | one_leg | 108 | 160.6 | 219.6 | N/A | N/A | 59.8% | 92.6% | 3.1% |
| rgbd+GP | n6 | round_table | 108 | 158.6 | 219.1 | N/A | N/A | 53.5% | 74.5% | 24.0% |
| rgbd+GP | n6 | lamp | 108 | 163.0 | 221.0 | N/A | N/A | 68.7% | 86.4% | 9.1% |
| rgbd+GP | n7 | one_leg | 108 | 317.7 | 436.4 | N/A | N/A | 40.3% | 64.1% | 4.3% |
| rgbd+GP | n7 | round_table | 108 | 322.9 | 441.8 | N/A | N/A | 35.6% | 55.2% | 20.8% |
| rgbd+GP | n7 | lamp | 108 | 323.4 | 441.8 | N/A | N/A | 40.7% | 56.0% | 13.9% |
| rgbd+GP | shuffle | one_leg | 72 | 206.5 | 251.3 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+GP | shuffle | round_table | 72 | 118.4 | 258.2 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+GP | shuffle | lamp | 72 | 131.9 | 191.3 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+colored GP | n0 | one_leg | 72 | 0.0 | 0.0 | N/A | N/A | 96.4% | 96.4% | 3.6% |
| rgbd+colored GP | n0 | round_table | 72 | 0.0 | 0.0 | N/A | N/A | 77.5% | 78.7% | 21.2% |
| rgbd+colored GP | n0 | lamp | 72 | 0.0 | 0.0 | N/A | N/A | 88.0% | 88.0% | 12.0% |
| rgbd+colored GP | n1 | one_leg | 72 | 5.0 | 6.9 | N/A | N/A | 97.7% | 97.7% | 2.3% |
| rgbd+colored GP | n1 | round_table | 72 | 5.0 | 6.9 | N/A | N/A | 73.1% | 76.3% | 23.7% |
| rgbd+colored GP | n1 | lamp | 72 | 5.1 | 7.0 | N/A | N/A | 89.0% | 89.0% | 11.0% |
| rgbd+colored GP | n2 | one_leg | 72 | 10.0 | 13.7 | N/A | N/A | 97.5% | 97.5% | 2.5% |
| rgbd+colored GP | n2 | round_table | 72 | 9.9 | 13.7 | N/A | N/A | 72.4% | 76.5% | 23.4% |
| rgbd+colored GP | n2 | lamp | 72 | 10.1 | 13.7 | N/A | N/A | 91.0% | 91.0% | 9.0% |
| rgbd+colored GP | n3 | one_leg | 72 | 20.0 | 27.6 | N/A | N/A | 91.7% | 95.5% | 4.5% |
| rgbd+colored GP | n3 | round_table | 72 | 19.9 | 27.6 | N/A | N/A | 73.5% | 79.9% | 20.1% |
| rgbd+colored GP | n3 | lamp | 72 | 20.2 | 27.8 | N/A | N/A | 88.4% | 90.0% | 10.0% |
| rgbd+colored GP | n4 | one_leg | 72 | 39.8 | 54.9 | N/A | N/A | 83.4% | 94.7% | 5.3% |
| rgbd+colored GP | n4 | round_table | 72 | 38.9 | 54.5 | N/A | N/A | 72.0% | 80.0% | 20.0% |
| rgbd+colored GP | n4 | lamp | 72 | 39.9 | 55.2 | N/A | N/A | 80.4% | 87.4% | 12.6% |
| rgbd+colored GP | n5 | one_leg | 108 | 79.7 | 109.6 | N/A | N/A | 75.8% | 97.4% | 2.6% |
| rgbd+colored GP | n5 | round_table | 108 | 80.2 | 109.8 | N/A | N/A | 66.9% | 79.0% | 20.9% |
| rgbd+colored GP | n5 | lamp | 108 | 81.0 | 110.5 | N/A | N/A | 76.1% | 88.5% | 11.4% |
| rgbd+colored GP | n6 | one_leg | 108 | 161.0 | 220.3 | N/A | N/A | 59.9% | 89.9% | 5.1% |
| rgbd+colored GP | n6 | round_table | 108 | 159.3 | 219.4 | N/A | N/A | 56.0% | 78.7% | 20.1% |
| rgbd+colored GP | n6 | lamp | 108 | 162.2 | 220.3 | N/A | N/A | 67.5% | 84.6% | 10.8% |
| rgbd+colored GP | n7 | one_leg | 108 | 321.0 | 439.1 | N/A | N/A | 39.9% | 65.1% | 3.4% |
| rgbd+colored GP | n7 | round_table | 108 | 317.8 | 440.7 | N/A | N/A | 36.4% | 56.2% | 19.9% |
| rgbd+colored GP | n7 | lamp | 108 | 324.5 | 439.1 | N/A | N/A | 39.7% | 56.9% | 11.3% |
| rgbd+colored GP | shuffle | one_leg | 72 | 208.1 | 251.8 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+colored GP | shuffle | round_table | 72 | 122.8 | 259.0 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+colored GP | shuffle | lamp | 72 | 130.2 | 191.4 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+GP+skill | n0 | one_leg | 72 | 0.0 | 0.0 | N/A | N/A | 97.4% | 97.4% | 2.6% |
| rgbd+GP+skill | n0 | round_table | 72 | 0.0 | 0.0 | N/A | N/A | 80.9% | 81.8% | 18.2% |
| rgbd+GP+skill | n0 | lamp | 72 | 0.0 | 0.0 | N/A | N/A | 86.9% | 86.9% | 13.1% |
| rgbd+GP+skill | n1 | one_leg | 72 | 5.0 | 6.9 | N/A | N/A | 96.5% | 96.5% | 3.5% |
| rgbd+GP+skill | n1 | round_table | 72 | 5.0 | 6.9 | N/A | N/A | 80.0% | 83.4% | 16.6% |
| rgbd+GP+skill | n1 | lamp | 72 | 5.1 | 6.9 | N/A | N/A | 85.8% | 85.8% | 14.2% |
| rgbd+GP+skill | n2 | one_leg | 72 | 10.1 | 13.8 | N/A | N/A | 96.9% | 96.9% | 3.1% |
| rgbd+GP+skill | n2 | round_table | 72 | 10.0 | 13.8 | N/A | N/A | 76.2% | 81.3% | 18.7% |
| rgbd+GP+skill | n2 | lamp | 72 | 10.1 | 13.9 | N/A | N/A | 88.2% | 88.2% | 11.8% |
| rgbd+GP+skill | n3 | one_leg | 72 | 20.0 | 27.6 | N/A | N/A | 93.2% | 97.4% | 2.6% |
| rgbd+GP+skill | n3 | round_table | 72 | 20.2 | 27.8 | N/A | N/A | 72.4% | 76.6% | 23.4% |
| rgbd+GP+skill | n3 | lamp | 72 | 20.3 | 27.9 | N/A | N/A | 88.7% | 90.2% | 9.8% |
| rgbd+GP+skill | n4 | one_leg | 72 | 40.1 | 55.2 | N/A | N/A | 85.3% | 97.2% | 2.8% |
| rgbd+GP+skill | n4 | round_table | 72 | 40.2 | 55.9 | N/A | N/A | 73.0% | 79.7% | 20.3% |
| rgbd+GP+skill | n4 | lamp | 72 | 40.5 | 55.8 | N/A | N/A | 84.6% | 92.3% | 7.7% |
| rgbd+GP+skill | n5 | one_leg | 108 | 80.7 | 110.1 | N/A | N/A | 74.8% | 96.1% | 3.7% |
| rgbd+GP+skill | n5 | round_table | 108 | 80.4 | 110.7 | N/A | N/A | 66.8% | 79.1% | 20.9% |
| rgbd+GP+skill | n5 | lamp | 108 | 81.2 | 109.9 | N/A | N/A | 76.6% | 88.9% | 11.1% |
| rgbd+GP+skill | n6 | one_leg | 108 | 160.0 | 219.4 | N/A | N/A | 60.2% | 92.5% | 2.8% |
| rgbd+GP+skill | n6 | round_table | 108 | 158.6 | 219.1 | N/A | N/A | 56.4% | 79.3% | 19.2% |
| rgbd+GP+skill | n6 | lamp | 108 | 162.0 | 220.6 | N/A | N/A | 70.6% | 88.1% | 6.6% |
| rgbd+GP+skill | n7 | one_leg | 108 | 321.3 | 439.1 | N/A | N/A | 41.0% | 65.3% | 1.9% |
| rgbd+GP+skill | n7 | round_table | 108 | 318.5 | 438.4 | N/A | N/A | 37.3% | 58.4% | 18.1% |
| rgbd+GP+skill | n7 | lamp | 108 | 328.2 | 444.9 | N/A | N/A | 40.8% | 56.0% | 11.5% |
| rgbd+GP+skill | shuffle | one_leg | 72 | 205.3 | 250.5 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+GP+skill | shuffle | round_table | 72 | 124.7 | 258.9 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+GP+skill | shuffle | lamp | 72 | 128.3 | 187.8 | N/A | N/A | 100.0% | 100.0% | 0.0% |
| rgbd+grasp-part | n0 | one_leg | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 95.9% | 95.9% | 4.1% |
| rgbd+grasp-part | n0 | round_table | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 81.3% | 81.8% | 18.2% |
| rgbd+grasp-part | n0 | lamp | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 90.2% | 90.2% | 9.8% |
| rgbd+grasp-part | n1 | one_leg | 72 | 5.0 | 6.9 | 4.2 | 5.8 | 95.6% | 95.6% | 4.4% |
| rgbd+grasp-part | n1 | round_table | 72 | 4.9 | 6.9 | 4.2 | 5.8 | 72.5% | 75.1% | 24.9% |
| rgbd+grasp-part | n1 | lamp | 72 | 5.0 | 6.9 | 4.2 | 5.7 | 89.5% | 89.5% | 10.5% |
| rgbd+grasp-part | n2 | one_leg | 72 | 10.0 | 13.8 | 8.5 | 11.6 | 95.9% | 95.9% | 4.1% |
| rgbd+grasp-part | n2 | round_table | 72 | 9.9 | 13.8 | 8.5 | 11.8 | 75.8% | 80.5% | 19.5% |
| rgbd+grasp-part | n2 | lamp | 72 | 10.1 | 13.9 | 8.4 | 11.5 | 92.2% | 92.2% | 7.8% |
| rgbd+grasp-part | n3 | one_leg | 72 | 20.1 | 27.6 | 16.9 | 23.0 | 92.3% | 96.4% | 3.6% |
| rgbd+grasp-part | n3 | round_table | 72 | 19.7 | 27.8 | 17.0 | 23.2 | 73.6% | 78.3% | 21.7% |
| rgbd+grasp-part | n3 | lamp | 72 | 20.0 | 27.7 | 17.0 | 23.3 | 88.0% | 89.6% | 10.4% |
| rgbd+grasp-part | n4 | one_leg | 72 | 40.2 | 55.2 | 34.0 | 46.0 | 85.1% | 96.7% | 3.3% |
| rgbd+grasp-part | n4 | round_table | 72 | 39.8 | 55.3 | 33.1 | 45.9 | 72.6% | 79.7% | 20.3% |
| rgbd+grasp-part | n4 | lamp | 72 | 40.3 | 55.6 | 33.7 | 46.1 | 83.4% | 91.0% | 9.0% |
| rgbd+grasp-part | n5 | one_leg | 108 | 80.4 | 110.0 | 67.2 | 92.5 | 73.5% | 95.0% | 5.0% |
| rgbd+grasp-part | n5 | round_table | 108 | 79.0 | 109.7 | 67.5 | 93.4 | 66.9% | 79.6% | 20.4% |
| rgbd+grasp-part | n5 | lamp | 108 | 80.6 | 110.2 | 67.2 | 92.8 | 73.9% | 86.9% | 13.1% |
| rgbd+grasp-part | n6 | one_leg | 108 | 160.4 | 219.3 | 100.9 | 139.5 | 59.4% | 93.1% | 3.0% |
| rgbd+grasp-part | n6 | round_table | 108 | 160.2 | 221.0 | 98.2 | 136.9 | 53.4% | 74.6% | 23.7% |
| rgbd+grasp-part | n6 | lamp | 108 | 161.9 | 220.4 | 100.6 | 139.5 | 68.2% | 85.4% | 10.3% |
| rgbd+grasp-part | n7 | one_leg | 108 | 320.2 | 439.0 | 133.3 | 172.6 | 40.8% | 65.6% | 3.6% |
| rgbd+grasp-part | n7 | round_table | 108 | 318.9 | 439.1 | 130.2 | 171.1 | 32.8% | 49.3% | 29.2% |
| rgbd+grasp-part | n7 | lamp | 108 | 317.5 | 439.1 | 131.5 | 171.6 | 44.7% | 59.9% | 10.6% |
| rgbd+grasp-part | shuffle | one_leg | 72 | 208.1 | 252.3 | 86.6 | 102.7 | 100.0% | 97.0% | 0.0% |
| rgbd+grasp-part | shuffle | round_table | 72 | 134.2 | 261.4 | 68.5 | 117.1 | 100.0% | 83.6% | 0.0% |
| rgbd+grasp-part | shuffle | lamp | 72 | 133.7 | 191.4 | 94.8 | 161.4 | 100.0% | 91.4% | 0.0% |
| rgbd+grasp-part | r180 | one_leg | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 96.0% | 96.0% | 4.0% |
| rgbd+grasp-part | r180 | round_table | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 78.3% | 79.0% | 21.0% |
| rgbd+grasp-part | r180 | lamp | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 90.4% | 90.4% | 9.6% |
| rgbd+grasp-part-colored | n0 | one_leg | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 95.9% | 95.9% | 4.1% |
| rgbd+grasp-part-colored | n0 | round_table | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 67.1% | 67.1% | 32.9% |
| rgbd+grasp-part-colored | n0 | lamp | 72 | 0.0 | 0.0 | 0.0 | 0.0 | 84.4% | 84.4% | 15.6% |
| rgbd+grasp-part-colored | n1 | one_leg | 72 | 5.0 | 6.9 | 4.3 | 5.8 | 96.5% | 96.5% | 3.5% |
| rgbd+grasp-part-colored | n1 | round_table | 72 | 5.0 | 7.0 | 4.2 | 5.8 | 76.3% | 80.5% | 19.5% |
| rgbd+grasp-part-colored | n1 | lamp | 72 | 5.0 | 7.0 | 4.3 | 5.8 | 89.8% | 89.8% | 10.2% |
| rgbd+grasp-part-colored | n2 | one_leg | 72 | 10.0 | 13.8 | 8.5 | 11.5 | 96.0% | 96.0% | 4.0% |
| rgbd+grasp-part-colored | n2 | round_table | 72 | 10.0 | 13.8 | 8.4 | 11.5 | 72.2% | 76.8% | 22.9% |
| rgbd+grasp-part-colored | n2 | lamp | 72 | 10.1 | 13.8 | 8.4 | 11.5 | 91.1% | 91.1% | 8.9% |
| rgbd+grasp-part-colored | n3 | one_leg | 72 | 20.0 | 27.4 | 16.9 | 23.1 | 88.6% | 92.8% | 7.2% |
| rgbd+grasp-part-colored | n3 | round_table | 72 | 19.9 | 27.4 | 16.9 | 23.1 | 75.6% | 81.2% | 18.8% |
| rgbd+grasp-part-colored | n3 | lamp | 72 | 20.1 | 27.8 | 17.2 | 23.3 | 86.8% | 88.4% | 11.6% |
| rgbd+grasp-part-colored | n4 | one_leg | 72 | 39.8 | 54.8 | 33.9 | 46.4 | 84.0% | 95.8% | 4.2% |
| rgbd+grasp-part-colored | n4 | round_table | 72 | 39.8 | 55.3 | 33.0 | 45.7 | 67.7% | 75.2% | 24.8% |
| rgbd+grasp-part-colored | n4 | lamp | 72 | 40.6 | 55.8 | 33.6 | 45.7 | 78.2% | 86.0% | 14.0% |
| rgbd+grasp-part-colored | n5 | one_leg | 108 | 80.5 | 109.8 | 67.4 | 93.0 | 75.0% | 97.1% | 2.9% |
| rgbd+grasp-part-colored | n5 | round_table | 108 | 79.7 | 111.7 | 67.2 | 92.3 | 58.1% | 68.0% | 31.9% |
| rgbd+grasp-part-colored | n5 | lamp | 108 | 81.2 | 110.2 | 66.8 | 93.2 | 71.5% | 85.0% | 15.0% |
| rgbd+grasp-part-colored | n6 | one_leg | 108 | 160.5 | 219.6 | 101.3 | 138.8 | 59.3% | 91.4% | 4.4% |
| rgbd+grasp-part-colored | n6 | round_table | 108 | 158.8 | 217.6 | 99.3 | 139.4 | 51.0% | 69.8% | 28.7% |
| rgbd+grasp-part-colored | n6 | lamp | 108 | 162.8 | 223.3 | 99.4 | 137.9 | 67.3% | 84.4% | 11.2% |
| rgbd+grasp-part-colored | n7 | one_leg | 108 | 319.2 | 435.8 | 132.1 | 171.7 | 38.9% | 64.1% | 5.5% |
| rgbd+grasp-part-colored | n7 | round_table | 108 | 318.0 | 439.0 | 130.0 | 170.9 | 33.4% | 51.6% | 26.8% |
| rgbd+grasp-part-colored | n7 | lamp | 108 | 326.2 | 442.1 | 131.2 | 171.9 | 39.2% | 56.4% | 11.8% |
| rgbd+grasp-part-colored | shuffle | one_leg | 72 | 207.8 | 256.3 | 87.1 | 103.1 | 100.0% | 94.2% | 0.0% |
| rgbd+grasp-part-colored | shuffle | round_table | 72 | 129.0 | 259.9 | 69.2 | 115.5 | 100.0% | 79.7% | 0.0% |
| rgbd+grasp-part-colored | shuffle | lamp | 72 | 132.4 | 192.2 | 90.9 | 155.8 | 100.0% | 89.1% | 0.0% |
| rgbd+grasp-part-colored | r180 | one_leg | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 95.3% | 95.3% | 4.7% |
| rgbd+grasp-part-colored | r180 | round_table | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 76.4% | 77.3% | 22.7% |
| rgbd+grasp-part-colored | r180 | lamp | 108 | 0.0 | 0.0 | 180.0 | 180.0 | 91.0% | 91.0% | 9.0% |

## 6. 数据产品

- `success_by_replicate.csv`: 三个 replicate 的 task success 与 Wilson CI。
- `success_overall_by_replicate.csv`: 三个 replicate 的 overall success 与 Wilson CI。
- `success_tracking_pooled.csv`: pooled success 与正式 tracking（含 72/108 样本量）。
- `three_task_pooled.csv`: 将三个 task 合并后的 success 与 position tracking summary；success 按 rollout 合并，tracking 按有效 skill-state 数加权。
- `vlm_sigma_by_task.csv`: 每个 VLM family×task 的合并条件、source trajectory 数、有效 control-step pair 数和 task-level Equivalent σ。
- `vlm_sigma_3task_pooled.csv`: Point/Grasp family 跨三个 task 的 position-equivalent σ；按有效点对数加权，供 pooled summary 图的粗实线使用。
- `vlm_orientation_tracking_equivalent.csv`: Grasp n0 clean-GT orientation tracking residual 映射到绑定 n0–n7 orientation schedule 的行为等效尺度；不冒充 raw VLM orientation residual。
- `noise_schedule.csv`: 固定的 n0–n7 position/orientation 绑定幅度，以及 r180 orientation-only endpoint；用于解释 position 横轴与 orientation 扰动的对应关系。
- `skill_progression_replicate_and_pooled.csv`: replicate 与 pooled skill-state success progression。
- `vlm_skill_error_reference.csv`: formal Grasp VLM 按 task×skill 汇总的 p50、RMS、p95 像素误差、p95-equivalent 以及 >40/>70/>100 px 尾部占比；skill-level 图读取此表。
- `vlm_skill_error_reference_pooled.csv`: 按 skill 跨 task/condition 的 valid-pair-weighted 汇总，供本章表格和文字使用。
- `skill_type_replicate_and_pooled.csv`: 5 类 skill 的 pooled success rate、tracking position/orientation/total 及对应样本量；skill-level 图读取此表。
- `annotation_noise_diagnostics.csv`: 实际扰动、workspace、projection 和 invalid 统计。
- `ordinal_trends.csv`: 仅 n0–n7 的预先固定 ordinal 描述性趋势。

- `table_validation.csv`: 对结果表的行唯一性、分母、成功率算术和 pooled/replicate 一致性校验；校验失败时不会生成图。
- `data_index.json`: 固定 `json -> tables -> figures` 数据索引；task-level 图读取对应的已校验 tables，三 task 合并图读取 `three_task_pooled.csv`。

- New manifest: `logs\annotation_noise_vlm_cover_108\manifest.jsonl`
- Read-only legacy manifest: `logs\annotation_noise_clean_train_fresh36_manifest.jsonl`

## 附录 A：VLM position-equivalent σ 与 orientation 对齐

下表给出图中 position-equivalent VLM σ 的完整 anchor table。`Valid control-step pairs` 是进入 VLM–GT 残差统计的有效点对数；这些 σ 只用于把真实 VLM 误差与 n0–n7 的数值噪声尺度对齐，不是额外的成功率观测。

| VLM family | Task | Combined conditions | Source trajectories | Valid control-step pairs | Equivalent position σ (mm/axis) |
| --- | --- | --- | --- | --- | --- |
| Point VLM | one_leg | rgbd+GP; rgbd+colored GP; rgbd+GP+skill | 108 | 45555 | 36.15 |
| Point VLM | round_table | rgbd+GP; rgbd+colored GP; rgbd+GP+skill | 108 | 93621 | 72.86 |
| Point VLM | lamp | rgbd+GP; rgbd+colored GP; rgbd+GP+skill | 108 | 73900 | 68.34 |
| Grasp VLM | one_leg | rgbd+grasp-part; rgbd+grasp-part-colored | 72 | 12168 | 170.09 |
| Grasp VLM | round_table | rgbd+grasp-part; rgbd+grasp-part-colored | 72 | 58626 | 100.28 |
| Grasp VLM | lamp | rgbd+grasp-part; rgbd+grasp-part-colored | 72 | 39492 | 96.26 |

三 task pooled 的 position-equivalent σ 按有效 control-step pair 数加权，供 pooled summary 图中的两条 position 参考线使用：

| VLM family | Task | Combined conditions | Source trajectories | Valid control-step pairs | Equivalent position σ (mm/axis) |
| --- | --- | --- | --- | --- | --- |
| Point VLM | 3 task pooled | rgbd+GP; rgbd+GP+skill; rgbd+colored GP | 324 | 213076 | 63.44 |
| Grasp VLM | 3 task pooled | rgbd+grasp-part; rgbd+grasp-part-colored | 216 | 110286 | 106.54 |

Grasp 的 orientation-equivalent scale 不在图中展示。它由 n0（未注入 orientation noise）的 clean-GT orientation tracking residual 与固定的 `0/2.5/5/10/20/40/60/90°` orientation schedule 线性匹配，再映射回绑定的 position schedule。该量表示下游策略的行为等效覆盖位置，而不是 raw VLM orientation σ。

| Source | Task | n0 tracking error (deg) | Matched orientation level | Behavioral orientation-equivalent σ (deg) | Position-axis coordinate (mm) | Tracking states |
| --- | --- | --- | --- | --- | --- | --- |
| Grasp VLM | one_leg | 14.74 | n3–n4 | 14.74 | 17.69 | 817 |
| Grasp VLM | round_table | 30.14 | n4–n5 | 30.14 | 36.17 | 996 |
| Grasp VLM | lamp | 44.45 | n5–n6 | 44.45 | 58.68 | 700 |
| Grasp VLM | 3task_pooled | 29.12 | n4–n5 | 29.12 | 34.95 | 2513 |
