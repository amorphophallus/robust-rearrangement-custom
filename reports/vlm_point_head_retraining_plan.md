# VLM guidance point 退化问题：loss 与 head 结构改造方案

日期：2026-08-17

> 状态（2026-08-19）：新版 `original_sft` 已移除独立 point head，并在 matched 300
> 样本诊断中基本消除旧式 regress-to-mean。本文保留为旧 structured-head 路线的设计
> 记录，不再是当前首选实现方案；新结果见
> `reports/vlm_original_sft_diagnostic_20260819.md`。

## 1. 结论先行

两个修改方向都成立，但作用层级不同：

1. **改 loss 是必要的训练修复，不是充分条件。** 当前 point loss 的相对权重、坐标尺度、样本不平衡和初始化都会削弱点回归对共享特征的影响。只把 `point_weight=0.02` 调大，或者把 SmoothL1 换成 MSE，不能解决 point head 缺乏空间信息、同一 coarse skill 内目标多峰的问题。
2. **让 skill/phase 条件化 point head 是更直接的结构修复。** 诊断数据表明，skill 信息本身对点位有很强解释力，特别是 lamp；但 5 个 coarse skill 仍然太粗，最终应使用任务内的 `skill_state/phase`，而不是只用 `pick/place/...`。
3. **推荐的近期候选是“phase/skill-conditioned point experts + 改进 loss/data sampler”。** 它改动较小、可与现有 API 兼容，也能用消融实验分别确认 loss 和 head 的贡献。
4. **推荐的长期结构是“语义 head 决定 what/phase，前视相机 visual token heatmap 决定 where”。** 当前从最后一个 language token 直接线性回归 `(x,y)`，很难保留精确空间结构；heatmap + DSNT/integral regression 更符合二维打点任务。

不建议第一版直接把 `argmax(skill_logits)` 拼给 point head。hard argmax 不可微；训练时喂 GT skill、推理时喂预测 skill 又会产生 teacher-forcing mismatch。应使用带 `stop-gradient` 的 soft probability/embedding，或者显式同时训练 oracle 和 deployment 两条路径。

## 2. 当前模型为什么会回归到均值

### 2.1 当前实现

ModelScope checkpoint 对应的本地代码位于 `/tmp/hy_furniture`。当前结构是：

```text
Qwen backbone
    └── 最后一个有效 token 的 hidden state h
          ├── Linear(h -> 5)     -> skill logits
          └── Linear(h -> 2)     -> sigmoid -> point (x, y)
```

loss 是：

```text
L = 1.0 * CrossEntropy(skill) + 0.02 * SmoothL1(point_x, point_y)
```

因此，两个 loss 不是完全独立训练；它们在两个 head 处分开，但都更新共享 backbone。

### 2.2 一个比 `0.02` 更重要的初始化问题

`skill_head.weight` 使用正态分布初始化，而 `point_head.weight` 被初始化为全零，point bias 对应图像中心。

对共享表示 `h`，首个训练 step 有：

```text
dL_point / dh = W_point^T * dL_point / dz = 0
```

也就是说，**训练初始阶段 point loss 完全不能推动 backbone 学习空间特征，skill loss 却可以立即塑造共享表示**。point head 在更新后才逐渐把梯度传回 backbone。当前 checkpoint 中 point head 的行范数约为 `0.0077–0.0096`，skill head 的行范数约为 `0.86–0.92`；绝对数值不能直接等同于有效梯度，但它足以说明必须记录真实梯度，而不能假定两个任务已被平衡。

### 2.3 数据和输出结构共同鼓励条件均值

- point loss 是 raw pixel 上的 SmoothL1，再乘固定的 `0.02`。权重的意义受分辨率、误差大小和 sigmoid 导数共同影响。
- 训练帧严重不均衡。长时间持续的 pick/screw 等阶段会远多于 insert 等短阶段；逐帧随机采样等价于让长阶段主导目标分布。
- 单个最后 token 的线性坐标 head 没有显式二维空间结构。当视觉表示不足以区分画面状态时，SmoothL1 的最优常数解接近条件中位数。
- `pick/place/...` 是 coarse skill。同一个 task 的同名 skill 可能对应不同零件或装配阶段。例如 `one_leg` 中有 tabletop pick 与 leg pick；`round_table` 中有 leg/base 两组 pick/place；lamp 也有 bulb/hood 阶段。因此即使 skill 完全正确，`task + coarse skill` 下仍可能是多峰坐标分布。

这解释了已经观察到的模式：task prompt 能把预测中心移动到该任务的中心附近；图像足以改变 skill；但 point 对同任务内的真实画面变化不敏感，因而停留在 task/skill 先验附近。

## 3. 本地数据对两个想法的收益估计

平衡诊断集包含 `3 task × 5 skill × 20 rollout-samples = 300` 条。下表比较：

- task-only：每个任务只输出一个 Huber 中心；
- oracle task×skill：使用 GT coarse skill 选择该 cell 的 Huber 中心；
- predicted-skill prototype：使用当前 VLM 的预测 skill 选择/混合 task×skill 中心；
- current VLM：当前真实点输出。

数值均为平均 pixel error，越低越好。

| task | task-only | oracle task×skill | predicted skill hard | predicted skill soft | current VLM |
|---|---:|---:|---:|---:|---:|
| one_leg | 17.18 | 9.26 | 11.90 | 11.62 | 14.24 |
| round_table | 20.60 | 13.96 | 16.14 | 16.26 | 19.82 |
| lamp | 32.74 | 5.94 | 11.99 | 13.77 | 28.89 |

这给出了 idea 2 的一个上限/下限判断：即使不从图像回归残差，只让当前 predicted skill 选择 prototype，平均误差也可能从：

- one_leg：`14.24 -> 11.6–11.9 px`；
- round_table：`19.82 -> 16.1–16.3 px`；
- lamp：`28.89 -> 12.0–13.8 px`。

所以 skill conditioning 很可能有效，lamp 的潜在收益尤其大。不过该 prototype 在同一批诊断样本上估计，数字偏乐观；更重要的是，它只能修复跨 skill 的中心差异，不能证明模型利用了图像。

当前 cross-skill image swap 诊断中，正确目标平均变化约 `26–47 px`，模型输出只变化约 `5–7 px`，沿正确目标变化方向的投影仅为 `5.7%–9.4%`。因此最终验收必须要求模型超过 task/phase prototype，而不能只看平均误差下降。

## 4. Idea 1：loss 应该怎么改

### 4.1 必做：修复初始化并记录真实梯度

point head 不再使用全零 weight。建议保留中心 bias，但将最后一层初始化为小的非零权重，例如 `Normal(0, 1e-3)`；具体 std 用一次短训练验证不能让初始输出分散过大。

训练日志至少增加：

- `||grad_skill||`、`||grad_point||`：两种 loss 在共享最后一层或 pooled hidden 上的梯度范数；
- 两个梯度的 cosine similarity；
- head weight/gradient norm；
- 每 task、每 phase 的 loss 和样本数；
- point prediction spread、bias、R²，以及 sigmoid 饱和比例。

先测量，再决定动态权重算法：

- 如果 point 梯度持续显著小于 skill，优先试 [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)，它按梯度大小和训练速度动态调权。
- 如果两个任务的共享梯度频繁出现显著负 cosine，再试 [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)。没有冲突证据时不应先增加它的复杂性。
- [Uncertainty weighting](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) 可以作为对照，但它可能把“较难、噪声较大”的 point 任务自动降权，不是当前退化问题的首选。

### 4.2 统一坐标尺度

建议把训练 target 统一到 `[0,1] × [0,1]`：

```text
y_norm = [x / (W-1), y / (H-1)]
```

point head 输出 normalized coordinate，loss 在 normalized 空间计算，同时继续在日志/API 中报告 pixel。SmoothL1 的 `beta` 可从 `0.02` 起步，相当于图像短边约 4.8 px 的二次区间；将 `beta ∈ {0.01, 0.02, 0.04}` 作为小规模搜索，而不是用 raw-pixel `beta=1` 再靠 `0.02` 抵消。

这不会自动产生 grounding，但能让 loss 权重跨分辨率更可解释，也降低 sigmoid 尺度与 magic coefficient 的耦合。

### 4.3 phase-balanced sampler 比单纯调权更重要

训练 batch 应按 `task × skill_state` 近似均衡，而不是按 frame 均匀采样。建议：

- 数据划分先按 rollout，再在 train 中采样，避免相邻帧泄漏；
- 每个 rollout/phase 限制最大帧数，或使用时间 stride；
- batch 中尽量包含同任务的不同 phase，供相对位移 loss 使用；
- validation/test 使用自然频率和 macro-balanced 两套指标。

现有 `TASK_PROGRESS_SCHEMA` 已定义稳定标签：

- one_leg：6 个 `skill_state`；
- round_table：9 个；
- lamp：7 个。

`SkillAnnotator` 和 rollout 内存结果已经产生 `skill_state`、`assembly_step`，但当前 `save_raw_rollout()` 没有把它们写进每帧 observation，`vlm_data_generator.py` 也没有写进训练 metadata/label。训练 v3 数据前应先贯通这两个字段。

### 4.4 增加直接反 collapse 的相对位移 loss

对同 task、不同 phase，或同 phase 中 GT point 距离足够大的成对样本 `(i,j)`，加入：

```text
L_delta = Huber((p_i - p_j) - (y_i - y_j))
```

若模型把两个画面都预测到相同中心，`p_i-p_j≈0`，这个 loss 会直接惩罚 collapse。它比只增大 absolute point loss 更针对当前问题。

第一版可使用：

```text
L_total = L_skill + lambda_point * L_point + lambda_delta * L_delta
```

`lambda_point` 不预设等于旧的 `0.02`；在 normalized coordinate 和非零初始化下，根据记录的共享梯度范数确定。`lambda_delta` 从使其共享梯度约占 point 分支 `10%–30%` 的范围起步。

### 4.5 不建议作为主方案的改法

- **只换 MSE**：异常点梯度更大，但不增加图像空间信息，也可能使优化更不稳。
- **只把 point weight 放大 10 倍**：可能增大 point head 的更新，却无法处理多峰 target；还可能破坏 skill accuracy。
- **只给各 skill 加 prototype**：会降低总体误差，但可能把“回归到 task 均值”变成“回归到 phase 均值”，仍不是视觉 grounding。

## 5. Idea 2：如何连接 skill head 和 point head

### 5.1 最小可行结构：soft skill embedding

```text
h = LayerNorm(last_token_hidden)
skill_logits = skill_head(h)
q = softmax(skill_logits / T)
e = stop_gradient(q) @ E_skill
point = point_mlp(concat(h, e, E_task))
```

其中 task 在 API 中已知，可以显式使用 task embedding，不必让模型再次从 prompt 猜 task。

首版对 `q` 使用 `stop_gradient`，理由是 point loss 不应该为了降低坐标误差而扭曲 skill 分类概率。后续可做一个不 detach 的消融，但必须观察 skill macro accuracy 和梯度冲突。

优点：改动小，可以直接检验“skill 条件是否减少跨 skill 平均化”。缺点：单个共享 MLP 仍可能忽略 embedding；coarse skill 仍有多峰。

### 5.2 更推荐的近期结构：条件 point experts

为每个 skill/phase 生成一个候选点：

```text
P_k = point_expert_k(h), k = 1...K
p_oracle = P[ground_truth_class]
p_soft = sum(stop_gradient(q_k) * P_k)
p_hard = P[argmax(q)]
```

同时训练：

```text
L_point = alpha * Huber(p_oracle, y)
        + (1-alpha) * Huber(p_soft, y)
```

训练早期 `alpha` 较大，让各 expert 学到对应分布；随后逐渐减小，确保 deployment path 能承受 skill 预测错误。这与 [scheduled sampling](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) 所处理的 train/inference mismatch 是同一类问题。

推理默认建议输出 `p_hard`，避免在两个不相容目标之间取平均；同时记录 `p_soft`、top-1 confidence 和两者距离。在 skill transition 且置信度低时，是否使用 soft 需要通过 rollout 决定，不能仅按 offline error 决定。

必须同时报告：

- oracle expert point error：衡量 point experts 自身；
- predicted expert point error：真实部署结果；
- oracle 与 predicted 的差值：衡量 skill/phase 分类错误造成的代价。

### 5.3 最终语义结构：预测 task-specific phase，而不是只有 5 个 skill

推荐在 v3 中加入 `phase_head`。已存在的 22 个 task-specific `skill_state` 可以作为 phase 类别；已知 task 用 mask 限制合法类别。

```text
phase_logits = phase_head(h)
q_phase = masked_softmax(phase_logits, valid_phases_for_task)
q_skill[k] = sum(q_phase[r] for phase r maps to skill k)
P_r = phase_conditioned_point_expert_r(h or spatial_tokens)
```

这样 skill 不再是与 point 平行、互不关联的另一个预测，而是 phase 的可解释聚合；client 仍可收到现有 5 类 skill，因此 API 可以向后兼容。

这一设计比“把 skill logits 拼到 point head”更强，原因是它区分：

- one_leg 的 `top-leg-pick` 与 `leg-top-pick`；
- round_table 的 `leg-top-*` 与 `base-leg-*`；
- lamp 的 `bulb-base-*` 与 `hood-base-*`。

如果真实条件分布在同一个 phase 内仍然多峰，单个 Huber 坐标仍会取中心。此时可输出 heatmap，或使用 mixture density；多值连续目标用显式混合分布建模的经典依据可参考 [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/)。

## 6. 推荐的最终空间 head

当前 `last token -> Linear(2)` 更偏语义摘要。准确二维定位应利用前视相机的二维 token grid：

```text
last-token semantic h ──> phase/skill logits ──> phase embedding/query
                                                │
front-camera visual token grid ── FiLM/cross-attention ──> 2-D heatmap
                                                             │
                                                      DSNT / integral
                                                             │
                                                           (x, y)
```

实现原则：

- 只用 front-camera token 形成二维输出，因为标注坐标属于 front camera；wrist image 和 state 可以进入 semantic query。
- 用 Gaussian target heatmap 的 CE/JS loss 加 DSNT/integral coordinate loss。
- 输出 heatmap entropy、peak probability、top-2 distance，作为置信度和 collapse 诊断。
- 若 visual tower 继续冻结，至少允许 Qwen 后段或一个轻量 spatial adapter 学习；否则冻结特征可能不足以恢复精确局部位置。

[DSNT](https://arxiv.org/abs/1801.07372) 指出纯 fully-connected 坐标回归缺乏空间泛化，而从归一化 heatmap 可微地取期望坐标能兼顾坐标监督与空间结构；[Integral Human Pose Regression](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.html) 给出了相近的 differentiable integral 思路。[CenterNet](https://arxiv.org/abs/1904.07850) 也将目标表示为 heatmap 上的点。

机器人领域里，[Transporter Networks](https://arxiv.org/abs/2010.14406) 和 [CLIPort](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf) 都强调保留空间结构，并把语义的 “what” 与空间的 “where” 分工。这与本问题中 skill/phase head 和 point head 的职责划分高度一致。用 skill embedding 调制空间特征可采用 [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671)；如果将来 expert 数量增加，可参考 [MMoE](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) 的共享 expert + task-specific gate，但当前规模无需一开始就引入完整 MMoE。

## 7. 分阶段实验与因果消融

所有实验必须使用相同 rollout-level split、相同训练样本预算、相同 seed 和相同 base backbone，禁止改变多项后只与旧模型比较。

| ID | loss/data | head | 目的 |
|---|---|---|---|
| E0 | 当前配置 | 两个 parallel linear heads | 可重复的基线 |
| E1 | 非零初始化、normalized loss、phase-balanced sampler、`L_delta`、梯度日志 | 保持 parallel heads | 单独验证 idea 1 |
| E2 | 当前 loss/data，补充必要日志 | soft skill embedding 或 5-skill experts | 单独验证 idea 2 |
| E3 | E1 | 5-skill experts，oracle+deployment loss | 近期可部署候选 |
| E4 | E1 | task-specific phase head + phase experts | 推荐语义候选 |
| E5 | E1 | phase-conditioned front-token heatmap + DSNT | 推荐最终候选 |

训练策略：

1. 先用 1 个固定 seed 筛 E0–E5，训练步数和有效样本数一致。
2. 选择 offline gate 通过且成本最低的前 2 个，用 3 seeds 重跑。
3. 只对 3-seed 结果稳定的模型做 VLM+DiT rollout smoke。
4. smoke 通过后再跑正式 3 condition × 3 task 矩阵。

如果资源有限，优先顺序是 `E0 -> E1 -> E2 -> E3 -> E4`；E5 可以在确认数据/监督本身有效后再投入。

## 8. 数据、代码和模型版本改造清单

### 8.1 Robust Rearrangement

1. `src/data_collection/io.py`
   - `save_raw_rollout()` 增加 `skill_states`、`assembly_steps`；
   - 每个 observation 保存对应字段；
   - 保持旧 pickle 读取兼容。
2. `src/eval/rollout.py`
   - 将已有 `rollout_data.skill_states/assembly_steps` 传给保存函数；
   - 对保存前后长度做断言。
3. `src/vlm_data_generator.py`
   - 输出 dataset schema v3；
   - metadata 增加 `skill_state`、`assembly_step`；
   - assistant label 可增加 `phase`，但线上 API 是否返回 phase 由 server schema 决定；
   - split manifest 保存 source rollout，确保无 frame leakage。
4. 数据审计脚本
   - task×phase frame/rollout 数量；
   - 每 cell point centroid、covariance、multimodality；
   - 重复帧和 target 静止段统计。

### 8.2 VLM 模型仓库/本地训练副本

1. model config `version: 2 -> 3`，记录 head 类型、phase schema、坐标规范和 expert inference mode。
2. 将 head 模块拆成 `semantic_head`、`point_head_v3`，支持 `parallel/skill_expert/phase_expert/heatmap` config，便于严格消融。
3. loss 返回所有原始项和加权项；gradient instrumentation 由 trainer callback 记录。
4. checkpoint 保存训练数据 manifest hash、split hash、phase schema hash 和 base model revision。
5. inference 保持原有 `skill`、`target_point_2d`，可选增加：

```json
{
  "phase": "leg-top-place",
  "skill_confidence": 0.91,
  "phase_confidence": 0.84,
  "point_confidence": 0.76,
  "policy_version": 3
}
```

自定义 structured head 继续走 Transformers/FastAPI server，不依赖 vLLM 对自定义输出 head 的支持；评测端维持严格 revision/version contract，不允许失败 fallback。

## 9. Offline 验收门槛

每 task、每 phase 都报告：

- skill/phase macro accuracy、confusion matrix；
- point mean、RMSE、median、p90、signed dx/dy bias；
- `R²`、prediction/GT spread ratio；
- oracle expert 与 predicted expert error；
- task-only、task×skill、task×phase prototype baseline；
- image/state/prompt intervention 的输出位移。

建议 gate：

1. skill macro accuracy 不低于 E0；phase macro accuracy先以 `>=80%` 为目标，并逐 task 检查。
2. 每个 task 的 point error 不仅低于 E0，还应显著低于 **held-out task×phase prototype**；否则只是把均值变细。
3. 每 task point `R² > 0.5` 作为第一版目标；prediction spread/GT spread ratio 在 `0.7–1.3` 附近，避免仍然过度收缩。
4. cross-phase image swap 的正确方向投影至少由当前 `5.7%–9.4%` 提升到 `>40%`，且绝对 point movement 随 GT movement 增加。
5. 按 phase 查看不能有高频 cell 掩盖低频 cell；任何关键 phase 系统偏差大于约 15 px 都需单独分析。
6. 同时报告 bootstrap 95% CI；只有 CI 和 3-seed 方向一致才进入 rollout。

门槛 3–5 是初始工程 gate，可在第一轮 E1/E2 分布出来后冻结为正式数值，但不能看完正式测试再调整。

## 10. Rollout 验收顺序

1. 每个候选先对 `one_leg/round_table/lamp` 各做离线真实序列回放，检查 phase transition 和 point continuity。
2. 每 task 3 rollout smoke，保留 VLM marker、phase/skill、confidence、GT point 的 debug MP4。
3. 异常 cell 做相同 launcher 的 scripted-GT 对照，区分 VLM 与 policy/environment。
4. smoke 中重点检查：
   - point 是否随 phase 和图像变化，而不是只在 phase transition 跳 prototype；
   - skill/phase 短暂误判是否让 point 跳到错误零件；
   - place/insert 临界阶段 hard expert 是否优于 soft mixture；
   - 新 point error 是否落入 DiT 已知可容忍噪声范围，并最终改善 success/tracking。
5. gate 通过后才进入正式 324 rollout；新模型结果与旧 invalid depth-missing 结果严格分开。

## 11. 推荐决策

### 近期实现

优先实现并比较 E1、E2、E3：

- E1 验证 loss/采样/初始化是否已足以恢复图像响应；
- E2 验证 head 条件化本身的收益；
- E3 是两者合并后的第一版可部署候选。

E3 中先使用 **5-skill experts + stop-gradient predicted probabilities + oracle/deployment 双 loss**，保持 API 不变。这样最快回答用户提出的两个因果问题。

### 随后升级

无论 E3 是否通过，都建议把数据 schema 升级为 phase-aware。若 E3 只达到 task×skill prototype、image swap 仍不敏感，直接推进 E4/E5，不再继续微调 coarse-skill loss 权重。

最终推荐 E5：**task-specific phase semantic head + phase-conditioned front-camera heatmap/DSNT spatial head**。它把“当前做什么”和“图像中的目标在哪里”建立显式但不过度耦合的关系，也从结构上减少直接坐标 head 回归到均值的倾向。

## 12. 风险与回退

- **skill/phase 错误被 point 放大**：保留 oracle/deployment 两组指标；推理记录置信度；不静默 fallback 到 scripted GT。
- **soft mixture 产生物理上无意义的中间点**：线上默认 hard expert，soft 只作训练辅助/消融。
- **phase label 来自自动机，存在边界抖动**：对 transition window 使用 soft/ignore label，或给相邻合法 phase 分配 label smoothing；不能把边界噪声当精确 GT。
- **prototype residual 再次掩盖 grounding**：若采用 phase center + residual，center 只能由 train split 估计，并强制要求超过 held-out phase prototype 和通过 image-swap gate。
- **heatmap head 读取错误视觉 token**：单元测试 front/wrist token mask 与 `image_grid_thw`，用合成单点图验证无 xy 交换、无重复缩放。
- **自适应 multi-task loss 降低 point 权重**：所有动态权重都记录原始 loss、有效梯度和最终权重，并保留固定权重对照。
