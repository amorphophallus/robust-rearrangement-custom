# 2D Guidance Point 是否降低 Vision-to-Action Loss：假设、公式与实验方案

> 状态：低优先级机制实验设计稿，尚未执行。本文档中的“预期结果”不是已有结论；该实验不应阻塞正文主结果、VLM 端到端实验或真机实验。
>
> 目标：把“guidance point 让 ID/OOD 视觉特征更接近”的机制假设，改写为一个与正确动作直接相关、可被证伪、并能适用于不同 vision-action policy 的 loss 假设。

## 0. 决策摘要

不再把“ID 与 OOD latent distance 变小”作为主公式或主结论。无条件特征对齐可能把本应对应不同动作的状态压到一起；域适应理论也表明，边缘特征分布对齐并不足以保证目标域预测正确 [11]。新的主张是：

> **在相同状态—专家动作数据、相同策略容量和训练预算下，把正确的 2D guidance point 绘制到 RGB 图像中，是否降低了 held-out、特别是 OOD 状态上的 vision-to-action conditional risk？**

论文证据按优先级排列：

1. **闭环行为证据（正文必需）**：相同 reset 下的 rollout success、tracking error、failure stage，以及 correct/noise/shuffle/erase guidance 的反事实比较；
2. **端到端证据（正文必需）**：VLM pointing/type error 与完整 pipeline success 的关系；
3. **held-out action loss（可选）**：若能低成本复用已有 checkpoints 和 expert action 数据，再计算 energy loss 与当前 DiT 的 denoising MSE；
4. **representation diagnostic（低优先级）**：冻结视觉特征后的 action probe、action-kNN、attention map 和 action-conditioned latent geometry。

当前判断是：**没有必要为了正文专门完成 action-conditioned vision feature comparison**。成功率和反事实行为实验已经直接回答 guidance 是否被策略以对任务有用的方式使用；feature comparison 只在需要解释“rendering 为什么有效”时提供机制线索，不能替代闭环结果，也不能单独建立因果机制。因此该实验列为低优先级，默认放附录或不做。

正文优先使用下面的行为层结论：

> **Behavioral and counterfactual evaluations show that the improvement arises from a usable correspondence between the visual guidance and the executed action.**

只有在低优先级 feature/action-loss 实验与行为结果方向一致时，才进一步讨论下面的表征假设：

> **Rendering the guidance point into the observation helps the visual encoder organize visually different states according to their control-relevant target.**

第二句目前是待检验的机制解释，不是已有实验结论。无论是否开展该实验，都不要写成“guidance makes OOD features closer to ID features”。

## 1. 核心论证与边界

### 1.1 一句话论证

在多任务家具拼装中，colored guidance point 把当前阶段的语义类型和二维目标位置显式写入 action expert 已经处理的图像坐标系；若这种接口确实改善了控制相关表征，那么在相同专家状态—动作对上，它应降低正确动作 chunk 的条件预测 loss，并且该差异应在初始零件位姿 OOD 时更明显。

### 1.2 能 claim 什么

- 可以 claim：在本实验的家具拼装任务、数据规模和 policy family 下，colored GP 降低了 held-out/OOD vision-to-action risk，并与更高 rollout success 一致。
- 若 information-matched 的 low-dimensional point vector 明显差于 rendered point，才可以进一步 claim：**image-aligned rendering** 比仅提供相同点信息更有利。
- 若只有 rendered GP 与 RGB(D) baseline 的比较，则只能 claim：显式点条件有用，不能把收益唯一归因于“绘制到图像上”。
- 若只有 latent distance 或 attention map 改善，不能 claim 动作对应关系改善。
- 本实验不能单独证明对所有任务、所有视觉编码器或所有 policy architecture 的普适性。通用的是评估公式，不是预先保证的实验结论。

### 1.3 是否需要做 action-conditioned feature comparison

它与 ID/OOD success 不等价，但在本文现阶段也不是必要指标：

- success 是最终闭环结果，还受到接触动力学、误差恢复和 visited-state distribution 的影响；
- held-out action loss 只测 expert states 上的动作对应，不能保证长时序任务成功；
- frozen action probe 更弱，只测动作信息是否容易从 latent 中读出。

因此，若论文只 claim“colored GP 在所测家具拼装任务中具有更好的多任务表现和空间鲁棒性”，主实验与 counterfactual behavior 已足够。只有当正文要进一步 claim“rendered GP 改善了 action-relevant visual representation”时，才需要本实验，而且必须同时看到 action-conditioned metric 与行为结果一致。若时间有限，优先级为：

1. information-matched rendered colored GP vs. low-dimensional $(x,y,\mathrm{color})$ 的闭环成功率；
2. correct/noise/shuffle/erase GP 的 success 与 tracking error；
3. VLM error–success curve；
4. held-out action loss；
5. frozen action probe、action-kNN 与 latent visualization。

### 1.4 术语与记号

| 记号 | 定义 |
|---|---|
| $o_t$ | 时刻 $t$ 的原始视觉观测；实现中包含 front/wrist RGB 或 RGB-D |
| $s_t$ | 机器人 proprioceptive state |
| $g_t=(u_t,v_t,c_t)$ | 2D 点坐标与类别颜色；uncolored GP 可令 $c_t$ 为常量 |
| $R_c(o_t,g_t)$ | condition $c$ 对图像的确定性渲染；$R_0(o_t,\varnothing)=o_t$ |
| $\tilde o_t^{\,c}$ | 渲染后的图像观测 $R_c(o_t,g_t)$ |
| $A_t^*$ | 专家 action chunk，形状为 $H\times D_a$ |
| $\pi_{\theta,c}(A_t\mid \tilde o_t^{\,c},s_t)$ | condition $c$ 下训练得到的 vision-action policy |
| $q\in\{\mathrm{ID},\mathrm{OOD}\}$ | 评估状态分布 |
| $z_t^c$ | policy 内部的视觉特征；本项目中固定取两个 encoder projection 的输出 |

## 2. 主公式：Vision-to-Action Conditional Risk

### 2.1 通用形式

对 condition $c$，先在同一训练 demonstrations 上学习：

$$
\hat\theta_c
=
\arg\min_{\theta}
\mathbb E_{(o_t,s_t,A_t^*)\sim\mathcal D_{\mathrm{train}}}
\left[
\ell_{\mathrm{native}}
\left(
\pi_{\theta,c}(\cdot\mid \tilde o_t^{\,c},s_t),
A_t^*
\right)
\right].
$$

然后在与训练隔离的 ID 或 OOD expert state-action 数据上定义：

$$
\boxed{
\mathcal L_{\mathrm{V2A}}^{q}(c)
=
\mathbb E_{(o_t,s_t,A_t^*)\sim\mathcal D_q}
\left[
\ell_{\mathrm{act}}
\left(
\pi_{\hat\theta_c,c}(\cdot\mid \tilde o_t^{\,c},s_t),
A_t^*
\right)
\right]
}
$$

其中 $\ell_{\mathrm{act}}$ 衡量预测动作分布与正确 action chunk 的对应关系。实验真正检验的量是 paired difference：

$$
\boxed{
\Delta_{\mathrm{GP}}^{q}
=
\mathcal L_{\mathrm{V2A}}^{q}(\mathrm{colored\ GP})
-
\mathcal L_{\mathrm{V2A}}^{q}(\mathrm{RGB(D)})
}
$$

预注册方向性假设为 $\Delta_{\mathrm{GP}}^{\mathrm{OOD}}<0$。ID 上的差异可能较小，因为 clean/low-randomness evaluation 可能处于 ceiling；OOD 才是主要检验。

这个写法继承了 behavior cloning 把策略学习视为 observation-to-action supervised prediction 的标准形式 [1]，但把 loss 放在独立的 ID/OOD 状态分布上，并明确比较 guidance condition。

### 2.2 为什么不能直接统一用 NLL 或 MSE

- 对显式确定性 policy，常见 loss 是 action MSE/L1。
- 对概率密度可 tractably evaluate 的 stochastic policy，可以使用 $-\log\pi(A_t^*\mid o_t,s_t)$。
- implicit policy、diffusion policy 和显式回归 policy 的原生训练目标不同；Implicit Behavioral Cloning 的比较也说明 MSE、mixture density、energy-based objective 具有不同的建模偏置 [4]。
- Diffusion Policy 的 denoising MSE 是 likelihood/score-learning 的 surrogate，而不是可直接与普通 action MSE 横向比较的同尺度数值 [2,3]。

因此本实验同时保留一个**跨 policy 的 evaluation loss**和一个**同 policy family 内灵敏的 native loss**。

## 3. 跨 Policy 的统一 Loss：Action Energy Loss

### 3.1 定义

对同一输入从 policy 采样 $K$ 个 normalized action chunks：

$$
\hat A_t^{(1)},\ldots,\hat A_t^{(K)}
\sim
\pi_{\hat\theta_c,c}(\cdot\mid\tilde o_t^{\,c},s_t).
$$

使用 energy score 的 loss 约定（即 Gneiting & Raftery [5] 中 score 的负号；本文统一为越小越好）：

$$
\boxed{
\hat\ell_{\mathrm{ES}}
=
\frac{1}{K}\sum_{k=1}^{K}
\left\|W(\hat A_t^{(k)}-A_t^*)\right\|_2
-
\frac{1}{2K(K-1)}
\sum_{k\neq l}
\left\|W(\hat A_t^{(k)}-\hat A_t^{(l)})\right\|_2
}
$$

- 第一项要求预测样本接近正确 action chunk。
- 第二项避免通过无意义地收缩或扩张预测分布来获得虚假的好分数；energy score 是 multivariate predictive distribution 的 proper scoring rule [5]。
- $W$ 使用训练集 action standard deviation 做 whitening；position、rotation、gripper 仍需单独报告，避免单一分量支配总分。
- 对确定性 policy，令 $K=1$ 且第二项为 0，退化为 normalized action distance。此时它不再评价分布校准，但仍是可比较的动作误差。

### 3.2 推荐实现

- diffusion/flow policy：$K=16$ 个独立采样 seed；若计算预算不足，先用 $K=8$ 做 pilot。
- deterministic policy：单次前向；同时报告 normalized MSE 和分量误差。
- 同一个 state-action 样本的采样 seed bank 在所有 conditions 间复用。
- 统计单位是 trajectory，不是 frame；bootstrap 时对 trajectory 重采样。

### 3.3 限制

单个状态通常只有一条示范 action chunk，而真实正确动作可能多模态。Energy loss 在整个 held-out dataset 上仍是 proper sample-based evaluation，但不能把一条 demonstration 当成唯一物理可行动作。因此：

- 不使用 best-of-$K$ error 作为主指标；它会奖励过度发散的预测分布。
- 同时报告 rollout success，避免“离 expert action 更近”被误写成“任务一定成功”。
- 对明显多模态的阶段，按 task/phase 分层报告，并检查示范者或轨迹模式。

## 4. 当前 DiT 的 Policy-Native Loss

### 4.1 与本仓库代码的对应关系

当前主实验使用 `rgbd/dit`：actor 是 diffusion policy，backbone 是纯 self-attention DiT，训练目标是 epsilon prediction MSE。代码路径：

- `src/config/experiment/rgbd/dit.yaml`
- `src/config/actor/diffusion.yaml`
- `src/config/actor/diffusion_model/dit.yaml`
- `src/behavior/diffusion.py::DiffusionPolicy.compute_loss`
- `src/models/dit_policy_transformer.py::DitPolicyTransformer`

对 normalized expert action chunk $A_t^*$，采样 diffusion timestep $k$ 和 $\epsilon\sim\mathcal N(0,I)$：

$$
A_t^{k}
=
\sqrt{\bar\alpha_k}A_t^*
+
\sqrt{1-\bar\alpha_k}\epsilon.
$$

本项目实际优化：

$$
\boxed{
\mathcal L_{\mathrm{denoise}}^{q}(c)
=
\mathbb E_{\mathcal D_q,k,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_{\hat\theta_c}
\left(
A_t^{k},k,
h_c(\tilde o_t^{\,c},s_t)
\right)
\right\|_2^2
\right]
}
$$

这与 Diffusion Policy 的 observation-conditioned denoising objective 一致 [3]，其基础来自 DDPM 的 denoising/variational formulation [2]。

### 4.2 公平比较所需的改动

当前 validation loss 每次会重新采样 timestep 和 Gaussian noise；小差异可能被 Monte Carlo 波动淹没。实验评估时应建立固定 evaluation bank：

1. 为每个 sample 固定 $M=8$ 或 $16$ 组 `(diffusion_timestep, epsilon_seed)`；
2. 所有 conditions、training seeds 和 checkpoints 复用同一 bank；
3. 报告 aggregate、task、phase、position/rotation/gripper 对应的 loss；
4. checkpoint 只能依据 ID validation criterion 或预设 epoch 选择，不能看 OOD loss/success 后选择；
5. `test_bc_loss` 只用于 DiT-with-DiT 比较，不能与 MLP/IBC/FMT 的 native loss 直接排序。

## 5. 低优先级 Representation-Level 诊断：若做，必须与动作挂钩

本节不是正文主证据的前置条件。只有在主实验完成、时间允许，并且作者仍希望解释 image-aligned rendering 的表征机制时才执行。即使结果符合预期，也优先作为附录分析；只有结果稳定、跨 seed 一致，并能解释 C1 与 information-matched C2 的行为差异时，才考虑在正文放一个紧凑 panel。

### 5.1 当前项目应取哪个 latent state

主分析固定使用：

$$
z_t^c
=
\operatorname{concat}
\left(
f_{\mathrm{front,proj}}(\tilde o_{t,\mathrm{front}}^{\,c}),
f_{\mathrm{wrist,proj}}(\tilde o_{t,\mathrm{wrist}}^{\,c})
\right).
$$

对应 `src/behavior/base.py` 中 `encoder1_proj(self.encoder1(image1))` 和 `encoder2_proj(self.encoder2(image2))` 的输出，在与 robot state/skill 拼接之前。若配置启用 `feature_layernorm`，取 layer norm 之后的值。原因：

- 它确实是 vision encoder 交给 action expert 的视觉表示；
- 尚未混入 proprioception 或 one-hot skill，便于隔离“点如何改变视觉特征”；
- 不使用 DiT 中某个任意 block 的 token，避免把 diffusion timestep 和 noisy action sample 一并混入 state 表示。

DiT 中间层只用于补充可视化；若使用，必须固定相同 $k$、$\epsilon$ 和 action input。

### 5.2 Frozen action probe loss

动作依赖 proprioception，因此 probe 输入必须同时包含 $z_t^c$ 和 $s_t$：

$$
\boxed{
\mathcal L_{\mathrm{probe}}^{q}(c)
=
\min_{\omega}
\mathbb E_{\mathcal D_q}
\left[
\left\|
\rho_\omega(z_t^c,s_t)-A_t^*
\right\|_{W}^{2}
\right]
}
$$

实验上并不在 $\mathcal D_q$ 上重新拟合：probe 只在 ID-train split 训练，在 ID-test 和 OOD-test 直接评估。对每个 condition 使用相同的 linear probe 和 two-layer MLP probe、相同 initialization seeds、optimizer、step 数和 early-stopping rule。

解释：更低的 OOD probe loss 表明该 visual latent 保留了更容易被同容量 decoder 读出的动作信息。它比“两个域的均值距离更小”更接近本文机制，但仍是诊断，不替代完整 policy 和 rollout。

### 5.3 无参数 action-kNN loss

为减少 probe 优化带来的自由度，在 task、phase 和 proprioception 邻域内，从 ID feature bank 为每个 OOD 样本寻找最近邻：

$$
j^*(i;c)
=
\arg\min_{j\in\mathcal N_{\mathrm{ID}}(i)}
d_z(z_i^c,z_j^c),
$$

$$
\boxed{
\mathcal L_{\mathrm{kNN-A}}^{\mathrm{OOD}}(c)
=
\frac{1}{N}
\sum_{i\in\mathcal D_{\mathrm{OOD}}}
\ell_A(A_i^*,A_{j^*(i;c)}^*)
}
$$

其中 $\mathcal N_{\mathrm{ID}}(i)$ 限制为相同 task、相同 assembly phase，并对 proprioception 做半径或最近邻约束。若 colored GP 的 OOD feature 最近邻对应更相似的专家动作，才说明 latent geometry 在控制意义上更好。

### 5.4 raw latent gap 的正确地位

可额外计算 task/phase/action-matched MMD、Fréchet distance 或 cosine distance，但只作为补充：

- `raw gap ↓ + probe/kNN loss ↓ + rollout success ↑`：与 action-relevant alignment 一致；
- `raw gap ↓` 但 `probe/kNN loss` 不降：可能是 representation collapse 或错误对齐；
- `raw gap` 不降但 action loss 降：仍可支持本文主 claim，说明 guidance 改善了决策边界而非简单做全局域对齐。

Invariance Through Latent Alignment 在 nuisance perceptual shift 下展示了 latent alignment 的控制价值 [10]，但本实验的 initial-part pose shift 会改变正确动作，不能无条件套用其“全局对齐越近越好”的解释。Conditional domain adaptation 也强调，域对齐需要结合 discriminative prediction information [12]。在本文中，这个 discriminative information 就是 expert action chunk。

### 5.5 attention map 的正确地位

ATM Figure 11 展示的是 spatial CLS token 对 RGB tokens 的 attention map：点轨迹输入让注意力更多集中在 task-relevant region [6]。它不是 OOD/ID latent-distance 实验。本文可以复用该可视化思路：

- 固定同一帧，比较 RGB(D) 与 colored GP policy 的 vision encoder saliency/DiT conditioning sensitivity；
- 展示 attention 是否落到点及目标物体附近；
- 必须同时给 action loss 和 point erase/shuffle 的反事实结果，不能由 attention 图单独得出机制结论。

## 6. 实验假设

### H1：正确的 colored GP 降低 OOD vision-to-action risk（主假设）

$$
\Delta_{\mathrm{GP}}^{\mathrm{OOD}}<0
$$

Primary endpoint：colored GP 与 RGB-D 的 OOD action energy loss paired difference，按 trajectory bootstrap 的 95% CI 完全低于 0。

### H2：收益来自正确的点—动作对应，而不是多画了一个圆点

正确 GP 应优于：

- point-coordinate shuffle：保持颜色、大小、分布和 task/phase，不保持当前帧的正确空间目标；
- appearance-matched random point：保留相同视觉扰动统计；
- point erase：对 trained colored-GP policy 在 eval 时擦除点；
- color shuffle：保持坐标，打乱类别颜色，用于拆分空间和语义贡献。

若 shuffled/random point 与 correct point 表现相同，说明 policy 可能忽略 point，或收益来自普通视觉正则化，而不是 vision-to-action correspondence。

### H3：image-aligned rendering 是否本身有贡献

加入 information-matched control：不在 RGB 上画点，而把

$$
(u/W,\ v/H,\ \operatorname{onehot}(c))
$$

作为低维 condition 拼接到 action expert。两组包含相同的点位置和类别信息，仅接口位置不同。

- `rendered colored GP < low-dim point vector`：支持 image-aligned rendering 更易被视觉策略利用；
- 两者相当：只能说明显式点信息有用，不能 claim 绘制到 RGB 带来额外优势；
- low-dim vector 更好：需要放弃“图像对齐是关键机制”的叙述。

### H4：action-relevant representation diagnostic 与主结果一致

预期 colored GP 同时降低 OOD probe loss 和 action-kNN loss。raw domain gap 是否降低不做方向性硬假设。

### H5：offline action risk 与 closed-loop success 同向

预期 condition/task/seed 层面的更低 OOD action risk 对应更高 rollout success，但不把相关性写成因果证明。长时序接触失败、动作误差累积和 policy-induced state distribution 都可能让低一步 loss 不能完全转化为成功。

## 7. 实验矩阵

### 7.1 核心训练 conditions

| ID | 输入 condition | 目的 | 优先级 |
|---|---|---|---|
| C0 | RGB-D only | 无显式 task/spatial guidance baseline | 必做 |
| C1 | RGB-D + rendered colored GP | hero condition | 必做 |
| C2 | RGB-D + low-dimensional `(x,y,color)` vector | information-matched interface control | 必做，若要 claim rendering 优势 |
| C3 | RGB-D + rendered uncolored GP | 分离空间位置与类别颜色 | 建议 |
| C4 | RGB-D + one-hot skill | 离散 task/phase condition 对照 | 建议 |
| C5 | RGB-D + GP + skill | 检查离散 skill 是否降低大扰动鲁棒性 | 建议，可复用现有结果 |

所有训练 conditions 必须：

- 使用同一批 raw trajectories、相同 action/state/depth、相同 trajectory split 和 sample order；
- 只通过 deterministic renderer 或 condition adapter 改变输入；
- 相同 DiT、vision encoder、augmentation、batch size、epoch/step budget 和 checkpoint rule；
- 至少 3 个 training seeds；若差异接近噪声，增加到 5 seeds，而不是增加未经配对的单 checkpoint rollouts。

现有 `reports/claude/batch_train_guide.md` 已要求同源 condition 间 action/state/depth/task order 一致；本实验应把该检查作为硬 gate。

### 7.2 Counterfactual evaluation conditions

无需为每种扰动重新训练；对 C1 checkpoint 在完全相同 held-out frames 上渲染：

| Eval condition | 保持 | 改变 | 识别的问题 |
|---|---|---|---|
| correct GP | 原始点和颜色 | 无 | 正常性能 |
| point noise | 正确颜色 | 点坐标加 $\epsilon\sim\mathcal N(0,\sigma^2I)$ | VLM spatial error robustness |
| point shuffle | task/phase、颜色、点分布 | 点来自另一帧 | 是否依赖正确空间对应 |
| color shuffle | 点坐标 | 类别颜色 | 语义颜色贡献 |
| random dot | 圆点外观统计 | 目标语义与位置 | 普通视觉正则化解释 |
| erase | 原始 RGB-D | 移除点 | test-time reliance |

`point shuffle` 应优先在同 task、同 phase、同 color 内打乱，避免模型仅凭明显不合理的颜色或阶段检测异常。另做跨 phase shuffle 检验 semantic shortcut，但两者必须分开报告。

## 8. 数据协议

### 8.1 ID expert state-action set

- low-randomness demonstrations；按 trajectory 划分 train/validation/test，禁止按 frame 随机切分；
- train split 用于 policy；validation 只用于 checkpoint selection；test 只用于最终 ID loss；
- 对所有 conditions 从同一 raw observation 现场确定性渲染，避免不同 LMDB 的压缩、顺序或预处理差异。

### 8.2 OOD expert state-action set

要计算“正确动作 loss”，OOD 数据必须包含专家 action labels。仅有 policy rollout 的成功/失败视频不够。优先方案：

1. 在 medium/high initial-part pose randomization 下重新采集 expert demonstrations；
2. 任务、phase 定义、控制频率、action chunk 切法与 low-randomness 数据一致；
3. 记录 oracle point/type，使同一 OOD frame 能渲染所有 conditions；
4. 不把 OOD demonstrations 用于 policy 或 probe 训练。

若暂时没有 OOD expert actions，第一阶段只能完成 ID loss、counterfactual point perturbation 和 rollout success；不得把 latent gap 当作 OOD action correctness 的替代品。

### 8.3 可选 on-policy oracle relabel

held-out expert states 衡量 expert distribution 上的对应关系，不能覆盖 policy rollout 进入的错误状态。如果 simulator 中有可查询的 scripted expert，可对 policy visited states 重新标注 $A_t^*$，再计算：

$$
\mathcal L_{\mathrm{V2A,on-policy}}(c)
=
\mathbb E_{s\sim d_{\pi_c}}
\left[\ell_{\mathrm{act}}(\pi_c(\cdot\mid s),A^*(s))\right].
$$

没有可靠 oracle 时不要伪造该指标；使用 failure stage、target tracking 和 success 作为 closed-loop 证据。

## 9. 执行步骤

### Priority Gate：先完成主线实验

只有在以下工作不受影响时再启动本报告中的 action-loss/feature 实验：

1. clean multitask、point-noise、initial-part-pose OOD 的主结果已经完成；
2. information-matched C1 vs. C2 闭环对照与 guidance counterfactual 已完成或已有明确排期；
3. VLM 端到端与真机实验没有更高优先级的数据缺口。

### Phase A：可选的低成本 action-loss pilot

1. 复用现有 RGB-D 与 colored-GP checkpoints，各选固定 checkpoint，不根据本次结果重新挑选。
2. 在同一 ID validation/test batch 上实现 fixed `(timestep, epsilon)` bank。
3. 计算 `denoising loss + sampled action MSE + energy loss`。
4. 对 colored-GP checkpoint 做 correct/noise/shuffle/erase counterfactual eval。
5. 若 correct GP 与 shuffle/erase 无差异，先排查 point 是否真的进入图像与 encoder，不立即扩大实验。

### Phase B：低优先级 feature 机制实验

1. 从同一 raw LMDB 训练 C0/C1/C2，至少 3 seeds。每个 seed 在不同 conditions 间尽量复用模型初始化、dataloader 顺序与图像增强随机流，使比较保持配对。
2. 在 ID test 和独立 OOD expert set 上计算三层 loss。
3. 抽取 post-projection vision features，训练 fixed probe 并计算 action-kNN loss。
4. 在 paired reset seeds 上做 ID/OOD rollouts，报告 success、tracking error 和 failure stage。

### Phase C：仅在结果稳定时扩展与出图

1. 增加 C3/C4/C5，分析空间点、颜色语义和离散 skill 的作用。
2. 加 point-noise curve，把 VLM empirical error distribution 映射到 loss/success robustness envelope。
3. 只在 action loss/probe/success 三者方向一致时加入 attention/latent 可视化。

## 10. 统计方案

### 10.1 可选实验内部的 primary endpoint

$$
\Delta_{\mathrm{GP}}^{\mathrm{OOD}}
=
\mathcal L_{\mathrm{ES}}^{\mathrm{OOD}}(\mathrm{C1})
-
\mathcal L_{\mathrm{ES}}^{\mathrm{OOD}}(\mathrm{C0}).
$$

- 在相同 trajectory、time index 和 action label 上做 paired difference；
- 按 trajectory 分层 bootstrap，不能把 frame 当独立样本；
- aggregate 时按 task 等权，同时给 pooled-by-transition 结果作为补充；
- 报告 mean difference、95% bootstrap CI、每 task difference 和 seed dispersion；
- 主假设通过标准：aggregate 95% CI 完全低于 0，且没有某个任务出现稳定、幅度相当的反向退化。

### 10.2 Secondary endpoints

- OOD fixed-bank denoising loss；
- OOD probe loss 与 action-kNN loss；
- C1 vs C2 的 information-matched comparison；
- correct vs noise/shuffle/erase 的 counterfactual differences；
- rollout success、tracking error、failure stage。

多重 secondary comparisons 使用 Holm correction，或明确标成 exploratory。成功率使用 paired reset 时优先报告 paired bootstrap / permutation；未配对时报告 Wilson interval 并避免把小差异写成显著。

### 10.3 不建议作为主统计的量

- 不同 policy family 的 raw training loss；
- 从单个 checkpoint 训练曲线取最小值；
- 把数千 frame 当作数千独立样本；
- 只对 condition/task aggregate 的 10 个左右点做相关性并报告显著 $p$ 值；
- 未控制 task/phase/action 的全局 t-SNE、MMD 或 centroid distance。

## 11. 预期结果与解释矩阵

### 11.1 预期结果

在结果符合当前故事时，预期观察为：

1. C1 colored GP 的 ID loss 与 C0 相近或更低，OOD energy/denoising/probe loss 明显更低；
2. C1 的 point-noise loss 随 $\sigma$ 增大而上升，但在 VLM 常见误差区间内仍优于 C0/C5；
3. correct GP 明显优于 same-phase point shuffle、random dot 和 erase，说明收益依赖正确 target-action correspondence；
4. C1 优于 C2 时，支持把 guidance 绘制进视觉坐标系的机制；若 C1≈C2，则将结论收缩为“显式二维目标信息降低 action risk”；
5. C1 的 action probe 和 action-kNN loss 更低；raw ID/OOD feature gap 可能降低，也可能不变；
6. offline loss 的 condition 排序与 OOD success 大体一致，但 contact-rich failure 可能使差异不完全一一对应。

### 11.2 证伪与改写规则

| 观察 | 正确解释 | 论文叙述调整 |
|---|---|---|
| raw latent gap 降，但 action loss/probe/success 不变 | 可能发生无任务意义的域对齐或 collapse | 删除 latent-alignment 机制 claim |
| denoising loss 降，但 energy loss/success 不变 | 更好拟合 diffusion surrogate，未改善动作执行 | native loss 仅放诊断 |
| energy/probe loss 降，但 success 不升 | 一步动作更接近 expert，但长时序接触/恢复仍是瓶颈 | 强调 correspondence，不 claim 完整任务提升机制 |
| success 升，但 expert-state action loss 不降 | 收益可能来自 closed-loop recovery 或不同 visited-state distribution | 补 on-policy oracle relabel；不硬解释为一步映射 |
| correct GP 与 shuffle/random 相同 | point 被忽略或只是视觉 augmentation | 不能 claim target guidance |
| C1≈C2 | 点信息有用，rendering 优势未建立 | 删除 image-alignment superiority |
| C2<C1 | low-dimensional point 更易用 | 重新评估最终接口，不坚持 rendered point |
| C1 只在 ID 降 loss | 更好拟合，不是 OOD 泛化 | 不把该机制用于解释 OOD success |
| C1 在部分 task 改善、部分 task 退化 | task/phase dependent | 按 place/insert/screw 等阶段限定结论 |

## 12. 可选图表计划

### Optional Appendix Figure A：从 raw feature gap 改为 action-risk 证据链

- **a** paired raw frame：RGB-D、rendered colored GP、low-dim point vector；
- **b** ID/OOD action energy loss，按 condition 和 task；
- **c** fixed-bank DiT denoising loss 与 frozen action-probe loss；
- **d** OOD rollout success；用连线或同序排列展示 loss 与 success 是否一致。

唯一结论：如果结果成立，colored GP 的优势与更低的正确动作条件风险一致，而不是仅在视觉特征空间中做无条件域对齐。该图默认进入附录，不作为正文主图。

### Figure B：反事实 point correctness

- x 轴：correct、noise levels、same-phase shuffle、color shuffle、random dot、erase；
- 左 y 轴：action energy/native loss；
- 右 y 轴：rollout success 或 tracking error；
- 按 task 分 panel，使用相同 reset/noise seed bank。

唯一结论：正确 point-action correspondence，而非点的存在本身，驱动 action expert。

### Internal Table A：claim gate

| Condition | OOD energy loss | OOD native loss | OOD probe loss | action-kNN loss | OOD success |
|---|---:|---:|---:|---:|---:|
| RGB-D | 待实验 | 待实验 | 待实验 | 待实验 | 已有/补测 |
| rendered colored GP | 待实验 | 待实验 | 待实验 | 待实验 | 已有/补测 |
| low-dim `(x,y,color)` | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| same-phase shuffled GP | counterfactual | counterfactual | 可选 | 可选 | counterfactual |

## 13. 与相关工作的关系

- **Behavior cloning / compounding error**：Ross et al. 把 imitation learning 写成 learner-induced state distribution 上的 surrogate action loss，并说明 sequential prediction 的分布偏移会放大错误 [1]。本文不提出新的通用 IL bound，而是把 guidance 的作用定位为降低指定 ID/OOD 分布上的 action risk。
- **Diffusion Policy**：直接复用 observation-conditioned denoising loss 的形式 [2,3]，并在本仓库用 fixed noise/timestep bank 降低比较方差。
- **Implicit Behavioral Cloning**：不同 policy parameterization 的 native loss 不同 [4]，因此增加 sample-based energy loss，而不是比较不可同尺度的 training loss。
- **ATM**：2D point tracks 被作为细粒度 subgoal，ATM policy 用 MSE 训练；其 Figure 11 是 task-relevant attention 的定性证据 [6]。本文借鉴 structured visual guidance 和 attention 展示，但把 action risk 作为更强的机制检验。
- **KITE**：2D keypoint 与 parameterized skill 共同提供 semantic/spatial grounding [7]。本文的实验进一步检验有限离散 skill condition 是否形成 shortcut，以及 colored point 能否在共享 DiT action expert 中替代一部分离散 task cue。
- **RT-Trajectory**：把粗轨迹画进图像作为可编辑 visual prompt，证明视觉轨迹可以改变 policy behavior mode [8]。本文使用单点而非完整轨迹，并通过 shuffle/erase loss 明确检验点是否参与动作预测。
- **R3M**：视觉表示的价值最终通过 downstream policy learning 与成功率验证 [9]。本文同样不把 representation geometry 本身当作最终指标。
- **ILA 与 domain adaptation**：ILA 在 nuisance perceptual shifts 下通过 latent alignment 改善控制 [10]；Zhao et al. 说明仅有 invariant feature 和小 source error 仍不足以保证 target error [11]；CDAN 进一步强调对齐应结合任务预测信息 [12]。因此本文使用 action-conditioned probe/kNN，而不是无条件 ID/OOD centroid gap。

## 14. 参考文献

1. Ross, S., Gordon, G. & Bagnell, D. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. Proceedings of AISTATS, PMLR **15**, 627–635 (2011). https://proceedings.mlr.press/v15/ross11a.html
2. Ho, J., Jain, A. N. & Abbeel, P. *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems **33** (2020). https://papers.nips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
3. Chi, C. et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. Robotics: Science and Systems XIX (2023). https://doi.org/10.15607/RSS.2023.XIX.026
4. Florence, P. et al. *Implicit Behavioral Cloning*. Proceedings of CoRL, PMLR **164**, 158–168 (2022). https://proceedings.mlr.press/v164/florence22a.html
5. Gneiting, T. & Raftery, A. E. *Strictly Proper Scoring Rules, Prediction, and Estimation*. Journal of the American Statistical Association **102**, 359–378 (2007). https://doi.org/10.1198/016214506000001437
6. Wen, C. et al. *Any-point Trajectory Modeling for Policy Learning*. Robotics: Science and Systems XX (2024). https://doi.org/10.15607/RSS.2024.XX.092
7. Sundaresan, P., Belkhale, S., Sadigh, D. & Bohg, J. *KITE: Keypoint-Conditioned Policies for Semantic Manipulation*. arXiv:2306.16605 (2023). https://arxiv.org/abs/2306.16605
8. Gu, J. et al. *RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches*. arXiv:2311.01977 (2023). https://arxiv.org/abs/2311.01977
9. Nair, S., Rajeswaran, A., Kumar, V., Finn, C. & Gupta, A. *R3M: A Universal Visual Representation for Robot Manipulation*. Proceedings of CoRL, PMLR **205**, 892–909 (2023). https://proceedings.mlr.press/v205/nair23a.html
10. Yoneda, T., Yang, G., Walter, M. R. & Stadie, B. C. *Invariance Through Latent Alignment*. Robotics: Science and Systems XVIII (2022). https://doi.org/10.15607/RSS.2022.XVIII.064
11. Zhao, H., Tachet des Combes, R., Zhang, K. & Gordon, G. J. *On Learning Invariant Representations for Domain Adaptation*. Proceedings of ICML, PMLR **97**, 7523–7532 (2019). https://proceedings.mlr.press/v97/zhao19a.html
12. Long, M., Cao, Z., Wang, J. & Jordan, M. I. *Conditional Adversarial Domain Adaptation*. Advances in Neural Information Processing Systems **31** (2018). https://papers.nips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html

## 15. 开始实验前的硬检查清单

- [ ] C0/C1/C2 使用逐字节一致的 raw action/state/depth 与 trajectory split。
- [ ] rendered 与 low-dim point 的 `(x,y,color)` 信息完全相同。
- [ ] point renderer 在 crop/resize/augmentation 后仍与目标像素一致。
- [ ] OOD set 有真实 expert action labels；否则不计算 OOD V2A loss。
- [ ] fixed diffusion timestep/noise bank 已保存 hash，所有 conditions 复用。
- [ ] energy score 使用 normalized action chunk，position/rotation/gripper 分项可追溯。
- [ ] feature hook 固定在 `encoder1_proj/encoder2_proj` 输出，不因 condition 改层。
- [ ] probe 仅在 ID train 拟合，OOD 只评估。
- [ ] checkpoint 不根据 OOD 指标选择。
- [ ] bootstrap 单位是 trajectory；所有表记录 trajectories、frames、model seeds 和 action samples $K$。
- [ ] correct/shuffle/random/erase 使用相同 frames 和采样 seeds。
- [ ] attention/latent figure 只有在 action loss 与 rollout evidence 支持时才进入主文。
