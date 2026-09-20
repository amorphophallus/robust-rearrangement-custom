# ICLR 2027 论文逐章逻辑大纲

> 整理日期：2026-09-19
> 固定依据：Overleaf `main.tex` 中的标题、摘要、章节标题和主文图表；`reports/iclr2026_experiment_results_story.md` 中的实验数据、统计口径与结论边界。
> 文档用途：先固定整篇论文的论证顺序，再逐章扩写和修改正文。除明确标注为“待补”的内容外，本大纲不引入新的实验结论。
> 写作定位：这是一篇以实验问题为中心的比较与分析型论文，不把贡献包装成通用的新架构。VLM--action expert 双系统是研究 conditioning interface 的应用背景和系统载体，而不是唯一的核心创新。

## 0. 固定内容

### 0.1 标题

**What Should a VLM Tell an Action Expert? Comparing Conditioning Interfaces for Long-Horizon Furniture Assembly**

标题对应全文的三个递进问题：

1. action expert 应该接收什么信息？
2. conditioning interface 对不准确的上游引导有多稳健？
3.这些结论能否延伸到更大的任务集合和真实系统？

### 0.2 摘要

> Long-horizon furniture assembly requires robots to switch across tasks and stages, localize the part and target, and execute contact-rich skills such as insertion and screwing; failure at any stage can invalidate the assembly. We use a vision-language model (VLM) for instruction understanding, task decomposition, and visual grounding, allowing a Diffusion Transformer (DiT) action expert to generate continuous actions from a compact conditioning interface rather than infer task stage and interaction target solely from high-dimensional observations. We ask which interface best improves downstream action generation while remaining robust to inevitable VLM prediction errors. Focusing on image-space targets, we compare 5 conditioning interfaces: a spatial-only guidance point (GP), a semantic-only low-dimensional skill condition, their explicit combination (GP+skill), a Target–Action Guidance Point (TAGPoint) that encodes low-dimensional information in point color, and a 6D grasp annotation that augments positional guidance with end-effector orientation. We evaluate these interfaces through FurnitureBench assembly, skill-level performance, controlled perturbations, end-to-end execution with real VLM predictions, and a larger set of single-step assembly tasks. Explicit 2D targets generally improve multitask performance, with the clearest gains in placement; skill conditioning alone provides limited benefits but becomes complementary when paired with a spatial target. We therefore propose TAGPoint, which encodes skill group in point color to combine spatial and semantic guidance in a compact, image-aligned representation. TAGPoint performs best among point-based interfaces under real VLM guidance, remains stable across target-point perturbation levels, and remains effective as task scale increases. Controlled perturbations make the role and failure boundaries of the intermediate interface measurable, providing guidance for representation design in VLM–action expert systems.

摘要已经锁定全文必须兑现的五项证据：

- FurnitureBench 完整长时程任务成功率；
- condition 对不同 skill 的影响；
- 受控 guidance perturbation 下的任务成功率与 tracking response；
- 使用真实 VLM prediction 的端到端评测；
- 在更大规模单步装配任务上的有效性。

### 0.3 一句话论文主线

在长时程家具拼装中，我们系统比较共享 DiT action expert 所接收的空间与语义条件，发现二维目标与低维 skill 信息具有互补作用，并以受控扰动、真实 VLM 引导和百任务级联合训练检验这些接口的收益、鲁棒性与任务规模边界；其中 TAGPoint 是本文推荐的图像对齐 point interface，真实系统中的有效性仍待真机评测完成后确认。

### 0.4 三级论证链

1. **信息内容：**先回答 action expert 需要空间目标、skill 语义还是二者的组合，并把完整任务收益定位到具体 manipulation skill。
2. **接口可靠性：**再回答上述接口面对空间误差时是否仍能完成任务、policy 是否在行为上实际使用了目标点，以及受控规律能否解释真实 VLM 引导下的系统表现。
3. **外部有效性：**最后回答结论能否在任务数量扩大和真实感知、深度与控制误差共同存在时保持。

这三层不能交换顺序。若不先证明 clean condition 有用，noise robustness 没有研究对象；若不先用受控扰动刻画边界，真实 VLM 结果缺少可解释的误差参照；若前三任务中的接口作用尚未明确，规模扩展和真机结果只能成为孤立 demo。

## 1. 术语锁定

| 规范术语 | 固定含义 | 不建议混用的说法 |
|---|---|---|
| conditioning interface | VLM 或 oracle 向 action expert 提供信息的接口形式 | prompt type、guidance type、condition type 随意切换 |
| DiT action expert | 接收观测、robot state 与 conditioning interface 并生成连续 action chunk 的下游策略 | VLA、low-level controller（除非上下文确实指控制器） |
| skill | 跨任务共享的五类操作能力之一：Pick、Place、Push、Insert 或 Screw | stage、subtask |
| stage | 某一装配任务中与具体对象和顺序绑定的操作实例，例如 `pick leg`；每个 stage 实例化一个 skill | 用 stage 指代五类 skill |
| guidance point (GP) | 只编码二维目标位置的图像空间点 | keypoint（引用相关工作时除外） |
| skill condition | 表示 skill identity 的 $N$ 维 one-hot 低维条件；$N$ 为 skill 数量，不包含具体 stage、对象或场景相关目标位置 | stage ID、task ID、language condition |
| GP+skill | GP 与独立的 $N$ 维 skill condition 的显式组合；作为 information-rich reference | theoretical upper bound、oracle upper bound |
| Target--Action Guidance Point (TAGPoint) | 用二维点位置编码 target，并用点颜色编码粗粒度 target-site gripper action mode | colored GP（只在历史图标签或说明旧名称时使用） |
| 6D grasp annotation | 在位置引导上补充末端旋转信息的 richer interface | grasp-part；正文统一删除 part |
| complete-task success | 完整家具装配成功率 | accuracy |
| conditional skill completion | 对所有属于同一 skill 的 task-specific stages，汇总已进入次数 $R$ 与完成次数 $C$ 后得到的 $C/R$ | 独立 skill success（会掩盖 stage-entry-state 依赖） |
| controlled spatial perturbation | 测试时对 guidance 施加的已知空间扰动 | VLM noise（两者分布并不相同） |
| VLM-generated guidance | 真实 VLM 预测并实际输入 action expert 的 condition | GT guidance、oracle guidance |

术语边界：TAGPoint 当前只验证两种颜色编码，对应两组粗粒度夹爪动作模式；不能把 RGB 的名义颜色空间写成已经验证的语义容量。GP+skill 提供完整固定 skill identity，但其 vocabulary-bound 属性是扩展性限制，不是已经由实验确认的鲁棒性缺陷。

## 2. Introduction

### 本章回答的问题

为什么长时程家具拼装需要一个显式的 VLM--action expert 中间接口，以及为什么“这个接口应该包含什么信息”本身是值得系统研究的问题？

### 推荐段落顺序（五段）

#### 第 1 段：从家具拼装难点建立任务需求

- 长时程家具拼装要求共享策略同时完成跨任务/阶段切换、交互目标定位和接触密集动作生成；任一阶段失败都可能改变后续状态并使完整装配失败，而有限示范难以覆盖这些偏移状态。
- 将矛盾收束为 action expert 必须回答的两个问题：当前“做什么”，以及“在哪里做”。第一段不介绍 TAGPoint，也不把结论泛化到通用机器人操作。

#### 第 2 段：引出双系统、二维接口与周期重锚

- VLM 负责开放式指令理解、任务分解和视觉 grounding，DiT action expert 负责连续、接触密集的动作生成；二维 image-space target 直接位于 VLM 与 action expert 共同使用的图像坐标系中，使上游无需输出完整机器人轨迹或精确 3D pose。
- 把 **periodically re-anchored, non-recursively accumulated guidance error** 明确写成双系统优势之一：action error 会经由状态转移影响后续 observation 并沿闭环传播，而每次 guidance refresh 都从当前 observation 重新预测 target，因此定位残差不会在接口变量中由上一时刻的 target 或 action prediction 递归相加。表述边界只需紧随一句：重新锚定不能撤销已经发生的接触失败或物体位姿偏移，也不意味着整个闭环不存在累积误差。
- **Figure 1a/c** 在此出现：panel a 对照“action error 经状态转移传播”与“guidance 根据当前图像周期重锚”，panel c 展示 VLM refresh 与连续 DiT control 的分工；图用于解释双系统的作用，而不是宣称新的系统架构。

#### 第 3 段：接口选择原则与 condition taxonomy

- 一旦语义推理与动作生成分属两个系统，中间接口既应传递上游已完成的决策，也不应迫使下游重新解决开放式理解或 grounding。本文因此不把具有空间歧义的原始语言作为主要条件，也不要求 image-based VLM 输出依赖 metric depth 与 calibration 的 point cloud 或 3D target；VLM 只输出 image-aligned 结果，由同时观察 RGB-D 与 robot state 的 action expert 补足 metric geometry。这是本文的设计取舍而非“2D 普遍优于 3D”的主张，RGB-only × RGB-D 比较用于测量 point condition 对显式 depth 的依赖。
- 在这一范围内，真正的未决问题是二维 target 还应携带多少语义或姿态信息：skill condition 仅回答“做什么”；GP 仅回答“在哪里做”；GP+skill 是包含完整固定 skill identity 的 information-rich reference；TAGPoint 用点颜色编码粗粒度夹爪动作模式，将空间和语义压缩为 image-aligned 表示；6D grasp annotation 则检验补充 end-effector orientation 是否值得更高的上游输出复杂度与误差成本。
- **Figure 1b** 把视觉中心留给本文实际比较的五种 conditioning interfaces；图解释设计空间的收束

#### 第 4 段：说明 clean success 不足以回答接口选择

- 真实 VLM 会产生语义错误、空间偏移和长尾误差；只用无噪声 oracle condition 会高估部署能力。
- 因此评价接口需要同时回答三件事：clean 条件下是否改善任务；误差增大时是否仍然可用；真实 VLM 输出是否能够驱动端到端执行。
- 提前概括但不展开数字：二维目标总体改善多任务表现，收益最集中于空间精度要求更高的 Place skill；skill condition 单独收益有限，但与空间 target 结合后具有互补作用；TAGPoint 在真实 VLM 下取得 point-based interfaces 中最高的观测表现。

#### 第 5 段：用三个研究问题预告全文结构

1. **What information should condition the action expert?**对应第 3 章的完整任务和 skill-level 分析。
2. **How robust are conditioning interfaces to imperfect guidance?**对应第 4 章的受控扰动、tracking 与真实 VLM。
3. **Can the findings extend to larger task sets and physical systems?**对应第 5 章的 AutoMate scale-up 和 real-world evaluation。

#### Contributions：三项贡献的分工

1. **Controlled comparison：**在同一个共享 DiT action expert 下比较空间、低维语义及其组合，并从 complete-task 与 skill level 两个尺度评估作用；同时测量受控 point perturbation 下的 success 和 tracking。
2. **Interface recommendation：**在本文比较的 point-based interfaces 中，将 TAGPoint 识别为推荐的二维中间表示；依据来自 clean、真实 VLM、noise window 和 task scale 的联合证据，而不是单一表格中的显著领先。
3. **System validation and scaling：**实现 VLM--TAGPoint--DiT 完整管线，把 VLM guidance error 与下游结果连接起来，并进一步在 AutoMate 联合训练和真实系统中测试外部有效性。

### 本章结尾必须守住的边界

- 不把 TAGPoint 写成所有 manipulation task 上通用最优的表示。
- 不把 Figure 1 写成架构创新；它解释研究对象和接口设计空间。
- 不声称二维点消除了整个闭环的长时程误差累积；准确说法是，它提供一个周期性重新锚定、其残差不在接口变量中递归相加的外部 target reference。
- 不直接声称 skill condition 导致 shortcut；扩大的 noisy eval 不支持这一结论。

## 3. Problem Setting and Controlled Study Design

### 本章角色

本章以紧凑的 preliminary 形式定义多任务家具拼装、conditioning interface 和统一的 DiT 训练目标，然后立即进入结果。主文目标为约 0.75--1 页、三个 subsection；模型 parameterization、训练超参数和数据细节统一放入附录，不在正文展开。

### 3.1 Multitask Furniture Assembly

#### 正文内容

- 本文研究 FurnitureBench 中的 one-leg、round-table 和 lamp 三个长时程装配任务。每个任务由顺序依赖的 stages 构成，每个 stage 对应 Pick、Place、Push、Insert 或 Screw 中的一个 skill；例如 `pick leg` 是一个具体 stage，其 skill 为 Pick。任一关键 stage 失败都可能使完整装配失败。
- 三个任务的 demonstrations 被共同用于训练一个共享 action expert。该策略在每个控制时刻接收 front/wrist observation、robot state 和可选 conditioning interface，并输出连续的 end-effector 与 gripper actions。传感器分辨率、action parameterization 和执行 horizon 不在本节展开，统一放入附录。


### 3.2 Conditioning Interfaces

#### 正文内容

1. **先定义上游接口的输入输出，不在本节描述 refresh schedule。**令

   $$
   o_t=(I_t^{\mathrm f},D_t^{\mathrm f},I_t^{\mathrm w},D_t^{\mathrm w},s_t),
   \qquad z_t^{(q)}=h_{\phi}^{(q)}(o_t,\ell_t),
   $$

   其中 $o_t$ 是 front/wrist RGB-D 与 robot state，$\ell_t$ 是与具体对象和任务进度绑定的 stage instruction（例如 `pick leg`），$q$ 表示 interface variant，$z_t^{(q)}$ 是上游输出。该式只回答“上游接收什么、输出什么”；VLM 的更新频率放到真实 VLM 实验的 protocol 中，oracle study 则以 scripted annotation 实例化同一个 $z_t^{(q)}$。

2. **再定义接口可能携带的三类信息。**令 $c_t\in\{0,1\}^{N}$ 表示 skill identity，$N$ 为 skill 数量；$p_t=(u_t,v_t)$ 表示 image-space target，$r_t\in\mathbb R^6$ 表示 target end-effector orientation。RGB/RGB-D 是无显式 condition 的 observation baselines，不计入五种 interface。具体取值 $N=5$ 只在附录交代。

   **Table 1（condition taxonomy）**在本节按普通正文表格编号；表注只需标题 *Information supplied by the five conditioning interfaces*。Skill semantics 一列保留 `Skill label`、`Coarse mode` 与 `—` 的区别，只有 2D target 和 EE orientation 两列使用 `✓`／`—`，不再添加符号或颜色分组的解释段。主实验表顺延为 Table 2。

| Interface | 形式 | Skill semantics | Image-space target location | Target end-effector orientation |
|---|---|---|---|---|
| skill condition | $c_t$ | Skill label | — | — |
| GP | $p_t$ | — | ✓ | — |
| TAGPoint | $(p_t,m_t)$ | Coarse mode | ✓ | — |
| GP+skill | $(p_t,c_t)$ | Skill label | ✓ | — |
| 6D grasp annotation | $(p_t,r_t)$ | —; colored variant: coarse two-group | ✓ | ✓ |

3. **在 TAGPoint 出现后再定义颜色变量。**TAGPoint 不额外预测一个独立语义标签，而是将完整 skill condition 压缩为

   $$
   m_t=g(c_t)\in\{0,1\},
   $$

   其中 $g$ 将 Pick/Screw 与 Place/Push/Insert 映射为两类 coarse gripper-action mode，并通过 point color 表示 $m_t$。skill condition 不含场景相关位置；GP 不显式给出 skill semantics；GP+skill 是 information-rich reference；6D grasp annotation 检验 orientation 是否值得更高的上游预测负担。GP 的位置分布可能间接携带 task 或 stage 信息，因此 “what/where” 并非严格正交。

4. **最后只解释 condition 如何进入 observation，不在这里展开 encoder 和 policy。**定义

   $$
   \widetilde I_t^{v,q}=\mathcal R_q(I_t^v,z_t^{(q)}),
   \qquad \eta_t^{(q)}=\psi_q(z_t^{(q)}).
   $$

   这里有两条独立注入路径：$\mathcal R_q$ 将 GP、TAGPoint 或 grasp marker 渲染到第 $v$ 个 RGB view；$\psi_q$ 提取需要独立拼接的低维变量，因此 skill-only 时 $\eta_t^{(q)}=c_t$，GP+skill 时同样保留 $c_t$，其余视觉接口的 $\eta_t^{(q)}$ 为空。这样，3.2 只定义 interface，视觉编码与 action generation 留到 3.3。

### 3.3 Shared DiT Action Expert

#### 逻辑任务

- front/wrist observations 分别由 ResNet-18 encoder 编码并投影。对于 RGB-D 输入，将 ResNet-18 首个卷积层由 3 个输入通道扩展为 4 个，保留预训练 RGB 权重，并将新增 depth 通道的卷积权重初始化为零。
- 在解释完 encoder 后再给 action expert 的输入输出：

  $$
  x_t^{(q)}=\big[E_{\mathrm f}(\widetilde I_t^{\mathrm f,q},D_t^{\mathrm f}),
  E_{\mathrm w}(\widetilde I_t^{\mathrm w,q},D_t^{\mathrm w}),s_t,\eta_t^{(q)}\big],
  \qquad \widehat{\mathbf a}_t\sim\pi_\theta(\cdot\mid x_t^{(q)}).
  $$

  $E_{\mathrm f}$ 与 $E_{\mathrm w}$ 分别编码 front/wrist view，$s_t$ 是 robot state，$\eta_t^{(q)}$ 是 3.2 定义的可选低维 condition；$x_t^{(q)}$ 是融合后的 observation representation，$\widehat{\mathbf a}_t$ 是预测的连续 action sequence。RGB-only 设置中省略 $D_t^{v}$。
- DiT 对 action tokens 进行纯 self-attention；diffusion timestep 与 $x_t^{(q)}$ 共同通过 adaptive LayerNorm 调制每个 block。backbone depth、hidden size 和 attention heads 等具体数值放入附录的 parameterization。
- 训练使用 DDPM forward noising 与 epsilon-prediction MSE；推理使用 DDIM。正文只给 forward process 与一个 loss，不复述完整 diffusion 教程，写法参考 *Much Ado About Noising* 先定义对象、再给训练目标。
- 明确 controlled-comparison contract：所有 variants 使用相同 demonstrations、action expert capacity、horizons 和 optimization；差别仅在 $\mathcal R_q$ 与 $\psi_q$ 所传递的 condition。评测 rollout、seed 和各实验特殊 protocol 在相应 Results subsection 首次出现时说明。

#### 正文公式

1. **Forward diffusion on action sequences**

   $$
   \epsilon\sim\mathcal N(0,I),\quad k\sim\mathrm{Unif}\{1,\ldots,K\},\quad
   \mathbf a_t^{(k)}=\sqrt{\bar\alpha_k}\,\mathbf a_t+
   \sqrt{1-\bar\alpha_k}\,\epsilon.
   $$

2. **Conditioned epsilon-prediction objective**

   $$
   \mathcal L_{\mathrm{DiT}}(\theta)=
   \mathbb E_{(x_t^{(q)},\mathbf a_t)\sim\mathcal D,\,k,\epsilon}
   \left[\left\|\epsilon-\epsilon_\theta
   \big(\mathbf a_t^{(k)},k,x_t^{(q)}\big)\right\|_2^2\right].
   $$

   训练 diffusion steps、DDIM inference steps、action-sequence horizon、执行 horizon 以及 DiT parameterization 均放入附录，不在本节列出。

### 与第 3 章的过渡句

在固定 demonstrations、DiT action expert 和训练目标后，首先比较不同接口在 clean guidance 下向共享策略提供了多少有效信息。

## 4. What Information Should Condition the Action Expert?

> 本章对应论文第 3 章。仿照 *Much Ado About Noising* 的结果组织方式：章首先交代结论及两个互相支撑的证据层次，接着在各小节中按“问题—协议—结果—解释—边界”推进。不要把小节变成只有一句结论的标题，也不必机械设置 Key Insight 列表。

### 章首：先交代答案，再给读图路线

**核心判断。** 在 clean scripted guidance 下，二维目标与低维 skill 语义向共享 action expert 提供互补信息：显式组合 GP+skill 的完整任务成功率最高；将粗粒度动作提示附着于点的 TAGPoint 优于单独 GP 的总体点估计，但多提供目标姿态的 grasp annotation 并不稳定。收益也不是均匀分布在每种操作中，而是在要求更高空间精度的 Place skill 上最清楚。

**章首正文写法（英文草稿，进入正文时再精修）：** “Under accurate scripted guidance, spatial targets and skill semantics provide complementary cues to the shared action expert, but more detailed geometry does not consistently improve assembly. We first compare complete-task success across the observation and interface variants (Sec. 3.1; Table 2). We then examine which operation skills account for the differences (Sec. 3.2; Fig. 2), before testing whether these clean-guidance conclusions survive upstream errors in the following section.”

章首不重复介绍 DiT、rendering 或全部实验参数；也不直接宣布 TAGPoint 是 clean 表中的最优行（此处 GP+skill 最高）。完整任务指标与 conditional skill completion 的分母不同，不能从后者反推前者。

### 4.1 Spatial Targets and Skill Semantics Play Complementary Roles in Multitask Assembly

#### A. 引入比较及读表协议（1 个紧凑段落）

- 用一句承接章首：“To identify what the action expert gains from each interface, we compare complete assemblies under clean scripted guidance.” 同一个 DiT 分别在 one-leg、round-table、lamp demonstrations 的并集上训练；除 condition 外，demonstrations、backbone/capacity、action horizon 和 optimization 固定。RGB 与 RGB-D 行没有显式 condition，用于隔离“只增加 depth”与“提供 task-relevant condition”。
- 每个 checkpoint 对每个任务评测 36 条完整 rollout（overall 为 108）。RGB/RGB-D 只用两个有效 supplemental training runs，其余各用三个；原始两条 baseline checkpoint 的错误 lineage 已剔除。先逐 checkpoint 计算 task/overall success，再跨独立训练 run 给 mean $\pm$ sample SD。即使 Table 2 删除 `$n$` 列，也要在 caption 保留这个不等重复数。不同策略评测不是 matched-reset paired comparison。
- **Table 2 前导句（英文草稿）：** “Table 2 reports complete-assembly success for observation-only baselines and six condition variants, using scripted guidance wherever an interface is required. This comparison isolates which information a shared action expert can exploit before introducing upstream prediction error.” 此处“五种 interface family”可对应六行 condition：6D grasp 另有 colored variant；全文不要写成八种全是 condition。

#### B. 先报告完整任务格局，再拆解信息增量（2 个结果段落）

1. **第一段直接读出主结果，不先做机制推测。** Overall：RGB `50.00±1.31%`、RGB-D `51.39±0.65%`、skill `53.70±1.60%`、GP `52.47±4.66%`、TAGPoint `57.41±5.78%`、GP+skill `63.27±4.18%`、grasp `46.30±8.49%`、colored grasp `47.53±5.58%`。GP+skill 是三个任务中最高的注册均值（one-leg 88.89%、round-table 56.48%、lamp 44.44%）；TAGPoint overall 第二（57.41%）。RGB/RGB-D 有效 checkpoint 不支持“只做单任务、不能多任务”的旧叙述；depth-only 的 overall 差值也很小，不将其夸大为显著优劣。
2. **第二段按两个受控方向拆信息贡献。**固定 GP 后加入完整 skill，one-leg / round-table / lamp 分别 `+6.48 / +14.81 / +11.11 pp`；固定 skill 后加入 GP，分别 `+11.11 / +7.41 / +10.18 pp`。这构成“what + where 互补”的观测证据，并非逐 rollout 的因果分解。TAGPoint 相比 GP 在三个任务分别 `+4.63 / +8.33 / +1.86 pp`、overall `+4.94 pp`；它将完整五类语义压缩为依附于位置的两组颜色，所以不是与 GP+skill 完全等信息量的替代。可用 1 句“Key insight”收束：**位置与操作语义同时提供时，完整任务成功率的观测均值最高；紧凑的颜色提示保留部分优势。**

#### C. 增加表征维度的反例与证据边界（1 个段落）

- 6D grasp / colored grasp 在 round-table 分别只有 `14.81±20.85%` 与 `22.22±19.25%`，而一条纯位置 GP 为 `41.67±12.11%`；one-leg 并未出现相同程度的失败。因此不能写“更丰富的几何必然更好”或“grasp 接口普遍无效”，只说在本多任务训练与这组 scripted annotations 中没有一致增益。真实 VLM grasp 在 lamp 可能有优势，留待后文，避免这里提前排除它。
- 用同 lineage 两 run 做方向核查：RGB-D `51.39±0.65%`，skill `54.17±1.96%`，TAGPoint `59.72±5.89%`，GP+skill `63.43±5.89%`；纯 GP 没有这组对应 run。注册主表跨 lineage 且重复数不同，不能声称小差值统计显著，更不能据此断言一套单点机制。这个边界作为段落末尾一句，不让它淹没主要发现。

#### Table 2 的入文、标题及图注任务

- **入文位置：**A 段末尾；文字先明确 clean scripted guidance 与统计单位，表格放在第一次读数之前。
- **标题建议：**`Complete-task success under clean guidance`。列为 Condition / One-leg / Round-table / Lamp / Overall，**不再单列 `$n$`**；RGB、RGB-D、RGB-D+skill、RGB-D+GP、RGB-D+TAGPoint、RGB-D+GP+skill、RGB-D+grasp、RGB-D+colored grasp 的顺序保留。
- **图注必须承担的信息：**单元是跨独立 training runs 的 mean ± sample SD；RGB/RGB-D 各两个有效 run、其余三个；每 run 每 task 36 rollout；粗体与下划线区分最佳/次佳、并列并列标记；GP+skill 是信息丰富的参考，TAGPoint 是后续 point-interface 主角；混合 lineage 意味着描述性排名，非精确 paired causal effect。不要让表格 caption 复制整个结果段落。

### 4.2 Spatial Conditioning Provides the Clearest Gains for Actions Requiring Higher Spatial Precision

#### A. 从完整任务到 skill 的分析问题及统计口径（1 个段落）

- 先说明 Table 2 将多个顺序依赖操作压缩为一个二元 outcome，无法区分“task/stage 识别”与“目标位置定位”的作用位置；分析底表按五类 **skill**（Pick/Place/Push/Insert/Screw）统计，Figure 2 展示 Push、Pick、Place and Insert、Screw 四类（仅去掉 Insert），`pick leg` 等具体任务节点始终称 **stage**。
- 读取与主实验相应的 66 份 task-level evaluation JSON：RGB-D 为两个有效 supplemental checkpoints，skill、GP、TAGPoint、GP+skill、grasp 和 colored grasp 各三个 checkpoint；每 checkpoint × task 为 36 条 rollout。每个 task-specific stage 统计进入 $R$ 与完成 $C$，同一 stage 在单条 rollout 中至多记一次，再按 skill 合并为 pooled conditional completion $\sum C/\sum R$。后续 stages 的进入状态取决于前面动作，故此指标是 workflow-level diagnostic，而不是 matched entry-state 的因果实验。
- **Figure 2 前导句（英文草稿）：** “To locate the behavioral source of the complete-task differences, Fig. 2 compares Push, Pick, Place, and Screw; Place has the clearest positive difference.” 每类 skill 的三根相邻柱按 skill、GP、GP+skill 排列，均直接相对 RGB-D 作差，不将 GP+skill 拆成带符号堆叠。

#### B. 先读全体 skill，识别饱和与退化（1 个结果段落）

- 有效 RGB-D checkpoints 的 Push `97.69% (211/216)`、Pick `98.31% (348/354)` 已接近饱和。相对 RGB-D，GP 的 Push/Pick 为 `+1.70/-2.31 pp`，skill-only 为 `-0.77/-1.98 pp`，GP+skill 为 `-0.15/-1.44 pp`；因此不能沿用旧 seed 的“condition 主要提升 Push/Pick”。Insert 不进入 Figure 2，也不需要在这一节正文单独解释。
- Screw 不是一律改善：RGB-D `90.77% (177/195)`，GP `78.29% (238/304)`，TAGPoint `88.81% (262/295)`，GP+skill `88.09% (281/319)`。GP 的降幅主要集中于 round-table（RGB-D `94/98=95.92%`，GP `114/144=79.17%`），特别是第一个 Screw stage（`54/54` 对 `69/88`）。可以解释为二维点只确定接触位置，而 Screw 还依赖旋转与接触过程；必须明确这是与数据相容的机制假设，不是 matched entry-state 对照所确立的因果结论。

#### C. Place 的主证据与具体 stages 的交叉核对（1–2 个结果段落）

- RGB-D Place 为 `77.45% (213/275)`；skill-only `77.12% (300/389)`、GP `83.84% (332/396)`、TAGPoint `80.15% (327/408)`、GP+skill `86.99% (361/415)`。对应相对 RGB-D 的 pooled difference 依次为 `-0.33 / +6.38 / +2.69 / +9.53 pp`。在已有 skill 条件下加入 GP，Place 增加 `+9.87 pp`。这才是“需要更高空间精度的动作获益”这一小节标题的直接证据。
- 附录四个具体 Place stages 中，GP+skill 相对 skill-only 均为正：one-leg leg `+5.07`、round-table leg `+4.99`、round-table base `+12.07`、lamp bulb `+18.10 pp`。用一句在正文提示一致方向，不在 Figure 2 重画 stage 细分，也不把 hood placement（标签覆盖不完整）纳入该四项比较。
- **可选的 Key insight 收束句：** “Spatial targets make their clearest observed contribution where the policy must resolve a precise placement location, even when the current skill is already specified.” 这是基于结果的解释，不声称 visual attention、latent alignment 或控制机制已经测得。

#### Figure 2 的入文、标题及图注任务

- **入文位置：**A 段末尾，第一次报告 skill 差值之前；不要把图挪到完整任务表前。
- **标题建议：**`Skill-level changes relative to RGB-D`。横轴依次为 Push、Pick、Place、Screw，纵轴为相对 RGB-D 的完成率差值（pp）；每组三根紧邻的竖柱依次为 skill、GP、GP+skill；配色、细边框和轻网格沿用 Place-step 对比图。
- **图注必须承担的信息：**RGB-D 使用两个有效 supplemental checkpoints，其余各三个；每 checkpoint/task 36 rollout；每根柱是对应 skill 直接相对 RGB-D 的 pooled proportion difference、不是 seed-mean difference 或配对的因果效果。四个 Place stages 与逐 checkpoint 不确定性在附录。

#### 段尾边界与向下章的连接

Complete-task 与 skill-level 两级证据共同支持“明确的空间 target 与低维语义互补，且 Place 是最清楚的受益环节”。但所有目标在这里均由正确 scripted annotations 提供；“更多信息在 clean setting 下可用”不自动意味着上游模型预测错误时也稳定，因此下章对注入误差和真实 VLM 输出作直接测量。

### 本章小结与下一章过渡

本章已经说明 clean setting 下什么信息被策略利用、收益最可能出现在哪类操作；下一章再问接口偏移、错位甚至由真实 VLM 产生时，这些信息是否仍然可用。不要在这里提前声称 colored point 的通用抗噪排名。

## 5. How Robust Are Conditioning Interfaces to Imperfect Guidance?

> 本章对应论文第 4 章。沿用 [*Much Ado About Noising*](https://arxiv.org/html/2512.01809) 的结果章节次序：章首先给读者一个有边界的答案，再说明后续小节怎样逐层检验这一答案；每小节按“问题与对照—主要观察—解释或反例—证据边界”展开，不机械设置 Key Insight 列表。

### 章首：先报告稳健性的层次，而不是先介绍噪声参数

**核心判断。** 五种 point/grasp 接口的完整任务成功率在测试的空间扰动范围内总体稳定，但稳定的 success 不意味着机器人忽略目标点：对实际显示目标测得的 tracking error 随扰动增大。真实 Point VLM 的典型定位误差落在受控实验覆盖的区间；在该误差窗口附近，TAGPoint 于较难的 round-table/lamp 上相对 GP 保持较高的观测成功率，而真实 VLM 输出也能驱动完整装配。这里没有“所有噪声下的统一最优接口”，也没有把 Gaussian 扰动等同于真实 VLM 错误。

**章首正文草稿（英文，之后写正文时精修）：** “Complete-task performance remains broadly stable under sizeable guidance perturbations, although the policies' target-tracking responses reveal that the perturbed points still affect control. We first measure assembly success under controlled errors (Sec. 4.1; Fig. 3), then inspect the motion relative to the displayed target (Sec. 4.2; Fig. 4), and finally test whether an actual VLM's predictions fall within a usable operating range (Sec. 4.3; Table 3).”

这三层分别回答 **能否完成 / 是否响应引导 / 真实上游是否可用**。不要在章首重复第 3 章的模型结构、把某条局部曲线的上升写成“noise helps”，或提前声称机制已由 tracking 识别。

### 5.1 Controlled Spatial Perturbations Reveal Interface Robustness

#### A. 问题、对照与协议（1 个紧凑段落）

- 首句承接 clean 结果：“The clean comparison cannot tell us whether a useful interface remains useful when the upstream target is misplaced.” 复用在 clean scripted guidance 上训练的 GP、TAGPoint、GP+skill、grasp、colored grasp checkpoint；只在 evaluation 时扰动接口，测的是 **test-time guidance-error response**，没有 noisy-condition retraining。
- n0--n7 对三维 target 加裁剪至 $[-2,2]$ 的 Gaussian position perturbation，per-axis $\sigma=\{0,3,6,12,24,48,96,192\}$ mm，再以同一相机投影成图像标记。grasp family 同时绑定 $\{0,2.5,5,10,20,40,60,90\}^{\circ}$ orientation schedule，因此它的曲线是位置与旋转的**联合扰动**。Shuffle 置换当前 stage 的 guidance，是 categorical semantic endpoint；r180 是 rotation endpoint，二者不混入连续尺度趋势。
- 每个 condition × noise × task 的 success 汇总三个 evaluation replicates、108 条完整 rollout，三任务 pooled cell 为 324 条。三个 replicates 是**同一训练 checkpoint 的评测 seed**，不能称为三个独立 training runs。以完整装配判据计算 success，并给 Wilson interval。详细 tracking 的不同样本量留到 §5.2 和图注交代。

#### B. 先报告全域观察，再定位真实 VLM 的工作窗口（2 个结果段落）

1. **全域结果先行。** n0→n7 的 pooled success 变化：GP `+4.0 pp`、TAGPoint `+0.7 pp`、GP+skill `+3.7 pp`、grasp `−1.8 pp`、colored grasp `−6.2 pp`；Wilson 区间重叠。写为“在这一测试范围内完整任务完成率总体稳定”，而不是噪声改善成功率或某接口在全域显著最鲁棒。Shuffle 单列为另一种语义扰动。
2. **再缩到部署相关窗口。** §5.3 实测 Point VLM 的 equivalent position scale 分别为 one-leg `36.15`、round-table `72.86`、lamp `68.34 mm/axis`；用任务各自的竖线标在 Figure 3 上，落在 n4--n5、n5--n6、n5--n6。该标线只提供读图坐标，不是额外成功率观测，也不预设 VLM 分布为 Gaussian。
3. **读任务差异而不是给全局排名。** Point-VLM window 附近 TAGPoint 与 GP 在 one-leg 接近，在 round-table/lamp 有更高的观测 task success；n5 时分别是 round-table `42.6%` 对 `40.7%`，lamp `42.6%` 对 `34.3%`。同区间位置 tracking 差异低于 `0.35 cm`，因此改善更体现在完整任务完成，而非简单“点跟得更近”。补充分析显示相对优势偏向 Pick/Screw，其他 skill 方向混合。
4. **处理反例。** grasp 在 n7 相对 GP 仍高 `3.1 pp`，主要由 lamp 驱动；one-leg/round-table 不一致。由于 grasp n0--n7 同时改变 position 和 orientation，只能讨论联合接口的表现，不可归因于某一维度。

#### Figure 3 的入文、标题及图注任务

- **入文位置：**A 段之后、B 段第一次读全域趋势之前；先说图要回答“接口在打点偏移后还可用吗”，再摆三任务分面图。
- **标题建议：**`Complete-task success under spatial guidance perturbations`。三个面板分别对应 one-leg、round-table、lamp；五种接口共享配色。连续 n0--n7 与右侧 Shuffle endpoint 要在坐标上断开，不能让读者读成 n7 的下一档数值噪声。
- **图注必须承担的信息：**clean-train/noisy-eval；n0--n7 的三维 per-axis σ 与投影关系；grasp 另有绑定的旋转 schedule；每 condition × level × task 为 108 rollout（同一训练 checkpoint 的三个 evaluation replicates）；Point/Grasp VLM task-specific position-equivalent 竖线的含义；若图上不画 replicate 散点或完整置信区间，明确其位置及 Wilson interval 的统计口径。不要把 Figure 3 caption 写成结论段落。
- **唯一读图结论：**完整任务 success 对大范围 point perturbation 总体稳定，真实 VLM 误差窗口暴露任务依赖的接口差异。

#### 段尾边界与通向 §5.2 的问题

成功率曲线本身无法说明稳定性来自“策略利用了点并容错”，还是“策略根本忽略了点”；因此下一节用同一批 rollout 的目标跟踪行为拆开这两种解释。禁止写“TAGPoint 在所有噪声下最鲁棒”“更大噪声提升了 policy”“Gaussian perturbation 完整模拟真实 VLM error”。

### 5.2 Target-Tracking Responses Reveal How Policies Use Perturbed Guidance

#### A. 为什么在 success 之外测 tracking（1 个紧凑段落）

- 开头设问：若 success 近似不变，policy 到底是否响应了被移动的点？使用 §5.1 的同一批 rollout，不引入新训练或评测 campaign。对每个 task-specific stage，在最后一个有效 control state 测 end-effector 到 **policy 实际收到的 displayed target** 的距离；这个 position tracking 不是到未扰动 clean GT 的距离，也不是完整任务 success 的替代指标。
- 五种 point/grasp 接口都有 position target，所以正文采用按有效 final stage-state 数加权的三任务 pooled position error。grasp 的 orientation/total error 放附录；否则 point 接口缺少对应 channel，主图不公平。正式 tracking 在 n0--n4/Shuffle 每 task 用两个新 evaluation replicates（72 rollout），在 n5--n7/r180 用三个 replicates（108 rollout），不能误写为整条曲线 fully matched。

#### B. 先报告共同响应，再检验语义置换这个反例（2 个结果段落）

1. **共同趋势。** 对五种接口，position tracking 随 position noise 增大，n7 接近 `25 cm`。这表示大幅偏离的 displayed target 更难被末端到达；不能把该曲线写成“相对 clean-GT 的动作误差随噪声线性增加”，也不把近似单调误称严格线性定律。
2. **数值偏移与语义置换对比。** n7 tracking 在 `15/15` 个 condition × task 单元中均高于 Shuffle，平均差 `11.0 cm`。n7 仍保留当前 stage 语义但把目标移远；Shuffle 可同时改变点所代表的 subtask 和 target。结果与策略对 supplied semantic-spatial cue 作出行为响应相容，不能推成已经识别了内部 attention/gating 机制。
3. **用 grasp 与 GP+skill 检查过度简化的解释。** Grasp family 在 n6→n7 的 pooled position/orientation/total error 为 `14.86→25.12 cm`、`98.4→124.8°`、`34.5→50.1`；放附录补充 orientation 维度，不在正文主图与 point family 混成单指标。GP+skill 的 pooled n0--n7 success range 为 `6.5 pp`，n6 Place 完成率 `88.0%`，高于 TAGPoint `82.7%` 与 GP `82.9%`；扩大样本后没有此前怀疑的 high-noise collapse。它作为 information-rich reference 的问题是固定 skill vocabulary 的扩展性，而不是已证实的不抗噪。

#### Figure 4 的入文、标题及图注任务

- **入文位置：**A 段末尾或 B 段共同趋势之前；Figure 3 看 success，Figure 4 才回答行为是否随 supplied target 变化。
- **标题建议：**`Three-task pooled position tracking under spatial guidance perturbations`。只画五种接口共享的 position metric；横轴连续噪声与右侧 Shuffle 分离；Point/Grasp VLM pooled equivalent scale 竖线只标参考尺度。
- **图注必须承担的信息：**按有效 final stage-state 数加权；tracking reference 是 displayed noisy target 而非 clean GT；n0--n4/Shuffle 每 task 72 rollout，n5--n7 每 task 108；额外 orientation/total grid 的附录位置；如图中仍标 `colored GP`，图注说明它对应 TAGPoint。不要在图注里解释内部表示机制。
- **唯一读图结论：**完整任务 success 的稳定性与 tracking 的目标敏感性可以同时成立；点条件在行为层面确实影响动作。

#### 段尾边界与通向 §5.3 的问题

受控位置噪声保持我们规定的分布，而 VLM 可同时犯语义和空间错误，且误差带有任务偏置与长尾。因此 Figure 3--4 确立的是 downstream response envelope，下一节必须用真实 VLM 输出检验这个 envelope 对系统评测是否有意义。

### 5.3 VLM-Generated Guidance Connects Controlled Robustness to End-to-End Performance

#### A. 从合成扰动到真实上游：建立公平的系统读表协议（1 个段落）

- 首句明确问题：“The synthetic curves alone do not establish that an actual VLM produces usable guidance.” 三个 FurnitureBench tasks 的 scripted trajectories 组成统一监督集，分别微调 Point VLM（输出五类 skill label 和 2D target）与 Grasp VLM（额外输出 target Rotation6D）。VLM 所接收的 stage instruction 指具体零件及当前步骤；五类 skill 是共享操作语义，不把二者混称。
- Rollout 中每 8 个 environment steps 更新一次引导，期间缓存输出；invalid structured output 不使用 oracle fallback。每个 VLM condition × task 为 36 条完整 rollout（overall 108）；无 VLM RGB-D baseline 合并两个有效 training runs，每 task 72 条（overall 216）。Point-family rows 共享 Point VLM，Grasp-family rows 共享 Grasp VLM；跨 family 同时改变预测头与接口，不是单变量 ablation。
- 正文主指标是 complete-task success；每个有效控制步还记录 VLM point 与 same-frame scripted GT projection 的 2D residual。no-VLM 与 VLM-guided policies 分别训练、样本量不同，评测起点不逐 rollout 配对，Table 3 仅作 **system-level descriptive comparison**。

#### B. 先给真实误差坐标，再解释它与 Figure 3 的关系（1 个结果段落）

1. 将每个有效控制步的 2D residual 与同相机 3D perturbation 的 projected RMSE 对齐：每 noise level 采样 200 个 camera-matched 扰动，再在相邻 levels 间插值得 equivalent `mm/axis`。这只是**误差幅度标尺**。Point VLM 的 task-specific equivalent $\sigma$ 为 `36.15--72.86 mm/axis`，Grasp VLM 为 `96.26--170.09 mm/axis`；n7 的 `192 mm/axis` 覆盖全部六个 task-level anchors。
2. 真实误差有结构：Point family bias norm `7.27--8.78 px`、anisotropy `3.91--5.73`；Grasp family 分别为 `26.41--26.50 px`、`2.27--2.30`。相对 n4 的 centered SWD/radial W1 非零且有长尾，因此不能将 Figure 3 的 clipped Gaussian curve 当作真实 VLM error 的同分布模拟。完整误差摘要表和 per-task anchors 归附录；主文保留对应 Figure 3 竖线所需的尺度与这个分布边界。

#### C. 读端到端结果：先 point family，再交代 grasp 反例（2 个结果段落）

1. 无 VLM RGB-D baseline 为 `111/216=51.4%`；三种 point interfaces 合计 `181/324=55.9%`。TAGPoint `64/108=59.3%` 是 point family 最高的**观测值**，GP `63/108=58.3%` 仅少一个 rollout，GP+skill `54/108=50.0%`。TAGPoint 相对无 VLM baseline 高 `7.9 pp`，但不同训练策略、样本量与未配对起点意味着不能单独归因于 VLM 或据此宣布 TAGPoint 显著优于 GP。
2. uncolored grasp 为全表最高 `68/108=63.0%`，尤其 lamp `22/36=61.1%`，相较三个 point rows 的 lamp `14--18/36` 更高；colored grasp 只有 `55/108=50.9%`，没有重现它的优势。将 rotation 描述为可能有用但**task dependent** 的额外表达通道，同时它要求 Grasp VLM 拟合更多输出并承受更大的上游误差；不把 grasp 的系统优势写成位置点本身无用。
3. **收束句可作为轻量 Key insight：**“The controlled study defines a downstream response envelope, while the actual-VLM study shows that imperfect predictions within the measured operating range can still support complete assembly.” 这不是 synthetic noise 曲线预测每条真实 rollout 成败的因果结论。

#### Table 3 的入文、标题及图注任务

- **入文位置：**A 段明确 family 与评测单位、B 段建立误差标尺之后；第一次读 success 数字之前。表格负责系统结果，正文负责解释 point-family 差距和 grasp 反例。
- **标题建议：**`System-level comparison with and without actual VLM guidance`。列为 Condition / One-leg / Round-table / Lamp / Overall；按 no-VLM baseline、Point-VLM family、Grasp-VLM family 分行组。行名统一为 RGB-D、RGB-D+GP、RGB-D+TAGPoint、RGB-D+GP+skill、RGB-D+grasp、RGB-D+colored grasp，避免旧报告中的 `grasp-part` 与 `colored GP` 漂移。
- **图注必须承担的信息：**no-VLM baseline 两个有效 training runs、每 task 72 rollout/overall 216；每 VLM-conditioned condition 每 task 36/overall 108；Point 与 Grasp 使用各自上游模型且后者另预测 rotation；所有策略独立训练、无 paired resets，所以是 system-level descriptive comparison，非“加 VLM”的单变量因果消融。可用粗体标全表最大值，但正文仍须明确 TAGPoint 的限定语为“among point-based interfaces”。
- **读表顺序：**先比较 no-VLM 与 point family，再比较 point family 内部，最后说明 grasp 在 lamp 的反例；不要只抓 overall 单列讲推荐接口。
- **唯一读表结论：**真实 VLM prediction 能驱动完整装配；TAGPoint 是 point family 最高观测值但领先 GP 仅一次成功，uncolored grasp 在全表最高且具有任务依赖性。

### 本章小结与下一章过渡

前三个 FurnitureBench 任务完成了从 clean scripted guidance、受控扰动到真实 VLM 的证据链，但不支持“接口对更大任务集合或真机普适有效”。下一章分开改变两个外部因素：先扩大 assembly identity 数量，再转向真实感知与控制系统；不能把两者合成一个笼统 generalization claim。

## 6. Scalability and Real-World Transfer of Conditioning Interfaces

> 本章对应论文第 5 章。按照上一章的“章首先给答案、小节分别用对照验证”的组织法，AutoMate 与真机分别检验 **assembly identity 数量** 和 **感知/控制系统域** 的变化，不作为并列的杂项补充实验。

### 章首：先说明已验证的规模边界，再引出真机问题

**核心判断。** 在 FurnitureBench 与 99 个 AutoMate training assemblies 的联合训练中，所有 condition 都保留了大量 AutoMate ID 完成能力，TAGPoint 给出最高 ID 观测值；但 FurnitureBench 的正迁移与干扰高度依赖 condition。单个 held-out assembly 提供初步 OOD 观察而非稳健泛化排名。真机目前已有可核查的 point-guided 执行片段，能够展示图像对齐接口如何工作；正式的 condition 对比与 VLM 端到端成功率仍需独立评测。这是“任务规模测试 + 真实系统验证”的两步，而不是一次证明通用性。

**章首正文草稿（英文，之后写正文时精修）：** “The three-task study does not establish whether the interface comparisons persist across a larger assembly set or on physical hardware. Joint FurnitureBench--AutoMate training retains substantial performance across 99 training assemblies, but its effects on the original long-horizon tasks depend on the conditioning interface. We first quantify this scale-up and its transfer cost (Sec. 5.1; Table 4), then examine how image-aligned guidance enters physical execution and evaluate the complete system under real perception and control errors (Sec. 5.2; Fig. 5 and the planned physical comparison).”

章首要区分**已有结果**（AutoMate ID/单个 OOD/FB joint，以及 scripted 真机示例）与**待完成结果**（真机 condition 排名和 VLM success）；不可先写“sim-to-real ordering holds”。

### 6.1 Scaling to a Larger Set of Assembly Tasks

#### A. 问题、联合训练和读表协议（1 个段落）

- 首句指出三个 FurnitureBench 长程任务可以控制 condition，却不能单独支持 task-scale claim。每个 condition 用 FurnitureBench 与 AutoMate demonstrations 进行 joint behavior cloning；AutoMate ID 覆盖 99 个 training assemblies，每个 assembly/condition 12 rollout，共 1188；OOD 仅一个训练未见的 assembly `00755`，每 condition 12 rollout。AutoMate 用 default $\times1$、fixed-hardest、SBC-off protocol。FurnitureBench joint formal checkpoint 在 one-leg/round-table/lamp 各 36 rollout（overall 108）。
- Table 4 同时呈现 AutoMate ID、单一 OOD assembly、joint-policy FurnitureBench overall，以及相对 **同 condition 的 main-experiment overall** 的 FB $\Delta$。AutoMate 面板主要使用 paired checkpoints，FurnitureBench 使用 formal checkpoints，AutoMate 的 RGB-D+skill 也是 formal；这是 condition-level cross-family evidence，不逐行声称严格 checkpoint matched。表前先定义 FB $\Delta$，让读者知道正负号不是对另一 condition 作差。

#### B. AutoMate ID 首先回答“扩到约百任务后是否仍可用”（1 个结果段落）

1. 八种 condition 在 99 个 AutoMate ID assemblies 上为 `59.2--65.1%`。TAGPoint 最高 `773/1188=65.1%`；RGB-D `761/1188=64.1%`，GP `754/1188=63.5%`，GP+skill `703/1188=59.2%`。这支持“接口在较多 assembly identities 下仍可学习/执行”，不支持“TAGPoint 与次高项有显著差异”或“一切任务规模下都最好”。
2. 对 held-out `00755`，GP 与 TAGPoint 都是 `10/12=83.3%`，RGB-D 为 `9/12=75.0%`，RGB 只有 `3/12=25.0%`。一项 OOD assembly、每接口 12 次，适合展示一个外部任务身份的可行性，不适合给出可靠的跨几何泛化排名。

#### C. 再检查原 FurnitureBench 任务的保留与干扰（1 个结果段落）

1. Joint policy 的 FB $\Delta$ 对 condition 敏感：RGB-D `+10.65 pp`、skill `+6.48 pp`，另六项下降 `2.78--10.49 pp`；八种 condition 平均 `−2.06 pp`。TAGPoint 的 FB joint overall `58/108=53.7%`、$\Delta=-3.70$；GP+skill 为 `57/108=52.8%`、`−10.49`。不把“平均只降 2.06 pp”写成所有接口都保留原能力，也不把 RGB-D 的正迁移归因于 depth 本身。
2. 一个可用的轻量 Key insight：**规模扩展并不统一奖励更丰富的 condition；新的任务家族可加入同一 policy，但原长程任务的迁移成本因接口而异。** AutoMate 当前只有 Insert annotations，因此实验测试 assembly identity 和任务家族尺度，尚未测试 Pick/Place/Push/Screw 全 vocabulary 的跨环境迁移。

#### Table 4 的入文、标题及图注任务

- **入文位置：**A 段之后、B 段第一次引用 65.1% 之前；表格是本小节唯一主文 display，不拆成 ID/OOD/FB 三张图来凑证据量。
- **标题建议：**`Cross-family joint training on FurnitureBench and AutoMate`。列为 Condition / AutoMate ID / AutoMate OOD (`00755`) / FB overall / FB $\Delta$ (pp)，行顺序与 clean Table 2 对齐。指标方向统一在表头标 success $\uparrow$；FB $\Delta$ 是保留/迁移诊断，不能只用正负号标“性能排名”。
- **图注必须承担的信息：**99 个 ID assemblies × 12 rollout = 1188/condition；单个 held-out `00755` 为 12/condition；FB 三 task × 36 = 108/condition；AutoMate default $\times1$、fixed-hardest、SBC-off；FB $\Delta$ 的减数是 condition-matched clean main result；AutoMate/FB 的 checkpoint provenance 不完全一致。粗体/下划线应说明最佳/次佳及并列，绝不可把 10/12 与 9/12 的排名措辞拔高为统计显著。
- **读表顺序：**AutoMate ID 的规模适应 → 单个 OOD 的初步外推 → FB $\Delta$ 的代价；不要只报告 TAGPoint 最高 ID 而省略 GP+skill 在 FB 上的干扰。
- **唯一读表结论：**joint training 获得 99-task AutoMate 能力，同时对原 FB 任务产生 condition-dependent 的正迁移或干扰。

#### 段尾边界与向 §6.2 的连接

AutoMate 保持了仿真域内的任务规模证据，但不覆盖真实相机深度、标定、执行器和接触误差。下一节因此必须另以 physical system 判断接口是否可部署，而不能把 AutoMate OOD 一行直接解释为 sim-to-real 证据。

### 6.2 Real-World Evaluation

#### A. 先让读者看到 policy 接收的点究竟如何工作（定性图 + 1 个段落）

- 开头设问：在真实 front-camera 输入中，image-aligned target 能否随操作阶段稳定指向零件和装配位点？Figure 5 取同一段 one-leg 真机 rollout 的 policy-visible frames，展示目标先在待抓零件上、后移到装配目标，再到机械臂接近目标位置。它解释接口如何进入真实 policy observation，**不声称这一序列证明完整任务成功**。
- 该片段使用 online scripted annotation，**不是 VLM 在线预测**；图中彩色点本就在 policy input 中，白环和放大窗只为论文阅读。青色对应 Pick/Screw 的粗粒度组，红色对应 Place/Push/Insert。素材是 one-leg，不能在图注写成 round-table。

#### Figure 5 的入文、标题及图注任务

- **入文位置：**§6.2 第一段之后，早于任何真机定量结果；功能是把 Figure 1 的接口概念变成真实输入示例，不是为物理成功率充当证据。
- **标题建议：**`Image-aligned guidance during a physical one-leg rollout`。顺序三帧：loose part（24 s）→ assembly target（52 s）→ gripper approaching marked site（128 s）；白色圈/放大窗与输入中真实彩色标点在视觉上区分。
- **图注必须承担的信息：**front-camera policy input、同一 rollout 的时刻和顺序、两种颜色的语义、白色辅助圈仅为阅读、guidance 来源为 online scripted annotation **而非 VLM**。不能写 demo 成功、任务完成率或“VLM generates these points”。
- **唯一读图结论：**目标点随操作阶段更新并直接进入真实 action expert 的图像输入。

#### B. 真机 condition comparison：先复核接口差异（待完成的定量小节）

- 在固定 camera、RealSense stereo depth 经 Prompt Depth Anything refinement 的 RGB-D preprocessing、控制器、demonstrations 和 evaluation starts 下，复现代表性 condition 对比。该实验对应 Table 2 的物理版本，检验仿真 interface ordering 在真实感知/控制误差下是否保留；不要先写“保持”或“下降不多”。
- **实验完成后本段的结果顺序：**先报告每个 condition/task 的完整装配成功率与 rollout 分母，再比较与模拟 clean Table 2 的相同/不同排序，最后按任务和失败阶段定位误差来源。若测得真实系统整体下降，只能说感知、深度、控制和接触误差的组合影响，不能独立归因于 depth。
- **正文必须补齐的协议字段：**具体 conditions、每 condition × task 的 trial 数、独立 policy runs、成功判据、timeout、初始零件配置是否 paired、VLM 是否关闭/改用 scripted guidance、失败与人工恢复规则。没有这些字段就保留 `[待填]`，不能填猜测数字。
- 真机已经综合覆盖真实 depth pipeline 的误差；除非出现无法解释的失败，不把额外 WAFT→depth 仿真消融设为第六章必要证据。真机可支持系统级容忍度，不能证明“depth error 本身只造成小幅成功率损失”。

#### C. 真机 end-to-end：再替换真实 VLM 引导（待完成的定量小节）

- 固定 B 的 camera/depth/control stack，将 scripted target 换成 VLM 预测的 TAGPoint；明确 VLM query frequency、invalid-output handling、人工恢复是否允许、每 task rollout 数与 task-success definition，默认不允许 oracle fallback。这样 B/C 才能分别回答“真实执行中的 interface effect”和“完整 VLM→TAGPoint→DiT pipeline 的系统可行性”。
- **实验完成后本段的结果顺序：**先报告 VLM 上游 point/type error 的分布及 failure categories，再报告完整装配 success，最后用同配置 scripted-vs-VLM 的对照解释差距。不能只放代表性成功视频，也不能将个别 demo 包装为大规模真机 benchmark。

#### 后续真实系统表格的入文、标题及图注占位（待实验）

- **建议 display：**单独一张 `Physical assembly success with scripted and VLM-generated guidance` 表；前半为物理 scripted condition comparison，后半为 VLM–TAGPoint end-to-end。若两组不能共享完全相同起点和控制协议，拆成两张紧凑表并在正文分别解读，不强行合并。不要与 Figure 5 的定性帧拼成一个视觉面板。
- **表前引导文案（待数字后落地）：**“We first compare conditioning interfaces under the same physical perception and control stack, then replace scripted targets with VLM predictions to evaluate the complete system.” 表中每行必须有 task-level $C/R$ 和 overall $C/R$，不能只有百分比。
- **图注必须承担的信息：**机器人/相机/depth preprocessing、condition 的真实来源、每 condition/task 的独立 runs 与 rollout 数、matched starts 与否、success/timeout/人工恢复规则。图注交代协议和统计单位，正文再解释排序与系统边界。
- **唯一待检验的信息：**真实感知/控制栈中 condition 的差异是否可复现，以及真实 VLM 输出是否能完成整个 physical assembly；目前不预写方向。

### 本章结尾边界

- AutoMate 的 task-scale 与真机的 physical-system transfer 是不同证据，不合成“通用泛化”。AutoMate OOD 仅一个 assembly；真机若 success 下降不多，可支持**整体 pipeline** 对真实 depth/control/contact 误差的容忍度，却不能识别单独的 depth 因果效应。
- 若真机定量尚未完成，摘要、贡献和结论只保留已完成的仿真和规模扩展结果；Figure 5 可作已经完成的 scripted qualitative demonstration，真机 success 仍只能写为 planned evaluation。

## 7. Related Work

### 本章角色

Related Work 不按论文逐篇罗列，而是围绕本文的接口问题建立三条研究脉络。每一小节最后都要落回“现有工作没有在同一 action expert 下系统比较接口信息与上游误差”。

### 7.1 Data-Efficient Furniture Assembly

- 从长时程依赖、接触精度、数据昂贵和状态覆盖不足组织相关工作。
- JUICER 等工作用于说明家具拼装为什么需要 task decomposition、数据扩展或 staged learning。
- 本文的区别不是声称替代所有训练方法，而是在固定 imitation-learning action expert 的情况下研究中间 condition。
- baseline 表述使用“prior multitask methods on FurnitureBench”或具体机制，不在核心贡献中直接把某一论文名称作为唯一对手。

### 7.2 Multitask Policy Conditioning

- 按 condition 所提供的信息分组：language/task identity、skill token、goal image、low-dimensional command。
- 比较这些方法是在告诉 policy “做什么”、提供 goal observation，还是定位“在哪里做”。
- 落到本文 gap：缺少在同一多任务、长时程 action expert 中对 spatial-only、semantic-only 和 combined interfaces 的受控比较，也缺少对不准确上游 prediction 的统一压力测试。

### 7.3 Keypoints and Visual Prompts for Robot Control

- KITE 说明“2D keypoint + finite skill library”可以连接语义 grounding 与执行。
- 本文的 GP+skill 提供相似的信息结构，但使用共享 DiT action expert，并在家具拼装中比较 clean/noisy/actual-VLM 表现。
- ATM 等 point/trajectory representation 工作用于说明 image-space target 的可解释性和行为诊断价值，不把 learn-from-video 的 motivation 搬到本文。
- 以 TAGPoint 收束：保留 image alignment，同时用颜色承载比固定完整 skill vocabulary 更粗粒度、可设计的 action cue。

## 8. Discussion and Limitations

### 推荐段落顺序

#### 第 1 段：中心解释

- 本文最稳定的发现不是“颜色点在所有设置中第一”，而是空间 target 与 skill/action 语义承担不同且互补的作用。
- Place 的提升支持二维 target 对 state-dependent 高精度目标的价值。
- TAGPoint 的优势在于把两类信息放在统一 image-aligned carrier 中，适合作为 VLM--action expert 接口。

#### 第 2 段：为什么不直接选择 GP+skill 或 6D grasp

- GP+skill 在 clean 和 noisy evaluation 中很强，是 information-rich reference；但固定 skill vocabulary 需要在扩展 skill 时重定义并重新训练接口。
- TAGPoint 当前只编码两组粗粒度 gripper action mode，信息更少但接口形式固定；其进一步可设计性是未来方向，不是已经验证的容量结论。
- 6D grasp 增加 orientation 表达能力，在 lamp 的真实 VLM 评测中有优势，但也增加上游输出误差维度，且不同组合下收益不稳定。

#### 第 3 段：如何理解 robustness

- success 在 0--192 mm 范围内总体稳定，不等于 policy 忽略 point；tracking response 表明行为会随 supplied target 改变。
- task success 与 tracking 衡量不同方面：前者是任务是否完成，后者是控制是否响应 guidance。
- 双系统的一个实际优势是将上游 grounding error 变成可单独记录、干预和测量的 interface residual：它在每次 VLM query 时由当前图像重新产生，而不是像 action error 一样直接通过状态转移递归积分。真实 residual 仍随 task、skill 和闭环状态变化，不能简化为全程固定或 i.i.d. 的噪声。
- synthetic perturbation 适合测 response curve，但不能复刻 VLM 的 bias、anisotropy 和 long tail。

#### 第 4 段：替代解释与数据限制

- GP 的空间分布本身可能携带 task/stage 信息，因此无法把 spatial 与 semantic 完全正交化。
- skill-level $C/R$ 受 policy-dependent entry state 影响，不能替代 fixed-state skill evaluation。
- 主表存在 mixed lineage、unequal train seeds；TAGPoint 与 GP 的真实 VLM 差距只有一次成功。
- AutoMate OOD 只有一个 assembly，且 cross-environment rows 并非全部 checkpoint-matched。
- 当前完整证据集中于一个 action-expert family 和 assembly domain。

#### 第 5 段：实际设计建议

- 若部署系统需要紧凑、可视化、可干预的 VLM interface，优先考虑 image-aligned spatial target，并在 target carrier 上编码必要的粗粒度 action semantics。
- 若 skill vocabulary 小且固定，GP+skill 仍是合理选择；若接触姿态是主要瓶颈，可以考虑 richer grasp interface，但需单独评估 rotation prediction error。
- 设计建议严格限定于本文研究的 interfaces 和 assembly setting。

## 9. Conclusion

### 推荐结构

1. **Contribution：**本文研究 VLM 应向共享 DiT action expert 提供什么条件信息，而不是提出一个全新的通用 control architecture。
2. **Decisive evidence：**二维 target 总体改善多任务表现，空间与 skill 语义互补，最清楚的 skill-level 收益位于 Place；受控扰动下 success 稳定而 tracking 响应目标位移。
3. **System evidence：**真实 VLM 可以驱动完整管线，TAGPoint 是 point interfaces 中的最高观测结果；AutoMate 说明接口在任务数量扩大后仍有效。
4. **Implication：**image-aligned point 为 VLM grounding 与连续动作生成提供了紧凑、可干预和可诊断的接口。
5. **Boundary：**结论限定于本文的家具装配任务、DiT action expert、实测 VLM error distribution 和当前真机规模。

Conclusion 不逐图复述数字，不引入新机制，不把 task-scale、OOD assembly 和 real-world transfer 合并成“通用泛化”。

## 10. 主文图表及其论证职责

| 顺序 | 固定内容 | 放置位置 | 唯一要表达的结论 | 状态 |
|---|---|---|---|---|
| Figure 1 | 家具拼装难点、接口选择边界、condition taxonomy、VLM→TAGPoint→DiT pipeline | Introduction | 上游应把已完成的语义与目标定位决策结构化地传给下游；本文因而聚焦以 2D target 为核心的接口，并以 TAGPoint 连接视觉 grounding 与连续控制 | 结构固定，待加入 text/3D 灰色排除分支、替换真实帧并补全全部 interface |
| Table 1 | five-interface information taxonomy | §2.2 | 明确区分 skill label、粗粒度 action mode、二维目标和末端旋转 | 已有；语义列保留 Skill label/Coarse mode，其他信息列使用 ✓/— |
| Table 2 | clean complete-task condition comparison | §3.1 | 空间与低维语义互补；GP+skill 是 information-rich reference，TAGPoint 是部署重点 | 已有；待加入 RGB-only 全 condition sweep 或同表分 panel |
| Figure 2 | multi-seed Push/Pick/Place/Screw contrasts | §3.2 | 四类 skill 的收益并不均匀，最清楚的正向差异位于 Place，GP 的 Screw 降幅须解释 | 已有竖向分组柱状图 PDF；横轴 skill、纵轴差值，仅去掉 Insert，Place stages 细分在附录 |
| Figure 3 | task-wise noisy complete-task success | §4.1 | success 对广泛扰动总体稳定，真实 VLM window 中存在 task-dependent interface difference | 已有 PDF |
| Figure 4 | pooled position tracking | §4.2 | policy 行为会响应被移动的二维 target | 已有 PDF；dense tracking grid 在附录 |
| Table 3 | actual-VLM system success | §4.3 | 真实 VLM 能驱动完整系统；TAGPoint 是 point interfaces 中最高观测结果但仅比 GP 多一次成功 | 已有 |
| Table 4 | FurnitureBench+AutoMate joint training | §5.1 | interface 在约百任务规模仍有效，但对原 FurnitureBench 的 transfer/interference 取决于 condition | 已有 |
| Figure 5 | physical one-leg front-camera guidance sequence | §5.2 | 展示 TAGPoint 的图像对齐目标如何随阶段切换；定性示例，不代表完整任务成功 | 已有 PDF；scripted guidance，非 VLM |
| Real-world quantitative display（待编号） | condition comparison + VLM end-to-end evaluation | §5.2 | 分别验证真实系统中的 interface ordering 与完整 VLM→policy 可行性 | 待实验 |

### 附录内容分流

- 四个 Place stages 的细化结果；
- pooled noisy success；
- task × position/orientation/total tracking dense grid；
- noisy skill-level success 与 tracking；
- VLM position-equivalent noise anchor table 与 calibration 公式；
- 网络、训练、数据、随机化与真机硬件细节；
- representative failure cases。

Low-train→Med-eval 的旧空间随机化结果不进入当前主论证链。其样本量较小、成功率整体接近零，最多作为附录 failure-boundary observation，不能恢复为独立的“space generalization”主章节。

## 11. 章节之间的连接句

这些句子用于保证论文不是若干实验的拼接，英文正文可在扩写时据此改写。

- **Introduction → Setup：**为比较中间接口本身，我们固定任务数据、action expert 和优化设置，只改变传递给策略的条件信息。
- **Setup → Clean comparison：**在统一实验基座上，我们首先检验每种接口在准确引导下能为完整装配提供多少有效信息。
- **Clean comparison → Skill analysis：**完整任务成功率复合多个依赖阶段，因此下一步定位 condition 收益具体发生在哪些 manipulation skill。
- **Clean chapter → Robustness：**clean oracle guidance 证明接口可能有用，但不能说明它能否容忍真实上游预测误差。
- **Noisy success → Tracking：**稳定的任务成功率既可能来自鲁棒使用，也可能来自忽略 condition，因此需要 tracking response 判断行为是否随目标变化。
- **Controlled noise → Actual VLM：**受控扰动给出下游 response curve，真实 VLM 评测则检验有偏、各向异性和非高斯误差下的完整系统表现。
- **Robustness → Scale/real world：**三个仿真任务中的闭环证据仍不足以建立外部有效性，因此进一步改变任务规模与系统域。
- **Results → Discussion：**在上述证据范围内，最后讨论为什么空间与语义信息互补、何时应选择 richer interface，以及结论在哪些条件下停止成立。

## 12. Claim--Evidence Map

| 主张 | 直接证据 | 当前强度 | 正文写法边界 |
|---|---|---|---|
| 空间 target 与 skill 语义互补 | Table 2 中 GP↔GP+skill、skill↔GP+skill 对比 | 支持，但并非完全 paired | 写“support complementary roles”，不写严格因果分解 |
| 二维 target 的收益最清楚地位于高空间精度动作 | Figure 2 与四个 Place stages | 支持，属于 diagnostic | 写 Place 差异，不写所有 spatial skill 都提升 |
| TAGPoint 是推荐的 point interface | clean、VLM-window、actual-VLM 64/108、AutoMate ID 65.1% | 多来源一致但优势小 | 写 recommended among studied point interfaces，不写 universally best |
| policy 在行为上使用二维 point | tracking 随 perturbation 增长；n7>Shuffle 为 15/15 | 行为证据较强 | 不外推到内部 attention/gating mechanism |
| conditioning interface 对上游误差稳定 | n0→n7 pooled success 小幅变化、区间重叠 | 支持总体稳定 | 不建立全局显著 robustness ranking |
| 真实 VLM 可驱动完整 pipeline | Table 3 complete-task success | 系统级支持 | no-VLM/VLM 非单变量 paired causal ablation |
| 接口可扩展到更大任务集合 | 99 AutoMate ID + joint FB | 支持 task-instance scale | AutoMate 仅 Insert，不写五-skill transfer |
| 接口可迁移到真实系统 | 待完成的真机 condition comparison 与 VLM end-to-end 定量表；Figure 5 只说明接口形式 | 待实验 | 结果出来前不进入最终 claim |
| condition 在没有 depth 时仍有效 | RGB × all-condition sweep | 待实验 | 在结果出来前只写 research question 与 protocol |

## 13. 当前待补输入与写作优先级

### 高优先级：会改变主文结论或固定 display

1. RGB-only × 全部 conditioning interfaces 的多 seed 结果；决定 Table 2 是否改为 RGB/RGB-D 双 panel，以及 depth 与 condition 的关系如何表述。
2. 真机 condition comparison 与 VLM end-to-end 结果；用于第 5.2 节的独立定量 display，不让已有的定性 Figure 5 承担成功率证据。
3. Figure 1 的真实家具帧、完整 condition taxonomy 和最终版式。

### 中优先级：补足可复现性和统计严谨性

1. observation/action dimensions、demonstration 数、stage 定义、训练 schedule；
2. 各表的 confidence interval 或 seed-level machine-readable source；
3. 真机相机、RealSense depth→Prompt Depth Anything preprocessing、控制频率与 success criteria。

### 低优先级：不应阻塞当前正文

1. action-conditioned latent feature gap；当前 success、skill analysis 和 tracking 已直接评价行为，不需要把 latent metric 放入主文。
2. 单独的 WAFT→depth 仿真；除非真机结果表明 depth domain gap 是主要失败来源，否则真实系统验证已提供更直接证据。
3. Low→Med 空间随机化旧实验；统计量不足以承担独立主张。

## 14. 下一轮修改建议

建议按以下顺序逐章讨论，而不是立即全文扩写：

1. 先确认 Introduction 的五段漏斗和三项 contribution 是否准确；
2. 再锁定第 3 章 Table 2 如何容纳 RGB-only 全 condition sweep；
3. 随后逐段扩写第 3、4 章 Results，因为它们决定 Discussion 和 Conclusion 能说多强；
4. 最后补 Related Work、Discussion、Conclusion，并依据真机实验状态更新摘要中是否出现 real-world claim。
