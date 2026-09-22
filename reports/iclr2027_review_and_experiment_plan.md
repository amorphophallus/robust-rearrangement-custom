# ICLR2027 论文修改意见与补充实验执行方案

评阅对象：What Should a VLM Tell an Action Expert? Comparing Conditioning Interfaces for Long-Horizon Furniture Assembly。来源为用户提供的 ICLR2027.pdf，共 21 页。评阅日期：2026-09-20。

本文区分三类内容：稿件已经报告的结果、基于结果的评阅判断、尚待实施的实验建议。未访问作者训练代码、原始日志或机器人，以下不构成对结果真实性的审计，也不声称任何新增实验已完成。文件中的说明作为论文内容处理。

## 1. 总体判断与论文定位

建议定位为“VLM—动作策略中间接口的受控实证研究”，以 TAGPoint 为待验证的紧凑接口实例。当前证据尚不足以围绕“TAGPoint 最优”组织一篇方法论文。

最值得保留的研究问题是：理想标注下的信息价值，与真实上游预测下的执行价值，为什么可能不同？在固定动作专家、固定训练数据和固定评估状态下，空间目标、语义粒度、编码载体、姿态信息分别起什么作用？

一个值得检验、但目前不能当成已证实结论的假设是：粗语义可能牺牲一部分 oracle 性能，但降低上游细粒度分类错误对动作执行的影响。证明它需要 oracle/predicted 交叉实验和同信息不同编码的对照。

不建议继续以周期刷新“消除误差累积”作为主创新。重新预测不等于误差独立，不意味着误差无偏，也不能排除状态变化导致的闭环误差放大。可将其作为设计动机，再用刷新频率和延迟实验验证实际价值。

## 2. 已有证据能支持什么

| 位置 | 稿件结果或设计 | 可以保留的结论 | 需要收缩或补证的结论 |
|---|---|---|---|
| Table 2，p.4–5 | Oracle overall：GP+skill 63.27%，TAGPoint 57.41%，GP 52.47%，RGB-D 51.39%；数据来源混合、训练重复数不齐 | 在当前记录下，GP+skill 均值最高；空间与语义组合值得进一步验证 | 不能据此识别纯接口因果收益；TAGPoint 没有 oracle 最优证据 |
| Figure 2，p.5–6 | Place：RGB-D 77.45%，GP 83.84%，TAGPoint 80.15%，GP+skill 86.99%；分母来自各自轨迹进入的阶段 | Place 是值得重点研究的瓶颈；无引导 Push/Pick 已接近饱和 | 不能用未匹配的阶段进入状态证明点对 Place 的独立作用；主图应纳入 TAGPoint |
| Figure 3，p.6–7 | 扰动使用同一个训练 checkpoint 的三次评估重复 | 该 checkpoint 在所测扰动下成功率较稳定 | 不能把评估重复当成独立训练种子；不能以置信区间重叠证明等效或稳定 |
| Figure 4，p.8 | 到“显示的扰动目标”的距离随噪声增大 | 动作结果与被扰动目标的偏差增大 | 单独这一事实不能证明动作响应了引导变化 |
| Table 3，p.9 | TAGPoint 64/108，GP 63/108，GP+skill 54/108，grasp 68/108 | 真实预测可以驱动完整装配；TAGPoint 是点接口中观察均值最高的一项 | 1 次成功的优势不足以推荐全局最优；grasp 是该表总体最高值 |
| Table 4，p.10 | TAGPoint AutoMate ID 773/1188，RGB-D 761/1188；OOD 仅一个几何体、12 次；部分面板 checkpoint 不一致 | 在较多已见装配身份上的可用性 | 不能证明跨几何泛化、技能词表扩展或 scaling law |
| Figure 5，p.10–11 | 真实 one-leg 的三个画面，使用脚本引导，没有完整任务成功率 | 界面能接入物理系统 | 不能命名为已经完成的 real-world evaluation，也不能支持 VLM 真机闭环成功 |

Table 3 中 TAGPoint 与 GP 的差值是 0.926 个百分点。仅作量级说明：把各自 108 次试验粗略当作独立同分布二项试验时，差值的正态近似 95% 区间半宽约 13.1 个百分点；正式分析还要考虑任务分层与训练种子，不能用这一近似代替主检验。

## 3. 最优先修复的论证缺口

### A 3.1 “使用了引导”的论证目前存在反例

令机器人最终位置为 x，真实目标为 p，显示目标为 p+δ。若策略完全忽略引导，x 不变，距离 ||x−(p+δ)|| 也通常会随 δ 增大。故 Figure 4 的趋势可以由评估参考点移动产生。

建议立即把“tracking responses reveal guidance use”等机制性标题改为中性的“Tracking relative to perturbed targets”，直到同状态干预实验完成。正文、Discussion 和 Conclusion 中依赖该推论的句子需同步修改。

### ? 3.2 TAGPoint 同时改变了语义粒度和编码位置

当前 GP+skill 使用五类向量标签，TAGPoint 使用二类颜色标签；两者比较同时改变了标签信息量、输入通道和空间绑定方式。至少增加 GP+二类向量、五色点两个条件，形成 2×2 因子设计。

Pick/Screw 与 Place/Push/Insert 的划分需要可检验的解释。不能未经验证就等同于瞬时夹爪开关状态。建议定义为 coarse action group，并给出每个 stage 的映射、接触阶段语义、是否全程固定、标签由何种可观测信息得到。

### ? 3.3 GP+skill 同样可以部署

Section 4.3 的 Point VLM 已经输出完整技能标签，再将标签折叠为 TAGPoint 两类。因此 GP+skill 并不天然比 TAGPoint 更难部署，当前实现也没有直接证明 TAGPoint 的上游输出 token 或时延更低。“information-rich reference”可以保留；与“deployable interface”对立的写法应删去。

如果要声称简化了上游预测负担，需另训直接预测二类 mode 的 Point VLM，并与“预测五类再映射”比较；这与动作专家侧的编码消融是两项不同实验。

### B 3.4 端到端的起点不清楚

Abstract/Introduction 说 VLM 负责任务分解，Section 4.3 却把 stage-level instruction 作为已给输入。需明确整任务指令如何变成当前 stage、谁判定切换、失败时谁决定重试。如果阶段切换依赖脚本或仿真真值，应称为 oracle-stage-guided execution；只有任务分解、阶段判定和动作生成均无 oracle 输入时，才可称完整任务级闭环。

同样要明确：前视点怎样传给腕视图？若使用深度反投影和相机外参，需要公开步骤及无效深度处理；若分别预测，需说明一致性约束；不能默许使用仿真真值投影完成真实 VLM 模式下的另一视角标注。

### C 3.5 鲁棒性没有分离错误类型

当前主要是位置扰动，grasp 同时叠加旋转扰动；无法分别归因位置与姿态。真实错误还包括错物体、错技能、阶段滞后、旧像素坐标缓存、格式错误和延迟。噪声强度的 RMSE 匹配只能校准幅度，不能替代这些错误分布。

## 4. 建议的文章结构

| 建议章节 | 必须回答的问题 | 主要证据及修改 |
|---|---|---|
| 1 Introduction | 为什么接口信息的多少不是唯一选择标准？ | 四段：任务困难；现有接口差异；上游误差与下游利用之间的缺口；研究问题与边界明确的贡献 |
| 2 Interface Design and Hypotheses | 究竟比较哪些变量？ | 内容：位置/语义/姿态；语义：无/二类/五类；载体：向量/图像；时间：刷新/缓存；将 TAGPoint 放在设计空间中定义 |
| 3 Experimental Protocol | 怎样保证比较公平？ | 数据、模型、VLM、任务切换、训练种子、初始状态、checkpoint 选择、统计单位集中交代 |
| 4 Controlled Interface Study | 正确引导下各信息有什么价值？ | 配对主表、2×2 编码消融、匹配阶段入口测试；区分整体任务与局部技能结论 |
| 5 Guidance Errors and Closed-Loop Execution | 什么错误导致收益消失？ | 同状态干预、点/类型交叉替换、真实错误分布、刷新与延迟；展示 oracle 到 predicted 的同 checkpoint 降幅 |
| 6 Generalization and Physical Validation | 结论能延伸多远？ | 多几何留出与迁移；有定量真机才保留 validation，否则将演示移入附录 |
| 7 Related Work | 本文新增的是哪种证据？ | 聚焦关键点、二维路径、语言层次、三维约束与装配研究，不泛列 VLA |
| 8 Discussion and Conclusion | 哪些条件下选哪种接口？ | 总结经过验证的条件性选择原则，不写普适排名 |

保留现有问题式标题是合理的。可把副标题稍作聚焦：A Controlled Study of Conditioning Interfaces for Long-Horizon Assembly。没有完成受控实验前，标题中的 Controlled 也应谨慎使用。

摘要建议按“问题—固定变量的设计—两项主结果—限制与设计启示”写，约 180–220 个英文词作为内部编辑目标；此范围不代表会议强制要求。去掉五种接口的逐一长句列举，以及没有量化支持的 best/recommended/scalable 表达。不要在新实验完成前填入预期提升。

引言建议四段：

1. 长程装配中，目标定位、动作模式与接触控制的职责需要划分。
2. 关键点、技能标签、轨迹、姿态提供不同信息，也带来不同预测误差。
3. 已有系统比较通常同时改变上游和下游；本文要隔离接口变量，并研究 oracle 排名能否迁移到真实预测。
4. 贡献写为受控比较、错误归因协议、任务条件下的设计发现；TAGPoint 是其中一个具体设计。

从正文删除“earlier single-task-collapse interpretation”“previously suspected high-noise collapse”等内部修订历史。读者应看到最终论证。数据审计和历史 run 的排除规则放入协议附录；关键局限保留，但无需每段重复。

## 5. 实验统一协议：所有新增比较先遵守这一层

### 5.1 数据与训练

- 冻结原始 demonstrations、训练/验证/测试切分和标注版本；按轨迹划分，避免相邻视频帧泄漏。
- 明确 target 是对象上的交互点、期望末端终点还是接触位点；标注是否用到未来轨迹或目标几何。Oracle 可以用特权信息，但需明确属于上界。
- 固定前/腕 RGB-D、图像尺寸、深度归一化、动作表示、坐标系、控制频率、chunk 32/执行 8、DiT 配置、DDIM 16 步及总训练更新数。
- 接口适配器输出同维度；记录参数量和 FLOPs，避免以额外模型容量解释收益。相同预算下各接口可有独立训练权重，这本身不是不公平；问题在于训练协议是否匹配。
- 首轮至少 3 个独立训练种子，关键结论建议 5 个。每种子每任务 100 个预先保存的 reset；成本受限时先用 50 个做诊断，不能将小试验当成稳定排名。
- 固定验证集 checkpoint 选择规则，不以最终测试成功率选模型。seed 排除只允许预先定义的技术故障，如损坏日志、配置偏离或训练数值失败，不能按低分排除。

### 5.2 配对与统计

- 相同 reset ID 用于各方法；同一训练种子的公共网络初始化及数据顺序尽量一致，允许接口头不同。
- 主要指标是每个训练种子上的三任务宏平均完整成功率。报告每任务计数、种子均值与标准差、配对成功率差及 95% 区间。
- bootstrap 需要保留方法间配对关系，并反映训练种子和 reset 的层次/交叉结构；若同一 reset 跨种子重复使用，不能把所有 rollout 简单视为独立样本。种子只有 3 个时区间估计仍不稳定，要展示所有种子。
- 单 checkpoint 的成功率可用 Wilson 区间，但其不能代表训练不确定性。离线 VLM 控制步也不能当成独立样本；按 episode 或几何体聚类。
- 少量预先指定的主对比优先于全表两两检验；探索性多重比较注明，并在适用时做 Holm 等校正。
- 如要声称“性能稳定”或“不劣于”，预先定义实际可接受降幅，例如 5 个百分点只是可讨论的候选，不是事后挑选阈值。区间重叠或 p>0.05 都不证明等效。

### 5.3 主表统一

在同一行列出 oracle、predicted 和二者差值，三者必须来自同一个动作 checkpoint。同一协议下不同接口可以各自训练，但不得将旧 oracle 表与新 predicted 表直接相减。

示例列：Interface / Seed count / Oracle SR / Predicted SR / Oracle-to-predicted change / Relative-to-GP paired difference / VLM latency / Valid output rate。

## 6. P0：投稿前最值得完成的实验

### x E0：实验台账审计与主表重跑

问题：现有排名是否来自接口，还是数据、checkpoint 和评估分布差异？

执行：保留原 8 个条件：RGB、RGB-D、skill、GP、TAGPoint、GP+skill、grasp、colored grasp；按上述统一协议训练与评估。若算力有限，先完成 RGB-D、skill、GP、TAGPoint、GP+skill、grasp 六个核心条件，明确其余为历史结果，不能混入新的主表排名。

完整规模：8 条件 × 3 训练种子 = 24 次训练；每 checkpoint 3 任务 × 100 reset，共 7,200 次 oracle rollout。六条件诊断版为 18 次训练、5,400 次 rollout。此为任务计数，不是 GPU 时长承诺。

输出：配对主表、所有训练种子散点、每任务失败分布、manifest。把“历史补跑”作为附录单独 panel，不与新协议混合计算总均值。

判断：如果优势在统一协议下消失，论文应转为信息价值与误差敏感性的研究，不继续以旧结果维护 TAGPoint 排名。

### ? E1：语义粒度 × 编码载体，决定 TAGPoint 的贡献到底是什么

|  | 二类语义 | 五类技能 |
|---|---|---|
| 点 + 独立向量 | 新增 GP+mode2 | 现有 GP+skill5 |
| 点颜色编码 | 现有 TAGPoint2 | 新增 ColorPoint5 |

所有条件使用相同点坐标、点大小、可见性规则和训练样本。两个向量条件用相同输出维度的 adapter。五色点控制颜色亮度、饱和度与可分辨性，至少换一次调色板验证不是某种颜色可见性造成的结果。

主要对比：TAGPoint2−GP+mode2 测试图像内绑定；ColorPoint5−GP+skill5 测试相同五类信息的载体效应；GP+skill5−GP+mode2 测试语义粒度。各自报告 oracle 和 predicted。

补充诊断：选择两个预先固定的 2+3 随机技能分组，与有语义依据的分组比较；仅改变颜色的训练/测试一致置换可检验颜色命名无关性，测试时单独换色则是错误干预，二者不可混淆。

成本：两个新增主条件 × 3 seed = 6 次训练，oracle 评估 1,800 次；可复用 E0 的另两个条件。

判断：若 TAGPoint≈GP+mode2，只能说二类语义有效，不能把收益主要归因颜色绑定；若任意分组同样有效，应考虑容量限制或正则化解释；若五类在 oracle 更好、二类在 predicted 更好，才接近目标机制。

### A E2：真实 VLM 的位置—语义交叉替换

对同一动作 checkpoint 运行以下四个单元，不改变任何其它控制设置：

| 点来源 | 语义来源 | 识别目标 |
|---|---|---|
| GT | GT | 同 checkpoint 的干净参照 |
| Predicted | GT | 位置错误代价 |
| GT | Predicted | 类型错误代价 |
| Predicted | Predicted | 两类错误的联合闭环代价 |

重点跑 TAGPoint2、GP+mode2、GP+skill5；GP 只需要点来源两档。每个有效实验单元使用 3 seed × 3 task × 100 reset；先以 50 reset 做诊断可减少成本。细粒度类型使用同一 Point VLM，TAGPoint 从该输出映射二类，避免更换上游模型造成混淆。

离线评估必须使用对各接口共同的 held-out 图像集，记录原始点误差、分位数、偏置、归一化图像坐标误差、五类与二类混淆矩阵、无效输出率。按 episode 抽样和 bootstrap。在线评估则允许每个方法访问自己的状态，不能将其它策略离线缓存的预测当成自身闭环预测。

关键统计：组内错分率 P(c_hat≠c 且 g(c_hat)=g(c))、跨组错分率、各类错分后的后续阶段失败概率。后者是关联诊断；固定状态注入相同类型错误才能识别具体错误的作用。

另报两种 pipeline：脚本阶段切换 + VLM 点/类型；VLM 阶段识别/切换 + VLM 点/类型。完整任务级成功必须对应后一种。

若要检验“上游更容易”，再用同一底座、同一数据和训练预算比较直接输出 mode2 与输出 skill5 后映射，并区分这是上游预测任务消融。

### A E3：同状态下干预引导，直接检验是否改变动作

建立未用于训练的阶段入口 snapshot bank。快照包含物体/机器人位置速度、夹爪状态、可恢复的接触信息、随机数状态、计时器、阶段计数器、缓存引导及相关控制器状态。先验证恢复后的短程轨迹可重复；物理引擎未暴露的接触缓存可能破坏精确重放。

每阶段先收集 50 个有效 snapshot，其中应包含正常状态和合理偏差状态。对同一状态恢复以下条件：正确点、向不同方向移动的点、同场景另一个候选对象/目标点、颜色翻转、删除点、固定点。删除或随机点属于分布外诊断，只能与其他证据共同解释。

每次保持 diffusion 初始噪声与随机数一致，比较下一动作 chunk 和前 K={1,8} 步行为。预测动作的平移、旋转、夹爪分量分别统计，避免把不同比例/单位直接相加。

建议指标：

- 动作差：D_a(δ)=E[||a(o,p+δ)−a(o,p)||]，分动作分量报告。
- 图像空间方向响应：G_K=〈x_img(t+K;δ)−x_img(t+K;0),δ〉/||δ||²，其中 x_img 是同一相机下的末端投影。
- 到干净目标的距离、到显示目标的距离、匹配入口的阶段完成率，分别报告。
- 引导不变时的多次采样作为随机动作波动底噪；引导改变时使用公共随机数。

先在接触前测方向响应，再在接触阶段测实际执行代价。点坐标本身可携带阶段信息，错误点的机制解释也应区分错空间目标和错阶段。

增加一个几乎不需新 rollout 的“忽略引导”参考：将干净运行的末端轨迹保持不变，只在离线评分中改变目标点。如果旧 Figure 4 趋势与该参考接近，就不能把原曲线解释为策略跟随。

最有辨识度的任务是同一观察下有两个都可达的装配目标，通过指令或点指定其一。此时还要向语言基线提供同样任务身份，区分指令跟随与自动完成默认目标。

### A E4：匹配入口的技能级实验

复用 E3 snapshot bank。优先测试四个已有 Place stage 以及表现异常的 round-table Screw，补齐 lamp hood 的标注覆盖；再扩展其它阶段。

每接口从相同 snapshot 运行至阶段结束，使用相同超时、成功定义和中断规则。报告每 stage 的成功率和配对差，再按预先规定权重聚合技能。完整任务 C/R 继续保留，作为自然到达分布下的描述指标；不要用它替代固定入口实验。

这可以区分 GP 对 Screw 的局部影响，和它使更多困难轨迹进入 Screw 后造成的样本组成变化。Push/Pick 接近饱和，不应据此推断空间引导在更困难分布下无用。

## 7. P1：支撑鲁棒性、现有工作比较与泛化

### E5：语义噪声、位置噪声与时序误差

先测试点噪声 × 类型错误的二维网格。位置可用 σ={0,12,48,96} mm；类型翻转率可用 q={0,0.1,0.3,0.5}。这些是起始设计，需根据训练外验证集的实际 VLM 分布调整并冻结，不能按测试优势挑点。

分别使用均匀错分与从验证集混淆矩阵采样的错分；二类和五类接口分别控制有效错误率，避免同一个五类错误概率在映射后二类错误率自然更低却未披露。

位置扰动分别考虑每次刷新独立抽样、整个 episode 固定偏置、与缓存时间相同的相关误差。直接向图像接口注入归一化像素误差也应测试：它比只使用 3D 投影噪声更贴合 Point VLM 的输出。

加入从验证集抽取的真实残差块，保留位置—类型—时间关联；将它与等 RMSE Gaussian 分开比较。不要把跨图像残差复制到另一目标后仍称为真实预测，只能称经验分布噪声。

对 grasp 采用 position-only、rotation-only、joint 三条轴。旋转在 SO(3) 上用 exp([ω]×)R 施加，角度用测地距离；声明物体对称性和等价姿态处理。不用“位置噪声的绑定角度档”代替独立姿态敏感性。

报告 raw curve、相对干净降幅、容许降幅内最大噪声及不确定性。若计算曲线面积，明确积分区间、坐标和归一化，避免横坐标变换改变排名却未解释。

### E6：刷新频率、延迟与缓存坐标系

在相同策略上测试刷新间隔 1/4/8/16/32 环境步及 stage-start-only，记录控制频率对应的秒数。加入 0/1/2 个 action chunk 的推理延迟。报告每任务 VLM 调用次数、p50/p95 时延、完整成功率和成功轨迹完成时间。

前视相机固定时的旧点与腕相机移动时的旧像素点含义不同。分别定义：缓存像素坐标；缓存三维目标再投影；跟踪更新目标。后一两项若引入深度或跟踪器，作为单独条件披露其信息和计算开销。

这一实验能检验“周期重新定位”的价值，而无需提出未被证明的误差不会累积论断。

### E7：外部方法与接口基线

现有主表主要是内部消融。建议增加以下比较，但把“同骨干接口比较”和“原方法系统比较”分表。

| 基线 | 推荐实现 | 回答的问题 | 优先级 |
|---|---|---|---|
| Language-conditioned DiT | 同 DiT 接入冻结文本 encoder + 等预算 adapter；比较任务级/阶段级文本及 GP+文本 | 用离散技能排除语言是否合理？ | 高 |
| HAMSTER-style path | 从相同示范自动提取下一阶段/固定时间窗的末端二维稀疏路径及夹爪事件，统一可预测时域；先 oracle 后 VLM；固定动作专家 | 单点相对轨迹的信息—预测难度权衡 | 高 |
| KITE-style execution | 共享骨干条件策略与 keypoint+skill 路由策略对比；记录每 skill 数据及总参数，必要时另给总参数匹配版 | 优势来自接口还是共享动作专家？ | 中 |
| 3D point+mode | 输入同目标的相机系/机器人系三维坐标，分别用 oracle geometry 和可获得的深度估计 | RGB-D 条件下 2D 接口是否仍有优势？ | 高 |
| Pose+mode | 保持位置、语义、标注目标一致，只增加旋转 | 姿态是否值得预测成本？ | 中 |
| 第二动作骨干 | 选已有可复现的另一类 diffusion policy，仅跑 RGB-D/GP/TAGPoint/GP+skill | 结论是否绑定 DiT/ResNet？ | 中 |

HAMSTER 已使用高层 VLM 预测二维路径，再交给具有三维感知的低层策略；这是比一般 VLA 更直接的参照。官方来源：[HAMSTER](https://hamster-robot.github.io/)。上述同 DiT 版本必须标为 HAMSTER-style interface adaptation，不能称原方法完整复现。

KITE 用二维关键点执行关键点条件技能，与你们的 GP+skill 有信息层面的联系，但共享 DiT 不等于复现技能库架构。官方来源：[KITE, CoRL 2023](https://proceedings.mlr.press/v229/sundaresan23a.html)。

语言层次可在相关工作中对照 [RT-H](https://rt-hierarchy.github.io/)，三维关系约束可对照 [ReKep](https://rekep-robot.github.io/)。二者的语言运动或约束优化系统与本文不完全同构，资源有限时不强求全系统复现。

截至评阅日，已有 [3D HAMSTER](https://arxiv.org/abs/2606.31329) 专门讨论二维引导与三维控制之间的几何问题。因此建议补充三维点/轨迹的相关工作，并完成一个同骨干三维点比较；不应仅以“设计选择”跳过所有三维参照。

[JUICER](https://arxiv.org/abs/2404.03729) 与 [AutoMate](https://arxiv.org/abs/2407.08028) 的训练数据和控制系统不同。若加入 published score，只能作为背景，不可混进协议匹配的主表声称超越。

### E8：多几何留出与真正的规模实验

原 AutoMate 有 100 个装配身份，当前 99/1 划分不适合强泛化结论。建议另立新协议，例如 70 train / 10 validation / 20 test，按几何族分层并尽量避免近重复几何跨 split；全部接口重新训练，已训练过的 99 个身份不能事后挑一部分直接当 OOD。

OOD 每几何每种子建议 20 次。3 种子、20 个 test 几何对应每接口 1,200 次评估。报告几何宏平均、几何间分布和最困难四分位数；bootstrap 保留训练种子和几何体聚类。

如保留 scalability 表述，使用训练身份 N={10,30,50,70} 的嵌套集合。区分两种预算：每身份示范数固定，测数据和身份共同增长；总示范数固定，测固定预算下的身份扩展。主文可选一条主轴，另一条作附录，避免全组合失控。

FB-only 与 FB+AutoMate 必须同 seed、同验证规则、同评估 reset 比较。至少加入 FB-only 等总更新数的计算量对照，以及联合训练中保持 FB 暴露次数的采样对照；否则新增数据、训练步数和旧任务暴露次数会混在一起。

AutoMate 标签只有 Insert 时，二类 mode 基本为常量，不能据它的好成绩推断 TAGPoint 的语义压缩优势。要验证跨技能扩展，需要多个技能或在 FB 中另做组合留出。

## 8. 条件性 P0：如果坚持真机结论，必须补齐定量验证

若论文主张仅限仿真，真机定量是增强证据；若保留 real-world transfer/validation 作为贡献，它就必须在投稿前完成。

第一阶段比较正确引导下的接口，第二阶段将同一套控制与深度处理替换为实际 VLM 引导。核心方法至少 RGB-D、GP、TAGPoint、GP+skill；grasp 若在仿真中是最高值，建议纳入第二阶段以避免只选择有利比较对象。

先以每方法每任务 10 次验证硬件和协议，正式评估建议至少 30 次、跨天分块；若目标差异较小需进一步增加样本。优先覆盖 one-leg 和 lamp 等不同接触要求的两至三个任务。这些规模用于建立可解释证据，不保证检测 1–5 个百分点的差异。

记录并公布：初始位姿集合与容差、方法运行顺序随机化、统一 timeout、自动成功判据、人工接触/重置是否计失败、失败恢复规则、VLM invalid output 处理、总试验数、整任务及分阶段成功、典型失败视频。每种子若仅用一个策略 checkpoint，要明确真机结论不包含训练方差。

若继续使用 Prompt Depth Anything，先让所有方法共享同一版本；要声称它有独立贡献时，另做原始 RealSense 深度与精化深度的配对消融。不要把引导、深度模型和控制器同时变动后归因为接口效果。

## 9. 可直接执行的工程拆分

以下是建议新增的模块职责，属于实现规格，不是已经存在或已经运行的代码。

| 模块 | 输入/输出 | 核心要求 |
|---|---|---|
| freeze_dataset | demonstrations → immutable manifest | trajectory split、原始样本哈希、标注版本、每任务示范数 |
| build_reset_bank | task seeds → reset/snapshot files | 完整状态恢复与短程一致性检查 |
| render_interface | raw observations + Guidance → policy inputs | 相同几何和图像变换，支持 GP/颜色/向量/pose/path，禁止偷偷读取 GT |
| guidance_provider | current observation + instruction → Guidance | 显式标记 oracle/predicted/noisy 和 stage 来源 |
| paired_evaluator | policies + reset bank + condition | 相同随机数规则，固定 timeout，不按成功率重试 |
| analyze_results | episode/step logs → tables/figures | 保留配对、任务、seed、geometry 层次，不重复计数缓存 query |

建议 Guidance schema：target_uv、camera_id、skill5、mode2、rotation6d（如适用）、source、timestamp、query_id、valid、stage_source。front 和 wrist 分开存 camera_id 与坐标，避免跨相机混用。

建议 run manifest：run_id、code_commit、dataset_hash、split_hash、annotation_hash、interface、train_seed、checkpoint_sha、train_steps、obs/action config、VLM model/version、VLM weights、prompt_hash、query_interval、reset_bank_hash、eval seed、success predicate version、exclusion_reason。

episode log：run_id、task、geometry、reset_id、seed、success、failure_stage、timeout/intervention/invalid_output、duration_steps、VLM query_count、latency。

step/query log：query_id、observation_timestamp、generation_timestamp、action_timestamp、cached_age、GT/predicted target、GT/predicted skill、mode、rotation error、point visibility、executed action、stage transition source。

评估流程伪代码：

```text
for each task and predefined reset:
    for each training seed:
        for each interface checkpoint:
            restore the same reset and controlled random state
            while not success/failure/timeout:
                observe current state
                update stage using the declared stage provider
                query guidance only when the fixed schedule requires it
                if counterfactual experiment:
                    replace only the specified guidance field
                render both views with the declared geometry pipeline
                predict action chunk and execute the first 8 actions
                log guidance source, cache age, actions, and transitions
            save one episode outcome regardless of success
```

VLM 输出缓存跨控制步重复使用时，query-level error、held-guidance error 和 episode-level performance 必须分开统计。Table 5 目前数十万 valid pairs 不等于数十万次独立 VLM 调用；应补充真实 query 数、刷新时误差、缓存后误差和 episode 等权统计。

## 10. 图表和具体写作修改

- Figure 1：建议用可编辑矢量图重画，显示实际输入、VLM 输出 schema、front/wrist 投影、stage 来源、缓存与 action chunk；把“误差传播对比”缩成待检验动机，避免视觉上暗示已证明的理论。
- Table 1：将 colored grasp 明确作为 grasp family 变体或加行；把 6D continuous rotation representation 与 6-DoF grasp pose 分清。当前二维位置加六维旋转编码不应被读者理解为完整的六自由度 metric pose 输入。
- Figure 2：纳入 TAGPoint；固定入口结果与自然轨迹 C/R 分图；每 stage 显示分母或附原始计数。说明 Insert 缺失或无覆盖原因。
- Figure 3：显示多训练种子和区间；0/3/6/12 mm 在当前线性轴左侧挤在一起，可使用明确标注的 symlog 或将低噪声区域放大；Shuffle 保持独立分类面板。
- Figure 4：增加固定轨迹重评分 null baseline，以及同状态动作响应；仅有扰动目标距离不宜占据机制证据的核心位置。
- Table 3：统一样本数与策略来源，改成 oracle/predicted 成对报告；1 次差异不宜依靠粗体/下划线营造稳定胜者印象。
- Table 4：同一行在 FB 与 AutoMate 使用同一 checkpoint；FB delta 对应同 seed、同数据协议的 FB-only 参照。AutoMate ID 的 12 次/几何样本较少，报告分布。
- Figure 5：未完成定量试验时改为“Qualitative physical interface demonstration”，移附录；不把未来实验计划写成已经完成的 Results。
- Figure 8–12：当前单图密度高、文字很小，重复近似曲线多。保留能回答特定问题的 stage 图，其余作为机器可读补充。Total Error 需给明确定义、单位归一化与权重，不直接混加厘米和角度。
- 修正 p.5 “Appendix 6”、p.8 “Appendix 8”等交叉引用，应指 Figure 6/8 in Appendix E；统一 TAGPoint/colored GP、grasp/grasp-part、GP+skill/GP+Skill 命名。
- Appendix A–D 补齐示范数、训练优化器与学习率、batch size、训练步数、VLM 底座与微调方式、数据切分、无效输出处理、完整 stage 列表和硬件信息。“will be supplied”不能替代可复现设置。

## 11. 推荐执行顺序与资源控制

第一步：E0 审计、冻结数据和 reset；同步改写论文主张。先修复不可解释的比较，避免新实验继续累积协议差异。

第二步：E1 两个关键编码控制与 E3/E4 snapshot 实验。前者决定 TAGPoint 的创新归属，后者验证论文的机制论证，可复用相同状态库。

第三步：E2 oracle/predicted 交叉替换，并完成同 checkpoint 主表。这是“接口最优取决于上游错误”的直接证据。

第四步：语言、HAMSTER-style path、三维点三个高价值比较；资源不足时先做 oracle 筛选，再对最相关候选做真实 VLM 测试。所有基线须给合理调参预算，不将仓促移植的低分作为方法优越性的证据。

第五步：根据最终主张选择 E5/E6、E8 与定量真机。若真机无法完成，就收缩真实迁移主张；若多几何评估无法完成，就将 Table 4 称作已见任务集扩展，取消 OOD 排名。

以原八条件 E0 加 E1 两个新条件为完整核心实验，共 30 次动作专家训练、9,000 次 oracle rollout；predicted、交叉替换、stage 和噪声实验另计。先测一次代表性训练与 rollout 实际成本，再估算 GPU 小时、VLM 调用和机器人时间。不要把增加同 checkpoint 评估次数当作增加训练种子。

结果不论是否有利于 TAGPoint，都应允许改变结论：颜色无优势则归因语义粒度；点响应弱则讨论冗余引导或不一致输入抑制；姿态更好则给任务条件下的选择原则；统一协议后差异小则报告边界与负结果。这样最终论文才是围绕可验证问题建立的研究。
