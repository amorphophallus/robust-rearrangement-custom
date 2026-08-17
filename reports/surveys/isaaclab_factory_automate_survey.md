# AutoMate 调研

## 最新维护入口

- Isaac Lab 环境总览中的 AutoMate 小节：
  <https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#automate>
- Isaac Lab 当前维护主线中的 `automate` 源码目录：
  <https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/automate>
- AutoMate 项目页：
  <https://bingjietang718.github.io/automate/>

这说明 AutoMate 现在的维护位置是 `Isaac Lab main`，不是已经归档的 `IsaacGymEnvs automate` 分支。

## 训练大致流程

官方当前流程是两阶段：

1. 先跑 `Isaac-AutoMate-Disassembly-Direct-v0` 收集 demonstration。
   文档明确说这是纯脚本过程，不是学习策略：物体初始已经处于装配状态，低层控制器先把手移动到预定义 `pre_grasp / grasp` 位姿，闭爪后把 plug 从 socket 中抬出，再移动到随机位姿并记录轨迹。

2. 再跑 `Isaac-AutoMate-Assembly-Direct-v0` 训练单个 assembly ID 的 specialist policy。
   官方描述的核心训练方法是：
   `PPO + imitation reward + dynamic time warping (DTW) + sampling-based curriculum`。
   其中 imitation reward 来自前一步收集到的 disassembly 轨迹；assembly 训练前需要先准备好这些轨迹。

官方文档里的当前入口是：

- 采集拆卸轨迹：
  `python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_disassembly_w_id.py --assembly_id=ASSEMBLY_ID --disassembly_dir=DISASSEMBLY_DIR`
- 训练装配策略：
  `python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --train`
- 评估已有 checkpoint：
  `python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --checkpoint=CHECKPOINT --log_eval`

Isaac Lab 文档还明确写了：

- 训练得到的 checkpoint 会自动保存到 `logs/rl_games/Assembly/test`
- 评估结果会保存为 `evaluation_{ASSEMBLY_ID}.h5`

## 预期结果

项目页和论文给出的公开结果是：

- specialist policies 对 `80` 个 assemblies 分别达到 `≈80%+` 的仿真成功率
- 一个 generalist policy 在 `20` 个 assemblies 上达到 `80%+` 的仿真成功率

这个结论来自 AutoMate 项目页和论文摘要，属于作者公开报告的结果，不是我本地复现得到的结果。

## 随机化程度

我对过当前代码后，AutoMate 的随机化主要集中在以下几类：

- socket 初始位置随机化：
  代码里显式对 socket 的 `xy` 位置加噪声
- plug 初始位姿随机化：
  assembly 训练时会对 plug 的初始 `xyz` 和旋转加随机扰动
- 机械臂初始状态随机化：
  reset 时会重新设定 Franka 初始 DOF，再移动到预定义 grasp pose
- curriculum 随机化：
  用 curriculum 控制 plug 初始插入深度 / 脱离深度范围，训练中会随着成功率调整难度
- 观测噪声：
  代码里对 socket 观测和目标相关量加入噪声

从任务范式上看，AutoMate 不是桌面 `pick-and-place`。它没有“先从桌面抓起物体再插入”的阶段，而是默认使用预定义 grasp：reset 后先把手移动到 `pre_grasp / grasp`，闭爪后再进入拆卸或装配过程。
