# Train Init State Eval — 实验总结

## 目标

测试 rgbd+colored GP 模型（lively-wind-5）在**训练集初始状态**上的成功率，判断 one_leg 0% 是过拟合（训练集能成功、泛化不到测试分布）还是根本不会（训练初始状态也失败）。

## 方法

1. 从服务器 236 上 `rgbd-skill-colored-1.lmdb` 提取 296 个训练 episode 的第一帧初始状态（parts_poses + robot joint positions），per-task 分布：lamp 97 / one_leg 102 / round_table 97
2. 修改 eval pipeline（`rollout.py`, `evaluate_model.py`, `furniture_rl_sim_env.py`）支持 `--init-state-file` 注入初始状态，通过 `env.reset_to()` 绕过环境随机重置
3. 先跑 lamp（最成功的 task，random init 38.89%）验证流程

## 结果

Lamp 训练初始状态上跑了 28/97 rollouts：**0/28 (0.0%)**

对比：random init lamp = **38.89% (14/36)**

## 结论

模型在训练集初始状态上表现**差于**随机初始状态——不是过拟合，更像反过来了。

### 可能原因（按优先级）

1. **LMDB 第一帧不是真正的"初始状态"**：expert demo 的第一帧物件可能已经离开起始位置，part_poses 记录了已经被移动过的状态，模型看到的是一个"半途"场景而非初始配置
2. **坐标系统不匹配**：LMDB 中的 `parts_poses` 可能是 AprilTag 坐标系，但 eval env 期望 simulator 坐标系，`april_coord_to_sim_coord` 转换可能有问题
3. **default joint positions 不匹配**：提取脚本用了默认 Franka 关节位置而非训练数据中的真实关节位置，导致机器人起点与 training 不一致
4. **训练数据中 expert 从困难初始配置开始**：expert demo 初始 part poses 本身就比 random init 更难

## 做的代码修改

| 文件 | 修改 |
|------|------|
| `scripts/extract_train_init_states.py` | 新建：从 LMDB 提取初始状态 |
| `src/eval/evaluate_model.py` | +`--init-state-file` CLI 参数，加载并过滤 per-task init states，修复 numpy.object_ 类型转换 |
| `src/eval/rollout.py` | `rollout()` / `calculate_success_rate()` 支持 `init_states` 参数，`reset_to` 后加 `refresh` |
| `furniture-bench/.../furniture_rl_sim_env.py` | 修复 `_reset_parts` 中 obstacle 全局索引→局部索引；修复 `reset_to` 缺少状态传播到 GPU |

## 下一步建议

1. 验证 LMDB 第一帧是否真的是"初始状态"——对比随机 reset 后的 part_poses 与 training 第一帧
2. 如果第一帧是 expert 中途状态 → 换用实际的初始 reset 配置
3. 如果坐标/关节位置不匹配 → 修正提取逻辑
