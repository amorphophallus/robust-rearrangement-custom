# 多任务 Condition 对比实验 — 评估结果汇总

**日期**: 2026-06-15
**任务**: `one_leg + round_table + lamp` (多任务)
**模型**: DiT (diffusion), 3×100 trajectories
**Project**: `multi-task-rgbd-skill-low-0610`
**Eval Settings**: `N_ENVS=3, N_ROLLOUTS=36`, image-based

---

## 1. 总览

| # | 实验 | RUN_ID | one_leg | round_table | lamp | Overall |
|---|---------|--------|---------|-------------|------|---------|
| 1 | rgbd+only skill | icy-vortex-9 | 86.11% (31/36) | 55.56% (20/36) | 30.56% (11/36) | 57.41% (62/108) |
| 2 | rgbd | clear-water-12 | 0.00% (0/36) | 41.67% (15/36) | 0.00% (0/36) | 13.89% (15/108) |
| 3 | rgbd+colored GP| absurd-voice-2 | **91.67% (33/36)** | 50.00% (18/36) | 41.67% (15/36) | 61.11% (66/108) |
| 4 | rgbd+GP | rare-monkey-4 | 83.33% (30/36) | 41.67% (15/36) | 33.33% (12/36) | 52.78% (57/108) |
| 4b | rgbd+GP | autumn-dust-13 | 77.78% (28/36) | 36.11% (13/36) | 41.67% (15/36) | 51.85% (56/108) |
| 5 | rgbd+GP+skill | fresh-tree-11 | 83.33% (30/36) | 50.00% (18/36) | **55.56% (20/36)** | **62.96% (68/108)** |
| 6 | rgb | true-firefly-8 | 0.00% (0/36) | 16.67% (6/36) | 0.00% (0/36) | 5.56% (6/108) |

> 注：Notion 实验记录表格中 exp4 的 run name 登录错误，autumn-dust-13 为正确 run name（训练至 3000 epoch，plain GP）。absurd-voice-2 仅训至 2000 epoch 且训练配置为 colored GP。经检查 rare-monkey-4（标签 "colored GP"）和 absurd-voice-2（标签 "plain GP"）的训练配置疑似对调：rare-monkey-4 的 `annotate_guidance_point_colored=false`，absurd-voice-2 的 `annotate_guidance_point_colored=true`。上表已按实际训练配置标注。

结论：
1. gp, skill one-hot 两种引导信息都能让多任务模型有区分不同任务的能力

## 2. 对比


### 2.1 round_table 多任务 vs 单任务

多任务数据 3*100 条，单任务数据 200 条。

结论：
1. 引导点在单任务时会导致模型过拟合，而多任务不存在过拟合问题
2. 多任务成功率还是比不过单任务的

### 2.2 3000 epoch vs 2000 epoch

统一使用 latest checkpoint 在 epoch 2000 和 epoch 3000 的 eval 结果对比（均使用 `N_ENVS=3, N_ROLLOUTS=36`）：

| # | 实验 | RUN_ID | 2000 epoch | 3000 epoch | Δ |
|---|------|--------|-----------|-----------|-----|
| 1 | rgbd+only skill | icy-vortex-9 | 45.37% (49/108) | 57.41% (62/108) | -12.04 |
| 2 | rgbd | clear-water-12 | 14.81% (16/108) | 13.89% (15/108) | +0.92 |
| 3 | rgbd+colored GP | rare-monkey-4 | 53.70% (58/108) | 52.78% (57/108) | +0.92 |
| 4 | rgbd+GP | autumn-dust-13 | 55.56% (60/108) | 51.85% (56/108) | +3.71 |
| 5 | rgbd+GP+skill | fresh-tree-11 | 52.78% (57/108) | 62.96% (68/108) | -10.18 |
| 6 | rgb | true-firefly-8 | 10.19% (11/108) | 5.56% (6/108) | +4.63 |

> autumn-dust-13 为 exp4 正选 run name。absurd-voice-2 仅训至 2000 epoch，无 3000 epoch checkpoint，不参与此对比。

结论：
1. **GP+skill 和 only skill 在 3000 epoch 显著优于 2000 epoch**（+10~12%），说明 skill-based 方法需要更长的训练。
2. **纯 GP 和 rgb/rgbd 在 epoch 间差异不大**（±5% 以内），2000 epoch 基本收敛。
3. **absurd-voice-2 在 2000 epoch 表现异常好 (61.11%)** 可能源于其 colored GP 训练数据，而非 epoch 差异。


### 2.3 sr vs test loss/action mse error


## 3. rollout 视频内容




