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
| 1 | rgbd+only skill | icy-vortex-9 | 86.11% (31/36) | **55.56% (20/36)** | 30.56% (11/36) | 57.41% (62/108) |
| 2 | rgbd | clear-water-12 | 0.00% (0/36) | 41.67% (15/36) | 0.00% (0/36) | 13.89% (15/108) |
| 3 | rgbd+colored GP | absurd-voice-2 | **91.67% (33/36)** | 27.78% (10/36) | 38.89% (14/36) | 52.78% (57/108) |
| 4a | rgbd+GP | rare-monkey-4 | 83.33% (30/36) | 41.67% (15/36) | 33.33% (12/36) | 52.78% (57/108) |
| 4b | rgbd+GP | autumn-dust-13 | 77.78% (28/36) | 36.11% (13/36) | 41.67% (15/36) | 51.85% (56/108) |
| 5 | rgbd+GP+skill | fresh-tree-11 | 83.33% (30/36) | 50.00% (18/36) | **55.56% (20/36)** | **62.96% (68/108)** |
| 6 | rgb | true-firefly-8 | 0.00% (0/36) | 16.67% (6/36) | 0.00% (0/36) | 5.56% (6/108) |


结论：
1. gp, skill one-hot 两种引导信息都能让多任务模型有区分不同任务的能力。
2. 多任务泛化性 rgbd+skill+gp > rgbd+skill > rgbd+gp=rgbd+colored gp。当前任务多样性有限，skill 就足够泛化，所以提升比 gp 多。想让 gp > skill 的两个方向：
    1. 提升任务泛化难度，med rand 和 permutation 肯定会有帮助。
    2. gp+grasp，提供 rot 信息。
3. rgbd 和 rgbd+skill 加 gp 都能获得提升说明 gp 确实能提供更多信息，促进泛化。
4. colored 并没有如预期地提供类似 one-hot skill 的信息。可能的原因：
    1. 信号大小：2px 点占图像的 0.008%。ResNet 的 32× 下采样后，这个点变成了 ~0.06 feature-map-pixel。颜色差异（红 vs 黄）在 deep features 里几乎不可区分。
    2. 信息量 vs 通路带宽：skill one-hot 贡献 5 个独占维度（531 维 conditioning 中的 5 维），而 colored GP 的 1 bit 信息和整个场景（物体、机械臂、桌面）"挤"在同一个 512-dim visual feature 空间里竞争。
    - 改成红色和蓝色的点来区分可能更好，单独放两个通道。但是信号大小的问题无法解决。
5. 为什么 rgbd 和 rgb 过拟合到 round_table 而不是最简单的 one_leg？



## 2. 对比


### 2.1 round_table 多任务 vs 单任务

多任务数据 3×100 条，单任务数据 200 条（单任务数据来源：[guidance point ablation](https://app.notion.com/p/guidance-point-ablation-34f6aab8287c802f97c2f0c57337f4ad)）。

| 实验条件 | 单任务 round_table | 多任务 round_table (3000 epoch) | Δ |
|---------|-------------------|-------------------------------|-----|
| rgbd | 41.67% (15/36) | 41.67% (15/36) (clear-water-12) | 0 |
| rgbd+GP | 33.33% (12/36) | 36.11-41.67% (autumn-dust-13 / rare-monkey-4) | **+3~8** |
| rgbd+GP+skill | **63.89% (23/36)** | 50.00% (18/36) (fresh-tree-11) | -13.89 |
| rgb | 36.11% (13/36) | 16.67% (6/36) (true-firefly-8) | -19.44 |
| rgbd+skill | 52.78% (19/36) | **55.56% (20/36) (icy-vortex-9)** | +2.78 |
| rgbd+colored GP | 50.00% (18/36) | 27.78% (10/36) (absurd-voice-2) | -22.22 |

结论：


### 2.2 3000 epoch vs 2000 epoch

统一使用 latest checkpoint 在 epoch 2000 和 epoch 3000 的 eval 结果对比（均使用 `N_ENVS=3, N_ROLLOUTS=36`）：

| # | 实验 | RUN_ID | 2000 epoch | 3000 epoch | Δ |
|---|------|--------|-----------|-----------|-----|
| 1 | rgbd+only skill | icy-vortex-9 | 45.37% (49/108) | 57.41% (62/108) | +12.04 |
| 2 | rgbd | clear-water-12 | 14.81% (16/108) | 13.89% (15/108) | -0.92 |
| 3 | rgbd+colored GP | absurd-voice-2 | 51.85% (56/108) | 52.78% (57/108) | +0.93 |
| 4a | rgbd+GP | rare-monkey-4 | 53.70% (58/108) | 52.78% (57/108) | -0.92 |
| 4b | rgbd+GP | autumn-dust-13 | 55.56% (60/108) | 51.85% (56/108) | -3.71 |
| 5 | rgbd+GP+skill | fresh-tree-11 | 52.78% (57/108) | 62.96% (68/108) | +10.18 |
| 6 | rgb | true-firefly-8 | 10.19% (11/108) | 5.56% (6/108) | -4.63 |



结论：
1. **GP+skill 和 only skill 在 3000 epoch 显著优于 2000 epoch**（+10~12%），说明 skill-based 方法需要更长的训练。
2. **纯 GP 和 rgb/rgbd 在 epoch 间差异不大**（±5% 以内），2000 epoch 基本收敛。
3. **absurd-voice-2 (colored GP) 在 2000 epoch 为 51.85%**，用错 eval flag (colored=false) 时为 50.00%（复测确认；首次 61.11% 为偶然波动），colored flag 差异在噪声范围内。


### 2.3 sr vs test loss/action mse error

wandb summary 中最后一次 eval 记录的 test loss 和 test action mse error（注：wandb test 数据对应最后一次 eval checkpoint，不一定是 3000 epoch；absurd-voice-2 对应 2000 epoch 训练）：

| # | RUN_ID | 实验 | SR (对应 epoch) | test_bc_loss | val_action_mse |
|---|--------|------|----------------|-------------|----------------|
| 1 | icy-vortex-9 | rgbd+skill | 57.41% (3000) | 0.0920 | 0.0266 |
| 2 | clear-water-12 | rgbd | 13.89% (3000) | 0.1472 | 0.0189 |
| 3 | absurd-voice-2 | colored GP | 50.00% (2000, retest) | 0.1186 | 0.0471 |
| 4a | rare-monkey-4 | GP | 52.78% (3000) | 0.1273 | 0.0456 |
| 4b | autumn-dust-13 | GP | 51.85% (3000) | 0.0879 | 0.0269 |
| 5 | fresh-tree-11 | GP+skill | 62.96% (3000) | 0.1009 | 0.0322 |
| 6 | true-firefly-8 | rgb | 5.56% (3000) | 0.2205 | 0.0390 |

![SR vs Metrics](./sr_vs_metrics.png)


## 3. rollout 视频内容


### 3.1 对比 rgbd+gp 和 rgbd+skill 的动作模式




### 3.2 对比 rgbd+colored gp 和 rgbd+gp 的动作模式


## 附录: 实验数据汇总

统一条件：`N_ENVS=3, N_ROLLOUTS=36`, image-based, checkpoint=latest

### A.1 3000 epoch eval

| # | RUN_ID | 训练配置 | eval GP colored | one_leg | round_table | lamp | Overall |
|---|--------|---------|-----------------|---------|-------------|------|---------|
| 1 | icy-vortex-9 | rgbd+skill | N/A | 86.11% (31/36) | 55.56% (20/36) | 30.56% (11/36) | 57.41% (62/108) |
| 2 | clear-water-12 | rgbd | N/A | 0.00% (0/36) | 41.67% (15/36) | 0.00% (0/36) | 13.89% (15/108) |
| 3 | absurd-voice-2 | colored GP | ❌ false (错) | 94.44% (34/36) | 33.33% (12/36) | 33.33% (12/36) | 53.70% (58/108) |
| 3 | absurd-voice-2 | colored GP | ✅ true (对) | **91.67% (33/36)** | 27.78% (10/36) | 38.89% (14/36) | 52.78% (57/108) |
| 4a | rare-monkey-4 | GP | ❌ true (错) | 83.33% (30/36) | 41.67% (15/36) | 33.33% (12/36) | 52.78% (57/108) |
| 4a | rare-monkey-4 | GP | ✅ false (对) | 83.33% (30/36) | 33.33% (12/36) | 27.78% (10/36) | 48.15% (52/108) |
| 4b | autumn-dust-13 | GP | ✅ false | 77.78% (28/36) | 36.11% (13/36) | 41.67% (15/36) | 51.85% (56/108) |
| 5 | fresh-tree-11 | GP+skill | N/A | 83.33% (30/36) | 50.00% (18/36) | **55.56% (20/36)** | **62.96% (68/108)** |
| 6 | true-firefly-8 | rgb | N/A | 0.00% (0/36) | 16.67% (6/36) | 0.00% (0/36) | 5.56% (6/108) |

### A.2 2000 epoch eval

| # | RUN_ID | 训练配置 | eval GP colored | one_leg | round_table | lamp | Overall |
|---|--------|---------|-----------------|---------|-------------|------|---------|
| 1 | icy-vortex-9 | rgbd+skill | N/A | 80.56% (29/36) | 33.33% (12/36) | 22.22% (8/36) | 45.37% (49/108) |
| 2 | clear-water-12 | rgbd | N/A | 0.00% (0/36) | 41.67% (15/36) | 2.78% (1/36) | 14.81% (16/108) |
| 3 | absurd-voice-2 | colored GP | ❌ false (错) | **91.67% (33/36)** | 50.00% (18/36) | 41.67% (15/36) | 61.11% (66/108) |
| 3 | absurd-voice-2 | colored GP | ✅ true (对) | 86.11% (31/36) | 33.33% (12/36) | 36.11% (13/36) | 51.85% (56/108) |
| 4a | rare-monkey-4 | GP | ❌ true (错) | 83.33% (30/36) | 41.67% (15/36) | 36.11% (13/36) | 53.70% (58/108) |
| 4a | rare-monkey-4 | GP | ✅ false (对) | 88.89% (32/36) | 38.89% (14/36) | 33.33% (12/36) | 53.70% (58/108) |
| 4b | autumn-dust-13 | GP | ✅ false | 80.56% (29/36) | 44.44% (16/36) | 41.67% (15/36) | 55.56% (60/108) |
| 5 | fresh-tree-11 | GP+skill | N/A | 86.11% (31/36) | 38.89% (14/36) | 33.33% (12/36) | 52.78% (57/108) |
| 6 | true-firefly-8 | rgb | N/A | 0.00% (0/36) | 30.56% (11/36) | 0.00% (0/36) | 10.19% (11/108) |

> eval GP colored 标记：✅ = 与训练配置一致，❌ = 与训练配置不一致

### A.3 wandb test metrics

wandb summary 最后一次 eval 记录的值（不一定是 3000 epoch 末尾；absurd-voice-2 来自 2000 epoch 训练）：

| RUN_ID | test_bc_loss | val_action_mse | val_mse_pos | val_mse_rot | val_mse_width |
|--------|-------------|----------------|-------------|-------------|---------------|
| icy-vortex-9 | 0.0920 | 0.0266 | 0.00016 | 0.00761 | 0.2196 |
| clear-water-12 | 0.1472 | 0.0189 | 0.00018 | 0.00643 | 0.1503 |
| absurd-voice-2 | 0.1186 | 0.0471 | 0.00014 | 0.01758 | 0.3653 |
| rare-monkey-4 | 0.1273 | 0.0456 | 0.00025 | 0.00423 | 0.4295 |
| autumn-dust-13 | 0.0879 | 0.0269 | 0.00011 | 0.00581 | 0.2339 |
| fresh-tree-11 | 0.1009 | 0.0322 | 0.00013 | 0.00882 | 0.2687 |
| true-firefly-8 | 0.2205 | 0.0390 | 0.00018 | 0.02559 | 0.2363 |

