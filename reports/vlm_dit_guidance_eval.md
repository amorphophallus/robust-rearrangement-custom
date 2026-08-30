# VLM + DiT guidance point 评测报告

## 验证链路（旧数据 / smoke / matched diagnostics / formal）

### 旧 324-rollout：invalid

- 判定：`invalid`。旧 324 个 VLM rollout 和旧 scripted-GT 对照均缺少 --save-depth-image；当前 evaluate_model.py 会因此不把 depth_image1/depth_image2 加入 RGBD policy observation，所以旧结果不能用于判断 checkpoint、VLM 或 condition 优劣。
- 旧数据只保留作故障溯源，不 resume、不拼接、不用于 condition 排名。缺少 `--save-depth-image` 会同时令 RGBD policy observation 缺少 depth；此外旧 one_leg 使用 700 steps，旧 rgbd+GP checkpoint 也与本轮固定 checkpoint 不同。

### ckpt_new 300-sample grounding gate

| Scope | n | Parse % | Valid point | Skill acc. % | Mean px | RMSE px | Median px | P90 px | Bias px | Spread | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 300 | 99.7 | 300/300 | 71.0 | 10.97 | 22.99 | 4.00 | 37.22 | 1.44 | 1.008 | 0.579 |
| one_leg | 100 | 99.0 | 100/100 | 70.0 | 8.17 | 15.63 | 3.80 | 15.79 | 1.36 | 1.121 | 0.557 |
| round_table | 100 | 100.0 | 100/100 | 75.0 | 12.31 | 25.30 | 3.16 | 42.26 | 3.00 | 0.986 | 0.221 |
| lamp | 100 | 100.0 | 100/100 | 68.0 | 12.42 | 26.49 | 4.30 | 31.68 | 0.58 | 0.958 | 0.559 |

Gate 结果：有效点 `300/300`，即该 300 样本中没有 null point；完整输出 parse 成功率 `99.7%`。整体 spread `1.008`、R² `0.579`，没有再次出现预测整体收缩到全局均值的旧式 collapse；task/skill 局部质量仍须结合下方真实 rollout 的 fresh-query 表判断。
原始诊断：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/grounding_300_current_images_cross_skill.json`；耗时 `485.30` s。

### 新 27-rollout smoke

本轮根据用户明确授权跳过 27-rollout smoke；manifest gate mode=`explicit_user_approved_bypass`，approval=`User requested full experiment after ckpt_new matched 300-sample diagnostic and 5/6 one_leg gate on 2026-08-22`。

Smoke gate：`pending`；manifest：`—`。

### Matched scripted diagnostics

| Condition/task | Success | Final skill | Failed skill | Tracking count | Tracking pos/rot mean |
| --- | --- | --- | --- | --- | --- |
| rgbd+GP / one_leg / scripted-GT | 3/3 | leg-top-screw | — | 18 | 6.83 cm / 9.98 deg |

Summary：`/data/hy/robust-rearrangement/logs/vlm_dit_depthfix_scripted_diag_20260817/summaries/rgbd_gp__one_leg.json`。

### 新正式 324-rollout

当前主 manifest 是 formal：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite/manifest.json`。

## 1. 实验状态

- 已完成 task-level 实验：`9/9`。
- 已完成 rollout：`324/324`。
- 成功 rollout：`181/324`。
- VLM：`http://10.71.106.240:8000`；revision：`9d36062d461e6d07f78d6148bea8039e1e019f92`。
- 原始 manifest：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite/manifest.json`。
- 阶段：`formal`；设计：3 个 condition × 3 个 task × 每格 36 rollout，共 324 rollout；每批 3 个并行环境。
- task 截止步数：one_leg=1000，round_table=1000，lamp=1000；randomness=low。
- 每个有效控制 step 的 GT/VLM 点对、每个 n0--n4 档位使用 `200` 个 3D Monte Carlo 投影样本。

## 2. Success rate

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 86.1% (31/36) | 44.4% (16/36) | 44.4% (16/36) | 58.3% (63/108) |
| rgbd+colored GP | 86.1% (31/36) | 41.7% (15/36) | 50.0% (18/36) | 59.3% (64/108) |
| rgbd+GP+skill | 83.3% (30/36) | 27.8% (10/36) | 38.9% (14/36) | 50.0% (54/108) |

### 2.1 每格成功率与 Wilson 95% CI

| Condition | Task | Success | Rate [Wilson 95% CI] |
| --- | --- | --- | --- |
| rgbd+GP | one_leg | 31/36 | 86.1% [71.3%, 93.9%] |
| rgbd+GP | round_table | 16/36 | 44.4% [29.5%, 60.4%] |
| rgbd+GP | lamp | 16/36 | 44.4% [29.5%, 60.4%] |
| rgbd+colored GP | one_leg | 31/36 | 86.1% [71.3%, 93.9%] |
| rgbd+colored GP | round_table | 15/36 | 41.7% [27.1%, 57.8%] |
| rgbd+colored GP | lamp | 18/36 | 50.0% [34.5%, 65.5%] |
| rgbd+GP+skill | one_leg | 30/36 | 83.3% [68.1%, 92.1%] |
| rgbd+GP+skill | round_table | 10/36 | 27.8% [15.8%, 44.0%] |
| rgbd+GP+skill | lamp | 14/36 | 38.9% [24.8%, 55.1%] |

## 3. Tracking error（clean GT pose）

每格为 `position cm / orientation deg / total (n)`；`total = pos_m / 0.01 + ori_deg / 5`，越低越好。VLM 只替换 policy 的 skill/2D point，shadow 自动机提供 clean guidance pose 作为共同 tracking target。

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 2.46/15.84/5.63 (n=207) | 3.05/32.75/9.60 (n=237) | 6.52/42.48/15.02 (n=172) | 3.82/29.79/9.78 (n=616) |
| rgbd+colored GP | 2.44/17.42/5.93 (n=207) | 3.22/30.88/9.40 (n=246) | 7.06/43.96/15.85 (n=171) | 4.02/30.00/10.02 (n=624) |
| rgbd+GP+skill | 2.66/14.96/5.65 (n=204) | 3.77/37.31/11.23 (n=218) | 6.62/41.63/14.95 (n=178) | 4.24/30.99/10.44 (n=600) |

### 3.1 每格 position / rotation 分布

| Condition | Task | n | Position mean/median/p90 cm | Rotation mean/median/p90 deg |
| --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 207 | 2.46/1.20/3.47 | 15.84/11.67/17.43 |
| rgbd+GP | round_table | 237 | 3.05/2.05/7.18 | 32.75/13.41/72.17 |
| rgbd+GP | lamp | 172 | 6.52/3.11/20.52 | 42.48/28.75/82.49 |
| rgbd+colored GP | one_leg | 207 | 2.44/1.37/3.61 | 17.42/12.06/19.97 |
| rgbd+colored GP | round_table | 246 | 3.22/2.02/7.28 | 30.88/13.01/72.06 |
| rgbd+colored GP | lamp | 171 | 7.06/3.03/22.63 | 43.96/36.26/87.28 |
| rgbd+GP+skill | one_leg | 204 | 2.66/1.35/3.63 | 14.96/11.75/17.95 |
| rgbd+GP+skill | round_table | 218 | 3.77/2.30/7.44 | 37.31/21.99/83.51 |
| rgbd+GP+skill | lamp | 178 | 6.62/2.92/20.39 | 41.63/31.10/85.70 |

### 3.2 Per-skill tracking

| Condition | Task | Skill | n | Position mean/median/p90 cm | Rotation mean/median/p90 deg |
| --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | leg-top-insert | 33 | 0.56/0.49/0.63 | 5.64/5.19/8.08 |
| rgbd+GP | one_leg | leg-top-pick | 35 | 1.09/1.10/1.45 | 15.27/13.46/14.83 |
| rgbd+GP | one_leg | leg-top-place | 35 | 0.91/0.80/1.45 | 12.14/11.31/16.05 |
| rgbd+GP | one_leg | leg-top-screw | 32 | 6.52/1.97/27.66 | 34.10/7.03/114.44 |
| rgbd+GP | one_leg | top-leg-pick | 36 | 1.92/1.60/3.35 | 13.07/12.61/17.57 |
| rgbd+GP | one_leg | top-leg-push | 36 | 3.98/3.17/4.48 | 15.89/11.60/13.69 |
| rgbd+GP | round_table | base-leg-insert | 19 | 1.78/1.32/3.12 | 16.42/8.24/9.40 |
| rgbd+GP | round_table | base-leg-pick | 23 | 2.45/1.02/2.03 | 13.75/10.62/16.09 |
| rgbd+GP | round_table | base-leg-place | 21 | 3.60/3.00/4.67 | 7.10/7.16/9.05 |
| rgbd+GP | round_table | base-leg-screw | 19 | 3.85/3.29/5.10 | 98.50/126.95/170.05 |
| rgbd+GP | round_table | leg-top-insert | 25 | 0.74/0.55/0.63 | 7.51/3.80/13.04 |
| rgbd+GP | round_table | leg-top-pick | 36 | 2.14/1.93/3.36 | 27.77/22.94/53.30 |
| rgbd+GP | round_table | leg-top-place | 34 | 2.07/1.24/2.83 | 19.60/6.45/56.62 |
| rgbd+GP | round_table | leg-top-screw | 24 | 2.44/2.20/2.87 | 38.74/34.35/48.20 |
| rgbd+GP | round_table | top-leg-push | 36 | 7.21/7.18/7.93 | 64.73/64.88/72.52 |
| rgbd+GP | lamp | base-bulb-push | 36 | 7.45/7.54/8.24 | 77.93/81.23/87.08 |
| rgbd+GP | lamp | bulb-base-insert | 22 | 1.50/1.48/1.74 | 30.82/25.43/39.53 |
| rgbd+GP | lamp | bulb-base-pick | 36 | 3.70/1.98/3.68 | 17.28/14.31/28.41 |
| rgbd+GP | lamp | bulb-base-place | 34 | 2.69/1.60/6.85 | 43.54/30.77/76.50 |
| rgbd+GP | lamp | bulb-base-screw | 21 | 12.77/13.90/25.26 | 40.96/35.68/76.11 |
| rgbd+GP | lamp | hood-base-pick | 12 | 4.62/3.30/3.53 | 19.79/20.25/23.77 |
| rgbd+GP | lamp | hood-base-place | 11 | 24.76/26.56/30.41 | 56.69/56.20/73.60 |
| rgbd+colored GP | one_leg | leg-top-insert | 34 | 0.69/0.54/0.77 | 6.74/5.98/9.84 |
| rgbd+colored GP | one_leg | leg-top-pick | 35 | 1.09/1.01/1.65 | 15.45/13.31/15.42 |
| rgbd+colored GP | one_leg | leg-top-place | 35 | 1.26/0.85/2.54 | 11.29/10.63/14.93 |
| rgbd+colored GP | one_leg | leg-top-screw | 31 | 5.54/1.87/25.14 | 43.86/22.36/167.65 |
| rgbd+colored GP | one_leg | top-leg-pick | 36 | 2.21/2.38/3.58 | 14.22/13.45/18.54 |
| rgbd+colored GP | one_leg | top-leg-push | 36 | 4.14/3.37/5.71 | 15.83/11.51/14.15 |
| rgbd+colored GP | round_table | base-leg-insert | 21 | 2.13/1.63/3.21 | 23.71/8.14/10.45 |
| rgbd+colored GP | round_table | base-leg-pick | 24 | 3.68/0.98/2.28 | 14.93/9.75/18.28 |
| rgbd+colored GP | round_table | base-leg-place | 21 | 2.75/2.77/3.02 | 7.11/7.47/8.00 |
| rgbd+colored GP | round_table | base-leg-screw | 20 | 5.50/2.95/16.66 | 73.65/57.74/168.96 |
| rgbd+colored GP | round_table | leg-top-insert | 28 | 0.76/0.56/0.76 | 15.05/5.75/20.64 |
| rgbd+colored GP | round_table | leg-top-pick | 36 | 2.24/1.87/4.03 | 20.90/23.90/31.33 |
| rgbd+colored GP | round_table | leg-top-place | 33 | 1.98/1.06/3.04 | 17.25/7.33/60.66 |
| rgbd+colored GP | round_table | leg-top-screw | 27 | 2.38/2.12/3.37 | 40.94/37.00/55.93 |
| rgbd+colored GP | round_table | top-leg-push | 36 | 7.26/7.28/7.81 | 63.03/60.81/74.33 |
| rgbd+colored GP | lamp | base-bulb-push | 36 | 7.74/7.80/8.60 | 78.80/78.82/89.00 |
| rgbd+colored GP | lamp | bulb-base-insert | 23 | 1.57/1.61/1.77 | 32.75/34.81/41.60 |
| rgbd+colored GP | lamp | bulb-base-pick | 36 | 2.87/1.88/3.71 | 15.05/13.59/25.28 |
| rgbd+colored GP | lamp | bulb-base-place | 35 | 3.98/1.71/11.80 | 56.40/39.77/119.27 |
| rgbd+colored GP | lamp | bulb-base-screw | 23 | 16.21/21.30/24.90 | 29.21/23.55/55.02 |
| rgbd+colored GP | lamp | hood-base-pick | 9 | 3.11/3.27/3.52 | 20.95/20.21/24.62 |
| rgbd+colored GP | lamp | hood-base-place | 9 | 27.71/27.15/29.66 | 61.16/51.87/94.65 |
| rgbd+GP+skill | one_leg | leg-top-insert | 32 | 0.54/0.52/0.70 | 6.00/5.66/8.78 |
| rgbd+GP+skill | one_leg | leg-top-pick | 34 | 1.01/1.01/1.42 | 13.58/13.82/15.08 |
| rgbd+GP+skill | one_leg | leg-top-place | 34 | 1.06/0.83/1.55 | 12.02/10.96/16.07 |
| rgbd+GP+skill | one_leg | leg-top-screw | 32 | 7.14/2.10/26.12 | 29.44/17.01/56.74 |
| rgbd+GP+skill | one_leg | top-leg-pick | 36 | 2.10/2.06/3.22 | 12.90/12.95/17.27 |
| rgbd+GP+skill | one_leg | top-leg-push | 36 | 4.19/3.39/4.11 | 16.20/11.17/13.32 |
| rgbd+GP+skill | round_table | base-leg-insert | 11 | 1.79/1.66/2.00 | 22.85/8.68/11.65 |
| rgbd+GP+skill | round_table | base-leg-pick | 21 | 3.59/1.84/6.04 | 18.72/14.07/39.49 |
| rgbd+GP+skill | round_table | base-leg-place | 17 | 4.71/2.94/10.70 | 15.26/8.53/40.49 |
| rgbd+GP+skill | round_table | base-leg-screw | 11 | 7.96/4.56/18.93 | 74.00/74.06/155.49 |
| rgbd+GP+skill | round_table | leg-top-insert | 26 | 0.58/0.54/0.64 | 16.71/5.25/41.63 |
| rgbd+GP+skill | round_table | leg-top-pick | 36 | 2.13/1.98/3.32 | 32.07/23.74/81.96 |
| rgbd+GP+skill | round_table | leg-top-place | 34 | 3.01/1.07/8.11 | 27.99/7.28/86.16 |
| rgbd+GP+skill | round_table | leg-top-screw | 26 | 4.27/3.40/6.32 | 58.10/45.13/97.62 |
| rgbd+GP+skill | round_table | top-leg-push | 36 | 7.07/7.05/7.77 | 65.67/65.30/72.61 |
| rgbd+GP+skill | lamp | base-bulb-push | 36 | 7.41/7.35/8.43 | 75.41/76.67/89.17 |
| rgbd+GP+skill | lamp | bulb-base-insert | 25 | 1.55/1.53/1.80 | 38.56/36.03/62.65 |
| rgbd+GP+skill | lamp | bulb-base-pick | 36 | 2.02/1.95/2.92 | 17.65/13.89/34.89 |
| rgbd+GP+skill | lamp | bulb-base-place | 36 | 3.66/1.63/11.32 | 57.20/45.52/112.18 |
| rgbd+GP+skill | lamp | bulb-base-screw | 25 | 15.26/19.69/22.33 | 20.42/21.83/27.91 |
| rgbd+GP+skill | lamp | hood-base-pick | 10 | 2.92/3.10/3.54 | 22.23/22.59/24.85 |
| rgbd+GP+skill | lamp | hood-base-place | 10 | 25.86/25.96/26.42 | 30.39/25.56/51.89 |

## 4. VLM 打点误差

逐 step 误差定义为 front camera 上 `||p_vlm - p_gt||₂`。缓存期间每个控制 step 都计入，因此这里直接给出你要求的 step average；投影参考也为每个有效控制 step 的 GT/VLM 点对单独生成，从而包含 action horizon 内 GT 移动与 VLM 点缓存造成的实际误差。

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 14.82/23.62 (n=15155) | 25.15/37.48 (n=30884) | 39.68/59.09 (n=24789) | 28.03/44.20 (n=70828) |
| rgbd+colored GP | 17.32/27.54 (n=15107) | 27.83/42.06 (n=30295) | 32.47/46.12 (n=24137) | 27.16/40.91 (n=69539) |
| rgbd+GP+skill | 12.99/20.35 (n=15293) | 35.67/50.78 (n=32442) | 25.04/39.29 (n=24974) | 27.25/42.05 (n=72709) |

表中每格为 `step mean px / step RMSE px (有效 step 数)`。

### 4.1 每格 VLM residual 分布

| Condition | Task | Valid/total | Skill acc. % | Mean px | RMSE px | Median px | P90 px | dx bias | dy bias | Bias norm | Spread | R² | >40/>70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 15155/15165 | 64.2 | 14.82 | 23.62 | 7.07 | 40.52 | -1.66 | -1.92 | 2.54 | 0.990 | 0.149 | 1637/584 |
| rgbd+GP | round_table | 30884/30974 | 33.6 | 25.15 | 37.48 | 15.30 | 82.01 | 6.59 | -0.56 | 6.61 | 0.756 | 0.063 | 5328/4740 |
| rgbd+GP | lamp | 24789/26900 | 65.0 | 39.68 | 59.09 | 18.03 | 97.31 | 13.70 | -12.01 | 18.21 | 0.650 | -0.036 | 10045/6204 |
| rgbd+colored GP | one_leg | 15107/15112 | 61.2 | 17.32 | 27.54 | 6.71 | 58.94 | -1.16 | -3.94 | 4.11 | 0.672 | 0.302 | 2765/147 |
| rgbd+colored GP | round_table | 30295/31152 | 41.0 | 27.83 | 42.06 | 12.53 | 85.38 | 5.15 | -1.76 | 5.44 | 0.795 | -0.061 | 6836/5961 |
| rgbd+colored GP | lamp | 24137/25764 | 64.8 | 32.47 | 46.12 | 16.64 | 80.01 | 11.49 | -10.30 | 15.43 | 0.774 | 0.028 | 9379/4468 |
| rgbd+GP+skill | one_leg | 15293/16073 | 64.5 | 12.99 | 20.35 | 6.08 | 38.83 | 2.14 | -1.11 | 2.41 | 0.871 | 0.383 | 1438/151 |
| rgbd+GP+skill | round_table | 32442/32574 | 36.5 | 35.67 | 50.78 | 24.21 | 89.36 | 4.17 | -2.00 | 4.62 | 0.756 | -0.255 | 9008/7741 |
| rgbd+GP+skill | lamp | 24974/27882 | 72.8 | 25.04 | 39.29 | 7.28 | 80.36 | 12.27 | -6.05 | 13.68 | 0.787 | 0.059 | 6956/3408 |

**指标解释。** `Valid/total` 是同时具有合法 shadow-GT 和 VLM 点的 control-step pair；production 中无效 VLM JSON 会直接终止 row，因此另从日志报告服务失败，不能把它混入 GT coverage。Skill accuracy 使用 shadow oracle coarse skill。Mean/RMSE/median/P90 都基于 `||p_vlm-p_gt||₂`；dx/dy 和 bias 描述系统偏移；spread 是 prediction/GT 二维标准差范数之比；`R²=1-SSE/SST`，0 等价于恒定预测该组 GT 均值，负值更差。R² 在 GT 无空间方差时记为 `—`。

**数据分析。**

- rgbd+GP / one_leg：n=15155；mean/RMSE/median/P90=14.82/23.62/7.07/40.52 px；>40/>70 px=1637/584；bias=2.54 px；spread=0.990；R²=0.149；skill acc=64.2%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP / round_table：n=30884；mean/RMSE/median/P90=25.15/37.48/15.30/82.01 px；>40/>70 px=5328/4740；bias=6.61 px；spread=0.756；R²=0.063；skill acc=33.6%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP / lamp：n=24789；mean/RMSE/median/P90=39.68/59.09/18.03/97.31 px；>40/>70 px=10045/6204；bias=18.21 px；spread=0.650；R²=-0.036；skill acc=65.0%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值；spread 明显收缩，需要检查 regress-to-mean。
- rgbd+colored GP / one_leg：n=15107；mean/RMSE/median/P90=17.32/27.54/6.71/58.94 px；>40/>70 px=2765/147；bias=4.11 px；spread=0.672；R²=0.302；skill acc=61.2%；RMSE 明显高于 mean，存在长尾；spread 明显收缩，需要检查 regress-to-mean。
- rgbd+colored GP / round_table：n=30295；mean/RMSE/median/P90=27.83/42.06/12.53/85.38 px；>40/>70 px=6836/5961；bias=5.44 px；spread=0.795；R²=-0.061；skill acc=41.0%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值。
- rgbd+colored GP / lamp：n=24137；mean/RMSE/median/P90=32.47/46.12/16.64/80.01 px；>40/>70 px=9379/4468；bias=15.43 px；spread=0.774；R²=0.028；skill acc=64.8%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP+skill / one_leg：n=15293；mean/RMSE/median/P90=12.99/20.35/6.08/38.83 px；>40/>70 px=1438/151；bias=2.41 px；spread=0.871；R²=0.383；skill acc=64.5%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP+skill / round_table：n=32442；mean/RMSE/median/P90=35.67/50.78/24.21/89.36 px；>40/>70 px=9008/7741；bias=4.62 px；spread=0.756；R²=-0.255；skill acc=36.5%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值。
- rgbd+GP+skill / lamp：n=24974；mean/RMSE/median/P90=25.04/39.29/7.28/80.36 px；>40/>70 px=6956/3408；bias=13.68 px；spread=0.787；R²=0.059；skill acc=72.8%；RMSE 明显高于 mean，存在长尾。

### 4.2 Fresh-query VLM point 质量

| Condition | Task | Valid/total | Skill acc. % | Mean px | RMSE px | Median px | P90 px | dx bias | dy bias | Bias norm | Spread | R² | >40/>70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 1907/1909 | 65.2 | 14.53 | 23.31 | 6.71 | 40.26 | -1.09 | -1.96 | 2.24 | 0.968 | 0.203 | 201/75 |
| rgbd+GP | round_table | 3868/3880 | 33.9 | 25.04 | 37.44 | 14.82 | 82.01 | 6.56 | -0.69 | 6.59 | 0.755 | 0.064 | 666/594 |
| rgbd+GP | lamp | 3105/3369 | 65.3 | 39.45 | 58.89 | 17.89 | 97.31 | 13.57 | -12.06 | 18.15 | 0.651 | -0.032 | 1249/773 |
| rgbd+colored GP | one_leg | 1902/1903 | 61.7 | 17.08 | 27.25 | 6.32 | 58.94 | -0.52 | -3.96 | 3.99 | 0.662 | 0.335 | 338/21 |
| rgbd+colored GP | round_table | 3788/3902 | 41.5 | 27.70 | 41.99 | 12.53 | 85.38 | 5.13 | -1.92 | 5.47 | 0.795 | -0.058 | 853/743 |
| rgbd+colored GP | lamp | 3026/3229 | 65.7 | 31.96 | 45.68 | 14.87 | 80.01 | 11.23 | -10.38 | 15.29 | 0.775 | 0.041 | 1153/546 |
| rgbd+GP+skill | one_leg | 1924/2023 | 65.5 | 12.62 | 19.91 | 6.00 | 37.60 | 2.58 | -1.16 | 2.83 | 0.855 | 0.428 | 172/20 |
| rgbd+GP+skill | round_table | 4058/4076 | 36.8 | 35.56 | 50.74 | 24.08 | 89.81 | 4.06 | -2.13 | 4.59 | 0.757 | -0.257 | 1127/969 |
| rgbd+GP+skill | lamp | 3132/3491 | 73.1 | 24.86 | 39.14 | 7.07 | 80.72 | 12.04 | -6.11 | 13.51 | 0.789 | 0.062 | 862/423 |

**指标解释。** 本表只保留 `step_idx == query_step` 的新 VLM 请求，衡量模型本身的即时输出；上一表统计每个 control step，包含 action-horizon 缓存造成的实际 stale-point 误差。其余公式完全相同。

**数据分析。**

- rgbd+GP / one_leg：n=1907；mean/RMSE/median/P90=14.53/23.31/6.71/40.26 px；>40/>70 px=201/75；bias=2.24 px；spread=0.968；R²=0.203；skill acc=65.2%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP / round_table：n=3868；mean/RMSE/median/P90=25.04/37.44/14.82/82.01 px；>40/>70 px=666/594；bias=6.59 px；spread=0.755；R²=0.064；skill acc=33.9%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP / lamp：n=3105；mean/RMSE/median/P90=39.45/58.89/17.89/97.31 px；>40/>70 px=1249/773；bias=18.15 px；spread=0.651；R²=-0.032；skill acc=65.3%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值；spread 明显收缩，需要检查 regress-to-mean。
- rgbd+colored GP / one_leg：n=1902；mean/RMSE/median/P90=17.08/27.25/6.32/58.94 px；>40/>70 px=338/21；bias=3.99 px；spread=0.662；R²=0.335；skill acc=61.7%；RMSE 明显高于 mean，存在长尾；spread 明显收缩，需要检查 regress-to-mean。
- rgbd+colored GP / round_table：n=3788；mean/RMSE/median/P90=27.70/41.99/12.53/85.38 px；>40/>70 px=853/743；bias=5.47 px；spread=0.795；R²=-0.058；skill acc=41.5%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值。
- rgbd+colored GP / lamp：n=3026；mean/RMSE/median/P90=31.96/45.68/14.87/80.01 px；>40/>70 px=1153/546；bias=15.29 px；spread=0.775；R²=0.041；skill acc=65.7%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP+skill / one_leg：n=1924；mean/RMSE/median/P90=12.62/19.91/6.00/37.60 px；>40/>70 px=172/20；bias=2.83 px；spread=0.855；R²=0.428；skill acc=65.5%；RMSE 明显高于 mean，存在长尾。
- rgbd+GP+skill / round_table：n=4058；mean/RMSE/median/P90=35.56/50.74/24.08/89.81 px；>40/>70 px=1127/969；bias=4.59 px；spread=0.757；R²=-0.257；skill acc=36.8%；RMSE 明显高于 mean，存在长尾；R²<0，平方误差差于恒定预测该组 GT 均值。
- rgbd+GP+skill / lamp：n=3132；mean/RMSE/median/P90=24.86/39.14/7.07/80.72 px；>40/>70 px=862/423；bias=13.51 px；spread=0.789；R²=0.062；skill acc=73.1%；RMSE 明显高于 mean，存在长尾。

### 4.3 Each skill step average（跨 task 聚合）

| Condition | Skill | Valid/total | Skill acc. % | Mean px | RMSE px | Median px | P90 px | Bias norm | Spread | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | push | 4386/4386 | 61.4 | 19.33 | 26.93 | 8.49 | 43.68 | 7.56 | 2.052 | -3.729 |
| rgbd+GP | pick | 40870/43081 | 38.7 | 40.52 | 56.02 | 24.33 | 90.20 | 15.78 | 0.699 | -0.093 |
| rgbd+GP | place | 9144/9144 | 81.1 | 10.37 | 16.40 | 6.08 | 27.07 | 1.87 | 0.848 | 0.703 |
| rgbd+GP | insert | 903/903 | 31.9 | 6.60 | 9.08 | 5.10 | 12.96 | 1.25 | 0.875 | 0.874 |
| rgbd+GP | screw | 15525/15525 | 68.2 | 9.25 | 16.89 | 5.10 | 15.81 | 3.37 | 1.254 | 0.268 |
| rgbd+colored GP | push | 4427/4427 | 59.6 | 20.20 | 27.84 | 9.22 | 43.17 | 6.11 | 2.027 | -3.520 |
| rgbd+colored GP | pick | 38853/41342 | 42.7 | 39.87 | 51.89 | 33.53 | 87.13 | 15.24 | 0.758 | 0.008 |
| rgbd+colored GP | place | 8170/8170 | 84.0 | 9.20 | 17.08 | 5.00 | 17.00 | 2.75 | 0.982 | 0.637 |
| rgbd+colored GP | insert | 929/929 | 20.8 | 6.19 | 10.80 | 4.24 | 11.52 | 0.61 | 1.010 | 0.814 |
| rgbd+colored GP | screw | 17160/17160 | 66.2 | 9.84 | 18.46 | 5.00 | 17.72 | 2.71 | 1.368 | 0.084 |
| rgbd+GP+skill | push | 5221/5221 | 53.7 | 20.63 | 26.90 | 10.20 | 41.98 | 10.42 | 1.886 | -3.273 |
| rgbd+GP+skill | pick | 39353/43173 | 41.2 | 40.51 | 54.13 | 29.15 | 91.35 | 14.46 | 0.789 | -0.215 |
| rgbd+GP+skill | place | 7774/7774 | 88.8 | 8.99 | 16.08 | 5.10 | 15.03 | 2.54 | 0.927 | 0.691 |
| rgbd+GP+skill | insert | 510/510 | 26.5 | 6.99 | 8.67 | 6.00 | 12.21 | 1.40 | 0.841 | 0.890 |
| rgbd+GP+skill | screw | 19851/19851 | 75.2 | 10.38 | 19.36 | 5.39 | 15.56 | 3.95 | 1.497 | -0.215 |

### 4.4 每个 task 的 per-skill point error

| Condition | Task | Skill | Valid/total | Skill acc. % | Mean px | RMSE px | Median px | P90 px | dx bias | dy bias | Bias norm | Spread | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | insert | 254/254 | 0.0 | 4.99 | 5.94 | 4.47 | 9.06 | -3.40 | -0.46 | 3.43 | 0.903 | 0.062 |
| rgbd+GP | one_leg | pick | 6986/6996 | 57.9 | 18.62 | 28.94 | 12.65 | 53.34 | -5.87 | -2.91 | 6.55 | 0.970 | 0.206 |
| rgbd+GP | one_leg | place | 2593/2593 | 92.5 | 4.87 | 6.35 | 3.61 | 9.06 | -1.92 | -0.89 | 2.11 | 0.999 | -0.002 |
| rgbd+GP | one_leg | push | 1671/1671 | 13.8 | 34.68 | 37.04 | 39.20 | 43.68 | 17.83 | -7.26 | 19.25 | 4.547 | -33.089 |
| rgbd+GP | one_leg | screw | 3651/3651 | 83.6 | 6.21 | 7.38 | 5.39 | 11.40 | -2.24 | 1.59 | 2.74 | 0.940 | -0.066 |
| rgbd+GP | round_table | insert | 544/544 | 51.7 | 6.29 | 8.78 | 5.10 | 9.85 | 2.15 | 1.01 | 2.38 | 0.977 | -0.026 |
| rgbd+GP | round_table | pick | 20776/20866 | 19.5 | 32.24 | 43.97 | 20.52 | 85.01 | 9.64 | -1.06 | 9.70 | 0.728 | 0.041 |
| rgbd+GP | round_table | place | 3574/3574 | 80.0 | 11.57 | 16.97 | 7.00 | 27.07 | 4.13 | -1.57 | 4.42 | 0.883 | -0.717 |
| rgbd+GP | round_table | push | 1065/1065 | 85.2 | 11.16 | 18.72 | 5.10 | 34.61 | -1.25 | -5.91 | 6.04 | 3.004 | -10.497 |
| rgbd+GP | round_table | screw | 4925/4925 | 46.7 | 10.25 | 18.95 | 6.32 | 13.45 | -2.32 | 3.28 | 4.02 | 2.101 | -3.209 |
| rgbd+GP | lamp | insert | 105/105 | 6.7 | 12.10 | 14.99 | 8.06 | 23.54 | -2.31 | 6.59 | 6.99 | 0.792 | -1.529 |
| rgbd+GP | lamp | pick | 13108/15219 | 56.1 | 65.31 | 79.21 | 68.62 | 124.15 | 27.89 | -25.33 | 37.68 | 0.697 | -0.678 |
| rgbd+GP | lamp | place | 2977/2977 | 72.4 | 13.73 | 21.10 | 7.28 | 39.62 | -1.11 | 7.97 | 8.05 | 0.969 | -1.105 |
| rgbd+GP | lamp | push | 1650/1650 | 94.1 | 9.06 | 17.69 | 5.00 | 12.04 | -2.48 | -2.44 | 3.48 | 2.725 | -7.011 |
| rgbd+GP | lamp | screw | 6949/6949 | 75.3 | 10.14 | 18.82 | 4.24 | 22.02 | -2.66 | 2.01 | 3.33 | 1.825 | -1.833 |
| rgbd+colored GP | one_leg | insert | 404/404 | 0.0 | 3.72 | 4.43 | 3.16 | 6.71 | -0.68 | 0.47 | 0.83 | 0.857 | 0.375 |
| rgbd+colored GP | one_leg | pick | 7165/7170 | 54.7 | 24.48 | 35.69 | 9.43 | 63.53 | -4.98 | -8.23 | 9.62 | 0.601 | 0.338 |
| rgbd+colored GP | one_leg | place | 2344/2344 | 93.9 | 4.39 | 5.25 | 4.00 | 8.06 | -1.04 | -0.40 | 1.12 | 0.815 | 0.315 |
| rgbd+colored GP | one_leg | push | 1653/1653 | 17.2 | 32.87 | 35.48 | 38.05 | 43.17 | 15.82 | -4.63 | 16.48 | 4.696 | -30.414 |
| rgbd+colored GP | one_leg | screw | 3541/3541 | 80.1 | 5.70 | 7.11 | 4.47 | 11.05 | -1.48 | 2.20 | 2.65 | 1.025 | -0.138 |
| rgbd+colored GP | round_table | insert | 468/468 | 40.8 | 8.25 | 14.27 | 5.83 | 15.00 | 1.32 | 0.60 | 1.45 | 1.501 | -1.526 |
| rgbd+colored GP | round_table | pick | 19869/20726 | 31.3 | 36.61 | 49.77 | 23.19 | 89.29 | 8.87 | -3.02 | 9.37 | 0.759 | -0.073 |
| rgbd+colored GP | round_table | place | 3184/3184 | 80.0 | 8.66 | 15.92 | 5.10 | 13.04 | 0.32 | 0.03 | 0.32 | 1.833 | -2.695 |
| rgbd+colored GP | round_table | push | 1063/1063 | 81.6 | 12.14 | 17.49 | 6.32 | 37.01 | 0.62 | -7.51 | 7.53 | 2.482 | -7.964 |
| rgbd+colored GP | round_table | screw | 5711/5711 | 46.9 | 12.47 | 23.51 | 6.32 | 31.06 | -3.98 | 2.49 | 4.69 | 2.449 | -4.997 |
| rgbd+colored GP | lamp | insert | 57/57 | 3.5 | 6.78 | 9.50 | 4.24 | 12.81 | 0.14 | -0.56 | 0.58 | 1.527 | -1.354 |
| rgbd+colored GP | lamp | pick | 11819/13446 | 53.9 | 54.68 | 62.58 | 60.53 | 87.13 | 25.87 | -22.87 | 34.53 | 0.893 | -0.788 |
| rgbd+colored GP | lamp | place | 2642/2642 | 80.0 | 14.14 | 23.92 | 5.83 | 39.29 | -3.93 | 7.54 | 8.50 | 1.444 | -2.161 |
| rgbd+colored GP | lamp | push | 1711/1711 | 86.8 | 12.96 | 24.47 | 5.66 | 56.14 | -5.80 | -3.22 | 6.63 | 4.072 | -16.657 |
| rgbd+colored GP | lamp | screw | 7908/7908 | 74.0 | 9.79 | 17.82 | 4.12 | 20.12 | -1.04 | 0.91 | 1.38 | 1.940 | -2.238 |
| rgbd+GP+skill | one_leg | insert | 246/246 | 0.0 | 4.95 | 5.93 | 3.61 | 10.00 | -2.16 | -0.94 | 2.36 | 1.059 | -0.106 |
| rgbd+GP+skill | one_leg | pick | 5710/6490 | 63.3 | 12.74 | 22.34 | 3.61 | 35.36 | -0.37 | -1.46 | 1.51 | 0.817 | 0.597 |
| rgbd+GP+skill | one_leg | place | 2761/2761 | 93.9 | 4.91 | 5.88 | 4.00 | 10.05 | -2.21 | -0.79 | 2.35 | 0.836 | 0.228 |
| rgbd+GP+skill | one_leg | push | 2425/2425 | 11.8 | 33.59 | 35.50 | 37.12 | 41.98 | 21.63 | -5.64 | 22.35 | 7.143 | -91.352 |
| rgbd+GP+skill | one_leg | screw | 4151/4151 | 81.4 | 7.15 | 8.86 | 6.08 | 13.60 | -2.63 | 1.82 | 3.20 | 1.355 | -0.018 |
| rgbd+GP+skill | round_table | insert | 189/189 | 69.8 | 9.82 | 11.38 | 10.05 | 15.00 | 6.97 | 1.39 | 7.11 | 0.822 | -0.427 |
| rgbd+GP+skill | round_table | pick | 24297/24429 | 29.2 | 42.65 | 56.71 | 27.80 | 97.01 | 7.19 | -2.85 | 7.74 | 0.714 | -0.243 |
| rgbd+GP+skill | round_table | place | 2490/2490 | 87.6 | 8.02 | 12.78 | 5.39 | 13.60 | 2.20 | 0.51 | 2.26 | 1.034 | -0.606 |
| rgbd+GP+skill | round_table | push | 1170/1170 | 79.9 | 12.55 | 19.39 | 6.40 | 35.44 | -2.06 | -7.48 | 7.75 | 2.902 | -10.265 |
| rgbd+GP+skill | round_table | screw | 4296/4296 | 35.4 | 19.71 | 32.97 | 8.25 | 74.01 | -10.21 | 2.71 | 10.56 | 3.899 | -15.947 |
| rgbd+GP+skill | lamp | insert | 75/75 | 4.0 | 6.57 | 8.36 | 5.10 | 11.03 | -1.19 | 1.57 | 1.97 | 0.908 | -0.371 |
| rgbd+GP+skill | lamp | pick | 9346/12254 | 53.4 | 51.91 | 60.59 | 51.00 | 93.23 | 35.89 | -19.11 | 40.66 | 1.129 | -1.850 |
| rgbd+GP+skill | lamp | place | 2523/2523 | 84.3 | 14.41 | 24.45 | 7.00 | 38.90 | -4.05 | 6.91 | 8.01 | 1.516 | -2.339 |
| rgbd+GP+skill | lamp | push | 1626/1626 | 97.4 | 7.11 | 13.20 | 5.10 | 9.85 | -1.11 | -1.69 | 2.02 | 1.996 | -3.795 |
| rgbd+GP+skill | lamp | screw | 11404/11404 | 87.9 | 8.05 | 14.64 | 4.24 | 13.04 | -1.47 | 1.12 | 1.85 | 1.599 | -1.267 |

## 5. VLM 对应 n0–n4 的哪个等级？

不能把 2D 像素误差直接除以一个固定 px/mm 系数，再与 3D 的 0/3/6/12/24 mm 比较。透视投影尺度随 GT 点深度、相机内参和偏移方向变化。主分析采用同坐标系比较：

1. 每个有效控制 step 形成一个配对样本：自动机给出当步 3D GT guidance point `P_gt`，同一 annotation util 给出 front-camera GT pixel `p_gt` 和相机内外参，实际送给 policy 的 VLM 点为 `p_vlm`。VLM 在 action horizon 内可以缓存，但每个当步 GT/VLM 点对都独立进入 step average 和投影分布。
2. 用由 `(episode, env, step, query_step)` 确定的 seed 采样 `200` 个 `z_j ~ N(0, I_3)`，随后逐分量 clip 到 `[-2, 2]`。这与现有 `annotation_noise.py` 完全一致。严格来说 clip 后的边际方差小于 1；这里的 n1--n4 名称和 σ 参数沿用原噪声实验，而不是声称截断后的实际标准差仍恰好等于 σ。
3. 五档使用同一组 `z_j`（common random numbers）以减小档位间 Monte Carlo 抖动。令 `σ_n ∈ {0, 3, 6, 12, 24} mm`，构造 `P_nj = P_gt + σ_n z_j`。
4. 使用 `skill_annotation_util.py` 相同的 robot-base→camera 变换、camera-y 翻转和内参投影，计算连续坐标 `e_nj = π(P_nj) - π(P_gt)`；VLM 残差为 `e_vlm = p_vlm - p_gt`。参考噪声投影不做整数取整，也不按图像边界裁剪，否则会人为压缩尾部。
5. 对每档全部投影样本精确累计一阶、二阶矩：`μ_n = (1/M)Σe_nj`，`Σ_n = (1/M)Σ(e_nj-μ_n)(e_nj-μ_n)^T`，并由全部样本计算 projected RMSE。VLM 的 mean/cov/RMSE 在所有有效 step 残差上计算，和 reference 的相机/深度条件完全配对。
6. 用 projected RMSE 在相邻 n 档之间线性插值得到主要“误差量级”，例如 `n1–n2`；超过 n4 时线性外推并标记 `>n4`。径向 W1 最近档作为大小分布的稳健交叉检查。
7. 完整 2D SWD 用于判断包含系统偏置的二维分布是否像某档噪声；centered SWD 在双方各自减均值后比较形状。它们不替代第 6 步的误差大小分类。

实现的性能策略：每个点对实际生成 `200` 个样本；mean/cov/RMSE 用全部样本的 sufficient statistics 精确合并；每档仅保留均匀抽取的 2000 个 reference residual 计算分位数、32-direction SWD、W1 和 KS，使 rollout summary 大小与后处理耗时有上界。在更接近最大单批规模的 3000-pair、200-sample 合成基准中，600,000 个样本/档的投影约 1.76 s，三 scope 汇总约 11.06 s，进程峰值 RSS 约 958 MiB；本机约有 19 GiB 可用内存，因此正式配置从 100 提高到 200。该数字只是性能基准，不是实验结果；正式运行仍记录实际 wall time。

以下第一张表按 condition 跨三个 task、按有效 step 聚合。

| Condition | Valid pairs | VLM RMSE px | P90 px | P95 px | Bias px | Anisotropy | Same-depth mm | Magnitude bracket | Equivalent σ mm | Closest radial W1 | Closest full SWD | Closest centered SWD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | 70828 | 44.20 | 83.60 | 93.02 | 8.78 | 4.70 | 60.01 | >n4 | 66.96 | n4 | n4 | n4 |
| rgbd+colored GP | 69539 | 40.91 | 80.00 | 88.28 | 7.92 | 3.91 | 60.13 | >n4 | 62.76 | n4 | n4 | n4 |
| rgbd+GP+skill | 72709 | 42.05 | 82.73 | 93.26 | 7.27 | 5.73 | 65.85 | >n4 | 65.95 | n4 | n4 | n4 |

### 5.1 每个 task 的误差量级

task-level 映射是主要诊断表；它避免不同任务的 GT 深度和透视尺度在总体聚合中互相抵消。

| Condition | Task | Valid pairs | VLM RMSE px | P90 px | Bias px | Magnitude bracket | Equivalent σ mm | Closest radial W1 | Closest full SWD | Closest centered SWD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 15155 | 23.62 | 40.52 | 2.54 | >n4 | 36.34 | n4 | n4 | n4 |
| rgbd+GP | round_table | 30884 | 37.48 | 82.01 | 6.61 | >n4 | 61.25 | n4 | n4 | n4 |
| rgbd+GP | lamp | 24789 | 59.09 | 97.31 | 18.21 | >n4 | 81.93 | n4 | n4 | n4 |
| rgbd+colored GP | one_leg | 15107 | 27.54 | 58.94 | 4.11 | >n4 | 41.07 | n4 | n4 | n4 |
| rgbd+colored GP | round_table | 30295 | 42.06 | 85.38 | 5.44 | >n4 | 69.68 | n4 | n4 | n4 |
| rgbd+colored GP | lamp | 24137 | 46.12 | 80.01 | 15.43 | >n4 | 66.20 | n4 | n4 | n4 |
| rgbd+GP+skill | one_leg | 15293 | 20.35 | 38.83 | 2.41 | >n4 | 31.09 | n3 | n3 | n4 |
| rgbd+GP+skill | round_table | 32442 | 50.78 | 89.36 | 4.62 | >n4 | 86.87 | n4 | n4 | n4 |
| rgbd+GP+skill | lamp | 24974 | 39.29 | 80.36 | 13.68 | >n4 | 56.91 | n4 | n4 | n4 |

### 5.2 与各档噪声分布的距离（condition overall）

| Condition | Level | Full 2D SWD px | Centered SWD px | Radial W1 px | Bias gap px | Radial KS | Projected RMSE px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | n0 | 17.84 | 19.11 | 28.02 | 8.78 | 0.968 | 0.00 |
| rgbd+GP | n1 | 16.71 | 18.11 | 26.25 | 8.78 | 0.771 | 1.97 |
| rgbd+GP | n2 | 15.59 | 17.13 | 24.47 | 8.77 | 0.618 | 3.95 |
| rgbd+GP | n3 | 13.48 | 15.24 | 21.20 | 8.77 | 0.453 | 7.90 |
| rgbd+GP | n4 | 10.40 | 11.85 | 17.03 | 8.75 | 0.276 | 15.83 |
| rgbd+colored GP | n0 | 17.29 | 18.43 | 27.16 | 7.92 | 0.975 | 0.00 |
| rgbd+colored GP | n1 | 16.19 | 17.43 | 25.42 | 7.92 | 0.733 | 1.95 |
| rgbd+colored GP | n2 | 15.12 | 16.44 | 23.68 | 7.92 | 0.584 | 3.90 |
| rgbd+colored GP | n3 | 13.20 | 14.52 | 20.58 | 7.92 | 0.421 | 7.80 |
| rgbd+colored GP | n4 | 10.60 | 11.05 | 16.93 | 7.90 | 0.327 | 15.63 |
| rgbd+GP+skill | n0 | 17.35 | 18.12 | 27.26 | 7.27 | 0.983 | 0.00 |
| rgbd+GP+skill | n1 | 16.27 | 17.13 | 25.55 | 7.27 | 0.778 | 1.91 |
| rgbd+GP+skill | n2 | 15.20 | 16.15 | 23.84 | 7.27 | 0.598 | 3.81 |
| rgbd+GP+skill | n3 | 13.21 | 14.26 | 20.59 | 7.26 | 0.436 | 7.63 |
| rgbd+GP+skill | n4 | 10.46 | 11.00 | 16.80 | 7.24 | 0.313 | 15.29 |

## 6. 偏移分布与噪声假设有多大区别？

建议同时看这些互补量，而不是只看一个平均半径：

- `full 2D SWD (px)`：同时反映尺度、系统偏置、方向和协方差形状；越小越接近该 n 档完整二维分布。它不是误差大小分类器：强单向 bias 下 full SWD 最近档可能是 n0。
- `centered SWD (px)`：VLM 和参考噪声各自减去均值后再计算，用来比较去除系统偏置后的分布形状。
- `radial W1 (px)`：只比较偏移半径，单位仍是像素，直观说明大小分布差多少。
- `bias norm (px)`：VLM 平均偏移向量的长度。噪声假设是零均值；bias 大说明存在系统性偏移，无法用增大零均值 σ 完全解释。
- `anisotropy = λmax/λmin`：协方差椭圆的长短轴比。接近 1 表示各向同性；明显大于 1 表示 VLM 偏移有方向偏好。径向 KS 作为辅助无量纲检验量。

same-depth mm 是把 VLM 像素在 GT 深度处反投影后的横向位移。它便于工程直觉和毫米量级展示，但深度方向不可由单个 2D 点观测，因此它不是完整 3D VLM 误差，也不能替代上面的投影分布匹配。

原噪声实验一个 skill phase 内保持同一个 3D 偏移；VLM 每个 query 更新、两次 query 之间缓存。上述 SWD 比较的是空间边缘分布，不声称二者的时间相关结构相同。

## 7. 实验方案与聚合口径

### 7.1 实验矩阵与控制变量

- Conditions：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`；仅 checkpoint/模型输入配置不同，VLM 服务、任务初始分布和评测代码相同。
- `rgbd+GP` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt`。
- `rgbd+colored GP` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-18/14-59-28.908152/models/absurd-voice-2_2026-06-18_14-59-48.700671/actor_chkpt_latest_3000.pt`。
- `rgbd+GP+skill` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/12-55-43.621615/models/fresh-tree-11_2026-06-13_12-56-10.422936/actor_chkpt_latest_3000.pt`。
- 当前主 manifest tasks：`one_leg`、`round_table`、`lamp`；每个 condition-task 36 rollout，共 `3×3×36=324`。Formal 固定为 36 rollout/格（324 总计）。
- 落盘上限：每格最多保存 `10` 条 pickle/MP4；全部 36 条仍在内存中累计 success、progress、tracking 和 VLM point/MC200 指标并写入 task summary。
- 本地并行：`n_envs=3`；三个 task 上限均为 1000 step；`randomness=low`。
- VLM：固定 readiness 中的 model revision；每次正式启动先 fail-fast 检查 `status=ready`、`policy_version=3`、`model_mode=original_sft`。各完成 cell 实际使用的 HTTP timeout 为 30/120 s；超时、schema/revision 不一致均直接终止，不 fallback 到自动机。
- Query：`--vlm-query-interval 0`，使用 checkpoint 的 `action_horizon=8`；每 8 个 environment step query 一次，其间缓存。
- Noise projection：每个有效控制 step 的 GT/VLM 点对、每档 `200` 个 clipped-standard-Gaussian 样本；reference reservoir 2000/档；SWD 32 个固定方向。
- Tracking target：shadow 自动机的 clean GT guidance pose；强制 `pose` 模式并报告 position cm / orientation deg / normalized total。VLM 控制 policy，自动机只负责 shadow GT 和指标。
- Initial state：当前默认方案与历史噪声实验一样，使用独立的 `randomness=low` reset，并非三个 condition 严格共享同一批初始状态。36 rollout/格和 Wilson CI 能反映抽样不确定性，但 condition 差值仍包含 reset 方差；若要求 paired comparison，应先从真实 env reset 额外建立并目视验证每 task 36 个固定初始状态的 bank，再给三种 condition 共同使用。不能复用仓库之前的 train-init bank，因为 `reports/train_init_eval.md` 已记录第一帧、坐标和关节状态有效性风险。

### 7.2 三类主要输出与聚合口径

1. **Success rate**：每 task 报 `success/36`，condition overall 报 `success/108` 和 Wilson 95% CI。condition 优劣的主要依据是 success rate，但 36 次/格仍应结合置信区间解释。
2. **Tracking error**：按所有有效控制记录加权，报告 position、orientation 和 `total = pos_m/0.01 + ori_deg/5`。它回答 policy 相对 clean GT pose 的跟踪程度。
3. **VLM 打点误差**：overall 与 each-skill 都按有效控制 step 加权报告 mean/RMSE pixel；全体有效 GT/VLM 点对另报 P90/P95、bias、covariance/anisotropy、same-depth lateral mm、n-level 等价量级和分布距离。按 skill 统计使用 oracle skill 标签，避免 VLM skill 误分类污染分组。

三个指标不合成一个总分：先以成功率比较 condition；tracking error 解释控制效果；point error 与 n-level/distribution metrics 诊断 VLM guidance 的空间质量。另保留 success-only / failure-only 打点统计，用于观察误差与任务失败的关联，但不作因果结论。

### 7.3 与历史噪声实验比较时的边界

- 主映射使用本报告的同相机 3D-noise→2D residual reference，不把 VLM 的 2D error 直接当作 3D mm。
- 历史噪声实验的偏移在 skill phase 内固定，而 VLM 以 8-step query/cache 更新；所以当前只比较空间边缘分布。若要比较时间相关结构，需要额外报告 autocorrelation 或按 phase 重跑。
- 本报告 tracking target 是 clean GT pose；若历史噪声 tracking 以加噪 pose 为 target，两者 tracking 数值不可直接映射。噪声等级结论只由投影残差比较给出。
- `same-depth mm` 只表示 GT 深度平面上的横向误差，不能恢复不可观测的深度方向，因此只作为工程辅助量。
- 正式运行使用独立的 rollout suffix/output 目录；旧的中断 rollout 不纳入任何表格或结论。

### 7.4 正式运行配置

已批准的固定项：`randomness=low` 独立 reset、`n_envs=3`、36 rollout/格、每格最多保存 10 条 pickle/MP4、query horizon=8、200 MC 样本/点/档、2000 reference reservoir/档、32-direction SWD、三个 task 的 max steps 均为 1000，以及 clean-GT pose tracking target。正式矩阵共 324 rollout。

## 8. 当前结论

- 跨三个 task，成功率最高的是 **rgbd+colored GP**：59.3%（64/108）。
- clean-GT pose tracking total 最低的是 **rgbd+GP**：9.78。
- step-weighted 打点 RMSE 最低的是 **rgbd+colored GP**：40.91 px。
- 三组 Wilson 95% CI 有重叠，因此“rgbd+colored GP 数值最高”应解释为当前 108-rollout/condition 下的最好观测值，而不是对所有 condition 差异都作显著性声明。

Condition 汇总：

- **rgbd+GP**：SR 58.3% （63/108，Wilson 95% CI 48.9%–67.2%）；tracking total 9.78；等价噪声 >n4 （外推 σ 66.96 mm）。
- **rgbd+colored GP**：SR 59.3% （64/108，Wilson 95% CI 49.8%–68.1%）；tracking total 10.02；等价噪声 >n4 （外推 σ 62.76 mm）。
- **rgbd+GP+skill**：SR 50.0% （54/108，Wilson 95% CI 40.7%–59.3%）；tracking total 10.44；等价噪声 >n4 （外推 σ 65.95 mm）。

full/centered SWD、radial W1、bias 和 anisotropy 表明 VLM 偏移具有显著系统偏置和方向性，不能只用增大零均值各向同性高斯的 σ 完整解释。

注意：本报告 VLM tracking target 是 clean GT pose，而历史噪声报告的 tracking target 是实际加噪 pose；二者 target 定义不同，不能直接把 tracking 数值映射成噪声等级。噪声等级的主比较必须使用第 5 节的同相机投影残差。

## 9. 复现命令

所有 scripted/VLM eval 必须通过 `/data/hy/gpu-snatcher/auto_eval.sh`；禁止人工直接调用 `evaluate_model`。以下示例会由 auto_eval 自动加入 `--save-depth-image`、保存 pickle/MP4 等固定参数。

本地单任务（以 rgbd+GP / one_leg 为例）：

```bash
export VLM_GUIDANCE_URL=http://10.71.106.240:8000
export VLM_API_TOKEN="$(sed -n 's/^VLM_API_TOKEN=//p' /mnt/nas/share/home/hy/vlm-guidance/server.env)"

/data/hy/gpu-snatcher/auto_eval.sh --steps eval \
  --local-path /data/hy/robust-rearrangement \
  --overwrite-wt-path /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt \
  --task one_leg --n-envs 3 --n-rollouts 36 \
  --randomness low --max-rollout-steps 1000 \
  --max-saved-rollouts 10 \
  --guidance-point-on-image --no-annotate-skill \
  --tracking-metric-type pose \
  --annotation-source vlm --vlm-base-url "$VLM_GUIDANCE_URL" \
  --vlm-timeout-seconds 30 --vlm-query-interval 0 \
  --vlm-noise-projection-samples 200 \
  --task-summary-out logs/vlm_dit_single/rgbd_gp__one_leg.json \
  --rollout-suffix-model-name vlm_dit_single/rgbd_gp/one_leg
```

完整矩阵由 gated runner 先做 print-command 审计，再执行。formal 默认使用已通过人工视频审查的 smoke manifest；用户明确授权直接 formal 时，必须用显式 bypass 和审批说明，由 manifest 留痕。

```bash
python3 scripts/run_vlm_dit_eval.py --phase print --stage formal \
  --namespace vlm_ckpt_new_formal_composite_mc200_low_20260822 \
  --output-dir /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite \
  --data-dir-raw /data/hy/robust-rearrangement/data \
  --allow-formal-without-smoke \
  --formal-approval-note 'User requested full experiment after ckpt_new matched 300-sample diagnostic and 5/6 one_leg gate on 2026-08-22'
```

只重新生成报告：

```bash
python scripts/generate_vlm_dit_report.py --manifest /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite/manifest.json --grounding-summary /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/grounding_300_current_images_cross_skill.json --output /data/hy/robust-rearrangement/reports/vlm_dit_guidance_eval.md
```

## 10. 原始导出

- task-level CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_task.csv`
- condition overall CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_overall.csv`
- each-skill step average CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_skill.csv`
- task×skill tracking CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_tracking_by_task_skill.csv`
- task×skill point error CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_point_error_by_task_skill.csv`
- source manifest：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite/manifest.json`
- 不复制 runtime manifest 到 reports；以上述 logs 路径为唯一源。

## 11. 数据来源与可追溯性

本节覆盖报告中的全部 15 张数据表。表格只由下列本地 JSON 源和确定性聚合公式生成；CSV 是便于继续分析的派生导出，不是第二套数据源。被 `.gitignore` 忽略的 runtime 数据不提交，但同机 session 可按绝对路径读取，并用 SHA256 检查是否仍是本报告使用的版本。

| 报告表格 | 原始来源 | 字段/计算来源 |
| --- | --- | --- |
| ckpt_new 300-sample grounding gate | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/grounding_300_current_images_cross_skill.json | `summaries.current.{overall,task/one_leg,task/round_table,task/lamp}` |
| Matched scripted diagnostics | /data/hy/robust-rearrangement/logs/vlm_dit_depthfix_scripted_diag_20260817/summaries/rgbd_gp__one_leg.json | `n_success`, `n_rollouts`, `tracking_error`, progress/skill counters |
| Success matrix；每格 Wilson 95% CI | 9 个 task summary（见下表） | `n_success`, `n_rollouts`；Wilson 由 `_wilson` 确定性计算 |
| Tracking matrix；task position/rotation 分布 | 9 个 task summary | `tracking_error.overall` |
| Per-skill tracking | 9 个 task summary | `tracking_error.by_skill` |
| VLM step mean/RMSE matrix；每格 residual 分布 | 9 个 task summary | `vlm_point_error.all.overall` |
| Fresh-query VLM point 质量 | 9 个 task summary | `vlm_point_error.all.fresh_queries.overall` |
| Each skill step average（跨 task） | 9 个 task summary | 合并 `vlm_point_error.all.by_skill` 的 sufficient statistics |
| Task × skill point error | 9 个 task summary | `vlm_point_error.all.by_skill` |
| Condition overall n0–n4 量级 | 9 个 task summary | 合并 `vlm_point_error.all.step_distribution` |
| 每 task n0–n4 量级 | 9 个 task summary | `vlm_point_error.all.step_distribution` |
| Condition × n0–n4 SWD/W1/KS | 9 个 task summary | 合并 `step_distribution.projection_reference.levels` 与 `step_distribution.vlm` |

说明：上表将版式相同、来源相同的成对表格合并为一行，因此 12 条来源映射覆盖 15 张 Markdown 表。报告结论只对这些表做排序和文字解释，不引入额外实验数据。

### 11.1 顶层来源文件

| Source | Absolute path | Bytes | SHA256 |
| --- | --- | --- | --- |
| formal composite manifest | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_composite/manifest.json | 72873 | b9d22d90655958ce9792a9b14b4bf74d0d042aa37cc0ca39b91b5e0c931276cc |
| 300-sample grounding summary | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/grounding_300_current_images_cross_skill.json | 481122 | e4563ac640b3400b2acd88bb3aadff7f07a4b21095386eb926dd905241f4651d |
| matched scripted summary | /data/hy/robust-rearrangement/logs/vlm_dit_depthfix_scripted_diag_20260817/summaries/rgbd_gp__one_leg.json | 10959 | 9a5b6edf6dc826489d13f2197db19d9cf8b4ccc119ca0d5835cfe72ee322133e |

### 11.2 九个正式 task summary

| Condition | Task | Success | Absolute summary path | Bytes | SHA256 |
| --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 31/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_runner/summaries/rgbd_gp__one_leg.json | 4455718 | 65aeec7b3acaf2406db407bbef1c25591ca449fa8c5b259b9e75f593e890fdfd |
| rgbd+GP | round_table | 16/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_runner/summaries/rgbd_gp__round_table.json | 6957177 | e130f24cf9e5784dc16b38c2824d06500a2777ae44e259832dda4c69973bee9b |
| rgbd+GP | lamp | 16/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_recovery_rgbd_gp_lamp_runner/summaries/rgbd_gp__lamp.json | 6007365 | f9fd7579a812fd8720359e4b5aa25ff9ec82f77c29265907b094a08f4753a20f |
| rgbd+colored GP | one_leg | 31/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_recovery_remaining_runner/summaries/rgbd_colored_gp__one_leg.json | 4436714 | cc35515b2836b17917409953411c0246eed594c17ea52feebe10c2c78a808b03 |
| rgbd+colored GP | round_table | 15/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_recovery_remaining_runner/summaries/rgbd_colored_gp__round_table.json | 6831463 | d10c46ede4e2c4250247e2e00627bfcbf8059192a140119f5fbe7ee38e9e617d |
| rgbd+colored GP | lamp | 18/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_colored_lamp_timeout120_runner/summaries/rgbd_colored_gp__lamp.json | 5847092 | 747accb8a1efe88df7fbc43ca6a237f6cc4f80515b1c047ce5e3fe329916a189 |
| rgbd+GP+skill | one_leg | 30/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_gp_skill_timeout120_runner/summaries/rgbd_gp_skill__one_leg.json | 4456167 | 9c50a7dd82e583e1a4785ec2b91b5d2ca29aa9b289e0aab882828692285cdad1 |
| rgbd+GP+skill | round_table | 10/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_gp_skill_timeout120_runner/summaries/rgbd_gp_skill__round_table.json | 7263329 | 07287779f47190d55e2bbc57da75d311565ead18a48a21cec2fa07a88b3c0ebc |
| rgbd+GP+skill | lamp | 14/36 | /data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_ckpt_new_9d36062_20260822/formal_gp_skill_timeout120_runner/summaries/rgbd_gp_skill__lamp.json | 5899252 | 6f379ca00cfd22a79d2705ee28ce7f1eac660807234b1770833a9a350d60b43c |

### 11.3 派生 CSV 与生成器

- task-level：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_task.csv`
- condition overall：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_overall.csv`
- cross-task skill：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_skill.csv`
- task × skill tracking：`/data/hy/robust-rearrangement/reports/data/vlm_dit_tracking_by_task_skill.csv`
- task × skill point：`/data/hy/robust-rearrangement/reports/data/vlm_dit_point_error_by_task_skill.csv`
- generator：`/data/hy/robust-rearrangement/scripts/generate_vlm_dit_report.py`；SHA256 `e475dbfdffea7bacec41b49f400455ab1c108838c9f27f7b203be57b5522ced5`。
- 所有 CSV 与 Markdown 均由同一次 generator 调用覆盖写出；重新生成命令见第 9 节。
