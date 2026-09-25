# RGB Extension 与 Main RGB-D FurnitureBench 对比登记

更新时间：2026-09-25（Asia/Shanghai）

## 1. 登记范围与统计口径

本文登记 FurnitureBench `one_leg`、`round_table`、`lamp` 三任务上的 RGB Extension 结果，并与 main experiment 最新有效 RGB/RGB-D 结果并列。根据当前报告范围，`skill` 和 `GP + skill` 不进入主表。

两类结果的统计协议不同，表中必须保留 `n_seed`：

- RGB Extension 的四个带 condition 方法使用 train seed `2026091801/02`，每个 checkpoint/task 评估 24 个 rollout；每个 task 的 `±` 是两个 train seed 之间的 sample standard deviation。
- Main experiment 的 RGB、RGB-D 及 RGB-D+condition 使用三个登记 train seed，每个 checkpoint/task 评估 36 个 rollout；`±` 是三个 train seed 之间的 sample standard deviation。
- 每个 train seed 的 Overall 先汇总三个 task 的成功数，再除以 `3 × n_rollouts_per_task`；主表中的 Overall 是 train-seed Overall 的 mean ± sample std。
- 因 rollout 数、train lineage 和注册规则不同，本表用于结果登记和横向观察，不把 RGB 与 RGB-D 的小百分点差异解释为严格 paired causal effect。

Main experiment 数值冻结于 RR `origin/main` commit `acae6da2813412f393376b0730516ce58a68af9b`，机器可读注册表为 [`main_experiment_registered_20260925.csv`](data/main_experiment_registered_20260925.csv)。

## 2. 主表

| Condition | Modality | n_seed | one_leg | round_table | lamp | Overall |
|---|---|---:|---:|---:|---:|---:|
| No condition | RGB | 3 | 84.26 ± 4.24% | 41.67 ± 8.33% | 18.52 ± 4.24% | 48.15 ± 3.34% |
| No condition | RGB-D | 3 | 82.41 ± 1.60% | 49.07 ± 1.60% | 25.93 ± 5.78% | 52.47 ± 1.93% |
| GP | RGB | 2 | 85.42 ± 2.95% | 45.83 ± 0.00% | 14.58 ± 2.95% | 48.61 ± 1.96% |
| GP | RGB-D | 3 | 85.19 ± 6.99% | 52.78 ± 15.47% | 33.33 ± 7.35% | 57.10 ± 5.10% |
| Colored GP | RGB | 2 | 87.50 ± 5.89% | 45.83 ± 0.00% | 10.42 ± 2.95% | 47.92 ± 2.95% |
| Colored GP | RGB-D | 3 | 83.33 ± 2.78% | 58.33 ± 12.11% | 34.26 ± 1.60% | 58.64 ± 4.57% |
| Grasp part | RGB | 2 | 85.42 ± 8.84% | 8.33 ± 0.00% | 12.50 ± 5.89% | 35.42 ± 0.98% |
| Grasp part | RGB-D | 3 | 87.04 ± 4.24% | 14.81 ± 20.85% | 37.04 ± 5.78% | 46.30 ± 8.49% |
| Colored grasp part | RGB | 2 | 85.42 ± 2.95% | 16.67 ± 11.79% | 14.58 ± 8.84% | 38.89 ± 7.86% |
| Colored grasp part | RGB-D | 3 | 87.04 ± 8.93% | 22.22 ± 19.25% | 33.33 ± 2.78% | 47.53 ± 5.58% |

## 3. RGB Extension 逐 seed 原始计数

| Condition | Train seed | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|---:|
| `rgb_gp` | `2026091801` | 21/24 | 11/24 | 4/24 | 36/72（50.00%） |
| `rgb_gp` | `2026091802` | 20/24 | 11/24 | 3/24 | 34/72（47.22%） |
| `rgb_colored_gp` | `2026091801` | 22/24 | 11/24 | 3/24 | 36/72（50.00%） |
| `rgb_colored_gp` | `2026091802` | 20/24 | 11/24 | 2/24 | 33/72（45.83%） |
| `rgb_grasp_part` | `2026091801` | 22/24 | 2/24 | 2/24 | 26/72（36.11%） |
| `rgb_grasp_part` | `2026091802` | 19/24 | 2/24 | 4/24 | 25/72（34.72%） |
| `rgb_grasp_part_colored` | `2026091801` | 21/24 | 6/24 | 5/24 | 32/72（44.44%） |
| `rgb_grasp_part_colored` | `2026091802` | 20/24 | 2/24 | 2/24 | 24/72（33.33%） |

### 3.1 固定评估协议

四个 RGB Extension condition 复用同一冻结 runner：

- checkpoint：最终 `actor_chkpt_last.pt`，epoch 2999 / step 300000；不使用 best checkpoint。
- `n_envs=12`，`n_rollouts=24`，eval seed `0`，randomness `low`，最多 1000 environment steps。
- action type `pos`，observation space `image`，EE pose/action frame 为 `robot-base`。
- wrist RGB 为 `240×320 → center-crop-224 → 224×224`，不做非等比 resize；纯 RGB 的 depth contract 为 `not_applicable`。
- annotation source 为 `scripted`，开启 annotation verify；front camera preset 为 `original`。
- success 使用 simulator `physics_reward`，tracking metric 为 position，并要求 tracking summary complete。
- 冻结代码：root `9dff9a4f8d9287345511a287a4ba88ff80954fda`，FurnitureBench `7d57047689fe86f24ad3ba362693ce3d34b19e9d`。

## 4. Main experiment 三 seed 具体数据

以下数值逐行来自最新注册 CSV，而不是从旧版 Markdown 手工抄录。

| Condition | Modality | Registered train seed / run | one_leg | round_table | lamp | Overall | Registration rule |
|---|---|---|---:|---:|---:|---:|---|
| No condition | RGB | `2026090701` | 32/36 | 15/36 | 8/36 | 55/108 | per-task min of eval seed 0/1 |
| No condition | RGB | `2026090702` | 30/36 | 18/36 | 5/36 | 53/108 | per-task min of eval seed 0/1 |
| No condition | RGB | `2026091701` | 29/36 | 12/36 | 7/36 | 48/108 | per-task min of eval seed 0/1 |
| No condition | RGB-D | `2026090701` | 30/36 | 18/36 | 7/36 | 55/108 | per-task min of eval seed 0/1 |
| No condition | RGB-D | `2026090702` | 29/36 | 17/36 | 10/36 | 56/108 | per-task min of eval seed 0/1 |
| No condition | RGB-D | `2026091701` | 30/36 | 18/36 | 11/36 | 59/108 | per-task min of eval seed 0/1 |
| GP | RGB-D | `2026092201` | 33/36 | 24/36 | 10/36 | 67/108 | single formal36 |
| GP | RGB-D | `autumn-dust-13` | 28/36 | 13/36 | 15/36 | 56/108 | single formal36 |
| GP | RGB-D | `icy-vortex-9` | 31/36 | 20/36 | 11/36 | 62/108 | single formal36 |
| Colored GP | RGB-D | `2026092201` | 29/36 | 19/36 | 13/36 | 61/108 | round_table max over two formal36 trials |
| Colored GP | RGB-D | `2026090701` | 31/36 | 26/36 | 12/36 | 69/108 | single aligned formal36 |
| Colored GP | RGB-D | `2026090702` | 30/36 | 18/36 | 12/36 | 60/108 | single aligned formal36 |
| Grasp part | RGB-D | `morning-glitter-1` | 31/36 | 14/36 | 15/36 | 60/108 | registered legacy formal36 |
| Grasp part | RGB-D | `2026090701` | 33/36 | 1/36 | 14/36 | 48/108 | per-task max of history/aligned |
| Grasp part | RGB-D | `2026090702` | 30/36 | 1/36 | 11/36 | 42/108 | per-task max of history/aligned |
| Colored grasp part | RGB-D | `eternal-cosmos-2` | 29/36 | 16/36 | 12/36 | 57/108 | registered legacy formal36 |
| Colored grasp part | RGB-D | `2026090701` | 30/36 | 4/36 | 11/36 | 45/108 | per-task max of history/aligned |
| Colored grasp part | RGB-D | `2026090702` | 35/36 | 4/36 | 13/36 | 52/108 | per-task max of history/aligned |

### 4.1 Main 注册规则边界

- `rgbd_gp/2026092201` 替换旧 GP 三 seed 中的 `rare-monkey-4`；其余两项为 `autumn-dust-13`、`icy-vortex-9`。
- `rgbd_colored_gp/2026092201/round_table` 的两次同协议 formal36 为 15/36 与 19/36，按既定注册规则取 19/36；selection receipt 已归档。
- RGB/RGB-D 的三个 train seed 都逐 task 取 eval seed 0/1 的较小成功数。
- 两个 RGB-D grasp condition 的 supplemental seed 逐 task 取历史与 aligned 评估的较大成功数。该口径是既有登记规则，不等同于统一单次 estimator。

## 5. Base 可审计来源索引

统一 bundle 根目录：

```text
base:/home/huyue/projects/robust-rearrangement-custom/logs/fb-main-rgbext-eval-0925
```

### 5.1 RGB Extension

| 内容 | Base 相对路径 |
|---|---|
| 24 个正式结果 JSON | `formal24/seed<TRAIN_SEED>/<CONDITION>/<TASK>.json` |
| 24 个 cell 完成标记 | `formal24/seed<TRAIN_SEED>/<CONDITION>/<TASK>.COMPLETE` |
| GP / colored-GP / grasp-part 队列日志 | `queue/formal24-seed12/queue.log` |
| colored grasp-part 补充队列日志 | `queue/formal24-rgb-grasp-part-colored-seed12/queue.log` |
| checkpoint SHA | `provenance/FINAL_CHECKPOINTS.sha256`、`provenance/RGB_GRASP_PART_COLORED_ADDENDUM.sha256` |
| 冻结代码 receipt | `provenance/eval-code-ready-v2.json` |
| 单 cell runner 与队列 | `provenance/run_rgbext_cell_bch.sh`、`provenance/queue_rgbext_formal24_seed12_bch.sh`、`provenance/prepare_and_queue_rgbext_colored_grasp_0925.sh` |

每个结果 JSON 都记录 `n_success`、`n_rollouts`、完整 eval command、resolved annotation config、训练配置、相机与 depth contract、tracking completion 和 checkpoint 路径。JSON 只有在合同检查通过后才生成对应 `.COMPLETE`。

### 5.2 Main experiment 注册输入

| 条件/来源 | Base 相对路径 |
|---|---|
| 最新机器可读注册表快照 | `provenance/main_experiment_registered_20260925.csv` |
| 最新 main 三 seed 报告快照 | `provenance/main_3seed_experiment_review_0913.md` |
| GP / colored-GP `2026092201` 与 colored-GP selection receipt | `main_registered_inputs/r218/main-gp-newseed-0924/` |
| supplemental aligned 结果 | `main_registered_inputs/r218/main-supplement-reeval-0914/results/` |
| RGB/RGB-D 独立 eval seed 1 结果 | `main_registered_inputs/r218/rgb-rgbd-highscore-reeval-0914/results/` |
| RGB/RGB-D `2026091701` canonical 结果 | `main_registered_inputs/r218/main3seed-rgb-rgbd-reeval-0921/formal36/results/` |
| RGB-D grasp 历史输入 | `main_registered_inputs/r218/rgbd-depthfix-0906/main_supplement_eval/`、`main_registered_inputs/r218/main-supplement-0906/` |
| GP 原 main 精确 JSON | `main_registered_inputs/base/logs/evaluate_model/` |
| grasp 原 main 精确 JSON | `main_registered_inputs/base/evaluate_model/` |
| `2026091701` conservative-min receipt 与哈希 | `main_registered_inputs/base/logs/main-retrain-corrected-repeat-seed1-0921/` |

### 5.3 Bundle 完整性

- `BUNDLE_COMPLETE`：登记 bundle 的生成时间和计数；当前为 24 JSON、24 COMPLETE、127 个 main 注册输入文件。
- `MANIFEST.sha256`：bundle 中除清单自身外所有文件的 SHA-256；当前包含 198 条记录。

## 6. 当前结论边界

- RGB-D GP 与 colored GP 在 lamp 上明显高于当前两 seed RGB 对应项，但两组并非相同 train lineage，也不是相同 rollout 数，不能只凭该表归因于 depth。
- 两种 grasp-part 表示都在 round_table 上显著低于 GP；这一模式在 RGB 与 RGB-D 中方向一致，但 RGB-D grasp 使用历史逐格 max，绝对值不应与 RGB formal24 当作严格同协议估计量。
- RGB Extension 当前只有两个完成 train seed。第三 seed 的 checkpoint 在本轮开始时仍是 partial，因此没有下载、评估或进入均值。
