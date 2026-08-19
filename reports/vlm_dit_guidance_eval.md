# VLM + DiT guidance point 评测报告

## 1. 实验状态

- 已完成 task-level 实验：`9/9`。
- 已完成 rollout：`324/324`。
- VLM：`http://10.71.106.240:8000`；revision：`933f15ce0ed0bc7108ec1f42074bf94d985a4cbf`。
- 原始 manifest：`/data/hy/robust-rearrangement/logs/vlm_dit_eval_20260813_approved_mc200_low/manifest.json`。
- 阶段：`legacy`；设计：3 个 condition × 3 个 task × 每格 36 rollout，共 324 rollout；每批 3 个并行环境。
- task 截止步数：one_leg=1000，round_table=1000，lamp=1000；randomness=low。
- 每个有效控制 step 的 GT/VLM 点对、每个 n0--n4 档位使用 `200` 个 3D Monte Carlo 投影样本。

> **旧 324-rollout 数据已判定 invalid：只保留作故障溯源，不得 resume、拼接或用于 condition 排名。**
>
> 原因：旧 324 个 VLM rollout 和旧 scripted-GT 对照均缺少 --save-depth-image；当前 evaluate_model.py 会因此不把 depth_image1/depth_image2 加入 RGBD policy observation，所以旧结果不能用于判断 checkpoint、VLM 或 condition 优劣。
> 2026-08-20 更新：新 `original_sft` 服务已完成三任务离线评测和各 1 条真实
> rollout 的人工视频审查。用户指定下一轮不再跑 27-rollout smoke，工作区审批后
> 直接运行新 324-rollout formal 矩阵。launcher 必须用显式 bypass 参数记录这次授权，
> 仍保留命令展开、checkpoint/depth/VLM revision 的 fail-fast 审计。

### 1.1 one_leg scripted-GT 诊断

| Run | Code | n_envs | Max steps | Success | Main failure |
| --- | --- | --- | --- | --- | --- |
| icy-3000 current code, local 3060 | 371a783 plus VLM worktree changes | 2 | 1000 | 0.0% (0/10) | leg-top-place |
| icy-3000 clean HEAD, remote 4090 | 371a783 | 3 | 1000 | 0.0% (0/3) | leg-top-place |
| icy-2500 clean HEAD, remote 4090 | 371a783 | 3 | 1000 | 0.0% (0/3) | leg-top-place |
| icy-3000 June-15 stack, remote 4090 | 5554145 with furniture-bench 71392f3 | 3 | 1000 | 0.0% (0/3) | leg-top-place |
| autumn latest-3000 clean HEAD, remote 4090 | 371a783 | 3 | 1000 | 8.3% (1/12) | leg-top-place for eleven rollouts |
| autumn best-val-3000 clean HEAD, remote 4090 | 371a783 | 3 | 1000 | 0.0% (0/3) | leg-top-place |
| fresh-tree latest-3000 clean HEAD, remote 4090 | 371a783 | 3 | 1000 | 33.3% (1/3) | leg-top-place for two rollouts |

已排除：


旧数据判定文件：`/data/hy/robust-rearrangement/reports/data/vlm_dit_baseline_gate.json`。新评测保留 smoke gate 能力，本轮 formal 按 2026-08-20 的用户明确授权记录 bypass。

## 2. Success rate

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 0.0% (0/36) | 25.0% (9/36) | 0.0% (0/36) | 8.3% (9/108) |
| rgbd+colored GP | 8.3% (3/36) | 33.3% (12/36) | 0.0% (0/36) | 13.9% (15/108) |
| rgbd+GP+skill | 8.3% (3/36) | 5.6% (2/36) | 0.0% (0/36) | 4.6% (5/108) |

## 3. Tracking error（clean GT pose）

每格为 `position cm / orientation deg / total (n)`；`total = pos_m / 0.01 + ori_deg / 5`，越低越好。VLM 只替换 policy 的 skill/2D point，shadow 自动机提供 clean guidance pose 作为共同 tracking target。

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 17.36/35.01/24.36 (n=36) | 13.83/79.69/29.77 (n=36) | 13.55/62.76/26.11 (n=36) | 14.91/59.15/26.74 (n=108) |
| rgbd+colored GP | 17.20/37.87/24.78 (n=36) | 13.91/80.62/30.04 (n=36) | 12.69/57.98/24.29 (n=36) | 14.60/58.82/26.37 (n=108) |
| rgbd+GP+skill | 15.91/36.23/23.15 (n=36) | 14.18/79.67/30.12 (n=36) | 13.39/62.70/25.93 (n=36) | 14.49/59.53/26.40 (n=108) |

## 4. VLM 打点误差

逐 step 误差定义为 front camera 上 `||p_vlm - p_gt||₂`。缓存期间每个控制 step 都计入，因此这里直接给出你要求的 step average；投影参考也为每个有效控制 step 的 GT/VLM 点对单独生成，从而包含 action horizon 内 GT 移动与 VLM 点缓存造成的实际误差。

| Condition | one_leg | round_table | lamp | Overall |
| --- | --- | --- | --- | --- |
| rgbd+GP | 23.09/28.67 (n=23231) | 32.33/41.66 (n=31896) | 61.58/67.62 (n=32589) | 40.75/50.47 (n=87716) |
| rgbd+colored GP | 25.07/29.43 (n=24156) | 30.09/39.99 (n=31592) | 56.36/59.96 (n=33661) | 38.62/46.39 (n=89409) |
| rgbd+GP+skill | 28.85/36.56 (n=20926) | 34.08/46.68 (n=35402) | 61.25/66.11 (n=34013) | 43.10/53.00 (n=90341) |

表中每格为 `step mean px / step RMSE px (有效 step 数)`。

### 4.1 Each skill step average

| Condition | Skill | Mean px | RMSE px | Steps | Same-depth mean mm |
| --- | --- | --- | --- | --- | --- |
| rgbd+GP | push | 30.32 | 32.03 | 4721 | 65.39 |
| rgbd+GP | pick | 45.97 | 55.02 | 71429 | 105.05 |
| rgbd+GP | place | 14.05 | 15.33 | 7551 | 27.79 |
| rgbd+GP | insert | 9.97 | 11.13 | 159 | 20.54 |
| rgbd+GP | screw | 10.32 | 12.06 | 3856 | 21.01 |
| rgbd+colored GP | push | 34.63 | 35.78 | 9717 | 74.29 |
| rgbd+colored GP | pick | 43.53 | 50.93 | 68610 | 97.16 |
| rgbd+colored GP | place | 13.20 | 14.82 | 7227 | 26.17 |
| rgbd+colored GP | insert | 10.35 | 11.65 | 283 | 21.47 |
| rgbd+colored GP | screw | 8.87 | 10.44 | 3572 | 18.72 |
| rgbd+GP+skill | push | 36.57 | 37.70 | 5794 | 77.54 |
| rgbd+GP+skill | pick | 48.89 | 58.22 | 71006 | 130.22 |
| rgbd+GP+skill | place | 14.14 | 17.72 | 7565 | 27.80 |
| rgbd+GP+skill | insert | 11.48 | 13.18 | 248 | 24.72 |
| rgbd+GP+skill | screw | 17.50 | 20.66 | 5728 | 38.08 |

## 5. VLM 对应 n0–n4 的哪个等级？

不能把 2D 像素误差直接除以一个固定 px/mm 系数，再与 3D 的 0/3/6/12/24 mm 比较。透视投影尺度随 GT 点深度、相机内参和偏移方向变化。主分析采用同坐标系比较：

1. 每个有效控制 step 形成一个配对样本：自动机给出当步 3D GT guidance point `P_gt`，同一 annotation util 给出 front-camera GT pixel `p_gt` 和相机内外参，实际送给 policy 的 VLM 点为 `p_vlm`。VLM 在 action horizon 内可以缓存，但每个当步 GT/VLM 点对都独立进入 step average 和投影分布。
2. 用由 `(episode, env, step, query_step)` 确定的 seed 采样 `200` 个 `z_j ~ N(0, I_3)`，随后逐分量 clip 到 `[-2, 2]`。这与现有 `annotation_noise.py` 完全一致。严格来说 clip 后的边际方差小于 1；这里的 n1--n4 名称和 σ 参数沿用原噪声实验，而不是声称截断后的实际标准差仍恰好等于 σ。
3. 五档使用同一组 `z_j`（common random numbers）以减小档位间 Monte Carlo 抖动。令 `σ_n ∈ {0, 3, 6, 12, 24} mm`，构造 `P_nj = P_gt + σ_n z_j`。
4. 使用 `skill_annotation_util.py` 相同的 sim-local→camera 变换、camera-y 翻转和内参投影，计算连续坐标 `e_nj = π(P_nj) - π(P_gt)`；VLM 残差为 `e_vlm = p_vlm - p_gt`。参考噪声投影不做整数取整，也不按图像边界裁剪，否则会人为压缩尾部。
5. 对每档全部投影样本精确累计一阶、二阶矩：`μ_n = (1/M)Σe_nj`，`Σ_n = (1/M)Σ(e_nj-μ_n)(e_nj-μ_n)^T`，并由全部样本计算 projected RMSE。VLM 的 mean/cov/RMSE 在所有有效 step 残差上计算，和 reference 的相机/深度条件完全配对。
6. 用 projected RMSE 在相邻 n 档之间线性插值得到主要“误差量级”，例如 `n1–n2`；超过 n4 时线性外推并标记 `>n4`。径向 W1 最近档作为大小分布的稳健交叉检查。
7. 完整 2D SWD 用于判断包含系统偏置的二维分布是否像某档噪声；centered SWD 在双方各自减均值后比较形状。它们不替代第 6 步的误差大小分类。

实现的性能策略：每个点对实际生成 `200` 个样本；mean/cov/RMSE 用全部样本的 sufficient statistics 精确合并；每档仅保留均匀抽取的 2000 个 reference residual 计算分位数、32-direction SWD、W1 和 KS，使 rollout summary 大小与后处理耗时有上界。在更接近最大单批规模的 3000-pair、200-sample 合成基准中，600,000 个样本/档的投影约 1.76 s，三 scope 汇总约 11.06 s，进程峰值 RSS 约 958 MiB；本机约有 19 GiB 可用内存，因此正式配置从 100 提高到 200。该数字只是性能基准，不是实验结果；正式运行仍记录实际 wall time。

以下第一张表按 condition 跨三个 task、按有效 step 聚合。

| Condition | Valid pairs | VLM RMSE px | P90 px | P95 px | Bias px | Anisotropy | Same-depth mm | Magnitude bracket | Equivalent σ mm | Closest radial W1 | Closest full SWD | Closest centered SWD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | 87716 | 50.47 | 80.21 | 101.33 | 25.02 | 4.69 | 92.42 | >n4 | 80.82 | n4 | n4 | n4 |
| rgbd+colored GP | 89409 | 46.39 | 78.34 | 83.11 | 25.74 | 3.28 | 85.56 | >n4 | 73.35 | n4 | n4 | n4 |
| rgbd+GP+skill | 90341 | 53.00 | 82.17 | 97.35 | 28.78 | 3.25 | 112.13 | >n4 | 84.85 | n4 | n4 | n4 |

### 5.1 每个 task 的误差量级

task-level 映射是主要诊断表；它避免不同任务的 GT 深度和透视尺度在总体聚合中互相抵消。

| Condition | Task | Valid pairs | VLM RMSE px | P90 px | Bias px | Magnitude bracket | Equivalent σ mm | Closest radial W1 | Closest full SWD | Closest centered SWD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | one_leg | 23231 | 28.67 | 49.27 | 13.78 | >n4 | 49.26 | n4 | n4 | n4 |
| rgbd+GP | round_table | 31896 | 41.66 | 78.49 | 11.82 | >n4 | 67.93 | n4 | n4 | n4 |
| rgbd+GP | lamp | 32589 | 67.62 | 102.79 | 48.15 | >n4 | 101.95 | n4 | n4 | n4 |
| rgbd+colored GP | one_leg | 24156 | 29.43 | 43.89 | 17.70 | >n4 | 47.13 | n4 | n4 | n4 |
| rgbd+colored GP | round_table | 31592 | 39.99 | 79.86 | 13.12 | >n4 | 67.02 | n4 | n4 | n4 |
| rgbd+colored GP | lamp | 33661 | 59.96 | 82.91 | 44.81 | >n4 | 89.50 | n4 | n4 | n4 |
| rgbd+GP+skill | one_leg | 20926 | 36.56 | 56.72 | 15.55 | >n4 | 60.12 | n4 | n4 | n4 |
| rgbd+GP+skill | round_table | 35402 | 46.68 | 80.47 | 16.26 | >n4 | 79.31 | n4 | n4 | n4 |
| rgbd+GP+skill | lamp | 34013 | 66.11 | 95.08 | 50.90 | >n4 | 98.74 | n4 | n4 | n4 |

### 5.2 与各档噪声分布的距离（condition overall）

| Condition | Level | Full 2D SWD px | Centered SWD px | Radial W1 px | Bias gap px | Radial KS | Projected RMSE px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rgbd+GP | n0 | 25.94 | 23.41 | 40.75 | 25.02 | 1.000 | 0.00 |
| rgbd+GP | n1 | 25.10 | 22.38 | 39.09 | 25.02 | 0.981 | 1.87 |
| rgbd+GP | n2 | 24.29 | 21.34 | 37.43 | 25.02 | 0.942 | 3.74 |
| rgbd+GP | n3 | 22.77 | 19.28 | 34.11 | 25.02 | 0.779 | 7.48 |
| rgbd+GP | n4 | 20.27 | 15.18 | 27.44 | 25.03 | 0.557 | 14.97 |
| rgbd+colored GP | n0 | 24.59 | 21.89 | 38.62 | 25.74 | 1.000 | 0.00 |
| rgbd+colored GP | n1 | 23.72 | 20.85 | 36.95 | 25.74 | 0.987 | 1.89 |
| rgbd+colored GP | n2 | 22.89 | 19.80 | 35.28 | 25.74 | 0.934 | 3.78 |
| rgbd+colored GP | n3 | 21.31 | 17.72 | 31.94 | 25.74 | 0.753 | 7.57 |
| rgbd+colored GP | n4 | 18.92 | 13.60 | 25.25 | 25.74 | 0.551 | 15.17 |
| rgbd+GP+skill | n0 | 27.43 | 25.13 | 43.10 | 28.78 | 1.000 | 0.00 |
| rgbd+GP+skill | n1 | 26.55 | 24.10 | 41.46 | 28.78 | 0.971 | 1.87 |
| rgbd+GP+skill | n2 | 25.68 | 23.08 | 39.81 | 28.78 | 0.908 | 3.74 |
| rgbd+GP+skill | n3 | 24.05 | 21.04 | 36.53 | 28.77 | 0.780 | 7.48 |
| rgbd+GP+skill | n4 | 21.34 | 16.96 | 29.94 | 28.77 | 0.585 | 14.98 |

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
- `rgbd+GP` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-24-37.790086/models/autumn-dust-13/actor_chkpt_latest_3000.pt`。
- `rgbd+colored GP` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-18/14-59-28.908152/models/absurd-voice-2_2026-06-18_14-59-48.700671/actor_chkpt_latest_3000.pt`。
- `rgbd+GP+skill` checkpoint：`/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/12-55-43.621615/models/fresh-tree-11_2026-06-13_12-56-10.422936/actor_chkpt_latest_3000.pt`。
- Tasks：`one_leg`、`round_table`、`lamp`；每个 condition-task 36 rollout，共 `3×3×36=324`。
- 本地并行：`n_envs=3`；三个 task 上限均为 1000 step；`randomness=low`。
- VLM：固定 readiness 中的 model revision；每次正式启动先 fail-fast 检查 `status=ready`、`policy_version=3`、`model_mode=original_sft`。HTTP timeout 30 s，失败或 schema/revision 不一致直接终止，不 fallback 到自动机。
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
- 正式运行将创建新的 timestamp output-dir；旧的中断 rollout 不纳入任何表格或结论。

### 7.4 正式运行配置

已批准的固定项：`randomness=low` 独立 reset、`n_envs=3`、36 rollout/格、query horizon=8、200 MC 样本/点/档、2000 reference reservoir/档、32-direction SWD、三个 task 的 max steps 均为 1000，以及 clean-GT pose tracking target。正式矩阵共 324 rollout。

## 8. 当前结论

旧 324-rollout 因缺少 --save-depth-image（且 one_leg 步数和 rgbd+GP checkpoint 不一致）而作废；只保留为故障诊断证据，不对三个 condition 排名。

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
  --guidance-point-on-image --no-annotate-skill \
  --tracking-metric-type pose \
  --annotation-source vlm --vlm-base-url "$VLM_GUIDANCE_URL" \
  --vlm-timeout-seconds 30 --vlm-query-interval 0 \
  --vlm-noise-projection-samples 200 \
  --task-summary-out logs/vlm_dit_single/rgbd_gp__one_leg.json \
  --rollout-suffix-model-name vlm_dit_single/rgbd_gp/one_leg
```

完整矩阵由 gated runner 先做 print-command 审计，再执行。本轮根据用户明确指示跳过 27-rollout smoke；使用 `--allow-formal-without-smoke` 时必须同时写入审批说明，并由 manifest 记录，不会默默关闭 gate。

```bash
python3 scripts/run_vlm_dit_eval.py --phase print --stage formal \
  --namespace vlm_original_sft_formal_mc200_low_20260820 \
  --output-dir logs/vlm_original_sft_formal_mc200_low_20260820 \
  --data-dir-raw /data/hy/robust-rearrangement/data \
  --allow-formal-without-smoke \
  --formal-approval-note "User approved direct formal after three-task original_sft preview on 2026-08-20"
```

只重新生成报告：

```bash
python scripts/generate_vlm_dit_report.py --manifest /data/hy/robust-rearrangement/logs/vlm_dit_eval_20260813_approved_mc200_low/manifest.json --output /data/hy/robust-rearrangement/reports/vlm_dit_guidance_eval.md
```

## 10. 原始导出

- task-level CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_task.csv`
- condition overall CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_overall.csv`
- each-skill step average CSV：`/data/hy/robust-rearrangement/reports/data/vlm_dit_guidance_by_skill.csv`
- source manifest：`/data/hy/robust-rearrangement/logs/vlm_dit_eval_20260813_approved_mc200_low/manifest.json`
- 不再保留冗余的 report-side manifest copy；以 `logs/.../manifest.json` 为唯一源。
