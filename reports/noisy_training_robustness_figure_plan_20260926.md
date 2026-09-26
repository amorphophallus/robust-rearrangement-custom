# Noisy-training robustness：最终图表方案

日期：2026-09-26

当前图对应单 checkpoint 实验：full rollout 每个 `Train×Eval×Task` 12 次，state bank 为 `n=24,m=1`。N3–N6（每轴噪声标准差 12–96 mm）作为 high-noise 区间统一标灰。

## 最终保留图

![Noisy-training robustness curves](./figures/noisy_train_noisy_eval/noisy_training_robustness_curves.png)

三联图只保留原 Figure 1 和原 Figure 2 的 B、C：

- A，full-rollout overall task SR：直接展示 noisy training 的端到端抗噪收益以及 clean-performance trade-off。N3–N6 mean SR 从 Train N0 的 50.0% 提高到 Train N2/N4 的 60.4%/62.5%；worst-case SR 从 38.9% 提高到 55.6%。
- B，state-bank `E_GT`：在相同 expert state 下测量 noisy rollout 终点到 clean GT 的距离。N0 随 eval noise 增大明显恶化，而 N2/N4 更稳定；高噪声平均相对 N0 分别降低 9.4%/10.8%。
- C，state-bank `Δx`：测量 noisy rollout 和同 state clean rollout 的行为终点位移。高噪声平均相对 N0 分别降低 21.4%/18.0%，说明 noisy training 降低了动作对错误 guidance 的敏感度。

这三个 panel 对应完整论证链：局部 clean-target error 更小、局部行为偏移更小，最终转化为长序列任务在高噪声下更高的 overall success。

## 建议图注

> Noisy training trades some clean-condition accuracy for improved robustness to corrupted guidance. From fixed expert states, N2/N4 training reduces clean-target error and behavior displacement as evaluation noise increases; in full rollouts, these local gains translate into higher mean and worst-case task success under high guidance noise. The shaded region denotes N3–N6 (12–96 mm standard deviation per axis).

## 结论边界

- 当前是单 checkpoint/condition，full rollout 每 task 12 次、state bank `m=1`，图中是点估计；三 seed 版本应改为 mean 和跨 seed 区间。
- Overall full-rollout 收益主要由 round table 驱动；one leg 接近饱和，lamp 没有提升。这个 task heterogeneity 必须在正文或附录中说明。
- Stage SR 有 ceiling effect，`E_input` 需要与 `E_GT` 联合解释，而 `G_parallel` 与 `P(G_parallel>0)` 当前信号较弱，因此不放入主图。

## 可追溯性

- 可复现绘图脚本：`reports/figures/noisy_train_noisy_eval/build_noisy_training_robustness_curves.py`
- 图中逐点数值及全部输入文件 SHA-256：`reports/figures/noisy_train_noisy_eval/noisy_training_robustness_curves_data.json`
- PNG：`reports/figures/noisy_train_noisy_eval/noisy_training_robustness_curves.png`
- PDF：`reports/figures/noisy_train_noisy_eval/noisy_training_robustness_curves.pdf`
- Full-rollout 输入：`logs/noisy-train-noisy-eval-20260925/full-rollout-pilot12/summaries/<train>/<seed>/<eval>/<task>.json`
- State-bank 输入：`logs/noisy-train-noisy-eval-20260925/state-bank-n24-m1-combined-v1/paired_summary.json`
