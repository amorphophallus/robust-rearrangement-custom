# Noisy-train × noisy-eval TAGPoint 实验计划（2026-09-25）

## 1. 研究问题与可支持的结论

目标是回答：在训练引导点含何种噪声时，TAGPoint 能在尽量保留 clean 性能的同时，最大程度抵抗测试时的标注误差？比较三种训练条件：N0、N2、N4。按已确认实验假设，三者来自相同的任务/示范分布，可直接比较；报告仍保留每份数据、checkpoint 和 seed 的真实 provenance，不把“同分布”写成“逐轨迹相同”。

两套评估回答不同问题：

- full rollout：从任务初始状态执行，测任务 SR、stage reach/completion，包含 compounding error；
- state bank：从相同 expert 中间状态恢复，只测当前阶段的局部操控与 spatial grounding，去除前序累计误差。

只有两套结果共同支持以下结论时，才称某训练噪声“更 robust”：它在 N1–N6 的任务/阶段成功率或 `E_GT` 上优于 N0-train，同时 N0-eval 的性能损失可接受。`E_input`、`Δx`、`G` 用来解释 policy 是跟随噪声、隐式去噪还是失稳，不能单独作为 robustness 排名。

## 2. 噪声轴与配对规则

| Level | 每轴位置标准差 | 采样方式 |
|---|---:|---|
| N0 | 0 mm | clean |
| N1 | 3 mm | clipped Gaussian |
| N2 | 6 mm | clipped Gaussian |
| N3 | 12 mm | clipped Gaussian |
| N4 | 24 mm | clipped Gaussian |
| N5 | 48 mm | clipped Gaussian |
| N6 | 96 mm | clipped Gaussian |

所有非零 level 使用 `Gaussian(0, σ²)` 后逐轴裁剪到 `[-2σ, 2σ]`。噪声在当前 semantic stage 内固定。state-bank 中，标准噪声 `z` 由 `(state_sha256, repeat_seed, annotation_noise_seed)` 唯一确定；同一 state/repeat 在所有 policy 和 N1–N6 共享 `z`，仅使用不同的 `σ`，因而 `δ=σz` 可严格配对。N0 与 noisy rollout 使用同一 state、repeat seed 和 policy checkpoint。

固定 annotation noise base seed 为 `2026092501`。原始记录必须保存 checkpoint hash、state manifest hash、`p0`、`pδ`、`δ`、`x0`、`xδ`、终止原因、FSM 转移、source episode 和 selection stratum。

## 3. State-bank 指标（必须实际计算）

令：

- `p0`：noisy rollout 终止帧的 clean GT 引导点；
- `pδ=p0+δ`：同一帧实际输入 policy 的 noisy 引导点；
- `x0`：同一 checkpoint、state、repeat 在 N0 下的终点 EE position；
- `xδ`：对应 noisy level 下的终点 EE position。

逐样本计算：

| 指标 | 定义 | 含义 |
|---|---|---|
| `E_GT(δ)` | `||xδ-p0||` | 相对真实目标的误差；主要 robustness 指标 |
| `E_input(δ)` | `||xδ-pδ||` | 相对输入引导点的误差；等同 noisy 条件下 position TE |
| `Δx(δ)` | `||xδ-x0||` | 引导噪声导致的实际行为终点位移 |
| `G_parallel(δ)` | `<xδ-x0,δ>/||δ||` | 沿噪声方向的有符号实际位移，单位 m；正值表示行为沿 guidance 偏移方向响应 |

N0 时 `δ=0`，`Δx` 与 `G_parallel` 记为 `null/—`，绝不写成 0；此时 `E_GT=E_input=TE.position`。解释时同时报告 `P(G_parallel>0)`：正值说明行为正确识别了 guidance 偏移方向，绝对值表示沿该方向实际移动的尺度；`G_parallel≈0` 且 `E_GT` 小更符合对错误 guidance 不敏感。旧的无量纲 `<xδ-x0,δ>/||δ||²` 只以 `guidance_gain` 字段保留作历史审计，不作为主指标。所有指标对全部 rollout（成功与失败）计算，另可报告 success-only 辅助值，禁止只保留成功样本。

成功标准保持现有 skill-level protocol：当前 stage 出现合法 FSM forward transition 即成功；任务最终 stage 继续使用 evaluator-side 的 `environment_done + assembly-pair` fallback，不修改仿真环境成功信号。

## 4. 实验矩阵

### 4.1 最小测试（先执行）

- train：N0/N2/N4 各取 registry 中第一个 checkpoint；
- eval：N0–N6；
- state bank：18 个 stage，每 stage 前 8 个已审计 state，`m=1`；
- 总量：`3 × 7 × 18 × 8 = 3,024` 个短 stage rollout；
- 先做单 stage、N0/N2 的 16-rollout GPU smoke，确认噪声、恢复、终止和 paired report，再跑完整最小矩阵。

最小测试通过条件：所有 restore gate 通过；N0 的 `δ_norm=0` 且 `Δx/G=null`；同一配对样本在 N1–N6 的 `δ/σ` 一致；记录内 `E_GT/E_input` 与离线重算误差 < `1e-6 m`；所有 stage 都能产出合法成功或失败终止。

### 4.2 Full-rollout pilot

- 同样先用 N0/N2/N4 各一个 checkpoint、N0–N6、3 tasks；
- 每个 `(train noise, eval noise, task)` 跑 12 rollout，共 `3×7×3×12=756` rollout，12 个环境同步执行；
- 该轮用于确认各 train/eval 组合的任务 SR 和 skill-level 轨迹均非 protocol artifact。单 cell 的标准误仍然较大，不将其作为三 seed 正式统计结论；若某 task 全部 train/eval cell 都为 0，先 debug，不直接解释为模型能力。

full rollout 同时报告：task SR、每 stage 的 `reached/total`、`completed/reached`、`completed/total`。三者缺一不可，避免把“没有到达 stage”误解为当前 stage 操控失败。

### 4.3 正式 state-bank 实验

- N0/N2/N4 各 3 个训练 seed；
- N0–N6 全部 eval level；
- 18 stages；每 stage `n=24` 个固定 state，`m=2`；
- 总量：`3 × 3 × 7 × 18 × 24 × 2 = 54,432` 个短 rollout。

以 source episode 为 cluster 做 bootstrap，避免同 episode 多个 state 被当作完全独立样本。报告逐 stage、逐 task macro、全部 stage macro；成功率给 cluster-bootstrap 95% CI，连续指标给 paired cluster-bootstrap CI。

## 5. 主比较与结果表

训练噪声 `t∈{N0,N2,N4}`、测试噪声 `e∈{N0…N6}`：

- clean cost：`C_t = SR_t(N0)-SR_N0(N0)`；
- degradation：`D_t(e)=SR_t(e)-SR_t(N0)`；
- robustness gain：`RG_t(e)=D_t(e)-D_N0(e)`；
- 汇总 N1–N6 macro/AUC、N6 worst case，并画 clean SR 与 robust AUC 的 Pareto 图。

逐 stage 主表模板（每个 task 单独一张，列为具体 stage）：

| Train / Eval / Metric | stage 1 | stage 2 | … |
|---|---:|---:|---:|
| N0 / N0 / SR |  |  |  |
| N0 / N1 / SR |  |  |  |
| … |  |  |  |
| N4 / N6 / SR |  |  |  |
| 对应 `E_GT`, `E_input`, `Δx`, `G` |  |  |  |

## 6. 实现与可追溯入口

- checkpoint registry：`reports/noisy_train_noisy_eval_checkpoint_registry_20260925.json`
- 单 cell evaluator：`scripts/evaluate_furniturebench_skill_level.py`
- state-bank matrix：`scripts/run_furniturebench_noisy_skill_level_matrix.py`
- paired metrics/report：`scripts/report_furniturebench_noisy_skill_level.py`
- 核心公式：`src/eval/noisy_skill_level.py`
- state bank：`logs/clean-skill-level-state-banks-20260923/{task}/{stage}`
- pilot 输出：`logs/noisy-train-noisy-eval-20260925/state-bank-pilot8-m1/`

每个输出目录保留 `run.json`、`records.jsonl`、`summary.json`；矩阵根目录保留 `matrix_run.json`、`matrix_status.json`、`paired_records.jsonl`、`paired_summary.json`。所有汇总值必须能回溯到单条 record 与 state hash。

## 7. 时间预算

依据 clean 24-state/m=1 矩阵的实测吞吐（5,184 个短 rollout 约 4.1–4.4 小时）：

- 单-stage smoke：约 5–15 分钟；
- 8-state/m=1 全部 3×7×18 pilot：r218 GPU0 串行约 2.5–3.5 小时（含 378 次 cell 初始化，可能更慢）；
- full-rollout 12 次/cell pilot：在 r218 GPU0 的实测完整轨迹吞吐约 4–6 sim steps/s 下，约 6–7 小时（含 63 次仿真/checkpoint 初始化）；
- 正式 state-bank 54,432 rollout：按当前 r218 串行实测吞吐，单 GPU 粗估 45–60 GPU 小时；应在验证 pilot 后按 task/stage 分片到已确认空闲的 GPU，保留单一矩阵 contract，再合并 paired report。

正式 full-rollout 的样本量不由本轮 `n=24,m=2` 约束；3-rollout pilot 通过后，再根据 pilot 方差和可用算力确定，不能把 pilot SR 当正式结论。
