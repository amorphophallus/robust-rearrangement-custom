# HY Furniture VLM 三任务 grounding 诊断

日期：2026-08-17

## 结论

当前 checkpoint 的问题不是 one_leg 独有。one_leg、round_table、lamp 都存在明显的任务条件化空间原型收缩：模型能识别 task，并能识别相当一部分 skill，但同一 task/skill 内的 guidance point 只对输入产生很弱的连续空间响应。

round_table 的 skill accuracy 略高，但 point grounding 并没有最好。它在空间相近的 place/insert/screw 阶段误差较小，在 push/pick 阶段仍严重失败。更多 round_table 训练帧主要改善了离散 skill 分类，没有解决连续坐标回归。

## 运行环境

- VLM server：`zju_4090_240`，`10.71.106.240:8000`，GPU 0。
- Model：Qwen3.5-2B + structured skill/point heads。
- Checkpoint revision：`933f15ce0ed0bc7108ec1f42074bf94d985a4cbf`。
- Checkpoint：`/mnt/nas/share/home/hy/zhouhangzhu--hy_furniture/snapshots/master/ckpt`。
- Attention backend：SDPA。
- 线上 service 在诊断前后均为 `ready`；诊断使用独立进程加载同一 checkpoint，结束后进程退出，没有修改 ModelScope 仓库和线上 service。

## 样本和消融设计

从 `data/processed/vlm/messages.jsonl` 中为每个 task×oracle-skill 选择 20 个不同 rollout，每个 rollout 最多取一帧：

- 3 tasks：one_leg、round_table、lamp；
- 5 skills：push、pick、place、insert、screw；
- 每 task 100 个样本，总计 300 个样本；
- 两张 320×240 RGB 图和 `state_info.base` 与训练输入一致。

先运行每 cell 1 个样本、七变体共 105 query 的 smoke，随后运行 300×7=2100 query 的正式诊断。两次均正常退出。

七个变体：

1. `current`：当前项目中的最新 system/user prompt；主结果。
2. `legacy`：本地训练数据文件中保存的旧短 prompt；只用于 prompt 敏感性检查，不用于断言实际训练 prompt。
3. `no_output_example`：移除 user prompt 末尾固定 JSON example。
4. `task_prompt_swapped`：one_leg 输入使用 round_table prompt，round_table 使用 lamp prompt，lamp 使用 one_leg prompt。
5. `black_images`：两张图都替换为黑图，state 不变。
6. `images_cross_skill`：同 task 内把两张图替换为另一个 skill 的配对图，state 不变。
7. `state_cross_skill`：同 task 内替换为另一个 skill 的 state，两张图不变。

## 当前 prompt 的三任务结果

每 task 均衡包含五个 skill，各 20 个样本。

| Task | Skill acc. | Mean px | RMSE px | Point R² | Pred/GT spread | 相对 task Huber 常数的 mean-error 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one_leg | 71% | 14.24 | 19.49 | 0.312 | 0.312 | 17.1% |
| round_table | 73% | 19.82 | 27.85 | 0.056 | 0.236 | 3.8% |
| lamp | 67% | 28.89 | 38.05 | 0.090 | 0.170 | 11.8% |

`Pred/GT spread` 是二维预测标准差范数与 GT 标准差范数之比。它只描述输出范围，不能单独证明 grounding；应结合 Point R² 和成对输入干预一起看。

round_table 只有 skill accuracy 略高。其 point R² 和相对常数原型改善均低于 one_leg，不支持“round_table 因 step 多而学会了更好的打点”。

## 每个 skill 的 point error

| Task | Push | Pick | Place | Insert | Screw |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_leg | 17.37 | 25.66 | 10.24 | 10.60 | 7.33 |
| round_table | 34.45 | 41.08 | 8.33 | 6.39 | 8.84 |
| lamp | 36.27 | 73.13 | 16.04 | 11.87 | 7.14 |

表内为 mean pixel error。三个任务都呈现相同结构：远离最终装配中心的 push/pick 较差，空间位置聚集的 place/insert/screw 较好。lamp pick 最严重；round_table 并没有免于该问题。

round_table 的均值进一步说明了塌缩位置：

| Skill | GT point mean | Predicted point mean | Mean error px |
| --- | --- | --- | ---: |
| push | `[197.6,175.9]` | `[182.2,145.7]` | 34.45 |
| pick | `[160.7,137.8]` | `[189.3,141.8]` | 41.08 |
| place | `[191.2,141.1]` | `[192.8,140.7]` | 8.33 |
| insert | `[190.8,137.6]` | `[193.1,138.9]` | 6.39 |
| screw | `[191.3,137.7]` | `[188.7,140.7]` | 8.84 |

place/insert/screw 的 GT 本来就在约 `[191,139]` 的小区域内，所以固定原型也能得到低误差。这不能证明模型根据当前图像正确打点。

## Task prompt 主导空间原型

交换 task prompt、保持图像和 state 不变时，平均预测点移动：

| 原输入 | 替换成的 task prompt | Mean displacement px | 交换后中心距目标 task 正常中心 |
| --- | --- | ---: | ---: |
| one_leg | round_table | 33.76 | 3.91 px |
| round_table | lamp | 24.71 | 2.13 px |
| lamp | one_leg | 23.28 | 2.51 px |

这是一条直接因果证据：仅替换 task 文本，输出中心就移动到目标 prompt 对应任务的正常预测中心附近。模型没有收敛到单一的全局多任务均值，而是从 system prompt 读取 task identity，选择 task-specific prototype。

## 图像和 state 的实际影响

同 task 内跨 skill 替换图像时：

| Task | GT donor target 平均位移 | VLM point 平均位移 | 沿正确 donor 方向的平均投影比例 | Skill 输出改变比例 |
| --- | ---: | ---: | ---: | ---: |
| one_leg | 26.13 px | 5.43 px | 8.2% | 51% |
| round_table | 33.97 px | 6.31 px | 9.4% | 77% |
| lamp | 46.62 px | 6.52 px | 5.7% | 64% |

图像明显影响 skill 分类，但 point 只沿正确目标方向响应约 6%--9%。这正好解释了“skill 在变化，guidance point 基本不变”。

跨 skill 替换 state 时，point 平均只移动 3.60--4.12 px；round_table 的 donor-direction 平均投影接近零。模型使用了一部分 state 信息，但空间定位仍主要停留在 task/phase prototype。

## Prompt 版本和固定 JSON example

当前 prompt 相对 legacy prompt 在三个任务上均更好：

| Task | Current mean px | Legacy mean px | Current skill acc. | Legacy skill acc. |
| --- | ---: | ---: | ---: | ---: |
| one_leg | 14.24 | 16.61 | 71% | 56% |
| round_table | 19.82 | 23.81 | 73% | 46% |
| lamp | 28.89 | 32.70 | 67% | 51% |

结果与“checkpoint 使用最新 prompt 训练”的判断一致，但不能仅凭推理结果证明训练文件版本。

完全移除固定 JSON example 后，one_leg/round_table 各有 99/100 个样本预测为 screw，输出 spread ratio 降至 0.055/0.037。由于模型使用最后一个有效 token 的 hidden state，删除整段 suffix 同时改变了 prompt 末尾结构和 token 位置。这证明模型对准确 prompt 模板非常敏感，但不能单独证明 `[160,153]` 数值是空间 bias 的来源。要隔离数值锚点，需要保持模板和 suffix 不变，只替换 example 中的坐标数值。

## 当前判断

1. 不是 one_leg 独有，三任务都有 conditional prototype collapse。
2. round_table 更多训练 step 对 skill head 有帮助，但对 point grounding 帮助有限。
3. 任务间输出不同主要由 task prompt 决定；所以不会塌到全局多任务均值。
4. task 内部输出会随 skill/图像变化，但幅度远小于真实目标变化，尤其是 push/pick。
5. 线上 rollout 中 lamp 和 round_table 后期阶段的较小 point error 很可能来自目标本身靠近高密度原型，不代表视觉 grounding 成功。
6. 当前 checkpoint 不适合直接启动正式 324-rollout VLM+DiT 排名实验；应先修复或重新训练 point head，并用本诊断作为离线 gate。

## 产物

- 原始正式结果：`logs/vlm_grounding_diagnostic_20260817/result.json`（临时诊断归档，不进入 Git）
- SHA256：`1118aa74de7dc88da61dc769546b2f2f7d8e27c07454b2937dd22b9a9d283c6c`
- 服务器副本（已从远端 `/home` 迁入 `DATA_DIR_RAW`）：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_grounding_diag_20260817/formal_results.json`
- Smoke（已从远端 `/home` 迁入 `DATA_DIR_RAW`）：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_grounding_diag_20260817/smoke_results.json`
- 样本包（已从远端 `/home` 迁入 `DATA_DIR_RAW`）：`/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_grounding_diag_20260817/vlm_grounding_3task_mc20_20260817`
- 采样脚本归档：`logs/vlm_grounding_diagnostic_20260817/tools/prepare_vlm_grounding_samples.py`
- 诊断脚本归档：`logs/vlm_grounding_diagnostic_20260817/tools/diagnose_vlm_grounding.py`
