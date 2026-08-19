# HY Furniture original_sft VLM 三任务诊断

日期：2026-08-19

## 结论

新版 `original_sft` 已经基本解决旧双 head checkpoint 的 regress-to-mean 问题。它不再使用
独立 point head，而是按照参考 `visualize_inference.sh`，由原生
`Qwen3_5ForConditionalGeneration` greedy-generate 包含 `skill` 和
`target_point_2d` 的 JSON。

在同一组 300 条平衡诊断样本上：

- coarse skill accuracy 为 **71.0%**；旧模型为 70.3%，skill 没有实质提升；
- 296/300 条产生有效二维点，point coverage 为 **98.7%**；
- 有效点 mean / RMSE / median / p90 为
  **11.51 / 24.57 / 4.12 / 34.78 px**；
- 整体 prediction/GT spread ratio 为 **1.015**，point R² 为 **0.522**；旧模型分别为
  0.531 和 0.310；
- 跨 skill 替换图像后，输出沿 donor 目标正确方向的响应由旧模型每任务约
  5.7%--9.4% 提升到 one_leg 49.1%、round_table 101.7%、lamp 76.6%。

因此不能再把主要错误概括成“所有 skill 输出同一个任务均值”。目前主要剩余问题是：

1. round_table 和 lamp 有少数离散模式/部件选择错误，形成 70--105 px 的长尾；
2. 4/300 条生成了合法 skill，但 `target_point_2d: null`；线上服务对此 fail closed，不会
   fallback 到 scripted annotator；
3. skill 分类仍有系统性混淆，例如 lamp 的 20 条 `insert` 没有一条被预测为 insert。

这组结果足以通过“是否仍 regress-to-mean”的离线检查，但还不能直接证明 VLM+DiT 的
rollout 成功率正常。2026-08-20 用户完成三条视频审查后决定不再跑 27-rollout
smoke；工作区和正式命令审批后直接进入 324-rollout formal 矩阵。

## 版本与运行环境

- 推理代码：`Superviro/hy_furniture` commit
  `7430c97b2861a6f9da5b7487a501f5745c573555`；
- checkpoint revision：
  `75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e`；
- `model.safetensors` SHA256：
  `b1c634a13951feade89d62a53adfe8832292b81e4647a5ce764d29d5366d2caf`；
- checkpoint：`/mnt/nas/share/home/hy/vlm-guidance/original_sft/ckpts`；
- VLM server：`zju_4090_240` GPU 0，`10.71.106.240:8000`；
- readiness：`policy_version=3`、`model_mode=original_sft`、attention backend `sdpa`；
- RR 本地和 NAS 基线 HEAD 均为
  `371a783bb6a938b5eb0925fcd280a38f51ee04a6`，适配改动尚未 commit。

服务端对 batch generation 显式设置 tokenizer `padding_side=left`。参考 visualizer 是
batch size 1，没有暴露 decoder-only 模型使用 right padding 时可能生成空字符串的问题。

## 数据与方法

沿用 2026-08-17 的固定样本包：从 `data/processed/vlm/messages.jsonl` 中按
`task × oracle skill` 各选 20 个不同 rollout 的帧，每个 rollout 最多一帧：

- task：one_leg、round_table、lamp；
- coarse skill：push、pick、place、insert、screw；
- 每个 task 100 条，总计 300 条；
- 输入是 front/wrist 320×240 RGB 图、当前最新版 task system prompt、user prompt 和
  `state_info.base`；
- greedy decoding，`max_new_tokens=256`，不采样；
- pixel error 是生成的 front-image `[x,y]` 与 scripted oracle 2-D point 的欧氏距离；
- point 统计排除 4 条 `null`，skill accuracy 仍以全部 300 条为分母。

这里的样本来自构造训练数据所用 messages 文件，适合与旧 checkpoint 做 matched 诊断，
但不是独立 held-out benchmark。71.0% 也是假设五类 skill 等频的 macro-like 结果，不等于
自然 rollout 帧分布下的准确率。

### Prompt provenance

服务端使用 `src/vlm_data_generator.py` 中当前 prompt。其 system prefix 与新上游 commit
`7430c97b...` 的 `rewrite_system_prompt.py::NEW_PREFIX` 逐字一致，user prompt 与本地
`messages.jsonl` 逐字一致（把每条 state 值还原成 `<state_info>` 后比较）。

本地固定样本包所来自的原始 `data/processed/vlm/messages.jsonl` 仍保存旧的短 system
prefix；诊断脚本没有照抄这个旧字段，而是明确使用最新版 prompt。上游
`visualize_inference.py` 的模型加载和 generation 路径已复现，但它本身会读取所指定
dataset 中的 prompt，所以“参考 visualizer”并不自动保证 prompt 版本。

上传 checkpoint 没有附带 `run_config.json`，ModelScope model card 也没有记录训练数据
文件哈希，因此仅凭 checkpoint 无法独立证明训练时最终读到的是重写后的文件。本报告的
结果证明的是：**该 checkpoint 在最新版 prompt 下的行为**；训练 prompt 版本仍依据训练
执行者给出的信息与上游 rewrite 脚本，而不是模型产物内的可审计 provenance。

为了判断模型是否真的读取图像，而不只是得到更好的常数原型，额外做
`images_cross_skill` 干预：保持 task 和 state 不变，把两张图替换成同 task 另一 skill 的
配对图，然后比较模型点位移动与 donor oracle target 移动的方向投影。

## 主结果

| Task | Valid point | Skill acc. | Mean px | RMSE px | Median px | P90 px | Bias norm px | Spread | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_leg | 98/100 | 74% | 6.10 | 9.72 | 4.00 | 11.25 | 1.68 | 0.948 | 0.831 |
| round_table | 98/100 | 71% | 15.64 | 31.91 | 4.06 | 77.47 | 5.46 | 1.099 | -0.225 |
| lamp | 100/100 | 68% | 12.77 | 26.39 | 4.79 | 39.09 | 0.60 | 0.938 | 0.562 |
| **overall** | **296/300** | **71%** | **11.51** | **24.57** | **4.12** | **34.78** | **1.97** | **1.015** | **0.522** |

round_table 的 median 只有 4.06 px，但 RMSE 31.91 px、p90 77.47 px；这不是整体收缩到
均值，而是混合了大量很准的预测和少量跳到错误空间模式的预测。98 个有效点中 15 个
误差超过 40 px，13 个超过 70 px，最大 105.08 px。lamp 分别有 10、8 个，one_leg
只有 2、0 个。

### 与旧双 head checkpoint 的 matched 比较

| Model | Skill acc. | Mean px | RMSE px | Median px | P90 px | Bias norm px | Spread | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 structured heads | 70.3% | 20.98 | 29.46 | 12.93 | 63.00 | 7.49 | 0.531 | 0.310 |
| 新 original_sft | 71.0% | 11.51 | 24.57 | 4.12 | 34.78 | 1.97 | 1.015 | 0.522 |

新版 mean error 降低 45.1%，median 降低 68.1%，RMSE 只降低 16.6%。RMSE 改善较小的
原因正是离散模式错误留下的长尾，不能据此重新解释为均值回归。

## 每个 skill 的结果

表内为 `skill accuracy / 有效点 mean pixel error`：

| Task | Push | Pick | Place | Insert | Screw |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_leg | 80% / 11.27 | 90% / 3.17 | 80% / 4.89 | 20% / 7.01 | 100% / 4.13 |
| round_table | 65% / 27.59 | 70% / 19.70 | 95% / 3.26 | 60% / 11.81 | 65% / 16.08 |
| lamp | 90% / 11.11 | 70% / 27.23 | 80% / 10.04 | 0% / 11.15 | 100% / 4.31 |

点位误差与 skill 是否正确高度耦合。特别是 round_table push/pick/screw 和 lamp pick，
少数样本把点生成到另一个部件附近。后续应同时报告“全部有效点误差”和
“skill 正确条件下的点误差”，否则无法区分 semantic routing 与视觉定位。

## 图像响应与 regress-to-mean 判断

| Task | Donor GT move px | Model move px | 正确方向响应 | Skill changed |
| --- | ---: | ---: | ---: | ---: |
| one_leg | 26.13 | 15.64 | 49.1% | 41% |
| round_table | 33.97 | 30.22 | 101.7% | 71% |
| lamp | 46.62 | 36.16 | 76.6% | 64% |

旧模型虽然会随图像改变 skill，但点位沿 donor 正确方向只响应 5.7%--9.4%。新版的输出
范围接近 GT，且跨图像干预产生大幅、方向相关的点位变化，所以旧式 task-specific
prototype collapse 基本消失。

one_leg 的响应仍只有 49.1%，说明“基本解决”不等于每个样本都充分 grounding；部分
phase 的目标本来接近，skill 错误也会降低成对响应。round_table 的负 R² 与 101.7%
响应并不矛盾：模型会大幅响应图像，但少数时候选择了错误的离散目标，导致平方误差
极大。

## 无效生成和线上行为

4 条无效 point 均生成 `skill: screw` 和 `target_point_2d: null`：

- one_leg：1 条 oracle place、1 条 oracle insert；
- round_table：1 条 oracle pick、1 条 oracle screw；
- lamp：0 条。

离线诊断保留 skill 并把 point 标成 invalid，以便统计 coverage；生产服务严格拒绝这些
结果，client 不会使用旧点或 scripted fallback。后续可以通过 output schema/constrained
decoding 降低 null 率，但不能把 null 默默替换为图像中心。

## 部署、适配与验证

RR 已适配两种 engine mode，并把线上协议升级为 version 3：

- `original_sft` 直接从 checkpoint 加载 `AutoModelForImageTextToText`；
- parser 严格校验 skill、raw-pixel `[x,y]` 和图像边界；
- 原生生成模型没有可校准的分类 head 概率，API 返回 null confidence/probabilities；
- readiness 暴露 model mode、revision 和 policy version；
- local client、eval matrix revision gate 与服务协议同步；
- NAS RR 已同步服务、client、rollout summary 和 VLM point metrics 相关文件。

验证结果：

- 本地相关测试：41/41 passed；
- 对 NAS RR 源码镜像运行相同测试：41/41 passed；
- 三个 task 各一张真实图片的 HTTP multipart smoke 通过；
- 服务重启后保持 ready，GPU0 约占 4.9 GiB。

真实 HTTP smoke 中，one_leg 和 lamp 首样本误差分别约 3.6 px 和 2.8 px；round_table
首样本发生 `push -> pick`，误差约 42.4 px，复现了正式诊断的离散长尾，而非传输坐标
交换或重复缩放。

## VLM+DiT 三任务单-rollout preview

按照正式评测相同的 RGBD/VLM contract，使用 `rgbd_gp` checkpoint、`randomness=low`、
`max_rollout_steps=1000`，在 one_leg、round_table、lamp 上各完成一条 rollout。每条命令均由
`gpu-snatcher/auto_eval.sh` 展开并确认包含 `--save-depth-image`，保存未压缩 pickle 和 MP4；
VLM revision 始终为 `75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e`。

三条用于人工检查的 `cam2_vlm_debug.mp4` 均为 H.264、320×240、20 FPS。视频包含红色 VLM
点 `V`、绿色 oracle 点 `G`、VLM/GT skill、pixel error 和 cache age。首/中/后段抽帧与
pickle 中的 `[x,y]`、skill、error 逐项一致，未发现 xy 交换、重复缩放或 marker 缺失。

以下 fresh-query 表只计算 `cache_age=0` 的新 VLM 调用，是判断 VLM 本身 point/skill 质量的
主表：

| Task | Success | Valid point | Skill acc. | Mean px | RMSE px | Median px | P90 px | P95 px | >40 px | >70 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_leg | 1/1 | 41/41 | 87.8% | 4.85 | 6.97 | 3.16 | 8.54 | 9.22 | 0 | 0 |
| round_table | 1/1 | 70/70 | 80.0% | 12.60 | 26.36 | 3.38 | 35.78 | 78.71 | 7 | 7 |
| lamp | 1/1 | 46/46 | 69.6% | 17.55 | 29.59 | 4.00 | 65.74 | 68.53 | 10 | 1 |

fresh-query 的 grounding、偏置与噪声比较如下：

| Task | Mean dx/dy px | Bias norm px | GT / pred spread px | Spread ratio | Corr u/v | R² | Magnitude-equivalent | Closest projected-noise distributions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| one_leg | +1.88 / +0.10 | 1.88 | 27.02 / 27.66 | 1.024 | 0.977 / 0.781 | 0.933 | 10.70 mm, n2–n3 | sliced/centered/radial: n2; RMSE: n3 |
| round_table | +5.27 / -1.50 | 5.48 | 40.56 / 41.20 | 1.016 | 0.797 / 0.858 | 0.578 | 45.35 mm, >n4 | sliced/centered/radial: n3; RMSE: n4 |
| lamp | -2.37 / +6.02 | 6.47 | 43.79 / 39.71 | 0.907 | 0.732 / 0.849 | 0.543 | 43.58 mm, >n4 | sliced/radial: n3; centered/RMSE: n4 |

作为补充，下面是每个 control step 都计算一次的 policy-facing 指标。它把同一次 VLM 输出在
8-action horizon 内的缓存帧也计入；由于 oracle target 可能在这 8 帧内移动，它衡量 policy
实际看到的 guidance error，而不是独立 VLM query 的泛化误差：

| Task | Valid point | Skill acc. | Mean px | RMSE px | Median px | P90 px | Bias norm px | Spread ratio | R² | >40 / >70 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_leg | 326/326 | 88.0% | 5.51 | 9.04 | 4.00 | 9.00 | 0.83 | 1.071 | 0.878 | 5 / 0 |
| round_table | 559/560 | 78.2% | 13.15 | 26.97 | 3.61 | 73.17 | 5.34 | 1.020 | 0.555 | 58 / 58 |
| lamp | 361/361 | 72.3% | 15.27 | 27.13 | 3.61 | 44.91 | 5.60 | 0.901 | 0.627 | 64 / 6 |

### 指标解释

- `Valid point` / coverage：GT 2-D point 与 VLM point 同时有效的样本数；无效 point 不进入
  pixel 统计，但不能从服务稳定性统计中消失。
- `Skill acc.`：VLM coarse skill 与 scripted oracle skill 相同的比例。当前 `rgbd_gp`
  checkpoint 不把 skill one-hot 输入 policy，但 skill 错误通常反映语义阶段或部件选择错误，
  并经常与 point 长尾共同出现。
- `Mean`、`RMSE`：分别是欧氏 pixel error 的算术均值和均方根。RMSE 对少数大误差更敏感；
  RMSE 远高于 median 通常表示长尾，而不是所有预测都普遍偏差很大。
- `Median/P90/P95`：50%、90%、95% 样本误差不超过该值；`>40/>70` 直接报告危险长尾个数。
- `dx/dy`：`VLM - GT`；x 正方向向右，y 正方向向下。`Bias norm` 是平均 `[dx,dy]` 的模，
  用于识别无法用 zero-mean noise 解释的系统偏置。
- `GT/pred spread`：点集到各自中心的二维 RMS 距离；`spread ratio=pred/GT`。接近 1 表示
  输出范围没有整体收缩，明显小于 1 才支持 regress-to-mean/过度收缩假设。
- `Corr u/v`：VLM 与 GT 在水平/垂直轴上的 Pearson correlation；只反映共同变化趋势，
  不惩罚固定偏置，因此必须与 bias、pixel error 一起看。
- `R² = 1 - SSE/SST`：SSE 是 VLM 相对 GT 的二维平方误差和，SST 是 GT 相对自身二维均值的
  总平方离差。1 为完美，0 等价于始终输出 GT 全局均值，负值表示比这个常数基线更差。
- `Magnitude-equivalent`：仅按 projected n0–n4 的 RMSE 曲线插值出的 3-D noise std；
  `sliced/centered/radial Wasserstein` 分别比较完整二维残差、去偏后的分布形状和径向误差。
  它们回答的问题不同，不能只凭一个平均 pixel error 声称“VLM 等价于 n4”。

### 数据分析

one_leg 的 fresh-query `R²=0.933`、spread ratio 1.024、P90 8.54 px，且没有 >40 px 点，
说明新版 VLM 在该 rollout 中既跟随目标移动，也没有向固定均值收缩。all-step 多出的 5 个
>40 px 帧来自 action horizon 内 GT 变化后仍使用缓存点的有效 guidance mismatch；这是既有
8-action 控制设计的一部分，不是网络延迟。

round_table 和 lamp 同样不符合旧式 regress-to-mean：spread ratio 分别为 1.016、0.907，
两个轴相关性均较高，R² 仍为正。但两者呈明显混合分布：median 只有 3–4 px，RMSE 却为
26–30 px；round_table 的 70 次 fresh query 中 7 次超过 70 px，lamp 的 46 次中 10 次超过
40 px。这与离线诊断的“少数错误部件/离散模式选择”一致。round_table 还有约 +5.3 px 的
水平偏置，lamp 有约 +6.0 px 的向下偏置。

因此噪声映射必须分开解释：one_leg 的分布距离最接近 n2、幅值处于 n2–n3；round_table
和 lamp 的 RMSE 幅值都超过 n4，但完整/径向分布的最近邻多为 n3。这并不矛盾——少数
极端离散错误会显著抬高 RMSE，而它们不是 n4 所假设的零均值、近各向同性连续噪声。

三条最终 preview 都成功，但运行期间第一条 round_table 尝试在 step 495 收到一次 HTTP
422，launcher 按 fail-closed 立即中止，未生成 summary 或正式产物；全新 low-randomness
重试完整成功。旧 client 丢弃了 FastAPI response detail，因此无法事后确定该次是 null、
越界还是非法 JSON。client 已改为在后续非 2xx 异常中保留限长的服务端 detail，不改变
严格解析和禁止 fallback 的行为。这次事件说明正式矩阵前仍需把 invalid-generation rate
作为服务稳定性 gate，不能因为重试成功而忽略。

### Preview 产物

- one_leg debug MP4：`/data/hy/robust-rearrangement/data/raw/diffik/sim/one_leg/rollout/low/rgbd-point-vlm/vlm_original_sft_preview_20260819/rgbd_gp/one_leg/success/2026-08-19T23-38-18.837599_cam2_vlm_debug.mp4`；
- round_table debug MP4：`/data/hy/robust-rearrangement/data/raw/diffik/sim/round_table/rollout/low/rgbd-point-vlm/vlm_original_sft_preview_round_retry_20260819/rgbd_gp/round_table/success/2026-08-19T23-50-42.321411_cam2_vlm_debug.mp4`；
- lamp debug MP4：`/data/hy/robust-rearrangement/data/raw/diffik/sim/lamp/rollout/low/rgbd-point-vlm/vlm_original_sft_preview_lamp_20260819/rgbd_gp/lamp/success/2026-08-19T23-55-00.143784_cam2_vlm_debug.mp4`；
- 完整输出根目录：`/data/hy/robust-rearrangement/data`，本次 preview 约 1.3 GiB，包含
  3 个未压缩 `.pkl`，没有 `.pkl.xz`。

## NAS 清理

新服务和真实 HTTP smoke 通过后，删除了以下无引用旧产物：

- `/mnt/nas/share/home/hy/zhouhangzhu--hy_furniture`：旧双 head 模型，约 4.2 GiB；
- `/mnt/nas/share/home/hy/vlm-guidance/Qwen3.5-2B`：旧独立 base model，约 4.3 GiB；
- `/mnt/nas/share/home/hy/vlm-guidance/manifest.json`：旧 manifest；
- `/mnt/nas/share/home/hy/vlm-guidance/original_sft/.git`：clone 后重复保存的 LFS 权重，约
  5.1 GiB。

合计释放约 13.6 GiB。删除不可从这些原路径直接恢复，但模型和代码都可从上游重新下载。
保留的新 checkpoint 文件和 SHA256 在删除前后均已核对；服务删除后仍为 ready。

## 产物

- 原始结果：`logs/vlm_original_sft_20260819/formal.json`（Git ignored）；
- 原始结果 SHA256：
  `76839695cac43e57568e986026efb171f4b10803a429b59baae5668e691181cb`；
- 服务器结果（已从远端 `/home` 迁入 `DATA_DIR_RAW`）：
  `/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_grounding_diag_20260817/original_sft_formal_20260819.json`；
- 固定样本包（已从远端 `/home` 迁入 `DATA_DIR_RAW`）：
  `/data/hy/robust-rearrangement/data/raw/vlm_diagnostics/vlm_grounding_diag_20260817/vlm_grounding_3task_mc20_20260817`；
- 临时工具放在 `logs/vlm_original_sft_20260819/tools/`，没有污染 `scripts/`。
