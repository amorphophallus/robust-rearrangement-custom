# Multi-task RGBD Skill 批量 Eval Agent 手册

本文档不是某一批 low/med/high 实验的参数快照。Agent 读完后，应先向用户确认本次 eval 配置，得到审批，再启动、监控并记录结果。训练和数据准备见 `reports/claude/batch_train_guide.md`。

## 1. 完成定义

只有同时满足以下条件，batch eval 才算完成：

1. 用户确认 eval 矩阵、checkpoint 来源、任务、随机度、rollout 数与资源策略。
2. 每个 checkpoint 的运行时配置和 eval 参数经过核对。
3. 全部实验完成，或明确记录失败原因和可恢复状态。
4. 每个任务及总计的成功率、assembly step 指标、日志和产物位置已入账。
5. 临时 rollout 的保留或删除严格执行用户批准的策略。

不要把“脚本已启动”或“进程仍存在”当作完成。

## 2. 启动前必须询问

Agent 先从仓库、launcher、checkpoint 和现有日志中发现可推断项，只向用户补问无法可靠推断的配置。至少形成以下配置表并请求一次审批：

| 类别 | 必填配置 |
|---|---|
| 实验标识 | batch 名、结果文档路径、W&B project（若使用） |
| 模型来源 | 每个实验的 W&B run ID 或本地 checkpoint 绝对路径 |
| 任务 | 例如 `one_leg, round_table, lamp` |
| 环境 | controller、domain、action type、observation space |
| 难度 | `low` / `med` / `high` 及 perturb 设置 |
| 评估量 | 每任务 rollout 数或目标成功数、并行 env 数、最大步数 |
| 比较公平性 | checkpoint 间是否共享初始状态；重复 seed/state bank 数；均值和方差口径 |
| annotation | skill debug、skill text、guidance point、grasp、part、colored |
| 资源 | 允许使用的服务器/GPU、是否串行、磁盘最低余量 |
| 产物 | 是否保存 pickle、depth、video；保留数量和清理审批规则 |

默认值必须标注为“建议值”，不能冒充用户决定。run ID、任务集、randomness、rollout 数和删除策略不得从历史批次静默继承。

## 3. Eval 矩阵

每行对应一个实际运行，建议使用下表提交审批：

| # | 名称 | checkpoint/run | experiment module | task | randomness | observation | rollout | annotation flags | save flags |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 待确认 | 待确认 | 待核对 | 待确认 | 待确认 | 待确认 | 待确认 | 待确认 | 待确认 |

常见 annotation 语义如下；实际可用 flag 必须以当前代码 `--help` 和 launcher 为准：

| 语义 | 典型开关 | 约束 |
|---|---|---|
| skill 状态/统计 | `annotate_skill` | image annotation 通常要求 image observation |
| skill 文字 | `skill_on_image` | 通常依赖 skill annotation |
| guidance point | `guidance_point_on_image` | 与 always-grasp/part 模式的互斥关系以代码为准 |
| colored point | `guidance_point_colored` | 只是配色，不应被当作启用 point 的替代 |
| always grasp | `grasp_annotation_on_image` | 所有适用 skill 绘制 grasp |
| part grasp | `grasp_part_annotate` | pick/place 绘 grasp，其余绘 point |
| colored grasp | `grasp_annotation_colored` | part-colored 通常同时要求 point/grasp colored |

数据集 suffix 是训练数据资产的标识，不应直接推导 eval annotation。eval flag 应由批准矩阵或 checkpoint 目标语义决定。

### 3.1 Checkpoint 输入与 pickle 输出必须分离

rollout 同时有两套 annotation 开关，不能混用：

- policy 输入图像：严格使用 checkpoint config 中的 `data.annotate_guidance_point`、`annotate_grasp`、`annotate_grasp_part` 及对应 colored 字段。它们改变 actor 的 observation contract。
- 保存到最终 pickle 的图像：严格使用本次命令行的 `--guidance-point-on-image`、`--grasp-annotation-on-image`、`--grasp-part-annotate`、`--guidance-point-colored`、`--grasp-annotation-colored`。checkpoint 的同名 policy 设置不得泄漏到保存结果。

因此，普通 state teacher 可以在不改变 policy 输入的情况下，用 CLI 选择独立 rollout campaign 的保存标注。如果目标资产是 LMDB，且当前 `pickle_to_lmdb` 已明确复用项目 annotation util、通过样本和视频验收，则可以按批准方案从同一批 pickle 离线生成不同标注 LMDB；否则不能假定离线补标注与 rollout 标注等价。

### 3.2 初始状态、公平比较与 seed

先确认随机性来源。若任务随机性只发生在 reset 时的初始状态采样，checkpoint 间的公平比较应优先使用 common random numbers：同一 task、同一重复组使用相同的初始状态序列，不同 checkpoint 复用该序列；不要让各 checkpoint 独立抽到难度不同的测试集。

建议方案是使用多个独立重复组，而不是只固定一个 seed：

1. 每个 task 选择 `K` 个从未用于训练数据采集的 eval seed 或 state bank。
2. 每个 seed/state bank 对每个 checkpoint 运行相同的 `N` 个初始状态。
3. 表格报告每个 checkpoint-task 的 `mean ± sample std`，样本是 `K` 个重复组各自的成功率；同时保留 pooled 成功数 `sum(success) / (K*N)`。
4. `K`、`N`、seed 列表、初始状态生成代码 commit 和 state bank 哈希必须入账。

固定 eval seed 本身不会造成训练集泄漏。泄漏风险来自复用了训练 episode 的初始状态、为特定 checkpoint 反复挑 seed，或根据测试结果调参后仍把同一测试集当最终结果。若当前代码不能只固定 reset 随机性，或无法证明 state bank 对所有 checkpoint 一致，应在审批表中明确降级为普通独立采样；此时不能报告成 paired/fair-seed 比较。

## 4. 基础设施审计

启动前动态确认，不能照抄历史机器状态：

```bash
date
git status --short
git rev-parse HEAD
conda run -n rr python -m src.eval.evaluate_model --help
df -h / /home /data 2>/dev/null
nvidia-smi
```

远端运行还需记录：hostname、代码目录和 commit、checkpoint 路径及 SHA256、GPU 型号/空闲显存、输出所在文件系统与剩余空间。快速盘与 HDD/NAS 要明确区分。

若保存 image rollout，先做小规模 smoke，再根据实测单轨迹大小估算全批次磁盘。历史上的“每实验约 60 GB”只能作旧批次参考，不能作为容量结论。

## 5. Checkpoint 核对

checkpoint 内嵌 config 是训练启动时的运行时快照，是重要证据，但仍需与 checkpoint 文件、训练日志、W&B run 和目标矩阵交叉核对；旧 checkpoint 可能缺字段，或来自错误启动。

```bash
conda run -n rr python - <<'PY'
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "/absolute/path/to/checkpoint.pt"
state = torch.load(path, map_location="cpu")
config = state.get("config", {})
data = config.get("data", {})
for key in (
    "suffix",
    "annotate_guidance_point",
    "annotate_guidance_point_colored",
    "annotate_grasp",
    "annotate_grasp_colored",
    "annotate_grasp_part",
    "annotate_skill_one_hot",
):
    print(f"{key}={data.get(key, '<missing>')}")
print(f"experiment_module={config.get('experiment_module', '<missing>')}")
PY
```

如果 checkpoint config 与批准矩阵冲突，停止该行 eval 并向用户报告；不要擅自把目标矩阵改成 checkpoint 的错误配置。

## 6. Smoke 与正式启动

先对每种不同 observation/annotation 组合做最小 smoke：

1. 单任务、单 rollout 或少量 target success。
2. 确认 checkpoint 可加载、环境可创建、图像/深度字段存在。
3. 若保存 annotation，抽查视频或 pickle 中标注确实出现且语义正确。
4. 记录耗时、显存和单轨迹磁盘占用。

smoke 必须覆盖 RGB-only 和 RGBD 两种保存路径。当前 evaluator 的 rollout 视频 serializer 可能在 RGB-only policy 下仍要求合法 depth 数组；若不加 `--save-depth-image` 会在 rollout 完成后的 crop/save 阶段失败。允许内部采集 depth 时，应在批准矩阵中注明“仅供 policy/save contract 使用”，成功合成 RGB 视频后删除临时 depth 诊断，并验收最终目录没有残留 depth 文件。不要把“模型不使用 depth”和“保存器不需要 depth”混为一谈。

正式命令必须由当前 launcher 的 CLI 参数构造。若 launcher 只能靠编辑 shell 常量切换实验，应先增加参数化 CLI 和 `--print-command`，再逐行打印并与批准矩阵比对。不要在共享脚本中反复手改常量后并发运行。

每个运行至少保存：

- 展开的最终命令；
- hostname、GPU ID、PID/session；
- checkpoint 路径与哈希；
- stdout/stderr 日志路径；
- 开始时间、完成时间和退出码。

## 7. 监控与判定

监控必须同时检查进程、日志进展、GPU 和磁盘：

```bash
pgrep -af 'src.eval.evaluate_model'
nvidia-smi
df -h / /home /data 2>/dev/null
rg -n 'Success rate|Assembly step success rates|Traceback|CUDA out of memory|No space left' /path/to/eval.log
```

判定规则：

| 状态 | 条件 |
|---|---|
| running | 进程存在，日志/计数持续推进，GPU 状态合理 |
| complete | 退出码为 0，预期任务都有最终统计，产物数量通过验收 |
| failed | 非零退出，或出现 OOM/ENOSPC/未处理 traceback |
| stale | 进程存在但超过合理窗口无日志、GPU 或文件进展；需诊断后才能重启 |

OOM 后不要直接并发补跑；先确认旧进程是否仍存活。append 模式重试前先数清已有 rollout，避免超采和重复。

## 8. 结果与资产记录

每个实验记录：

| 字段 | 内容 |
|---|---|
| identity | batch/experiment/run/checkpoint SHA |
| config | task、randomness、rollout 数、完整 annotation/save flags |
| runtime | host、GPU、开始/结束、退出码、日志 |
| metrics | 每任务成功率、all tasks、assembly step 分项 |
| artifacts | pickle/video/depth 的绝对路径、数量、字节数 |
| retention | 保留、归档或待审批删除 |

删除任何旧数据或本批 rollout 前，先列出绝对路径、owner、文件数、字节数和所在文件系统，得到用户明确批准后再执行。权限不足的残留要如实记录，不得用不相关账号绕过。

若采用多个 seed/state bank，成功率表必须明确 `mean ± sample std` 的重复组数量和每组 rollout 数。只有一个普通采样批次时，报告原始成功率与计数，不得伪造 seed 方差或把单次 rollout 的 Bernoulli 标准误写成跨 seed 标准差。

## 9. 已验证参考案例（不可静默继承）

`med_train_med_eval_0828` 于 2026-08-30 完成：8 个 checkpoint、3 个 task、每 cell 36 rollouts，共 864 rollouts；`randomness=med`，3 个并行环境，最大 1000 steps。该批最终按审批使用现有随机 reset，不固定 seed/state bank，因此报告原始成功率而不是 `mean ± std`。每个 cell 保留 3 个合成 RGB 视频，不保留 pickle 或 depth；24 个 cell 全部成功，最终 72 个视频。结果见 `reports/med_train_med_eval_0828.md`，运行清单见 `logs/med_train_med_eval_0828/formal_manifest.json`。

## 10. Agent 执行模板

用户可直接提供：

```text
请按 reports/claude/batch_eval_guide.md 执行 multi-task-rgbd-skill batch eval。
先审计当前仓库、launcher、checkpoint、GPU 和磁盘，然后把仍需我决定的配置及 eval 矩阵列给我审批。
未经审批不要删除数据或正式启动。审批后完成 smoke、逐项启动、持续监控、结果汇总和资产记录；遇到失败先确认旧进程状态和已有产物，再恢复。
```
