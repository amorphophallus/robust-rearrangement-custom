# Multi-task RGBD Skill 批量训练 Agent 手册

## 核心代码修改审批规则

Agent 在修改仓库内核心代码逻辑前，必须先向用户列出拟修改的文件、行为变化、兼容性影响和验证方案，等待用户明确审批。核心代码包括 `src/` 下的 rollout、数据处理、dataset、model、训练 launcher 及其直接依赖；未获审批不得直接修改。

实验执行期间，Agent 可以自行决定并调整已批准方案内的运行编排，例如参数展开、任务轮询、服务器选择、传输重试、断点续传、日志记录、marker 检查和临时清理，但不得借此改变核心算法或数据语义。临时操作脚本优先放在 `logs/` 下，避免污染 `scripts/` 和用户的 git 工作区。

收到核心代码修改需求时，先输出“修改审批摘要”，至少包含：

| 项目 | 必须说明 |
|---|---|
| 文件 | 绝对路径和涉及函数/入口 |
| 逻辑 | 修改前后行为，尤其是 checkpoint、CLI、rollout、pickle、LMDB 之间的边界 |
| 风险 | 对既有实验、数据兼容性和训练配置的影响 |
| 验证 | 单测、smoke、数据校验和回滚方式 |
| 状态 | `待审批`、`已批准` 或 `已拒绝` |

只有状态为 `已批准` 后才能编辑核心代码；文档、日志和用户明确要求的资产台账可直接更新。

本文档用于让 agent 从零询问配置、准备数据、调度双卡训练并监控到完成。它不是某次 low/med 实验的静态记录。服务器容量、GPU、数据位置和 checkpoint 状态必须在每次 campaign 开始时实时审计，禁止照抄历史快照。

> ppu96 上“zstd LMDB 放根盘、四个独立 world-size=1 实验按每卡两个并行”的专用方案，不使用本文的双卡 DDP 启动章节；应改读 `reports/claude/ppu96_single_card_4way_zstd_runbook.md`。该专用文档同时记录了 1/3/4 路压力测试、固定参数、磁盘与 checkpoint 布局、正式启动模板和恢复 gate。

## 1. 完成定义

一个 campaign 只有同时满足以下条件才算完成：

1. 用户批准删除清单、数据方案和训练矩阵三项配置。
2. 每个物理数据集达到批准的逐任务轨迹数，metadata、provenance、loader 和统计校验通过。
3. 数据位于训练服务器的 SSD/NVMe，上传 manifest 一致，传输路由和平均速度有记录。
4. 所有训练的实际 Hydra config 与批准矩阵一致。
5. 每个 run 达到目标 epoch，last checkpoint 存在，W&B 状态和进程退出状态正常。
6. `logs/` 下的数据资产与实验台账已更新。

只“启动训练”不算完成。agent 应持续轮询和恢复，直到全部 run 完成，或遇到必须由用户处理的真实阻塞。

## 2. 启动时必须询问的配置

先从仓库、checkpoint 和现有日志中发现可确定的信息，再向用户询问仍不明确的项。至少要确认：

| 类别 | 必填配置 |
|---|---|
| Campaign | campaign ID、W&B project、W&B mode |
| 数据范围 | task 列表、randomness、每任务成功轨迹数、是否保存 failure |
| Teacher | 每任务 checkpoint/run ID、checkpoint 选择规则、期望成功率 |
| Annotation | 实验列表、每组 suffix、六个 annotation flag、是否允许同一物理数据集复用 |
| 模型 | `experiment`、vision encoder 及额外 Hydra override |
| 训练 | global batch size、epoch、steps/epoch、保存间隔、worker、GPU 数、resume 规则 |
| 基础设施 | 允许使用的服务器、快速盘、上传速度阈值、哪些旧资产允许删除 |

推荐默认的三任务 condition matrix 如下，但必须把最终表展示给用户审批，不能把它当作永远固定的配置。这里列出的是 8 个逻辑实验、5 个物理 LMDB：`rgbd`、`rgbd-skill-point`、`rgbd-skill-point-colored`、`rgbd-skill-grasp-part`、`rgbd-skill-grasp-part-colored`。

| 实验 | suffix | GP | skill | GP colored | grasp | grasp colored | grasp-part | experiment |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Exp1: rgbd | `rgbd` | F | F | F | F | F | F | `rgbd/dit` |
| Exp2: rgbd+GP | `rgbd-skill-point` | T | F | F | F | F | F | `rgbd/dit` |
| Exp3: rgbd+colored GP | `rgbd-skill-point-colored` | T | F | T | F | F | F | `rgbd/dit` |
| Exp4: rgbd+only skill | `rgbd` | F | T | F | F | F | F | `rgbd/dit` |
| Exp5: rgbd+GP+skill | `rgbd-skill-point` | T | T | F | F | F | F | `rgbd/dit` |
| Exp6: image baseline | `rgbd` | F | F | F | F | F | F | `image/dit` + `vision_encoder=resnet`, `pretrained=false` |
| Exp7: rgbd+grasp-part | `rgbd-skill-grasp-part` | F | F | F | F | F | T | `rgbd/dit` |
| Exp8: rgbd+colored grasp-part | `rgbd-skill-grasp-part-colored` | F | F | T | F | T | T | `rgbd/dit` |

`data.suffix` 描述物理图像形态，annotation flag 描述训练输入。source rollout 只生成一次共同的 pickle，并保存 skill/2D annotation metadata；各物理 LMDB 在 `pickle_to_lmdb` 阶段从同一批 pickle 调用共享 annotation util 确定性生成。不要为每个 annotation mode 重复 rollout，也不要仅为逻辑命名复制数百 GB 数据。

## 3. 三个审批门

在 `logs/<campaign>_plan.md` 中先写好以下三张表，并给出精确值和依据。

### 审批 1：删除清单

逐项列出：host、精确绝对路径、owner、大小、挂载点、盘型、删除后可用空间、删除理由。审批前只允许只读命令：`find`、`du`、`df`、`findmnt`、`lsblk`、`stat`。

删除前再次检查：

```bash
test -e <exact-path>
du -sh <exact-path>
stat -c '%U:%G %a %n' <exact-path>
findmnt -T <exact-path>
```

只删除用户逐项批准且本人有权限的精确路径。禁止宽泛 glob、`git reset --hard`、清除其他用户数据或用全局 `pkill`。删除后记录路径不存在和 `df` 前后变化。

### 审批 2：数据方案

表中必须包含：

- 每任务 success 数和总数，不能用“每个 LMDB N 条”这种有歧义的表述。
- teacher checkpoint 路径、SHA256、训练配置和 smoke 成功率。
- 需要几个物理数据集，哪些实验复用它们。
- 在线采集参数、离线派生方式、预计 raw/LMDB 容量。
- provenance、manifest、sample/full validation 和 loader smoke test。

### 审批 3：训练矩阵

同时列出共同参数和每组差异参数。至少包含 tasks、randomness、suffix、六个 annotation flag、experiment、encoder overrides、global batch size、epoch、steps/epoch、保存频率、GPU 数和 W&B project。

用户明确批准三项后才能执行有副作用的删除、采集、上传和训练。

## 4. 实时基础设施审计

历史服务器编号可以作为扫描候选，但不能把历史容量当作当前事实。对每台候选机采集：

```bash
hostname
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
df -h / /home /data 2>/dev/null
lsblk -o NAME,TYPE,ROTA,TRAN,SIZE,FSTYPE,MOUNTPOINTS
findmnt -T <candidate-data-root>
```

规则：

- `ROTA=0` 且 `findmnt` 命中预期 SSD/NVMe 挂载点，才可作为训练数据盘。
- `/data` 不是天然快速盘；同一批服务器上它可能是 NVMe，也可能是 HDD。
- 数据容量、训练 checkpoint、临时目录和 10% 余量都要计入空间预算。
- owner 不是当前用户且无写权限的数据只记录为权限阻塞，不尝试绕过。
- GPU “空闲”应同时参考显存、utilization 和进程，不只看 utilization 瞬时值。

所有结果写入 `logs/<campaign>_assets.md`，不要依赖某个外部 Notion 页面存在。

## 5. 数据采集与派生

### 5.1 Teacher smoke

每个 teacher 先收集少量目标成功轨迹，建议 10 条：

```bash
python -m src.eval.evaluate_model \
  --wt-path <checkpoint> \
  --gpu <gpu> \
  --n-envs <batch-envs> \
  --n-rollouts <batch-envs> \
  --target-successes 10 \
  -f <task> \
  --randomness <randomness> \
  --action-type pos \
  --observation-space image \
  --save-rollouts \
  --save-depth-image \
  --annotate-skill \
  --output-only-pickle \
  --max-saved-rollouts 10 \
  --rollout-suffix-model-name <campaign-smoke>
```

`target-successes` 按批运行，eval 成功数可能超过目标；`max-saved-rollouts` 才保证落盘数量精确。检查每个 pickle 的 success、task、动作长度、非零 skill、2D metadata 和非占位深度。

### 5.2 同轨迹 annotation 派生

若多组实验只在可视化 annotation 上不同，优先采集一份原始 RGBD + skill/2D metadata，再在 pickle 到 LMDB 转换时确定性渲染。这样所有 condition 使用完全相同的状态、动作、深度和轨迹分布。

当前转换器支持：

```text
none
guidance-point
guidance-point-colored
grasp-part
grasp-part-colored
```

使用 `src.data_processing.process_pickles_to_lmdb --image-annotation-mode <mode> --provenance-json <json>`。生成大数据前先用一个真实 pickle 做五种 mode 集成测试，确认只允许目标 RGB 相机像素不同。

不要假定 eval CLI 的图像 annotation flag 一定覆盖 checkpoint config。先读 `_resolve_eval_annotation_settings` 和实际保存代码；若保存行为由 checkpoint 决定，应使用离线派生或先修复并测试覆盖逻辑。

### 5.3 路径与配额

多任务目录名由 task 列表规范化生成；例如三个任务常为：

```text
processed/diffik/sim/lamp-one_leg-round_table/rollout/<randomness>/success/<suffix>.lmdb
```

这只是示例，不是路径重写规则。同一 campaign 的 legacy 数据、重新派生数据和同机
split placement 可能分别使用下划线或连字符目录（例如 `round_table` 与
`round-table`）；agent 不得凭记忆或“统一格式”替换其中任一字符。资产路径应从
实际 `data.data_paths_override`、远端进程命令、manifest/marker 和只读 `find` 结果
交叉确认。清理、恢复和后续上传前都要对精确父目录执行 `test -d`、`findmnt -T`，
路径不存在时先判定为拼写/登记不一致，不能把它直接解释成数据已删除。

转换时使用逐任务限制，例如：

```bash
--task-episode-limit one_leg=200 round_table=200 lamp=200
```

不要用全局 `--num-pickles 600` 替代平衡配额。转换 metadata 至少记录：task counts、selected pickle list、teacher SHA256、代码 commit/patch hash、annotation mode 和 source manifest hash。

### 5.4 数据验收

每个 LMDB 必须完成：

1. metadata 的 episode 数、task counts、randomness、suffix 和 provenance 检查。
2. `scripts/validate_lmdb_dataset.py` sample 校验。
3. 一次 full stats/全 episode 索引校验。
4. 使用对应 annotation flag 的 dataset loader 读取至少一个 batch。
5. 同源 condition 间 action/state/depth/task order 一致性检查。

## 6. 上传与资产台账

### 6.1 有线网络确认

传输前记录源和目标：

```bash
ip route get <peer-ip>
ip -br link
ethtool <wired-interface>
```

路由必须走已确认的有线接口。Wi-Fi、`scp -3` 和本机双 SSH tar 管道都不适合大 LMDB。优先 `rsync` 直传；源和目标都应为快速盘。

### 6.2 速度与完整性

记录开始/结束时间和实际字节数：

```text
MB/s = bytes / duration_seconds / 1,000,000
```

默认目标约 100 MB/s。低于 80 MB/s 先检查路由、源盘、目标盘和 CPU；持续低于 50 MB/s 时停止并换接口或数据源。不要只采信工具瞬时进度。

传输后比较：

- `du --bytes` 或文件总字节数。
- 相对文件列表。
- SHA256 manifest。
- 目标服务器上的 LMDB metadata 和 loader smoke。

资产表每行包含 dataset ID、suffix、episodes、task counts、annotation、source manifest、host、快速盘绝对路径、盘型、字节数、校验状态和训练占用状态。移动、复制、删除都追加事件，不覆盖历史。

共享源 pickle 的生命周期必须由所有物理消费者的 gate 共同管理：在每一个需要该 source 的物理 LMDB 都完成远端传输、`data.mdb` hash、loader/full-stats 和 aggregate 轨迹数校验前，保留源 pickle。并行上传某一个 LMDB 通过 gate 不等于可以删除 source；删除 watcher 必须显式等待其余并行 `pickle_to_lmdb` 消费者的 aggregate-ready marker，确认最后一个物理消费者通过后，才按任务逐一核对批准的文件数并删除源 pickle。删除后保留 source manifest、provenance、hash 和删除 marker，并在资产台账记录事件；禁止用“本地 LMDB 已生成”替代远端验收。

### 6.3 非交互 SSH 的运行环境

不要假设远端非交互 SSH 的 `PATH` 已包含 `conda`。在每台服务器首次使用时记录实际环境入口（例如绝对路径 `.../miniconda3/bin/conda`），远端校验和 loader smoke 使用该绝对路径，或在同一条命令中显式 source `conda.sh`。本地转换器和远端 validator 的 Python 环境要分开核对；validator 参数也必须匹配其覆盖范围：单 shard 使用结构/loader smoke，只有四个 shard 齐全时才使用要求 600 条轨迹的 campaign 聚合 validator。

若 NAS 上的旧版 `validate_lmdb_dataset.py --full-stats` 在多任务数据的可变形状字段（例如 `parts_poses`）上出现 broadcast/shape mismatch，先记录为 validator incompatibility，不得把它当作 LMDB 损坏，也不得切换到未经批准的本地代码。应继续使用 NAS 环境和 NAS `src.dataset.lmdb` 做只读全量结构扫描：metadata/index、600/200/200 轨迹配额、success、每条 episode 的 lowdim dtype/leading dimension，以及每条 episode 的起止/中间 frame payload；结构扫描通过后才可写 ready marker，并在资产台账中同时记录原 validator 错误和 fallback 结果。

## 7. 双卡训练启动

启动器为 `/data/hy/gpu-snatcher/auto_train_multi_card.sh`。先运行 `--help`，以当前脚本实际支持的 CLI 为准。多个实验必须通过 CLI 传值，禁止为了每个实验反复改默认变量。

典型命令：

```bash
bash /data/hy/gpu-snatcher/auto_train_multi_card.sh \
  --ssh-name <host> \
  --gpu-id <gpu0,gpu1> \
  --num-gpus 2 \
  --data-dir-processed <fast-data-root> \
  --task-spec '[one_leg, round_table, lamp]' \
  --data-suffix <suffix> \
  --experiment rgbd/dit \
  --randomness <randomness> \
  --storage-format lmdb \
  --load-into-memory false \
  --dataloader-workers 4 \
  --batch-size 512 \
  --num-epochs 3000 \
  --steps-per-epoch 100 \
  --save-per-epoch 500 \
  --wandb-project <project> \
  --wandb-mode online \
  --ddp-shard-enabled true \
  --dryrun false \
  <six annotation switches>
```

`training.batch_size` 在当前 DDP 训练中按 global batch 口径审批。不要在 OOM 后自动改变它，因为这会改变实验；先记录失败并由用户批准配置变更，或按相同配置换机器。

Image-only 实验应显式传 `--vision-encoder resnet --vision-encoder-pretrained false`，这两个参数已是 launcher CLI，不是共享默认值。如果其他必要 Hydra override 没有 launcher CLI，必须先增加经过测试的显式选项，或使用完整记录的 `torchrun` 命令。Hydra/W&B 最终配置可能把 launcher 的 `resnet` alias 展开为具体注册名 `vision_encoder.model=resnet18`；配置校验应比较展开后的最终值，而不是把 alias 当作错误。禁止临时取消注释共享脚本中的默认行后并行启动。

启动前还必须把 launcher 实际展开的 Hydra overrides 与当前 config schema 对照。当前训练 config 不定义 `data.annotation_noise_*`，因此不能把 gpu-snatcher 默认注入的这些键直接传给 `bc_ddp`，否则 Hydra 会在训练进程启动前退出。实验期可以在 `logs/` 使用经过验证的兼容编排脚本去掉无效默认项，但不得修改 annotation 语义或训练参数。调度器只有在 launcher 返回 `status: started` 后才能写入 `started` registry；`status: failed` 必须保留失败记录并继续正常频率轮询。

### 7.1 调度条件

每个 job 只在同时满足以下条件时启动：

- 两张 GPU 空闲且无人预留。
- 该服务器快速盘上已有完整且校验通过的目标 LMDB。
- 代码版本和数据 provenance 可追踪。
- checkpoint 输出目录与临时目录有足够空间。
- W&B 登录和网络正常。

优先使用已有数据的服务器，减少复制。不要把一个 LMDB 放在 HDD 后通过 symlink 伪装成快速盘路径。

### 7.1.1 动态选机规则

服务器不是实验配置的一部分，不得把 `228`、`230` 或其他主机写成永久映射。每轮调度按以下顺序实时筛选：

1. 使用 `gpu-snatcher/check_zju_gpu.sh` 获取所有候选主机的 GPU 状态，并通过远端 `nvidia-smi` 确认一对未被占用的 GPU。
2. 按数据资产台账反查该 annotation 数据集的完整物理布局；四片目录、完成 marker、字节数、`data.mdb` hash 和 loader smoke 必须全部通过。
3. 对数据路径运行 `findmnt`/`lsblk`/`df`，确认路径位于本地非旋转盘且有足够余量。`/var/tmp`、HDD、NFS/NAS 和 symlink 不能作为训练数据落盘。
4. 读取 `MemFree`、`MemAvailable`、`SwapFree` 和 `earlyoom`/`systemd-oomd` 状态；容量门禁以可回收 page cache 后的 `MemAvailable` 为主，`MemFree` 只作诊断，不能因为 page cache 较大而误判内存不足。如果主机处于内存或 swap 保护线以下，跳过该主机并记录 `pending: host-memory-pressure`。GPU 空闲不等于 Python 训练可安全启动。
5. 只有同一主机同时满足数据 gate、快速盘 gate、系统资源 gate 和双卡 gate，才调用 launcher；否则记录 pending 原因，继续轮询，不把数据不完整或资源不安全的空卡当作可运行条件。

GPU snapshot 到远端训练进程启动之间存在抢卡竞态。launcher 必须在远端创建
tmux 前，对选中的每张卡重新读取 `memory.total`、`memory.used` 和 compute PID；
任一卡不再满足空闲阈值就应在创建 W&B run 之前退出，让调度器稍后重试。训练
启动后还应核对选中卡上的 compute PID owner；若出现外部进程共享，记录实际
SM 利用率、速度和 ETA，不能把受争用速度误判为模型或数据性能。

tmux 使用 `remain-on-exit` 或训练命令外还有交互 shell 时，pane 本身存活不代表训练
进程存活。launcher 的启动后门禁必须同时确认 `pane_dead=0` 且
`pane_current_command` 仍为 `python`/`torchrun`，再核对实际 `bc_ddp` 参数和两张卡上的
rank PID；训练命令已经退出而 pane 回到 `bash` 时必须返回 `status: failed`，不得写入
`started` registry。迁移恢复还必须看到同一 W&B run ID、明确的 checkpoint epoch 和
`start_epoch=checkpoint_epoch+1` 后才记为 `resumed`。

如果一台主机有多个本地非旋转盘、但单个挂载点不足以容纳完整数据集，可以使用“同机多快盘分片”布局：每个 shard 必须落在实际 `ROTA=0` 的挂载点，训练通过 `data.data_paths_override` 使用四个绝对路径；必须分别完成字节数、`data.mdb` hash、loader smoke 和 aggregate gate。不得用 symlink、HDD、NFS/NAS 或未经登记的 `/var/tmp` 冒充这种布局；临时根盘路径只能在台账中明确标记，并在训练结束后清理。

同机 split 数据释放和下一批数据接力必须使用两个不同 gate：先由完成监控证明 run、
最终 checkpoint 和 W&B 状态，再由 cleanup watcher 核对“远端无对应进程 + 当前实际
`data_paths_override` 的每个 shard/marker 一致”并写 cleanup-complete marker。下一批
uploader 只能等待 cleanup-complete，不能只等待 training release marker。新布局必须和
调度器识别的 shard-to-mount 映射完全一致，并分别按挂载点计算预计释放后空间、目标
分片字节数和至少约 20GB/批准比例的余量；不能只看两块盘的合计空间。

数据准备可以先于 GPU 空闲进行，但训练启动必须重新执行上述三项检查。候选主机释放后优先把完整数据通过有线网卡迁移过去，使用带超时和外层重试的 `rsync --partial --inplace --append-verify --timeout=<seconds>`；只写 rsync 的断点参数而没有失败重试循环，网络 broken pipe 后仍会让整个 watcher 退出。服务器之间不能直接认证时，可以使用本机快速盘逐 shard 中转，但每次只暂存一个 shard，并在 source/relay/target hash 一致后立即删除 relay。完成 hash/loader gate 后才能写 ready marker。

多个生成或迁移 watcher 可能同时选择同一目标盘。容量 gate 除当前已用空间外，
还必须扣除其他在建数据的剩余预计字节数，并用 owner/reservation marker 持久化；
aggregate-ready 或明确失败清理后才能释放 reservation。禁止两个 watcher 各自只按
当前 `df` 余量通过 gate，否则即使它们单独都能放下，合计仍可能写满快速盘。

生成与上传并行时优先使用单流交错队列：第 N 个分片完成后上传、hash/full-stats、
删除生成端输出并启动第 N+1 个 writer，同时用一个 `--bwlimit` 受控的断点流搬运另一
个已验证分片。这样既不在容量有限的生成盘堆积多个 LMDB，也不让多路 100MB/s 流
争抢目标盘。coordinator 重启后的容量门禁应扣除目标上已经验收的分片，只要求
“剩余预计字节 + 余量”；重复要求初始整批空闲容量会错误阻止合法断点恢复。

不稳定主机应放在 placement 列表末尾，并在每轮调度中缓存 SSH transport failure：
首次连接超时后，本轮后续实验直接跳过该主机，下一轮再重试。远端 marker/data
predicate 返回非零不等于 SSH 不可达，不能把普通的“数据不存在”缓存为主机故障。
GPU snapshot 自身和 placement 检查应共享或合并可达性结果，避免一个低优先级主机
把整轮抢卡扫描从几十秒拖到数分钟。

### 7.1.2 campaign 编排入口与 legacy 隔离

仓库中带有 campaign-specific 的旧脚本时，必须先检查其 source suffix、manifest、服务器和 GPU 映射；不能因为脚本名称相似就复用。对于当前 med campaign，唯一允许的编排入口是 `logs/med_0801_*`：

- `logs/med_0801_dynamic_inventory.md`：当前主机、快盘、数据资产和实验配置的实时登记。
- `logs/med_0801_generate_upload_point_remote.sh`：当前 point 数据的远端 NAS-only 分片生成流程。
- `logs/med_0801_sequential_annotation_pipeline.sh`：串行编排的可审计 fallback；它等待前一阶段完成后再重建各物理 LMDB。只有在确认没有对应的并行 worker、owner marker 或上传 watcher 时才能使用。
- 当前 med-0801 的 colored grasp-part 生成使用 `logs/med_0801_build_grasp_part_colored_after_release.sh`（shard-1/释放协调）、`logs/med_0801_build_grasp_part_colored_parallel_2.sh`（shard-2）、`logs/med_0801_build_grasp_part_colored_parallel_34.sh`（shard-3/4）和 `logs/med_0801_upload_grasp_part_colored_236.sh`（236 快盘上传）。四个分片都从同一 shared pickle 独立执行完整 `pickle_to_lmdb`，不重新 rollout；并行 worker 的 owner marker、full-stats 和 aggregate gate 是强制边界。
- `logs/med_0801_release_watch.sh`：按已验证的 Exp2/5、Exp3、Exp7、Exp8 完成状态生成阶段 release marker；marker 不绑定固定服务器。
- `logs/med_0801_completion_monitor.sh`：小时级对账 W&B run ID、最终 config、目标 epoch、NAS checkpoint metadata 和远端进程退出；全部通过后写不可重复启动的 experiment completion marker。release watcher 不能依赖停止更新的状态快照。
- `logs/med_0801_dynamic_training_poller.sh`：调用 `check_zju_gpu.sh`，按数据布局动态选择训练主机，不使用固定 host 映射。
- `logs/med_0801_launch_compat.sh`：实验期启动适配器，复用远端 tmux/conda 环境并避免向当前 Hydra schema 注入无效的 annotation-noise 键。
- `logs/med_0801_watch_rgbd_upload_236.sh`：当前 RGBD 训练副本的低频断点上传验收；只在远端 rsync 成功、`data.mdb` 字节数/hash 和 LMDB full-stats 全部通过后写入 236 的 ready marker。

`logs/med_0801_poll_stage_a_training.sh`、`logs/med_0801_build_upload_shared_stage_a.sh` 和 `logs/med_0801_build_upload_shared_stage_b.sh` 保留用于审计，但其中的历史 host/path 映射不能直接用于新的 campaign；使用前必须经过动态资产表和当前挂载审计。当前 campaign 不使用旧 Stage-B 脚本；实际采用串行或并行编排时，以当前运行台账、owner marker 和锁文件为准，不能同时启动两套生成器。

所有会长期运行或可由人工恢复的 campaign 编排器都必须使用 `flock` 单实例锁，并在启动前核对已有 PID、tmux 和 marker；Stage-A、Stage-B、GPU poller 不能并行存在多个副本。发现重复等待器时只停止重复的编排进程，不停止训练、数据转换或上传进程。

manifest 只是资产索引，不能作为生成完成 gate。恢复生成器看到已有 manifest 行时，仍须
核对对应 LMDB 的完成 marker、metadata、episode 数、annotation mode、字节数和
`data.mdb` hash；任一不一致都应把该行标为失效，而不是直接复用。执行带
`--overwrite` 的 pickle-to-LMDB 前，还必须确认没有其他 owner/lock holder，且目标
路径没有通过上述 gate 的有效资产。若有效资产已存在，禁止覆盖；若未完成残片需要重建，
应先在台账记录 invalidation，并优先写到新路径或新主机，完成验收后再精确清理旧残片。

长时间 watcher、GPU poller 和速度监控应由 user-level systemd、supervisord 或其他
脱离当前终端的持久 supervisor 托管；单次交互 shell 中的 `nohup` 不能视为持久化。
重启前先用 `fuser`/`lslocks` 检查 lock holder，避免父 shell 退出后遗留的 `sleep`
子进程继续持锁而使新实例静默退出。调度器启动后必须验证 supervisor 状态、日志
时间戳和实际 PID，而不是只看启动命令返回成功。数据 ready marker 应放在 LMDB
目录的同级资产目录，并让 validator、poller 和资产台账使用同一个 marker 路径约定。
`systemd-run --user` 创建的 transient unit 可能在 user manager 重建后消失；每次 agent
续接、上下文恢复或状态汇报都应先用 `systemctl --user is-active`、日志最新时间和实际
PID 三项复核。unit 不存在时按原批准间隔和同一单实例锁重建，不能把历史的
`systemctl start` 成功或旧 PID 当作当前监控仍存活的证据。

`scripts/prepare_upload_med_0801.sh`、`scripts/schedule_med_0801_training.sh` 等旧入口如果仍引用旧 source/provenance、旧 host 或旧 GPU，标记为 `legacy/no-touch`，不得用于当前 campaign。旧 `rgbd` backup 也不能绕过当前 source manifest gate。

### 7.2 启动后强制核对

从 tmux 的真实 Hydra config 和进程状态核对：

```text
task
randomness
data.suffix
data.annotate_guidance_point
data.annotate_skill_one_hot
data.annotate_guidance_point_colored
data.annotate_grasp
data.annotate_grasp_colored
data.annotate_grasp_part
data.storage_format
data.ddp_shard_enabled
training.batch_size
training.num_epochs
training.steps_per_epoch
wandb.project / wandb.mode
experiment / vision encoder
world size = 2 and both ranks alive
```

任何一项与审批表不符，都只向该 tmux session 发送 `C-c` 并清理该 session；不要使用宽泛 `pkill`。

## 8. 监控与恢复

启动后首小时每 10 分钟检查，稳定后每 1 小时检查；若运行状态稳定且用户批准，可进一步降低频率：

- tmux 是否存在，两个 rank 是否都存活。
- GPU 显存/utilization，系统 RAM，快速盘剩余空间和 I/O wait。
- epoch/step 是否推进，是否出现 Traceback、OOM、Killed、NCCL 或 LMDB 错误。
- W&B run ID/name/state 和同步状态。
- last/periodic checkpoint 是否按期生成。

每个 run 在 `logs/<campaign>_runs.md` 记录 experiment、host、GPU、tmux、run ID、run name、数据路径、实际 config 摘要、开始时间、当前 epoch、checkpoint 和状态。

W&B 初始化可能晚于 launcher 返回。若启动记录暂时只有 `run_id=-`，小时级监控必须按启动时间和完整实验 config 从 W&B 项目中对账，随后向 append-only registry 补写 run name/ID；不能让缺失 ID 的 run 脱离速度、恢复和完成监控。发现 run 已经 `finished` 时，先核对 config、目标 epoch、最终 checkpoint metadata 和进程退出，再写 completion marker；动态轮询器必须在检查 tmux 之前先检查该 marker，禁止把已完成实验重新从 epoch 0 启动。

W&B 的远端 `state` 不是训练进程存活性的单一事实源。若 run 突然显示 `crashed` 或
summary epoch 停止更新，先同时检查 tmux、torchrun/rank、GPU compute process、最近
checkpoint metadata，以及本地 `wandb-core`/`debug-internal.log`。若进程仍存活、
checkpoint 已超过 W&B epoch，且日志明确显示 `file_stream` HTTP 5xx、timeout 或其他
同步故障，应登记为“训练存活、W&B sync degraded”，不能停止或重复启动训练。

### 8.1 训练时间预算与速度告警

训练启动或恢复时必须记录 `baseline_epoch`、`baseline_time` 和当前 run ID。稳定速度只
使用至少两次观测计算：

```text
rate_epoch_h = (current_epoch - baseline_epoch) / elapsed_hours
full_run_days = target_epoch / rate_epoch_h / 24
remaining_eta_days = (target_epoch - current_epoch) / rate_epoch_h / 24
```

三天门禁使用 `full_run_days`，不是 `remaining_eta_days`。单个 3000-epoch run 的完整
投影在 2--3 天内属于可接受范围；`full_run_days > 3` 时即使已经训练过半、剩余时间
少于三天，也必须向用户告警并做一次诊断。诊断至少包括：两个 rank 是否持续推进、
每张 GPU 的 utilization/显存、CPU load 和 I/O wait、数据所在快盘余量和吞吐、LMDB
dataloader 是否阻塞或报错、其他用户的 CPU/GPU 竞争，以及最近 checkpoint 的 mtime。
不能静默修改审批过的参数或直接重启。

fresh run 可以用 baseline epoch 0；恢复 run 绝不能用累计 `current_epoch / 本次恢复时长`
估速，否则会制造虚假高速。短于 15 分钟的手工复查只报告 live epoch，不更新稳定速度
baseline；首个有效窗口仍标记为 `STARTING`。连续两个检查周期 epoch 不增长，或
checkpoint 超过一个周期未更新时，标记为 `WARN_STALLED` 并先定位进程、数据加载或
资源问题；没有证据前不得把它当成正常慢速。速度监控可以和数据生成、上传、GPU
轮询并行，稳定后维持小时级检查，只有状态变化或告警才输出通知。

W&B summary 落后时，速度监控可以使用经过解析的 NAS checkpoint metadata 中的
`epoch/global_step` 作为进度源，但必须在台账记录 `progress_source=checkpoint`、
W&B epoch、checkpoint 路径和哈希。首次从 W&B 切换到 checkpoint 时重新建立速度
baseline，不能把两个来源的观测直接相减；最终完成前仍需直接加载 checkpoint 验证。

恢复流程：

1. 精确识别失败 run 和原因。
2. 保存错误日志、最后 epoch、run ID 和 checkpoint 路径。
3. 只停止该 tmux session，并验证对应 PID 退出。
4. 使用原批准配置和精确 run ID 恢复；不同 host 需先验证相同数据 manifest。
5. 确认起始 epoch 大于零且 W&B 继续同一 run。

恢复启动必须显式传入原 W&B run ID，例如
`--wandb-continue-run-id <run_id>`；该参数映射到
`wandb.continue_run_id`，由训练程序从该 run 的最新可用 checkpoint 恢复模型、优化器、epoch
和 global step。不要用 `training.load_checkpoint_run_id` 替代它：后者是加载权重但创建新 run
的路径，不满足断点续训要求。动态 GPU 轮询器在确认原 tmux 已退出后，会从 registry 最新的
`started/resumed` 行读取精确 run ID 并自动传入该参数。

`continue_run_id` 只能证明续接同一个 W&B run，不能单独证明选中了磁盘上最新的
checkpoint。恢复前先直接读取所有候选 checkpoint 的 `epoch` 和 `global_step`，确定
唯一的最新有效文件；若 W&B 文件、缓存或历史 exact-name 输出目录可能比该文件旧，
必须使用训练程序已有的显式本地 fallback 配置把已验证文件加入解析路径，并在创建
tmux 前设置最低 epoch 门槛。不得为此临时修改核心训练代码。启动后还必须同时核对：
日志指向预期 checkpoint、运行配置中的 `training.start_epoch=checkpoint_epoch+1`、
W&B run ID 未变化。任一项不符就只停止本次错误启动，登记为失败后再修正恢复入口。

若 checkpoint 位于所有训练服务器都可访问的 NAS，共享 checkpoint 不构成 host
亲和性：恢复调度应在任意满足条件的稳定服务器上抢占双卡。只需要把本地训练数据
搬到目标服务器的快速盘并完成 manifest/hash/full-stats 门禁；恢复命令继续引用同一
NAS checkpoint 绝对路径和原 W&B run ID。不要因为 checkpoint 最初由某台服务器写入，
就无条件等待该服务器空卡。

每次启动或恢复都要在 `logs/<campaign>_runs.md` 记录 host、GPU、tmux、W&B run name/ID、
checkpoint 路径、epoch/global step、SHA256、训练代码 Git commit 和 Python/PyTorch/CUDA
环境版本；`logs/<campaign>_training_speed.tsv` 记录同一 run ID 的当前 epoch、有效速度
`rate_epoch_h`、`full_run_days`、`remaining_eta_days` 和告警状态。恢复后的速度 baseline
从恢复时间和恢复起始 epoch 重新建立，不得把恢复前的速率混入新窗口，也不得让短周期
人工检查覆盖最后一个稳定 baseline。

收到 `SIGTERM` 时，先检查 `dmesg`、`journalctl`、`earlyoom`、`systemd-oomd`、GPU 进程和数据错误；不能仅凭 pane 中的 `Killed` 判断为 OOM，也不能因为 SSH 断开就删除数据。若 checkpoint 可用，恢复命令必须同时保留原 W&B run ID、数据 manifest、annotation flags 和训练配置。W&B 已有较后 log step 而 checkpoint 较早时，恢复初期可能出现 step 非单调警告；确认 epoch 和 loss 正常推进后再继续运行。

资产/训练 registry 中最新状态为 `started` 或 `resumed` 时都表示该实验已经登记，轮询器不得重复创建新 run；只有在确认进程退出且存在可恢复 checkpoint 时，才按精确 run ID 执行恢复。

完成验证不能只看 tmux 消失：正常路径必须同时检查最终 W&B config、目标 epoch、last
checkpoint metadata 中的 epoch/global step、日志无未处理异常、远端训练进程退出和
W&B `finished` 状态。完成 marker 是调度终止锁，不是人工备注；只有上述证据齐全才能
写入，写入后 poller 不得再次启动该 experiment。

如果 W&B 已被证明发生持续同步故障，可使用严格的降级完成门禁替代远端
`finished`：最终 checkpoint 必须可直接加载且精确达到目标 epoch/global step；已保存
或仍可读取的最终配置必须通过全部实验映射校验；远端 tmux 和精确训练 PID 均已退出；
训练日志没有未处理的 Traceback/NCCL/OOM；本地 W&B run bundle 与明确的 sync/API
错误日志必须保留。完成 registry/event 必须记录原 run ID、`wandb_state` 和降级原因。
任一证据缺失时不得写 completion marker，也不得把中间 checkpoint 当成完成。

轮询器可能先观察到训练进程退出并追加 `failed/process-dead`，而 W&B 和异步 checkpoint
随后已经完成收尾。完成监控不得因为这条失败行丢弃最近的 `started/resumed` run 身份；
仍需用原 run ID 执行完整完成门禁。若 W&B `finished`、最终 epoch/checkpoint 和远端退出
全部通过，应登记为 `finished` 并写 completion marker，而不是从最终 checkpoint 再启动一次。

### 8.2 最终 checkpoint 本地归档

每个 run 通过完成门禁后，使用 `/data/hy/gpu-snatcher/auto_eval.sh` 的 `download` step
下载最终 checkpoint，并让脚本生成文件名；不要用 `--overwrite-wt-path` 绕过下载和命名。
先运行 `--help` 核对当前 CLI。典型调用为：

```bash
bash /data/hy/gpu-snatcher/auto_eval.sh \
  --steps download \
  --run-id <models-directory中的run-name> \
  --project <wandb-project> \
  --local-path <local-checkout> \
  --task one_leg+round_table+lamp \
  --remote-ssh-host '' \
  --checkpoint-pattern '*last*.pt'
```

当前脚本的 `--run-id` 用于匹配 outputs 路径；若 W&B ID 不出现在模型目录名中，应传
实际 run name，并检查脚本打印的 selected output/checkpoint 正是完成门禁对应文件。
NAS 在本机不可见时设置实际 SSH host；可见时使用空 host 从挂载路径复制。脚本的目标
目录可能仍包含历史 randomness 默认值，资产台账必须记录实际绝对路径，不能凭目录名
推断 checkpoint 配置。

下载后逐个直接 `torch.load(..., map_location='cpu')` 核对 epoch/global step，比较源目标
字节数和 SHA256，并在 `logs/<campaign>_checkpoint_downloads.tsv` 记录 experiment、run
name/ID、源路径、脚本生成的目标路径、bytes、hash 和验证结果。若最终文件只存在于某台
服务器本地盘，先直接加载和计算 hash，再用可续传 rsync 补入相同 NAS output 结构；NAS
ACL 不允许保留 owner/group/perms 时显式关闭这些属性，临时文件通过 `cmp`/SHA256 后原子
改名。之后仍由原 auto-eval download 流程发现和命名，不手工伪造本地归档文件名。

## 9. Agent 执行模板

用户可直接给 agent 以下指令：

```text
请按 reports/claude/batch_train_guide.md 创建一个新的 multi-task-rgbd-skill campaign。
先从仓库和服务器实时审计已知配置，再询问我仍缺少的配置。
在 logs 下持久化 plan、数据资产表和 run 表。
先给我审批：1) 精确删除目录；2) 数据集/teacher/采集实现；3) 完整训练矩阵。
三项批准后按顺序执行，上传只使用有线网络和快速盘，随后轮询空闲双卡。
持续验证和恢复，直到所有批准的训练达到目标 epoch 并有可用 checkpoint。
```

本手册中的示例值都必须被本次审批表覆盖。任何历史数据位置、服务器容量、W&B run 或 checkpoint 都要在使用前重新验证。
