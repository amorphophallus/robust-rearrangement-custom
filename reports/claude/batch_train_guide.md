# 批量训练实验操作手册

## 1. 实验名称与超参对照表

**共同固定参数** (所有 8 个实验):

```
data.demo_source=rollout
# data.data_subset 不传（默认 None = 使用全部数据）
data.demo_outcome=success
data.storage_format=lmdb
data.load_into_memory=false
data.dataloader_workers=4
training.batch_size=512
training.num_epochs=3000
training.steps_per_epoch=100
training.save_per_epoch=500
wandb.project=multi-task-rgbd-skill-low-0608
wandb.mode=online
randomness=low
dryrun=false
data.ddp_shard_enabled=true
"task=[one_leg, round_table, lamp]"
```

**各实验差异参数**:

| # | 实验 | `data.suffix` | `data.annotate_guidance_point` | `data.annotate_skill_one_hot` | `data.annotate_guidance_point_colored` | `data.annotate_grasp` | `data.annotate_grasp_colored` | `data.annotate_grasp_part` | `experiment` | `vision_encoder` | 数据来源 |
|---|------|---------------|:---:|:---:|:---:|:---:|:---:|:---:|------------|----------------|---------|
| 1 | rgbd | `rgbd` | false | false | false | false | false | false | `rgbd/dit` | (none) | 228/data, 230/data, 236/home, 243/home |
| 2 | rgbd+GP | `rgbd-skill-point` | true | false | false | false | false | false | `rgbd/dit` | (none) | point 数据集 |
| 3 | rgbd+colored GP | `rgbd-skill-point-colored` | true | false | true | false | false | false | `rgbd/dit` | (none) | colored point 数据集 |
| 4 | rgbd+only skill | `rgbd-only-skill` | false | true | false | false | false | false | `rgbd/dit` | (none) | only-skill 数据集 |
| 5 | rgbd+GP+skill | `rgbd-skill-point` | true | true | false | false | false | false | `rgbd/dit` | (none) | 同 #2 |
| 6 | rgb | `rgbd` | false | false | false | false | false | false | `image/dit` | `resnet, pretrained=false` | 同 #1 |
| 7 | rgbd+grasp-part | `rgbd-skill-grasp-part` | false | false | false | false | false | true | `rgbd/dit` | (none) | grasp-part 数据集 |
| 8 | rgbd+colored grasp-part | `rgbd-skill-grasp-part-colored` | false | false | true | false | true | true | `rgbd/dit` | (none) | colored grasp-part 数据集 |

**对应的 auto_train_multi_card.sh 变量设置**:

```
Exp1: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=rgbd/dit
Exp2: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=rgbd/dit
Exp3: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="true"  DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=rgbd/dit
Exp4: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="true"  DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=rgbd/dit
Exp5: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="true"  DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=rgbd/dit
Exp6: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="false" +experiment=image/dit
Exp7: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="false" DATA_ANNOTATE_GRASP_PART="true"  +experiment=rgbd/dit
Exp8: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="true"  DATA_ANNOTATE_GRASP="false" DATA_ANNOTATE_GRASP_COLORED="true"  DATA_ANNOTATE_GRASP_PART="true"  +experiment=rgbd/dit
```

> `data.suffix` 描述的是数据集/图像 annotation 形态，不等于模型输入开关。
> 例如 Exp2 与 Exp5 读取同一个 `rgbd-skill-point` 数据集，但通过 `data.annotate_skill_one_hot` 区分 checkpoint 配置。
> Exp6 注意事项：需要取消注释 `vision_encoder=resnet` 和 `vision_encoder.pretrained=false`，并把 `+experiment=rgbd/dit` 改为 `+experiment=image/dit`。

---

## 2. 4090 服务器白名单与磁盘信息

| 服务器 | SSH Host | 目标盘 | 顺序读 MB/s | 当前可用 | 训练用数据路径 | 已有数据 (在快速盘上) | 备注 |
|--------|----------|--------|------------|---------|---------------|---------------------|------|
| 228 | `zju_4090_228` | NVMe `/data` + `/home` | 1,800 | `/data` 50G, `/home` 218G | 均可 | (无 lmdb 找到) | 双 NVMe，均可用; 受保护 rgbd 应在 /data 但路径待确认 |
| 230 | `zju_4090_230` | NVMe `/data` | 1,900 | 162G | `/data/hy/robust-rearrangement-custom/data/` | (无 lmdb 找到) | `/` 已满 (0 可用); 需先清空间 |
| 232 | `zju_4090_232` | NVMe `/` | 3,300 | **595G** | `~/robust-rearrangement-custom/data/` | (无 lmdb 在 NVMe 上) | **优先使用**; 空间最充裕; /data 是 HDD (仅作中转) |
| 236 | `zju_4090_236` | SATA SSD RAID `/` | 4,900 | 167G | `~/robust-rearrangement-custom/data/` | (无 lmdb 找到) | 空间紧张 |
| 238 | `zju_4090_238` | SATA SSD `/` | 4,300 | 315G | `~/robust-rearrangement-custom/data/` | **rgbd** 466G (5 shards) | 已有 rgbd 数据可直接用于 Exp1/4/6 |
| 240 | `zju_4090_240` | SATA SSD `/` | 3,700 | 233G | `/data/hy/robust-rearrangement-custom/data/` | (无 lmdb 找到) | |
| 243 | `zju_4090_243` | NVMe SSD `/` | 4,400 | **15G** | `~/robust-rearrangement-custom/data/` | (NVMe 几乎满) | NVMe 不可用; /data HDD 有 `rgbd-skill-point*` 全量 (~640G) 仅作 SCP 源 |

> **重要变化 (2026-06-08 实测)**: 与 5 月 benchmark 时期相比，多台服务器磁盘使用率大幅上升。
> 243 的 NVMe 仅剩 15G，无法训练。232 的 NVMe (595G) 是当前最优选择。
> 数据集实际大小: 单个 lmdb shard ≈100-120G，rgbd 完整数据集 ≈466G (5 shards)，point/grasp-part 系列通常与 `rgbd-skill-point*` 同量级。

> **⚠️ HDD 禁令**: 训练数据**绝对禁止**放在 HDD 上。即使系统盘是 SSD，同时读写 HDD 也会导致同服务器上所有 SSD 训练卡顿（IO 竞争拖慢 page cache）。
> - 232 `/data` (HDD sda1)、243 `/data` (HDD sda1) 只能用于数据备份，不能用于训练
> - 训练数据路径必须指向 NVMe/SSD: `~/` (系统盘) 或 230/228 的 `/data` (NVMe)
> - 传输数据时也要从 HDD 源 SCP 到目标 SSD，不能直接从 HDD 训练

---

## 3. 数据集位置与备份策略

### 3.1 路径结构关键发现

`add_subdir(task=["one_leg","round_table","lamp"])` → 目录名 `lamp-one_leg-round_table`

训练代码期望的完整路径:
```
{DATA_DIR_PROCESSED}/processed/diffik/sim/lamp-one_leg-round_table/rollout/low/success/{suffix}.lmdb/
```

> `round_table/` 是旧的单任务格式，多任务训练不会找到该路径下的数据。

### 3.2 实测数据分布 (2026-06-08 全面审计)

**多任务格式 (`lamp-one_leg-round_table/`)** — 可直接使用:

| 服务器 | 盘 | 数据集 | shard | 大小 |
|--------|-----|--------|-------|------|
| 228 | NVMe `/data` | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 230 | NVMe `/data` | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 230 | NVMe `/data` | rgbd-skill-point-colored | rgbd-skill-point-colored-1 | 125G |
| 236 | SSD `/home` | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 236 | SSD `/home` | rgbd-skill-point-colored | rgbd-skill-point-colored-1 | 125G |
| 240 | SSD `/data` | rgbd-skill-point-colored | rgbd-skill-point-colored-1 | 125G |
| 243 | NVMe `/home` (15G!) | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 243 | NVMe `/home` (15G!) | rgbd-skill-point | rgbd-skill-point-1 | 125G |
| 243 | HDD `/data` | rgbd-skill-point-colored | rgbd-skill-point-colored-1 | 125G |

**旧格式 (`round_table/`)** — 需移/连到多任务路径:

| 服务器 | 盘 | 数据集 | shard | 大小 |
|--------|-----|--------|-------|------|
| 238 | SSD `/home` | rgbd | rgbd-1~5 | 466G (5 shards) |
| 243 | HDD `/data` | rgbd-skill-point | rgbd-skill-point, rgbd-skill-point-1~5 | 640G (6 shards) |
| 243 | HDD `/data` | rgbd-only-skill | rgbd-only-skill-1 | 118G |
| 232 | HDD `/data` | rgbd-skill-point | rgbd-skill-point | 116G |

### 3.3 三个不可删除的备份数据集

| 数据集 | 位置 |
|--------|------|
| rgbd | **未找到多任务格式** — 238 有 `round_table/rgbd-*` (466G)，需移路径 |
| rgbd+GP | 243 HDD `/data` `round_table/rgbd-skill-point*` (640G) |
| rgbd+colored GP | 240 `/data` `lamp-one_leg-round_table/rgbd-skill-point-colored-1` (125G) |

### 3.4 各实验数据准备方案

| # | 实验 | 数据后缀 | 已有数据 (快速盘+正确格式) | 方案 |
|---|------|---------|--------------------------|------|
| 1 | rgbd | rgbd | **无** (仅 238 有旧格式) | 从 238 SCP rgbd-* 到目标服务器，建 symlink 或改名到 `lamp-one_leg-round_table/` 下 |
| 2 | rgbd+GP | rgbd-skill-point | 243 NVMe 有 1 shard (125G) 但盘满 | 从 243 HDD SCP 到 232 NVMe; 需移路径 |
| 3 | rgbd+colored GP | rgbd-skill-point-colored | 236 SSD 125G, 240 SSD 125G, 230 NVMe 125G | **可直接用** 236 或 240 |
| 4 | rgbd+only skill | rgbd-only-skill | 228 NVMe 123G, 230 NVMe 123G, 236 SSD 123G | **可直接用** 228/236; 或 fallback 到 rgbd |
| 5 | rgbd+GP+skill | rgbd-skill-point | 同 #2 | 同 #2 |
| 7 | rgbd+grasp-part | rgbd-skill-grasp-part | 暂无历史快照 | 用新数据收集流程生成 |
| 8 | rgbd+colored grasp-part | rgbd-skill-grasp-part-colored | 暂无历史快照 | 用新数据收集流程生成 |
| 6 | rgb | rgbd | 同 #1 | 同 #1 |

### 3.5 数据传输流程

```bash
# 1. 检查目标服务器 SSH 连通性
ssh -o ConnectTimeout=5 <target_host> echo ok

# 2. 检查目标空间 (需要 ~650G)
ssh <target_host> "df -h <data_path>"

# 3. 如果空间不足，清理非受保护数据
# 注意: 绝对不能删除:
#   240 /data: rgbd-skill-point-colored*.lmdb
#   251 /data: rgbd-skill-point*.lmdb
#   228 /data: rgbd*.lmdb
# 其他都可以清理

# 4. SCP 传输 (从备份源到目标)
# 示例: 从 228 传 rgbd 数据到 232
ssh zju_4090_228 "cd /data/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success && tar czf - rgbd*.lmdb" | \
  ssh zju_4090_232 "cd ~/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success && tar xzf -"

# 5. 传输完成后验证数据完整性
ssh <target_host> "du -sh <data_path>/<suffix>*.lmdb && ls <data_path>/<suffix>*.lmdb/data.mdb"

# 6. 更新 Notion 数据准备表
```

### 3.6 数据集路径验证 (每次训练启动前)

S CP 完成后，必须验证数据集在目标服务器上的路径与 `auto_train_multi_card.sh` 中的 `DATA_DIR_PROCESSED` + suffix 拼接结果一致:

```bash
# 训练代码期望的完整路径结构:
# ${DATA_DIR_PROCESSED}/processed/diffik/sim/{task}/rollout/{randomness}/{outcome}/{suffix}.lmdb/
#
# 示例: DATA_DIR_PROCESSED=~/robust-rearrangement-custom/data/
# 完整路径应为: ~/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/rgbd*.lmdb/

# 验证命令:
ssh <target_host> "ls -d \$(eval echo <DATA_DIR_PROCESSED>)/processed/diffik/sim/round_table/rollout/low/success/<suffix>*.lmdb"
```

验证失败常见原因:
- lmdb 放错目录层级
- `DATA_DIR_PROCESSED` 设置成了 lmdb 的父目录而非 `data/` 目录
- lmdb 目录名称不匹配 suffix（如 `rgbd-skill-point.lmdb` vs `rgbd-skill-point-colored.lmdb`）

---

## 4. auto_train_multi_card.sh 关键参数说明

脚本位置：`~/projects/gpu-snatcher/auto_train_multi_card.sh`

### 4.1 需修改的参数 (行号)

| 参数 | 说明 |
|------|------|
| `DATA_ANNOTATE_GUIDANCE_POINT` | 模型是否额外输入 guidance point |
| `DATA_ANNOTATE_SKILL_ONE_HOT` | 模型是否额外输入 skill one-hot |
| `DATA_GUIDANCE_POINT_COLORED` | point annotation 是否 colored；在 `grasp-part-colored` 中也要一起开启 |
| `DATA_ANNOTATE_GRASP` | 数据集是否为 grasp 全量 annotation 模式 |
| `DATA_ANNOTATE_GRASP_COLORED` | grasp annotation 是否 colored |
| `DATA_ANNOTATE_GRASP_PART` | 数据集是否为 grasp-part 模式 (`pick/place`=grasp, 其他=point) |
| `+experiment=...` | `rgbd/dit` 或 `image/dit` |
| `vision_encoder=resnet` 等 | Exp6 需取消注释 |
| `SSH_NAME` | 目标服务器编号 (不含 `zju_4090_` 前缀) |
| `DATA_DIR_PROCESSED` | 训练数据根目录 |

### 4.2 修改前检查清单

- [ ] 确认目标服务器代码已 pull 到最新 (`ssh <host> "cd <project> && git pull"`)
- [ ] 确认 conda 环境 rr 存在
- [ ] 确认 `RUNTIME_TMP_ROOT` 路径存在（默认 `/data/hy/tmp`）
- [ ] 确认 `wandb.project` 正确
- [ ] 确认 `training.num_epochs=3000`（非 2000）

### 4.3 串行启动与原子锁（防止配置错误）

**严重教训**: 连续修改 `auto_train_multi_card.sh` 启动多个实验时，极易忘记切换某个参数（如 `DATA_GUIDANCE_POINT_COLORED`），导致实验配置错误。**所有实验启动必须是串行原子操作。**

**原子锁流程**（每个实验）:
```bash
LOCKFILE=/tmp/auto_train.lock

# 1. 获取锁
exec 9>$LOCKFILE
flock 9 || exit 1

# 2. 修改 auto_train_multi_card.sh 参数（所有关键变量一次性修改）
#    DATA_ANNOTATE_GUIDANCE_POINT, DATA_ANNOTATE_SKILL_ONE_HOT,
#    DATA_GUIDANCE_POINT_COLORED, DATA_ANNOTATE_GRASP,
#    DATA_ANNOTATE_GRASP_COLORED, DATA_ANNOTATE_GRASP_PART,
#    +experiment, SSH_NAME, GPU_ID, WANDB_CONTINUE_RUN_ID

# 3. 验证参数正确（grep 所有关键变量，打印到 stdout 确认）
grep -E '^(DATA_ANNOTATE|DATA_GUIDANCE|WANDB_CONTINUE|SSH_NAME|GPU_ID)' auto_train_multi_card.sh
grep 'experiment=' auto_train_multi_card.sh

# 4. 启动训练
bash ./auto_train_multi_card.sh

# 5. 确认训练启动成功（GPU 内存 > 10GB）

# 6. 释放锁
flock -u 9
```

**禁用并行修改**: 绝不在同一时刻修改 auto_train_multi_card.sh 启动两个实验。前一个完全确认跑起来后再改参数启动下一个。

### 4.4 训练启动后必做: Training Config 双检 (Double Check) ⚠️

**每次实验启动后必须执行，不允许跳过。** 脚本的参数设置（bash 变量 → 命令行参数）可能因拼写、copy-paste 或默认覆盖而偏离预期。唯一完全真实的参数来源是训练代码启动时打印的 Hydra config。

**执行时机**: 训练启动后，从 tmux 输出中获取 config dump（约在 wandb init 之前打印）。

**双检项**（对照 §1 实验超参表）:

| 检查项 | config key | 正则 |
|--------|-----------|------|
| GP 输入开关 | `annotate_guidance_point` | `annotate_guidance_point: (true\|false)` |
| Skill One-hot 开关 | `annotate_skill_one_hot` | `annotate_skill_one_hot: (true\|false)` |
| Colored Point 开关 | `annotate_guidance_point_colored` | `annotate_guidance_point_colored: (true\|false)` |
| Grasp 开关 | `annotate_grasp` | `annotate_grasp: (true\|false)` |
| Colored Grasp 开关 | `annotate_grasp_colored` | `annotate_grasp_colored: (true\|false)` |
| Grasp-part 开关 | `annotate_grasp_part` | `annotate_grasp_part: (true\|false)` |
| 数据后缀 | `suffix` | `suffix: <expected>` |
| Experiment | `experiment` | 隐含在 vision_encoder 加载信息中 |

**执行方法**:
```bash
# 从 tmux 输出中提取 config 关键字段
ssh <host> "tmux capture-pane -pt <session>:train -S -200" 2>/dev/null | \
  grep -E 'annotate_guidance_point:|annotate_skill_one_hot:|annotate_guidance_point_colored:|annotate_grasp:|annotate_grasp_colored:|annotate_grasp_part:|suffix:'

# 示例正确输出 (Exp4 rgbd+only skill):
#   annotate_guidance_point: false
#   annotate_guidance_point_colored: false
#   annotate_skill_one_hot: true
#   suffix: rgbd-only-skill
```

**判定**: 三项开关 + suffix 必须与 §1 表中该实验的要求完全一致。任何一项不匹配 → **立即 `tmux send-keys C-c` 终止训练**，修正脚本参数后重新启动。

**示例 — Exp4 (rgbd+only skill) 双检实录 (2026-06-27)**:
```
annotate_guidance_point: false          ✅ (期望 false)
annotate_guidance_point_colored: false  ✅ (期望 false)
annotate_skill_one_hot: true            ✅ (期望 true)
suffix: rgbd-only-skill                 ✅ (期望 rgbd-only-skill)
```

> **⚠️ 强制规则**: 此双检流程写入本文档后，对所有后续实验生效。任何实验跳过双检直接运行的，视为配置未验证，结果不可信。

### 4.5 同服务器多实验并行策略

当一台服务器有空闲的 2+ 张额外 GPU 时，可以在同一台服务器上同时跑多个实验。

**前提条件**:
- 数据盘为 NVMe/SSD（非 HDD）。NVMe 随机读 >700MB/s，2 个训练并行 IO 影响小
- HDD 服务器绝对不要并行 — 随机 IOPS 瓶颈会导致 10x 降速（参考 benchmark report 5.4 节）
- 每个实验需 2 张 GPU（共 2×20GB ≈ 40GB 显存），确保目标 GPU 空闲
- 内存充足（每实验 ~10-20GB RSS 增长）

**操作步骤**:
1. 确认目标服务器 GPU 状态: `./check_zju_gpu.sh | grep <host>`
2. 用 `GPU_ID="<a>,<b>"` 显式指定 GPU（避免与已有实验抢卡）
3. 确认该服务器有所需数据集（或提前 SCP）
4. 修改 `auto_train_multi_card.sh` 后直接启动 — 脚本自动分配不同 tmux session 名

**示例**: 230 上同时跑 Exp3 (GPUs 0,1, tmux: atlas) + Exp4 (GPUs 2,4, tmux: birch)

```bash
# Exp4 启动前设置
SSH_NAME="230"
GPU_ID="2,4"              # 显式指定，避让 Exp3 的 0,1
DATA_DIR_PROCESSED="/data/hy/robust-rearrangement-custom/data/"
```

**注意事项**:
- `wandb.project` 同一项目下自动分配不同 run_name，无需修改
- 每个实验独立的 tmux session，互不干扰
- 如果某实验 crash (OOM)，优先降低 `training.batch_size` 到 256
- 3 个及以上实验并行时注意监控内存 + page cache 竞争

---

## 5. 执行流程

### 阶段 1: 前置准备 (数据摸底)

在启动任何训练前，先对全部 7 台 4090 服务器做数据摸底：

```bash
# 每台服务器搜索所有 lmdb 数据
for host in 228 230 232 236 238 240 243; do
    echo "=== zju_4090_$host ==="
    ssh zju_4090_$host "find /data ~ -maxdepth 10 -name '*.lmdb' -type d 2>/dev/null | head -30; echo '---disk---'; df -h / /data 2>/dev/null | grep -v tmp"
done
```

然后根据实际找到的数据位置更新 §3.2 数据分布表，再开始调度实验。

同时:
- [ ] 确认所有服务器代码已 pull 到最新
- [ ] 确认 conda 环境 rr 存在
- [ ] 确认 wandb 登录状态
- [ ] 确认 `RUNTIME_TMP_ROOT` 路径存在

### 阶段 2: 紧循环调度 (while 直到无 GPU 或无空间)

**核心原则**: 不在实验之间插入不必要的等待。找到一个空闲双卡就立即启动下一个实验，直到所有待跑实验完成或没有任何服务器能承接新实验。

```
待跑队列: [Exp1, Exp2, Exp3, Exp4, Exp5, Exp6]  (按实验号顺序)
每个实验状态: pending | preparing | running | done | failed

while 有待跑实验 (pending):
    1. bash ./check_zju_gpu.sh → 找有空闲双卡的 4090 服务器
    2. if 无任何服务器有 2+ 空闲 GPU:
       - 如果已有 running 实验: 检查它们状态，等待 GPU 释放
       - 如果没有任何 running 实验: sleep 60，continue
    3. if 有空闲双卡: 立即从待跑队列取下一个 pending 实验，不等待
    4. 选一台可用服务器 — 优先同一台已有数据且有空闲额外 GPU 的 (可并行)
    5. 检查/准备数据:
       a. 数据必须放在快速盘 (NVMe/SSD) 上 — **绝对禁止从 HDD 读训练数据**
       b. 如果 /data 是 HDD (如 232/243)，必须用 ~/ 路径 (系统 SSD)
       c. ssh <host> "ls <data_path>/<suffix>*.lmdb" → 数据存在?
       d. 不存在 → 检查快速盘空间，必要时清腾非受保护数据，SCP 传输
       e. 更新 Notion 数据准备表
    6. 修改 auto_train_multi_card.sh 超参数 (按 §1 表)
       - 如果目标服务器已有其他实验，设置 GPU_ID="<a>,<b>" 避让
    7. 确认 GPU 仍空闲 → bash ./auto_train_multi_card.sh
    8. 启动成功后: **立即登记到 Notion 实验过程表**（ssh, tmux, wandb proj, run_name），标记 running。⚠️ 不可遗漏!
      获取 run name: 从 tmux 输出或 `wandb.Api().runs()` 查最新 run
    9. 回到步骤 1 立即检查是否还能再启动一个
```

### 阶段 3: 训练监控 (分级)

**第 1 小时** — 每 10 分钟检查一次:
- 检查 crash (Traceback/Error/Killed/OOM)
- 检查训练速度 (目标 ~2min/epoch，即 >0.8 it/s for batch_size=512)
- 确认 wandb 正常同步

**1 小时后** — 如果训练稳定 (无 crash，速度正常，loss 正常下降):
- 降频到每 6 小时检查一次
- 仍通过 cron 自动执行

```bash
# 检查脚本
ssh <host> "tmux capture-pane -pt <session>:train -S -30" | \
  grep -E "Traceback|Error|Killed|OOM|it/s|epoch.*3000|CheckpointSaver"
```

**常见异常处理**:
| 异常 | 处理 |
|------|------|
| OOM | batch_size 降到 256 |
| 速度显著慢 (>3min/epoch) | 检查 DATA_DIR_PROCESSED 是否在 HDD；迁移到 SSD |
| tmux 僵死 | ssh <host> "tmux send-keys -t <session>:train C-c" 重启 |
| 磁盘满 | 清理非受保护数据 |

### 阶段 4: NAS 监控（每 30 分钟）

NAS (`/mnt/nas/share`) 断连或满盘是训练崩溃的首要原因。必须持续监控。

```bash
# 检查 NAS 连通性和空间
ssh <any_server> "df -h /mnt/nas/share | tail -1; ls /mnt/nas/share/home/hy/robust-rearrangement-custom/.git >/dev/null 2>&1 && echo NAS_OK || echo NAS_DOWN"
```

**阈值**:
- NAS 空间 < 50G: ⚠️ 警告，通知用户清理
- NAS 空间 < 10G: 🔴 紧急，训练 checkpoint 将失败
- NAS 不可达 (ls 失败): 🔴 停止启动新实验，已运行实验可能崩溃
- NAS 恢复后: 检查是否有 D-state 残留进程，清理后重启

**NAS 空间清理优先级**:
1. 旧 checkpoint (`outputs/` 下超过 7 天的目录)
2. 旧 wandb runs (wandb 云端的本地缓存)
3. 其他用户的大文件（需沟通）

```bash
# 清理 7 天前的 outputs
ssh <server> "find /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/ -maxdepth 2 -type d -mtime +7 -exec rm -rf {} \;"
```

---

## 6. 常见问题与处理

| 异常 | 现象 | 处理 |
|------|------|------|
| SSH 不通 | check_zju_gpu.sh 报 DOWN | 跳过该服务器，等下一轮 |
| GPU 被抢占 | 启动后发现 GPU busy | kill tmux，放回队列等待 |
| 数据空间不足 | df 显示 <650G | 删除非受保护数据，或换服务器 |
| SCP 超时/中断 | tar 管道断开 | 检查 SSH，用 rsync 替代 |
| 训练 crash (OOM) | CUDA out of memory | batch_size 降到 256 |
| 训练速度慢 | >3min/epoch | 检查 DATA_DIR_PROCESSED 是否在 SSD 上 |
| wandb 登录失效 | wandb login 过期 | ssh <host> "wandb login --relogin" |
| tmux session 名冲突 | 所有候选名都被占用 | 手动指定一个未使用的名字 |

---

## 7. Notion 同步要求

### 7.1 数据准备表更新

每当数据在服务器间移动后，更新 Notion 页面 "多任务 condition 对比实验" (date: 2026-06-08) 中的「数据准备」表格:
- 在对应标注类型的「存在服务器」列追加新服务器

### 7.2 实验过程表更新

每个实验启动后，填写「实验过程」表格:
| 列 | 值 |
|----|-----|
| 实验 | 实验名 (如 rgbd+colored GP) |
| ssh | 服务器编号 (如 236) |
| tmux | session 名 (如 atlas) |
| wandb proj | multi-task-rgbd-skill-low-0608 |
| wandb run_name | 从训练输出解析 |

### 7.3 更新工具

使用 Notion MCP tools:
- `notion-update-page` 更新页面内容
- `notion-search` 查找相关页面

---

## 8. 调度优化策略

### 8.1 服务器分配优先级

为减少数据传输，尽量将实验分配到已有对应数据的服务器:

```
Exp1 (rgbd):            238 (本地) > 232 > 228
Exp2 (rgbd+GP):         232 (SCP from 243 HDD) > 238 (需清腾)
Exp3 (rgbd+colored GP): 待确认数据位置, 优先 232
Exp4 (rgbd+only skill): 238 (本地 + fallback) > 232 > 228
Exp5 (rgbd+GP+skill):   同 #2
Exp6 (rgb):             同 #1
```

> 232 NVMe (595G) 建议留给 `rgbd-skill-point*` / `rgbd-skill-grasp-part*` 数据集，rgbd 数据集优先用 238 (本地已有)。

### 8.2 并行策略

- 每台服务器同时只跑 1 个训练（避免 IO 竞争导致 10x 降速）
- 如果多台服务器同时有空闲 GPU，可并行启动多个实验
- 并行时优先将实验分配到不同服务器上

### 8.3 传数据策略

- 优先使用已有数据的服务器
- 如果必须传输，首选带宽大的源-目标对
- 一次传输整个 lmdb 目录（含所有 shard）
- 传输完成后**立即更新 Notion 数据准备表**，供后续实验复用
- **数据集有任何传输或删除操作，必须登记到 Notion 数据准备表**。格式: `服务器 /路径 (状态)`，例如 `240 /home SSD (rgbd-skill-point 125G)`
- 这是必做步骤，避免后续实验找不到数据

---

## 9. 训练中断与断点重训

### 9.1 检测中断

每轮监控时检查以下信号:
- tmux pane 中是否有 `Traceback` / `Error` / `Killed` / `OOM`
- GPU 显存是否异常归零（进程 crash）
- wandb 面板是否显示 `crashed`
- SSH 连接是否持续失败（服务器宕机/网络故障）

### 9.2 中断分类与处理

| 类型 | 症状 | 处理 |
|------|------|------|
| **OOM** | CUDA out of memory | `training.batch_size` 降到 256，重启 |
| **磁盘满** | No space left on device | 清理非保护数据，重启 |
| **NAS 挂死** | 进程 D-state, kill -9 无效 | 等 NAS 恢复后进程自动解除；清理僵尸进程后重启 |
| **服务器宕机** | SSH timeout 持续 | 等服务器恢复；检查/清理僵尸进程；在新服务器上恢复 |
| **进程被抢占** | GPU 显存被其他进程占用 | 找其他空闲 GPU/服务器，继续运行 |
| **wandb 同步断** | 训练正常但 wandb 显示 crashed | 训练本身不受影响，checkpoint 仍在保存；可选重启修复 wandb 显示 |

### 9.3 断点重训流程 (Resume)

当确认训练中断后:

```
1. 记录中断信息:
   - 实验编号、wandb run_id、中断原因、当前 epoch (从 tmux 或 wandb 估算)

2. 清理残留:
   - ssh <host> "tmux kill-session -t <session>"  # 清理旧 tmux
   - ssh <host> "pkill -9 -u hy -f bc_ddp"         # 清理残留进程 (注意: 只杀目标实验!)
   - 确认 GPU 释放

3. 检查 checkpoint:
   - checkpoint 保存在 NAS: /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/<date>/<time>/models/<run_name>/
   - 确认最新 checkpoint 存在: actor_chkpt_last.pt
   - 如果 NAS 不可达，checkpoint 丢失，需要从头训练

4. 选择恢复服务器:
   - 优先同服务器（数据和 checkpoint 都在）
   - 否则选任何有空闲 GPU 且有数据的服务器
   - 如果没有数据，从备份源 SCP (见 §3.3)

5. 修改 auto_train_multi_card.sh:
   - 设置 WANDB_CONTINUE_RUN_ID="<run_id>"  (注意: 是 run_id 如 rx6nfry4，不是 run_name)
   - 其他参数与原始实验一致
   - DATA_DIR_PROCESSED 和 SSH_NAME 根据目标服务器调整
   - GPU_ID 显式指定空闲 GPU（避免冲突）

6. 启动训练:
   - bash ./auto_train_multi_card.sh
   - 等待 wandb 确认 resume 成功 (epoch 不为 0)
   - 更新 Notion 实验过程表

7. 验证恢复:
   - 确认 continue_run_id 生效: tmux 输出中应显示 continue_run_id=<run_id>
   - 确认从 checkpoint 恢复: 起始 epoch > 0
   - 确认 wandb 状态恢复为 running
```

### 9.4 continue_run_id 经验总结

**正确用法**:
- `WANDB_CONTINUE_RUN_ID` 的值是 wandb **run ID**（如 `rx6nfry4`），不是 run name（如 `hopeful-planet-1`）
- run ID 可以从 wandb 面板 URL 获取，或通过 `wandb.Api().runs()` 查询
- 训练代码会先尝试从 wandb artifacts 找 checkpoint → 失败后 fallback 到 NAS 本地路径

**已知问题**:
- 如果 checkpoint 从未上传到 wandb artifacts（wandb.mode=online 不主动上传大文件），resume 依赖 NAS fallback
- NAS 不可达时 resume 失败（找不到 checkpoint）
- 部分情况下 wandb 状态无法从 crashed 恢复为 running（训练实际正常）
- wandb sync 进程 crash 后不会自动重连，需重启训练才能修复 wandb 显示
- **磁盘监控**: 多实验并行时 `/home/hy/tmp` 易满，需定时清理 pip/torch cache
- 清理命令: `rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface`
- 设置 2h cron 监控 `df -h /`，<2G 时主动清理
- `pkill -9 -u hy -f bc_ddp` 会杀死**所有** bc_ddp 进程，务必先锁定目标实验的 tmux/gpu
- **禁止 `pkill -9 -u hy`** — 会误杀同服务器所有实验
- **禁止 `for pid in $(nvidia-smi ...); do kill -9 $pid; done`** — 会杀死 GPU 上所有进程（包括其他实验和他人进程）
- **唯一安全的杀进程方法**: 先获取 GPU 上 PID，再验证该 PID 属于目标实验，最后 `kill -9 <PID>`
  ```bash
  # 正确做法：
  # 1. 找目标 GPU 上的 PID
  TARGET_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -N)
  # 2. 验证 PID 属于目标实验（检查 cmdline）
  cat /proc/$TARGET_PID/cmdline | grep -q '<wandb_run_name_or_continue_id>' && kill -9 $TARGET_PID
  # 或使用 pkill 精确过滤：
  pkill -f 'continue_run_id=<exact_id>'   # 只杀特定 continue_run_id 的进程
  pkill -f '<wandb_run_name>'              # 只杀特定 run name 的进程
  ```
- **最推荐: `tmux send-keys C-c`** — 只杀指定 tmux session 内的进程，绝不影响同服其他实验:
  ```bash
  ssh <host> "tmux send-keys -t <session>:train C-c"   # 发 Ctrl+C
  sleep 3
  ssh <host> "tmux kill-session -t <session>"           # 清理 session
  ```
  优点: 天然隔离，不可能误杀其他 tmux 的进程。知道实验在哪个 session 就能精确杀。

**SCP 数据源优先级**:
- 优先从备份数据集 SCP（见 §3.3 三个不可删除的备份）
- 使用 `ssh <target> "scp -r <source>:/path <target_path>"` 实现服务器间直传（不走本机）
- 需要先设置 SSH key 互信：将目标服务器的 `~/.ssh/id_rsa.pub` 添加到源服务器的 `authorized_keys`
- 修复 `chmod 600 ~/.ssh/id_rsa` 权限问题
- 内网直传速度 ~80-100MB/s (SSD 源)，HDD 源约 3-5MB/s

### 9.5 SCP 传输经验

**速度要求**: 1 小时必须传完（125G 需 >35MB/s），否则换方案。

**传输方式速度对比**:
| 方式 | 命令 | 速度 | 适用场景 |
|------|------|------|---------|
| 服务器间直传 (SSD源) | `ssh <target> "scp -r <src>:/path <dst>"` | **80-100 MB/s** (~20min/125G) | ✅ 首选 |
| 服务器间直传 (HDD源) | 同上 | **3-5 MB/s** (~7h/125G) | ❌ 太慢，避免 |
| scp -3 (本机中转) | `scp -3 <src> <dst>` | **2-3 MB/s** (~12h/125G) | ❌ 禁止使用 |
| tar pipe (本机中转) | `ssh <A> "tar c ..." \| ssh <B> "tar x"` | **<1 MB/s** | ❌ 禁止使用 |

**服务器间直传设置步骤**:
```bash
# 1. 获取目标服务器公钥
ssh <target> "cat ~/.ssh/id_rsa.pub"

# 2. 添加到源服务器 authorized_keys
ssh <source> "echo '<key>' >> ~/.ssh/authorized_keys"

# 3. 修复权限（常见问题）— ⚠️ 绝不覆盖现有私钥
# 先检查文件内容，确认是公钥而非私钥后再操作
ssh <target> "ls -la ~/.ssh/id_rsa ~/.ssh/id_rsa_1 2>/dev/null; head -1 ~/.ssh/id_rsa 2>/dev/null"
# 只修复权限，不修改文件内容！
ssh <target> "chmod 600 ~/.ssh/id_rsa 2>/dev/null; chmod 600 ~/.ssh/id_rsa_1 2>/dev/null"
# ⚠️ 绝不 echo > ~/.ssh/id_rsa（这会覆盖私钥！）

# 4. 测试连通性
ssh <target> "ssh <source_ip> 'echo OK'"

# 5. 执行传输（从目标拉取）
ssh <target> "scp -r <source_ip>:/path/to/data.lmdb /local/path/"

# 6. 验证
ssh <target> "ls /local/path/data.lmdb/data.mdb && du -sh /local/path/data.lmdb"
```

**传输决策树**:
```
需要传输数据?
├─ 目标服务器已有数据? → 跳过传输
├─ 源服务器是 SSD? → 服务器间直传 (20min)
├─ 源服务器是 HDD?
│   ├─ 有其他 SSD 源? → 优先用 SSD 源
│   └─ 唯一副本在 HDD? → 接受慢速传输 (但标注风险)
└─ 无法设置直传?
    ├─ 尝试 ssh -A agent forwarding
    └─ 最后手段: scp -3 (标注 12h+)
```

**数据备份源 (见 §3.3)**:
- rgbd: 228 /data (NVMe)
- rgbd+GP (`rgbd-skill-point`): 251 /data (HDD) → 慢，优先找 SSD 副本
- rgbd+colored GP (`rgbd-skill-point-colored`): 251 /data (HDD) → 慢，236/240 有 SSD 副本
