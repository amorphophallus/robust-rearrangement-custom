# 批量训练实验操作手册

## 1. 实验名称与超参对照表

**共同固定参数** (所有 6 个实验):

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

| # | 实验 | GP | One-hot Skill | Colored GP | experiment | vision_encoder | 数据后缀 | 数据来源 |
|---|------|:---:|:---:|:---:|------------|----------------|---------|---------|
| 1 | rgbd | false | false | false | rgbd/dit | (none) | rgbd | 228/data, 230/data, 236/home, 243/home |
| 2 | rgbd+GP | true | false | false | rgbd/dit | (none) | rgbd-skill | 251/data, 243/home |
| 3 | rgbd+colored GP | true | false | true | rgbd/dit | (none) | rgbd-skill-colored | 240/data, 251/data, 230/data, 243/data, 236/home |
| 4 | rgbd+only skill | false | true | false | rgbd/dit | (none) | rgbd-only-skill | 复用 rgbd（见下方说明） |
| 5 | rgbd+GP+skill | true | true | false | rgbd/dit | (none) | rgbd-skill | 同 #2 |
| 6 | rgb | false | false | false | image/dit | resnet, pretrained=false | rgbd | 同 #1 |

**对应的 auto_train_multi_card.sh 变量设置**:

```
Exp1: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" +experiment=rgbd/dit
Exp2: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" +experiment=rgbd/dit
Exp3: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="true"  +experiment=rgbd/dit
Exp4: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="true"  DATA_GUIDANCE_POINT_COLORED="false" +experiment=rgbd/dit
Exp5: DATA_ANNOTATE_GUIDANCE_POINT="true"  DATA_ANNOTATE_SKILL_ONE_HOT="true"  DATA_GUIDANCE_POINT_COLORED="false" +experiment=rgbd/dit
Exp6: DATA_ANNOTATE_GUIDANCE_POINT="false" DATA_ANNOTATE_SKILL_ONE_HOT="false" DATA_GUIDANCE_POINT_COLORED="false" +experiment=image/dit  # vision_encoder=resnet vision_encoder.pretrained=false 取消注释
```

> **Exp4 数据说明**: 当前脚本在 `GP=false` + `skill=true` 时进入 elif 分支，`DATA_SUFFIX="rgbd-only-skill"`，fallback 为空。
> 需修改 elif 分支添加 `DATA_SUFFIX_FALLBACK="rgbd"`，使 Python 层在找不到 rgbd-only-skill 数据时自动回退到 rgbd（skill one-hot 由 dataloader 运行时标注）。

> **Exp6 注意事项**: 需要取消注释 `vision_encoder=resnet` 和 `vision_encoder.pretrained=false`，并把 `+experiment=rgbd/dit` 改为 `+experiment=image/dit`。

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
| 243 | `zju_4090_243` | NVMe SSD `/` | 4,400 | **15G** | `~/robust-rearrangement-custom/data/` | (NVMe 几乎满) | NVMe 不可用; /data HDD 有 rgbd-skill 全量 (~640G) 仅作 SCP 源 |

> **重要变化 (2026-06-08 实测)**: 与 5 月 benchmark 时期相比，多台服务器磁盘使用率大幅上升。
> 243 的 NVMe 仅剩 15G，无法训练。232 的 NVMe (595G) 是当前最优选择。
> 数据集实际大小: 单个 lmdb shard ≈100-120G，rgbd 完整数据集 ≈466G (5 shards)，rgbd-skill ≈640G (6 shards)。

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
| 230 | NVMe `/data` | rgbd-skill-colored | rgbd-skill-colored-1 | 125G |
| 236 | SSD `/home` | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 236 | SSD `/home` | rgbd-skill-colored | rgbd-skill-colored-1 | 125G |
| 240 | SSD `/data` | rgbd-skill-colored | rgbd-skill-colored-1 | 125G |
| 243 | NVMe `/home` (15G!) | rgbd-only-skill | rgbd-only-skill-1 | 123G |
| 243 | NVMe `/home` (15G!) | rgbd-skill | rgbd-skill-1 | 125G |
| 243 | HDD `/data` | rgbd-skill-colored | rgbd-skill-colored-1 | 125G |

**旧格式 (`round_table/`)** — 需移/连到多任务路径:

| 服务器 | 盘 | 数据集 | shard | 大小 |
|--------|-----|--------|-------|------|
| 238 | SSD `/home` | rgbd | rgbd-1~5 | 466G (5 shards) |
| 243 | HDD `/data` | rgbd-skill | rgbd-skill, rgbd-skill-1~5 | 640G (6 shards) |
| 243 | HDD `/data` | rgbd-only-skill | rgbd-only-skill-1 | 118G |
| 232 | HDD `/data` | rgbd-skill | rgbd-skill | 116G |

### 3.3 三个不可删除的备份数据集

| 数据集 | 位置 |
|--------|------|
| rgbd | **未找到多任务格式** — 238 有 `round_table/rgbd-*` (466G)，需移路径 |
| rgbd+GP | 243 HDD `/data` `round_table/rgbd-skill*` (640G) |
| rgbd+colored GP | 240 `/data` `lamp-one_leg-round_table/rgbd-skill-colored-1` (125G) |

### 3.4 各实验数据准备方案

| # | 实验 | 数据后缀 | 已有数据 (快速盘+正确格式) | 方案 |
|---|------|---------|--------------------------|------|
| 1 | rgbd | rgbd | **无** (仅 238 有旧格式) | 从 238 SCP rgbd-* 到目标服务器，建 symlink 或改名到 `lamp-one_leg-round_table/` 下 |
| 2 | rgbd+GP | rgbd-skill | 243 NVMe 有 1 shard (125G) 但盘满 | 从 243 HDD SCP 到 232 NVMe; 需移路径 |
| 3 | rgbd+colored GP | rgbd-skill-colored | 236 SSD 125G, 240 SSD 125G, 230 NVMe 125G | **可直接用** 236 或 240 |
| 4 | rgbd+only skill | rgbd-only-skill | 228 NVMe 123G, 230 NVMe 123G, 236 SSD 123G | **可直接用** 228/236; 或 fallback 到 rgbd |
| 5 | rgbd+GP+skill | rgbd-skill | 同 #2 | 同 #2 |
| 6 | rgb | rgbd | 同 #1 | 同 #1 |

### 3.5 数据传输流程

```bash
# 1. 检查目标服务器 SSH 连通性
ssh -o ConnectTimeout=5 <target_host> echo ok

# 2. 检查目标空间 (需要 ~650G)
ssh <target_host> "df -h <data_path>"

# 3. 如果空间不足，清理非受保护数据
# 注意: 绝对不能删除:
#   240 /data: rgbd-skill-colored*.lmdb
#   251 /data: rgbd-skill*.lmdb
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
- lmdb 目录名称不匹配 suffix（如 `rgbd-skill.lmdb` vs `rgbd-skill-colored.lmdb`）

---

## 4. auto_train_multi_card.sh 关键参数说明

脚本位置：`~/projects/gpu-snatcher/auto_train_multi_card.sh`

### 4.1 需修改的参数 (行号)

| 行号 | 参数 | 说明 |
|------|------|------|
| 32 | `DATA_ANNOTATE_GUIDANCE_POINT` | true/false |
| 33 | `DATA_ANNOTATE_SKILL_ONE_HOT` | true/false |
| 34 | `DATA_GUIDANCE_POINT_COLORED` | true/false |
| 55 | `+experiment=rgbd/dit` | rgbd/dit 或 image/dit |
| 56-57 | `vision_encoder=resnet` 等 | Exp6 需取消注释 |
| 90 | `SSH_NAME` | 目标服务器编号 (不含 zju_4090_ 前缀) |
| 94 | `DATA_DIR_PROCESSED` | `~/robust-rearrangement-custom/data/` (236/238/232/243 home) 或 `/data/hy/robust-rearrangement-custom/data/` (240 local) |

### 4.2 修改前检查清单

- [ ] 确认目标服务器代码已 pull 到最新 (`ssh <host> "cd <project> && git pull"`)
- [ ] 确认 conda 环境 rr 存在
- [ ] 确认 `RUNTIME_TMP_ROOT` 路径存在（默认 `/data/hy/tmp`）
- [ ] 确认 `wandb.project` 正确
- [ ] 确认 `training.num_epochs=3000`（非 2000）

### 4.3 同服务器多实验并行策略

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
    8. 启动成功后: 记录到 Notion，标记实验为 running
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

> 232 NVMe (595G) 建议留给 rgbd-skill 数据集 (Exp2/5)，rgbd 数据集优先用 238 (本地已有)。

### 8.2 并行策略

- 每台服务器同时只跑 1 个训练（避免 IO 竞争导致 10x 降速）
- 如果多台服务器同时有空闲 GPU，可并行启动多个实验
- 并行时优先将实验分配到不同服务器上

### 8.3 传数据策略

- 优先使用已有数据的服务器
- 如果必须传输，首选带宽大的源-目标对
- 一次传输整个 lmdb 目录（含所有 shard）
- 传输完成后立即更新 Notion 数据表，供后续实验复用
