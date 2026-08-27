# ppu96：zstd LMDB + 4 路单卡训练执行手册

**状态**：基础设施方案已固定并完成容量实测；正式科学实验的四行 condition matrix 必须在启动前登记。
**基准日期**：2026-08-27。
**适用机器**：SSH 别名 `ppu96`，2 张 96 GiB PPU。
**目标读者**：后续接手任务、只拿本文作为主要上下文的 Codex/agent。

本文是本次方案的权威执行 runbook。`reports/claude/batch_train_guide.md` 仍然适用于通用的数据审批、资产台账和完成定义，但其中的“双卡 DDP”启动方式不适用于本方案。本方案的每个实验都是 `WORLD_SIZE=1`，四个独立实验按 `2 + 2` 共享两张物理 PPU。

## 0. 给接手 Codex 的强制说明

1. 先完整阅读本文，再执行任何远程写操作或训练启动。
2. 本文固定的是数据格式、机器、磁盘、并发拓扑、单进程训练入口和共同性能参数。不要把压力测试中四个同配置进程误当成四个科学 condition。
3. 正式启动前，必须把第 4 节的四行科学实验矩阵填满。若用户没有给出四个 condition、W&B project 或 resume 规则，应只询问这些缺项，不得根据旧报告猜测。
4. 当前验证过的单卡入口是 `src.train.bc_ddp` 的 world-size=1 模式。不要退回长期未同步双卡修复的 `src.train.bc`。
5. 不要用 `git reset`、`git checkout --`、`git clean` 或覆盖式同步来“还原”仓库。基准测试时 rr 和 gpu-snatcher 都有用户工作区修改；先审计、保存和比较差异。
6. 不要直接连续调用 gpu-snatcher 的单卡 launcher 来启动同一 PPU 上的第二个实验。当前 launcher 会把显存超过阈值的 GPU 判为非空闲，而且它面向 NVIDIA/ZJU 服务器。ppu96 的 2×2 共享编排应使用第 9 节记录到 campaign 目录的直接 tmux 命令。
7. 不要把压力测试脚本当正式训练 launcher。压力测试明确关闭了 W&B、评估和 checkpoint，只跑一个 100-step epoch。
8. 长任务只能在 `tmux` 中运行。停止任务时只能定位到精确 tmux session、run ID 或 PID，禁止按用户批量杀进程。
9. ppu96 的根盘属于当前容器的 overlay。它速度合适但不应视为永久归档；数据可以保留在根盘加速训练，checkpoint 和最终运行台账必须同步到 CPFS 等持久位置。

## 1. 最终架构与固定决策

### 1.1 执行拓扑

| 项目 | 固定值 |
|---|---|
| 训练任务数 | 4 个独立实验 |
| 物理加速卡 | PPU 0、PPU 1，各 96 GiB |
| 分配 | PPU 0：slot 1、slot 2；PPU 1：slot 3、slot 4 |
| 每实验进程数 | 1 rank，`--nproc_per_node=1` |
| 训练入口 | `python -m torch.distributed.run ... -m src.train.bc_ddp` |
| 单实验 batch | global batch 256；world-size=1 时 per-rank 也是 256 |
| DataLoader | 每实验 4 workers，persistent workers，prefetch factor 2 |
| Device async prefetch | `false` |
| DDP 数据分片 | `data.ddp_shard_enabled=false` |
| CPU 线程 | 每实验 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` |
| 数据格式 | LMDB，frame payload 使用 zstd level 1 无损压缩 |
| 数据位置 | ppu96 根盘 `/root/rr-local-data/...`，不跨网挂载训练 |
| 训练时长基准 | 3000 epochs × 100 optimizer steps/epoch |
| checkpoint | 本地根盘生成 snapshot，异步复制到持久 CPFS；根盘作为 fallback |

拓扑图：

```text
同一份或已登记的 zstd LMDB（ppu96 根盘）
                  │
        ┌─────────┴─────────┐
        │                   │
   PPU 0 / 96 GiB      PPU 1 / 96 GiB
   ├─ slot 1, rank 0    ├─ slot 3, rank 0
   └─ slot 2, rank 0    └─ slot 4, rank 0
        │                   │
   8 loader workers     8 loader workers
        └─────────┬─────────┘
              16 workers total
```

### 1.2 为什么压缩逻辑属于 rr

zstd 是无损通用压缩算法；level 1 侧重压缩/解压速度。这里把一次性的造数写入成本和训练期间的少量 CPU 解压成本，换成显著更少的磁盘读取量。

职责边界固定如下：

- rr 的 `src/dataset/lmdb.py` 定义 frame 压缩元数据、写入和透明解压，保证 reader/writer 与旧未压缩 LMDB 的兼容性。
- rr 的 `src/data_processing/process_pickles_to_lmdb.py` 默认写 zstd level 1。
- rr 的 `requirements.txt` 固定 `zstandard==0.25.0`。
- gpu-snatcher 的 `auto_data_preparation.sh` 只负责把 `--frame-compression zstd --frame-compression-level 1` 显式传给 rr，并做依赖预检。
- 训练配置不需要额外的“解压开关”；reader 从 LMDB metadata 自动判断。

不要把 codec 实现复制进 gpu-snatcher，否则会出现数据格式与训练 reader 分叉。

### 1.3 为什么单卡也使用 `bc_ddp.py`

Git 历史审计显示：`5bccbf2` 曾把 bc 的 runtime/checkpoint 能力同步到 DDP；此后 LMDB、shard-aware dataset、global/per-rank batch、异步 checkpoint、本地 fallback、resume seed、W&B 磁盘容错和 source sampling 等修复主要继续落在 `bc_ddp.py`。继续维护 `bc.py` 单卡分支会再次产生行为漂移。

world-size=1 已在 ppu96 上完成 5-step smoke、单任务 100-step 基线、3 任务和 4 任务压力测试。四路测试全部正常退出。

## 2. 机器、磁盘和环境快照

以下只是 2026-08-27 的已验证快照。每次正式 campaign 仍必须实时检查。

| 项目 | 已验证值 |
|---|---|
| SSH | `ssh ppu96` |
| 容器 hostname | `dsw-81468-cf5f94d94-8pshx` |
| 远端用户 | `root` |
| rr 远端代码目录 | `/mnt/cpfs/users/hy/robust-rearrangement-custom` |
| PPU SDK | `/usr/local/PPU_SDK/envsetup.sh` |
| Python env | 远端 rr 目录下 `.venv` |
| Torch | 2.9.0（PPU CUDA 兼容接口） |
| CPU | 32 vCPU |
| 内存 | 300 GiB，无 swap |
| PPU | 2 × PPU-ZW805，96 GiB/卡 |
| 根文件系统 | overlay，约 886 GiB |
| 基准时根盘使用量 | 169 GiB used，672 GiB available，21% |
| `/tmp` | 独立约 30 GiB；不得放 LMDB 或大量 checkpoint |

ppu96 当前没有 `tailscale` 命令。仍按远程操作规范先确认身份：

```bash
ssh ppu96 'hostname; whoami; tailscale ip -4 2>/dev/null || true'
```

预期至少看到上述 hostname 和 `root`。若 hostname、用户、挂载或卡数不同，应停止并重新审计，不要把历史快照当事实。

## 3. 当前已验证的数据资产

### 3.1 压缩数据

```text
/root/rr-local-data/processed/diffik/sim/one_leg/rollout/med/success/
└── rgbd-only-skill-zstd.lmdb/
    ├── data.mdb
    └── lock.mdb
```

| 字段 | 值 |
|---|---|
| task | `one_leg` |
| demo source | `rollout` |
| randomness | `med` |
| outcome | `success` |
| 物理 suffix | `rgbd-only-skill-zstd` |
| episodes | 400 |
| frames/actions | 118,290 |
| frame compression | `{"codec": "zstd", "level": 1}` |
| `data.mdb` bytes | 53,174,714,368 |
| `data.mdb` 大小 | 49.52 GiB |
| SHA-256 | `bbdded6027fe62ca36ffb3ea11022a3e5e5c376d4a890f41d8a3d6b32b146f3a` |

对应的原始未压缩 LMDB 为：

```text
/mnt/cpfs/users/hy/robust-rearrangement-custom/data/processed/diffik/sim/
one_leg/rollout/med/success/rgbd-only-skill.lmdb/data.mdb
```

原始文件 127,474,061,312 bytes（约 118.72 GiB），SHA-256 为 `2fce87c6a0e3afdd5a7d0b2b7462720ff3cd9467843a9fd3481c3a30b46bfdb4`。压缩后节省约 69.20 GiB，即 58.3%。

当前压缩资产已经完成：

- 118,290 帧逐帧 zstd round-trip 一致性验证；
- 400 episode validator；
- metadata/index/frame count 校验；
- 单卡训练和四路训练读取；
- 完整 `data.mdb` SHA-256。

当前 artifact 最初由一次性 repack 工具从旧 LMDB 重打包得到；该过程工具已在任务结束后按项目约定清理。**新数据不要重新创建或依赖临时迁移脚本**；新转换器已经能从 pickle 直接生成 zstd LMDB。

### 3.2 适用边界

压力测试的四个任务共享同一份 49.52 GiB LMDB。若正式四个 condition 读取不同物理 LMDB，必须先用真实四路径做同样的 100-step gate：

- 四份各约 49.5 GiB 的数据虽能放进 672 GiB 根盘；
- 但训练进程已经占约 112.6 GiB host RSS，四份数据总计约 198 GiB，无法保证全部同时驻留 300 GiB RAM；
- 因此 page cache 和冷缓存磁盘压力可能高于本次共享数据测试。

不同数据路径未经 exact-matrix smoke 时，不得直接沿用“每任务 1.05 step/s”的 ETA。

## 4. 正式四实验矩阵：启动前必须填满

### 4.1 共同训练参数

除非用户明确批准新的科学配置，下面是本方案的共同值：

| 参数 | 固定值 |
|---|---|
| `training.num_epochs` | 3000 |
| `training.steps_per_epoch` | 100 |
| `training.batch_size` | 256 |
| `data.dataloader_workers` | 4 |
| `data.persistent_workers` | true |
| `data.prefetch_factor` | 2 |
| `data.async_device_prefetch` | false |
| `data.storage_format` | lmdb |
| `data.load_into_memory` | false |
| `data.ddp_shard_enabled` | false |
| `data.test_split` | 0.05 |
| checkpoint 类型 | `last`、`best_val_action_mse_error` |
| `training.save_last_every` | 10 |
| `training.save_per_epoch` | 500 |
| `training.async_checkpoint_saver` | true |
| `rollout.rollouts` | false |

`training.batch_size` 在 world-size=1 时既是 global batch，也是 per-rank batch。不要把双卡方案的 global batch 512 原样塞进单卡任务。

### 4.2 必须登记的四行

下面的 `<未登记>` 必须被精确值替换，并写入 `logs/<campaign>/manifest.md`。只给本文作为 context 时，Codex 应先检查该 manifest；不存在或有空字段就询问用户。

| Slot | Device | Seed | Run label | `+experiment` | task | randomness | LMDB 绝对路径 | suffix 与六个 annotation flags | W&B project/name | resume run ID |
|---|---:|---:|---|---|---|---|---|---|---|---|
| 1 | 0 | 41001 | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `null` 或精确 ID |
| 2 | 0 | 41002 | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `null` 或精确 ID |
| 3 | 1 | 41003 | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `null` 或精确 ID |
| 4 | 1 | 41004 | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `<未登记>` | `null` 或精确 ID |

本次容量验证使用的参考 condition 是：

```text
+experiment=rgbd/dit
task=one_leg
randomness=med
data.demo_source=rollout
data.demo_outcome=success
data.suffix=rgbd-only-skill
data.annotate_guidance_point=false
data.annotate_skill_one_hot=true
data.annotate_guidance_point_colored=false
data.annotate_grasp=false
data.annotate_grasp_colored=false
data.annotate_grasp_part=false
```

这只能作为已验证的性能参考。除非用户明确说“四个不同 seed 重复此 condition”，不能自动把正式矩阵填成四个相同实验。

## 5. 代码状态与单卡兼容性 gate

### 5.1 需要存在的实现

rr 至少应包含以下语义：

- `src/dataset/lmdb.py`：默认 zstd level 1、按 metadata 压缩/解压、未知 codec 拒绝、老 LMDB 无 metadata 时按未压缩读取；
- `src/data_processing/process_pickles_to_lmdb.py`：`--frame-compression` 和 `--frame-compression-level`，默认 zstd/1；
- `src/dataset/dataloader.py`：固定步数 wraparound 不得使用会永久缓存 batch 的 `itertools.cycle`；
- `src/train/bc_ddp.py`：world-size=1 可运行，打印 `TRAIN_TIMING`，保留 checkpoint/resume 修复；
- `src/config/data.yaml`：workers 4、persistent true、prefetch 2、async false；
- `requirements.txt`：`zstandard==0.25.0`。

2026-08-27 压测时远端 Git HEAD 是 `cde42d2e197f334faa13f94f513a2bdccdf4f34f`，但上述实现当时仍是 dirty-worktree patch。关键文件参考哈希如下：

| 文件 | SHA-256 |
|---|---|
| `src/dataset/lmdb.py` | `b1bc19c69344bc7c7a0ce353e597527f70eb1a0c77b40da41088f12d51cbae33` |
| `src/data_processing/process_pickles_to_lmdb.py` | `eda7f5eedcf6ba46385fbcf72feb7b15594b825c5ef959716bde68dcecc3b169` |
| `src/dataset/dataloader.py` | `9c34b95da8795ba64b14e8db584aef25f1702b71fa5e7907b9068ae997d4c8a7` |
| `src/train/bc_ddp.py` | `d111daadb164acee7ec02092cefd9db486d2ddeead47edb662a0213b5a393bea` |
| `src/config/data.yaml` | `d60290c0cc0f053aaa18da28e2f339bbae59a5977fad5b95a4d7ba245c3dd0ca` |

未来代码发生正常演进时，不要求永远保持这些哈希；但必须审查差异是否仍保留上述语义，不能为了匹配旧哈希覆盖新代码。

### 5.2 启动前代码检查

在 ppu96 上执行：

```bash
cd /mnt/cpfs/users/hy/robust-rearrangement-custom
source /usr/local/PPU_SDK/envsetup.sh >/dev/null
source .venv/bin/activate

git rev-parse HEAD
git status --short
git diff --check
python -c 'import torch, zstandard; print(torch.__version__, torch.cuda.device_count(), zstandard.__version__)'
python -m pytest -q tests/test_dataloader.py tests/test_lmdb_frame_compression.py
```

已验证结果是 7 tests passed。warning 来自 Python 3.12 下 pytest assertion rewrite 的弃用提示，不是测试失败。

再确认每个物理 PPU 单独暴露为一个逻辑 device：

```bash
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES=1 python -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
```

## 6. 新数据造数：从 pickle 直接写 zstd LMDB

### 6.1 首选 gpu-snatcher 编排入口

在保存源 pickle、且拥有 rr 环境的源机器上执行。以下命令对应当前 one-leg/med/400 数据，只是模板；正式 condition 的 task、randomness、配额和 suffix 必须来自第 4 节矩阵。

```bash
bash /home/huyue/projects/gpu-snatcher/auto_data_preparation.sh \
  --steps process_pickles \
  --tasks one_leg \
  --local-path /home/huyue/projects/robust-rearrangement-custom \
  --randomness med \
  --n-rollouts 400 \
  --process-batch-size 2 \
  --process-suffix rgbd-only-skill \
  --process-output-suffix rgbd-only-skill-zstd \
  --frame-compression zstd \
  --frame-compression-level 1
```

运行前先阅读脚本顶部的用户配置和 `git diff`。该脚本已有用户本地参数修改，禁止覆盖。CLI 参数只覆盖本次运行。

### 6.2 直接 rr converter 入口

当 gpu-snatcher 的路径配置不适合当前源机器时，可以直接调用 rr。必须使用 staging 目标，验证通过后再同文件系统原子改名；目标已存在时停止，禁止默认 `--overwrite`。

```bash
cd <SOURCE_RR_ROOT>
source <RR_PYTHON_ENV>/bin/activate

RAW_DIR=<absolute-pickle-directory-containing-only-the-approved-one-leg-data>
STAGING=<absolute-output-parent>/rgbd-only-skill-zstd.building.lmdb
FINAL=<absolute-output-parent>/rgbd-only-skill-zstd.lmdb

test -d "$RAW_DIR"
test ! -e "$STAGING"
test ! -e "$FINAL"
PICKLE_COUNT=$(find "$RAW_DIR" -type f -name '*.pkl*' | wc -l)
test "$PICKLE_COUNT" -eq 400

python -m src.data_processing.process_pickles_to_lmdb \
  -c diffik \
  -d sim \
  -f one_leg \
  -s rollout \
  -r med \
  -o success \
  --input-dir "$RAW_DIR" \
  --output-dir "$STAGING" \
  --suffix rgbd-only-skill \
  --output-suffix rgbd-only-skill-zstd \
  --batch-size 2 \
  --map-size-gb 128 \
  --frame-compression zstd \
  --frame-compression-level 1 \
  --debug-storage-stats
```

`--input-dir` 与 `--task-episode-limit` 不能同时使用，所以这个直接入口只适用于内容已经被隔离、且文件数经过上面断言的单任务目录。多任务或需要逐任务截断时，使用 6.1 节的 gpu-snatcher 入口，让 converter 根据规范数据路径应用 `task=limit`。如果源目录、多任务合并顺序、轨迹配额或 annotation mode 不同，应精确修改对应参数并写进 manifest。不要用全局 `--num-pickles` 代替多任务逐任务配额。

### 6.3 造数后验收

在 writer 完全退出、LMDB 关闭后执行：

```bash
python scripts/validate_lmdb_dataset.py --sample-episodes 400 "$STAGING"

python - "$STAGING" <<'PY'
import sys
from pathlib import Path
from src.dataset.lmdb import read_lmdb_episode_index, read_lmdb_meta

path = Path(sys.argv[1])
meta = read_lmdb_meta(path)
index = read_lmdb_episode_index(path)
compression = meta["frame_specs"].get("compression")
frames = sum(int(ep["frame_end"]) - int(ep["frame_start"]) for ep in index)
print("episodes", len(index))
print("frames", frames)
print("compression", compression)
assert compression == {"codec": "zstd", "level": 1}
PY

stat -c '%n %s bytes' "$STAGING/data.mdb"
sha256sum "$STAGING/data.mdb"
du -sh "$STAGING"
```

对当前参考数据应额外断言 400 episodes 和 118,290 frames。对新数据使用其批准配额，不得照抄。

验收通过后才允许：

```bash
mv "$STAGING" "$FINAL"
```

必须把 metadata 摘要、文件字节数、SHA-256、源 pickle manifest hash、rr commit/patch 状态和完成时间写入 `logs/<campaign>/assets.md`。

## 7. 传到 ppu96 根盘

### 7.1 原则

- 由保存源 LMDB 的 Ubuntu 机器直接传到 ppu96，不经过控制机中转。
- 训练数据必须落到 ppu96 本地根盘；不能直接从 CPFS/NAS/互联网挂载训练。
- 先确认目标父目录属于 `/`，确认剩余空间并预留 checkpoint、W&B 和至少 100 GiB 安全余量。
- 传输时 LMDB 不得有 writer；训练也不应读取不完整目标。
- 使用独立 staging 目录和完成 marker，禁止让 loader 看见半份数据。

### 7.2 传输步骤

```bash
ssh ppu96 'hostname; whoami; findmnt -T /root; df -h /; mkdir -p /root/rr-local-data/processed/diffik/sim/one_leg/rollout/med/success'

rsync -aP --partial --append-verify \
  <SOURCE_DATASET>.lmdb/ \
  ppu96:/root/rr-local-data/processed/diffik/sim/one_leg/rollout/med/success/<DATASET>.lmdb.incoming/
```

传完后在 ppu96 验证字节数、SHA 和 metadata。源、目标的 `data.mdb` SHA 必须一致：

```bash
ssh ppu96 '
  stat -c "%n %s bytes" /root/rr-local-data/.../<DATASET>.lmdb.incoming/data.mdb
  sha256sum /root/rr-local-data/.../<DATASET>.lmdb.incoming/data.mdb
  df -h /
'
```

然后在 ppu96 的 rr 环境运行 validator 和一个 loader smoke。全部通过后，将 `.incoming` 原子改名为最终目录并写 `.transfer-complete`。现有最终目录非空时必须停止并请求用户处理，不能覆盖或删除。

## 8. Campaign 目录与持久化布局

定义唯一 campaign ID，例如 `ppu96-oneleg-med-20260827-a`。不要复用历史目录。

```text
本地高速根盘：
/root/rr-runs/<campaign>/
├── slot-1/
│   ├── hydra/
│   ├── wandb/
│   ├── checkpoint-tmp/
│   ├── checkpoint-fallback/
│   └── train.log
├── slot-2/
├── slot-3/
└── slot-4/

持久 CPFS：
/mnt/cpfs/users/hy/robust-rearrangement-custom/outputs/<campaign>/
├── slot-1/models/
├── slot-2/models/
├── slot-3/models/
└── slot-4/models/

仓库忽略区台账：
/mnt/cpfs/users/hy/robust-rearrangement-custom/logs/<campaign>/
├── manifest.md
├── commands/
├── monitors/
├── assets.md
└── completion.md
```

`checkpoint_saver_tmp_dir` 指向根盘 slot 目录，`training.model_save_dir` 指向 CPFS slot 目录，`checkpoint_fallback_dir` 指向根盘。这样 torch serialization 不直接阻塞在 CPFS 上，异步 saver 负责复制，CPFS 失败时 fallback 保留本地 checkpoint 并由恢复线程重试。

启动前：

```bash
test ! -e "/root/rr-runs/$CAMPAIGN"
test ! -e "/mnt/cpfs/users/hy/robust-rearrangement-custom/outputs/$CAMPAIGN"
mkdir -p "/root/rr-runs/$CAMPAIGN"
mkdir -p "/mnt/cpfs/users/hy/robust-rearrangement-custom/outputs/$CAMPAIGN"
mkdir -p "/mnt/cpfs/users/hy/robust-rearrangement-custom/logs/$CAMPAIGN/commands"
```

若目录已存在，先判断是 resume 还是 ID 冲突，不得直接复用。

## 9. 正式启动流程

### 9.1 启动前资源 gate

每次远程操作先确认身份，然后在加载 PPU SDK 后检查：

```bash
cd /mnt/cpfs/users/hy/robust-rearrangement-custom
source /usr/local/PPU_SDK/envsetup.sh >/dev/null
source .venv/bin/activate

df -h / /tmp /mnt/cpfs
free -h
findmnt -T /root/rr-local-data
tmux list-sessions 2>/dev/null || true
pgrep -af 'src[.]train[.]bc_ddp|torch[.]distributed[.]run' || true
ppu-smi --query-ppu=timestamp,index,utilization.ppu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits
```

启动条件：

- 两张 PPU 上没有未登记进程；
- 根盘至少保留 100 GiB，CPFS checkpoint 目标可写；
- host `MemAvailable` 足够，建议启动前至少 200 GiB；
- 四个数据路径及 SHA/metadata gate 均通过；
- 四行矩阵、四条完整命令、四个 tmux 名和 W&B 名均已写入 manifest；
- 没有同名 tmux、Hydra、model 或 W&B run 冲突。

### 9.2 先做正式配置 smoke

在启动四路前，用 slot 1 的真实模型和真实数据跑 5 steps，但关闭 W&B、评估、rollout 和 checkpoint。smoke 只能写到新的 `logs/<campaign>/smoke/` 和根盘临时目录。

smoke 必须证明：

- world size 为 1；
- 可见 device 数为 1；
- global/per-rank batch 都是 256；
- exact LMDB 路径、episode/frame count 正确；
- forward、backward、optimizer step 和 DataLoader wraparound 正常；
- 进程退出码 0；
- 不产生正式 W&B run 或正式 checkpoint。

### 9.3 每个 slot 的标准命令

把下面模板渲染成四个完整、无占位符的命令文件，例如 `logs/<campaign>/commands/slot-1.sh`。临时编排脚本必须放 `logs/<campaign>/tools/`，不要放进 tracked `scripts/`。

```bash
#!/usr/bin/env bash
# PPU SDK 的 envsetup.sh 会读取可选位置参数，不兼容提前启用 nounset。
set -eo pipefail

RR_ROOT=/mnt/cpfs/users/hy/robust-rearrangement-custom
CAMPAIGN=<campaign>
SLOT=<1-4>
DEVICE=<0-or-1>
SEED=<seed>
RUN_LABEL=<unique-run-label>
DATASET=<absolute-zstd-lmdb-path>
LOCAL_RUN_ROOT=/root/rr-runs/$CAMPAIGN/slot-$SLOT
DURABLE_MODEL_ROOT=$RR_ROOT/outputs/$CAMPAIGN/slot-$SLOT/models

mkdir -p \
  "$LOCAL_RUN_ROOT/hydra" \
  "$LOCAL_RUN_ROOT/wandb" \
  "$LOCAL_RUN_ROOT/checkpoint-tmp" \
  "$LOCAL_RUN_ROOT/checkpoint-fallback" \
  "$DURABLE_MODEL_ROOT"

cd "$RR_ROOT"
source /usr/local/PPU_SDK/envsetup.sh >/dev/null
source .venv/bin/activate
set -u

export CUDA_VISIBLE_DEVICES="$DEVICE"
export DATA_DIR_PROCESSED="$RR_ROOT/data"
export RUN_OUTPUT_DIR="$LOCAL_RUN_ROOT/hydra"
export WANDB_DIR="$LOCAL_RUN_ROOT/wandb"
export TMPDIR="/tmp/rr-$CAMPAIGN-slot-$SLOT"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

exec python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m src.train.bc_ddp \
  +experiment=<matrix-experiment> \
  task=<matrix-task> \
  data.demo_source=rollout \
  data.demo_outcome=success \
  data.suffix=<matrix-suffix> \
  data.annotate_guidance_point=<true-or-false> \
  data.annotate_skill_one_hot=<true-or-false> \
  data.annotate_guidance_point_colored=<true-or-false> \
  data.annotate_grasp=<true-or-false> \
  data.annotate_grasp_colored=<true-or-false> \
  data.annotate_grasp_part=<true-or-false> \
  data.storage_format=lmdb \
  data.load_into_memory=false \
  data.dataloader_workers=4 \
  data.persistent_workers=true \
  data.prefetch_factor=2 \
  data.async_device_prefetch=false \
  data.data_subset=null \
  data.test_split=0.05 \
  "data.data_paths_override=[$DATASET]" \
  data.ddp_shard_enabled=false \
  training.batch_size=256 \
  training.num_epochs=3000 \
  training.steps_per_epoch=100 \
  training.eval_every=100 \
  training.sample_every=100 \
  'training.save_checkpoints=[last,best_val_action_mse_error]' \
  training.save_last_every=10 \
  training.save_per_epoch=500 \
  training.async_checkpoint_saver=true \
  "training.checkpoint_saver_tmp_dir=$LOCAL_RUN_ROOT/checkpoint-tmp" \
  "training.checkpoint_fallback_dir=$LOCAL_RUN_ROOT/checkpoint-fallback" \
  "training.model_save_dir=$DURABLE_MODEL_ROOT" \
  rollout.rollouts=false \
  wandb.project=<matrix-wandb-project> \
  wandb.mode=online \
  "wandb.name=$RUN_LABEL" \
  wandb.continue_run_id=<null-or-exact-run-id> \
  randomness=<matrix-randomness> \
  +seed="$SEED" \
  dryrun=false \
  "hydra.run.dir=$LOCAL_RUN_ROOT/hydra" \
  >"$LOCAL_RUN_ROOT/train.log" 2>&1
```

模型专属 override（例如 vision encoder）必须作为第 4 节矩阵的一部分追加到对应 slot，不要偷偷写成四个任务的共同默认。

### 9.4 用四个独立 tmux 启动

先给四个渲染后的命令执行 `bash -n`，把文件 SHA-256 写入 manifest。然后依次创建：

```bash
tmux new-session -d -s "${CAMPAIGN}-s1" "bash <absolute-slot-1-command>"
tmux new-session -d -s "${CAMPAIGN}-s3" "bash <absolute-slot-3-command>"
tmux new-session -d -s "${CAMPAIGN}-s2" "bash <absolute-slot-2-command>"
tmux new-session -d -s "${CAMPAIGN}-s4" "bash <absolute-slot-4-command>"
```

顺序先给两张卡各启动一个，再立即补第二个；四个任务应在约 10 秒内全部发起，以复现 2×2 并发条件。不要因为第一批已经占显存而改用别的卡。

### 9.5 启动后验收

启动后 30–120 秒内必须检查：

1. 四个 tmux 均存在，pane 当前仍为 Python/torchrun，而不是命令已经失败后回到 shell。
2. 四个 `bc_ddp` rank 均存在；每个 rank 的 `CUDA_VISIBLE_DEVICES` 与 slot 表一致。
3. 每个日志明确显示 world size 1、global/per-rank batch 256、exact data path、正确 episode/frame count。
4. 四个 W&B run name/ID 与 manifest 一一对应，没有意外 resume。
5. 两张 PPU 显存都应约 34–36 GiB，稳定训练窗口平均利用率目标约 85–95%。
6. 根盘、CPFS、内存和 `/tmp` 没有快速耗尽。
7. 从每个日志抄录首个 `TRAIN_TIMING`；若单任务低于约 0.85 step/s，先诊断再继续估时。

只看到 tmux session 不算启动成功。必须同时确认训练 PID、日志进度、PPU 占用和 W&B ID。

## 10. 监控、故障和恢复

### 10.1 最低监控集

campaign 开始后，把监控输出写到 `logs/<campaign>/monitors/`：

```bash
ppu-smi \
  --query-ppu=timestamp,index,utilization.ppu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 \
  -f logs/<campaign>/monitors/ppu.csv

vmstat 1 > logs/<campaign>/monitors/vmstat.log
```

另每 5 分钟记录：

```bash
date -u
df -h / /tmp /mnt/cpfs
free -h
ps -eo pid,ppid,etimes,rss,args | rg 'src[.]train[.]bc_ddp|torch[.]distributed[.]run'
```

每个 epoch 的 `TRAIN_TIMING` 至少保留：data wait、compute、step/s、samples/s、raw GiB/s。W&B 之外仍要保留文本日志，因为网络或 W&B 可暂时失败。

### 10.2 诊断阈值

以下是调查阈值，不是自动杀任务条件：

| 信号 | 调查条件 | 优先检查 |
|---|---|---|
| PPU 利用率 | 任一卡连续 5 分钟低于 80% | data wait、其他进程、worker 退出、数据路径 |
| 单任务速度 | 连续 epoch 低于 0.85 step/s | PPU 共享、exact LMDB、checkpoint/评估阶段 |
| data wait | 持续超过 20% | 根盘 I/O、不同 LMDB page cache、worker 数 |
| CPU | run queue 长期 >32 且 idle 很低 | 进程/worker 数、线程变量是否为 1 |
| I/O wait | 长期 >25% | 根盘读取、CPFS checkpoint copy、page cache 抖动 |
| Host 内存 | `MemAvailable < 60 GiB` | RSS、page cache、异常 batch 缓存 |
| 根盘 | available <100 GiB | W&B、Hydra、fallback checkpoint、无关临时文件 |
| PPU 显存 | 明显高于约 36 GiB/卡或持续增长 | 模型配置差异、缓存/泄漏 |

不要为了“跑满 CPU”盲目增加 worker。本次四路测试 CPU 仍有约 60% idle，但两张 PPU 已约 92% 利用率；目标是训练吞吐，不是 CPU 100%。此前纯读取测试也表明读者过多会降低 zstd LMDB 吞吐。

### 10.3 async 的固定决策

`data.async_device_prefetch=false` 是本方案的固定值。本次没有做 true/false A/B；关闭时，4×4 DataLoader workers 已把训练 data wait 压到约 4–5%，并把两张 PPU 推到约 92%。因此没有证据支持在正式 campaign 中临时开启。

若未来模型、数据或 PPU runtime 改变，必须作为独立 100-step A/B 重新测试，不能在四个科学实验中混用，否则实验条件不一致。

### 10.4 精确停止与恢复

正常情况下让任务自行结束。需要停止单个任务时：

```bash
tmux send-keys -t <exact-campaign-slot-session> C-c
```

等待对应 rank 和 worker 退出，再处理 session。禁止 `pkill -u`、宽泛 `pkill python` 或终止其他 slot。

恢复单个 slot 时：

1. 保留原 W&B run ID、原数据路径和原科学 overrides；
2. 将 `wandb.continue_run_id` 设置为精确 run ID；
3. 保持原 `training.model_save_dir` 和 `checkpoint_fallback_dir`；
4. 启动后必须看到明确的 local/W&B checkpoint 恢复消息；
5. 检查 checkpoint epoch 与 `start_epoch=checkpoint_epoch+1`；
6. `bc_ddp.py` 会恢复 checkpoint 中的 seed，必须确认日志；
7. 只在 resume 验收完成后把 registry 状态改成 `resumed`。

如果 CPFS 暂时失败，异步 saver 会把 checkpoint 放入根盘 fallback。不要释放 ppu96 实例，直到 fallback 已成功同步到持久目标并有 SHA/epoch 记录。

## 11. 完成与归档定义

四实验 campaign 只有同时满足下列条件才算完成：

1. 四个 run 均达到批准的 3000 epochs，或有用户批准的 early-stop 结论；
2. 每个 run 的最终 checkpoint 和必要 archive checkpoint 位于持久 CPFS；
3. checkpoint 能读取，payload 中 epoch、config、optimizer/scheduler state 与 run 匹配；
4. W&B run 状态、ID、name 和最终 summary 已记录；
5. 四个训练进程、workers 和 tmux 已自然退出；
6. 根盘 fallback 中没有尚未同步的唯一 checkpoint；
7. `logs/<campaign>/completion.md` 记录：开始/结束时间、源码 commit/patch、数据 SHA、最终 Hydra config、每任务实际速度、失败/恢复事件、checkpoint 路径和 SHA；
8. 只有在持久归档验收后，才可以请求用户批准清理根盘 run artifacts。数据集是否保留由用户单独决定。

## 12. 压力测试方法与结果

### 12.1 方法

测试日期：2026-08-27 UTC。压力测试和分析使用的是 `logs/lmdb400-ppu96/tools/` 下的任务专用临时工具；任务完成后这些过程脚本已清理，方法、参数和汇总结果完整保存在本文，原始监控与训练日志继续保留在下列目录。

共同条件：

- 同一份当前 zstd LMDB；
- 在每个 scenario 前请求逐出该 LMDB file cache，测试冷缓存；
- 每任务 batch 256、4 workers、persistent true、prefetch 2、async false；
- 每任务 world-size=1、100 optimizer steps、1 epoch；
- 禁用 W&B、eval、rollout、sampling 和全部 checkpoint；
- 每任务独立 TMPDIR，BLAS/OpenMP 线程设为 1；
- 同时采集 PPU CSV、`vmstat` 和聚合 Python RSS；
- 计时只使用训练 loop 的 `TRAIN_TIMING`，PPU 利用率按实际 loop 重叠窗口统计，不混入编译启动和结束后的空闲时间。

原始日志：

```text
logs/lmdb400-ppu96/collected/single-baseline-cold-100step/
logs/lmdb400-ppu96/collected/single-3jobs-cold-100step/
logs/lmdb400-ppu96/collected/single-4jobs-cold-100step/
```

三个 scenario 的所有 job exit code 均为 0。

### 12.2 单任务基线

| 指标 | 结果 |
|---|---:|
| loop duration / 100 steps | 68.049 s |
| optimizer steps/s | 1.470374 |
| samples/s | 376.416 |
| data wait/step | 0.192748 s |
| compute/step | 0.487351 s |
| data wait fraction | 28.341% |
| PPU 0 平均利用率 | 65.824% |
| PPU 0 峰值显存 | 18,541 MiB |
| CPU user/system/idle/iowait | 4.49% / 1.61% / 87.19% / 6.68% |
| Python RSS 峰值 | 27.584 GiB |

### 12.3 三任务：PPU 0 两个，PPU 1 一个

| Job | Device | duration | steps/s | data wait | compute | wait fraction |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 95.138 s | 1.051537 | 0.039301 s | 0.911687 s | 4.133% |
| 2 | 0 | 95.102 s | 1.051954 | 0.039248 s | 0.911363 s | 4.129% |
| 3 | 1 | 65.637 s | 1.524448 | 0.167527 s | 0.488448 s | 25.539% |

资源结果：

| 指标 | 结果 |
|---|---:|
| aggregate steps/s | 3.627939 |
| PPU 0 平均利用率（双任务） | 92.170% |
| PPU 1 平均利用率（单任务） | 67.815% |
| PPU 0 / PPU 1 峰值显存 | 35,172 / 18,538 MiB |
| CPU user/system/idle/iowait | 13.07% / 4.02% / 69.98% / 12.97% |
| `vmstat bi` 平均 | 257,330 blocks/s |
| run queue p95 | 9 |
| Python RSS 峰值 | 76.764 GiB |

两个共享 PPU 的任务几乎完全一致。它们的 data wait 只有约 4.1%，但 compute/step 从单任务约 0.49 s 增至约 0.91 s，说明主要减速来自同卡两个训练进程的 PPU 计算/调度竞争，而不是 CPU 解压。

### 12.4 四任务：每张 PPU 两个

| Job | Device | duration | steps/s | data wait | compute | wait fraction |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 94.738 s | 1.056027 | 0.043261 s | 0.903685 s | 4.569% |
| 2 | 0 | 94.659 s | 1.056884 | 0.039001 s | 0.907176 s | 4.122% |
| 3 | 1 | 95.251 s | 1.050282 | 0.044551 s | 0.907574 s | 4.679% |
| 4 | 1 | 95.242 s | 1.050407 | 0.043123 s | 0.908889 s | 4.530% |

资源结果：

| 指标 | 结果 |
|---|---:|
| aggregate steps/s | 4.213600 |
| PPU 0 / PPU 1 平均利用率 | 92.000% / 91.574% |
| 两卡 p95 PPU 利用率 | 100% / 100% |
| PPU 0 / PPU 1 峰值显存 | 35,180 / 35,170 MiB |
| memory-util 平均 | 32.606% / 32.511% |
| CPU user/system/idle/iowait | 18.83% / 5.17% / 59.95% / 15.95% |
| `vmstat bi` 平均 | 300,995 blocks/s |
| run queue p95 | 11 |
| Python RSS 峰值 | 112.589 GiB |

### 12.5 对比结论

| 对比 | 结论 |
|---|---|
| 四路单任务速度 vs 独占单卡 | 每任务约慢 28.4%，epoch wall time 约增加 40% |
| 四路 aggregate vs 一任务 | 2.866× |
| 四路 aggregate vs 两张卡各一个任务 | 约 1.433×，即整机吞吐提高约 43.3% |
| 四路 aggregate vs 三路 | 提高约 16.1% |
| 三路与四路最慢任务完成时间 | 约 95.1 s vs 95.3 s，几乎相同 |

三路时，PPU 1 的独占任务提前约 29.5 秒结束，之后该卡空闲；四路在几乎不延长 campaign makespan 的情况下多完成一个任务。因此当前机器上“每卡两个、总共四个”明显优于三路。

显存和 host RAM 都不是四路瓶颈。CPU 也没有跑满；四路时仍约 60% idle。冷缓存下约 16% I/O wait 说明根盘读取有压力，但 16 个 DataLoader workers 已把训练线程看到的 data wait 隐藏到约 4–5%，而两张 PPU 达到约 92%。当前主要瓶颈是每张卡两个独立模型的 PPU 计算竞争。

## 13. 3000 epoch 时间和容量预算

本次配置是 100 optimizer steps/epoch，因此：

```text
独占单卡：68.049 s/epoch × 3000 = 56.71 h = 2.36 days
四路并行：95.251 s/epoch × 3000 = 79.38 h = 3.31 days
```

四个实验同时开始时，纯训练 loop 预计约 3.31 天全部结束。若只允许每张卡一个任务，四个实验要分两批，约 4.72 天；因此 2×2 共享可把四实验总排期缩短约 1.41 天。

生产 ETA 必须注明：

- 压测关闭了 eval、checkpoint、sampling 和在线 W&B；
- 正式运行需要包含这些开销；
- 根盘 page cache 在长训练中会变热，数据等待可能比冷缓存测试更低；
- 不同 LMDB、外部 PPU 争用、CPFS checkpoint 或不同模型大小会改变速度。

实际排期建议为 3.6–4.0 天，并在首个正常生产 epoch 后用真实 `TRAIN_TIMING` 更新：

```text
ETA_hours = measured_seconds_per_epoch × remaining_epochs / 3600
```

若 `steps_per_epoch` 不是 100，则基于本测试的粗略缩放为：

```text
time ≈ reference_time × actual_steps_per_epoch / 100
```

但模型结构、评估周期或数据路径同时改变时，必须以真实 smoke/首 epoch 为准，不能只做线性外推。

## 14. 不应执行的方案

- 不要使用老 `python -m src.train.bc training.gpu_id=...` 单卡入口。
- 不要把两张 PPU 合成一个双卡 DDP 实验；本方案的目标是四个独立科学实验。
- 不要在同一卡第二个任务启动时依赖 gpu-snatcher 的 free-GPU 门禁。
- 不要把四个任务都放在 PPU 0，或按 3+1 分配。
- 不要把数据放在 `/tmp`、CPFS、NAS 或跨互联网 mount 后直接训练。
- 不要为四个任务复制四份相同 LMDB；共享只读物理数据。
- 不要为了 CPU 利用率好看把 workers 从 4 盲目提高到 8/16。
- 不要在正式四个实验中混用 async prefetch true/false。
- 不要把当前一次性 repack、传输或 benchmark 工具移入 tracked `scripts/`。
- 不要覆盖已存在的 LMDB、run 目录、W&B run 或 checkpoint；先明确是新 run 还是 resume。
- 不要在只完成“启动”后宣称任务结束；按第 11 节持续监控到 checkpoint 和持久归档验收完成。
