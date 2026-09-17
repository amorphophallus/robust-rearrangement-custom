# AutoMate 90/10 held-out assembly 实验：冻结划分与训练计划

更新时间：2026-09-17（Asia/Shanghai）
状态：**split、索引格式和训练参数已冻结；新 joint training 尚未启动。**

## 1. 实验目标

本实验检验一个视觉 joint policy 能否在保持 FurnitureBench 能力的同时，从 90 个 AutoMate assembly 的 demonstration 中学习，并对 10 个训练中完全不可见的 assembly 产生 zero-shot task generalization。训练继续使用已有 FurnitureBench 和 AutoMate 数据，不重新采集、不重新标注，也不修改 LMDB 中的样本。

主张边界固定为：**generalization to 10 pre-registered unseen assembly identities**。其中 9 个任务来自 AutoMate 论文公开的几何代表集合；`00755` 是预先指定、没有 specialist demonstration 的任务。实验结果可以支持跨 unseen assembly 的泛化证据；是否属于更强的 geometry-shift OOD，需结合 held-out task 到 Train90 的 PointNet latent 距离单独判断。

## 2. Test10 的固定选择

AutoMate 论文从每个 plug/socket mesh 表面采样点云，用 PointNet autoencoder 学习 32 维几何表示，并用 `perplexity=6` 的 t-SNE 展示 assembly 几何分布。论文 Figure 5 给出的 10 个跨 cluster 代表任务为：

`00015, 00296, 00320, 00340, 00681, 00731, 00768, 01036, 01041, 01129`。

本实验必须把没有 specialist demonstration 的 `00755` 放入测试集，因此采用以下预登记规则：

1. 保留论文 10 个代表任务中的 9 个，不根据现有 generalist 成功率重新挑选。
2. 用独立的 raw-mesh plug/socket 多分辨率 Chamfer audit，只决定论文代表集合中由谁让位；`01129` 是该集合里与 `00755` 最近的任务，距离为 `0.627038240`，因此用 `00755` 替换 `01129`。
3. `01129` 回到 Train90。该替换使 mandatory holdout 对论文原几何覆盖的扰动尽可能小。

最终划分如下。任务 ID 和顺序在训练前冻结，不允许根据新 policy 的结果修改。

| Held-out task | 来源 | Specialist final train SR | 本轮角色 |
|---|---|---:|---|
| `00015` | AutoMate Figure 5 | 91.9% | Test only |
| `00296` | AutoMate Figure 5 | 83.2% | Test only |
| `00320` | AutoMate Figure 5 | 99.8% | Test only |
| `00340` | AutoMate Figure 5 | 80.2% | Test only |
| `00681` | AutoMate Figure 5 | 98.3% | Test only |
| `00731` | AutoMate Figure 5 | 100.0% | Test only |
| `00755` | Mandatory holdout | N/A | Test only；无 specialist demonstration |
| `00768` | AutoMate Figure 5 | 97.3% | Test only |
| `01036` | AutoMate Figure 5 | 98.9% | Test only |
| `01041` | AutoMate Figure 5 | 100.0% | Test only |

这组任务具有代表性的依据是：9/10 直接沿用 AutoMate 论文用 PointNet+t-SNE 展示几何覆盖时公开的代表集合；唯一替换由 mandatory task 决定，并让与它几何最近的原代表退出 Test10。因此它保留了论文选择覆盖的主要区域，同时避免用 policy SR 进行事后选择。PointNet latent 用于定义和可视化几何邻域；t-SNE 只用于展示，因为二维投影会扭曲全局距离，不应用来计算“离训练集多远”。

## 3. 数据实现：保留 LMDB，只过滤 episode index

原始 LMDB 全部保持只读。AutoMate `data.mdb` 仍包含原来的 99 个任务和 4,950 个 episodes；训练入口读取外部 `rr-episode-selection-v1` JSON，在构建 episode manifest 后只保留 Train90 的 4,500 个 episodes。索引执行 fail-closed 校验：输入必须恰好是 99 tasks/4,950 episodes，输出必须恰好是 90 tasks/4,500 episodes，任一 Test10 任务不得进入训练 manifest。

`00755` 原本就不在 AutoMate99 LMDB；外部索引另外排除：

`00015, 00296, 00320, 00340, 00681, 00731, 00768, 01036, 01041`。

`original` 数据对应 `rgbd-skill` 物理数据。其 selection index 上传到 ModelScope 数据集的固定目录：

```text
indices/automate90_paper_tsne_plus_00755_v1/
├── automate_train90_index.json
├── heldout_10.txt
└── train_90.txt
```

ModelScope commit 固定为 `723b63da4d56616b82196f3ff9a6c65e006a10af`。公开重新 clone 后，三份文件的 SHA-256 分别为：index `d697be1daa63b46f0f7ba072940d737b2e47ae88d1e404dbc954638b7da18c0b`、Test10 `4ccf9ed4e97df0cb1f3945494e4bd6437ff6205bf38a7bf5e643cb672c0c8d90`、Train90 `e804fa0dd91e81a5bf261836de08006873cc65503e931d132edc8e340f79c61a`。

其他 logical conditions 使用五份物理 LMDB 和各自带 source hash 的本地索引。映射固定为：

| Logical condition | Physical AutoMate data | `data.mdb` SHA-256 |
|---|---|---|
| `rgb`, `rgbd`, `rgbd_skill` | `rgbd-skill` | `acf0083d647c9279d662379abbb80c9a9ef398694058b9f129b4fc0642e19b2b` |
| `rgbd_gp`, `rgbd_gp_skill` | `rgbd-gp-skill` | `30adfdd47bd7282a57c3ba66de902065b2afb6680fde7bc81c5bef211de5461b` |
| `rgbd_colored_gp` | `rgbd-colored-gp` | `e768c25db6f98e1c8aef4e6f45762c200201a6dfeb1027bc09a4460a385f9a95` |
| `rgbd_grasp_part` | `rgbd-grasp-part` | `a2993f43ff7e64c32bddb44d09484d0e95e3af75e45a4e3f6bbd906df6c87333` |
| `rgbd_grasp_part_colored` | `rgbd-grasp-part-colored` | `7b13f7f9debe67f02413e07c994a709ad776f68e4aa691f36f2025d663b218a0` |

冻结索引在 r218 的路径为：

```text
/data/hy/robust-rearrangement/logs/automate-90-10-pointnet-selection-0917/
```

NAS 备份路径为：

```text
/mnt/nas/datasets_tmp/rr_joint_training_0821_scripted_prod_20260901_v1/manifests/automate90_paper_tsne_plus_00755_v1/
```

历史训练结束后，NAS 上数百 GB 的 condition-specific 完整 LMDB 已按原清理策略释放。本实验保留的是可校验的无损备份链，而不是声称这些大 LMDB 仍在 NAS：ModelScope 上的 `original` LMDB 是规范基底；NAS 的 `reconstruction_backup/` 保存全帧审计过的 sparse sidecar、manifest 和 `reconstruct_lmdb_variants.py`，能精确恢复其余四种物理 condition。AutoMate sidecar SHA-256 为 `728d56c6abfda43b6085c4f6e39503b08d44609903c2722ee571bc9abfcbae15`；恢复后必须达到上表目标 `data.mdb` hash，才允许 staging 到训练机。五份 condition index 已在 Windows workspace、r218 和 NAS 三处逐文件核对 SHA-256 一致。

## 4. 固定训练矩阵

| 项目 | 冻结设置 |
|---|---|
| Conditions | `rgb`, `rgbd`, `rgbd_colored_gp`, `rgbd_gp`, `rgbd_gp_skill`, `rgbd_skill`, `rgbd_grasp_part`, `rgbd_grasp_part_colored` |
| Train seeds | `2026091801`, `2026091802`, `2026091803`；每个 condition 3 seeds，共 24 runs |
| Training data | 原 FurnitureBench LMDB + 原 AutoMate99 LMDB + 对应 Train90 外部索引 |
| Source sampling | 每个 epoch 全局样本严格 `FurnitureBench:AutoMate = 50%:50%` |
| Model/config | 沿用 formal joint experiment 的 `+experiment=<condition>/dit` |
| Distributed training | 2 GPUs，DDP，global batch size `512` |
| Schedule | `steps_per_epoch=100`, `num_epochs=3000`, `save_per_epoch=500` |
| Per-epoch quota | 51,200 samples：25,600 FurnitureBench + 25,600 AutoMate |
| Primary checkpoint | epoch 3000 final checkpoint |
| Selection leakage | 不用 FB、ID90 或 Test10 SR 选择 checkpoint、超参、seed 或任务划分 |

每个 run 必须包含以下 overrides；`episode_selection_index` 按 condition 指向登记表中的物理索引：

```text
data.episode_selection_index=<absolute-path>/automate_train90_index.json
data.env_sampling_weights={FurnitureBench:0.5,AutoMate:0.5}
data.ddp_shard_enabled=true
data.minority_class_power=false
training.batch_size=512
training.steps_per_epoch=100
training.num_epochs=3000
training.save_per_epoch=500
training.async_checkpoint_saver=true
early_stopper.patience=inf
```

run name 固定为 `automate90_fb50_<condition>_seed<seed>_formal`。训练前登记 hostname、GPU、Git commit、Conda environment、LMDB 路径及 SHA-256、selection index 路径及 SHA-256、W&B run ID。数据需 staging 到本地 SSD/NVMe，不跨 NAS 直接训练。

## 5. 部署训练 agent 的固定执行计划

1. 读取本报告、`condition_index_registry.tsv` 和对应 index；核对 Test10 与 Train90 互斥且并集为 100 个候选任务。
2. 对目标服务器做 GPU、磁盘与数据盘审计。验证 condition 所需 `data.mdb` hash；不复制或派生新的 AutoMate90 LMDB。
3. 在启动命令中显式传入 selection index 和 50:50 source weights。预运行 manifest audit，必须得到 AutoMate `99→90 tasks`、`4950→4500 episodes`。
4. 首先启动 `rgbd_colored_gp` seed `2026091801` 作为 plumbing pilot。首个 epoch 必须记录全局 `25,600 FB / 25,600 AutoMate`，并检查 DDP shard、loss、吞吐、checkpoint 和 W&B。结构检查通过后该 run 继续作为正式 seed。
5. 按 GPU 空闲情况启动其余 23 个 runs。不得根据中途成功率修改 split、sampling ratio 或超参；异常只能用同一 run ID 和最近 checkpoint 恢复。
6. 仅评估 epoch-3000 final checkpoint：FB formal 每 task 36 rollouts；AutoMate ID90 每 task 12 rollouts；Test10 每 task、每 train seed 100 rollouts，分配到 5 个固定 evaluation seeds，每 seed 20 rollouts。
7. Test10 报告 task-macro SR、rollout-micro SR、task-level bootstrap 95% CI 和逐任务 SR。九个有 specialist 的任务同时报告 generalist SR、specialist SR、差值与比值；`00755` 单独报告 generalist SR。

### 5.1 AutoMate success 口径

ID90 和 Test10 的主 SR 固定使用 AutoMate/IsaacLab 官方 success predicate，以与 AutoMate 原文和已有 100-task 面板保持同口径；训练和评测 agent 不得在本轮 90/10 实验中自行改阈值或用事后人工视频判定覆盖官方 SR。

已知限制是，官方判据只组合高度窗口与全局 `15 mm` 平均 keypoint-distance 阈值，不检查孔腔包含、接触、实际插入深度或持续稳定。`00320` 的视频审计显示，该判据可在 peg 扫过孔口而未形成可见稳定插入时触发。本轮仍按官方 SR 登记结果，并将该问题作为独立的后续度量学任务；如未完成新的预登记严格判据，不在主结果中声称“稳定物理插入率”。

## 6. 启动门槛

正式启动前必须同时满足：

- 训练代码能读取外部 index，并通过 manifest/filter/source-sampling 回归测试；
- ModelScope `original` index 的公开下载内容与本地 SHA-256 一致；
- 五份物理 condition index 在 r218 与 NAS 各有一份 hash 一致的备份；
- 目标训练机具备对应 condition 数据，且 `data.mdb` hash 与本报告一致；
- 首 epoch 的 source quota 精确为 50%/50%。

任何一项失败都停止该 run，修复数据或配置后再用同一冻结 split 启动。
