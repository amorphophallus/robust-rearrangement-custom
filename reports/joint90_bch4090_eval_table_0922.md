# Joint90 bch_4090 live evaluation table

## Table 4-style Joint90 result (authoritative, camera-fixed)

Registered: `2026-09-26`. FurnitureBench uses the completed 36-rollout formal
evaluation. AutoMate uses the camera-synchronization-fixed run at
`/home/bch/rr-joint90-0921/runs/joint90-automate-validation12-default1-hardest-fabric-camera-fix-0925`.
The retained AutoMate matrix contains exactly seven conditions and `700/700`
validated cells; every cell has 12 rollouts, enabled Fabric, a valid dynamic RGB
stream, and no rollout error. `rgbd_skill` was removed before it started at the
user's request. The presentation below also omits the completed
`rgbd_gp_skill` row, matching the requested Table 4 layout.

Each FurnitureBench subscript is `Joint90 - original FB` in percentage points.
The original FB reference is the frozen three-training-seed mean in
[`main_3seed_experiment_review_0913.md`](main_3seed_experiment_review_0913.md).
Joint90 is one formal training seed whereas the reference is a three-seed mean,
so these deltas are diagnostic rather than paired-seed causal estimates.
AutoMate columns have no original-FB counterpart and therefore have no
subscript.

| Condition | FB One-leg | FB Round table | FB Lamp | FB Overall | AutoMate ID90 | AutoMate OOD10 |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 88.9%<sub>+4.63</sub> | 61.1%<sub>+19.44</sub> | 13.9%<sub>-4.63</sub> | 54.6%<sub>+6.48</sub> | 54.8% (592/1080) | 41.7% (50/120) |
| RGB-D | **97.2%**<sub>+14.81</sub> | 55.6%<sub>+6.49</sub> | 25.0%<sub>-0.93</sub> | 59.3%<sub>+6.79</sub> | 60.2% (650/1080) | 56.7% (68/120) |
| RGB-D + Colored GP | 91.7%<sub>+8.34</sub> | 58.3%<sub>0.00</sub> | **33.3%**<sub>-0.93</sub> | **61.1%**<sub>+2.47</sub> | 61.4% (663/1080) | 56.7% (68/120) |
| RGB-D + GP | **97.2%**<sub>+12.03</sub> | 52.8%<sub>0.00</sub> | **33.3%**<sub>0.00</sub> | **61.1%**<sub>+4.01</sub> | 62.1% (671/1080) | **59.2%** (71/120) |
| RGB-D + Grasp Part | 88.9%<sub>+1.85</sub> | 0.0%<sub>-14.81</sub> | **33.3%**<sub>-3.71</sub> | 40.7%<sub>-5.56</sub> | **62.6%** (676/1080) | 55.0% (66/120) |
| RGB-D + Colored Grasp Part | 94.4%<sub>+7.40</sub> | 0.0%<sub>-22.22</sub> | 8.3%<sub>-25.00</sub> | 34.3%<sub>-13.27</sub> | 61.4% (663/1080) | 55.0% (66/120) |

The FurnitureBench numerators for these rows are, respectively: RGB
`32/36, 22/36, 5/36`; RGB-D `35/36, 20/36, 9/36`; colored GP
`33/36, 21/36, 12/36`; GP `35/36, 19/36, 12/36`; grasp part
`32/36, 0/36, 12/36`; and colored grasp part `34/36, 0/36, 3/36`.

### 结论与结论边界

1. **当前的 target-conditioned policy 范式能够扩展到跨仿真环境以及显著更大的任务集合，现有结果最有力地支持的是紧凑几何 target 的有效性，而不是所有附加编码都有效。**
   在 AutoMate ID90 上，四种带空间条件的表示均超过 RGB-D 基线
   （`60.2%`），增幅为 `+1.2` 至 `+2.4 pp`；其中 GP 在 OOD10 上也取得
   最高成功率 `59.2%`，比 RGB-D 高 `2.5 pp`。更值得注意的是，带 target
   条件的策略表现出较强的零样本跨装配泛化能力：GP 的 ID90 与 OOD10
   成功率仅相差 `2.9 pp`，而 RGB 和 RGB-D 的差距分别为 `13.1 pp` 和
   `3.5 pp`。这说明紧凑的几何 target 接口能够支持跨 assembly 的知识迁移。
   但这些结果不能被解读为“OOD 本身更容易”或“所有 target 编码都会改善
   OOD”：Colored GP 在 OOD10 上仅与 RGB-D 持平，两种 grasp 表示则均比
   RGB-D 低 `1.7 pp`。

2. **额外的 cue 与包含旋转信息的表示在当前对比中没有表现出可靠增益。**
   Colored GP 在 ID90 和 OOD10 上分别比 plain GP 低 `1.0 pp` 和
   `2.5 pp`。加入部件/位姿信息（包括 rotation）的 grasp 表示没有改善跨域
   综合表现，并在 FurnitureBench Round-table 上出现任务级 collapse
   （`0/36`）；Colored grasp 在 Lamp 上也降至 `8.3%`。这一现象与优化不稳定
   或策略过度依赖高维位姿通道的解释相符。不过，本实验没有将 rotation 与
   表示类型、标注方式及 checkpoint 差异完全解耦。因此，当前能够严谨支持的
   结论是：rotation-rich conditioning 在本轮训练中与任务级 collapse 相关，
   而不是 rotation 已被证明是 collapse 的单独原因。

3. **混入大量插入任务对 FurnitureBench 产生的是选择性正迁移，而不是所有
   家具任务上的一致提升。该结论为后续混训的任务选择提供参考，相似任务和几何形状更容易形成正迁移。** 相比原始三 seed 的 FB 基准，四个非 grasp
   condition 的 overall 提升 `+2.47` 至 `+6.79 pp`，而 grasp 和 colored
   grasp 分别下降 `5.56` 和 `13.27 pp`。对表中六个 condition 等权平均后，
   FB overall 仅提高 `0.15 pp`。分任务结果更清楚地揭示了迁移来源：One-leg
   平均提升 `8.18 pp`，Round-table 和 Lamp 则分别变化 `-1.85 pp` 和
   `-5.87 pp`。因此，插入任务占比较高的混训显著改善了 peg 形状与 automate 更接近的
   One-leg，并大体维持了 FB 的整体水平，但尚未形成对长时程家具装配任务的
   普遍正迁移。

> **Invalid AutoMate panel (2026-09-25):** every AutoMate number in this
> historical table was generated through a bch_4090 Isaac Sim 4.5 workaround
> that disabled Fabric. Physics advanced, but wrist/front RGB-D geometry was
> frozen, so these values cannot rank visual policies. See
> [the camera-sync audit](automate_eval_camera_sync_audit_0925.md).
> FurnitureBench values are unaffected.

Updated: `2026-09-24T06:40:00+08:00`. FurnitureBench Validation36 is complete: all 21 condition/task cells passed the 36-rollout contract (seed 0 base 12 plus seed 12 supplement 24). The run used root commit `9dff9a4f8d9287345511a287a4ba88ff80954fda` and the authoritative FurnitureBench gitlink `7d57047689fe86f24ad3ba362693ce3d34b19e9d`; result root: `/home/bch/rr-joint90-0921/runs/joint90-fb-validation36-joint-protocol-v2-0923`. The old AutoMate rows below are retained only as raw diagnostics because they used the wrong `training2x + training/SBC` reset protocol. The corrected `default1 + hardest + SBC-off` AutoMate evaluation is now complete across all eight formal checkpoint conditions.

| Domain | Condition | Task | Rollout budget | State | Success |
|---|---|---|---:|---|---|
| FB | `rgb` | `one_leg` | 36 | complete | 32/36 (88.89%) |
| FB | `rgb` | `round_table` | 36 | complete | 22/36 (61.11%) |
| FB | `rgb` | `lamp` | 36 | complete | 5/36 (13.89%) |
| FB | `rgbd` | `one_leg` | 36 | complete | 35/36 (97.22%) |
| FB | `rgbd` | `round_table` | 36 | complete | 20/36 (55.56%) |
| FB | `rgbd` | `lamp` | 36 | complete | 9/36 (25.00%) |
| FB | `rgbd_colored_gp` | `one_leg` | 36 | complete | 33/36 (91.67%) |
| FB | `rgbd_colored_gp` | `round_table` | 36 | complete | 21/36 (58.33%) |
| FB | `rgbd_colored_gp` | `lamp` | 36 | complete | 12/36 (33.33%) |
| FB | `rgbd_gp` | `one_leg` | 36 | complete | 35/36 (97.22%) |
| FB | `rgbd_gp` | `round_table` | 36 | complete | 19/36 (52.78%) |
| FB | `rgbd_gp` | `lamp` | 36 | complete | 12/36 (33.33%) |
| FB | `rgbd_gp_skill` | `one_leg` | 36 | complete | 32/36 (88.89%) |
| FB | `rgbd_gp_skill` | `round_table` | 36 | complete | 23/36 (63.89%) |
| FB | `rgbd_gp_skill` | `lamp` | 36 | complete | 16/36 (44.44%) |
| FB | `rgbd_grasp_part_colored` | `one_leg` | 36 | complete | 34/36 (94.44%) |
| FB | `rgbd_grasp_part_colored` | `round_table` | 36 | complete | 0/36 (0.00%) |
| FB | `rgbd_grasp_part_colored` | `lamp` | 36 | complete | 3/36 (8.33%) |
| FB | `rgbd_skill` | `one_leg` | 36 | complete | 32/36 (88.89%) |
| FB | `rgbd_skill` | `round_table` | 36 | complete | 26/36 (72.22%) |
| FB | `rgbd_skill` | `lamp` | 36 | complete | 14/36 (38.89%) |
| AutoMate ID90 | `rgb` | aggregate | 1080 | invalid protocol | 543/1080 (diagnostic only) |
| AutoMate TEST10 | `rgb` | aggregate | 120 | invalid protocol | 62/120 (diagnostic only) |
| AutoMate ID90 | `rgbd` | aggregate | 1080 | invalid protocol | 503/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd` | aggregate | 120 | invalid protocol | 60/120 (diagnostic only) |
| AutoMate ID90 | `rgbd_colored_gp` | aggregate | 1080 | invalid protocol | 551/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd_colored_gp` | aggregate | 120 | invalid protocol | 74/120 (diagnostic only) |
| AutoMate ID90 | `rgbd_gp` | aggregate | 1080 | invalid protocol | 601/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd_gp` | aggregate | 120 | invalid protocol | 73/120 (diagnostic only) |
| AutoMate ID90 | `rgbd_gp_skill` | aggregate | 1080 | invalid protocol | 599/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd_gp_skill` | aggregate | 120 | invalid protocol | 73/120 (diagnostic only) |
| AutoMate ID90 | `rgbd_grasp_part_colored` | aggregate | 1080 | invalid protocol | 596/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd_grasp_part_colored` | aggregate | 120 | invalid protocol | 75/120 (diagnostic only) |
| AutoMate ID90 | `rgbd_skill` | aggregate | 1080 | invalid protocol | 589/1080 (diagnostic only) |
| AutoMate TEST10 | `rgbd_skill` | aggregate | 120 | invalid protocol | 73/120 (diagnostic only) |

Contracts: FB uses raw 240×320 RGB-D, synchronized center crop `[8:232, 48:272]` to 224×224 without interpolation, finite positive metric depth, original front camera, and scripted annotations. The formal AutoMate rerun uses scripted annotations, `default1` reset randomness, fixed hardest curriculum, SBC disabled, and the official task predicate.

## FurnitureBench aggregate

| Condition | one_leg | round_table | lamp | Aggregate |
|---|---:|---:|---:|---:|
| `rgb` | 32/36 | 22/36 | 5/36 | 59/108 (54.63%) |
| `rgbd` | 35/36 | 20/36 | 9/36 | 64/108 (59.26%) |
| `rgbd_colored_gp` | 33/36 | 21/36 | 12/36 | 66/108 (61.11%) |
| `rgbd_gp` | 35/36 | 19/36 | 12/36 | 66/108 (61.11%) |
| `rgbd_gp_skill` | 32/36 | 23/36 | 16/36 | 71/108 (65.74%) |
| `rgbd_grasp_part_colored` | 34/36 | 0/36 | 3/36 | 37/108 (34.26%) |
| `rgbd_skill` | 32/36 | 26/36 | 14/36 | 72/108 (66.67%) |
| **All FB conditions** | **233/252 (92.46%)** | **131/252 (51.98%)** | **71/252 (28.17%)** | **435/756 (57.54%)** |

## AutoMate corrected rerun (final)

All `800/800` cells are complete: seven original conditions contributed 700 cells and the `rgbd_grasp_part` addendum contributed 100 cells. Every cell passed the fixed contract: 12 rollouts, 12 environments, `start_seed=9143000`, `max_steps=75`, `default1`, hardest curriculum, SBC disabled, scripted annotations, `curr_max_disp` fixed at its upper bound, zero initial `task_success`, zero initial `inserted_height`, and no rollout errors. The aggregate contract audit reported `0` failures.

| Condition | ID90 | TEST10 | All |
|---|---:|---:|---:|
| `rgb` | 284/1080 (26.30%) | 34/120 (28.33%) | 318/1200 (26.50%) |
| `rgbd` | 144/1080 (13.33%) | 12/120 (10.00%) | 156/1200 (13.00%) |
| `rgbd_colored_gp` | 162/1080 (15.00%) | 15/120 (12.50%) | 177/1200 (14.75%) |
| `rgbd_gp` | 203/1080 (18.80%) | 22/120 (18.33%) | 225/1200 (18.75%) |
| `rgbd_gp_skill` | 110/1080 (10.19%) | 10/120 (8.33%) | 120/1200 (10.00%) |
| `rgbd_grasp_part` | 126/1080 (11.67%) | 10/120 (8.33%) | 136/1200 (11.33%) |
| `rgbd_grasp_part_colored` | 181/1080 (16.76%) | 16/120 (13.33%) | 197/1200 (16.42%) |
| `rgbd_skill` | 105/1080 (9.72%) | 8/120 (6.67%) | 113/1200 (9.42%) |
| **All conditions** |  |  | **1442/9600 (15.02%)** |

The corrected protocol removes the earlier reset-success artifact: the early audit observed only 1 first-step success in 216 rollouts (0.46%), versus 44.10% in the invalid protocol. The final ID90/TEST10 differences are small and do not reproduce the invalid run's anomalously higher OOD success rates.

## AutoMate checkpoint completeness audit

The first-round training produced eight formal checkpoint conditions, while the running corrected AutoMate queue was created with seven. The seven present local copies all passed `/home/bch/rr-joint90-0921/checkpoints/AUTHORITATIVE_CHECKPOINTS.sha256`; their condition names match the formal source list. The omitted condition is `rgbd_grasp_part` (non-colored), which must receive the same `90 ID90 + 10 TEST10` cells, 12 rollouts per cell, after its formal checkpoint is available on bch_4090.

| Condition | bch_4090 checkpoint | SHA-256 status | Evaluation status |
|---|---|---|---|
| `rgb`, `rgbd`, `rgbd_colored_gp`, `rgbd_gp`, `rgbd_gp_skill`, `rgbd_grasp_part_colored`, `rgbd_skill` | present | manifest verified | completed in the 700-cell main queue |
| `rgbd_grasp_part` | present | source and bch SHA-256 match: `da24fe1097d8313c0b08eace4d740c413022b0c6fd8137c3f9a7665f7a0ecfa9` | completed in the 100-cell addendum |

The source NAS is `10.71.106.246:/volume2/datasets_tmp`. It was reached through the approved `base -> zju_4090_238 -> NAS` path and the 589 MB formal checkpoint was copied to bch_4090 via base. Its bch_4090 SHA-256 matches the source. The addendum tmux session completed the omitted 90 ID90 plus 10 TEST10 cells after the main queue, without GPU overlap. The final AutoMate table therefore contains 800 validated cells across eight conditions, rather than the original 700-cell queue alone.

## Corrected protocol divergence

The earlier all-zero non-RGB rerun was invalid. Its root command-line arguments matched the Joint90 protocol, but the server had manually advanced the `furniture-bench` submodule to `f0d9cdec6944dfe717c4372351e456d9ec47fc78`, which did not implement `depth_positive_meters`. The root evaluator passed that keyword through `**kwargs`, so it was silently ignored and raw negative Isaac Gym depth reached checkpoints trained on positive metric depth. The old JSON field `eval_depth_contract=positive_meters` was inferred from the requested checkpoint mode rather than measured from the actual observation path, producing a false-positive contract check.

The corrected run restored the root gitlink `7d57047689fe86f24ad3ba362693ce3d34b19e9d`, verified the active `abs()` depth conversion before launch, isolated the configured environment with `PYTHONNOUSERSITE=1`, and rejected any result whose annotation preprocessing did not match its checkpoint training configuration. All 18 new result JSON files passed the independent contract audit; the first gate changed from 0/12 to `rgbd/one_leg = 11/12`.

Reliability note: AutoMate Train90/Test10 shows no task-level leakage in the retained split/index/checkpoint evidence, but the entire performance panel used the wrong reset protocol and is invalid for formal reporting. See [the protocol audit](automate_joint90_eval_protocol_audit_0923.md) and the now-superseded [OOD/ID reliability audit](automate_ood_id_reliability_audit_0923.md).
