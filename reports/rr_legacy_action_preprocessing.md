# RR legacy action 预处理：保留决定与已知风险

状态：**暂时解决（保留 legacy 行为，后续单独调参）**
记录日期：2026-08-27

## 当前决定

当前真机/仿真混训继续复用 Robust Rearrangement（RR）的历史 action 预处理流程，不修改现有 pickle、LMDB、Dataset、Policy 或控制器语义：

- pickle 的 `actions` 按 8D delta 读取：base-frame `delta xyz`、局部右乘 `delta quat_xyzw`、gripper；
- 预处理先限制 delta 平移和旋转；
- 再由当前 eepose 与处理后的 delta 构造 `action/pos`；
- Policy 继续训练和输出 10D absolute pose：`xyz + rotation_6d + gripper`；
- real/sim 直接拼接训练，不增加 sample weight。

作出这一决定的主要原因是：RR 原始代码和已有仿真 Policy 都使用了该流程，当前优先保证与论文实验及历史 checkpoint 的可比性。这里的“暂时解决”表示风险已经识别并记录，但暂不改变训练标签定义。

实现位置：

- [`process_pickle_file`](../src/data_processing/process_pickles.py) 中先处理 delta，再构造 `action/pos`；
- [`clip_quat_xyzw_magnitude`](../src/data_processing/utils.py) 实现旋转缩放；
- [`process_pickles_to_lmdb`](../src/data_processing/process_pickles_to_lmdb.py) 复用同一处理函数，因此 LMDB 路径具有相同行为。

## 两个待调超参数

### 1. 旋转缩放系数

当前硬编码值：

```text
rotation_episode_clip_mag_rad = 0.35
```

当前实现对整条 episode 的所有 rotation vector 计算一个 Frobenius 范数：

```text
M = sqrt(sum_t ||r_t||^2)
s = min(1, rotation_episode_clip_mag_rad / M)
r'_t = s * r_t
```

这不是逐 timestep 的角度裁剪。它会让同一帧的缩放程度依赖 episode 长度和 episode 内其他帧的旋转量。

现有数据测得：

| 数据域 | episode 平均缩放系数 |
|---|---:|
| 真机 | 0.0743 |
| 仿真 | 0.1146 |

虽然这两个系数差异明显，但在当前 absolute-pose 标签中的旋转偏差分布接近：真机/仿真中位数分别约为 5.80°/5.94°，p95 均约为 17.6°。因此不能只根据 episode scale 判断实际 sim-real 标签差异。

### 2. xyz 裁剪系数

当前硬编码值：

```text
translation_clip_per_axis_m = 0.025
```

当前实现对 `delta x/y/z` 分量分别裁剪到 `[-0.025, 0.025] m`，不是对三维平移范数做裁剪。

现有数据中各分量达到裁剪边界的比例为：

- 真机：约 1.8%–2.9%；
- 仿真：约 3.5%–4.8%。

若以“任意一个 xyz 分量被修改”为标准，受影响 timestep 约为真机 6.25%、仿真 10.92%。

## 已知问题

1. 两种处理都会静默修改训练标签，LMDB 当前没有完整记录具体缩放结果。
2. episode-level rotation scaling 具有非局部性，并不是真正的逐 timestep 角速度限制。
3. xyz 的逐轴裁剪依赖坐标轴，不等价于限制末端线速度范数。
4. 这些离线处理只能改变训练 target，不能保证推理时 Policy 输出满足速度或安全限制。
5. 仿真 legacy pickle 顶层可能记录 `action_type=pos`，但保存的 `actions` 已在落盘前转成 delta；该字段表示采集/控制模式，不能直接当作 payload 类型。

## 后续调参原则

后续只有在独立实验中调整这两个超参数，不覆盖现有数据：

1. 将硬编码值参数化，并写入 LMDB metadata 和训练配置；
2. 使用新 output suffix 重建数据，保留 `rr_legacy` LMDB 和 checkpoint；
3. 至少比较 legacy、较弱限制、无离线限制三组设置；
4. 同时报告 Policy target-current 的平移/旋转分布、裁剪率、任务成功率和轨迹平滑度；
5. 真机安全限速应在 Policy 输出后、控制器执行前逐 timestep 完成，并分别保存 raw Policy action 与实际执行 action，不能依赖本离线预处理。

## 本次不处理的范围

- eepose 的坐标参考系和物理控制点对齐；
- 真机/仿真 observation-action 时间对齐；
- RGB、depth 和 skill 的域差；
- 真机在线速度、角速度及加速度限制。

这些问题分别审计，不与本次 legacy action 预处理决定混合。
