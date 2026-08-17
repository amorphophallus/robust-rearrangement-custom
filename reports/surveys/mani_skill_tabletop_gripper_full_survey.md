# ManiSkill Tabletop Gripper 调研

| Task | 满足 `Franka 单臂桌面` | 任务/多样性 | 官方任务页信号 | 成功判定条件 | 官方可下载资产（精确 episode 数） | 官方 MP solution | 官方 MP 是否直接暴露可提取的 subtask TCP target pose | 官方 RL 证据 | 你拿到 200 条的最现实路径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AssemblingKits-v1` | 是 | kit 几何随机；20 种 shape；misplaced piece 初始位姿随机 | Dense `❌`；Demos `❌`；Max steps `200` | 任务页无官方 `Success/Fail` 信号；按语义应为 piece 插入匹配槽位 | 无官方 demo zip（404） | 无 | 否 | 无 | 需要从零自采。第一批成本高。 |
| `LiftPegUpright-v1` | 是 | 单 peg；初始 xy 随机；无物体类别多样性 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；peg 被提起并立正/稳定 upright | RL `993` (`pd_joint_delta_pos`)；RL `1015` (`pd_ee_delta_pose`)；官方 ckpt `2` 个 | 有 | 是。`reach_pose / grasp_pose / lift_pose / final_pose / lower_pose` | 有官方 checkpoint | 可直接用官方 RL 或自己跑官方 MP 采 `200+`。 |
| `PegInsertionSide-v1` | 是 | peg/box 尺寸随机；peg 和 box 初始位姿都随机 | Dense `✅`；Demos `✅`；Max steps `100` | 有官方 success 信号；peg 成功插入 side hole / goal pose 达成 | RL `1000`（无 checkpoint）；MP `1000` | 有 | 是。`grasp_pose / pre_insert_pose / insert_pose` | 有官方 RL rollouts only；属于官方 RL small benchmark | 最稳的是直接复用官方 MP `1000` 条。 |
| `PickClutterYCB-v1` | 是 | YCB 多物体 clutter；多样性强 | Dense `❌`；Success/Fail `❌`；Demos `❌`；Max steps `100` | 任务页无官方 `Success/Fail` 信号；按语义应为目标物体被提起并移到 goal | 无官方 demo zip（404） | 无 | 否 | 无 | 不适合第一批。需要自己做 expert collector。 |
| `PickCube-v1` | 是 | 单 cube；初始位姿随机；goal 3D 位置随机 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；cube 被抓起并移动到 goal position | RL `997` (`pd_joint_delta_pos`)；RL `1013` (`pd_ee_delta_pose`)；RL `1022` (`pd_ee_delta_pos`)；Teleop `10`；MP `1000`；官方 ckpt `3` 个 | 有 | 是。`reach_pose / grasp_pose / goal_pose` | 有官方 checkpoint；属于官方 RL small benchmark | 直接可用，`200` 条没有数量问题。 |
| `PickCubeSO100-v1` | 否 | 与 `PickCube` 同语义，但 robot 是 SO100 | Dense `✅`；Demos `❌`；Max steps `50` | 有官方 success 信号；同 `PickCube` 语义 | 无官方 demo zip（404） | 无 | 否 | 无 | 不建议纳入主 benchmark。 |
| `PickCubeWidowXAI-v1` | 否 | 与 `PickCube` 同语义，但 robot 是 WidowXAI | Dense `✅`；Demos `❌`；Max steps `50` | 有官方 success 信号；同 `PickCube` 语义 | 无官方 demo zip（404） | 无 | 否 | 无 | 不建议纳入主 benchmark。 |
| `PickSingleYCB-v1` | 是 | 单 YCB 目标物体；物体类别随机；初始位姿随机；goal 3D 位置随机 | Dense `✅`；Demos `❌`；Max steps `50`；官方建议 GPU 下至少 `128` env 覆盖全部 YCB 物体 | 有官方 success 信号；目标 YCB 物体被拿起并移到 3D goal position | 无官方 demo zip（404） | 无 | 否 | 无 | 定义非常适合你，但要自己采。 |
| `PlaceSphere-v1` | 是 | sphere + bin；两者位置都随机；类别多样性低 | Dense `✅`；Demos `❌`；Max steps `50` | 有官方 success 信号；sphere 被放入 bin / goal receptacle | 无官方 demo zip（404） | 有 | 是。`grasp_pose / lift_pose / goal_pose / align_pose` | 无 | 没有官方 demo，但有官方 MP，自己采 `200+` 很可行。 |
| `PlugCharger-v1` | 是 | charger 与 receptacle 初始位姿都随机 | Dense `❌`；Demos `✅`；Max steps `200` | charger head 成功插入 receptacle / goal pose 达成 | MP `1000` | 有 | 是。`grasp_pose / pre_insert_pose / insert_pose` | 无官方 RL 轨迹/ckpt | 直接复用官方 MP `1000` 条即可。 |
| `PokeCube-v1` | 是，但不是 pick-place | peg + cube；cube goal 位置显式定义 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；cube 被 poke/push 到 goal region | RL `960` (`pd_joint_delta_pos`)；RL `579` (`pd_ee_delta_pose`)；RL `761` (`pd_ee_delta_pos`)；官方 ckpt `3` 个 | 无 | 否 | 有官方 checkpoint | 可做对照任务，不作为主线。 |
| `PullCube-v1` | 是，但不是 pick-place | 单 cube；goal region 显式定义 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；cube 被拉到 goal region | RL `1024` (`pd_joint_delta_pos`)；RL `1024` (`pd_ee_delta_pose`)；RL `1024` (`pd_ee_delta_pos`)；官方 ckpt `3` 个 | 有 | 是。`reach_pose / goal_pose` | 有官方 checkpoint | 数量完全够，但不是 pick-place。 |
| `PullCubeTool-v1` | 是，但不是 pick-place | tool + out-of-reach cube；需要工具使用 | Dense `✅`；Demos `✅`；Max steps `100` | 有官方 success 信号；借助工具把 cube 拉到目标区域 | MP `1000` | 有 | 是。`grasp_pose / lift_pose / approach_pose / hook_pose / target_pose` | 无 | 官方 MP 已够采 `200+`。 |
| `PushCube-v1` | 是，但不是 pick-place | 单 cube；goal region 显式定义 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；cube 被推到 goal region | RL `1023` (`pd_joint_delta_pos`)；RL `1018` (`pd_ee_delta_pose`)；RL `1021` (`pd_ee_delta_pos`)；MP `1000`；官方 ckpt `3` 个 | 有 | 是。`reach_pose / goal_pose` | 有官方 checkpoint；属于官方 RL small benchmark | 采数最容易，但任务偏 pushing。 |
| `PushT-v1` | 是，但不是 pick-place | T-block 位姿和 goal T 区域随机；更偏 planar manipulation | Dense `✅`；Demos `✅`；Max steps `100` | 有官方 success 信号；T 形物体与目标 T 区域对齐/重合 | RL `999` (`pd_joint_delta_pos`)；RL `719` (`pd_ee_delta_pose`)；RL `888` (`pd_ee_delta_pos`)；官方 ckpt `3` 个 | 无 | 否 | 有官方 checkpoint；属于官方 RL small benchmark | 可做对照 benchmark，不适合主 pick-place 路线。 |
| `RollBall-v1` | 是，但不是 pick-place | ball 和目标区域位置随机 | Dense `✅`；Demos `✅`；Max steps `80` | 有官方 success 信号；ball 被滚到目标区域 | RL `830` (`pd_joint_delta_pos`)；RL `855` (`pd_ee_delta_pose`)；RL `688` (`pd_ee_delta_pos`)；官方 ckpt `3` 个 | 无 | 否 | 有官方 checkpoint | 可做 rolling 对照，不建议放主集合。 |
| `StackCube-v1` | 是 | 两个 cube；两者初始位姿随机；pick + place 两阶段清晰 | Dense `✅`；Demos `✅`；Max steps `50` | 有官方 success 信号；一个 cube 被稳定放到另一个 cube 顶部 | RL `932` (`pd_joint_delta_pos`)；RL `995` (`pd_ee_delta_pose`)；RL `902` (`pd_ee_delta_pos`)；MP `1000`；官方 ckpt `3` 个 | 有 | 是。`grasp_pose / lift_pose / align_pose`，`goal_pose = cubeB.pose + z` | 有官方 checkpoint | 非常强，`200` 条轻松。 |
| `StackPyramid-v1` | 是 | 3 个 cube；长时程两次 place；初始位姿随机 | Dense `❌`；Demos `✅`；Max steps `250` | 最终三块搭成金字塔/bridge 结构 | MP `1000` | 有 | 部分是。第二阶段 target pose 清楚，第一阶段 side-by-side pose 更 heuristic | 无 | 可以采 `200+`，但比 `StackCube` 更长、更脆。 |
| `TurnFaucet-v1` | 是，但不是 pick-place | articulation turning；初始状态通常依赖 faucet 几何 | Dense `❌`；Demos `❌`；Max steps `200` | 任务页无官方 `Success/Fail` 信号；按语义应为阀门角度转到目标开合状态 | 无官方 demo zip（404） | 无 | 否 | 无 | 不建议第一批。 |
| `TwoRobotPickCube-v1` | 否 | 两机器人协作 pick-place | Dense `✅`；Demos `✅`；Max steps `100` | 有官方 success 信号；双机器人协作把 cube 拿到 goal | RL `983`；官方 ckpt `1` 个 | 无 | 否 | 有官方 checkpoint | 排除，不满足单臂。 |
| `TwoRobotStackCube-v1` | 否 | 两机器人协作 stack | Dense `✅`；Demos `✅`；Max steps `100` | 有官方 success 信号；双机器人协作完成 stack | RL `1007`；官方 ckpt `1` 个 | 无 | 否 | 有官方 checkpoint | 排除，不满足单臂。 |

| 优先级 | Task | 直接开工理由 |
| --- | --- | --- |
| S | `PickCube-v1` | 官方 MP/RL/teleop 都有；MP 里直接有 `goal_pose`；`200` 条现成够；target pose 几乎零调参。 |
| S | `StackCube-v1` | 标准二阶段 pick-place；MP 显式给 `grasp/lift/align`；非常适合迁移到你的标注管线。 |
| S | `PlaceSphere-v1` | 没有官方 demo，但有官方 MP solution；自己批量采数据最省事；`goal_pose / align_pose` 明确。 |
| S | `PegInsertionSide-v1` | 官方 MP 直接给 `pre_insert/insert`；对 pose-conditioned skill 很强；MP/rollout 数量充足。 |
| S | `PlugCharger-v1` | 官方 MP 直接给 `pre_insert/insert`；初始化随机，适合测泛化；`1000` 条现成 MP 数据。 |
| A | `PickSingleYCB-v1` | 最符合“多物体、多摆放、pick-place”；goal 明确；缺点是没有官方 demo/MP，需要你自己补 collector。 |
| B | `AssemblingKits-v1` | 任务有价值，但 slot 匹配和 target pose 计算更费人。 |
| B | `StackPyramid-v1` | 可以做长时程扩展，但比 `StackCube` 更难稳，第一阶段 pose 设计更 heuristic。 |
