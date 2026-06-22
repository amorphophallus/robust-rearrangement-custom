# Skill Annotation 坐标系文档

## 1. 坐标系层级

```
World (Sim Global)
  │  Isaac Gym rb_states 在此坐标系
  │  env_offset = base_pos_world - franka_origin
  │  单 env 时 env_offset ≈ 0，多 env 时为 grid 偏移
  ▼
Env-Local (Sim Local)
  │  _make_env_local_annotation_inputs 将 rb_states 从 world 转到此坐标系
  │  Camera (front_cam_pos) 在此坐标系
  │  franka_from_origin_mat: Env-Local → Robot
  ▼
Robot (Franka Base)
  │  ee_pos = hand_world - base_world (相对基座的 hand 位置)
  │  base_tag_from_robot_mat: Robot → April
  ▼
April (基座 AprilTag)
     sim_to_april_mat = inv(T_tag) @ inv(T_franka): Env-Local → April
     april_to_robot_mat = T_tag: April → Robot
```

## 2. 关键矩阵

| 矩阵 | 数值 | 方向 |
|------|------|------|
| `franka_from_origin_mat` | `get_mat((-0.3, 0, 0.415), (0,0,0))` | Env-Local → Robot |
| `base_tag_from_robot_mat` | `get_mat((0.3015, 0, 0), (π, 0, π/2))` | April → Robot |
| `sim_to_april_mat` | `inv(T_tag) @ inv(T_franka)` | Env-Local → April |
| `april_to_robot_mat` | `T_tag` | April → Robot |
| `robot_to_ee_mat` | `rot_mat((π, 0, 0))` | Robot → EE |

恒等式：`april_to_robot @ sim_to_april = inv(T_franka)`（纯平移 0.3, 0, -0.415）

## 3. Annotation 输入各量的坐标系

| 变量 | 坐标系 | 来源 |
|------|--------|------|
| `rb_states[:,:3]` | Env-Local | world pos - env_offset |
| `rb_states[:,3:7]` | World | 未修改（纯平移不改变旋转） |
| `ee_pos` | Robot | `hand_world - base_world` |
| `ee_quat` | World (=Robot) | base 无旋转，两坐标系等价 |
| `sim_to_april_mat` | 期望 Env-Local 输入 | 常量 |
| `april_to_robot_mat` | April → Robot | 常量 |
| `base_pos` | Env-Local | `= franka_origin` |
| guidance_point | World | `gp_robot + base_pos` |

## 4. `Leg._compute_skill_place_target` 变量坐标系

| 变量 | 坐标系 |
|------|--------|
| `table_pose_env` | Env-Local |
| `leg_pose_env` | Env-Local |
| `table_pose_april` | April |
| `leg_pose_april` | April |
| `leg_pose_robot` | Robot |
| `table_pose_robot` | Robot |
| `table_hole_pose_robot` | Robot |
| `target_leg_pose_robot` | Robot |
| `rel_robot` | Robot |
| `ee_pose_robot`（输入） | Robot |
| 返回值 | Robot |

## 5. `env_offset` 说明

```python
env_offset = base_pos_world - franka_origin
```
- `base_pos_world`: rb_states[base_idxs[env_idx], :3] — Franka base 在 world 坐标
- `franka_origin`: franka_from_origin_mat[:3,3] — 设计的 base 位置

单 env 时两者应相等 → env_offset ≈ 0。实际运行中可能有微小差异。
多 env 时 env_offset = 该 env 的 grid 偏移。

## 6. 已知问题

- **gp z 偏差**：one_leg place 阶段 guidance_point z ≈ 0.522（world），EE z ≈ 0.016。
  计算链自洽（gp = target_robot_z + franka_origin_z），但 table_env_z 和预期桌板位置不匹配，env_offset 可能非零。
- **_find_leg_pose_x_look_front_skill**：place 阶段每帧重算，离散朝向切换可能导致 guidance_point 跳变 ~1-2cm。
