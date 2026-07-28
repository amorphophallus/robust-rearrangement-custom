# VLM Data Generator

`src.vlm_data_generator` 用于生成 VLM SFT 数据集。它有两层能力：

- 批量调用 `src.eval.evaluate_model` 生成 `rgbd-only-skill` rollout pickle。
- 从 rollout pickle 转成 VLM SFT 友好的单帧双图数据格式。

当前工具不会在图片上画任何 point、grasp 或 skill 文本。front/wrist 两路图片都是原始 RGB 帧。`target_point_2d` 只读取 pickle 中已有的 front camera `guidance_point_2d.color_image2`。

## 坐标系约定

`target_point_2d`

2D target point 是 front camera 图像上的像素坐标 `[u, v]`，不再区分 front/wrist 两个字段。`u` 是从左到右增加的水平像素坐标，`v` 是从上到下增加的垂直像素坐标。坐标是在保存到数据集的 resized front image 上定义的，所以可以直接对应 `images/..._front.png` 的像素位置。没有 front 投影时 `target_point_2d = null`。

`target_point_3d`

3D target point 是 `sim_local` 坐标系下的位置，单位是米。这个点来自 rollout pickle 中的 `observation["guidance_point"]`，代码中由 `skill_annotation_util.project_3d_to_2d(point_sim_local, camera_info)` 投影到 2D。

`state_info.base`

`state_info.base` 是放进 user prompt 的主要本体信息，固定包含 `ee_pos_sim`、`ee_quat_sim`、`ee_pos_vel`、`ee_ori_vel`、`gripper_width`。其中 `ee_pos_sim` / `ee_quat_sim` 来自 rollout pickle 的 `robot_state.ee_pos_sim` / `robot_state.ee_quat_sim`，和 `target_point_3d` 使用同一个 `sim_local` 坐标系。默认情况下，旧 pickle 如果没有 `ee_pos_sim` 会被跳过，并在 manifest 的 `skipped.missing_sim_local_eepose` 中计数；只有显式设置 `--allow-legacy-eepose` 时才允许转换旧数据。

`state_info.extra`

`state_info.extra` 是放进 user prompt 的额外本体信息，固定包含 `joint_positions`、`joint_velocities`、`joint_torques`。如果 pickle 中有 `parts_poses`，也会放到 `state_info.extra.parts_poses`。

坐标语义还会记录在每条样本的 `metadata.coordinate_frames` 和 `manifest.schema.coordinate_frames` 中，方便训练或检查脚本读取；但不会放进 `state_info`，避免把说明文字拼到 user prompt。

## 推荐用法

端到端生成一个新数据集。下面的命令可以直接复制粘贴运行：使用本地 `one_leg` RPPO low checkpoint，生成 100 条成功 rollout，并输出到 `${DATA_PROCESSED_DIR}/vlm`。如果当前 shell 没有设置 `DATA_PROCESSED_DIR`，命令会默认使用 `/data/hy/robust-rearrangement/data/processed`。

```bash
cd /data/hy/robust-rearrangement
export DATA_PROCESSED_DIR="${DATA_PROCESSED_DIR:-/data/hy/robust-rearrangement/data/processed}"
export DATA_DIR_RAW="${DATA_DIR_RAW:-/data/hy/robust-rearrangement}"
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib:${LD_LIBRARY_PATH:-}"

/home/hy/anaconda3/envs/rr/bin/python -m src.vlm_data_generator generate \
  --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
  --task-rollout one_leg=100 \
  --output-dir "${DATA_PROCESSED_DIR}/vlm" \
  --gpu 0 \
  --randomness low \
  --n-envs 3 \
  --rollout-run-name vlm_rppo_one_leg_low_direct \
  --target-successes \
  --output-mode overwrite
```

用三个不同 task-specific RPPO checkpoint 生成同一个数据集。第一条命令用 `overwrite` 初始化 `${DATA_PROCESSED_DIR}/vlm`，后两条命令用 `append` 追加到同一个数据集：

```bash
cd /data/hy/robust-rearrangement
source ~/.bashrc
export DATA_PROCESSED_DIR="${DATA_PROCESSED_DIR:-/data/hy/robust-rearrangement/data/processed}"
export DATA_DIR_RAW="${DATA_DIR_RAW:-/data/hy/robust-rearrangement}"
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib:${LD_LIBRARY_PATH:-}"
conda activate rr

python -m src.vlm_data_generator generate \
  --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
  --task-rollout one_leg=100 \
  --output-dir "${DATA_PROCESSED_DIR}/vlm" \
  --gpu 0 \
  --randomness low \
  --n-envs 3 \
  --rollout-run-name vlm_rppo_one_leg_low_direct \
  --target-successes \
  --output-mode overwrite && \
python -m src.vlm_data_generator generate \
  --wt-path checkpoints/rppo/round_table/low/actor_chkpt.pt \
  --task-rollout round_table=100 \
  --output-dir "${DATA_PROCESSED_DIR}/vlm" \
  --gpu 0 \
  --randomness low \
  --n-envs 3 \
  --rollout-run-name vlm_rppo_round_table_low_direct \
  --target-successes \
  --output-mode append && \
python -m src.vlm_data_generator generate \
  --wt-path checkpoints/rppo/lamp/low/actor_chkpt.pt \
  --task-rollout lamp=100 \
  --output-dir "${DATA_PROCESSED_DIR}/vlm" \
  --gpu 0 \
  --randomness low \
  --n-envs 3 \
  --rollout-run-name vlm_rppo_lamp_low_direct \
  --target-successes \
  --output-mode append
```

只转换已有 pickle。`--input-dir` 可以传目录，也可以直接传具体 `.pkl` 文件；传目录时会递归扫描该目录下的 pickle。

```bash
cd /data/hy/robust-rearrangement
export DATA_PROCESSED_DIR="${DATA_PROCESSED_DIR:-/data/hy/robust-rearrangement/data/processed}"

/home/hy/anaconda3/envs/rr/bin/python -m src.vlm_data_generator convert \
  --task-rollout one_leg=100 \
  --task-rollout round_table=100 \
  --task-rollout lamp=100 \
  --input-dir /data/hy/robust-rearrangement/raw/diffik/sim/one_leg/rollout/low/rgbd-only-skill \
  --input-dir /data/hy/robust-rearrangement/raw/diffik/sim/round_table/rollout/low/rgbd-only-skill \
  --input-dir /data/hy/robust-rearrangement/raw/diffik/sim/lamp/rollout/low/rgbd-only-skill \
  --output-dir "${DATA_PROCESSED_DIR}/vlm" \
  --output-mode overwrite
```

## 子命令

`rollouts`

只跑原始 rollout pickle，不转换 VLM 数据。

`convert`

只把已有 pickle 转成 VLM SFT 数据。

`generate`

先跑 rollout，再转换本次生成的 pickle。

## 关键参数

`--task-rollout TASK=COUNT`

每个任务要多少条 rollout。可重复指定，例如 `--task-rollout one_leg=100 --task-rollout lamp=100`。兼容别名 `round_tabl -> round_table`。

`--tasks ... --rollouts-per-task N`

给多个任务统一设置 rollout 数量。

`--wt-path / --run-id / --sweep-id / --project-id`

rollout checkpoint 来源，四选一。`convert` 不需要这些参数。

`LD_LIBRARY_PATH`

Isaac Gym native binding 需要能找到 conda 环境中的 `libpython3.8.so.1.0`。README 的命令已经显式导出 `/home/hy/anaconda3/envs/rr/lib`；工具也会自动把当前 Python 环境的 `sys.prefix/lib` 传给内部 `evaluate_model` 子进程，避免直接运行 `python -m src.vlm_data_generator ...` 时漏掉这个路径。

`--output-dir`

VLM 数据集输出目录。

`--output-mode {error,append,overwrite}`

输出目录已有数据时的处理方式。`error` 是默认值，防止误覆盖；`append` 会读取已有 `messages.jsonl` 或 `qwen_llava_sharegpt.json` 后合并新样本；`overwrite` 会删除本工具生成的 `messages.jsonl`、`qwen_llava_sharegpt.json`、`llamafactory_*.json`、`manifest.json`、`images/`、`depth/` 后重建。

`--format {both,all,messages-jsonl,sharegpt-json,llamafactory-json}`

选择输出数据格式。默认 `both` 会输出历史格式 `messages.jsonl` 和 `qwen_llava_sharegpt.json`。`llamafactory-json` 只输出 `llamafactory_<state_mode>.json` 和对应的 `llamafactory_<state_mode>_dataset_info.json`。`all` 会同时输出三种索引格式。

`--llamafactory-state-mode {placeholder,base,base-extra}`

写 LLaMAFactory JSON 时如何处理 user prompt 里的 `<state_info>` 占位符。`base` 会替换为 `{"base": state_info.base}`，这是默认值；`base-extra` 会同时包含 `extra`；`placeholder` 会保留占位符，通常只用于调试。

`--rollout-run-name`

插入 rollout raw path 的额外目录名。用多个 checkpoint 追加同一个 VLM 数据集时建议设置为 `ckpt_a`、`ckpt_b` 等，便于追踪来源。

`--target-successes`

把 `TASK=COUNT` 解释为目标成功 rollout 数。开启后，rollout 阶段会按 `gpu-snatcher/auto_data_preparation.sh` 的模式传 `--n-rollouts ${n_envs}` 和 `--target-successes COUNT`，直到 evaluator 收集到指定数量的成功 rollout。

`--frame-stride`

从每条 rollout 采样帧的步长。默认 `1` 表示每帧都导出。

`--max-frames-per-rollout`

每条 rollout 最多导出多少帧。默认 `0` 表示不限制。

`--allow-legacy-eepose`

允许转换缺少 `robot_state.ee_pos_sim` 的旧 pickle。默认不开启，因为这样无法严格保证 `target_point_3d` 和 `state_info.base.ee_pos_sim` 在同一个坐标系。

`--system-prompt`

默认不需要设置。工具内置了 `one_leg`、`round_table`、`lamp` 三个 task-specific system prompt，分别用自然语言描述该任务的拼装过程：第一步做什么、第二步做什么，以及这些步骤对应的 skill progression。设置该参数会覆盖所有任务的内置 prompt。

内置 prompt 的作用是给 VLM 提供任务流程先验；它不是 assistant label。assistant 仍然只监督 `skill`、`target_point_2d`、`target_point_3d` 三个字段。

`--no-save-depth-npy`

默认会把 depth image 以 `.npy` 保存，并在 metadata 中记录路径。加这个参数可关闭 depth 落盘。

## Rollout 配置

`generate` 和 `rollouts` 直接调用 `src.eval.evaluate_model`，但 rollout 参数对齐 `/data/hy/gpu-snatcher/auto_data_preparation.sh` 的任务配置。开启 `--target-successes` 时，实际调用模式如下：

```bash
/home/hy/anaconda3/envs/rr/bin/python -m src.eval.evaluate_model \
  --n-envs 3 \
  --n-rollouts 3 \
  --target-successes 100 \
  -f one_leg \
  --if-exists overwrite \
  --max-rollout-steps 700 \
  --action-type pos \
  --observation-space image \
  --randomness low \
  --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
  --save-rollouts \
  --save-depth-image \
  --output-only-pickle \
  --annotate-skill \
  --max-saved-rollouts 100 \
  --rollout-after-success 200 \
  --rollout-suffix-model-name vlm_rppo_one_leg_low_direct
```

这对应 `rgbd-only-skill`：

- 保存 RGBD 和 skill 元信息。
- 不保存 failure pickle，转换时只读取 `success/`。
- 不启用 `--guidance-point-on-image`。
- 不启用 `--grasp-annotation-on-image`。
- 不启用 `--skill-on-image`。
- 图片上没有任何可视化标注。

task-specific rollout 参数：

| task | `--max-rollout-steps` | `--rollout-after-success` |
|---|---:|---:|
| `one_leg` | 700 | 200 |
| `round_table` | 1000 | 100 |
| `lamp` | 1000 | 20 |

一次检查 rollout 中观察到的 `one_leg` skill 顺序为：

```text
top-leg-pick -> top-leg-push -> leg-top-pick -> leg-top-place -> leg-top-insert -> leg-top-screw
```

据此，`one_leg` 的内置 system prompt 描述为：先接近 tabletop connector 区域，推动/调整 top-leg 接触点；然后抓取 leg，把 leg 放到 tabletop socket；最后插入并旋紧 leg。

## 输出目录结构

```text
<output-dir>/
  messages.jsonl
  qwen_llava_sharegpt.json
  llamafactory_base.json
  llamafactory_base_dataset_info.json
  manifest.json
  images/
    <task>/
      <sample_id>_front.png
      <sample_id>_wrist.png
  depth/
    <task>/
      <sample_id>_front_depth.npy
      <sample_id>_wrist_depth.npy
```

## 使用数据时的 Prompt 拼接

数据集里的 user prompt 不直接写入真实本体 JSON，而是保留 `<state_info>` 占位符。真实本体数据保存在同一条样本的顶层 `state_info` 字段里，训练或评测 dataloader 在最终喂给 VLM 前再选择如何替换 `<state_info>`。

当前 user prompt 的组合逻辑：

- `messages[*].content` 中的 user prompt 先说明 front camera，然后给第一个 `<image>` 占位符。
- 接着说明 wrist camera，然后给第二个 `<image>` 占位符。
- 然后说明本体感知信息，并给 `<state_info>` 占位符。
- `<state_info>` 在数据集文件中保持原样，不在导出阶段替换。
- dataloader 可以把 `<state_info>` 替换为只包含 `state_info.base` 的 JSON，也可以替换为 `state_info.base + state_info.extra` 的 JSON。
- assistant label 不包含本体信息，只监督 `skill`、`target_point_2d`、`target_point_3d`。

数据集中保存的默认 user prompt 形式如下：

```text
This is the front camera image:
<image>
This is the wrist camera image:
<image>
This is the robot proprioceptive state information:
<state_info>
Please analyze the images and state information, then provide the current skill and target point. Return the answer in JSON format exactly like this example: {"skill": "pick", "target_point_2d": [160.0, 153.0], "target_point_3d": [0.160508, 0.000166, 0.430685]}
```

最小输入使用 `state_info.base`：

```text
This is the front camera image:
<image>
This is the wrist camera image:
<image>
This is the robot proprioceptive state information:
<state_info>
...

替换后：
This is the front camera image:
<image>
This is the wrist camera image:
<image>
This is the robot proprioceptive state information:
{"base": <state_info.base>}
...
```

如果希望模型同时看到更多关节状态，使用 `state_info.base + state_info.extra`：

```text
This is the front camera image:
<image>
This is the wrist camera image:
<image>
This is the robot proprioceptive state information:
<state_info>
...

替换后：
This is the front camera image:
<image>
This is the wrist camera image:
<image>
This is the robot proprioceptive state information:
{"base": <state_info.base>, "extra": <state_info.extra>}
...
```

默认 user prompt 还包含一个输出 JSON 示例，用来约束模型输出格式：

```json
{"skill": "pick", "target_point_2d": [160.0, 153.0], "target_point_3d": [0.160508, 0.000166, 0.430685]}
```

## `messages.jsonl` 格式

`messages.jsonl` 是整个 VLM SFT 数据集的主索引文件。每一行是一条训练样本，包含样本 id、图片路径、顶层 `state_info`、chat messages 和 metadata。训练 dataloader 通常读取这一行，加载 `images` 指向的两张图，然后把 `messages` 作为多模态对话样本喂给模型。

简单说：`messages.jsonl` 是“外层样本容器”。

```json
{
  "id": "round_table_00000_2026-05-26T18-24-23.138263_frame_00000",
  "images": [
    "images/round_table/..._front.png",
    "images/round_table/..._wrist.png"
  ],
  "state_info": {
    "base": {
      "ee_pos_sim": [0.610314, 0.099169, 0.173151],
      "ee_quat_sim": [0.856925, 0.510935, 0.052267, 0.0435],
      "ee_pos_vel": [0.0, 0.0, 0.0],
      "ee_ori_vel": [0.0, 0.0, 0.0],
      "gripper_width": [0.065]
    },
    "extra": {
      "joint_positions": [],
      "joint_velocities": [],
      "joint_torques": [],
      "parts_poses": []
    }
  },
  "messages": [
    {"role": "system", "content": "You are a vision-language robot policy assistant ..."},
    {"role": "user", "content": "This is the front camera image:\n<image>\nThis is the wrist camera image:\n<image>\nThis is the robot proprioceptive state information:\n<state_info>\nPlease analyze ... example: {\"skill\": \"pick\", ...}"},
    {"role": "assistant", "content": "{\"skill\":\"push\", \"target_point_2d\":..., \"target_point_3d\":...}"}
  ],
  "metadata": {
    "task": "round_table",
    "source_pickle": "/abs/path/to/source.pkl",
    "frame_index": 0,
    "success": true,
    "camera_map": {"front": "color_image2", "wrist": "color_image1"},
    "coordinate_frames": {
      "target_point_2d": "front-camera resized image pixel coordinates [u, v]; u increases left-to-right and v increases top-to-bottom",
      "target_point_3d": "sim_local position in meters, same frame as state_info.base.ee_pos_sim"
    },
    "depth": {
      "front": "depth/round_table/..._front_depth.npy",
      "wrist": "depth/round_table/..._wrist_depth.npy"
    }
  }
}
```

## Assistant JSON 格式

Assistant JSON 是 `messages.jsonl` 中 `messages[-1].content` 的内容，也就是模型要学习输出的 label/answer。它不是另一个单独的数据文件，而是被包在每条 `messages.jsonl` 样本里的 assistant 回答。当前设计中，本体信息不放在 assistant label 中；真实本体信息保存在顶层 `state_info`，user prompt 中只保留 `<state_info>` 占位符。

简单说：Assistant JSON 是“外层样本容器里的监督标签”。

`messages[-1].content` 是严格 JSON 字符串：

```json
{
  "skill": "push",
  "target_point_2d": [189.0, 175.0],
  "target_point_3d": [0.28161, 0.066043, 0.41625]
}
```

## `qwen_llava_sharegpt.json` 格式

同一批样本还会导出 ShareGPT/LLaVA/Qwen-VL 常用格式：

```json
{
  "id": "sample_id",
  "image": ["images/..._front.png", "images/..._wrist.png"],
  "state_info": {"base": {}, "extra": {}},
  "conversations": [
    {"from": "human", "value": "This is the front camera image:\n<image>\nThis is the wrist camera image:\n<image>\nThis is the robot proprioceptive state information:\n<state_info>\nPlease analyze ... example: {\"skill\": \"pick\", ...}"},
    {"from": "gpt", "value": "{\"skill\":\"push\", \"target_point_2d\":..., \"target_point_3d\":...}"}
  ],
  "metadata": {}
}
```

## `llamafactory_base.json` 格式

`--format llamafactory-json` 或 `to-llamafactory` 会导出 LLaMAFactory 直接可用的 ShareGPT 多模态格式。它和 `qwen_llava_sharegpt.json` 的主要区别是：

- 图片字段使用 LLaMAFactory 常用的 `images`。
- 顶层包含 `system`，保留 task-specific system prompt。
- user prompt 里的 `<state_info>` 已经按 `--llamafactory-state-mode` 替换，默认只写 `state_info.base`。

```json
{
  "id": "sample_id",
  "images": ["images/..._front.png", "images/..._wrist.png"],
  "state_info": {"base": {}, "extra": {}},
  "system": "task-specific system prompt",
  "conversations": [
    {"from": "human", "value": "This is the front camera image:\n<image>\n... {\"base\": {...}}\nPlease analyze ..."},
    {"from": "gpt", "value": "{\"skill\":\"push\", \"target_point_2d\":..., \"target_point_3d\":...}"}
  ],
  "metadata": {}
}
```

对应的 LLaMAFactory `dataset_info.json` 片段：

```json
{
  "rr_vlm_base": {
    "file_name": "llamafactory_base.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images",
      "system": "system"
    }
  }
}
```

把现有 `qwen_llava_sharegpt.json` 转成 LLaMAFactory 格式：

```bash
cd /data/hy/robust-rearrangement

python -m src.vlm_data_generator to-llamafactory \
  --input-file /data/hy/robust-rearrangement/data/processed/vlm/qwen_llava_sharegpt.json \
  --output-file /data/hy/robust-rearrangement/data/processed/vlm/llamafactory_base.json \
  --dataset-info-file /data/hy/robust-rearrangement/data/processed/vlm/llamafactory_base_dataset_info.json \
  --dataset-name rr_vlm_base \
  --llamafactory-state-mode base
```

## 注意事项

- `color_image2` 映射为 front camera，`color_image1` 映射为 wrist camera。
- `depth_image2` 映射为 front depth，`depth_image1` 映射为 wrist depth。
- 如果要让数据只包含成功轨迹，保持默认 `--demo-outcome success`。
- 多 checkpoint 追加时，第一轮用 `--output-mode overwrite`，后续用 `--output-mode append`。
- `append` 会跳过重复 `sample_id`，避免重复转换同一个 pickle 时产生重复样本。
