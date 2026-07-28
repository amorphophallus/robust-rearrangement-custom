---
license: apache-2.0
task: visual-question-answering
task_categories:
- visual-question-answering
language:
- en
tags:
- vision-language
- robotics
- imitation-learning
- furniture-assembly
- llamafactory
size_categories:
- 100K<n<1M
configs:
- config_name: preview
  data_files:
  - split: train
    path: preview/llamafactory_preview.jsonl
  default: true
- config_name: llamafactory_base
  data_files:
  - split: train
    path: llamafactory_base.json
---

# Robust Rearrangement VLM SFT 数据集说明

## 使用方法

推荐直接使用 `llamafactory_base.json` 训练 LLaMAFactory。这个文件已经是 LLaMAFactory ShareGPT 多模态格式，并且已经把 user prompt 里的 `<state_info>` 占位符替换成 `state_info.base`。

数据集根目录：

```bash
/data/hy/robust-rearrangement/data/processed/vlm
```

ModelScope 页面默认预览使用 README YAML 中的 `preview` config，指向 `preview/llamafactory_preview.jsonl`，只包含前 100 条样本，用于触发网页预览和快速检查字段。完整训练数据仍在 `llamafactory_base.json`，对应 `llamafactory_base` config。完整 RGB 图像和 depth 文件以 tar 包形式分发，下载后需要解压到数据集根目录再训练；tar 包不作为 ModelScope preview 的 `data_files`，避免平台把大二进制压缩包当成可解析样本文件。若需要网页预览直接显示图片，需要额外上传展开后的图片文件。

LLaMAFactory 数据文件：

```text
llamafactory_base.json
llamafactory_base_dataset_info.json
```

把 `llamafactory_base_dataset_info.json` 里的内容加入 LLaMAFactory 的 `dataset_info.json`：

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

### 必须拼接 `state_info`

这个数据集的原始 user prompt 里有一个 `<state_info>` 占位符。使用数据时一定要把它替换成本体感知信息，否则模型只能看到字面量 `<state_info>`，看不到机器人状态。

有两个推荐选项：

```json
{"base": state_info.base}
```

或者：

```json
{"base": state_info.base, "extra": state_info.extra}
```

当前已经生成好的 `llamafactory_base.json` 使用第一种，也就是只拼接 `base`。如果要使用 `base + extra`，重新生成：

```bash
cd /data/hy/robust-rearrangement

python -m src.vlm_data_generator to-llamafactory \
  --input-file /data/hy/robust-rearrangement/data/processed/vlm/qwen_llava_sharegpt.json \
  --output-file /data/hy/robust-rearrangement/data/processed/vlm/llamafactory_base_extra.json \
  --dataset-info-file /data/hy/robust-rearrangement/data/processed/vlm/llamafactory_base_extra_dataset_info.json \
  --dataset-name rr_vlm_base_extra \
  --llamafactory-state-mode base-extra
```

## LLaMAFactory 样本格式例子

`llamafactory_base.json` 中每条样本的结构如下：

```json
{
  "id": "one_leg_00000_2026-07-20T21-37-47.683643_frame_00000",
  "images": [
    "images/one_leg/one_leg_00000_2026-07-20T21-37-47.683643_frame_00000_front.png",
    "images/one_leg/one_leg_00000_2026-07-20T21-37-47.683643_frame_00000_wrist.png"
  ],
  "state_info": {
    "base": {
      "ee_pos_sim": [0.285126, -0.014924, 0.585789],
      "ee_quat_sim": [0.934382, 0.35589, 0.012423, 0.010877],
      "ee_pos_vel": [0.000001, -0.0, 0.0],
      "ee_ori_vel": [-0.0, -0.000002, 0.0],
      "gripper_width": [0.065]
    },
    "extra": {
      "joint_positions": [],
      "joint_velocities": [],
      "joint_torques": [],
      "parts_poses": []
    }
  },
  "system": "task-specific system prompt",
  "conversations": [
    {
      "from": "human",
      "value": "This is the front camera image:\n<image>\nThis is the wrist camera image:\n<image>\nThis is the robot proprioceptive state information:\n{\"base\":{\"ee_pos_sim\":[0.285126,-0.014924,0.585789],\"ee_quat_sim\":[0.934382,0.35589,0.012423,0.010877],\"ee_pos_vel\":[0.000001,-0.0,0.0],\"ee_ori_vel\":[-0.0,-0.000002,0.0],\"gripper_width\":[0.065]}}\nPlease analyze the images and state information, then provide the current skill and target point. Return the answer in JSON format exactly like this example: {\"skill\": \"pick\", \"target_point_2d\": [160.0, 153.0], \"target_point_3d\": [0.160508, 0.000166, 0.430685]}"
    },
    {
      "from": "gpt",
      "value": "{\"skill\": \"pick\", \"target_point_2d\": [159.0, 153.0], \"target_point_3d\": [0.160002, -0.001944, 0.430685]}"
    }
  ],
  "metadata": {}
}
```

概念上，训练时输入和标签是：

```json
{
  "system": "task-specific system prompt",
  "user": "front image <image> + wrist image <image> + 拼接后的 state_info + 输出格式要求",
  "images": ["front.png", "wrist.png"],
  "y": "{\"skill\": \"pick\", \"target_point_2d\": [159.0, 153.0], \"target_point_3d\": [0.160002, -0.001944, 0.430685]}"
}
```

`y` 是 assistant label，必须是严格 JSON，只包含：

```text
skill
target_point_2d
target_point_3d
```

## 数据集规模

```text
samples total: 130910
one_leg:       30494
round_table:   54910
lamp:          45506

RGB PNG files:   261820
depth NPY files: 261820
dataset size:    ~85G
```

每条样本包含两张 RGB 图片：

```text
front camera: color_image2
wrist camera: color_image1
```

depth 以 `.npy` 保存，并在 `metadata.depth` 里记录路径。

## 文件说明

```text
README.md
manifest.json
messages.jsonl
qwen_llava_sharegpt.json
llamafactory_base.json
llamafactory_base_dataset_info.json
images/
  one_leg/
  round_table/
  lamp/
depth/
  one_leg/
  round_table/
  lamp/
```

`messages.jsonl`

内部完整格式。每行包含 `id`、`images`、顶层 `state_info`、OpenAI-style `messages` 和 `metadata`。其中 user prompt 保留 `<state_info>` 占位符。

`qwen_llava_sharegpt.json`

ShareGPT/LLaVA/Qwen-VL 风格格式。字段是 `image`，不是 `images`；user prompt 仍保留 `<state_info>` 占位符。

`llamafactory_base.json`

推荐训练文件。字段是 LLaMAFactory 常用的 `images`，并且已经将 `<state_info>` 替换为 `{"base": state_info.base}`。

## 标注语义

`target_point_2d`

front camera 图像像素坐标 `[u, v]`。`u` 从左到右增加，`v` 从上到下增加。没有 wrist target point。

`target_point_3d`

`sim_local` 坐标系下的 3D 目标点，单位米，和 `state_info.base.ee_pos_sim` 使用同一个坐标系。

`state_info.base`

```text
ee_pos_sim
ee_quat_sim
ee_pos_vel
ee_ori_vel
gripper_width
```

`state_info.extra`

```text
joint_positions
joint_velocities
joint_torques
parts_poses
```

## ModelScope 上传

不要直接上传展开后的 `images/` 和 `depth/` 目录。ModelScope 官方上传文档说明了这些限制：单文件不超过 50 GB，总文件数不超过 100,000，单个子目录文件数不超过 10,000，非 LFS 文件总大小建议不超过 500 MB。当前展开后有超过 52 万个 image/depth 文件，因此上传脚本会先按 task 打包成 tar。

上传脚本在仓库的 `scripts/` 目录：

```bash
cd /data/hy/robust-rearrangement

export MODELSCOPE_TOKEN="<your_token>"
export REPO_ID="<your_namespace>/<your_dataset_name>"

scripts/upload_vlm_to_modelscope.sh \
  --local-dir /data/hy/robust-rearrangement/data/processed/vlm
```

默认行为是：

```text
prepare: 生成 /data/hy/robust-rearrangement/data/processed/vlm_modelscope_upload
upload:  调用 modelscope upload --repo-type dataset
```

只打包不上传：

```bash
scripts/upload_vlm_to_modelscope.sh prepare
```

只上传已有 staging 目录：

```bash
scripts/upload_vlm_to_modelscope.sh upload
```

上传布局：

```text
vlm_modelscope_upload/
  README.md
  manifest.json
  messages.jsonl
  qwen_llava_sharegpt.json
  llamafactory_base.json
  llamafactory_base_dataset_info.json
  images_one_leg.tar
  images_round_table.tar
  images_lamp.tar
  depth_one_leg.tar
  depth_round_table.tar
  depth_lamp.tar
```

下载后恢复展开目录：

```bash
cd /path/to/downloaded/dataset
for f in images_*.tar depth_*.tar; do
  [ -f "$f" ] && tar -xf "$f"
done
```
