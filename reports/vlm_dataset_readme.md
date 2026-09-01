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

## 2026-08-31 Scripted Target Rotation6D 增补

本 revision 在原有 121,944 条有效样本上原位增加
`target_rotation_6d`，没有重新采集 rollout，也没有修改 RGB/depth tar。
远端索引记录的 300 个 `source_pickle` 与本地三个原 campaign 的
pickle 集合完全相等（`one_leg`、`round_table`、`lamp` 各 100 个）。

旋转来自 scripted geometry GT `guidance_pose`，不是 VLM 预测。每条原始
wire message 为了保留完整标注仍包含：

```json
{
  "skill": "pick",
  "target_point_2d": [159.0, 153.0],
  "target_point_3d": [0.160002, -0.001944, 0.430685],
  "target_rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}
```

这是数据集存储格式，不是推荐的最终模型 assistant contract；推荐的 Ver1/Ver2
输出都会在训练预处理时删除 `target_point_3d`，具体见下一节。

完整 enrichment 校验结果见 `rotation6d_enrichment_audit_20260831.json`。

## 推荐训练契约：Ver1 与 Ver2

下面的 loss 面向“共享视觉/状态 backbone + 结构化分类/回归 head”的训练方式。
如果直接使用 LLaMAFactory 做自回归文本 SFT，默认仍是对完整 assistant JSON 做
token-level cross entropy；要使用下面的分项权重，需要自定义 head 或 Trainer。

当前 revision 的数据集 wire message 仍完整保留 `target_point_3d` 标注，便于几何
审计和未来扩展；但推荐的 Ver1/Ver2 模型 assistant 输出都不包含该字段。使用者
应在训练预处理时自行裁剪 message：Ver1 同时删除 `target_point_3d` 和
`target_rotation_6d`，Ver2 只删除 `target_point_3d`。下面的 baseline loss 不监督
3D 点，也不要把米制 3D 坐标混进像素 `L_point`。数据集中的原始 3D 标注不修改。

### Ver1：skill + 2D point

模型 head 输出：

```text
skill_logits: [batch, 5]
point_2d_pred: [batch, 2]   # [u, v]，直接使用原图像素，不归一化
```

建议固定类别映射，例如
`push=0, pick=1, place=2, insert=3, screw=4`，并在训练、评估和部署侧共用
同一份映射。Ver1 assistant 输出格式：

```json
{
  "skill": "pick",
  "target_point_2d": [159.0, 153.0]
}
```

推荐损失：

```text
L_ver1 = 1.0 * L_skill + 0.02 * L_point

L_skill = CrossEntropy(skill_logits, skill_gt)

L_point =
    SmoothL1(u_pred - u_gt, beta=1.0)
  + SmoothL1(v_pred - v_gt, beta=1.0)
```

先对每条样本的两个像素维度求和，再对 batch 求平均。像素坐标不归一化时，
单轴 50 px 误差的 SmoothL1 约为 49.5，乘以 `0.02` 后约为 `0.99`，
与常见 skill CE 量级接近。SmoothL1 比 L2 不容易被少量大误差点主导。

### Ver2：skill + 2D point + Rotation6D

模型 head 输出：

```text
skill_logits:     [batch, 5]
point_2d_pred:    [batch, 2]
rotation_6d_pred: [batch, 6]
```

Ver2 assistant 输出格式：

```json
{
  "skill": "pick",
  "target_point_2d": [159.0, 153.0],
  "target_rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}
```

推荐第一版损失：

```text
L_ver2 =
    1.0 * L_skill
  + 0.02 * L_point
  + 1.0 * L_rotation

L_rotation = L_rot6d + 0.5 * L_geo

L_rot6d =
    mean_i SmoothL1(rotation_6d_pred[i] - rotation_6d_gt[i], beta=0.1)

L_geo = rotation_angle(R_pred, R_gt) / pi
```

`L_rot6d` 对六维输出提供稳定的逐维监督，避免只使用几何角度时原始 6D
向量出现尺度漂移或退化；`L_geo` 直接优化最终姿态角误差。由于解码已经将
结果投影到 `SO(3)`，不需要额外的 orthogonality loss。建议从
`lambda_rotation=1.0` 开始：若角度误差不下降可试 `2.0`；若 skill
accuracy 明显受损可降到 `0.5`。

对应的 PyTorch loss 骨架：

```python
loss_skill = F.cross_entropy(skill_logits, skill_gt)

loss_point = F.smooth_l1_loss(
    point_2d_pred,
    point_2d_gt,
    beta=1.0,
    reduction="none",
).sum(dim=-1).mean()

loss_rot6d = F.smooth_l1_loss(
    rotation_6d_pred,
    rotation_6d_gt,
    beta=0.1,
    reduction="none",
).mean(dim=-1).mean()

R_pred = rotation_6d_to_matrix_rows(rotation_6d_pred)
R_gt = rotation_6d_to_matrix_rows(rotation_6d_gt)
loss_geo = rotation_geodesic_loss(R_pred, R_gt)

loss_rotation = loss_rot6d + 0.5 * loss_geo
loss = loss_skill + 0.02 * loss_point + loss_rotation
```

### Ver2 解码 tips

本数据集保存的是 `guidance_pose[:3, :3]` 的前两**行**：

```text
[r00, r01, r02, r10, r11, r12]
```

不要直接套用按“前两列”定义的 Rotation6D 实现。按行 Gram--Schmidt 解码：

```python
def rotation_6d_to_matrix_rows(rotation_6d):
    row1_raw = rotation_6d[..., 0:3]
    row2_raw = rotation_6d[..., 3:6]

    row1 = F.normalize(row1_raw, dim=-1, eps=1e-6)
    row2_raw = row2_raw - (
        row1 * row2_raw
    ).sum(dim=-1, keepdim=True) * row1
    row2 = F.normalize(row2_raw, dim=-1, eps=1e-6)
    row3 = torch.cross(row1, row2, dim=-1)

    # dim=-2 表示三个向量是旋转矩阵的三行。
    return torch.stack([row1, row2, row3], dim=-2)
```

几何误差推荐使用 `atan2(sin(theta), cos(theta))`，避免直接 `acos`
在接近 0 或 pi 时的数值问题：

```python
def rotation_geodesic_loss(R_pred, R_gt):
    relative = R_pred @ R_gt.transpose(-1, -2)
    cos_theta = (
        relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    cos_theta = cos_theta.clamp(-1.0, 1.0)

    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = 0.5 * torch.sqrt(skew.square().sum(dim=-1) + 1e-8)
    theta = torch.atan2(sin_theta, cos_theta)
    return (theta / torch.pi).mean()
```

训练和评估时还应注意：

- Rotation6D head 最后一层可用小权重，并把 bias 初始化为单位旋转
  `[1, 0, 0, 0, 1, 0]`，避免初始两个行向量为零或近似平行。
- 训练日志分别记录未加权/加权的 `L_skill`、`L_point`、
  `L_rot6d`、`L_geo`，同时报告 skill accuracy、像素距离和
  `theta * 180 / pi` 的角度误差。
- 文本 VLM 推理时先严格解析六个有限浮点数，再执行 Gram--Schmidt；不要把
  未正交化的六维输出直接当旋转矩阵使用。
- 当前标签是连续浮点数。若后续改用 action token，需要在解码 token 后仍执行
  同样的 Gram--Schmidt，并以角度而非 6D 欧氏距离作为最终旋转指标。


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

## LLaMAFactory 原始 wire 样本格式例子

`llamafactory_base.json` 中每条样本的结构如下：

下例忠实展示磁盘中的原始 message，因此仍能看到保留的 `target_point_3d`。
使用推荐 Ver1/Ver2 contract 时，预处理器应同时从 user prompt 的输出格式要求和
assistant JSON 中删除这个字段；本 README 的展示不会修改底层标注。

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
      "value": "This is the front camera image:\n<image>\nThis is the wrist camera image:\n<image>\nThis is the robot proprioceptive state information:\n{\"base\":{\"ee_pos_sim\":[0.285126,-0.014924,0.585789],\"ee_quat_sim\":[0.934382,0.35589,0.012423,0.010877],\"ee_pos_vel\":[0.000001,-0.0,0.0],\"ee_ori_vel\":[-0.0,-0.000002,0.0],\"gripper_width\":[0.065]}}\nPlease analyze the images and state information, then provide the current skill, target point, and target orientation. Return strict JSON with skill, target_point_2d, target_point_3d, and target_rotation_6d."
    },
    {
      "from": "gpt",
      "value": "{\"skill\": \"pick\", \"target_point_2d\": [159.0, 153.0], \"target_point_3d\": [0.160002, -0.001944, 0.430685], \"target_rotation_6d\": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}"
    }
  ],
  "metadata": {}
}
```

按推荐的 Ver2 contract 调整后，训练时输入和标签是：

```json
{
  "system": "task-specific system prompt",
  "user": "front image <image> + wrist image <image> + 拼接后的 state_info + 只要求 skill、target_point_2d、target_rotation_6d",
  "images": ["front.png", "wrist.png"],
  "y": "{\"skill\": \"pick\", \"target_point_2d\": [159.0, 153.0], \"target_rotation_6d\": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}"
}
```

推荐 Ver2 的 `y` 是严格 JSON，只包含：

```text
skill
target_point_2d
target_rotation_6d
```

## 数据集规模

```text
indexed samples total: 121944
one_leg:              27477
round_table:          51729
lamp:                 42738

referenced RGB PNG files:   243888
referenced depth NPY files: 243888
dataset size:                ~85G
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
rotation6d_enrichment_audit_20260831.json
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

`target_rotation_6d`

同一个 `sim_local` 坐标系下的目标末端旋转。设 `guidance_pose` 的旋转矩阵为
`R`，字段按以下顺序保存其前两行：

```text
[r00, r01, r02, r10, r11, r12]
```

解码时先分别归一化第一行和第二行，并从第二行减去它在第一行上的投影；
第三行取前两行的叉乘。这一 Gram--Schmidt 过程会把模型输出投回合法的
`SO(3)` 旋转矩阵。数据集保存的是 6 个浮点数，未预先离散化。

### 可选的独立 Rotation6D tokenization

若使用纯 autoregressive VLM 并希望避免直接生成小数，可以在训练侧对六个维度
分别处理：

1. 将每个值裁剪到 `[-1, 1]`。
2. 每维均匀量化为 256 个 bin，并映射为专用 action token。
3. 推理时把 token 解码为 bin center。
4. 对六维结果执行上述 Gram--Schmidt，再用于机器人控制或旋转误差评估。

这是可选训练方案，不是数据集 wire format；原始浮点 Rotation6D 始终保留，便于
采用 regression head、flow/diffusion action head 或不同精度的 tokenizer。

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

# 二选一：持久登录，或仅为当前 shell 设置官方环境变量。
modelscope login
# export MODELSCOPE_API_TOKEN="<your_token>"
export REPO_ID="huyue233/furniture-bench-2d-point-annotation"
export MODELSCOPE_BYPASS_PROXY=1
export MAX_WORKERS=1

scripts/upload_vlm_to_modelscope.sh upload-index-update \
  --upload-dir /data/hy/robust-rearrangement/data/processed/vlm_rotation6d_update
```

这次是原位索引更新；上传目录不含媒体 tar，原仓库中已有 tar 保持不变。上传布局：

当前节点直连 `www.modelscope.cn` 和 ModelScope LFS endpoint 比环境代理稳定，因此
上述命令显式绕过代理并串行上传。脚本显式启用 ModelScope 上传缓存；它能在重跑时
跳过已成功文件，但 ModelScope 当前的大文件 PUT 仍是整文件请求，不能从单文件中间
的字节位置续传。

```text
vlm_modelscope_upload/
  README.md
  manifest.json
  messages.jsonl
  qwen_llava_sharegpt.json
  llamafactory_base.json
  llamafactory_base_dataset_info.json
  rotation6d_enrichment_audit_20260831.json
  preview/llamafactory_preview.jsonl
```
