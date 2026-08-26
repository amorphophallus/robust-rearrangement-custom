# 真机 one-leg skill annotation

`real_skill_annotation_util.py` 为 `deoxys_furniturebench_raw_v2` 的真机 pickle
补充 FurnitureBench/robust-rearrangement 使用的 skill、机器人基座坐标系 guidance
point/pose，以及 front/wrist 二维投影。目前只支持 `one_leg`。

## 离线标注 pickle

在 robust-rearrangement 仓库根目录执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/robust-rearrangement-custom

python -m src.eval.real_skill_annotation_util \
  "$DATA_DIR_RAW/raw/osc/real/one_leg/teleop/low/success/示例.pkl"
```

默认保留原文件，并在同目录生成 `示例.annotated.pkl`。常用形式：

```shell
# 指定单个输出文件
python -m src.eval.real_skill_annotation_util input.pkl \
  --output output.annotated.pkl

# 原子覆盖输入文件；只有明确需要原地更新时才使用
python -m src.eval.real_skill_annotation_util input.pkl --overwrite

# 批量处理目录中的未标注 pickle
python -m src.eval.real_skill_annotation_util /path/to/success/

# 同时生成带 guidance point 和 skill 文本的检查视频
python -m src.eval.real_skill_annotation_util input.pkl \
  --video-output input.annotation.mp4 \
  --video-fps 10
```

默认位姿策略依次使用当前帧 AprilTag、夹取后的 EE 刚体传播和 last-known pose。
如果已经由 `recover_tabletop_pose_sam2.py` 生成 tabletop recovery JSON，可为单个
pickle 添加：

```shell
python -m src.eval.real_skill_annotation_util input.pkl \
  --sam2-tabletop-recovery recovery.json
```

每个 observation 会新增或更新：

- `skill`、`skill_state`、`assembly_step`
- `guidance_point`、`guidance_pose`、`guidance_gripper_width`
- `guidance_point_2d`、`grasp_annotation_2d`
- `guidance`、`real_annotation_debug`

pickle 根目录写入 `annotation_source=real_skill_annotation_util`，完整配置和统计位于
`metadata.real_skill_annotation`。RGB、原始 `parts_poses` 和 `parts_founds` 不会被
绘制或覆盖；SAM2 pose 也只作为标注计算的临时 overlay。

## 实时逐帧接口

`RealSkillAnnotationSession` 是离线命令和实时数采共用的有状态 API。一个 session
只能对应一个按时间排序的 episode；每个实际保存的 observation 只调用一次：

```python
from src.eval.real_skill_annotation_util import RealSkillAnnotationSession

session = RealSkillAnnotationSession(
    "one_leg",
    camera_info,
    mode="online",
)

for observation in saved_observations:
    session.annotate_observation(observation)  # 原地写入 annotation 字段

payload = {
    "observations": saved_observations,
    "actions": actions,
    "metadata": {"schema": "deoxys_furniturebench_raw_v2"},
}
session.update_trajectory_metadata(payload)
```

不要用保存 episode 的 session 处理预览中未保存的中间帧，否则其状态转换和统计将
无法由最终 pickle 复现。Deoxys 数采脚本已经为预览和保存分别维护 session，并按
`camera_capture_wall_time_ns` 避免对同一相机帧重复推进。

## 在 Deoxys 数采中实时使用

FrankaControl 上执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg deoxys/config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --task-name one_leg \
  --draw-part-poses \
  --real-skill-annotation
```

front 预览显示紫色 guidance point 和 `skill/skill_state`，但保存的 RGB 保持原样。
pickle 中保存完整在线 annotation，且
`metadata.real_skill_annotation.mode=online`、`complete=true`。如果中途异常，原始
遥操作数据仍可保存，`complete=false`，随后用上面的离线命令重新标注。

运行 Deoxys 前必须保证 robust-rearrangement 根目录在 `PYTHONPATH` 中；当前
FrankaControl 的 `~/.bashrc` 已配置，启动图形终端后仍应先执行 `source ~/.bashrc`。
