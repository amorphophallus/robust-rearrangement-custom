# 真机 one-leg / round-table / lamp skill annotation

`real_skill_annotation_util.py` 为 Deoxys 真机 pickle
补充 FurnitureBench/robust-rearrangement 使用的 skill、机器人基座坐标系 guidance
point/pose，以及 front/wrist 二维投影。支持 `one_leg`、`round_table`、`lamp`；
当前数采脚本按 `e` 后对最终 10 Hz observation 离线运行这一标注器。

## 离线标注 pickle

在 robust-rearrangement 仓库根目录执行：

```shell
source ~/.bashrc
conda activate rr-real
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
新任务的状态转移不会仅由陈旧的 last-known pose 触发。`round_table` 和 `lamp`
的旧 pickle 没有障碍物位姿，push 目标使用 Deoxys setup 的固定标定位姿；
`metadata.real_skill_annotation.obstacle_pose_source=configured_default` 会注明来源。
如果已经由 `recover_tabletop_pose_sam2.py` 生成 tabletop recovery JSON，可为单个
`one_leg` pickle 添加：

```shell
python -m src.eval.real_skill_annotation_util input.pkl \
  --sam2-tabletop-recovery recovery.json
```

每个 observation 会新增或更新：

- `skill`、`skill_state`、`assembly_step`
- `guidance_point`、`guidance_pose`、`guidance_gripper_width`
- `guidance_point_2d`、`grasp_annotation_2d`
- `guidance`、`real_annotation_debug`

pickle 根目录写入 `annotation_source=real_skill_annotation_util` 和
`annotation_status=annotated`，完整配置和统计位于
`metadata.real_skill_annotation`。RGB、原始 `parts_poses` 和 `parts_founds` 不会被
绘制或覆盖；SAM2 pose 也只作为标注计算的临时 overlay。
其中 `complete` 表示标注计算完成，`task_fsm_complete` 才表示所有装配对满足
FurnitureBench 几何完成条件。后者为 false 不会伪装成任务成功，也不会阻止保存。

### round_table / lamp 状态转移

- round_table：`top-leg-push → leg-top-pick → place → insert → screw`，
  几何装配完成后进入 `base-leg-pick → place → insert → screw → done`。
- lamp：`base-bulb-push → bulb-base-pick → place → insert → screw`，
  几何装配完成后进入 `hood-base-pick → place → done`。
- 两个 push 均在零件到达 FurnitureBench 目标（位置误差 L1 < 5 cm）时立即推进，
  **不等待松爪**。夹持期间 AprilTag 缺失时，使用附着时固定的 EE–零件变换推断位置。
  place/装配完成仍使用各零件原有几何阈值；lamp hood 还要求松爪，不能仅凭动作顺序完成。
- pick 连续两帧满足夹取代理条件后进入 place；insert 张爪后进入 screw。
  place 阶段确认零件掉落时回到 pick。实时和离线 FSM 分别使用独立 session，
  绝不把预览状态直接写入 pickle。

## 标注与未标注数据分层

同一批数采数据统一使用以下目录，避免仅凭文件名猜测标注状态：

```text
success/
  annotated/    # 可直接用于后续处理；根 key 为 annotation_status=annotated
  unannotated/  # 保留的原始数据；根 key 为 annotation_status=unannotated
```

离线标注时从 `unannotated/` 读取，并通过 `--output` 把结果写入
`annotated/`，不要覆盖唯一的原始数据。标注程序会自动写入 `annotated` 状态；给
原始数据分层时需要显式写入 `unannotated` 状态。现有训练数据处理器会递归发现
pickle，因此允许在 `success/` 下增加这两层目录；但处理时必须把输入目录明确
指向 `success/annotated/`，不要指向同时包含两类数据的父目录 `success/`。

## 实时逐帧接口

`RealSkillAnnotationSession` 是离线命令和实时数采共用的有状态 API。一个 session
只能对应一个按时间排序的 episode；每个实际保存的 observation 只调用一次：

```python
from src.eval.real_skill_annotation_util import RealSkillAnnotationSession

session = RealSkillAnnotationSession(
    "round_table",  # 或 one_leg / lamp
    camera_info,
    mode="online",
)

for observation in saved_observations:
    session.annotate_observation(observation)  # 原地写入 annotation 字段

payload = {
    "observations": saved_observations,
    "actions": actions,
    "metadata": {"schema": "deoxys_furniturebench_raw_v6_offline_buffered"},
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
当前数采脚本的预览标注仅用于显示；保存时会对最终时间线重新计算，
`metadata.real_skill_annotation.mode=offline`。如果中途异常，原始遥操作数据
仍可保存到 `incomplete/`，并在同名 `.txt` 记录问题，之后可用上述离线命令重新标注。

运行 Deoxys 前必须保证 robust-rearrangement 根目录在 `PYTHONPATH` 中；当前
FrankaControl 的 `~/.bashrc` 已配置，启动图形终端后仍应先执行 `source ~/.bashrc`。
