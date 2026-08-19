# HY Furniture VLM 服务端部署手册（4090 + NAS Conda）

本文档对应当前实际部署环境：

- 推理服务器：`zju_4090_240`；
- 推理 GPU：物理 GPU0，NVIDIA RTX 4090 24 GB；
- NVIDIA Driver：`550.120`；
- Python 环境：NAS 上独立 Conda prefix；
- 服务框架：Transformers + PyTorch + FastAPI，单进程、单 GPU；
- 客户端：当前 3060 机器上的 `evaluate_model.py`；
- 协议：内网 HTTP + Bearer Token。

下面每一步都可以直接复制执行。没有特别说明时，“服务器命令”均在
`zju_4090_240` 上执行，“本地命令”均在运行 diffusion policy 的 3060 机器执行。

## 1. 最终架构

```text
3060 本地机器
  evaluate_model.py
    ├─ 本地自动机：只生成 shadow GT，不参与 VLM 模式下的 policy 控制
    ├─ VLMGuidanceClient
    └─ diffusion policy
             │
             │ HTTP multipart，front/wrist RGB + state_info.base
             ▼
zju_4090_240:8000
  FastAPI /v1/guidance/predict
    └─ Qwen3_5ForConditionalGeneration（original_sft）
         └─ greedy generate assistant JSON
              ├─ skill
              └─ target_point_2d（raw 320×240 pixel）
```

VLM 返回：

- `skill`：`push/pick/place/insert/screw`；
- `point_1000`：Qwen `[0,1000]` 坐标；
- `point_px`：front camera 的 `320x240` 像素坐标；
- 原始生成文本、模型 revision 和服务端耗时。

`original_sft` 没有独立的 skill classification head，因此 `skill_confidence` 和
`skill_probabilities` 为 `null`，客户端不得伪造概率。

## 2. 为什么不直接使用 vLLM

当前 checkpoint 是标准原生 Qwen text-generation checkpoint，理论上已经具备评估
vLLM 的前提；旧版“自定义 point head 无法由 vLLM 执行”的限制不再适用。

本轮仍先使用 Transformers，是为了逐项复现 GitHub commit `7430c97b...` 中
`visualize_inference.sh` 的参考路径：`AutoModelForImageTextToText`、greedy generation、
相同 chat template 和 JSON parser。在离线 point/skill 指标确认前切换推理后端，会同时
改变 checkpoint 与 runtime，无法判断 regress-to-mean 是否真的修复。vLLM 可以在
Transformers 基线通过后作为单独的吞吐/一致性实验。

参考 visualizer 每次只推理一条样本；服务端支持 batch，因此在 `original_sft` 模式显式
设置 tokenizer `padding_side=left`。decoder-only generation 使用 right padding 会从 pad
token 后继续生成，可能得到空或损坏的 JSON。

## 3. 本次部署使用的固定路径和版本

登录服务器：

```bash
ssh zju_4090_240
```

设置本次部署变量：

```bash
export RR_REPO=/mnt/nas/share/home/hy/robust-rearrangement-custom
export VLM_ENV=/mnt/nas/share/home/hy/miniconda3/envs/rr-vlm-guidance-runtime
export VLM_ROOT=/mnt/nas/share/home/hy/vlm-guidance
export VLM_CHECKPOINT_DIR="$VLM_ROOT/original_sft/ckpts"
export VLM_MANIFEST_PATH="$VLM_ROOT/manifest.original_sft.json"
export VLM_SERVER_ENV_FILE="$VLM_ROOT/server.env"
```

固定 revision：

```text
Superviro/hy_furniture inference code:
7430c97b2861a6f9da5b7487a501f5745c573555

zhouhangzhu/hy_furniture_weights original_sft:
75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e
```

服务器驱动 `550.120` 不适合 CUDA 12.8，所以本机实际使用：

```text
Python 3.11
PyTorch 2.5.1 + CUDA 12.4
torchvision 0.20.1 + CUDA 12.4
transformers 5.5.4
attention backend: sdpa
```

Transformers 5.x 支持 PyTorch 2.4+。checkpoint 作者提供的 Conv3d workaround 只会在
PyTorch 2.9 上启用；PyTorch 2.5.1 使用原生 Conv3d，不走该 patch。

## 4. 同步服务代码

仓库已经存在时：

```bash
cd "$RR_REPO"
git checkout main
git pull --ff-only origin main
```

第一次部署时：

```bash
mkdir -p /mnt/nas/share/home/hy
git clone git@github.com:amorphophallus/robust-rearrangement-custom.git "$RR_REPO"
cd "$RR_REPO"
git checkout main
```

确认服务文件齐全：

```bash
cd "$RR_REPO"
test -f services/vlm_guidance/app.py
test -f services/vlm_guidance/engine.py
test -f services/vlm_guidance/modeling.py
test -x services/vlm_guidance/conda_server.sh
```

四条命令都没有输出即表示通过。

## 5. 创建 NAS Conda 环境

不要修改已有 `rr` 或 `track` 环境。本次已经准备好的独立 prefix 是
`$VLM_ENV`；下面的创建命令只在该目录尚不存在时执行：

```bash
/mnt/nas/share/home/hy/miniconda3/bin/conda create -y \
  --override-channels \
  -c conda-forge \
  -p "$VLM_ENV" \
  python=3.11 \
  pip
```

这里显式使用 `conda-forge`，不会要求代替用户接受 Anaconda defaults channel 的 ToS，
也不会改写全局 Conda channel 配置。

确认 Python：

```bash
"$VLM_ENV/bin/python" --version
```

预期是 Python 3.11.x。

### 5.1 安装与驱动兼容的 PyTorch

NAS ACL 在部分服务器上可能让 `os.access()` 误报不可写，pip 随后会打印
`Defaulting to user installation` 并污染 `~/.local`。因此下面始终显式设置
`--prefix` 和 `PYTHONNOUSERSITE=1`：

```bash
env PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$VLM_ENV/bin/python" -m pip install \
  --prefix="$VLM_ENV" \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

安装服务依赖和 ModelScope 下载工具：

```bash
env PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$VLM_ENV/bin/python" -m pip install \
  --prefix="$VLM_ENV" \
  -r "$RR_REPO/services/vlm_guidance/requirements.txt" \
  modelscope-hub
```

检查所有包确实来自该环境：

```bash
env PYTHONNOUSERSITE=1 "$VLM_ENV/bin/python" - <<'PY'
import fastapi
import torch
import torchvision
import transformers

print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("fastapi:", fastapi.__version__)
print("torch file:", torch.__file__)
PY
```

`torch file` 必须位于 `$VLM_ENV` 下，不能位于 `/home/hy/.local`。

## 6. 下载 original_sft checkpoint

创建模型目录：

```bash
mkdir -p "$VLM_ROOT"
```

该仓库包含完整 Qwen 权重和 processor，不再需要另外下载 base model：

```bash
git clone https://www.modelscope.cn/zhouhangzhu/hy_furniture_weights.git \
  "$VLM_ROOT/original_sft"

git -c safe.directory="$VLM_ROOT/original_sft" \
  -C "$VLM_ROOT/original_sft" rev-parse HEAD
```

检查模型文件：

```bash
test -f "$VLM_CHECKPOINT_DIR/config.json"
test -f "$VLM_CHECKPOINT_DIR/model.safetensors"
test -f "$VLM_CHECKPOINT_DIR/processor_config.json"
test -f "$VLM_CHECKPOINT_DIR/tokenizer.json"
```

revision 必须是 `75dc7b8a...`。`config.json` 的 architecture 必须是
`Qwen3_5ForConditionalGeneration`，且不应包含 `hy_furniture_policy`。

## 7. 生成并校验 manifest

manifest 会固定文件大小和 SHA256，防止误用旧输出头或被修改的权重：

```bash
cd "$RR_REPO"
env PYTHONNOUSERSITE=1 PYTHONPATH="$RR_REPO" \
  "$VLM_ENV/bin/python" -m services.vlm_guidance.prepare_manifest \
  --checkpoint-dir "$VLM_CHECKPOINT_DIR" \
  --model-mode original_sft \
  --checkpoint-revision 75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e \
  --output "$VLM_MANIFEST_PATH"

test -s "$VLM_MANIFEST_PATH"
```

计算大权重文件的 SHA256 需要几分钟。如果 architecture 或 model mode 校验失败，必须
停止部署，不能 fallback 到旧 point head。

## 8. 创建服务配置和 Token

```bash
mkdir -p "$VLM_ROOT"
umask 077
VLM_NEW_TOKEN="$(openssl rand -hex 32)"

{
  printf '%s\n' \
    "VLM_CHECKPOINT_DIR=$VLM_CHECKPOINT_DIR" \
    "VLM_MANIFEST_PATH=$VLM_MANIFEST_PATH" \
    'VLM_MODEL_MODE=original_sft' \
    'VLM_MODEL_REVISION=75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e' \
    'VLM_DEVICE=cuda:0' \
    'VLM_ATTENTION_BACKEND=sdpa' \
    'VLM_MAX_LENGTH=4096' \
    'VLM_IMAGE_MAX_PIXELS=262144' \
    'VLM_MAX_MICRO_BATCH_SIZE=4' \
    'VLM_MAX_NEW_TOKENS=256' \
    "VLM_API_TOKEN=$VLM_NEW_TOKEN"
} > "$VLM_SERVER_ENV_FILE"

chmod 600 "$VLM_SERVER_ENV_FILE"
unset VLM_NEW_TOKEN
```

不要提交 `server.env`，不要把 Token 复制到公开日志或 W&B config。

## 9. 释放占卡程序，然后启动服务

模型加载前必须显式释放 GPU0，不能只依赖自动检测，以免大模型首次分配显存时发生
竞态 OOM。

在 3060 本地机器执行：

```bash
cd /data/hy/gpu-snatcher
./reserve_gpu.sh release --host zju_4090_240 --gpu 0
./reserve_gpu.sh status --host zju_4090_240 --gpu 0
```

确认 `process_alive=false` 且 GPU 使用率低于 10%。

回到 `zju_4090_240` 启动服务：

```bash
cd "$RR_REPO"
export VLM_CONDA_ENV="$VLM_ENV"
export VLM_SERVER_ENV_FILE="$VLM_SERVER_ENV_FILE"
export VLM_GPU_ID=0

./services/vlm_guidance/conda_server.sh start
```

脚本会再次检查：

- GPU0 是否存在；
- 当前显存占用是否小于 10%；
- 是否已有任意 CUDA compute PID；
- 8000 端口是否空闲；
- Conda Python 和 `server.env` 是否存在；
- 是否已有同一服务 PID。

任一检查失败都会直接退出，不会抢占或终止别人的进程。

查看模型加载日志：

```bash
./services/vlm_guidance/conda_server.sh logs
```

模型启动依次执行 manifest SHA256 校验、原生 checkpoint 加载和一次 greedy-generation
warmup，可能需要几分钟。

## 10. 检查 readiness

仍在服务器执行：

```bash
cd "$RR_REPO"
export VLM_CONDA_ENV="$VLM_ENV"
export VLM_SERVER_ENV_FILE="$VLM_SERVER_ENV_FILE"
export VLM_GPU_ID=0

./services/vlm_guidance/conda_server.sh status
```

成功时最后一行类似：

```json
{"status":"ready","model_revision":"75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e","policy_version":3,"model_mode":"original_sft","device":"cuda:0","attention_backend":"sdpa"}
```

只要不是 `status=ready`，就不要启动本地 rollout。

## 11. 完整 HTTP smoke test

生成两张接口尺寸正确的黑图：

```bash
env PYTHONNOUSERSITE=1 "$VLM_ENV/bin/python" - <<'PY'
from PIL import Image

Image.new("RGB", (320, 240), "black").save("/tmp/vlm-front.png")
Image.new("RGB", (320, 240), "black").save("/tmp/vlm-wrist.png")
PY
```

生成 metadata：

```bash
printf '%s' \
  '{"task":"one_leg","items":[{"request_id":"smoke-0","state_info":{"base":{"ee_pos_sim":[0.0,0.0,0.0],"ee_quat_sim":[0.0,0.0,0.0,1.0],"ee_pos_vel":[0.0,0.0,0.0],"ee_ori_vel":[0.0,0.0,0.0],"gripper_width":0.0}}}]}' \
  > /tmp/vlm-metadata.json
```

加载 Token 并请求：

```bash
set -a
source "$VLM_SERVER_ENV_FILE"
set +a

curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  -F 'metadata=</tmp/vlm-metadata.json;type=application/json' \
  -F 'front_0=@/tmp/vlm-front.png;type=image/png' \
  -F 'wrist_0=@/tmp/vlm-wrist.png;type=image/png' \
  http://127.0.0.1:8000/v1/guidance/predict
```

黑图预测没有业务意义；这里只验证双图预处理、greedy generation、JSON parsing 和 HTTP schema。

## 12. 从 3060 本地机器访问

服务器地址是 `10.71.106.240`。在 3060 本地机器执行：

```bash
export VLM_GUIDANCE_URL=http://10.71.106.240:8000
export VLM_API_TOKEN="$(
  sed -n 's/^VLM_API_TOKEN=//p' \
    /mnt/nas/share/home/hy/vlm-guidance/server.env
)"
test -n "$VLM_API_TOKEN"

curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  "$VLM_GUIDANCE_URL/health/ready"
```

如果服务器本机 readiness 正常而本地失败，检查服务器防火墙或机房 ACL 是否允许 3060
机器访问 TCP 8000。当前是内网 HTTP；不要把 8000 直接暴露到公网。跨不可信网络应使用
SSH tunnel、VPN 或 HTTPS reverse proxy。

## 13. 运行 VLM + diffusion policy

本地评测环境只需要原有依赖、`requests` 和 `Pillow`，不需要安装服务端 Transformers。
Isaac Gym 还需要显式找到 rr 环境里的 `libpython3.8.so.1.0`：

```bash
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

### 13.1 真实 VLM + DiT policy 正式评测命令

所有 scripted/VLM eval 必须由最新版 `gpu-snatcher/auto_eval.sh` 启动，不要手工调用
`python -m src.eval.evaluate_model`。`auto_eval.sh` 会统一添加 RGBD 所需的
`--save-depth-image`、rollout 保存参数和 observation/action 配置。

下面是 `rgbd+GP` 在 `one_leg` 上的完整 36-rollout 命令：

```bash
cd /data/hy/robust-rearrangement

export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLM_GUIDANCE_URL=http://10.71.106.240:8000
export VLM_API_TOKEN="$(
  sed -n 's/^VLM_API_TOKEN=//p' \
    /mnt/nas/share/home/hy/vlm-guidance/server.env
)"
test -n "$VLM_API_TOKEN"

mkdir -p logs/vlm_dit_single/summaries

/data/hy/gpu-snatcher/auto_eval.sh \
  --steps eval \
  --local-path /data/hy/robust-rearrangement \
  --overwrite-wt-path /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt \
  --task one_leg \
  --n-envs 3 \
  --n-rollouts 36 \
  --randomness low \
  --max-rollout-steps 1000 \
  --annotation-source vlm \
  --tracking-metric-type pose \
  --vlm-base-url "$VLM_GUIDANCE_URL" \
  --vlm-timeout-seconds 30 \
  --vlm-query-interval 0 \
  --vlm-noise-projection-samples 200 \
  --task-summary-out /data/hy/robust-rearrangement/logs/vlm_dit_single/summaries/rgbd_gp__one_leg.json \
  --rollout-suffix-model-name vlm_dit_single/rgbd_gp/one_leg \
  --guidance-point-on-image
```

关键参数的实际含义：

- `--n-envs 3 --n-rollouts 36`：每轮并行 3 个仿真环境，共收集 36 个 episode；
- `--task one_leg`：当前评测任务。三个正式任务为 `one_leg`、`round_table`、`lamp`；
- `--max-rollout-steps 1000`：one_leg 的 episode 上限，与项目默认
  `task_timeout(one_leg)` 以及噪声基线保持一致；round_table 和 lamp 也使用 1000；
- `--randomness low`：使用与 clean-train/noise 实验相同的 low-randomness 仿真设置；
- `--overwrite-wt-path`：强制使用给定的本地 DiT checkpoint，不从 W&B 下载；
- `--tracking-metric-type pose`：强制输出 position/orientation/total tracking，便于生成
  P/O/T 表；
- `--annotation-source vlm`：policy 的 skill 和 2D point 都来自远端 VLM；自动机只作为
  shadow GT，不控制 policy，也不会在 VLM 失败时静默 fallback；
- `--vlm-base-url`：远端 FastAPI 服务地址；
- `--vlm-timeout-seconds 30`：单次 HTTP batch 请求最多等待 30 秒；
- `--vlm-query-interval 0`：跟随 checkpoint 的 `actor.action_horizon`。这三个 checkpoint
  都是 8，所以每 8 个 environment step 重新 query 一次，其间复用缓存；
- `--vlm-noise-projection-samples 200`：每个有效控制 step 的 GT/VLM 点对、每个 n0--n4 档位投影
  200 个 3D Monte Carlo 偏移样本，用于 3D 噪声与 2D VLM error 的同坐标系比较；
- `--task-summary-out`：保存 success、tracking、step-average point error、per-skill point
  error 和 projected-noise distribution 的完整 JSON。

三个 condition 只需要替换 checkpoint：

```text
rgbd+GP:
/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt

rgbd+colored GP:
/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-18/14-59-28.908152/models/absurd-voice-2_2026-06-18_14-59-48.700671/actor_chkpt_latest_3000.pt

rgbd+GP+skill:
/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/12-55-43.621615/models/fresh-tree-11_2026-06-13_12-56-10.422936/actor_chkpt_latest_3000.pt
```

完整的 3 condition × 3 task × 36 rollout 矩阵由可续跑启动器执行：

```bash
python scripts/run_vlm_dit_eval.py
```

启动器生成 9 个 task-level cell，并让每个 cell 经过同一个 `auto_eval.sh` 入口；正式运行前
先使用 `--print-command` 检查展开后的最终命令。

### 13.2 16-step 链路 smoke

链路 smoke 也使用同一个入口，只缩小 rollout 数和最大步数：

```bash
/data/hy/gpu-snatcher/auto_eval.sh \
  --steps eval \
  --local-path /data/hy/robust-rearrangement \
  --overwrite-wt-path /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt \
  --task one_leg \
  --n-envs 1 \
  --n-rollouts 1 \
  --max-rollout-steps 16 \
  --randomness low \
  --annotation-source vlm \
  --tracking-metric-type pose \
  --vlm-base-url "$VLM_GUIDANCE_URL" \
  --vlm-timeout-seconds 30 \
  --vlm-query-interval 0 \
  --vlm-noise-projection-samples 200 \
  --task-summary-out /data/hy/robust-rearrangement/logs/vlm_dit_single/summaries/smoke.json \
  --rollout-suffix-model-name vlm_dit_single/smoke \
  --guidance-point-on-image
```

16 step 通常不足以完成装配，因此这个命令的成功率没有实验意义；它用于检查仿真、
远端 VLM、DP 输入和误差统计链路。正式实验应改回 task 对应的完整
`--max-rollout-steps`，并增加 `--n-rollouts`。

参数说明：

- `--vlm-query-interval 0`：使用 `actor.action_horizon` 作为查询间隔；
- 正整数：每隔指定 environment steps 查询一次，中间复用缓存；
- 多环境 batch 或 NAS/网络抖动时，timeout 建议先用 30 秒；
- API、policy version、revision 或响应 schema 不匹配时会停止 rollout，不会静默回退自动机。

VLM 模式下：

- policy 使用 VLM 的 `skill + point_px`；
- 自动机同步运行，只产生 oracle/shadow GT；
- 每个实际控制 step 计算 VLM point 与 oracle point 的欧氏像素距离；
- 同一 VLM query 在 action horizon 内复用时，每个控制 step 都会计入 step-average；
- 输出 overall 和按 oracle skill 分组的 mean/RMSE/coverage；
- 同时输出 success-only 与 failure-only 统计；
- 每个有效控制 step 都在当步 3D GT point 周围生成 200 个 clipped-standard-Gaussian
  Monte Carlo 样本，并把 n0--n4 缩放后的点通过同一 front camera 投到 2D；
- projected-noise 的 mean/covariance/RMSE 使用全部 Monte Carlo 样本精确累计；SWD、W1、
  KS 和分位数使用有界 reference reservoir，避免 summary JSON 随 rollout 数量失控；
- rollout pickle 保存逐 step 的 VLM point、oracle point、误差、query step 和 cache age。

聚合 JSON/W&B 中的主字段是：

```text
vlm_point_error/all/overall/mean_error_px
vlm_point_error/all/overall/rmse_px
vlm_point_error/all/by_skill/<skill>/mean_error_px
vlm_point_error/all/step_distribution/projection_reference/...
vlm_point_error/all/step_distribution/noise_levels/<n0-n4>/projected/...
vlm_point_error/all/step_distribution/magnitude_equivalent_bracket
vlm_point_error/all/step_distribution/magnitude_equivalent_std_mm
vlm_point_error/success_only/...
vlm_point_error/failure_only/...
```

这里的 `mean_error_px` 就是要求的 step-average 打点误差；
`by_skill/<skill>/mean_error_px` 是 each-skill step-average 打点误差。

## 14. 日常运维

在 `zju_4090_240`：

```bash
cd /mnt/nas/share/home/hy/robust-rearrangement-custom

export VLM_CONDA_ENV=/mnt/nas/share/home/hy/miniconda3/envs/rr-vlm-guidance-runtime
export VLM_SERVER_ENV_FILE=/mnt/nas/share/home/hy/vlm-guidance/server.env
export VLM_GPU_ID=0

# 状态 + readiness
./services/vlm_guidance/conda_server.sh status

# 最近 200 行日志
./services/vlm_guidance/conda_server.sh logs

# 持续查看日志
tail -f /home/hy/.cache/rr-vlm-guidance/server.log

# 安全停止；只发送 SIGTERM，不发送 SIGKILL
./services/vlm_guidance/conda_server.sh stop

# 重启前仍会重新执行 GPU <10% 和 compute PID 检查
./services/vlm_guidance/conda_server.sh restart
```

服务必须保持一个 Uvicorn worker。不要增加 `--workers`，否则每个 worker 都会复制模型。

## 15. 常见故障

### pip 打印 `Defaulting to user installation`

说明没有使用本手册的显式 prefix。中止安装并重新执行带有下面两项的命令：

```text
PYTHONNOUSERSITE=1
--prefix="$VLM_ENV"
```

### `GPU is not idle enough` 或 `already has compute PID`

服务脚本不会抢卡。先执行 `nvidia-smi`，确认目标进程；如果是占卡 worker，从 3060
本地机器执行 `reserve_gpu.sh release`，不要手工 `kill -9`。

### `CUDA was requested but is unavailable`

检查：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "$VLM_ENV/bin/python" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY
```

本机应显示 PyTorch `2.5.1`、CUDA runtime `12.4`、CUDA 可用和 RTX 4090。

### `Qwen3VLVideoProcessor requires the Torchvision library`

Qwen3.5 processor 初始化时也会注册视频处理器；即使本服务只发送图片，环境中仍必须
安装与 PyTorch/CUDA 匹配的 `torchvision 0.20.1+cu124`。重新执行第 5.1 节安装命令，
再用第 5.1 节的 import 检查确认版本。

### `libpython3.8.so.1.0: cannot open shared object file`

这是 3060 客户端的 Isaac Gym 动态库路径问题，不是 VLM 服务故障。在启动评测前执行
第 13 节的 `LD_LIBRARY_PATH` 导出命令即可，不需要修改系统库或使用 sudo。

### strict load 报 key mismatch

确认 base model 和 checkpoint revision 正确、checkpoint 路径以 `/ckpt` 结尾。不要改成
`strict=False`。manifest SHA256 不匹配时也不要直接删除 manifest 绕过。

### CUDA OOM

先确认没有占卡 worker 或其他进程。再把 `server.env` 中：

```text
VLM_MAX_MICRO_BATCH_SIZE=4
```

改成 `2` 或 `1`，然后 stop/start。不要增加 Uvicorn worker。

### readiness 连接失败

```bash
./services/vlm_guidance/conda_server.sh logs
ps -fp "$(cat /home/hy/.cache/rr-vlm-guidance/server.pid)"
nvidia-smi -i 0
```

检查 manifest、模型路径、依赖导入、GPU OOM 和端口占用。

### 本地请求超时

依次确认：服务器本机 readiness、本地到 `10.71.106.240:8000` 的连通性、服务日志中的
forward 时间、`--vlm-timeout-seconds`，以及 batch size。

## 16. Docker 说明

仓库保留 `services/vlm_guidance/Dockerfile` 作为其他服务器的备选方案，但本次
`zju_4090_240` 不使用 Docker。当前 Docker 基础镜像是 CUDA 12.8 / PyTorch 2.9，和
本机 `550.120` 驱动组合不合适；不要照搬该镜像到这台服务器。

## 17. 参考

- [ModelScope Hub CLI](https://github.com/modelscope/modelscope_hub)
- [PyTorch 历史版本安装命令](https://docs.pytorch.org/get-started/previous-versions/)
- [Transformers 官方仓库与运行要求](https://github.com/huggingface/transformers)
- [NVIDIA nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
