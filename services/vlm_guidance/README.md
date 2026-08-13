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
    └─ FurniturePolicyModel
         ├─ Qwen3.5-2B backbone
         ├─ 5 类 skill head
         └─ 2-D point regression head
```

VLM 返回：

- `skill`：`push/pick/place/insert/screw`；
- `point_1000`：Qwen `[0,1000]` 坐标；
- `point_px`：front camera 的 `320x240` 像素坐标；
- skill 概率、模型 revision 和服务端耗时。

## 2. 为什么不直接使用 vLLM

这个 checkpoint 不是标准的自回归文本生成 checkpoint。训练代码丢弃了 Qwen 的
language-model head，新增了一个五分类 head 和一个二维回归 head。推理时必须取得最后
一个有效 token 的 hidden state，再执行这两个自定义 head。

vLLM 的标准生成接口或 OpenAI-compatible API 不会自动执行项目里的
`FurniturePolicyModel.forward`，也不能把二维回归结果作为标准生成结果返回。即使底层
vLLM 支持 Qwen3.5 backbone，直接指向这个 checkpoint 仍会遇到输出 head 和 state-dict
不匹配。因此当前可靠方案是按照模型仓库的 `visualize_inference.py`，用 Transformers
重建 backbone + heads，并用 `strict=True` 加载 checkpoint。

不要通过 `strict=False`、补一个临时 LM head 或把 point 转成文本来绕开，这些做法会
改变模型语义或隐藏 checkpoint 不匹配。

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
export VLM_BASE_MODEL_DIR="$VLM_ROOT/Qwen3.5-2B"
export VLM_CHECKPOINT_DIR=/mnt/nas/share/home/hy/zhouhangzhu--hy_furniture/snapshots/master/ckpt
export VLM_MANIFEST_PATH="$VLM_ROOT/manifest.json"
export VLM_SERVER_ENV_FILE="$VLM_ROOT/server.env"
```

固定 revision：

```text
Qwen/Qwen3.5-2B:
c00cc5fd7803c60b7788e053dcce33d0d26b11ef

zhouhangzhu/hy_furniture:
933f15ce0ed0bc7108ec1f42074bf94d985a4cbf
```

服务器驱动 `550.120` 不适合 CUDA 12.8，所以本机实际使用：

```text
Python 3.11
PyTorch 2.5.1 + CUDA 12.4
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
import transformers

print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("fastapi:", fastapi.__version__)
print("torch file:", torch.__file__)
PY
```

`torch file` 必须位于 `$VLM_ENV` 下，不能位于 `/home/hy/.local`。

## 6. 下载 base model 和 checkpoint

创建模型目录：

```bash
mkdir -p "$VLM_ROOT"
```

下载并固定 Qwen3.5-2B：

```bash
"$VLM_ENV/bin/ms-hub" download Qwen/Qwen3.5-2B \
  --revision c00cc5fd7803c60b7788e053dcce33d0d26b11ef \
  --local-dir "$VLM_BASE_MODEL_DIR"
```

下载器支持断点续传。超时时重新执行同一条命令即可。

本次服务器已经有 HY Furniture checkpoint：

```text
/mnt/nas/share/home/hy/zhouhangzhu--hy_furniture/snapshots/master/ckpt
```

如果将来需要从零下载，可执行：

```bash
"$VLM_ENV/bin/ms-hub" download zhouhangzhu/hy_furniture \
  --revision 933f15ce0ed0bc7108ec1f42074bf94d985a4cbf \
  --local-dir "$VLM_ROOT/hy_furniture_source"

export VLM_CHECKPOINT_DIR="$VLM_ROOT/hy_furniture_source/ckpt"
```

检查模型文件：

```bash
test -f "$VLM_BASE_MODEL_DIR/config.json"
find "$VLM_BASE_MODEL_DIR" -maxdepth 1 -name '*.safetensors' -print

test -f "$VLM_CHECKPOINT_DIR/config.json"
test -f "$VLM_CHECKPOINT_DIR/model.safetensors"
test -f "$VLM_CHECKPOINT_DIR/processor_config.json"
test -f "$VLM_CHECKPOINT_DIR/tokenizer.json"
```

base model 的 `find` 至少应打印一个权重文件。

## 7. 生成并校验 manifest

manifest 会固定文件大小和 SHA256，防止误用旧输出头或被修改的权重：

```bash
cd "$RR_REPO"
env PYTHONNOUSERSITE=1 PYTHONPATH="$RR_REPO" \
  "$VLM_ENV/bin/python" -m services.vlm_guidance.prepare_manifest \
  --base-model-dir "$VLM_BASE_MODEL_DIR" \
  --checkpoint-dir "$VLM_CHECKPOINT_DIR" \
  --base-revision c00cc5fd7803c60b7788e053dcce33d0d26b11ef \
  --checkpoint-revision 933f15ce0ed0bc7108ec1f42074bf94d985a4cbf \
  --output "$VLM_MANIFEST_PATH"

test -s "$VLM_MANIFEST_PATH"
```

计算两个大权重文件的 SHA256 需要几分钟。如果出现 `unexpected point policy` 或
`unexpected skill order`，说明 checkpoint 不是当前新版输出头，必须停止部署。

## 8. 创建服务配置和 Token

```bash
mkdir -p "$VLM_ROOT"
umask 077
VLM_NEW_TOKEN="$(openssl rand -hex 32)"

{
  printf '%s\n' \
    "VLM_BASE_MODEL_DIR=$VLM_BASE_MODEL_DIR" \
    "VLM_CHECKPOINT_DIR=$VLM_CHECKPOINT_DIR" \
    "VLM_MANIFEST_PATH=$VLM_MANIFEST_PATH" \
    'VLM_MODEL_REVISION=933f15ce0ed0bc7108ec1f42074bf94d985a4cbf' \
    'VLM_DEVICE=cuda:0' \
    'VLM_ATTENTION_BACKEND=sdpa' \
    'VLM_MAX_LENGTH=4096' \
    'VLM_IMAGE_MAX_PIXELS=262144' \
    'VLM_MAX_MICRO_BATCH_SIZE=4' \
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

模型启动依次执行 manifest SHA256 校验、base model 加载、checkpoint strict load 和一次
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
{"status":"ready","model_revision":"933f15ce0ed0bc7108ec1f42074bf94d985a4cbf","policy_version":2,"device":"cuda:0","attention_backend":"sdpa"}
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

黑图预测没有业务意义；这里只验证双图预处理、GPU forward、自定义 heads 和 HTTP schema。

## 12. 从 3060 本地机器访问

服务器地址是 `10.71.106.240`。在 3060 本地机器执行：

```bash
export VLM_GUIDANCE_URL=http://10.71.106.240:8000
export VLM_API_TOKEN='<从 server.env 复制 VLM_API_TOKEN 的值>'

curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  "$VLM_GUIDANCE_URL/health/ready"
```

如果服务器本机 readiness 正常而本地失败，检查服务器防火墙或机房 ACL 是否允许 3060
机器访问 TCP 8000。当前是内网 HTTP；不要把 8000 直接暴露到公网。跨不可信网络应使用
SSH tunnel、VPN 或 HTTPS reverse proxy。

## 13. 运行 VLM + diffusion policy

本地评测环境只需要原有依赖、`requests` 和 `Pillow`，不需要安装服务端 Transformers。

在原有评测命令后增加：

```bash
--annotation-source vlm \
--vlm-base-url "$VLM_GUIDANCE_URL" \
--vlm-timeout-seconds 30 \
--vlm-query-interval 0
```

示例骨架：

```bash
cd /data/hy/robust-rearrangement

export VLM_GUIDANCE_URL=http://10.71.106.240:8000
export VLM_API_TOKEN='<与服务器一致的 Token>'

python -m src.eval.evaluate_model \
  <保留原有 checkpoint、task、n-envs、n-rollouts 等参数> \
  --annotation-source vlm \
  --vlm-base-url "$VLM_GUIDANCE_URL" \
  --vlm-timeout-seconds 30 \
  --vlm-query-interval 0
```

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
- rollout pickle 保存逐 step 的 VLM point、oracle point、误差、query step 和 cache age。

聚合 JSON/W&B 中的主字段是：

```text
vlm_point_error/all/overall/mean_error_px
vlm_point_error/all/overall/rmse_px
vlm_point_error/all/by_skill/<skill>/mean_error_px
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
