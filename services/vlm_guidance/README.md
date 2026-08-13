# HY Furniture VLM 服务端部署手册

本文档用于把 `zhouhangzhu/hy_furniture` 部署到一台 NVIDIA RTX 4090
服务器，并向运行 diffusion policy 的本地机器提供 HTTP 推理接口。

推荐方案是：服务代码继续放在 `robust-rearrangement` 仓库中，但使用独立
Docker 镜像运行。这样服务端和客户端共享 prompt、相机坐标约定和接口版本，
同时不会污染本地 diffusion policy 的 Python 环境。

## 1. 为什么不用 vLLM

这个 checkpoint 不是标准文本生成模型。训练代码删除了 Qwen 的 LM head，新增了：

- 五分类 skill head：`push/pick/place/insert/screw`；
- 二维 point regression head：输出 Qwen `[0, 1000]` 坐标，随后转换到
  front camera 的 `320x240` 像素坐标。

推理需要取最后一个有效 token 的 hidden state，再调用这两个自定义 head。
vLLM 的标准生成/OpenAI 接口不会执行这里的 `FurniturePolicyModel.forward`，
因此不能直接加载和部署这个 checkpoint。

本服务按照模型仓库里的 `visualize_inference.py` 实现：

1. Transformers 加载原始 `Qwen3.5-2B`；
2. 丢弃 LM head，挂载自定义 skill/point heads；
3. 使用 `strict=True` 加载 `hy_furniture/ckpt/model.safetensors`；
4. FastAPI 对外提供结构化预测接口。

整体调用链如下：

```text
本地 evaluate_model.py
  -> VLMGuidanceClient
  -> HTTP POST /v1/guidance/predict
  -> 4090 服务器上的 FastAPI
  -> Transformers + FurniturePolicyModel
  -> skill + point_px
  -> 本地 diffusion policy
```

## 2. 部署前提

下面命令默认服务器使用 Linux，并约定：

```bash
export RR_REPO=/data/hy/robust-rearrangement
export VLM_ROOT=/data/hy/models/hy_furniture
```

如果服务器目录不同，只需要修改这两个变量。后续命令都不要使用本地机器的路径。

先确认 GPU、Docker 和 NVIDIA Container Toolkit 可用：

```bash
nvidia-smi
docker --version
docker info | sed -n '/Runtimes/,+3p'
```

`nvidia-smi` 应该能看到 RTX 4090。Docker runtime 信息中应出现
`nvidia`。如果 `docker` 无权限，需要让管理员把当前用户加入 Docker 用户组，
或者在所有 Docker 命令前加 `sudo`。

服务器还需要有足够空间存放：

- Qwen3.5-2B 原始权重；
- HY Furniture 完整 checkpoint；
- PyTorch CUDA Docker 镜像；
- 构建后的服务镜像。

建议至少预留 35 GB。

## 3. 把本项目同步到服务器

服务端必须使用包含 `services/vlm_guidance` 的这个版本。可以在服务器重新 clone，
也可以通过已有代码同步流程把当前工作区传过去。

如果服务器能够访问项目的 GitHub origin，并且这些改动已经 push，可以执行：

```bash
mkdir -p /data/hy
git clone git@github.com:amorphophallus/robust-rearrangement-custom.git \
  "$RR_REPO"
cd "$RR_REPO"
git checkout main
git pull --ff-only origin main
```

如果仓库已经存在，只执行最后三条。若改动还没有 push，则先从本地同步当前工作区；
不要在服务器使用一个尚未包含 `services/vlm_guidance` 的旧 commit。

完成后确认：

```bash
cd "$RR_REPO"
test -f services/vlm_guidance/Dockerfile
test -f services/vlm_guidance/app.py
test -f src/vlm_data_generator.py
```

三条命令都没有输出即表示文件存在。

## 4. 下载并固定模型

### 4.1 创建下载工具环境

模型只需要下载一次。建议使用单独 venv：

```bash
mkdir -p "$VLM_ROOT"
python3 -m venv "$VLM_ROOT/download-env"
source "$VLM_ROOT/download-env/bin/activate"
python -m pip install --upgrade pip
python -m pip install modelscope-hub Pillow
```

### 4.2 下载 HY Furniture checkpoint

固定使用下面的 checkpoint commit：

```text
933f15ce0ed0bc7108ec1f42074bf94d985a4cbf
```

下载：

```bash
ms-hub download zhouhangzhu/hy_furniture \
  --revision 933f15ce0ed0bc7108ec1f42074bf94d985a4cbf \
  --local-dir "$VLM_ROOT/source"
```

### 4.3 下载 Qwen3.5-2B base model

当前实现固定到下面的 base model commit：

```text
c00cc5fd7803c60b7788e053dcce33d0d26b11ef
```

下载：

```bash
ms-hub download Qwen/Qwen3.5-2B \
  --revision c00cc5fd7803c60b7788e053dcce33d0d26b11ef \
  --local-dir "$VLM_ROOT/base_model"
```

如果 ModelScope 命令出现超时，可以重新执行同一条命令；下载器会复用已完成文件。

### 4.4 检查目录

执行：

```bash
test -f "$VLM_ROOT/source/ckpt/model.safetensors"
test -f "$VLM_ROOT/source/ckpt/config.json"
test -f "$VLM_ROOT/source/ckpt/processor_config.json"
test -f "$VLM_ROOT/source/ckpt/tokenizer.json"
test -f "$VLM_ROOT/base_model/config.json"
find "$VLM_ROOT/base_model" -maxdepth 1 -name '*.safetensors' -print
```

最后一条命令应至少打印一个 base-model 权重文件。预期目录结构是：

```text
/data/hy/models/hy_furniture/
├── base_model/
│   ├── config.json
│   └── *.safetensors
└── source/
    └── ckpt/
        ├── chat_template.jinja
        ├── config.json
        ├── model.safetensors
        ├── processor_config.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

## 5. 生成模型 manifest

manifest 用来防止部署时误用了其他 checkpoint、旧 point head 或被修改的权重。
生成过程会计算两个模型权重的 SHA256，因此可能需要几分钟。

```bash
cd "$RR_REPO"
source "$VLM_ROOT/download-env/bin/activate"

python -m services.vlm_guidance.prepare_manifest \
  --base-model-dir "$VLM_ROOT/base_model" \
  --checkpoint-dir "$VLM_ROOT/source/ckpt" \
  --base-revision c00cc5fd7803c60b7788e053dcce33d0d26b11ef \
  --checkpoint-revision 933f15ce0ed0bc7108ec1f42074bf94d985a4cbf \
  --output "$VLM_ROOT/manifest.json"

test -s "$VLM_ROOT/manifest.json"
```

如果这里报 `unexpected point policy` 或 `unexpected skill order`，说明下载的不是本服务
支持的新版输出头，不要跳过检查继续部署。

## 6. 构建 Docker 镜像

必须从 `robust-rearrangement` 仓库根目录构建：

```bash
cd "$RR_REPO"
docker build \
  -f services/vlm_guidance/Dockerfile \
  -t rr-vlm-guidance:hy-furniture-933f15ce \
  .
```

镜像以 `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` 为基础，并固定
`transformers==5.5.4`。第一次构建需要下载较大的 CUDA/PyTorch 镜像。

确认镜像存在：

```bash
docker image inspect rr-vlm-guidance:hy-furniture-933f15ce \
  --format '{{.Id}}'
```

## 7. 创建服务配置和 API Token

生成一个随机 Token，并写入只允许当前用户读取的环境文件：

```bash
VLM_TOKEN="$(openssl rand -hex 32)"
umask 077

printf '%s\n' \
  'VLM_BASE_MODEL_DIR=/models/base_model' \
  'VLM_CHECKPOINT_DIR=/models/source/ckpt' \
  'VLM_MANIFEST_PATH=/models/manifest.json' \
  'VLM_MODEL_REVISION=933f15ce0ed0bc7108ec1f42074bf94d985a4cbf' \
  'VLM_DEVICE=cuda:0' \
  'VLM_ATTENTION_BACKEND=sdpa' \
  'VLM_MAX_LENGTH=4096' \
  'VLM_IMAGE_MAX_PIXELS=262144' \
  'VLM_MAX_MICRO_BATCH_SIZE=8' \
  "VLM_API_TOKEN=$VLM_TOKEN" \
  > "$VLM_ROOT/server.env"

chmod 600 "$VLM_ROOT/server.env"
```

不要把 `server.env` 提交到 Git，也不要把 Token 发到公开日志里。本地客户端稍后需要
使用同一个 Token。

## 8. 启动服务

如果服务器只有这一张 4090，可以直接给容器全部 GPU：

```bash
docker run -d \
  --name rr-vlm-guidance \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  --env-file "$VLM_ROOT/server.env" \
  -v "$VLM_ROOT:/models:ro" \
  rr-vlm-guidance:hy-furniture-933f15ce
```

如果服务器有多张 GPU，并且只想使用编号为 0 的 GPU，把 `--gpus all` 替换为：

```bash
--gpus '"device=0"'
```

服务只允许启动一个 Uvicorn worker。不要通过 `--workers` 启动多个进程，否则每个
worker 都会复制一份模型到同一张 GPU。

查看启动日志：

```bash
docker logs -f --tail 100 rr-vlm-guidance
```

启动过程会依次执行：manifest 校验、base model 加载、checkpoint strict load 和
一次 warmup。manifest 首次校验和模型加载可能需要几分钟。看到 Uvicorn 开始监听后，
按 `Ctrl-C` 退出日志跟踪不会停止容器。

查看容器状态和显存：

```bash
docker ps --filter name=rr-vlm-guidance
nvidia-smi
```

## 9. 在服务器本机检查 readiness

从环境文件读取 Token，只用于当前 shell：

```bash
set -a
source "$VLM_ROOT/server.env"
set +a

curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  http://127.0.0.1:8000/health/ready
```

成功时会返回类似：

```json
{
  "status": "ready",
  "model_revision": "933f15ce0ed0bc7108ec1f42074bf94d985a4cbf",
  "policy_version": 2,
  "device": "cuda:0",
  "attention_backend": "sdpa"
}
```

只要不是 `status=ready`，本地评测就不应开始。

## 10. 执行一次完整推理 smoke test

先生成两张符合接口尺寸要求的测试图：

```bash
source "$VLM_ROOT/download-env/bin/activate"

python - <<'PY'
from PIL import Image
Image.new("RGB", (320, 240), "black").save("/tmp/vlm-front.png")
Image.new("RGB", (320, 240), "black").save("/tmp/vlm-wrist.png")
PY
```

创建请求 metadata：

```bash
printf '%s' \
  '{"task":"one_leg","items":[{"request_id":"smoke-0","state_info":{"base":{"ee_pos_sim":[0.0,0.0,0.0],"ee_quat_sim":[0.0,0.0,0.0,1.0],"ee_pos_vel":[0.0,0.0,0.0],"ee_ori_vel":[0.0,0.0,0.0],"gripper_width":0.0}}}]}' \
  > /tmp/vlm-metadata.json
```

请求推理接口：

```bash
curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  -F 'metadata=</tmp/vlm-metadata.json;type=application/json' \
  -F 'front_0=@/tmp/vlm-front.png;type=image/png' \
  -F 'wrist_0=@/tmp/vlm-wrist.png;type=image/png' \
  http://127.0.0.1:8000/v1/guidance/predict
```

成功响应应包含：

- `policy_version: 2`；
- 与 manifest 一致的 `model_revision`；
- `predictions[0].skill`；
- `predictions[0].skill_probabilities`；
- `predictions[0].point_1000`；
- `predictions[0].point_px`；
- `timing_ms`。

黑图预测本身没有业务意义，这一步只验证完整的图片预处理、GPU forward 和自定义输出
head 都能工作。

## 11. 允许本地机器访问

查询服务器局域网 IP：

```bash
hostname -I
```

假设服务器 IP 是 `192.168.1.20`，在运行 diffusion policy 的本地机器执行：

```bash
export VLM_GUIDANCE_URL=http://192.168.1.20:8000
export VLM_API_TOKEN='<复制服务器 server.env 中的 VLM_API_TOKEN>'

curl --fail --show-error \
  -H "Authorization: Bearer $VLM_API_TOKEN" \
  "$VLM_GUIDANCE_URL/health/ready"
```

如果服务器本机可以访问但本地机器不行，需要检查服务器防火墙、云安全组或机房网络
ACL 是否允许本地机器访问 TCP 8000。只应允许可信内网来源，不建议把 8000 端口直接
暴露到公网。

当前接口使用 HTTP。跨不可信网络时，应在前面增加 HTTPS reverse proxy 或使用 VPN/
SSH 隧道；Bearer Token 本身不能加密传输内容。

## 12. 本地运行 VLM + diffusion policy

本地机器不需要安装服务端的 Transformers 环境，只需要当前项目原有评测环境以及
`requests`、`Pillow`。

在原有 `evaluate_model.py` 命令后增加：

```bash
--annotation-source vlm \
--vlm-timeout-seconds 10 \
--vlm-query-interval 0
```

完整形式示意：

```bash
cd /data/hy/robust-rearrangement

export VLM_GUIDANCE_URL=http://192.168.1.20:8000
export VLM_API_TOKEN='<服务器生成的 Token>'

python -m src.eval.evaluate_model \
  <原有 checkpoint、task、rollout 参数> \
  --annotation-source vlm \
  --vlm-timeout-seconds 10 \
  --vlm-query-interval 0
```

参数含义：

- `--vlm-query-interval 0`：自动使用 `actor.action_horizon` 作为查询间隔；
- 设置为正整数：每隔固定数量的 environment steps 请求一次 VLM，中间复用缓存；
- `--vlm-timeout-seconds`：单次 HTTP 请求超时，多环境大 batch 时可提高到 30；
- API、schema、policy version 或模型 revision 不匹配时直接停止 rollout，不会偷偷回退
  到自动机。

VLM 模式下，diffusion policy 实际使用 VLM 给出的 skill 和 2D point。自动机只作为
shadow GT 同步运行，用来计算：

- 所有实际控制 step 的平均像素误差和 RMSE；
- 按 oracle skill 分组的 step-average 像素误差；
- success-only 和 failure-only 的同类指标；
- 无效/出画 GT 数量和 coverage。

这些聚合结果会写入 console、evaluation JSON 和 W&B。保存 rollout 时，每帧还会保留
VLM prediction、oracle skill/point、cache age 和 point error，方便定位失败案例。

## 13. 常用运维命令

```bash
# 查看日志
docker logs --tail 200 rr-vlm-guidance

# 持续查看日志
docker logs -f rr-vlm-guidance

# 重启服务
docker restart rr-vlm-guidance

# 停止服务
docker stop rr-vlm-guidance

# 再次启动
docker start rr-vlm-guidance

# 查看容器状态
docker inspect rr-vlm-guidance --format '{{.State.Status}}'
```

更新服务代码后，需要重新 build 镜像并重建容器：

```bash
cd "$RR_REPO"
docker build \
  -f services/vlm_guidance/Dockerfile \
  -t rr-vlm-guidance:hy-furniture-933f15ce \
  .

docker stop rr-vlm-guidance
docker rm rr-vlm-guidance
```

然后重新执行第 8 节的 `docker run`。模型目录是只读挂载，不需要重新下载。

## 14. 常见问题

### `CUDA was requested but is unavailable`

容器没有拿到 GPU。检查：

```bash
docker run --rm --gpus all \
  pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime \
  python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

如果失败，修复 NVIDIA 驱动或 NVIDIA Container Toolkit 后再启动服务。

### readiness 一直返回 503

先看：

```bash
docker logs --tail 300 rr-vlm-guidance
```

常见原因包括模型文件路径错误、manifest 校验失败、checkpoint/base model 不匹配，或
warmup 发生 CUDA OOM。

### `checkpoint/model key mismatch` 或 strict load 报错

不要设置 `strict=False` 绕过。确认使用的是：

- base：`Qwen/Qwen3.5-2B` commit
  `c00cc5fd7803c60b7788e053dcce33d0d26b11ef`；
- checkpoint：`zhouhangzhu/hy_furniture` commit
  `933f15ce0ed0bc7108ec1f42074bf94d985a4cbf`；
- checkpoint 路径指向 `source/ckpt`，不是 `source`。

### `401 unauthorized`

客户端 Token 与服务器 `server.env` 不一致，或者请求没有携带：

```text
Authorization: Bearer <token>
```

### `422` 且提示图片尺寸错误

接口只接受两张 `320x240` RGB 图片。项目客户端会在请求前使用现有 rollout resize/
crop 逻辑处理图片；手写客户端时也必须遵守这个尺寸和 front/wrist 顺序。

### CUDA OOM

先把 `server.env` 中的：

```text
VLM_MAX_MICRO_BATCH_SIZE=8
```

改成 `4`、`2` 或 `1`，然后执行：

```bash
docker stop rr-vlm-guidance
docker rm rr-vlm-guidance
```

再重新执行第 8 节的 `docker run`。修改 `server.env` 后只执行
`docker restart` 不会更新容器已有的环境变量，必须重建容器。

不要通过增加 Uvicorn workers 解决吞吐问题；多个 worker 会复制模型并进一步增加显存。

### `flash_attn` 缺失

默认配置是 `VLM_ATTENTION_BACKEND=sdpa`，不依赖 FlashAttention。除非自行构建并验证了
与 PyTorch 2.9/CUDA 12.8 匹配的 FlashAttention，否则不要改成
`flash_attention_2`。

### 服务端正常，但本地请求超时

依次检查：

1. 本地能否访问 `http://服务器IP:8000/health/ready`；
2. 防火墙/安全组是否允许 TCP 8000；
3. `--vlm-timeout-seconds` 是否过小；
4. 服务器 `nvidia-smi` 是否有其他进程占满 GPU；
5. `docker logs` 中实际 forward 时间和错误信息。

## 15. 不使用 Docker 时的备选启动方式

只有服务器无法使用 Docker 时才建议这样部署：

```bash
python3 -m venv /data/hy/venvs/rr-vlm-guidance
source /data/hy/venvs/rr-vlm-guidance/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$RR_REPO/services/vlm_guidance/requirements.txt"

set -a
source "$VLM_ROOT/server.env"
set +a

# Docker 内路径改成服务器真实路径
export VLM_BASE_MODEL_DIR="$VLM_ROOT/base_model"
export VLM_CHECKPOINT_DIR="$VLM_ROOT/source/ckpt"
export VLM_MANIFEST_PATH="$VLM_ROOT/manifest.json"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$RR_REPO"

cd "$RR_REPO"
uvicorn services.vlm_guidance.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

这种方式需要自行配置进程守护和开机重启，因此生产/长期实验仍推荐 Docker 的
`--restart unless-stopped` 方案。
