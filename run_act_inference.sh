#!/usr/bin/env bash
# 独立 ACT 试验入口。默认 DRY-RUN；真机必须显式设置 NOVA_ACT_EXECUTE=1。
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${NOVA_INFERENCE_PYTHON:-}" ]]; then
  python_bin=$NOVA_INFERENCE_PYTHON
else
  python_bin=/home/nvidia/miniforge3/envs/single/bin/python
fi
if [[ ! -x "$python_bin" ]]; then
  echo "没有找到推理 Python：$python_bin" >&2
  exit 2
fi

export NOVA_ACT_EXECUTE=${NOVA_ACT_EXECUTE:-0}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$root"

mode=DRY-RUN
[[ "$NOVA_ACT_EXECUTE" == "1" ]] && mode=真机执行
echo "[ACT启动] 模式：$mode"
echo "[ACT启动] Python：$python_bin"
act_checkpoint=$("$python_bin" -c 'from Dobot_Inference import settings_act; print(settings_act.CHECKPOINT_DIR)')
echo "[ACT启动] 权重：$act_checkpoint"

mkdir -p logs
top_pid=""
cleanup() {
  if [[ -n "$top_pid" ]] && kill -0 "$top_pid" 2>/dev/null; then
    kill -INT "$top_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

camera_ready() {
  "$python_bin" -c 'import json,urllib.request; from Dobot_Inference import settings_act; u=settings_act.TOP_CAMERA_URL.rsplit("/",1)[0]+"/health"; r=urllib.request.urlopen(u,timeout=1); raise SystemExit(0 if json.load(r).get("ok") else 1)' 2>/dev/null
}
if camera_ready; then
  echo "[相机] 复用已运行的 Top 服务"
else
  echo "[相机] 正在启动 Top 服务……"
  "$python_bin" -m Dobot_Init.top_camera --no-preview >>logs/top_camera.log 2>&1 &
  top_pid=$!
  ready=0
  for ((attempt=0; attempt<40; attempt++)); do
    if camera_ready; then ready=1; break; fi
    if ! kill -0 "$top_pid" 2>/dev/null; then break; fi
    sleep 0.25
  done
  if [[ "$ready" != 1 ]]; then
    echo "[相机] Top 启动失败，日志：$root/logs/top_camera.log" >&2
    tail -n 30 logs/top_camera.log >&2 || true
    exit 1
  fi
fi

"$python_bin" -m Dobot_Inference.infer_act
