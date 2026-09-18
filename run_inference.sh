#!/usr/bin/env bash
# π0.5 推理入口。默认只读；真机执行必须显式设置 NOVA_EXECUTE=1。

_nova_inference_main() (
  set -euo pipefail
  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

  if [[ -n "${NOVA_INFERENCE_PYTHON:-}" ]]; then
    python_bin=$NOVA_INFERENCE_PYTHON
  else
    python_bin=""
    for candidate in \
      /home/nvidia/miniforge3/envs/single/bin/python \
      /root/miniforge3/envs/dobot-lerobot/bin/python \
      "$root/lerobot-main/.venv/bin/python"; do
      if [[ -x "$candidate" ]]; then
        python_bin=$candidate
        break
      fi
    done
  fi

  if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "没有找到可执行的推理 Python；请设置 NOVA_INFERENCE_PYTHON。" >&2
    exit 2
  fi

  export NOVA_EXECUTE=${NOVA_EXECUTE:-0}
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1

  if [[ "$NOVA_EXECUTE" == "1" ]]; then
    mode="真机执行"
  else
    mode="DRY-RUN"
  fi

  cd "$root"
  readarray -t resolved_paths < <(
    "$python_bin" -c 'from Dobot_Inference import settings; print(settings.CHECKPOINT_DIR); print(settings.resolve_dataset_meta_root())'
  )
  echo "[启动] 模式：$mode"
  echo "[启动] Python：$python_bin"
  echo "[启动] 权重：${resolved_paths[0]}"
  echo "[启动] 数据：${resolved_paths[1]}"

  # 只回收本入口启动的相机；已存在的服务由其原启动者管理。
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
    "$python_bin" -c 'import json, urllib.request; from Dobot_Inference import settings; url = settings.TOP_CAMERA_URL.rsplit("/", 1)[0] + "/health"; response = urllib.request.urlopen(url, timeout=1); raise SystemExit(0 if json.load(response).get("ok") else 1)' 2>/dev/null
  }

  if camera_ready; then
    echo "[相机] 复用已运行的 Top 服务"
  else
    echo "[相机] 正在启动 Top 服务……"
    "$python_bin" -m Dobot_Init.top_camera --no-preview >>logs/top_camera.log 2>&1 &
    top_pid=$!
    ready=0
    for ((attempt=0; attempt<40; attempt++)); do
      if camera_ready; then
        ready=1
        break
      fi
      if ! kill -0 "$top_pid" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if [[ "$ready" != 1 ]]; then
      echo "[相机] Top 启动失败，日志：$root/logs/top_camera.log" >&2
      tail -n 30 logs/top_camera.log >&2 || true
      exit 1
    fi
  fi

  # 不用exec：退出推理后需要执行相机清理。双路预览由推理进程共享相机帧。
  mkdir -p logs
  console_log="$root/logs/inference_console_$(date +%Y%m%d_%H%M%S).log"
  echo "[日志] 终端输出：$console_log"
  inference_status=0
  "$python_bin" -m Dobot_Inference.infer 2>&1 | tee "$console_log" || inference_status=${PIPESTATUS[0]}
  if [[ "$inference_status" != 0 ]]; then
    echo "[推理异常] Python 状态码=$inference_status；完整 traceback 已保存：$console_log" >&2
  fi
  return "$inference_status"
)

_nova_inference_status=0
_nova_inference_main "$@" || _nova_inference_status=$?
if [[ "$_nova_inference_status" != 0 ]]; then
  echo "[入口] 推理已停止，当前终端保留；状态码=$_nova_inference_status" >&2
fi
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return "$_nova_inference_status"
fi
exit "$_nova_inference_status"
