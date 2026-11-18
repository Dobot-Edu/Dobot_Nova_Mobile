#!/usr/bin/env bash
# 一键启动 Top 相机和 VR 遥操。
# 用法：bash run_teleop.sh [full|arm_only|test]

_nova_teleop_main() (
  set -euo pipefail

  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  source "$root/config/runtime.env"
  profile=${1:-full}
  python_bin=${NOVA_PYTHON:-$NOVA_BOARD_PYTHON}

  case "$profile" in
    full|arm_only|test) ;;
    *)
      echo "模式必须是 full、arm_only、test，实际为：$profile" >&2
      return 2
      ;;
  esac
  if [[ ! -x "$python_bin" ]]; then
    echo "Python 不可执行：$python_bin；可通过 NOVA_PYTHON 指定。" >&2
    return 2
  fi

  cd "$root"
  mkdir -p logs
  top_pid=""

  cleanup() {
    if [[ -n "$top_pid" ]] && kill -0 "$top_pid" 2>/dev/null; then
      kill -INT "$top_pid" 2>/dev/null || true
      wait "$top_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  top_health_url=$("$python_bin" -c 'from config.devices import TOP_CAMERA_URL; print(TOP_CAMERA_URL.rsplit("/", 1)[0] + "/health")')

  camera_ready() {
    curl -fsS --max-time 1 "$top_health_url" 2>/dev/null \
      | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true'
  }

  if camera_ready; then
    echo "[一键启动] 复用已经运行的 Top 相机服务"
  else
    echo "[一键启动] 正在启动 Top 相机……"
    "$python_bin" -m Dobot_Init.top_camera --no-preview \
      >>logs/top_camera.log 2>&1 &
    top_pid=$!
    ready=0
    for _ in $(seq 1 40); do
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
      echo "Top 相机启动失败，日志：$root/logs/top_camera.log" >&2
      tail -n 30 logs/top_camera.log >&2 || true
      return 1
    fi
  fi

  echo "[一键启动] 遥操模式：$profile"
  "$python_bin" -m Dobot_Collection.run_teleop \
    --profile "$profile" --execute --enable-robot
)

_nova_teleop_main "$@"
status=$?
unset -f _nova_teleop_main
return "$status" 2>/dev/null || exit "$status"
