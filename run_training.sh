#!/usr/bin/env bash
# 在已经安装 LeRobot π0.5 依赖的训练服务器上启动训练。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${NOVA_TRAIN_PYTHON:-python3}
cd "$root"
exec "$python_bin" -m Dobot_Train.train_pi05

