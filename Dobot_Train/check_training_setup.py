#!/usr/bin/env python3
"""不下载模型、不占用GPU，只检查训练服务器的软件和数据集配置。"""

from __future__ import annotations

import torch

import constants as settings
from lerobot_source import use_local_lerobot
from Dobot_Train.train_pi05 import preflight


def main() -> int:
    use_local_lerobot()
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    preflight()
    config = PI05Config(
        pretrained_path=settings.PRETRAINED_MODEL,
        device="cuda",
        dtype="bfloat16",
        gradient_checkpointing=True,
        train_expert_only=False,
        freeze_vision_encoder=False,
        chunk_size=settings.POLICY_CHUNK_SIZE,
        n_action_steps=settings.POLICY_ACTION_STEPS,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("训练服务器CUDA不可用")
    print(f"CUDA设备：{torch.cuda.get_device_name(0)}")
    print(
        f"π0.5全量微调配置检查通过：chunk={config.chunk_size}, "
        f"execute={config.n_action_steps}"
    )
    print(
        f"数据集v3、{settings.STATE_DIM}维状态、{settings.ACTION_DIM}维动作"
        "和q01/q99统计量均正常。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
